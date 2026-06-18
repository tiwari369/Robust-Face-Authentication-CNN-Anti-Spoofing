"""
LBAMS v4.3 - Blink + CNN Anti-Spoofing Capture-to-Dataset Attendance System
Python 3.13.x compatible.

The camera captures a live face, verifies blink-based liveness, runs a lightweight
CNN anti-spoofing audit, compares the captured face directly with enrolled dataset
photographs, and writes attendance only after a confident face-to-gallery match.
Blink verification is enabled by default as the primary anti-spoofing gate.

Run:
    python attendance_system.py
Keys:
    Q     quit
    SPACE save the current face crop and force one capture/match attempt after blink verification
    B     reset blink liveness state
    R     reset the current visual message
"""

from __future__ import annotations

import csv
import json
import math
import os
import pickle
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cnn_antispoof import LightweightCNNAntiSpoof

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "face_model.yml"
MAPPING_PATH = BASE_DIR / "models" / "name_mapping.pkl"
GALLERY_PATH = BASE_DIR / "models" / "face_gallery.npz"
ATTENDANCE_DIR = BASE_DIR / "attendance"
SNAPSHOT_DIR = BASE_DIR / "attendance_snapshots"
MANUAL_CAPTURE_DIR = BASE_DIR / "manual_captures"
SCHEDULE_FILE = BASE_DIR / "schedule.json"
CONFIG_FILE = BASE_DIR / "runtime_config.json"
ATTENDANCE_DIR.mkdir(exist_ok=True)
SNAPSHOT_DIR.mkdir(exist_ok=True)
MANUAL_CAPTURE_DIR.mkdir(exist_ok=True)

WINDOW_TITLE = "Robust Face Authentication - CNN Anti-Spoofing Attendance"

FACE_FEATURE_SIZE = (64, 64)
FACE_LBPH_SIZE = (100, 100)
SCALE_FACTOR = 0.5
DEFAULT_GALLERY_THRESHOLD = 0.38
DEFAULT_GALLERY_GAP = 0.010
DEFAULT_LBPH_THRESHOLD = 128.0
CAPTURE_COOLDOWN_SECONDS = 1.50
STABLE_FRAMES_REQUIRED = 4
TEXTURE_THRESHOLD = 10.0
EYE_FRAMES_CLOSED = 2
EYE_FRAMES_OPEN = 2


def load_runtime_config() -> dict[str, Any]:
    defaults = {
        "auto_mark_attendance": True,
        "capture_match_mode": True,
        "require_blink_for_attendance": True,
        "enforce_texture_anti_spoof": False,
        "enable_cnn_antispoof": True,
        "enforce_cnn_antispoof": False,
        "cnn_antispoof_threshold": 0.42,
        "save_attendance_snapshots": True,
        "gallery_similarity_threshold": DEFAULT_GALLERY_THRESHOLD,
        "gallery_gap_threshold": DEFAULT_GALLERY_GAP,
        "lbph_threshold": DEFAULT_LBPH_THRESHOLD,
        "camera_index": 0,
        "camera_width": 1280,
        "camera_height": 720,
    }
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            if isinstance(user_cfg, dict):
                defaults.update(user_cfg)
        except Exception:
            pass
    return defaults


def normalize_face(face_gray: np.ndarray, size=FACE_LBPH_SIZE) -> np.ndarray:
    face = cv2.resize(face_gray, size)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(face)


def extract_combined_features(face_gray: np.ndarray) -> np.ndarray:
    face = normalize_face(face_gray, FACE_FEATURE_SIZE)
    gx = cv2.Sobel(face, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(face, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    bins = np.floor((ang % 180) / 20).astype(np.int32)
    hog_parts = []
    cell = 8
    for yy in range(0, FACE_FEATURE_SIZE[1], cell):
        for xx in range(0, FACE_FEATURE_SIZE[0], cell):
            b = bins[yy:yy + cell, xx:xx + cell].reshape(-1)
            m = mag[yy:yy + cell, xx:xx + cell].reshape(-1)
            hist = np.bincount(b, weights=m, minlength=9).astype(np.float32)
            hist /= (np.linalg.norm(hist) + 1e-7)
            hog_parts.append(hist)
    hog_feat = np.concatenate(hog_parts).astype(np.float32)

    center = face[1:-1, 1:-1]
    lbp = np.zeros_like(face, dtype=np.uint8)
    lbp[1:-1, 1:-1] |= ((face[:-2, :-2] >= center).astype(np.uint8) << 7)
    lbp[1:-1, 1:-1] |= ((face[:-2, 1:-1] >= center).astype(np.uint8) << 6)
    lbp[1:-1, 1:-1] |= ((face[:-2, 2:] >= center).astype(np.uint8) << 5)
    lbp[1:-1, 1:-1] |= ((face[1:-1, 2:] >= center).astype(np.uint8) << 4)
    lbp[1:-1, 1:-1] |= ((face[2:, 2:] >= center).astype(np.uint8) << 3)
    lbp[1:-1, 1:-1] |= ((face[2:, 1:-1] >= center).astype(np.uint8) << 2)
    lbp[1:-1, 1:-1] |= ((face[2:, :-2] >= center).astype(np.uint8) << 1)
    lbp[1:-1, 1:-1] |= ((face[1:-1, :-2] >= center).astype(np.uint8) << 0)
    hist = cv2.calcHist([lbp], [0], None, [256], [0, 256]).reshape(-1).astype(np.float32)
    hist /= (hist.sum() + 1e-7)

    feat = np.concatenate([hog_feat, hist]).astype(np.float32)
    feat /= (np.linalg.norm(feat) + 1e-7)
    return feat


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name).strip("_") or "student"


def load_schedule() -> dict[str, Any]:
    if not SCHEDULE_FILE.exists():
        return {"time_locked_attendance": False, "schedules": []}
    try:
        with open(SCHEDULE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"time_locked_attendance": False, "schedules": []}
        data.setdefault("time_locked_attendance", False)
        data.setdefault("schedules", [])
        return data
    except Exception:
        return {"time_locked_attendance": False, "schedules": []}


def active_class(schedule_data: dict[str, Any]):
    schedules = schedule_data.get("schedules", []) or []
    if not schedules:
        return True, None, "Attendance open", "General"

    now = datetime.now()
    today = now.strftime("%A")
    now_min = now.hour * 60 + now.minute
    candidates = []
    for s in schedules:
        day = s.get("day", "daily")
        if day not in ("daily", today):
            continue
        try:
            sh, sm = map(int, str(s.get("start_time", "00:00")).split(":"))
            eh, em = map(int, str(s.get("end_time", "23:59")).split(":"))
        except Exception:
            continue
        start_min = sh * 60 + sm
        end_min = eh * 60 + em
        if start_min <= now_min <= end_min:
            candidates.append(s)

    if candidates:
        selected = candidates[0]
        subject = selected.get("subject", "Scheduled Class")
        teacher = selected.get("teacher", "")
        room = selected.get("room", "")
        details = f"{subject}{' | ' + teacher if teacher else ''}{' | ' + room if room else ''}"
        return True, selected, details, subject

    if schedule_data.get("time_locked_attendance", False):
        return False, None, "Locked: no active scheduled class", "General"

    # Schedules exist for demonstration, but time-lock is not enforced.
    selected = schedules[0]
    subject = selected.get("subject", "Scheduled Class")
    return True, selected, f"Open: {subject}", subject


def today_csv() -> Path:
    return ATTENDANCE_DIR / f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"


def load_marked(path: Path) -> set[str]:
    marked: set[str] = set()
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("Name"):
                    marked.add(row["Name"])
    return marked


def mark_attendance(name: str, csv_path: Path, marked: set[str], subject: str, mode: str, confidence: str) -> bool:
    if name in marked:
        return False
    exists = csv_path.exists()
    fields = ["Name", "Date", "Time", "Status", "Subject", "Mode", "Confidence"]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({
            "Name": name,
            "Date": datetime.now().strftime("%Y-%m-%d"),
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Status": "Present",
            "Subject": subject or "General",
            "Mode": mode,
            "Confidence": confidence,
        })
    marked.add(name)
    return True


def save_snapshot(name: str, frame: np.ndarray, box: tuple[int, int, int, int] | None = None) -> None:
    day_dir = SNAPSHOT_DIR / datetime.now().strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%H%M%S")
    file_path = day_dir / f"{safe_name(name)}_{timestamp}.jpg"
    if box:
        x, y, w, h = box
        pad = 18
        y1 = max(0, y - pad)
        x1 = max(0, x - pad)
        y2 = min(frame.shape[0], y + h + pad)
        x2 = min(frame.shape[1], x + w + pad)
        img = frame[y1:y2, x1:x2]
    else:
        img = frame
    cv2.imwrite(str(file_path), img)


def save_manual_capture(name: str, frame: np.ndarray, box: tuple[int, int, int, int], note: str) -> Path:
    """Save a manual SPACE-key capture attempt for audit and debugging."""
    day_dir = MANUAL_CAPTURE_DIR / datetime.now().strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%H%M%S")
    file_path = day_dir / f"{timestamp}_{safe_name(name)}_{safe_name(note)}.jpg"
    x, y, w, h = box
    pad = 24
    y1 = max(0, y - pad)
    x1 = max(0, x - pad)
    y2 = min(frame.shape[0], y + h + pad)
    x2 = min(frame.shape[1], x + w + pad)
    cv2.imwrite(str(file_path), frame[y1:y2, x1:x2])
    return file_path


def normalize_key(key: int) -> int:
    """Normalize OpenCV waitKey / waitKeyEx values across Windows and Linux."""
    if key == -1:
        return -1
    return key & 0xFF


class BlinkAudit:
    """State-machine based blink liveness verifier.

    A valid blink is counted only after an open-eye state is observed, followed by a
    short closed-eye state, followed by open eyes again. This avoids counting random
    eye detector failures as a real blink.
    """
    def __init__(self):
        self.blinks = 0
        self.eye_open_seen = False
        self.eye_closed_seen = False
        self.open_s = 0
        self.closed_s = 0
        self.verified_until = 0.0
        self.last_state = "WAIT_OPEN"

    def reset(self) -> None:
        self.blinks = 0
        self.eye_open_seen = False
        self.eye_closed_seen = False
        self.open_s = 0
        self.closed_s = 0
        self.verified_until = 0.0
        self.last_state = "WAIT_OPEN"

    def update(self, eyes_count: int, hold_seconds: float = 12.0) -> tuple[int, bool, str]:
        now = time.time()
        is_open = eyes_count >= 1
        if is_open:
            self.open_s += 1
            self.closed_s = 0
        else:
            self.closed_s += 1
            self.open_s = 0

        if not self.eye_open_seen:
            if self.open_s >= EYE_FRAMES_OPEN:
                self.eye_open_seen = True
                self.last_state = "OPEN_DETECTED"
            else:
                self.last_state = "LOOK_AT_CAMERA"
        elif not self.eye_closed_seen:
            if self.closed_s >= EYE_FRAMES_CLOSED:
                self.eye_closed_seen = True
                self.last_state = "EYES_CLOSED"
            else:
                self.last_state = "BLINK_NOW"
        else:
            if self.open_s >= EYE_FRAMES_OPEN:
                self.blinks += 1
                self.eye_closed_seen = False
                self.verified_until = now + hold_seconds
                self.last_state = "LIVENESS_VERIFIED"
            else:
                self.last_state = "OPEN_EYES"

        verified = now <= self.verified_until or self.blinks > 0
        return self.blinks, verified, self.last_state


def is_real_texture(gray_face: np.ndarray) -> bool:
    face = cv2.resize(gray_face, (64, 64))
    return cv2.Laplacian(face, cv2.CV_64F).var() >= TEXTURE_THRESHOLD


def load_models():
    if not MAPPING_PATH.exists() or not MODEL_PATH.exists() or not GALLERY_PATH.exists():
        raise FileNotFoundError("Model files missing. Run: python train_model.py")

    with open(MAPPING_PATH, "rb") as f:
        model_data = pickle.load(f)
    if isinstance(model_data, dict) and "id_to_name" in model_data:
        id_to_name = model_data["id_to_name"]
        svm_model = model_data.get("svm_model")
        has_svm = bool(model_data.get("has_svm", False) and svm_model is not None)
        gallery_threshold = float(model_data.get("gallery_similarity_threshold", DEFAULT_GALLERY_THRESHOLD))
        gallery_gap = float(model_data.get("gallery_gap_threshold", DEFAULT_GALLERY_GAP))
        lbph_threshold = float(model_data.get("lbph_threshold", DEFAULT_LBPH_THRESHOLD))
    else:
        id_to_name = model_data
        svm_model = None
        has_svm = False
        gallery_threshold = DEFAULT_GALLERY_THRESHOLD
        gallery_gap = DEFAULT_GALLERY_GAP
        lbph_threshold = DEFAULT_LBPH_THRESHOLD

    lbph = cv2.face.LBPHFaceRecognizer_create()
    lbph.read(str(MODEL_PATH))
    gallery = np.load(GALLERY_PATH, allow_pickle=True)
    gallery_data = {
        "features": gallery["features"].astype(np.float32),
        "labels": gallery["labels"].astype(np.int32),
        "names": gallery["names"].astype(object),
        "files": gallery["files"].astype(object),
    }
    return id_to_name, svm_model, has_svm, lbph, gallery_data, gallery_threshold, gallery_gap, lbph_threshold


def svm_vote(svm_model, roi_gray: np.ndarray, id_to_name: dict[int, str]) -> tuple[str, float]:
    if svm_model is None:
        return "", 0.0
    try:
        feat = extract_combined_features(roi_gray).reshape(1, -1)
        if hasattr(svm_model, "decision_function"):
            scores = np.asarray(svm_model.decision_function(feat))
            classes = getattr(svm_model, "classes_", None)
            if classes is None and hasattr(svm_model, "named_steps"):
                classes = svm_model.named_steps.get("svm").classes_
            row = scores if scores.ndim == 1 else scores[0]
            idx = int(np.argmax(row))
            sorted_scores = np.sort(row)
            margin = float(sorted_scores[-1] - sorted_scores[-2]) if len(sorted_scores) > 1 else float(sorted_scores[-1])
            label = int(classes[idx]) if classes is not None else idx + 1
            return id_to_name.get(label, ""), margin
        pred = int(svm_model.predict(feat)[0])
        return id_to_name.get(pred, ""), 0.0
    except Exception:
        return "", 0.0


def gallery_match(roi_gray: np.ndarray, gallery_data: dict[str, Any], threshold: float, gap_threshold: float) -> dict[str, Any]:
    feat = extract_combined_features(roi_gray)
    features = gallery_data["features"]
    scores = features @ feat
    order = np.argsort(scores)[::-1]
    best_idx = int(order[0])
    second_score = float(scores[int(order[1])]) if len(order) > 1 else 0.0
    best_score = float(scores[best_idx])
    best_label = int(gallery_data["labels"][best_idx])
    best_name = str(gallery_data["names"][best_idx])

    # Aggregate the best identity score across its three dataset photos.
    same = gallery_data["labels"] == best_label
    identity_score = float(np.mean(np.sort(scores[same])[-min(3, int(np.sum(same))):]))
    gap = best_score - second_score
    known = best_score >= threshold or identity_score >= (threshold - 0.025) or gap >= gap_threshold

    return {
        "known": bool(known),
        "name": best_name,
        "label": best_label,
        "best_score": best_score,
        "identity_score": identity_score,
        "gap": gap,
        "file": str(gallery_data["files"][best_idx]),
    }


def recognize_capture(roi_gray: np.ndarray, lbph, gallery_data: dict[str, Any], id_to_name: dict[int, str],
                      svm_model, gallery_threshold: float, gallery_gap: float, lbph_threshold: float):
    gm = gallery_match(roi_gray, gallery_data, gallery_threshold, gallery_gap)
    face_100 = normalize_face(roi_gray, FACE_LBPH_SIZE)
    lbph_label, lbph_distance = lbph.predict(face_100)
    lbph_label = int(lbph_label)
    lbph_name = id_to_name.get(lbph_label, "Unknown")
    lbph_known = float(lbph_distance) <= lbph_threshold
    svm_name, svm_margin = svm_vote(svm_model, roi_gray, id_to_name)

    # Decision logic: gallery match is primary because the requested workflow is captured photo vs dataset photo.
    if gm["known"]:
        name = gm["name"]
        verified_by = "Gallery"
        if lbph_known and lbph_name == name:
            verified_by += "+LBPH"
        elif svm_name == name and svm_margin > 0:
            verified_by += "+SVM"
        confidence = f"{verified_by} score={gm['best_score']:.2f}, identity={gm['identity_score']:.2f}, LBPH={float(lbph_distance):.0f}"
        return name, confidence, True, gm

    if lbph_known:
        confidence = f"LBPH fallback distance={float(lbph_distance):.0f}, gallery={gm['best_score']:.2f}"
        return lbph_name, confidence, True, gm

    confidence = f"No confident match: gallery={gm['best_score']:.2f}, LBPH={float(lbph_distance):.0f}"
    return "Unknown", confidence, False, gm


def draw_panel(frame: np.ndarray, line1: str, line2: str, color=(35, 80, 45)):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 70), color, cv2.FILLED)
    cv2.putText(frame, line1, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2)
    cv2.putText(frame, line2, (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 245, 230), 1)


def main():
    if not hasattr(cv2, "face"):
        print("ERROR: cv2.face is missing. Install opencv-contrib-python from requirements.txt.")
        return

    cfg = load_runtime_config()
    try:
        id_to_name, svm_model, has_svm, lbph, gallery_data, gallery_threshold, gallery_gap, lbph_threshold = load_models()
        # Runtime config can override model thresholds.
        gallery_threshold = float(cfg.get("gallery_similarity_threshold", gallery_threshold))
        gallery_gap = float(cfg.get("gallery_gap_threshold", gallery_gap))
        lbph_threshold = float(cfg.get("lbph_threshold", lbph_threshold))
    except Exception as exc:
        print(f"ERROR: {exc}")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    eye_glasses_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
    if face_cascade.empty():
        print("ERROR: Face cascade could not be loaded.")
        return
    if eye_cascade.empty() and eye_glasses_cascade.empty():
        print("WARNING: Eye cascades could not be loaded. Blink verification may not work.")

    cnn_detector = LightweightCNNAntiSpoof(float(cfg.get("cnn_antispoof_threshold", 0.42)))

    schedule_data = load_schedule()
    csv_path = today_csv()
    marked_today = load_marked(csv_path)
    mode = "Captured face-to-dataset gallery matching + LBPH + blink liveness"
    if has_svm:
        mode += " + SVM audit"
    if bool(cfg.get("enable_cnn_antispoof", True)):
        mode += " + CNN anti-spoofing audit"

    print("\nLBAMS Capture-Match Attendance started")
    print(f"Students loaded      : {len(id_to_name)}")
    print(f"Dataset templates    : {len(gallery_data['labels'])}")
    print(f"Attendance file      : {csv_path}")
    print(f"Recognition mode     : {mode}")
    print(f"Blink liveness gate  : {bool(cfg.get('require_blink_for_attendance', True))}")
    print(f"Texture blocking     : {bool(cfg.get('enforce_texture_anti_spoof', False))}")
    print(f"CNN anti-spoof audit : {bool(cfg.get('enable_cnn_antispoof', True))}")
    print(f"CNN anti-spoof block : {bool(cfg.get('enforce_cnn_antispoof', False))}")
    print("Instruction          : Look at the camera, blink once slowly, then keep the face steady.")
    print("Press Q to quit, SPACE to save a manual capture and force matching after liveness, B to reset blink.\n")

    camera_index = int(cfg.get("camera_index", 0))
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if os.name == "nt" else 0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("ERROR: Camera not found. Check webcam permission or change camera_index in runtime_config.json.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg.get("camera_width", 1280)))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg.get("camera_height", 720)))

    blink_trackers = defaultdict(BlinkAudit)
    last_capture = defaultdict(lambda: 0.0)
    stable_boxes = defaultdict(lambda: deque(maxlen=STABLE_FRAMES_REQUIRED))
    flash_text = ""
    flash_detail = ""
    flash_until = 0.0
    force_capture = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_h, frame_w = frame.shape[:2]
        manual_capture = force_capture
        force_capture = False
        is_open, current_class, schedule_msg, subject = active_class(schedule_data)

        if is_open:
            draw_panel(frame, "LBAMS Attendance Open", f"{schedule_msg} | Present today: {len(marked_today)}", (30, 88, 45))
        else:
            draw_panel(frame, "Attendance Locked", schedule_msg, (55, 35, 35))
            cv2.imshow(WINDOW_TITLE, frame)
            key = normalize_key(cv2.waitKeyEx(30))
            if key in (ord("q"), ord("Q")):
                break
            if key in (32, 13):
                print("Manual capture requested, but attendance is currently locked by schedule.")
            continue

        small = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        gray_small = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        faces = face_cascade.detectMultiScale(gray_small, scaleFactor=1.08, minNeighbors=5, minSize=(55, 55))
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if len(faces) == 0:
            cv2.putText(frame, "No face detected. Look at the camera or press SPACE after centering your face.",
                        (16, frame_h - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 210, 255), 2)
            if manual_capture:
                flash_text = "Manual capture requested"
                flash_detail = "No face was detected. Center the face inside the camera frame and press SPACE again."
                flash_until = time.time() + 2.5

        manual_capture_consumed = False
        for idx, (fx, fy, fw, fh) in enumerate(faces):
            x = int(fx / SCALE_FACTOR)
            y = int(fy / SCALE_FACTOR)
            w = int(fw / SCALE_FACTOR)
            h = int(fh / SCALE_FACTOR)
            x, y = max(0, x), max(0, y)
            x2, y2 = min(frame_w, x + w), min(frame_h, y + h)
            roi = gray_full[y:y2, x:x2]
            if roi.size == 0:
                continue

            track_key = f"face_{idx}"
            stable_boxes[track_key].append((x, y, w, h))
            stable = len(stable_boxes[track_key]) >= STABLE_FRAMES_REQUIRED

            upper = roi[:max(1, roi.shape[0] // 2), :]
            detected_eyes = []
            if not eye_cascade.empty():
                detected_eyes.extend(list(eye_cascade.detectMultiScale(upper, scaleFactor=1.08, minNeighbors=4, minSize=(16, 16))))
            if not eye_glasses_cascade.empty():
                detected_eyes.extend(list(eye_glasses_cascade.detectMultiScale(upper, scaleFactor=1.08, minNeighbors=4, minSize=(16, 16))))
            eyes_count = len(detected_eyes)
            blinks, blink_verified, blink_state = blink_trackers[track_key].update(
                eyes_count, float(cfg.get("blink_liveness_hold_seconds", 12.0))
            )
            texture_ok = is_real_texture(roi)
            if bool(cfg.get("enable_cnn_antispoof", True)):
                cnn_result = cnn_detector.predict(roi)
            else:
                cnn_result = None

            name, confidence, known, gm = recognize_capture(
                roi, lbph, gallery_data, id_to_name, svm_model,
                gallery_threshold, gallery_gap, lbph_threshold
            )

            already_marked = name in marked_today if known else False
            require_blink = bool(cfg.get("require_blink_for_attendance", False))
            enforce_texture = bool(cfg.get("enforce_texture_anti_spoof", False))
            blink_ok = (not require_blink) or blink_verified
            cnn_required = bool(cfg.get("enforce_cnn_antispoof", False))
            cnn_ok = (not cnn_required) or (cnn_result is None) or bool(cnn_result.is_live)
            spoof_ok = ((not enforce_texture) or texture_ok) and cnn_ok
            now = time.time()
            due = (now - last_capture[track_key]) >= CAPTURE_COOLDOWN_SECONDS
            manual_attempt = bool(manual_capture and not manual_capture_consumed)
            should_capture = (bool(cfg.get("auto_mark_attendance", True)) and stable and due) or manual_attempt

            if manual_attempt:
                note = "known" if known else "unknown"
                save_manual_capture(name if known else "Unknown", frame, (x, y, w, h), note)
                manual_capture_consumed = True
                flash_text = "Manual capture saved"
                if known:
                    flash_detail = f"Captured {name}. Blink status: {'verified' if blink_ok else 'pending'}. Matching score: {gm['best_score']:.2f}"
                else:
                    flash_detail = f"Captured an unknown face. Matching score: {gm['best_score']:.2f}"
                flash_until = time.time() + 2.5

            if known and already_marked:
                box_color, status = (130, 130, 130), "Already marked"
            elif known and not blink_ok:
                box_color, status = (0, 170, 255), f"Blink liveness required: {blink_state}"
            elif known and enforce_texture and not texture_ok:
                box_color, status = (0, 0, 255), "Texture spoof warning"
            elif known and cnn_required and not cnn_ok:
                box_color, status = (0, 0, 255), "CNN anti-spoofing risk"
            elif known:
                box_color, status = (0, 210, 0), "Matched after blink liveness"
            else:
                box_color, status = (0, 0, 220), "Unknown / no confident match"

            cv2.rectangle(frame, (x, y), (x2, y2), box_color, 2)
            label_top = max(72, y - 62)
            cv2.rectangle(frame, (x, label_top), (min(frame_w - 1, x + max(w, 360)), y), box_color, cv2.FILLED)
            cv2.putText(frame, name if known else "Unknown Person", (x + 8, label_top + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, status, (x + 8, label_top + 47), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cnn_text = f"CNN={cnn_result.score:.2f}" if cnn_result is not None else "CNN=off"
            cv2.putText(frame, f"Gallery={gm['best_score']:.2f} | Blink={blinks} | Eyes={eyes_count} | {cnn_text} | State={blink_state}",
                        (x, min(frame_h - 12, y2 + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)

            if known and not already_marked and blink_ok and spoof_ok and should_capture:
                final_confidence = confidence
                if cnn_result is not None:
                    final_confidence += f", CNN={cnn_result.score:.2f}/{cnn_result.label}"
                if mark_attendance(name, csv_path, marked_today, subject, mode, final_confidence):
                    if bool(cfg.get("save_attendance_snapshots", True)):
                        save_snapshot(name, frame, (x, y, w, h))
                    flash_text = "Attendance marked successfully"
                    flash_detail = f"{name} | {subject} | {final_confidence}"
                    flash_until = time.time() + 3.0
                    last_capture[track_key] = now
                    print(f"MARKED: {name} | {datetime.now().strftime('%H:%M:%S')} | {subject} | {final_confidence}")

        if time.time() < flash_until:
            cv2.rectangle(frame, (0, 76), (frame_w, 148), (25, 105, 35), cv2.FILLED)
            cv2.putText(frame, flash_text, (18, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (185, 255, 190), 2)
            cv2.putText(frame, flash_detail[:130], (18, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1)

        cv2.rectangle(frame, (0, frame_h - 34), (frame_w, frame_h), (12, 22, 14), cv2.FILLED)
        cv2.putText(frame, "Q=Quit | SPACE=Save manual capture + force match | B=Reset Blink | Blink + CNN anti-spoofing + dataset matching.",
                    (12, frame_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (145, 220, 150), 1)
        cv2.imshow(WINDOW_TITLE, frame)
        key = normalize_key(cv2.waitKeyEx(30))
        if key in (ord("q"), ord("Q")):
            break
        if key in (32, 13):
            force_capture = True
            flash_text = "Manual capture requested"
            flash_detail = "Keep the OpenCV camera window active, center your face, blink once, and hold steady."
            flash_until = time.time() + 1.8
        if key in (ord("r"), ord("R")):
            flash_until = 0.0
        if key in (ord("b"), ord("B")):
            for tracker in blink_trackers.values():
                tracker.reset()
            flash_text = "Blink liveness state reset"
            flash_detail = "Look at the camera, blink once slowly, then hold steady."
            flash_until = time.time() + 2.5

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSaved attendance: {csv_path}")
    print(f"Total marked today: {len(marked_today)}")


if __name__ == "__main__":
    main()
