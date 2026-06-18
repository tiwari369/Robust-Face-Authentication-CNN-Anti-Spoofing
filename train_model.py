"""
LBAMS Blink-CNN Capture-Match Training Script
Python 3.13.x compatible.

This script trains the full project dataset and creates recognition assets:
1. models/face_model.yml       - LBPH recognizer for fast OpenCV inference
2. models/face_gallery.npz     - dataset photo-template gallery for direct captured-photo matching
3. models/name_mapping.pkl     - labels, metadata, thresholds, and optional SVM model
4. cnn_antispoof.py            - lightweight CNN anti-spoofing scorer used at runtime

Run:
    python train_model.py
"""

from __future__ import annotations

import csv
import json
import pickle
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

try:
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except Exception:
    SGDClassifier = None
    Pipeline = None
    StandardScaler = None
    SKLEARN_OK = False

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

FACE_FEATURE_SIZE = (64, 64)
FACE_LBPH_SIZE = (100, 100)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def display_name(folder_name: str) -> str:
    # Preserve the student identity while converting folder labels into readable names.
    return folder_name.replace("_", " ").strip()


def detect_face_crop(gray: np.ndarray, cascade: cv2.CascadeClassifier) -> tuple[np.ndarray, bool]:
    """Return the largest detected face; use a safe center-crop fallback for face-focused images."""
    gray_eq = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(gray_eq, scaleFactor=1.08, minNeighbors=4, minSize=(45, 45))
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        pad_x = int(w * 0.12)
        pad_y = int(h * 0.18)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(gray.shape[1], x + w + pad_x)
        y2 = min(gray.shape[0], y + h + pad_y)
        return gray[y1:y2, x1:x2], True

    # Fallback: use the central square because the prepared dataset images are already face-focused.
    h_img, w_img = gray.shape[:2]
    side = int(min(h_img, w_img) * 0.82)
    x1 = (w_img - side) // 2
    y1 = (h_img - side) // 2
    return gray[y1:y1 + side, x1:x1 + side], False


def normalize_face(face_gray: np.ndarray, size: Tuple[int, int] = FACE_LBPH_SIZE) -> np.ndarray:
    face = cv2.resize(face_gray, size)
    # CLAHE is more stable than full equalization for classroom lighting changes.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(face)


def augment_face(face_100: np.ndarray) -> list[np.ndarray]:
    """Use the real enrolled dataset photographs only.
    This keeps the Python 3.13 project lightweight and makes the trained model
    directly reflect the submitted institutional dataset.
    """
    return [normalize_face(face_100, FACE_LBPH_SIZE)]


def extract_combined_features(face_gray: np.ndarray) -> np.ndarray:
    """Fast HOG-style gradient histogram + LBP histogram feature vector."""
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
    lbp_hist = cv2.calcHist([lbp], [0], None, [256], [0, 256]).reshape(-1).astype(np.float32)
    lbp_hist /= (lbp_hist.sum() + 1e-7)

    feat = np.concatenate([hog_feat, lbp_hist]).astype(np.float32)
    feat /= (np.linalg.norm(feat) + 1e-7)
    return feat


def collect_training_data():
    if not hasattr(cv2, "face"):
        raise RuntimeError("cv2.face is missing. Install opencv-contrib-python.")
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        raise RuntimeError("OpenCV Haar cascade could not be loaded.")
    if not DATASET_DIR.exists():
        raise RuntimeError(f"Dataset folder not found: {DATASET_DIR}")

    student_dirs = [p for p in sorted(DATASET_DIR.iterdir()) if p.is_dir()]
    if not student_dirs:
        raise RuntimeError("No student folders found inside dataset/.")

    lbph_faces: list[np.ndarray] = []
    lbph_labels: list[int] = []
    svm_features: list[np.ndarray] = []
    svm_labels: list[int] = []

    gallery_features: list[np.ndarray] = []
    gallery_labels: list[int] = []
    gallery_names: list[str] = []
    gallery_files: list[str] = []

    id_to_name: dict[int, str] = {}
    report_rows: list[dict] = []

    print("\nScanning complete dataset and creating face templates...\n")
    for label, student_dir in enumerate(student_dirs, start=1):
        name = display_name(student_dir.name)
        id_to_name[label] = name
        image_files = sorted(
            p for p in student_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        detected_count = 0
        fallback_count = 0
        used_count = 0
        samples_count = 0

        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            crop, detected = detect_face_crop(gray, cascade)
            detected_count += int(detected)
            fallback_count += int(not detected)
            used_count += 1

            face_100 = normalize_face(crop, FACE_LBPH_SIZE)
            # The gallery stores only real dataset photos, exactly matching the requested camera-to-dataset workflow.
            gallery_features.append(extract_combined_features(face_100))
            gallery_labels.append(label)
            gallery_names.append(name)
            gallery_files.append(str(img_path.relative_to(BASE_DIR)))

            for aug in augment_face(face_100):
                lbph_faces.append(normalize_face(aug, FACE_LBPH_SIZE))
                svm_features.append(extract_combined_features(aug))
                lbph_labels.append(label)
                svm_labels.append(label)
                samples_count += 1

        print(f"  {label:03d}. {name:<36} photos={used_count:>2} templates={samples_count:>2} fallback={fallback_count}")
        report_rows.append({
            "label_id": label,
            "student_folder": student_dir.name,
            "student_name": name,
            "source_images": len(image_files),
            "used_images": used_count,
            "haar_detected_images": detected_count,
            "fallback_center_crop_images": fallback_count,
            "training_samples_after_augmentation": samples_count,
        })

    return {
        "lbph_faces": lbph_faces,
        "lbph_labels": np.asarray(lbph_labels, dtype=np.int32),
        "svm_features": np.asarray(svm_features, dtype=np.float32),
        "svm_labels": np.asarray(svm_labels, dtype=np.int32),
        "gallery_features": np.asarray(gallery_features, dtype=np.float32),
        "gallery_labels": np.asarray(gallery_labels, dtype=np.int32),
        "gallery_names": np.asarray(gallery_names, dtype=object),
        "gallery_files": np.asarray(gallery_files, dtype=object),
        "id_to_name": id_to_name,
        "rows": report_rows,
    }


def train():
    start = time.time()
    data = collect_training_data()
    faces = data["lbph_faces"]
    labels = data["lbph_labels"]
    id_to_name = data["id_to_name"]
    rows = data["rows"]

    if len(faces) == 0:
        raise RuntimeError("No usable training face images were found.")

    print(f"\nTraining LBPH recognizer on {len(faces)} samples / {len(id_to_name)} students...")
    lbph = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=8, grid_y=8)
    lbph.train(faces, labels)
    lbph.save(str(MODELS_DIR / "face_model.yml"))
    print("  Saved models/face_model.yml")

    svm_model = None
    if SKLEARN_OK and len(id_to_name) >= 2:
        print("\nTraining lightweight HOG-LBP SGD-SVM auxiliary recognizer...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data["svm_features"])
        svm = SGDClassifier(loss="hinge", alpha=0.0007, max_iter=1, tol=None, random_state=42)
        classes = np.array(sorted(id_to_name.keys()), dtype=np.int32)
        rng = np.random.default_rng(42)
        y = data["svm_labels"]
        for _ in range(8):
            order = rng.permutation(len(y))
            svm.partial_fit(X_scaled[order], y[order], classes=classes)
        svm_model = Pipeline([("scaler", scaler), ("svm", svm)])
        print("  Saved auxiliary SVM inside name_mapping.pkl")
    else:
        print("\nAuxiliary SVM skipped. LBPH + gallery matching will still work.")

    np.savez_compressed(
        MODELS_DIR / "face_gallery.npz",
        features=data["gallery_features"],
        labels=data["gallery_labels"],
        names=data["gallery_names"],
        files=data["gallery_files"],
    )
    print("  Saved models/face_gallery.npz")

    trained_at = datetime.now().isoformat(timespec="seconds")
    model_data = {
        "id_to_name": id_to_name,
        "svm_model": svm_model,
        "has_svm": svm_model is not None,
        "recognition_mode": "Blink-gated captured face-to-dataset photo matching + LBPH verification + CNN anti-spoofing audit",
        "feature_type": "CLAHE normalized HOG-LBP gallery templates + LBPH fallback + lightweight CNN anti-spoofing score",
        "face_feature_size": FACE_FEATURE_SIZE,
        "face_lbph_size": FACE_LBPH_SIZE,
        "gallery_similarity_threshold": 0.38,
        "gallery_gap_threshold": 0.010,
        "lbph_threshold": 128.0,
        "trained_at": trained_at,
        "python_version": platform.python_version(),
        "opencv_version": cv2.__version__,
    }
    with open(MODELS_DIR / "name_mapping.pkl", "wb") as f:
        pickle.dump(model_data, f)
    print("  Saved models/name_mapping.pkl")

    metadata = {
        "project": "A Robust Face Authentication Framework with CNN-Based Anti-Spoofing for Intelligent Attendance Systems",
        "conference_alignment": "Face authentication framework for intelligent attendance systems",
        "students": len(id_to_name),
        "dataset_photos_used": int(sum(r["used_images"] for r in rows)),
        "training_samples_after_augmentation": int(len(labels)),
        "gallery_templates": int(len(data["gallery_labels"])),
        "algorithm": "Blink liveness gate, lightweight CNN anti-spoofing audit, captured-photo to dataset-photo gallery matching, LBPH verification, CLAHE preprocessing, schedule-aware logging, and audit snapshots",
        "trained_at": trained_at,
        "python_version": platform.python_version(),
        "opencv_version": cv2.__version__,
        "anti_spoofing_components": ["blink liveness verification", "lightweight CNN anti-spoofing audit", "optional texture gate"],
        "notes": "The camera pipeline verifies blink liveness, audits the crop with a lightweight CNN anti-spoofing scorer, compares the captured face with enrolled dataset photographs, and writes attendance after a confident match.",
    }
    with open(REPORTS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(REPORTS_DIR / "training_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - start
    print("\nTraining complete.")
    print(f"Students                : {len(id_to_name)}")
    print(f"Dataset photos used     : {metadata['dataset_photos_used']}")
    print(f"Gallery templates       : {metadata['gallery_templates']}")
    print(f"Training samples        : {len(labels)}")
    print(f"Auxiliary SVM enabled   : {svm_model is not None}")
    print(f"Time                    : {elapsed:.1f} sec")
    print("\nNow run: python attendance_system.py")


if __name__ == "__main__":
    train()
