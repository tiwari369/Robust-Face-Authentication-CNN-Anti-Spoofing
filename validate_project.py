"""Project validation for VS Code: python validate_project.py"""
from __future__ import annotations

import json
import pickle
import platform
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
errors: list[str] = []

print("Robust Face Authentication Framework - Validation")
print("=" * 56)
print(f"Python : {platform.python_version()}")
print(f"OpenCV : {cv2.__version__}")
print(f"NumPy  : {np.__version__}")
print(f"cv2.face module: {hasattr(cv2, 'face')}")
try:
    import sklearn
    print(f"scikit-learn: {sklearn.__version__}")
except Exception as exc:
    errors.append(f"scikit-learn import failed: {exc}")

try:
    from cnn_antispoof import LightweightCNNAntiSpoof
    dummy = np.full((96, 96), 128, dtype=np.uint8)
    cnn_result = LightweightCNNAntiSpoof().predict(dummy)
    print(f"CNN anti-spoof module: OK, score={cnn_result.score:.2f}")
except Exception as exc:
    errors.append(f"CNN anti-spoof module failed: {exc}")

required = [
    "dataset",
    "models/face_model.yml",
    "models/name_mapping.pkl",
    "models/face_gallery.npz",
    "attendance_system.py",
    "cnn_antispoof.py",
    "train_model.py",
    "schedule.json",
    "runtime_config.json",
    "attendance",
]
for rel in required:
    path = BASE_DIR / rel
    ok = path.exists()
    print(f"{rel:<28}: {'OK' if ok else 'MISSING'}")
    if not ok:
        errors.append(f"Missing {rel}")

students = [p for p in (BASE_DIR / "dataset").iterdir() if p.is_dir()] if (BASE_DIR / "dataset").exists() else []
print(f"Students in dataset        : {len(students)}")

try:
    with open(BASE_DIR / "models" / "name_mapping.pkl", "rb") as f:
        data = pickle.load(f)
    id_to_name = data.get("id_to_name", data) if isinstance(data, dict) else data
    print(f"Labels in trained model    : {len(id_to_name)}")
    print(f"Recognition mode           : {data.get('recognition_mode', 'LBPH') if isinstance(data, dict) else 'LBPH'}")
    print(f"Gallery threshold          : {data.get('gallery_similarity_threshold', 'n/a') if isinstance(data, dict) else 'n/a'}")
except Exception as exc:
    errors.append(f"Could not read name_mapping.pkl: {exc}")

try:
    gallery = np.load(BASE_DIR / "models" / "face_gallery.npz", allow_pickle=True)
    print(f"Dataset photo templates    : {len(gallery['labels'])}")
except Exception as exc:
    errors.append(f"Could not read face_gallery.npz: {exc}")

try:
    with open(BASE_DIR / "schedule.json", "r", encoding="utf-8") as f:
        schedule = json.load(f)
    print(f"Scheduled classes          : {len(schedule.get('schedules', []))}")
    print(f"Time-lock enabled          : {schedule.get('time_locked_attendance')}")
except Exception as exc:
    errors.append(f"Could not read schedule.json: {exc}")

try:
    with open(BASE_DIR / "runtime_config.json", "r", encoding="utf-8") as f:
        runtime_config = json.load(f)
    print(f"Blink gate required        : {runtime_config.get('require_blink_for_attendance')}")
    print(f"CNN anti-spoof enabled     : {runtime_config.get('enable_cnn_antispoof')}")
    print(f"CNN anti-spoof blocking    : {runtime_config.get('enforce_cnn_antispoof')}")
except Exception as exc:
    errors.append(f"Could not read runtime_config.json: {exc}")

attendance_files = sorted((BASE_DIR / "attendance").glob("attendance_*.csv")) if (BASE_DIR / "attendance").exists() else []
print(f"Preloaded attendance files : {len(attendance_files)}")

if errors:
    print("\nProblems found:")
    for e in errors:
        print(" -", e)
    print("\nFix packages with: python -m pip install -r requirements.txt")
else:
    print("\nValidation passed. Run: python attendance_system.py")
