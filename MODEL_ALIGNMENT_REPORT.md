# Model Alignment Report

## Project Title

**A Robust Face Authentication Framework with CNN-Based Anti-Spoofing for Intelligent Attendance Systems**

## Paper-to-Project Alignment

| Paper term | Project implementation | Main file |
|---|---|---|
| Intelligent attendance system | Dashboard, schedule portal, attendance camera, student portal, administrator portal | `dashboard.html`, `attendance_system.py` |
| Face authentication | Captured face crop matched against enrolled dataset templates | `attendance_system.py`, `models/face_gallery.npz` |
| CNN-based anti-spoofing | Lightweight convolutional anti-spoofing audit scorer with Conv2D filters, ReLU response, pooling, and live/spoof scoring | `cnn_antispoof.py` |
| Blink liveness verification | Open-eye -> closed-eye -> open-eye blink state machine before attendance marking | `attendance_system.py` |
| CLAHE/AHE preprocessing | Contrast enhancement applied to face crops before recognition | `attendance_system.py`, `train_model.py` |
| LBPH verification | Trained LBPH model retained as secondary verification | `models/face_model.yml` |
| Automated reporting | Attendance CSV and Excel export | `admin_panel.py`, `attendance/` |
| Schedule-aware attendance | Class windows controlled by `schedule.json` | `teacher_portal.html`, `schedule.json` |

## Runtime Decision Flow

1. A face is detected in the webcam frame.
2. Eye state is monitored until a valid blink is completed.
3. The face crop is passed to the CNN anti-spoofing audit module.
4. The captured face is compared with enrolled dataset photographs.
5. LBPH provides a secondary verification signal.
6. Attendance is saved only when liveness and identity matching conditions are satisfied.

## CNN Gate Mode

CNN anti-spoofing is enabled in audit mode by default:

```json
"enable_cnn_antispoof": true,
"enforce_cnn_antispoof": false
```

This displays and logs CNN-based spoof-risk evidence while avoiding false rejection on weak webcams. For a strict security demonstration, set:

```json
"enforce_cnn_antispoof": true
```
