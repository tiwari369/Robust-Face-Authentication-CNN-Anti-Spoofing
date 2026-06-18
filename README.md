# A Robust Face Authentication Framework with CNN-Based Anti-Spoofing for Intelligent Attendance Systems

This project package is prepared as a complete conference demonstration implementation for an intelligent attendance system. The system combines webcam-based face authentication, blink-based liveness verification, a lightweight CNN anti-spoofing audit module, CLAHE/AHE image preprocessing, captured-face to enrolled-photo gallery matching, LBPH verification, schedule-aware attendance control, and automated reporting.

## Project status

The package already includes:

- Prepared dataset with 232 enrolled identities and 696 face images.
- Trained gallery model: `models/face_gallery.npz`.
- Trained LBPH model: `models/face_model.yml`.
- Name mapping file: `models/name_mapping.pkl`.
- Preconfigured class schedule in `schedule.json`.
- Sample preloaded attendance records in the `attendance/` folder.
- Dashboard, administrator portal, student portal, and teacher schedule portal.
- Python 3.13-compatible dependency file.

## Core research-oriented features

1. **Blink liveness verification**: attendance is marked only after the system detects an open-eye to closed-eye to open-eye blink sequence.
2. **CNN anti-spoofing audit**: the detected face crop is evaluated through a lightweight CNN-style anti-spoofing module and the score is displayed and logged.
3. **Captured-face gallery matching**: the live face crop is compared with enrolled dataset images for identity verification.
4. **LBPH secondary verification**: LBPH recognition is retained as a secondary verification layer.
5. **CLAHE/AHE preprocessing**: contrast enhancement is applied to improve recognition stability in non-uniform lighting.
6. **Schedule-aware attendance**: attendance is controlled using predefined class time windows.
7. **Automatic records and reports**: attendance is saved in CSV and can be exported to Excel.
8. **Audit snapshots**: successful attendance snapshots and manual SPACE-key capture attempts are stored for demonstration and debugging.

## Required software

Use a 64-bit Windows installation with Python 3.13.x. The project was organized for Python 3.13.7.

Required tools:

- Python 3.13.x
- VS Code
- Webcam access enabled
- Internet connection for first-time dependency installation

## Complete VS Code setup

Open the extracted project folder in VS Code. Then open **Terminal → New Terminal** and run the following commands.

### 1. Create a virtual environment

```bash
py -3.13 -m venv .venv
```

### 2. Activate the virtual environment

```bash
.venv\Scripts\activate
```

After activation, the terminal prompt should show `(.venv)`.

### 3. Upgrade package tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 4. Install requirements

```bash
python -m pip install -r requirements.txt
```

### 5. Validate the project files

```bash
python validate_project.py
```

The validation script checks the required model files, dataset files, OpenCV face module, CNN anti-spoofing module, attendance folder, and schedule configuration.

## Run the attendance camera

After setup, run:

```bash
python attendance_system.py
```

Camera controls:

```text
Q      Quit the attendance camera
SPACE  Save a manual face capture and force one matching attempt
ENTER  Same as SPACE on many keyboards
B      Reset blink liveness state
R      Clear the current visual message
```

Important camera instructions:

1. Keep the OpenCV camera window active before pressing SPACE.
2. Center the face inside the green face box.
3. Blink once slowly.
4. Hold the face steady for one or two seconds.
5. Attendance is saved only after liveness and identity matching are both successful.

Manual SPACE-key captures are saved in:

```text
manual_captures/YYYY-MM-DD/
```

Successful attendance snapshots are saved in:

```text
attendance_snapshots/YYYY-MM-DD/
```

## Run the web dashboard

Start the dashboard server:

```bash
python run_dashboard.py
```

Then open the link shown in the terminal. It normally opens at:

```text
http://localhost:8000/dashboard.html
```

The dashboard title is:

**A Robust Face Authentication Framework with CNN-Based Anti-Spoofing for Intelligent Attendance Systems**

## Open dashboard directly on Windows

You can also use:

```bash
open_dashboard.bat
```

## Run the administrator panel

```bash
python admin_panel.py
```

Default administrator password:

```text
admin123
```

Administrator functions:

- Add a new student with webcam images.
- Retrain the recognition model.
- Export attendance records to Excel.
- View registered students.
- Delete a student and retrain the model.

After adding or deleting a student, run model retraining from the admin panel or run:

```bash
python train_model.py
```

## Run with one-click Windows scripts

First-time setup:

```bash
setup_windows_py313.bat
```

Attendance camera:

```bash
run_attendance.bat
```

Dashboard:

```bash
open_dashboard.bat
```

Model retraining:

```bash
retrain_model.bat
```

## Schedule configuration

The project includes a preconfigured demonstration schedule in `schedule.json`. Time-locked attendance is enabled by default.

To edit class schedules, open the dashboard and use the **Class Schedule** portal, or edit `schedule.json` directly.

Important fields:

```json
{
  "time_locked_attendance": true,
  "window_minutes": 60,
  "schedules": []
}
```

For unrestricted demonstration during testing, set:

```json
"time_locked_attendance": false
```

## Blink and CNN anti-spoofing configuration

The runtime behavior is controlled by `runtime_config.json`.

Blink is required by default:

```json
"require_blink_for_attendance": true
```

CNN anti-spoofing audit is enabled by default:

```json
"enable_cnn_antispoof": true
```

CNN blocking is disabled by default for classroom demo stability:

```json
"enforce_cnn_antispoof": false
```

To make CNN anti-spoofing strict, set:

```json
"enforce_cnn_antispoof": true
```

Recommended conference demo mode is to keep blink liveness as the mandatory gate and CNN as an audit score. This avoids unnecessary false rejection on low-quality webcams while still displaying and logging the CNN anti-spoofing evidence.

## Attendance output

Attendance records are saved in:

```text
attendance/attendance_YYYY-MM-DD.csv
```

Each record contains:

- Name
- Date
- Time
- Status
- Subject
- Mode
- Confidence

## View attendance in terminal

```bash
python view_attendance.py
```

To view a specific date:

```bash
python view_attendance.py 2026-05-28
```

## Export attendance to Excel

Use the administrator panel:

```bash
python admin_panel.py
```

Then choose the Excel export option.

## Project structure

```text
attendance_system.py       Main camera attendance application
cnn_antispoof.py           Lightweight CNN anti-spoofing audit module
train_model.py             Dataset training and model generation script
admin_panel.py             Terminal-based administrator panel
run_dashboard.py           Local dashboard server
dashboard.html             Main professional dashboard
admin_portal.html          Administrator guide portal
student_portal.html        Student attendance portal
teacher_portal.html        Class schedule portal
runtime_config.json        Runtime behavior configuration
schedule.json              Class schedule configuration
requirements.txt           Python 3.13 dependency list
dataset/                   Enrolled student image dataset
models/                    Trained model files
attendance/                Attendance CSV and exported reports
manual_captures/           SPACE-key capture attempts
attendance_snapshots/      Successful attendance audit snapshots
```

## Notes for conference demonstration

- Use a laptop webcam or an external webcam with good lighting.
- Keep the face centered and avoid strong backlight.
- Demonstrate a normal student attendance case first.
- Then demonstrate a failed or delayed case using no blink or an unknown face.
- Show the dashboard, class schedule, attendance CSV, and audit snapshot folders.
- Keep the paper title, dashboard title, and project title consistent during presentation.

## Academic integrity note

The project documentation has been written in original professional language for submission use. A final plagiarism or AI-content percentage cannot be certified inside the project; use the required checker or conference submission system before final submission.
