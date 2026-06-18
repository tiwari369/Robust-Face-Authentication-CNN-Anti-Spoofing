# A Robust Face Authentication Framework with CNN-Based Anti-Spoofing for Intelligent Attendance Systems

A research-oriented intelligent attendance system that combines face authentication, blink-based liveness verification, and CNN-based anti-spoofing to improve the security and reliability of biometric attendance systems.

This project is designed to reduce fake attendance attempts caused by printed photos, mobile screen replays, static face images, and other spoofing attacks.

> **Project Type:** Major Project / Conference Research Implementation
> **Status:** Repository kept private until conference paper acceptance/publication confirmation.

---

## Author

**Riteek Raj Tiwari**
B.Tech Computer Science and Engineering
Mahatma Gandhi Central University, Motihari, Bihar

---

## Project Title

**A Robust Face Authentication Framework with CNN-Based Anti-Spoofing for Intelligent Attendance Systems**

---

## Project Overview

Traditional face recognition attendance systems can be vulnerable to spoofing attacks. A person may try to mark fake attendance using a printed photograph, mobile screen replay, or static face image.

To address this issue, this project introduces a multi-layer verification framework that uses:

* Face detection
* Face recognition
* Blink-based liveness verification
* CNN-based anti-spoofing audit
* Attendance marking only after valid identity and liveness verification

The system is built for academic demonstration, conference presentation, and research-based implementation of secure biometric attendance.

---

## Key Objectives

The main objectives of this project are:

1. To build a secure face-based attendance system.
2. To prevent fake attendance using printed photo or screen replay attacks.
3. To verify whether the detected face is live using blink detection.
4. To use CNN-based anti-spoofing for spoof audit and additional security.
5. To maintain attendance records in a structured format.
6. To provide dashboard and portal-based interaction for demonstration.
7. To create a practical academic implementation of intelligent attendance security.

---

## Core Features

### 1. Face Authentication

The system captures a live face from the camera and compares it with enrolled student face data. Attendance is marked only when the detected face matches a registered identity.

### 2. Blink-Based Liveness Detection

The system checks whether the person in front of the camera performs a real blink. This helps confirm that the face is live and not a static image.

### 3. CNN-Based Anti-Spoofing

A CNN-based anti-spoofing module is used to evaluate whether the detected face appears real or spoofed. This helps identify attacks such as:

* Printed photo attack
* Mobile screen replay attack
* Static image attack
* Non-live face attempt

### 4. Attendance Marking

After successful face recognition and liveness verification, attendance is recorded automatically with time and date details.

### 5. Admin Panel

The admin panel allows project-level operations such as:

* Adding a new student
* Capturing face images
* Retraining the recognition model
* Viewing registered students
* Exporting attendance records
* Deleting student data when required

### 6. Dashboard Interface

The project includes dashboard and portal pages for a better demonstration experience. These pages make the project easier to present during viva, project evaluation, and conference demonstration.

### 7. Schedule-Based Attendance Support

The system supports class schedule configuration using JSON files. Attendance can be controlled using predefined class time windows.

---

## Tech Stack

* Python 3.13
* OpenCV
* NumPy
* CNN-based anti-spoofing logic
* Face recognition logic
* LBPH-based verification support
* HTML
* CSS
* JavaScript
* JSON configuration files
* Windows batch scripts

---

## Project Workflow

```text
Camera Input
     ↓
Face Detection
     ↓
Preprocessing / Enhancement
     ↓
Blink Liveness Verification
     ↓
CNN Anti-Spoofing Audit
     ↓
Face Recognition / Identity Matching
     ↓
Attendance Marking
     ↓
CSV / Report Output
```

---

## System Architecture

The system follows a modular architecture:

```text
User Face Capture
       ↓
OpenCV Camera Module
       ↓
Face Detection Module
       ↓
Liveness Detection Module
       ↓
CNN Anti-Spoofing Module
       ↓
Recognition Module
       ↓
Attendance Storage Module
       ↓
Dashboard / Admin Panel
```

---

## Project Structure

```text
attendance_system.py              Main camera-based attendance system
admin_panel.py                    Terminal-based admin panel
cnn_antispoof.py                  CNN anti-spoofing module
train_model.py                    Model training script
run_dashboard.py                  Local dashboard server
view_attendance.py                Attendance viewing script
validate_project.py               Project validation script

dashboard.html                    Main dashboard page
admin_portal.html                 Admin portal page
student_portal.html               Student portal page
teacher_portal.html               Teacher portal page

runtime_config.json               Runtime configuration file
schedule.json                     Class schedule configuration
requirements.txt                  Python dependency list
PROJECT_RUN_GUIDE.txt             Project running guide
CONFERENCE_IMPACT_NOTES.md        Conference/project impact notes

open_dashboard.bat                Windows script to open dashboard
run_attendance.bat                Windows script to run attendance system
retrain_model.bat                 Windows script to retrain model
setup_windows_py313.bat           Windows setup script
```

---

## Sensitive Files Not Uploaded

The following folders/files are intentionally excluded from GitHub using `.gitignore`:

```text
dataset/
models/
attendance/
reports/
manual_captures/
attendance_snapshots/
*.csv
*.db
*.xlsx
*.log
```

These may contain biometric face images, trained model files, attendance records, generated reports, or private student data.

---

## How to Run the Project

Follow these steps on Windows using VS Code.

### Step 1: Clone the Repository

```bash
git clone https://github.com/tiwari369/Robust-Face-Authentication-CNN-Anti-Spoofing.git
```

### Step 2: Open the Project Folder

```bash
cd Robust-Face-Authentication-CNN-Anti-Spoofing
```

### Step 3: Create Virtual Environment

```bash
py -3.13 -m venv .venv
```

If the above command does not work, use:

```bash
python -m venv .venv
```

### Step 4: Activate Virtual Environment

For Windows PowerShell:

```bash
.venv\Scripts\activate
```

After activation, terminal should show:

```text
(.venv)
```

### Step 5: Upgrade pip

```bash
python -m pip install --upgrade pip setuptools wheel
```

### Step 6: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 7: Validate Project Files

```bash
python validate_project.py
```

This checks whether required project files and configurations are available.

### Step 8: Run Attendance System

```bash
python attendance_system.py
```

---

## Camera Controls

While the attendance camera window is open:

```text
Q      Quit the attendance camera
SPACE  Save manual face capture / force matching attempt
ENTER  Same as SPACE on many keyboards
B      Reset blink liveness state
R      Clear current visual message
```

---

## How Attendance Works

1. The camera captures the live face.
2. The system detects the face using OpenCV.
3. The person performs a blink for liveness verification.
4. The CNN anti-spoofing module checks whether the face appears real or fake.
5. The recognition module matches the face with registered student data.
6. If verification is successful, attendance is marked.
7. Attendance data is saved locally.

---

## Run the Dashboard

Start the dashboard server:

```bash
python run_dashboard.py
```

Then open the dashboard link shown in the terminal.

Common local dashboard URL:

```text
http://localhost:8000/dashboard.html
```

You can also use:

```bash
open_dashboard.bat
```

---

## Run the Admin Panel

```bash
python admin_panel.py
```

Default admin password:

```text
admin123
```

Admin panel options include:

* Add student
* Capture student face images
* Retrain model
* View student list
* Delete student
* Export attendance records

---

## Retrain the Model

After adding or deleting student data, retrain the model:

```bash
python train_model.py
```

Or use:

```bash
retrain_model.bat
```

---

## Runtime Configuration

Runtime behavior is controlled by:

```text
runtime_config.json
```

Important configuration options include:

```json
{
  "require_blink_for_attendance": true,
  "enable_cnn_antispoof": true,
  "enforce_cnn_antispoof": false
}
```

Recommended conference demo mode:

```text
Blink liveness: mandatory
CNN anti-spoofing: enabled as audit score
CNN strict blocking: optional
```

This keeps the demo stable while still showing anti-spoofing evidence.

---

## Schedule Configuration

Class schedule is controlled through:

```text
schedule.json
```

Example configuration:

```json
{
  "time_locked_attendance": true,
  "window_minutes": 60,
  "schedules": []
}
```

For unrestricted testing/demo, set:

```json
"time_locked_attendance": false
```

---

## Output

Attendance records are stored locally in attendance files. A typical attendance record may include:

* Student name
* Date
* Time
* Status
* Subject
* Mode
* Confidence score

---

## Conference Demonstration Points

During project demonstration, the following points can be shown:

1. Registered student face authentication.
2. Blink-based liveness verification.
3. CNN anti-spoofing score/audit.
4. Successful attendance marking.
5. Failed or delayed attendance for no blink/unknown face.
6. Admin panel features.
7. Dashboard interface.
8. Attendance output records.

---

## Why This Project Is Important

Face attendance systems are convenient but can be attacked using photos or videos. This project improves the reliability of such systems by adding liveness detection and anti-spoofing mechanisms.

The project is useful for:

* Smart classroom attendance
* Secure biometric attendance research
* Computer vision project demonstration
* Anti-spoofing research prototype
* Academic conference implementation

---

## Limitations

* Performance may depend on webcam quality.
* Very low lighting can reduce recognition accuracy.
* CNN anti-spoofing may require better training for real-world deployment.
* The system is designed for academic demonstration and needs further validation before production use.

---

## Future Scope

Possible future improvements include:

* Deep learning-based face embeddings
* Improved CNN anti-spoofing training
* LSTM/GRU-based temporal liveness detection
* Vision Transformer-based spoof detection
* Cloud-based attendance dashboard
* Mobile app integration
* Better database integration
* Role-based login system
* Real-time analytics dashboard

---

## Privacy and Ethical Note

This project may involve biometric data such as face images. Such data should not be made public without proper consent.

Before making this repository public, carefully check and remove:

```text
student face images
attendance snapshots
CSV attendance files
trained biometric models
reports containing personal details
conference paper files if unpublished
```

---

## Academic Note

This project is developed for academic, research, and conference demonstration purposes. The repository does not include private biometric datasets, report files, or unpublished paper files.

---

## Author

**Riteek Raj Tiwari**
B.Tech Computer Science and Engineering
Mahatma Gandhi Central University, Motihari, Bihar

---

## Disclaimer

This repository is intended for academic and research demonstration purposes only. It should not be used directly in a real-world biometric attendance environment without proper testing, ethical review, consent, and security validation.
