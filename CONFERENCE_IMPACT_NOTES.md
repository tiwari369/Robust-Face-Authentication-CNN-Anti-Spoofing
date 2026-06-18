# Conference Demonstration Notes

## Project Title

**A Robust Face Authentication Framework with CNN-Based Anti-Spoofing for Intelligent Attendance Systems**

## Technical Positioning

This project presents a practical face-authentication framework for intelligent attendance systems. It reduces manual roll-call effort, improves attendance transparency, and addresses proxy attendance by combining biometric matching, blink-based liveness verification, CNN anti-spoofing audit, image preprocessing, scheduled class validation, and automatic attendance report generation.

## Demonstration Pipeline

1. The teacher configures a class schedule.
2. The camera opens during the permitted class window.
3. The student appears in front of the webcam.
4. Blink liveness is verified.
5. The face crop is evaluated by the CNN anti-spoofing audit module.
6. The captured face is matched with enrolled dataset photographs.
7. LBPH provides secondary verification.
8. Attendance is saved with subject, timestamp, confidence, and audit details.

## Impact Points

- Contactless biometric attendance.
- Blink-based liveness verification before attendance marking.
- CNN anti-spoofing audit score for spoof-risk analysis.
- Captured-face to enrolled-photo matching using the prepared dataset.
- Schedule-aware attendance control for classroom use.
- CSV and Excel reporting for academic administration.
- Dashboard-based monitoring for administrators, teachers, and students.

## Suggested Conference Demonstration Sequence

1. Open the dashboard and show the project title.
2. Show the teacher schedule portal and active class window.
3. Start `attendance_system.py`.
4. Complete one blink-based successful attendance marking case.
5. Press SPACE to demonstrate manual face capture and forced matching.
6. Show the saved CSV attendance record.
7. Show the audit snapshot and manual capture folders.
8. Show the admin panel and retraining option.
