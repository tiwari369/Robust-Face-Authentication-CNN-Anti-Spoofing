# A Robust Face Authentication Framework with CNN-Based Anti-Spoofing for Intelligent Attendance Systems

This repository contains a research-oriented implementation of an intelligent attendance system using face authentication, blink-based liveness verification, and CNN-based anti-spoofing. The project is designed as a conference demonstration system for improving the reliability of biometric attendance against spoofing attacks such as printed photos, replayed videos, and static face attempts.

> Status: Academic conference/research project. The repository is currently private until paper acceptance/publication confirmation.

## Overview

Traditional face recognition attendance systems can be vulnerable to spoofing attacks. This project adds multiple verification layers to improve system robustness:

- Webcam-based face authentication
- Blink-based liveness verification
- CNN-based anti-spoofing audit
- CLAHE/AHE-based preprocessing
- Captured-face matching with enrolled gallery images
- LBPH-based secondary verification
- Schedule-aware attendance control
- Dashboard-based demonstration workflow
- Attendance records and export support

## Core Features

1. **Blink Liveness Verification**  
   Attendance is marked only after the system detects a valid open-eye to closed-eye to open-eye blink sequence.

2. **CNN Anti-Spoofing Audit**  
   The detected face crop is evaluated using a lightweight CNN-based anti-spoofing module. The anti-spoofing score is displayed and logged for audit purposes.

3. **Face Authentication**  
   The live face crop is compared with enrolled student face images for identity verification.

4. **LBPH Secondary Verification**  
   LBPH recognition is retained as an additional verification layer.

5. **CLAHE/AHE Preprocessing**  
   Contrast enhancement is used to improve face recognition stability under uneven lighting conditions.

6. **Schedule-Aware Attendance**  
   Attendance marking can be controlled using predefined class time windows.

7. **Dashboard and Portals**  
   The project includes dashboard, administrator portal, student portal, and teacher schedule portal pages.

8. **Attendance Records and Reports**  
   Attendance is saved in CSV format and can be exported for reporting.

## Tech Stack

- Python 3.13
- OpenCV
- NumPy
- CNN-based anti-spoofing logic
- LBPH face recognition
- HTML/CSS dashboard pages
- JSON-based runtime configuration
- Windows batch scripts for easy execution

## Required Software

Use a 64-bit Windows system with Python 3.13.x.

Required tools:

- Python 3.13.x
- VS Code
- Webcam access
- Internet connection for first-time dependency installation

## Complete VS Code Setup

Open the project folder in VS Code. Then open:

```text
Terminal → New Terminal
