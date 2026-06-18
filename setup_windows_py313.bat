@echo off
setlocal
cd /d "%~dp0"
echo ==========================================
echo LBAMS v4.3 Blink-CNN Capture-Match Python 3.13 setup
echo ==========================================
py -3.13 -m venv .venv
if errorlevel 1 (
  echo Could not create venv with Python 3.13. Check Python installation.
  pause
  exit /b 1
)
call .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python validate_project.py
pause
