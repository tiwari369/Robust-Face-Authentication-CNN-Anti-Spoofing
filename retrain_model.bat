@echo off
cd /d "%~dp0"
if exist .venv\Scripts\activate call .venv\Scripts\activate
python train_model.py
pause
