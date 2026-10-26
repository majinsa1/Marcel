@echo off
cd /d "%~dp0\code"
python -m pip install -r requirements.txt
if errorlevel 1 pause & exit /b 1
python test_group01.py
pause
