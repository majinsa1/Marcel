@echo off
cd /d "%~dp0\code"
python -m pip install -r requirements.txt
if errorlevel 1 pause & exit /b 1
python group01.py --boundary mirror --N 5000 --K 0 0.5 1.5 3 --end 60 --root ../my_runs --plot --movies
if errorlevel 1 pause & exit /b 1
start "" "..\my_runs\mirror\index.html"
pause
