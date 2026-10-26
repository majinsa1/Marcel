@echo off
cd /d "%~dp0\code"
python group01.py --boundary sliding --N 5000 --K 0 0.5 1.5 3 --end 60 --root ../my_runs --plot --movies
if errorlevel 1 pause & exit /b 1
start "" "..\my_runs\sliding\index.html"
pause
