@echo off
cd /d "%~dp0\code"
python group01.py --boundary soft --N 5000 --K 0 0.5 1.5 3 --end 60 --wall-width 2 --wall-k 4 --root ../my_runs --plot --movies
if errorlevel 1 pause & exit /b 1
start "" "..\my_runs\soft\index.html"
pause
