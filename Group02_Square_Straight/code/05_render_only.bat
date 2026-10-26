@echo off
cd /d "%~dp0"
python plot_square.py --root "%USERPROFILE%\N5000_square_runs" --movies --max-frames 120
if errorlevel 1 goto :failed
start "" "%USERPROFILE%\N5000_square_runs\index.html"
pause
exit /b 0
:failed
echo Failed: check result root and Python dependencies.
pause
exit /b 1
