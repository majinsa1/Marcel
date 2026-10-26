@echo off
cd /d "%~dp0"
python group_square.py --boundaries mirror sliding mixed soft periodic --N 5000 --K 0.5 --end 300 --root "%USERPROFILE%\N5000_square_runs" --resume --plot --movies --max-frames 120
if errorlevel 1 goto :failed
python group_square.py --boundaries mirror periodic mixed soft sliding --N 5000 --K 3 --end 300 --root "%USERPROFILE%\N5000_square_runs" --resume --plot --movies --max-frames 120
if errorlevel 1 goto :failed
start "" "%USERPROFILE%\N5000_square_runs\index.html"
pause
exit /b 0
:failed
echo Incomplete group. Completed cases remain saved. Run this BAT again to resume.
pause
exit /b 1
