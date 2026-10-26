@echo off
cd /d "%~dp0"
python group_square.py --boundaries mirror sliding mixed soft periodic --N 500 --K 0 1.5 --end 8 --root "%USERPROFILE%\N5000_square_quick" --resume --plot --movies --max-frames 20
if errorlevel 1 goto :failed
start "" "%USERPROFILE%\N5000_square_quick\index.html"
pause
exit /b 0
:failed
echo Failed: please photograph the error text.
pause
exit /b 1
