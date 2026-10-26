@echo off
cd /d "%~dp0"
python -m pip install -r requirements.txt
if errorlevel 1 goto :failed
python test_group_square.py
if errorlevel 1 goto :failed
echo.
echo ALL CHECKS PASSED.
pause
exit /b 0
:failed
echo An error occurred. Keep this window open and take a screenshot.
pause
exit /b 1
