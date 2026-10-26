@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
echo Checking time-endpoint handling...
python test_clock_fix.py
if errorlevel 1 (
  echo Time-endpoint test FAILED. Check Python or copied files.
  pause
  exit /b 1
)
echo.
echo Resuming ONLY K=1.5 to t=300 (K=0 is already finished)...
python group01.py --boundary mirror --N 5000 --K 1.5 --end 300 --root ../my_runs --resume --plot --movies
if errorlevel 1 (
  echo.
  echo Simulation stopped with an error. Keep the data and send a screenshot.
  pause
  exit /b 1
)
echo.
echo Completed. Open ..\my_runs\mirror\index.html
pause
