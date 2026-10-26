@echo off
cd /d "%~dp0"
REM Example: ONLY K=1.5, mirror and periodic, resume or start to t=300.
python group_square.py --boundaries mirror periodic --N 5000 --K 1.5 --end 300 --root "%USERPROFILE%\N5000_square_runs" --resume --plot --movies
pause
