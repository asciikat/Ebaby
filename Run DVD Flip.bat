@echo off
REM Launch the Red-Box DVD Flip desktop app natively on Windows.
REM (Replaces the old WSL launcher; the app now runs on the project's .venv.
REM  The GUI auto-fixes the venv Tcl/Tk paths in redboxflip/gui/app.py.)
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Could not find .venv\Scripts\python.exe
    echo Create it with:  py -m venv .venv  ^&^&  .venv\Scripts\pip install -r requirements-redbox.txt
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m redboxflip
pause
