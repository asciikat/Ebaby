@echo off
REM Launch Crop Studio (batch browser cropper) natively on Windows.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Could not find .venv\Scripts\python.exe
    echo Create it with:  py -m venv .venv  ^&^&  .venv\Scripts\pip install -r requirements-redbox.txt
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m cropstudio
pause
