@echo off
REM Build the Crop Studio desktop exe (PyInstaller onedir).
REM Output: dist\CropStudio\CropStudio.exe
cd /d "%~dp0"

if not exist ".venv\Scripts\pyinstaller.exe" (
    echo pyinstaller not installed in .venv
    echo Install it with:  .venv\Scripts\pip install pyinstaller pywebview
    pause
    exit /b 1
)

".venv\Scripts\pyinstaller.exe" --noconfirm CropStudio.spec
echo.
echo Done. Exe at: dist\CropStudio\CropStudio.exe
pause
