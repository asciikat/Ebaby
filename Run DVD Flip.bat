@echo off
REM Launch the Red-Box DVD Flip desktop app inside WSL2 / Ubuntu (WSLg shows the GUI).
wsl.exe -e bash -lc "cd '/mnt/c/Users/mardi/Documents/Ebay code' && source ~/ebay-venv/venv/bin/activate && python3 -m redboxflip"
pause
