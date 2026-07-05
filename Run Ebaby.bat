@echo off
echo Starting Ebaby server in WSL...
start "" wsl.exe -d Debian bash -lc "source ~/Ebabyv2/bin/activate && cd '/mnt/c/Users/mardi/Documents/Ebay code' && python3 -m ebaby.server"
timeout /t 3 /nobreak >nul
start "" http://localhost:8765
