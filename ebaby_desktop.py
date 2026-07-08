"""EBABY as a standalone desktop app (no browser).

Runs on WINDOWS Python (the repo .venv, which has pywebview). The pipeline
itself still runs inside WSL Debian — this shell boots that server if it
isn't already up, waits for the port, then opens a native window on it.
Closing the window shuts the server down again (unless it was already
running before we started, in which case we leave it alone).
"""
import json
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

PORT = 8765
SERVER_DIR = "/mnt/c/Users/mardi/Documents/Ebay code/.claude/worktrees/busy-napier-89a58d"
BOOT = (f"source ~/Ebabyv2/bin/activate && cd '{SERVER_DIR}' "
        f"&& python3 -m ebaby.server")


def port_open() -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", PORT)) == 0


def _watch_for_quit(window, stop_event):
    """Poll the server's quit flag (set by the 'Accept Hustle?' button) and
    close the window when it fires.

    This intentionally does NOT use pywebview's JS->Python bridge
    (window.pywebview.api / js_api=). That bridge proved unreliable in
    practice: sometimes unavailable to the page at all, and when it WAS
    reachable, calling window.destroy() from inside the bridge's own callback
    thread deadlocked the GUI on some backends (the window would blank and
    Windows would mark it "(Not Responding)"). Polling from a plain thread
    that this script itself owns — never invoked BY the webview, only ever
    calling INTO it — sidesteps both failure modes: destroy() is called the
    same way pywebview's own docs recommend (from a background thread while
    webview.start() runs the GUI loop on the main thread)."""
    url = f"http://127.0.0.1:{PORT}/api/quit-status"
    while not stop_event.is_set():
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if json.load(resp).get("quit"):
                    window.destroy()
                    return
        except (urllib.error.URLError, OSError, ValueError):
            pass  # server hiccup / not up yet — just keep polling
        stop_event.wait(0.5)


def main() -> int:
    try:
        import webview
    except ImportError:
        print("pywebview missing — run: .venv\\Scripts\\pip install pywebview")
        return 1

    started_here = False
    if not port_open():
        subprocess.Popen(
            ["wsl.exe", "-d", "Debian", "bash", "-lc", BOOT],
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        started_here = True
        deadline = time.time() + 90  # cold WSL + cv2 import can be slow
        while time.time() < deadline and not port_open():
            time.sleep(0.5)
        if not port_open():
            print("EBABY engine failed to start — check WSL Debian + ~/Ebabyv2")
            return 1

    window = webview.create_window(
        "EBABY — PLASTIC IN. PAPER OUT.",
        f"http://127.0.0.1:{PORT}",
        width=1180, height=980, background_color="#0a0a0d",
    )
    stop_event = threading.Event()
    watcher = threading.Thread(target=_watch_for_quit, args=(window, stop_event), daemon=True)
    watcher.start()
    webview.start()
    stop_event.set()  # window closed some other way (the X button) — stop polling

    if started_here:
        subprocess.run(
            ["wsl.exe", "-d", "Debian", "bash", "-lc", "pkill -f ebaby.server"],
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
