import subprocess
import threading
import webbrowser

import uvicorn

from .webapp import build_session, create_app

HOST, PORT = "127.0.0.1", 8753


def _open_browser():
    url = f"http://{HOST}:{PORT}/"
    # Try Windows-interop openers FIRST: under WSL, webbrowser uses xdg-open,
    # which has no Linux browser to launch (it "succeeds" but opens nothing).
    for cmd in (["cmd.exe", "/c", "start", "", url],
                ["wslview", url],
                ["powershell.exe", "-NoProfile", "-Command", f"Start-Process '{url}'"]):
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return
        except Exception:
            continue
    try:
        webbrowser.open(url)
    except Exception:
        pass


def _serve(session, open_browser=True):
    app = create_app(session)
    if open_browser:
        threading.Timer(1.0, _open_browser).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


def run(open_browser=True):
    print("Processing photos from 'Images in' …")
    session = build_session(progress=print)
    print(f"Done. Open the browser to review {len(session.groups)} DVD(s).")
    _serve(session, open_browser=open_browser)


if __name__ == "__main__":
    run()
