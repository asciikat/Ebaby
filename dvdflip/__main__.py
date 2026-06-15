import threading
import webbrowser

import uvicorn

from . import vision
from .webapp import build_session, create_app

HOST, PORT = "127.0.0.1", 8753


def _serve(session, open_browser=True):
    app = create_app(session)
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{HOST}:{PORT}/")).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


def run(open_browser=True):
    if not vision.ollama_available():
        print("Ollama is not reachable at localhost:11434.")
        print("Start Ollama (and `ollama pull qwen2.5vl:7b`), then run again.")
        return
    print("Processing photos from 'Images in' …")
    session = build_session(progress=print)
    print(f"Done. Open the browser to review {len(session.groups)} DVD(s).")
    _serve(session, open_browser=open_browser)


if __name__ == "__main__":
    run()
