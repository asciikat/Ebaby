"""Launch Crop Studio: start uvicorn on 127.0.0.1:8765 and open the browser."""
import threading
import webbrowser

import uvicorn

from .app import create_app

HOST = "127.0.0.1"
PORT = 8765
URL = f"http://{HOST}:{PORT}/"


def build_app():
    """Construct the FastAPI app (own run dir + settings). Separated for testing."""
    return create_app()


def main():
    app = build_app()
    threading.Timer(1.0, lambda: webbrowser.open(URL)).start()
    print(f"Crop Studio running at {URL}  (Ctrl+C to stop)")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
