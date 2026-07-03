"""Desktop entry point: uvicorn on a background thread + a pywebview window.

This is what the PyInstaller exe runs. `python -m cropstudio` (uvicorn +
default browser tab) stays untouched for development.
"""
import socket
import threading

import uvicorn

from .app import create_app

HOST = "127.0.0.1"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def main():
    import webview   # lazy: heavy GUI import only when actually launching

    port = _free_port()
    app = create_app()
    server = uvicorn.Server(uvicorn.Config(app, host=HOST, port=port,
                                           log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()

    window = webview.create_window("Crop Studio", f"http://{HOST}:{port}/",
                                   width=1400, height=950)
    webview.start()          # blocks until the window closes
    server.should_exit = True


if __name__ == "__main__":
    main()
