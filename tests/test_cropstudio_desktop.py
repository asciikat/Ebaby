import threading
import time
import urllib.request

import uvicorn

from cropstudio.desktop import HOST, _free_port


def test_free_port_is_bindable_and_distinct():
    a, b = _free_port(), _free_port()
    assert a > 0 and b > 0     # OS-assigned ephemeral ports


def test_server_thread_serves_health(tmp_path, monkeypatch):
    monkeypatch.setenv("CROPSTUDIO_OUTPUT", str(tmp_path))
    from cropstudio.app import create_app
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(), host=HOST, port=port,
                                           log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 10
    last_err = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://{HOST}:{port}/health", timeout=2) as r:
                assert b'"ok":true' in r.read()
                break
        except Exception as e:
            last_err = e
            time.sleep(0.3)
    else:
        raise AssertionError(f"server never came up: {last_err}")
    server.should_exit = True
