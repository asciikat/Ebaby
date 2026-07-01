import cv2
import numpy as np
from fastapi.testclient import TestClient

from cropstudio.app import create_app
from cropstudio import manifest
import cropstudio.service as service


def _jpeg(color=(20, 20, 20)):
    img = np.full((40, 30, 3), color, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def _client(tmp_path, monkeypatch, title="Goober"):
    # Force the title reader to be deterministic (no live Ollama).
    monkeypatch.setattr(service.vlm, "available", lambda: True)
    monkeypatch.setattr(service.vlm, "title_from_cover", lambda bgr, s: title)
    return TestClient(create_app(run_dir=tmp_path, settings=None))


def _post(client, dvd_index, slot, data):
    return client.post("/shot",
                       data={"dvd_index": dvd_index, "slot": slot},
                       files={"image": ("s.jpg", data, "image/jpeg")})


def test_health(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.get("/health")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_three_shots_save_a_dvd(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, title="Goober")
    assert _post(c, 0, 0, _jpeg()).json()["status"] == "buffered"
    assert _post(c, 0, 1, _jpeg()).json()["status"] == "buffered"
    r = _post(c, 0, 2, _jpeg()).json()
    assert r["status"] == "saved" and r["title"] == "Goober"
    assert (tmp_path / "Goober - Inside.jpg").exists()
    assert r["used_stock"] is True


def test_finish_flushes_partial(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, title="Solo")
    _post(c, 0, 0, _jpeg())
    _post(c, 0, 1, _jpeg())
    r = c.post("/finish").json()
    assert r["status"] == "saved"
    assert (tmp_path / "Solo - Front Cover.jpg").exists()
    assert r["new_stock"] is True


def test_bad_image_is_rejected(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = _post(c, 0, 0, b"not-an-image")
    assert r.status_code == 400


def test_index_served(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.get("/")
    assert r.status_code == 200 and "Ebaby" in r.text


def test_saving_a_dvd_appends_a_manifest_entry(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, title="Goober")
    _post(c, 0, 0, _jpeg())
    _post(c, 0, 1, _jpeg())
    _post(c, 0, 2, _jpeg())
    entries = manifest.load(tmp_path)
    assert len(entries) == 1
    assert entries[0]["title"] == "Goober"
    assert entries[0]["used_stock"] is True
