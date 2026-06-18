import numpy as np

from redboxflip import vlm
from redboxflip.models import Settings


def test_looks_like_title_filters_junk():
    assert vlm._looks_like_title("OPEN WATER")
    assert vlm._looks_like_title("The Matrix")
    assert not vlm._looks_like_title("")
    assert not vlm._looks_like_title("x")
    assert not vlm._looks_like_title("NONE")
    assert not vlm._looks_like_title("I cannot determine the title")
    assert not vlm._looks_like_title("a" * 80)
    assert not vlm._looks_like_title("line one\nline two")


def test_clean_strips_quotes_and_space():
    assert vlm._clean('  "Open Water" ') == "Open Water"
    assert vlm._clean("Open   Water") == "Open Water"


def test_title_from_cover_returns_none_when_no_curl(monkeypatch):
    monkeypatch.setattr(vlm, "_curl_cmd", lambda: None)
    img = np.full((400, 300, 3), 255, np.uint8)
    assert vlm.title_from_cover(img, Settings()) is None


def test_title_from_cover_parses_response(monkeypatch):
    import json

    class FakeProc:
        returncode = 0
        stdout = json.dumps({"response": "  OPEN WATER \n"})
        stderr = ""

    monkeypatch.setattr(vlm, "_curl_cmd", lambda: "curl.exe")
    monkeypatch.setattr(vlm.subprocess, "run", lambda *a, **k: FakeProc())
    img = np.full((400, 300, 3), 200, np.uint8)
    assert vlm.title_from_cover(img, Settings()) == "OPEN WATER"


def test_title_from_cover_rejects_refusal(monkeypatch):
    import json

    class FakeProc:
        returncode = 0
        stdout = json.dumps({"response": "I cannot determine the title from this image."})
        stderr = ""

    monkeypatch.setattr(vlm, "_curl_cmd", lambda: "curl.exe")
    monkeypatch.setattr(vlm.subprocess, "run", lambda *a, **k: FakeProc())
    img = np.full((400, 300, 3), 200, np.uint8)
    assert vlm.title_from_cover(img, Settings()) is None


def _fake_resp(payload):
    import json

    class FakeProc:
        returncode = 0
        stdout = json.dumps({"response": json.dumps(payload)})
        stderr = ""

    return lambda *a, **k: FakeProc()


def test_is_upright_parses_bool(monkeypatch):
    monkeypatch.setattr(vlm, "_curl_cmd", lambda: "curl.exe")
    monkeypatch.setattr(vlm.subprocess, "run", _fake_resp({"upright": False}))
    img = np.full((400, 300, 3), 200, np.uint8)
    assert vlm.is_upright(img, Settings()) is False


def test_is_upright_none_without_curl(monkeypatch):
    monkeypatch.setattr(vlm, "_curl_cmd", lambda: None)
    img = np.full((400, 300, 3), 200, np.uint8)
    assert vlm.is_upright(img, Settings()) is None


def test_extract_face_returns_dict(monkeypatch):
    payload = {"upright": True, "all_text": "region 4 pal", "title": "Amadeus",
               "region": "Region 4", "low_confidence": ["studio"]}
    monkeypatch.setattr(vlm, "_curl_cmd", lambda: "curl.exe")
    monkeypatch.setattr(vlm.subprocess, "run", _fake_resp(payload))
    img = np.full((400, 300, 3), 200, np.uint8)
    out = vlm.extract_face(img, Settings())
    assert out["title"] == "Amadeus"
    assert out["upright"] is True


def test_extract_face_empty_on_failure(monkeypatch):
    monkeypatch.setattr(vlm, "_curl_cmd", lambda: None)
    img = np.full((400, 300, 3), 200, np.uint8)
    assert vlm.extract_face(img, Settings()) == {}
