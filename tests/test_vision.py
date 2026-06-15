import json
import numpy as np
import pytest
from dvdflip import vision
from dvdflip.models import ClassifyResult

WHITE = np.full((40, 30, 3), 255, dtype=np.uint8)

def _fake_chat(content):
    def _call(messages, **kw):
        return content
    return _call

def test_classify_parses_clean_json(monkeypatch):
    payload = json.dumps({"side": "front", "rotation_cw": 180,
                          "title": "King Kong Escapes", "year": 1967,
                          "confidence": 0.9})
    monkeypatch.setattr(vision, "_chat", _fake_chat(payload))
    r = vision.classify_photo(WHITE, size_mm=(135, 190), barcode=None)
    assert isinstance(r, ClassifyResult)
    assert r.side == "front" and r.rotation_cw == 180 and r.year == 1967

def test_classify_clamps_bad_rotation(monkeypatch):
    payload = json.dumps({"side": "spaceship", "rotation_cw": 47,
                          "title": "", "year": "n/a", "confidence": 5})
    monkeypatch.setattr(vision, "_chat", _fake_chat(payload))
    r = vision.classify_photo(WHITE, size_mm=(0, 0), barcode=None)
    assert r.rotation_cw in (0, 90, 180, 270)
    assert r.side == "other"
    assert r.year is None
    assert 0.0 <= r.confidence <= 1.0

def test_classify_handles_garbage(monkeypatch):
    monkeypatch.setattr(vision, "_chat", _fake_chat("not json at all"))
    r = vision.classify_photo(WHITE, size_mm=(0, 0), barcode=None)
    assert r.side == "other" and r.confidence == 0.0

def test_health_false_when_down(monkeypatch):
    def boom(*a, **k):
        raise OSError("refused")
    monkeypatch.setattr(vision.requests, "get", boom)
    assert vision.ollama_available() is False
