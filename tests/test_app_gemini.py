"""Tests for the Gemini call in app.py.

These pin the fix for the historical 404: a current model (gemini-2.5-flash) on
the v1beta endpoint, with the API key carried in the x-goog-api-key header rather
than the URL query string (so the key never leaks into logs).
"""
import json
import numpy as np
import pytest

import app

WHITE = np.full((40, 30, 3), 255, dtype=np.uint8)


class FakeResp:
    def __init__(self, payload, ok=True, status_code=200, text=""):
        self._payload = payload
        self.ok = ok
        self.status_code = status_code
        self.text = text or json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status_code}")


def _gemini_ok(meta):
    """Shape a successful generateContent response around `meta`."""
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(meta)}]}}]}


def test_defaults_are_current_not_retired():
    # The whole point of the fix: do not ship a retired model/endpoint.
    assert app.GEMINI_MODEL == "gemini-2.5-flash"
    assert app.GEMINI_API_VERSION == "v1beta"
    assert "1.5" not in app.GEMINI_MODEL
    assert "2.0" not in app.GEMINI_MODEL


def test_gemini_url_has_no_key_and_correct_path():
    url = app._gemini_url("generateContent")
    assert url == (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-2.5-flash:generateContent"
    )
    assert "key=" not in url  # secret must never live in the URL


def test_analyze_sends_key_in_header_and_parses(monkeypatch):
    meta = {"side": "back", "rotation_cw": 90, "barcode": "0025192828928",
            "title": "King Kong Escapes DVD", "description": "Toho classic",
            "genre": "Sci-Fi", "studio": "Universal", "year": 1967}
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers or {}
        return FakeResp(_gemini_ok(meta))

    monkeypatch.setattr(app, "API_KEY", "FAKE_KEY")
    monkeypatch.setattr(app.requests, "post", fake_post)

    out = app.analyze_with_gemini(WHITE)
    assert out == meta
    # Endpoint + model are the fixed, current ones...
    assert captured["url"].endswith("/v1beta/models/gemini-2.5-flash:generateContent")
    # ...and the key travels in the header, not the URL.
    assert captured["headers"].get("x-goog-api-key") == "FAKE_KEY"
    assert "FAKE_KEY" not in captured["url"]
    assert "key=" not in captured["url"]


def test_analyze_404_returns_safe_fallback(monkeypatch):
    monkeypatch.setattr(app, "API_KEY", "FAKE_KEY")
    monkeypatch.setattr(app.requests, "post",
                        lambda *a, **k: FakeResp({}, ok=False, status_code=404,
                                                 text="model not found"))
    # On 404 the code asks for the model list; keep that offline too.
    monkeypatch.setattr(app.requests, "get",
                        lambda *a, **k: FakeResp({"models": []}))

    out = app.analyze_with_gemini(WHITE)
    assert out == app._empty_meta()
    assert out["title"] == "Unknown DVD" and out["side"] == "other"


def test_analyze_without_key_skips_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not hit the network without a key")

    monkeypatch.setattr(app, "API_KEY", "")
    monkeypatch.setattr(app.requests, "post", boom)
    assert app.analyze_with_gemini(WHITE) == app._empty_meta()


def test_list_available_models_filters_by_method(monkeypatch):
    payload = {"models": [
        {"name": "models/gemini-2.5-flash",
         "supportedGenerationMethods": ["generateContent", "countTokens"]},
        {"name": "models/embedding-001",
         "supportedGenerationMethods": ["embedContent"]},
        {"name": "models/gemini-3.5-flash",
         "supportedGenerationMethods": ["generateContent"]},
    ]}
    monkeypatch.setattr(app.requests, "get", lambda *a, **k: FakeResp(payload))
    usable = app.list_available_models("FAKE_KEY")
    assert usable == ["gemini-2.5-flash", "gemini-3.5-flash"]
