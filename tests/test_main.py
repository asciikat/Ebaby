import types
from dvdflip import __main__ as m

def test_main_has_run(monkeypatch):
    called = {}
    def fake_build(**k):
        called["built"] = True
        return types.SimpleNamespace(groups=[])
    monkeypatch.setattr(m, "build_session", fake_build)
    monkeypatch.setattr(m, "_serve", lambda session, open_browser: called.setdefault("served", True))
    monkeypatch.setattr(m.vision, "ollama_available", lambda: True)
    m.run(open_browser=False)
    assert called.get("built") and called.get("served")

def test_main_warns_when_ollama_down(monkeypatch, capsys):
    monkeypatch.setattr(m.vision, "ollama_available", lambda: False)
    m.run(open_browser=False)
    out = capsys.readouterr().out.lower()
    assert "ollama" in out
