from redboxflip import titles


def test_clean_title_strips_format_noise():
    assert titles.clean_title("The Matrix (DVD)") == "The Matrix"
    assert titles.clean_title("Alien - Blu-ray") == "Alien"
    assert titles.clean_title("Heat [DVD Region 4]") == "Heat"


def test_cache_get_set(tmp_path, monkeypatch):
    monkeypatch.setattr(titles, "TITLE_CACHE_PATH", tmp_path / "c.json")
    cache = titles.load_cache()
    assert cache == {}
    cache["123"] = "The Matrix"
    titles.save_cache(cache)
    assert titles.load_cache()["123"] == "The Matrix"


def test_resolve_prefers_manual(monkeypatch):
    monkeypatch.setattr(titles, "lookup_online", lambda *a, **k: "WRONG")
    title, source = titles.resolve_title("123", do_lookup=True, cache={},
                                         manual="My Title")
    assert title == "My Title" and source == "manual"


def test_resolve_uses_cache_before_network(monkeypatch):
    called = {"n": 0}
    def fake(*a, **k):
        called["n"] += 1
        return "NET"
    monkeypatch.setattr(titles, "lookup_online", fake)
    title, source = titles.resolve_title("123", do_lookup=True,
                                          cache={"123": "Cached"})
    assert title == "Cached" and source == "cache"
    assert called["n"] == 0


def test_resolve_falls_back_to_lookup_and_caches(monkeypatch):
    monkeypatch.setattr(titles, "lookup_online", lambda *a, **k: "From Web")
    cache = {}
    title, source = titles.resolve_title("999", do_lookup=True, cache=cache)
    assert title == "From Web" and source == "lookup"
    assert cache["999"] == "From Web"


def test_resolve_returns_empty_when_nothing(monkeypatch):
    monkeypatch.setattr(titles, "lookup_online", lambda *a, **k: "")
    title, source = titles.resolve_title("", do_lookup=True, cache={})
    assert title == "" and source == "none"
