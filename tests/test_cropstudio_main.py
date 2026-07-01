import cropstudio.__main__ as m


def test_build_app_returns_fastapi(tmp_path, monkeypatch):
    monkeypatch.setenv("CROPSTUDIO_OUTPUT", str(tmp_path))
    app = m.build_app()
    routes = {r.path for r in app.routes}
    assert "/shot" in routes and "/health" in routes
