import io

import pytest
from fastapi.testclient import TestClient

from ebaby import batch, server


@pytest.fixture(autouse=True)
def isolated_root(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "BATCHES_ROOT", tmp_path / "Ebaby Runs")
    return tmp_path


@pytest.fixture
def client():
    return TestClient(server.app)


def test_create_and_list_batches(client):
    resp = client.post("/api/batches", json={"name": "run1"})
    assert resp.status_code == 200
    resp = client.get("/api/batches")
    assert resp.json() == ["run1"]


def test_create_duplicate_batch_returns_409(client):
    client.post("/api/batches", json={"name": "run1"})
    resp = client.post("/api/batches", json={"name": "run1"})
    assert resp.status_code == 409


def test_get_batch_state(client):
    client.post("/api/batches", json={"name": "run1"})
    resp = client.get("/api/batches/run1/state")
    assert resp.status_code == 200
    assert resp.json()["stage"] == "upload"


def test_upload_and_rename_plan_used_zone(client, tmp_path):
    client.post("/api/batches", json={"name": "run1"})
    for name in ("s1.png", "s2.png", "s3.png"):
        client.post(
            "/api/batches/run1/upload/used",
            files={"file": (name, io.BytesIO(b"x"), "image/png")},
        )
    resp = client.post("/api/batches/run1/rename/plan", json={"zone": "used"})
    assert resp.status_code == 200
    plan = resp.json()["plan"]
    assert len(plan) == 3
    assert any(p["new_name"] == "a_front.png" for p in plan)


def test_rename_plan_uneven_count_returns_422(client):
    client.post("/api/batches", json={"name": "run1"})
    client.post("/api/batches/run1/upload/used",
                files={"file": ("s1.png", io.BytesIO(b"x"), "image/png")})
    resp = client.post("/api/batches/run1/rename/plan", json={"zone": "used"})
    assert resp.status_code == 422


def test_rename_apply_advances_state_once_both_zones_done(client):
    client.post("/api/batches", json={"name": "run1"})
    for n in ("s1.png", "s2.png", "s3.png"):
        client.post("/api/batches/run1/upload/used",
                    files={"file": (n, io.BytesIO(b"x"), "image/png")})
    for n in ("n1.png", "n2.png"):
        client.post("/api/batches/run1/upload/new",
                    files={"file": (n, io.BytesIO(b"x"), "image/png")})

    client.post("/api/batches/run1/rename/apply", json={"zone": "used"})
    state = client.get("/api/batches/run1/state").json()
    assert state["stage"] == "upload"  # only one of two zones done so far

    client.post("/api/batches/run1/rename/apply", json={"zone": "new"})
    state = client.get("/api/batches/run1/state").json()
    assert state["stage"] == "color"  # both zones renamed -> advance
