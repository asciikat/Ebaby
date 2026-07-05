from unittest.mock import patch

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


def _make_batch_at_stage(client, tmp_path, stage):
    client.post("/api/batches", json={"name": "run1"})
    d = batch.batch_dir("run1")
    state = batch.read_state(d)
    state["stage"] = stage
    batch.write_state(d, state)
    return d


@patch("ebaby.server.color.color_correct_file")
def test_color_run_advances_to_barcode(mock_cc, client, tmp_path):
    d = _make_batch_at_stage(client, tmp_path, "color")
    (d / "1_originals/used/a_front.dng").write_bytes(b"x")
    resp = client.post("/api/batches/run1/color/run")
    assert resp.status_code == 200
    assert mock_cc.called
    assert batch.read_state(d)["stage"] == "barcode"


@patch("ebaby.server.barcode_decode.decode_barcodes", return_value=["400638133393"])
@patch("ebaby.server.barcode_locate.locate_and_crop", return_value=True)
def test_barcode_run_records_digits_and_advances(mock_locate, mock_decode, client, tmp_path):
    d = _make_batch_at_stage(client, tmp_path, "barcode")
    (d / "2_color/a_back.png").write_bytes(b"x")
    resp = client.post("/api/batches/run1/barcode/run")
    assert resp.status_code == 200
    body = resp.json()
    assert body["results"]["a"] == "400638133393"
    assert batch.read_state(d)["stage"] == "ebay"


def test_barcode_manual_override_records_digits(client, tmp_path):
    d = _make_batch_at_stage(client, tmp_path, "barcode")
    resp = client.post("/api/batches/run1/barcode/manual",
                       json={"key": "a", "digits": "400638133393"})
    assert resp.status_code == 200
    state = batch.read_state(d)
    assert state["barcodes"]["a"] == "400638133393"


@patch("ebaby.server.ebay.write_csv")
@patch("ebaby.server.ebay.fetch_all")
@patch("ebaby.server.ebay.get_token", return_value="tok")
def test_ebay_run_writes_csv_and_advances(mock_token, mock_fetch, mock_write, client, tmp_path):
    d = _make_batch_at_stage(client, tmp_path, "ebay")
    state = batch.read_state(d)
    state["barcodes"] = {"a": "400638133393"}
    batch.write_state(d, state)
    mock_fetch.return_value = [{"Barcode": "400638133393", "Image Set Name": "Matrix",
                                "Title": "The Matrix"}]
    resp = client.post("/api/batches/run1/ebay/run")
    assert resp.status_code == 200
    assert batch.read_state(d)["stage"] == "crop"  # rename_title auto-applies then advances
    assert mock_write.called


@patch("ebaby.server.ebay.write_csv")
@patch("ebaby.server.ebay.fetch_all")
@patch("ebaby.server.ebay.get_token", return_value="tok")
def test_ebay_run_writes_listing_txt(mock_token, mock_fetch, mock_write, client, tmp_path):
    d = _make_batch_at_stage(client, tmp_path, "ebay")
    state = batch.read_state(d)
    state["barcodes"] = {"a": "400638133393"}
    batch.write_state(d, state)
    mock_fetch.return_value = [{"Barcode": "400638133393", "Image Set Name": "Matrix",
                                "Title": "The Matrix"}]
    body = client.post("/api/batches/run1/ebay/run").json()
    txt = d / "Ebaby Listings.txt"
    assert txt.exists()
    content = txt.read_text(encoding="utf-8")
    assert "400638133393" in content and "The Matrix" in content
    assert body["listing_txt"].endswith("Ebaby Listings.txt")


def test_serve_file_returns_batch_image_and_404s_outside(client, tmp_path):
    d = _make_batch_at_stage(client, tmp_path, "crop")
    (d / "4_renamed").mkdir(parents=True, exist_ok=True)
    (d / "4_renamed/Matrix_front.png").write_bytes(b"pngbytes")
    ok = client.get("/api/files/run1/4_renamed/Matrix_front.png")
    assert ok.status_code == 200
    assert ok.content == b"pngbytes"
    missing = client.get("/api/files/run1/4_renamed/nope.png")
    assert missing.status_code == 404


@patch("ebaby.server.crop.crop_and_compose")
@patch("ebaby.server.crop.detect_crop_box")
def test_crop_run_returns_seeded_quads_per_set(mock_detect, mock_compose, client, tmp_path):
    import numpy as np
    d = _make_batch_at_stage(client, tmp_path, "crop")
    (d / "4_renamed/Matrix_front.png").write_bytes(b"x")
    mock_detect.return_value = (np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype="float32"), 0.5)
    mock_compose.return_value = np.zeros((10, 10, 3), dtype="uint8")
    resp = client.post("/api/batches/run1/crop/run")
    assert resp.status_code == 200
    body = resp.json()
    assert "Matrix_front.png" in body["quads"]
    assert batch.read_state(d)["stage"] == "done"
