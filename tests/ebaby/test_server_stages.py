import io
from unittest.mock import patch

import cv2
import numpy as np
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


def test_color_run_passes_plain_jpg_through_to_2_color(client, tmp_path):
    """A JPG-only batch must not sail through the pipeline with zero output —
    every later stage reads from 2_color."""
    d = _make_batch_at_stage(client, tmp_path, "color")
    img = np.full((40, 30, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(d / "1_originals/used/a_front.jpg"), img)
    resp = client.post("/api/batches/run1/color/run")
    assert resp.status_code == 200
    assert resp.json()["colored"] == 1
    assert (d / "2_color/a_front.png").exists()


def test_color_run_on_wrong_stage_is_a_clean_409(client, tmp_path):
    _make_batch_at_stage(client, tmp_path, "upload")
    resp = client.post("/api/batches/run1/color/run")
    assert resp.status_code == 409
    assert "stage" in resp.json()["error"]


def test_rename_apply_is_idempotent_no_front_back_swap(client, tmp_path):
    """Re-running rename on a renamed zone would re-group alphabetically and
    swap front/back (a_back sorts before a_front). Second call must no-op."""
    client.post("/api/batches", json={"name": "run1"})
    for n in ("s1.png", "s2.png", "s3.png"):
        client.post("/api/batches/run1/upload/used",
                    files={"file": (n, io.BytesIO(b"x"), "image/png")})
    first = client.post("/api/batches/run1/rename/apply", json={"zone": "used"}).json()
    assert first["renamed"] == 3
    d = batch.batch_dir("run1")
    assert (d / "1_originals/used/a_front.png").exists()
    second = client.post("/api/batches/run1/rename/apply", json={"zone": "used"}).json()
    assert second.get("already_done") is True
    assert second["renamed"] == 0
    assert (d / "1_originals/used/a_front.png").exists()  # not swapped


def test_upload_strips_path_components_and_dedupes(client, tmp_path):
    client.post("/api/batches", json={"name": "run1"})
    r = client.post("/api/batches/run1/upload/used",
                    files={"file": ("..\\..\\evil.png", io.BytesIO(b"a"), "image/png")})
    assert r.status_code == 200
    d = batch.batch_dir("run1")
    inside = list((d / "1_originals/used").glob("*.png"))
    assert len(inside) == 1 and ".." not in inside[0].name
    # duplicate camera filename must not overwrite the first upload
    client.post("/api/batches/run1/upload/used",
                files={"file": ("IMG_1.png", io.BytesIO(b"one"), "image/png")})
    client.post("/api/batches/run1/upload/used",
                files={"file": ("IMG_1.png", io.BytesIO(b"two"), "image/png")})
    names = sorted(p.name for p in (d / "1_originals/used").glob("IMG_1*"))
    assert names == ["IMG_1.png", "IMG_1_2.png"]


def test_barcode_manual_sanitizes_digits_and_rejects_junk(client, tmp_path):
    d = _make_batch_at_stage(client, tmp_path, "barcode")
    ok = client.post("/api/batches/run1/barcode/manual",
                     json={"key": "a", "digits": " 4006-3813 3393 "})
    assert ok.status_code == 200
    assert batch.read_state(d)["barcodes"]["a"] == "400638133393"
    bad = client.post("/api/batches/run1/barcode/manual",
                      json={"key": "a", "digits": "not/a/barcode"})
    assert bad.status_code == 422


def test_crop_run_on_unfinished_batch_is_409_not_empty_success(client, tmp_path):
    _make_batch_at_stage(client, tmp_path, "upload")
    resp = client.post("/api/batches/run1/crop/run")
    assert resp.status_code == 409


def test_crop_manual_degenerate_quad_is_422(client, tmp_path):
    d = _make_batch_at_stage(client, tmp_path, "crop")
    (d / "4_renamed").mkdir(parents=True, exist_ok=True)
    img = np.full((60, 40, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(d / "4_renamed/Movie_front.png"), img)
    resp = client.post("/api/batches/run1/crop/manual",
                       json={"filename": "Movie_front.png",
                             "quad": [[0, 0], [1, 0], [1, 1], [0, 1]]})
    assert resp.status_code == 422
    missing = client.post("/api/batches/run1/crop/manual",
                          json={"filename": "nope.png",
                                "quad": [[0, 0], [30, 0], [30, 50], [0, 50]]})
    assert missing.status_code == 404


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


@patch("ebaby.server.ebay.get_token",
       side_effect=RuntimeError("eBay credentials missing"))
def test_ebay_run_without_creds_still_advances_with_barcode_names(mock_token, client, tmp_path):
    d = _make_batch_at_stage(client, tmp_path, "ebay")
    state = batch.read_state(d)
    state["barcodes"] = {"a": "400638133393"}
    batch.write_state(d, state)
    (d / "2_color").mkdir(parents=True, exist_ok=True)
    (d / "2_color/a_front.png").write_bytes(b"x")
    resp = client.post("/api/batches/run1/ebay/run")
    assert resp.status_code == 200
    body = resp.json()
    assert "eBay lookup skipped" in body["warning"]
    assert batch.read_state(d)["stage"] == "crop"
    # fell back to barcode-digit naming, file still moved to 4_renamed
    assert (d / "4_renamed/400638133393_front.png").exists()


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
    d = _make_batch_at_stage(client, tmp_path, "crop")
    (d / "4_renamed").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "4_renamed/Matrix_front.png"),
                np.full((50, 40, 3), 90, dtype=np.uint8))
    mock_detect.return_value = (np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype="float32"), 0.5)
    mock_compose.return_value = np.zeros((10, 10, 3), dtype="uint8")
    resp = client.post("/api/batches/run1/crop/run")
    assert resp.status_code == 200
    body = resp.json()
    assert "Matrix_front.png" in body["quads"]
    assert batch.read_state(d)["stage"] == "done"


@patch("ebaby.server.crop.crop_and_compose")
@patch("ebaby.server.crop.detect_crop_box")
def test_crop_run_persists_quads_and_progress_in_state(mock_detect, mock_compose, client, tmp_path):
    d = _make_batch_at_stage(client, tmp_path, "crop")
    (d / "4_renamed").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "4_renamed/Movie_front.png"),
                np.full((50, 40, 3), 90, dtype=np.uint8))
    mock_detect.return_value = (np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype="float32"), 0.5)
    mock_compose.return_value = np.zeros((10, 10, 3), dtype="uint8")
    client.post("/api/batches/run1/crop/run")
    s = batch.read_state(d)
    assert "Movie_front.png" in s["quads"]          # refresh-resume source
    assert s["progress"] == {"stage": "crop", "done": 1, "total": 1}


@patch("ebaby.server.color.color_correct_file")
def test_color_run_writes_progress_counter(mock_cc, client, tmp_path):
    d = _make_batch_at_stage(client, tmp_path, "color")
    (d / "1_originals/used/a_front.dng").write_bytes(b"x")
    (d / "1_originals/used/a_back.dng").write_bytes(b"x")
    client.post("/api/batches/run1/color/run")
    s = batch.read_state(d)
    assert s["progress"] == {"stage": "color", "done": 2, "total": 2}


@patch("ebaby.server.ebay.write_csv")
@patch("ebaby.server.ebay.fetch_all")
@patch("ebaby.server.ebay.get_token", return_value="tok")
def test_ebay_run_persists_rows_for_resume(mock_token, mock_fetch, mock_write, client, tmp_path):
    d = _make_batch_at_stage(client, tmp_path, "ebay")
    state = batch.read_state(d)
    state["barcodes"] = {"a": "400638133393"}
    batch.write_state(d, state)
    mock_fetch.return_value = [{"Barcode": "400638133393", "Image Set Name": "Matrix",
                                "Title": "The Matrix"}]
    client.post("/api/batches/run1/ebay/run")
    s = batch.read_state(d)
    assert s["ebay_rows"][0]["Title"] == "The Matrix"
    assert s["listing_txt"].endswith("Ebaby Listings.txt")


@patch("ebaby.server.color.color_correct_file")
def test_color_run_raw_plus_jpeg_pair_prefers_raw(mock_cc, client, tmp_path):
    """RAW+JPEG cameras save both per shot with the SAME stem. The plain JPEG
    must not clobber the colour-corrected RAW conversion of the same shot."""
    d = _make_batch_at_stage(client, tmp_path, "color")
    (d / "1_originals/used/a_front.dng").write_bytes(b"raw")
    cv2.imwrite(str(d / "1_originals/used/a_front.jpg"),
                np.full((30, 20, 3), 90, dtype=np.uint8))
    resp = client.post("/api/batches/run1/color/run")
    assert resp.status_code == 200
    assert resp.json()["colored"] == 1          # one shot, not two
    assert mock_cc.call_count == 1              # and it was the RAW that won
    assert mock_cc.call_args[0][0].suffix == ".dng"


@patch("ebaby.server.color.color_correct_file")
def test_color_run_shares_one_white_balance_across_a_set(mock_cc, client, tmp_path):
    """front/back/inside of one disc must be colour-corrected with the SAME
    white balance (first shot computes it, the rest reuse it)."""
    mock_cc.return_value = (1.1, 1.0, 0.9)  # the gains the first shot "computed"
    d = _make_batch_at_stage(client, tmp_path, "color")
    for role in ("front", "back", "inside"):
        (d / f"1_originals/used/a_{role}.dng").write_bytes(b"x")
    resp = client.post("/api/batches/run1/color/run")
    assert resp.status_code == 200
    assert resp.json()["colored"] == 3
    passed = [c.kwargs.get("gains") for c in mock_cc.call_args_list]
    assert passed[0] is None                 # first shot: compute fresh
    assert passed[1] == (1.1, 1.0, 0.9)      # rest: reuse the set's balance
    assert passed[2] == (1.1, 1.0, 0.9)


@patch("ebaby.server.ebay.fetch_all")
@patch("ebaby.server.ebay.get_token", return_value="tok")
def test_ebay_run_keeps_only_the_relevant_condition_price(mock_token, mock_fetch, client, tmp_path):
    """A disc uploaded to the USED zone (letter key) must not carry the
    new-copy price around — the user asked to only see the relevant one."""
    d = _make_batch_at_stage(client, tmp_path, "ebay")
    state = batch.read_state(d)
    state["barcodes"] = {"a": "400638133393"}          # letter key => USED
    batch.write_state(d, state)
    mock_fetch.return_value = [{
        "Barcode": "400638133393", "Image Set Name": "Matrix", "Title": "The Matrix",
        "Lowest Price New (AUD)": 40.0, "Lowest Price Used (AUD)": 25.0,
        "Your Price New (AUD)": 35.99, "Your Price Used (AUD)": 22.5,
    }]
    client.post("/api/batches/run1/ebay/run")
    row = batch.read_state(d)["ebay_rows"][0]
    assert row["Stock"] == "used"
    assert row["Lowest Price Used (AUD)"] == 25.0 and row["Your Price Used (AUD)"] == 22.5
    assert row["Lowest Price New (AUD)"] == "" and row["Your Price New (AUD)"] == ""
    txt = (d / "Ebaby Listings.txt").read_text(encoding="utf-8")
    assert "list USED at" in txt and "list NEW" not in txt


@patch("ebaby.server.ebay.write_csv")
@patch("ebaby.server.ebay.fetch_all")
@patch("ebaby.server.ebay.get_token", return_value="tok")
def test_listing_txt_totals_only_the_ten_percent_prices(mock_token, mock_fetch, mock_write, client, tmp_path):
    """The overall total sums the suggested (10%-under) prices across discs —
    NOT the lowest-comp prices."""
    d = _make_batch_at_stage(client, tmp_path, "ebay")
    state = batch.read_state(d)
    state["barcodes"] = {"a": "111", "2": "222"}   # 'a' => used, '2' => new
    batch.write_state(d, state)
    mock_fetch.return_value = [
        {"Barcode": "111", "Title": "A", "Lowest Price Used (AUD)": 25.0,
         "Your Price Used (AUD)": 22.99, "Lowest Price New (AUD)": 40.0,
         "Your Price New (AUD)": 35.99},
        {"Barcode": "222", "Title": "B", "Lowest Price New (AUD)": 12.0,
         "Your Price New (AUD)": 9.99, "Lowest Price Used (AUD)": 7.0,
         "Your Price Used (AUD)": 5.99},
    ]
    client.post("/api/batches/run1/ebay/run")
    content = (d / "Ebaby Listings.txt").read_text(encoding="utf-8")
    # used 22.99 (disc A) + new 9.99 (disc B) = 32.98 — the relevant ones only
    assert "TOTAL TAKE" in content
    assert "$32.98" in content
