import io
import shutil
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
    (d / "2_color").mkdir(parents=True, exist_ok=True)
    # a *readable* back photo: decode only runs when cv2.imread succeeds, so a
    # valid image is what actually exercises the record-and-advance path
    cv2.imwrite(str(d / "2_color/a_back.png"), np.full((20, 20, 3), 255, dtype=np.uint8))
    resp = client.post("/api/batches/run1/barcode/run")
    assert resp.status_code == 200
    body = resp.json()
    assert body["results"]["a"] == "400638133393"
    assert batch.read_state(d)["stage"] == "ebay"


@patch("ebaby.server.barcode_locate.locate_and_crop", return_value=True)
def test_barcode_run_unreadable_photo_is_a_miss_not_a_batch_crash(mock_locate, client, tmp_path):
    """locate_and_crop can report success while both the crop and the back
    photo are unreadable (cv2.imread -> None). That one item must record as a
    miss and the batch must still advance — it used to crash the whole run and
    lose every result. See test_barcode_decode.test_..._none_...as well."""
    d = _make_batch_at_stage(client, tmp_path, "barcode")
    (d / "2_color").mkdir(parents=True, exist_ok=True)
    (d / "2_color/a_back.png").write_bytes(b"not a real png")  # imread -> None
    resp = client.post("/api/batches/run1/barcode/run")
    assert resp.status_code == 200
    assert resp.json()["results"]["a"] is None      # clean miss
    assert batch.read_state(d)["stage"] == "ebay"    # batch still advanced


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
    # rename empties 2_color — the now-dead directory must not be left behind
    assert not (d / "2_color").exists()


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
    """A disc uploaded to the USED zone (letter key) must collapse to ONE
    Condition/price pair — the user asked to only see the relevant one, with
    no dead New-condition columns left blank."""
    d = _make_batch_at_stage(client, tmp_path, "ebay")
    state = batch.read_state(d)
    state["barcodes"] = {"a": "400638133393"}          # letter key => USED
    batch.write_state(d, state)
    mock_fetch.return_value = [{
        "Barcode": "400638133393", "Image Set Name": "Matrix", "Title": "The Matrix",
        "_lowest_new": 40.0, "_lowest_used": 25.0,
    }]
    client.post("/api/batches/run1/ebay/run")
    row = batch.read_state(d)["ebay_rows"][0]
    assert row["Stock"] == "used"
    assert row["Condition"] == "Used"
    assert row["Lowest Price (AUD)"] == 25.0
    assert row["Your Price (AUD)"] == 22.99   # 25 * 0.9 = 22.50 -> charm .99
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
        {"Barcode": "111", "Title": "A", "_lowest_used": 25.0, "_lowest_new": 40.0},
        {"Barcode": "222", "Title": "B", "_lowest_new": 12.0, "_lowest_used": 7.0},
    ]
    client.post("/api/batches/run1/ebay/run")
    content = (d / "Ebaby Listings.txt").read_text(encoding="utf-8")
    # used 25*0.9->22.99 (disc A) + new 12*0.9->10.99 (disc B) = 33.98
    assert "TOTAL TAKE" in content
    assert "$33.98" in content


def _seed_edit_batch(client, d):
    """A 'crop'-stage batch with one used disc (wrong eBay title) whose renamed
    + cropped photos and quads exist on disk, ready for a title/edit."""
    state = batch.read_state(d)
    state["stage"] = "crop"
    state["barcodes"] = {"a": "9327478001218"}
    state["ebay_rows"] = [{
        "Barcode": "9327478001218", "Image Set Name": "Intruder_Dvd_2009",
        "Title": "Intruder (DVD, 2009)", "Condition": "Used",
        "Lowest Price (AUD)": 14.95, "Your Price (AUD)": 12.99, "Stock": "used",
    }]
    state["quads"] = {
        "Intruder_Dvd_2009_front.png": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "Intruder_Dvd_2009_back.png": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "Intruder_Dvd_2009_inside.png": [[0, 0], [1, 0], [1, 1], [0, 1]],
    }
    batch.write_state(d, state)
    for role in ("front", "back", "inside"):
        (d / "4_renamed").mkdir(parents=True, exist_ok=True)
        (d / "5_cropped").mkdir(parents=True, exist_ok=True)
        (d / "4_renamed" / f"Intruder_Dvd_2009_{role}.png").write_bytes(b"png")
        (d / "5_cropped" / f"Intruder_Dvd_2009_{role}.jpg").write_bytes(b"jpg")
    return d


def test_title_edit_reslugs_renames_photos_and_rewrites_files(client, tmp_path):
    d = _make_batch_at_stage(client, tmp_path, "crop")
    _seed_edit_batch(client, d)
    resp = client.post("/api/batches/run1/title/edit", json={
        "image_set_name": "Intruder_Dvd_2009", "title": "One Step Beyond", "price": "9.99"})
    assert resp.status_code == 200
    row = resp.json()["rows"][0]
    assert row["Title"] == "One Step Beyond"
    assert row["Image Set Name"] == "One_Step_Beyond"
    assert row["Your Price (AUD)"] == 9.99
    # photos renamed in BOTH stage folders, old names gone
    for role in ("front", "back", "inside"):
        assert (d / "4_renamed" / f"One_Step_Beyond_{role}.png").is_file()
        assert (d / "5_cropped" / f"One_Step_Beyond_{role}.jpg").is_file()
        assert not (d / "4_renamed" / f"Intruder_Dvd_2009_{role}.png").exists()
    # quads remapped to the new filenames
    quads = batch.read_state(d)["quads"]
    assert "One_Step_Beyond_front.png" in quads
    assert "Intruder_Dvd_2009_front.png" not in quads
    # CSV + listing rewritten with the corrected title
    csv_text = (d / "Ebay_Details.csv").read_text(encoding="utf-8-sig")
    assert "One Step Beyond" in csv_text and "Intruder" not in csv_text
    assert "One Step Beyond" in (d / "Ebaby Listings.txt").read_text(encoding="utf-8")


def test_title_edit_price_only_keeps_the_slug_and_files(client, tmp_path):
    d = _make_batch_at_stage(client, tmp_path, "crop")
    _seed_edit_batch(client, d)
    # same title (slug unchanged) — only the price moves; files must NOT churn
    resp = client.post("/api/batches/run1/title/edit", json={
        "image_set_name": "Intruder_Dvd_2009", "title": "Intruder (DVD, 2009)", "price": "5"})
    assert resp.status_code == 200
    assert resp.json()["rows"][0]["Your Price (AUD)"] == 5.0
    assert (d / "4_renamed" / "Intruder_Dvd_2009_front.png").is_file()  # untouched


def test_title_edit_rejects_empty_title_and_bad_price(client, tmp_path):
    d = _make_batch_at_stage(client, tmp_path, "crop")
    _seed_edit_batch(client, d)
    assert client.post("/api/batches/run1/title/edit", json={
        "image_set_name": "Intruder_Dvd_2009", "title": "   "}).status_code == 422
    assert client.post("/api/batches/run1/title/edit", json={
        "image_set_name": "Intruder_Dvd_2009", "title": "Fine", "price": "cheap"}).status_code == 422
    # nothing should have been renamed on a rejected edit
    assert (d / "4_renamed" / "Intruder_Dvd_2009_front.png").is_file()


def test_title_edit_unknown_disc_is_404(client, tmp_path):
    d = _make_batch_at_stage(client, tmp_path, "crop")
    _seed_edit_batch(client, d)
    assert client.post("/api/batches/run1/title/edit", json={
        "image_set_name": "Nope", "title": "Whatever"}).status_code == 404


def test_title_edit_dedupes_against_another_disc(client, tmp_path):
    d = _make_batch_at_stage(client, tmp_path, "crop")
    _seed_edit_batch(client, d)
    state = batch.read_state(d)
    # a second disc already occupies the slug we're about to collide with
    state["ebay_rows"].append({
        "Barcode": "111", "Image Set Name": "One_Step_Beyond",
        "Title": "One Step Beyond", "Condition": "Used",
        "Lowest Price (AUD)": 8.0, "Your Price (AUD)": 6.99, "Stock": "used"})
    batch.write_state(d, state)
    resp = client.post("/api/batches/run1/title/edit", json={
        "image_set_name": "Intruder_Dvd_2009", "title": "One Step Beyond"})
    assert resp.status_code == 200
    edited = next(r for r in resp.json()["rows"] if r["Barcode"] == "9327478001218")
    assert edited["Image Set Name"] == "One_Step_Beyond_2"  # deduped, not clobbered


def _seed_done_batch_with_all_folders(client, tmp_path):
    d = _make_batch_at_stage(client, tmp_path, "done")
    for sub in ("1_originals/used", "3_barcodes", "4_renamed", "2_color", "5_cropped"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    (d / "1_originals/used/a_front.dng").write_bytes(b"raw")
    (d / "3_barcodes/a_back_barcode.png").write_bytes(b"crop")
    (d / "4_renamed/Movie_front.png").write_bytes(b"png")
    (d / "5_cropped/Movie_front.jpg").write_bytes(b"jpg")
    (d / "Ebay_Details.csv").write_text("Barcode\n123\n", encoding="utf-8")
    (d / "Ebaby Listings.txt").write_text("stuff", encoding="utf-8")
    return d


def test_accept_deletes_intermediates_keeps_cropped_and_paperwork(client, tmp_path):
    d = _seed_done_batch_with_all_folders(client, tmp_path)
    resp = client.post("/api/batches/run1/accept")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["removed"]) == {"1_originals", "3_barcodes", "4_renamed", "2_color"}
    assert not (d / "1_originals").exists()
    assert not (d / "3_barcodes").exists()
    assert not (d / "4_renamed").exists()
    assert not (d / "2_color").exists()
    # the stuff we're supposed to keep survives untouched
    assert (d / "5_cropped/Movie_front.jpg").is_file()
    assert (d / "Ebay_Details.csv").is_file()
    assert (d / "Ebaby Listings.txt").is_file()
    assert (d / "state.json").is_file()


def test_accept_is_idempotent_when_folders_already_gone(client, tmp_path):
    d = _seed_done_batch_with_all_folders(client, tmp_path)
    first = client.post("/api/batches/run1/accept")
    assert first.status_code == 200
    second = client.post("/api/batches/run1/accept")
    assert second.status_code == 200
    assert second.json()["removed"] == []          # nothing left to remove
    assert (d / "5_cropped/Movie_front.jpg").is_file()  # still untouched


def test_accept_refuses_a_batch_that_isnt_done_yet(client, tmp_path):
    d = _make_batch_at_stage(client, tmp_path, "crop")
    (d / "1_originals/used").mkdir(parents=True, exist_ok=True)
    (d / "1_originals/used/a_front.dng").write_bytes(b"raw")
    resp = client.post("/api/batches/run1/accept")
    assert resp.status_code == 409
    assert (d / "1_originals/used/a_front.dng").is_file()  # nothing deleted


def test_accept_404s_on_an_unknown_batch(client, tmp_path):
    resp = client.post("/api/batches/ghost/accept")
    assert resp.status_code == 404


def test_accept_sets_the_accepted_flag_in_state(client, tmp_path):
    d = _seed_done_batch_with_all_folders(client, tmp_path)
    client.post("/api/batches/run1/accept")
    assert batch.read_state(d)["accepted"] is True


def test_accept_reports_a_locked_file_but_still_removes_the_rest(client, tmp_path, monkeypatch):
    """A locked file (Windows AV/Explorer holding a handle) must not abort
    cleanup of the OTHER folders, and must surface a clear error instead of
    an unhandled 500 — a partial failure used to be silently swallowed."""
    d = _seed_done_batch_with_all_folders(client, tmp_path)
    real_rmtree = shutil.rmtree
    def flaky_rmtree(path, *a, **kw):
        if "3_barcodes" in str(path):
            raise PermissionError("Access is denied")
        return real_rmtree(path, *a, **kw)
    monkeypatch.setattr(server.shutil, "rmtree", flaky_rmtree)
    resp = client.post("/api/batches/run1/accept")
    assert resp.status_code == 200                 # not an unhandled 500
    body = resp.json()
    assert "3_barcodes" in body["error"]
    assert "1_originals" in body["removed"]         # the other dirs still went
    assert not (d / "1_originals").exists()
    assert (d / "3_barcodes").exists()              # the locked one survives
    assert batch.read_state(d)["accepted"] is True  # still marked finalized


def test_accept_retry_finishes_a_partially_failed_cleanup(client, tmp_path, monkeypatch):
    d = _seed_done_batch_with_all_folders(client, tmp_path)
    real_rmtree = shutil.rmtree
    def flaky_once(path, *a, **kw):
        if "3_barcodes" in str(path):
            raise PermissionError("Access is denied")
        return real_rmtree(path, *a, **kw)
    monkeypatch.setattr(server.shutil, "rmtree", flaky_once)
    client.post("/api/batches/run1/accept")
    assert (d / "3_barcodes").exists()
    monkeypatch.setattr(server.shutil, "rmtree", real_rmtree)  # "file" is closed now
    retry = client.post("/api/batches/run1/accept")
    assert retry.status_code == 200
    assert retry.json()["removed"] == ["3_barcodes"]
    assert not (d / "3_barcodes").exists()


def test_crop_run_refuses_once_batch_is_accepted(client, tmp_path):
    d = _seed_done_batch_with_all_folders(client, tmp_path)
    client.post("/api/batches/run1/accept")
    resp = client.post("/api/batches/run1/crop/run")
    assert resp.status_code == 409
    assert "Accept Hustle" in resp.json()["error"]


def test_crop_manual_refuses_once_batch_is_accepted(client, tmp_path):
    d = _seed_done_batch_with_all_folders(client, tmp_path)
    client.post("/api/batches/run1/accept")
    resp = client.post("/api/batches/run1/crop/manual", json={
        "filename": "Movie_front.png", "quad": [[0, 0], [1, 0], [1, 1], [0, 1]]})
    assert resp.status_code == 409
    assert "Accept Hustle" in resp.json()["error"]


def test_title_edit_refuses_once_batch_is_accepted(client, tmp_path):
    d = _make_batch_at_stage(client, tmp_path, "crop")
    _seed_edit_batch(client, d)
    state = batch.read_state(d)
    state["stage"] = "done"
    state["accepted"] = True
    batch.write_state(d, state)
    resp = client.post("/api/batches/run1/title/edit", json={
        "image_set_name": "Intruder_Dvd_2009", "title": "One Step Beyond"})
    assert resp.status_code == 409
    assert "Accept Hustle" in resp.json()["error"]


def test_batch_lock_returns_the_same_lock_object_for_the_same_name(tmp_path):
    """The whole point of _batch_lock: two calls for the same batch name must
    serialize on ONE lock, not two different ones."""
    assert server._batch_lock("run1") is server._batch_lock("run1")
    assert server._batch_lock("run1") is not server._batch_lock("run2")


def test_quit_clear_resets_a_stale_flag(client):
    """A reused server can hold quit=True from a previous Accept Hustle —
    the wrapper clears it at launch so its window doesn't close instantly."""
    try:
        client.post("/api/quit")
        assert client.get("/api/quit-status").json()["quit"] is True
        assert client.post("/api/quit/clear").json()["ok"] is True
        assert client.get("/api/quit-status").json()["quit"] is False
    finally:
        server._QUIT_REQUESTED.clear()


def test_quit_status_starts_false_and_flips_after_request(client):
    # _QUIT_REQUESTED is a module-level singleton (mirrors the real desktop
    # app's lifetime) — clear it after so this test can't leak into others.
    assert not server._QUIT_REQUESTED.is_set()
    try:
        assert client.get("/api/quit-status").json()["quit"] is False
        assert client.post("/api/quit").json()["ok"] is True
        assert client.get("/api/quit-status").json()["quit"] is True
    finally:
        server._QUIT_REQUESTED.clear()
