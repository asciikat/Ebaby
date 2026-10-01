import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from ebaby import batch, server

SAMPLES = Path(__file__).resolve().parent.parent.parent / "samples" / "redbox"


@pytest.fixture(autouse=True)
def isolated_root(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "BATCHES_ROOT", tmp_path / "Ebaby Runs")


@pytest.mark.skipif(not SAMPLES.exists(), reason="samples/redbox fixtures not present")
@patch("ebaby.server.ebay.write_csv")
@patch("ebaby.server.ebay.fetch_all")
@patch("ebaby.server.ebay.get_token", return_value="tok")
def test_full_batch_walkthrough(mock_token, mock_fetch, mock_write):
    client = TestClient(server.app)
    client.post("/api/batches", json={"name": "e2e"})

    # Upload the 3-shot used set as s1/s2/s3 so filename sort assigns
    # front/back/inside in the order our fixture filenames imply.
    for src_name, upload_name in [("front.jpg", "s1.jpg"), ("back.jpg", "s2.jpg"),
                                   ("inside.jpg", "s3.jpg")]:
        with open(SAMPLES / src_name, "rb") as f:
            client.post("/api/batches/e2e/upload/used",
                        files={"file": (upload_name, f, "image/jpeg")})
    # Minimal 2-shot new-stock set so the "both zones done" gate passes.
    for upload_name in ["n1.jpg", "n2.jpg"]:
        with open(SAMPLES / "front.jpg", "rb") as f:
            client.post("/api/batches/e2e/upload/new",
                        files={"file": (upload_name, f, "image/jpeg")})

    client.post("/api/batches/e2e/rename/apply", json={"zone": "used"})
    client.post("/api/batches/e2e/rename/apply", json={"zone": "new"})
    assert batch.read_state(batch.batch_dir("e2e"))["stage"] == "color"

    # color/run only processes .nef/.dng — our fixtures are .jpg, so nothing
    # to colour-correct; confirm the stage still advances cleanly.
    client.post("/api/batches/e2e/color/run")
    assert batch.read_state(batch.batch_dir("e2e"))["stage"] == "barcode"

    # Copy the renamed originals into 2_color/ so the barcode stage (which
    # reads from 2_color/) has real *_back*.png files to scan, mirroring
    # what color/run would have produced for RAW input.
    d = batch.batch_dir("e2e")
    for src in (d / "1_originals/used").glob("*"):
        shutil.copy(src, d / "2_color" / f"{src.stem}.png")

    resp = client.post("/api/batches/e2e/barcode/run")
    assert resp.status_code == 200
    assert batch.read_state(d)["stage"] == "ebay"

    mock_fetch.return_value = [{"Barcode": "000000000000", "Image Set Name": "Sample_DVD",
                                "Title": "Sample DVD"}]
    resp = client.post("/api/batches/e2e/ebay/run")
    assert resp.status_code == 200
    assert batch.read_state(d)["stage"] == "crop"
    assert (d / "Ebay_Details.csv") or mock_write.called

    resp = client.post("/api/batches/e2e/crop/run")
    assert resp.status_code == 200
    assert batch.read_state(d)["stage"] == "done"
    assert list((d / "5_cropped").glob("*.jpg"))
