from pathlib import Path
import shutil
from fastapi.testclient import TestClient

from dvdflip import webapp, pipeline
from dvdflip.models import ClassifyResult

FIXED = {
    "front": ClassifyResult("front", 0, "King Kong Escapes", 1967, 0.95),
    "back": ClassifyResult("back", 0, "King Kong Escapes", 1967, 0.96),
    "center": ClassifyResult("center", 0, "King Kong Escapes", 1967, 0.80),
}

def test_three_samples_group_and_save(monkeypatch, tmp_path, sample_dngs):
    in_dir = tmp_path / "Images in"; in_dir.mkdir()
    for p in sample_dngs:
        shutil.copy(p, in_dir / p.name)

    def fake_classify(bgr, size_mm, barcode):
        if barcode == "0025192828928":
            return FIXED["back"]
        if size_mm[1] >= 230:
            return FIXED["center"]
        return FIXED["front"]
    monkeypatch.setattr(pipeline.vision, "classify_photo", fake_classify)
    monkeypatch.setattr(webapp.vision, "draft_listing",
        lambda f, b, bc: {"listing_title": "King Kong Escapes DVD 1967",
                          "description": "Classic Toho kaiju film.", "genre": "Sci-Fi",
                          "region": "2", "runtime": "96 min", "studio": "Toho",
                          "year": 1967})

    session = webapp.build_session(input_dir=in_dir, output_base=tmp_path / "out",
                                   progress=lambda *a: None)
    assert len(session.groups) == 1
    g = session.groups[0]
    sides = sorted(p.side for p in g.photos)
    assert sides == ["back", "center", "front"]
    assert g.barcode == "0025192828928"

    client = TestClient(webapp.create_app(session))
    r = client.post("/api/dvd/0/save", json={
        "listing_title": g.listing_title, "description": g.description,
        "genre": g.genre, "region": g.region, "runtime": g.runtime,
        "studio": g.studio, "condition": "Very good", "price": "9.99",
        "photos": [{"side": p.side, "deleted": False} for p in g.photos]})
    assert r.status_code == 200
    dvd_dir = session.run_dir / "King-Kong-Escapes"
    jpgs = sorted(x.name for x in dvd_dir.glob("*.jpg"))
    assert len(jpgs) == 3
    assert (dvd_dir / "listing.txt").exists()
