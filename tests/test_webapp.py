from pathlib import Path
from fastapi.testclient import TestClient
from dvdflip import webapp
from dvdflip.models import Photo, DvdGroup

def _session(tmp_path):
    work = tmp_path / "_work"; work.mkdir(parents=True)
    img = work / "a.jpg"
    import numpy as np, cv2
    cv2.imwrite(str(img), np.full((10, 10, 3), 255, np.uint8))
    p = Photo(source_path=Path("a.dng"), side="front", title="King Kong Escapes")
    p.work_image = img
    g = DvdGroup(title="King Kong Escapes", listing_title="King Kong Escapes DVD",
                 photos=[p])
    return webapp.Session(run_dir=tmp_path, work_dir=work, groups=[g])

def test_review_page_loads(tmp_path):
    app = webapp.create_app(_session(tmp_path))
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200 and "King Kong Escapes" in r.text

def test_save_dvd_writes_outputs(tmp_path):
    sess = _session(tmp_path)
    app = webapp.create_app(sess)
    client = TestClient(app)
    r = client.post("/api/dvd/0/save", json={
        "listing_title": "King Kong Escapes DVD 1967", "description": "d",
        "genre": "Sci-Fi", "region": "2", "runtime": "96 min", "studio": "Toho",
        "condition": "Very good", "price": "9.99",
        "photos": [{"side": "front", "deleted": False}]})
    assert r.status_code == 200
    dvd_dir = tmp_path / "King-Kong-Escapes"
    assert (dvd_dir / "listing.txt").exists()
    assert any(dvd_dir.glob("king-kong-escapes_front.jpg"))

def test_photo_image_served(tmp_path):
    app = webapp.create_app(_session(tmp_path))
    client = TestClient(app)
    r = client.get("/work/a.jpg")
    assert r.status_code == 200

def test_review_has_controls(tmp_path):
    app = webapp.create_app(_session(tmp_path))
    client = TestClient(app)
    html = client.get("/").text
    assert 'id="save-all"' in html
    assert "/work/a.jpg" in html
    assert "Condition" in html
