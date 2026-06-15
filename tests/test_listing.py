import csv
from pathlib import Path
from dvdflip.models import Photo, DvdGroup
from dvdflip.listing import group_photos, write_listing_txt, write_batch_csv

def _photo(name, side, title, barcode=None):
    p = Photo(source_path=Path(name), side=side, title=title, barcode=barcode)
    p.work_image = Path(name).with_suffix(".jpg")
    return p

def test_group_by_title():
    photos = [
        _photo("a.dng", "front", "King Kong Escapes"),
        _photo("b.dng", "back", "king kong escapes", barcode="0025192828928"),
        _photo("c.dng", "front", "Godzilla"),
    ]
    groups = group_photos(photos)
    assert len(groups) == 2
    kk = next(g for g in groups if "king kong" in g.title.lower())
    assert len(kk.photos) == 2
    assert kk.barcode == "0025192828928"

def test_untitled_photos_group_by_time(tmp_path):
    a = tmp_path / "a.dng"; a.write_bytes(b"x")
    b = tmp_path / "b.dng"; b.write_bytes(b"x")
    p1 = _photo(str(a), "front", ""); p2 = _photo(str(b), "back", "")
    groups = group_photos([p1, p2])
    assert len(groups) == 1
    assert groups[0].title.startswith("Untitled")

def test_write_csv_has_blank_condition_price(tmp_path):
    g = DvdGroup(title="King Kong Escapes", listing_title="King Kong Escapes DVD",
                 barcode="0025192828928", photos=[_photo("a.dng", "front", "x")])
    path = write_batch_csv([g], tmp_path)
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert rows[0]["condition"] == "" and rows[0]["price"] == ""
    assert rows[0]["barcode"] == "0025192828928"
