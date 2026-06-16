from redboxflip import naming
from redboxflip.models import Face, ShotResult, DvdGroup


def test_safe_stem():
    assert naming.safe_stem("The Matrix: Reloaded") == "The Matrix Reloaded"
    assert naming.safe_stem("a/b\\c") == "a b c"
    assert naming.safe_stem("   ") == "dvd"


def test_output_filename():
    assert naming.output_filename("The Matrix", Face.BACK) == "The Matrix - Back Cover.jpg"
    assert naming.output_filename("The Matrix", Face.FRONT) == "The Matrix - Front Cover.jpg"
    assert naming.output_filename("The Matrix", Face.INSIDE) == "The Matrix - Inside.jpg"


def test_write_listing_creates_txt_and_csv(tmp_path):
    g = DvdGroup(index=1, barcode="9325336022306", title="Open Water",
                 region="Region 4 (PAL, Australia)",
                 shots=[ShotResult(input_path="b.jpg", face=Face.BACK,
                                   output_path="Open Water - Back Cover.jpg")])
    txt, csv_path = naming.write_listing(tmp_path, [g])
    assert txt.exists() and csv_path.exists()
    body = txt.read_text(encoding="utf-8")
    assert "Open Water" in body
    assert "9325336022306" in body
    assert "Region 4 (PAL, Australia)" in body


def test_write_run_log(tmp_path):
    g = DvdGroup(index=1, barcode=None, title="X", region="R", shots=[])
    p = naming.write_run_log(tmp_path, [g], {"engine": "rembg"})
    assert p.exists()
    import json
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["settings"]["engine"] == "rembg"
    assert data["dvds"][0]["title"] == "X"
