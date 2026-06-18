import json

from redboxflip import naming
from redboxflip.models import (Face, Field, DvdScan, ShotResult, DvdGroup, HIGH)


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
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["settings"]["engine"] == "rembg"
    assert data["dvds"][0]["title"] == "X"


# --- per-DVD self-contained folder files ----------------------------------- #

def _dvd_group(tmp_path, scan=None, barcode=None):
    shots = [
        ShotResult(input_path="a.jpg", face=Face.BACK,
                   output_path=str(tmp_path / "X - Back Cover.jpg")),
        ShotResult(input_path="b.jpg", face=Face.FRONT,
                   output_path=str(tmp_path / "X - Front Cover.jpg")),
        ShotResult(input_path="c.jpg", face=Face.INSIDE,
                   output_path=str(tmp_path / "X - Inside.jpg")),
    ]
    return DvdGroup(index=1, barcode=barcode, title="X",
                    region="Region 4 (PAL, Australia)", shots=shots, scan=scan)


def test_write_dvd_files_creates_three_files(tmp_path):
    txt, csv_path, scan_json = naming.write_dvd_files(tmp_path, _dvd_group(tmp_path))
    assert txt.exists() and csv_path.exists() and scan_json.exists()
    assert (txt.name, csv_path.name, scan_json.name) == (
        "ebay_listing.txt", "ebay_listing.csv", "qwen_scan.json")


def test_listing_photos_keep_full_title_names(tmp_path):
    naming.write_dvd_files(tmp_path, _dvd_group(tmp_path))
    body = (tmp_path / "ebay_listing.txt").read_text(encoding="utf-8")
    assert "X - Front Cover.jpg" in body
    assert "X - Back Cover.jpg" in body
    assert "X - Inside.jpg" in body


def test_listing_flags_unknown_barcode_for_manual_check(tmp_path):
    naming.write_dvd_files(tmp_path, _dvd_group(tmp_path, barcode=None))
    body = (tmp_path / "ebay_listing.txt").read_text(encoding="utf-8")
    assert "MANUAL CHECK NEEDED" in body
    assert "barcode" in body.lower()


def test_listing_includes_known_barcode(tmp_path):
    scan = DvdScan(barcode=Field("9325336010945", HIGH, False, "decoder"))
    naming.write_dvd_files(tmp_path, _dvd_group(tmp_path, scan=scan,
                                                barcode="9325336010945"))
    body = (tmp_path / "ebay_listing.txt").read_text(encoding="utf-8")
    assert "9325336010945" in body


def test_qwen_scan_json_is_valid_and_has_scan(tmp_path):
    scan = DvdScan(title=Field("X", HIGH, False, "front"))
    naming.write_dvd_files(tmp_path, _dvd_group(tmp_path, scan=scan,
                                                barcode="123456789012"))
    data = json.loads((tmp_path / "qwen_scan.json").read_text(encoding="utf-8"))
    assert data["title"] == "X"
    assert data["barcode"] == "123456789012"
    assert data["scan"]["title"]["value"] == "X"
    assert len(data["photos"]) == 3


def test_unknown_title_is_flagged_not_marked_high(tmp_path):
    # An "Untitled DVD N" placeholder must not claim High confidence.
    scan = DvdScan(title=Field("Untitled DVD 3", "Unknown", True, "title"))
    g = _dvd_group(tmp_path, scan=scan, barcode=None)
    g.title = "Untitled DVD 3"
    naming.write_dvd_files(tmp_path, g)
    body = (tmp_path / "ebay_listing.txt").read_text(encoding="utf-8")
    # The title row itself carries the manual-check flag (no contradiction).
    title_line = next(ln for ln in body.splitlines() if ln.startswith("DVD Title:"))
    assert "needs manual check" in title_line
    csv_body = (tmp_path / "ebay_listing.csv").read_text(encoding="utf-8")
    assert "Untitled DVD 3,Unknown,yes" in csv_body


def test_no_scan_still_warns_manual_check(tmp_path):
    # No Qwen scan at all -> the listing must still flag that fields are unknown.
    g = _dvd_group(tmp_path, scan=None, barcode=None)
    naming.write_dvd_files(tmp_path, g)
    body = (tmp_path / "ebay_listing.txt").read_text(encoding="utf-8")
    assert "MANUAL CHECK NEEDED" in body
