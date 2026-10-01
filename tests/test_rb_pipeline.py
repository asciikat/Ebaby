from pathlib import Path

import cv2
import numpy as np

from redboxflip import pipeline
from redboxflip.models import Face, Settings
from rb_helpers import make_redbox_image


def test_gather_inputs_sorted_jpgs(tmp_path):
    (tmp_path / "b.jpg").write_bytes(b"x")
    (tmp_path / "a.JPG").write_bytes(b"x")
    (tmp_path / "note.txt").write_bytes(b"x")
    got = [p.name for p in pipeline.gather_inputs(tmp_path)]
    assert got == ["a.JPG", "b.jpg"]


def test_assign_groups_chunks_of_three():
    paths = [Path(f"{i}.jpg") for i in range(6)]
    groups = pipeline.assign_groups(paths)
    assert len(groups) == 2
    assert groups[0] == [Path("0.jpg"), Path("1.jpg"), Path("2.jpg")]
    assert groups[1][0] == Path("3.jpg")


def test_assign_groups_handles_trailing_partial():
    paths = [Path(f"{i}.jpg") for i in range(4)]
    groups = pipeline.assign_groups(paths)
    assert len(groups) == 2
    assert len(groups[1]) == 1


def test_unique_dir_avoids_collision(tmp_path):
    # Two DVDs with the same title must not share a folder (would overwrite).
    a = pipeline._unique_dir(tmp_path, "Sexy Beast")
    a.mkdir()
    b = pipeline._unique_dir(tmp_path, "Sexy Beast")
    assert a.name == "Sexy Beast"
    assert b.name == "Sexy Beast (2)"
    assert a != b


def test_process_shot_returns_square_image(tmp_path):
    img = make_redbox_image()
    p = tmp_path / "shot.jpg"
    cv2.imwrite(str(p), img)
    s = Settings(cutout_engine="geometric", colour_tidy=False, margin_pct=5,
                 max_edge_px=400, qwen_extract=False, auto_orient=False,
                 title_engine="ocr")
    pil, result = pipeline.process_shot(p, Face.FRONT, s)
    assert pil.width == pil.height
    assert pil.mode == "RGB"
    assert result.face == Face.FRONT
    assert result.cutout_method in ("geometric", "deskew")


def test_run_batch_end_to_end(tmp_path, monkeypatch):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    img = make_redbox_image()
    for name in ("01.jpg", "02.jpg", "03.jpg"):
        cv2.imwrite(str(in_dir / name), img)
    out_dir = tmp_path / "out"

    # Stub OCR so the front cover returns a known title
    monkeypatch.setattr(pipeline.ocr, "extract_title", lambda img: "Test DVD")

    s = Settings(input_dir=str(in_dir), output_dir=str(out_dir),
                 cutout_engine="geometric", colour_tidy=False, title_lookup=False,
                 title_engine="ocr", qwen_extract=False, auto_orient=False,
                 max_edge_px=400)
    run_dir, groups = pipeline.run_batch(s)
    assert len(groups) == 1
    g = groups[0]
    assert g.title == "Test DVD"
    names = sorted(Path(sh.output_path).name for sh in g.shots)
    assert names == ["Test DVD - Back Cover.jpg",
                     "Test DVD - Front Cover.jpg",
                     "Test DVD - Inside.jpg"]
    assert all(Path(sh.output_path).exists() for sh in g.shots)
    assert (run_dir / "dvd_listing.txt").exists()
    assert (run_dir / "run_log.json").exists()
    # Ebaby is title-only: no per-DVD Qwen scan / listing files are written.
    dvd_dir = run_dir / "Test DVD"
    assert not (dvd_dir / "ebay_listing.txt").exists()
    assert not (dvd_dir / "qwen_scan.json").exists()
    assert g.scan is None and g.barcode is None
