import shutil
from pathlib import Path

import pytest

SAMPLES = Path(__file__).resolve().parent.parent / "samples" / "redbox"


def _have_samples():
    return all((SAMPLES / n).exists() for n in ("back.jpg", "front.jpg", "inside.jpg"))


@pytest.mark.skipif(not _have_samples(), reason="redbox sample scans not present")
def test_real_batch_makes_three_square_jpgs(tmp_path):
    from PIL import Image
    from redboxflip import pipeline
    from redboxflip.models import Settings

    in_dir = tmp_path / "in"; in_dir.mkdir()
    for n in ("back.jpg", "front.jpg", "inside.jpg"):
        shutil.copy(SAMPLES / n, in_dir / n)
    out_dir = tmp_path / "out"

    s = Settings(input_dir=str(in_dir), output_dir=str(out_dir),
                 cutout_engine="grabcut", colour_tidy=True, title_lookup=False,
                 max_edge_px=1200)
    run_dir, groups = pipeline.run_batch(s)

    assert len(groups) == 1
    g = groups[0]
    assert len(g.shots) == 3
    for shot in g.shots:
        assert shot.output_path and Path(shot.output_path).exists()
        im = Image.open(shot.output_path)
        assert im.width == im.height            # square
        # corners are white (background replaced, no red)
        assert im.convert("RGB").getpixel((2, 2)) == (255, 255, 255)


@pytest.mark.skipif(not _have_samples(), reason="redbox sample scans not present")
def test_barcode_reads_from_real_back():
    import cv2
    from redboxflip import barcode
    if not barcode.available_decoders():
        pytest.skip("no barcode decoder installed")
    bgr = cv2.imread(str(SAMPLES / "back.jpg"))
    digits, method, rot = barcode.decode(bgr)
    assert digits and digits.isdigit() and len(digits) >= 8
