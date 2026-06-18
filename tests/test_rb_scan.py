"""scan.osd_rotation: confident upright/flip verdict, else None."""
import numpy as np

from redboxflip import scan


def test_osd_rotation_none_on_blank():
    blank = np.full((600, 400, 3), 255, np.uint8)
    assert scan.osd_rotation(blank) is None


def test_osd_rotation_accepts_bgra_input():
    bgra = np.zeros((600, 400, 4), np.uint8)
    bgra[:, :, 3] = 255
    # No text -> OSD can't decide -> None (and must not raise on 4 channels).
    assert scan.osd_rotation(bgra) is None


def test_osd_rotation_returns_zero_or_180_or_none():
    img = np.random.default_rng(0).integers(0, 255, (500, 400, 3), dtype=np.uint8)
    assert scan.osd_rotation(img) in (0, 180, None)


def test_flip_by_text_none_when_no_text():
    blank = np.full((600, 400, 3), 255, np.uint8)
    assert scan.flip_by_text(blank) is None


def test_flip_by_text_detects_upside_down():
    # Render real text, then check upright stays and flipped is detected.
    import cv2
    up = np.full((400, 700, 3), 255, np.uint8)
    for i, line in enumerate(("THE LIFE OF DAVID GALE", "WIDESCREEN COLLECTION",
                              "DIGITAL SURROUND SOUND", "ACADEMY AWARD WINNER")):
        cv2.putText(up, line, (20, 90 + i * 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, (0, 0, 0), 3, cv2.LINE_AA)
    assert scan.flip_by_text(up) is False                       # already upright
    flipped = cv2.rotate(up, cv2.ROTATE_180)
    assert scan.flip_by_text(flipped) is True                   # needs flipping
