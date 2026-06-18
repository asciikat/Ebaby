"""Test OCR-confidence voting for 180 detection: text scores higher upright.
For each face, score the deskewed cover and its 180 flip; save the higher-scoring
orientation so we can eyeball whether the winner is truly upright."""
import os

import cv2
import numpy as np
import pytesseract

from redboxflip import scan
from redboxflip.models import Face, Settings

INPUTS = {
    "amadeus_back": ("IMG_20260617_224148.jpg", Face.BACK),
    "amadeus_front": ("IMG_20260617_224135.jpg", Face.FRONT),
    "amadeus_inside": ("IMG_20260617_224208.jpg", Face.INSIDE),
    "gale_back": ("IMG_20260617_224014.jpg", Face.BACK),
    "gale_front": ("1781700457950.jpg", Face.FRONT),
    "gale_inside": ("IMG_20260617_224044.jpg", Face.INSIDE),
}
IN = "/mnt/c/Users/mardi/Documents/Ebay code/Images in"
OUT = "/mnt/c/Users/mardi/Documents/Ebay code/scripts/_orientcheck"
os.makedirs(OUT, exist_ok=True)
S = Settings()


def score(bgr):
    """Sum of positive word confidences + count of confident words."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if max(gray.shape) < 1400:
        f = 1400.0 / max(gray.shape)
        gray = cv2.resize(gray, None, fx=f, fy=f)
    try:
        d = pytesseract.image_to_data(gray, config="--psm 3",
                                      output_type=pytesseract.Output.DICT)
    except Exception:
        return 0.0, 0
    confs = [int(c) for c in d["conf"] if int(c) > 0]
    strong = sum(1 for i, c in enumerate(d["conf"])
                 if int(c) >= 60 and len(d["text"][i].strip()) >= 3)
    return float(sum(confs)), strong


for label, (fname, face) in INPUTS.items():
    bgr = cv2.imread(f"{IN}/{fname}")
    sc = scan.scan_case(bgr, S.rembg_model)
    cover = np.ascontiguousarray(scan.orient_and_tighten(sc[0], face)[:, :, :3])
    flipped = cv2.rotate(cover, cv2.ROTATE_180)
    s0, n0 = score(cover)
    s1, n1 = score(flipped)
    chosen = cover if (n0, s0) >= (n1, s1) else flipped
    cv2.imwrite(f"{OUT}/{label}.jpg",
                cv2.resize(chosen, (330, int(330 * chosen.shape[0] / chosen.shape[1]))),
                [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f"{label:15s} as-is(strong={n0},sum={s0:.0f})  flip(strong={n1},sum={s1:.0f})  "
          f"-> {'AS-IS' if chosen is cover else 'FLIP'}")
