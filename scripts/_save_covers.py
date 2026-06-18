"""Save the 6 deskewed covers (no flip correction) so we can eyeball ground truth."""
import cv2
import numpy as np

from redboxflip import scan
from redboxflip.models import Face, Settings

INPUTS = {
    "IMG_20260617_224148.jpg": "amadeus_back",
    "IMG_20260617_224135.jpg": "amadeus_front",
    "IMG_20260617_224208.jpg": "amadeus_inside",
    "IMG_20260617_224014.jpg": "gale_back",
    "1781700457950.jpg":       "gale_front",
    "IMG_20260617_224044.jpg": "gale_inside",
}
FACE = {"back": Face.BACK, "front": Face.FRONT, "inside": Face.INSIDE}
IN = "/mnt/c/Users/mardi/Documents/Ebay code/Images in"
OUT = "/mnt/c/Users/mardi/Documents/Ebay code/scripts/_covers"
S = Settings()

import os
os.makedirs(OUT, exist_ok=True)
for fname, label in INPUTS.items():
    face = FACE[label.split("_")[1]]
    bgr = cv2.imread(f"{IN}/{fname}")
    sc = scan.scan_case(bgr, S.rembg_model)
    hi = scan.orient_and_tighten(sc[0], face)
    cover = np.ascontiguousarray(hi[:, :, :3])
    small = cv2.resize(cover, None, fx=0.45, fy=0.45)
    cv2.imwrite(f"{OUT}/{label}.jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print("saved", label, cover.shape)
