"""Can a real barcode decoder read the EAN from the high-res back covers?"""
import cv2

from redboxflip import scan
from redboxflip.models import Face, Settings

IN = "/mnt/c/Users/mardi/Documents/Ebay code/Images in"
S = Settings()
BACKS = {"Amadeus": "IMG_20260617_224148.jpg", "Gale": "IMG_20260617_224014.jpg"}


def try_pyzbar(bgr):
    try:
        from pyzbar.pyzbar import decode
    except Exception as e:
        return f"pyzbar import fail: {e}"
    res = []
    for rot in (0, 1, 2, 3):
        im = bgr if rot == 0 else cv2.rotate(bgr, [None, cv2.ROTATE_90_CLOCKWISE,
              cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE][rot])
        for d in decode(im):
            res.append(f"rot{rot*90}:{d.type}:{d.data.decode(errors='replace')}")
    return res or "none"


for name, fname in BACKS.items():
    bgr = cv2.imread(f"{IN}/{fname}")
    print(f"\n## {name} full-res {bgr.shape}")
    print("  full :", try_pyzbar(bgr))
    sc = scan.scan_case(bgr, S.rembg_model)
    if sc:
        cover = cv2.cvtColor(scan.orient_and_tighten(sc[0], Face.BACK), cv2.COLOR_BGRA2BGR)
        print("  cover:", try_pyzbar(cover))
