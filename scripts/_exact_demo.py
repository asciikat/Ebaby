"""Demo: contrast/edge-based case quad + exact perspective crop (no matte).

Proves the 'exact' cutout on a real transparent-case photo (Open Water inside),
and previews whether an edge detector can find the clear-case outline that the
user can see by eye. Saves before/after for visual review.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from redboxflip import cutout, clean
from redboxflip.imaging import resize_max_pil

SRC = Path(r"C:\Users\mardi\Documents\Ebay code\Images in\ss (3).jpg")
OUT = Path(r"C:\Users\mardi\Documents\Ebay code\processed\_exact_demo")
OUT.mkdir(parents=True, exist_ok=True)


def contrast_quad(bgr):
    """Find the case outline by edge energy (the visible clear/white boundary)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    edges = cv2.Canny(g, 30, 90)
    edges = cv2.dilate(edges, np.ones((7, 7), np.uint8), iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             np.ones((25, 25), np.uint8), iterations=3)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.15 * bgr.shape[0] * bgr.shape[1]:
        return None
    return cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)


def main():
    bgr = cv2.imread(str(SRC))
    print("source:", SRC.name, bgr.shape)

    quad = contrast_quad(bgr)
    if quad is None:
        print("contrast detector: no confident quad found")
        return
    print("quad:", quad.tolist())

    rgba, method = cutout.make_cutout(bgr, quad, "exact", feather_px=0)
    print("method:", method, "cutout shape:", rgba.shape,
          "opaque:", bool((rgba[:, :, 3] == 255).all()))

    composed = clean.compose_on_white_square(rgba, margin_pct=4)
    composed = resize_max_pil(composed, 1100)
    composed.save(OUT / "open_water_inside_exact.jpg", quality=92)

    # also dump the detected edges + quad overlay for inspection
    overlay = bgr.copy()
    cv2.polylines(overlay, [quad.astype(int)], True, (0, 0, 255), 6)
    cv2.imwrite(str(OUT / "open_water_inside_quad_overlay.jpg"), overlay)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
