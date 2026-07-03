"""RAW decode + surrounding-white color correction, ported from the WSL
pipeline's Colorprodawgv1.py (proven on real DVD photos 2026-07-03).

The math functions (mask/gains/enhance) are pure OpenCV/NumPy. rawpy is
imported lazily inside the decode functions so this module loads — and the
math stays testable — even on an environment without the rawpy wheel.
"""
import io

import cv2
import numpy as np


# ----------------------------------------------------------------------
#  WHITE BALANCE  (uses the white paper SURROUNDING the DVD)
# ----------------------------------------------------------------------
def get_surrounding_white_mask(bgr_img):
    """Mask of the white paper that SURROUNDS the DVD.

    1. Find bright, low-saturation pixels (white paper).
    2. Keep only white blobs that TOUCH the image border — that's the
       background paper. White patches inside the cover art get ignored.
    """
    h, w = bgr_img.shape[:2]
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    kernel = np.ones((5, 5), np.uint8)

    # Try strict thresholds first, relax if not enough paper found
    for v_thr, s_thr in ((150, 45), (120, 60), (100, 80)):
        white = ((v > v_thr) & (s < s_thr)).astype(np.uint8)
        if white.sum() == 0:
            continue

        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel)
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel)

        num, labels = cv2.connectedComponents(white)

        border_labels = set()
        border_labels.update(labels[0, :].tolist())
        border_labels.update(labels[-1, :].tolist())
        border_labels.update(labels[:, 0].tolist())
        border_labels.update(labels[:, -1].tolist())
        border_labels.discard(0)

        if not border_labels:
            continue

        surrounding = np.isin(labels, list(border_labels)).astype(np.uint8) * 255

        # Need at least ~2% of the frame to trust it
        if (surrounding.sum() / 255) > (h * w * 0.02):
            return surrounding

    # Fallback: any white pixels, no border constraint
    return ((v > 120) & (s < 60)).astype(np.uint8) * 255


def get_white_balance_gains(bgr_img):
    """B, G, R multipliers that make the surrounding white paper neutral."""
    mask = get_surrounding_white_mask(bgr_img)
    if mask.sum() == 0:
        return 1.0, 1.0, 1.0

    mean_b = cv2.mean(bgr_img[:, :, 0], mask=mask)[0]
    mean_g = cv2.mean(bgr_img[:, :, 1], mask=mask)[0]
    mean_r = cv2.mean(bgr_img[:, :, 2], mask=mask)[0]

    if mean_b == 0 or mean_g == 0 or mean_r == 0:
        return 1.0, 1.0, 1.0

    target = (mean_b + mean_g + mean_r) / 3.0
    gain_b = float(np.clip(target / mean_b, 0.5, 2.0))
    gain_g = float(np.clip(target / mean_g, 0.5, 2.0))
    gain_r = float(np.clip(target / mean_r, 0.5, 2.0))
    return gain_b, gain_g, gain_r


def apply_white_balance(bgr_img, gains):
    b, g, r = cv2.split(bgr_img.astype("float32"))
    b *= gains[0]
    g *= gains[1]
    r *= gains[2]
    return np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
#  ENHANCEMENT  (slight contrast + slight saturation + mild sharpen)
# ----------------------------------------------------------------------
def create_s_curve_lut(strength=1.0):
    """S-curve contrast LUT. 0 = none, 1.0 = slight, 2.5 = strong."""
    if strength == 0.0:
        return np.arange(256, dtype=np.uint8)
    x = np.arange(256)
    nx = (x / 255.0 - 0.5) * 2
    y = 1 / (1 + np.exp(-strength * nx))
    y = (y - y.min()) / (y.max() - y.min()) * 255
    return np.clip(y, 0, 255).astype(np.uint8)


def enhance_image(bgr_img, contrast=1.0, saturation=1.1, sharpen=1.2):
    img_contrast = cv2.LUT(bgr_img, create_s_curve_lut(contrast))

    if saturation != 1.0:
        hsv = cv2.cvtColor(img_contrast, cv2.COLOR_BGR2HSV).astype("float32")
        hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
        img_color = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    else:
        img_color = img_contrast

    if sharpen > 1.0:
        blurred = cv2.GaussianBlur(img_color, (0, 0), 3.0)
        factor = sharpen - 1.0
        return cv2.addWeighted(img_color, 1.0 + factor, blurred, -factor, 0)
    return img_color
