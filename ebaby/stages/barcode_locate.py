"""Locate the barcode on a case photo (back cover first, but the caller may
try any shot of the set) and save a tight crop of just that region — ported
from locate_and_crop_barcodes.py.

For .NEF/.DNG this pulls the embedded full-res JPEG preview via exiftool
(fast, avoids a full RAW decode). Falls back to reading the file directly
with OpenCV for any other format. If no barcode-like region can be
confidently located, the whole frame is returned instead so the decode stage
downstream still gets a shot at it.
"""
import subprocess

import cv2
import numpy as np

RAW_EXTS = {".nef", ".dng"}
MAX_DIM = 2400


def extract_raw_preview(raw_path):
    """Embedded JPEG preview from a RAW file via exiftool, or None."""
    for tag in ("-PreviewImage", "-JpgFromRaw"):
        try:
            result = subprocess.run(
                ["exiftool", "-b", tag, str(raw_path)],
                capture_output=True, check=False, timeout=30,
            )
        except FileNotFoundError:
            return None
        except subprocess.TimeoutExpired:
            return None
        if result.returncode == 0 and result.stdout:
            buf = np.frombuffer(result.stdout, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is not None:
                return img
    return None


def load_image(path):
    """Load any supported source file (RAW preview or standard image) as BGR."""
    path = str(path)
    ext = path.lower().rsplit(".", 1)[-1]
    if f".{ext}" in RAW_EXTS:
        return extract_raw_preview(path)
    return cv2.imread(path)


def locate_barcode_crop(gray):
    """(x0, y0, x1, y1) padded bounding box of the most barcode-like region,
    or None if nothing found. Barcodes are wide-and-short with a strong local
    gradient (alternating bars)."""
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=-1)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=-1)
    grad = cv2.convertScaleAbs(cv2.subtract(cv2.convertScaleAbs(grad_x),
                                            cv2.convertScaleAbs(grad_y)))
    blurred = cv2.blur(grad, (9, 9))
    _, thresh = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    if w < 40 or h < 15 or w / max(h, 1) < 1.5:
        return None
    pad_x, pad_y = int(w * 0.15), int(h * 0.6)
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(gray.shape[1], x + w + pad_x), min(gray.shape[0], y + h + pad_y)
    return x0, y0, x1, y1


def downscale_if_needed(img, max_dim=MAX_DIM):
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def locate_and_crop(photo_path, out_path) -> bool:
    """Save a tight barcode crop (or full-frame fallback) to out_path.
    Returns True if a confident crop was made, False if it fell back to the
    whole frame."""
    img = load_image(photo_path)
    if img is None:
        raise ValueError(f"could not read image: {photo_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    box = locate_barcode_crop(gray)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if box:
        x0, y0, x1, y1 = box
        cv2.imwrite(str(out_path), downscale_if_needed(img[y0:y1, x0:x1]))
        return True
    cv2.imwrite(str(out_path), downscale_if_needed(img))
    return False
