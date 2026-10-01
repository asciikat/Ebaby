"""Decode a barcode from an image using many high-contrast views — ported
from dvd_barcode_scanner.py's decode_barcodes(). Input is expected to already
be a tight barcode crop (or a full-frame fallback) from
ebaby.stages.barcode_locate; this module focuses on scale/contrast/sharpness
variations to coax a read out of it.
"""
import cv2
from pyzbar import pyzbar

PRODUCT_SYMBOLS = {"EAN13", "UPCA", "EAN8", "UPCE"}


def _locate_barcode_crop(gray):
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
    pad = 15
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
    crop = gray[y0:y1, x0:x1]
    return crop if crop.size > 0 else None


_MAX_VIEW_DIM = 3200  # upscaling a 2400px full-frame fallback x3 = 7200px
                      # through 7 contrast variants each = minutes per miss


def _candidates(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    sources = [gray]
    crop = _locate_barcode_crop(gray)
    if crop is not None:
        sources.append(crop)
    for src in sources:
        for scale in (1, 2, 3):
            if scale > 1 and max(src.shape[:2]) * scale > _MAX_VIEW_DIM:
                continue  # tight crops still get x2/x3; huge frames don't
            g = cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1 else src
            yield g
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
            yield clahe
            _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            yield otsu
            yield cv2.bitwise_not(otsu)
            blur = cv2.GaussianBlur(g, (0, 0), 3)
            yield cv2.addWeighted(g, 1.5, blur, -0.5, 0)
            yield cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 15, 4)


def _normalize_barcode(value, sym_type):
    """A UPC-A code comes back from zbar as a 13-digit EAN13 with a leading
    '0' — strip it so the saved barcode matches the 12-digit UPC-A actually
    printed on the case."""
    if sym_type == "EAN13" and len(value) == 13 and value.startswith("0"):
        return value[1:]
    return value


def decode_barcodes(image, product_only=True):
    """Unique digit strings in first-seen order, normalized. Stops as soon as
    a product barcode is found. A None image (unreadable/corrupt crop) is a
    clean miss, not a crash — the caller loops over a whole batch and one bad
    file must not abort the rest."""
    if image is None:
        return []
    seen = []
    for view in _candidates(image):
        for sym in pyzbar.decode(view):
            if product_only and sym.type not in PRODUCT_SYMBOLS:
                continue
            raw_value = sym.data.decode("utf-8", "ignore").strip()
            value = _normalize_barcode(raw_value, sym.type)
            if value and value not in seen:
                seen.append(value)
        if seen:
            break
    return seen
