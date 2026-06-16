"""Robust barcode decoding.

Reads the digits off a barcode using up to three decoders (pyzbar/ZBar,
zxing-cpp, OpenCV) across rotations, scales, and preprocessings. Designed to be
run on the FULL-RESOLUTION original back photo, never the downscaled output.
"""
import cv2
import numpy as np

_ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE,
              180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
_SCALES = (1.0, 1.5, 2.0)


def available_decoders():
    """Names of decoders importable in this environment."""
    names = []
    try:
        import pyzbar.pyzbar  # noqa: F401
        names.append("pyzbar")
    except Exception:
        pass
    try:
        import zxingcpp  # noqa: F401
        names.append("zxingcpp")
    except Exception:
        pass
    if hasattr(cv2, "barcode") and hasattr(cv2.barcode, "BarcodeDetector"):
        names.append("opencv")
    return names


def check_dependencies():
    """(ok, message) for a clear startup error if no decoder is available."""
    if available_decoders():
        return True, ""
    return False, (
        "No barcode decoder available. Install at least one:\n"
        "    pip install pyzbar zxing-cpp\n"
        "    sudo apt install libzbar0\n"
    )


def _preprocs(gray):
    yield gray
    yield cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    yield cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 5)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    yield cv2.addWeighted(gray, 1.5, blur, -0.5, 0)   # unsharp


def _try_pyzbar(img):
    try:
        from pyzbar.pyzbar import decode
        for b in decode(img):
            if b.data:
                return b.data.decode("utf-8", "replace")
    except Exception:
        pass
    return None


def _try_zxing(img):
    try:
        import zxingcpp
        results = zxingcpp.read_barcodes(img)
        for r in results:
            if r.text:
                return r.text
    except Exception:
        pass
    return None


def _try_opencv(img):
    try:
        det = cv2.barcode.BarcodeDetector()
        ok, infos, _types, _pts = det.detectAndDecodeMulti(img)
        if ok:
            for s in infos:
                if s:
                    return s
    except Exception:
        pass
    return None


def decode(bgr: np.ndarray):
    """Return (digits or None, method, rotation_degrees)."""
    gray0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    for deg, flag in _ROTATIONS.items():
        gray = gray0 if flag is None else cv2.rotate(gray0, flag)
        for scale in _SCALES:
            scaled = gray if scale == 1.0 else cv2.resize(
                gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            for proc in _preprocs(scaled):
                for name, fn in (("pyzbar", _try_pyzbar),
                                 ("zxingcpp", _try_zxing),
                                 ("opencv", _try_opencv)):
                    digits = fn(proc)
                    if digits:
                        return digits, name, deg
    return None, "", 0
