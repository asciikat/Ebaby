import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageOps

from .config import RAW_EXT, HEIC_EXT, SUPPORTED_EXT

try:
    import rawpy
    RAWPY_OK = True
except ImportError:
    RAWPY_OK = False

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_OK = True
except ImportError:
    HEIC_OK = False


def load_bgr(path):
    """Load any supported image to a BGR uint8 array, orientation-corrected."""
    path = Path(path)
    if path.suffix.lower() in RAW_EXT:
        if not RAWPY_OK:
            raise RuntimeError("rawpy not installed; cannot read .dng")
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=8,
                output_color=rawpy.ColorSpace.sRGB,
                user_flip=-1,
            )
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    with Image.open(path) as im:
        im.load()
        rgb = ImageOps.exif_transpose(im).convert("RGB")
    return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)


def gather_inputs(folder, exclude_dir=None):
    """Return supported image files under `folder`, sorted, excluding `exclude_dir`."""
    folder = Path(folder)
    out = []
    try:
        exclude = Path(exclude_dir).resolve() if exclude_dir else None
    except Exception:
        exclude = None
    if not folder.is_dir():
        return out
    for p in sorted(folder.rglob("*")):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf not in SUPPORTED_EXT:
            continue
        if suf in RAW_EXT and not RAWPY_OK:
            continue
        if suf in HEIC_EXT and not HEIC_OK:
            continue
        if exclude is not None:
            try:
                p.resolve().relative_to(exclude)
                continue
            except (ValueError, OSError):
                pass
        out.append(p)
    return out
