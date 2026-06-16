"""Image loading, BGR<->PIL conversion, resize, and JPEG saving."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


def load_image_bgr(path) -> np.ndarray:
    """Load an image as a BGR uint8 array, honouring EXIF orientation."""
    with Image.open(path) as src:
        src.load()
        rgb = ImageOps.exif_transpose(src).convert("RGB")
    return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def pil_to_bgr(pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR)


def resize_max(bgr: np.ndarray, max_edge: int) -> np.ndarray:
    """Downscale so the longest edge is <= max_edge. No upscaling."""
    if max_edge <= 0:
        return bgr
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return bgr
    scale = max_edge / float(longest)
    return cv2.resize(bgr, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def resize_max_pil(pil: Image.Image, max_edge: int) -> Image.Image:
    if max_edge <= 0:
        return pil
    w, h = pil.size
    if max(w, h) <= max_edge:
        return pil
    scale = max_edge / float(max(w, h))
    return pil.resize((max(1, round(w * scale)), max(1, round(h * scale))), LANCZOS)


def save_jpeg(pil: Image.Image, path, quality: int = 92) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pil.convert("RGB").save(path, quality=quality, optimize=True, progressive=True)
