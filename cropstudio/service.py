"""Turn a buffered DVD (slot->jpeg bytes) into named files under the run folder.

Reuses redboxflip's title reader (Qwen), filename builder, and JPEG saver so the
output is identical to the main pipeline's per-DVD folders.
"""
import cv2
import numpy as np
from PIL import Image

from redboxflip import naming, titles, vlm
from redboxflip.imaging import save_jpeg
from redboxflip.models import FACE_ORDER


def _decode(data: bytes):
    """JPEG bytes -> (PIL RGB, BGR ndarray). Raises ValueError on junk."""
    arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode image")
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), bgr


def ensure_decodable(data: bytes) -> None:
    """Raise ValueError if the bytes are not a decodable image."""
    _decode(data)


def resolve_title(shots: dict, settings, reader=None) -> str:
    """Read the DVD title from Front (slot 1), else Back (0), else Inside (2).

    `reader(bgr, settings) -> str|None` is injectable for tests. By default uses
    Qwen via redboxflip.vlm, but only when it is reachable.
    """
    if reader is None:
        if not vlm.available():
            return ""
        reader = vlm.title_from_cover
    for slot in (1, 0, 2):
        if slot in shots:
            _, bgr = _decode(shots[slot])
            t = reader(bgr, settings)
            return titles.clean_title(t) if t else ""
    return ""


def _unique_dir(parent, stem):
    cand = parent / stem
    n = 2
    while cand.exists():
        cand = parent / f"{stem} ({n})"
        n += 1
    return cand


def save_dvd(shots: dict, run_dir, settings, dvd_counter: int,
             reader=None, quality: int = 92) -> dict:
    """Write one DVD's shots. Returns {title, dir, files}."""
    title = resolve_title(shots, settings, reader) or f"Untitled DVD {dvd_counter}"
    dvd_dir = _unique_dir(run_dir, naming.safe_stem(title))
    files = []
    for slot, data in sorted(shots.items()):
        pil, _ = _decode(data)
        out = dvd_dir / naming.output_filename(title, FACE_ORDER[slot])
        save_jpeg(pil, out, quality)
        files.append(out.name)
    return {"title": title, "dir": str(dvd_dir), "files": files}
