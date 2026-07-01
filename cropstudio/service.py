"""Turn a buffered DVD (slot->jpeg bytes) into named files under the run folder.

Reuses redboxflip's title reader (Qwen), filename builder, and JPEG saver so the
output is identical to the main pipeline's per-DVD folders.
"""
import cv2
import numpy as np
from PIL import Image

from redboxflip import barcode, naming, titles, vlm
from redboxflip.imaging import save_jpeg
from redboxflip.models import FACE_ORDER

from . import ebay as _ebay_module
from .ebay import EbayUnavailable
from .ebay_config import load_ebay_config


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


def resolve_identity(shots: dict, settings, ebay_config=None,
                      barcode_decoder=None, ebay_client=None,
                      qwen_reader=None) -> dict:
    """Resolve a DVD's title + supporting identity data from its shots.

    Priority: a barcode decoded off the Back shot (slot 0), matched against
    the eBay Browse API, is the PRIMARY source (grounded in real catalog
    data). Qwen vision (`qwen_reader`) is the FALLBACK — it only runs when
    the barcode doesn't decode, or decodes but eBay has no match for it.
    Qwen never overrides a barcode-resolved title.

    `barcode_decoder(bgr) -> (digits|None, method, rotation)` defaults to
    `redboxflip.barcode.decode`. `ebay_client` needs `.get_token(config)` and
    `.lookup_barcode(barcode, token, config)`, defaulting to the
    `cropstudio.ebay` module. `qwen_reader(bgr, settings) -> str|None`
    defaults to `redboxflip.vlm.title_from_cover` when reachable.
    """
    barcode_decoder = barcode_decoder or barcode.decode
    ebay_client = ebay_client or _ebay_module
    ebay_config = ebay_config if ebay_config is not None else load_ebay_config()

    result = {"title": "", "title_source": "none", "barcode": None,
              "ebay_match": None}

    if 0 in shots:
        _, back_bgr = _decode(shots[0])
        digits, _method, _rotation = barcode_decoder(back_bgr)
        if digits:
            result["barcode"] = digits
            if ebay_config.configured:
                try:
                    token = ebay_client.get_token(ebay_config)
                    match = ebay_client.lookup_barcode(digits, token, ebay_config)
                    if match.get("found"):
                        result["title"] = titles.clean_title(match["title"])
                        result["title_source"] = "ebay"
                        result["ebay_match"] = match
                except EbayUnavailable:
                    pass   # falls through to Qwen below

    if not result["title"]:
        reader = qwen_reader
        if reader is None and vlm.available():
            reader = vlm.title_from_cover
        if reader is not None:
            for slot in (1, 0, 2):
                if slot in shots:
                    _, bgr = _decode(shots[slot])
                    t = reader(bgr, settings)
                    if t:
                        result["title"] = titles.clean_title(t)
                        result["title_source"] = "qwen"
                    break

    return result


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
