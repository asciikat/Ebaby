"""Turn a buffered DVD (slot->jpeg bytes) into named files under the run folder.

Reuses redboxflip's title reader (Qwen), filename builder, and JPEG saver so the
output is identical to the main pipeline's per-DVD folders.
"""
from pathlib import Path

import re

import cv2
import numpy as np
from PIL import Image

from redboxflip import barcode, naming, titles, vlm
from redboxflip.imaging import save_jpeg
from redboxflip.models import FACE_ORDER

from . import ebay as _ebay_module
from .ebay import EbayUnavailable
from .ebay_config import load_ebay_config

# Max seconds resolve_identity spends trying to decode the Back shot's
# barcode before falling through to Qwen. Backs with a readable barcode
# decode in well under a second; this only bounds the hopeless cases.
_BARCODE_DEADLINE_S = 12.0


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


def _plausible_retail_barcode(digits) -> bool:
    """True only for a checksum-valid EAN-8 / UPC-A / EAN-13.

    The decoders can hallucinate — zxing has produced GS1 reads like
    '(01)8998...' from pure noise — and a junk code sent to eBay returns a
    confidently wrong title. Retail DVDs carry EAN/UPC only, so anything
    else is treated as no-barcode and falls through to Qwen.
    """
    if not digits or not re.fullmatch(r"\d{8}|\d{12}|\d{13}", str(digits)):
        return False
    ds = [int(c) for c in str(digits)]
    check = ds.pop()
    # EAN/UPC checksum: weights alternate 3,1 from the rightmost payload digit.
    total = sum(d * (3 if i % 2 == 0 else 1) for i, d in enumerate(reversed(ds)))
    return (10 - total % 10) % 10 == check


def _title_word_seen_in_text(title: str, text: str) -> bool:
    """True if any 4+ letter word from `title` appears in `text` (case-insensitive).
    Best-effort heuristic — not proof, just enough to flag an obvious mismatch."""
    if not title or not text:
        return False
    words = re.findall(r"[A-Za-z]{4,}", title)
    text_low = text.lower()
    return any(w.lower() in text_low for w in words)


def resolve_identity(shots: dict, settings, ebay_config=None,
                      barcode_decoder=None, ebay_client=None,
                      qwen_reader=None, front_text_reader=None) -> dict:
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
    if barcode_decoder is None:
        # Interactive app: cap the sweep. A barcode-less shot otherwise burns
        # ~2 minutes exhausting every rotation/scale/preproc combination.
        barcode_decoder = lambda bgr: barcode.decode(bgr, deadline_s=_BARCODE_DEADLINE_S)
    ebay_client = ebay_client or _ebay_module
    ebay_config = ebay_config if ebay_config is not None else load_ebay_config()

    result = {"title": "", "title_source": "none", "barcode": None,
              "ebay_match": None, "mismatch_warning": False}

    if 0 in shots:
        _, back_bgr = _decode(shots[0])
        digits, _method, _rotation = barcode_decoder(back_bgr)
        if digits and not _plausible_retail_barcode(digits):
            digits = None                      # spurious read — treat as no barcode
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

            if result["title_source"] == "ebay" and 1 in shots:
                _, front_bgr = _decode(shots[1])
                front_reader = front_text_reader
                if front_reader is None:
                    front_reader = lambda bgr, s: vlm.extract_face(bgr, s).get("all_text", "")
                front_text = front_reader(front_bgr, settings) or ""
                if front_text and not _title_word_seen_in_text(result["title"], front_text):
                    result["mismatch_warning"] = True

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


def _unique_title(run_dir, title, faces) -> str:
    """A variant of `title` whose output filenames don't already exist
    directly under run_dir (flat layout — no per-DVD subfolder)."""
    candidate = title
    n = 2
    while any((run_dir / naming.output_filename(candidate, face)).exists()
              for face in faces):
        candidate = f"{title} ({n})"
        n += 1
    return candidate


def save_dvd(shots: dict, run_dir, settings, dvd_counter: int,
             reader=None, ebay_config=None, barcode_decoder=None,
             ebay_client=None, front_text_reader=None, quality: int = 92) -> dict:
    """Write one DVD's shots flat into run_dir. Returns
    {title, dir, files, title_source, barcode, new_stock, used_stock,
    mismatch_warning}."""
    run_dir = Path(run_dir)
    identity = resolve_identity(shots, settings, ebay_config=ebay_config,
                                barcode_decoder=barcode_decoder,
                                ebay_client=ebay_client, qwen_reader=reader,
                                front_text_reader=front_text_reader)
    title = identity["title"] or f"Untitled DVD {dvd_counter}"
    faces_present = [FACE_ORDER[slot] for slot in sorted(shots.keys())]
    unique_title = _unique_title(run_dir, title, faces_present)

    files = []
    for slot, data in sorted(shots.items()):
        pil, _ = _decode(data)
        out = run_dir / naming.output_filename(unique_title, FACE_ORDER[slot])
        save_jpeg(pil, out, quality)
        files.append(out.name)

    return {
        "title": unique_title,
        "dir": str(run_dir),
        "files": files,
        "title_source": identity["title_source"],
        "barcode": identity["barcode"],
        "new_stock": len(shots) < 3,
        "used_stock": len(shots) >= 3,
        "mismatch_warning": identity.get("mismatch_warning", False),
    }
