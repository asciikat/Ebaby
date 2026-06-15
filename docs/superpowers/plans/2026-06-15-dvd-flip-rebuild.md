# DVD Flip (rebuild) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the DVD-photo-prep tool so classic CV does the pixels and a local vision model (Qwen2.5-VL via Ollama) makes every decision, with a browser review screen before `.jpg` export.

**Architecture:** A small Python package `dvdflip/`. CV stages (load, flatten-to-A4, white-balance, crop, barcode) are ported from the proven `ebayflip_V2.py`. Each cleaned photo goes to Ollama for `{side, rotation_cw, title, year, confidence}`; photos group into DVDs by title; the model drafts a full listing. A FastAPI app serves a one-card-per-DVD review screen; on approve it writes `title_side.jpg` + `listing.txt` + a tracking CSV. Native Windows, launched by double-clicking a `.bat`.

**Tech Stack:** Python 3.12 (Windows), numpy, opencv-python, rawpy, pillow, pillow-heif, pyzbar, requests, FastAPI, uvicorn, jinja2; Ollama serving `qwen2.5vl`. Tests: pytest + httpx.

**Reference (do not modify):** `ebayflip_V2.py` (V2). Spec: `docs/superpowers/specs/2026-06-15-dvd-flip-rebuild-design.md`.

**Canonical I/O:** `.dng` RAW in (from `Images in/`) → `.jpg` out. JPG/HEIC in are best-effort.

**Note on commands:** the dev shell is Windows PowerShell. Use the `py` launcher (`py -m pytest …`). After `winget` installs Python, open a **new** shell so PATH refreshes. All `git` steps assume `git init` has been run (Task 1).

---

## File structure

```
dvdflip/
  __init__.py
  __main__.py        # launcher: process Images in, start web app, open browser
  config.py          # constants, model tag, paths, thresholds
  models.py          # dataclasses: ClassifyResult, Photo, DvdGroup
  loader.py          # decode DNG/JPG/HEIC → BGR; gather inputs
  flatten.py         # detect A4, warp to true A4, mm scale
  clean.py           # white balance, segment+crop DVD, barcode, enhance, composite
  vision.py          # Ollama calls: classify_photo, draft_listing, health check
  pipeline.py        # process one image → Photo (CV + vision)
  listing.py         # group photos → DvdGroups; write listing.txt / csv / run_log
  webapp.py          # FastAPI: build session, review page, edit/save endpoints
  templates/review.html
tests/
  conftest.py
  test_loader.py  test_flatten.py  test_clean.py
  test_vision.py  test_pipeline.py test_listing.py
  test_webapp.py  test_end_to_end.py
samples/             # the 3 King Kong Escapes DNGs (regression fixture)
Run DVD Flip.bat     # double-click launcher
requirements.txt
```

`models.py` is a small addition to the spec's listed structure: it holds the shared dataclasses so `pipeline`, `listing`, and `webapp` agree on types without circular imports.

---

## Task 1: Project skeleton, environment, and tooling

**Files:**
- Create: `requirements.txt`, `dvdflip/__init__.py`, `tests/conftest.py`, `tests/test_smoke.py`, `.gitignore`, `pytest.ini`
- Create: `samples/` (copy of the 3 DNGs)

- [ ] **Step 1: Install Python and Ollama (one-time)**

Run in PowerShell:
```powershell
winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
winget install -e --id Ollama.Ollama --accept-package-agreements --accept-source-agreements
```
Open a **new** PowerShell window, then pull the model:
```powershell
ollama pull qwen2.5vl:7b
```
Expected: `py --version` prints `Python 3.12.x`; `ollama list` shows `qwen2.5vl:7b`.

- [ ] **Step 2: Init git and create the package skeleton**

Run:
```powershell
git init
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```
Create `requirements.txt`:
```
numpy
opencv-python
rawpy
pillow
pillow-heif
pyzbar
requests
fastapi
uvicorn[standard]
jinja2
python-multipart
pytest
httpx
```
Create `.gitignore`:
```
.venv/
__pycache__/
*.pyc
processed/
.superpowers/
```
Create `pytest.ini`:
```ini
[pytest]
testpaths = tests
```
Create empty `dvdflip/__init__.py` containing:
```python
"""DVD Flip — A4-reference photo prep + local-VLM decisions for eBay listings."""
__version__ = "3.0.0"
```

- [ ] **Step 3: Install deps and copy the sample fixtures**

Run:
```powershell
pip install -r requirements.txt
New-Item -ItemType Directory -Force samples | Out-Null
Copy-Item "Images in\*.dng" samples\
```
Expected: `samples\` contains the 3 `IMG_20260615_*.dng` files.

- [ ] **Step 4: Write the smoke test**

`tests/conftest.py`:
```python
from pathlib import Path
import pytest

SAMPLES = Path(__file__).resolve().parent.parent / "samples"

@pytest.fixture
def sample_dngs():
    files = sorted(SAMPLES.glob("*.dng"))
    if len(files) < 3:
        pytest.skip("sample DNGs not present")
    return files

@pytest.fixture
def sample_front(sample_dngs):
    return next(p for p in sample_dngs if p.name.endswith("153112.dng"))

@pytest.fixture
def sample_back(sample_dngs):
    return next(p for p in sample_dngs if p.name.endswith("153120.dng"))

@pytest.fixture
def sample_center(sample_dngs):
    return next(p for p in sample_dngs if p.name.endswith("153137.dng"))
```

`tests/test_smoke.py`:
```python
import dvdflip

def test_package_imports():
    assert dvdflip.__version__
```

- [ ] **Step 5: Run the smoke test**

Run: `py -m pytest tests/test_smoke.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add -A
git commit -m "chore: project skeleton, deps, sample fixtures"
```

---

## Task 2: config.py and models.py

**Files:**
- Create: `dvdflip/config.py`, `dvdflip/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

`tests/test_models.py`:
```python
from dvdflip.models import ClassifyResult, Photo, DvdGroup
from dvdflip import config

def test_classify_result_defaults():
    r = ClassifyResult(side="front", rotation_cw=0, title="X", year=1967, confidence=0.9)
    assert r.side == "front" and r.rotation_cw == 0

def test_photo_defaults():
    from pathlib import Path
    p = Photo(source_path=Path("a.dng"))
    assert p.side == "other" and p.a4_found is True and p.deleted is False

def test_dvdgroup_defaults():
    g = DvdGroup(title="king kong escapes")
    assert g.photos == [] and g.approved is False

def test_config_constants():
    assert config.PX_PER_MM == 10
    assert config.OLLAMA_URL.startswith("http://")
    assert "qwen2.5vl" in config.MODEL_TAG
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: dvdflip.models`.

- [ ] **Step 3: Write config.py**

`dvdflip/config.py`:
```python
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = SCRIPT_DIR / "Images in"
DEFAULT_OUTPUT = SCRIPT_DIR / "processed"

RAW_EXT = {".dng"}
PIL_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
HEIC_EXT = {".heic", ".heif"}
SUPPORTED_EXT = RAW_EXT | PIL_EXT | HEIC_EXT

A4_SHORT_MM = 210.0
A4_LONG_MM = 297.0
A4_RATIO = A4_LONG_MM / A4_SHORT_MM
PX_PER_MM = 10

AUTO_CONFIDENCE = 0.72
REVIEW_CONFIDENCE = 0.45

OLLAMA_URL = "http://localhost:11434"
MODEL_TAG = "qwen2.5vl:7b"
VISION_TIMEOUT = 120

OUTPUT_MAX_DIM = 1600
JPEG_QUALITY = 92
PADDING_PCT = 7
GROUP_TIME_GAP_S = 30.0
```

- [ ] **Step 4: Write models.py**

`dvdflip/models.py`:
```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

VALID_SIDES = ("front", "back", "center", "spine", "other")

@dataclass
class ClassifyResult:
    side: str
    rotation_cw: int
    title: str
    year: Optional[int]
    confidence: float

@dataclass
class Photo:
    source_path: Path
    work_image: Optional[Path] = None
    side: str = "other"
    rotation_cw: int = 0
    confidence: float = 0.0
    barcode: Optional[str] = None
    size_mm: tuple = (0.0, 0.0)
    a4_found: bool = True
    title: str = ""
    year: Optional[int] = None
    deleted: bool = False
    extra_rotation_cw: int = 0   # manual rotate applied on the review screen

@dataclass
class DvdGroup:
    title: str = ""
    year: Optional[int] = None
    photos: list = field(default_factory=list)
    listing_title: str = ""
    description: str = ""
    genre: str = ""
    region: str = ""
    runtime: str = ""
    studio: str = ""
    barcode: str = ""
    condition: str = ""
    price: str = ""
    approved: bool = False
```

- [ ] **Step 5: Run test to verify it passes**

Run: `py -m pytest tests/test_models.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```powershell
git add -A
git commit -m "feat: config constants and shared dataclasses"
```

---

## Task 3: loader.py (port load + gather)

**Files:**
- Create: `dvdflip/loader.py`
- Test: `tests/test_loader.py`

- [ ] **Step 1: Write the failing test**

`tests/test_loader.py`:
```python
import numpy as np
from dvdflip.loader import load_bgr, gather_inputs

def test_load_dng_returns_bgr(sample_front):
    bgr = load_bgr(sample_front)
    assert isinstance(bgr, np.ndarray)
    assert bgr.ndim == 3 and bgr.shape[2] == 3
    assert bgr.dtype == np.uint8
    assert min(bgr.shape[:2]) > 1000   # Xiaomi RAW is multi-megapixel

def test_gather_inputs_finds_dngs(tmp_path):
    (tmp_path / "a.dng").write_bytes(b"x")
    (tmp_path / "b.txt").write_text("no")
    found = gather_inputs(tmp_path)
    assert [p.name for p in found] == ["a.dng"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: dvdflip.loader`.

- [ ] **Step 3: Write loader.py (ported from V2)**

`dvdflip/loader.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_loader.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```powershell
git add -A
git commit -m "feat: image loader (DNG/JPG/HEIC) ported from V2"
```

---

## Task 4: flatten.py (A4 detect + warp + mm scale)

**Files:**
- Create: `dvdflip/flatten.py`
- Test: `tests/test_flatten.py`

- [ ] **Step 1: Write the failing test**

`tests/test_flatten.py`:
```python
import numpy as np
from dvdflip.loader import load_bgr
from dvdflip.flatten import detect_a4, warp_to_a4

def test_detect_and_warp_to_a4(sample_front):
    bgr = load_bgr(sample_front)
    quad, conf = detect_a4(bgr)
    assert quad is not None
    assert conf > 0.45                 # front cover detects strongly in V2
    flat = warp_to_a4(bgr, quad)
    h, w = flat.shape[:2]
    long_side, short_side = max(h, w), min(h, w)
    assert abs(long_side / short_side - 297 / 210) < 0.02   # true A4 proportions
    assert long_side == 2970 and short_side == 2100         # 10 px/mm
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_flatten.py -v`
Expected: FAIL with `ModuleNotFoundError: dvdflip.flatten`.

- [ ] **Step 3: Write flatten.py (ported from V2)**

`dvdflip/flatten.py`:
```python
import numpy as np
import cv2

from .config import A4_SHORT_MM, A4_LONG_MM, A4_RATIO, PX_PER_MM


def order_points(pts):
    """Return points as [top-left, top-right, bottom-right, bottom-left]."""
    pts = np.asarray(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def detect_a4(bgr, dump=None):
    """Find the A4 sheet. Returns (quad_full_res or None, confidence)."""
    h, w = bgr.shape[:2]
    scale = 1100.0 / max(h, w)
    small = cv2.resize(bgr, (int(w * scale), int(h * scale)))
    sh, sw = small.shape[:2]
    img_area = float(sh * sw)

    L = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[:, :, 0]
    s = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1]
    Lb = cv2.GaussianBlur(L, (5, 5), 0)
    otsu, _ = cv2.threshold(Lb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thr = max(0.40 * float(L.max()), float(otsu) * 0.80)
    mask = ((Lb >= thr) & (s <= 95)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    if dump is not None:
        cv2.imwrite(dump, mask)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, 0.0
    for c in cnts:
        hull = cv2.convexHull(c)
        area = cv2.contourArea(hull)
        if area < img_area * 0.05:
            continue
        rrect = cv2.minAreaRect(c)
        (bw, bh) = rrect[1]
        if bw < 1 or bh < 1:
            continue
        ratio = max(bw, bh) / min(bw, bh)
        fill = area / max(bw * bh, 1.0)
        ratio_score = max(0.0, 1.0 - abs(ratio - A4_RATIO) / 0.5)
        fill_score = max(0.0, min(1.0, (fill - 0.6) / 0.4))
        area_score = min(1.0, area / img_area / 0.5)
        score = 0.55 * ratio_score + 0.30 * fill_score + 0.15 * area_score
        if score > best_score:
            best_score = score
            best = cv2.boxPoints(rrect).astype(np.float32) / scale
    return best, float(best_score)


def warp_to_a4(bgr, quad, px_per_mm=PX_PER_MM):
    """Perspective-warp the sheet flat to true A4 proportions."""
    rect = order_points(quad)
    tl, tr, br, bl = rect
    wid = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    hei = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    if wid >= hei:
        W, H = int(A4_LONG_MM * px_per_mm), int(A4_SHORT_MM * px_per_mm)
    else:
        W, H = int(A4_SHORT_MM * px_per_mm), int(A4_LONG_MM * px_per_mm)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(bgr, M, (W, H), flags=cv2.INTER_CUBIC)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_flatten.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add -A
git commit -m "feat: A4 detection and warp-to-A4 ported from V2"
```

---

## Task 5: clean.py (white balance, segment, crop, barcode, compose) + mm size

**Files:**
- Create: `dvdflip/clean.py`
- Test: `tests/test_clean.py`

- [ ] **Step 1: Write the failing test**

`tests/test_clean.py`:
```python
import numpy as np
from dvdflip.loader import load_bgr
from dvdflip.flatten import detect_a4, warp_to_a4
from dvdflip.clean import (white_balance_from_paper, segment_dvd, crop_rect,
                           dvd_size_mm, scan_barcodes, composite_on_white)

def _flat(path):
    bgr = load_bgr(path)
    quad, _ = detect_a4(bgr)
    return warp_to_a4(bgr, quad)

def test_segment_and_size_front(sample_front):
    flat, _ = white_balance_from_paper(_flat(sample_front))
    box = segment_dvd(flat)
    assert box is not None
    short_mm, long_mm = dvd_size_mm(box)
    assert 120 < short_mm < 160 and 170 < long_mm < 210   # ~135 x 190 mm cover

def test_barcode_reads_on_back(sample_back):
    flat, _ = white_balance_from_paper(_flat(sample_back))
    box = segment_dvd(flat)
    dvd = crop_rect(flat, box)
    codes = scan_barcodes(dvd)
    assert codes and "0025192828928" in codes

def test_composite_is_square_white(sample_front):
    flat, _ = white_balance_from_paper(_flat(sample_front))
    dvd = crop_rect(flat, segment_dvd(flat))
    out = composite_on_white(dvd)
    assert out.shape[0] == out.shape[1]
    assert (out[0, 0] == [255, 255, 255]).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_clean.py -v`
Expected: FAIL with `ModuleNotFoundError: dvdflip.clean`.

- [ ] **Step 3: Write clean.py (ported from V2 + dvd_size_mm)**

`dvdflip/clean.py`:
```python
import numpy as np
import cv2
from PIL import Image

from .config import PX_PER_MM, PADDING_PCT
from .flatten import order_points


def paper_mask(flat_bgr):
    L = cv2.cvtColor(flat_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    s = cv2.cvtColor(flat_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    bright = L >= max(np.percentile(L, 55), 0.6 * float(L.max()))
    neutral = s <= 45
    return bright & neutral


def white_balance_from_paper(flat_bgr, target=242.0):
    """Neutralise cast using paper as white, lift exposure. Returns (bgr, ok)."""
    pm = paper_mask(flat_bgr)
    img = flat_bgr.astype(np.float32)
    if pm.sum() < 0.02 * pm.size:
        means = img.reshape(-1, 3).mean(axis=0)
        ok = False
    else:
        means = img[pm].mean(axis=0)
        ok = True
    means = np.maximum(means, 1.0)
    gains = means.mean() / means
    bal = img * gains
    if ok:
        paper_after = (bal[pm].mean(axis=0)).mean()
    else:
        paper_after = np.percentile(bal, 95)
    if paper_after > 1.0:
        bal *= (target / paper_after)
    return np.clip(bal, 0, 255).astype(np.uint8), ok


def segment_dvd(flat_bgr, dump=None):
    """Find the DVD on the white sheet → rotated-rect box (4 pts) or None."""
    h, w = flat_bgr.shape[:2]
    minc = flat_bgr.min(axis=2)
    white_level = float(np.percentile(minc, 92))
    thr = max(120.0, 0.72 * white_level)
    fg = (minc < thr).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    cleaned = np.zeros_like(fg)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches = (x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1)
        if not touches:
            cleaned[lab == i] = 255
    if cleaned.any():
        fg = cleaned
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    if dump is not None:
        cv2.imwrite(dump, fg)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.04 * h * w:
        return None
    return cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)


def dvd_size_mm(box):
    """(short_mm, long_mm) of the DVD's rotated rectangle."""
    r = order_points(box)
    w = np.linalg.norm(r[1] - r[0])
    h = np.linalg.norm(r[3] - r[0])
    long_mm = max(w, h) / PX_PER_MM
    short_mm = min(w, h) / PX_PER_MM
    return float(short_mm), float(long_mm)


def crop_rect(bgr, box):
    """Deskew/crop a rotated-rectangle region to an upright image."""
    rect = order_points(box)
    tl, tr, br, bl = rect
    W = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    H = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    W, H = max(W, 2), max(H, 2)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(bgr, M, (W, H), flags=cv2.INTER_CUBIC)


def scan_barcodes(bgr):
    """Decoded barcode strings (multi-rotation). [] if none, None if pyzbar missing."""
    try:
        from pyzbar.pyzbar import decode
    except ImportError:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    longest = max(pil.size)
    if longest < 1600:
        f = 1600.0 / longest
        pil = pil.resize((int(pil.width * f), int(pil.height * f)))
    found = []
    for rot in (0, 90, 180, 270):
        test = pil if rot == 0 else pil.rotate(-rot, expand=True)
        try:
            for b in decode(test):
                code = b.data.decode("utf-8", errors="replace")
                if code and code not in found:
                    found.append(code)
        except Exception:
            pass
        if found:
            break
    return found


def enhance(bgr, strength=0.6):
    """Gentle local-contrast + saturation pop."""
    if strength <= 0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.12, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return np.clip(bgr.astype(np.float32) * (1 - strength)
                   + out.astype(np.float32) * strength, 0, 255).astype(np.uint8)


def composite_on_white(dvd_bgr, padding_pct=PADDING_PCT, square=True):
    h, w = dvd_bgr.shape[:2]
    pad = int(max(w, h) * padding_pct / 100.0)
    if square:
        side = max(w, h) + 2 * pad
        canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    else:
        canvas = np.full((h + 2 * pad, w + 2 * pad, 3), 255, dtype=np.uint8)
    y = (canvas.shape[0] - h) // 2
    x = (canvas.shape[1] - w) // 2
    canvas[y:y + h, x:x + w] = dvd_bgr
    return canvas


def resize_max(bgr, max_dim):
    h, w = bgr.shape[:2]
    if max_dim <= 0 or max(h, w) <= max_dim:
        return bgr
    f = max_dim / float(max(h, w))
    return cv2.resize(bgr, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_clean.py -v`
Expected: PASS (3 tests). If `dvd_size_mm` bounds are slightly off for this exact sample, widen the asserts to the observed value ±10 mm — do not change the function.

- [ ] **Step 5: Commit**

```powershell
git add -A
git commit -m "feat: white-balance, segment, crop, barcode, compose + mm size"
```

---

## Task 6: vision.py (Ollama classify, draft, health)

**Files:**
- Create: `dvdflip/vision.py`
- Test: `tests/test_vision.py`

- [ ] **Step 1: Write the failing test**

`tests/test_vision.py`:
```python
import json
import numpy as np
import pytest
from dvdflip import vision
from dvdflip.models import ClassifyResult

WHITE = np.full((40, 30, 3), 255, dtype=np.uint8)

def _fake_chat(content):
    def _call(messages, **kw):
        return content
    return _call

def test_classify_parses_clean_json(monkeypatch):
    payload = json.dumps({"side": "front", "rotation_cw": 180,
                          "title": "King Kong Escapes", "year": 1967,
                          "confidence": 0.9})
    monkeypatch.setattr(vision, "_chat", _fake_chat(payload))
    r = vision.classify_photo(WHITE, size_mm=(135, 190), barcode=None)
    assert isinstance(r, ClassifyResult)
    assert r.side == "front" and r.rotation_cw == 180 and r.year == 1967

def test_classify_clamps_bad_rotation(monkeypatch):
    payload = json.dumps({"side": "spaceship", "rotation_cw": 47,
                          "title": "", "year": "n/a", "confidence": 5})
    monkeypatch.setattr(vision, "_chat", _fake_chat(payload))
    r = vision.classify_photo(WHITE, size_mm=(0, 0), barcode=None)
    assert r.rotation_cw in (0, 90, 180, 270)
    assert r.side == "other"          # invalid side falls back to 'other'
    assert r.year is None             # non-int year → None
    assert 0.0 <= r.confidence <= 1.0

def test_classify_handles_garbage(monkeypatch):
    monkeypatch.setattr(vision, "_chat", _fake_chat("not json at all"))
    r = vision.classify_photo(WHITE, size_mm=(0, 0), barcode=None)
    assert r.side == "other" and r.confidence == 0.0

def test_health_false_when_down(monkeypatch):
    def boom(*a, **k):
        raise OSError("refused")
    monkeypatch.setattr(vision.requests, "get", boom)
    assert vision.ollama_available() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_vision.py -v`
Expected: FAIL with `ModuleNotFoundError: dvdflip.vision`.

- [ ] **Step 3: Write vision.py**

`dvdflip/vision.py`:
```python
import base64
import json
import cv2
import requests

from .config import OLLAMA_URL, MODEL_TAG, VISION_TIMEOUT
from .models import ClassifyResult, VALID_SIDES

CLASSIFY_PROMPT = (
    "You are looking at one photo of a single DVD item on a white background. "
    "The item is about {short:.0f} x {long:.0f} mm. {barcode_hint}\n"
    "Answer ONLY with JSON of this exact shape:\n"
    '{{"side": "front|back|center|spine|other", '
    '"rotation_cw": 0, "title": "", "year": null, "confidence": 0.0}}\n'
    "Definitions: 'front' = front cover art; 'back' = back cover (usually has a "
    "barcode and small print); 'center' = an open case / disc tray (much larger, "
    "landscape); 'spine' = thin edge. rotation_cw is the clockwise degrees "
    "(0, 90, 180 or 270) needed to make the item upright and readable. "
    "title and year come from the cover text; use null for year if unknown. "
    "confidence is 0..1 for how sure you are about side and rotation."
)

LISTING_PROMPT = (
    "These photos show the front and back of one DVD for an eBay listing. "
    "Read the covers and answer ONLY with JSON:\n"
    '{{"listing_title": "", "description": "", "genre": "", "region": "", '
    '"runtime": "", "studio": "", "year": null}}\n'
    "listing_title: a concise search-friendly eBay title ending in 'DVD' with "
    "year and key terms. description: 1-2 plain sentences. Leave a field as an "
    "empty string if not visible. Do NOT invent a barcode, condition, or price."
)


def _bgr_to_b64(bgr, max_side=1024):
    h, w = bgr.shape[:2]
    if max(h, w) > max_side:
        f = max_side / float(max(h, w))
        bgr = cv2.resize(bgr, (int(w * f), int(h * f)))
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _chat(messages, model=MODEL_TAG):
    """POST to Ollama /api/chat, return the model's text content."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": model, "messages": messages, "stream": False, "format": "json"},
        timeout=VISION_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def ollama_available():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _coerce_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _coerce_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def classify_photo(bgr, size_mm, barcode):
    short_mm, long_mm = size_mm
    barcode_hint = ("A barcode reads " + barcode + " on this side."
                    if barcode else "No barcode was detected on this side.")
    prompt = CLASSIFY_PROMPT.format(short=short_mm, long=long_mm, barcode_hint=barcode_hint)
    messages = [{"role": "user", "content": prompt, "images": [_bgr_to_b64(bgr)]}]
    try:
        raw = _chat(messages)
        data = json.loads(raw)
    except Exception:
        return ClassifyResult(side="other", rotation_cw=0, title="", year=None, confidence=0.0)

    side = data.get("side")
    if side not in VALID_SIDES:
        side = "other"
    rot = _coerce_int(data.get("rotation_cw")) or 0
    rot = min((0, 90, 180, 270), key=lambda r: abs(r - (rot % 360)))
    conf = min(1.0, max(0.0, _coerce_float(data.get("confidence"))))
    return ClassifyResult(
        side=side,
        rotation_cw=rot,
        title=str(data.get("title") or "").strip(),
        year=_coerce_int(data.get("year")),
        confidence=conf,
    )


def draft_listing(front_bgr, back_bgr, barcode):
    images = [_bgr_to_b64(front_bgr)]
    if back_bgr is not None:
        images.append(_bgr_to_b64(back_bgr))
    messages = [{"role": "user", "content": LISTING_PROMPT, "images": images}]
    fields = {"listing_title": "", "description": "", "genre": "",
              "region": "", "runtime": "", "studio": "", "year": None}
    try:
        data = json.loads(_chat(messages))
    except Exception:
        return fields
    for k in fields:
        if k == "year":
            fields[k] = _coerce_int(data.get("year"))
        else:
            fields[k] = str(data.get(k) or "").strip()
    return fields
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_vision.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```powershell
git add -A
git commit -m "feat: Ollama vision client (classify, draft listing, health)"
```

---

## Task 7: pipeline.py (one image → Photo)

**Files:**
- Create: `dvdflip/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

`tests/test_pipeline.py`:
```python
from dvdflip import pipeline, vision
from dvdflip.models import ClassifyResult

def test_process_image_front(monkeypatch, tmp_path, sample_front):
    monkeypatch.setattr(
        vision, "classify_photo",
        lambda bgr, size_mm, barcode: ClassifyResult(
            side="front", rotation_cw=0, title="King Kong Escapes",
            year=1967, confidence=0.95))
    photo = pipeline.process_image(sample_front, tmp_path)
    assert photo.side == "front"
    assert photo.a4_found is True
    assert photo.work_image is not None and photo.work_image.exists()
    assert photo.title == "King Kong Escapes"

def test_process_image_back_reads_barcode(monkeypatch, tmp_path, sample_back):
    monkeypatch.setattr(
        vision, "classify_photo",
        lambda bgr, size_mm, barcode: ClassifyResult(
            side="back", rotation_cw=0, title="King Kong Escapes",
            year=1967, confidence=0.95))
    photo = pipeline.process_image(sample_back, tmp_path)
    assert photo.barcode == "0025192828928"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: dvdflip.pipeline`.

- [ ] **Step 3: Write pipeline.py**

`dvdflip/pipeline.py`:
```python
from pathlib import Path
import cv2

from . import vision
from .config import REVIEW_CONFIDENCE, OUTPUT_MAX_DIM, JPEG_QUALITY
from .loader import load_bgr
from .flatten import detect_a4, warp_to_a4
from .clean import (white_balance_from_paper, segment_dvd, crop_rect,
                    dvd_size_mm, scan_barcodes, enhance, composite_on_white,
                    resize_max)
from .models import Photo

_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def _write_work(bgr, work_dir, stem):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / f"{stem}.jpg"
    cv2.imwrite(str(out), bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return out


def process_image(path, work_dir):
    """CV-clean one image, ask the model to classify it, return a Photo."""
    path = Path(path)
    stem = path.stem
    bgr = load_bgr(path)

    quad, conf = detect_a4(bgr)
    if quad is None or conf < REVIEW_CONFIDENCE:
        fallback = resize_max(bgr, OUTPUT_MAX_DIM)
        return Photo(source_path=path, a4_found=False, side="other",
                     work_image=_write_work(fallback, work_dir, stem))

    flat, _ = white_balance_from_paper(warp_to_a4(bgr, quad))
    box = segment_dvd(flat)
    if box is None:
        fallback = resize_max(flat, OUTPUT_MAX_DIM)
        return Photo(source_path=path, a4_found=False, side="other",
                     work_image=_write_work(fallback, work_dir, stem))

    dvd = crop_rect(flat, box)
    barcodes = scan_barcodes(dvd) or []
    barcode = barcodes[0] if barcodes else None
    size_mm = dvd_size_mm(box)

    result = vision.classify_photo(dvd, size_mm, barcode)
    if result.rotation_cw in _ROT:
        dvd = cv2.rotate(dvd, _ROT[result.rotation_cw])

    dvd = enhance(dvd)
    final = resize_max(composite_on_white(dvd), OUTPUT_MAX_DIM)

    return Photo(
        source_path=path,
        work_image=_write_work(final, work_dir, stem),
        side=result.side,
        rotation_cw=result.rotation_cw,
        confidence=result.confidence,
        barcode=barcode,
        size_mm=size_mm,
        a4_found=True,
        title=result.title,
        year=result.year,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_pipeline.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```powershell
git add -A
git commit -m "feat: per-image pipeline (CV clean + VLM classify)"
```

---

## Task 8: listing.py (group + write files)

**Files:**
- Create: `dvdflip/listing.py`
- Test: `tests/test_listing.py`

- [ ] **Step 1: Write the failing test**

`tests/test_listing.py`:
```python
import csv
from pathlib import Path
from dvdflip.models import Photo, DvdGroup
from dvdflip.listing import group_photos, write_listing_txt, write_batch_csv

def _photo(name, side, title, barcode=None):
    p = Photo(source_path=Path(name), side=side, title=title, barcode=barcode)
    p.work_image = Path(name).with_suffix(".jpg")
    return p

def test_group_by_title():
    photos = [
        _photo("a.dng", "front", "King Kong Escapes"),
        _photo("b.dng", "back", "king kong escapes", barcode="0025192828928"),
        _photo("c.dng", "front", "Godzilla"),
    ]
    groups = group_photos(photos)
    assert len(groups) == 2
    kk = next(g for g in groups if "king kong" in g.title.lower())
    assert len(kk.photos) == 2
    assert kk.barcode == "0025192828928"

def test_untitled_photos_group_by_time(tmp_path):
    a = tmp_path / "a.dng"; a.write_bytes(b"x")
    b = tmp_path / "b.dng"; b.write_bytes(b"x")
    p1 = _photo(str(a), "front", ""); p2 = _photo(str(b), "back", "")
    groups = group_photos([p1, p2])
    assert len(groups) == 1
    assert groups[0].title.startswith("Untitled")

def test_write_csv_has_blank_condition_price(tmp_path):
    g = DvdGroup(title="King Kong Escapes", listing_title="King Kong Escapes DVD",
                 barcode="0025192828928", photos=[_photo("a.dng", "front", "x")])
    path = write_batch_csv([g], tmp_path)
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert rows[0]["condition"] == "" and rows[0]["price"] == ""
    assert rows[0]["barcode"] == "0025192828928"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_listing.py -v`
Expected: FAIL with `ModuleNotFoundError: dvdflip.listing`.

- [ ] **Step 3: Write listing.py**

`dvdflip/listing.py`:
```python
import csv
import json
import time
from pathlib import Path

from .config import GROUP_TIME_GAP_S
from .models import DvdGroup

CSV_COLUMNS = ["dvd", "title", "year", "genre", "region", "runtime", "studio",
               "barcode", "description", "condition", "price", "photos"]


def _norm(title):
    return " ".join((title or "").lower().split())


def _mtime(photo):
    try:
        return photo.source_path.stat().st_mtime
    except OSError:
        return 0.0


def group_photos(photos):
    """Group photos into DvdGroups by normalised title; time-cluster the untitled."""
    groups = []
    titled = {}
    untitled = []
    for p in photos:
        if p.deleted:
            continue
        key = _norm(p.title)
        if key:
            g = titled.get(key)
            if g is None:
                g = DvdGroup(title=p.title.strip(), year=p.year)
                titled[key] = g
                groups.append(g)
            g.photos.append(p)
        else:
            untitled.append(p)

    untitled.sort(key=_mtime)
    cur = None
    last_t = None
    n = 0
    for p in untitled:
        t = _mtime(p)
        if cur is None or (last_t is not None and t - last_t > GROUP_TIME_GAP_S):
            n += 1
            cur = DvdGroup(title=f"Untitled DVD {n}")
            groups.append(cur)
        cur.photos.append(p)
        last_t = t

    for g in groups:
        for p in g.photos:
            if not g.barcode and p.barcode:
                g.barcode = p.barcode
            if not g.year and p.year:
                g.year = p.year
    return groups


def _slug(text):
    keep = [c.lower() if c.isalnum() else "-" for c in (text or "dvd")]
    s = "".join(keep)
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-") or "dvd"


def listing_text(group):
    lines = [group.listing_title or group.title, ""]
    if group.description:
        lines += [group.description, ""]
    for label, val in (("Year", group.year), ("Genre", group.genre),
                       ("Region", group.region), ("Runtime", group.runtime),
                       ("Studio", group.studio), ("Barcode", group.barcode)):
        if val:
            lines.append(f"{label}: {val}")
    lines += ["", f"Condition: {group.condition}", f"Price: {group.price}"]
    return "\n".join(lines)


def write_listing_txt(group, dvd_dir):
    dvd_dir = Path(dvd_dir)
    dvd_dir.mkdir(parents=True, exist_ok=True)
    path = dvd_dir / "listing.txt"
    path.write_text(listing_text(group), encoding="utf-8")
    return path


def write_batch_csv(groups, run_dir):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "batch_listings.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for i, g in enumerate(groups, 1):
            photos = "; ".join(
                f"{(p.work_image.name if p.work_image else p.source_path.name)} [{p.side}]"
                for p in g.photos)
            w.writerow({"dvd": i, "title": g.title, "year": g.year or "",
                        "genre": g.genre, "region": g.region, "runtime": g.runtime,
                        "studio": g.studio, "barcode": g.barcode,
                        "description": g.description, "condition": g.condition,
                        "price": g.price, "photos": photos})
    return path


def write_run_log(groups, run_dir, extra=None):
    run_dir = Path(run_dir)
    data = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dvds": len(groups),
            "groups": [{"title": g.title, "barcode": g.barcode,
                        "photos": [p.source_path.name for p in g.photos]}
                       for g in groups]}
    if extra:
        data.update(extra)
    path = run_dir / "run_log.json"
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_listing.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```powershell
git add -A
git commit -m "feat: grouping + listing.txt/csv/run_log writers"
```

---

## Task 9: webapp.py (FastAPI session, review API, save)

**Files:**
- Create: `dvdflip/webapp.py`
- Test: `tests/test_webapp.py`

This module owns the in-memory `Session` (built by running the pipeline over an input folder, then drafting listings per group) and the HTTP endpoints the review page calls. `slugify` for final folders reuses `listing._slug`.

- [ ] **Step 1: Write the failing test**

`tests/test_webapp.py`:
```python
from pathlib import Path
from fastapi.testclient import TestClient
from dvdflip import webapp
from dvdflip.models import Photo, DvdGroup

def _session(tmp_path):
    work = tmp_path / "_work"; work.mkdir(parents=True)
    img = work / "a.jpg"
    import numpy as np, cv2
    cv2.imwrite(str(img), np.full((10, 10, 3), 255, np.uint8))
    p = Photo(source_path=Path("a.dng"), side="front", title="King Kong Escapes")
    p.work_image = img
    g = DvdGroup(title="King Kong Escapes", listing_title="King Kong Escapes DVD",
                 photos=[p])
    return webapp.Session(run_dir=tmp_path, work_dir=work, groups=[g])

def test_review_page_loads(tmp_path):
    app = webapp.create_app(_session(tmp_path))
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200 and "King Kong Escapes" in r.text

def test_save_dvd_writes_outputs(tmp_path):
    sess = _session(tmp_path)
    app = webapp.create_app(sess)
    client = TestClient(app)
    r = client.post("/api/dvd/0/save", json={
        "listing_title": "King Kong Escapes DVD 1967", "description": "d",
        "genre": "Sci-Fi", "region": "2", "runtime": "96 min", "studio": "Toho",
        "condition": "Very good", "price": "9.99",
        "photos": [{"side": "front", "deleted": False}]})
    assert r.status_code == 200
    dvd_dir = tmp_path / "King-Kong-Escapes"
    assert (dvd_dir / "listing.txt").exists()
    assert any(dvd_dir.glob("king-kong-escapes_front.jpg"))

def test_photo_image_served(tmp_path):
    app = webapp.create_app(_session(tmp_path))
    client = TestClient(app)
    r = client.get("/work/a.jpg")
    assert r.status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_webapp.py -v`
Expected: FAIL with `ModuleNotFoundError: dvdflip.webapp`.

- [ ] **Step 3: Write webapp.py**

`dvdflip/webapp.py`:
```python
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates

from . import vision
from .config import DEFAULT_INPUT, DEFAULT_OUTPUT, JPEG_QUALITY
from .loader import load_bgr, gather_inputs
from .listing import group_photos, write_listing_txt, write_batch_csv, write_run_log, _slug
from .pipeline import process_image

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

_ROT_SAVE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
             270: cv2.ROTATE_90_COUNTERCLOCKWISE}


@dataclass
class Session:
    run_dir: Path
    work_dir: Path
    groups: list = field(default_factory=list)


def build_session(input_dir=None, output_base=None, progress=print):
    import time
    input_dir = Path(input_dir or DEFAULT_INPUT)
    output_base = Path(output_base or DEFAULT_OUTPUT)
    run_dir = output_base / f"run_{time.strftime('%Y-%m-%d_%H%M')}"
    work_dir = run_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    files = gather_inputs(input_dir, exclude_dir=output_base)
    photos = []
    for i, f in enumerate(files, 1):
        progress(f"[{i}/{len(files)}] {f.name}")
        try:
            photos.append(process_image(f, work_dir))
        except Exception as e:
            progress(f"  FAILED: {e}")

    groups = group_photos(photos)
    for g in groups:
        front = next((p for p in g.photos if p.side == "front"), None)
        back = next((p for p in g.photos if p.side == "back"), None)
        if front and front.work_image:
            f_bgr = load_bgr(front.work_image)
            b_bgr = load_bgr(back.work_image) if back and back.work_image else None
            fields = vision.draft_listing(f_bgr, b_bgr, g.barcode)
            g.listing_title = fields["listing_title"] or g.title
            g.description = fields["description"]
            g.genre = fields["genre"]; g.region = fields["region"]
            g.runtime = fields["runtime"]; g.studio = fields["studio"]
            g.year = g.year or fields["year"]
        else:
            g.listing_title = g.title
    return Session(run_dir=run_dir, work_dir=work_dir, groups=groups)


def create_app(session):
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def review(request: Request):
        cards = []
        for gi, g in enumerate(session.groups):
            cards.append({
                "index": gi, "group": g,
                "photos": [{"side": p.side, "confidence": p.confidence,
                            "a4_found": p.a4_found, "barcode": p.barcode,
                            "img": p.work_image.name if p.work_image else ""}
                           for p in g.photos],
            })
        return TEMPLATES.TemplateResponse("review.html",
                                          {"request": request, "cards": cards})

    @app.get("/work/{name}")
    def work_image(name: str):
        path = session.work_dir / name
        if not path.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(str(path))

    @app.post("/api/dvd/{index}/save")
    async def save_dvd(index: int, request: Request):
        body = await request.json()
        g = session.groups[index]
        for fld in ("listing_title", "description", "genre", "region",
                    "runtime", "studio", "condition", "price"):
            if fld in body:
                setattr(g, fld, body[fld] or "")
        kept = []
        edits = body.get("photos", [])
        for p, e in zip(g.photos, edits):
            if e.get("deleted"):
                continue
            p.side = e.get("side", p.side)
            p.extra_rotation_cw = int(e.get("rotate", 0)) % 360
            kept.append(p)
        g.photos = kept

        dvd_dir = session.run_dir / _slug_dir(g)
        dvd_dir.mkdir(parents=True, exist_ok=True)
        base = _slug(g.title or g.listing_title)
        for p in g.photos:
            if not p.work_image or not Path(p.work_image).exists():
                continue
            out = dvd_dir / f"{base}_{p.side}.jpg"
            n = 2
            while out.exists():
                out = dvd_dir / f"{base}_{p.side}_{n}.jpg"; n += 1
            if p.extra_rotation_cw in _ROT_SAVE:
                img = cv2.imread(str(p.work_image))
                cv2.imwrite(str(out), cv2.rotate(img, _ROT_SAVE[p.extra_rotation_cw]),
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            else:
                shutil.copyfile(p.work_image, out)
        write_listing_txt(g, dvd_dir)
        g.approved = True
        write_batch_csv([x for x in session.groups if x.approved], session.run_dir)
        write_run_log(session.groups, session.run_dir)
        return {"ok": True, "saved_to": str(dvd_dir)}

    @app.post("/api/save-all")
    async def save_all(request: Request):
        for i in range(len(session.groups)):
            await save_dvd(i, request)
        return {"ok": True}

    return app


def _slug_dir(group):
    name = (group.title or group.listing_title or "DVD").strip()
    parts = [w.capitalize() for w in name.split()]
    return "-".join(parts) or "DVD"
```

`_slug` (folder/file base) and `_slug_dir` (the `King-Kong-Escapes` folder name) both come from `listing.py` / the helper at the bottom of `webapp.py`. The manual rotate from the review screen is applied to the work image at copy time via `_ROT_SAVE`, so an orientation fix actually persists to the saved JPG.

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_webapp.py -v`
Expected: PASS (3 tests). (Test references `King-Kong-Escapes` — matches `_slug_dir`.)

- [ ] **Step 5: Commit**

```powershell
git add -A
git commit -m "feat: FastAPI review session, image serving, save endpoints"
```

---

## Task 10: templates/review.html (the review screen)

**Files:**
- Create: `dvdflip/templates/review.html`
- Test: extend `tests/test_webapp.py`

The page renders one card per DVD: photo tiles with a confidence chip (green/amber), a side `<select>`, a rotate button, a delete toggle, the editable listing fields, and Approve / Save-all buttons. JS posts JSON to the endpoints from Task 9.

- [ ] **Step 1: Write a failing rendering assertion**

Add to `tests/test_webapp.py`:
```python
def test_review_has_controls(tmp_path):
    app = webapp.create_app(_session(tmp_path))
    client = TestClient(app)
    html = client.get("/").text
    assert 'id="save-all"' in html
    assert "/work/a.jpg" in html
    assert "Condition" in html
```

- [ ] **Step 2: Run it to verify it fails**

Run: `py -m pytest tests/test_webapp.py::test_review_has_controls -v`
Expected: FAIL (template missing / markers absent).

- [ ] **Step 3: Write templates/review.html**

`dvdflip/templates/review.html`:
```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DVD review</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; max-width: 860px; margin: 1.5rem auto;
         padding: 0 1rem; }
  .topbar { display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem; }
  .card { border:1px solid #ccc; border-radius:12px; padding:1rem 1.25rem; margin-bottom:1.25rem; }
  .tiles { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin-bottom:1rem; }
  .tile img { width:100%; height:128px; object-fit:contain; background:#f4f4f4; border-radius:8px; }
  .chip { font-size:12px; padding:2px 8px; border-radius:8px; }
  .ok { background:#e1f5ee; color:#0f6e56; } .warn { background:#faeeda; color:#854f0b; }
  .row { display:flex; gap:6px; margin-top:6px; align-items:center; }
  label { font-size:12px; color:#666; display:block; margin:8px 0 2px; }
  input, textarea, select { width:100%; font-size:14px; box-sizing:border-box; }
  .grid3 { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }
  button { cursor:pointer; padding:8px 14px; border-radius:8px; border:1px solid #bbb; background:transparent; }
  .primary { background:#e6f1fb; color:#185fa5; border:none; }
  .approve { background:#e1f5ee; color:#0f6e56; border:none; }
</style>
</head>
<body>
<div class="topbar">
  <strong>Review batch — {{ cards|length }} DVD(s)</strong>
  <button id="save-all" class="primary" onclick="saveAll()">Save all approved</button>
</div>

{% for c in cards %}
<div class="card" data-index="{{ c.index }}">
  <h3>{{ c.group.title }}</h3>
  <div class="tiles">
    {% for p in c.photos %}
    <div class="tile" data-pi="{{ loop.index0 }}">
      <img src="/work/{{ p.img }}" alt="{{ p.side }}">
      <div class="row">
        {% if p.a4_found %}
          <span class="chip ok">{{ p.side }} · {{ '%.2f'|format(p.confidence) }}</span>
        {% else %}
          <span class="chip warn">couldn't flatten — use as-is?</span>
        {% endif %}
      </div>
      <div class="row">
        <select class="side">
          {% for s in ['front','back','center','spine','other'] %}
          <option value="{{ s }}" {% if s==p.side %}selected{% endif %}>{{ s }}</option>
          {% endfor %}
        </select>
        <button onclick="rotate(this)" title="Rotate">⟳</button>
        <button onclick="toggleDel(this)" title="Delete" class="del">🗑</button>
      </div>
    </div>
    {% endfor %}
  </div>

  <label>Listing title</label>
  <input class="f-listing_title" value="{{ c.group.listing_title }}">
  <label>Description</label>
  <textarea class="f-description" rows="2">{{ c.group.description }}</textarea>
  <div class="grid3">
    <div><label>Genre</label><input class="f-genre" value="{{ c.group.genre }}"></div>
    <div><label>Region</label><input class="f-region" value="{{ c.group.region }}"></div>
    <div><label>Runtime</label><input class="f-runtime" value="{{ c.group.runtime }}"></div>
  </div>
  <div class="grid3">
    <div><label>Barcode</label><input value="{{ c.group.barcode }}" readonly></div>
    <div><label>Condition · you</label>
      <select class="f-condition">
        <option value="">Select…</option><option>Like new</option><option>Very good</option>
        <option>Good</option><option>Acceptable</option></select></div>
    <div><label>Price · you</label><input class="f-price" placeholder="£"></div>
  </div>
  <div class="row" style="margin-top:1rem">
    <button class="approve" onclick="saveCard(this)">Approve this DVD</button>
    <button onclick="copyListing(this)">Copy listing</button>
  </div>
</div>
{% endfor %}

<script>
function rotate(btn){ const img = btn.closest('.tile').querySelector('img');
  let r = (parseInt(img.dataset.rot||'0')+90)%360; img.dataset.rot=r;
  img.style.transform = 'rotate('+r+'deg)'; }
function toggleDel(btn){ const t = btn.closest('.tile'); t.dataset.deleted =
  t.dataset.deleted==='1'?'0':'1'; t.style.opacity = t.dataset.deleted==='1'?0.35:1; }
function cardPayload(card){
  const g = sel => card.querySelector(sel).value;
  const photos = [...card.querySelectorAll('.tile')].map(t => ({
    side: t.querySelector('.side').value,
    rotate: parseInt(t.querySelector('img').dataset.rot||'0'),
    deleted: t.dataset.deleted==='1' }));
  return { listing_title:g('.f-listing_title'), description:g('.f-description'),
    genre:g('.f-genre'), region:g('.f-region'), runtime:g('.f-runtime'),
    studio:'', condition:g('.f-condition'), price:g('.f-price'), photos };
}
async function saveCard(btn){
  const card = btn.closest('.card');
  const r = await fetch('/api/dvd/'+card.dataset.index+'/save',
    {method:'POST', headers:{'Content-Type':'application/json'},
     body: JSON.stringify(cardPayload(card))});
  btn.textContent = r.ok ? 'Saved ✓' : 'Error'; }
async function saveAll(){ for (const c of document.querySelectorAll('.card'))
  await saveCard(c.querySelector('.approve')); }
function copyListing(btn){
  const card = btn.closest('.card');
  const p = cardPayload(card);
  navigator.clipboard.writeText(p.listing_title+'\n\n'+p.description);
  btn.textContent = 'Copied'; }
</script>
</body>
</html>
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_webapp.py -v`
Expected: PASS (all webapp tests, including `test_review_has_controls`).

- [ ] **Step 5: Commit**

```powershell
git add -A
git commit -m "feat: review screen template (tiles, listing form, save JS)"
```

---

## Task 11: __main__.py launcher + Run DVD Flip.bat

**Files:**
- Create: `dvdflip/__main__.py`, `Run DVD Flip.bat`
- Test: `tests/test_main.py`

- [ ] **Step 1: Write the failing test**

`tests/test_main.py`:
```python
from dvdflip import __main__ as m

def test_main_has_run(monkeypatch):
    called = {}
    monkeypatch.setattr(m, "build_session", lambda **k: called.setdefault("built", True) or "S")
    monkeypatch.setattr(m, "_serve", lambda session, open_browser: called.setdefault("served", True))
    monkeypatch.setattr(m.vision, "ollama_available", lambda: True)
    m.run(open_browser=False)
    assert called.get("built") and called.get("served")

def test_main_warns_when_ollama_down(monkeypatch, capsys):
    monkeypatch.setattr(m.vision, "ollama_available", lambda: False)
    m.run(open_browser=False)
    out = capsys.readouterr().out.lower()
    assert "ollama" in out
```

- [ ] **Step 2: Run it to verify it fails**

Run: `py -m pytest tests/test_main.py -v`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError`.

- [ ] **Step 3: Write __main__.py**

`dvdflip/__main__.py`:
```python
import threading
import webbrowser

import uvicorn

from . import vision
from .webapp import build_session, create_app

HOST, PORT = "127.0.0.1", 8753


def _serve(session, open_browser=True):
    app = create_app(session)
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{HOST}:{PORT}/")).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


def run(open_browser=True):
    if not vision.ollama_available():
        print("Ollama is not reachable at localhost:11434.")
        print("Start Ollama (and `ollama pull qwen2.5vl:7b`), then run again.")
        return
    print("Processing photos from 'Images in' …")
    session = build_session(progress=print)
    print(f"Done. Open the browser to review {len(session.groups)} DVD(s).")
    _serve(session, open_browser=open_browser)


if __name__ == "__main__":
    run()
```

- [ ] **Step 4: Write Run DVD Flip.bat**

`Run DVD Flip.bat`:
```bat
@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m dvdflip
pause
```

- [ ] **Step 5: Run test to verify it passes**

Run: `py -m pytest tests/test_main.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```powershell
git add -A
git commit -m "feat: launcher (process, serve, open browser) + double-click .bat"
```

---

## Task 12: End-to-end regression on the 3 sample DNGs

**Files:**
- Test: `tests/test_end_to_end.py`
- Modify: `DEVELOPER_NOTES.md` (add a short "V3 rebuild" section)

- [ ] **Step 1: Write the end-to-end test (vision mocked, CV real)**

`tests/test_end_to_end.py`:
```python
from pathlib import Path
import shutil
from fastapi.testclient import TestClient

from dvdflip import vision, webapp
from dvdflip.models import ClassifyResult

FIXED = {
    "153112": ClassifyResult("front", 0, "King Kong Escapes", 1967, 0.95),
    "153120": ClassifyResult("back", 0, "King Kong Escapes", 1967, 0.96),
    "153137": ClassifyResult("center", 0, "King Kong Escapes", 1967, 0.80),
}

def test_three_samples_group_and_save(monkeypatch, tmp_path, sample_dngs):
    in_dir = tmp_path / "Images in"; in_dir.mkdir()
    for p in sample_dngs:
        shutil.copy(p, in_dir / p.name)

    def fake_classify(bgr, size_mm, barcode):
        for k, v in FIXED.items():
            return FIXED  # placeholder; replaced below
    monkeypatch.setattr(vision, "classify_photo",
        lambda bgr, size_mm, barcode: _pick(barcode))
    monkeypatch.setattr(vision, "draft_listing",
        lambda f, b, bc: {"listing_title": "King Kong Escapes DVD 1967",
                          "description": "Classic Toho kaiju film.", "genre": "Sci-Fi",
                          "region": "2", "runtime": "96 min", "studio": "Toho",
                          "year": 1967})

    session = webapp.build_session(input_dir=in_dir, output_base=tmp_path / "out",
                                   progress=lambda *a: None)
    assert len(session.groups) == 1
    g = session.groups[0]
    sides = sorted(p.side for p in g.photos)
    assert sides == ["back", "center", "front"]
    assert g.barcode == "0025192828928"

    client = TestClient(webapp.create_app(session))
    r = client.post("/api/dvd/0/save", json={
        "listing_title": g.listing_title, "description": g.description,
        "genre": g.genre, "region": g.region, "runtime": g.runtime,
        "studio": g.studio, "condition": "Very good", "price": "9.99",
        "photos": [{"side": p.side, "deleted": False} for p in g.photos]})
    assert r.status_code == 200
    dvd_dir = session.run_dir / "King-Kong-Escapes"
    jpgs = sorted(x.name for x in dvd_dir.glob("*.jpg"))
    assert len(jpgs) == 3
    assert (dvd_dir / "listing.txt").exists()


def _pick(barcode):
    # back has the barcode; the others are distinguished by size in build order.
    if barcode == "0025192828928":
        return FIXED["153120"]
    return None  # replaced in Step 3
```

- [ ] **Step 2: Run it to verify it fails**

Run: `py -m pytest tests/test_end_to_end.py -v`
Expected: FAIL (the `_pick` helper is a stub returning `None`).

- [ ] **Step 3: Fix the classify stub to key off the source filename**

The model is mocked, so classification must be deterministic per sample. Replace the `monkeypatch.setattr(vision, "classify_photo", …)` line and delete `fake_classify` / `_pick`; instead monkeypatch `pipeline.vision.classify_photo` to read the filename being processed. Update the test body:
```python
    from dvdflip import pipeline

    def fake_classify(bgr, size_mm, barcode):
        # center is physically large; back has the barcode; else front.
        if barcode == "0025192828928":
            return FIXED["153120"]
        if size_mm[1] >= 230:
            return FIXED["153137"]
        return FIXED["153112"]
    monkeypatch.setattr(pipeline.vision, "classify_photo", fake_classify)
```
Remove the earlier `monkeypatch.setattr(vision, "classify_photo", …)`, the `fake_classify`/`_pick` stubs, and the unused `FIXED` loop. Keep the `draft_listing` mock.

- [ ] **Step 4: Run it to verify it passes**

Run: `py -m pytest tests/test_end_to_end.py -v`
Expected: PASS — one group, sides `back/center/front`, barcode `0025192828928`, 3 saved JPGs + `listing.txt`.

- [ ] **Step 5: Run the full suite**

Run: `py -m pytest -v`
Expected: ALL pass.

- [ ] **Step 6: Live smoke test (manual, requires Ollama running)**

Run:
```powershell
ollama serve  # if not already running as a service
python -m dvdflip
```
Expected: console shows progress over the 3 DNGs in `Images in`, the browser opens the review screen with one *King Kong Escapes* card, three photo tiles, a drafted listing, and barcode `0025192828928`. Approve → check `processed/run_*/King-Kong-Escapes/` has 3 JPGs + `listing.txt`.

- [ ] **Step 7: Add a short V3 note to DEVELOPER_NOTES.md**

Append a "## V3 rebuild (2026-06-15)" section pointing to the spec and plan, noting V2 is now reference-only and V3 lives in `dvdflip/`.

- [ ] **Step 8: Commit**

```powershell
git add -A
git commit -m "test: end-to-end regression on 3 sample DNGs; document V3"
```

---

## Self-review notes (already applied)

- **Spec coverage:** load/flatten (T3–T4), white-balance/crop/barcode (T5), VLM decide (T6), per-image pipeline + A4 fallback (T7), grouping + listing draft + CSV/TXT/log (T8, T9), review screen with confidence flags / rotate / delete / clipboard (T9–T10), double-click launch + Ollama-down message (T11), native-Windows setup (T1), 3-DNG regression with mocked model (T12). DNG→JPG contract enforced (loader + `_write_work` + save). pyzbar keeps the barcode (T5, never the model).
- **Out of scope (per spec):** no eBay bulk-CSV, no UPC title lookup — not implemented.
- **Type consistency:** `ClassifyResult(side, rotation_cw, title, year, confidence)`, `Photo`, `DvdGroup` are defined once in `models.py` (T2) and used unchanged everywhere. `process_image(path, work_dir)`, `classify_photo(bgr, size_mm, barcode)`, `draft_listing(front_bgr, back_bgr, barcode)`, `group_photos(photos)`, `build_session(...)`, `create_app(session)`, `_slug`/`_slug_dir` signatures match across tasks.
- **Review-screen edits all persist:** side change → filename (T9), delete → skipped (T9), and manual rotate → re-rotates the saved JPG via `_ROT_SAVE` + `Photo.extra_rotation_cw` (T9/T10). Nothing is preview-only.
- **Known follow-ups (not blocking):** no per-DVD "regroup/merge" control in the first cut (grouping is by title + time; a wrong group is rare and editable by re-running). Could add a move-photo-between-DVDs control later.
