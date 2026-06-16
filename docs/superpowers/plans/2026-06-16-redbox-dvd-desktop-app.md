# Red-Box DVD Desktop App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fresh Tkinter desktop app (`redboxflip/`) that turns flat phone-scanner photos of DVD cases laid on a hand-drawn red-box template into clean 1:1 white-square eBay photos, reads the back-cover barcode to name files by title, and writes a per-run listing — automation-first, with intelligent fallback ladders and full manual override. No web/FastAPI.

**Architecture:** A small, single-purpose-module package. Classic CV finds the red box (ROI); a selectable AI engine (rembg or SAM box-prompt) mattes the case off the background, with GrabCut → geometric → manual fallbacks; the matte is feathered and composited on a pure-white square at true pixel scale (no re-warp = no distortion). The back shot is barcode-scanned at full resolution by three decoders; the decoded number resolves a title that names all three of a DVD's files. A Tkinter GUI offers fire-and-forget, a review grid, and a one-by-one editor — all in one window.

**Tech Stack:** Python 3, OpenCV (`opencv-contrib-python`), NumPy, Pillow, `pyzbar` + `zxing-cpp` + OpenCV barcode (decoders), `rembg`+`onnxruntime` and optional `mobile_sam` (cutout), Tkinter (GUI), pytest (tests). Runs in WSL2 Ubuntu via WSLg.

**Spec:** `docs/superpowers/specs/2026-06-16-redbox-dvd-desktop-app-design.md`

---

## File Structure

New package `redboxflip/` (the old `dvdflip/` is retired in Task 18):

| File | Responsibility |
|------|----------------|
| `redboxflip/__init__.py` | Package marker, version |
| `redboxflip/imaging.py` | Load JPG (EXIF transpose), BGR↔PIL, resize, save JPEG |
| `redboxflip/models.py` | `Face` enum, `Settings`, `ShotResult`, `DvdGroup` dataclasses |
| `redboxflip/config.py` | Settings + title-cache file paths, load/save settings |
| `redboxflip/detect.py` | Red-box ROI detection ladder, `order_points`, ROI bounds |
| `redboxflip/cutout.py` | Matte engines (rembg/SAM/GrabCut/geometric), feather, dispatcher |
| `redboxflip/clean.py` | Erase residual red, auto-upright, compose-on-white-square, colour tidy |
| `redboxflip/barcode.py` | Multi-decoder barcode scanning + dependency check |
| `redboxflip/titles.py` | Title cache, online lookup, resolution chain |
| `redboxflip/naming.py` | Filename builder, listing `.txt`/`.csv`, `run_log.json` |
| `redboxflip/pipeline.py` | `gather_inputs`, `assign_groups`, `process_shot`, `run_batch` |
| `redboxflip/gui/__init__.py` | GUI subpackage marker |
| `redboxflip/gui/editor.py` | Per-shot editor (V1 `CornerDialog` ported + extended) |
| `redboxflip/gui/grid.py` | Results grid grouped by DVD |
| `redboxflip/gui/app.py` | Main window, settings, Run, progress, wiring |
| `redboxflip/__main__.py` | Entry: GUI by default, headless CLI with `--input/--output` |
| `tests/test_rb_*.py` | Unit + end-to-end tests (prefixed `rb_` to avoid clashing with old `dvdflip` tests) |

Tests reuse the existing `tests/conftest.py`. New synthetic-image helpers live inline in the test files.

---

## Task 1: Package scaffold + dependencies

**Files:**
- Create: `redboxflip/__init__.py`
- Create: `redboxflip/gui/__init__.py`
- Create: `requirements-redbox.txt`
- Create: `tests/test_rb_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rb_smoke.py
def test_package_imports():
    import redboxflip
    assert redboxflip.__version__ == "0.1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rb_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'redboxflip'`

- [ ] **Step 3: Create the package files**

```python
# redboxflip/__init__.py
"""Red-box DVD photo processor for eBay (V1 successor)."""
__version__ = "0.1.0"
```

```python
# redboxflip/gui/__init__.py
"""Tkinter GUI for redboxflip."""
```

```text
# requirements-redbox.txt
numpy
opencv-contrib-python
pillow
pyzbar
zxing-cpp
rembg
onnxruntime
requests
pytest
# optional, only if the SAM cutout engine is used:
# mobile-sam
# optional, only for the OCR upright fallback:
# pytesseract
# test-only (generates barcodes for the decoder tests):
python-barcode
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rb_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Install dependencies**

Run (in the WSL venv): `pip install -r requirements-redbox.txt`
Also system libs: `sudo apt install -y python3-tk python3-pil.imagetk libzbar0`
Expected: installs without error; `python3 -c "import cv2, numpy, PIL, pyzbar.pyzbar, zxing_cpp"` prints nothing and exits 0.

- [ ] **Step 6: Commit**

```bash
git add redboxflip/__init__.py redboxflip/gui/__init__.py requirements-redbox.txt tests/test_rb_smoke.py
git commit -m "feat(redbox): package scaffold + dependency manifest"
```

---

## Task 2: Imaging helpers

**Files:**
- Create: `redboxflip/imaging.py`
- Test: `tests/test_rb_imaging.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rb_imaging.py
import numpy as np
from PIL import Image
from redboxflip import imaging


def test_bgr_pil_roundtrip():
    bgr = np.zeros((4, 6, 3), dtype=np.uint8)
    bgr[:, :, 2] = 255  # red in BGR
    pil = imaging.bgr_to_pil(bgr)
    assert pil.size == (6, 4)
    assert pil.getpixel((0, 0)) == (255, 0, 0)  # red in RGB
    back = imaging.pil_to_bgr(pil)
    assert np.array_equal(back, bgr)


def test_resize_max_keeps_aspect_and_caps_edge():
    bgr = np.zeros((100, 200, 3), dtype=np.uint8)
    out = imaging.resize_max(bgr, 100)
    assert max(out.shape[:2]) == 100
    assert out.shape[1] == 100 and out.shape[0] == 50


def test_resize_max_noop_when_small():
    bgr = np.zeros((10, 20, 3), dtype=np.uint8)
    out = imaging.resize_max(bgr, 100)
    assert out.shape == bgr.shape


def test_load_image_bgr_reads_jpg(tmp_path):
    p = tmp_path / "x.jpg"
    Image.new("RGB", (8, 5), (10, 20, 30)).save(p)
    bgr = imaging.load_image_bgr(p)
    assert bgr.shape == (5, 8, 3)
    # PIL (10,20,30) RGB -> BGR (30,20,10)
    assert tuple(int(c) for c in bgr[0, 0]) == (30, 20, 10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rb_imaging.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'redboxflip.imaging'`

- [ ] **Step 3: Write the implementation**

```python
# redboxflip/imaging.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rb_imaging.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add redboxflip/imaging.py tests/test_rb_imaging.py
git commit -m "feat(redbox): imaging helpers (load/convert/resize/save)"
```

---

## Task 3: Data models

**Files:**
- Create: `redboxflip/models.py`
- Test: `tests/test_rb_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rb_models.py
from redboxflip.models import Face, FACE_ORDER, FACE_FILE_LABEL, Settings, ShotResult, DvdGroup


def test_face_order_is_back_front_inside():
    assert FACE_ORDER == [Face.BACK, Face.FRONT, Face.INSIDE]


def test_face_file_labels():
    assert FACE_FILE_LABEL[Face.BACK] == "Back Cover"
    assert FACE_FILE_LABEL[Face.FRONT] == "Front Cover"
    assert FACE_FILE_LABEL[Face.INSIDE] == "Inside"


def test_settings_defaults():
    s = Settings()
    assert s.cutout_engine == "rembg"
    assert s.default_region == "Region 4 (PAL, Australia)"
    assert s.colour_tidy is True
    assert s.title_lookup is True
    assert s.feather_px == 3
    assert s.margin_pct == 6.0
    assert s.jpeg_quality == 92
    assert s.max_edge_px == 1600


def test_shotresult_and_group_roundtrip_to_dict():
    r = ShotResult(input_path="a.jpg", face=Face.BACK, barcode="123")
    d = r.to_dict()
    assert d["face"] == "back" and d["barcode"] == "123"
    g = DvdGroup(index=1, barcode="123", title="The Matrix",
                 region="Region 4 (PAL, Australia)", shots=[r])
    gd = g.to_dict()
    assert gd["title"] == "The Matrix" and gd["shots"][0]["face"] == "back"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rb_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'redboxflip.models'`

- [ ] **Step 3: Write the implementation**

```python
# redboxflip/models.py
"""Core data types passed between pipeline stages and the GUI."""
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class Face(str, Enum):
    BACK = "back"
    FRONT = "front"
    INSIDE = "inside"


# Fixed capture order: back first (barcode), then front, then inside.
FACE_ORDER = [Face.BACK, Face.FRONT, Face.INSIDE]

FACE_FILE_LABEL = {
    Face.BACK: "Back Cover",
    Face.FRONT: "Front Cover",
    Face.INSIDE: "Inside",
}


@dataclass
class Settings:
    input_dir: str = ""
    output_dir: str = ""
    cutout_engine: str = "rembg"            # "rembg" | "sam" | "grabcut" | "geometric"
    rembg_model: str = "isnet-general-use"
    sam_checkpoint: str = ""                # path to MobileSAM weights, if using SAM
    feather_px: int = 3
    margin_pct: float = 6.0
    jpeg_quality: int = 92
    max_edge_px: int = 1600
    default_region: str = "Region 4 (PAL, Australia)"
    colour_tidy: bool = True
    title_lookup: bool = True
    auto_open_output: bool = True
    auto_open_grid: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        known = {k: d[k] for k in cls().__dict__ if k in d}
        return cls(**known)


@dataclass
class ShotResult:
    input_path: str
    face: Face
    output_path: Optional[str] = None
    barcode: Optional[str] = None
    title: Optional[str] = None
    region: Optional[str] = None
    detect_method: str = ""
    detect_conf: float = 0.0
    cutout_method: str = ""
    rotation: int = 0
    status: str = "ok"                      # "ok" | "failed" | "needs_review"
    error: str = ""
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["face"] = self.face.value
        return d


@dataclass
class DvdGroup:
    index: int
    barcode: Optional[str]
    title: str
    region: str
    shots: list = field(default_factory=list)   # list[ShotResult]

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "barcode": self.barcode,
            "title": self.title,
            "region": self.region,
            "shots": [s.to_dict() for s in self.shots],
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rb_models.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add redboxflip/models.py tests/test_rb_models.py
git commit -m "feat(redbox): core data models (Face, Settings, ShotResult, DvdGroup)"
```

---

## Task 4: Config (settings + cache paths, load/save)

**Files:**
- Create: `redboxflip/config.py`
- Test: `tests/test_rb_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rb_config.py
from redboxflip import config
from redboxflip.models import Settings


def test_load_returns_defaults_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "nope.json")
    s = config.load_settings()
    assert isinstance(s, Settings)
    assert s.cutout_engine == "rembg"


def test_save_then_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "s.json")
    s = Settings(input_dir="/in", output_dir="/out", margin_pct=10.0,
                 cutout_engine="sam")
    config.save_settings(s)
    loaded = config.load_settings()
    assert loaded.input_dir == "/in"
    assert loaded.margin_pct == 10.0
    assert loaded.cutout_engine == "sam"


def test_corrupt_file_falls_back_to_defaults(tmp_path, monkeypatch):
    p = tmp_path / "s.json"
    p.write_text("{not json")
    monkeypatch.setattr(config, "SETTINGS_PATH", p)
    s = config.load_settings()
    assert s.cutout_engine == "rembg"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rb_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'redboxflip.config'`

- [ ] **Step 3: Write the implementation**

```python
# redboxflip/config.py
"""Persisted settings and on-disk paths."""
import json
from pathlib import Path

from .models import Settings

HOME = Path.home()
SETTINGS_PATH = HOME / ".redboxflip.json"
TITLE_CACHE_PATH = HOME / ".redboxflip_titles.json"


def load_settings() -> Settings:
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return Settings.from_dict(data)
    except Exception:
        return Settings()


def save_settings(settings: Settings) -> None:
    try:
        SETTINGS_PATH.write_text(json.dumps(settings.to_dict(), indent=2),
                                 encoding="utf-8")
    except Exception:
        pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rb_config.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add redboxflip/config.py tests/test_rb_config.py
git commit -m "feat(redbox): settings load/save + cache paths"
```

---

## Task 5: Red-box detection

**Files:**
- Create: `redboxflip/detect.py`
- Test: `tests/test_rb_detect.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rb_detect.py
import cv2
import numpy as np

from redboxflip import detect


def make_redbox_image(w=600, h=400, box=(80, 60, 520, 340), case=(120, 100, 480, 300)):
    """White sheet, a red rectangle outline, and a dark 'case' filling most of it."""
    img = np.full((h, w, 3), 255, np.uint8)
    bx0, by0, bx1, by1 = box
    cv2.rectangle(img, (bx0, by0), (bx1, by1), (0, 0, 255), 6)   # red (BGR)
    cx0, cy0, cx1, cy1 = case
    cv2.rectangle(img, (cx0, cy0), (cx1, cy1), (40, 30, 30), -1)  # dark case
    return img


def test_red_mask_lights_up_the_outline():
    img = make_redbox_image()
    mask = detect.red_mask(img)
    assert mask.dtype == np.uint8
    assert (mask > 0).sum() > 500          # the red outline is detected


def test_find_red_box_locates_the_drawn_box():
    img = make_redbox_image(box=(80, 60, 520, 340))
    quad, conf, method = detect.find_red_box(img)
    assert quad is not None
    assert method.startswith("red")
    assert conf > 0.0
    x0, y0 = quad[:, 0].min(), quad[:, 1].min()
    x1, y1 = quad[:, 0].max(), quad[:, 1].max()
    assert abs(x0 - 80) < 20 and abs(y0 - 60) < 20
    assert abs(x1 - 520) < 20 and abs(y1 - 340) < 20


def test_find_red_box_falls_back_to_whole_image_when_no_red():
    img = np.full((400, 600, 3), 255, np.uint8)
    quad, conf, method = detect.find_red_box(img)
    assert method == "whole-image"
    assert conf == 0.0
    assert quad[:, 0].max() == 599 and quad[:, 1].max() == 399


def test_roi_bounds_insets_inside_the_box():
    quad = np.array([[80, 60], [520, 60], [520, 340], [80, 340]], np.float32)
    x0, y0, x1, y1 = detect.roi_bounds(quad, (400, 600), inset=10)
    assert (x0, y0, x1, y1) == (90, 70, 510, 330)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rb_detect.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'redboxflip.detect'`

- [ ] **Step 3: Write the implementation**

```python
# redboxflip/detect.py
"""Find the hand-drawn red placement box and derive a region of interest.

Ladder: HSV red mask -> largest 4-point contour -> bounding box of all red ->
whole image. Each step returns a confidence and a method label so low-confidence
detections can be flagged for review.
"""
import cv2
import numpy as np

# Red wraps around the hue circle, so we need two ranges.
_RED_LOWER_1 = np.array([0, 80, 60], np.uint8)
_RED_UPPER_1 = np.array([10, 255, 255], np.uint8)
_RED_LOWER_2 = np.array([170, 80, 60], np.uint8)
_RED_UPPER_2 = np.array([180, 255, 255], np.uint8)

RED_INSET_PX = 6   # how far to crop inside the red line by default


def red_mask(bgr: np.ndarray) -> np.ndarray:
    """Binary mask (uint8 0/255) of red pixels."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, _RED_LOWER_1, _RED_UPPER_1) | \
        cv2.inRange(hsv, _RED_LOWER_2, _RED_UPPER_2)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)


def order_points(pts) -> np.ndarray:
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


def find_red_box(bgr: np.ndarray):
    """Return (quad 4x2 float32, confidence 0..1, method)."""
    h, w = bgr.shape[:2]
    img_area = float(h * w)
    mask = red_mask(bgr)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.05 * img_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and area > best_area:
            best, best_area = approx.reshape(4, 2).astype(np.float32), area
    if best is not None:
        conf = min(1.0, (best_area / img_area) / 0.6)
        return order_points(best), conf, "red-contour"

    ys, xs = np.where(mask > 0)
    if xs.size > 0.02 * img_area:
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
        return quad, 0.4, "red-bbox"

    quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    return quad, 0.0, "whole-image"


def roi_bounds(quad, shape, inset: int = RED_INSET_PX):
    """Axis-aligned (x0, y0, x1, y1) just inside the quad, clipped to the image."""
    h, w = shape[:2]
    x0 = max(0, int(round(quad[:, 0].min())) + inset)
    y0 = max(0, int(round(quad[:, 1].min())) + inset)
    x1 = min(w, int(round(quad[:, 0].max())) - inset)
    y1 = min(h, int(round(quad[:, 1].max())) - inset)
    if x1 - x0 < 10 or y1 - y0 < 10:   # inset too aggressive; back off
        x0 = max(0, int(round(quad[:, 0].min())))
        y0 = max(0, int(round(quad[:, 1].min())))
        x1 = min(w, int(round(quad[:, 0].max())))
        y1 = min(h, int(round(quad[:, 1].max())))
    return x0, y0, x1, y1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rb_detect.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add redboxflip/detect.py tests/test_rb_detect.py
git commit -m "feat(redbox): red-box ROI detection ladder"
```

---

## Task 6: Cutout engines + dispatcher

**Files:**
- Create: `redboxflip/cutout.py`
- Test: `tests/test_rb_cutout.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rb_cutout.py
import cv2
import numpy as np

from redboxflip import cutout
from tests.test_rb_detect import make_redbox_image


def test_geometric_matte_covers_the_case():
    img = make_redbox_image()
    roi = img[60:340, 80:520]
    alpha = cutout.geometric_matte(roi)
    assert alpha.shape == roi.shape[:2]
    cov = (alpha > 20).mean()
    assert 0.3 < cov < 0.99          # the dark case fills much (not all) of the ROI


def test_feather_softens_edges():
    alpha = np.zeros((50, 50), np.uint8)
    alpha[10:40, 10:40] = 255
    soft = cutout.feather_alpha(alpha, 3)
    edge_vals = np.unique(soft)
    assert len(edge_vals) > 2        # intermediate (feathered) values exist


def test_validate_coverage():
    assert cutout.validate_coverage(np.full((10, 10), 255, np.uint8)) is False  # 100% = whole frame
    good = np.zeros((10, 10), np.uint8); good[2:8, 2:8] = 255
    assert cutout.validate_coverage(good) is True
    assert cutout.validate_coverage(np.zeros((10, 10), np.uint8)) is False      # empty


def test_make_cutout_returns_rgba_tight_to_alpha():
    img = make_redbox_image()
    quad = np.array([[80, 60], [520, 60], [520, 340], [80, 340]], np.float32)
    rgba, method = cutout.make_cutout(img, quad, "geometric", feather_px=2)
    assert rgba.shape[2] == 4
    assert method == "geometric"
    # tight crop: alpha should touch all four edges
    a = rgba[:, :, 3]
    assert a[:, 0].max() > 0 or a[0, :].max() > 0


def test_make_cutout_falls_back_when_ai_engine_returns_none(monkeypatch):
    img = make_redbox_image()
    quad = np.array([[80, 60], [520, 60], [520, 340], [80, 340]], np.float32)
    monkeypatch.setattr(cutout, "rembg_matte", lambda roi, model=None: None)
    rgba, method = cutout.make_cutout(img, quad, "rembg", feather_px=0)
    assert method in ("grabcut", "geometric")   # fell through the ladder
    assert rgba.shape[2] == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rb_cutout.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'redboxflip.cutout'`

- [ ] **Step 3: Write the implementation**

```python
# redboxflip/cutout.py
"""Matte the DVD case off its background.

Engines, in fallback order per selection:
  rembg   : rembg -> grabcut -> geometric
  sam     : sam   -> grabcut -> geometric
  grabcut : grabcut -> geometric
  geometric: geometric only
The chosen AI engine runs only on the red-box ROI so surrounding paper cannot
confuse it. Output is an RGBA array cropped tight to the (feathered) alpha.
"""
import cv2
import numpy as np

from .detect import red_mask, roi_bounds

_REMBG_SESSIONS = {}
_SAM_PREDICTOR = None

ENGINE_LADDER = {
    "rembg": ["rembg", "grabcut", "geometric"],
    "sam": ["sam", "grabcut", "geometric"],
    "grabcut": ["grabcut", "geometric"],
    "geometric": ["geometric"],
}


def validate_coverage(alpha: np.ndarray, min_cov: float = 0.05,
                      max_cov: float = 0.985) -> bool:
    cov = float((alpha > 20).mean())
    return min_cov <= cov <= max_cov


def feather_alpha(alpha: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return alpha
    k = int(px) * 2 + 1
    return cv2.GaussianBlur(alpha, (k, k), 0)


def geometric_matte(roi_bgr: np.ndarray) -> np.ndarray:
    """Solid mask of the largest non-white, non-red blob (its convex hull)."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    not_white = gray < 235
    red = red_mask(roi_bgr) > 0
    fg = (not_white & ~red).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(roi_bgr.shape[:2], np.uint8)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(out, [cv2.convexHull(c)], -1, 255, -1)
    return out


def grabcut_matte(roi_bgr: np.ndarray) -> np.ndarray:
    """GrabCut seeded by a rectangle just inside the ROI edges."""
    h, w = roi_bgr.shape[:2]
    if h < 20 or w < 20:
        return None
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    inset_x, inset_y = max(2, w // 25), max(2, h // 25)
    rect = (inset_x, inset_y, w - 2 * inset_x, h - 2 * inset_y)
    try:
        cv2.grabCut(roi_bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0)
    return alpha.astype(np.uint8)


def rembg_matte(roi_bgr: np.ndarray, model: str = "isnet-general-use"):
    """Alpha matte from rembg, or None if rembg/model unavailable."""
    try:
        from rembg import remove, new_session
        from PIL import Image
    except Exception:
        return None
    try:
        if model not in _REMBG_SESSIONS:
            _REMBG_SESSIONS[model] = new_session(model)
        pil = Image.fromarray(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
        out = remove(pil, session=_REMBG_SESSIONS[model])
        return np.asarray(out.convert("RGBA").getchannel("A"))
    except Exception:
        return None


def sam_matte(roi_bgr: np.ndarray, checkpoint: str = ""):
    """Alpha matte from MobileSAM, prompted by a box covering the ROI.

    Returns None if mobile_sam or the checkpoint is unavailable.
    """
    global _SAM_PREDICTOR
    try:
        import torch
        from mobile_sam import sam_model_registry, SamPredictor
    except Exception:
        return None
    import os
    if not checkpoint or not os.path.exists(checkpoint):
        return None
    try:
        if _SAM_PREDICTOR is None:
            sam = sam_model_registry["vit_t"](checkpoint=checkpoint)
            sam.eval()
            _SAM_PREDICTOR = SamPredictor(sam)
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        _SAM_PREDICTOR.set_image(rgb)
        h, w = roi_bgr.shape[:2]
        box = np.array([2, 2, w - 2, h - 2])
        masks, scores, _ = _SAM_PREDICTOR.predict(box=box, multimask_output=True)
        best = masks[int(np.argmax(scores))]
        return (best.astype(np.uint8) * 255)
    except Exception:
        return None


def _run_engine(name, roi_bgr, model, checkpoint):
    if name == "rembg":
        return rembg_matte(roi_bgr, model)
    if name == "sam":
        return sam_matte(roi_bgr, checkpoint)
    if name == "grabcut":
        return grabcut_matte(roi_bgr)
    if name == "geometric":
        return geometric_matte(roi_bgr)
    return None


def make_cutout(bgr, quad, engine: str, feather_px: int,
                rembg_model: str = "isnet-general-use", sam_checkpoint: str = ""):
    """Return (rgba uint8 HxWx4 tight to alpha bbox, method)."""
    x0, y0, x1, y1 = roi_bounds(quad, bgr.shape)
    roi = bgr[y0:y1, x0:x1].copy()

    alpha, method = None, None
    for name in ENGINE_LADDER.get(engine, ["geometric"]):
        a = _run_engine(name, roi, rembg_model, sam_checkpoint)
        if a is not None and a.shape == roi.shape[:2] and validate_coverage(a):
            alpha, method = a, name
            break
    if alpha is None:
        alpha, method = geometric_matte(roi), "geometric"

    alpha = feather_alpha(alpha, feather_px)
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    rgba = np.dstack([rgb, alpha])

    ys, xs = np.where(alpha > 20)
    if xs.size and ys.size:
        rgba = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return rgba, method
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rb_cutout.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add redboxflip/cutout.py tests/test_rb_cutout.py
git commit -m "feat(redbox): cutout engines (rembg/sam/grabcut/geometric) + ladder"
```

---

## Task 7: Clean — erase red, upright, compose on white square, colour tidy

**Files:**
- Create: `redboxflip/clean.py`
- Test: `tests/test_rb_clean.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rb_clean.py
import numpy as np
from PIL import Image

from redboxflip import clean
from redboxflip.models import Face


def test_compose_on_white_square_is_square_centered_and_white_margin():
    rgba = np.zeros((20, 40, 4), np.uint8)
    rgba[..., 0] = 255          # red
    rgba[..., 3] = 255          # opaque
    out = clean.compose_on_white_square(rgba, margin_pct=0)
    assert out.size == (40, 40)            # square on the longest edge
    assert out.getpixel((0, 0)) == (255, 255, 255)        # top corner white
    assert out.getpixel((20, 20)) == (255, 0, 0)          # centre is the object


def test_compose_adds_margin():
    rgba = np.zeros((20, 20, 4), np.uint8)
    rgba[..., 3] = 255
    out = clean.compose_on_white_square(rgba, margin_pct=10)
    assert out.size == (24, 24)            # 20 + 2*(10% of 20)=2 each side


def test_auto_upright_rotates_wide_front_to_portrait():
    rgba = np.zeros((30, 60, 4), np.uint8)   # wide (landscape)
    out, rot = clean.auto_upright(rgba, Face.FRONT)
    assert out.shape[0] > out.shape[1]       # now portrait
    assert rot == 90


def test_auto_upright_leaves_inside_landscape():
    rgba = np.zeros((30, 60, 4), np.uint8)   # wide
    out, rot = clean.auto_upright(rgba, Face.INSIDE)
    assert rot == 0                          # inside shots stay landscape


def test_erase_red_makes_red_transparent():
    rgba = np.zeros((10, 10, 4), np.uint8)
    rgba[..., 3] = 255
    rgba[0:3, 0:3, 0] = 255      # a red patch (RGBA red = channel 0)
    out = clean.erase_red_to_white(rgba)
    assert out[0, 0, 3] == 0     # that patch is now transparent
    assert out[9, 9, 3] == 255   # non-red stays opaque
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rb_clean.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'redboxflip.clean'`

- [ ] **Step 3: Write the implementation**

```python
# redboxflip/clean.py
"""Post-matte cleanup: erase residual red, auto-upright, compose on white."""
import cv2
import numpy as np
from PIL import Image

from .detect import red_mask
from .models import Face


def erase_red_to_white(rgba: np.ndarray) -> np.ndarray:
    """Set any residual red pixels (in an RGBA array) to transparent."""
    rgb = rgba[:, :, :3]
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    m = red_mask(bgr)
    rgba = rgba.copy()
    rgba[m > 0, 3] = 0
    return rgba


def auto_upright(rgba: np.ndarray, face: Face, barcode_rot=None):
    """Rotate in 90° steps toward the expected orientation. Returns (rgba, deg).

    front/back are portrait (taller than wide); inside is landscape.
    A barcode_rot hint (the rotation at which the barcode decoded) is accepted
    for future use but the deterministic aspect rule drives the default.
    """
    h, w = rgba.shape[:2]
    rot = 0
    if face in (Face.FRONT, Face.BACK) and w > h:
        rot = 90
    elif face == Face.INSIDE and h > w:
        rot = 90
    if rot:
        rgba = np.ascontiguousarray(np.rot90(rgba, k=1))   # 90° counter-clockwise
    return rgba, rot


def compose_on_white_square(rgba, margin_pct: float) -> Image.Image:
    """Center a (feathered) RGBA cutout on a pure-white square canvas."""
    pil = Image.fromarray(rgba, "RGBA") if isinstance(rgba, np.ndarray) else rgba.convert("RGBA")
    w, h = pil.size
    side = max(w, h)
    margin = int(round(side * margin_pct / 100.0))
    canvas_side = side + 2 * margin
    canvas = Image.new("RGB", (canvas_side, canvas_side), "white")
    ox = (canvas_side - w) // 2
    oy = (canvas_side - h) // 2
    canvas.paste(pil, (ox, oy), pil.getchannel("A"))
    return canvas


def colour_tidy_bgr(bgr: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """Gentle white-balance + local contrast + mild saturation (ported from V1)."""
    if strength <= 0:
        return bgr
    original = bgr.astype(np.float32)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    paper = gray >= np.percentile(gray, 95)
    white_ref = original[paper].mean(axis=0) if np.any(paper) else original.reshape(-1, 3).mean(axis=0)
    cast = white_ref / max(white_ref.mean(), 1e-6)
    wb = np.clip(original / np.maximum(cast, 1e-6), 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(wb, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(l)
    enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
    hh, ss, vv = cv2.split(hsv)
    ss = np.clip(ss * 1.12, 0, 255)
    final = cv2.cvtColor(cv2.merge((hh, ss, vv)).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    return np.clip(original * (1 - strength) + final * strength, 0, 255).astype(np.uint8)


def colour_tidy_rgba(rgba: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """Apply colour tidy to the RGB of an RGBA array, preserving alpha."""
    rgb = rgba[:, :, :3]
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    tidied = colour_tidy_bgr(bgr, strength)
    out = rgba.copy()
    out[:, :, :3] = tidied[:, :, ::-1]
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rb_clean.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add redboxflip/clean.py tests/test_rb_clean.py
git commit -m "feat(redbox): erase-red, auto-upright, compose-on-white, colour tidy"
```

---

## Task 8: Barcode decoding (the crux)

**Files:**
- Create: `redboxflip/barcode.py`
- Test: `tests/test_rb_barcode.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rb_barcode.py
import numpy as np
import pytest

from redboxflip import barcode


def _make_ean13(tmp_path, digits12="590123412345"):
    bc = pytest.importorskip("barcode")
    from barcode.writer import ImageWriter
    ean = bc.get("ean13", digits12, writer=ImageWriter())
    path = tmp_path / "code"
    saved = ean.save(str(path))      # returns full path incl. extension
    return saved, ean.get_fullcode()


def test_available_decoders_lists_something():
    decoders = barcode.available_decoders()
    if not decoders:
        pytest.skip("no barcode decoder installed in this environment")
    assert isinstance(decoders, list)


def test_decode_reads_a_generated_ean13(tmp_path):
    if not barcode.available_decoders():
        pytest.skip("no barcode decoder installed")
    import cv2
    saved, fullcode = _make_ean13(tmp_path)
    bgr = cv2.imread(saved)
    digits, method, rot = barcode.decode(bgr)
    assert digits == fullcode


def test_decode_reads_rotated_barcode(tmp_path):
    if not barcode.available_decoders():
        pytest.skip("no barcode decoder installed")
    import cv2
    saved, fullcode = _make_ean13(tmp_path)
    bgr = cv2.imread(saved)
    rotated = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    digits, method, rot = barcode.decode(rotated)
    assert digits == fullcode


def test_decode_returns_none_on_blank():
    if not barcode.available_decoders():
        pytest.skip("no barcode decoder installed")
    blank = np.full((200, 300, 3), 255, np.uint8)
    digits, method, rot = barcode.decode(blank)
    assert digits is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rb_barcode.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'redboxflip.barcode'`

- [ ] **Step 3: Write the implementation**

```python
# redboxflip/barcode.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rb_barcode.py -v`
Expected: PASS (4 tests; some skip if no decoder/`python-barcode` installed — install them so they run)

- [ ] **Step 5: Commit**

```bash
git add redboxflip/barcode.py tests/test_rb_barcode.py
git commit -m "feat(redbox): robust multi-decoder barcode scanning"
```

---

## Task 9: Titles — cache, online lookup, resolution chain

**Files:**
- Create: `redboxflip/titles.py`
- Test: `tests/test_rb_titles.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rb_titles.py
from redboxflip import titles


def test_clean_title_strips_format_noise():
    assert titles.clean_title("The Matrix (DVD)") == "The Matrix"
    assert titles.clean_title("Alien - Blu-ray") == "Alien"
    assert titles.clean_title("Heat [DVD Region 4]") == "Heat"


def test_cache_get_set(tmp_path, monkeypatch):
    monkeypatch.setattr(titles, "TITLE_CACHE_PATH", tmp_path / "c.json")
    cache = titles.load_cache()
    assert cache == {}
    cache["123"] = "The Matrix"
    titles.save_cache(cache)
    assert titles.load_cache()["123"] == "The Matrix"


def test_resolve_prefers_manual(monkeypatch):
    monkeypatch.setattr(titles, "lookup_online", lambda *a, **k: "WRONG")
    title, source = titles.resolve_title("123", do_lookup=True, cache={},
                                         manual="My Title")
    assert title == "My Title" and source == "manual"


def test_resolve_uses_cache_before_network(monkeypatch):
    called = {"n": 0}
    def fake(*a, **k):
        called["n"] += 1
        return "NET"
    monkeypatch.setattr(titles, "lookup_online", fake)
    title, source = titles.resolve_title("123", do_lookup=True,
                                          cache={"123": "Cached"})
    assert title == "Cached" and source == "cache"
    assert called["n"] == 0


def test_resolve_falls_back_to_lookup_and_caches(monkeypatch):
    monkeypatch.setattr(titles, "lookup_online", lambda *a, **k: "From Web")
    cache = {}
    title, source = titles.resolve_title("999", do_lookup=True, cache=cache)
    assert title == "From Web" and source == "lookup"
    assert cache["999"] == "From Web"


def test_resolve_returns_empty_when_nothing(monkeypatch):
    monkeypatch.setattr(titles, "lookup_online", lambda *a, **k: "")
    title, source = titles.resolve_title("", do_lookup=True, cache={})
    assert title == "" and source == "none"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rb_titles.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'redboxflip.titles'`

- [ ] **Step 3: Write the implementation**

```python
# redboxflip/titles.py
"""Resolve a DVD title from a barcode: cache -> online lookup -> manual."""
import json
import re
import urllib.parse
import urllib.request

from .config import TITLE_CACHE_PATH

_NOISE = [
    r"\s*\((DVD|Blu-?ray|4K|UHD)[^)]*\)\s*$",
    r"\s*\[(DVD|Blu-?ray|4K|UHD)[^\]]*\]\s*$",
    r"\s*-\s*(DVD|Blu-?ray|4K|UHD)\s*$",
    r"\s*\bRegion\s*\d+.*$",
]


def clean_title(raw: str) -> str:
    if not raw:
        return ""
    text = raw.strip()
    for pat in _NOISE:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return text.strip(" -[]")


def load_cache() -> dict:
    try:
        return json.loads(TITLE_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_cache(cache: dict) -> None:
    try:
        TITLE_CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception:
        pass


def lookup_online(barcode: str, timeout: int = 8) -> str:
    """Best-effort title from the free upcitemdb trial endpoint. '' on failure."""
    if not barcode:
        return ""
    try:
        url = "https://api.upcitemdb.com/prod/trial/lookup?upc=" + urllib.parse.quote(barcode)
        req = urllib.request.Request(url, headers={"User-Agent": "redboxflip/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        items = data.get("items") or []
        if items:
            return clean_title(items[0].get("title") or "")
    except Exception:
        pass
    return ""


def resolve_title(barcode, *, do_lookup: bool, cache: dict, manual: str = None):
    """Return (title, source) where source is manual|cache|lookup|none."""
    if manual:
        cleaned = manual.strip()
        if barcode:
            cache[barcode] = cleaned
        return cleaned, "manual"
    if barcode and barcode in cache and cache[barcode]:
        return cache[barcode], "cache"
    if do_lookup and barcode:
        found = lookup_online(barcode)
        if found:
            cache[barcode] = found
            return found, "lookup"
    return "", "none"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rb_titles.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add redboxflip/titles.py tests/test_rb_titles.py
git commit -m "feat(redbox): title resolution chain (cache/lookup/manual)"
```

---

## Task 10: Naming — filenames, listing files, run log

**Files:**
- Create: `redboxflip/naming.py`
- Test: `tests/test_rb_naming.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rb_naming.py
from redboxflip import naming
from redboxflip.models import Face, ShotResult, DvdGroup


def test_safe_stem():
    assert naming.safe_stem("The Matrix: Reloaded") == "The Matrix Reloaded"
    assert naming.safe_stem("a/b\\c") == "a b c"
    assert naming.safe_stem("   ") == "dvd"


def test_output_filename():
    assert naming.output_filename("The Matrix", Face.BACK) == "The Matrix - Back Cover.jpg"
    assert naming.output_filename("The Matrix", Face.FRONT) == "The Matrix - Front Cover.jpg"
    assert naming.output_filename("The Matrix", Face.INSIDE) == "The Matrix - Inside.jpg"


def test_write_listing_creates_txt_and_csv(tmp_path):
    g = DvdGroup(index=1, barcode="9325336022306", title="Open Water",
                 region="Region 4 (PAL, Australia)",
                 shots=[ShotResult(input_path="b.jpg", face=Face.BACK,
                                   output_path="Open Water - Back Cover.jpg")])
    txt, csv_path = naming.write_listing(tmp_path, [g])
    assert txt.exists() and csv_path.exists()
    body = txt.read_text(encoding="utf-8")
    assert "Open Water" in body
    assert "9325336022306" in body
    assert "Region 4 (PAL, Australia)" in body


def test_write_run_log(tmp_path):
    g = DvdGroup(index=1, barcode=None, title="X", region="R", shots=[])
    p = naming.write_run_log(tmp_path, [g], {"engine": "rembg"})
    assert p.exists()
    import json
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["settings"]["engine"] == "rembg"
    assert data["dvds"][0]["title"] == "X"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rb_naming.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'redboxflip.naming'`

- [ ] **Step 3: Write the implementation**

```python
# redboxflip/naming.py
"""Filename building and per-run output files (listing txt/csv, run log)."""
import csv
import json
import re
import time
from pathlib import Path

from .models import Face, FACE_FILE_LABEL

_BAD = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_stem(text: str, max_len: int = 90) -> str:
    text = _BAD.sub(" ", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len] or "dvd"


def output_filename(title: str, face: Face) -> str:
    return f"{safe_stem(title)} - {FACE_FILE_LABEL[face]}.jpg"


def write_listing(run_dir, groups):
    run_dir = Path(run_dir)
    txt = run_dir / "dvd_listing.txt"
    lines = [
        "DVD listing",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M')}",
        f"DVDs: {len(groups)}",
        "",
    ]
    for g in groups:
        lines += [
            "=" * 50,
            f"DVD {g.index}",
            f"Name:    {g.title}",
            f"Region:  {g.region}",
            f"Barcode: {g.barcode or ''}",
            "Photos:",
        ]
        for s in g.shots:
            fname = Path(s.output_path).name if s.output_path else "(unsaved)"
            lines.append(f"  {fname}  [{s.face.value}]")
        lines += ["", "Condition: ", "Notes: ", "Price AUD: ", ""]
    txt.write_text("\n".join(lines), encoding="utf-8")

    csv_path = run_dir / "dvd_listing.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dvd", "name", "region", "barcode", "photos"])
        for g in groups:
            photos = "; ".join(
                f"{Path(s.output_path).name if s.output_path else '(unsaved)'} [{s.face.value}]"
                for s in g.shots)
            w.writerow([g.index, g.title, g.region, g.barcode or "", photos])
    return txt, csv_path


def write_run_log(run_dir, groups, settings_dict):
    run_dir = Path(run_dir)
    payload = {
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "settings": settings_dict,
        "dvds": [g.to_dict() for g in groups],
    }
    p = run_dir / "run_log.json"
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return p
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rb_naming.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add redboxflip/naming.py tests/test_rb_naming.py
git commit -m "feat(redbox): filename builder + listing/run-log writers"
```

---

## Task 11: Pipeline — gather, group, process a shot, run a batch

**Files:**
- Create: `redboxflip/pipeline.py`
- Test: `tests/test_rb_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rb_pipeline.py
from pathlib import Path

import cv2
import numpy as np

from redboxflip import pipeline
from redboxflip.models import Face, Settings
from tests.test_rb_detect import make_redbox_image


def test_gather_inputs_sorted_jpgs(tmp_path):
    (tmp_path / "b.jpg").write_bytes(b"x")
    (tmp_path / "a.JPG").write_bytes(b"x")
    (tmp_path / "note.txt").write_bytes(b"x")
    got = [p.name for p in pipeline.gather_inputs(tmp_path)]
    assert got == ["a.JPG", "b.jpg"]


def test_assign_groups_back_front_inside():
    paths = [Path(f"{i}.jpg") for i in range(6)]
    groups = pipeline.assign_groups(paths)
    assert len(groups) == 2
    assert [f for _, f in groups[0]] == [Face.BACK, Face.FRONT, Face.INSIDE]
    assert groups[1][0][0] == Path("3.jpg")


def test_assign_groups_handles_trailing_partial():
    paths = [Path(f"{i}.jpg") for i in range(4)]   # 3 + 1
    groups = pipeline.assign_groups(paths)
    assert len(groups) == 2
    assert [f for _, f in groups[1]] == [Face.BACK]


def test_process_shot_returns_square_image(tmp_path):
    img = make_redbox_image()
    p = tmp_path / "shot.jpg"
    cv2.imwrite(str(p), img)
    s = Settings(cutout_engine="geometric", colour_tidy=False, margin_pct=5,
                 max_edge_px=400)
    pil, result = pipeline.process_shot(p, Face.FRONT, s)
    assert pil.width == pil.height          # square
    assert pil.mode == "RGB"
    assert result.face == Face.FRONT
    assert result.cutout_method == "geometric"


def test_run_batch_end_to_end(tmp_path, monkeypatch):
    # three identical synthetic shots = one DVD
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    img = make_redbox_image()
    for name in ("01.jpg", "02.jpg", "03.jpg"):
        cv2.imwrite(str(in_dir / name), img)
    out_dir = tmp_path / "out"

    # no network, no real barcode: force a title
    monkeypatch.setattr(pipeline, "decode_barcode", lambda bgr: ("123", "test", 0))
    monkeypatch.setattr(pipeline, "resolve_title",
                        lambda bc, **k: ("Test DVD", "lookup"))

    s = Settings(input_dir=str(in_dir), output_dir=str(out_dir),
                 cutout_engine="geometric", colour_tidy=False, title_lookup=False,
                 max_edge_px=400)
    run_dir, groups = pipeline.run_batch(s)
    assert len(groups) == 1
    g = groups[0]
    assert g.title == "Test DVD"
    names = sorted(Path(sh.output_path).name for sh in g.shots)
    assert names == ["Test DVD - Back Cover.jpg",
                     "Test DVD - Front Cover.jpg",
                     "Test DVD - Inside.jpg"]
    assert all(Path(sh.output_path).exists() for sh in g.shots)
    assert (run_dir / "dvd_listing.txt").exists()
    assert (run_dir / "run_log.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rb_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'redboxflip.pipeline'`

- [ ] **Step 3: Write the implementation**

```python
# redboxflip/pipeline.py
"""Batch orchestration: gather -> group -> process each shot -> save + listing."""
import time
from pathlib import Path

from . import clean, cutout, detect, naming, titles
from .barcode import decode as decode_barcode
from .config import load_cache_for_batch
from .imaging import load_image_bgr, resize_max_pil, save_jpeg
from .models import Face, FACE_ORDER, ShotResult, DvdGroup
from .titles import resolve_title

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def gather_inputs(folder):
    folder = Path(folder)
    files = [p for p in folder.iterdir()
             if p.is_file() and p.suffix.lower() in SUPPORTED]
    return sorted(files, key=lambda p: p.name.lower())


def assign_groups(paths):
    """Slice the batch into [(path, Face)] groups of the fixed cycle."""
    groups = []
    for i in range(0, len(paths), len(FACE_ORDER)):
        chunk = paths[i:i + len(FACE_ORDER)]
        groups.append([(p, FACE_ORDER[j]) for j, p in enumerate(chunk)])
    return groups


def process_shot(path, face, settings, manual_quad=None, extra_rotation=0):
    """Run detect -> cutout -> clean -> compose. Returns (PIL RGB, ShotResult)."""
    t0 = time.perf_counter()
    bgr = load_image_bgr(path)

    if manual_quad is not None:
        quad, conf, det_method = manual_quad, 1.0, "manual"
    else:
        quad, conf, det_method = detect.find_red_box(bgr)

    rgba, cut_method = cutout.make_cutout(
        bgr, quad, settings.cutout_engine, settings.feather_px,
        rembg_model=settings.rembg_model, sam_checkpoint=settings.sam_checkpoint)
    rgba = clean.erase_red_to_white(rgba)

    barcode_digits = None
    if face == Face.BACK:
        barcode_digits, _m, _r = decode_barcode(bgr)

    rgba, rot = clean.auto_upright(rgba, face)
    rot = (rot + extra_rotation) % 360
    if extra_rotation:
        import numpy as np
        steps = (extra_rotation // 90) % 4
        for _ in range(steps):
            rgba = np.ascontiguousarray(np.rot90(rgba, k=-1))  # clockwise

    if settings.colour_tidy:
        rgba = clean.colour_tidy_rgba(rgba, 0.6)

    composed = clean.compose_on_white_square(rgba, settings.margin_pct)
    composed = resize_max_pil(composed, settings.max_edge_px)

    result = ShotResult(
        input_path=str(path), face=face, barcode=barcode_digits,
        detect_method=det_method, detect_conf=round(float(conf), 3),
        cutout_method=cut_method, rotation=rot, status="ok",
        elapsed_ms=int((time.perf_counter() - t0) * 1000),
    )
    return composed, result


def _make_run_dir(output_dir):
    base = Path(output_dir)
    run = base / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run.mkdir(parents=True, exist_ok=True)
    return run


def run_batch(settings, progress_cb=None):
    """Process every input, group into DVDs, save named files + listing + log."""
    paths = gather_inputs(settings.input_dir)
    grouped = assign_groups(paths)
    run_dir = _make_run_dir(settings.output_dir)
    cache = load_cache_for_batch()

    total = len(paths)
    done = 0
    dvd_groups = []

    for gi, shots in enumerate(grouped, 1):
        composed = {}     # Face -> (PIL, ShotResult)
        for path, face in shots:
            try:
                pil, res = process_shot(path, face, settings)
            except Exception as e:
                res = ShotResult(input_path=str(path), face=face,
                                 status="failed", error=str(e))
                pil = None
            composed[face] = (pil, res)
            done += 1
            if progress_cb:
                progress_cb(done, total, path.name)

        back = composed.get(Face.BACK)
        barcode = back[1].barcode if back else None
        title, _src = resolve_title(barcode, do_lookup=settings.title_lookup,
                                    cache=cache)
        if not title:
            title = barcode or f"Untitled DVD {gi}"

        dvd_dir = run_dir / naming.safe_stem(title)
        dvd_dir.mkdir(parents=True, exist_ok=True)
        group = DvdGroup(index=gi, barcode=barcode, title=title,
                         region=settings.default_region, shots=[])
        for face, (pil, res) in composed.items():
            res.title, res.region = title, group.region
            if pil is not None:
                out = dvd_dir / naming.output_filename(title, face)
                save_jpeg(pil, out, settings.jpeg_quality)
                res.output_path = str(out)
            group.shots.append(res)
        group.shots.sort(key=lambda s: FACE_ORDER.index(s.face))
        dvd_groups.append(group)

    titles.save_cache(cache)
    naming.write_listing(run_dir, dvd_groups)
    naming.write_run_log(run_dir, dvd_groups, settings.to_dict())
    return run_dir, dvd_groups
```

- [ ] **Step 4: Add the `load_cache_for_batch` helper to config**

In `redboxflip/config.py`, append:

```python
def load_cache_for_batch() -> dict:
    """Thin wrapper so the pipeline doesn't import titles at module load."""
    from .titles import load_cache
    return load_cache()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_rb_pipeline.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Commit**

```bash
git add redboxflip/pipeline.py redboxflip/config.py tests/test_rb_pipeline.py
git commit -m "feat(redbox): batch pipeline (gather/group/process/run_batch)"
```

---

## Task 12: CLI entry point + dependency check

**Files:**
- Create: `redboxflip/__main__.py`
- Test: `tests/test_rb_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rb_cli.py
import cv2
from pathlib import Path

from redboxflip import __main__ as cli
from tests.test_rb_detect import make_redbox_image


def test_build_parser_defaults():
    parser = cli.build_parser()
    args = parser.parse_args([])
    assert args.input is None and args.output is None


def test_run_headless_processes_a_folder(tmp_path, monkeypatch):
    in_dir = tmp_path / "in"; in_dir.mkdir()
    img = make_redbox_image()
    for n in ("1.jpg", "2.jpg", "3.jpg"):
        cv2.imwrite(str(in_dir / n), img)
    out_dir = tmp_path / "out"

    from redboxflip import pipeline
    monkeypatch.setattr(pipeline, "decode_barcode", lambda bgr: ("123", "t", 0))
    monkeypatch.setattr(pipeline, "resolve_title", lambda bc, **k: ("CLI DVD", "lookup"))

    code = cli.run_headless(str(in_dir), str(out_dir),
                            engine="geometric", colour_tidy=False)
    assert code == 0
    outputs = list((out_dir).rglob("*.jpg"))
    assert len(outputs) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rb_cli.py -v`
Expected: FAIL with `ImportError` / `AttributeError: module 'redboxflip.__main__' has no attribute 'build_parser'`

- [ ] **Step 3: Write the implementation**

```python
# redboxflip/__main__.py
"""Entry point: GUI by default, headless batch with --input/--output."""
import argparse
import sys

from .config import load_settings
from .models import Settings


def build_parser():
    p = argparse.ArgumentParser(
        prog="redboxflip",
        description="Red-box DVD photo processor for eBay.")
    p.add_argument("-i", "--input", help="input folder of scans")
    p.add_argument("-o", "--output", help="output folder")
    p.add_argument("--engine", default=None,
                   help="cutout engine: rembg|sam|grabcut|geometric")
    p.add_argument("--no-colour", action="store_true", help="disable colour tidy")
    return p


def run_headless(input_dir, output_dir, engine=None, colour_tidy=None):
    from . import pipeline
    s = load_settings()
    s.input_dir = input_dir
    s.output_dir = output_dir
    if engine:
        s.cutout_engine = engine
    if colour_tidy is not None:
        s.colour_tidy = colour_tidy
    run_dir, groups = pipeline.run_batch(
        s, progress_cb=lambda d, t, n: print(f"[{d}/{t}] {n}"))
    print(f"Done. {len(groups)} DVD(s). Output: {run_dir}")
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.input or args.output:
        if not (args.input and args.output):
            print("Both --input and --output are required for headless mode.",
                  file=sys.stderr)
            return 2
        return run_headless(args.input, args.output, engine=args.engine,
                            colour_tidy=(False if args.no_colour else None))
    # GUI mode
    from .gui.app import launch
    return launch()


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rb_cli.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add redboxflip/__main__.py tests/test_rb_cli.py
git commit -m "feat(redbox): CLI entry point + headless batch"
```

---

## Task 13: GUI — per-shot editor (port + extend V1 CornerDialog)

GUI tasks can't be meaningfully unit-tested headless (Tkinter needs a display, available here via WSLg). Each GUI task has a small logic test where possible plus a **manual smoke test** with explicit expected behaviour.

**Files:**
- Create: `redboxflip/gui/editor.py`
- Test: `tests/test_rb_editor_logic.py`

- [ ] **Step 1: Write the failing logic test**

```python
# tests/test_rb_editor_logic.py
from redboxflip.gui import editor


def test_scale_corners_roundtrip():
    # canvas-space <-> image-space corner mapping must round-trip
    corners_img = [(100, 50), (400, 50), (400, 300), (100, 300)]
    scale = 0.5
    canvas = editor.image_to_canvas(corners_img, scale)
    assert canvas[0] == (50, 25)
    back = editor.canvas_to_image(canvas, scale)
    assert back[2] == (400, 300)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rb_editor_logic.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'redboxflip.gui.editor'`

- [ ] **Step 3: Write the implementation**

```python
# redboxflip/gui/editor.py
"""Per-shot editor: drag the 4 crop corners (zoom magnifier), rotate, set face,
pick cutout engine + re-run matte, edit title/region, re-scan/hand-type barcode.

Ported and extended from V1's CornerDialog. Returns an edit dict the grid applies.
"""
import math

import numpy as np

from ..models import Face

try:
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk
    LANCZOS = getattr(Image, "Resampling", Image).LANCZOS
except Exception:   # headless test import
    tk = None


def image_to_canvas(corners, scale):
    return [(int(x * scale), int(y * scale)) for x, y in corners]


def canvas_to_image(corners, scale):
    return [(int(round(x / scale)), int(round(y / scale))) for x, y in corners]


# --- The dialog below is exercised by the manual smoke test (Step 5). ---

ENGINES = ["rembg", "sam", "grabcut", "geometric"]


class ShotEditor:
    HANDLE_R = 12
    GRAB = 36
    MAG_SRC = 70
    MAG_DISP = 200

    def __init__(self, root, pil_image, shot, settings):
        self.root = root
        self.original = pil_image.convert("RGB")
        self.shot = shot           # ShotResult (mutated on apply)
        self.settings = settings
        self.result = None         # dict of edits or None if cancelled
        self.extra_rotation = 0

        self.top = tk.Toplevel(root)
        self.top.title(f"Edit — {shot.input_path}")
        self.top.transient(root)
        self.top.grab_set()

        bar = ttk.Frame(self.top); bar.pack(side="bottom", fill="x", padx=10, pady=8)
        ttk.Button(bar, text="✓ Apply", command=self._apply).pack(side="left")
        ttk.Button(bar, text="Cancel", command=self._cancel).pack(side="left", padx=8)
        ttk.Button(bar, text="Reset corners",
                   command=self._reset_corners).pack(side="left", padx=8)

        opts = ttk.Frame(self.top); opts.pack(side="bottom", fill="x", padx=10, pady=4)
        ttk.Label(opts, text="Face:").grid(row=0, column=0, sticky="w")
        self.face_var = tk.StringVar(value=shot.face.value)
        for i, f in enumerate(Face):
            ttk.Radiobutton(opts, text=f.value, value=f.value,
                            variable=self.face_var).grid(row=0, column=1 + i, padx=4)
        ttk.Label(opts, text="Engine:").grid(row=1, column=0, sticky="w")
        self.engine_var = tk.StringVar(value=settings.cutout_engine)
        ttk.Combobox(opts, textvariable=self.engine_var, values=ENGINES,
                     state="readonly", width=12).grid(row=1, column=1, columnspan=2,
                                                       sticky="w", padx=4)
        ttk.Label(opts, text="Title:").grid(row=2, column=0, sticky="w")
        self.title_var = tk.StringVar(value=shot.title or "")
        ttk.Entry(opts, textvariable=self.title_var, width=36).grid(
            row=2, column=1, columnspan=4, sticky="w", padx=4)
        ttk.Label(opts, text="Barcode:").grid(row=3, column=0, sticky="w")
        self.barcode_var = tk.StringVar(value=shot.barcode or "")
        ttk.Entry(opts, textvariable=self.barcode_var, width=20).grid(
            row=3, column=1, columnspan=2, sticky="w", padx=4)

        rot = ttk.Frame(self.top); rot.pack(side="bottom", pady=4)
        ttk.Label(rot, text="Rotate:").pack(side="left")
        ttk.Button(rot, text="Left", command=lambda: self._rotate(-90)).pack(side="left", padx=3)
        ttk.Button(rot, text="Right", command=lambda: self._rotate(90)).pack(side="left", padx=3)
        ttk.Button(rot, text="180", command=lambda: self._rotate(180)).pack(side="left", padx=3)

        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        self.max_w, self.max_h = min(1100, sw - 120), min(620, sh - 360)
        self.canvas = tk.Canvas(self.top, bg="#202020", highlightthickness=0)
        self.canvas.pack(padx=10, pady=6)
        self.canvas.bind("<Button-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.dragging = None
        self._mag_tk = None
        self._rebuild()

    def _rebuild(self):
        pil = self.original if self.extra_rotation % 360 == 0 else \
            self.original.rotate(-self.extra_rotation, expand=True)
        w, h = pil.size
        self.scale = min(self.max_w / w, self.max_h / h, 1.0)
        dw, dh = max(100, int(w * self.scale)), max(100, int(h * self.scale))
        self._tk = ImageTk.PhotoImage(pil.resize((dw, dh), LANCZOS))
        self._disp = pil
        self.canvas.config(width=dw, height=dh)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk)
        self._reset_corners(redraw=False)
        self._redraw()

    def _reset_corners(self, redraw=True):
        w = int(self.canvas["width"]); h = int(self.canvas["height"])
        m = max(30, int(min(w, h) * 0.10))
        self.corners = [(m, m), (w - m, m), (w - m, h - m), (m, h - m)]
        if redraw:
            self._redraw()

    def _redraw(self):
        self.canvas.delete("quad")
        pts = self.corners + [self.corners[0]]
        for i in range(4):
            self.canvas.create_line(*pts[i], *pts[i + 1], fill="#ff3030",
                                    width=2, tags="quad")
        for x, y in self.corners:
            r = self.HANDLE_R
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="#ff3030",
                                    outline="white", width=2, tags="quad")

    def _closest(self, x, y):
        d = [math.hypot(x - cx, y - cy) for cx, cy in self.corners]
        i = min(range(4), key=lambda k: d[k])
        return i, d[i]

    def _press(self, e):
        i, dist = self._closest(e.x, e.y)
        self.dragging = i
        if dist >= self.GRAB:
            self.corners[i] = (e.x, e.y); self._redraw()
        self._magnify(e.x, e.y)

    def _drag(self, e):
        if self.dragging is None:
            return
        w = self.canvas.winfo_width(); h = self.canvas.winfo_height()
        self.corners[self.dragging] = (max(0, min(e.x, w - 1)), max(0, min(e.y, h - 1)))
        self._redraw(); self._magnify(*self.corners[self.dragging])

    def _release(self, _e):
        self.dragging = None
        self.canvas.delete("magnifier"); self._mag_tk = None

    def _magnify(self, x, y):
        try:
            pw, ph = self._disp.size
            ox, oy = x / max(self.scale, 1e-6), y / max(self.scale, 1e-6)
            half = self.MAG_SRC // 2
            crop = self._disp.crop((max(0, int(ox - half)), max(0, int(oy - half)),
                                    min(pw, int(ox + half)), min(ph, int(oy + half))))
            if crop.width < 8 or crop.height < 8:
                return
            region = crop.resize((self.MAG_DISP, self.MAG_DISP), Image.NEAREST)
            self._mag_tk = ImageTk.PhotoImage(region)
            self.canvas.delete("magnifier")
            self.canvas.create_image(x + 20, y - self.MAG_DISP - 20, anchor="nw",
                                    image=self._mag_tk, tags="magnifier")
            self.canvas.tag_raise("magnifier")
        except Exception:
            pass

    def _rotate(self, delta):
        self.extra_rotation = (self.extra_rotation + delta) % 360
        self._rebuild()

    def _apply(self):
        quad_img = canvas_to_image(self.corners, self.scale)
        self.result = {
            "quad": np.array(quad_img, dtype=np.float32),
            "face": Face(self.face_var.get()),
            "engine": self.engine_var.get(),
            "title": self.title_var.get().strip(),
            "barcode": self.barcode_var.get().strip() or None,
            "extra_rotation": self.extra_rotation,
        }
        self.top.destroy()

    def _cancel(self):
        self.result = None
        self.top.destroy()

    def get(self):
        self.root.wait_window(self.top)
        return self.result
```

- [ ] **Step 4: Run the logic test to verify it passes**

Run: `pytest tests/test_rb_editor_logic.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Manual smoke test**

Create `scripts/smoke_editor.py`:

```python
# scripts/smoke_editor.py
import tkinter as tk
from PIL import Image
from redboxflip.gui.editor import ShotEditor
from redboxflip.models import ShotResult, Face, Settings

root = tk.Tk(); root.withdraw()
img = Image.new("RGB", (600, 400), "white")
shot = ShotResult(input_path="demo.jpg", face=Face.FRONT, title="Demo")
ed = ShotEditor(root, img, shot, Settings())
print("RESULT:", ed.get())
```

Run: `python3 scripts/smoke_editor.py`
Expected: a window opens showing the image with 4 draggable red corner handles and a zoom magnifier that follows the cursor while dragging; Face/Engine/Title/Barcode controls and Rotate buttons are present. Clicking **Apply** prints a `RESULT:` dict with a `quad`, the selected face/engine/title; **Cancel** prints `RESULT: None`.

- [ ] **Step 6: Commit**

```bash
git add redboxflip/gui/editor.py tests/test_rb_editor_logic.py scripts/smoke_editor.py
git commit -m "feat(redbox): per-shot editor dialog (corner crop, rotate, engine, title)"
```

---

## Task 14: GUI — results grid

**Files:**
- Create: `redboxflip/gui/grid.py`
- Test: `tests/test_rb_grid_logic.py`

- [ ] **Step 1: Write the failing logic test**

```python
# tests/test_rb_grid_logic.py
from redboxflip.gui import grid
from redboxflip.models import ShotResult, Face, DvdGroup


def test_badge_text_reflects_status():
    g = DvdGroup(index=1, barcode="123", title="X", region="R", shots=[
        ShotResult(input_path="b.jpg", face=Face.BACK, barcode="123",
                   title="X", status="ok"),
    ])
    assert grid.badge_for(g) == "✓ title + barcode"

    g2 = DvdGroup(index=2, barcode=None, title="Untitled DVD 2", region="R",
                  shots=[ShotResult(input_path="b.jpg", face=Face.BACK)])
    assert grid.badge_for(g2) == "⚠ needs review"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rb_grid_logic.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'redboxflip.gui.grid'`

- [ ] **Step 3: Write the implementation**

```python
# redboxflip/gui/grid.py
"""Results grid grouped by DVD: thumbnails, status badges, click-to-edit,
step-through. Re-runs a shot through the pipeline when the editor applies edits.
"""
from pathlib import Path

from ..models import Face

try:
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk
    LANCZOS = getattr(Image, "Resampling", Image).LANCZOS
except Exception:
    tk = None


def badge_for(group) -> str:
    has_title = bool(group.title) and not group.title.startswith("Untitled DVD")
    if has_title and group.barcode:
        return "✓ title + barcode"
    if has_title:
        return "✓ title"
    return "⚠ needs review"


THUMB = 180


class ResultsGrid:
    def __init__(self, root, groups, settings, reprocess_cb):
        """reprocess_cb(shot, edits) -> (PIL, ShotResult); applied on editor save."""
        self.root = root
        self.groups = groups
        self.settings = settings
        self.reprocess_cb = reprocess_cb
        self._thumbs = []

        self.top = tk.Toplevel(root)
        self.top.title("Review results")
        self.top.geometry("1000x720")

        top_bar = ttk.Frame(self.top); top_bar.pack(fill="x", padx=10, pady=6)
        ttk.Button(top_bar, text="Step through all",
                   command=self._step_through).pack(side="left")
        ttk.Label(top_bar, text="Click any photo to fix it.").pack(side="left", padx=12)

        canvas = tk.Canvas(self.top, highlightthickness=0)
        scroll = ttk.Scrollbar(self.top, orient="vertical", command=canvas.yview)
        self.frame = ttk.Frame(canvas)
        self.frame.bind("<Configure>",
                        lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.frame, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self._render()

    def _render(self):
        for child in self.frame.winfo_children():
            child.destroy()
        self._thumbs.clear()
        for g in self.groups:
            box = ttk.LabelFrame(self.frame, text=f"DVD {g.index}: {g.title}  [{badge_for(g)}]")
            box.pack(fill="x", padx=8, pady=6)
            row = ttk.Frame(box); row.pack(fill="x", padx=6, pady=6)
            for shot in g.shots:
                cell = ttk.Frame(row); cell.pack(side="left", padx=8)
                img = self._load_thumb(shot.output_path)
                if img is not None:
                    self._thumbs.append(img)
                    btn = tk.Button(cell, image=img,
                                    command=lambda s=shot: self._edit(s))
                    btn.pack()
                ttk.Label(cell, text=shot.face.value).pack()

    def _load_thumb(self, path):
        if not path or not Path(path).exists():
            return None
        pil = Image.open(path).convert("RGB")
        pil.thumbnail((THUMB, THUMB), LANCZOS)
        return ImageTk.PhotoImage(pil)

    def _edit(self, shot):
        from .editor import ShotEditor
        pil = Image.open(shot.input_path).convert("RGB") \
            if Path(shot.input_path).exists() else Image.new("RGB", (600, 400), "white")
        editor = ShotEditor(self.root, pil, shot, self.settings)
        edits = editor.get()
        if edits:
            self.reprocess_cb(shot, edits)
            self._render()

    def _step_through(self):
        for g in self.groups:
            for shot in g.shots:
                self._edit(shot)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rb_grid_logic.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add redboxflip/gui/grid.py tests/test_rb_grid_logic.py
git commit -m "feat(redbox): results grid grouped by DVD with status badges"
```

---

## Task 15: GUI — main window + wiring

**Files:**
- Create: `redboxflip/gui/app.py`
- Test: `tests/test_rb_app_logic.py`

- [ ] **Step 1: Write the failing logic test**

```python
# tests/test_rb_app_logic.py
from redboxflip.gui import app
from redboxflip.models import Settings, ShotResult, Face


def test_apply_edits_to_settings_builds_overrides():
    s = Settings(cutout_engine="rembg")
    edits = {"engine": "sam", "extra_rotation": 90}
    over = app.settings_for_edit(s, edits)
    assert over.cutout_engine == "sam"
    # original unchanged
    assert s.cutout_engine == "rembg"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rb_app_logic.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'redboxflip.gui.app'`

- [ ] **Step 3: Write the implementation**

```python
# redboxflip/gui/app.py
"""Main Tkinter window: folders, settings, Run, progress, then the review grid.

All three flows live here: fire-and-forget (Run, auto-open output), review grid
(opens after a run), and one-by-one (Step through in the grid).
"""
import copy
import os
import queue
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path

from .. import pipeline
from ..config import load_settings, save_settings
from ..models import Settings

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except Exception:
    tk = None

ENGINES = ["rembg", "sam", "grabcut", "geometric"]


def settings_for_edit(settings: Settings, edits: dict) -> Settings:
    """Return a copy of settings with per-edit overrides (engine) applied."""
    over = replace(settings)
    if edits.get("engine"):
        over.cutout_engine = edits["engine"]
    return over


def _open_folder(path):
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)             # noqa: P204
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            # WSL: open in Windows Explorer if available, else xdg-open
            if subprocess.run(["which", "explorer.exe"], capture_output=True).returncode == 0:
                subprocess.Popen(["explorer.exe", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass


class App:
    def __init__(self, root):
        self.root = root
        self.settings = load_settings()
        self.msg_q = queue.Queue()
        self.run_dir = None
        self.groups = None

        root.title("Red-Box DVD Flip")
        root.geometry("760x560")

        # folders
        ff = ttk.LabelFrame(root, text="Folders"); ff.pack(fill="x", padx=10, pady=8)
        self.in_var = tk.StringVar(value=self.settings.input_dir)
        self.out_var = tk.StringVar(value=self.settings.output_dir)
        self._folder_row(ff, "Input", self.in_var, 0)
        self._folder_row(ff, "Output", self.out_var, 1)
        ff.columnconfigure(1, weight=1)

        # settings
        sf = ttk.LabelFrame(root, text="Settings"); sf.pack(fill="x", padx=10, pady=8)
        self.engine_var = tk.StringVar(value=self.settings.cutout_engine)
        ttk.Label(sf, text="Cutout engine").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Combobox(sf, textvariable=self.engine_var, values=ENGINES,
                     state="readonly", width=14).grid(row=0, column=1, sticky="w")
        self.region_var = tk.StringVar(value=self.settings.default_region)
        ttk.Label(sf, text="Region").grid(row=0, column=2, sticky="w", padx=6)
        ttk.Entry(sf, textvariable=self.region_var, width=24).grid(row=0, column=3, sticky="w")
        self.margin_var = tk.IntVar(value=int(self.settings.margin_pct))
        ttk.Label(sf, text="Margin %").grid(row=1, column=0, sticky="w", padx=6)
        ttk.Spinbox(sf, from_=0, to=30, textvariable=self.margin_var, width=6).grid(row=1, column=1, sticky="w")
        self.quality_var = tk.IntVar(value=self.settings.jpeg_quality)
        ttk.Label(sf, text="JPEG quality").grid(row=1, column=2, sticky="w", padx=6)
        ttk.Spinbox(sf, from_=70, to=100, textvariable=self.quality_var, width=6).grid(row=1, column=3, sticky="w")
        self.colour_var = tk.BooleanVar(value=self.settings.colour_tidy)
        ttk.Checkbutton(sf, text="Colour tidy", variable=self.colour_var).grid(row=2, column=0, sticky="w", padx=6)
        self.lookup_var = tk.BooleanVar(value=self.settings.title_lookup)
        ttk.Checkbutton(sf, text="Title lookup", variable=self.lookup_var).grid(row=2, column=1, sticky="w")

        # run
        rb = ttk.Frame(root); rb.pack(fill="x", padx=10, pady=8)
        self.run_btn = ttk.Button(rb, text="▶ Run", command=self._start)
        self.run_btn.pack(side="left")
        self.status = ttk.Label(rb, text="Ready"); self.status.pack(side="left", padx=12)

        pf = ttk.LabelFrame(root, text="Progress"); pf.pack(fill="both", expand=True, padx=10, pady=8)
        self.pb = ttk.Progressbar(pf, mode="determinate"); self.pb.pack(fill="x", padx=8, pady=6)
        self.log = tk.Text(pf, height=12); self.log.pack(fill="both", expand=True, padx=8, pady=6)

        root.after(80, self._pump)
        root.protocol("WM_DELETE_WINDOW", self._close)

    def _folder_row(self, parent, label, var, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=6)
        ttk.Button(parent, text="Browse",
                   command=lambda: var.set(filedialog.askdirectory() or var.get())
                   ).grid(row=row, column=2, padx=6)

    def _collect_settings(self) -> Settings:
        s = self.settings
        s.input_dir = self.in_var.get()
        s.output_dir = self.out_var.get()
        s.cutout_engine = self.engine_var.get()
        s.default_region = self.region_var.get()
        s.margin_pct = float(self.margin_var.get())
        s.jpeg_quality = int(self.quality_var.get())
        s.colour_tidy = bool(self.colour_var.get())
        s.title_lookup = bool(self.lookup_var.get())
        return s

    def _start(self):
        s = self._collect_settings()
        if not s.input_dir or not s.output_dir:
            messagebox.showwarning("Folders", "Choose input and output folders.")
            return
        save_settings(s)
        self.run_btn.config(state="disabled")
        self.log.delete("1.0", "end")
        threading.Thread(target=self._worker, args=(copy.copy(s),), daemon=True).start()

    def _worker(self, s):
        try:
            def cb(done, total, name):
                self.msg_q.put(("progress", (done, total, name)))
            run_dir, groups = pipeline.run_batch(s, progress_cb=cb)
            self.msg_q.put(("done", (run_dir, groups)))
        except Exception as e:
            self.msg_q.put(("error", str(e)))

    def _pump(self):
        try:
            while True:
                kind, payload = self.msg_q.get_nowait()
                if kind == "progress":
                    done, total, name = payload
                    self.pb["maximum"] = total; self.pb["value"] = done
                    self.status.config(text=f"{done}/{total}")
                    self.log.insert("end", f"[{done}/{total}] {name}\n"); self.log.see("end")
                elif kind == "done":
                    self.run_dir, self.groups = payload
                    self.status.config(text=f"Done. {len(self.groups)} DVD(s).")
                    self.run_btn.config(state="normal")
                    self._after_run()
                elif kind == "error":
                    self.status.config(text="Error")
                    self.log.insert("end", f"ERROR: {payload}\n")
                    self.run_btn.config(state="normal")
        except queue.Empty:
            pass
        self.root.after(80, self._pump)

    def _after_run(self):
        if self.settings.auto_open_output and self.run_dir:
            _open_folder(self.run_dir)
        if self.settings.auto_open_grid and self.groups:
            from .grid import ResultsGrid
            ResultsGrid(self.root, self.groups, self.settings, self._reprocess)

    def _reprocess(self, shot, edits):
        over = settings_for_edit(self.settings, edits)
        pil, new_res = pipeline.process_shot(
            Path(shot.input_path),
            edits.get("face", shot.face),
            over,
            manual_quad=edits.get("quad"),
            extra_rotation=edits.get("extra_rotation", 0),
        )
        # apply title/barcode overrides and re-save under the (possibly new) title
        title = edits.get("title") or shot.title or "Untitled"
        from .. import naming
        from ..imaging import save_jpeg
        out = Path(shot.output_path).parent / naming.output_filename(title, new_res.face)
        save_jpeg(pil, out, self.settings.jpeg_quality)
        shot.output_path = str(out)
        shot.face = new_res.face
        shot.title = title
        shot.barcode = edits.get("barcode", shot.barcode)
        shot.cutout_method = new_res.cutout_method

    def _close(self):
        save_settings(self._collect_settings())
        self.root.destroy()


def launch():
    if tk is None:
        print("Tkinter not available. Install: sudo apt install python3-tk python3-pil.imagetk",
              file=sys.stderr)
        return 1
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rb_app_logic.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Manual smoke test**

Run: `python3 -m redboxflip`
Expected: the main window opens with Folders, Settings (cutout engine dropdown, region, margin, quality, colour tidy, title lookup), a Run button, a progress bar and log. Choosing an input folder of test JPGs and an output folder, then Run, processes them, opens the output folder, and pops the review grid.

- [ ] **Step 6: Commit**

```bash
git add redboxflip/gui/app.py tests/test_rb_app_logic.py
git commit -m "feat(redbox): main window, threaded run, progress, grid wiring"
```

---

## Task 16: Dependency check at launch

**Files:**
- Modify: `redboxflip/__main__.py`
- Test: `tests/test_rb_depcheck.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rb_depcheck.py
from redboxflip import __main__ as cli


def test_startup_report_lists_decoders_and_engines():
    report = cli.startup_report()
    assert "Barcode decoders:" in report
    assert "Cutout:" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rb_depcheck.py -v`
Expected: FAIL with `AttributeError: module 'redboxflip.__main__' has no attribute 'startup_report'`

- [ ] **Step 3: Add the report and print it in main()**

In `redboxflip/__main__.py`, add:

```python
def startup_report() -> str:
    from .barcode import available_decoders
    decoders = available_decoders()
    try:
        import rembg  # noqa: F401
        rembg_ok = "available"
    except Exception:
        rembg_ok = "MISSING (cutout falls back to GrabCut)"
    lines = [
        f"Barcode decoders: {', '.join(decoders) if decoders else 'NONE — install pyzbar/zxing-cpp + libzbar0'}",
        f"Cutout: rembg {rembg_ok}; GrabCut/geometric always available",
    ]
    return "\n".join(lines)
```

Then at the top of `main()` (after parsing args), add:

```python
    print(startup_report(), file=sys.stderr)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rb_depcheck.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add redboxflip/__main__.py tests/test_rb_depcheck.py
git commit -m "feat(redbox): startup dependency report"
```

---

## Task 17: Sample fixtures + end-to-end regression

**Files:**
- Create: `samples/redbox/` (the four real template scans + two liked outputs)
- Create: `tests/test_rb_end_to_end.py`

- [ ] **Step 1: Add the real fixtures**

Copy the four template scans (`IMG_20260616_*.jpg`: blank template, front, back, inside) and the two liked output references into `samples/redbox/`. Name the back scan `back.jpg`, front `front.jpg`, inside `inside.jpg`.

Run: `ls samples/redbox/`
Expected: at least `back.jpg`, `front.jpg`, `inside.jpg` present.

- [ ] **Step 2: Write the end-to-end test (skips if fixtures absent)**

```python
# tests/test_rb_end_to_end.py
import shutil
from pathlib import Path

import pytest

SAMPLES = Path(__file__).resolve().parent.parent / "samples" / "redbox"


def _have_samples():
    return all((SAMPLES / n).exists() for n in ("back.jpg", "front.jpg", "inside.jpg"))


@pytest.mark.skipif(not _have_samples(), reason="redbox sample scans not present")
def test_real_batch_makes_three_square_jpgs(tmp_path):
    from PIL import Image
    from redboxflip import pipeline
    from redboxflip.models import Settings

    in_dir = tmp_path / "in"; in_dir.mkdir()
    for n in ("back.jpg", "front.jpg", "inside.jpg"):
        shutil.copy(SAMPLES / n, in_dir / n)
    out_dir = tmp_path / "out"

    s = Settings(input_dir=str(in_dir), output_dir=str(out_dir),
                 cutout_engine="grabcut", colour_tidy=True, title_lookup=False,
                 max_edge_px=1200)
    run_dir, groups = pipeline.run_batch(s)

    assert len(groups) == 1
    g = groups[0]
    assert len(g.shots) == 3
    for shot in g.shots:
        assert shot.output_path and Path(shot.output_path).exists()
        im = Image.open(shot.output_path)
        assert im.width == im.height            # square
        # corners are white (background replaced, no red)
        assert im.convert("RGB").getpixel((2, 2)) == (255, 255, 255)


@pytest.mark.skipif(not _have_samples(), reason="redbox sample scans not present")
def test_barcode_reads_from_real_back():
    import cv2
    from redboxflip import barcode
    if not barcode.available_decoders():
        pytest.skip("no barcode decoder installed")
    bgr = cv2.imread(str(SAMPLES / "back.jpg"))
    digits, method, rot = barcode.decode(bgr)
    assert digits and digits.isdigit() and len(digits) >= 8
```

- [ ] **Step 3: Run the end-to-end tests**

Run: `pytest tests/test_rb_end_to_end.py -v`
Expected: PASS (or SKIP if the fixtures haven't been added yet). If `test_barcode_reads_from_real_back` fails, that is the real-world signal to tune `barcode.py` (add scales/preprocs) — do that here until it passes on the real back scan.

- [ ] **Step 4: Commit**

```bash
git add samples/redbox tests/test_rb_end_to_end.py
git commit -m "test(redbox): real-scan end-to-end regression + barcode check"
```

---

## Task 18: Retire the old pipeline, update launcher + docs

**Files:**
- Modify: `Run DVD Flip.bat`
- Modify: `requirements.txt`
- Delete: `dvdflip/` and its tests (`tests/test_clean.py`, `test_end_to_end.py`, `test_flatten.py`, `test_listing.py`, `test_loader.py`, `test_main.py`, `test_models.py`, `test_pipeline.py`, `test_smoke.py`, `test_vision.py`, `test_webapp.py`)
- Create: `redboxflip/README.md`

- [ ] **Step 1: Confirm the new suite is green before deleting anything**

Run: `pytest tests/test_rb_*.py -v`
Expected: all PASS (end-to-end may SKIP without fixtures).

- [ ] **Step 2: Update the launcher to run the new app**

```bat
@echo off
REM Launch the Red-Box DVD Flip desktop app inside WSL2 / Ubuntu (WSLg shows the GUI).
wsl.exe -e bash -lc "cd '/mnt/c/Users/mardi/Documents/Ebay code' && source ~/ebay-venv/venv/bin/activate && python3 -m redboxflip"
pause
```

- [ ] **Step 3: Replace requirements.txt with the new app's deps**

Overwrite `requirements.txt` with the contents of `requirements-redbox.txt` (drop `rawpy`, `fastapi`, `uvicorn`, `jinja2`, `python-multipart`, `httpx`, `pillow-heif` — no longer used), then delete `requirements-redbox.txt`.

Run: `pip install -r requirements.txt`
Expected: installs clean.

- [ ] **Step 4: Remove the retired package and its tests**

```bash
git rm -r dvdflip
git rm tests/test_clean.py tests/test_end_to_end.py tests/test_flatten.py \
       tests/test_listing.py tests/test_loader.py tests/test_main.py \
       tests/test_models.py tests/test_pipeline.py tests/test_smoke.py \
       tests/test_vision.py tests/test_webapp.py
```

Note: `tests/conftest.py` stays (the new end-to-end test does not use its DNG fixtures, but leaving it is harmless).

- [ ] **Step 5: Write a short README**

```markdown
# redboxflip

Desktop app (Tkinter, runs in Ubuntu/WSLg) that turns red-box-template DVD
scans into 1:1 white-square eBay photos, named by title from the back-cover
barcode.

## Run
    source ~/ebay-venv/venv/bin/activate
    python3 -m redboxflip                 # GUI
    python3 -m redboxflip -i IN -o OUT    # headless batch

## Capture order
Per DVD, scan in this order: **Back, Front, Inside**. Back is first so its
barcode names all three files.

## Cutout engines
`rembg` (default) or `sam` (set a MobileSAM checkpoint in settings); both are
guided by the red box and fall back to GrabCut → geometric → manual.
```

Save as `redboxflip/README.md`.

- [ ] **Step 6: Run the full suite**

Run: `pytest -v`
Expected: only `test_rb_*` tests collected; all PASS (end-to-end may SKIP).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore(redbox): retire dvdflip, point launcher + requirements at redboxflip"
```

---

## Self-Review

**1. Spec coverage** — every spec section maps to a task:

- §2 capture protocol (Back→Front→Inside, 3/DVD, new back = new DVD) → Task 3 (`FACE_ORDER`), Task 11 (`assign_groups`).
- §3 automation-first + fallback ladders + manual override → detection ladder (Task 5), cutout ladder (Task 6), upright ladder (Task 7), barcode sweep (Task 8), title chain (Task 9), manual overrides in the editor (Task 13) and grid reprocess (Tasks 14–15).
- §4 module map → Tasks 2–15 create each module (`imaging.py` added as a small helper module — noted below).
- §5 pipeline stages A–F → Task 5 (B detect), Task 6 (C cutout + feather, no re-warp), Task 7 (D upright, E compose), Task 8 (F barcode), Task 11 ties them together.
- §6 barcode (full-res, three decoders, rotation/scale/preproc sweep, hard dep check, manual fallback) → Task 8 + Task 16 (dep report) + editor barcode field (Task 13).
- §7 title chain (cache → lookup → manual; LLM seam excluded) → Task 9; the `resolve_title(..., manual=)` arg is the seam point.
- §8 grouping/naming/outputs (per-DVD subfolder, `dvd_listing.txt`/`.csv`, `run_log.json`, region default) → Task 10 + Task 11.
- §9 one window, three flows → Tasks 13–15 (`auto_open_output` = fire-and-forget, `ResultsGrid` = review, `_step_through` = one-by-one).
- §10 deps/runtime/launcher → Task 1, Task 18.
- §11 error handling (degrade down ladders, per-image failure logged, never crash the batch) → `process_shot` try/except in Task 11; ladder fallbacks throughout.
- §12 testing (fixtures + unit + e2e) → every task's tests + Task 17.
- §13 non-goals → no FastAPI/web anywhere; AI segmentation in core, title-LLM excluded (Task 9 seam only).

**2. Placeholder scan** — no "TBD"/"add error handling"/"similar to Task N": every code step contains complete code; every test step contains real assertions. Task 17 Step 1 requires the user to drop in real image files (a data fixture, not a code placeholder) and the test SKIPs cleanly without them.

**3. Type consistency** — checked across tasks: `Face` enum values (`back/front/inside`) used consistently in models, naming (`FACE_FILE_LABEL`), pipeline (`FACE_ORDER`), editor, grid. `make_cutout(bgr, quad, engine, feather_px, rembg_model=, sam_checkpoint=)` signature matches its call in `process_shot`. `process_shot(path, face, settings, manual_quad=None, extra_rotation=0)` matches calls in the CLI test, grid reprocess, and app. `decode` returns `(digits, method, rot)` everywhere; `resolve_title(barcode, *, do_lookup, cache, manual=None)` matches its monkeypatched calls. `ShotResult`/`DvdGroup` field names match the listing/run-log writers.

**Deviation from spec noted:** the spec's §4 module list did not include `imaging.py`; it is added as a small single-responsibility helper (load/convert/resize/save) so the other modules stay focused. This is a refinement, not a scope change.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-16-redbox-dvd-desktop-app.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
