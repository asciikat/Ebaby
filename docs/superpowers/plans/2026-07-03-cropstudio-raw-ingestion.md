# Crop Studio RAW Ingestion (Plan 2 of 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let cropstudio accept `.DNG`/`.NEF` camera RAW uploads directly — the server decodes, white-balances off the surrounding paper, and enhances them (ported from the WSL pipeline's `Colorprodawgv1.py`) before the browser crop UI sees the shot, killing the manual WSL pre-processing step.

**Architecture:** One new leaf module `cropstudio/colorcorrect.py` holds the ported pure image math (white-balance mask/gains, S-curve/saturation/sharpen enhance) plus rawpy decode — rawpy is imported lazily so the math functions work even where rawpy is absent. One new endpoint `POST /rawdecode` in `app.py` turns uploaded RAW bytes into a corrected JPEG. The browser's existing RAW-extension guard (added 2026-07-03, commit 885a63c) flips from "fail loudly" to "upload to `/rawdecode`, load the returned JPEG into the canvas" — everything downstream (crop, save, barcode, title) is unchanged.

**Tech Stack:** Python (rawpy, OpenCV, FastAPI, pytest), vanilla JS in `cropstudio/static/index.html`.

**Verified context (2026-07-03):** the exact port source `Pipeline/Final/Colorprodawgv1.py` (WSL Debian) was run tonight on the 3 real DNGs in `Images in/` — all 3 converted clean, and the results passed through `service.save_dvd` correctly (barcode `0883316276402` decoded, Qwen read "GOOBER AND THE GHOST CHASERS"). The port below copies those proven functions verbatim where possible.

**Out of scope (noted findings, separate follow-ups):** the ~114 s worst-case `redboxflip.barcode.decode` sweep on barcode-less shots (needs a deadline cap — touches shared redboxflip code, not this plan); pywebview/PyInstaller packaging (Plan 3).

---

### Task 1: Install rawpy into the Windows venv

**Files:**
- Modify: `requirements-redbox.txt`

- [ ] **Step 1: Install the wheel**

```bash
cd "C:/Users/mardi/Documents/Ebay code"
".venv/Scripts/python.exe" -m pip install rawpy
```

Expected: installs a `cp313`-tagged Windows wheel (rawpy ships them; WSL Debian runs 0.27.0). If NO Windows wheel exists for Python 3.13, STOP — report to the user; do not attempt a source build (needs libraw toolchain).

- [ ] **Step 2: Verify import + a real decode**

```bash
".venv/Scripts/python.exe" -c "
import rawpy, glob
p = glob.glob('Images in/*.dng')[0]
with rawpy.imread(p) as raw:
    rgb = raw.postprocess(half_size=True, use_camera_wb=True)
print('decoded', rgb.shape)
"
```

Expected: prints something like `decoded (1536, 2040, 3)`.

- [ ] **Step 3: Add to requirements**

In `requirements-redbox.txt`, after the `onnxruntime` line, add:

```
rawpy
```

- [ ] **Step 4: Commit**

```bash
git add requirements-redbox.txt
git commit -m "feat(cropstudio): add rawpy dependency for RAW ingestion"
```

---

### Task 2: `colorcorrect.py` — white-balance + enhance math (no rawpy needed)

**Files:**
- Create: `cropstudio/colorcorrect.py`
- Test: `tests/test_cropstudio_colorcorrect.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cropstudio_colorcorrect.py
import numpy as np

from cropstudio import colorcorrect


def _bordered_image(border_bgr, center_bgr, size=200, border=40):
    """Synthetic photo: a colored 'paper' border around a 'DVD' center block."""
    img = np.full((size, size, 3), border_bgr, np.uint8)
    img[border:-border, border:-border] = center_bgr
    return img


def test_wb_gains_neutralize_a_warm_cast():
    # Paper border has a warm (red-heavy) cast; gains should push R down / B up.
    img = _bordered_image(border_bgr=(180, 200, 230), center_bgr=(30, 30, 30))
    gb, gg, gr = colorcorrect.get_white_balance_gains(img)
    assert gb > 1.0        # blue lifted
    assert gr < 1.0        # red pulled down
    corrected = colorcorrect.apply_white_balance(img, (gb, gg, gr))
    b, g, r = corrected[10, 10].astype(int)   # a border (paper) pixel
    assert abs(b - g) <= 6 and abs(g - r) <= 6   # near-neutral after correction


def test_wb_gains_are_clipped_to_sane_range():
    # Extreme cast must not blow out: gains stay within [0.5, 2.0].
    img = _bordered_image(border_bgr=(40, 120, 250), center_bgr=(0, 0, 0))
    gains = colorcorrect.get_white_balance_gains(img)
    assert all(0.5 <= g <= 2.0 for g in gains)


def test_wb_gains_neutral_image_stays_neutral():
    img = _bordered_image(border_bgr=(210, 210, 210), center_bgr=(50, 50, 50))
    gb, gg, gr = colorcorrect.get_white_balance_gains(img)
    assert abs(gb - 1.0) < 0.02 and abs(gg - 1.0) < 0.02 and abs(gr - 1.0) < 0.02


def test_surrounding_mask_ignores_white_inside_the_cover():
    # White block INSIDE the dark center must not join the border mask.
    img = _bordered_image(border_bgr=(220, 220, 220), center_bgr=(30, 30, 30))
    img[90:110, 90:110] = (250, 250, 250)     # white patch inside the "cover"
    mask = colorcorrect.get_surrounding_white_mask(img)
    assert mask[100, 100] == 0                # inner white patch excluded
    assert mask[10, 10] > 0                   # border paper included


def test_enhance_image_returns_same_shape_uint8():
    img = _bordered_image(border_bgr=(220, 220, 220), center_bgr=(90, 120, 150))
    out = colorcorrect.enhance_image(img)
    assert out.shape == img.shape and out.dtype == np.uint8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `".venv/Scripts/python.exe" -m pytest tests/test_cropstudio_colorcorrect.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cropstudio.colorcorrect'`

- [ ] **Step 3: Write the implementation (verbatim port of the proven WSL functions)**

```python
# cropstudio/colorcorrect.py
"""RAW decode + surrounding-white color correction, ported from the WSL
pipeline's Colorprodawgv1.py (proven on real DVD photos 2026-07-03).

The math functions (mask/gains/enhance) are pure OpenCV/NumPy. rawpy is
imported lazily inside the decode functions so this module loads — and the
math stays testable — even on an environment without the rawpy wheel.
"""
import io

import cv2
import numpy as np


# ----------------------------------------------------------------------
#  WHITE BALANCE  (uses the white paper SURROUNDING the DVD)
# ----------------------------------------------------------------------
def get_surrounding_white_mask(bgr_img):
    """Mask of the white paper that SURROUNDS the DVD.

    1. Find bright, low-saturation pixels (white paper).
    2. Keep only white blobs that TOUCH the image border — that's the
       background paper. White patches inside the cover art get ignored.
    """
    h, w = bgr_img.shape[:2]
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    kernel = np.ones((5, 5), np.uint8)

    # Try strict thresholds first, relax if not enough paper found
    for v_thr, s_thr in ((150, 45), (120, 60), (100, 80)):
        white = ((v > v_thr) & (s < s_thr)).astype(np.uint8)
        if white.sum() == 0:
            continue

        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel)
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel)

        num, labels = cv2.connectedComponents(white)

        border_labels = set()
        border_labels.update(labels[0, :].tolist())
        border_labels.update(labels[-1, :].tolist())
        border_labels.update(labels[:, 0].tolist())
        border_labels.update(labels[:, -1].tolist())
        border_labels.discard(0)

        if not border_labels:
            continue

        surrounding = np.isin(labels, list(border_labels)).astype(np.uint8) * 255

        # Need at least ~2% of the frame to trust it
        if (surrounding.sum() / 255) > (h * w * 0.02):
            return surrounding

    # Fallback: any white pixels, no border constraint
    return ((v > 120) & (s < 60)).astype(np.uint8) * 255


def get_white_balance_gains(bgr_img):
    """B, G, R multipliers that make the surrounding white paper neutral."""
    mask = get_surrounding_white_mask(bgr_img)
    if mask.sum() == 0:
        return 1.0, 1.0, 1.0

    mean_b = cv2.mean(bgr_img[:, :, 0], mask=mask)[0]
    mean_g = cv2.mean(bgr_img[:, :, 1], mask=mask)[0]
    mean_r = cv2.mean(bgr_img[:, :, 2], mask=mask)[0]

    if mean_b == 0 or mean_g == 0 or mean_r == 0:
        return 1.0, 1.0, 1.0

    target = (mean_b + mean_g + mean_r) / 3.0
    gain_b = float(np.clip(target / mean_b, 0.5, 2.0))
    gain_g = float(np.clip(target / mean_g, 0.5, 2.0))
    gain_r = float(np.clip(target / mean_r, 0.5, 2.0))
    return gain_b, gain_g, gain_r


def apply_white_balance(bgr_img, gains):
    b, g, r = cv2.split(bgr_img.astype("float32"))
    b *= gains[0]
    g *= gains[1]
    r *= gains[2]
    return np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
#  ENHANCEMENT  (slight contrast + slight saturation + mild sharpen)
# ----------------------------------------------------------------------
def create_s_curve_lut(strength=1.0):
    """S-curve contrast LUT. 0 = none, 1.0 = slight, 2.5 = strong."""
    if strength == 0.0:
        return np.arange(256, dtype=np.uint8)
    x = np.arange(256)
    nx = (x / 255.0 - 0.5) * 2
    y = 1 / (1 + np.exp(-strength * nx))
    y = (y - y.min()) / (y.max() - y.min()) * 255
    return np.clip(y, 0, 255).astype(np.uint8)


def enhance_image(bgr_img, contrast=1.0, saturation=1.1, sharpen=1.2):
    img_contrast = cv2.LUT(bgr_img, create_s_curve_lut(contrast))

    if saturation != 1.0:
        hsv = cv2.cvtColor(img_contrast, cv2.COLOR_BGR2HSV).astype("float32")
        hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
        img_color = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    else:
        img_color = img_contrast

    if sharpen > 1.0:
        blurred = cv2.GaussianBlur(img_color, (0, 0), 3.0)
        factor = sharpen - 1.0
        return cv2.addWeighted(img_color, 1.0 + factor, blurred, -factor, 0)
    return img_color
```

(The RAW decode functions land in Task 3 — this task is the rawpy-free math only.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `".venv/Scripts/python.exe" -m pytest tests/test_cropstudio_colorcorrect.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add cropstudio/colorcorrect.py tests/test_cropstudio_colorcorrect.py
git commit -m "feat(cropstudio): port surrounding-white WB + enhance from WSL pipeline"
```

---

### Task 3: RAW decode → corrected JPEG bytes

**Files:**
- Modify: `cropstudio/colorcorrect.py` (append)
- Test: `tests/test_cropstudio_colorcorrect.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cropstudio_colorcorrect.py`:

```python
import glob
import os

import cv2
import pytest

_REAL_DNGS = sorted(glob.glob(os.path.join("Images in", "*.dng")))


def test_raw_to_jpeg_rejects_non_raw_bytes():
    with pytest.raises(ValueError):
        colorcorrect.raw_to_jpeg(b"definitely not a raw file")


@pytest.mark.skipif(not _REAL_DNGS, reason="no real DNG fixture on this machine")
def test_raw_to_jpeg_decodes_a_real_dng():
    data = open(_REAL_DNGS[0], "rb").read()
    jpeg = colorcorrect.raw_to_jpeg(data)
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    # half_size decode of a 12MP phone DNG is still comfortably above the
    # browser canvas's 1200px working width
    assert max(img.shape[:2]) >= 1200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `".venv/Scripts/python.exe" -m pytest tests/test_cropstudio_colorcorrect.py -v -k raw_to_jpeg`
Expected: FAIL with `AttributeError: module 'cropstudio.colorcorrect' has no attribute 'raw_to_jpeg'`

- [ ] **Step 3: Append the decode functions**

Append to `cropstudio/colorcorrect.py`:

```python
# ----------------------------------------------------------------------
#  RAW decode (rawpy imported lazily — see module docstring)
# ----------------------------------------------------------------------
def _denoise_levels():
    import rawpy
    return {
        "off":    (rawpy.FBDDNoiseReductionMode.Off,  None,  0),
        "light":  (rawpy.FBDDNoiseReductionMode.Full, None,  1),
        "medium": (rawpy.FBDDNoiseReductionMode.Full, 100.0, 1),
        "strong": (rawpy.FBDDNoiseReductionMode.Full, 250.0, 2),
    }


def decode_raw(data: bytes, denoise: str = "light"):
    """RAW container bytes -> BGR ndarray. Raises ValueError on junk input.

    half_size=True quarters the pixel count (12MP -> 3MP, ~4x faster). The
    browser canvas works at 1200px wide, so nothing downstream loses detail.
    """
    import rawpy
    fbdd, noise_thr, median_passes = _denoise_levels().get(
        denoise, _denoise_levels()["light"])
    kwargs = dict(
        use_camera_wb=True,
        fbdd_noise_reduction=fbdd,
        median_filter_passes=median_passes,
        half_size=True,
    )
    if noise_thr is not None:
        kwargs["noise_thr"] = noise_thr
    try:
        with rawpy.imread(io.BytesIO(data)) as raw:
            rgb = raw.postprocess(**kwargs)
    except Exception as e:
        raise ValueError(f"could not decode RAW data: {e}") from e
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def raw_to_jpeg(data: bytes, denoise: str = "light", quality: int = 92) -> bytes:
    """Full pipeline: decode -> WB off surrounding paper -> enhance -> JPEG."""
    bgr = decode_raw(data, denoise)
    bgr = apply_white_balance(bgr, get_white_balance_gains(bgr))
    bgr = enhance_image(bgr)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("could not encode corrected image")
    return buf.tobytes()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `".venv/Scripts/python.exe" -m pytest tests/test_cropstudio_colorcorrect.py -v`
Expected: PASS (7 tests; the real-DNG one takes a few seconds)

- [ ] **Step 5: Commit**

```bash
git add cropstudio/colorcorrect.py tests/test_cropstudio_colorcorrect.py
git commit -m "feat(cropstudio): RAW bytes -> color-corrected JPEG via rawpy"
```

---

### Task 4: `POST /rawdecode` endpoint

**Files:**
- Modify: `cropstudio/app.py`
- Test: `tests/test_cropstudio_app.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cropstudio_app.py`:

```python
import cropstudio.colorcorrect as colorcorrect


def test_rawdecode_returns_corrected_jpeg(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(colorcorrect, "raw_to_jpeg", lambda data: b"\xff\xd8fakejpeg")
    r = c.post("/rawdecode",
               files={"image": ("x.dng", b"raw-bytes-here", "application/octet-stream")})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content == b"\xff\xd8fakejpeg"


def test_rawdecode_bad_input_is_400(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/rawdecode",
               files={"image": ("x.dng", b"not raw", "application/octet-stream")})
    assert r.status_code == 400
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `".venv/Scripts/python.exe" -m pytest tests/test_cropstudio_app.py -v -k rawdecode`
Expected: FAIL with `assert 404 == 200` (route doesn't exist yet)

- [ ] **Step 3: Add the endpoint**

In `cropstudio/app.py`, change the imports:

```python
from fastapi.responses import HTMLResponse, Response

from . import batch, colorcorrect, manifest, paths, service
```

(replacing the current `from fastapi.responses import HTMLResponse` and
`from . import batch, manifest, paths, service` lines), then add this route
inside `create_app`, after the `/shot` route:

```python
    @app.post("/rawdecode")
    async def rawdecode(image: UploadFile = File(...)):
        data = await image.read()
        try:
            jpeg = colorcorrect.raw_to_jpeg(data)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return Response(content=jpeg, media_type="image/jpeg")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `".venv/Scripts/python.exe" -m pytest tests/test_cropstudio_app.py -v`
Expected: PASS (all, including the 2 new rawdecode tests)

- [ ] **Step 5: Commit**

```bash
git add cropstudio/app.py tests/test_cropstudio_app.py
git commit -m "feat(cropstudio): POST /rawdecode serves RAW uploads as corrected JPEG"
```

---

### Task 5: Browser — RAW uploads go through `/rawdecode` instead of failing

**Files:**
- Modify: `cropstudio/static/index.html` (the `loadCurrent` function and its
  RAW guard, added in commit 885a63c)

- [ ] **Step 1: Extract the shared canvas-draw helper and rewrite the RAW branch**

In `cropstudio/static/index.html`, replace the whole block from
`const RAW_EXTS = ...` through the end of `function loadCurrent() { ... }`:

```js
        // Camera RAW containers the browser cannot decode. Server-side RAW
        // ingestion (rawpy) is a planned separate feature; until then, fail
        // loudly instead of hanging on "Loading image..." forever.
        const RAW_EXTS = /\.(dng|nef|cr2|cr3|arw|raf|orf|rw2|pef|srw)$/i;

        function _unreadable(file, why) {
            imageLoaded = false;
            setStatus(`Can't open ${file.name}`, 'text-red-600');
            showToast(`Can't open "${file.name}" — ${why}`, false);
        }

        function loadCurrent() {
            const file = queue[qIndex];
            if (!file) return;
            updatePos();
            if (RAW_EXTS.test(file.name)) {
                _unreadable(file, 'RAW files are not supported yet. Convert to JPG first (or use the phone camera\'s JPG output).');
                return;
            }
            setStatus('Loading image...', 'text-red-500');
            const reader = new FileReader();
            reader.onload = (event) => {
                const img = new Image();
                img.onload = () => {
                    const MAX_WIDTH = 1200;
                    let width = img.width, height = img.height;
                    if (width > MAX_WIDTH) { height = Math.round((height * MAX_WIDTH) / width); width = MAX_WIDTH; }
                    sourceCanvas.width = width; sourceCanvas.height = height;
                    sourceCanvas.getContext('2d', { willReadFrequently: true }).drawImage(img, 0, 0, width, height);
                    imageLoaded = true;
                    outputRotation = 0;
                    corners = null;              // new image: drop the previous quad
                    if (isCvLoaded) runAutoDetect();
                };
                img.onerror = () => _unreadable(file, 'not a browser-readable image.');
                img.src = event.target.result;
            };
            reader.onerror = () => _unreadable(file, 'could not read the file.');
            reader.readAsDataURL(file);
        }
```

with:

```js
        // Camera RAW containers the browser cannot decode. The server decodes
        // + color-corrects them via POST /rawdecode (rawpy port of the WSL
        // pipeline's Colorprodawgv1), so RAW uploads work like any other file.
        const RAW_EXTS = /\.(dng|nef|cr2|cr3|arw|raf|orf|rw2|pef|srw)$/i;

        function _unreadable(file, why) {
            imageLoaded = false;
            setStatus(`Can't open ${file.name}`, 'text-red-600');
            showToast(`Can't open "${file.name}" — ${why}`, false);
        }

        function _drawLoadedImage(img) {
            const MAX_WIDTH = 1200;
            let width = img.width, height = img.height;
            if (width > MAX_WIDTH) { height = Math.round((height * MAX_WIDTH) / width); width = MAX_WIDTH; }
            sourceCanvas.width = width; sourceCanvas.height = height;
            sourceCanvas.getContext('2d', { willReadFrequently: true }).drawImage(img, 0, 0, width, height);
            imageLoaded = true;
            outputRotation = 0;
            corners = null;                      // new image: drop the previous quad
            if (isCvLoaded) runAutoDetect();
        }

        function loadCurrent() {
            const file = queue[qIndex];
            if (!file) return;
            updatePos();
            if (RAW_EXTS.test(file.name)) {
                setStatus('Decoding RAW on server...', 'text-red-500');
                const fd = new FormData();
                fd.append('image', file, file.name);
                fetch('/rawdecode', { method: 'POST', body: fd })
                    .then(r => {
                        if (!r.ok) throw new Error('server ' + r.status);
                        return r.blob();
                    })
                    .then(blob => {
                        const img = new Image();
                        img.onload = () => { URL.revokeObjectURL(img.src); _drawLoadedImage(img); };
                        img.onerror = () => _unreadable(file, 'server RAW decode returned an unreadable image.');
                        img.src = URL.createObjectURL(blob);
                    })
                    .catch(err => _unreadable(file, 'RAW decode failed (' + err.message + ').'));
                return;
            }
            setStatus('Loading image...', 'text-red-500');
            const reader = new FileReader();
            reader.onload = (event) => {
                const img = new Image();
                img.onload = () => _drawLoadedImage(img);
                img.onerror = () => _unreadable(file, 'not a browser-readable image.');
                img.src = event.target.result;
            };
            reader.onerror = () => _unreadable(file, 'could not read the file.');
            reader.readAsDataURL(file);
        }
```

- [ ] **Step 2: Manual verify — DNG straight into the browser**

Start the server (`".venv/Scripts/python.exe" -m cropstudio` or the preview
tool), open the page, upload the 3 real DNGs from `Images in/`
(`IMG_20260630_194814.dng` first — it's the Back). Confirm:
1. Status shows "Decoding RAW on server..." then the color-corrected photo
   appears on the canvas (expect a few seconds per shot for the decode).
2. Auto-detect draws a box on the case.
3. Save & Next works; after the third shot + Finish batch, the run folder
   has 3 correctly-titled JPEGs and a manifest entry with the barcode.
4. Also upload one junk file renamed to `.dng` — confirm the red
   "RAW decode failed (server 400)" toast, and Save & Next skips it.

- [ ] **Step 3: Commit**

```bash
git add cropstudio/static/index.html
git commit -m "feat(cropstudio): browser uploads RAW via /rawdecode instead of failing"
```

---

### Task 6: Full suite + end-to-end + spec bookkeeping

**Files:**
- Modify: `docs/superpowers/specs/2026-07-02-cropstudio-full-pipeline-design.md`
  (only if behavior diverged from the spec during implementation)

- [ ] **Step 1: Full cropstudio suite**

Run: `".venv/Scripts/python.exe" -m pytest tests/test_cropstudio_*.py -v`
Expected: all PASS (49 = 42 existing + 5 WB/enhance + 2 rawdecode; the
real-DNG decode test may be skipped on machines without the fixture).

- [ ] **Step 2: No regression in the rest of the suite**

Run: `".venv/Scripts/python.exe" -m pytest -q`
Expected: only the 3 pre-existing retired-`dvdflip/` failures
(`test_main.py` x2, `test_models.py` x1), unchanged.

- [ ] **Step 3: End-to-end sanity (scriptable, no browser)**

```bash
".venv/Scripts/python.exe" -c "
from pathlib import Path
from cropstudio import colorcorrect
data = Path('Images in/IMG_20260630_194814.dng').read_bytes()
jpeg = colorcorrect.raw_to_jpeg(data)
print('corrected jpeg:', len(jpeg), 'bytes')
import cv2, numpy as np
from redboxflip import barcode
img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
digits, method, rot = barcode.decode(img)
print('barcode off corrected image:', digits, method)
"
```

Expected: a multi-hundred-KB JPEG and barcode `0883316276402` — proving the
corrected output keeps the barcode readable end to end.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "test(cropstudio): verified RAW ingestion end-to-end" --allow-empty
```
