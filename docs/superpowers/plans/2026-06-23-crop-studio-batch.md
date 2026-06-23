# Crop Studio — Batch Browser Cropper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local FastAPI helper that serves the manual slider-cropper page, lets the user step through a batch of DVD photos, auto-saves each finished crop into `processed/run_<ts>/<Title>/`, and names files via a Qwen title read — reusing redboxflip's naming/vlm/imaging modules.

**Architecture:** New `cropstudio/` package. Pure buffering logic (`batch.py`) and disk/title logic (`service.py`) are separated from the FastAPI wiring (`app.py`) so the core is unit-testable without a server. The browser page (`static/index.html`, a batch-aware evolution of `ebaby_gemini_fixed.html`) is served same-origin, so `fetch('/shot')` has no CORS issues and Qwen is reached server-side.

**Tech Stack:** Python 3.13 (`.venv`), FastAPI + uvicorn + python-multipart (already installed), OpenCV/Pillow (already used by redboxflip), OpenCV.js in the browser.

**Conventions:**
- Run all commands from `C:\Users\mardi\Documents\Ebay code`.
- Python is `.venv\Scripts\python.exe` (Git Bash: `.venv/Scripts/python.exe`).
- Tests live in `tests/`, run with `.venv/Scripts/python.exe -m pytest`.

---

### Task 1: Package skeleton + output-path resolution

**Files:**
- Create: `cropstudio/__init__.py`
- Create: `cropstudio/paths.py`
- Test: `tests/test_cropstudio_paths.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cropstudio_paths.py
import os
from pathlib import Path

from cropstudio import paths


def test_output_root_defaults_to_project_processed():
    os.environ.pop("CROPSTUDIO_OUTPUT", None)
    root = paths.output_root()
    assert root.name == "processed"
    # project root is the parent of the cropstudio package
    assert root.parent == Path(paths.__file__).resolve().parent.parent


def test_output_root_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CROPSTUDIO_OUTPUT", str(tmp_path / "out"))
    assert paths.output_root() == tmp_path / "out"


def test_make_run_dir_creates_timestamped_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("CROPSTUDIO_OUTPUT", str(tmp_path))
    d = paths.make_run_dir(now="20260623_151004")
    assert d == tmp_path / "run_20260623_151004"
    assert d.is_dir()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cropstudio_paths.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cropstudio'`

- [ ] **Step 3: Write minimal implementation**

```python
# cropstudio/__init__.py
"""Crop Studio: local batch cropper helper for the Ebaby DVD workflow."""
```

```python
# cropstudio/paths.py
"""Where Crop Studio writes its output (native Windows path, not the WSL default)."""
import os
import time
from pathlib import Path


def project_root() -> Path:
    """The Ebaby project root = the parent of the cropstudio package."""
    return Path(__file__).resolve().parent.parent


def output_root() -> Path:
    """processed/ root. CROPSTUDIO_OUTPUT overrides; default is <project>/processed."""
    env = os.environ.get("CROPSTUDIO_OUTPUT")
    return Path(env) if env else project_root() / "processed"


def make_run_dir(now: str = None) -> Path:
    """Create and return processed/run_<timestamp>/ for this session."""
    ts = now or time.strftime("%Y%m%d_%H%M%S")
    d = output_root() / f"run_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cropstudio_paths.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add cropstudio/__init__.py cropstudio/paths.py tests/test_cropstudio_paths.py
git commit -m "feat(cropstudio): package skeleton + output path resolution"
```

---

### Task 2: BatchState buffering logic (pure, no I/O)

**Files:**
- Create: `cropstudio/batch.py`
- Test: `tests/test_cropstudio_batch.py`

`BatchState` buffers shots for one DVD keyed by slot (0 Back, 1 Front, 2 Inside).
`add()` returns a `FlushRequest` when a DVD is ready to write (3 slots filled, or a
higher `dvd_index` arrives meaning the previous DVD is done). `finish()` flushes a
trailing partial DVD.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cropstudio_batch.py
from cropstudio.batch import BatchState


def test_full_dvd_flushes_on_third_slot():
    s = BatchState()
    assert s.add(0, 0, b"back") is None
    assert s.add(0, 1, b"front") is None
    req = s.add(0, 2, b"inside")
    assert req is not None
    assert req.dvd_index == 0
    assert req.shots == {0: b"back", 1: b"front", 2: b"inside"}


def test_resending_same_slot_overwrites():
    s = BatchState()
    s.add(0, 0, b"first")
    s.add(0, 0, b"second")           # Prev -> re-crop -> Next
    s.add(0, 1, b"front")
    req = s.add(0, 2, b"inside")
    assert req.shots[0] == b"second"


def test_higher_dvd_index_flushes_partial_previous():
    s = BatchState()
    s.add(0, 0, b"back")             # only 1 shot for DVD 0
    req = s.add(1, 0, b"next-back")  # DVD 1 begins -> DVD 0 flushes partial
    assert req is not None
    assert req.dvd_index == 0
    assert req.shots == {0: b"back"}


def test_finish_flushes_trailing_partial():
    s = BatchState()
    s.add(0, 0, b"back")
    s.add(0, 1, b"front")
    req = s.finish()
    assert req.shots == {0: b"back", 1: b"front"}
    assert s.finish() is None        # nothing left


def test_finish_empty_returns_none():
    assert BatchState().finish() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cropstudio_batch.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cropstudio.batch'`

- [ ] **Step 3: Write minimal implementation**

```python
# cropstudio/batch.py
"""In-memory buffering of crop shots into DVDs of three (Back, Front, Inside).

Keyed by slot so re-cropping (Prev -> Next) overwrites rather than duplicating.
Pure logic: no disk, no network — the caller writes the returned FlushRequest.
"""
from dataclasses import dataclass


@dataclass
class FlushRequest:
    dvd_index: int
    shots: dict          # {slot:int -> bytes}, 1..3 entries


class BatchState:
    def __init__(self):
        self.current_dvd_index = None
        self.buffer = {}                 # slot -> bytes

    def add(self, dvd_index: int, slot: int, data: bytes):
        """Buffer a shot. Return a FlushRequest if a DVD became ready, else None."""
        flush = None
        if (self.current_dvd_index is not None
                and dvd_index != self.current_dvd_index and self.buffer):
            # A new DVD started; the previous one is done (possibly partial).
            flush = FlushRequest(self.current_dvd_index, dict(self.buffer))
            self.buffer = {}
        self.current_dvd_index = dvd_index
        self.buffer[slot] = data
        if flush is None and len(self.buffer) == 3:
            flush = FlushRequest(self.current_dvd_index, dict(self.buffer))
            self.buffer = {}
            self.current_dvd_index = None
        return flush

    def finish(self):
        """Flush a trailing partial DVD (1-2 shots). None if nothing buffered."""
        if not self.buffer:
            return None
        req = FlushRequest(self.current_dvd_index, dict(self.buffer))
        self.buffer = {}
        self.current_dvd_index = None
        return req
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cropstudio_batch.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add cropstudio/batch.py tests/test_cropstudio_batch.py
git commit -m "feat(cropstudio): slot-keyed batch buffering with flush rules"
```

---

### Task 3: Title resolution + DVD saving (reuses redboxflip)

**Files:**
- Create: `cropstudio/service.py`
- Test: `tests/test_cropstudio_service.py`

Decodes the buffered JPEG bytes, reads the title off the Front shot (Qwen via an
injectable `reader`), builds a collision-safe folder via `redboxflip.naming`, and
writes the three faces with `redboxflip.imaging.save_jpeg`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cropstudio_service.py
import cv2
import numpy as np
import pytest

from cropstudio import service


def _jpeg(color):
    img = np.full((40, 30, 3), color, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def test_save_dvd_names_from_front_and_writes_three(tmp_path):
    shots = {0: _jpeg((10, 10, 10)), 1: _jpeg((20, 20, 20)), 2: _jpeg((30, 30, 30))}
    out = service.save_dvd(shots, tmp_path, settings=None, dvd_counter=1,
                           reader=lambda bgr, s: "Goober And The Ghost Chasers")
    assert out["title"] == "Goober And The Ghost Chasers"
    d = tmp_path / "Goober And The Ghost Chasers"
    assert (d / "Goober And The Ghost Chasers - Back Cover.jpg").exists()
    assert (d / "Goober And The Ghost Chasers - Front Cover.jpg").exists()
    assert (d / "Goober And The Ghost Chasers - Inside.jpg").exists()


def test_save_dvd_falls_back_when_reader_returns_none(tmp_path):
    shots = {0: _jpeg((10, 10, 10)), 1: _jpeg((20, 20, 20)), 2: _jpeg((30, 30, 30))}
    out = service.save_dvd(shots, tmp_path, settings=None, dvd_counter=2,
                           reader=lambda bgr, s: None)
    assert out["title"] == "Untitled DVD 2"
    assert (tmp_path / "Untitled DVD 2" / "Untitled DVD 2 - Front Cover.jpg").exists()


def test_save_dvd_uniquifies_duplicate_title(tmp_path):
    shots = {0: _jpeg((1, 1, 1)), 1: _jpeg((2, 2, 2)), 2: _jpeg((3, 3, 3))}
    r = lambda bgr, s: "Heat"
    service.save_dvd(shots, tmp_path, None, 1, reader=r)
    out2 = service.save_dvd(shots, tmp_path, None, 2, reader=r)
    assert out2["dir"].endswith("Heat (2)")


def test_partial_dvd_uses_back_when_no_front(tmp_path):
    shots = {0: _jpeg((1, 1, 1))}        # only Back
    out = service.save_dvd(shots, tmp_path, None, 1, reader=lambda bgr, s: "Solo")
    assert (tmp_path / "Solo" / "Solo - Back Cover.jpg").exists()


def test_ensure_decodable_rejects_junk():
    with pytest.raises(ValueError):
        service.ensure_decodable(b"not a jpeg")
    service.ensure_decodable(_jpeg((5, 5, 5)))   # does not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cropstudio_service.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cropstudio.service'`

- [ ] **Step 3: Write minimal implementation**

```python
# cropstudio/service.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cropstudio_service.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add cropstudio/service.py tests/test_cropstudio_service.py
git commit -m "feat(cropstudio): title resolution + DVD file saving via redboxflip"
```

---

### Task 4: FastAPI app (`/`, `/health`, `/shot`, `/finish`)

**Files:**
- Create: `cropstudio/app.py`
- Test: `tests/test_cropstudio_app.py`

`create_app(run_dir=..., settings=...)` is a factory so tests inject a tmp run dir.
The title reader defaults to Qwen; tests monkeypatch `redboxflip.vlm`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cropstudio_app.py
import cv2
import numpy as np
from fastapi.testclient import TestClient

from cropstudio.app import create_app
import cropstudio.service as service


def _jpeg(color=(20, 20, 20)):
    img = np.full((40, 30, 3), color, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def _client(tmp_path, monkeypatch, title="Goober"):
    # Force the title reader to be deterministic (no live Ollama).
    monkeypatch.setattr(service.vlm, "available", lambda: True)
    monkeypatch.setattr(service.vlm, "title_from_cover", lambda bgr, s: title)
    return TestClient(create_app(run_dir=tmp_path, settings=None))


def _post(client, dvd_index, slot, data):
    return client.post("/shot",
                       data={"dvd_index": dvd_index, "slot": slot},
                       files={"image": ("s.jpg", data, "image/jpeg")})


def test_health(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.get("/health")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_three_shots_save_a_dvd(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, title="Goober")
    assert _post(c, 0, 0, _jpeg()).json()["status"] == "buffered"
    assert _post(c, 0, 1, _jpeg()).json()["status"] == "buffered"
    r = _post(c, 0, 2, _jpeg()).json()
    assert r["status"] == "saved" and r["title"] == "Goober"
    assert (tmp_path / "Goober" / "Goober - Inside.jpg").exists()


def test_finish_flushes_partial(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, title="Solo")
    _post(c, 0, 0, _jpeg())
    _post(c, 0, 1, _jpeg())
    r = c.post("/finish").json()
    assert r["status"] == "saved"
    assert (tmp_path / "Solo" / "Solo - Front Cover.jpg").exists()


def test_bad_image_is_rejected(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = _post(c, 0, 0, b"not-an-image")
    assert r.status_code == 400


def test_index_served(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.get("/")
    assert r.status_code == 200 and "Ebaby" in r.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cropstudio_app.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cropstudio.app'`
(Note: `test_index_served` will also fail until Task 6 creates the HTML; that is expected and re-run there.)

- [ ] **Step 3: Write minimal implementation**

```python
# cropstudio/app.py
"""FastAPI server: serves the cropper page and saves posted crops to processed/."""
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from redboxflip import vlm
from redboxflip.config import load_settings

from . import batch, paths, service

_STATIC = Path(__file__).resolve().parent / "static"


def create_app(run_dir=None, settings=None):
    app = FastAPI(title="Crop Studio")
    state = batch.BatchState()
    settings = settings if settings is not None else load_settings()
    run_dir = run_dir or paths.make_run_dir()
    counter = {"n": 0}

    def _flush(req):
        counter["n"] += 1
        return service.save_dvd(req.shots, run_dir, settings, counter["n"])

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (_STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health():
        return {"ok": True, "qwen_available": vlm.available(),
                "output_dir": str(run_dir)}

    @app.post("/shot")
    async def shot(image: UploadFile = File(...), dvd_index: int = Form(...),
                   slot: int = Form(...)):
        data = await image.read()
        try:
            service.ensure_decodable(data)
        except ValueError:
            raise HTTPException(status_code=400, detail="could not decode image")
        req = state.add(dvd_index, slot, data)
        if req is None:
            return {"status": "buffered", "have": sorted(state.buffer.keys())}
        return {"status": "saved", **_flush(req)}

    @app.post("/finish")
    def finish():
        req = state.finish()
        if req is None:
            return {"status": "empty"}
        return {"status": "saved", **_flush(req)}

    return app
```

- [ ] **Step 4: Run test to verify it passes (except the HTML test)**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cropstudio_app.py -v -k "not index_served"`
Expected: PASS (4 passed). `test_index_served` is deselected; it passes after Task 6.

- [ ] **Step 5: Commit**

```bash
git add cropstudio/app.py tests/test_cropstudio_app.py
git commit -m "feat(cropstudio): FastAPI routes for shot buffering + save"
```

---

### Task 5: Entry point (`python -m cropstudio`)

**Files:**
- Create: `cropstudio/__main__.py`
- Test: `tests/test_cropstudio_main.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cropstudio_main.py
import cropstudio.__main__ as m


def test_build_app_returns_fastapi(tmp_path, monkeypatch):
    monkeypatch.setenv("CROPSTUDIO_OUTPUT", str(tmp_path))
    app = m.build_app()
    routes = {r.path for r in app.routes}
    assert "/shot" in routes and "/health" in routes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cropstudio_main.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cropstudio.__main__'`

- [ ] **Step 3: Write minimal implementation**

```python
# cropstudio/__main__.py
"""Launch Crop Studio: start uvicorn on 127.0.0.1:8765 and open the browser."""
import threading
import webbrowser

import uvicorn

from .app import create_app

HOST = "127.0.0.1"
PORT = 8765
URL = f"http://{HOST}:{PORT}/"


def build_app():
    """Construct the FastAPI app (own run dir + settings). Separated for testing."""
    return create_app()


def main():
    app = build_app()
    threading.Timer(1.0, lambda: webbrowser.open(URL)).start()
    print(f"Crop Studio running at {URL}  (Ctrl+C to stop)")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cropstudio_main.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add cropstudio/__main__.py tests/test_cropstudio_main.py
git commit -m "feat(cropstudio): python -m cropstudio entry point"
```

---

### Task 6: Batch-aware cropper page (`static/index.html`)

**Files:**
- Create: `cropstudio/static/index.html` (start from `C:\Users\mardi\Downloads\ebaby_gemini_fixed.html`)

This task ports the existing fixed cropper and adds: multi-file upload, a file
queue with Prev/Next, a position+face indicator, auto-save POST on Next, a
Finish-batch button, and a toast. There is no unit test; verification is manual
(the `test_index_served` test from Task 4 confirms it is served).

- [ ] **Step 1: Copy the working cropper as the starting point**

```bash
mkdir -p cropstudio/static
cp "C:/Users/mardi/Downloads/ebaby_gemini_fixed.html" cropstudio/static/index.html
```

- [ ] **Step 2: Add the batch toolbar markup**

In `cropstudio/static/index.html`, find the header action buttons block:

```html
                    <button id="downloadBtn" class="bg-red-900 hover:bg-red-950 text-white px-5 py-2.5 rounded-lg transition-colors font-semibold shadow-sm hidden">Download</button>
                    <span id="statusText" class="text-sm font-semibold text-red-400 bg-red-50 px-3 py-1.5 rounded-md border border-red-100">Waiting...</span>
```

Replace it with (multi-file upload + nav + position indicator; Download kept as a manual fallback):

```html
                    <button id="prevBtn" class="bg-white border border-red-200 hover:bg-red-50 text-red-800 px-4 py-2.5 rounded-lg font-semibold hidden">‹ Prev</button>
                    <button id="nextBtn" class="bg-red-600 hover:bg-red-700 text-white px-5 py-2.5 rounded-lg font-semibold shadow-sm hidden">Save &amp; Next ›</button>
                    <button id="finishBtn" class="bg-red-900 hover:bg-red-950 text-white px-5 py-2.5 rounded-lg font-semibold shadow-sm hidden">Finish batch</button>
                    <span id="posText" class="text-sm font-bold text-red-700 bg-red-50 px-3 py-1.5 rounded-md border border-red-100 hidden">—</span>
                    <span id="statusText" class="text-sm font-semibold text-red-400 bg-red-50 px-3 py-1.5 rounded-md border border-red-100">Waiting...</span>
```

Then change the file input to accept multiple files. Find:

```html
                        <input type="file" id="imageInput" accept="image/*" class="hidden">
```

Replace with:

```html
                        <input type="file" id="imageInput" accept="image/*" multiple class="hidden">
```

- [ ] **Step 3: Add a toast container before `</body>`**

Find the closing of the main container and the OpenCV script tag:

```html
    </div>

    <script async src="https://docs.opencv.org/4.8.0/opencv.js" onload="onOpenCvLoaded();" type="text/javascript"></script>
```

Replace with (adds a fixed toast div):

```html
    </div>

    <div id="toast" class="fixed bottom-5 right-5 z-50 hidden max-w-sm px-4 py-3 rounded-lg shadow-lg text-sm font-semibold"></div>

    <script async src="https://docs.opencv.org/4.8.0/opencv.js" onload="onOpenCvLoaded();" type="text/javascript"></script>
```

- [ ] **Step 4: Replace the single-image load handler with a batch queue**

Find the existing file-input change handler (the block starting
`document.getElementById('imageInput').addEventListener('change', (e) => {` and
ending at its closing `});`) and replace the WHOLE block with:

```javascript
        // ---- batch queue ----------------------------------------------------
        let queue = [];            // File[]
        let qIndex = 0;            // current file index
        const FACE_LABEL = ['Back', 'Front', 'Inside'];

        document.getElementById('imageInput').addEventListener('change', (e) => {
            const files = Array.from(e.target.files || []);
            if (!files.length) return;
            queue = files; qIndex = 0;
            document.getElementById('prevBtn').classList.remove('hidden');
            document.getElementById('nextBtn').classList.remove('hidden');
            document.getElementById('posText').classList.remove('hidden');
            loadCurrent();
        });

        function updatePos() {
            const face = FACE_LABEL[qIndex % 3];
            document.getElementById('posText').textContent =
                `Shot ${qIndex + 1} / ${queue.length} — ${face}`;
            const last = qIndex >= queue.length - 1;
            document.getElementById('nextBtn').classList.toggle('hidden', last);
            document.getElementById('finishBtn').classList.toggle('hidden', !last);
        }

        function loadCurrent() {
            const file = queue[qIndex];
            if (!file) return;
            setStatus('Loading image...', 'text-red-500');
            updatePos();
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
                    if (isCvLoaded) runAutoDetect();
                };
                img.src = event.target.result;
            };
            reader.readAsDataURL(file);
        }
```

- [ ] **Step 5: Wire Prev / Save&Next / Finish to the server**

Immediately AFTER the block added in Step 4, add:

```javascript
        function showToast(msg, ok) {
            const t = document.getElementById('toast');
            t.textContent = msg;
            t.className = 'fixed bottom-5 right-5 z-50 max-w-sm px-4 py-3 rounded-lg shadow-lg text-sm font-semibold text-white ' +
                (ok ? 'bg-red-700' : 'bg-red-900');
            t.classList.remove('hidden');
            clearTimeout(window._tt); window._tt = setTimeout(() => t.classList.add('hidden'), 4000);
        }

        function currentBlob() {
            return new Promise(res => outputCanvas.toBlob(res, 'image/jpeg', 0.95));
        }

        async function postCurrentShot() {
            const blob = await currentBlob();
            if (!blob) throw new Error('no crop to save');
            const fd = new FormData();
            fd.append('image', blob, 'shot.jpg');
            fd.append('dvd_index', Math.floor(qIndex / 3));
            fd.append('slot', qIndex % 3);
            const r = await fetch('/shot', { method: 'POST', body: fd });
            if (!r.ok) throw new Error('server ' + r.status);
            return r.json();
        }

        document.getElementById('nextBtn').addEventListener('click', async () => {
            try {
                const res = await postCurrentShot();
                if (res.status === 'saved') showToast(`Saved "${res.title}" (${res.files.length} files)`, true);
            } catch (err) {
                showToast('Save failed: ' + err.message + ' — click Save & Next to retry', false);
                return;   // do NOT advance; keep the crop on screen
            }
            if (qIndex < queue.length - 1) { qIndex++; loadCurrent(); }
        });

        document.getElementById('finishBtn').addEventListener('click', async () => {
            try {
                const res = await postCurrentShot();           // save the last shot
                if (res.status === 'saved') showToast(`Saved "${res.title}"`, true);
                const f = await fetch('/finish', { method: 'POST' });
                const fj = await f.json();
                if (fj.status === 'saved') showToast(`Saved "${fj.title}" — batch complete`, true);
                else showToast('Batch complete', true);
            } catch (err) {
                showToast('Finish failed: ' + err.message, false);
            }
        });

        document.getElementById('prevBtn').addEventListener('click', () => {
            if (qIndex > 0) { qIndex--; loadCurrent(); }
        });
```

- [ ] **Step 6: Update the ready status text**

Find:

```javascript
                setStatus("Engine Ready — upload a photo", "text-red-600");
```

Replace with:

```javascript
                setStatus("Engine Ready — upload a batch of photos", "text-red-600");
```

- [ ] **Step 7: Verify the page is served and parses**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cropstudio_app.py::test_index_served -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add cropstudio/static/index.html
git commit -m "feat(cropstudio): batch-aware cropper page with auto-save"
```

---

### Task 7: Windows launcher

**Files:**
- Create: `Run Crop Studio.bat`

- [ ] **Step 1: Create the launcher**

```bat
@echo off
REM Launch Crop Studio (batch browser cropper) natively on Windows.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Could not find .venv\Scripts\python.exe
    echo Create it with:  py -m venv .venv  ^&^&  .venv\Scripts\pip install -r requirements-redbox.txt
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m cropstudio
pause
```

- [ ] **Step 2: Verify the server boots (smoke, 5s) and stops**

Run:
```bash
cd "C:/Users/mardi/Documents/Ebay code"
.venv/Scripts/python.exe -c "import cropstudio.__main__ as m; app=m.build_app(); from fastapi.testclient import TestClient; print(TestClient(app).get('/health').json())"
```
Expected: prints `{'ok': True, 'qwen_available': ..., 'output_dir': '...processed...run_...'}`

- [ ] **Step 3: Commit**

```bash
git add "Run Crop Studio.bat"
git commit -m "feat(cropstudio): native Windows launcher"
```

---

### Task 8: Full suite + manual end-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: Run the whole cropstudio suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cropstudio_*.py -v`
Expected: all PASS (paths 3, batch 5, service 5, app 5, main 1)

- [ ] **Step 2: Confirm no regression in the rest of the suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: the cropstudio tests pass; the only failures are the 3 pre-existing
retired-`dvdflip/` failures (test_main.py x2, test_models.py x1) — unchanged.

- [ ] **Step 3: Manual end-to-end**

1. Double-click `Run Crop Studio.bat`. A console window opens; the browser opens to `http://127.0.0.1:8765/`.
2. Click **Upload Photo**, select 3 (or 6) images from `Images in` (Back, Front, Inside order).
3. For each: drag the 4 corners over the case, rotate if upside-down, then **Save & Next**.
4. On the last shot, click **Finish batch**.
5. Confirm a toast shows the read title, and that `processed/run_<ts>/<Title>/` contains `<Title> - Back Cover.jpg`, `- Front Cover.jpg`, `- Inside.jpg`.
6. Close the console window to stop the server.

- [ ] **Step 4: Final commit (docs/notes if any tweaks were needed)**

```bash
git add -A
git commit -m "test(cropstudio): verified batch end-to-end" --allow-empty
```

---

## Notes for the implementer

- **Do not** touch `redboxflip/` in this plan — Crop Studio only imports it.
- The multi-disc inside-crop bug in `redboxflip/scan.py` is a **separate** task; ignore it here.
- If `git commit` reports CRLF warnings on the `.html`/`.bat`, that is expected on Windows and harmless.
- Keep `cropstudio/__main__.py`'s `build_app()` separate from `main()` so tests never start uvicorn.
