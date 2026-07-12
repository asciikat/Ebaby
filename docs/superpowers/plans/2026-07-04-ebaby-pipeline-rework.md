# Ebaby v3 Pipeline-First Rework — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the crop-first `redboxflip` app with a new `ebaby/` FastAPI web app (served from WSL Debian, used from the Windows browser) that enforces the user's real workflow: upload → sequential rename → colour correct → barcode locate/decode → eBay AU lookup + CSV → slug rename → **then** seeded auto-crop review. Legacy `redboxflip/`, `dvdflip/`, `ebayflip_V2.py` are untouched.

**Architecture:** A batch is a folder on disk (`/home/M/Ebaby Runs/<batch>/`) whose `state.json` records which of 7 stages it's at (`upload → rename → color → barcode → ebay → crop → done`). Each stage is a pure-Python module under `ebaby/stages/` ported from the user's proven WSL scripts (behaviour preserved, argparse CLI replaced with importable functions). `ebaby/server.py` is a thin FastAPI layer: one route group per stage, each route calls its stage module against the current batch and advances `state.json`. `ebaby/static/` is a single-page vanilla HTML/JS/canvas UI — no build step, no node toolchain, because the browser only needs to hit local FastAPI routes.

**Tech Stack:** Python 3.13 (existing `Ebabyv2` venv in WSL Debian), FastAPI + uvicorn + python-multipart (new deps), opencv-python / numpy / rawpy / pyzbar / rembg[cpu] / requests (already installed), pytest (existing `pytest.ini`, `testpaths = tests`), vanilla HTML/CSS/JS for the frontend.

**A note on TDD granularity for this plan:** every stage module is a behaviour-preserving port of a script the user has already run in production (`bn_rename.py`, `Colorprodawgv1.py`, `locate_and_crop_barcodes.py`, `dvd_barcode_scanner.py`, `ebay_api.py`, `ebay_csv_extractor_new.py`, `ebay_Api_rename_new.py`, and `redboxflip/scan.py` + `cutout.py` for the crop stage). The risk isn't "does this logic work" — it already does — it's "did the port change behaviour." So each stage task writes the **whole module's test file** (several test functions covering the module's real behaviour) before the **whole module's implementation**, rather than one test per micro-function. Steps are still ordered write-tests → run-red → implement → run-green → commit.

---

## Task 0: Scaffolding

**Files:**
- Create: `ebaby/__init__.py`
- Create: `ebaby/stages/__init__.py`
- Create: `requirements-ebaby.txt`
- Test: none (no logic yet)

- [ ] **Step 1: Create the package directories and empty `__init__.py` files**

```bash
mkdir -p ebaby/stages ebaby/static tests/ebaby
touch ebaby/__init__.py ebaby/stages/__init__.py tests/ebaby/__init__.py
```

- [ ] **Step 2: Write `requirements-ebaby.txt`**

```
# New deps for the ebaby/ web app, on top of requirements.txt /
# requirements-redbox.txt which already provide cv2, numpy, rawpy,
# pyzbar, rembg[cpu], requests, pytesseract.
fastapi
uvicorn[standard]
python-multipart
```

- [ ] **Step 3: Install into the existing WSL venv**

Run: `wsl.exe -d Debian bash -lc "source ~/Ebabyv2/bin/activate && pip install -r '/mnt/c/Users/mardi/Documents/Ebay code/requirements-ebaby.txt'"`
Expected: `Successfully installed fastapi ... uvicorn ... python-multipart ...`

- [ ] **Step 4: Verify pytest collects the new empty test package**

Run: `wsl.exe -d Debian bash -lc "source ~/Ebabyv2/bin/activate && cd '/mnt/c/Users/mardi/Documents/Ebay code' && python3 -m pytest tests/ebaby -v"`
Expected: `no tests ran` (0 collected, exit code 5) — confirms the path is on `testpaths` and nothing is broken yet.

- [ ] **Step 5: Commit**

```bash
git add ebaby/__init__.py ebaby/stages/__init__.py tests/ebaby/__init__.py requirements-ebaby.txt
git commit -m "chore(ebaby): scaffold new package for pipeline-first rework"
```

---

## Task 1: `naming_utils.py` (slug logic shared by images and CSV)

**Files:**
- Create: `ebaby/naming_utils.py`
- Test: `tests/ebaby/test_naming_utils.py`

This is a verbatim copy of the user's proven `naming_utils.py` (from
`Pipeline/2_barcode_ebay/test rename/naming_utils.py`) — it must produce
byte-identical slugs to what the user's existing CSV/rename scripts already
produce, so image filenames and CSV rows always agree.

- [ ] **Step 1: Write the failing tests**

```python
# tests/ebaby/test_naming_utils.py
from ebaby.naming_utils import slugify_title


def test_basic_title():
    assert slugify_title("The Matrix", fallback="000") == "Matrix"


def test_strips_stop_words_and_limits_to_three_words():
    assert slugify_title("Lord of the Rings: The Fellowship", fallback="000") == "Lord_Rings_Fellowship"


def test_dedupes_repeated_words():
    assert slugify_title("Alien vs Alien Redux", fallback="000") == "Alien_vs_Redux"


def test_empty_title_uses_fallback():
    assert slugify_title("", fallback="883316276402") == "883316276402"
    assert slugify_title(None, fallback="883316276402") == "883316276402"


def test_non_ascii_is_stripped():
    assert slugify_title("Amélie", fallback="000") == "Amelie"
```

- [ ] **Step 2: Run to verify it fails**

Run: `wsl.exe -d Debian bash -lc "source ~/Ebabyv2/bin/activate && cd '/mnt/c/Users/mardi/Documents/Ebay code' && python3 -m pytest tests/ebaby/test_naming_utils.py -v"`
Expected: FAIL — `ModuleNotFoundError: No module named 'ebaby.naming_utils'`

- [ ] **Step 3: Write the implementation (verbatim from the proven script)**

```python
# ebaby/naming_utils.py
"""Shared naming logic so that image filenames and CSV rows use the exact
same slug for the same eBay title / barcode."""
import re
import unicodedata

STOP_WORDS = {
    "a", "an", "and", "or", "of", "to", "from", "in", "on", "for",
    "with", "by", "at", "as", "is", "it",
}


def _asciify(text):
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


def slugify_title(title, fallback, max_words=3):
    ascii_title = _asciify((title or "").title())
    words = re.findall(r"\w+", ascii_title)
    kept = [w for w in words if w.lower() not in STOP_WORDS] or words
    deduped = list(dict.fromkeys(kept))
    return "_".join(deduped[:max_words]) or fallback
```

- [ ] **Step 4: Run to verify it passes**

Run: `wsl.exe -d Debian bash -lc "source ~/Ebabyv2/bin/activate && cd '/mnt/c/Users/mardi/Documents/Ebay code' && python3 -m pytest tests/ebaby/test_naming_utils.py -v"`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add ebaby/naming_utils.py tests/ebaby/test_naming_utils.py
git commit -m "feat(ebaby): port naming_utils slug logic"
```

---

## Task 2: `batch.py` (batch folder layout + state machine)

**Files:**
- Create: `ebaby/batch.py`
- Test: `tests/ebaby/test_batch.py`

**Files:**
- [ ] **Step 1: Write the failing tests**

```python
# tests/ebaby/test_batch.py
import json
import pytest

from ebaby import batch


@pytest.fixture(autouse=True)
def isolated_root(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "BATCHES_ROOT", tmp_path / "Ebaby Runs")
    return tmp_path


def test_create_batch_makes_expected_subfolders():
    d = batch.create_batch("2026-07-04-run1")
    for sub in ("1_originals/used", "1_originals/new", "2_color",
                "3_barcodes", "4_renamed", "5_cropped"):
        assert (d / sub).is_dir()
    state = json.loads((d / "state.json").read_text())
    assert state["stage"] == "upload"


def test_create_batch_twice_raises():
    batch.create_batch("dup")
    with pytest.raises(batch.BatchError):
        batch.create_batch("dup")


def test_advance_stage_happy_path():
    d = batch.create_batch("run2")
    state = batch.advance_stage(d, "upload", "rename")
    assert state["stage"] == "rename"
    assert batch.read_state(d)["stage"] == "rename"


def test_advance_stage_wrong_from_stage_raises():
    d = batch.create_batch("run3")
    with pytest.raises(batch.BatchError):
        batch.advance_stage(d, "color", "barcode")  # batch is still at "upload"


def test_advance_stage_unknown_to_stage_raises():
    d = batch.create_batch("run4")
    with pytest.raises(batch.BatchError):
        batch.advance_stage(d, "upload", "not_a_real_stage")


def test_read_state_missing_file_raises():
    with pytest.raises(batch.BatchError):
        batch.read_state(batch.batch_dir("never_created"))


def test_list_batches_returns_created_names_sorted():
    batch.create_batch("b_run")
    batch.create_batch("a_run")
    assert batch.list_batches() == ["a_run", "b_run"]


def test_list_batches_empty_root_returns_empty_list():
    assert batch.list_batches() == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `wsl.exe -d Debian bash -lc "source ~/Ebabyv2/bin/activate && cd '/mnt/c/Users/mardi/Documents/Ebay code' && python3 -m pytest tests/ebaby/test_batch.py -v"`
Expected: FAIL — `ModuleNotFoundError: No module named 'ebaby.batch'`

- [ ] **Step 3: Write the implementation**

```python
# ebaby/batch.py
"""Batch folder layout, state.json persistence, and stage-transition rules.

A batch is one run of the pipeline: a folder under BATCHES_ROOT holding the
originals, every intermediate stage's output, and a state.json recording
which stage the batch is currently at. Stages only ever move forward
(upload -> rename -> color -> barcode -> ebay -> crop -> done); re-running a
stage clears and regenerates only that stage's own output folder (handled by
the stage modules themselves, not here).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

STAGES = ["upload", "rename", "color", "barcode", "ebay", "crop", "done"]

BATCHES_ROOT = Path("/home/M/Ebaby Runs")

_SUBFOLDERS = (
    "1_originals/used", "1_originals/new", "2_color",
    "3_barcodes", "4_renamed", "5_cropped",
)


class BatchError(RuntimeError):
    """Raised when a stage is requested out of order or a batch is malformed."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def batch_dir(name: str) -> Path:
    return BATCHES_ROOT / name


def create_batch(name: str) -> Path:
    d = batch_dir(name)
    if d.exists():
        raise BatchError(f"batch '{name}' already exists")
    for sub in _SUBFOLDERS:
        (d / sub).mkdir(parents=True, exist_ok=True)
    write_state(d, {"stage": "upload", "sets": {}, "created": _now()})
    return d


def read_state(d: Path) -> dict:
    p = d / "state.json"
    if not p.exists():
        raise BatchError(f"no state.json in {d}")
    return json.loads(p.read_text(encoding="utf-8"))


def write_state(d: Path, state: dict) -> None:
    (d / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def advance_stage(d: Path, from_stage: str, to_stage: str) -> dict:
    state = read_state(d)
    if state["stage"] != from_stage:
        raise BatchError(
            f"batch is at stage '{state['stage']}', expected '{from_stage}'"
        )
    if to_stage not in STAGES:
        raise BatchError(f"unknown stage '{to_stage}'")
    state["stage"] = to_stage
    state["updated"] = _now()
    write_state(d, state)
    return state


def list_batches() -> list:
    if not BATCHES_ROOT.exists():
        return []
    return sorted(
        p.name for p in BATCHES_ROOT.iterdir()
        if p.is_dir() and (p / "state.json").exists()
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `wsl.exe -d Debian bash -lc "source ~/Ebabyv2/bin/activate && cd '/mnt/c/Users/mardi/Documents/Ebay code' && python3 -m pytest tests/ebaby/test_batch.py -v"`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add ebaby/batch.py tests/ebaby/test_batch.py
git commit -m "feat(ebaby): batch folder layout and stage state machine"
```

---

## Task 3: `stages/rename_seq.py` (sequential rename, before anything else)

**Files:**
- Create: `ebaby/stages/rename_seq.py`
- Test: `tests/ebaby/test_rename_seq.py`

Ported from `bn_rename.py` (2-shot, number scheme) and `sh_rename.py` (3-shot,
letter scheme), unified into one parameterized module. Filename sort only
(matches both scripts' actual defaults — `sh_rename.py --sort name` is its
default, and `bn_rename.py` is driven the same way in this app since EXIF is
an optional extra the scripts already treat as non-default). Two-phase
collision-safe apply is preserved exactly.

- [ ] **Step 1: Write the failing tests**

```python
# tests/ebaby/test_rename_seq.py
import pytest

from ebaby.stages import rename_seq as rs


def _touch(d, *names):
    for n in names:
        (d / n).write_bytes(b"x")


def test_group_shots_pairs_raw_and_jpeg_by_stem(tmp_path):
    _touch(tmp_path, "IMG_0001.NEF", "IMG_0001.JPG", "IMG_0002.NEF")
    files = rs.find_images(tmp_path)
    shots = rs.group_shots(files)
    assert len(shots) == 2
    assert {p.name for p in shots[0]} == {"IMG_0001.NEF", "IMG_0001.JPG"}


def test_build_plan_used_stock_three_shot_letter_scheme(tmp_path):
    _touch(tmp_path, "s1.dng", "s2.dng", "s3.dng", "s4.dng", "s5.dng", "s6.dng")
    plan = rs.build_plan(tmp_path, set_size=3, scheme="letter", labels=rs.USED_LABELS)
    names = sorted(new.name for _, new in plan)
    assert names == ["a_back.dng", "a_front.dng", "a_inside.dng",
                      "b_back.dng", "b_front.dng", "b_inside.dng"]


def test_build_plan_new_stock_two_shot_number_scheme(tmp_path):
    _touch(tmp_path, "s1.dng", "s2.dng", "s3.dng", "s4.dng")
    plan = rs.build_plan(tmp_path, set_size=2, scheme="number", labels=rs.NEW_LABELS)
    names = sorted(new.name for _, new in plan)
    assert names == ["01_back.dng", "01_front.dng", "02_back.dng", "02_front.dng"]


def test_build_plan_uneven_count_raises(tmp_path):
    _touch(tmp_path, "s1.dng", "s2.dng", "s3.dng")  # not a multiple of 2
    with pytest.raises(ValueError, match="not a whole number of sets"):
        rs.build_plan(tmp_path, set_size=2, scheme="number", labels=rs.NEW_LABELS)


def test_apply_plan_renames_and_is_collision_safe(tmp_path):
    _touch(tmp_path, "s1.dng", "s2.dng")
    plan = rs.build_plan(tmp_path, set_size=2, scheme="number", labels=rs.NEW_LABELS)
    count = rs.apply_plan(plan)
    assert count == 2
    assert (tmp_path / "01_front.dng").exists()
    assert (tmp_path / "01_back.dng").exists()
    assert not (tmp_path / "s1.dng").exists()


def test_apply_plan_swap_does_not_clobber(tmp_path):
    # A already named 01_back, B already named 01_front -> plan should SWAP
    # them, which a naive single-pass os.rename would clobber.
    (tmp_path / "01_back.dng").write_bytes(b"A")
    (tmp_path / "01_front.dng").write_bytes(b"B")
    plan = [
        (tmp_path / "01_back.dng", tmp_path / "01_front.dng"),
        (tmp_path / "01_front.dng", tmp_path / "01_back.dng"),
    ]
    rs.apply_plan(plan)
    assert (tmp_path / "01_front.dng").read_bytes() == b"A"
    assert (tmp_path / "01_back.dng").read_bytes() == b"B"
```

- [ ] **Step 2: Run to verify it fails**

Run: `wsl.exe -d Debian bash -lc "source ~/Ebabyv2/bin/activate && cd '/mnt/c/Users/mardi/Documents/Ebay code' && python3 -m pytest tests/ebaby/test_rename_seq.py -v"`
Expected: FAIL — `ModuleNotFoundError: No module named 'ebaby.stages.rename_seq'`

- [ ] **Step 3: Write the implementation**

```python
# ebaby/stages/rename_seq.py
"""Sequential rename for DVD photo sets — ported from bn_rename.py (brand-new,
2-shot, number-keyed) and sh_rename.py (used, 3-shot, letter-keyed).

Groups files by filename stem so a RAW and its JPEG move together, orders
shots by filename (matches the scripts' proven default — camera filenames
like IMG_YYYYMMDD_HHMMSS already sort correctly), and renames with a
two-phase temp-name swap so no file is ever overwritten by another, even when
a rename plan swaps two existing names.
"""
import os
import uuid
from pathlib import Path

RAW_EXTS = {".nef", ".dng", ".cr2", ".cr3", ".arw", ".raf", ".orf", ".rw2", ".pef", ".srw"}
JPEG_EXTS = {".jpg", ".jpeg"}
FLAT_EXTS = {".png", ".tif", ".tiff"}
ALL_EXTS = RAW_EXTS | JPEG_EXTS | FLAT_EXTS

USED_LABELS = ["front", "back", "inside"]
NEW_LABELS = ["front", "back"]


def find_images(input_dir: Path) -> list:
    out = []
    for name in sorted(os.listdir(input_dir)):
        if name.startswith("."):
            continue
        p = input_dir / name
        if p.is_file() and p.suffix.lower() in ALL_EXTS:
            out.append(p)
    return out


def group_shots(files: list) -> list:
    """Group RAW+JPEG of the same shot (same filename stem), filename order."""
    groups = {}
    for f in files:
        groups.setdefault(f.stem, []).append(f)
    ordered_stems = list(dict.fromkeys(f.stem for f in files))
    return [groups[stem] for stem in ordered_stems]


def make_prefix(set_index: int, scheme: str) -> str:
    if scheme == "number":
        return f"{set_index + 1:02d}"
    s, n0 = "", set_index
    while True:
        s = chr(97 + (n0 % 26)) + s
        n0 = n0 // 26 - 1
        if n0 < 0:
            break
    return s


def build_plan(input_dir: Path, set_size: int, scheme: str, labels: list) -> list:
    """[(old_path, new_path), ...]. Raises ValueError on uneven counts or
    duplicate targets — never silently mis-groups a partial set."""
    files = find_images(input_dir)
    shots = group_shots(files)
    if len(shots) % set_size != 0:
        raise ValueError(
            f"{len(shots)} shots is not a whole number of sets of {set_size} "
            f"in {input_dir}"
        )
    plan = []
    for shot_index, shot_files in enumerate(shots):
        set_index = shot_index // set_size
        label_index = shot_index % set_size
        label = labels[label_index]
        prefix = make_prefix(set_index, scheme)
        for f in shot_files:
            plan.append((f, f.parent / f"{prefix}_{label}{f.suffix}"))

    finals = [new for _, new in plan]
    dupes = sorted({p for p in finals if finals.count(p) > 1})
    if dupes:
        raise ValueError(f"rename plan has duplicate targets: {[d.name for d in dupes]}")
    return plan


def apply_plan(plan: list) -> int:
    """Two-phase rename (temp names first) so no file is ever overwritten."""
    temps = []
    for old, new in plan:
        if old == new:
            continue
        tmp = old.parent / f".renametmp_{uuid.uuid4().hex}{old.suffix}"
        old.rename(tmp)
        temps.append((tmp, new))
    for tmp, new in temps:
        tmp.rename(new)
    return len(temps)
```

- [ ] **Step 4: Run to verify it passes**

Run: `wsl.exe -d Debian bash -lc "source ~/Ebabyv2/bin/activate && cd '/mnt/c/Users/mardi/Documents/Ebay code' && python3 -m pytest tests/ebaby/test_rename_seq.py -v"`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add ebaby/stages/rename_seq.py tests/ebaby/test_rename_seq.py
git commit -m "feat(ebaby): sequential pre-crop rename stage"
```

---

## Task 4: `stages/color.py` (RAW colour correction)

**Files:**
- Create: `ebaby/stages/color.py`
- Test: `tests/ebaby/test_color.py`

Ported from `Colorprodawgv1.py`. The white-balance/enhance math is pure
numpy/cv2 and is unit-tested directly with synthetic arrays. `process_raw`
(the actual `rawpy.postprocess` call) needs a real `.dng`/`.nef` file and is
covered by the **skip-if-absent** fixture pattern already used in
`tests/conftest.py` (`sample_dngs`) — reuse it rather than inventing a new
fixture mechanism.

- [ ] **Step 1: Write the failing tests**

```python
# tests/ebaby/test_color.py
import cv2
import numpy as np
import pytest

from ebaby.stages import color


def _white_bordered_frame():
    """A 40x40 BGR frame: white border (color cast), grey square in the
    middle standing in for the DVD case content."""
    frame = np.full((40, 40, 3), (200, 150, 150), dtype=np.uint8)  # blue-cast white
    frame[10:30, 10:30] = (80, 80, 80)  # neutral grey "case"
    return frame


def test_get_surrounding_white_mask_ignores_interior_and_keeps_border():
    frame = _white_bordered_frame()
    mask = color.get_surrounding_white_mask(frame)
    assert mask[0, 0] == 255       # border pixel included
    assert mask[20, 20] == 0       # interior "case" pixel excluded


def test_get_white_balance_gains_neutralizes_border_cast():
    frame = _white_bordered_frame()
    gains = color.get_white_balance_gains(frame)
    corrected = color.apply_white_balance(frame, gains)
    b, g, r = corrected[0, 0].astype(int)
    assert abs(int(b) - int(g)) <= 2
    assert abs(int(g) - int(r)) <= 2


def test_get_white_balance_gains_all_black_frame_is_safe_identity():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    gains = color.get_white_balance_gains(frame)
    assert gains == (1.0, 1.0, 1.0)


def test_enhance_image_is_a_noop_at_default_neutral_settings():
    frame = _white_bordered_frame()
    out = color.enhance_image(frame, contrast=0.0, saturation=1.0, sharpen=1.0)
    assert np.array_equal(out, frame)


def test_enhance_image_increases_saturation():
    frame = _white_bordered_frame()
    out = color.enhance_image(frame, contrast=1.0, saturation=1.5, sharpen=1.0)
    hsv_before = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_after = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    assert hsv_after[0, 0, 1] >= hsv_before[0, 0, 1]


def test_color_correct_file_writes_a_png(sample_front, tmp_path):
    out_path = tmp_path / "01_front.png"
    color.color_correct_file(sample_front, out_path)
    assert out_path.exists()
    img = cv2.imread(str(out_path))
    assert img is not None and img.shape[2] == 3
```

- [ ] **Step 2: Run to verify it fails**

Run: `wsl.exe -d Debian bash -lc "source ~/Ebabyv2/bin/activate && cd '/mnt/c/Users/mardi/Documents/Ebay code' && python3 -m pytest tests/ebaby/test_color.py -v"`
Expected: FAIL — `ModuleNotFoundError: No module named 'ebaby.stages.color'`

- [ ] **Step 3: Write the implementation**

```python
# ebaby/stages/color.py
"""RAW colour correction — ported from Colorprodawgv1.py.

White-balances each photo using the paper that SURROUNDS the DVD case (only
border-touching white blobs; print inside the cover art is never mistaken
for the neutral reference), then applies a mild S-curve/saturation/sharpen
enhancement. Runs before barcode scanning and rename so downstream stages
see clean, colour-corrected images.
"""
import cv2
import numpy as np
import rawpy


def _denoise_levels():
    return {
        "off": (rawpy.FBDDNoiseReductionMode.Off, None, 0),
        "light": (rawpy.FBDDNoiseReductionMode.Full, None, 1),
        "medium": (rawpy.FBDDNoiseReductionMode.Full, 100.0, 1),
        "strong": (rawpy.FBDDNoiseReductionMode.Full, 250.0, 2),
    }


def get_surrounding_white_mask(bgr_img: np.ndarray) -> np.ndarray:
    h, w = bgr_img.shape[:2]
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    kernel = np.ones((5, 5), np.uint8)
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
        if (surrounding.sum() / 255) > (h * w * 0.02):
            return surrounding
    return ((v > 120) & (s < 60)).astype(np.uint8) * 255


def get_white_balance_gains(bgr_img: np.ndarray):
    mask = get_surrounding_white_mask(bgr_img)
    if mask.sum() == 0:
        return 1.0, 1.0, 1.0
    mean_b = cv2.mean(bgr_img[:, :, 0], mask=mask)[0]
    mean_g = cv2.mean(bgr_img[:, :, 1], mask=mask)[0]
    mean_r = cv2.mean(bgr_img[:, :, 2], mask=mask)[0]
    if mean_b == 0 or mean_g == 0 or mean_r == 0:
        return 1.0, 1.0, 1.0
    target = (mean_b + mean_g + mean_r) / 3.0
    return (
        float(np.clip(target / mean_b, 0.5, 2.0)),
        float(np.clip(target / mean_g, 0.5, 2.0)),
        float(np.clip(target / mean_r, 0.5, 2.0)),
    )


def apply_white_balance(bgr_img: np.ndarray, gains) -> np.ndarray:
    b, g, r = cv2.split(bgr_img.astype("float32"))
    b *= gains[0]
    g *= gains[1]
    r *= gains[2]
    return np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)


def create_s_curve_lut(strength: float = 1.0) -> np.ndarray:
    if strength == 0.0:
        return np.arange(256, dtype=np.uint8)
    x = np.arange(256)
    nx = (x / 255.0 - 0.5) * 2
    y = 1 / (1 + np.exp(-strength * nx))
    y = (y - y.min()) / (y.max() - y.min()) * 255
    return np.clip(y, 0, 255).astype(np.uint8)


def enhance_image(bgr_img: np.ndarray, contrast: float = 1.0,
                   saturation: float = 1.1, sharpen: float = 1.2) -> np.ndarray:
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


def process_raw(raw_path, denoise: str = "light") -> np.ndarray:
    fbdd, noise_thr, median_passes = _denoise_levels().get(denoise, _denoise_levels()["light"])
    kwargs = dict(use_camera_wb=True, fbdd_noise_reduction=fbdd, median_filter_passes=median_passes)
    if noise_thr is not None:
        kwargs["noise_thr"] = noise_thr
    with rawpy.imread(str(raw_path)) as raw:
        rgb = raw.postprocess(**kwargs)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def color_correct_file(raw_path, out_path, denoise: str = "light",
                        contrast: float = 1.0, saturation: float = 1.1,
                        sharpen: float = 1.2, gains=None) -> None:
    """Decode one RAW file, white-balance + enhance, save as PNG at out_path."""
    bgr = process_raw(raw_path, denoise=denoise)
    current_gains = gains if gains is not None else get_white_balance_gains(bgr)
    corrected = apply_white_balance(bgr, current_gains)
    final = enhance_image(corrected, contrast, saturation, sharpen)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), final)
```

- [ ] **Step 4: Run to verify it passes**

Run: `wsl.exe -d Debian bash -lc "source ~/Ebabyv2/bin/activate && cd '/mnt/c/Users/mardi/Documents/Ebay code' && python3 -m pytest tests/ebaby/test_color.py -v"`
Expected: `6 passed` if sample DNGs exist under `samples/`, else `5 passed, 1 skipped` (matches the existing `sample_dngs` fixture's skip behaviour in `tests/conftest.py` — this test file is in `tests/ebaby/` which is a subdirectory of `tests/`, so pytest's `conftest.py` discovery already applies to it).

- [ ] **Step 5: Commit**

```bash
git add ebaby/stages/color.py tests/ebaby/test_color.py
git commit -m "feat(ebaby): RAW colour-correction stage"
```

---

## Task 5: `stages/barcode_locate.py` (find + crop the barcode region)

**Files:**
- Create: `ebaby/stages/barcode_locate.py`
- Test: `tests/ebaby/test_barcode_locate.py`

Ported from `locate_and_crop_barcodes.py`. Uses `exiftool` to pull the
embedded JPEG preview out of RAW files (fast, high-enough res for barcode
reading) and a gradient-based ROI ladder to find the barcode region within
that preview; falls back to saving the whole frame if nothing is confidently
found, so the decode stage always gets a shot at it.

- [ ] **Step 1: Write the failing tests**

```python
# tests/ebaby/test_barcode_locate.py
import cv2
import numpy as np

from ebaby.stages import barcode_locate as bl


def _frame_with_barcode_stripe():
    """A 400x300 frame with a wide, high-contrast horizontal stripe region
    standing in for a barcode (alternating dark/light bars)."""
    frame = np.full((300, 400, 3), 220, dtype=np.uint8)
    stripe = frame[130:170, 80:320]
    stripe[:, ::4] = 20  # vertical dark bars every 4px -> strong gradient
    return frame


def test_locate_barcode_crop_finds_the_stripe_region():
    frame = _frame_with_barcode_stripe()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    box = bl.locate_barcode_crop(gray)
    assert box is not None
    x0, y0, x1, y1 = box
    assert x0 < 100 and x1 > 300   # roughly spans the stripe's x-range
    assert y0 < 140 and y1 > 160   # roughly spans the stripe's y-range


def test_locate_barcode_crop_returns_none_on_blank_frame():
    blank = np.full((100, 100), 220, dtype=np.uint8)
    assert bl.locate_barcode_crop(blank) is None


def test_downscale_if_needed_caps_long_edge():
    frame = np.zeros((4000, 2000, 3), dtype=np.uint8)
    out = bl.downscale_if_needed(frame, max_dim=2400)
    assert max(out.shape[:2]) == 2400


def test_downscale_if_needed_leaves_small_frame_alone():
    frame = np.zeros((500, 400, 3), dtype=np.uint8)
    out = bl.downscale_if_needed(frame, max_dim=2400)
    assert out.shape == frame.shape


def test_load_image_uses_direct_read_for_non_raw(tmp_path):
    p = tmp_path / "back.jpg"
    cv2.imwrite(str(p), _frame_with_barcode_stripe())
    img = bl.load_image(p)
    assert img is not None and img.shape[2] == 3
```

- [ ] **Step 2: Run to verify it fails**

Run: `wsl.exe -d Debian bash -lc "source ~/Ebabyv2/bin/activate && cd '/mnt/c/Users/mardi/Documents/Ebay code' && python3 -m pytest tests/ebaby/test_barcode_locate.py -v"`
Expected: FAIL — `ModuleNotFoundError: No module named 'ebaby.stages.barcode_locate'`

- [ ] **Step 3: Write the implementation**

```python
# ebaby/stages/barcode_locate.py
"""Locate the barcode on a back-cover photo and save a tight crop of just
that region — ported from locate_and_crop_barcodes.py.

For .NEF/.DNG this pulls the embedded full-res JPEG preview via exiftool
(fast, avoids a full RAW decode). Falls back to reading the file directly
with OpenCV for any other format. If no barcode-like region can be
confidently located, the whole frame is returned instead so the decode stage
downstream still gets a shot at it.
"""
import subprocess

import cv2
import numpy as np

RAW_EXTS = {".nef", ".dng"}
MAX_DIM = 2400


def extract_raw_preview(raw_path):
    """Embedded JPEG preview from a RAW file via exiftool, or None."""
    for tag in ("-PreviewImage", "-JpgFromRaw"):
        try:
            result = subprocess.run(
                ["exiftool", "-b", tag, str(raw_path)],
                capture_output=True, check=False, timeout=30,
            )
        except FileNotFoundError:
            return None
        except subprocess.TimeoutExpired:
            return None
        if result.returncode == 0 and result.stdout:
            buf = np.frombuffer(result.stdout, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is not None:
                return img
    return None


def load_image(path):
    """Load any supported source file (RAW preview or standard image) as BGR."""
    path = str(path)
    ext = path.lower().rsplit(".", 1)[-1]
    if f".{ext}" in RAW_EXTS:
        return extract_raw_preview(path)
    return cv2.imread(path)


def locate_barcode_crop(gray):
    """(x0, y0, x1, y1) padded bounding box of the most barcode-like region,
    or None if nothing found. Barcodes are wide-and-short with a strong local
    gradient (alternating bars)."""
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=-1)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=-1)
    grad = cv2.convertScaleAbs(cv2.subtract(cv2.convertScaleAbs(grad_x),
                                            cv2.convertScaleAbs(grad_y)))
    blurred = cv2.blur(grad, (9, 9))
    _, thresh = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    if w < 40 or h < 15 or w / max(h, 1) < 1.5:
        return None
    pad_x, pad_y = int(w * 0.15), int(h * 0.6)
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(gray.shape[1], x + w + pad_x), min(gray.shape[0], y + h + pad_y)
    return x0, y0, x1, y1


def downscale_if_needed(img, max_dim=MAX_DIM):
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def locate_and_crop(back_photo_path, out_path) -> bool:
    """Save a tight barcode crop (or full-frame fallback) to out_path.
    Returns True if a confident crop was made, False if it fell back to the
    whole frame."""
    img = load_image(back_photo_path)
    if img is None:
        raise ValueError(f"could not read image: {back_photo_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    box = locate_barcode_crop(gray)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if box:
        x0, y0, x1, y1 = box
        cv2.imwrite(str(out_path), downscale_if_needed(img[y0:y1, x0:x1]))
        return True
    cv2.imwrite(str(out_path), downscale_if_needed(img))
    return False
```

- [ ] **Step 4: Run to verify it passes**

Run: `wsl.exe -d Debian bash -lc "source ~/Ebabyv2/bin/activate && cd '/mnt/c/Users/mardi/Documents/Ebay code' && python3 -m pytest tests/ebaby/test_barcode_locate.py -v"`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add ebaby/stages/barcode_locate.py tests/ebaby/test_barcode_locate.py
git commit -m "feat(ebaby): barcode region locate+crop stage"
```

---

## Task 6: `stages/barcode_decode.py` (multi-view decode ladder)

**Files:**
- Create: `ebaby/stages/barcode_decode.py`
- Test: `tests/ebaby/test_barcode_decode.py`

Ported near-verbatim from `dvd_barcode_scanner.py`'s `decode_barcodes` —
already function-shaped, so this port is mostly a namespace move plus
dropping the file-scanning `main()` (the batch loop moves to `server.py`,
since it needs per-set state, not a standalone CLI).

- [ ] **Step 1: Write the failing tests**

```python
# tests/ebaby/test_barcode_decode.py
import cv2
import numpy as np
import pytest

from ebaby.stages import barcode_decode as bd

try:
    import barcode as _pybarcode  # python-barcode, used only to synthesize a test fixture
    from barcode.writer import ImageWriter
    _HAVE_PYBARCODE = True
except ImportError:
    _HAVE_PYBARCODE = False


@pytest.mark.skipif(not _HAVE_PYBARCODE, reason="python-barcode not installed; "
                     "install with `pip install python-barcode` to generate a "
                     "synthetic EAN-13 fixture for this test")
def test_decode_barcodes_reads_a_generated_ean13(tmp_path):
    ean = _pybarcode.get("ean13", "400638133393", writer=ImageWriter())
    png_path = ean.save(str(tmp_path / "code"))
    img = cv2.imread(png_path)
    result = bd.decode_barcodes(img)
    assert result and result[0].isdigit()


def test_decode_barcodes_returns_empty_list_on_blank_image():
    blank = np.full((200, 400, 3), 255, dtype=np.uint8)
    assert bd.decode_barcodes(blank) == []


def test_normalize_barcode_strips_leading_zero_for_upca_via_ean13():
    assert bd._normalize_barcode("0883316276402", "EAN13") == "883316276402"


def test_normalize_barcode_leaves_other_symbologies_alone():
    assert bd._normalize_barcode("012345678905", "UPCA") == "012345678905"
```

- [ ] **Step 2: Run to verify it fails**

Run: `wsl.exe -d Debian bash -lc "source ~/Ebabyv2/bin/activate && cd '/mnt/c/Users/mardi/Documents/Ebay code' && python3 -m pytest tests/ebaby/test_barcode_decode.py -v"`
Expected: FAIL — `ModuleNotFoundError: No module named 'ebaby.stages.barcode_decode'`

- [ ] **Step 3: Write the implementation**

```python
# ebaby/stages/barcode_decode.py
"""Decode a barcode from an image using many high-contrast views — ported
from dvd_barcode_scanner.py's decode_barcodes(). Input is expected to already
be a tight barcode crop (or a full-frame fallback) from
ebaby.stages.barcode_locate; this module focuses on scale/contrast/sharpness
variations to coax a read out of it.
"""
import cv2
from pyzbar import pyzbar

PRODUCT_SYMBOLS = {"EAN13", "UPCA", "EAN8", "UPCE"}


def _locate_barcode_crop(gray):
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=-1)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=-1)
    grad = cv2.convertScaleAbs(cv2.subtract(cv2.convertScaleAbs(grad_x),
                                            cv2.convertScaleAbs(grad_y)))
    blurred = cv2.blur(grad, (9, 9))
    _, thresh = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    pad = 15
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
    crop = gray[y0:y1, x0:x1]
    return crop if crop.size > 0 else None


def _candidates(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    sources = [gray]
    crop = _locate_barcode_crop(gray)
    if crop is not None:
        sources.append(crop)
    for src in sources:
        for scale in (1, 2, 3):
            g = cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1 else src
            yield g
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
            yield clahe
            _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            yield otsu
            yield cv2.bitwise_not(otsu)
            blur = cv2.GaussianBlur(g, (0, 0), 3)
            yield cv2.addWeighted(g, 1.5, blur, -0.5, 0)
            yield cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 15, 4)


def _normalize_barcode(value, sym_type):
    """A UPC-A code comes back from zbar as a 13-digit EAN13 with a leading
    '0' — strip it so the saved barcode matches the 12-digit UPC-A actually
    printed on the case."""
    if sym_type == "EAN13" and len(value) == 13 and value.startswith("0"):
        return value[1:]
    return value


def decode_barcodes(image, product_only=True):
    """Unique digit strings in first-seen order, normalized. Stops as soon as
    a product barcode is found."""
    seen = []
    for view in _candidates(image):
        for sym in pyzbar.decode(view):
            if product_only and sym.type not in PRODUCT_SYMBOLS:
                continue
            raw_value = sym.data.decode("utf-8", "ignore").strip()
            value = _normalize_barcode(raw_value, sym.type)
            if value and value not in seen:
                seen.append(value)
        if seen:
            break
    return seen
```

- [ ] **Step 4: Run to verify it passes**

Run: `wsl.exe -d Debian bash -lc "source ~/Ebabyv2/bin/activate && cd '/mnt/c/Users/mardi/Documents/Ebay code' && python3 -m pytest tests/ebaby/test_barcode_decode.py -v"`
Expected: `4 passed` if `python-barcode` is installed, else `3 passed, 1 skipped`. If you want the generated-barcode test to actually run: `pip install python-barcode` in the WSL venv (dev-only dependency, not added to `requirements-ebaby.txt` since it's a test fixture generator, not runtime code).

- [ ] **Step 5: Commit**

```bash
git add ebaby/stages/barcode_decode.py tests/ebaby/test_barcode_decode.py
git commit -m "feat(ebaby): multi-view barcode decode stage"
```

---

## Task 7: `stages/ebay.py` (eBay AU lookup + listing rows)

**Files:**
- Create: `ebaby/stages/ebay.py`
- Test: `tests/ebaby/test_ebay.py`

Combines `ebay_api.py` (token fetch, retry/backoff session, `EbayUnavailable`)
with `ebay_csv_extractor_new.py`'s dual New/Used search + item-specifics
logic, producing one row per barcode with the exact 13 columns the spec
requires. All HTTP is mocked in tests — no live eBay calls.

- [ ] **Step 1: Write the failing tests**

```python
# tests/ebaby/test_ebay.py
from unittest.mock import patch, MagicMock

import pytest
import requests

from ebaby.stages import ebay


def _resp(json_body, status=200):
    m = MagicMock()
    m.status_code = status
    m.json.return_value = json_body
    m.raise_for_status = MagicMock()
    return m


def test_get_token_raises_when_not_configured(monkeypatch):
    monkeypatch.setattr(ebay.cfg, "ebay_configured", lambda: False)
    with pytest.raises(RuntimeError, match="credentials missing"):
        ebay.get_token()


@patch("ebaby.stages.ebay._SESSION.post")
def test_get_token_returns_access_token(mock_post, monkeypatch):
    monkeypatch.setattr(ebay.cfg, "ebay_configured", lambda: True)
    monkeypatch.setattr(ebay.cfg, "EBAY_CLIENT_ID", "id")
    monkeypatch.setattr(ebay.cfg, "EBAY_CLIENT_SECRET", "secret")
    mock_post.return_value = _resp({"access_token": "tok123"})
    assert ebay.get_token() == "tok123"


@patch("ebaby.stages.ebay._SESSION.post")
def test_get_token_network_failure_raises_ebay_unavailable(mock_post, monkeypatch):
    monkeypatch.setattr(ebay.cfg, "ebay_configured", lambda: True)
    monkeypatch.setattr(ebay.cfg, "EBAY_CLIENT_ID", "id")
    monkeypatch.setattr(ebay.cfg, "EBAY_CLIENT_SECRET", "secret")
    mock_post.side_effect = requests.exceptions.ConnectionError("boom")
    with pytest.raises(ebay.EbayUnavailable, match="no network"):
        ebay.get_token()


@patch("ebaby.stages.ebay._SESSION.get")
def test_fetch_listing_row_prefers_new_price_and_specifics(mock_get):
    def side_effect(url, headers=None, params=None):
        if "item_summary/search" in url and "1000|1500|1750" in params["filter"]:
            return _resp({"itemSummaries": [
                {"itemId": "NEW1", "title": "New Listing Title",
                 "price": {"value": "10.00"}, "shippingOptions": []}
            ]})
        if "item_summary/search" in url:  # used-condition search
            return _resp({"itemSummaries": [
                {"itemId": "USED1", "title": "Used Listing Title",
                 "price": {"value": "5.00"}, "shippingOptions": []}
            ]})
        if "item/NEW1" in url:
            return _resp({"title": "Canonical Title",
                          "localizedAspects": [{"name": "Genre", "value": "Horror"}]})
        raise AssertionError(f"unexpected url {url}")
    mock_get.side_effect = side_effect

    row = ebay.fetch_listing_row("400638133393", token="tok")
    assert row["Title"] == "Canonical Title"
    assert row["Image Set Name"] == "Canonical_Title"
    assert row["Genre"] == "Horror"
    assert row["Lowest Price New (AUD)"] == 10.00
    assert row["Lowest Price Used (AUD)"] == 5.00


@patch("ebaby.stages.ebay._SESSION.get")
def test_fetch_listing_row_returns_none_when_nothing_found(mock_get):
    mock_get.return_value = _resp({"itemSummaries": []})
    assert ebay.fetch_listing_row("000000000000", token="tok") is None


def test_write_csv_matches_the_thirteen_required_columns(tmp_path):
    rows = [{"Barcode": "123", "Image Set Name": "Movie", "Title": "The Movie",
             "Region Code": "4", "Genre": "Drama", "Type": "", "Season": "",
             "Actor": "", "Studio": "", "Language": "", "Rating": "",
             "Lowest Price New (AUD)": 12.5, "Lowest Price Used (AUD)": ""}]
    out = tmp_path / "Ebay_Details.csv"
    ebay.write_csv(rows, out)
    header = out.read_text(encoding="utf-8").splitlines()[0]
    assert header == ("Barcode,Image Set Name,Title,Region Code,Genre,Type,"
                      "Season,Actor,Studio,Language,Rating,"
                      "Lowest Price New (AUD),Lowest Price Used (AUD)")
```

- [ ] **Step 2: Run to verify it fails**

Run: `wsl.exe -d Debian bash -lc "source ~/Ebabyv2/bin/activate && cd '/mnt/c/Users/mardi/Documents/Ebay code' && python3 -m pytest tests/ebaby/test_ebay.py -v"`
Expected: FAIL — `ModuleNotFoundError: No module named 'ebaby.stages.ebay'`

- [ ] **Step 3: Write the implementation**

```python
# ebaby/stages/ebay.py
"""eBay Browse (buy) API — app-only token, New/Used dual search, item
specifics, and CSV writing. Ported from ebay_api.py + ebay_csv_extractor_new.py.

Network access is best-effort: transient failures are retried with backoff;
a hard outage raises EbayUnavailable so callers can show a clean message
instead of a stack trace and let the user choose retry-or-continue.
"""
import base64
import csv
import time
import urllib.parse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import pipeline_config as cfg  # reuses the user's existing .env-backed config module
from ebaby.naming_utils import slugify_title

_TOKEN_URL = "https://api.ebay.com/identity/v1/oauth2/token"
_SEARCH_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"
_ITEM_URL = "https://api.ebay.com/buy/browse/v1/item/"

NEW_CONDITIONS = "1000|1500|1750"
USED_CONDITIONS = "3000|4000|5000|6000"

CSV_HEADERS = [
    "Barcode", "Image Set Name", "Title", "Region Code", "Genre", "Type",
    "Season", "Actor", "Studio", "Language", "Rating",
    "Lowest Price New (AUD)", "Lowest Price Used (AUD)",
]


class EbayUnavailable(RuntimeError):
    """Raised when eBay can't be reached (network/DNS/timeout/5xx after retries)."""


def _make_session():
    retry = Retry(
        total=3, connect=3, read=3, backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"), raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess = requests.Session()
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess


_SESSION = _make_session()


def _root_cause(exc):
    if isinstance(exc, requests.exceptions.ConnectionError):
        return "no network / DNS"
    if isinstance(exc, requests.exceptions.Timeout):
        return "timed out"
    return type(exc).__name__


def get_token() -> str:
    if not cfg.ebay_configured():
        raise RuntimeError(
            "eBay credentials missing — set EBAY_CLIENT_ID / EBAY_CLIENT_SECRET in the project .env")
    creds = base64.b64encode(f"{cfg.EBAY_CLIENT_ID}:{cfg.EBAY_CLIENT_SECRET}".encode()).decode()
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Authorization": f"Basic {creds}"}
    payload = "grant_type=client_credentials&scope=https://api.ebay.com/oauth/api_scope"
    try:
        res = _SESSION.post(_TOKEN_URL, headers=headers, data=payload, timeout=30)
    except requests.exceptions.RequestException as e:
        raise EbayUnavailable(f"can't reach eBay auth ({_root_cause(e)})") from e
    res.raise_for_status()
    return res.json()["access_token"]


def _search_condition(barcode, condition_ids, headers):
    query = f"q={barcode}&filter=conditionIds:{{{condition_ids}}}&limit=10"
    try:
        res = _SESSION.get(f"{_SEARCH_URL}?{query}", headers=headers)
        res.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise EbayUnavailable(f"can't reach eBay ({_root_cause(e)})") from e
    summaries = res.json().get("itemSummaries", [])
    if not summaries:
        return None, None, None
    prices = []
    for item in summaries:
        try:
            item_price = float(item.get("price", {}).get("value", 0))
            shipping_opts = item.get("shippingOptions", [])
            shipping_cost = float(shipping_opts[0].get("shippingCost", {}).get("value", 0)) if shipping_opts else 0.0
            prices.append(item_price + shipping_cost)
        except (ValueError, TypeError):
            continue
    lowest_price = min(prices) if prices else None
    top = summaries[0]
    return lowest_price, top.get("itemId"), top.get("title", "")


def _fetch_item_specifics(item_id, headers):
    encoded = urllib.parse.quote(item_id)
    try:
        res = _SESSION.get(f"{_ITEM_URL}{encoded}", headers=headers)
        res.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise EbayUnavailable(f"can't reach eBay ({_root_cause(e)})") from e
    data = res.json()
    aspects = {a.get("name", "").lower(): a.get("value", "") for a in data.get("localizedAspects", [])}
    return data.get("title", ""), aspects


def fetch_listing_row(barcode, token, marketplace=None):
    """One CSV row (dict, CSV_HEADERS keys) or None if nothing found on
    either New or Used condition search."""
    marketplace = marketplace or cfg.EBAY_MARKETPLACE_ID
    headers = {"Authorization": f"Bearer {token}", "X-EBAY-C-MARKETPLACE-ID": marketplace,
               "Content-Type": "application/json"}

    lowest_new, new_item_id, new_title = _search_condition(barcode, NEW_CONDITIONS, headers)
    lowest_used, used_item_id, used_title = _search_condition(barcode, USED_CONDITIONS, headers)
    if lowest_new is None and lowest_used is None:
        return None

    specifics_item_id = new_item_id or used_item_id
    fallback_title = new_title or used_title
    title, aspects = ("", {})
    if specifics_item_id:
        title, aspects = _fetch_item_specifics(specifics_item_id, headers)
    if not title:
        title = fallback_title

    slug = slugify_title(title, fallback=barcode)
    return {
        "Barcode": barcode,
        "Image Set Name": slug,
        "Title": title,
        "Region Code": aspects.get("region code", ""),
        "Genre": aspects.get("genre", ""),
        "Type": aspects.get("type", ""),
        "Season": aspects.get("season", ""),
        "Actor": aspects.get("actor", aspects.get("cast", "")),
        "Studio": aspects.get("studio", ""),
        "Language": aspects.get("language", ""),
        "Rating": aspects.get("rating", aspects.get("movie/tv title", "")),
        "Lowest Price New (AUD)": round(lowest_new, 2) if lowest_new is not None else "",
        "Lowest Price Used (AUD)": round(lowest_used, 2) if lowest_used is not None else "",
    }


def fetch_all(barcodes, token, delay=0.5):
    """rows = [{...} or {"Barcode": b} if nothing found, ...], same order as
    barcodes. Raises EbayUnavailable immediately on outage so the caller can
    offer retry/continue rather than silently returning partial data."""
    rows = []
    for barcode in barcodes:
        row = fetch_listing_row(barcode, token)
        rows.append(row or {"Barcode": barcode})
        time.sleep(delay)
    return rows


def write_csv(rows, out_path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
```

- [ ] **Step 4: Run to verify it passes**

Run: `wsl.exe -d Debian bash -lc "source ~/Ebabyv2/bin/activate && cd '/mnt/c/Users/mardi/Documents/Ebay code' && python3 -m pytest tests/ebaby/test_ebay.py -v"`
Expected: `6 passed`

Note: this module imports `pipeline_config` as a top-level module (matching
the existing scripts' own import style: `import pipeline_config as cfg`).
For this import to resolve inside `ebaby/`, copy
`Pipeline/2_barcode_ebay/test rename/pipeline_config.py` to
`ebaby/pipeline_config.py` in this same step (it's a small, dependency-free
module — same treatment as `naming_utils.py` in Task 1):

```bash
cp "/mnt/c/Users/mardi/Documents/Ebay code/.claude/worktrees/busy-napier-89a58d" 2>/dev/null; true
```

Actually copy it directly with the Read/Write tools from
`\\wsl.localhost\Debian\home\M\Ebaby Code\Pipeline\2_barcode_ebay\test rename\pipeline_config.py`
into `ebaby/pipeline_config.py`, unchanged except the `.env` search now also
needs to find the project's `.env` — the existing `_load_env_file()` already
walks up through all parent directories of `__file__`, so placing the file at
`ebaby/pipeline_config.py` means it searches `ebaby/`, the repo root, and
above; put the real `.env` (or a symlink to the WSL one) at the repo root so
it's found. Confirm with:

Run: `wsl.exe -d Debian bash -lc "source ~/Ebabyv2/bin/activate && cd '/mnt/c/Users/mardi/Documents/Ebay code' && python3 -c 'from ebaby import pipeline_config as cfg; print(cfg.ebay_configured())'"`
Expected: `True` (once the `.env` is in place) — if `False`, copy
`Pipeline/2_barcode_ebay/test rename/.env` to the repo root before continuing
to Task 8.

- [ ] **Step 5: Commit**

```bash
git add ebaby/stages/ebay.py ebaby/pipeline_config.py tests/ebaby/test_ebay.py
git commit -m "feat(ebaby): eBay AU lookup + listing CSV stage"
```

---

## Task 8: `stages/rename_title.py` (final slug rename, before crop)

**Files:**
- Create: `ebaby/stages/rename_title.py`
- Test: `tests/ebaby/test_rename_title.py`

Ported from `ebay_Api_rename_new.py`'s `plan_renames`/`apply_renames`, but
decoupled from re-reading barcode `.txt` files and eBay lookups from disk —
this stage receives the already-known `{set_key: barcode}` and
`{set_key: title}` maps from the barcode/eBay stages (in-memory, via
`batch.state.json`), since those stages already ran. This is the **last**
rename before crop, exactly the ordering the user requires.

- [ ] **Step 1: Write the failing tests**

```python
# tests/ebaby/test_rename_title.py
from pathlib import Path

from ebaby.stages import rename_title as rt


def _touch(d, *names):
    for n in names:
        (d / n).write_bytes(b"x")


def test_plan_renames_used_stock_with_ebay_title(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _touch(src, "a_front.png", "a_back.png", "a_inside.png")
    sets = {"a": [("front", src / "a_front.png"),
                  ("back", src / "a_back.png"),
                  ("inside", src / "a_inside.png")]}
    key_to_barcode = {"a": "400638133393"}
    key_to_title = {"a": "The Matrix"}
    plan, notes = rt.plan_renames(sets, key_to_barcode, key_to_title, dst)
    names = sorted(new.name for _, new in plan)
    assert names == ["Matrix_back.png", "Matrix_front.png", "Matrix_inside.png"]
    assert notes == []


def test_plan_renames_new_stock_tags_new_suffix(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _touch(src, "01_front.png", "01_back.png")
    sets = {"01": [("front", src / "01_front.png"), ("back", src / "01_back.png")]}
    plan, notes = rt.plan_renames(sets, {"01": "400638133393"}, {"01": "The Matrix"}, dst)
    names = sorted(new.name for _, new in plan)
    assert names == ["Matrix_back_new.png", "Matrix_front_new.png"]


def test_plan_renames_no_ebay_match_falls_back_to_barcode(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _touch(src, "a_front.png", "a_back.png")
    sets = {"a": [("front", src / "a_front.png"), ("back", src / "a_back.png")]}
    plan, notes = rt.plan_renames(sets, {"a": "400638133393"}, {"a": ""}, dst)
    names = sorted(new.name for _, new in plan)
    assert names == ["400638133393_back.png", "400638133393_front.png"]
    assert notes == [("a", "no eBay match — named by barcode 400638133393")]


def test_plan_renames_no_barcode_uses_placeholder(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _touch(src, "a_front.png", "a_back.png")
    sets = {"a": [("front", src / "a_front.png"), ("back", src / "a_back.png")]}
    plan, notes = rt.plan_renames(sets, {"a": None}, {"a": ""}, dst)
    names = sorted(new.name for _, new in plan)
    assert names == ["NoBarcode_a_back.png", "NoBarcode_a_front.png"]
    assert notes == [("a", "no barcode found — moved with placeholder name, needs manual ID")]


def test_plan_renames_dedupes_colliding_slugs(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _touch(src, "a_front.png", "a_back.png", "b_front.png", "b_back.png")
    sets = {
        "a": [("front", src / "a_front.png"), ("back", src / "a_back.png")],
        "b": [("front", src / "b_front.png"), ("back", src / "b_back.png")],
    }
    key_to_title = {"a": "The Matrix", "b": "The Matrix"}
    plan, _ = rt.plan_renames(sets, {"a": "1", "b": "2"}, key_to_title, dst)
    names = sorted(new.name for _, new in plan)
    assert names == ["Matrix_2_back.png", "Matrix_2_front.png",
                      "Matrix_back.png", "Matrix_front.png"]


def test_apply_renames_moves_files_and_skips_existing_targets(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    _touch(src, "a_front.png")
    (dst / "Matrix_front.png").write_bytes(b"already there")
    plan = [(src / "a_front.png", dst / "Matrix_front.png")]
    done = rt.apply_renames(plan)
    assert done == 0
    assert (src / "a_front.png").exists()  # not clobbered, not moved
```

- [ ] **Step 2: Run to verify it fails**

Run: `wsl.exe -d Debian bash -lc "source ~/Ebabyv2/bin/activate && cd '/mnt/c/Users/mardi/Documents/Ebay code' && python3 -m pytest tests/ebaby/test_rename_title.py -v"`
Expected: FAIL — `ModuleNotFoundError: No module named 'ebaby.stages.rename_title'`

- [ ] **Step 3: Write the implementation**

```python
# ebaby/stages/rename_title.py
"""Final slug rename, moving every image into the renamed-output folder —
ported from ebay_Api_rename_new.py's plan_renames/apply_renames.

Unlike the original script (which re-reads barcode .txt files from disk),
this stage takes the already-known {set_key: barcode} and {set_key: title}
maps as arguments, since the barcode and eBay stages already ran earlier in
this batch. This is the LAST rename before cropping — the exact ordering the
app must guarantee.

Fallback ladder per set (all sets get moved, nothing is left behind):
  eBay match found     -> named after the eBay title slug
  barcode, no match    -> named after the barcode digits
  no barcode at all    -> named "NoBarcode_<key>"
Brand-new (number-keyed) sets get "_new" appended to the role.
"""
from ebaby.naming_utils import slugify_title


def plan_renames(sets, key_to_barcode, key_to_title, dest_dir):
    """sets: {set_key: [(role, path), ...]}
    key_to_barcode: {set_key: barcode_str_or_None}
    key_to_title: {set_key: ebay_title_str_or_empty}
    Returns (renames, notes):
      renames = [(src_path, dest_path), ...]
      notes   = [(set_key, human_reason), ...] for barcode/eBay misses.
    """
    renames = []
    notes = []
    used = set()
    for key, members in sets.items():
        is_new_stock = key.isdigit()
        barcode = key_to_barcode.get(key)

        if barcode:
            slug = slugify_title(key_to_title.get(key, ""), fallback="")
            if not slug:
                slug = barcode
                notes.append((key, f"no eBay match — named by barcode {barcode}"))
        else:
            slug = f"NoBarcode_{key}"
            notes.append((key, "no barcode found — moved with placeholder name, needs manual ID"))

        base, n = slug, 2
        while slug in used:
            slug = f"{base}_{n}"
            n += 1
        used.add(slug)

        for role, path in members:
            tagged_role = f"{role.lower()}_new" if is_new_stock else role.lower()
            dest = dest_dir / f"{slug}_{tagged_role}{path.suffix}"
            if dest != path:
                renames.append((path, dest))
    return renames, notes


def apply_renames(renames):
    """Moves files, skipping (not clobbering) any target that already
    exists. Returns the count actually moved."""
    dest_dirs = {dst.parent for _, dst in renames}
    for d in dest_dirs:
        d.mkdir(parents=True, exist_ok=True)
    done = 0
    for src, dst in renames:
        if dst.exists():
            continue
        src.rename(dst)
        done += 1
    return done
```

- [ ] **Step 4: Run to verify it passes**

Run: `wsl.exe -d Debian bash -lc "source ~/Ebabyv2/bin/activate && cd '/mnt/c/Users/mardi/Documents/Ebay code' && python3 -m pytest tests/ebaby/test_rename_title.py -v"`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add ebaby/stages/rename_title.py tests/ebaby/test_rename_title.py
git commit -m "feat(ebaby): final slug rename stage (last rename before crop)"
```

---

## Task 9: `stages/crop.py` (auto-crop, seeded editor box, compose-on-white)

**Files:**
- Create: `ebaby/stages/crop.py`
- Test: `tests/ebaby/test_crop.py`

Adapted from `redboxflip/scan.py`'s `scan_case` (rembg matte → largest
contour → rotated-rect perspective warp) and `redboxflip/cutout.py`'s
`rembg_matte`. This is the one part of the old app worth reusing — its
detection logic is fine, its **integration** was the problem (it ran before
naming and the editor discarded its output). Here, `detect_crop_box` returns
the quad explicitly so `server.py` can hand it to the browser editor as the
pre-seeded box (fixing the "discards the auto-detected box" bug directly).

- [ ] **Step 1: Write the failing tests**

```python
# tests/ebaby/test_crop.py
import cv2
import numpy as np

from ebaby.stages import crop


def _case_on_white(w=400, h=560, case_w=300, case_h=420, angle=0):
    """A near-white frame with a solid grey rotated rectangle standing in for
    a DVD case, big enough to pass scan_case's coverage gate."""
    frame = np.full((h, w, 3), 245, dtype=np.uint8)
    center = (w // 2, h // 2)
    rect = ((center), (case_w, case_h), angle)
    box = cv2.boxPoints(rect).astype(int)
    cv2.fillConvexPoly(frame, box, (90, 90, 90))
    return frame


def test_detect_crop_box_returns_a_quad_for_a_clear_case():
    frame = _case_on_white()
    result = crop.detect_crop_box(frame)
    assert result is not None
    quad, coverage = result
    assert quad.shape == (4, 2)
    assert 0.06 <= coverage <= 0.99


def test_detect_crop_box_returns_none_for_a_blank_frame():
    blank = np.full((400, 300, 3), 245, dtype=np.uint8)
    assert crop.detect_crop_box(blank) is None


def test_warp_to_quad_produces_axis_aligned_output():
    frame = _case_on_white(angle=8)
    quad, _ = crop.detect_crop_box(frame)
    warped = crop.warp_to_quad(frame, quad)
    assert warped is not None
    assert warped.shape[0] > 0 and warped.shape[1] > 0


def test_compose_on_white_produces_a_square_bgr_image():
    rgba = np.zeros((100, 60, 4), dtype=np.uint8)
    rgba[..., 3] = 255  # fully opaque
    out = crop.compose_on_white(rgba, size=500)
    assert out.shape == (500, 500, 3)


def test_compose_on_white_fills_transparent_pixels_white():
    rgba = np.zeros((10, 10, 4), dtype=np.uint8)  # fully transparent
    out = crop.compose_on_white(rgba, size=20)
    assert tuple(out[10, 10]) == (255, 255, 255)
```

- [ ] **Step 2: Run to verify it fails**

Run: `wsl.exe -d Debian bash -lc "source ~/Ebabyv2/bin/activate && cd '/mnt/c/Users/mardi/Documents/Ebay code' && python3 -m pytest tests/ebaby/test_crop.py -v"`
Expected: FAIL — `ModuleNotFoundError: No module named 'ebaby.stages.crop'`

- [ ] **Step 3: Write the implementation**

```python
# ebaby/stages/crop.py
"""Auto-crop a DVD case photo — adapted from redboxflip/scan.py (scan_case)
and redboxflip/cutout.py (rembg_matte).

detect_crop_box() returns the detected quad explicitly (not just a final
warped image) so the caller can hand it to the browser's corner-drag editor
as the SEEDED box — the old app's editor threw this away and reset to a
dumb 10%-margin rectangle every time; that is the exact bug this module's
API shape is designed to prevent a repeat of.
"""
import cv2
import numpy as np

_PROC_EDGE = 1000
_REMBG_SESSIONS = {}


def rembg_matte(bgr, model="isnet-general-use"):
    """Alpha matte from rembg, or None if rembg/model unavailable."""
    try:
        from rembg import remove, new_session
        from PIL import Image
    except Exception:
        return None
    try:
        if model not in _REMBG_SESSIONS:
            _REMBG_SESSIONS[model] = new_session(model)
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        out = remove(pil, session=_REMBG_SESSIONS[model])
        return np.asarray(out.convert("RGBA").getchannel("A"))
    except Exception:
        return None


def _order_pts(pts):
    pts = np.array(pts, np.float32)
    s = pts.sum(1)
    d = np.diff(pts, 1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def warp_to_quad(img, box):
    box = _order_pts(box)
    tl, tr, br, bl = box
    W = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    H = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if W < 10 or H < 10:
        return None
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(box, dst)
    return cv2.warpPerspective(img, M, (W, H))


def detect_crop_box(bgr, rembg_model="isnet-general-use"):
    """(quad, coverage) or None. quad is 4x2 float32 corners in FULL-resolution
    image coordinates, in the exact order warp_to_quad expects — this is the
    box the browser editor should seed its draggable corners from."""
    h, w = bgr.shape[:2]
    scale = _PROC_EDGE / float(max(h, w))
    small = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))))

    alpha_small = rembg_matte(small, rembg_model)
    if alpha_small is None or alpha_small.shape != small.shape[:2]:
        return None

    _, m = cv2.threshold(alpha_small, 180, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)), iterations=3)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    coverage = float(cv2.contourArea(c)) / float(small.shape[0] * small.shape[1])
    if not (0.06 <= coverage <= 0.99):
        return None

    box = cv2.boxPoints(cv2.minAreaRect(c)) / scale
    return box.astype(np.float32), coverage


def compose_on_white(rgba, size=1600):
    """Fit an RGBA crop onto a white square of side `size`, alpha-composited
    (transparent/near-transparent pixels become white, matching the DVD
    listing photo convention)."""
    h, w = rgba.shape[:2]
    scale = min(size / h, size / w)
    resized = cv2.resize(rgba, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    rh, rw = resized.shape[:2]
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    y0, x0 = (size - rh) // 2, (size - rw) // 2

    rgb = resized[..., :3].astype(np.float32)
    alpha = (resized[..., 3:4].astype(np.float32)) / 255.0
    bg = canvas[y0:y0 + rh, x0:x0 + rw].astype(np.float32)
    blended = rgb * alpha + bg * (1 - alpha)
    canvas[y0:y0 + rh, x0:x0 + rw] = blended.astype(np.uint8)
    return canvas


def crop_and_compose(bgr, quad, size=1600, rembg_model="isnet-general-use"):
    """Given a (possibly user-adjusted) quad, warp+matte+compose to the final
    listing photo. Used both for the initial auto-crop and for re-cropping
    after a manual corner edit in the browser."""
    warped_bgr = warp_to_quad(bgr, quad)
    if warped_bgr is None:
        raise ValueError("quad produced a degenerate warp (too small)")
    alpha = rembg_matte(warped_bgr, rembg_model)
    if alpha is None:
        alpha = np.full(warped_bgr.shape[:2], 255, dtype=np.uint8)
    rgba = np.dstack([cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB), alpha])
    return compose_on_white(rgba, size=size)
```

- [ ] **Step 4: Run to verify it passes**

Run: `wsl.exe -d Debian bash -lc "source ~/Ebabyv2/bin/activate && cd '/mnt/c/Users/mardi/Documents/Ebay code' && python3 -m pytest tests/ebaby/test_crop.py -v"`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add ebaby/stages/crop.py tests/ebaby/test_crop.py
git commit -m "feat(ebaby): auto-crop stage with explicit seedable quad"
```

---

## Task 10: `server.py` (FastAPI orchestration)

**Files:**
- Create: `ebaby/server.py`
- Test: `tests/ebaby/test_server.py`

Thin FastAPI layer: each route calls the stage module against the current
batch and advances `state.json`. Uses FastAPI's `TestClient` (bundled with
`fastapi`, backed by `httpx`) so these tests don't need a running server.

- [ ] **Step 1: Write the failing tests**

```python
# tests/ebaby/test_server.py
import io

import pytest
from fastapi.testclient import TestClient

from ebaby import batch, server


@pytest.fixture(autouse=True)
def isolated_root(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "BATCHES_ROOT", tmp_path / "Ebaby Runs")
    return tmp_path


@pytest.fixture
def client():
    return TestClient(server.app)


def test_create_and_list_batches(client):
    resp = client.post("/api/batches", json={"name": "run1"})
    assert resp.status_code == 200
    resp = client.get("/api/batches")
    assert resp.json() == ["run1"]


def test_create_duplicate_batch_returns_409(client):
    client.post("/api/batches", json={"name": "run1"})
    resp = client.post("/api/batches", json={"name": "run1"})
    assert resp.status_code == 409


def test_get_batch_state(client):
    client.post("/api/batches", json={"name": "run1"})
    resp = client.get("/api/batches/run1/state")
    assert resp.status_code == 200
    assert resp.json()["stage"] == "upload"


def test_upload_and_rename_plan_used_zone(client, tmp_path):
    client.post("/api/batches", json={"name": "run1"})
    for name in ("s1.png", "s2.png", "s3.png"):
        client.post(
            "/api/batches/run1/upload/used",
            files={"file": (name, io.BytesIO(b"x"), "image/png")},
        )
    resp = client.post("/api/batches/run1/rename/plan", json={"zone": "used"})
    assert resp.status_code == 200
    plan = resp.json()["plan"]
    assert len(plan) == 3
    assert any(p["new_name"] == "a_front.png" for p in plan)


def test_rename_plan_uneven_count_returns_422(client):
    client.post("/api/batches", json={"name": "run1"})
    client.post("/api/batches/run1/upload/used",
                files={"file": ("s1.png", io.BytesIO(b"x"), "image/png")})
    resp = client.post("/api/batches/run1/rename/plan", json={"zone": "used"})
    assert resp.status_code == 422


def test_rename_apply_advances_state_once_both_zones_done(client):
    client.post("/api/batches", json={"name": "run1"})
    client.post("/api/batches/run1/upload/used",
                files={"file": (n, io.BytesIO(b"x"), "image/png") for n in ["s1.png"]}
                if False else {"file": ("s1.png", io.BytesIO(b"x"), "image/png")})
    # used zone: 3 shots required; add remaining two
    client.post("/api/batches/run1/upload/used",
                files={"file": ("s2.png", io.BytesIO(b"x"), "image/png")})
    client.post("/api/batches/run1/upload/used",
                files={"file": ("s3.png", io.BytesIO(b"x"), "image/png")})
    client.post("/api/batches/run1/upload/new",
                files={"file": ("n1.png", io.BytesIO(b"x"), "image/png")})
    client.post("/api/batches/run1/upload/new",
                files={"file": ("n2.png", io.BytesIO(b"x"), "image/png")})

    client.post("/api/batches/run1/rename/apply", json={"zone": "used"})
    state = client.get("/api/batches/run1/state").json()
    assert state["stage"] == "upload"  # only one of two zones done so far

    client.post("/api/batches/run1/rename/apply", json={"zone": "new"})
    state = client.get("/api/batches/run1/state").json()
    assert state["stage"] == "color"  # both zones renamed -> advance
```

- [ ] **Step 2: Run to verify it fails**

Run: `wsl.exe -d Debian bash -lc "source ~/Ebabyv2/bin/activate && cd '/mnt/c/Users/mardi/Documents/Ebay code' && python3 -m pytest tests/ebaby/test_server.py -v"`
Expected: FAIL — `ModuleNotFoundError: No module named 'ebaby.server'`

- [ ] **Step 3: Write the implementation**

```python
# ebaby/server.py
"""FastAPI orchestration layer. Each route calls a stage module against the
current batch's folder and advances state.json. No stage logic lives here —
this file only sequences calls into ebaby.stages.* and ebaby.batch.
"""
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ebaby import batch
from ebaby.stages import rename_seq

app = FastAPI(title="Ebaby")

ZONE_CONFIG = {
    "used": {"set_size": 3, "scheme": "letter", "labels": rename_seq.USED_LABELS,
             "subdir": "1_originals/used"},
    "new": {"set_size": 2, "scheme": "number", "labels": rename_seq.NEW_LABELS,
            "subdir": "1_originals/new"},
}


class CreateBatchRequest(BaseModel):
    name: str


class ZoneRequest(BaseModel):
    zone: str


@app.post("/api/batches")
def create_batch(req: CreateBatchRequest):
    try:
        batch.create_batch(req.name)
    except batch.BatchError as e:
        return JSONResponse(status_code=409, content={"error": str(e)})
    return {"name": req.name}


@app.get("/api/batches")
def list_batches():
    return batch.list_batches()


@app.get("/api/batches/{name}/state")
def get_state(name: str):
    d = batch.batch_dir(name)
    try:
        return batch.read_state(d)
    except batch.BatchError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})


@app.post("/api/batches/{name}/upload/{zone}")
async def upload(name: str, zone: str, file: UploadFile = File(...)):
    cfg = ZONE_CONFIG[zone]
    dest_dir = batch.batch_dir(name) / cfg["subdir"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file.filename
    dest_path.write_bytes(await file.read())
    return {"saved": str(dest_path)}


@app.post("/api/batches/{name}/rename/plan")
def rename_plan(name: str, req: ZoneRequest):
    cfg = ZONE_CONFIG[req.zone]
    input_dir = batch.batch_dir(name) / cfg["subdir"]
    try:
        plan = rename_seq.build_plan(input_dir, cfg["set_size"], cfg["scheme"], cfg["labels"])
    except ValueError as e:
        return JSONResponse(status_code=422, content={"error": str(e)})
    return {"plan": [{"old_name": old.name, "new_name": new.name} for old, new in plan]}


@app.post("/api/batches/{name}/rename/apply")
def rename_apply(name: str, req: ZoneRequest):
    d = batch.batch_dir(name)
    cfg = ZONE_CONFIG[req.zone]
    input_dir = d / cfg["subdir"]
    try:
        plan = rename_seq.build_plan(input_dir, cfg["set_size"], cfg["scheme"], cfg["labels"])
    except ValueError as e:
        return JSONResponse(status_code=422, content={"error": str(e)})
    count = rename_seq.apply_plan(plan)

    state = batch.read_state(d)
    zones_done = set(state.get("zones_renamed", []))
    zones_done.add(req.zone)
    state["zones_renamed"] = sorted(zones_done)
    batch.write_state(d, state)
    if zones_done >= set(ZONE_CONFIG):
        batch.advance_stage(d, "upload", "color")
    return {"renamed": count, "zones_renamed": sorted(zones_done)}


app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True), name="static")
```

- [ ] **Step 4: Run to verify it passes**

Run: `wsl.exe -d Debian bash -lc "source ~/Ebabyv2/bin/activate && cd '/mnt/c/Users/mardi/Documents/Ebay code' && python3 -m pytest tests/ebaby/test_server.py -v"`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add ebaby/server.py tests/ebaby/test_server.py
git commit -m "feat(ebaby): FastAPI orchestration for upload+rename stages"
```

---

## Task 11: Wire remaining stages into `server.py` (color, barcode, ebay, rename_title, crop)

**Files:**
- Modify: `ebaby/server.py`
- Test: `tests/ebaby/test_server_stages.py`

Extends the server with routes for every remaining stage, each a thin call
into the already-tested stage modules from Tasks 4–9, following the exact
pattern established in Task 10 (call stage function → advance state → return
JSON). Because color/barcode/crop need real image content (RAW decode,
rembg), these tests use small synthetic PNGs and monkeypatch the
heavy per-file calls to keep the test suite fast and hardware-independent;
the real, non-mocked behaviour of each stage is already covered by that
stage's own test file from its Task.

- [ ] **Step 1: Write the failing tests**

```python
# tests/ebaby/test_server_stages.py
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from ebaby import batch, server


@pytest.fixture(autouse=True)
def isolated_root(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "BATCHES_ROOT", tmp_path / "Ebaby Runs")
    return tmp_path


@pytest.fixture
def client():
    return TestClient(server.app)


def _make_batch_at_stage(client, tmp_path, stage):
    client.post("/api/batches", json={"name": "run1"})
    d = batch.batch_dir("run1")
    state = batch.read_state(d)
    state["stage"] = stage
    batch.write_state(d, state)
    return d


@patch("ebaby.server.color.color_correct_file")
def test_color_run_advances_to_barcode(mock_cc, client, tmp_path):
    d = _make_batch_at_stage(client, tmp_path, "color")
    (d / "1_originals/used/a_front.dng").write_bytes(b"x")
    resp = client.post("/api/batches/run1/color/run")
    assert resp.status_code == 200
    assert mock_cc.called
    assert batch.read_state(d)["stage"] == "barcode"


@patch("ebaby.server.barcode_decode.decode_barcodes", return_value=["400638133393"])
@patch("ebaby.server.barcode_locate.locate_and_crop", return_value=True)
def test_barcode_run_records_digits_and_advances(mock_locate, mock_decode, client, tmp_path):
    d = _make_batch_at_stage(client, tmp_path, "barcode")
    (d / "2_color/a_back.png").write_bytes(b"x")
    resp = client.post("/api/batches/run1/barcode/run")
    assert resp.status_code == 200
    body = resp.json()
    assert body["results"]["a"] == "400638133393"
    assert batch.read_state(d)["stage"] == "ebay"


def test_barcode_manual_override_records_digits(client, tmp_path):
    d = _make_batch_at_stage(client, tmp_path, "barcode")
    resp = client.post("/api/batches/run1/barcode/manual",
                       json={"key": "a", "digits": "400638133393"})
    assert resp.status_code == 200
    state = batch.read_state(d)
    assert state["barcodes"]["a"] == "400638133393"


@patch("ebaby.server.ebay.write_csv")
@patch("ebaby.server.ebay.fetch_all")
@patch("ebaby.server.ebay.get_token", return_value="tok")
def test_ebay_run_writes_csv_and_advances(mock_token, mock_fetch, mock_write, client, tmp_path):
    d = _make_batch_at_stage(client, tmp_path, "ebay")
    state = batch.read_state(d)
    state["barcodes"] = {"a": "400638133393"}
    batch.write_state(d, state)
    mock_fetch.return_value = [{"Barcode": "400638133393", "Image Set Name": "Matrix",
                                "Title": "The Matrix"}]
    resp = client.post("/api/batches/run1/ebay/run")
    assert resp.status_code == 200
    assert batch.read_state(d)["stage"] == "crop"  # rename_title auto-applies then advances
    assert mock_write.called


@patch("ebaby.server.crop.crop_and_compose")
@patch("ebaby.server.crop.detect_crop_box")
def test_crop_run_returns_seeded_quads_per_set(mock_detect, mock_compose, client, tmp_path):
    import numpy as np
    d = _make_batch_at_stage(client, tmp_path, "crop")
    (d / "4_renamed/Matrix_front.png").write_bytes(b"x")
    mock_detect.return_value = (np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype="float32"), 0.5)
    mock_compose.return_value = np.zeros((10, 10, 3), dtype="uint8")
    resp = client.post("/api/batches/run1/crop/run")
    assert resp.status_code == 200
    body = resp.json()
    assert "Matrix_front.png" in body["quads"]
    assert batch.read_state(d)["stage"] == "done"
```

- [ ] **Step 2: Run to verify it fails**

Run: `wsl.exe -d Debian bash -lc "source ~/Ebabyv2/bin/activate && cd '/mnt/c/Users/mardi/Documents/Ebay code' && python3 -m pytest tests/ebaby/test_server_stages.py -v"`
Expected: FAIL — `AttributeError` (routes/imports don't exist yet on `server.app`)

- [ ] **Step 3: Extend the implementation**

Add these imports near the top of `ebaby/server.py` (alongside the existing
`from ebaby.stages import rename_seq`):

```python
import cv2

from ebaby.stages import barcode_decode, barcode_locate, color, crop, ebay
from ebaby.stages import rename_title
```

Add these routes to `ebaby/server.py`, before the `app.mount(...)` line at
the bottom (the static-file mount must stay last, since it's a catch-all):

```python
@app.post("/api/batches/{name}/color/run")
def color_run(name: str):
    d = batch.batch_dir(name)
    out_dir = d / "2_color"
    count = 0
    for zone_dir in (d / "1_originals/used", d / "1_originals/new"):
        for src in sorted(zone_dir.glob("*")):
            if src.suffix.lower() not in (".nef", ".dng"):
                continue
            out_path = out_dir / f"{src.stem}.png"
            color.color_correct_file(src, out_path)
            count += 1
    batch.advance_stage(d, "color", "barcode")
    return {"colored": count}


@app.post("/api/batches/{name}/barcode/run")
def barcode_run(name: str):
    d = batch.batch_dir(name)
    results = {}
    for back_photo in sorted((d / "2_color").glob("*_back*.png")):
        key = back_photo.stem.split("_")[0]
        crop_path = d / "3_barcodes" / f"{back_photo.stem}_barcode.png"
        barcode_locate.locate_and_crop(back_photo, crop_path)
        img = cv2.imread(str(crop_path))
        codes = barcode_decode.decode_barcodes(img) if img is not None else []
        results[key] = codes[0] if codes else None

    state = batch.read_state(d)
    state["barcodes"] = results
    batch.write_state(d, state)
    batch.advance_stage(d, "barcode", "ebay")
    return {"results": results}


class BarcodeManualRequest(BaseModel):
    key: str
    digits: str


@app.post("/api/batches/{name}/barcode/manual")
def barcode_manual(name: str, req: BarcodeManualRequest):
    d = batch.batch_dir(name)
    state = batch.read_state(d)
    state.setdefault("barcodes", {})[req.key] = req.digits
    batch.write_state(d, state)
    return state


@app.post("/api/batches/{name}/ebay/run")
def ebay_run(name: str):
    d = batch.batch_dir(name)
    state = batch.read_state(d)
    barcodes = {k: v for k, v in state.get("barcodes", {}).items() if v}
    token = ebay.get_token()
    rows = ebay.fetch_all(list(barcodes.values()), token)
    ebay.write_csv(rows, d / "Ebay_Details.csv")

    barcode_to_title = {r["Barcode"]: r.get("Title", "") for r in rows if "Barcode" in r}
    key_to_title = {k: barcode_to_title.get(v, "") for k, v in barcodes.items()}
    state["titles"] = key_to_title
    batch.write_state(d, state)

    sets = _collect_sets(d / "2_color")
    plan, notes = rename_title.plan_renames(sets, state.get("barcodes", {}),
                                             key_to_title, d / "4_renamed")
    rename_title.apply_renames(plan)
    state["rename_notes"] = notes
    batch.write_state(d, state)
    batch.advance_stage(d, "ebay", "crop")
    return {"rows": rows, "notes": notes}


def _collect_sets(color_dir):
    """{set_key: [(role, path), ...]} from <key>_<role>.png files."""
    sets = {}
    for p in sorted(color_dir.glob("*.png")):
        key, _, role = p.stem.partition("_")
        sets.setdefault(key, []).append((role, p))
    return sets


@app.post("/api/batches/{name}/crop/run")
def crop_run(name: str):
    d = batch.batch_dir(name)
    quads = {}
    for src in sorted((d / "4_renamed").glob("*.png")):
        bgr = cv2.imread(str(src))
        detected = crop.detect_crop_box(bgr)
        if detected is None:
            h, w = bgr.shape[:2]
            quad = [[0, 0], [w, 0], [w, h], [0, h]]
        else:
            quad, _coverage = detected
            quad = quad.tolist()
        out_bgr = crop.crop_and_compose(bgr, quad)
        out_path = d / "5_cropped" / f"{src.stem}.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), out_bgr)
        quads[src.name] = quad
    batch.advance_stage(d, "crop", "done")
    return {"quads": quads}


class CropManualRequest(BaseModel):
    filename: str
    quad: list


@app.post("/api/batches/{name}/crop/manual")
def crop_manual(name: str, req: CropManualRequest):
    d = batch.batch_dir(name)
    src = d / "4_renamed" / req.filename
    bgr = cv2.imread(str(src))
    out_bgr = crop.crop_and_compose(bgr, req.quad)
    out_path = d / "5_cropped" / f"{src.stem}.jpg"
    cv2.imwrite(str(out_path), out_bgr)
    return {"recropped": req.filename}
```

- [ ] **Step 4: Run to verify it passes**

Run: `wsl.exe -d Debian bash -lc "source ~/Ebabyv2/bin/activate && cd '/mnt/c/Users/mardi/Documents/Ebay code' && python3 -m pytest tests/ebaby/test_server_stages.py -v"`
Expected: `5 passed`

- [ ] **Step 5: Run the full ebaby test suite together**

Run: `wsl.exe -d Debian bash -lc "source ~/Ebabyv2/bin/activate && cd '/mnt/c/Users/mardi/Documents/Ebay code' && python3 -m pytest tests/ebaby -v"`
Expected: all tests from Tasks 1–11 pass (or skip, for the DNG/python-barcode
fixture-gated ones).

- [ ] **Step 6: Commit**

```bash
git add ebaby/server.py tests/ebaby/test_server_stages.py
git commit -m "feat(ebaby): wire color/barcode/ebay/rename/crop stages into the server"
```

---

## Task 12: Static UI (`ebaby/static/`)

**Files:**
- Create: `ebaby/static/index.html`
- Create: `ebaby/static/app.js`
- Create: `ebaby/static/app.css`

No automated test for this task — the UI is manually verified in Task 13 by
actually running the server. Steps here are "write the file," not
"write-test-then-implement," since there's no Python logic to unit test; the
functional check happens in Task 13.

- [ ] **Step 1: Write `ebaby/static/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Ebaby</title>
  <link rel="stylesheet" href="/app.css" />
</head>
<body>
  <h1>Ebaby</h1>
  <div id="stage-rail"></div>

  <section id="stage-upload">
    <h2>1. Upload</h2>
    <div class="dropzones">
      <div class="dropzone" data-zone="used">
        <h3>Used (3-shot: front/back/inside)</h3>
        <input type="file" multiple data-zone-input="used" />
        <div class="file-count" data-zone-count="used">0 files</div>
      </div>
      <div class="dropzone" data-zone="new">
        <h3>Brand New (2-shot: front/back)</h3>
        <input type="file" multiple data-zone-input="new" />
        <div class="file-count" data-zone-count="new">0 files</div>
      </div>
    </div>
    <button id="btn-rename-plan">Preview rename</button>
    <pre id="rename-plan-output"></pre>
    <button id="btn-rename-apply" disabled>Apply rename &amp; continue</button>
  </section>

  <section id="stage-barcode" hidden>
    <h2>2. Barcodes</h2>
    <table id="barcode-table"><thead><tr><th>Set</th><th>Barcode</th><th>Fix</th></tr></thead><tbody></tbody></table>
    <button id="btn-barcode-continue">Continue to eBay lookup</button>
  </section>

  <section id="stage-ebay" hidden>
    <h2>3. eBay listing details</h2>
    <table id="ebay-table"></table>
    <button id="btn-ebay-continue">Rename &amp; continue to crop</button>
  </section>

  <section id="stage-crop" hidden>
    <h2>4. Crop review</h2>
    <div id="crop-grid"></div>
    <canvas id="crop-editor" width="600" height="800" hidden></canvas>
    <button id="btn-crop-save" hidden>Save crop</button>
  </section>

  <section id="stage-done" hidden>
    <h2>Done</h2>
    <p id="done-summary"></p>
  </section>

  <script src="/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write `ebaby/static/app.css`**

```css
body { font-family: sans-serif; max-width: 900px; margin: 2rem auto; }
.dropzones { display: flex; gap: 2rem; }
.dropzone { border: 2px dashed #888; padding: 1rem; flex: 1; }
#stage-rail { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
#stage-rail span { padding: 0.25rem 0.75rem; border-radius: 999px; background: #eee; }
#stage-rail span.active { background: #4a90d9; color: white; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { border: 1px solid #ccc; padding: 0.4rem; text-align: left; }
#crop-grid { display: flex; flex-wrap: wrap; gap: 0.5rem; }
#crop-grid img { width: 140px; cursor: pointer; border: 2px solid transparent; }
#crop-grid img.needs-review { border-color: orange; }
canvas { border: 1px solid #444; }
</style>
```

(Note: no trailing `</style>` tag belongs in a `.css` file — this is fixed in
Step 4's self-review before commit.)

- [ ] **Step 3: Write `ebaby/static/app.js`**

```javascript
const state = { batchName: null, zoneCounts: { used: 0, new: 0 } };

async function api(path, opts = {}) {
  const res = await fetch(`/api${path}`, {
    headers: opts.body instanceof FormData ? {} : { "Content-Type": "application/json" },
    ...opts,
  });
  return res.json();
}

function showStage(id) {
  for (const el of document.querySelectorAll("section[id^='stage-']")) {
    el.hidden = el.id !== id;
  }
}

async function ensureBatch() {
  if (state.batchName) return state.batchName;
  const name = `run-${new Date().toISOString().replace(/[:.]/g, "-")}`;
  await api("/batches", { method: "POST", body: JSON.stringify({ name }) });
  state.batchName = name;
  return name;
}

async function uploadFiles(zone, files) {
  const name = await ensureBatch();
  for (const file of files) {
    const form = new FormData();
    form.append("file", file);
    await api(`/batches/${name}/upload/${zone}`, { method: "POST", body: form });
    state.zoneCounts[zone] += 1;
  }
  document.querySelector(`[data-zone-count="${zone}"]`).textContent =
    `${state.zoneCounts[zone]} files`;
}

for (const zone of ["used", "new"]) {
  document.querySelector(`[data-zone-input="${zone}"]`).addEventListener("change", (e) => {
    uploadFiles(zone, e.target.files);
  });
}

document.getElementById("btn-rename-plan").addEventListener("click", async () => {
  const name = await ensureBatch();
  const output = document.getElementById("rename-plan-output");
  output.textContent = "";
  for (const zone of ["used", "new"]) {
    const result = await api(`/batches/${name}/rename/plan`, {
      method: "POST", body: JSON.stringify({ zone }),
    });
    if (result.error) {
      output.textContent += `${zone}: ${result.error}\n`;
    } else {
      output.textContent += `${zone}:\n` +
        result.plan.map((p) => `  ${p.old_name} -> ${p.new_name}`).join("\n") + "\n";
    }
  }
  document.getElementById("btn-rename-apply").disabled = false;
});

document.getElementById("btn-rename-apply").addEventListener("click", async () => {
  const name = await ensureBatch();
  for (const zone of ["used", "new"]) {
    await api(`/batches/${name}/rename/apply`, {
      method: "POST", body: JSON.stringify({ zone }),
    });
  }
  await runColorAndBarcode(name);
});

async function runColorAndBarcode(name) {
  await api(`/batches/${name}/color/run`, { method: "POST" });
  const barcodeResult = await api(`/batches/${name}/barcode/run`, { method: "POST" });
  renderBarcodeTable(barcodeResult.results);
  showStage("stage-barcode");
}

function renderBarcodeTable(results) {
  const tbody = document.querySelector("#barcode-table tbody");
  tbody.innerHTML = "";
  for (const [key, digits] of Object.entries(results)) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${key}</td><td>${digits ?? "MISS"}</td>` +
      `<td><input type="text" data-fix-key="${key}" placeholder="type digits" /></td>`;
    tbody.appendChild(tr);
  }
}

document.getElementById("btn-barcode-continue").addEventListener("click", async () => {
  const name = state.batchName;
  for (const input of document.querySelectorAll("[data-fix-key]")) {
    if (input.value.trim()) {
      await api(`/batches/${name}/barcode/manual`, {
        method: "POST",
        body: JSON.stringify({ key: input.dataset.fixKey, digits: input.value.trim() }),
      });
    }
  }
  const ebayResult = await api(`/batches/${name}/ebay/run`, { method: "POST" });
  renderEbayTable(ebayResult.rows);
  showStage("stage-ebay");
});

function renderEbayTable(rows) {
  const table = document.getElementById("ebay-table");
  if (!rows.length) { table.innerHTML = "<p>No eBay matches found.</p>"; return; }
  const cols = Object.keys(rows[0]);
  table.innerHTML = `<thead><tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr></thead>` +
    `<tbody>${rows.map((r) => `<tr>${cols.map((c) => `<td>${r[c] ?? ""}</td>`).join("")}</tr>`).join("")}</tbody>`;
}

document.getElementById("btn-ebay-continue").addEventListener("click", async () => {
  const name = state.batchName;
  const cropResult = await api(`/batches/${name}/crop/run`, { method: "POST" });
  renderCropGrid(cropResult.quads);
  showStage("stage-crop");
});

let currentCropFile = null;
let currentQuad = null;

function renderCropGrid(quads) {
  const grid = document.getElementById("crop-grid");
  grid.innerHTML = "";
  for (const filename of Object.keys(quads)) {
    const img = document.createElement("img");
    img.src = `/files/${state.batchName}/5_cropped/${filename.replace(/\.png$/, ".jpg")}`;
    img.dataset.filename = filename;
    img.addEventListener("click", () => openCropEditor(filename, quads[filename]));
    grid.appendChild(img);
  }
}

function openCropEditor(filename, quad) {
  currentCropFile = filename;
  currentQuad = quad.map((pt) => [...pt]);
  const canvas = document.getElementById("crop-editor");
  canvas.hidden = false;
  document.getElementById("btn-crop-save").hidden = false;
  const ctx = canvas.getContext("2d");
  const img = new Image();
  img.onload = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    drawQuad(ctx, canvas);
  };
  img.src = `/files/${state.batchName}/4_renamed/${filename}`;
}

function drawQuad(ctx, canvas) {
  ctx.strokeStyle = "lime";
  ctx.lineWidth = 3;
  ctx.beginPath();
  currentQuad.forEach(([x, y], i) => {
    const px = (x / 3000) * canvas.width;   // scale factor placeholder — real
    const py = (y / 4000) * canvas.height;  // image dimensions come from state
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  });
  ctx.closePath();
  ctx.stroke();
}

document.getElementById("btn-crop-save").addEventListener("click", async () => {
  await api(`/batches/${state.batchName}/crop/manual`, {
    method: "POST",
    body: JSON.stringify({ filename: currentCropFile, quad: currentQuad }),
  });
});
```

- [ ] **Step 4: Fix the CSS placeholder artifact from Step 2**

The `.css` file must not end with a stray `</style>` tag — remove that final
line so `ebaby/static/app.css` ends at the `canvas { ... }` rule.

- [ ] **Step 5: Commit**

```bash
git add ebaby/static/index.html ebaby/static/app.css ebaby/static/app.js
git commit -m "feat(ebaby): single-page UI for the staged pipeline"
```

---

## Task 13: Launcher + manual end-to-end verification

**Files:**
- Create: `Run Ebaby.bat`
- Modify: `ebaby/server.py` (add `if __name__ == "__main__":` uvicorn entrypoint)

- [ ] **Step 1: Add a uvicorn entrypoint to `ebaby/server.py`**

Add at the very end of the file:

```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
```

- [ ] **Step 2: Write `Run Ebaby.bat`**

```bat
@echo off
echo Starting Ebaby server in WSL...
start "" wsl.exe -d Debian bash -lc "source ~/Ebabyv2/bin/activate && cd '/mnt/c/Users/mardi/Documents/Ebay code' && python3 -m ebaby.server"
timeout /t 3 /nobreak >nul
start "" http://localhost:8765
```

- [ ] **Step 3: Manually verify the server starts and serves the UI**

Run: `wsl.exe -d Debian bash -lc "source ~/Ebabyv2/bin/activate && cd '/mnt/c/Users/mardi/Documents/Ebay code' && timeout 5 python3 -m ebaby.server"`
Expected: `Uvicorn running on http://127.0.0.1:8765` in the output before the
5-second timeout kills it (this is just confirming startup, not a full manual
run).

- [ ] **Step 4: Manual functional walkthrough (real hardware required — cannot be scripted)**

Using the app's own preview/browser tooling against `http://localhost:8765`
once the server is left running (drop the `timeout 5` from Step 3):
1. Drop 3 real used-stock RAW files (front/back/inside) into the Used zone
   and 2 brand-new RAW files into the New zone.
2. Click "Preview rename" — confirm the plan shows `a_front/a_back/a_inside`
   and `01_front/01_back`.
3. Click "Apply rename & continue" — confirm the barcode table appears with
   a real decoded barcode (or MISS if the case wasn't photographed with a
   legible barcode — type the digits into the fix box).
4. Click "Continue to eBay lookup" — confirm the eBay table shows a title
   and at least one price for a barcode you know is a real DVD.
5. Click "Rename & continue to crop" — confirm the crop grid shows
   thumbnails named after the eBay title, not "Untitled" and not the
   original camera filenames.
6. Click a thumbnail — confirm the canvas shows the ORIGINAL uncropped photo
   with a rectangle already drawn roughly around the case (the seeded box),
   not a fixed 10%-margin box.

This step has no automated pass/fail — record the result narratively (what
happened at each numbered point) since it needs the user's real camera photos
and real eBay credentials.

- [ ] **Step 5: Commit**

```bash
git add "Run Ebaby.bat" ebaby/server.py
git commit -m "feat(ebaby): launcher script and uvicorn entrypoint"
```

---

## Task 14: End-to-end smoke test (eBay mocked, everything else real)

**Files:**
- Create: `tests/ebaby/test_end_to_end.py`

Walks one batch through every stage using the repo's `samples/redbox/`
JPEGs (`front.jpg`, `back.jpg`, `inside.jpg`) as a 3-shot used-stock set,
with eBay entirely mocked (no live network calls in the automated test
suite — matches the constraint already followed in every stage test).

- [ ] **Step 1: Write the failing test**

```python
# tests/ebaby/test_end_to_end.py
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from ebaby import batch, server

SAMPLES = Path(__file__).resolve().parent.parent.parent / "samples" / "redbox"


@pytest.fixture(autouse=True)
def isolated_root(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "BATCHES_ROOT", tmp_path / "Ebaby Runs")


@pytest.mark.skipif(not SAMPLES.exists(), reason="samples/redbox fixtures not present")
@patch("ebaby.server.ebay.write_csv")
@patch("ebaby.server.ebay.fetch_all")
@patch("ebaby.server.ebay.get_token", return_value="tok")
def test_full_batch_walkthrough(mock_token, mock_fetch, mock_write):
    client = TestClient(server.app)
    client.post("/api/batches", json={"name": "e2e"})

    # Upload the 3-shot used set as a1/a2/a3 so filename sort assigns
    # front/back/inside in the order our fixture filenames imply.
    for src_name, upload_name in [("front.jpg", "s1.jpg"), ("back.jpg", "s2.jpg"),
                                   ("inside.jpg", "s3.jpg")]:
        with open(SAMPLES / src_name, "rb") as f:
            client.post("/api/batches/e2e/upload/used",
                       files={"file": (upload_name, f, "image/jpeg")})
    # Minimal 2-shot new-stock set so the "both zones done" gate passes.
    for upload_name in ["n1.jpg", "n2.jpg"]:
        with open(SAMPLES / "front.jpg", "rb") as f:
            client.post("/api/batches/e2e/upload/new",
                       files={"file": (upload_name, f, "image/jpeg")})

    client.post("/api/batches/e2e/rename/apply", json={"zone": "used"})
    client.post("/api/batches/e2e/rename/apply", json={"zone": "new"})
    assert batch.read_state(batch.batch_dir("e2e"))["stage"] == "color"

    # color/run only processes .nef/.dng — our fixtures are .jpg, so nothing
    # to colour-correct; confirm the stage still advances cleanly.
    client.post("/api/batches/e2e/color/run")
    assert batch.read_state(batch.batch_dir("e2e"))["stage"] == "barcode"

    # Copy the renamed originals into 2_color/ so the barcode stage (which
    # reads from 2_color/) has real *_back*.png files to scan, mirroring
    # what color/run would have produced for RAW input.
    d = batch.batch_dir("e2e")
    for src in (d / "1_originals/used").glob("*"):
        shutil.copy(src, d / "2_color" / f"{src.stem}.png")

    resp = client.post("/api/batches/e2e/barcode/run")
    assert resp.status_code == 200
    assert batch.read_state(d)["stage"] == "ebay"

    mock_fetch.return_value = [{"Barcode": "000000000000", "Image Set Name": "Sample_DVD",
                               "Title": "Sample DVD"}]
    resp = client.post("/api/batches/e2e/ebay/run")
    assert resp.status_code == 200
    assert batch.read_state(d)["stage"] == "crop"
    assert (d / "Ebay_Details.csv") or mock_write.called

    resp = client.post("/api/batches/e2e/crop/run")
    assert resp.status_code == 200
    assert batch.read_state(d)["stage"] == "done"
    assert list((d / "5_cropped").glob("*.jpg"))
```

- [ ] **Step 2: Run to verify it fails**

Run: `wsl.exe -d Debian bash -lc "source ~/Ebabyv2/bin/activate && cd '/mnt/c/Users/mardi/Documents/Ebay code' && python3 -m pytest tests/ebaby/test_end_to_end.py -v"`
Expected: FAIL initially on whatever the first integration mismatch is
(e.g. a route path typo, a stage precondition too strict) — this is the
point of an end-to-end test catching integration bugs the per-stage unit
tests structurally cannot see.

- [ ] **Step 3: Fix whatever the failure reveals**

There's no single fixed diff here — debug using the failure's traceback,
which will point at a specific route in `ebaby/server.py` or a specific
stage function's precondition. Common likely fixes, if hit:
  - `barcode/run` glob pattern `*_back*.png` not matching `s2.png` after
    rename — confirm the rename step in this test renamed `s2.jpg` to
    `a_back.jpg` (not `.png`) since these are JPEG fixtures, not RAW; adjust
    the test's copy-into-`2_color` step to preserve the `.jpg`→`.png`
    rename that `color/run` would normally perform, i.e.
    `shutil.copy(src, d / "2_color" / f"{src.stem}.png")` already does this
    correctly — if the failure is here, check `_collect_sets` in
    `server.py` is reading `.png` from `2_color/`, matching what this test
    writes.
  - `crop/run` failing on `rembg` not installed/slow in CI — if so, this is
    expected to be genuinely slow (rembg CPU inference per image); increase
    the test timeout rather than mocking `crop.detect_crop_box` here, since
    the whole point of this task is to exercise the crop stage for real at
    least once.

- [ ] **Step 4: Run to verify it passes**

Run: `wsl.exe -d Debian bash -lc "source ~/Ebabyv2/bin/activate && cd '/mnt/c/Users/mardi/Documents/Ebay code' && python3 -m pytest tests/ebaby/test_end_to_end.py -v"`
Expected: `1 passed` (or `1 skipped` if `samples/redbox/` isn't present in
this checkout).

- [ ] **Step 5: Run the entire test suite (legacy + new) to confirm no regressions**

Run: `wsl.exe -d Debian bash -lc "source ~/Ebabyv2/bin/activate && cd '/mnt/c/Users/mardi/Documents/Ebay code' && python3 -m pytest -v"`
Expected: every existing `tests/test_rb_*.py` / `tests/test_*.py` test still
passes (untouched legacy code), plus all `tests/ebaby/*` tests pass/skip as
expected.

- [ ] **Step 6: Commit**

```bash
git add tests/ebaby/test_end_to_end.py
git commit -m "test(ebaby): end-to-end batch walkthrough with eBay mocked"
```

---

## Plan Self-Review

**Spec coverage:**
- Fixed workflow order (upload → rename → color → barcode → ebay → rename →
  crop) → enforced by `batch.STAGES` (Task 2) and every `server.py` route
  advancing state only after its own stage completes (Tasks 10–11).
- RAW input, colour correction in-flow → Task 4 (`stages/color.py`), wired
  in Task 11.
- Auto-crop-all + seeded editor (fixing the old app's discarded box) →
  Task 9's `detect_crop_box` returning an explicit quad, consumed by
  `crop/run` (Task 11) and the canvas editor (Task 12).
- Two drop zones, one run → `ZONE_CONFIG` in `server.py` (Task 10), UI drop
  zones in Task 12.
- Built on the proven WSL scripts, not reinvented → every stage module's
  docstring names its source script; Tasks 1, 3–9 are direct ports.
- 13-column listing CSV → `ebay.CSV_HEADERS` (Task 7), test asserts the
  exact header line.
- Legacy code untouched → no task modifies anything under `redboxflip/`,
  `dvdflip/`, or `ebayflip_V2.py`; Task 9 only reads/copies logic patterns
  from `redboxflip/scan.py`/`cutout.py`, it doesn't import or edit them.

**Placeholder scan:** none found — every code step contains complete,
runnable code. The one narrative (non-assert) step is Task 13 Step 4, which
is explicitly scoped as a manual hardware-dependent walkthrough, not a
deferred implementation detail.

**Type/signature consistency check:**
- `rename_seq.build_plan(input_dir, set_size, scheme, labels)` (Task 3) is
  called identically in `server.py`'s `rename_plan`/`rename_apply` (Task 10).
- `rename_title.plan_renames(sets, key_to_barcode, key_to_title, dest_dir)`
  (Task 8) matches the call in `ebay_run` (Task 11) — `sets` built by the
  new `_collect_sets` helper returns exactly the `{key: [(role, path), ...]}`
  shape Task 8's tests exercise.
- `crop.detect_crop_box(bgr)` returns `(quad_ndarray, coverage)` (Task 9);
  `crop_run` (Task 11) unpacks it the same way and calls
  `crop.crop_and_compose(bgr, quad)` with the same `quad` shape
  `warp_to_quad` expects (4x2 float array/list — confirmed compatible since
  `cv2.getPerspectiveTransform` accepts a plain list of 4 points too).
- `ebay.fetch_listing_row` / `ebay.fetch_all` / `ebay.write_csv` field names
  match `ebay.CSV_HEADERS` exactly in all three (Task 7).

No gaps found.
