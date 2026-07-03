# Crop Studio Desktop Exe (Plan 3 of 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package Crop Studio as a single double-clickable Windows app — a pywebview window over the existing FastAPI server, built with PyInstaller — per the 2026-07-02 design spec.

**Architecture:** `cropstudio/desktop.py` starts uvicorn on a random free localhost port in a daemon thread and opens a native pywebview window at it; `cropstudio_desktop.py` (repo root) is the PyInstaller entry. `paths.output_root()` learns a frozen-mode branch so output lands next to the exe instead of inside the bundle. A committed `CropStudio.spec` + `Build Crop Studio Exe.bat` make the build reproducible. Dev mode (`python -m cropstudio`, browser tab) is untouched.

**Tech Stack:** pywebview 6.2 (EdgeWebView2 backend), PyInstaller 6.21 (onedir, `--windowed`), existing venv.

**Spike results (2026-07-04, de-risking done before this plan):** onedir build succeeds in ~60 s with `--add-data` (absolute path!) for `cropstudio/static`, `--hidden-import pyzbar.pyzbar --hidden-import zxingcpp --collect-all pyzbar`. The built exe boots, serves `/` and `/health`, and `/rawdecode` returns a byte-identical corrected JPEG for a real DNG (rawpy DLLs bundle cleanly). rembg/onnxruntime are NOT needed (cropstudio never imports `redboxflip.cutout`). Confirmed defect to fix: output_dir resolved to `dist\CropStudio\_internal\processed\...`.

---

### Task 1: Frozen-mode output path

**Files:**
- Modify: `cropstudio/paths.py`
- Test: `tests/test_cropstudio_paths.py` (append)

- [ ] **Step 1: Append the failing test**

```python
import sys


def test_output_root_next_to_exe_when_frozen(tmp_path, monkeypatch):
    monkeypatch.delenv("CROPSTUDIO_OUTPUT", raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "CropStudio.exe"))
    from cropstudio import paths
    assert paths.output_root() == tmp_path / "processed"
```

- [ ] **Step 2: Run to verify failure**

Run: `".venv/Scripts/python.exe" -m pytest tests/test_cropstudio_paths.py -v`
Expected: the new test FAILS (output_root returns `<project>/processed`, not the exe-relative path); 3 pre-existing tests still pass.

- [ ] **Step 3: Implement**

In `cropstudio/paths.py`, add `import sys` to the imports and replace `output_root()`:

```python
def output_root() -> Path:
    """processed/ root. CROPSTUDIO_OUTPUT overrides; frozen exe writes next
    to itself; dev default is <project>/processed."""
    env = os.environ.get("CROPSTUDIO_OUTPUT")
    if env:
        return Path(env)
    if getattr(sys, "frozen", False):          # PyInstaller bundle
        return Path(sys.executable).resolve().parent / "processed"
    return project_root() / "processed"
```

- [ ] **Step 4: Run to verify pass**

Run: `".venv/Scripts/python.exe" -m pytest tests/test_cropstudio_paths.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add cropstudio/paths.py tests/test_cropstudio_paths.py
git commit -m "fix(cropstudio): frozen exe writes processed/ next to itself"
```

---

### Task 2: Desktop entry point (formalize the spike draft)

**Files:**
- Already drafted in spike: `cropstudio/desktop.py`, `cropstudio_desktop.py` (repo root)
- Test: `tests/test_cropstudio_desktop.py`

- [ ] **Step 1: Write the test** (covers the headless-testable half: free-port helper + server thread pattern)

```python
# tests/test_cropstudio_desktop.py
import threading
import time
import urllib.request

import uvicorn

from cropstudio.desktop import HOST, _free_port


def test_free_port_is_bindable_and_distinct():
    a, b = _free_port(), _free_port()
    assert a > 0 and b > 0     # OS-assigned ephemeral ports


def test_server_thread_serves_health(tmp_path, monkeypatch):
    monkeypatch.setenv("CROPSTUDIO_OUTPUT", str(tmp_path))
    from cropstudio.app import create_app
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(), host=HOST, port=port,
                                           log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 10
    last_err = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://{HOST}:{port}/health", timeout=2) as r:
                assert b'"ok":true' in r.read()
                break
        except Exception as e:
            last_err = e
            time.sleep(0.3)
    else:
        raise AssertionError(f"server never came up: {last_err}")
    server.should_exit = True
```

- [ ] **Step 2: Run the tests**

Run: `".venv/Scripts/python.exe" -m pytest tests/test_cropstudio_desktop.py -v`
Expected: PASS (desktop.py already exists from the spike; if it doesn't in a
fresh checkout, create it with the content below).

`cropstudio/desktop.py` (as drafted+verified in the spike):

```python
"""Desktop entry point: uvicorn on a background thread + a pywebview window.

This is what the PyInstaller exe runs. `python -m cropstudio` (uvicorn +
default browser tab) stays untouched for development.
"""
import socket
import threading

import uvicorn

from .app import create_app

HOST = "127.0.0.1"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def main():
    import webview   # lazy: heavy GUI import only when actually launching

    port = _free_port()
    app = create_app()
    server = uvicorn.Server(uvicorn.Config(app, host=HOST, port=port,
                                           log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()

    window = webview.create_window("Crop Studio", f"http://{HOST}:{port}/",
                                   width=1400, height=950)
    webview.start()          # blocks until the window closes
    server.should_exit = True


if __name__ == "__main__":
    main()
```

`cropstudio_desktop.py` (repo root):

```python
"""PyInstaller entry point for the Crop Studio desktop exe."""
from cropstudio.desktop import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Add pywebview + pyinstaller to requirements**

In `requirements-redbox.txt`, after `rawpy`, add:

```
pywebview
# build-only (exe packaging):
pyinstaller
```

- [ ] **Step 4: Commit**

```bash
git add cropstudio/desktop.py cropstudio_desktop.py tests/test_cropstudio_desktop.py requirements-redbox.txt
git commit -m "feat(cropstudio): pywebview desktop entry point"
```

---

### Task 3: Reproducible build — spec file + build script

**Files:**
- Create: `CropStudio.spec` (repo root)
- Create: `Build Crop Studio Exe.bat` (repo root)

- [ ] **Step 1: Create `CropStudio.spec`**

```python
# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the Crop Studio desktop exe.
# Build with:  .venv\Scripts\pyinstaller.exe --noconfirm CropStudio.spec
from PyInstaller.utils.hooks import collect_all

pyzbar_datas, pyzbar_binaries, pyzbar_hidden = collect_all('pyzbar')

a = Analysis(
    ['cropstudio_desktop.py'],
    pathex=[],
    binaries=pyzbar_binaries,
    datas=[('cropstudio/static', 'cropstudio/static')] + pyzbar_datas,
    hiddenimports=['pyzbar.pyzbar', 'zxingcpp'] + pyzbar_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['rembg', 'onnxruntime'],   # cropstudio never imports redboxflip.cutout
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CropStudio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='CropStudio',
)
```

- [ ] **Step 2: Create `Build Crop Studio Exe.bat`**

```bat
@echo off
REM Build the Crop Studio desktop exe (PyInstaller onedir).
REM Output: dist\CropStudio\CropStudio.exe
cd /d "%~dp0"

if not exist ".venv\Scripts\pyinstaller.exe" (
    echo pyinstaller not installed in .venv
    echo Install it with:  .venv\Scripts\pip install pyinstaller pywebview
    pause
    exit /b 1
)

".venv\Scripts\pyinstaller.exe" --noconfirm CropStudio.spec
echo.
echo Done. Exe at: dist\CropStudio\CropStudio.exe
pause
```

- [ ] **Step 3: Build from the spec and verify**

```bash
cd "C:/Users/mardi/Documents/Ebay code"
".venv/Scripts/pyinstaller.exe" --noconfirm CropStudio.spec
```

Expected: `Build complete!`; `dist/CropStudio/CropStudio.exe` exists.
Then verify the built exe end-to-end (PowerShell): launch it, find its
listening port via `Get-NetTCPConnection -OwningProcess`, check `/health`
returns `ok:true` with an `output_dir` NEXT TO THE EXE (`dist\CropStudio\
processed\run_*` — proves Task 1), check `/` serves the page, POST a real
DNG to `/rawdecode` (expect HTTP 200 image/jpeg), and — the one thing the
spike didn't cover — POST 3 shots through `/shot`/`/finish` and confirm the
flush works in-exe (exercises pyzbar/zxingcpp/Qwen paths inside the bundle).
Kill the process when done.

- [ ] **Step 4: Add `dist/`, `build/`, and generated runtime junk to `.gitignore`**

Append to `.gitignore`:

```
dist/
build/
```

- [ ] **Step 5: Commit**

```bash
git add CropStudio.spec "Build Crop Studio Exe.bat" .gitignore
git commit -m "feat(cropstudio): reproducible PyInstaller build (spec + bat)"
```

---

### Task 4: Full suite + final verification

- [ ] **Step 1:** `".venv/Scripts/python.exe" -m pytest tests/test_cropstudio_*.py -q` — expect 54 passed (51 + 1 paths + 2 desktop).
- [ ] **Step 2:** `".venv/Scripts/python.exe" -m pytest -q` — only the 3 pre-existing retired-`dvdflip` failures.
- [ ] **Step 3:** Human check: double-click `dist\CropStudio\CropStudio.exe`, crop one real DVD end to end, confirm output lands in `dist\CropStudio\processed\`.
- [ ] **Step 4:** `git commit --allow-empty -m "test(cropstudio): desktop exe verified end-to-end"`
