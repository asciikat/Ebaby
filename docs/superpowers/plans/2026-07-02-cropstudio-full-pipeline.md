# Crop Studio Full Pipeline (Plan 1 of 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the crop auto-detect defect that silently produced a near-blank crop (root cause of the "Sexy Beast" -> "Pool Boy" mistitle), and flip cropstudio's title resolution to be barcode+eBay-primary with Qwen as a grounded fallback, output a flat (no per-DVD subfolder) layout, and log every DVD to a run manifest.

**Architecture:** Extends the existing `cropstudio/` FastAPI + vanilla-JS package in place. Two new leaf modules (`ebay_config.py`, `ebay.py`) ported from the WSL pipeline's `ebay_api.py` with dependency-injectable HTTP sessions for testing. `service.resolve_identity()` is a new function that tries barcode decode -> eBay match first and only calls the existing Qwen path when that fails; `service.save_dvd()` is rewired to call it and to write flat output. A new `manifest.py` appends one JSON entry per DVD, written from `app.py`'s existing `_flush` helper. Two `static/index.html` changes hunt down the crop defect (an aspect-ratio prior in the auto-detect scoring, plus a content-fraction guard before every save) and one JS change adds client-side DVD/slot counters (replacing the fixed `qIndex / 3` math, which breaks once a DVD can have 2 shots instead of 3) plus the new "Finish DVD (2 shots only)" button.

**Tech Stack:** Python (FastAPI, pytest), vanilla JS + OpenCV.js (`cropstudio/static/index.html`), `requests` for the eBay Browse API.

**Scope note:** This is Plan 1 of 3 for the full pipeline-merge design
(`docs/superpowers/specs/2026-07-02-cropstudio-full-pipeline-design.md`).
RAW/DNG ingestion (color-correction) and the pywebview/PyInstaller single-exe
packaging are independent subsystems and get their own plans after this one
ships — packaging in particular needs a bundling-risk spike (rawpy/rembg/
onnxruntime/pyzbar/zxingcpp under PyInstaller) that shouldn't block this work.
The batch-queue thumbnail UI is deferred to the manifest-consuming follow-up
plan since it's read-only sugar on top of the manifest this plan writes.

---

### Task 1: Crop auto-detect — DVD-case aspect-ratio prior

**Files:**
- Modify: `cropstudio/static/index.html:390-426` (the `autoDetect` function)

- [ ] **Step 1: Add the aspect-ratio penalty to the contour scoring loop**

In `cropstudio/static/index.html`, inside `function autoDetect(src) { ... }`,
replace the scoring loop (currently lines 410-423):

```js
            const imgArea = src.cols * src.rows;
            let bestBox = null, bestScore = 0;
            for (let i = 0; i < contours.size(); i++) {
                const c = contours.get(i);
                const area = cv.contourArea(c);
                // Reject specks and the whole-frame background contour.
                if (area >= 0.06 * imgArea && area <= 0.92 * imgArea) {
                    const rr = cv.minAreaRect(c);
                    const rArea = rr.size.width * rr.size.height;
                    const fill = rArea > 0 ? area / rArea : 0;     // 1.0 == perfect rectangle
                    const score = fill * (area / imgArea);          // big AND rectangle-like wins
                    if (fill > 0.55 && score > bestScore) { bestScore = score; bestBox = rotatedRectCorners(rr); }
                }
                c.delete();
            }
```

with:

```js
            const imgArea = src.cols * src.rows;
            // Front/Back covers have a known aspect ratio (redboxflip.scan's
            // proven DVD-case ratio). Penalizing boxes that deviate from it
            // stops a spurious thin high-contrast band (e.g. a glare or
            // reflection edge) from out-scoring the true case boundary just
            // by being a cleaner rectangle. Inside shots (slot 2) vary too
            // much in aspect to apply this prior.
            const DVD_COVER_RATIO = 0.711;
            const currentSlot = (typeof qIndex !== 'undefined') ? (currentDvdSlot()) : 0;
            let bestBox = null, bestScore = 0;
            for (let i = 0; i < contours.size(); i++) {
                const c = contours.get(i);
                const area = cv.contourArea(c);
                // Reject specks and the whole-frame background contour.
                if (area >= 0.06 * imgArea && area <= 0.92 * imgArea) {
                    const rr = cv.minAreaRect(c);
                    const rArea = rr.size.width * rr.size.height;
                    const fill = rArea > 0 ? area / rArea : 0;     // 1.0 == perfect rectangle
                    let score = fill * (area / imgArea);            // big AND rectangle-like wins
                    if (currentSlot !== 2 && rr.size.width > 0 && rr.size.height > 0) {
                        const boxRatio = Math.min(rr.size.width, rr.size.height) /
                                         Math.max(rr.size.width, rr.size.height);
                        const deviation = Math.abs(boxRatio - DVD_COVER_RATIO) / DVD_COVER_RATIO;
                        score *= Math.max(0.15, 1 - deviation);
                    }
                    if (fill > 0.55 && score > bestScore) { bestScore = score; bestBox = rotatedRectCorners(rr); }
                }
                c.delete();
            }
```

This references a `currentDvdSlot()` helper that Task 8 introduces (it
replaces the old `qIndex % 3` slot math). Add a temporary version now so
Task 1 is independently testable/committable:

Add this function directly above `function runAutoDetect() {` (currently
line 373):

```js
        // Which face (0=Back,1=Front,2=Inside) is currently loaded. Task 8
        // replaces the body of this with real client-side DVD/slot counters;
        // for now it mirrors the existing fixed-stride assumption.
        function currentDvdSlot() { return qIndex % 3; }

```

- [ ] **Step 2: Manually verify with the existing sample photos**

This file has no JS test harness (existing project convention — see the
2026-06-23 plan's Task 6, which also relied on manual verification). Run:

```
.venv/Scripts/python.exe -m cropstudio
```

Open `http://127.0.0.1:8765/`, upload a Back+Front+Inside triplet from
`Images in/`, and confirm the auto-detected box still lands on the case
outline (not degraded) for a normal, non-glare photo. This step only proves
no regression — Step 2 of Task 2 is where the actual defect gets a repro.

- [ ] **Step 3: Commit**

```bash
git add cropstudio/static/index.html
git commit -m "fix(cropstudio): add DVD-case aspect-ratio prior to crop auto-detect"
```

---

### Task 2: Crop content-fraction guard before every save

**Files:**
- Modify: `cropstudio/static/index.html` (new `contentFraction()` helper;
  `nextBtn` and `finishBtn` click handlers)

- [ ] **Step 1: Add the `contentFraction()` measurement helper**

Add this function directly above `document.getElementById('nextBtn')...`
(currently line 306), right after `async function postCurrentShot() { ... }`:

```js
        function contentFraction() {
            // Non-white-row span / canvas height on the final composited
            // square. This is the exact measurement used to diagnose a
            // mis-cropped shot in production: a bad crop shows up as mostly
            // white padding with a thin band of real content in the middle
            // (measured range across a real 60-shot batch was 0.5-0.95 for
            // every good shot; the one bad DVD measured 0.20 and 0.27).
            const w = outputCanvas.width, h = outputCanvas.height;
            if (w === 0 || h === 0) return 1;
            const ctx = outputCanvas.getContext('2d');
            const data = ctx.getImageData(0, 0, w, h).data;
            let minRow = -1, maxRow = -1;
            for (let y = 0; y < h; y++) {
                let hasContent = false;
                for (let x = 0; x < w; x += 4) {   // sample every 4th pixel column
                    const i = (y * w + x) * 4;
                    if (data[i] < 245 || data[i + 1] < 245 || data[i + 2] < 245) { hasContent = true; break; }
                }
                if (hasContent) { if (minRow < 0) minRow = y; maxRow = y; }
            }
            if (minRow < 0) return 0;
            return (maxRow - minRow + 1) / h;
        }

        const MIN_CONTENT_FRACTION = 0.4;
```

- [ ] **Step 2: Reproduce the defect against the guard (manual repro)**

Run `.venv/Scripts/python.exe -m cropstudio`, open the app, load the same
"Sexy Beast" source photo that produced the bad crop this session (or any
photo with a strong horizontal glare band), and manually drag the corners
into a thin sliver like the one seen in production. Confirm
`contentFraction()` (check via the browser console:
`console.log(contentFraction())`) reports well under `0.4`. This confirms
the measurement catches the real defect before wiring the block below.

- [ ] **Step 3: Wire the guard into Save & Next and Finish batch**

Replace the `nextBtn` click handler (currently lines 306-315):

```js
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
```

with:

```js
        document.getElementById('nextBtn').addEventListener('click', async () => {
            const frac = contentFraction();
            if (frac < MIN_CONTENT_FRACTION) {
                showToast(`Crop looks mostly blank (${Math.round(frac * 100)}% content) — check the corners`, false);
                return;   // do NOT save or advance
            }
            try {
                const res = await postCurrentShot();
                if (res.status === 'saved') showToast(`Saved "${res.title}" (${res.files.length} files)`, true);
            } catch (err) {
                showToast('Save failed: ' + err.message + ' — click Save & Next to retry', false);
                return;   // do NOT advance; keep the crop on screen
            }
            if (qIndex < queue.length - 1) { qIndex++; loadCurrent(); }
        });
```

And replace the `finishBtn` click handler (currently lines 317-328):

```js
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
```

with:

```js
        document.getElementById('finishBtn').addEventListener('click', async () => {
            const frac = contentFraction();
            if (frac < MIN_CONTENT_FRACTION) {
                showToast(`Crop looks mostly blank (${Math.round(frac * 100)}% content) — check the corners`, false);
                return;
            }
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
```

- [ ] **Step 4: Manually verify the guard blocks the bad crop and allows a good one**

Re-run the Step 2 repro: with the sliver crop still showing, click Save &
Next — confirm it shows the red "Crop looks mostly blank" toast and does
**not** advance. Then drag the corners to the correct case boundary and
click Save & Next again — confirm it saves and advances normally.

- [ ] **Step 5: Commit**

```bash
git add cropstudio/static/index.html
git commit -m "fix(cropstudio): block Save & Next on a near-blank crop"
```

---

### Task 3: eBay credentials config

**Files:**
- Create: `cropstudio/ebay_config.py`
- Test: `tests/test_cropstudio_ebay_config.py`
- Modify: `.gitignore` (ensure `.env` is ignored — check first, it may already be)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cropstudio_ebay_config.py
from cropstudio.ebay_config import load_ebay_config


def test_load_ebay_config_reads_env_file(tmp_path):
    (tmp_path / ".env").write_text(
        "EBAY_CLIENT_ID=abc123\nEBAY_CLIENT_SECRET=secret456\n", encoding="utf-8")
    cfg = load_ebay_config(project_root=tmp_path)
    assert cfg.client_id == "abc123"
    assert cfg.client_secret == "secret456"
    assert cfg.marketplace_id == "EBAY_AU"     # default when not set
    assert cfg.configured is True


def test_load_ebay_config_honors_custom_marketplace(tmp_path):
    (tmp_path / ".env").write_text(
        "EBAY_CLIENT_ID=abc\nEBAY_CLIENT_SECRET=def\nEBAY_MARKETPLACE_ID=EBAY_GB\n",
        encoding="utf-8")
    cfg = load_ebay_config(project_root=tmp_path)
    assert cfg.marketplace_id == "EBAY_GB"


def test_load_ebay_config_missing_file_is_unconfigured(tmp_path):
    cfg = load_ebay_config(project_root=tmp_path)
    assert cfg.client_id == ""
    assert cfg.configured is False


def test_load_ebay_config_ignores_comments_and_blank_lines(tmp_path):
    (tmp_path / ".env").write_text(
        "# eBay creds\n\nEBAY_CLIENT_ID=abc\nEBAY_CLIENT_SECRET=def\n", encoding="utf-8")
    cfg = load_ebay_config(project_root=tmp_path)
    assert cfg.configured is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cropstudio_ebay_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cropstudio.ebay_config'`

- [ ] **Step 3: Write minimal implementation**

```python
# cropstudio/ebay_config.py
"""Reads eBay Browse API credentials from THIS project's own .env.

Deliberately separate from the WSL pipeline's copy (Pipeline/2_barcode_ebay/
.env) — that one was exposed in a chat session and must not be reused as-is.
"""
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EbayConfig:
    client_id: str = ""
    client_secret: str = ""
    marketplace_id: str = "EBAY_AU"

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret)


def _parse_env_file(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def load_ebay_config(project_root=None) -> EbayConfig:
    """Reads EBAY_CLIENT_ID / EBAY_CLIENT_SECRET / EBAY_MARKETPLACE_ID from
    <project_root>/.env, falling back to the process environment.
    `project_root` defaults to this repo's root (parent of `cropstudio/`).
    """
    root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent
    env_path = root / ".env"
    values = dict(os.environ)
    if env_path.exists():
        values.update(_parse_env_file(env_path.read_text(encoding="utf-8")))
    return EbayConfig(
        client_id=values.get("EBAY_CLIENT_ID", ""),
        client_secret=values.get("EBAY_CLIENT_SECRET", ""),
        marketplace_id=values.get("EBAY_MARKETPLACE_ID", "EBAY_AU"),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cropstudio_ebay_config.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Ensure `.env` is gitignored**

Check `.gitignore` for a `.env` entry; if missing, add it:

```bash
grep -q '^\.env$' .gitignore || echo '.env' >> .gitignore
```

- [ ] **Step 6: Commit**

```bash
git add cropstudio/ebay_config.py tests/test_cropstudio_ebay_config.py .gitignore
git commit -m "feat(cropstudio): read eBay credentials from this repo's .env"
```

---

### Task 4: eBay Browse API client

**Files:**
- Create: `cropstudio/ebay.py`
- Test: `tests/test_cropstudio_ebay.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cropstudio_ebay.py
import pytest
import requests

from cropstudio.ebay import EbayUnavailable, get_token, lookup_barcode
from cropstudio.ebay_config import EbayConfig


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = json_data or {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(str(self.status_code))


class _FakeSession:
    def __init__(self, post_response=None, get_response=None, raise_exc=None):
        self._post_response = post_response
        self._get_response = get_response
        self._raise_exc = raise_exc

    def post(self, *a, **kw):
        if self._raise_exc:
            raise self._raise_exc
        return self._post_response

    def get(self, *a, **kw):
        if self._raise_exc:
            raise self._raise_exc
        return self._get_response


def test_get_token_raises_when_not_configured():
    with pytest.raises(RuntimeError, match="credentials missing"):
        get_token(EbayConfig(), session=_FakeSession())


def test_get_token_returns_access_token():
    cfg = EbayConfig(client_id="id", client_secret="secret")
    session = _FakeSession(post_response=_FakeResponse(200, {"access_token": "tok123"}))
    assert get_token(cfg, session=session) == "tok123"


def test_get_token_raises_ebay_unavailable_on_connection_error():
    cfg = EbayConfig(client_id="id", client_secret="secret")
    session = _FakeSession(raise_exc=requests.exceptions.ConnectionError("boom"))
    with pytest.raises(EbayUnavailable, match="no network"):
        get_token(cfg, session=session)


def test_lookup_barcode_found():
    cfg = EbayConfig(client_id="id", client_secret="secret")
    item = {
        "title": "Sexy Beast (DVD, 2001)",
        "price": {"value": "8.50", "currency": "AUD"},
        "categories": [{"categoryId": "617"}],
        "image": {"imageUrl": "http://x/y.jpg"},
        "itemWebUrl": "http://x/item",
    }
    session = _FakeSession(get_response=_FakeResponse(200, {"itemSummaries": [item]}))
    out = lookup_barcode("9325336022306", "tok", cfg, session=session)
    assert out["found"] is True
    assert out["title"] == "Sexy Beast (DVD, 2001)"
    assert out["price"] == "8.50"
    assert out["currency"] == "AUD"


def test_lookup_barcode_not_found():
    cfg = EbayConfig(client_id="id", client_secret="secret")
    session = _FakeSession(get_response=_FakeResponse(200, {"itemSummaries": []}))
    out = lookup_barcode("0000000000000", "tok", cfg, session=session)
    assert out["found"] is False
    assert out["barcode"] == "0000000000000"


def test_lookup_barcode_raises_ebay_unavailable_on_timeout():
    cfg = EbayConfig(client_id="id", client_secret="secret")
    session = _FakeSession(raise_exc=requests.exceptions.Timeout("slow"))
    with pytest.raises(EbayUnavailable, match="timed out"):
        lookup_barcode("123", "tok", cfg, session=session)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cropstudio_ebay.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cropstudio.ebay'`

- [ ] **Step 3: Write minimal implementation**

```python
# cropstudio/ebay.py
"""eBay Browse (buy) API client — app-only (client-credentials) token.

Ported from the WSL pipeline's ebay_api.py. Used to look up a barcode's
matched title/price/specifics. Never used to override a title that came
from Qwen — see service.resolve_identity, which decides priority.
"""
import base64

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_TOKEN_URL = "https://api.ebay.com/identity/v1/oauth2/token"
_SEARCH_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"


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


def get_token(config, session=None) -> str:
    """Fetch an application access token for the Browse API."""
    if not config.configured:
        raise RuntimeError(
            "eBay credentials missing — set EBAY_CLIENT_ID / EBAY_CLIENT_SECRET "
            "in this project's .env")
    session = session or _make_session()
    creds = base64.b64encode(
        f"{config.client_id}:{config.client_secret}".encode()).decode()
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {creds}",
    }
    payload = ("grant_type=client_credentials"
               "&scope=https://api.ebay.com/oauth/api_scope")
    try:
        res = session.post(_TOKEN_URL, headers=headers, data=payload, timeout=30)
    except requests.exceptions.RequestException as e:
        raise EbayUnavailable(f"can't reach eBay auth ({_root_cause(e)})") from e
    res.raise_for_status()
    return res.json()["access_token"]


def lookup_barcode(barcode, token, config, session=None) -> dict:
    """Research one barcode. Returns a dict (always includes 'barcode' and
    'found'); on a hit it also has title/price/currency/category/image_url/
    item_url. Raises EbayUnavailable if the network is unreachable."""
    session = session or _make_session()
    headers = {
        "Authorization": f"Bearer {token}",
        "X-EBAY-C-MARKETPLACE-ID": config.marketplace_id,
    }
    try:
        res = session.get(_SEARCH_URL, headers=headers,
                          params={"q": barcode, "limit": 1}, timeout=30)
    except requests.exceptions.RequestException as e:
        raise EbayUnavailable(f"can't reach eBay ({_root_cause(e)})") from e
    out = {"barcode": barcode, "found": False, "title": "", "price": "",
           "currency": "", "category": "", "image_url": "", "item_url": ""}
    if res.status_code == 200 and res.json().get("itemSummaries"):
        item = res.json()["itemSummaries"][0]
        out.update(
            found=True,
            title=item.get("title", ""),
            price=item.get("price", {}).get("value", ""),
            currency=item.get("price", {}).get("currency", ""),
            category=(item.get("categories", [{}]) or [{}])[0].get("categoryId", ""),
            image_url=item.get("image", {}).get("imageUrl", ""),
            item_url=item.get("itemWebUrl", ""),
        )
    return out


def _root_cause(exc):
    if isinstance(exc, requests.exceptions.ConnectionError):
        return "no network / DNS"
    if isinstance(exc, requests.exceptions.Timeout):
        return "timed out"
    return type(exc).__name__
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cropstudio_ebay.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add cropstudio/ebay.py tests/test_cropstudio_ebay.py
git commit -m "feat(cropstudio): eBay Browse API client for barcode price/specifics lookup"
```

---

### Task 5: `resolve_identity` — barcode+eBay-primary title resolution

**Files:**
- Modify: `cropstudio/service.py` (add `resolve_identity`, keep `resolve_title` untouched for now — Task 6 removes it)
- Test: `tests/test_cropstudio_service.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cropstudio_service.py` (keep the existing tests in the
file untouched — this task is purely additive):

```python
from cropstudio.ebay import EbayUnavailable
from cropstudio.ebay_config import EbayConfig


class _FakeEbayClient:
    def __init__(self, match):
        self._match = match

    def get_token(self, config):
        return "tok"

    def lookup_barcode(self, barcode, token, config):
        return self._match


class _UnreachableEbayClient:
    def get_token(self, config):
        raise EbayUnavailable("no network")

    def lookup_barcode(self, barcode, token, config):
        raise EbayUnavailable("no network")


def test_resolve_identity_uses_ebay_when_barcode_matches():
    shots = {0: _jpeg((1, 1, 1)), 1: _jpeg((2, 2, 2))}
    qwen_calls = []
    result = service.resolve_identity(
        shots, settings=None,
        ebay_config=EbayConfig(client_id="id", client_secret="secret"),
        barcode_decoder=lambda bgr: ("9325336022306", "pyzbar", 0),
        ebay_client=_FakeEbayClient({"found": True, "title": "Sexy Beast (DVD)"}),
        qwen_reader=lambda bgr, s: qwen_calls.append(1) or "SHOULD NOT BE USED")
    assert result["title"] == "Sexy Beast (DVD)"
    assert result["title_source"] == "ebay"
    assert result["barcode"] == "9325336022306"
    assert qwen_calls == []          # Qwen never invoked when barcode+eBay found it


def test_resolve_identity_falls_back_to_qwen_when_no_barcode():
    shots = {0: _jpeg((1, 1, 1)), 1: _jpeg((2, 2, 2))}
    result = service.resolve_identity(
        shots, settings=None, ebay_config=EbayConfig(),
        barcode_decoder=lambda bgr: (None, "", 0),
        qwen_reader=lambda bgr, s: "Amadeus")
    assert result["title"] == "Amadeus"
    assert result["title_source"] == "qwen"
    assert result["barcode"] is None


def test_resolve_identity_falls_back_to_qwen_when_no_ebay_match():
    shots = {0: _jpeg((1, 1, 1)), 1: _jpeg((2, 2, 2))}
    result = service.resolve_identity(
        shots, settings=None,
        ebay_config=EbayConfig(client_id="id", client_secret="secret"),
        barcode_decoder=lambda bgr: ("000000000000", "pyzbar", 0),
        ebay_client=_FakeEbayClient({"found": False}),
        qwen_reader=lambda bgr, s: "King Kong")
    assert result["title"] == "King Kong"
    assert result["title_source"] == "qwen"
    assert result["barcode"] == "000000000000"


def test_resolve_identity_falls_back_to_qwen_when_ebay_unreachable():
    shots = {0: _jpeg((1, 1, 1)), 1: _jpeg((2, 2, 2))}
    result = service.resolve_identity(
        shots, settings=None,
        ebay_config=EbayConfig(client_id="id", client_secret="secret"),
        barcode_decoder=lambda bgr: ("123", "pyzbar", 0),
        ebay_client=_UnreachableEbayClient(),
        qwen_reader=lambda bgr, s: "Fallback Title")
    assert result["title"] == "Fallback Title"
    assert result["title_source"] == "qwen"


def test_resolve_identity_no_barcode_no_qwen_reader_returns_empty():
    shots = {0: _jpeg((1, 1, 1))}
    result = service.resolve_identity(
        shots, settings=None, ebay_config=EbayConfig(),
        barcode_decoder=lambda bgr: (None, "", 0),
        qwen_reader=None)
    assert result["title"] == ""
    assert result["title_source"] == "none"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cropstudio_service.py -v -k resolve_identity`
Expected: FAIL with `AttributeError: module 'cropstudio.service' has no attribute 'resolve_identity'`

- [ ] **Step 3: Write minimal implementation**

In `cropstudio/service.py`, add these imports at the top (alongside the
existing `from redboxflip import naming, titles, vlm`):

```python
from redboxflip import barcode
from .ebay import EbayUnavailable
from . import ebay as _ebay_module
from .ebay_config import load_ebay_config
```

Then add this function (anywhere after `_decode`, e.g. right before
`resolve_title`):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cropstudio_service.py -v`
Expected: PASS — the 5 new `resolve_identity` tests plus all pre-existing
tests in the file (this task doesn't touch `save_dvd` or `resolve_title`).

- [ ] **Step 5: Commit**

```bash
git add cropstudio/service.py tests/test_cropstudio_service.py
git commit -m "feat(cropstudio): barcode+eBay-primary title resolution (resolve_identity)"
```

---

### Task 6: Rewire `save_dvd` onto `resolve_identity` + flat output layout

**Files:**
- Modify: `cropstudio/service.py` (`save_dvd`, remove now-unused `resolve_title` and `_unique_dir`, add `_unique_title`)
- Modify: `tests/test_cropstudio_service.py` (update the 4 pre-existing `save_dvd` tests to the flat layout)
- Modify: `tests/test_cropstudio_app.py` (update 2 pre-existing tests to the flat layout)

This is the behavior-changing task: no more `<Title>/` subfolder per DVD —
every file saves straight into the run folder, with the title in the
filename exactly as resolved (per the locked "no folders... don't change the
titles" decision).

- [ ] **Step 1: Update the pre-existing tests to expect flat output (write first, watch them fail)**

In `tests/test_cropstudio_service.py`, replace these four tests:

```python
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
```

with:

```python
def test_save_dvd_names_from_front_and_writes_three_flat(tmp_path):
    shots = {0: _jpeg((10, 10, 10)), 1: _jpeg((20, 20, 20)), 2: _jpeg((30, 30, 30))}
    out = service.save_dvd(shots, tmp_path, settings=None, dvd_counter=1,
                           reader=lambda bgr, s: "Goober And The Ghost Chasers")
    assert out["title"] == "Goober And The Ghost Chasers"
    assert out["dir"] == str(tmp_path)          # no per-DVD subfolder
    assert (tmp_path / "Goober And The Ghost Chasers - Back Cover.jpg").exists()
    assert (tmp_path / "Goober And The Ghost Chasers - Front Cover.jpg").exists()
    assert (tmp_path / "Goober And The Ghost Chasers - Inside.jpg").exists()
    assert out["used_stock"] is True
    assert out["new_stock"] is False


def test_save_dvd_falls_back_when_reader_returns_none(tmp_path):
    shots = {0: _jpeg((10, 10, 10)), 1: _jpeg((20, 20, 20)), 2: _jpeg((30, 30, 30))}
    out = service.save_dvd(shots, tmp_path, settings=None, dvd_counter=2,
                           reader=lambda bgr, s: None)
    assert out["title"] == "Untitled DVD 2"
    assert (tmp_path / "Untitled DVD 2 - Front Cover.jpg").exists()


def test_save_dvd_uniquifies_duplicate_title_by_filename(tmp_path):
    shots = {0: _jpeg((1, 1, 1)), 1: _jpeg((2, 2, 2)), 2: _jpeg((3, 3, 3))}
    r = lambda bgr, s: "Heat"
    service.save_dvd(shots, tmp_path, None, 1, reader=r)
    out2 = service.save_dvd(shots, tmp_path, None, 2, reader=r)
    assert out2["title"] == "Heat (2)"
    assert (tmp_path / "Heat (2) - Back Cover.jpg").exists()
    assert (tmp_path / "Heat - Back Cover.jpg").exists()   # first one untouched


def test_partial_dvd_uses_back_when_no_front_flat(tmp_path):
    shots = {0: _jpeg((1, 1, 1))}        # only Back
    out = service.save_dvd(shots, tmp_path, None, 1, reader=lambda bgr, s: "Solo")
    assert (tmp_path / "Solo - Back Cover.jpg").exists()
    assert out["new_stock"] is True    # 1 shot counts as new/sealed (< 3), matches 2-shot rule
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cropstudio_service.py -v`
Expected: FAIL — the 4 updated tests fail against the old subfolder-based
`save_dvd` (e.g. `assert out["dir"] == str(tmp_path)` fails because `dir`
currently ends with `/Goober And The Ghost Chasers`).

- [ ] **Step 3: Rewrite `save_dvd`, remove `resolve_title` and `_unique_dir`**

In `cropstudio/service.py`, delete the `resolve_title` function and the
`_unique_dir` function entirely (both now superseded), and replace `save_dvd`
with:

```python
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
             ebay_client=None, quality: int = 92) -> dict:
    """Write one DVD's shots flat into run_dir. Returns
    {title, dir, files, title_source, barcode, new_stock, used_stock}."""
    run_dir = Path(run_dir)
    identity = resolve_identity(shots, settings, ebay_config=ebay_config,
                                barcode_decoder=barcode_decoder,
                                ebay_client=ebay_client, qwen_reader=reader)
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
    }
```

Add `from pathlib import Path` to the top of `cropstudio/service.py` if not
already present (it is not, currently).

- [ ] **Step 4: Update the two pre-existing app-level tests to the flat layout**

In `tests/test_cropstudio_app.py`, replace:

```python
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
```

with:

```python
def test_three_shots_save_a_dvd(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, title="Goober")
    assert _post(c, 0, 0, _jpeg()).json()["status"] == "buffered"
    assert _post(c, 0, 1, _jpeg()).json()["status"] == "buffered"
    r = _post(c, 0, 2, _jpeg()).json()
    assert r["status"] == "saved" and r["title"] == "Goober"
    assert (tmp_path / "Goober - Inside.jpg").exists()
    assert r["used_stock"] is True


def test_finish_flushes_partial(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, title="Solo")
    _post(c, 0, 0, _jpeg())
    _post(c, 0, 1, _jpeg())
    r = c.post("/finish").json()
    assert r["status"] == "saved"
    assert (tmp_path / "Solo - Front Cover.jpg").exists()
    assert r["new_stock"] is True
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cropstudio_service.py tests/test_cropstudio_app.py -v`
Expected: PASS (all tests in both files)

- [ ] **Step 6: Commit**

```bash
git add cropstudio/service.py tests/test_cropstudio_service.py tests/test_cropstudio_app.py
git commit -m "feat(cropstudio): flat output layout + new/used stock tag from shot count"
```

---

### Task 7: Back-cover-barcode vs Front-cover mismatch guard

**Files:**
- Modify: `cropstudio/service.py` (`resolve_identity`)
- Test: `tests/test_cropstudio_service.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cropstudio_service.py`:

```python
def test_resolve_identity_warns_on_front_back_mismatch():
    shots = {0: _jpeg((1, 1, 1)), 1: _jpeg((2, 2, 2))}
    result = service.resolve_identity(
        shots, settings=None,
        ebay_config=EbayConfig(client_id="id", client_secret="secret"),
        barcode_decoder=lambda bgr: ("9325336022306", "pyzbar", 0),
        ebay_client=_FakeEbayClient({"found": True, "title": "Sexy Beast (DVD)"}),
        front_text_reader=lambda bgr, s: "AMERICAN DAD VOLUME 4 SEASON FOUR")
    assert result["title"] == "Sexy Beast (DVD)"
    assert result["mismatch_warning"] is True


def test_resolve_identity_no_warning_when_front_text_matches():
    shots = {0: _jpeg((1, 1, 1)), 1: _jpeg((2, 2, 2))}
    result = service.resolve_identity(
        shots, settings=None,
        ebay_config=EbayConfig(client_id="id", client_secret="secret"),
        barcode_decoder=lambda bgr: ("9325336022306", "pyzbar", 0),
        ebay_client=_FakeEbayClient({"found": True, "title": "Sexy Beast (DVD)"}),
        front_text_reader=lambda bgr, s: "SEXY BEAST a film by jonathan glazer")
    assert result["mismatch_warning"] is False


def test_resolve_identity_no_warning_when_title_came_from_qwen():
    # Qwen-sourced titles have nothing to cross-check against — no guard needed.
    shots = {0: _jpeg((1, 1, 1)), 1: _jpeg((2, 2, 2))}
    result = service.resolve_identity(
        shots, settings=None, ebay_config=EbayConfig(),
        barcode_decoder=lambda bgr: (None, "", 0),
        qwen_reader=lambda bgr, s: "Amadeus",
        front_text_reader=lambda bgr, s: "totally unrelated text")
    assert result["mismatch_warning"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cropstudio_service.py -v -k mismatch`
Expected: FAIL with `TypeError: resolve_identity() got an unexpected keyword argument 'front_text_reader'`

- [ ] **Step 3: Add the guard to `resolve_identity`**

Add `import re` to the top of `cropstudio/service.py`. Add this helper above
`resolve_identity`:

```python
def _title_word_seen_in_text(title: str, text: str) -> bool:
    """True if any 4+ letter word from `title` appears in `text` (case-insensitive).
    Best-effort heuristic — not proof, just enough to flag an obvious mismatch."""
    if not title or not text:
        return False
    words = re.findall(r"[A-Za-z]{4,}", title)
    text_low = text.lower()
    return any(w.lower() in text_low for w in words)
```

Change the `resolve_identity` signature to add `front_text_reader=None`, and
add `"mismatch_warning": False` to the initial `result` dict. Then replace
the `if digits:` block (added in Task 5) in full:

```python
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
```

with:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cropstudio_service.py -v`
Expected: PASS (all tests, including the 3 new mismatch-guard ones)

- [ ] **Step 5: Thread `mismatch_warning` through `save_dvd`'s return dict**

In `cropstudio/service.py`, `save_dvd`, add `"mismatch_warning":
identity["mismatch_warning"]` to the returned dict (alongside the existing
`title_source`/`barcode`/`new_stock`/`used_stock` keys).

Add this test to `tests/test_cropstudio_service.py`:

```python
def test_save_dvd_surfaces_mismatch_warning(tmp_path):
    shots = {0: _jpeg((1, 1, 1)), 1: _jpeg((2, 2, 2))}
    out = service.save_dvd(
        shots, tmp_path, settings=None, dvd_counter=1,
        ebay_config=EbayConfig(client_id="id", client_secret="secret"),
        barcode_decoder=lambda bgr: ("123", "pyzbar", 0),
        ebay_client=_FakeEbayClient({"found": True, "title": "Sexy Beast"}),
        # reader= is the qwen fallback param, unused here since eBay resolved it;
        # front_text_reader isn't a save_dvd param yet — this test only checks
        # the key exists and defaults sanely with no injected front reader.
    )
    assert "mismatch_warning" in out
```

Run: `.venv/Scripts/python.exe -m pytest tests/test_cropstudio_service.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add cropstudio/service.py tests/test_cropstudio_service.py
git commit -m "feat(cropstudio): warn on Back-barcode vs Front-cover title mismatch"
```

---

### Task 8: Client-side DVD/slot counters + "Finish DVD (2 shots only)" button

**Files:**
- Modify: `cropstudio/static/index.html`

The existing client derives `dvd_index = floor(qIndex / 3)` and `slot =
qIndex % 3` from a fixed 3-shots-per-DVD assumption. That breaks the moment
one DVD in the batch has only 2 shots (new/sealed stock) — every DVD after
it would be misaligned. This task replaces the fixed-stride math with real
counters that only advance on explicit user action.

- [ ] **Step 1: Add the counters and the new button's HTML**

Replace this line (currently line 235):

```js
        let qIndex = 0;            // current file index
```

with:

```js
        let qIndex = 0;            // current file index (position in the upload queue)
        let currentDvdIndex = 0;   // which DVD is being built (increments on Save&Next-past-slot-2 or Finish DVD)
        let currentSlot = 0;       // 0=Back, 1=Front, 2=Inside — NOT derived from qIndex, so a
                                   // 2-shot new/sealed DVD doesn't misalign every DVD after it
```

Replace the temporary `currentDvdSlot()` helper added in Task 1:

```js
        // Which face (0=Back,1=Front,2=Inside) is currently loaded. Task 8
        // replaces the body of this with real client-side DVD/slot counters;
        // for now it mirrors the existing fixed-stride assumption.
        function currentDvdSlot() { return qIndex % 3; }
```

with:

```js
        function currentDvdSlot() { return currentSlot; }
```

In the HTML, add the new button right after `finishBtn` (currently line 49):

```html
                    <button id="finishBtn" class="bg-red-900 hover:bg-red-950 text-white px-5 py-2.5 rounded-lg font-semibold shadow-sm hidden">Finish batch</button>
```

becomes:

```html
                    <button id="finishBtn" class="bg-red-900 hover:bg-red-950 text-white px-5 py-2.5 rounded-lg font-semibold shadow-sm hidden">Finish batch</button>
                    <button id="finishDvdBtn" class="bg-red-800 hover:bg-red-900 text-white px-5 py-2.5 rounded-lg font-semibold shadow-sm hidden">Finish DVD (2 shots only)</button>
```

- [ ] **Step 2: Reset counters on a new batch load and use them in `postCurrentShot`/`updatePos`**

Replace (currently lines 238-246):

```js
        document.getElementById('imageInput').addEventListener('change', (e) => {
            const files = Array.from(e.target.files || []);
            if (!files.length) return;
            queue = files; qIndex = 0;
            document.getElementById('prevBtn').classList.remove('hidden');
            document.getElementById('nextBtn').classList.remove('hidden');
            document.getElementById('posText').classList.remove('hidden');
            loadCurrent();
        });
```

with:

```js
        document.getElementById('imageInput').addEventListener('change', (e) => {
            const files = Array.from(e.target.files || []);
            if (!files.length) return;
            queue = files; qIndex = 0; currentDvdIndex = 0; currentSlot = 0;
            document.getElementById('prevBtn').classList.remove('hidden');
            document.getElementById('nextBtn').classList.remove('hidden');
            document.getElementById('posText').classList.remove('hidden');
            loadCurrent();
        });
```

Replace `updatePos` (currently lines 248-255):

```js
        function updatePos() {
            const face = FACE_LABEL[qIndex % 3];
            document.getElementById('posText').textContent =
                `Shot ${qIndex + 1} / ${queue.length} — ${face}`;
            const last = qIndex >= queue.length - 1;
            document.getElementById('nextBtn').classList.toggle('hidden', last);
            document.getElementById('finishBtn').classList.toggle('hidden', !last);
        }
```

with:

```js
        function updatePos() {
            const face = FACE_LABEL[currentSlot];
            document.getElementById('posText').textContent =
                `Shot ${qIndex + 1} / ${queue.length} — ${face}`;
            const last = qIndex >= queue.length - 1;
            document.getElementById('nextBtn').classList.toggle('hidden', last);
            document.getElementById('finishBtn').classList.toggle('hidden', !last);
            // Only offer the 2-shot finish while viewing the Front shot (slot 1) —
            // that's exactly the "Back+Front, no Inside" new/sealed case.
            document.getElementById('finishDvdBtn').classList.toggle('hidden', last || currentSlot !== 1);
        }
```

Replace `postCurrentShot` (currently lines 294-304):

```js
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
```

with:

```js
        async function postCurrentShot() {
            const blob = await currentBlob();
            if (!blob) throw new Error('no crop to save');
            const fd = new FormData();
            fd.append('image', blob, 'shot.jpg');
            fd.append('dvd_index', currentDvdIndex);
            fd.append('slot', currentSlot);
            const r = await fetch('/shot', { method: 'POST', body: fd });
            if (!r.ok) throw new Error('server ' + r.status);
            return r.json();
        }
```

- [ ] **Step 3: Advance the counters correctly on Save & Next, and add the Finish-DVD handler**

Replace the `nextBtn` handler from Task 2 (it currently ends with
`if (qIndex < queue.length - 1) { qIndex++; loadCurrent(); }`):

```js
        document.getElementById('nextBtn').addEventListener('click', async () => {
            const frac = contentFraction();
            if (frac < MIN_CONTENT_FRACTION) {
                showToast(`Crop looks mostly blank (${Math.round(frac * 100)}% content) — check the corners`, false);
                return;   // do NOT save or advance
            }
            try {
                const res = await postCurrentShot();
                if (res.status === 'saved') showToast(`Saved "${res.title}" (${res.files.length} files)`, true);
            } catch (err) {
                showToast('Save failed: ' + err.message + ' — click Save & Next to retry', false);
                return;   // do NOT advance; keep the crop on screen
            }
            if (currentSlot === 2) { currentSlot = 0; currentDvdIndex++; }
            else { currentSlot++; }
            if (qIndex < queue.length - 1) { qIndex++; loadCurrent(); }
        });
```

Add a new handler right after the `finishBtn` handler:

```js
        document.getElementById('finishDvdBtn').addEventListener('click', async () => {
            const frac = contentFraction();
            if (frac < MIN_CONTENT_FRACTION) {
                showToast(`Crop looks mostly blank (${Math.round(frac * 100)}% content) — check the corners`, false);
                return;
            }
            try {
                const res = await postCurrentShot();          // saves Front (slot 1) into the 2-shot buffer
                const f = await fetch('/finish', { method: 'POST' });
                const fj = await f.json();
                if (fj.status === 'saved') showToast(`Saved "${fj.title}" (new/sealed, ${fj.files.length} files)`, true);
                else showToast('Nothing to finish', false);
            } catch (err) {
                showToast('Finish DVD failed: ' + err.message, false);
                return;
            }
            currentSlot = 0; currentDvdIndex++;
            if (qIndex < queue.length - 1) { qIndex++; loadCurrent(); }
        });
```

- [ ] **Step 4: Manually verify a mixed 2-shot + 3-shot batch**

Run `.venv/Scripts/python.exe -m cropstudio`. Upload 5 photos: Back+Front
of a "new/sealed" DVD, then Back+Front+Inside of a normal DVD. On shot 2
(Front of the first DVD), confirm the "Finish DVD (2 shots only)" button is
visible; click it. Confirm it saves a 2-file DVD and advances to shot 3
(which should now load as Back of DVD 2, not get misassigned). Complete the
remaining 3-shot DVD normally with Save & Next / Finish batch and confirm
both DVDs' files appear correctly named with no cross-contamination.

- [ ] **Step 5: Commit**

```bash
git add cropstudio/static/index.html
git commit -m "feat(cropstudio): client-side DVD/slot counters + Finish DVD (2 shots only)"
```

---

### Task 9: Run manifest

**Files:**
- Create: `cropstudio/manifest.py`
- Modify: `cropstudio/app.py` (`_flush`)
- Test: `tests/test_cropstudio_manifest.py`
- Modify: `tests/test_cropstudio_app.py` (assert a manifest entry appears)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cropstudio_manifest.py
from cropstudio import manifest


def test_append_creates_and_grows_manifest(tmp_path):
    manifest.append(tmp_path, {"title": "Amadeus"})
    manifest.append(tmp_path, {"title": "King Kong"})
    entries = manifest.load(tmp_path)
    assert len(entries) == 2
    assert entries[0]["title"] == "Amadeus"
    assert entries[1]["title"] == "King Kong"


def test_load_missing_manifest_returns_empty_list(tmp_path):
    assert manifest.load(tmp_path) == []


def test_manifest_path_lives_in_run_dir(tmp_path):
    assert manifest.manifest_path(tmp_path) == tmp_path / "run_manifest.json"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cropstudio_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cropstudio.manifest'`

- [ ] **Step 3: Write minimal implementation**

```python
# cropstudio/manifest.py
"""Append-only JSON log of every DVD processed in a run.

One entry per DVD, in save order: shot count/new-used tag, barcode, which
title source won (ebay/qwen/none), crop-guard/mismatch flags, output
filenames. Backs the batch-queue view (a later, separate plan).
"""
import json
from pathlib import Path


def manifest_path(run_dir) -> Path:
    return Path(run_dir) / "run_manifest.json"


def load(run_dir) -> list:
    p = manifest_path(run_dir)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def append(run_dir, entry: dict) -> None:
    entries = load(run_dir)
    entries.append(entry)
    manifest_path(run_dir).write_text(json.dumps(entries, indent=2), encoding="utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cropstudio_manifest.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Wire it into `app.py`'s `_flush`**

In `cropstudio/app.py`, add `from . import manifest` to the imports
(alongside `from . import batch, paths, service`), and replace `_flush`:

```python
    def _flush(req):
        counter["n"] += 1
        result = service.save_dvd(req.shots, run_dir, settings, counter["n"])
        manifest.append(run_dir, {
            "dvd_number": counter["n"],
            "title": result["title"],
            "title_source": result["title_source"],
            "barcode": result["barcode"],
            "new_stock": result["new_stock"],
            "used_stock": result["used_stock"],
            "mismatch_warning": result.get("mismatch_warning", False),
            "files": result["files"],
        })
        return result
```

- [ ] **Step 6: Add an app-level test that a manifest entry is written**

Add to `tests/test_cropstudio_app.py`:

```python
from cropstudio import manifest


def test_saving_a_dvd_appends_a_manifest_entry(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, title="Goober")
    _post(c, 0, 0, _jpeg())
    _post(c, 0, 1, _jpeg())
    _post(c, 0, 2, _jpeg())
    entries = manifest.load(tmp_path)
    assert len(entries) == 1
    assert entries[0]["title"] == "Goober"
    assert entries[0]["used_stock"] is True
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cropstudio_manifest.py tests/test_cropstudio_app.py -v`
Expected: PASS (all tests in both files)

- [ ] **Step 8: Commit**

```bash
git add cropstudio/manifest.py cropstudio/app.py tests/test_cropstudio_manifest.py tests/test_cropstudio_app.py
git commit -m "feat(cropstudio): run manifest logging every DVD's processing result"
```

---

### Task 10: Full suite + manual end-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: Run the whole cropstudio suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cropstudio_*.py -v`
Expected: all PASS.

- [ ] **Step 2: Confirm no regression in the rest of the suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: the only failures are the 3 pre-existing retired-`dvdflip/`
failures (`test_main.py` x2, `test_models.py` x1) — unchanged from before
this plan.

- [ ] **Step 3: Manual end-to-end — the actual Sexy Beast repro**

1. Double-click `Run Crop Studio.bat`.
2. Upload the same Sexy Beast Back/Front/Inside photos (or any set that
   previously triggered a bad auto-detect).
3. Confirm the aspect-ratio prior (Task 1) either detects the correct box
   directly, or — if it still mis-detects on a particularly bad glare photo —
   confirm the content-fraction guard (Task 2) blocks the save with the
   "Crop looks mostly blank" toast instead of silently saving garbage.
4. Drag corners to the correct boundary, confirm it saves.
5. Confirm the saved title is NOT hallucinated garbage: if a barcode was
   decoded and eBay is configured, the title should be the real matched
   title; otherwise Qwen's guess, same as before.
6. Check `processed/run_<ts>/run_manifest.json` — confirm one entry per DVD
   with the fields from Task 9.
7. Check the output folder is flat — no `<Title>/` subfolders.

- [ ] **Step 4: Manual end-to-end — mixed new/used batch with real eBay credentials**

(Skip this step if `.env` isn't populated with real eBay credentials yet —
document that it was skipped rather than faking a pass.) Populate
`.env` per Task 3, run a batch with one 2-shot (new/sealed) and one 3-shot
(used) DVD, confirm:
- The 2-shot DVD's manifest entry has `"new_stock": true`.
- If its barcode matches a real eBay listing, `"title_source": "ebay"` and
  the title is the real eBay-matched title, not a Qwen guess.

- [ ] **Step 5: Final commit (docs/notes if any tweaks were needed)**

```bash
git add -A
git commit -m "test(cropstudio): verified full pipeline plan 1 end-to-end" --allow-empty
```
