"""FastAPI orchestration layer. Each route calls a stage module against the
current batch's folder and advances state.json. No stage logic lives here —
this file only sequences calls into ebaby.stages.* and ebaby.batch.
"""
import re
import shutil
import threading
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import cv2

from ebaby import batch, naming_utils
from ebaby.stages import barcode_decode, barcode_locate, color, crop, ebay
from ebaby.stages import rename_seq, rename_title

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


def _stage_gate(name, *allowed):
    """(batch_dir, state, None) when the batch exists and is at an allowed
    stage; otherwise (.., .., JSONResponse) with a clean 404/409 instead of
    letting stage modules 500 halfway through their side effects."""
    d = batch.batch_dir(name)
    try:
        state = batch.read_state(d)
    except batch.BatchError as e:
        return d, None, JSONResponse(status_code=404, content={"error": str(e)})
    if allowed and state.get("stage") not in allowed:
        return d, state, JSONResponse(status_code=409, content={
            "error": f"batch is at stage '{state.get('stage')}', "
                     f"this step needs {' or '.join(allowed)}"})
    return d, state, None


# accept_hustle deletes 1_originals/3_barcodes/4_renamed/2_color; crop_run,
# crop_manual and title_edit all read or write files in that same set. All
# four routes run on FastAPI's thread pool, so two of them can genuinely
# overlap on the same batch (e.g. a re-chop still in flight when Accept
# Hustle is clicked) — this lock serializes them per batch name, and the
# "accepted" flag lets the other three refuse cleanly once cleanup has run,
# instead of hitting missing files mid-operation.
_BATCH_LOCKS: dict = {}
_BATCH_LOCKS_GUARD = threading.Lock()


def _batch_lock(name: str) -> threading.Lock:
    with _BATCH_LOCKS_GUARD:
        return _BATCH_LOCKS.setdefault(name, threading.Lock())


def _refuse_if_accepted(state):
    if state.get("accepted"):
        return JSONResponse(status_code=409, content={
            "error": "This batch was finalized with Accept Hustle — its "
                     "working files are gone, only the cropped photos and "
                     "eBay files remain."})
    return None


@app.post("/api/batches/{name}/upload/{zone}")
async def upload(name: str, zone: str, file: UploadFile = File(...)):
    if zone not in ZONE_CONFIG:
        return JSONResponse(status_code=422, content={"error": f"unknown zone '{zone}'"})
    d, _state, err = _stage_gate(name, "upload")
    if err:
        return err
    # basename only — an uploaded filename must never walk out of the batch
    # (split on BOTH separators: the server runs on POSIX, browsers on Windows)
    safe = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not safe or safe in (".", ".."):
        return JSONResponse(status_code=422, content={"error": "file has no usable name"})
    dest_dir = d / ZONE_CONFIG[zone]["subdir"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path, n = dest_dir / safe, 2
    while dest_path.exists():  # duplicate camera names must not overwrite
        dest_path = dest_dir / f"{Path(safe).stem}_{n}{Path(safe).suffix}"
        n += 1
    dest_path.write_bytes(await file.read())
    return {"saved": str(dest_path)}


@app.post("/api/batches/{name}/rename/plan")
def rename_plan(name: str, req: ZoneRequest):
    if req.zone not in ZONE_CONFIG:
        return JSONResponse(status_code=422, content={"error": f"unknown zone '{req.zone}'"})
    d, _state, err = _stage_gate(name)
    if err:
        return err
    cfg = ZONE_CONFIG[req.zone]
    input_dir = d / cfg["subdir"]
    input_dir.mkdir(parents=True, exist_ok=True)
    try:
        plan = rename_seq.build_plan(input_dir, cfg["set_size"], cfg["scheme"], cfg["labels"])
    except ValueError as e:
        return JSONResponse(status_code=422, content={"error": str(e)})
    return {"plan": [{"old_name": old.name, "new_name": new.name} for old, new in plan]}


@app.post("/api/batches/{name}/rename/apply")
def rename_apply(name: str, req: ZoneRequest):
    if req.zone not in ZONE_CONFIG:
        return JSONResponse(status_code=422, content={"error": f"unknown zone '{req.zone}'"})
    d, state, err = _stage_gate(name, "upload")
    if err:
        return err
    # Idempotence guard: re-running the rename on an already-renamed zone
    # would re-group alphabetically and SWAP front/back (a_back sorts first).
    zones_done = set(state.get("zones_renamed", []))
    if req.zone in zones_done:
        return {"renamed": 0, "zones_renamed": sorted(zones_done), "already_done": True}
    cfg = ZONE_CONFIG[req.zone]
    input_dir = d / cfg["subdir"]
    input_dir.mkdir(parents=True, exist_ok=True)
    try:
        plan = rename_seq.build_plan(input_dir, cfg["set_size"], cfg["scheme"], cfg["labels"])
    except ValueError as e:
        return JSONResponse(status_code=422, content={"error": str(e)})
    count = rename_seq.apply_plan(plan)

    zones_done.add(req.zone)
    state["zones_renamed"] = sorted(zones_done)
    batch.write_state(d, state)
    if zones_done >= set(ZONE_CONFIG):
        batch.advance_stage(d, "upload", "color")
    return {"renamed": count, "zones_renamed": sorted(zones_done)}


# RAW formats rawpy can decode; plain formats pass through so a JPG batch
# doesn't silently produce an empty run (2_color feeds every later stage).
RAW_EXTS = (".nef", ".dng", ".cr2", ".cr3", ".arw", ".raf", ".orf", ".rw2")
PLAIN_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


def _set_progress(d, state, stage, done, total):
    """Written per file so the browser can poll /state and show 3/7 instead
    of staring at a spinner for minutes."""
    state["progress"] = {"stage": stage, "done": done, "total": total}
    batch.write_state(d, state)


@app.post("/api/batches/{name}/color/run")
def color_run(name: str):
    d, state, err = _stage_gate(name, "color")
    if err:
        return err
    out_dir = d / "2_color"
    out_dir.mkdir(parents=True, exist_ok=True)
    # RAW+JPEG cameras save BOTH per shot with the same stem; both would
    # write the same <stem>.png and the plain JPEG (sorting after .dng/.nef)
    # would clobber the colour-corrected RAW conversion. One file per stem,
    # RAW preferred.
    by_stem = {}
    for zone_dir in (d / "1_originals/used", d / "1_originals/new"):
        for src in sorted(zone_dir.glob("*")):
            ext = src.suffix.lower()
            if ext not in RAW_EXTS + PLAIN_EXTS:
                continue
            key = (zone_dir.name, src.stem)
            if key not in by_stem or (ext in RAW_EXTS and
                                      by_stem[key].suffix.lower() not in RAW_EXTS):
                by_stem[key] = src
    # group the deduped shots by SET (zone + key before the first "_") so the
    # front/back/inside of one disc share a single white balance and don't
    # come out with three slightly different colour casts
    sets = {}
    for (zone, stem), src in by_stem.items():
        sets.setdefault((zone, stem.split("_")[0]), []).append(src)
    total = sum(len(g) for g in sets.values())
    count, i = 0, 0
    _set_progress(d, state, "color", 0, total)
    for group in sets.values():
        shared = None
        for src in sorted(group):
            out_path = out_dir / f"{src.stem}.png"
            if src.suffix.lower() in RAW_EXTS:
                g = color.color_correct_file(src, out_path, gains=shared)
                count += 1
            else:
                g = color.color_correct_plain_file(src, out_path, gains=shared)
                if g is not None:
                    count += 1
            if shared is None and g is not None:
                shared = g  # first shot sets the balance for the whole set
            i += 1
            _set_progress(d, state, "color", i, total)
    batch.advance_stage(d, "color", "barcode")
    return {"colored": count}


@app.post("/api/batches/{name}/barcode/run")
def barcode_run(name: str):
    d, state, err = _stage_gate(name, "barcode")
    if err:
        return err
    results, crops = {}, {}
    backs = sorted((d / "2_color").glob("*_back*.png"))
    _set_progress(d, state, "barcode", 0, len(backs))
    for i, back_photo in enumerate(backs, 1):
        key = back_photo.stem.split("_")[0]
        crop_path = d / "3_barcodes" / f"{back_photo.stem}_barcode.png"
        located = barcode_locate.locate_and_crop(back_photo, crop_path)
        # prefer the (downscaled) crop even when locate fell back — decoding
        # the full-resolution original hangs for minutes per miss
        img = cv2.imread(str(crop_path))
        if img is None:
            img = cv2.imread(str(back_photo))
        # `located` is deliberately NOT part of this guard: locate_and_crop can
        # report success while the crop it wrote is unreadable, and if the back
        # photo is unreadable too, img is None — decoding None must be skipped
        # (decode_barcodes also guards, but keep the crash off the hot path).
        codes = barcode_decode.decode_barcodes(img) if img is not None else []
        results[key] = codes[0] if codes else None
        if crop_path.is_file():
            crops[key] = crop_path.name  # fix screen shows the actual barcode
        _set_progress(d, state, "barcode", i, len(backs))

    state["barcodes"] = results
    state["barcode_crops"] = crops
    batch.write_state(d, state)
    batch.advance_stage(d, "barcode", "ebay")
    return {"results": results, "crops": crops}


class BarcodeManualRequest(BaseModel):
    key: str
    digits: str


@app.post("/api/batches/{name}/barcode/manual")
def barcode_manual(name: str, req: BarcodeManualRequest):
    d, state, err = _stage_gate(name)
    if err:
        return err
    # digits only — this string later becomes a FILENAME when eBay has no
    # title, so slashes/quotes in here would nest or crash the final rename
    digits = re.sub(r"\D", "", req.digits)
    if not 8 <= len(digits) <= 14:
        return JSONResponse(status_code=422, content={
            "error": f"'{req.digits}' doesn't look like a barcode (need 8-14 digits)"})
    state.setdefault("barcodes", {})[req.key] = digits
    batch.write_state(d, state)
    return state


def _windows_path(p):
    """/mnt/c/... -> C:\\... so the user can paste it into Explorer."""
    s = str(p)
    if s.startswith("/mnt/") and len(s) > 6:
        drive = s[5].upper()
        return f"{drive}:" + s[6:].replace("/", "\\")
    return s


def _your_price(row):
    """The row's suggested (10%-under) price as a float, or 0.0 if none."""
    v = row.get("Your Price (AUD)", "")
    if v != "" and v is not None:
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    return 0.0


def _write_listing_txt(rows, out_path):
    lines = [
        "=" * 52,
        "  E B A B Y  —  T H E  S C O R E",
        "=" * 52,
        "  Plastic in. Paper out.",
        "  Every disc below is one step closer to guns,",
        "  drugs and hookers money.*",
        "",
        "  * or rent. Let's be honest, it's rent.",
        "=" * 52,
        "",
    ]
    for i, row in enumerate(rows, 1):
        lines.append(f"--- SCORE #{i} " + "-" * 36)
        lines.append(f"Barcode: {row.get('Barcode', '?')}")
        for k, v in row.items():
            if k in ("Barcode", "Stock", "Your Price (AUD)") or k.startswith("_"):
                continue
            if v != "" and v is not None:
                lines.append(f"  {k}: {v}")
        your_price = row.get("Your Price (AUD)", "")
        if your_price != "":
            condition = row.get("Condition", "").upper() or "IT"
            lines.append(f"  >> YOUR MOVE — list {condition} at ${your_price}  (10% under the lowest, delivered)")
        lines.append("")
    total = sum(_your_price(row) for row in rows)
    lines += [
        "=" * 52,
        f"  {len(rows)} disc(s) fenced. Now go list 'em on eBay",
        "  before the heat comes down.",
        "",
        f"  TOTAL TAKE (all your 10%-under prices): ${total:.2f}",
        "=" * 52,
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


@app.post("/api/batches/{name}/ebay/run")
def ebay_run(name: str):
    d, state, err = _stage_gate(name, "ebay")
    if err:
        return err
    barcodes = {k: v for k, v in state.get("barcodes", {}).items() if v}
    warning = None
    try:
        token = ebay.get_token()
        rows = ebay.fetch_all(list(barcodes.values()), token)
    except (RuntimeError, ebay.EbayUnavailable) as e:
        # No creds / eBay down: keep going — files fall back to barcode names.
        warning = f"eBay lookup skipped: {e}"
        rows = [{"Barcode": b} for b in barcodes.values()]

    # Collapse to the ONE condition the disc is actually sold as — the user is
    # selling this disc as new OR used, so the other copy's price is just
    # noise. rows come back in barcodes-dict order; a barcode sold in BOTH
    # zones keeps its own label.
    for (key, _bc), row in zip(barcodes.items(), rows):
        stock = "new" if key.isdigit() else "used"
        row["Stock"] = stock
        row["Condition"] = "New" if stock == "new" else "Used"
        lowest = row.pop("_lowest_new" if stock == "new" else "_lowest_used", None)
        row.pop("_lowest_new", None)
        row.pop("_lowest_used", None)
        row["Lowest Price (AUD)"] = lowest if lowest is not None else ""
        row["Your Price (AUD)"] = ebay._undercut(lowest)

    barcode_to_title = {r["Barcode"]: r.get("Title", "") for r in rows if "Barcode" in r}
    key_to_title = {k: barcode_to_title.get(v, "") for k, v in barcodes.items()}
    state["titles"] = key_to_title

    sets = _collect_sets(d / "2_color")
    slugs = rename_title.compute_slugs(sets, state.get("barcodes", {}), key_to_title)
    # Make each row's Image Set Name the ACTUAL deduped file slug, so the CSV,
    # the listing text, and the renamed photos always agree — and the
    # title-edit route can locate a disc's files from its row.
    for (key, _bc), row in zip(barcodes.items(), rows):
        if key in slugs:
            row["Image Set Name"] = slugs[key]

    try:
        ebay.write_csv(rows, d / "Ebay_Details.csv")
        _write_listing_txt(rows, d / "Ebaby Listings.txt")
    except OSError as e:
        # e.g. the CSV is open in Excel — don't lose the paid eBay fetch
        warning = (warning or "") + f" Couldn't write listing files ({e}); close them and re-run."

    plan, notes = rename_title.plan_renames(sets, state.get("barcodes", {}),
                                             key_to_title, d / "4_renamed")
    rename_title.apply_renames(plan)
    # the rename above MOVES every file out of 2_color, so it's always empty
    # afterward — an empty leftover directory with no further purpose.
    color_dir = d / "2_color"
    if color_dir.is_dir() and not any(color_dir.iterdir()):
        color_dir.rmdir()
    state["rename_notes"] = notes
    # persisted so a page refresh can rebuild the results screen (resume)
    state["ebay_rows"] = rows
    state["ebay_warning"] = warning
    state["listing_txt"] = _windows_path(d / "Ebaby Listings.txt")
    state["folder"] = _windows_path(d)
    batch.write_state(d, state)
    batch.advance_stage(d, "ebay", "crop")
    return {"rows": rows, "notes": notes, "warning": warning,
            "listing_txt": state["listing_txt"], "folder": state["folder"]}


def _collect_sets(color_dir):
    """{set_key: [(role, path), ...]} from <key>_<role>.png files."""
    sets = {}
    for p in sorted(color_dir.glob("*.png")):
        key, _, role = p.stem.partition("_")
        sets.setdefault(key, []).append((role, p))
    return sets


class TitleEditRequest(BaseModel):
    image_set_name: str          # the disc's current slug (row["Image Set Name"])
    title: str
    price: str | None = None     # optional override of "Your Price (AUD)"


# every filename suffix a disc's photos can carry (used: 3 shots; new: +_new)
_ROLE_TOKENS = ("front", "back", "inside", "front_new", "back_new")


def _parse_price(raw):
    """('' | float, error_or_None) from user price text. '' clears the price."""
    s = str(raw).strip().lstrip("$").replace(",", "")
    if s == "":
        return "", None
    try:
        return round(float(s), 2), None
    except ValueError:
        return None, f"'{raw}' isn't a price"


def _rename_disc_files(d, old_slug, new_slug):
    """Rename a disc's photos old_slug_* -> new_slug_* across 4_renamed (.png)
    and 5_cropped (.jpg). Returns {old_png_name: new_png_name} to remap
    state['quads']. Two-phase with rollback so a locked file can't leave the
    set half-renamed; raises OSError if it can't complete."""
    pairs = []
    for sub, ext in (("4_renamed", ".png"), ("5_cropped", ".jpg")):
        for role in _ROLE_TOKENS:
            src = d / sub / f"{old_slug}_{role}{ext}"
            if src.is_file():
                pairs.append((src, d / sub / f"{new_slug}_{role}{ext}"))
    for _src, dst in pairs:
        if dst.exists():                      # never clobber an unrelated file
            raise OSError(f"target already exists: {dst.name}")
    done = []
    try:
        for src, dst in pairs:
            src.rename(dst)
            done.append((src, dst))
    except OSError:
        for src, dst in reversed(done):
            dst.rename(src)                   # roll back so files stay consistent
        raise
    return {f"{old_slug}_{role}.png": f"{new_slug}_{role}.png" for role in _ROLE_TOKENS}


@app.post("/api/batches/{name}/title/edit")
def title_edit(name: str, req: TitleEditRequest):
    """Correct a disc's Title (and optionally its list price) after the fact —
    for when eBay returned the wrong match. Re-slugs the Image Set Name,
    renames the photos to match, remaps the crop quads, and rewrites the CSV +
    listing text so everything stays in step."""
    d, state, err = _stage_gate(name)        # any batch that has eBay rows
    if err:
        return err
    with _batch_lock(name):
        state = batch.read_state(d)          # re-read under the lock
        refused = _refuse_if_accepted(state)
        if refused:
            return refused
        rows = state.get("ebay_rows") or []
        row = next((r for r in rows if r.get("Image Set Name") == req.image_set_name), None)
        if row is None:
            return JSONResponse(status_code=404, content={
                "error": f"no disc named '{req.image_set_name}' in this batch"})

        new_title = (req.title or "").strip()
        if not new_title:
            return JSONResponse(status_code=422, content={"error": "title can't be empty"})

        new_price = None
        if req.price is not None:
            new_price, price_err = _parse_price(req.price)
            if price_err:
                return JSONResponse(status_code=422, content={"error": price_err})

        old_slug = req.image_set_name
        others = {r.get("Image Set Name") for r in rows if r is not row}
        new_slug = naming_utils.slugify_title(new_title, fallback=(row.get("Barcode") or old_slug))
        base, n = new_slug, 2
        while new_slug in others:             # keep every disc's slug unique
            new_slug = f"{base}_{n}"
            n += 1

        if new_slug != old_slug:
            try:
                quad_map = _rename_disc_files(d, old_slug, new_slug)
            except OSError as e:
                return JSONResponse(status_code=422, content={
                    "error": f"couldn't rename the photos ({e}); close them and retry"})
            quads = state.get("quads") or {}
            state["quads"] = {quad_map.get(k, k): v for k, v in quads.items()}

        row["Title"] = new_title
        row["Image Set Name"] = new_slug
        if new_price is not None:
            row["Your Price (AUD)"] = new_price

        # keep state["titles"] in step so a later re-run/rename stays consistent
        barcode = row.get("Barcode")
        if barcode:
            for k, v in (state.get("barcodes") or {}).items():
                if v == barcode:
                    state.setdefault("titles", {})[k] = new_title

        warning = None
        try:
            ebay.write_csv(rows, d / "Ebay_Details.csv")
            _write_listing_txt(rows, d / "Ebaby Listings.txt")
        except OSError as e:
            warning = f"Couldn't rewrite the listing files ({e}); close them and retry."
        state["ebay_rows"] = rows
        batch.write_state(d, state)
        return {"rows": rows, "quads": state.get("quads", {}), "warning": warning}


@app.post("/api/batches/{name}/crop/run")
def crop_run(name: str):
    # 'done' is allowed too: RE-CHOP on a finished batch is legitimate,
    # but a batch still at upload/color/barcode gets a clean 409, not {}.
    d, state, err = _stage_gate(name, "crop", "done")
    if err:
        return err
    with _batch_lock(name):
        state = batch.read_state(d)          # re-read under the lock
        refused = _refuse_if_accepted(state)
        if refused:
            return refused
        quads = {}
        srcs = sorted((d / "4_renamed").glob("*.png"))
        _set_progress(d, state, "crop", 0, len(srcs))
        for i, src in enumerate(srcs, 1):
            bgr = cv2.imread(str(src))
            if bgr is None:
                continue
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
            _set_progress(d, state, "crop", i, len(srcs))
        state["quads"] = quads  # persisted for refresh-resume of the chop shop
        batch.write_state(d, state)
        if state.get("stage") == "crop":
            batch.advance_stage(d, "crop", "done")
        return {"quads": quads}


# "Accept Hustle?" deletes everything except the final listing photos and
# paperwork. Safe: 1_originals is always a COPY the upload step made — the
# user's actual source files live elsewhere on their disk — so this can never
# destroy anything irreplaceable, only this batch's working copies.
_ACCEPT_REMOVE_DIRS = ("1_originals", "3_barcodes", "4_renamed", "2_color")


_ACCEPT_KEPT = ["5_cropped", "Ebay_Details.csv", "Ebaby Listings.txt", "state.json"]


@app.post("/api/batches/{name}/accept")
def accept_hustle(name: str):
    d, state, err = _stage_gate(name, "done")
    if err:
        return err
    with _batch_lock(name):
        state = batch.read_state(d)          # re-read under the lock
        # NOT gated on state["accepted"] — a call that only partially
        # succeeded (one locked file) must be re-triable, and re-running is
        # naturally idempotent: the is_dir() check below just skips whatever
        # a previous call already removed.
        removed, errors = [], []
        for sub in _ACCEPT_REMOVE_DIRS:
            p = d / sub
            if p.is_dir():
                try:
                    shutil.rmtree(p)
                    removed.append(sub)
                except OSError as e:
                    errors.append(f"{sub}: {e}")
        # Set even on partial failure: the user has declared this batch
        # finalized, so crop/title routes should refuse from now on — a retry
        # of THIS route is still how the remaining folders get cleaned up.
        state["accepted"] = True
        batch.write_state(d, state)
        result = {"removed": removed, "kept": _ACCEPT_KEPT}
        if errors:
            result["error"] = ("Couldn't fully clean up: " + "; ".join(errors) +
                                " — close any program using those files and try again.")
        return result


@app.get("/api/files/{name}/{stage}/{filename}")
def serve_file(name: str, stage: str, filename: str):
    """Serve a batch image to the browser (crop grid + editor). Path pieces
    are constrained to a known batch subfolder so this can't walk the disk."""
    d = batch.batch_dir(name)
    p = (d / stage / filename).resolve()
    if d.resolve() not in p.parents or not p.is_file():
        return JSONResponse(status_code=404, content={"error": "not found"})
    return FileResponse(p)


class CropManualRequest(BaseModel):
    filename: str
    quad: list


@app.post("/api/batches/{name}/crop/manual")
def crop_manual(name: str, req: CropManualRequest):
    d, _state, err = _stage_gate(name)
    if err:
        return err
    with _batch_lock(name):
        _state = batch.read_state(d)         # re-read under the lock
        refused = _refuse_if_accepted(_state)
        if refused:
            return refused
        src = d / "4_renamed" / Path(req.filename).name
        bgr = cv2.imread(str(src))
        if bgr is None:
            return JSONResponse(status_code=404,
                                content={"error": f"no such photo: {req.filename}"})
        try:
            out_bgr = crop.crop_and_compose(bgr, req.quad)
        except (ValueError, cv2.error) as e:
            # e.g. all four corners dragged into a heap — degenerate warp
            return JSONResponse(status_code=422, content={
                "error": f"that crop box is too small or twisted ({e})"})
        out_path = d / "5_cropped" / f"{src.stem}.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), out_bgr)
        _state.setdefault("quads", {})[src.name] = req.quad
        batch.write_state(d, _state)
        return {"recropped": req.filename}


app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
