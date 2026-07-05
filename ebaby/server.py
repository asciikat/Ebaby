"""FastAPI orchestration layer. Each route calls a stage module against the
current batch's folder and advances state.json. No stage logic lives here —
this file only sequences calls into ebaby.stages.* and ebaby.batch.
"""
import re
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import cv2

from ebaby import batch
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


@app.post("/api/batches/{name}/color/run")
def color_run(name: str):
    d, _state, err = _stage_gate(name, "color")
    if err:
        return err
    out_dir = d / "2_color"
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for zone_dir in (d / "1_originals/used", d / "1_originals/new"):
        for src in sorted(zone_dir.glob("*")):
            ext = src.suffix.lower()
            out_path = out_dir / f"{src.stem}.png"
            if ext in RAW_EXTS:
                color.color_correct_file(src, out_path)
                count += 1
            elif ext in PLAIN_EXTS:
                if color.color_correct_plain_file(src, out_path):
                    count += 1
    batch.advance_stage(d, "color", "barcode")
    return {"colored": count}


@app.post("/api/batches/{name}/barcode/run")
def barcode_run(name: str):
    d, state, err = _stage_gate(name, "barcode")
    if err:
        return err
    results = {}
    for back_photo in sorted((d / "2_color").glob("*_back*.png")):
        key = back_photo.stem.split("_")[0]
        crop_path = d / "3_barcodes" / f"{back_photo.stem}_barcode.png"
        located = barcode_locate.locate_and_crop(back_photo, crop_path)
        # prefer the (downscaled) crop even when locate fell back — decoding
        # the full-resolution original hangs for minutes per miss
        img = cv2.imread(str(crop_path))
        if img is None:
            img = cv2.imread(str(back_photo))
        codes = barcode_decode.decode_barcodes(img) if (located or img is not None) else []
        results[key] = codes[0] if codes else None

    state["barcodes"] = results
    batch.write_state(d, state)
    batch.advance_stage(d, "barcode", "ebay")
    return {"results": results}


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
            if k != "Barcode" and v:
                lines.append(f"  {k}: {v}")
        lines.append("")
    lines += [
        "=" * 52,
        f"  {len(rows)} disc(s) fenced. Now go list 'em on eBay",
        "  before the heat comes down.",
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
    try:
        ebay.write_csv(rows, d / "Ebay_Details.csv")
        _write_listing_txt(rows, d / "Ebaby Listings.txt")
    except OSError as e:
        # e.g. the CSV is open in Excel — don't lose the paid eBay fetch
        warning = (warning or "") + f" Couldn't write listing files ({e}); close them and re-run."

    # tag each row new/used AFTER the CSV is written (extra keys would break
    # DictWriter). rows come back in barcodes-dict order, so zip keys to rows
    # — a barcode sold in BOTH zones must not collapse to one stock label.
    for (key, _bc), row in zip(barcodes.items(), rows):
        row["Stock"] = "new" if key.isdigit() else "used"

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
    return {"rows": rows, "notes": notes, "warning": warning,
            "listing_txt": _windows_path(d / "Ebaby Listings.txt"),
            "folder": _windows_path(d)}


def _collect_sets(color_dir):
    """{set_key: [(role, path), ...]} from <key>_<role>.png files."""
    sets = {}
    for p in sorted(color_dir.glob("*.png")):
        key, _, role = p.stem.partition("_")
        sets.setdefault(key, []).append((role, p))
    return sets


@app.post("/api/batches/{name}/crop/run")
def crop_run(name: str):
    # 'done' is allowed too: RE-CHOP on a finished batch is legitimate,
    # but a batch still at upload/color/barcode gets a clean 409, not {}.
    d, state, err = _stage_gate(name, "crop", "done")
    if err:
        return err
    quads = {}
    for src in sorted((d / "4_renamed").glob("*.png")):
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
    if state.get("stage") == "crop":
        batch.advance_stage(d, "crop", "done")
    return {"quads": quads}


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
    return {"recropped": req.filename}


app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
