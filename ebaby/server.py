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
