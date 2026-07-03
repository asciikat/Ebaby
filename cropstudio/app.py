"""FastAPI server: serves the cropper page and saves posted crops to processed/."""
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response

from redboxflip import vlm
from redboxflip.config import load_settings

from . import batch, colorcorrect, manifest, paths, service

_STATIC = Path(__file__).resolve().parent / "static"


def create_app(run_dir=None, settings=None):
    app = FastAPI(title="Crop Studio")
    state = batch.BatchState()
    settings = settings if settings is not None else load_settings()
    run_dir = run_dir or paths.make_run_dir()
    counter = {"n": 0}

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

    @app.post("/rawdecode")
    async def rawdecode(image: UploadFile = File(...)):
        data = await image.read()
        try:
            jpeg = colorcorrect.raw_to_jpeg(data)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return Response(content=jpeg, media_type="image/jpeg")

    @app.post("/finish")
    def finish():
        req = state.finish()
        if req is None:
            return {"status": "empty"}
        return {"status": "saved", **_flush(req)}

    return app
