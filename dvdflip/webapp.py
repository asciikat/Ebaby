import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates

from .config import DEFAULT_INPUT, DEFAULT_OUTPUT, JPEG_QUALITY
from .loader import load_bgr, gather_inputs
from .listing import group_photos, write_listing_txt, write_batch_csv, write_run_log, _slug
from .pipeline import process_image, reconcile_orientation

TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

_ROT_SAVE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
             270: cv2.ROTATE_90_COUNTERCLOCKWISE}


@dataclass
class Session:
    run_dir: Path
    work_dir: Path
    groups: list = field(default_factory=list)


def _slug_dir(group):
    name = (group.title or group.listing_title or "DVD").strip()
    parts = [w.capitalize() for w in name.split()]
    return "-".join(parts) or "DVD"


def build_session(input_dir=None, output_base=None, progress=print):
    import time
    input_dir = Path(input_dir or DEFAULT_INPUT)
    output_base = Path(output_base or DEFAULT_OUTPUT)
    run_dir = output_base / f"run_{time.strftime('%Y-%m-%d_%H%M')}"
    work_dir = run_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    files = gather_inputs(input_dir, exclude_dir=output_base)
    photos = []
    for i, f in enumerate(files, 1):
        progress(f"[{i}/{len(files)}] {f.name}")
        try:
            photos.append(process_image(f, work_dir))
        except Exception as e:
            progress(f"  FAILED: {e}")

    groups = group_photos(photos)
    for g in groups:
        reconcile_orientation(g.photos)
        # Listing text is filled in by hand in the review screen (model dropped).
        g.listing_title = g.title
    return Session(run_dir=run_dir, work_dir=work_dir, groups=groups)


def create_app(session):
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def review(request: Request):
        cards = []
        for gi, g in enumerate(session.groups):
            cards.append({
                "index": gi, "group": g,
                "photos": [{"side": p.side, "confidence": p.confidence,
                            "a4_found": p.a4_found, "barcode": p.barcode,
                            "img": p.work_image.name if p.work_image else ""}
                           for p in g.photos],
            })
        return TEMPLATES.TemplateResponse(request, "review.html",
                                          {"cards": cards})

    @app.get("/work/{name}")
    def work_image(name: str):
        path = session.work_dir / name
        if not path.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(str(path))

    @app.post("/api/dvd/{index}/save")
    async def save_dvd(index: int, request: Request):
        body = await request.json()
        g = session.groups[index]
        for fld in ("listing_title", "description", "genre", "region",
                    "runtime", "studio", "condition", "price"):
            if fld in body:
                setattr(g, fld, body[fld] or "")
        kept = []
        edits = body.get("photos", [])
        for p, e in zip(g.photos, edits):
            if e.get("deleted"):
                continue
            p.side = e.get("side", p.side)
            p.extra_rotation_cw = int(e.get("rotate", 0)) % 360
            kept.append(p)
        g.photos = kept

        dvd_dir = session.run_dir / _slug_dir(g)
        dvd_dir.mkdir(parents=True, exist_ok=True)
        base = _slug(g.title or g.listing_title)
        for p in g.photos:
            if not p.work_image or not Path(p.work_image).exists():
                continue
            out = dvd_dir / f"{base}_{p.side}.jpg"
            n = 2
            while out.exists():
                out = dvd_dir / f"{base}_{p.side}_{n}.jpg"; n += 1
            if p.extra_rotation_cw in _ROT_SAVE:
                img = cv2.imread(str(p.work_image))
                cv2.imwrite(str(out), cv2.rotate(img, _ROT_SAVE[p.extra_rotation_cw]),
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            else:
                shutil.copyfile(p.work_image, out)
        write_listing_txt(g, dvd_dir)
        g.approved = True
        write_batch_csv([x for x in session.groups if x.approved], session.run_dir)
        write_run_log(session.groups, session.run_dir)
        return {"ok": True, "saved_to": str(dvd_dir)}

    return app
