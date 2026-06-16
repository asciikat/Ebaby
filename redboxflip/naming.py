"""Filename building and per-run output files (listing txt/csv, run log)."""
import csv
import json
import re
import time
from pathlib import Path

from .models import Face, FACE_FILE_LABEL

_BAD = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_stem(text: str, max_len: int = 90) -> str:
    text = _BAD.sub(" ", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len] or "dvd"


def output_filename(title: str, face: Face) -> str:
    return f"{safe_stem(title)} - {FACE_FILE_LABEL[face]}.jpg"


def write_listing(run_dir, groups):
    run_dir = Path(run_dir)
    txt = run_dir / "dvd_listing.txt"
    lines = [
        "DVD listing",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M')}",
        f"DVDs: {len(groups)}",
        "",
    ]
    for g in groups:
        lines += [
            "=" * 50,
            f"DVD {g.index}",
            f"Name:    {g.title}",
            f"Region:  {g.region}",
            f"Barcode: {g.barcode or ''}",
            "Photos:",
        ]
        for s in g.shots:
            fname = Path(s.output_path).name if s.output_path else "(unsaved)"
            lines.append(f"  {fname}  [{s.face.value}]")
        lines += ["", "Condition: ", "Notes: ", "Price AUD: ", ""]
    txt.write_text("\n".join(lines), encoding="utf-8")

    csv_path = run_dir / "dvd_listing.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dvd", "name", "region", "barcode", "photos"])
        for g in groups:
            photos = "; ".join(
                f"{Path(s.output_path).name if s.output_path else '(unsaved)'} [{s.face.value}]"
                for s in g.shots)
            w.writerow([g.index, g.title, g.region, g.barcode or "", photos])
    return txt, csv_path


def write_run_log(run_dir, groups, settings_dict):
    run_dir = Path(run_dir)
    payload = {
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "settings": settings_dict,
        "dvds": [g.to_dict() for g in groups],
    }
    p = run_dir / "run_log.json"
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return p
