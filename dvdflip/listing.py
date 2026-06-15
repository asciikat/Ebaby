import csv
import json
import time
from pathlib import Path

from .config import GROUP_TIME_GAP_S
from .models import DvdGroup

CSV_COLUMNS = ["dvd", "title", "year", "genre", "region", "runtime", "studio",
               "barcode", "description", "condition", "price", "photos"]


def _norm(title):
    return " ".join((title or "").lower().split())


def _mtime(photo):
    try:
        return photo.source_path.stat().st_mtime
    except OSError:
        return 0.0


def group_photos(photos):
    """Group photos into DvdGroups by normalised title; time-cluster the untitled."""
    groups = []
    titled = {}
    untitled = []
    for p in photos:
        if p.deleted:
            continue
        key = _norm(p.title)
        if key:
            g = titled.get(key)
            if g is None:
                g = DvdGroup(title=p.title.strip(), year=p.year)
                titled[key] = g
                groups.append(g)
            g.photos.append(p)
        else:
            untitled.append(p)

    untitled.sort(key=_mtime)
    cur = None
    last_t = None
    n = 0
    for p in untitled:
        t = _mtime(p)
        if cur is None or (last_t is not None and t - last_t > GROUP_TIME_GAP_S):
            n += 1
            cur = DvdGroup(title=f"Untitled DVD {n}")
            groups.append(cur)
        cur.photos.append(p)
        last_t = t

    for g in groups:
        for p in g.photos:
            if not g.barcode and p.barcode:
                g.barcode = p.barcode
            if not g.year and p.year:
                g.year = p.year
    return groups


def _slug(text):
    keep = [c.lower() if c.isalnum() else "-" for c in (text or "dvd")]
    s = "".join(keep)
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-") or "dvd"


def listing_text(group):
    lines = [group.listing_title or group.title, ""]
    if group.description:
        lines += [group.description, ""]
    for label, val in (("Year", group.year), ("Genre", group.genre),
                       ("Region", group.region), ("Runtime", group.runtime),
                       ("Studio", group.studio), ("Barcode", group.barcode)):
        if val:
            lines.append(f"{label}: {val}")
    lines += ["", f"Condition: {group.condition}", f"Price: {group.price}"]
    return "\n".join(lines)


def write_listing_txt(group, dvd_dir):
    dvd_dir = Path(dvd_dir)
    dvd_dir.mkdir(parents=True, exist_ok=True)
    path = dvd_dir / "listing.txt"
    path.write_text(listing_text(group), encoding="utf-8")
    return path


def write_batch_csv(groups, run_dir):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "batch_listings.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for i, g in enumerate(groups, 1):
            photos = "; ".join(
                f"{(p.work_image.name if p.work_image else p.source_path.name)} [{p.side}]"
                for p in g.photos)
            w.writerow({"dvd": i, "title": g.title, "year": g.year or "",
                        "genre": g.genre, "region": g.region, "runtime": g.runtime,
                        "studio": g.studio, "barcode": g.barcode,
                        "description": g.description, "condition": g.condition,
                        "price": g.price, "photos": photos})
    return path


def write_run_log(groups, run_dir, extra=None):
    run_dir = Path(run_dir)
    data = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dvds": len(groups),
            "groups": [{"title": g.title, "barcode": g.barcode,
                        "photos": [p.source_path.name for p in g.photos]}
                       for g in groups]}
    if extra:
        data.update(extra)
    path = run_dir / "run_log.json"
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path
