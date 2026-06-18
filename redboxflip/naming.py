"""Filename building, per-DVD listing files, and per-run summary files."""
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


# --------------------------------------------------------------------------- #
# Per-DVD eBay listing (one self-contained set of files inside each DVD folder)
# --------------------------------------------------------------------------- #

def _fmt_field(f) -> str:
    """Render one extracted field as 'value (Confidence)' or a check warning."""
    if f is None or not f.known:
        return "Unknown  [needs manual check]"
    tail = f" ({f.confidence})" if f.confidence else ""
    if f.needs_check:
        tail += "  [needs manual check]"
    return f"{f.value}{tail}"


def _description(group) -> str:
    s = group.scan
    if s is None:
        return f"{group.title}. DVD. {group.region}."
    parts = []
    lead = s.title.value if s.title.known else group.title
    edition = f" ({s.edition.value})" if s.edition.known else ""
    parts.append(f"{lead}{edition} on DVD.")
    if s.studio.known:
        parts.append(f"Released by {s.studio.value}" +
                     (f" in {s.release_year.value}." if s.release_year.known else "."))
    elif s.release_year.known:
        parts.append(f"Released {s.release_year.value}.")
    if s.region.known or s.pal_ntsc.known:
        rp = " ".join(x for x in (s.region.value if s.region.known else "",
                                  s.pal_ntsc.value if s.pal_ntsc.known else "") if x)
        parts.append(f"Region/standard: {rp}.")
    if s.rating.known:
        parts.append(f"Rated {s.rating.value}.")
    if s.genre.known:
        parts.append(f"Genre: {s.genre.value}.")
    if s.special_features.known:
        parts.append(f"Special features: {s.special_features.value}.")
    if s.condition.known:
        parts.append(f"Condition notes: {s.condition.value}.")
    return " ".join(parts)


_LISTING_ROWS = [
    ("DVD Title", "title"), ("Format", "format"), ("Region", "region"),
    ("PAL/NTSC", "pal_ntsc"), ("Barcode", None), ("Rating", "rating"),
    ("Release Year", "release_year"), ("Studio/Distributor", "studio"),
    ("Edition", "edition"), ("Number of Discs", "num_discs"),
    ("Languages", "languages"), ("Subtitles", "subtitles"),
    ("Genre", "genre"), ("Special Features", "special_features"),
]


def render_ebay_text(group) -> str:
    s = group.scan
    L = []
    suggested = (s.suggested_title if s and s.suggested_title
                 else f"{group.title} DVD {group.region}")
    L += ["=" * 52, "eBay Listing", "=" * 52, "",
          f"Suggested eBay Title: {suggested}", ""]

    title_uncertain = group.title.startswith("Untitled DVD") or (
        s is not None and s.title.needs_check)
    for label, key in _LISTING_ROWS:
        if key == "title":
            # Carry the title's real confidence like every other field.
            if s is not None:
                L.append(f"{label}: {_fmt_field(s.title)}")
            elif title_uncertain:
                L.append(f"{label}: {group.title}  [needs manual check]")
            else:
                L.append(f"{label}: {group.title}")
        elif key is None:                       # barcode comes off the group
            bc = group.barcode or (s.barcode.value if s and s.barcode.known else "")
            L.append(f"Barcode: {bc if bc else 'Unknown  [needs manual check]'}")
        else:
            L.append(f"{label}: {_fmt_field(getattr(s, key) if s else None)}")

    cond = _fmt_field(s.condition) if s else "Unknown  [needs manual check]"
    included = s.included_items if s and s.included_items else "DVD case, disc(s)"
    tested = s.tested if s else "Not tested - sold as is"
    L += ["",
          f"Condition: {cond}",
          f"Included Items: {included}",
          f"Tested: {tested}",
          "",
          "Description:",
          _description(group),
          ""]
    if s and s.notes.known:
        L += [f"Notes: {s.notes.value}", ""]

    # When no scan ran, every field is unknown — say so loudly rather than
    # silently omitting the warning.
    checks = s.manual_checks() if s else ["all fields (no automatic scan was run)"]
    if not (group.barcode or (s and s.barcode.known)):
        checks = ["barcode"] + [c for c in checks if c != "barcode"]
    if checks:
        L += ["⚠ MANUAL CHECK NEEDED: " + ", ".join(checks), ""]

    L += ["Photos:"]
    for sh in group.shots:
        fname = Path(sh.output_path).name if sh.output_path else "(unsaved)"
        L.append(f"  {fname}  [{sh.face.value}]")
    L += ["", f"Region (default): {group.region}", ""]
    return "\n".join(L)


def write_dvd_files(dvd_dir, group):
    """Write ebay_listing.txt, ebay_listing.csv, qwen_scan.json into a DVD folder."""
    dvd_dir = Path(dvd_dir)
    dvd_dir.mkdir(parents=True, exist_ok=True)
    s = group.scan

    txt = dvd_dir / "ebay_listing.txt"
    txt.write_text(render_ebay_text(group), encoding="utf-8")

    csv_path = dvd_dir / "ebay_listing.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["field", "value", "confidence", "needs_manual_check"])
        title_uncertain = group.title.startswith("Untitled DVD") or (
            s is not None and s.title.needs_check)
        w.writerow(["Suggested eBay Title",
                    (s.suggested_title if s and s.suggested_title else group.title),
                    "", "yes" if title_uncertain else "no"])
        for label, key in _LISTING_ROWS:
            if key == "title":
                if s is not None:
                    w.writerow([label, s.title.value, s.title.confidence,
                                "yes" if s.title.needs_check else "no"])
                else:
                    w.writerow([label, group.title, "Unknown" if title_uncertain else "",
                                "yes" if title_uncertain else "no"])
            elif key is None:
                bc = group.barcode or (s.barcode.value if s and s.barcode.known else "")
                w.writerow([label, bc, "High" if bc else "Unknown",
                            "no" if bc else "yes"])
            else:
                fld = getattr(s, key) if s else None
                if fld and fld.known:
                    w.writerow([label, fld.value, fld.confidence,
                                "yes" if fld.needs_check else "no"])
                else:
                    w.writerow([label, "Unknown", "Unknown", "yes"])
        if s:
            w.writerow(["Condition", s.condition.value, s.condition.confidence,
                        "yes" if s.condition.needs_check else "no"])
            w.writerow(["Included Items", s.included_items, "", ""])
            w.writerow(["Tested", s.tested, "", ""])

    scan_json = dvd_dir / "qwen_scan.json"
    payload = {
        "title": group.title,
        "barcode": group.barcode,
        "region_default": group.region,
        "scan": s.to_dict() if s else None,
        "photos": [{"face": sh.face.value,
                    "file": Path(sh.output_path).name if sh.output_path else None}
                   for sh in group.shots],
    }
    scan_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return txt, csv_path, scan_json


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
