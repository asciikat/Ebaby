"""Sequential rename for DVD photo sets — ported from bn_rename.py (brand-new,
2-shot, number-keyed) and sh_rename.py (used, 3-shot, letter-keyed).

Groups files by filename stem so a RAW and its JPEG move together, orders
shots by filename (matches the scripts' proven default — camera filenames
like IMG_YYYYMMDD_HHMMSS already sort correctly), and renames with a
two-phase temp-name swap so no file is ever overwritten by another, even when
a rename plan swaps two existing names.
"""
import os
import uuid
from pathlib import Path

RAW_EXTS = {".nef", ".dng", ".cr2", ".cr3", ".arw", ".raf", ".orf", ".rw2", ".pef", ".srw"}
JPEG_EXTS = {".jpg", ".jpeg"}
FLAT_EXTS = {".png", ".tif", ".tiff"}
ALL_EXTS = RAW_EXTS | JPEG_EXTS | FLAT_EXTS

USED_LABELS = ["front", "back", "inside"]
NEW_LABELS = ["front", "back"]


def find_images(input_dir: Path) -> list:
    out = []
    for name in sorted(os.listdir(input_dir)):
        if name.startswith("."):
            continue
        p = input_dir / name
        if p.is_file() and p.suffix.lower() in ALL_EXTS:
            out.append(p)
    return out


def group_shots(files: list) -> list:
    """Group RAW+JPEG of the same shot (same filename stem), filename order."""
    groups = {}
    for f in files:
        groups.setdefault(f.stem, []).append(f)
    ordered_stems = list(dict.fromkeys(f.stem for f in files))
    return [groups[stem] for stem in ordered_stems]


def make_prefix(set_index: int, scheme: str) -> str:
    if scheme == "number":
        return f"{set_index + 1:02d}"
    s, n0 = "", set_index
    while True:
        s = chr(97 + (n0 % 26)) + s
        n0 = n0 // 26 - 1
        if n0 < 0:
            break
    return s


def build_plan(input_dir: Path, set_size: int, scheme: str, labels: list) -> list:
    """[(old_path, new_path), ...]. Raises ValueError on uneven counts or
    duplicate targets — never silently mis-groups a partial set."""
    files = find_images(input_dir)
    shots = group_shots(files)
    if len(shots) % set_size != 0:
        raise ValueError(
            f"{len(shots)} shots is not a whole number of sets of {set_size} "
            f"in {input_dir}"
        )
    plan = []
    for shot_index, shot_files in enumerate(shots):
        set_index = shot_index // set_size
        label_index = shot_index % set_size
        label = labels[label_index]
        prefix = make_prefix(set_index, scheme)
        for f in shot_files:
            plan.append((f, f.parent / f"{prefix}_{label}{f.suffix}"))

    finals = [new for _, new in plan]
    dupes = sorted({p for p in finals if finals.count(p) > 1})
    if dupes:
        raise ValueError(f"rename plan has duplicate targets: {[d.name for d in dupes]}")
    return plan


def apply_plan(plan: list) -> int:
    """Two-phase rename (temp names first) so no file is ever overwritten."""
    temps = []
    for old, new in plan:
        if old == new:
            continue
        tmp = old.parent / f".renametmp_{uuid.uuid4().hex}{old.suffix}"
        old.rename(tmp)
        temps.append((tmp, new))
    for tmp, new in temps:
        tmp.rename(new)
    return len(temps)
