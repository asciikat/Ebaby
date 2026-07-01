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
