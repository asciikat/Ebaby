"""Merge per-face Qwen reads into a validated DvdScan for the eBay listing.

The vision model reads printed words well but will confidently invent facts it
cannot see (in testing it produced a wrong region, studio and barcode). So every
sensitive field is accepted ONLY when the value also appears in the text the
model transcribed for that same face — a value pulled from the model's prior
knowledge won't be in the transcription and is dropped to "Unknown / needs
manual check". Barcodes never come from the model; the pipeline decodes those.
"""
import re

from .models import (Face, Field, DvdScan, HIGH, MEDIUM, LOW, UNKNOWN)

# Fields the user said must never be guessed — only kept when verified in-text.
SENSITIVE = {"region", "pal_ntsc", "rating", "release_year"}

# Which faces to consult for each field, in order of preference.
FACE_PRIORITY = {
    "title": [Face.FRONT, Face.INSIDE, Face.BACK],
    "format": [Face.BACK, Face.FRONT, Face.INSIDE],
    "region": [Face.BACK, Face.INSIDE, Face.FRONT],
    "pal_ntsc": [Face.BACK, Face.INSIDE, Face.FRONT],
    "rating": [Face.BACK, Face.FRONT],
    "release_year": [Face.BACK, Face.FRONT, Face.INSIDE],
    "studio": [Face.BACK, Face.FRONT, Face.INSIDE],
    "edition": [Face.FRONT, Face.BACK],
    "num_discs": [Face.BACK, Face.FRONT, Face.INSIDE],
    "languages": [Face.BACK, Face.INSIDE],
    "subtitles": [Face.BACK, Face.INSIDE],
    "genre": [Face.BACK, Face.FRONT],
    "special_features": [Face.BACK, Face.FRONT],
    "notes": [Face.FRONT, Face.BACK],
}

_RATING_RE = re.compile(
    r"\b(G|PG|M|MA15\+?|MA|R18\+?|R|RC|NR|PG-?13|NC-?17|E|AO|CTC|TV-?[A-Z0-9]+)\b",
    re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19[0-9]{2}|20[0-9]{2})\b")
_FORMAT_RE = re.compile(r"\b(4K UHD|UHD|4K|Blu-?ray|DVD)\b", re.IGNORECASE)
_REGION_RE = re.compile(r"(region\s*(?:free|all|[0-9](?:\s*[,&/]\s*[0-9])*)|all\s*regions)",
                        re.IGNORECASE)
_WORD_NUM = {"single": 1, "one": 1, "two": 2, "three": 3, "four": 4,
             "double": 2, "triple": 3}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _in_text(value: str, all_text: str) -> bool:
    v = _norm(value)
    return bool(v) and v in _norm(all_text)


def _known(v) -> bool:
    return bool(v) and str(v).strip().lower() not in ("", "unknown", "n/a", "none", "-")


# --- per-field validators: return a cleaned value or None ------------------- #

def _v_text(v):
    v = (v or "").strip()
    return v if _known(v) and len(v) >= 2 else None


def _v_title(v):
    v = (v or "").strip().strip('"“”')
    if not _known(v) or not (2 <= len(v) <= 80) or "\n" in v:
        return None
    return v


def _v_rating(v):
    m = _RATING_RE.search(v or "")
    return m.group(0).upper().replace(" ", "") if m else None


def _v_year(v):
    m = _YEAR_RE.search(v or "")
    return m.group(0) if m else None


def _v_format(v):
    m = _FORMAT_RE.search(v or "")
    return m.group(0).upper().replace("BLURAY", "Blu-ray").replace("BLU-RAY", "Blu-ray") if m else None


def _v_region(v):
    m = _REGION_RE.search(v or "")
    if not m:
        return None
    return re.sub(r"\s+", " ", m.group(0).strip()).title()


def _v_palntsc(v):
    m = re.search(r"\b(PAL|NTSC)\b", v or "", re.IGNORECASE)
    return m.group(1).upper() if m else None


def _v_discs(v):
    v = (v or "").lower()
    m = re.search(r"\b([1-9])\b", v)
    if m:
        return m.group(1)
    for word, n in _WORD_NUM.items():
        if word in v:
            return str(n)
    return None


_VALIDATORS = {
    "title": _v_title, "format": _v_format, "region": _v_region,
    "pal_ntsc": _v_palntsc, "rating": _v_rating, "release_year": _v_year,
    "num_discs": _v_discs,
}
# every other field falls back to free text
_KEY_ALIAS = {"condition": "condition_notes"}


def _pick(per_face, key) -> Field:
    """Choose the best validated value for one field across faces."""
    validator = _VALIDATORS.get(key, _v_text)
    sensitive = key in SENSITIVE
    src_key = _KEY_ALIAS.get(key, key)
    for face in FACE_PRIORITY[key]:
        data = per_face.get(face)
        if not data:
            continue
        clean = validator(data.get(src_key))
        if not clean:
            continue
        all_text = data.get("all_text", "")
        in_txt = _in_text(clean, all_text) or _in_text(data.get(src_key, ""), all_text)
        is_low = key in {_norm_lc(x) for x in data.get("low_confidence", [])}

        if sensitive and not in_txt:
            continue            # unverifiable sensitive claim -> drop it
        if key == "title":
            conf = MEDIUM if not in_txt else HIGH
        elif sensitive:
            conf = LOW if is_low else HIGH
        else:
            conf = LOW if is_low else (HIGH if in_txt else MEDIUM)
        needs = conf in (LOW, UNKNOWN) or (sensitive and conf != HIGH)
        return Field(clean, conf, needs, face.value)
    return Field()


def _norm_lc(x) -> str:
    return re.sub(r"[^a-z_]", "", str(x).lower())


# --- mine fields straight from the transcription -------------------------- #
# The model transcribes text well but often leaves the structured fields
# "Unknown". Parsing them back out of all_text is deterministic and grounded in
# what was actually read, so it adds detail without risking a hallucination.

def _mine_text(all_text: str, key: str):
    t = all_text or ""
    if key == "num_discs":
        m = re.search(r"\b([1-9])\s*[-\s]?\s*discs?\b", t, re.IGNORECASE)
        if m:
            return m.group(1)
        for word, n in _WORD_NUM.items():
            if re.search(rf"\b{word}[-\s]?discs?\b", t, re.IGNORECASE):
                return str(n)
        return None
    if key == "pal_ntsc":
        m = re.search(r"\b(PAL|NTSC)\b", t, re.IGNORECASE)
        return m.group(1).upper() if m else None
    if key == "region":
        m = re.search(r"region[s]?\s*(free|all|[0-9][0-9\s,&/]*)", t, re.IGNORECASE)
        if not m:
            return None
        return ("Region " + re.sub(r"\s+", " ", m.group(1).strip())).title()
    if key == "rating":
        # Only real multi-char badges — never a stray single letter in prose.
        m = re.search(r"\b(MA\s?15\+?|R\s?18\+?|PG-?13|NC-?17|M\s?15\+?|PG|MA)\b",
                      t, re.IGNORECASE)
        return m.group(1).upper().replace(" ", "") if m else None
    if key == "subtitles":
        m = re.search(r"subtitles?\s*[:\-]\s*([^\n.;]{3,80})", t, re.IGNORECASE)
        return m.group(1).strip(" .;,") if m else None
    if key == "special_features":
        m = re.search(r"special\s+features?\s*[:\-]?\s*(.+)", t,
                      re.IGNORECASE | re.DOTALL)
        if not m:
            return None
        seg = re.split(r"subtitles?\s*[:\-]", m.group(1), flags=re.IGNORECASE)[0]
        seg = re.sub(r"\s+", " ", seg).strip(" .;,")
        if len(seg) > 140:
            seg = seg[:140].rsplit(" ", 1)[0] + "…"
        return seg or None
    return None


def _mine_field(per_face, key) -> Field:
    sensitive = key in SENSITIVE
    for face in FACE_PRIORITY[key]:
        data = per_face.get(face) or {}
        mined = _mine_text(data.get("all_text", ""), key)
        if mined:
            # Parsed from real transcription: trustworthy but worth a glance for
            # the sensitive fields the user never wants guessed.
            return Field(mined, MEDIUM, sensitive, face.value)
    return Field()


def _condition(per_face) -> Field:
    """Combine any visible condition notes across all faces."""
    notes = []
    for face in (Face.BACK, Face.FRONT, Face.INSIDE):
        data = per_face.get(face) or {}
        c = (data.get("condition_notes") or "").strip()
        if _known(c) and c.lower() not in (n.lower() for n in notes):
            notes.append(c)
    if not notes:
        return Field("Unknown", UNKNOWN, True, "")
    return Field("; ".join(notes), MEDIUM, True, "photos")


def suggested_ebay_title(scan: DvdScan) -> str:
    """Build an eBay-style title (<=80 chars) from the confident fields."""
    bits = []
    if scan.title.known:
        bits.append(scan.title.value)
    if scan.edition.known and scan.edition.value.lower() not in scan.title.value.lower():
        bits.append(scan.edition.value)
    bits.append("DVD")
    if scan.region.known:
        bits.append(scan.region.value)
    if scan.pal_ntsc.known:
        bits.append(scan.pal_ntsc.value)
    if scan.release_year.known:
        bits.append(scan.release_year.value)
    title = " ".join(b for b in bits if b).strip()
    title = re.sub(r"\s+", " ", title)
    return title[:80].rstrip()


def _sanitize(d: dict) -> dict:
    """Coerce a raw model dict to safe types (small models return odd shapes).

    String fields -> str (model may emit a number/null); low_confidence -> list.
    Keeps validators and the text-miner from crashing the whole batch on garbage.
    """
    out = {}
    for k, v in (d or {}).items():
        if k == "low_confidence":
            out[k] = v if isinstance(v, list) else []
        elif k == "upright":
            out[k] = v
        else:
            out[k] = "" if v is None else (v if isinstance(v, str) else str(v))
    return out


def merge_faces(per_face: dict, default_region: str = "") -> DvdScan:
    """Combine raw per-face Qwen dicts (keyed by Face) into a validated DvdScan."""
    per_face = {f: _sanitize(d) for f, d in (per_face or {}).items() if d}
    scan = DvdScan()
    minable = {"num_discs", "pal_ntsc", "region", "rating", "subtitles",
               "special_features"}
    for name in ("title", "format", "region", "pal_ntsc", "rating",
                 "release_year", "studio", "edition", "num_discs", "languages",
                 "subtitles", "genre", "special_features", "notes"):
        fld = _pick(per_face, name)
        if not fld.known and name in minable:    # structured field empty -> mine the text
            fld = _mine_field(per_face, name)
        setattr(scan, name, fld)
    scan.condition = _condition(per_face)

    # A DVD is a DVD: if nothing read, default the format but flag for a check.
    if not scan.format.known:
        scan.format = Field("DVD", LOW, True, "default")

    scan.raw_text = {f.value: (d.get("all_text") or "")
                     for f, d in per_face.items() if d}
    booklet = any("booklet" in t.lower() for t in scan.raw_text.values())
    discs = scan.num_discs.value if scan.num_discs.known and scan.num_discs.value.isdigit() else None
    items = ["DVD case", (f"{discs} disc{'s' if discs != '1' else ''}" if discs else "disc(s)")]
    if booklet:
        items.append("booklet")
    scan.included_items = ", ".join(items)

    scan.suggested_title = suggested_ebay_title(scan)
    return scan
