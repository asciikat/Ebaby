"""iCollect Everything enrichment — barcode -> the site's collector
"Automatic Estimated Value", turned into a per-condition recommended price.

Lookups go through the LOCAL barcode index (data/icollect.sqlite, built once
by tools/build_icollect_index.py from the site's public database pages —
robots.txt allows it). Only a matched disc costs one page fetch, for the
fresh value. Until the index file exists every lookup returns None and the
feature simply stays dark.

The site quotes values in USD for what a typical circulating copy changes
hands at — closest to a USED disc. A factory-sealed copy trades above that,
so the NEW recommendation applies a premium.
"""
import re
import sqlite3
from pathlib import Path

INDEX_DB = Path(__file__).resolve().parent.parent.parent / "data" / "icollect.sqlite"

USED_MULT = 1.0
NEW_MULT = 1.5   # sealed premium over the circulating-copy estimate

# Values are user-locale strings — "~$6.99" (USD) but also things like
# "en_SE 24310" (Swedish entry, ambiguous units). Only a clean $ amount is
# trustworthy; anything else reads as no value. Matches both the visible
# HTML and the JSON-LD block.
_VALUE = re.compile(
    r"Automatic Estimated Value[^$\d]{0,80}?~?\$([\d,]+\.\d{2})")

_SESSION = None


def _sess():
    global _SESSION
    if _SESSION is None:
        from curl_cffi import requests as creq
        s = creq.Session(impersonate="chrome")
        # the site's bot gate: any client carrying this cookie is "human"
        s.cookies.set("ice_human", "1", domain="www.icollecteverything.com")
        _SESSION = s
    return _SESSION


def _barcode_forms(barcode):
    """The index stores barcodes AS PRINTED on each case, which varies:
    12-digit UPC-A, the same code with a leading 0 (EAN-13 form), stripped
    zeros... query every spelling of ours."""
    forms = {barcode, barcode.lstrip("0")}
    if len(barcode) == 12:
        forms.add("0" + barcode)
    if len(barcode) == 13 and barcode.startswith("0"):
        forms.add(barcode[1:])
    return [f for f in forms if f]


def find_items(barcode, db_path=None):
    """[(item_id, title), ...] from the local index (the same barcode often
    has several catalog entries — different countries/editions), or []."""
    db = Path(db_path) if db_path else INDEX_DB
    if not db.is_file():
        return []
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=3)
    except sqlite3.Error:
        return []
    try:
        forms = _barcode_forms(barcode)
        marks = ",".join("?" * len(forms))
        return con.execute(
            f"SELECT item_id, title FROM items WHERE barcode IN ({marks})",
            forms).fetchall()
    except sqlite3.OperationalError:
        return []      # index mid-build (writer holds the lock) — stay dark
    finally:
        con.close()


def fetch_value_usd(item_id):
    """The item page's 'Automatic Estimated Value' in USD, or None."""
    try:
        res = _sess().get(
            f"https://www.icollecteverything.com/db/item/movie/{item_id}/",
            timeout=30)
        if res.status_code != 200:
            return None
        m = _VALUE.search(res.text)
        return float(m.group(1).replace(",", "")) if m else None
    except Exception:   # enrichment must never sink the batch
        return None


def lookup(barcode, usd_to_aud, db_path=None):
    """{'title', 'value_aud', 'reco_new', 'reco_used', 'url'} or None.
    usd_to_aud: AUD per 1 USD (None -> no lookup; a reco in the wrong
    currency is worse than no reco)."""
    if not usd_to_aud:
        return None
    # try up to 3 catalog entries for this barcode — many carry a junk or
    # foreign-locale value; the first clean $ estimate wins
    item_id = title = value = None
    for item_id, title in find_items(barcode, db_path=db_path)[:3]:
        value = fetch_value_usd(item_id)
        if value is not None:
            break
    if value is None:
        return None
    aud = value * usd_to_aud
    return {
        "title": title,
        "value_aud": round(aud, 2),
        "reco_used": round(aud * USED_MULT, 2),
        "reco_new": round(aud * NEW_MULT, 2),
        "url": f"https://www.icollecteverything.com/db/item/movie/{item_id}/",
    }
