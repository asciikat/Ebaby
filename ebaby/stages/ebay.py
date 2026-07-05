"""eBay Browse (buy) API — app-only token, New/Used dual search, item
specifics, and CSV writing. Ported from ebay_api.py + ebay_csv_extractor_new.py.

Network access is best-effort: transient failures are retried with backoff;
a hard outage raises EbayUnavailable so callers can show a clean message
instead of a stack trace and let the user choose retry-or-continue.
"""
import base64
import csv
import time
import urllib.parse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ebaby import pipeline_config as cfg
from ebaby.naming_utils import slugify_title

_TOKEN_URL = "https://api.ebay.com/identity/v1/oauth2/token"
_SEARCH_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"
_ITEM_URL = "https://api.ebay.com/buy/browse/v1/item/"

NEW_CONDITIONS = "1000|1500|1750"
USED_CONDITIONS = "3000|4000|5000|6000"

CSV_HEADERS = [
    "Barcode", "Image Set Name", "Title", "Region Code", "Genre", "Type",
    "Season", "Actor", "Studio", "Language", "Rating",
    "Lowest Price New (AUD)", "Lowest Price Used (AUD)",
]


class EbayUnavailable(RuntimeError):
    """Raised when eBay can't be reached (network/DNS/timeout/5xx after retries)."""


def _make_session():
    retry = Retry(
        total=3, connect=3, read=3, backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"), raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess = requests.Session()
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess


_SESSION = _make_session()


def _root_cause(exc):
    if isinstance(exc, requests.exceptions.ConnectionError):
        return "no network / DNS"
    if isinstance(exc, requests.exceptions.Timeout):
        return "timed out"
    return type(exc).__name__


def get_token() -> str:
    if not cfg.ebay_configured():
        raise RuntimeError(
            "eBay credentials missing — set EBAY_CLIENT_ID / EBAY_CLIENT_SECRET in the project .env")
    creds = base64.b64encode(f"{cfg.EBAY_CLIENT_ID}:{cfg.EBAY_CLIENT_SECRET}".encode()).decode()
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Authorization": f"Basic {creds}"}
    payload = "grant_type=client_credentials&scope=https://api.ebay.com/oauth/api_scope"
    try:
        res = _SESSION.post(_TOKEN_URL, headers=headers, data=payload, timeout=30)
        res.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise EbayUnavailable(f"can't reach eBay auth ({_root_cause(e)})") from e
    return res.json()["access_token"]


def _delivered_price(item):
    """(total, has_postage): total is price + CHEAPEST quoted postage when the
    seller quotes postage, or the bare price when they don't. None when the
    item has no usable price at all (never default a missing price to $0 —
    that fabricates bargains)."""
    try:
        price = float(item["price"]["value"])
    except (KeyError, TypeError, ValueError):
        return None
    postage = []
    for opt in item.get("shippingOptions", []):
        try:
            postage.append(float(opt["shippingCost"]["value"]))
        except (KeyError, TypeError, ValueError):
            continue
    if postage:
        return price + min(postage), True
    return price, False


def _search_condition(barcode, condition_ids, headers):
    # FIXED_PRICE only: an auction sitting at $0.99 with 6 days left is not
    # a real "lowest price". deliveryCountry pins postage quotes to AU.
    params = {
        "q": barcode,
        "filter": (f"conditionIds:{{{condition_ids}}},"
                   "buyingOptions:{FIXED_PRICE},deliveryCountry:AU"),
        "sort": "price",  # eBay sorts ascending by price + postage
        "limit": "20",
    }
    try:
        res = _SESSION.get(_SEARCH_URL, headers=headers, params=params)
        res.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise EbayUnavailable(f"can't reach eBay ({_root_cause(e)})") from e
    summaries = res.json().get("itemSummaries", [])
    if not summaries:
        return None, None, None

    with_postage, bare = [], []
    for item in summaries:
        dp = _delivered_price(item)
        if dp is None:
            continue
        (with_postage if dp[1] else bare).append((dp[0], item))

    # Sellers who quote postage give the honest delivered total; listings
    # with no postage info only count when nobody quotes postage at all.
    pool = with_postage or bare
    if not pool:
        return None, None, None
    lowest_price, best = min(pool, key=lambda t: t[0])
    return lowest_price, best.get("itemId"), best.get("title", "")


def _fetch_item_specifics(item_id, headers):
    encoded = urllib.parse.quote(item_id)
    try:
        res = _SESSION.get(f"{_ITEM_URL}{encoded}", headers=headers)
        res.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise EbayUnavailable(f"can't reach eBay ({_root_cause(e)})") from e
    data = res.json()
    aspects = {a.get("name", "").lower(): a.get("value", "") for a in data.get("localizedAspects", [])}
    return data.get("title", ""), aspects


def fetch_listing_row(barcode, token, marketplace=None):
    """One CSV row (dict, CSV_HEADERS keys) or None if nothing found on
    either New or Used condition search."""
    marketplace = marketplace or cfg.EBAY_MARKETPLACE_ID
    headers = {"Authorization": f"Bearer {token}", "X-EBAY-C-MARKETPLACE-ID": marketplace,
               "Content-Type": "application/json"}

    lowest_new, new_item_id, new_title = _search_condition(barcode, NEW_CONDITIONS, headers)
    lowest_used, used_item_id, used_title = _search_condition(barcode, USED_CONDITIONS, headers)
    if lowest_new is None and lowest_used is None:
        return None

    specifics_item_id = new_item_id or used_item_id
    fallback_title = new_title or used_title
    title, aspects = ("", {})
    if specifics_item_id:
        title, aspects = _fetch_item_specifics(specifics_item_id, headers)
    if not title:
        title = fallback_title

    slug = slugify_title(title, fallback=barcode)
    return {
        "Barcode": barcode,
        "Image Set Name": slug,
        "Title": title,
        "Region Code": aspects.get("region code", ""),
        "Genre": aspects.get("genre", ""),
        "Type": aspects.get("type", ""),
        "Season": aspects.get("season", ""),
        "Actor": aspects.get("actor", aspects.get("cast", "")),
        "Studio": aspects.get("studio", ""),
        "Language": aspects.get("language", ""),
        "Rating": aspects.get("rating", aspects.get("movie/tv title", "")),
        "Lowest Price New (AUD)": round(lowest_new, 2) if lowest_new is not None else "",
        "Lowest Price Used (AUD)": round(lowest_used, 2) if lowest_used is not None else "",
    }


def fetch_all(barcodes, token, delay=0.5):
    """rows = [{...} or {"Barcode": b} if nothing found, ...], same order as
    barcodes. Raises EbayUnavailable immediately on outage so the caller can
    offer retry/continue rather than silently returning partial data."""
    rows = []
    for barcode in barcodes:
        row = fetch_listing_row(barcode, token)
        rows.append(row or {"Barcode": barcode})
        time.sleep(delay)
    return rows


def write_csv(rows, out_path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
