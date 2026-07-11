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
_INSIGHTS_URL = "https://api.ebay.com/buy/marketplace_insights/v1_beta/item_sales/search"

NEW_CONDITIONS = "1000|1500|1750"
USED_CONDITIONS = "3000|4000|5000|6000"

CSV_HEADERS = [
    "Barcode", "Image Set Name", "Title", "Condition", "Region Code", "Genre",
    "Type", "Season", "Actor", "Studio", "Language", "Rating",
    "Lowest Price (AUD)", "Last Sold (AUD)", "Your Price (AUD)",
]

# Columns written even when every row in the batch is blank for them — the
# identity/pricing fields the user always wants a slot for. Everything else
# in CSV_HEADERS is metadata eBay sometimes doesn't return (Region Code,
# Season, etc.); a column that's blank on EVERY row in this batch is dropped
# so the sheet isn't full of dead space.
_ALWAYS_KEPT_HEADERS = {
    "Barcode", "Image Set Name", "Title", "Condition",
    "Lowest Price (AUD)", "Your Price (AUD)",
}

# Undercut the cheapest comparable DELIVERED price by this much, so the listing
# sits at the top of the buyer's price-sorted results without giving away
# margin. DVDs are low-dollar and heavily comparison-shopped — 10% reads as
# clearly cheaper where 5% barely registers on a results page.
UNDERCUT_RATIO = 0.90


def _charm(price):
    """Round to the nearest whole dollar and knock off a cent, so every
    suggested price ends in .99 — $26.99, $8.99 — the classic eBay look.
    (round-half-up, not banker's rounding, so it's predictable.)"""
    if price < 1.0:
        return round(price, 2)          # too cheap to charm sensibly
    return round(int(price + 0.5) - 0.01, 2)


def _undercut(price):
    """A competitive delivered price ~`UNDERCUT_RATIO` of the cheapest comp,
    charm-rounded to a .99/.00, or "" when there was no comp to undercut. This
    is a target DELIVERED total (item + your postage), since that's the number
    buyers actually compare and eBay sorts on."""
    if price is None:
        return ""
    return _charm(price * UNDERCUT_RATIO)


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


def _search_condition(query, condition_ids, headers, by_gtin=False):
    # FIXED_PRICE only: an auction sitting at $0.99 with 6 days left is not
    # a real "lowest price". deliveryCountry pins postage quotes to AU.
    #
    # by_gtin: barcodes must go through eBay's gtin= param (exact product-
    # identifier match), NOT q=. q= is fuzzy text search — measured live, it
    # ranked a bogus "Intruder (DVD, 2009)" listing above four genuine "One
    # Step Beyond" listings for barcode 9327478001218, which then became the
    # canonical title for the whole disc. gtin= returned only true matches.
    params = {
        ("gtin" if by_gtin else "q"): query,
        "filter": (f"conditionIds:{{{condition_ids}}},"
                   "buyingOptions:{FIXED_PRICE},deliveryCountry:AU"),
        "sort": "price",  # eBay sorts ascending by price + postage
        "limit": "20",
    }
    try:
        res = _SESSION.get(_SEARCH_URL, headers=headers, params=params, timeout=30)
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


# Marketplace Insights (sold history) is a limited-release eBay API — most
# keys get 401/403. One refusal turns it off for the rest of the process so a
# batch doesn't burn a failing call per disc.
_INSIGHTS_AVAILABLE = True


def _search_last_sold(query, condition_ids, headers, by_gtin=False):
    """Most recent SOLD price for the query, or (None, None, None).
    Same (price, itemId, title) shape as _search_condition."""
    global _INSIGHTS_AVAILABLE
    if not _INSIGHTS_AVAILABLE:
        return None, None, None
    params = {
        ("gtin" if by_gtin else "q"): query,
        "filter": f"conditionIds:{{{condition_ids}}}",
        "limit": "20",
    }
    try:
        res = _SESSION.get(_INSIGHTS_URL, headers=headers, params=params, timeout=30)
        if res.status_code in (401, 403):
            _INSIGHTS_AVAILABLE = False
            return None, None, None
        res.raise_for_status()
    except requests.exceptions.RequestException:
        return None, None, None   # best-effort fallback — never sink the batch
    best = None                   # the NEWEST sale, not the cheapest
    for sale in res.json().get("itemSales", []):
        price = (sale.get("lastSoldPrice") or {}).get("value")
        if price is None:
            continue
        when = sale.get("lastSoldDate", "")
        if best is None or when > best[0]:
            best = (when, float(price), sale.get("itemId"), sale.get("title", ""))
    if best is None:
        return None, None, None
    return best[1], best[2], best[3]


def _fetch_item_specifics(item_id, headers):
    encoded = urllib.parse.quote(item_id)
    try:
        res = _SESSION.get(f"{_ITEM_URL}{encoded}", headers=headers, timeout=30)
        res.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise EbayUnavailable(f"can't reach eBay ({_root_cause(e)})") from e
    data = res.json()
    aspects = {a.get("name", "").lower(): a.get("value", "") for a in data.get("localizedAspects", [])}
    return data.get("title", ""), aspects


def _guarded_min(barcode_price, title_price, floor_ratio=0.3):
    """Lowest of the barcode- and title-search delivered prices, EXCEPT a
    title match is rejected when it's below floor_ratio of the
    barcode-confirmed price. A title-only listing that cheap is almost always
    the wrong item — an empty case, a single disc where ours is a boxset, or a
    generic 'DVD lot' — and letting it through is exactly how prices come out
    implausibly low. With no barcode price to check against, the title price
    is taken as-is (better a rough figure than none)."""
    if title_price is None:
        return barcode_price
    if barcode_price is None:
        return title_price
    if title_price < barcode_price * floor_ratio:
        return barcode_price
    return min(barcode_price, title_price)


def fetch_listing_row(barcode, token, marketplace=None):
    """One CSV row (dict, CSV_HEADERS keys) or None if nothing found on
    either New or Used condition search.

    Two searches per condition: first by BARCODE, then a second by TITLE
    (many sellers never enter the barcode, so barcode-only search misses
    their — often cheaper — listings). The lowest delivered price across
    both searches wins; the same postage rules apply to each."""
    marketplace = marketplace or cfg.EBAY_MARKETPLACE_ID
    headers = {"Authorization": f"Bearer {token}", "X-EBAY-C-MARKETPLACE-ID": marketplace,
               "Content-Type": "application/json"}

    lowest_new, new_item_id, new_title = _search_condition(
        barcode, NEW_CONDITIONS, headers, by_gtin=True)
    lowest_used, used_item_id, used_title = _search_condition(
        barcode, USED_CONDITIONS, headers, by_gtin=True)
    # nothing on sale right now -> what did it LAST SELL for?
    sold_new = sold_used = None
    if lowest_new is None:
        sold_new, sn_id, sn_title = _search_last_sold(
            barcode, NEW_CONDITIONS, headers, by_gtin=True)
        new_item_id, new_title = new_item_id or sn_id, new_title or sn_title
    if lowest_used is None:
        sold_used, su_id, su_title = _search_last_sold(
            barcode, USED_CONDITIONS, headers, by_gtin=True)
        used_item_id, used_title = used_item_id or su_id, used_title or su_title
    if lowest_new is None and lowest_used is None and             sold_new is None and sold_used is None:
        return None

    specifics_item_id = new_item_id or used_item_id
    fallback_title = new_title or used_title
    title, aspects = ("", {})
    if specifics_item_id:
        title, aspects = _fetch_item_specifics(specifics_item_id, headers)
    if not title:
        title = fallback_title

    if title:
        # 'DVD' keeps the title search from matching Blu-rays/CDs/posters.
        # Best-effort: a transient failure here must not throw away the
        # barcode-pass prices (or, via fetch_all, the whole batch's rows).
        q = title if "dvd" in title.lower() else f"{title} DVD"
        try:
            title_new, _, _ = _search_condition(q, NEW_CONDITIONS, headers)
            title_used, _, _ = _search_condition(q, USED_CONDITIONS, headers)
        except EbayUnavailable:
            title_new = title_used = None
        lowest_new = _guarded_min(lowest_new, title_new)
        lowest_used = _guarded_min(lowest_used, title_used)
        if lowest_new is None and sold_new is None:
            sold_new, _, _ = _search_last_sold(q, NEW_CONDITIONS, headers)
        if lowest_used is None and sold_used is None:
            sold_used, _, _ = _search_last_sold(q, USED_CONDITIONS, headers)

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
        # not real CSV columns (extrasaction="ignore" drops them on write) —
        # the caller doesn't yet know if THIS barcode is the new or used copy,
        # so both lowest prices are carried until server.py picks the one
        # that matches how the disc was actually uploaded.
        "_lowest_new": round(lowest_new, 2) if lowest_new is not None else None,
        "_lowest_used": round(lowest_used, 2) if lowest_used is not None else None,
        "_sold_new": round(sold_new, 2) if sold_new is not None else None,
        "_sold_used": round(sold_used, 2) if sold_used is not None else None,
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


def _prune_blank_columns(rows):
    """CSV_HEADERS minus any optional column that's blank on EVERY row in
    this batch (e.g. a run with no TV box sets never has a Season value)."""
    return [h for h in CSV_HEADERS if h in _ALWAYS_KEPT_HEADERS
            or any(row.get(h, "") not in ("", None) for row in rows)]


def write_csv(rows, out_path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = _prune_blank_columns(rows)
    # utf-8-sig: without the BOM, Excel renders accented titles as mojibake
    with open(out_path, mode="w", newline="", encoding="utf-8-sig") as f:
        # extrasaction="ignore": rows carry internal keys (Stock, _lowest_*)
        # that aren't CSV columns — drop them instead of raising.
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
