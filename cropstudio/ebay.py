"""eBay Browse (buy) API client — app-only (client-credentials) token.

Ported from the WSL pipeline's ebay_api.py. Used to look up a barcode's
matched title/price/specifics. Never used to override a title that came
from Qwen — see service.resolve_identity, which decides priority.
"""
import base64

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_TOKEN_URL = "https://api.ebay.com/identity/v1/oauth2/token"
_SEARCH_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"


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


def get_token(config, session=None) -> str:
    """Fetch an application access token for the Browse API."""
    if not config.configured:
        raise RuntimeError(
            "eBay credentials missing — set EBAY_CLIENT_ID / EBAY_CLIENT_SECRET "
            "in this project's .env")
    session = session or _make_session()
    creds = base64.b64encode(
        f"{config.client_id}:{config.client_secret}".encode()).decode()
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {creds}",
    }
    payload = ("grant_type=client_credentials"
               "&scope=https://api.ebay.com/oauth/api_scope")
    try:
        res = session.post(_TOKEN_URL, headers=headers, data=payload, timeout=30)
    except requests.exceptions.RequestException as e:
        raise EbayUnavailable(f"can't reach eBay auth ({_root_cause(e)})") from e
    res.raise_for_status()
    return res.json()["access_token"]


def lookup_barcode(barcode, token, config, session=None) -> dict:
    """Research one barcode. Returns a dict (always includes 'barcode' and
    'found'); on a hit it also has title/price/currency/category/image_url/
    item_url. Raises EbayUnavailable if the network is unreachable."""
    session = session or _make_session()
    headers = {
        "Authorization": f"Bearer {token}",
        "X-EBAY-C-MARKETPLACE-ID": config.marketplace_id,
    }
    try:
        res = session.get(_SEARCH_URL, headers=headers,
                          params={"q": barcode, "limit": 1}, timeout=30)
    except requests.exceptions.RequestException as e:
        raise EbayUnavailable(f"can't reach eBay ({_root_cause(e)})") from e
    out = {"barcode": barcode, "found": False, "title": "", "price": "",
           "currency": "", "category": "", "image_url": "", "item_url": ""}
    if res.status_code == 200 and res.json().get("itemSummaries"):
        item = res.json()["itemSummaries"][0]
        out.update(
            found=True,
            title=item.get("title", ""),
            price=item.get("price", {}).get("value", ""),
            currency=item.get("price", {}).get("currency", ""),
            category=(item.get("categories", [{}]) or [{}])[0].get("categoryId", ""),
            image_url=item.get("image", {}).get("imageUrl", ""),
            item_url=item.get("itemWebUrl", ""),
        )
    return out


def _root_cause(exc):
    if isinstance(exc, requests.exceptions.ConnectionError):
        return "no network / DNS"
    if isinstance(exc, requests.exceptions.Timeout):
        return "timed out"
    return type(exc).__name__
