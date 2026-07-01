import pytest
import requests

from cropstudio.ebay import EbayUnavailable, get_token, lookup_barcode
from cropstudio.ebay_config import EbayConfig


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = json_data or {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(str(self.status_code))


class _FakeSession:
    def __init__(self, post_response=None, get_response=None, raise_exc=None):
        self._post_response = post_response
        self._get_response = get_response
        self._raise_exc = raise_exc

    def post(self, *a, **kw):
        if self._raise_exc:
            raise self._raise_exc
        return self._post_response

    def get(self, *a, **kw):
        if self._raise_exc:
            raise self._raise_exc
        return self._get_response


def test_get_token_raises_when_not_configured():
    with pytest.raises(RuntimeError, match="credentials missing"):
        get_token(EbayConfig(), session=_FakeSession())


def test_get_token_returns_access_token():
    cfg = EbayConfig(client_id="id", client_secret="secret")
    session = _FakeSession(post_response=_FakeResponse(200, {"access_token": "tok123"}))
    assert get_token(cfg, session=session) == "tok123"


def test_get_token_raises_ebay_unavailable_on_connection_error():
    cfg = EbayConfig(client_id="id", client_secret="secret")
    session = _FakeSession(raise_exc=requests.exceptions.ConnectionError("boom"))
    with pytest.raises(EbayUnavailable, match="no network"):
        get_token(cfg, session=session)


def test_lookup_barcode_found():
    cfg = EbayConfig(client_id="id", client_secret="secret")
    item = {
        "title": "Sexy Beast (DVD, 2001)",
        "price": {"value": "8.50", "currency": "AUD"},
        "categories": [{"categoryId": "617"}],
        "image": {"imageUrl": "http://x/y.jpg"},
        "itemWebUrl": "http://x/item",
    }
    session = _FakeSession(get_response=_FakeResponse(200, {"itemSummaries": [item]}))
    out = lookup_barcode("9325336022306", "tok", cfg, session=session)
    assert out["found"] is True
    assert out["title"] == "Sexy Beast (DVD, 2001)"
    assert out["price"] == "8.50"
    assert out["currency"] == "AUD"


def test_lookup_barcode_not_found():
    cfg = EbayConfig(client_id="id", client_secret="secret")
    session = _FakeSession(get_response=_FakeResponse(200, {"itemSummaries": []}))
    out = lookup_barcode("0000000000000", "tok", cfg, session=session)
    assert out["found"] is False
    assert out["barcode"] == "0000000000000"


def test_lookup_barcode_raises_ebay_unavailable_on_timeout():
    cfg = EbayConfig(client_id="id", client_secret="secret")
    session = _FakeSession(raise_exc=requests.exceptions.Timeout("slow"))
    with pytest.raises(EbayUnavailable, match="timed out"):
        lookup_barcode("123", "tok", cfg, session=session)
