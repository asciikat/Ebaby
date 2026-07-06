from unittest.mock import patch, MagicMock

import pytest
import requests

from ebaby.stages import ebay


def _resp(json_body, status=200):
    m = MagicMock()
    m.status_code = status
    m.json.return_value = json_body
    m.raise_for_status = MagicMock()
    return m


def _error_resp(status):
    m = MagicMock()
    m.status_code = status
    m.raise_for_status.side_effect = requests.exceptions.HTTPError(f"{status} error")
    return m


def test_get_token_raises_when_not_configured(monkeypatch):
    monkeypatch.setattr(ebay.cfg, "ebay_configured", lambda: False)
    with pytest.raises(RuntimeError, match="credentials missing"):
        ebay.get_token()


@patch("ebaby.stages.ebay._SESSION.post")
def test_get_token_returns_access_token(mock_post, monkeypatch):
    monkeypatch.setattr(ebay.cfg, "ebay_configured", lambda: True)
    monkeypatch.setattr(ebay.cfg, "EBAY_CLIENT_ID", "id")
    monkeypatch.setattr(ebay.cfg, "EBAY_CLIENT_SECRET", "secret")
    mock_post.return_value = _resp({"access_token": "tok123"})
    assert ebay.get_token() == "tok123"


@patch("ebaby.stages.ebay._SESSION.post")
def test_get_token_network_failure_raises_ebay_unavailable(mock_post, monkeypatch):
    monkeypatch.setattr(ebay.cfg, "ebay_configured", lambda: True)
    monkeypatch.setattr(ebay.cfg, "EBAY_CLIENT_ID", "id")
    monkeypatch.setattr(ebay.cfg, "EBAY_CLIENT_SECRET", "secret")
    mock_post.side_effect = requests.exceptions.ConnectionError("boom")
    with pytest.raises(ebay.EbayUnavailable, match="no network"):
        ebay.get_token()


@patch("ebaby.stages.ebay._SESSION.post")
def test_get_token_http_error_status_raises_ebay_unavailable(mock_post, monkeypatch):
    # A 401 (e.g. bad/expired app credentials) must become a clean
    # EbayUnavailable, not an uncaught requests.HTTPError leaking to the caller.
    monkeypatch.setattr(ebay.cfg, "ebay_configured", lambda: True)
    monkeypatch.setattr(ebay.cfg, "EBAY_CLIENT_ID", "id")
    monkeypatch.setattr(ebay.cfg, "EBAY_CLIENT_SECRET", "secret")
    mock_post.return_value = _error_resp(401)
    with pytest.raises(ebay.EbayUnavailable):
        ebay.get_token()


@patch("ebaby.stages.ebay._SESSION.get")
def test_fetch_listing_row_prefers_new_price_and_specifics(mock_get):
    def side_effect(url, headers=None, params=None, timeout=None):
        if "item_summary/search" in url and "1000|1500|1750" in params["filter"]:
            return _resp({"itemSummaries": [
                {"itemId": "NEW1", "title": "New Listing Title",
                 "price": {"value": "10.00"}, "shippingOptions": []}
            ]})
        if "item_summary/search" in url:  # used-condition search
            return _resp({"itemSummaries": [
                {"itemId": "USED1", "title": "Used Listing Title",
                 "price": {"value": "5.00"}, "shippingOptions": []}
            ]})
        if "item/NEW1" in url:
            return _resp({"title": "Canonical Title",
                          "localizedAspects": [{"name": "Genre", "value": "Horror"}]})
        raise AssertionError(f"unexpected url {url}")
    mock_get.side_effect = side_effect

    row = ebay.fetch_listing_row("400638133393", token="tok")
    assert row["Title"] == "Canonical Title"
    assert row["Image Set Name"] == "Canonical_Title"
    assert row["Genre"] == "Horror"
    assert row["Lowest Price New (AUD)"] == 10.00
    assert row["Lowest Price Used (AUD)"] == 5.00


@patch("ebaby.stages.ebay._SESSION.get")
def test_fetch_listing_row_returns_none_when_nothing_found(mock_get):
    mock_get.return_value = _resp({"itemSummaries": []})
    assert ebay.fetch_listing_row("000000000000", token="tok") is None


@patch("ebaby.stages.ebay._SESSION.get")
def test_lowest_price_includes_postage(mock_get):
    """$8 item + $4 postage ($12 delivered) beats a $9 item + $5 postage."""
    def side_effect(url, headers=None, params=None, timeout=None):
        if "item_summary/search" in url and "1000|1500|1750" in params["filter"]:
            return _resp({"itemSummaries": []})
        if "item_summary/search" in url:
            return _resp({"itemSummaries": [
                {"itemId": "A", "title": "A", "price": {"value": "9.00"},
                 "shippingOptions": [{"shippingCost": {"value": "5.00"}}]},
                {"itemId": "B", "title": "B", "price": {"value": "8.00"},
                 "shippingOptions": [{"shippingCost": {"value": "4.00"}}]},
            ]})
        if "item/B" in url:
            return _resp({"title": "B", "localizedAspects": []})
        raise AssertionError(f"unexpected url {url}")
    mock_get.side_effect = side_effect
    row = ebay.fetch_listing_row("5021456189472", token="tok")
    assert row["Lowest Price Used (AUD)"] == 12.00


@patch("ebaby.stages.ebay._SESSION.get")
def test_no_postage_listing_ignored_when_others_quote_postage(mock_get):
    """A $2 pickup-only listing must not undercut a $10+$3 delivered quote."""
    def side_effect(url, headers=None, params=None, timeout=None):
        if "item_summary/search" in url and "1000|1500|1750" in params["filter"]:
            return _resp({"itemSummaries": []})
        if "item_summary/search" in url:
            return _resp({"itemSummaries": [
                {"itemId": "PICKUP", "title": "Pickup", "price": {"value": "2.00"}},
                {"itemId": "POSTED", "title": "Posted", "price": {"value": "10.00"},
                 "shippingOptions": [{"shippingCost": {"value": "3.00"}}]},
            ]})
        if "item/POSTED" in url:
            return _resp({"title": "Posted", "localizedAspects": []})
        raise AssertionError(f"unexpected url {url}")
    mock_get.side_effect = side_effect
    row = ebay.fetch_listing_row("5021456189472", token="tok")
    assert row["Lowest Price Used (AUD)"] == 13.00


@patch("ebaby.stages.ebay._SESSION.get")
def test_search_filters_out_auctions_and_pins_au_delivery(mock_get):
    seen = {}
    def side_effect(url, headers=None, params=None, timeout=None):
        if "item_summary/search" in url:
            seen.setdefault("filters", []).append(params["filter"])
            return _resp({"itemSummaries": []})
        raise AssertionError(f"unexpected url {url}")
    mock_get.side_effect = side_effect
    assert ebay.fetch_listing_row("5021456189472", token="tok") is None
    for f in seen["filters"]:
        assert "buyingOptions:{FIXED_PRICE}" in f
        assert "deliveryCountry:AU" in f


@patch("ebaby.stages.ebay._SESSION.get")
def test_cheapest_of_multiple_postage_options_is_used(mock_get):
    def side_effect(url, headers=None, params=None, timeout=None):
        if "item_summary/search" in url and "1000|1500|1750" in params["filter"]:
            return _resp({"itemSummaries": [
                {"itemId": "N", "title": "N", "price": {"value": "10.00"},
                 "shippingOptions": [{"shippingCost": {"value": "9.00"}},
                                      {"shippingCost": {"value": "2.50"}}]},
            ]})
        if "item_summary/search" in url:
            return _resp({"itemSummaries": []})
        if "item/N" in url:
            return _resp({"title": "N", "localizedAspects": []})
        raise AssertionError(f"unexpected url {url}")
    mock_get.side_effect = side_effect
    row = ebay.fetch_listing_row("5021456189472", token="tok")
    assert row["Lowest Price New (AUD)"] == 12.50


def test_write_csv_matches_the_thirteen_required_columns(tmp_path):
    rows = [{"Barcode": "123", "Image Set Name": "Movie", "Title": "The Movie",
             "Region Code": "4", "Genre": "Drama", "Type": "", "Season": "",
             "Actor": "", "Studio": "", "Language": "", "Rating": "",
             "Lowest Price New (AUD)": 12.5, "Lowest Price Used (AUD)": ""}]
    out = tmp_path / "Ebay_Details.csv"
    ebay.write_csv(rows, out)
    header = out.read_text(encoding="utf-8-sig").splitlines()[0]
    assert header == ("Barcode,Image Set Name,Title,Region Code,Genre,Type,"
                      "Season,Actor,Studio,Language,Rating,"
                      "Lowest Price New (AUD),Lowest Price Used (AUD)")


@patch("ebaby.stages.ebay._SESSION.get")
def test_title_search_can_beat_barcode_price(mock_get):
    """Sellers who never enter a barcode are only findable by title — their
    cheaper delivered total must win. Postage rules still apply per search."""
    def side_effect(url, headers=None, params=None, timeout=None):
        if "item_summary/search" in url:
            by_title = "Some Movie" in params["q"]
            if "1000|1500|1750" in params["filter"]:
                return _resp({"itemSummaries": []})  # nothing new anywhere
            if by_title:
                return _resp({"itemSummaries": [
                    {"itemId": "T1", "title": "Some Movie DVD",
                     "price": {"value": "8.00"},
                     "shippingOptions": [{"shippingCost": {"value": "2.00"}}]},
                ]})
            return _resp({"itemSummaries": [
                {"itemId": "B1", "title": "Some Movie",
                 "price": {"value": "12.00"},
                 "shippingOptions": [{"shippingCost": {"value": "3.00"}}]},
            ]})
        if "item/B1" in url:
            return _resp({"title": "Some Movie", "localizedAspects": []})
        raise AssertionError(f"unexpected url {url}")
    mock_get.side_effect = side_effect
    row = ebay.fetch_listing_row("5021456189472", token="tok")
    assert row["Lowest Price Used (AUD)"] == 10.00  # 8 + 2 beats 12 + 3
    assert row["Title"] == "Some Movie"


@patch("ebaby.stages.ebay._SESSION.get")
def test_title_search_appends_dvd_keyword(mock_get):
    seen = []
    def side_effect(url, headers=None, params=None, timeout=None):
        if "item_summary/search" in url:
            seen.append(params["q"])
            if params["q"] == "5021456189472" and "3000" in params["filter"]:
                return _resp({"itemSummaries": [
                    {"itemId": "B1", "title": "Blade Runner",
                     "price": {"value": "9.00"},
                     "shippingOptions": [{"shippingCost": {"value": "3.00"}}]},
                ]})
            return _resp({"itemSummaries": []})
        if "item/B1" in url:
            return _resp({"title": "Blade Runner", "localizedAspects": []})
        raise AssertionError(f"unexpected url {url}")
    mock_get.side_effect = side_effect
    ebay.fetch_listing_row("5021456189472", token="tok")
    assert "Blade Runner DVD" in seen  # DVD keyword keeps Blu-rays/CDs out


@patch("ebaby.stages.ebay._SESSION.get")
def test_title_search_failure_keeps_barcode_prices(mock_get):
    """A transient outage during the TITLE pass must not throw away the
    barcode-pass prices (or, via fetch_all, every other disc's row)."""
    def side_effect(url, headers=None, params=None, timeout=None):
        if "item_summary/search" in url:
            if "Some Movie" in params["q"]:
                raise requests.exceptions.ConnectionError("mid-run outage")
            if "1000|1500|1750" in params["filter"]:
                return _resp({"itemSummaries": []})
            return _resp({"itemSummaries": [
                {"itemId": "B1", "title": "Some Movie",
                 "price": {"value": "12.00"},
                 "shippingOptions": [{"shippingCost": {"value": "3.00"}}]},
            ]})
        if "item/B1" in url:
            return _resp({"title": "Some Movie", "localizedAspects": []})
        raise AssertionError(f"unexpected url {url}")
    mock_get.side_effect = side_effect
    row = ebay.fetch_listing_row("5021456189472", token="tok")
    assert row["Lowest Price Used (AUD)"] == 15.00  # barcode pass survives


@patch("ebaby.stages.ebay._SESSION.get")
def test_title_search_absurdly_low_match_is_rejected(mock_get):
    """A title match far below the barcode-confirmed price (empty case /
    single disc vs boxset / generic lot) must NOT drag the price down."""
    def side_effect(url, headers=None, params=None, timeout=None):
        if "item_summary/search" in url:
            by_title = "Boxset" in params["q"]
            if "1000|1500|1750" in params["filter"]:
                return _resp({"itemSummaries": []})
            if by_title:  # a $1.50-delivered wrong match
                return _resp({"itemSummaries": [
                    {"itemId": "T1", "title": "Boxset DVD",
                     "price": {"value": "1.00"},
                     "shippingOptions": [{"shippingCost": {"value": "0.50"}}]},
                ]})
            return _resp({"itemSummaries": [  # barcode-confirmed $30 delivered
                {"itemId": "B1", "title": "Boxset",
                 "price": {"value": "27.00"},
                 "shippingOptions": [{"shippingCost": {"value": "3.00"}}]},
            ]})
        if "item/B1" in url:
            return _resp({"title": "Boxset", "localizedAspects": []})
        raise AssertionError(f"unexpected url {url}")
    mock_get.side_effect = side_effect
    row = ebay.fetch_listing_row("5021456189472", token="tok")
    assert row["Lowest Price Used (AUD)"] == 30.00  # $1.50 match rejected


def test_guarded_min_accepts_title_price_when_no_barcode_price():
    # nothing to sanity-check against -> take the title figure
    assert ebay._guarded_min(None, 4.0) == 4.0
    # plausible title price wins
    assert ebay._guarded_min(20.0, 15.0) == 15.0
    # implausible (below 30% of barcode price) rejected
    assert ebay._guarded_min(20.0, 2.0) == 20.0
