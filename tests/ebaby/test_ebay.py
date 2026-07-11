from unittest.mock import patch, MagicMock

import pytest
import requests

from ebaby.stages import ebay


@pytest.fixture(autouse=True)
def _no_network_scrape(monkeypatch):
    """Tests must NEVER hit the real eBay or the rates API — the tests that
    cover the scraper/rates patch them explicitly."""
    monkeypatch.setattr(ebay, "_SCRAPE_AVAILABLE", False)
    monkeypatch.setattr(ebay, "_RATES", {"GBP": 2.0, "USD": 1.5})


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
        if "marketplace_insights" in url:
            return _resp({}, status=403)
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
    # both conditions' lowest prices are carried internally — server.py picks
    # the one that matches how the disc was actually uploaded
    assert row["_lowest_new"] == 10.00
    assert row["_lowest_used"] == 5.00


@patch("ebaby.stages.ebay._SESSION.get")
def test_fetch_listing_row_returns_none_when_nothing_found(mock_get):
    mock_get.return_value = _resp({"itemSummaries": []})
    assert ebay.fetch_listing_row("000000000000", token="tok") is None


@patch("ebaby.stages.ebay._SESSION.get")
def test_lowest_price_includes_postage(mock_get):
    """$8 item + $4 postage ($12 delivered) beats a $9 item + $5 postage."""
    def side_effect(url, headers=None, params=None, timeout=None):
        if "marketplace_insights" in url:
            return _resp({}, status=403)
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
    assert row["_lowest_used"] == 12.00


@patch("ebaby.stages.ebay._SESSION.get")
def test_no_postage_listing_ignored_when_others_quote_postage(mock_get):
    """A $2 pickup-only listing must not undercut a $10+$3 delivered quote."""
    def side_effect(url, headers=None, params=None, timeout=None):
        if "marketplace_insights" in url:
            return _resp({}, status=403)
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
    assert row["_lowest_used"] == 13.00


@patch("ebaby.stages.ebay._SESSION.get")
def test_barcode_pass_searches_by_gtin_title_pass_by_text(mock_get):
    """Barcode lookups must use eBay's gtin= (exact product-identifier match).
    q= is fuzzy text search — measured live, it ranked a bogus 'Intruder'
    listing above four genuine 'One Step Beyond' listings for barcode
    9327478001218 and the wrong title became the whole disc's name. The
    title pass stays q= on purpose (recall for sellers who skip the barcode)."""
    calls = []
    def side_effect(url, headers=None, params=None, timeout=None):
        if "marketplace_insights" in url:
            return _resp({}, status=403)
        if "item_summary/search" in url:
            calls.append(dict(params))
            if params.get("gtin"):  # barcode pass finds the disc
                return _resp({"itemSummaries": [
                    {"itemId": "B1", "title": "One Step Beyond Volume 1",
                     "price": {"value": "14.95"},
                     "shippingOptions": [{"shippingCost": {"value": "0.00"}}]},
                ]})
            return _resp({"itemSummaries": []})  # title pass: nothing cheaper
        if "item/B1" in url:
            return _resp({"title": "One Step Beyond Volume 1", "localizedAspects": []})
        raise AssertionError(f"unexpected url {url}")
    mock_get.side_effect = side_effect
    row = ebay.fetch_listing_row("9327478001218", token="tok")
    assert row["Title"] == "One Step Beyond Volume 1"
    barcode_calls = [c for c in calls if c.get("gtin")]
    title_calls = [c for c in calls if c.get("q")]
    assert len(barcode_calls) == 2                       # New + Used, both gtin
    assert all(c["gtin"] == "9327478001218" and "q" not in c for c in barcode_calls)
    assert title_calls and all("gtin" not in c for c in title_calls)


@patch("ebaby.stages.ebay._SESSION.get")
def test_search_filters_out_auctions_and_pins_au_delivery(mock_get):
    seen = {}
    def side_effect(url, headers=None, params=None, timeout=None):
        if "marketplace_insights" in url:
            return _resp({}, status=403)
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
        if "marketplace_insights" in url:
            return _resp({}, status=403)
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
    assert row["_lowest_new"] == 12.50


def test_write_csv_matches_the_required_columns(tmp_path):
    rows = [{"Barcode": "123", "Image Set Name": "Movie", "Title": "The Movie",
             "Condition": "Used", "Region Code": "4", "Genre": "Drama", "Type": "",
             "Season": "", "Actor": "", "Studio": "", "Language": "", "Rating": "",
             "Lowest Price (AUD)": 12.5, "Your Price (AUD)": 11.25}]
    out = tmp_path / "Ebay_Details.csv"
    ebay.write_csv(rows, out)
    header = out.read_text(encoding="utf-8-sig").splitlines()[0]
    # Type/Season/Actor/Studio/Language/Rating are blank on this one row, so
    # every one of those gets pruned — only Region Code+Genre have data
    assert header == ("Barcode,Image Set Name,Title,Condition,Region Code,Genre,"
                      "Lowest Price (AUD),Your Price (AUD)")


def test_write_csv_keeps_a_metadata_column_if_any_row_has_it(tmp_path):
    rows = [
        {"Barcode": "1", "Image Set Name": "A", "Title": "A", "Condition": "Used",
         "Season": "", "Lowest Price (AUD)": 10, "Your Price (AUD)": 8.99},
        {"Barcode": "2", "Image Set Name": "B", "Title": "B", "Condition": "New",
         "Season": "3", "Lowest Price (AUD)": 20, "Your Price (AUD)": 17.99},
    ]
    out = tmp_path / "Ebay_Details.csv"
    ebay.write_csv(rows, out)
    header = out.read_text(encoding="utf-8-sig").splitlines()[0]
    assert "Season" in header  # one row has it -> keep it for the whole batch


def test_write_csv_never_drops_the_core_columns_even_if_all_blank(tmp_path):
    rows = [{"Barcode": "", "Image Set Name": "", "Title": "", "Condition": "",
             "Lowest Price (AUD)": "", "Your Price (AUD)": ""}]
    out = tmp_path / "Ebay_Details.csv"
    ebay.write_csv(rows, out)
    header = out.read_text(encoding="utf-8-sig").splitlines()[0]
    for core in ("Barcode", "Image Set Name", "Title", "Condition",
                 "Lowest Price (AUD)", "Your Price (AUD)"):
        assert core in header


def test_undercut_is_ten_percent_off_charmed_and_blank_when_no_comp():
    assert ebay._undercut(30.0) == 26.99    # 27.00 -> .99
    assert ebay._undercut(22.20) == 19.99   # 22.20*0.9 = 19.98 -> 19.99
    assert ebay._undercut(None) == ""       # no comp -> blank


def test_charm_always_ends_in_ninety_nine():
    assert ebay._charm(27.85) == 27.99
    assert ebay._charm(27.20) == 26.99
    assert ebay._charm(27.43) == 26.99
    assert ebay._charm(8.955) == 8.99
    assert ebay._charm(4.50) == 4.99


@patch("ebaby.stages.ebay._SESSION.get")
def test_title_search_can_beat_barcode_price(mock_get):
    """Sellers who never enter a barcode are only findable by title — their
    cheaper delivered total must win. Postage rules still apply per search."""
    def side_effect(url, headers=None, params=None, timeout=None):
        if "marketplace_insights" in url:
            return _resp({}, status=403)
        if "item_summary/search" in url:
            by_title = "Some Movie" in params.get("q", "")
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
    assert row["_lowest_used"] == 10.00  # 8 + 2 beats 12 + 3
    assert row["Title"] == "Some Movie"


@patch("ebaby.stages.ebay._SESSION.get")
def test_title_search_appends_dvd_keyword(mock_get):
    seen = []
    def side_effect(url, headers=None, params=None, timeout=None):
        if "marketplace_insights" in url:
            return _resp({}, status=403)
        if "item_summary/search" in url:
            seen.append(params.get("q") or params.get("gtin"))
            if params.get("gtin") == "5021456189472" and "3000" in params["filter"]:
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
        if "marketplace_insights" in url:
            return _resp({}, status=403)
        if "item_summary/search" in url:
            if "Some Movie" in params.get("q", ""):
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
    assert row["_lowest_used"] == 15.00  # barcode pass survives


@patch("ebaby.stages.ebay._SESSION.get")
def test_title_search_absurdly_low_match_is_rejected(mock_get):
    """A title match far below the barcode-confirmed price (empty case /
    single disc vs boxset / generic lot) must NOT drag the price down."""
    def side_effect(url, headers=None, params=None, timeout=None):
        if "marketplace_insights" in url:
            return _resp({}, status=403)
        if "item_summary/search" in url:
            by_title = "Boxset" in params.get("q", "")
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
    assert row["_lowest_used"] == 30.00  # $1.50 match rejected


def test_guarded_min_accepts_title_price_when_no_barcode_price():
    # nothing to sanity-check against -> take the title figure
    assert ebay._guarded_min(None, 4.0) == 4.0
    # plausible title price wins
    assert ebay._guarded_min(20.0, 15.0) == 15.0
    # implausible (below 30% of barcode price) rejected
    assert ebay._guarded_min(20.0, 2.0) == 20.0


@patch("ebaby.stages.ebay._SESSION.get")
def test_last_sold_fallback_when_nothing_on_sale(mock_get):
    """No active listings at all -> the row still comes back, carrying the
    most RECENT (not cheapest) sold price from Marketplace Insights."""
    ebay._INSIGHTS_AVAILABLE = True

    def side_effect(url, headers=None, params=None, timeout=None):
        if "marketplace_insights" in url:
            return _resp({"itemSales": [
                {"itemId": "S1", "title": "Rare Movie DVD",
                 "lastSoldPrice": {"value": "9.50"},
                 "lastSoldDate": "2026-06-01T00:00:00Z"},
                {"itemId": "S2", "title": "Rare Movie DVD",
                 "lastSoldPrice": {"value": "4.00"},
                 "lastSoldDate": "2026-07-01T00:00:00Z"},
            ]})
        if "item_summary/search" in url:
            return _resp({"itemSummaries": []})   # nothing on sale anywhere
        if "/item/" in url:
            return _resp({"title": "Rare Movie DVD", "localizedAspects": []})
        raise AssertionError(f"unexpected url {url}")

    mock_get.side_effect = side_effect
    row = ebay.fetch_listing_row("5021456189472", "tok")
    assert row is not None
    assert row["_lowest_new"] is None and row["_lowest_used"] is None
    assert row["_sold_new"] == 4.00    # newest sale wins, not the cheapest
    assert row["Title"] == "Rare Movie DVD"


@patch("ebaby.stages.ebay._SESSION.get")
def test_insights_403_disables_itself_and_returns_none_row(mock_get):
    ebay._INSIGHTS_AVAILABLE = True

    def side_effect(url, headers=None, params=None, timeout=None):
        if "marketplace_insights" in url:
            return _resp({}, status=403)
        if "item_summary/search" in url:
            return _resp({"itemSummaries": []})
        raise AssertionError(f"unexpected url {url}")

    mock_get.side_effect = side_effect
    assert ebay.fetch_listing_row("5021456189472", "tok") is None
    assert ebay._INSIGHTS_AVAILABLE is False   # one 403 turns it off for the run
    ebay._INSIGHTS_AVAILABLE = True            # don't leak into other tests


@patch("ebaby.stages.ebay._scrape_last_sold")
@patch("ebaby.stages.ebay._SESSION.get")
def test_sold_page_scrape_backs_up_insights(mock_get, mock_scrape):
    """Insights refused (403) -> the sold-listings page scrape supplies the
    last-sold price instead."""
    ebay._INSIGHTS_AVAILABLE = True

    def side_effect(url, headers=None, params=None, timeout=None):
        if "marketplace_insights" in url:
            return _resp({}, status=403)
        if "item_summary/search" in url:
            return _resp({"itemSummaries": []})
        raise AssertionError(f"unexpected url {url}")

    mock_get.side_effect = side_effect
    mock_scrape.return_value = (6.50, None, None)
    row = ebay.fetch_listing_row("5021456189472", "tok")
    assert row is not None
    assert row["_sold_new"] == 6.50 and row["_sold_used"] == 6.50
    ebay._INSIGHTS_AVAILABLE = True


@patch("ebaby.stages.ebay._SESSION.get")
def test_worldwide_listing_fills_in_when_au_is_empty(mock_get):
    """AU has nothing on sale -> the UK listing (GBP, postage to AU quoted)
    wins, converted to AUD: (9.99 + 2.50) * 2.0 = 24.98."""
    ebay._INSIGHTS_AVAILABLE = True

    def side_effect(url, headers=None, params=None, timeout=None):
        if "marketplace_insights" in url:
            return _resp({}, status=403)
        if "item_summary/search" in url:
            if headers.get("X-EBAY-C-MARKETPLACE-ID") == "EBAY_GB":
                return _resp({"itemSummaries": [
                    {"itemId": "UK1", "title": "The Who From the Bush DVD",
                     "price": {"value": "9.99", "currency": "GBP"},
                     "shippingOptions": [{"shippingCost": {"value": "2.50",
                                                           "currency": "GBP"}}]},
                ]})
            return _resp({"itemSummaries": []})   # AU + US empty
        if "/item/" in url:
            return _resp({"title": "The Who From the Bush DVD",
                          "localizedAspects": []})
        raise AssertionError(f"unexpected url {url}")

    mock_get.side_effect = side_effect
    row = ebay.fetch_listing_row("5021456189472", "tok")
    assert row is not None
    assert row["_lowest_new"] == 24.98 and row["_ww_new"] is True
    assert row["Title"] == "The Who From the Bush DVD"
