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


@patch("ebaby.stages.ebay._SESSION.get")
def test_fetch_listing_row_prefers_new_price_and_specifics(mock_get):
    def side_effect(url, headers=None, params=None):
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


def test_write_csv_matches_the_thirteen_required_columns(tmp_path):
    rows = [{"Barcode": "123", "Image Set Name": "Movie", "Title": "The Movie",
             "Region Code": "4", "Genre": "Drama", "Type": "", "Season": "",
             "Actor": "", "Studio": "", "Language": "", "Rating": "",
             "Lowest Price New (AUD)": 12.5, "Lowest Price Used (AUD)": ""}]
    out = tmp_path / "Ebay_Details.csv"
    ebay.write_csv(rows, out)
    header = out.read_text(encoding="utf-8").splitlines()[0]
    assert header == ("Barcode,Image Set Name,Title,Region Code,Genre,Type,"
                      "Season,Actor,Studio,Language,Rating,"
                      "Lowest Price New (AUD),Lowest Price Used (AUD)")
