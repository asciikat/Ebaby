import sqlite3
from unittest.mock import patch

from ebaby.stages import icollect


def _index(tmp_path, rows):
    db = tmp_path / "icollect.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE items (barcode TEXT, item_id INTEGER, title TEXT,"
                " PRIMARY KEY (barcode, item_id))")
    con.executemany("INSERT INTO items VALUES (?, ?, ?)", rows)
    con.commit()
    con.close()
    return db


def test_barcode_forms_cover_upc_ean_spellings():
    assert "0036000291452" in icollect._barcode_forms("036000291452")  # 12 -> +0
    assert "036000291452" in icollect._barcode_forms("0036000291452")  # 13 -> -0
    assert "36000291452" in icollect._barcode_forms("036000291452")    # stripped


def test_find_items_matches_any_spelling(tmp_path):
    db = _index(tmp_path, [("0025391747488", 745195, "The Goonies (1985)")])
    # ours is the normalized 12-digit UPC-A; the site stored the 13-digit form
    assert icollect.find_items("025391747488", db_path=db) == [(745195, "The Goonies (1985)")]
    assert icollect.find_items("999999999999", db_path=db) == []


def test_find_items_without_index_is_empty(tmp_path):
    assert icollect.find_items("025391747488", db_path=tmp_path / "nope.sqlite") == []


def test_lookup_skips_junk_value_entries(tmp_path):
    """First catalog entry has a foreign-locale value -> the next one wins."""
    db = _index(tmp_path, [("025391747488", 1, "SE entry"),
                           ("025391747488", 2, "US entry")])
    vals = {1: None, 2: 6.99}
    from unittest.mock import patch as _p
    with _p.object(icollect, "fetch_value_usd", side_effect=lambda i: vals[i]):
        info = icollect.lookup("025391747488", usd_to_aud=1.0, db_path=db)
    assert info and info["value_aud"] == 6.99 and info["title"] == "US entry"


def test_value_regex_parses_the_real_page_shape():
    html = ("<dt>Automatic Estimated Value:</dt>\n<dd>~$6.99</dd>"
            "<dt>Automatic Estimated Date:</dt><dd>2026-01-29</dd>")
    assert icollect._VALUE.search(html).group(1) == "6.99"


def test_lookup_converts_and_applies_sealed_premium(tmp_path):
    db = _index(tmp_path, [("025391747488", 745195, "The Goonies (1985)")])
    with patch.object(icollect, "fetch_value_usd", return_value=6.99):
        info = icollect.lookup("025391747488", usd_to_aud=1.5, db_path=db)
    aud = 6.99 * 1.5
    assert info["value_aud"] == round(aud, 2)
    assert info["reco_used"] == round(aud * icollect.USED_MULT, 2)
    assert info["reco_new"] == round(aud * icollect.NEW_MULT, 2)
    assert info["reco_new"] > info["reco_used"]       # sealed premium is real


def test_lookup_without_fx_rate_stays_dark(tmp_path):
    db = _index(tmp_path, [("025391747488", 745195, "The Goonies")])
    assert icollect.lookup("025391747488", usd_to_aud=None, db_path=db) is None
