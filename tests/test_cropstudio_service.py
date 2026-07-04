import cv2
import numpy as np
import pytest

from cropstudio import service
from cropstudio.ebay import EbayUnavailable
from cropstudio.ebay_config import EbayConfig


def _jpeg(color):
    img = np.full((40, 30, 3), color, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def test_save_dvd_names_from_front_and_writes_three_flat(tmp_path):
    shots = {0: _jpeg((10, 10, 10)), 1: _jpeg((20, 20, 20)), 2: _jpeg((30, 30, 30))}
    out = service.save_dvd(shots, tmp_path, settings=None, dvd_counter=1,
                           reader=lambda bgr, s: "Goober And The Ghost Chasers")
    assert out["title"] == "Goober And The Ghost Chasers"
    assert out["dir"] == str(tmp_path)          # no per-DVD subfolder
    assert (tmp_path / "Goober And The Ghost Chasers - Back Cover.jpg").exists()
    assert (tmp_path / "Goober And The Ghost Chasers - Front Cover.jpg").exists()
    assert (tmp_path / "Goober And The Ghost Chasers - Inside.jpg").exists()
    assert out["used_stock"] is True
    assert out["new_stock"] is False


def test_save_dvd_falls_back_when_reader_returns_none(tmp_path):
    shots = {0: _jpeg((10, 10, 10)), 1: _jpeg((20, 20, 20)), 2: _jpeg((30, 30, 30))}
    out = service.save_dvd(shots, tmp_path, settings=None, dvd_counter=2,
                           reader=lambda bgr, s: None)
    assert out["title"] == "Untitled DVD 2"
    assert (tmp_path / "Untitled DVD 2 - Front Cover.jpg").exists()


def test_save_dvd_uniquifies_duplicate_title_by_filename(tmp_path):
    shots = {0: _jpeg((1, 1, 1)), 1: _jpeg((2, 2, 2)), 2: _jpeg((3, 3, 3))}
    r = lambda bgr, s: "Heat"
    service.save_dvd(shots, tmp_path, None, 1, reader=r)
    out2 = service.save_dvd(shots, tmp_path, None, 2, reader=r)
    assert out2["title"] == "Heat (2)"
    assert (tmp_path / "Heat (2) - Back Cover.jpg").exists()
    assert (tmp_path / "Heat - Back Cover.jpg").exists()   # first one untouched


def test_partial_dvd_uses_back_when_no_front_flat(tmp_path):
    shots = {0: _jpeg((1, 1, 1))}        # only Back
    out = service.save_dvd(shots, tmp_path, None, 1, reader=lambda bgr, s: "Solo")
    assert (tmp_path / "Solo - Back Cover.jpg").exists()
    assert out["new_stock"] is True    # 1 shot counts as new/sealed (< 3), matches 2-shot rule


def test_ensure_decodable_rejects_junk():
    with pytest.raises(ValueError):
        service.ensure_decodable(b"not a jpeg")
    service.ensure_decodable(_jpeg((5, 5, 5)))   # does not raise


def test_save_dvd_surfaces_mismatch_warning(tmp_path):
    shots = {0: _jpeg((1, 1, 1)), 1: _jpeg((2, 2, 2))}
    out = service.save_dvd(
        shots, tmp_path, settings=None, dvd_counter=1,
        ebay_config=EbayConfig(client_id="id", client_secret="secret"),
        barcode_decoder=lambda bgr: ("123", "pyzbar", 0),
        ebay_client=_FakeEbayClient({"found": True, "title": "Sexy Beast"}),
        front_text_reader=lambda bgr, s: "SEXY BEAST",
    )
    assert "mismatch_warning" in out
    assert out["mismatch_warning"] is False


class _FakeEbayClient:
    def __init__(self, match):
        self._match = match

    def get_token(self, config):
        return "tok"

    def lookup_barcode(self, barcode, token, config):
        return self._match


class _UnreachableEbayClient:
    def get_token(self, config):
        raise EbayUnavailable("no network")

    def lookup_barcode(self, barcode, token, config):
        raise EbayUnavailable("no network")


def test_resolve_identity_uses_ebay_when_barcode_matches():
    shots = {0: _jpeg((1, 1, 1)), 1: _jpeg((2, 2, 2))}
    qwen_calls = []
    result = service.resolve_identity(
        shots, settings=None,
        ebay_config=EbayConfig(client_id="id", client_secret="secret"),
        barcode_decoder=lambda bgr: ("9325336022306", "pyzbar", 0),
        ebay_client=_FakeEbayClient({"found": True, "title": "Sexy Beast (DVD)"}),
        qwen_reader=lambda bgr, s: qwen_calls.append(1) or "SHOULD NOT BE USED",
        front_text_reader=lambda bgr, s: "SEXY BEAST")
    assert result["title"] == "Sexy Beast"     # titles.clean_title strips the "(DVD)" noise
    assert result["title_source"] == "ebay"
    assert result["barcode"] == "9325336022306"
    assert qwen_calls == []          # Qwen never invoked when barcode+eBay found it


def test_resolve_identity_falls_back_to_qwen_when_no_barcode():
    shots = {0: _jpeg((1, 1, 1)), 1: _jpeg((2, 2, 2))}
    result = service.resolve_identity(
        shots, settings=None, ebay_config=EbayConfig(),
        barcode_decoder=lambda bgr: (None, "", 0),
        qwen_reader=lambda bgr, s: "Amadeus")
    assert result["title"] == "Amadeus"
    assert result["title_source"] == "qwen"
    assert result["barcode"] is None


def test_resolve_identity_falls_back_to_qwen_when_no_ebay_match():
    shots = {0: _jpeg((1, 1, 1)), 1: _jpeg((2, 2, 2))}
    result = service.resolve_identity(
        shots, settings=None,
        ebay_config=EbayConfig(client_id="id", client_secret="secret"),
        barcode_decoder=lambda bgr: ("000000000000", "pyzbar", 0),
        ebay_client=_FakeEbayClient({"found": False}),
        qwen_reader=lambda bgr, s: "King Kong")
    assert result["title"] == "King Kong"
    assert result["title_source"] == "qwen"
    assert result["barcode"] == "000000000000"


def test_resolve_identity_falls_back_to_qwen_when_ebay_unreachable():
    shots = {0: _jpeg((1, 1, 1)), 1: _jpeg((2, 2, 2))}
    result = service.resolve_identity(
        shots, settings=None,
        ebay_config=EbayConfig(client_id="id", client_secret="secret"),
        barcode_decoder=lambda bgr: ("123", "pyzbar", 0),
        ebay_client=_UnreachableEbayClient(),
        qwen_reader=lambda bgr, s: "Fallback Title")
    assert result["title"] == "Fallback Title"
    assert result["title_source"] == "qwen"


def test_plausible_retail_barcode_accepts_valid_ean13_upca_ean8():
    assert service._plausible_retail_barcode("9325336022306") is True   # real EAN-13
    assert service._plausible_retail_barcode("883316276402") is True    # real UPC-A
    assert service._plausible_retail_barcode("96385074") is True        # valid EAN-8


def test_plausible_retail_barcode_rejects_junk():
    assert service._plausible_retail_barcode(None) is False
    assert service._plausible_retail_barcode("") is False
    assert service._plausible_retail_barcode("(01)89984225219249") is False  # GS1 noise read
    assert service._plausible_retail_barcode("9325336022307") is False  # bad check digit
    assert service._plausible_retail_barcode("48154254") is False       # bad EAN-8 checksum


def test_resolve_identity_ignores_implausible_barcode():
    # A spurious decoder read (zxing can hallucinate GS1 codes) must not
    # reach eBay — it should fall through to Qwen like a failed decode.
    shots = {0: _jpeg((1, 1, 1)), 1: _jpeg((2, 2, 2))}
    ebay_calls = []
    class _SpyEbay:
        def get_token(self, config):
            ebay_calls.append("token")
            return "tok"
        def lookup_barcode(self, barcode, token, config):
            ebay_calls.append("lookup")
            return {"found": True, "title": "WRONG"}
    result = service.resolve_identity(
        shots, settings=None,
        ebay_config=EbayConfig(client_id="id", client_secret="secret"),
        barcode_decoder=lambda bgr: ("(01)89984225219249", "zxingcpp", 0),
        ebay_client=_SpyEbay(),
        qwen_reader=lambda bgr, s: "Right Title")
    assert result["title"] == "Right Title"
    assert result["title_source"] == "qwen"
    assert result["barcode"] is None
    assert ebay_calls == []


def test_resolve_identity_warns_on_front_back_mismatch():
    shots = {0: _jpeg((1, 1, 1)), 1: _jpeg((2, 2, 2))}
    result = service.resolve_identity(
        shots, settings=None,
        ebay_config=EbayConfig(client_id="id", client_secret="secret"),
        barcode_decoder=lambda bgr: ("9325336022306", "pyzbar", 0),
        ebay_client=_FakeEbayClient({"found": True, "title": "Sexy Beast (DVD)"}),
        front_text_reader=lambda bgr, s: "AMERICAN DAD VOLUME 4 SEASON FOUR")
    assert result["title"] == "Sexy Beast"
    assert result["mismatch_warning"] is True


def test_resolve_identity_no_warning_when_front_text_matches():
    shots = {0: _jpeg((1, 1, 1)), 1: _jpeg((2, 2, 2))}
    result = service.resolve_identity(
        shots, settings=None,
        ebay_config=EbayConfig(client_id="id", client_secret="secret"),
        barcode_decoder=lambda bgr: ("9325336022306", "pyzbar", 0),
        ebay_client=_FakeEbayClient({"found": True, "title": "Sexy Beast (DVD)"}),
        front_text_reader=lambda bgr, s: "SEXY BEAST a film by jonathan glazer")
    assert result["mismatch_warning"] is False


def test_resolve_identity_no_warning_when_title_came_from_qwen():
    # Qwen-sourced titles have nothing to cross-check against — no guard needed.
    shots = {0: _jpeg((1, 1, 1)), 1: _jpeg((2, 2, 2))}
    result = service.resolve_identity(
        shots, settings=None, ebay_config=EbayConfig(),
        barcode_decoder=lambda bgr: (None, "", 0),
        qwen_reader=lambda bgr, s: "Amadeus",
        front_text_reader=lambda bgr, s: "totally unrelated text")
    assert result["mismatch_warning"] is False


def test_resolve_identity_no_barcode_no_qwen_reader_returns_empty(monkeypatch):
    # Force Qwen unreachable (Ollama may genuinely be live on this dev
    # machine, which would otherwise make a real hallucination-prone call).
    monkeypatch.setattr(service.vlm, "available", lambda: False)
    shots = {0: _jpeg((1, 1, 1))}
    result = service.resolve_identity(
        shots, settings=None, ebay_config=EbayConfig(),
        barcode_decoder=lambda bgr: (None, "", 0),
        qwen_reader=None)
    assert result["title"] == ""
    assert result["title_source"] == "none"
