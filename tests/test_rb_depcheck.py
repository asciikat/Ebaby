from redboxflip import __main__ as cli


def test_startup_report_lists_decoders_and_engines():
    report = cli.startup_report()
    assert "Barcode decoders:" in report
    assert "Cutout:" in report
