from cropstudio.ebay_config import load_ebay_config


def test_load_ebay_config_reads_env_file(tmp_path):
    (tmp_path / ".env").write_text(
        "EBAY_CLIENT_ID=abc123\nEBAY_CLIENT_SECRET=secret456\n", encoding="utf-8")
    cfg = load_ebay_config(project_root=tmp_path)
    assert cfg.client_id == "abc123"
    assert cfg.client_secret == "secret456"
    assert cfg.marketplace_id == "EBAY_AU"     # default when not set
    assert cfg.configured is True


def test_load_ebay_config_honors_custom_marketplace(tmp_path):
    (tmp_path / ".env").write_text(
        "EBAY_CLIENT_ID=abc\nEBAY_CLIENT_SECRET=def\nEBAY_MARKETPLACE_ID=EBAY_GB\n",
        encoding="utf-8")
    cfg = load_ebay_config(project_root=tmp_path)
    assert cfg.marketplace_id == "EBAY_GB"


def test_load_ebay_config_missing_file_is_unconfigured(tmp_path):
    cfg = load_ebay_config(project_root=tmp_path)
    assert cfg.client_id == ""
    assert cfg.configured is False


def test_load_ebay_config_ignores_comments_and_blank_lines(tmp_path):
    (tmp_path / ".env").write_text(
        "# eBay creds\n\nEBAY_CLIENT_ID=abc\nEBAY_CLIENT_SECRET=def\n", encoding="utf-8")
    cfg = load_ebay_config(project_root=tmp_path)
    assert cfg.configured is True
