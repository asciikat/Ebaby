from cropstudio import manifest


def test_append_creates_and_grows_manifest(tmp_path):
    manifest.append(tmp_path, {"title": "Amadeus"})
    manifest.append(tmp_path, {"title": "King Kong"})
    entries = manifest.load(tmp_path)
    assert len(entries) == 2
    assert entries[0]["title"] == "Amadeus"
    assert entries[1]["title"] == "King Kong"


def test_load_missing_manifest_returns_empty_list(tmp_path):
    assert manifest.load(tmp_path) == []


def test_manifest_path_lives_in_run_dir(tmp_path):
    assert manifest.manifest_path(tmp_path) == tmp_path / "run_manifest.json"
