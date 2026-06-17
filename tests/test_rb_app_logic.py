from redboxflip.gui import app
from redboxflip.models import Settings, ShotResult, Face


def test_apply_edits_to_settings_builds_overrides():
    s = Settings(cutout_engine="rembg")
    edits = {"engine": "sam", "extra_rotation": 90}
    over = app.settings_for_edit(s, edits)
    assert over.cutout_engine == "sam"
    # original unchanged
    assert s.cutout_engine == "rembg"
