from redboxflip.gui import grid
from redboxflip.models import ShotResult, Face, DvdGroup


def test_badge_text_reflects_status():
    g = DvdGroup(index=1, barcode="123", title="X", region="R", shots=[
        ShotResult(input_path="b.jpg", face=Face.BACK, barcode="123",
                   title="X", status="ok"),
    ])
    assert grid.badge_for(g) == "✓ title + barcode"

    g2 = DvdGroup(index=2, barcode=None, title="Untitled DVD 2", region="R",
                  shots=[ShotResult(input_path="b.jpg", face=Face.BACK)])
    assert grid.badge_for(g2) == "⚠ needs review"
