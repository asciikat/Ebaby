from redboxflip.models import Face, FACE_ORDER, FACE_FILE_LABEL, Settings, ShotResult, DvdGroup


def test_face_order_is_back_front_inside():
    assert FACE_ORDER == [Face.BACK, Face.FRONT, Face.INSIDE]


def test_face_file_labels():
    assert FACE_FILE_LABEL[Face.BACK] == "Back Cover"
    assert FACE_FILE_LABEL[Face.FRONT] == "Front Cover"
    assert FACE_FILE_LABEL[Face.INSIDE] == "Inside"


def test_settings_defaults():
    s = Settings()
    assert s.cutout_engine == "geometric"
    assert s.default_region == "Region 4 (PAL, Australia)"
    assert s.colour_tidy is True
    assert s.title_lookup is True
    assert s.feather_px == 3
    assert s.margin_pct == 6.0
    assert s.jpeg_quality == 92
    assert s.max_edge_px == 1600


def test_shotresult_and_group_roundtrip_to_dict():
    r = ShotResult(input_path="a.jpg", face=Face.BACK, barcode="123")
    d = r.to_dict()
    assert d["face"] == "back" and d["barcode"] == "123"
    g = DvdGroup(index=1, barcode="123", title="The Matrix",
                 region="Region 4 (PAL, Australia)", shots=[r])
    gd = g.to_dict()
    assert gd["title"] == "The Matrix" and gd["shots"][0]["face"] == "back"
