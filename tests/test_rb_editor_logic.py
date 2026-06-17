from redboxflip.gui import editor


def test_scale_corners_roundtrip():
    # canvas-space <-> image-space corner mapping must round-trip
    corners_img = [(100, 50), (400, 50), (400, 300), (100, 300)]
    scale = 0.5
    canvas = editor.image_to_canvas(corners_img, scale)
    assert canvas[0] == (50, 25)
    back = editor.canvas_to_image(canvas, scale)
    assert back[2] == (400, 300)
