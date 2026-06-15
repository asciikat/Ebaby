from dvdflip import pipeline, vision
from dvdflip.models import ClassifyResult

def test_process_image_front(monkeypatch, tmp_path, sample_front):
    monkeypatch.setattr(
        vision, "classify_photo",
        lambda bgr, size_mm, barcode: ClassifyResult(
            side="front", rotation_cw=0, title="King Kong Escapes",
            year=1967, confidence=0.95))
    photo = pipeline.process_image(sample_front, tmp_path)
    assert photo.side == "front"
    assert photo.a4_found is True
    assert photo.work_image is not None and photo.work_image.exists()
    assert photo.title == "King Kong Escapes"

def test_process_image_back_reads_barcode(monkeypatch, tmp_path, sample_back):
    monkeypatch.setattr(
        vision, "classify_photo",
        lambda bgr, size_mm, barcode: ClassifyResult(
            side="back", rotation_cw=0, title="King Kong Escapes",
            year=1967, confidence=0.95))
    photo = pipeline.process_image(sample_back, tmp_path)
    assert photo.barcode == "0025192828928"
