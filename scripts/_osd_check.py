"""Throwaway: check Tesseract OSD verdict on the flipped Amadeus back cover."""
import cv2
import pytesseract

base = "processed/_review_165926/run_20260618_165926/Amadeus Director's Cut"
for name in ("Back Cover", "Front Cover", "Inside"):
    p = f"{base}/Amadeus Director's Cut - {name}.jpg"
    img = cv2.imread(p)
    if img is None:
        print(name, "-> could not load", p)
        continue
    small = cv2.resize(img, (int(img.shape[1] * 0.6), int(img.shape[0] * 0.6)))
    print("====", name, img.shape, "====")
    try:
        print(pytesseract.image_to_osd(small))
    except Exception as e:
        print("OSD failed:", e)
