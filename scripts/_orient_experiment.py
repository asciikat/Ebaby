"""Find a RELIABLE upside-down detector. Test OSD (several preprocs) and Qwen
on the high-res deskewed cover for all 6 real faces. Ground truth known by eye."""
import base64
import json
import subprocess
import time

import cv2
import numpy as np
import pytesseract

from redboxflip import scan
from redboxflip.models import Face, Settings

# input path -> (face, human-known correct? we just record OSD/Qwen verdicts)
INPUTS = {
    "IMG_20260617_224148.jpg": ("Amadeus", Face.BACK),
    "IMG_20260617_224135.jpg": ("Amadeus", Face.FRONT),
    "IMG_20260617_224208.jpg": ("Amadeus", Face.INSIDE),
    "IMG_20260617_224014.jpg": ("Gale", Face.BACK),
    "1781700457950.jpg":       ("Gale", Face.FRONT),
    "IMG_20260617_224044.jpg": ("Gale", Face.INSIDE),
}
IN = "/mnt/c/Users/mardi/Documents/Ebay code/Images in"
S = Settings()


def osd_verdict(bgr, label):
    """Try OSD a few ways; return rotation int or None."""
    attempts = []
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    attempts.append(("raw", bgr))
    # upscale 1.5x
    attempts.append(("up", cv2.resize(bgr, None, fx=1.5, fy=1.5)))
    # otsu threshold (text crisp)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    attempts.append(("otsu", cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)))
    # inverted otsu (white-on-dark covers)
    attempts.append(("otsu_inv", cv2.cvtColor(255 - th, cv2.COLOR_GRAY2BGR)))
    out = []
    for nm, im in attempts:
        try:
            o = pytesseract.image_to_osd(im, output_type=pytesseract.Output.DICT)
            out.append(f"{nm}:rot{o['rotate']}@{o['orientation_conf']:.1f}")
        except Exception as e:
            msg = str(e).split('.')[0][:30]
            out.append(f"{nm}:FAIL")
    return " ".join(out)


def qwen_orient(bgr):
    h, w = bgr.shape[:2]
    s = 768.0 / max(h, w)
    if s < 1.0:
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)))
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    prompt = ("This is a photo of one face of a DVD case. Read the largest title "
              "or heading text. Decide if the image is upright or upside-down. "
              "Answer upright=true only if the main text reads normally left-to-right.")
    fmt = {"type": "object",
           "properties": {"upright": {"type": "boolean"},
                          "largest_text_read": {"type": "string"}},
           "required": ["upright", "largest_text_read"]}
    body = {"model": "qwen3.5:4b", "prompt": prompt,
            "images": [base64.b64encode(buf).decode()],
            "stream": False, "think": False, "keep_alive": "5m",
            "options": {"num_predict": 80, "temperature": 0}, "format": fmt}
    r = subprocess.run(["curl.exe", "-s", "-m", "180",
                        "http://127.0.0.1:11434/api/generate",
                        "-H", "Content-Type: application/json", "-d", "@-"],
                       input=json.dumps(body), capture_output=True, text=True)
    try:
        return json.loads(r.stdout).get("response", "")
    except Exception:
        return "FAIL"


for fname, (dvd, face) in INPUTS.items():
    bgr = cv2.imread(f"{IN}/{fname}")
    sc = scan.scan_case(bgr, S.rembg_model)
    if sc is None:
        print(f"{dvd:8s} {face.value:7s} SCAN FAILED")
        continue
    hi = scan.orient_and_tighten(sc[0], face)
    cover = np.ascontiguousarray(hi[:, :, :3])
    print(f"\n## {dvd} {face.value}  cover={cover.shape[1]}x{cover.shape[0]}")
    print("   OSD :", osd_verdict(cover, face.value))
    print("   QWEN:", qwen_orient(cover).strip())
