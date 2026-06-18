"""Throwaway probe: can qwen3.5:4b (a) detect orientation and (b) extract
eBay fields as JSON from the real Amadeus covers? Uses curl.exe (WSL->Win)."""
import base64
import json
import subprocess
import time

import cv2

OLLAMA = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen3.5:4b"
BASE = "processed/_review_165926/run_20260618_165926/Amadeus Director's Cut"


def call(prompt, bgr, fmt=None, num_predict=400):
    h, w = bgr.shape[:2]
    s = 768.0 / max(h, w)
    if s < 1.0:
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)))
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    body = {
        "model": MODEL, "prompt": prompt,
        "images": [base64.b64encode(buf).decode()],
        "stream": False, "think": False, "keep_alive": "5m",
        "options": {"num_predict": num_predict, "temperature": 0},
    }
    if fmt is not None:
        body["format"] = fmt
    t0 = time.time()
    r = subprocess.run(["curl.exe", "-s", "-m", "180", OLLAMA,
                        "-H", "Content-Type: application/json", "-d", "@-"],
                       input=json.dumps(body), capture_output=True, text=True)
    dt = time.time() - t0
    try:
        resp = json.loads(r.stdout).get("response", "")
    except Exception:
        resp = "PARSE_FAIL: " + r.stdout[:300]
    return resp, dt


# --- Test 1: orientation on each face ---
ori_prompt = ("Look at this photo of a DVD case. Is the text and artwork "
              "right-side up? Reply with ONE number: the clockwise rotation in "
              "degrees needed to make it upright. Choose 0, 90, 180, or 270. "
              "Number only.")
ori_fmt = {"type": "object",
           "properties": {"rotation": {"type": "integer", "enum": [0, 90, 180, 270]}},
           "required": ["rotation"]}
print("===== ORIENTATION =====")
for name in ("Back Cover", "Front Cover", "Inside"):
    img = cv2.imread(f"{BASE}/Amadeus Director's Cut - {name}.jpg")
    resp, dt = call(ori_prompt, img, fmt=ori_fmt, num_predict=40)
    print(f"{name:12s} {dt:5.1f}s  {resp.strip()}")

# --- Test 2: field extraction from the back cover ---
extract_prompt = (
    "You are reading photos of a DVD case to build an eBay listing. Read ONLY "
    "text that is actually printed and legible in the image. Do NOT guess. For "
    "any field you cannot clearly read, use \"Unknown\". Return the data."
)
extract_fmt = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "rating": {"type": "string"},
        "region": {"type": "string"},
        "studio": {"type": "string"},
        "release_year": {"type": "string"},
        "num_discs": {"type": "string"},
        "runtime": {"type": "string"},
        "barcode": {"type": "string"},
        "edition": {"type": "string"},
        "languages": {"type": "string"},
        "subtitles": {"type": "string"},
    },
    "required": ["title", "rating", "region", "studio", "barcode"],
}
print("\n===== BACK COVER EXTRACTION =====")
img = cv2.imread(f"{BASE}/Amadeus Director's Cut - Back Cover.jpg")
resp, dt = call(extract_prompt, img, fmt=extract_fmt, num_predict=500)
print(f"{dt:.1f}s")
try:
    print(json.dumps(json.loads(resp), indent=2))
except Exception:
    print(resp)
