import os
import sys
import time
import json
import base64
import csv
import shutil
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# Automatically load API Key
load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")

# --- GEMINI MODEL / ENDPOINT ---
# Google retires model names aggressively. gemini-1.5-flash / -latest AND even
# gemini-2.0-flash are now shut down and return HTTP 404 ("model not found for
# API version ...") for any key created after their cutoff — that is the 404
# this app used to hit. gemini-2.5-flash is the current low-latency, multimodal,
# JSON-capable Flash model. Both values are env-overridable so you never have to
# edit code when Google moves the goalposts again (e.g. GEMINI_MODEL=gemini-3.5-flash).
# Run `python app.py --list-models` to print exactly what YOUR key can use.
GEMINI_BASE = "https://generativelanguage.googleapis.com"
GEMINI_API_VERSION = os.environ.get("GEMINI_API_VERSION", "v1beta")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# --- CONFIGURATION ---
INPUT_DIR = Path("Images in")
OUTPUT_DIR = Path("processed")
WORK_DIR = OUTPUT_DIR / "_work"

# Ensure directories exist
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
WORK_DIR.mkdir(exist_ok=True)

# Global State for the Web App
APP_STATE = {"dvd_data": []}

# Try to load HEIC support for iPhone photos
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

# --- CORE IMAGE PROCESSING ---
def load_image(path):
    """Loads image securely, respecting EXIF rotation (fixes phone photos)."""
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def order_points(pts):
    """Orders corners: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect

def robust_crop(bgr_img):
    """Finds the DVD using Edge Detection instead of relying on A4 paper."""
    h, w = bgr_img.shape[:2]
    scale = 800.0 / max(h, w)
    small = cv2.resize(bgr_img, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100) # Edge detection ignores soft shadows

    # Close gaps in the edges to form a solid shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return bgr_img # Fallback

    largest_cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(largest_cnt) < (small.shape[0] * small.shape[1] * 0.1):
        return bgr_img # Fallback if shape is too small

    rect = cv2.minAreaRect(largest_cnt)
    box = np.int32(cv2.boxPoints(rect)) / scale
    box = order_points(box)

    (tl, tr, br, bl) = box
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0], [maxWidth-1, 0], [maxWidth-1, maxHeight-1], [0, maxHeight-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(box, dst)
    warped = cv2.warpPerspective(bgr_img, M, (maxWidth, maxHeight))

    # Add a clean 5% white padding boundary suitable for eBay
    pad = int(max(maxWidth, maxHeight) * 0.05)
    return cv2.copyMakeBorder(warped, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))

def enhance_image(img):
    """Applies a gentle contrast boost."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

# --- GEMINI AI PIPELINE ---
def _gemini_url(action, model=None):
    """Build a Generative Language REST URL for `action` (e.g. 'generateContent').

    The API key is sent in the 'x-goog-api-key' header, NOT the query string, so
    the URL contains no secret and is safe to print in logs.
    """
    return f"{GEMINI_BASE}/{GEMINI_API_VERSION}/models/{model or GEMINI_MODEL}:{action}"

def list_available_models(key, action="generateContent"):
    """Ask the API which models THIS key can actually call for `action`.

    Turns an opaque 404 into an actionable list. Returns the bare model codes
    (e.g. 'gemini-2.5-flash'), or [] if the call fails.
    """
    key = (key or "").strip()
    if not key:
        return []
    try:
        resp = requests.get(
            f"{GEMINI_BASE}/{GEMINI_API_VERSION}/models",
            headers={"x-goog-api-key": key},
            timeout=30,
        )
        resp.raise_for_status()
        models = resp.json().get("models", [])
    except Exception as e:
        print(f"      [!] Could not list models for this key: {e}")
        return []
    usable = []
    for m in models:
        if action in m.get("supportedGenerationMethods", []):
            # 'models/gemini-2.5-flash' -> 'gemini-2.5-flash'
            usable.append(m.get("name", "").split("/", 1)[-1])
    return usable

def _empty_meta():
    """The neutral metadata record used whenever AI analysis is unavailable."""
    return {"side": "other", "rotation_cw": 0, "barcode": "", "title": "Unknown DVD",
            "description": "", "genre": "", "studio": "", "year": 0}

def analyze_with_gemini(bgr_img):
    """Sends the cropped image to Gemini for analysis, OCR, and rotation detection."""
    clean_key = (API_KEY or "").strip()
    if not clean_key:
        print("      [!] GEMINI_API_KEY is not set; skipping AI analysis.")
        return _empty_meta()

    h, w = bgr_img.shape[:2]
    if max(h, w) > 1200:
        scale = 1200.0 / max(h, w)
        bgr_img = cv2.resize(bgr_img, (int(w * scale), int(h * scale)))

    _, buf = cv2.imencode(".jpg", bgr_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64_img = base64.b64encode(buf.tobytes()).decode("ascii")

    prompt = """Analyze this photo of a DVD cover. Return ONLY a valid JSON object with these exact keys:
    {
        "side": "string: exactly one of 'front', 'back', 'center', or 'other'",
        "rotation_cw": <integer: degrees to rotate clockwise to make the text perfectly upright. Must be 0, 90, 180, or 270>,
        "barcode": "string: read the UPC barcode digits if visible, otherwise empty string",
        "title": "string: A concise eBay listing title ending in 'DVD'",
        "description": "string: Short description of the item",
        "genre": "string",
        "studio": "string",
        "year": <integer: release year or 0>
    }"""

    url = _gemini_url("generateContent")
    headers = {"x-goog-api-key": clean_key}
    payload = {
        "contents": [{"parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/jpeg", "data": b64_img}}]}],
        "generationConfig": {"responseMimeType": "application/json"}
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)

        # Deep Diagnostic Logger (URL carries no key, so it is safe to print)
        if not resp.ok:
            print(f"\n      [!] Google API Rejected the Request!")
            print(f"      [!] URL: {url}")
            print(f"      [!] Model: {GEMINI_MODEL}  |  API version: {GEMINI_API_VERSION}")
            print(f"      [!] Status Code: {resp.status_code}")
            print(f"      [!] Error Details: {resp.text}\n")
            if resp.status_code == 404:
                usable = list_available_models(clean_key)
                if usable:
                    print(f"      [i] Models your key CAN use for generateContent:")
                    for name in usable:
                        print(f"            - {name}")
                    print(f"      [i] Set GEMINI_MODEL in your .env to one of the above.\n")

        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"      [!] Pipeline Error: {e}")
        return _empty_meta()

def process_all_images():
    """Reads folder, groups by time, crops, asks Gemini, and straightens."""
    files = sorted(INPUT_DIR.glob("*.*"), key=lambda f: f.stat().st_mtime)
    valid_files = [f for f in files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.heic']]

    if not valid_files:
        return []

    # Group photos taken within 30 seconds of each other
    groups, current_group, last_time = [], [], 0
    for f in valid_files:
        mtime = f.stat().st_mtime
        if not current_group or (mtime - last_time) <= 30.0:
            current_group.append(f)
        else:
            groups.append(current_group)
            current_group = [f]
        last_time = mtime
    if current_group: groups.append(current_group)

    processed_data = []
    for i, group_files in enumerate(groups):
        print(f"\nProcessing DVD #{i+1} ({len(group_files)} photos)...")
        dvd_data = {"id": i, "photos": [], "title": "Unknown DVD", "barcode": "", "description": "", "genre": "", "year": 0, "studio": ""}

        for file in group_files:
            print(f"  -> Cropping & Enhancing: {file.name}")
            img = load_image(file)
            cropped = robust_crop(img)

            print(f"  -> Gemini Analysis...")
            meta = analyze_with_gemini(cropped)

            # Apply Gemini's intelligent orientation fix
            rot = meta.get("rotation_cw", 0)
            if rot == 90: cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)
            elif rot == 180: cropped = cv2.rotate(cropped, cv2.ROTATE_180)
            elif rot == 270: cropped = cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Final Contrast Pop
            final_img = enhance_image(cropped)

            out_name = f"{file.stem}_work.jpg"
            cv2.imwrite(str(WORK_DIR / out_name), final_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

            dvd_data["photos"].append({"filename": out_name, "side": meta.get("side", "other"), "original": file.name})

            # Merge Best Metadata
            if meta.get("barcode"): dvd_data["barcode"] = meta["barcode"]
            if meta.get("title") and meta.get("title") != "Unknown DVD": dvd_data["title"] = meta["title"]
            if meta.get("description"): dvd_data["description"] = meta["description"]
            if meta.get("genre"): dvd_data["genre"] = meta["genre"]
            if meta.get("year"): dvd_data["year"] = meta["year"]
            if meta.get("studio"): dvd_data["studio"] = meta["studio"]

        processed_data.append(dvd_data)
    return processed_data

# --- FASTAPI WEB SERVER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not (API_KEY or "").strip():
        print("\n[CRITICAL ERROR]: GEMINI_API_KEY is missing from your .env file!")
        os._exit(1)
    print("\n========================================")
    print(" 🤖 DVD Auto-Pipeline Starting...")
    print(f"    Model: {GEMINI_MODEL}  ({GEMINI_API_VERSION})")
    print("========================================")
    APP_STATE["dvd_data"] = process_all_images()
    print("\n========================================")
    print(" ✅ Processing Complete. Starting Web UI...")
    print(" 🌐 Open http://127.0.0.1:8000 in your browser")
    print("========================================\n")
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/work", StaticFiles(directory=WORK_DIR, check_dir=False), name="work")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>eBay Flip Review</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 p-8">
    <div class="max-w-6xl mx-auto">
        <div class="flex justify-between items-center mb-8">
            <h1 class="text-3xl font-bold text-gray-800">Review eBay Listings</h1>
            <button onclick="exportData()" class="bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded-lg font-bold shadow-lg transition">Export CSV & Save Photos</button>
        </div>
        <div id="app"></div>
    </div>

    <script>
        const data = {{ data_json | safe }};

        function render() {
            const app = document.getElementById('app');
            if (data.length === 0) {
                app.innerHTML = '<div class="bg-white p-8 rounded-lg text-center text-gray-500">No images processed. Drop photos in "Images in" folder and restart script.</div>';
                return;
            }
            app.innerHTML = data.map((dvd, i) => `
                <div class="bg-white p-6 rounded-lg shadow-md border border-gray-200 mb-6 flex gap-8">
                    <div class="w-1/2 space-y-4">
                        <div>
                            <label class="block text-xs font-bold text-gray-500 uppercase tracking-wide">Listing Title</label>
                            <input class="w-full border-b border-gray-300 py-2 focus:outline-none focus:border-blue-500 font-bold text-lg" value="${dvd.title}" onchange="data[${i}].title=this.value">
                        </div>
                        <div class="grid grid-cols-2 gap-4">
                            <div>
                                <label class="block text-xs font-bold text-gray-500 uppercase tracking-wide">Barcode</label>
                                <input class="w-full border rounded p-2 mt-1 bg-gray-50" value="${dvd.barcode || ''}" onchange="data[${i}].barcode=this.value">
                            </div>
                            <div>
                                <label class="block text-xs font-bold text-gray-500 uppercase tracking-wide">Year</label>
                                <input class="w-full border rounded p-2 mt-1 bg-gray-50" value="${dvd.year || ''}" onchange="data[${i}].year=this.value">
                            </div>
                            <div>
                                <label class="block text-xs font-bold text-gray-500 uppercase tracking-wide">Genre</label>
                                <input class="w-full border rounded p-2 mt-1 bg-gray-50" value="${dvd.genre || ''}" onchange="data[${i}].genre=this.value">
                            </div>
                            <div>
                                <label class="block text-xs font-bold text-gray-500 uppercase tracking-wide">Studio</label>
                                <input class="w-full border rounded p-2 mt-1 bg-gray-50" value="${dvd.studio || ''}" onchange="data[${i}].studio=this.value">
                            </div>
                        </div>
                        <div>
                            <label class="block text-xs font-bold text-gray-500 uppercase tracking-wide mb-1">Description</label>
                            <textarea class="w-full border rounded p-2 h-24 text-gray-700 bg-gray-50" onchange="data[${i}].description=this.value">${dvd.description || ''}</textarea>
                        </div>
                    </div>
                    <div class="w-1/2 grid grid-cols-2 gap-4 bg-gray-50 p-4 rounded-lg border">
                        ${dvd.photos.map(p => `
                            <div class="flex flex-col items-center">
                                <img src="/work/${p.filename}" class="w-full h-48 object-contain bg-white border shadow-sm rounded">
                                <span class="mt-2 text-xs font-bold px-2 py-1 bg-blue-100 text-blue-800 rounded uppercase tracking-wider">${p.side}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `).join('');
        }

        async function exportData() {
            const res = await fetch('/export', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            });
            if(res.ok) {
                alert('Export Successful! CSV and finalized Images are in the /processed/ folder.');
            } else {
                alert('Export Failed. Check console.');
            }
        }
        render();
    </script>
</body>
</html>
"""

@app.get("/")
def review():
    return HTMLResponse(HTML_TEMPLATE.replace("{{ data_json | safe }}", json.dumps(APP_STATE["dvd_data"])))

@app.post("/export")
async def export_data(request: Request):
    data = await request.json()
    csv_path = OUTPUT_DIR / "batch_listings.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Title", "Barcode", "Genre", "Year", "Studio", "Description", "Photos"])

        for dvd in data:
            slug = "".join(c if c.isalnum() else "_" for c in dvd["title"])
            dvd_folder = OUTPUT_DIR / slug
            dvd_folder.mkdir(exist_ok=True)

            photo_names = []
            for p in dvd["photos"]:
                src = WORK_DIR / p["filename"]
                dst = dvd_folder / p["filename"]
                if src.exists():
                    shutil.copy(src, dst)
                    photo_names.append(p["filename"])

            writer.writerow([dvd["title"], dvd.get("barcode", ""), dvd.get("genre", ""), dvd.get("year", ""), dvd.get("studio", ""), dvd.get("description", ""), ";".join(photo_names)])

    return {"status": "success"}

def _print_model_diagnostics():
    """`python app.py --list-models`: show what the current key can call."""
    key = (API_KEY or "").strip()
    if not key:
        print("[CRITICAL] GEMINI_API_KEY is missing from your .env file.")
        return 1
    print(f"Querying {GEMINI_BASE}/{GEMINI_API_VERSION}/models for this key ...")
    usable = list_available_models(key)
    if not usable:
        print("No models returned (check the key, its restrictions, billing, or your network).")
        return 1
    print("Models available to this key for generateContent:")
    for name in usable:
        marker = "   <-- current GEMINI_MODEL" if name == GEMINI_MODEL else ""
        print(f"  - {name}{marker}")
    if GEMINI_MODEL not in usable:
        print(f"\n[!] Your configured GEMINI_MODEL '{GEMINI_MODEL}' is NOT in the list above.")
        print("    Set GEMINI_MODEL in your .env to one of the models listed.")
    return 0

if __name__ == "__main__":
    if "--list-models" in sys.argv:
        sys.exit(_print_model_diagnostics())
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
