"""End-to-end on the real photos with Qwen ON. Print listings; dump previews."""
import os
import time

import cv2
from PIL import Image

from redboxflip import pipeline
from redboxflip.models import Settings

s = Settings()
s.input_dir = "/mnt/c/Users/mardi/Documents/Ebay code/Images in"
s.output_dir = "/mnt/c/Users/mardi/Documents/Ebay code/processed"
s.qwen_extract = True
s.auto_orient = True
s.colour_tidy = True

t0 = time.time()
run_dir, groups = pipeline.run_batch(
    s, progress_cb=lambda d, t, n: print(f"  [{d}/{t}] {n}", flush=True))
print(f"\nTOTAL {time.time()-t0:.1f}s  run_dir={run_dir}\n")

prev = "/mnt/c/Users/mardi/Documents/Ebay code/scripts/_covers/out"
os.makedirs(prev, exist_ok=True)
for g in groups:
    print("=" * 60)
    listing = run_dir / pipeline.naming.safe_stem(g.title) / "ebay_listing.txt"
    print(listing.read_text(encoding="utf-8"))
    for sh in g.shots:
        if sh.output_path:
            im = cv2.imread(sh.output_path)
            small = cv2.resize(im, (360, 360))
            tag = f"{pipeline.naming.safe_stem(g.title)}_{sh.face.value}.jpg"
            cv2.imwrite(f"{prev}/{tag}", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
print("previews in", prev)
