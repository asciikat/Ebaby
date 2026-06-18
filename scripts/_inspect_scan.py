import json
import sys

p = sys.argv[1]
d = json.load(open(p, encoding="utf-8"))
s = d["scan"]
print("=== per-field values ===")
for k in ("title", "region", "pal_ntsc", "rating", "release_year", "studio",
          "edition", "num_discs", "languages", "subtitles", "genre",
          "special_features"):
    print(f"{k:16s} {s[k]['value']!r:40s} | {s[k]['confidence']}")
print("\n=== RAW TEXT (what Qwen transcribed) ===")
for k, v in (s["raw_text"] or {}).items():
    print(f"--- {k} ---")
    print((v or "")[:700])
    print()
