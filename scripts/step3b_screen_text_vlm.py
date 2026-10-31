# -*- coding: utf-8 -*-
"""Step 3b: build annotations/screen_text.csv from VLM-read keyframes (.tmp/vlm_out/<ad_id>.json). status = automatic."""
import csv, json, os, re
ROOT = r"E:\Research_FF"
ANN = os.path.join(ROOT, "ad_dataset", "annotations")
SRC = os.path.join(ROOT, ".tmp", "vlm_out")
TYPES = {"brand", "warning", "helpline", "disclaimer", "tagline", "caption", "credit", "other"}
WARN_RE = re.compile(r"statutory warning|warning|injurious|हानिकारक|चेतावनी", re.I)
NOT_AD_RE = re.compile(r"subscribe|notout90s|wildfilms", re.I)  # uploader/stock-footage overlays, not ad text

ids = [r["ad_id"] for r in csv.DictReader(open(os.path.join(ANN, "ads.csv"), encoding="utf-8"))]
rows = []
for a in ids:
    f = os.path.join(SRC, a + ".json")
    if not os.path.exists(f):
        continue
    for o in json.load(open(f, encoding="utf-8")):
        t = re.sub(r"[ \t]*\n[ \t]*", " / ", o["text"]).strip()
        if not t or NOT_AD_RE.search(t):
            continue
        ty = o["text_type"] if o["text_type"] in TYPES else "other"
        if ty != "helpline" and WARN_RE.search(t):
            ty = "warning"
        lang = o["language"] if o["language"] in ("Hindi", "English", "Mixed") else "English"
        rows.append([a, o["start"], max(o["end"], o["start"]), t, ty, lang, "automatic"])
rows.sort(key=lambda r: (r[0], r[1], r[2]))
with open(os.path.join(ANN, "screen_text.csv"), "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["ad_id", "start_second", "end_second", "original_text", "text_type", "language", "correction_status"])
    w.writerows(rows)
print(len(rows), "screen_text rows from", len({r[0] for r in rows}), "ads")
