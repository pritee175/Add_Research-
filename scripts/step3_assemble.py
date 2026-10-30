# -*- coding: utf-8 -*-
"""Step 3: merge cached ASR/OCR into annotations/speech.csv and screen_text.csv (status = automatic)."""
import csv, json, os, re
ROOT = r"E:\Research_FF"
ANN = os.path.join(ROOT, "ad_dataset", "annotations")
CACHE = os.path.join(ROOT, ".scratch", "asr_ocr_cache")
WARN_RE = re.compile(r"warning|injurious|health|cancer|kills|quit|1800|tobacco|तंबाकू|तम्बाकू|चेतावनी|स्वास्थ्य|कैंसर|हानिकारक", re.I)
ids = [r["ad_id"] for r in csv.DictReader(open(os.path.join(ANN, "ads.csv"), encoding="utf-8"))]
sp, st = [], []
for a in ids:
    f = os.path.join(CACHE, a + ".asr_sarvam.json")  # Sarvam preferred; whisper cache is the fallback
    if not os.path.exists(f):
        f = os.path.join(CACHE, a + ".asr.json")
    if os.path.exists(f):
        d = json.load(open(f, encoding="utf-8"))
        lang = (d["language"] or "").split("-")[0]
        sp += [[a, s["start"], s["end"], s["text"], lang, "automatic"] for s in d["speech"]]
    f = os.path.join(CACHE, a + ".ocr.json")
    if os.path.exists(f):
        for o in json.load(open(f, encoding="utf-8"))["ocr"]:
            t = "warning" if WARN_RE.search(o["text"]) else "other"
            l = "Hindi" if re.search(r"[\u0900-\u097F]", o["text"]) else "English"
            st.append([a, o["start"], o["end"], o["text"], t, l, "automatic"])
for name, hdr, rows in [("speech.csv", ["ad_id","start_second","end_second","original_text","language","transcription_status"], sp),
                        ("screen_text.csv", ["ad_id","start_second","end_second","original_text","text_type","language","correction_status"], st)]:
    with open(os.path.join(ANN, name), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
print(len(sp), "speech rows;", len(st), "screen_text rows")
