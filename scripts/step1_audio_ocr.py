# -*- coding: utf-8 -*-
"""Step 1: extract audio and run EasyOCR for every downloaded ad.
Resumable: per-ad results cached in .scratch/asr_ocr_cache/<ad_id>.json.
Writes ad_dataset/annotations/speech.csv and screen_text.csv (status = automatic).
Run with .venv_media\\Scripts\\python.exe"""
import csv, json, os, re, subprocess, sys

import cv2
import easyocr
import imageio_ffmpeg

ROOT = r"E:\Research_FF"
DS = os.path.join(ROOT, "ad_dataset")
ANN = os.path.join(DS, "annotations")
CACHE = os.path.join(ROOT, ".scratch", "asr_ocr_cache")
MODEL_CACHE = os.path.join(ROOT, ".model_cache")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
OCR_STEP = 2.0
WARN_RE = re.compile(r"warning|injurious|health|cancer|kills|quit|1800|tobacco|तंबाकू|तम्बाकू|चेतावनी|स्वास्थ्य|कैंसर|हानिकारक", re.I)

os.makedirs(CACHE, exist_ok=True)
os.makedirs(os.path.join(DS, "audio"), exist_ok=True)

ads = list(csv.DictReader(open(os.path.join(ANN, "ads.csv"), encoding="utf-8")))
todo = [r["ad_id"] for r in ads if r["video_status"] == "downloaded"]
if len(sys.argv) > 1:
    todo = sys.argv[1:]

print("loading models", flush=True)
reader = easyocr.Reader(["en", "hi"], gpu=False,
                        model_storage_directory=os.path.join(MODEL_CACHE, "easyocr"),
                        user_network_directory=os.path.join(MODEL_CACHE, "easyocr"))


def norm(t):
    return re.sub(r"\W+", "", t.lower())


def process(ad_id):
    out = os.path.join(CACHE, ad_id + ".ocr.json")
    if os.path.exists(out):
        return
    video = os.path.join(DS, "videos", ad_id + ".mp4")
    wav = os.path.join(DS, "audio", ad_id + ".wav")
    subprocess.run([FFMPEG, "-y", "-i", video, "-ar", "16000", "-ac", "1", wav],
                   check=True, capture_output=True)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    obs = []  # (t, [(text, conf)])
    t = 0.0
    while t < dur:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if w > 960:
            frame = cv2.resize(frame, (960, int(h * 960 / w)))
        res = reader.readtext(frame, detail=1, paragraph=False)
        obs.append((round(t, 1), [(x[1].strip(), float(x[2])) for x in res if x[2] > 0.4 and len(x[1].strip()) > 1]))
        t += OCR_STEP
    cap.release()

    # merge identical text across consecutive samples into segments
    open_ = {}
    ocr = []
    for t, items in obs:
        keys = {norm(x): x for x, _ in items if norm(x)}
        for k in list(open_):
            if k not in keys:
                s = open_.pop(k)
                ocr.append({"start": s[0], "end": round(s[1] + OCR_STEP, 1), "text": s[2]})
        for k, txt in keys.items():
            if k in open_:
                open_[k][1] = t
            else:
                open_[k] = [t, t, txt]
    for k, s in open_.items():
        ocr.append({"start": s[0], "end": round(min(s[1] + OCR_STEP, dur), 1), "text": s[2]})
    ocr.sort(key=lambda x: (x["start"], x["text"]))

    json.dump({"duration": round(dur, 1), "ocr": ocr},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(ad_id, len(ocr), "ocr", flush=True)


for a in todo:
    try:
        process(a)
    except Exception as e:
        print("FAILED", a, e, flush=True)

