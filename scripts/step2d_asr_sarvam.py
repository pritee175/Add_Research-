# -*- coding: utf-8 -*-
"""Step 2 (cloud): Sarvam AI speech-to-text (saaras:v4) on ad_dataset/audio/*.wav.
Uploads AUDIO ONLY (16 kHz mono wav chunks, never the videos). Key is read from .secrets/speech_key.txt
and is never printed or written anywhere else.
Long ads are split into CHUNK_S-second pieces (REST API is meant for <30 s), timestamps shifted back to ad time.
Output: .scratch/asr_ocr_cache/<ad>.asr_sarvam.json   (resumable)
usage: step2d_asr_sarvam.py [ad_id ...]   (run with .venv_media python; needs `requests`)"""
import glob, json, os, subprocess, sys, tempfile, time
import requests

ROOT = r"E:\Research_FF"
CACHE = os.path.join(ROOT, ".scratch", "asr_ocr_cache")
FFMPEG = os.path.join(ROOT, ".tmp", "bin", "ffmpeg.exe")
URL = "https://api.sarvam.ai/speech-to-text"
CHUNK_S = 25
import re
_raw = open(os.path.join(ROOT, ".secrets", "speech_key.txt"), encoding="utf-8").read()
_m = re.search(r"sk_[A-Za-z0-9_\-]+", _raw)  # file may contain extra pasted text; use only the key
KEY = _m.group(0) if _m else _raw.strip()
only = sys.argv[1:]
tmp = os.path.join(ROOT, ".tmp", "sarvam_chunks")
os.makedirs(tmp, exist_ok=True)


def transcribe_chunk(path):
    for attempt in range(4):
        with open(path, "rb") as f:
            r = requests.post(URL, headers={"api-subscription-key": KEY},
                              files={"file": (os.path.basename(path), f, "audio/wav")},
                              data={"model": "saaras:v4", "language_code": "unknown", "with_timestamps": "true"},
                              timeout=120)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503):
            time.sleep(5 * (attempt + 1))
            continue
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    raise RuntimeError("retries exhausted")


for wav in sorted(glob.glob(os.path.join(ROOT, "ad_dataset", "audio", "*.wav"))):
    ad = os.path.basename(wav)[:-4]
    if only and ad not in only:
        continue
    out = os.path.join(CACHE, f"{ad}.asr_sarvam.json")
    if os.path.exists(out):
        continue
    for old in glob.glob(os.path.join(tmp, ad + "_*.wav")):
        os.remove(old)
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", wav, "-f", "segment", "-segment_time", str(CHUNK_S),
                    "-c", "copy", os.path.join(tmp, ad + "_%03d.wav")], check=True)
    segs, langs = [], []
    try:
        for i, ch in enumerate(sorted(glob.glob(os.path.join(tmp, ad + "_*.wav")))):
            off = i * CHUNK_S
            j = transcribe_chunk(ch)
            if j.get("language_code"):
                langs.append(j["language_code"])
            ts = j.get("timestamps") or {}
            words, st, en = ts.get("words") or [], ts.get("start_time_seconds") or [], ts.get("end_time_seconds") or []
            if words:
                for w, a, b in zip(words, st, en):
                    if w.strip():
                        segs.append({"start": round(off + a, 1), "end": round(off + b, 1), "text": w.strip()})
            elif (j.get("transcript") or "").strip():
                segs.append({"start": float(off), "end": float(off + CHUNK_S), "text": j["transcript"].strip()})
    except Exception as e:
        print("FAILED", ad, str(e)[:200], flush=True)
        continue
    finally:
        for old in glob.glob(os.path.join(tmp, ad + "_*.wav")):
            os.remove(old)
    lang = max(set(langs), key=langs.count) if langs else None
    json.dump({"language": lang, "model": "sarvam-saaras:v4", "speech": segs},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(ad, lang, len(segs), flush=True)
