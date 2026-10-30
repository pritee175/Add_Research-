# -*- coding: utf-8 -*-
"""Step 2 (alt): openai-whisper using models cached in .model_cache/whisper. Run with .venv_media python.
usage: step2b_asr_openai.py [small|base]"""
import glob, json, os, sys
import whisper
ROOT = r"E:\Research_FF"
CACHE = os.path.join(ROOT, ".scratch", "asr_ocr_cache")
os.makedirs(CACHE, exist_ok=True)
name = sys.argv[1] if len(sys.argv) > 1 else "small"
m = whisper.load_model(name, device="cpu", download_root=os.path.join(ROOT, ".model_cache", "whisper"))
print("model", name, "loaded", flush=True)
for wav in sorted(glob.glob(os.path.join(ROOT, "ad_dataset", "audio", "*.wav"))):
    ad = os.path.basename(wav)[:-4]
    out = os.path.join(CACHE, ad + ".asr.json")
    if os.path.exists(out):
        continue
    r = m.transcribe(wav, verbose=None, fp16=False, condition_on_previous_text=False)
    segs = [{"start": round(s["start"], 1), "end": round(s["end"], 1), "text": s["text"].strip(),
             "no_speech": round(s.get("no_speech_prob", 0), 2)} for s in r["segments"] if s["text"].strip()]
    json.dump({"language": r.get("language"), "model": name, "speech": segs},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(ad, r.get("language"), len(segs), flush=True)
