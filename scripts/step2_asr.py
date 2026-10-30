# -*- coding: utf-8 -*-
"""Step 2: faster-whisper (int8, CPU) on ad_dataset/audio/*.wav -> .scratch/asr_ocr_cache/<ad>.asr.json
Run with .venv\Scripts\python.exe (low memory)."""
import glob, json, os, sys
from faster_whisper import WhisperModel

ROOT = r"E:\Research_FF"
CACHE = os.path.join(ROOT, ".scratch", "asr_ocr_cache")
model = WhisperModel(sys.argv[1] if len(sys.argv) > 1 else "small", device="cpu", compute_type="int8",
                     download_root=os.path.join(ROOT, ".model_cache", "faster_whisper"))
for wav in sorted(glob.glob(os.path.join(ROOT, "ad_dataset", "audio", "*.wav"))):
    ad = os.path.basename(wav)[:-4]
    out = os.path.join(CACHE, ad + ".asr.json")
    if os.path.exists(out):
        continue
    segs, info = model.transcribe(wav, vad_filter=True, condition_on_previous_text=False)
    segs = [{"start": round(s.start, 1), "end": round(s.end, 1), "text": s.text.strip(),
             "no_speech": round(s.no_speech_prob, 2)} for s in segs if s.text.strip()]
    json.dump({"language": info.language, "lang_prob": round(info.language_probability, 2), "speech": segs},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(ad, info.language, len(segs), flush=True)
