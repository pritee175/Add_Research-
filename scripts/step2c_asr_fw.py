# -*- coding: utf-8 -*-
"""Step 2 (improved): faster-whisper int8 with language detection restricted to languages seen in Indian ads.
Output: .scratch/asr_ocr_cache/<ad>.asr.json (overwrites base results only if --replace) or <ad>.asr_<model>.json
usage: step2c_asr_fw.py <model: small|medium|large-v3-turbo> [ad_id ...]   (run with .venv python)"""
import glob, json, os, sys
import numpy as np
from faster_whisper import WhisperModel, decode_audio

ROOT = r"E:\Research_FF"
CACHE = os.path.join(ROOT, ".scratch", "asr_ocr_cache")
ALLOWED = ["hi", "en", "mr", "ta", "bn", "te", "kn", "ml", "gu", "pa"]
name = sys.argv[1]
only = sys.argv[2:]
model = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=2,
                     download_root=os.path.join(ROOT, ".model_cache", "faster_whisper"))
print("model", name, "loaded", flush=True)
for wav in sorted(glob.glob(os.path.join(ROOT, "ad_dataset", "audio", "*.wav"))):
    ad = os.path.basename(wav)[:-4]
    if only and ad not in only:
        continue
    out = os.path.join(CACHE, f"{ad}.asr_{name}.json")
    if os.path.exists(out):
        continue
    audio = decode_audio(wav, sampling_rate=16000)
    lang, p, probs = model.detect_language(audio, vad_filter=True, language_detection_segments=4)
    cand = {l: pr for l, pr in probs if l in ALLOWED}
    lang = max(cand, key=cand.get) if cand else "hi"
    def run(beam):
        sg, _ = model.transcribe(audio, language=lang, task="transcribe", vad_filter=True,
                                 beam_size=beam, condition_on_previous_text=False,
                                 compression_ratio_threshold=2.0, no_speech_threshold=0.6)
        return list(sg)
    try:
        segs = run(5)
    except RuntimeError as e:
        print(ad, "mem error, retry beam=1", flush=True)
        import gc; gc.collect()
        try:
            segs = run(1)
        except RuntimeError as e2:
            print("FAILED", ad, e2, flush=True)
            continue
    segs = [{"start": round(s.start, 1), "end": round(s.end, 1), "text": s.text.strip(),
             "no_speech": round(s.no_speech_prob, 2), "avg_logprob": round(s.avg_logprob, 2)}
            for s in segs if s.text.strip()]
    json.dump({"language": lang, "lang_prob": round(cand.get(lang, 0), 2), "model": name, "speech": segs},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(ad, lang, round(cand.get(lang, 0), 2), len(segs), flush=True)
