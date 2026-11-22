"""Classify one new video and print JSON. Called by the web demo's upload box (and usable on its own).
Loads CLIP only when run, so the web server itself stays light.
Usage: python baselines/predict_video.py VIDEO [--transcript TEXT] [--screen TEXT]"""
import sys, json, argparse, pathlib
import numpy as np, joblib
from clip_embed import embed
from fused_model import FusedModel
B = pathlib.Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument('video'); ap.add_argument('--transcript', default=''); ap.add_argument('--screen', default=''); a = ap.parse_args()
mod = joblib.load(B / 'models' / 'model_all.joblib')
z = np.load(B / 'features' / 'clip_vitl14_frames.npz'); import csv
ads = {r['ad_id']: r for r in csv.DictReader(open(B.parent / 'ad_dataset' / 'annotations' / 'ads.csv', encoding='utf-8'))}
by = {}
for e, i in zip(z['emb'], z['ad_id']): by.setdefault(str(i), []).append(e)
ids = [k for k in by if ads.get(k, {}).get('main_label') in ('promotional', 'preventive')]
X = np.stack([np.mean(by[k], 0) / np.linalg.norm(np.mean(by[k], 0)) for k in ids])
x = embed(a.video); txt = (a.transcript + ' ||| ' + a.screen) if (a.transcript.strip() or a.screen.strip()) else None
lab, p, parts = mod.predict(txt, x); pos, neg = mod.top_words(txt) if txt else ([], [])
s = X @ x; order = np.argsort(-s)[:3]
print(json.dumps(dict(pred=lab, p=round(p, 3), parts={k: round(v, 3) for k, v in parts.items()}, pos=pos, neg=neg, image_only=txt is None,
                      neighbours=[dict(id=ids[i], label=ads[ids[i]]['main_label'], sim=round(float(s[i]), 2), title=ads[ids[i]]['video_title'][:70]) for i in order]), ensure_ascii=False))
