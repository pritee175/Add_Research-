"""Demo: classify an ad as promotional or preventive and show why.
  python baselines/demo.py --test-set                      # the 15 held-out test ads (honest scores)
  python baselines/demo.py --ad_id A_0033                  # one dataset ad (warns if it was in the training data)
  python baselines/demo.py --video path/to/ad.mp4          # a brand-new video (image-only unless text is given)
  python baselines/demo.py --video ad.mp4 --transcript "..." --screen "..."   # adds the text model
Explanation shown: which modality voted what, the words pushing each way, and the 3 most similar dataset ads (CLIP).
This is a research baseline on provisional single-annotator labels; read baselines/README.md before quoting any number."""
import os, sys, argparse, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / '.hf_cache'))
import numpy as np, joblib
import image_fusion_baseline as m
from fused_model import FusedModel
B = pathlib.Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument('--test-set', action='store_true'); ap.add_argument('--ad_id'); ap.add_argument('--video')
ap.add_argument('--transcript', default=''); ap.add_argument('--screen', default=''); a = ap.parse_args()
idx = {x['ad_id']: i for i, x in enumerate(m.ads)}
def neighbours(x, exclude=None, k=3):
    s = m.X_img @ x; order = [i for i in np.argsort(-s) if m.ads[i]['ad_id'] != exclude][:k]
    return [(m.ads[i]['ad_id'], m.ads[i]['main_label'], float(s[i])) for i in order]
def show(name, truth, lab, p, parts, text, mod, nn, note=''):
    print(f"\n=== {name} ===" + (f"  (true label: {truth})" if truth else ''))
    print(f"prediction: {lab.upper()}   P(promotional) = {p:.2f}   " + '  '.join(f'{k}={v:.2f}' for k, v in parts.items()))
    if note: print('note:', note)
    if text and 'text' in parts:
        pos, neg = mod.top_words(text); print('words pushing promotional:', pos or '-'); print('words pushing preventive :', neg or '-')
    print('most similar dataset ads (CLIP):', ', '.join(f'{i} [{l}] {s:.2f}' for i, l, s in nn))
from clip_embed import embed
if a.test_set:
    mod = joblib.load(B / 'models' / 'model_trainval.joblib'); ok = 0; te = [i for i in range(len(m.ads)) if m.split[i] == 'test']
    print(f"{'ad':8}{'true':13}{'predicted':13}{'P(promo)':9}  title")
    for i in te:
        lab, p, parts = mod.predict(m.text[i], m.X_img[i]); ok += lab == m.ads[i]['main_label']
        print(f"{m.ads[i]['ad_id']:8}{m.ads[i]['main_label']:13}{lab:13}{p:<9.2f}  {m.ads[i]['video_title'][:48]}" + ('' if lab == m.ads[i]['main_label'] else '   <-- WRONG'))
    print(f"\n{ok}/{len(te)} correct on the held-out test ads (model trained on train+val only). 15 ads is small: 95% interval about 80-100%.")
elif a.ad_id:
    i = idx[a.ad_id]; t = m.split[i] == 'test'
    mod = joblib.load(B / 'models' / ('model_trainval.joblib' if t else 'model_all.joblib'))
    lab, p, parts = mod.predict(m.text[i], m.X_img[i])
    show(a.ad_id + ' ' + m.ads[i]['video_title'][:50], m.ads[i]['main_label'], lab, p, parts, m.text[i], mod, neighbours(m.X_img[i], a.ad_id),
         '' if t else 'this ad was in the training data, so a correct answer here is not a fair test. Use --test-set or --video.')
elif a.video:
    mod = joblib.load(B / 'models' / 'model_all.joblib'); x = embed(a.video)
    txt = (a.transcript + ' ||| ' + a.screen) if (a.transcript or a.screen) else None
    lab, p, parts = mod.predict(txt, x)
    show(pathlib.Path(a.video).name, '', lab, p, parts, txt, mod, neighbours(x),
         '' if txt else 'image-only prediction (no transcript or screen text supplied); cross-validated F1 for image-only is about 0.94.')
else:
    ap.print_help()
