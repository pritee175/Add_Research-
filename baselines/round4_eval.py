"""Second held-out evaluation: the frozen fused model (trained on train+val only, models/model_trainval.joblib) scored on the
staged Round 4 ads in ad_dataset/round4/. No Round 4 ad was used for training, tuning or feature selection.
Labels are AI labels pending human review, so treat the scores as provisional. Writes baselines/results_round4.json.
Needs the CLIP image tower (about 1.5 GB free RAM). Usage: python baselines/round4_eval.py"""
import csv, json, math, pathlib, collections
import numpy as np, joblib
from sklearn.metrics import f1_score
from clip_embed import embed
from fused_model import FusedModel
B = pathlib.Path(__file__).resolve().parent; R4 = B.parent / 'ad_dataset' / 'round4'
rd = lambda f: list(csv.DictReader(open(R4 / f, encoding='utf-8')))
ads = [a for a in rd('ads.csv') if a['main_label'] in ('promotional', 'preventive')]
sp, st = collections.defaultdict(list), collections.defaultdict(list)
for r in sorted(rd('speech.csv'), key=lambda r: (r['ad_id'], float(r['start_second'] or 0))): sp[r['ad_id']].append(r['original_text'])
for r in sorted(rd('screen_text.csv'), key=lambda r: (r['ad_id'], float(r['start_second'] or 0))): st[r['ad_id']].append(r['original_text'])
mod = joblib.load(B / 'models' / 'model_trainval.joblib'); rows = []
for a in ads:
    text = ' '.join(sp[a['ad_id']]) + ' ||| ' + ' '.join(st[a['ad_id']]); x = embed(str(B.parent / 'ad_dataset' / a['video_file']))
    out = {}
    for name, t, i in (('fused', text, x), ('text', text, None), ('image', None, x)):
        lab, p, _ = mod.predict(t, i); out[name] = (lab, round(p, 3))
    rows.append(dict(ad_id=a['ad_id'], true=a['main_label'], promotion_type=a['promotion_type'], product=a['product_category'], title=a['video_title'][:50], **{k: v[0] for k, v in out.items()}, **{'p_' + k: v[1] for k, v in out.items()}))
    print(f"{a['ad_id']} {a['main_label']:11} fused={out['fused'][0]:11}({out['fused'][1]:.2f}) text={out['text'][0]:11}({out['text'][1]:.2f}) image={out['image'][0]:11}({out['image'][1]:.2f}) {a['promotion_type']:18} {a['video_title'][:34]}", flush=True)
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return [round((c - h) / d, 3), round((c + h) / d, 3)]
res = {}
for name in ('fused', 'text', 'image'):
    k = sum(r[name] == r['true'] for r in rows); n = len(rows)
    res[name] = dict(n=n, correct=k, acc=round(k / n, 3), wilson95=wilson(k, n), macro_f1=round(float(f1_score([r['true'] for r in rows], [r[name] for r in rows], average='macro')), 3),
                     by_group={g: [sum(r[name] == r['true'] for r in rows if g == r['promotion_type'] or g == r['true']), sum(1 for r in rows if g == r['promotion_type'] or g == r['true'])] for g in ('preventive', 'direct', 'surrogate_verified')})
res['per_ad'] = rows; res['note'] = 'Frozen model trained on train+val (86 ads). Round 4 labels are AI labels pending human review. Wills (A_0024) and Royal Stag (A_0012) ads are in the v1 TEST split, not in training.'
json.dump(res, open(B / 'results_round4.json', 'w'), indent=1, ensure_ascii=False)
for name in ('fused', 'text', 'image'): print(name, res[name]['correct'], '/', res[name]['n'], 'acc', res[name]['acc'], 'Wilson95', res[name]['wilson95'], 'macroF1', res[name]['macro_f1'], res[name]['by_group'])
