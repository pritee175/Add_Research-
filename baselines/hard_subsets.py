"""Style-vs-message check. Where do the text, image and fused baselines fail?
Subsets are defined from metadata that is NEVER a model input:
  surrogate promotional  : promotion_type in surrogate_verified / surrogate_possible / uncertain (22 ads)
  preventive non-institutional : preventive ads whose source_type is 'other' or 'brand' (10 ads; film-style, not NGO/government PSAs)
  controls: direct promotional (41), preventive NGO/government (28)
Predictions are out-of-fold from 5x5 grouped CV; each ad's score is the share of repeats it was classified correctly.
Usage: python baselines/hard_subsets.py"""
import json, pathlib, collections
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import image_fusion_baseline as m
n = len(m.ads); correct = {k: np.zeros(n) for k in ('text', 'image', 'text+image')}; reps = 5
for seed in range(reps):
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(np.zeros(n), m.y, m.g):
        for k in correct: correct[k][b] += (m.preds(k, a, b) == m.y[b])
for k in correct: correct[k] /= reps
A = m.ads
subs = {
 'surrogate promotional': [i for i, a in enumerate(A) if a['promotion_type'].startswith('surrogate') or a['promotion_type'] == 'uncertain'],
 'direct promotional (control)': [i for i, a in enumerate(A) if a['promotion_type'] == 'direct'],
 'preventive non-institutional': [i for i, a in enumerate(A) if a['main_label'] == 'preventive' and a['source_type'] in ('other', 'brand')],
 'preventive NGO/government (control)': [i for i, a in enumerate(A) if a['main_label'] == 'preventive' and a['source_type'] in ('NGO', 'government')],
 'all': list(range(n))}
rng = np.random.default_rng(0); out = {}
print(f"{'subset':38}{'n':>4}  " + '  '.join(f'{k:>22}' for k in correct))
for nm, ix in subs.items():
    row = {}
    for k in correct:
        v = correct[k][ix]; bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
        row[k] = dict(acc=float(v.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    out[nm] = dict(n=len(ix), **row)
    print(f'{nm:38}{len(ix):>4}  ' + '  '.join(f"{row[k]['acc']:.2f} [{row[k]['ci95'][0]:.2f},{row[k]['ci95'][1]:.2f}]".rjust(22) for k in correct))
hard = {k: [A[i]['ad_id'] for i in subs['surrogate promotional'] + subs['preventive non-institutional'] if correct[k][i] < .5] for k in correct}
print('hard ads wrong in most repeats:', hard)
out['hard_ads_mostly_wrong'] = hard
json.dump(out, open(pathlib.Path(__file__).with_name('results_hard_subsets.json'), 'w'), indent=1)
