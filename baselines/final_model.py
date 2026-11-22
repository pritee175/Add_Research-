"""Train and save the fused model, and score it on the held-out test split.
  models/model_trainval.joblib : trained on train+val (87 ads) -> honest scores on the 15 test ads, and the demo for held-out ads
  models/model_all.joblib      : trained on all 101 binary ads -> use for brand-new videos
Writes results_final_test.json and results_final_test.csv (per-ad predictions).
Usage: python baselines/final_model.py"""
import csv, json, pathlib, math
import numpy as np, joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
import image_fusion_baseline as m
from fused_model import FusedModel
B = pathlib.Path(__file__).resolve().parent
(B / 'models').mkdir(exist_ok=True)
tr = np.where(np.isin(m.split, ['train', 'val']))[0]; te = np.where(m.split == 'test')[0]
mod = FusedModel().fit([m.text[i] for i in tr], m.X_img[tr], m.y[tr])
joblib.dump(mod, B / 'models' / 'model_trainval.joblib', compress=3)
joblib.dump(FusedModel().fit(m.text, m.X_img, m.y), B / 'models' / 'model_all.joblib', compress=3)
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 3), round((c + h) / d, 3)]
rows, res = [], {}
for name, kw in (('fused', dict(use_t=True, use_i=True)), ('text', dict(use_t=True, use_i=False)), ('image', dict(use_t=False, use_i=True))):
    yp = []
    for i in te:
        lab, p, parts = mod.predict(m.text[i] if kw['use_t'] else None, m.X_img[i] if kw['use_i'] else None)
        yp.append(int(lab == 'promotional'))
        if name == 'fused':
            rows.append(dict(ad_id=m.ads[i]['ad_id'], true=m.ads[i]['main_label'], predicted=lab, p_promotional=round(p, 3),
                             p_text=round(parts['text'], 3), p_image=round(parts['image'], 3), correct=int(lab == m.ads[i]['main_label'])))
    yp = np.array(yp); yt = m.y[te]; k = int((yp == yt).sum())
    pr, rc, f, _ = precision_recall_fscore_support(yt, yp, labels=[0, 1], zero_division=0)
    res[name] = dict(n=len(te), correct=k, acc=k / len(te), acc_ci95_wilson=wilson(k, len(te)), macro_f1=float(f1_score(yt, yp, average='macro')),
                     preventive=dict(precision=float(pr[0]), recall=float(rc[0])), promotional=dict(precision=float(pr[1]), recall=float(rc[1])),
                     confusion_rows_true_prev_promo=confusion_matrix(yt, yp, labels=[0, 1]).tolist())
    print(f"{name:6} test {k}/{len(te)} = {k/len(te):.2f}  Wilson95% {res[name]['acc_ci95_wilson']}  macro-F1 {res[name]['macro_f1']:.2f}  confusion(prev,promo) {res[name]['confusion_rows_true_prev_promo']}")
res['trained_on'] = 'train+val (%d ads)' % len(tr)
json.dump(res, open(B / 'results_final_test.json', 'w'), indent=1)
with open(B / 'results_final_test.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('wrong on test:', [(r['ad_id'], r['true'], r['p_promotional']) for r in rows if not r['correct']])
