"""Frame-based and fused baselines. Same task, data and evaluation protocol as text_baseline.py.
Image features: frozen CLIP ViT-L/14 embeddings (extract_clip_features.py), mean-pooled over each ad's sampled frames.
Fusion = average of predicted probabilities from the text model (transcript + screen text) and the image model.
Run extract_clip_features.py first. Usage: python baselines/image_fusion_baseline.py"""
import csv, json, pathlib, collections
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
B = pathlib.Path(__file__).resolve().parent
R = B.parent / 'ad_dataset' / 'annotations'
rd = lambda f: list(csv.DictReader(open(R / f, encoding='utf-8')))
ads = [a for a in rd('ads.csv') if a['video_status'] == 'downloaded' and a['main_label'] in ('promotional', 'preventive')]
sp, st = collections.defaultdict(list), collections.defaultdict(list)
for r in sorted(rd('speech.csv'), key=lambda r: (r['ad_id'], float(r['start_second'] or 0))): sp[r['ad_id']].append(r['original_text'])
for r in sorted(rd('screen_text.csv'), key=lambda r: (r['ad_id'], float(r['start_second'] or 0))): st[r['ad_id']].append(r['original_text'])
def grp(a):
    if a['ad_id'] in ('B_0002', 'B_0017'): return 'DUP_PASSIVE'
    if a['campaign_id']: return a['campaign_id']
    b = a['brand']; return 'BRAND_' + b if b and b not in ('unknown', 'not_applicable') else a['ad_id']
y = np.array([a['main_label'] == 'promotional' for a in ads]).astype(int)
g = np.array([grp(a) for a in ads]); split = np.array([a['split'] for a in ads])
text = [' '.join(sp[a['ad_id']]) + ' ||| ' + ' '.join(st[a['ad_id']]) for a in ads]
z = np.load(B / 'features' / 'clip_vitl14_frames.npz'); by = collections.defaultdict(list)
for e, i in zip(z['emb'], z['ad_id']): by[str(i)].append(e)
X_img = np.stack([np.mean(by[a['ad_id']], 0) for a in ads]); X_img /= np.linalg.norm(X_img, axis=1, keepdims=True)
def p_text(tri, tei):
    w = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r'(?u)\b\w+\b')
    c = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=60000)
    tr = [text[i] for i in tri]; te = [text[i] for i in tei]
    Xtr = hstack([w.fit_transform(tr), c.fit_transform(tr)]).tocsr(); Xte = hstack([w.transform(te), c.transform(te)]).tocsr()
    return LogisticRegression(C=10, max_iter=2000, class_weight='balanced').fit(Xtr, y[tri]).predict_proba(Xte)[:, 1]
def p_img(tri, tei): return LogisticRegression(C=10, max_iter=2000, class_weight='balanced').fit(X_img[tri], y[tri]).predict_proba(X_img[tei])[:, 1]
def preds(name, tri, tei):
    if name == 'image': return p_img(tri, tei) > .5
    if name == 'text': return p_text(tri, tei) > .5
    return (p_text(tri, tei) + p_img(tri, tei)) / 2 > .5
sc = lambda yt, yp: dict(acc=accuracy_score(yt, yp), macro_f1=f1_score(yt, yp, average='macro'))
if __name__ == '__main__':
    res = {}
    for name in ['image', 'text', 'text+image']:
        tr, va, te = (np.where(split == s)[0] for s in ('train', 'val', 'test'))
        out = {nm: sc(y[ix], preds(name, tr, ix)) for nm, ix in (('val', va), ('test', te))}
        out['test']['confusion[prev,promo]'] = confusion_matrix(y[te], preds(name, tr, te), labels=[0, 1]).tolist()
        f1s, accs, wrong = [], [], collections.Counter()
        for seed in range(5):
            for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, g):
                p = preds(name, a, b); s = sc(y[b], p); f1s.append(s['macro_f1']); accs.append(s['acc'])
                for i in b[p != y[b]]: wrong[ads[i]['ad_id']] += 1
        out['cv'] = dict(macro_f1_mean=float(np.mean(f1s)), macro_f1_sd=float(np.std(f1s)), acc_mean=float(np.mean(accs)), folds=len(f1s))
        out['most_often_wrong'] = [k for k, _ in wrong.most_common(10)]; res[name] = out
        print(f"{name:11} test acc {out['test']['acc']:.2f} F1 {out['test']['macro_f1']:.2f} | val F1 {out['val']['macro_f1']:.2f} | CV F1 {out['cv']['macro_f1_mean']:.2f}±{out['cv']['macro_f1_sd']:.2f} acc {out['cv']['acc_mean']:.2f} | wrong often: {out['most_often_wrong'][:8]}")
    json.dump(res, open(B / 'results_image_fusion.json', 'w'), indent=1)
