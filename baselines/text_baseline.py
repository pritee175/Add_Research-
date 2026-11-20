"""Text baselines: promotional vs preventive from transcript and/or on-screen text.
Never uses main_label, final_reason, search_target, evidence.direction or any annotator field as input.
Evaluation: (a) the frozen train/val/test split, (b) 5-fold grouped CV over all ads (groups = campaign/brand,
same rule as make_splits.py) repeated with 5 seeds. unclear_mixed (1 ad) is dropped.
Usage: python baselines/text_baseline.py"""
import csv, json, pathlib, collections
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.dummy import DummyClassifier
R = pathlib.Path(__file__).resolve().parents[1] / 'ad_dataset' / 'annotations'
rd = lambda f: list(csv.DictReader(open(R / f, encoding='utf-8')))
ads = [a for a in rd('ads.csv') if a['video_status'] == 'downloaded' and a['main_label'] in ('promotional', 'preventive')]
sp, st, st2 = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
for r in sorted(rd('speech.csv'), key=lambda r: (r['ad_id'], float(r['start_second'] or 0))): sp[r['ad_id']].append(r['original_text'])
for r in sorted(rd('screen_text.csv'), key=lambda r: (r['ad_id'], float(r['start_second'] or 0))):
    st[r['ad_id']].append(r['original_text'])
    if r['text_type'] in ('brand', 'tagline', 'caption', 'other'): st2[r['ad_id']].append(r['original_text'])  # drops warning/helpline/disclaimer/credit cues
def grp(a):
    if a['ad_id'] in ('B_0002', 'B_0017'): return 'DUP_PASSIVE'
    if a['campaign_id']: return a['campaign_id']
    b = a['brand']; return 'BRAND_' + b if b and b not in ('unknown', 'not_applicable') else a['ad_id']
y = np.array([a['main_label'] == 'promotional' for a in ads]).astype(int)
g = np.array([grp(a) for a in ads]); split = np.array([a['split'] for a in ads])
T = {'speech': [' '.join(sp[a['ad_id']]) for a in ads], 'screen': [' '.join(st[a['ad_id']]) for a in ads]}
T['screen_no_cues'] = [' '.join(st2[a['ad_id']]) for a in ads]
T['speech+screen'] = [x + ' ||| ' + z for x, z in zip(T['speech'], T['screen'])]
def feats(tr, te):
    w = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r'(?u)\b\w+\b')
    c = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=60000)
    Xtr = hstack([w.fit_transform(tr), c.fit_transform(tr)]).tocsr(); Xte = hstack([w.transform(te), c.transform(te)]).tocsr()
    return Xtr, Xte
def fit_pred(name, tri, tei, texts, model):
    tr = [texts[i] for i in tri]; te = [texts[i] for i in tei]
    if model == 'majority': m = DummyClassifier(strategy='most_frequent').fit(np.zeros((len(tri), 1)), y[tri]); return m.predict(np.zeros((len(tei), 1)))
    Xtr, Xte = feats(tr, te)
    return LogisticRegression(C=10, max_iter=2000, class_weight='balanced').fit(Xtr, y[tri]).predict(Xte)
def score(yt, yp): return dict(acc=accuracy_score(yt, yp), macro_f1=f1_score(yt, yp, average='macro'))
res = {}
for name in ['majority', 'speech', 'screen', 'screen_no_cues', 'speech+screen']:
    texts = T['speech+screen'] if name == 'majority' else T[name]; model = 'majority' if name == 'majority' else 'lr'
    tr = np.where(split == 'train')[0]; va = np.where(split == 'val')[0]; te = np.where(split == 'test')[0]
    out = {}
    for nm, idx in (('val', va), ('test', te)):
        p = fit_pred(name, tr, idx, texts, model); out[nm] = score(y[idx], p); out[nm]['n'] = len(idx)
        if nm == 'test': out['test']['confusion[prev,promo]'] = confusion_matrix(y[idx], p, labels=[0, 1]).tolist()
    f1s, accs, wrong = [], [], collections.Counter()
    for seed in range(5):
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, g):
            p = fit_pred(name, a, b, texts, model); s = score(y[b], p); f1s.append(s['macro_f1']); accs.append(s['acc'])
            for i in b[p != y[b]]: wrong[ads[i]['ad_id']] += 1
    out['cv'] = dict(macro_f1_mean=float(np.mean(f1s)), macro_f1_sd=float(np.std(f1s)), acc_mean=float(np.mean(accs)), acc_sd=float(np.std(accs)), folds=len(f1s))
    out['most_often_wrong'] = [k for k, _ in wrong.most_common(8)]
    res[name] = out
    print(f"{name:14} test acc {out['test']['acc']:.2f} F1 {out['test']['macro_f1']:.2f} | val F1 {out['val']['macro_f1']:.2f} | CV F1 {out['cv']['macro_f1_mean']:.2f}±{out['cv']['macro_f1_sd']:.2f} acc {out['cv']['acc_mean']:.2f} | wrong often: {out['most_often_wrong'][:6]}")
res['_n'] = dict(ads=len(ads), promotional=int(y.sum()), preventive=int(len(y) - y.sum()), test=int((split == 'test').sum()))
json.dump(res, open(pathlib.Path(__file__).with_name('results_text.json'), 'w'), indent=1)
