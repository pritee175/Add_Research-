"""Fair comparison: current TF-IDF text model vs multilingual text embeddings, alone and fused with CLIP images.

NEW file. It only IMPORTS the existing pipeline (image_fusion_baseline.py) to reuse its ads, labels, groups, frozen
split, TF-IDF model, CLIP features and the same 5x5 StratifiedGroupKFold protocol. Nothing existing is edited or re-run.

What is compared (same 101 ads, same labels, same folds):
    tfidf              current text model (word 1-2 + char 2-5 TF-IDF, logistic regression)
    embed              multilingual text embedding + logistic regression (same C=10, class_weight=balanced, no tuning)
    clip               CLIP ViT-L/14 image model (existing)
    tfidf+clip         existing fused model (reference)
    embed+clip         does the new text method help the combined model?
    tfidf+embed        both text views
    tfidf+embed+clip   everything
Fusion = average of probabilities, as in the existing fused model. Threshold 0.5.

Reported: frozen val/test (trained on train only, like the existing tables), 5x5 grouped-CV macro-F1/accuracy,
PAIRED fold-by-fold difference against tfidf (same folds, so the noise cancels), the ads each method gets wrong,
ads fixed / broken relative to tfidf, and the style-vs-message subsets from hard_subsets.py.
Never used as input: main_label, final_reason, search_target, source_type, evidence.direction, annotator fields
(labels are used only as training targets and for scoring, exactly as in the existing baselines).

Usage: python baselines/text_embedding_comparison.py [--features baselines/features/text_multilingual-e5-base.npz] [--out FILE.json]
Run extract_text_embeddings.py first.
"""
import sys, json, argparse, pathlib, collections
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

B = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(B))
import image_fusion_baseline as m          # existing pipeline: ads, y, g, split, text, X_img, p_text, p_img   (import only)

ap = argparse.ArgumentParser()
ap.add_argument('--features', default=str(B / 'features' / 'text_multilingual-e5-base.npz'))
ap.add_argument('--out', default=None)
ap.add_argument('--repeats', type=int, default=5)
args = ap.parse_args()
feat = pathlib.Path(args.features)
out_path = pathlib.Path(args.out) if args.out else B / f'results_text_embeddings_{feat.stem.replace("text_", "")}.json'

# ---- load embeddings and align them to m.ads ------------------------------------------------------------
z = np.load(feat); pos = {str(i): k for k, i in enumerate(z['ad_id'])}
missing = [a['ad_id'] for a in m.ads if a['ad_id'] not in pos]
assert not missing, f'embeddings missing for {missing[:5]}... re-run extract_text_embeddings.py'
X_txt = np.stack([z['emb'][pos[a['ad_id']]] for a in m.ads]).astype(np.float64)
X_txt /= np.linalg.norm(X_txt, axis=1, keepdims=True)
y, g, split, ads = m.y, m.g, m.split, m.ads
n = len(y)


def p_embed(tri, tei):
    return LogisticRegression(C=10, max_iter=2000, class_weight='balanced').fit(X_txt[tri], y[tri]).predict_proba(X_txt[tei])[:, 1]


def all_probs(tri, tei):
    """Base-model probabilities for the test indices; fused methods are averages of these."""
    P = {'tfidf': m.p_text(tri, tei), 'embed': p_embed(tri, tei), 'clip': m.p_img(tri, tei)}
    P['tfidf+clip'] = (P['tfidf'] + P['clip']) / 2
    P['embed+clip'] = (P['embed'] + P['clip']) / 2
    P['tfidf+embed'] = (P['tfidf'] + P['embed']) / 2
    P['tfidf+embed+clip'] = (P['tfidf'] + P['embed'] + P['clip']) / 3
    return P


METHODS = ['tfidf', 'embed', 'clip', 'tfidf+clip', 'embed+clip', 'tfidf+embed', 'tfidf+embed+clip']
sc = lambda yt, yp: dict(acc=float(accuracy_score(yt, yp)), macro_f1=float(f1_score(yt, yp, average='macro')))

# ---- frozen split (trained on train only, like the existing result tables) ------------------------------
tr, va, te = (np.where(split == s)[0] for s in ('train', 'val', 'test'))
frozen = {k: {} for k in METHODS}
for nm, ix in (('val', va), ('test', te)):
    P = all_probs(tr, ix)
    for k in METHODS:
        yp = (P[k] > .5).astype(int); frozen[k][nm] = sc(y[ix], yp); frozen[k][nm]['n'] = len(ix)
        if nm == 'test': frozen[k][nm]['confusion[prev,promo]'] = confusion_matrix(y[ix], yp, labels=[0, 1]).tolist()

# ---- grouped CV: identical folds to the existing scripts (same splitter, seeds 0..4, same groups) --------
f1s = {k: [] for k in METHODS}; accs = {k: [] for k in METHODS}
correct = {k: np.zeros(n) for k in METHODS}          # per ad: share of repeats classified correctly (out-of-fold)
for seed in range(args.repeats):
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(np.zeros(n), y, g):
        P = all_probs(a, b)
        for k in METHODS:
            yp = (P[k] > .5).astype(int); s = sc(y[b], yp); f1s[k].append(s['macro_f1']); accs[k].append(s['acc'])
            correct[k][b] += (yp == y[b])
for k in METHODS: correct[k] /= args.repeats

cv = {k: dict(macro_f1_mean=float(np.mean(f1s[k])), macro_f1_sd=float(np.std(f1s[k])), acc_mean=float(np.mean(accs[k])), folds=len(f1s[k])) for k in METHODS}
# paired comparison against the current text model: same folds, so the difference is far less noisy than comparing two SDs
paired = {}
for k in METHODS:
    d = np.array(f1s[k]) - np.array(f1s['tfidf'])
    paired[k] = dict(mean_diff_f1_vs_tfidf=float(d.mean()), sd_of_diff=float(d.std()), se=float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None,
                     folds_better=int((d > 1e-9).sum()), folds_worse=int((d < -1e-9).sum()), folds_equal=int((np.abs(d) <= 1e-9).sum()))
wrong = {k: [ads[i]['ad_id'] for i in np.argsort(correct[k]) if correct[k][i] < 0.5][:12] for k in METHODS}
# ads that change status relative to the current text model (usually wrong = under half of the repeats right)
fixed = {k: [ads[i]['ad_id'] for i in range(n) if correct['tfidf'][i] < .5 <= correct[k][i]] for k in METHODS if k != 'tfidf'}
broken = {k: [ads[i]['ad_id'] for i in range(n) if correct[k][i] < .5 <= correct['tfidf'][i]] for k in METHODS if k != 'tfidf'}

# ---- style-vs-message subsets (definitions copied from hard_subsets.py; metadata only, never a model input) -
subs = {
    'surrogate promotional': [i for i, a in enumerate(ads) if a['promotion_type'].startswith('surrogate') or a['promotion_type'] == 'uncertain'],
    'direct promotional (control)': [i for i, a in enumerate(ads) if a['promotion_type'] == 'direct'],
    'preventive non-institutional': [i for i, a in enumerate(ads) if a['main_label'] == 'preventive' and a['source_type'] in ('other', 'brand')],
    'preventive NGO/government (control)': [i for i, a in enumerate(ads) if a['main_label'] == 'preventive' and a['source_type'] in ('NGO', 'government')],
    'all': list(range(n))}
rng = np.random.default_rng(0); subset_res = {}
for name, idx in subs.items():
    subset_res[name] = {'n': len(idx)}
    for k in METHODS:
        v = correct[k][idx]; bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
        subset_res[name][k] = dict(acc=float(v.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
film = [ads[i]['ad_id'] for i in subs['preventive non-institutional']]
film_detail = {k: {a: float(correct[k][i]) for a, i in zip(film, subs['preventive non-institutional'])} for k in METHODS}

# ---- sanity check: our re-run of the existing TF-IDF / CLIP / fused numbers must match the saved ones ------
check = {}
try:
    ref = json.load(open(B / 'results_image_fusion.json', encoding='utf-8'))
    for mine, theirs in (('tfidf', 'text'), ('clip', 'image'), ('tfidf+clip', 'text+image')):
        check[mine] = dict(rerun=round(cv[mine]['macro_f1_mean'], 4), saved=round(ref[theirs]['cv']['macro_f1_mean'], 4),
                           match=abs(cv[mine]['macro_f1_mean'] - ref[theirs]['cv']['macro_f1_mean']) < 1e-6)
except Exception as e:
    check = {'error': repr(e)}

res = dict(features=str(feat.name), n_ads=n, promotional=int(y.sum()), preventive=int(n - y.sum()), frozen=frozen, cv=cv,
           paired_vs_tfidf=paired, most_often_wrong=wrong, fixed_vs_tfidf=fixed, broken_vs_tfidf=broken,
           subsets=subset_res, film_style_preventive_per_ad=film_detail, sanity_check_vs_saved_results=check)
json.dump(res, open(out_path, 'w'), indent=1)

# ---- print ---------------------------------------------------------------------------------------------
print(f'embeddings: {feat.name}   ads: {n}   folds: {len(f1s["tfidf"])}\n')
print(f"{'method':18}{'test acc':>9}{'val F1':>8}{'CV F1':>16}{'CV acc':>8}{'paired dF1 vs tfidf':>26}")
for k in METHODS:
    p = paired[k]
    print(f"{k:18}{frozen[k]['test']['acc']:9.2f}{frozen[k]['val']['macro_f1']:8.2f}  {cv[k]['macro_f1_mean']:.3f}±{cv[k]['macro_f1_sd']:.3f}{cv[k]['acc_mean']:8.3f}"
          f"   {p['mean_diff_f1_vs_tfidf']:+.3f} (better {p['folds_better']}/worse {p['folds_worse']}/same {p['folds_equal']})")
print('\nsanity check (re-run of existing models vs saved results_image_fusion.json):', check)
print('\nsubsets (out-of-fold accuracy):')
print(f"{'subset':38}{'n':>4}" + ''.join(f'{k:>18}' for k in ('tfidf', 'embed', 'clip', 'tfidf+clip', 'embed+clip')))
for name, d in subset_res.items():
    print(f"{name:38}{d['n']:4d}" + ''.join(f"{d[k]['acc']:18.2f}" for k in ('tfidf', 'embed', 'clip', 'tfidf+clip', 'embed+clip')))
print('\nads wrong most often:');  [print(f'  {k:18}{wrong[k][:8]}') for k in METHODS]
print('\nfixed relative to tfidf (wrong for tfidf, right for the method):'); [print(f'  {k:18}{v}') for k, v in fixed.items()]
print('broken relative to tfidf (right for tfidf, wrong for the method):'); [print(f'  {k:18}{v}') for k, v in broken.items()]
print('\nsaved', out_path)
