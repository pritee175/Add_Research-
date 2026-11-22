"""Precompute everything the web demo shows into baselines/web/demo_data.json, so the web server itself needs only the
standard library (low memory). Re-run after retraining (python baselines/final_model.py) or after the data changes.
Usage: python baselines/build_demo_data.py"""
import json, pathlib
import numpy as np, joblib
import image_fusion_baseline as m
from fused_model import FusedModel

B = pathlib.Path(__file__).resolve().parent; ROOT = B.parent
VID = ROOT / 'ad_dataset' / 'videos'
M_TV = joblib.load(B / 'models' / 'model_trainval.joblib'); M_ALL = joblib.load(B / 'models' / 'model_all.joblib')
n = len(m.ads); jl = lambda f: json.load(open(B / f, encoding='utf-8'))
RT, RI, RF, RH = jl('results_text.json'), jl('results_image_fusion.json'), jl('results_final_test.json'), jl('results_hard_subsets.json')

def neighbours(x, exclude=None, k=3):
    s = m.X_img @ x; order = [i for i in np.argsort(-s) if m.ads[i]['ad_id'] != exclude][:k]
    return [dict(id=m.ads[i]['ad_id'], label=m.ads[i]['main_label'], sim=round(float(s[i]), 2), title=m.ads[i]['video_title'][:70]) for i in order]

def ad_result(i):
    test = m.split[i] == 'test'; mod = M_TV if test else M_ALL
    lab, p, parts = mod.predict(m.text[i], m.X_img[i]); pos, neg = mod.top_words(m.text[i]); a = m.ads[i]
    return dict(id=a['ad_id'], title=a['video_title'], product=a['product_category'], split=m.split[i], true=a['main_label'], pred=lab, p=round(p, 3),
                parts={k: round(v, 3) for k, v in parts.items()}, correct=lab == a['main_label'], pos=pos, neg=neg,
                neighbours=neighbours(m.X_img[i], a['ad_id']), in_training=not test, has_video=(VID / f"{a['ad_id']}.mp4").exists())

sp = {}
for s in ('train', 'val', 'test'):
    ix = [i for i in range(n) if m.split[i] == s]
    sp[s] = dict(n=len(ix), promotional=sum(m.ads[i]['main_label'] == 'promotional' for i in ix), preventive=sum(m.ads[i]['main_label'] == 'preventive' for i in ix))
te = [i for i in range(n) if m.split[i] == 'test']; tr = [i for i in range(n) if m.split[i] != 'test']; S = m.X_img @ m.X_img.T
nn = [float(S[i, tr].max()) for i in te]; tt = [float(max(S[i, [j for j in tr if j != i]])) for i in tr]
G = lambda s: {m.g[i] for i in range(n) if m.split[i] == s}; V = lambda s: {m.ads[i]['youtube_video_id'] for i in range(n) if m.split[i] == s}
cv = lambda d: dict(f1=round(d['cv']['macro_f1_mean'], 3), sd=round(d['cv']['macro_f1_sd'], 3), acc=round(d['cv']['acc_mean'], 3))
hs = {k: {mm: round(v[mm]['acc'], 2) for mm in ('text', 'image', 'text+image')} | dict(n=v['n']) for k, v in RH.items() if isinstance(v, dict) and 'n' in v}
data = dict(n_ads=n, splits=sp, test_confusion=RF['fused']['confusion_rows_true_prev_promo'], test=dict(fused=RF['fused'], image=RF['image'], text=RF['text']),
            cv=dict(majority=cv(RT['majority']), transcript=cv(RT['speech']), screen=cv(RT['screen']), text=cv(RT['speech+screen']), image=cv(RI['image']), fused=cv(RI['text+image'])),
            hard=hs, disjoint=dict(shared_groups=len(G('test') & (G('train') | G('val'))), shared_videos=len(V('test') & (V('train') | V('val'))),
                                   test_to_train_median=round(float(np.median(nn)), 2), test_to_train_max=round(max(nn), 2),
                                   train_to_train_median=round(float(np.median(tt)), 2), train_to_train_p90=round(float(np.percentile(tt, 90)), 2)),
            test_ads=[ad_result(i) for i in te],
            other_ads=[dict(id=m.ads[i]['ad_id'], title=m.ads[i]['video_title'][:60], split=m.split[i]) for i in range(n) if m.split[i] != 'test'],
            ads={m.ads[i]['ad_id']: ad_result(i) for i in range(n)})
out = B / 'web' / 'demo_data.json'; out.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
print('wrote', out, round(out.stat().st_size / 1024), 'KB;', len(data['ads']), 'ads')
