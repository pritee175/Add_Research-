"""Append an ann2 export to annotators.csv and report Cohen's kappa vs ann1.
Usage: python merge_ann2.py path/to/annotators_ann2.csv
Kappa uses the original ann1 and ann2 labels (before adjudication)."""
import csv, sys, pathlib, collections
p = pathlib.Path(__file__).resolve().parents[1] / 'annotations' / 'annotators.csv'
new = list(csv.DictReader(open(sys.argv[1], encoding='utf-8')))
cur = list(csv.DictReader(open(p, encoding='utf-8')))
have = {(r['ad_id'], r['annotator_id']) for r in cur}
add = [r for r in new if (r['ad_id'], 'ann2') not in have and r['annotator_id'] == 'ann2']
with open(p, 'a', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(cur[0].keys()))
    w.writerows(add)
rows = cur + add
l1 = {r['ad_id']: r['chosen_label'] for r in rows if r['annotator_id'] == 'ann1'}
l2 = {r['ad_id']: r['chosen_label'] for r in rows if r['annotator_id'] == 'ann2'}
ids = sorted(set(l1) & set(l2)); n = len(ids)
po = sum(l1[i] == l2[i] for i in ids) / n
c1, c2 = collections.Counter(l1[i] for i in ids), collections.Counter(l2[i] for i in ids)
pe = sum(c1[k] * c2[k] for k in c1) / n ** 2
print(f'added {len(add)} rows; overlap {n}; agreement {po:.3f}; kappa {(po-pe)/(1-pe) if pe<1 else float("nan"):.3f}')
print('disagreements (need adjudicator):', [i for i in ids if l1[i] != l2[i]])
