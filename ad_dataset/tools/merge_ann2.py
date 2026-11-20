"""Append an annotator export to annotators.csv and report Cohen's kappa against a reference annotator.
Usage: python merge_ann2.py export.csv [annotator_id=ann2] [reference_id=ai_ann2]
Only ads present in the export are compared. Labels are the originals (before adjudication).
Note: if the reference is an AI annotator, the result is agreement with an AI, not human inter-annotator agreement."""
import csv, sys, pathlib, collections
AID = sys.argv[2] if len(sys.argv) > 2 else 'ann2'
REF = sys.argv[3] if len(sys.argv) > 3 else 'ai_ann2'
p = pathlib.Path(__file__).resolve().parents[1] / 'annotations' / 'annotators.csv'
raw = p.read_bytes(); nl = '\r\n' if b'\r\n' in raw else '\n'
new = [r for r in csv.DictReader(open(sys.argv[1], encoding='utf-8-sig')) if r['annotator_id'] == AID]
cur = list(csv.DictReader(open(p, encoding='utf-8')))
have = {(r['ad_id'], r['annotator_id']) for r in cur}
add = [r for r in new if (r['ad_id'], AID) not in have]
if add:
    with open(p, 'a', encoding='utf-8', newline='') as f:
        if not raw.endswith(b'\n'): f.write(nl)
        csv.DictWriter(f, fieldnames=list(cur[0].keys()), lineterminator=nl).writerows(add)
rows = cur + add
l1 = {r['ad_id']: r['chosen_label'] for r in rows if r['annotator_id'] == REF}
l2 = {r['ad_id']: r['chosen_label'] for r in rows if r['annotator_id'] == AID}
ids = sorted(set(l1) & set(l2) & {r['ad_id'] for r in new}); n = len(ids)
po = sum(l1[i] == l2[i] for i in ids) / n
c1, c2 = collections.Counter(l1[i] for i in ids), collections.Counter(l2[i] for i in ids)
pe = sum(c1[k] * c2[k] for k in c1) / n ** 2
print(f'added {len(add)} rows; compared {AID} vs {REF} on {n} ads; agreement {po:.3f}; kappa {(po-pe)/(1-pe) if pe<1 else float("nan"):.3f}')
labs = sorted(set(l1.values()) | set(l2.values()))
print(f'{REF} (rows) vs {AID} (cols):', labs)
for a in labs: print(f'  {a:14}', [sum(1 for i in ids if l1[i] == a and l2[i] == b) for b in labs])
print('disagreements (need adjudicator):', [(i, REF + '=' + l1[i], AID + '=' + l2[i]) for i in ids if l1[i] != l2[i]])
