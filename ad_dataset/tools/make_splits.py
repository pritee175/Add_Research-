"""Assign train/val/test in ads.csv, grouped so related ads never straddle splits.
Group = campaign_id, else brand (commercial ads), else the ad itself. B_0002/B_0017 are
duplicate content and forced into one group. Stratified by search_target. Seed fixed."""
import csv, random, collections, pathlib
p = pathlib.Path(__file__).resolve().parents[1] / 'annotations' / 'ads.csv'
raw = p.read_bytes(); nl = '\r\n' if b'\r\n' in raw else '\n'
rows = list(csv.reader(open(p, encoding='utf-8', newline=''))); h = rows[0]
c = {k: h.index(k) for k in ('ad_id', 'brand', 'campaign_id', 'search_target', 'split', 'video_status', 'main_label')}
def grp(r):
    if r[c['ad_id']] in ('B_0002', 'B_0017'): return 'DUP_PASSIVE'
    if r[c['campaign_id']]: return r[c['campaign_id']]
    b = r[c['brand']]
    return 'BRAND_' + b if b and b not in ('unknown', 'not_applicable') else r[c['ad_id']]
use = [r for r in rows[1:] if r[c['video_status']] == 'downloaded' and r[c['main_label']] != 'excluded']
for r in rows[1:]: r[c['split']] = ''
G = collections.defaultdict(list)
for r in use: G[(r[c['search_target']], grp(r))].append(r)
tgt = {'train': .7, 'val': .15, 'test': .15}; rnd = random.Random(42)
for t in {k[0] for k in G}:
    gs = sorted((k for k in G if k[0] == t), key=lambda k: (-len(G[k]), k)); rnd.shuffle(gs)
    gs.sort(key=lambda k: -len(G[k]))
    n = sum(len(G[k]) for k in gs); cnt = collections.Counter()
    for k in gs:
        s = max(tgt, key=lambda s: tgt[s] * n - cnt[s]); cnt[s] += len(G[k])
        for r in G[k]: r[c['split']] = s
with open(p, 'w', encoding='utf-8', newline='') as f: csv.writer(f, lineterminator=nl).writerows(rows)
print(collections.Counter((r[c['search_target']], r[c['split']]) for r in use))
