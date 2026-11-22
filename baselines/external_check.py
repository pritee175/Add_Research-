"""Informal out-of-sample check on round-4 'keep' candidates (videos in .tmp/r4_videos, NOT in the dataset, never used for training).
Image-only predictions from model_all. Expected labels are my judgement from titles/descriptions, NOT human-annotated:
the 6 drink-driving films are preventive, the rest are promotional. Treat as a sanity check, not a result.
Usage: python baselines/external_check.py"""
import csv, pathlib, joblib
import numpy as np
from clip_embed import embed
R = pathlib.Path(__file__).resolve().parents[1]
mod = joblib.load(pathlib.Path(__file__).with_name('models') / 'model_all.joblib')
PREV = {'q9CGd9qMcjQ', 'B3GMFO9B_iU', '57HwsYBwJaE', '-GdRAJpuEl4', 'dazqdI-etLY', 'tdTryL4ELoI'}
rows = [x for x in csv.DictReader(open(R / 'ad_dataset/candidates/candidates_round4.csv', encoding='utf-8-sig')) if x['decision'].startswith('keep')]
ok = 0; n = 0
for x in rows:
    f = R / '.tmp' / 'r4_videos' / (x['youtube_video_id'] + '.mp4')
    if not f.exists(): continue
    c = f.with_suffix('.npy')  # cache: the run can resume after a memory crash on a small machine
    if c.exists(): e = np.load(c)
    else: e = embed(f); np.save(c, e)
    lab, p, _ = mod.predict(None, e); exp = 'preventive' if x['youtube_video_id'] in PREV else 'promotional'; n += 1; ok += lab == exp
    print(f"{x['youtube_video_id']:13}{exp:12}{lab:12}P(promo)={p:.2f}  {'' if lab == exp else '<-- differs  '}{x['title'][:48]}")
print(f'\n{ok}/{n} agree with my expected label (image-only, unseen videos, expected labels not human-verified)')
