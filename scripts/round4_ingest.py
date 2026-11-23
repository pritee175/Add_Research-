"""Round 4 ingest, stage 1: copy the 15 screened 'keep' videos from .tmp/r4_videos into ad_dataset/videos under new ids,
extract 16 kHz mono audio, and cut keyframes (same settings as extract_keyframes.py) plus 5 evenly spaced review frames.
Does not touch any existing annotation file. Usage: python scripts/round4_ingest.py"""
import csv, os, shutil, subprocess, sys, json
import cv2
ROOT = 'E:/Research_FF'; FF = ROOT + '/.tmp/bin/ffmpeg.exe'
# new id -> youtube id. A_ = promotional-targeted, B_ = preventive-targeted (project convention)
MAP = {'B_0047': 'q9CGd9qMcjQ', 'B_0048': 'B3GMFO9B_iU', 'B_0049': '57HwsYBwJaE', 'B_0050': '-GdRAJpuEl4', 'B_0051': 'dazqdI-etLY', 'B_0052': 'tdTryL4ELoI',
       'A_0074': 'wIKA6pAewpg', 'A_0075': '1DGSaLVJby0', 'A_0076': 'XPQj5_5nF4o', 'A_0077': 'kfjNCbNJRqs', 'A_0078': 'k0tELT5NVWY',
       'A_0079': 'jtNxY_98A-s', 'A_0080': 'sON8zyqQhJ4', 'A_0081': 'CphlkB7uYfY', 'A_0082': 'ck0IjzlWabw'}
os.makedirs(ROOT + '/ad_dataset/audio', exist_ok=True)
info = {}
for ad, yt in MAP.items():
    src = f'{ROOT}/.tmp/r4_videos/{yt}.mp4'; dst = f'{ROOT}/ad_dataset/videos/{ad}.mp4'
    if os.path.exists(dst): raise SystemExit(f'{dst} already exists; refusing to overwrite')
    shutil.copy2(src, dst)
    subprocess.run([FF, '-y', '-loglevel', 'error', '-i', dst, '-ar', '16000', '-ac', '1', f'{ROOT}/ad_dataset/audio/{ad}.wav'], check=True)
    cap = cv2.VideoCapture(dst); fps = cap.get(cv2.CAP_PROP_FPS) or 25; dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    # 5 evenly spaced review frames (10%,30%,50%,70%,90% of the video), saved under frames/round4/<ad>/
    os.makedirs(f'{ROOT}/ad_dataset/frames/round4/{ad}', exist_ok=True); fr = []
    for q in (.1, .3, .5, .7, .9):
        t = round(dur * q, 1); cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, f = cap.read()
        if ok:
            p = f'frames/round4/{ad}/t{t}s.jpg'; cv2.imwrite(f'{ROOT}/ad_dataset/{p}', f); fr.append((ad, t, p, 'regular'))
    info[ad] = dict(yt=yt, duration=round(dur), frames=fr)
    print(ad, yt, round(dur), 's', len(fr), 'frames', flush=True)
json.dump(info, open(ROOT + '/.tmp/round4_info.json', 'w'), indent=1)
