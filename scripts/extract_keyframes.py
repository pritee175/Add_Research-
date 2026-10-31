"""Extract deduplicated keyframes (one candidate every STEP s) for VLM screen-text reading."""
import cv2, os, sys
import numpy as np

STEP = 2.0
LONG_STEP = 3.0
OUT = "E:/Research_FF/.tmp/vlm_frames"

def run(ad_id, max_frames=90):
    cap = cv2.VideoCapture(f"E:/Research_FF/ad_dataset/videos/{ad_id}.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    step = STEP if dur <= 60 else max(LONG_STEP, dur / max_frames)
    os.makedirs(f"{OUT}/{ad_id}", exist_ok=True)
    prev, kept, t = None, 0, 0.0
    while t < dur:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(f, (960, int(f.shape[0] * 960 / f.shape[1])))
        g = cv2.cvtColor(cv2.resize(f, (64, 36)), cv2.COLOR_BGR2GRAY).astype(float)
        if prev is None or np.abs(g - prev).mean() > 2.5:
            cv2.imwrite(f"{OUT}/{ad_id}/{t:07.1f}.jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])
            kept += 1
            prev = g
        t += step
    print(ad_id, f"dur={dur:.0f}s step={step:.1f} kept={kept}")

for a in sys.argv[1:]:
    run(a)
