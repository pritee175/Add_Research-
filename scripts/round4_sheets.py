"""Round 4: tile keyframes into 3x2 labelled contact sheets (.tmp/r4_sheets/<ad>/NN.jpg) for reading on-screen text.
Ads with more than 30 kept frames use every 2nd frame (about 6 s spacing) to keep the number of sheets manageable."""
import cv2, os, sys, glob
import numpy as np
SRC = "E:/Research_FF/.tmp/vlm_frames"; OUT = "E:/Research_FF/.tmp/r4_sheets"; W = 560
for ad in sys.argv[1:]:
    fs = sorted(glob.glob(f"{SRC}/{ad}/*.jpg"), key=lambda f: float(os.path.basename(f)[:-4]))
    if len(fs) > 30: fs = fs[::2]
    os.makedirs(f"{OUT}/{ad}", exist_ok=True)
    for n in range(0, len(fs), 6):
        tiles = []
        for f in fs[n:n+6]:
            im = cv2.imread(f); im = cv2.resize(im, (W, int(im.shape[0]*W/im.shape[1])))
            t = float(os.path.basename(f)[:-4])
            cv2.rectangle(im, (0,0), (92,26), (0,0,0), -1); cv2.putText(im, f"t={t:.0f}s", (5,19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            tiles.append(im)
        h = max(i.shape[0] for i in tiles); tiles = [cv2.copyMakeBorder(i, 0, h-i.shape[0], 0, 0, cv2.BORDER_CONSTANT) for i in tiles]
        while len(tiles) < 6: tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(f"{OUT}/{ad}/{n//6:02d}.jpg", np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(ad, len(fs), "frames", (len(fs)+5)//6, "sheets")
