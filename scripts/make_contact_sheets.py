"""Tile extracted keyframes (.tmp/vlm_frames/<ad>/<t>.jpg) into labelled 2x2 contact sheets in .tmp/vlm_sheets/<ad>/NN.jpg."""
import cv2, os, sys, glob
import numpy as np
SRC = "E:/Research_FF/.tmp/vlm_frames"; OUT = "E:/Research_FF/.tmp/vlm_sheets"; W = 720
for ad in sys.argv[1:]:
    fs = sorted(glob.glob(f"{SRC}/{ad}/*.jpg"))
    os.makedirs(f"{OUT}/{ad}", exist_ok=True)
    for n in range(0, len(fs), 4):
        tiles = []
        for f in fs[n:n+4]:
            im = cv2.imread(f); im = cv2.resize(im, (W, int(im.shape[0]*W/im.shape[1])))
            t = float(os.path.basename(f)[:-4])
            cv2.rectangle(im, (0,0), (110,30), (0,0,0), -1)
            cv2.putText(im, f"t={t:.0f}s", (6,22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            tiles.append(im)
        h = max(i.shape[0] for i in tiles)
        tiles = [cv2.copyMakeBorder(i, 0, h-i.shape[0], 0, 0, cv2.BORDER_CONSTANT) for i in tiles]
        while len(tiles) < 4: tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(f"{OUT}/{ad}/{n//4:02d}.jpg", np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(ad, len(fs), "frames", (len(fs)+3)//4, "sheets")
