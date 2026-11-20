"""Extract frozen CLIP image embeddings for the sampled frames of every usable ad.
Model is pinned by commit for reproducibility; weights are cached on the project disk (.hf_cache, git-ignored).
Output: baselines/features/clip_vitl14_frames.npz (+ .json with versions). No labels are read here.
Usage: python baselines/extract_clip_features.py"""
import os, csv, json, pathlib, platform
ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / '.hf_cache'))
import numpy as np, torch, transformers
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import CLIPModel, CLIPProcessor
MODEL, REV = 'openai/clip-vit-large-patch14', '32bd64288804d66eefd0ccbe215aa642df71cc41'
path = snapshot_download(MODEL, revision=REV, allow_patterns=['*.json', '*.txt', 'model.safetensors'])
torch.manual_seed(0); torch.set_num_threads(os.cpu_count() or 4)
model = CLIPModel.from_pretrained(path).eval(); proc = CLIPProcessor.from_pretrained(path)
D = ROOT / 'ad_dataset'
ids = {a['ad_id'] for a in csv.DictReader(open(D / 'annotations/ads.csv', encoding='utf-8')) if a['video_status'] == 'downloaded' and a['main_label'] != 'excluded'}
rows = [r for r in csv.DictReader(open(D / 'annotations/frames.csv', encoding='utf-8')) if r['ad_id'] in ids]
import re
for r in rows:  # some end-of-video frames have an empty `second`; the time is in the file name (t58.0s_end.jpg)
    r['second'] = r['second'] or re.search(r't([\d.]+)s', r['frame_file']).group(1)
rows.sort(key=lambda r: (r['ad_id'], float(r['second'])))
E, B = [], 16
with torch.no_grad():
    for i in range(0, len(rows), B):
        ims = [Image.open(D / r['frame_file']).convert('RGB') for r in rows[i:i + B]]
        f = model.get_image_features(**proc(images=ims, return_tensors='pt'))
        f = f.pooler_output if hasattr(f, 'pooler_output') else f
        E.append(torch.nn.functional.normalize(f, dim=-1).numpy()); print(i + len(ims), '/', len(rows), flush=True)
out = ROOT / 'baselines' / 'features'; out.mkdir(exist_ok=True)
np.savez_compressed(out / 'clip_vitl14_frames.npz', emb=np.concatenate(E).astype('float32'), ad_id=np.array([r['ad_id'] for r in rows]), second=np.array([float(r['second']) for r in rows]))
json.dump(dict(model=MODEL, revision=REV, torch=torch.__version__, transformers=transformers.__version__, python=platform.python_version(), n_frames=len(rows), n_ads=len({r['ad_id'] for r in rows})), open(out / 'clip_vitl14_frames.json', 'w'), indent=1)
