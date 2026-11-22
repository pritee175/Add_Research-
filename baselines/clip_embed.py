"""Embed a video as the mean of CLIP ViT-L/14 image embeddings of 5 evenly spaced frames.
Same model and revision as extract_clip_features.py (openai/clip-vit-large-patch14 @ 32bd642), but low-memory:
only the image tower is read from the safetensors file (plain file reads, no memory map) and kept in float16 (~0.6 GB),
so it also runs on a laptop with little free RAM. Embeddings match the stored float32 ones to cosine >= 0.999 (checked in README)."""
import os, sys, json, pathlib
os.environ.setdefault('HF_HOME', str(pathlib.Path(__file__).resolve().parents[1] / '.hf_cache'))
import numpy as np
_C = {}
REV = '32bd64288804d66eefd0ccbe215aa642df71cc41'
_DT = {'F32': np.float32, 'F16': np.float16}


def _load():
    import torch
    from huggingface_hub import snapshot_download
    from transformers import CLIPConfig, CLIPVisionModelWithProjection, CLIPImageProcessor
    p = snapshot_download('openai/clip-vit-large-patch14', revision=REV, allow_patterns=['*.json', '*.txt', 'model.safetensors'], local_files_only=True)
    cfg = CLIPConfig.from_pretrained(p)
    torch.set_default_dtype(torch.float16)
    try: model = CLIPVisionModelWithProjection(cfg.vision_config).eval()
    finally: torch.set_default_dtype(torch.float32)
    params = dict(model.named_parameters()); done = set()
    with open(os.path.join(p, 'model.safetensors'), 'rb') as f:  # copy one tensor at a time: keeps peak memory near the model size
        n = int.from_bytes(f.read(8), 'little'); hdr = json.loads(f.read(n)); base = 8 + n
        for k, v in hdr.items():
            if k not in params: continue
            b, e = v['data_offsets']; f.seek(base + b)
            with torch.no_grad(): params[k].copy_(torch.from_numpy(np.frombuffer(f.read(e - b), dtype=_DT[v['dtype']]).reshape(v['shape']).copy()))
            done.add(k)
    if done != set(params): raise RuntimeError('missing weights: %s' % sorted(set(params) - done)[:5])
    _C['m'] = (model, CLIPImageProcessor.from_pretrained(p))


def embed(path):
    import cv2, torch
    from PIL import Image
    if 'm' not in _C: _load()
    clip, proc = _C['m']
    cap = cv2.VideoCapture(str(path)); n = cap.get(cv2.CAP_PROP_FRAME_COUNT); ims = []
    for f in (.1, .3, .5, .7, .9):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * f)); ok, fr = cap.read()
        if ok: ims.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    if not ims: sys.exit('could not read frames from the video')
    return embed_images(ims)


def embed_images(ims):
    import torch
    if 'm' not in _C: _load()
    clip, proc = _C['m']
    import gc
    out = []
    with torch.no_grad():  # one frame at a time keeps peak memory low on small machines
        for im in ims:
            px = proc(images=[im], return_tensors='pt')['pixel_values'].half()
            out.append(torch.nn.functional.normalize(clip(pixel_values=px).image_embeds.float(), dim=-1).numpy()[0]); gc.collect()
    o = np.stack(out)
    v = o.mean(0); return v / np.linalg.norm(v)
