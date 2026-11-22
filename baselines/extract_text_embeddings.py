"""Extract frozen multilingual TEXT embeddings, one vector per usable ad, for the text-model comparison.

This is the text counterpart of extract_clip_features.py. It is a NEW file: no existing baseline code is touched.
Input per ad (same recipe as image_fusion_baseline.py / text_baseline.py):
    ' '.join(speech rows) + ' ||| ' + ' '.join(screen-text rows), rows sorted by start_second.
No labels and no answer-key fields are read here (only ads.csv for the id list and video_status/main_label != excluded).

Long ads: transformer models accept at most 512 tokens, so each ad's text is cut into chunks that fit, every chunk is
embedded, and the chunk vectors are averaged (weighted by chunk length) and L2-normalised. Nothing is silently truncated.

Default model: intfloat/multilingual-e5-base (XLM-RoBERTa-base backbone, further trained for sentence embeddings,
covers Hindi/Bengali/Marathi/Tamil/English). Other choices work too, e.g.
    --model sentence-transformers/LaBSE --pooling cls --prefix ""
    --model google/muril-base-cased --pooling mean --prefix ""
    --model xlm-roberta-base --pooling mean --prefix ""        (raw XLM-R: expected to be a weak embedder)

Output: baselines/features/text_<model-slug>.npz  (ad_id, emb)   +   .json (model, resolved commit, library versions).
Usage:  python baselines/extract_text_embeddings.py [--model NAME] [--revision COMMIT] [--pooling mean|cls] [--prefix "passage: "]
Tip: set HF_HOME to a folder OUTSIDE OneDrive (the model is ~1.1 GB); the default is <repo>/.hf_cache (git-ignored).
"""
import os, re, csv, json, argparse, pathlib, platform, collections

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / '.hf_cache'))

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='intfloat/multilingual-e5-base')
ap.add_argument('--revision', default=None, help='pin a commit (recommended once you have run it; the JSON records the one used)')
ap.add_argument('--pooling', default='mean', choices=['mean', 'cls'])
ap.add_argument('--prefix', default=None, help='text prepended to every chunk; e5 models need "query: " or "passage: " (default: "query: " for e5, else none)')
ap.add_argument('--max-len', type=int, default=512)
ap.add_argument('--out', default=None, help='output .npz path (default baselines/features/text_<slug>.npz)')
args = ap.parse_args()
if args.prefix is None:
    args.prefix = 'query: ' if 'e5' in args.model.lower() else ''
slug = re.sub(r'[^A-Za-z0-9]+', '-', args.model.split('/')[-1]).strip('-').lower()
out_npz = pathlib.Path(args.out) if args.out else ROOT / 'baselines' / 'features' / f'text_{slug}.npz'

import numpy as np, torch, transformers
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

A = ROOT / 'ad_dataset' / 'annotations'
rd = lambda f: list(csv.DictReader(open(A / f, encoding='utf-8')))
# same ad set as the existing baselines: downloaded, binary label (the single unclear_mixed ad is dropped)
ads = [a for a in rd('ads.csv') if a['video_status'] == 'downloaded' and a['main_label'] in ('promotional', 'preventive')]
sp, st = collections.defaultdict(list), collections.defaultdict(list)
for r in sorted(rd('speech.csv'), key=lambda r: (r['ad_id'], float(r['start_second'] or 0))): sp[r['ad_id']].append(r['original_text'])
for r in sorted(rd('screen_text.csv'), key=lambda r: (r['ad_id'], float(r['start_second'] or 0))): st[r['ad_id']].append(r['original_text'])
texts = [' '.join(sp[a['ad_id']]) + ' ||| ' + ' '.join(st[a['ad_id']]) for a in ads]   # identical to m.text in image_fusion_baseline.py

path = snapshot_download(args.model, revision=args.revision,
                         ignore_patterns=['*.onnx', '*.onnx_data', 'onnx/*', 'openvino/*', '*.msgpack', '*.h5', '*.ot', 'flax_*', 'tf_*', '*.tflite', 'original/*'])
commit = os.path.basename(path.rstrip('/\\'))
torch.manual_seed(0); torch.set_num_threads(os.cpu_count() or 4)
tok = AutoTokenizer.from_pretrained(path); tok.model_max_length = 10 ** 9   # we chunk ourselves; silence the length warning
model = AutoModel.from_pretrained(path).eval()
prefix_ids = tok(args.prefix, add_special_tokens=False)['input_ids'] if args.prefix else []
n_special = len(tok.build_inputs_with_special_tokens([]))
body = args.max_len - n_special - len(prefix_ids)
assert body > 16, 'max-len too small for this prefix'


@torch.no_grad()
def embed_chunk(ids):
    inp = tok.build_inputs_with_special_tokens(prefix_ids + ids)
    x = torch.tensor([inp]); hs = model(input_ids=x, attention_mask=torch.ones_like(x)).last_hidden_state[0]
    v = hs[0] if args.pooling == 'cls' else hs.mean(0)   # batch of one chunk -> no padding, so a plain mean is the masked mean
    return torch.nn.functional.normalize(v, dim=-1).numpy(), len(ids)


E, n_chunks = [], []
for k, (a, t) in enumerate(zip(ads, texts)):
    ids = tok(t, add_special_tokens=False)['input_ids'] or tok('empty', add_special_tokens=False)['input_ids']
    parts = [embed_chunk(ids[i:i + body]) for i in range(0, len(ids), body)]
    v = sum(e * w for e, w in parts) / sum(w for _, w in parts)
    E.append(v / np.linalg.norm(v)); n_chunks.append(len(parts))
    print(f"{k + 1:3d}/{len(ads)} {a['ad_id']}  tokens={len(ids):5d}  chunks={len(parts)}", flush=True)

out_npz.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out_npz, emb=np.stack(E).astype('float32'), ad_id=np.array([a['ad_id'] for a in ads]))
json.dump(dict(model=args.model, resolved_commit=commit, pooling=args.pooling, prefix=args.prefix, max_len=args.max_len,
               torch=torch.__version__, transformers=transformers.__version__, python=platform.python_version(),
               n_ads=len(ads), emb_dim=int(E[0].shape[0]), ads_needing_chunking=int(sum(c > 1 for c in n_chunks)), max_chunks=int(max(n_chunks))),
          open(out_npz.with_suffix('.json'), 'w'), indent=1)
print('saved', out_npz)
