"""Local web demo for the promotional-vs-preventive ad classifier (research prototype). Standard library only; works offline.
  python baselines/web_demo.py          then open http://127.0.0.1:8765
Shows: how the data was split, the models, the cross-validated results, the 15 held-out test ads with video + explanation,
and an optional upload for a brand-new video (image-only; needs ~2 GB free RAM, the CLIP model loads on first upload).
Nothing is sent over the network."""
import os, sys, json, re, pathlib, mimetypes, urllib.parse, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import numpy as np, joblib
import image_fusion_baseline as m
from fused_model import FusedModel

B = pathlib.Path(__file__).resolve().parent; ROOT = B.parent
VID = ROOT / 'ad_dataset' / 'videos'; UP = ROOT / '.tmp' / 'uploads'; UP.mkdir(parents=True, exist_ok=True)
M_TV = joblib.load(B / 'models' / 'model_trainval.joblib'); M_ALL = joblib.load(B / 'models' / 'model_all.joblib')
idx = {a['ad_id']: i for i, a in enumerate(m.ads)}; n = len(m.ads)
jl = lambda f: json.load(open(B / f, encoding='utf-8'))
RT, RI, RF, RH = jl('results_text.json'), jl('results_image_fusion.json'), jl('results_final_test.json'), jl('results_hard_subsets.json')

def neighbours(x, exclude=None, k=3):
    s = m.X_img @ x; order = [i for i in np.argsort(-s) if m.ads[i]['ad_id'] != exclude][:k]
    return [dict(id=m.ads[i]['ad_id'], label=m.ads[i]['main_label'], sim=round(float(s[i]), 2), title=m.ads[i]['video_title'][:70]) for i in order]

def ad_result(ad_id):
    i = idx[ad_id]; test = m.split[i] == 'test'; mod = M_TV if test else M_ALL
    lab, p, parts = mod.predict(m.text[i], m.X_img[i]); pos, neg = mod.top_words(m.text[i])
    return dict(id=ad_id, title=m.ads[i]['video_title'], product=m.ads[i]['product_category'], split=m.split[i], true=m.ads[i]['main_label'],
                pred=lab, p=round(p, 3), parts={k: round(v, 3) for k, v in parts.items()}, correct=lab == m.ads[i]['main_label'],
                pos=pos, neg=neg, neighbours=neighbours(m.X_img[i], ad_id), in_training=not test, has_video=(VID / f'{ad_id}.mp4').exists())

def build_summary():
    sp = {}
    for s in ('train', 'val', 'test'):
        ix = [i for i in range(n) if m.split[i] == s]
        sp[s] = dict(n=len(ix), promotional=sum(m.ads[i]['main_label'] == 'promotional' for i in ix), preventive=sum(m.ads[i]['main_label'] == 'preventive' for i in ix))
    te = [i for i in range(n) if m.split[i] == 'test']; tr = [i for i in range(n) if m.split[i] != 'test']; S = m.X_img @ m.X_img.T
    nn = [float(S[i, tr].max()) for i in te]; tt = [float(max(S[i, [j for j in tr if j != i]])) for i in tr]
    G = lambda s: {m.g[i] for i in range(n) if m.split[i] == s}; V = lambda s: {m.ads[i]['youtube_video_id'] for i in range(n) if m.split[i] == s}
    cv = lambda d: dict(f1=round(d['cv']['macro_f1_mean'], 3), sd=round(d['cv']['macro_f1_sd'], 3), acc=round(d['cv']['acc_mean'], 3))
    hs = {k: {mm: round(v[mm]['acc'], 2) for mm in ('text', 'image', 'text+image')} | dict(n=v['n']) for k, v in RH.items() if isinstance(v, dict) and 'n' in v}
    return dict(n_ads=n, splits=sp, test_confusion=RF['fused']['confusion_rows_true_prev_promo'],
                test=dict(fused=RF['fused'], image=RF['image'], text=RF['text']),
                cv=dict(majority=cv(RT['majority']), transcript=cv(RT['speech']), screen=cv(RT['screen']), text=cv(RT['speech+screen']), image=cv(RI['image']), fused=cv(RI['text+image'])),
                hard=hs, disjoint=dict(shared_groups=len(G('test') & (G('train') | G('val'))), shared_videos=len(V('test') & (V('train') | V('val'))),
                                       test_to_train_median=round(float(np.median(nn)), 2), test_to_train_max=round(max(nn), 2),
                                       train_to_train_median=round(float(np.median(tt)), 2), train_to_train_p90=round(float(np.percentile(tt, 90)), 2)),
                test_ads=[ad_result(m.ads[i]['ad_id']) for i in te],
                other_ads=[dict(id=m.ads[i]['ad_id'], title=m.ads[i]['video_title'][:60], split=m.split[i]) for i in range(n) if m.split[i] != 'test'])
SUMMARY = build_summary()

def predict_upload(path, transcript, screen):
    from clip_embed import embed
    x = embed(path); txt = (transcript + ' ||| ' + screen) if (transcript.strip() or screen.strip()) else None
    lab, p, parts = M_ALL.predict(txt, x); pos, neg = M_ALL.top_words(txt) if txt else ([], [])
    return dict(pred=lab, p=round(p, 3), parts={k: round(v, 3) for k, v in parts.items()}, pos=pos, neg=neg, neighbours=neighbours(x), image_only=txt is None)

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send_json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode('utf-8'); self.send_response(code); self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
        if u.path in ('/', '/index.html'):
            b = (B / 'web' / 'index.html').read_bytes(); self.send_response(200); self.send_header('Content-Type', 'text/html; charset=utf-8'); self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)
        elif u.path == '/api/summary': self.send_json(SUMMARY)
        elif u.path == '/api/ad':
            ad = q.get('id', [''])[0]
            self.send_json(ad_result(ad) if ad in idx else dict(error='unknown ad'), 200 if ad in idx else 404)
        elif u.path.startswith('/video/'):
            f = VID / (re.sub(r'[^A-Za-z0-9_]', '', u.path[7:].replace('.mp4', '')) + '.mp4')
            if not f.exists(): self.send_error(404); return
            size = f.stat().st_size; r = re.match(r'bytes=(\d*)-(\d*)', self.headers.get('Range', '')); a, z = 0, size - 1
            if r: a = int(r.group(1) or 0); z = int(r.group(2)) if r.group(2) else size - 1
            z = min(z, size - 1); self.send_response(206 if r else 200); self.send_header('Content-Type', 'video/mp4'); self.send_header('Accept-Ranges', 'bytes')
            if r: self.send_header('Content-Range', f'bytes {a}-{z}/{size}')
            self.send_header('Content-Length', str(z - a + 1)); self.end_headers()
            with open(f, 'rb') as fh:
                fh.seek(a); left = z - a + 1
                while left > 0:
                    chunk = fh.read(min(1 << 16, left))
                    if not chunk: break
                    try: self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError): return
                    left -= len(chunk)
        else: self.send_error(404)
    def do_POST(self):
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
        if u.path != '/api/upload': self.send_error(404); return
        ln = int(self.headers.get('Content-Length', 0))
        if ln <= 0 or ln > 300 * 1024 * 1024: self.send_json(dict(error='file missing or larger than 300 MB'), 400); return
        f = UP / f'upload_{int(time.time())}.mp4'
        with open(f, 'wb') as fh:
            left = ln
            while left > 0:
                c = self.rfile.read(min(1 << 20, left))
                if not c: break
                fh.write(c); left -= len(c)
        try: self.send_json(predict_upload(f, q.get('transcript', [''])[0], q.get('screen', [''])[0]))
        except (MemoryError, OSError) as e:
            self.send_json(dict(error='Not enough free memory to load the image model (%s). Close other programs (browser tabs, editors) and try again; the held-out test ads above need no extra memory.' % str(e)[:80]), 500)
        except Exception as e: self.send_json(dict(error='%s: %s' % (type(e).__name__, str(e)[:160])), 500)
        finally:
            try: f.unlink()
            except OSError: pass

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f'Research demo running at http://127.0.0.1:{port}  (Ctrl+C to stop)', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), H).serve_forever()
