"""Local web demo for the promotional-vs-preventive ad classifier (research prototype). Standard library only, ~30 MB RAM.
  python baselines/web_demo.py          then open http://127.0.0.1:8765
All numbers and per-ad predictions come from baselines/web/demo_data.json (built by build_demo_data.py), so the server does
not load scikit-learn or CLIP. Only the optional upload box starts a separate process (predict_video.py) that loads CLIP.
Temp files go to .tmp/ on the project disk. Nothing is sent over the network."""
import os, sys, json, re, pathlib, subprocess, urllib.parse, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

B = pathlib.Path(__file__).resolve().parent; ROOT = B.parent
VID = ROOT / 'ad_dataset' / 'videos'; TMP = ROOT / '.tmp'; UP = TMP / 'uploads'; UP.mkdir(parents=True, exist_ok=True)
DATA = json.load(open(B / 'web' / 'demo_data.json', encoding='utf-8'))
ADS = DATA.pop('ads'); SUMMARY = DATA

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
            self.send_json(ADS[ad] if ad in ADS else dict(error='unknown ad'), 200 if ad in ADS else 404)
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
        try:
            env = dict(os.environ, TEMP=str(TMP), TMP=str(TMP), HF_HOME=str(ROOT / '.hf_cache'), PYTHONIOENCODING='utf-8')
            p = subprocess.run([sys.executable, str(B / 'predict_video.py'), str(f), '--transcript', q.get('transcript', [''])[0], '--screen', q.get('screen', [''])[0]],
                               capture_output=True, text=True, encoding='utf-8', timeout=600, env=env, cwd=str(ROOT))
            line = [l for l in p.stdout.splitlines() if l.startswith('{')]
            if line: self.send_json(json.loads(line[-1]))
            else:
                err = (p.stderr or '')[-300:]
                msg = 'Not enough free memory to load the image model. Close other programs and try again; the held-out test ads above need no extra memory.' if ('Memory' in err or 'paging' in err) else 'Prediction failed: ' + err[-160:]
                self.send_json(dict(error=msg), 500)
        except Exception as e: self.send_json(dict(error='%s: %s' % (type(e).__name__, str(e)[:160])), 500)
        finally:
            try: f.unlink()
            except OSError: pass

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f'Research demo running at http://127.0.0.1:{port}  (Ctrl+C to stop)', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), H).serve_forever()
