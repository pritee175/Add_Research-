"""Round 4 candidate search (metadata only, nothing is downloaded).
Targets the gaps listed in PROJECT_STATUS.md Section 0: preventive ads in commercial style, preventive pan masala/gutka,
promotional tobacco, promotional ads in PSA/CSR style. Skips videos already in ads.csv or candidates_all.csv.
Output: ad_dataset/candidates/candidates_round4.csv (decision column left blank for human review).
Usage: python scripts/search_candidates_round4.py"""
import csv, json, subprocess, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / 'ad_dataset'
seen = {r['youtube_video_id'] for r in csv.DictReader(open(D / 'candidates/candidates_all.csv', encoding='utf-8-sig'))}
seen |= {r['youtube_video_id'] for r in csv.DictReader(open(D / 'annotations/ads.csv', encoding='utf-8')) if r['youtube_video_id']}
Q = {
 'preventive_alcohol_commercial': [
  'drink and drive ad film India Hindi', 'dont drink and drive advertisement India agency', 'responsible drinking campaign India film',
  'Diageo India responsible drinking campaign', 'Pernod Ricard India responsible drinking', 'Carlsberg India drink responsibly campaign',
  'drunk driving awareness ad Bangalore traffic police', 'Hyderabad police drunk driving awareness film', 'Chennai traffic police drink drive ad',
  'Kolkata police drink drive awareness ad', 'sharab peekar gaadi mat chalao vigyapan', 'alcohol harm awareness ad India Hindi short ad'],
 'preventive_pan_masala_gutka': [
  'gutka ban awareness ad India Hindi', 'pan masala cancer awareness ad Hindi', 'gutkha khane se cancer vigyapan', 'mouth cancer gutka awareness ad Marathi',
  'tobacco chewing awareness ad Tamil', 'pan masala harmful to health public service ad India', 'khaini gutka tobacco cancer ad Gujarati', 'supari paan masala cancer awareness video India'],
 'promotional_tobacco': [
  'Gold Flake cigarette advertisement India old', 'Four Square cigarette ad India', 'Charms cigarettes ad India', 'Classic cigarettes ad India',
  'Panama cigarettes India ad', 'Wills Lifestyle cigarette ad India', 'Scissors cigarette ad India', 'Capstan cigarette ad India old', 'bidi advertisement India old tvc', 'cigarette tvc india 1990s doordarshan'],
 'promotional_psa_or_csr_style': [
  'Rajnigandha CSR film', 'Vimal pan masala social message ad', 'Kamla Pasand social message film', 'Royal Stag Large short films',
  'McDowells soda social message ad', 'Imperial Blue social message ad', 'Bagpiper soda social ad', 'Blenders Pride fashion tour film']}
rows, ids = [], set(seen)
for cat, qs in Q.items():
    for q in qs:
        try:
            p = subprocess.run(['yt-dlp', f'ytsearch12:{q}', '--flat-playlist', '--dump-json', '--no-warnings', '--skip-download'],
                               capture_output=True, text=True, encoding='utf-8', timeout=120)
        except subprocess.TimeoutExpired:
            print('timeout', q, file=sys.stderr); continue
        n = 0
        for line in p.stdout.splitlines():
            try: j = json.loads(line)
            except Exception: continue
            v = j.get('id')
            if not v or v in ids: continue
            dur = j.get('duration') or 0
            if dur and (dur < 8 or dur > 600): continue
            ids.add(v); n += 1
            rows.append(dict(target=cat, youtube_video_id=v, youtube_url=f'https://youtu.be/{v}', duration_s=int(dur) if dur else '',
                             title=j.get('title', ''), channel=j.get('channel') or j.get('uploader') or '', query=q, decision=''))
        print(f'{cat[:22]:22} {q[:55]:55} +{n}', flush=True)
out = D / 'candidates' / 'candidates_round4.csv'
with open(out, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(len(rows), 'new candidates ->', out)
