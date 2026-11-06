"""Build a blind, self-contained annotation sheet (ad_dataset/annotation_sheet.html).

Shows video, transcript and screen text only. Never embeds main_label, existing
annotator answers, source_type, title or search_target. Exports rows in the
annotators.csv format.  Usage: python build_annotation_sheet.py
"""
import csv, json, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
ann = root / 'annotations'
rd = lambda n: list(csv.DictReader(open(ann / n, encoding='utf-8')))
ads = [a for a in rd('ads.csv') if a['video_status'] == 'downloaded' and a['main_label'] != 'excluded']
by = lambda rows: {k: [r for r in rows if r['ad_id'] == k] for k in {r['ad_id'] for r in rows}}
sp, st = by(rd('speech.csv')), by(rd('screen_text.csv'))
data = []
for a in ads:
    i = a['ad_id']
    data.append(dict(
        id=i, video=a['video_file'], dur=a['duration_seconds'], cat=a['product_category'],
        brand=a['brand'], shown=a['shown_product'], implied=a['implied_product'],
        warn=a['statutory_warning_present'],
        speech=[[r['start_second'], r['original_text']] for r in sp.get(i, [])],
        screen=[[r['start_second'], r['text_type'], r['original_text']] for r in st.get(i, [])]))
tpl = r'''<!doctype html><html><head><meta charset="utf-8"><title>Annotation Sheet</title>
<style>body{font:15px system-ui;margin:0;display:flex;height:100vh}#l{width:150px;overflow:auto;border-right:1px solid #ccc}
#l div{padding:6px 10px;cursor:pointer}#l .d{background:#d7f5d7}#l .c{font-weight:bold;outline:2px solid #36c}
#m{flex:1;overflow:auto;padding:14px}video{max-width:560px;width:100%}.b{background:#f4f4f4;padding:8px;margin:8px 0;max-height:170px;overflow:auto;white-space:pre-wrap}
label{display:block;margin:3px 0}textarea{width:100%;height:70px}button{padding:6px 14px;margin-right:8px}</style></head><body>
<div id="l"></div><div id="m"></div>
<script>
const D=__DATA__;let cur=0;const S=JSON.parse(localStorage.ann2||'{}');
const L=[['promotional','Main message makes product/brand attractive, encourages trial/purchase or normalizes use (a small required warning does not change this)'],
['preventive','Main message warns about harm, discourages use or supports quitting'],
['unclear_mixed','Substantial messages both ways, or purpose genuinely unclear (explain in reason)'],
['excluded','Not an ad/campaign, duplicate, incomplete or too poor to judge (explain in reason)']];
function save(){localStorage.ann2=JSON.stringify(S)}
function list(){l.innerHTML=D.map((x,i)=>`<div class="${S[x.id]&&S[x.id].lab?'d':''} ${i==cur?'c':''}" onclick="cur=${i};show()">${x.id}</div>`).join('')}
function show(){const x=D[cur],s=S[x.id]||{};list();
m.innerHTML=`<h3>${x.id} <small>(${x.dur}s)</small></h3><video controls src="${x.video}"></video>
<div><b>Product:</b> ${x.cat} | <b>Brand:</b> ${x.brand} | <b>Shown:</b> ${x.shown} | <b>Implied:</b> ${x.implied} | <b>Warning (auto, confirm):</b> ${x.warn}</div>
<b>Speech</b><div class="b">${x.speech.map(r=>`[${r[0]}s] ${r[1]}`).join('\n')||'(none)'}</div>
<b>Screen text</b><div class="b">${x.screen.map(r=>`[${r[0]}s ${r[1]}] ${r[2]}`).join('\n')||'(none)'}</div>
${L.map(o=>`<label><input type=radio name=lab value=${o[0]} ${s.lab==o[0]?'checked':''} onchange="up()"> <b>${o[0]}</b> - ${o[1]}</label>`).join('')}
<p>Confidence: ${['high','medium','low'].map(c=>`<label style="display:inline"><input type=radio name=conf value=${c} ${s.conf==c?'checked':''} onchange="up()"> ${c}</label>`).join(' ')}</p>
<textarea id=why placeholder="Reason (cite what you saw/heard)" oninput="up()">${s.why||''}</textarea><p>
<button onclick="go(-1)">Prev</button><button onclick="go(1)">Next</button><button onclick="exp()">Export CSV</button> Done: ${Object.values(S).filter(v=>v.lab).length}/${D.length}</p>`}
function up(){const x=D[cur],v=n=>(document.querySelector(`input[name=${n}]:checked`)||{}).value||'';
S[x.id]={lab:v('lab'),conf:v('conf'),why:document.getElementById('why').value};save();list()}
function go(d){cur=Math.max(0,Math.min(D.length-1,cur+d));show()}
function exp(){const q=t=>'"'+String(t).replace(/"/g,'""')+'"';
const r=['ad_id,annotator_id,chosen_label,confidence,reason,agreed_flag,final_label,adjudicator_id'];
D.forEach(x=>{const s=S[x.id];if(s&&s.lab)r.push([x.id,'ann2',s.lab,s.conf,q(s.why),'','',''].join(','))});
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([r.join('\n')],{type:'text/csv'}));a.download='annotators_ann2.csv';a.click()}
show();
</script></body></html>'''
out = root / 'annotation_sheet.html'
out.write_text(tpl.replace('__DATA__', json.dumps(data, ensure_ascii=False)), encoding='utf-8')
print(len(data), 'ads ->', out)
