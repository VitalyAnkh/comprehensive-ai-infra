#!/usr/bin/env python3
"""Validate chapter coverage, evidence, arithmetic, figures and offline rendering."""
from pathlib import Path
from fractions import Fraction
from urllib.parse import unquote,urlsplit
import hashlib,html,json,re,sys,xml.etree.ElementTree as ET
from PIL import Image
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE.parent))
from preview_output import preview_path
md=HERE.parent/'08-推理优化.md';raw=md.read_text();page=preview_path(md).read_text();errors=[];checks=0
def check(ok,msg):
 global checks
 checks+=1
 if not ok:errors.append(msg)
def load(p):return json.loads(p.read_text())
outline=next(p for p in [ROOT/'outlines'/md.name,ROOT/'archive/outlines'/md.name,ROOT/'archive/outlines/08-单实例推理.md'] if p.exists()).read_text()
check(re.findall(r'^#{2,3} (8\.\d+(?:\.\d+)?) ',raw,re.M)==re.findall(r'^#{2,3} (8\.\d+(?:\.\d+)?) ',outline,re.M),'Outline section numbering/order differs')
check(re.findall(r'^> \*\*练习 (8-\d+)',raw,re.M)==[f'8-{i}' for i in range(1,14)],'Exercise sequence')
check(re.findall(r'^> \*\*练习 (8-\d+) · 核心',raw,re.M)==['8-2','8-4','8-9'],'Core selection')
figure_count=len(load(HERE/'figure-index.json'))
check(re.findall(r'^\*图 (8-\d+)[：　]',raw,re.M)==[f'8-{i}' for i in range(1,figure_count+1)],'External caption sequence')
check(len(re.findall(r'^!\[',raw,re.M))==figure_count,'Image count')
refs=re.findall(r'\[\^([^\]]+)\](?!:)',raw);defs=re.findall(r'^\[\^([^\]]+)\]:',raw,re.M);check(set(refs)==set(defs),'Footnote coverage');check(len(defs)==len(set(defs)),'Duplicate footnote')
for u in re.findall(r'\]\(([^)]+)\)',raw):
 if re.match(r'\w+:',u):continue
 parts=urlsplit(u);p=(md.parent/unquote(parts.path)).resolve()
 check(p.exists(),'Missing link '+u)
for p in sorted(HERE.glob('figure-*.svg'))+sorted(HERE.glob('ch08-*.svg')):
 tree=ET.parse(p);texts=[''.join(t.itertext()) for t in tree.iter() if t.tag.endswith('}text')];check(not any(re.search(r'图\s*\d+[-－]\d+',t) for t in texts),'Number inside illustration '+p.name)
 for ext in ['png','pdf']:check(p.with_suffix('.'+ext).exists(),'Missing image format '+p.name)
 with Image.open(p.with_suffix('.png')) as im:im.verify()
for row in load(HERE/'sources.json')['sources']:check(hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256'],'Source hash '+row['path'])
for row in load(HERE/'manifest.json')['outputs']:check(hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256'],'Output hash '+row['path'])
layout=load(HERE/'teaching-layout-validation.json')
check(len(layout)==figure_count and all(x['width_pt']==420 and x['min_label_pt']>=11 and not x['text_extent_warnings'] for x in layout),'Book-size readable figures')
check('MATHPLACEHOLDER' not in page and 'katex-error' not in page,'Unrendered math');check(page.count('src="data:image/png;base64,')==figure_count,'Offline image embedding');check('url(fonts/' not in page,'External math font')
ids=re.findall(r'\bid="([^"]+)"',page);check(len(ids)==len(set(ids)),'Duplicate HTML IDs')
for ref in re.findall(r'href="#([^"]+)"',page):check(html.unescape(unquote(ref)) in ids,'HTML anchor '+ref)
# Independent small calculations supporting the prose; compare figure payloads to archived data.
check(2*36*8*128*2==147456,'KV bytes per token');check(8192*147456==1152*2**20,'8K KV')
check(8192*2*36*8*128//32*34==612*2**20,'q8_0 metadata');check(8192*2*36*8*128//32*18==324*2**20,'q4_0 metadata')
check((15136811008+2048*147456-1)//(2048*147456)==51,'2K crossover');check((15136811008+8192*147456-1)//(8192*147456)==13,'8K crossover')
check(512*7680+512*513//2==4063488,'Causal pairs');check(95+4*9==131 and 4*(95+9)==416,'Page references')
check(69178275840+16506720256+6006016==85691002112,'GGUF byte conservation')
check(96000000000-8*2**30-85691002112>=1577058304,'Q2_K 8K necessary capacity');check(96000000000-8*2**30-85691002112<6308233216,'Q2_K 32K capacity fails')
d=load(HERE/'figure-data.json');check(sum((L+3)//4*4 for L in d['pages']['lengths'])==52,'Page allocation');check(52-8==d['pages']['shared'],'Page sharing')
for row in d['speculation']['drafts']:
 a=Fraction(1 if row['draft']=='AAAA' else 3,4);n=sum(a**j for j in range(5));check(float(n)==row['expected_output'],'Expected output');check(abs((d['speculation']['verification_ms']+d['speculation']['query_ms'])/float(n)-row['ms_per_output'])<1e-9,'Cost per output')
service=load(ROOT/'experiments/ch08/08-09/analysis.json');check(sum(r['qualified'] for r in service['groups'])==155,'Qualified results');check(sum(r['correct'] for r in service['groups'])==294,'Correct results');check(sum(r['requests'] for r in service['groups'])==336,'Request total')
for policy in ['fixed','continuous','chunked']:check(d[policy]==load(ROOT/f'calculations/results/iteration-batching-pro6000-{policy}.json')['batching_steps'],'Exact time bars '+policy)
check(d['chunk_history']==load(ROOT/'calculations/results/chunk-history-book.json')['chunk_history_rows'],'Exact chunk samples')
# Recompute authored design choices and compare task-aligned figure cells to source rows.
import subprocess
subprocess.run([sys.executable,str(HERE/'teaching-check.py')],check=True,capture_output=True,text=True)
check(load(HERE/'teaching-validation.json')['passed'],'Teaching design arithmetic')
quality=load(ROOT/'experiments/ch08/08-08/results/summary.json');control=load(ROOT/'experiments/ch08/08-08/results/q-control-summary.json')
sets=[[r for r in c['requests'] if r['mode']=='natural' and r['trial']!='warm'] for c in quality['configurations']]+[[r for r in control['records'] if r['mode']=='natural']]
for records,matrix in zip(sets,d['kv']['correct_by_task']):
 index={(r['task_id'],r['concurrency'],r['trial']):int(bool(r.get('strict_correct',r.get('correct')))) for r in records}
 expected=[[index[(task,c,t)] for c,t in d['kv']['columns']] for task in d['kv']['task_ids']]
 check(matrix==expected,'Task-aligned quality cells')
check(d['batch']['shared_weight_bytes']==15136811008 and d['batch']['kv_bytes_per_position']==147456,'Batch curve input')
# New mechanism diagrams must agree with the examples they explain.
life=d['lifecycle']
check(abs(life['queue_s']+life['prefill_s']+life['output_intervals']*life['interval_s']-life['completion_s'])<1e-12,'Lifecycle timeline arithmetic')
for row in d['attention_geometry']:
 c=row['new'];h=row['history']
 check(row['history_pairs']==c*h and row['within_pairs']==c*(c+1)//2 and row['total_pairs']==c*h+c*(c+1)//2,'Attention cell geometry')
check(d['kv_layout']['total_bytes']==[64,34,18] and [x+y for x,y in zip(d['kv_layout']['payload_bytes'],d['kv_layout']['scale_bytes'])]==[64,34,18],'KV byte layout')
cache=d['cache_choice'];worked=load(HERE/'teaching-validation.json');check(cache['B']['count']*cache['B']['each_mib']==cache['A']['size_mib']==cache['capacity_mib'] and abs(cache['A']['net_ms']-worked['cache']['A_gen5'][2]*1e3)<1e-9 and abs(cache['B']['each_net_ms']-worked['cache']['B_gen5'][2]*1e3)<1e-9,'Equal-capacity cache comparison')
example=d['verification_example'];check(example['output']==example['draft'][:example['first_mismatch']-1]+[example['replacement']],'Draft correction diagram')
reference={r['name']:r for r in load(HERE/'teaching-validation.json')['design_candidates']}
for r in d['design_plane']['configurations']:
 check(all(abs(r[k]-reference[r['name']][k])<1e-12 for k in ['memory_gib','time_s']),'Configuration diagram '+r['name'])
report={'passed':not errors,'checks':checks,'sections':6,'subsections':22,'exercises':13,'figures':figure_count,'formulas':load(HERE/'math-validation.json')['expressions'],'chinese_characters':len(re.findall(r'[\u4e00-\u9fff]',raw)),'errors':errors}
(HERE/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2));sys.exit(bool(errors))
