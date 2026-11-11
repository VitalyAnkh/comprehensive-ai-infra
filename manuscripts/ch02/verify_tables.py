from pathlib import Path
import json,re
H=Path(__file__).resolve().parent;R=H.parents[1]
md=(H.parent/'02-模型架构.md').read_text();models=[['qwen3-8b-prefill-8192'],['qwen36-prefill-8192'],['v3-forward-prefill'],['attention-deepseek-v4-flash-b1-t8192-s0','experts-deepseek-v4-flash-b64-balanced'],['k3-kda-b1-t1-s8192','k3-mla-compact-b1-t8192-s0','experts-kimi-k3-b64-balanced'],['v41-forward-decode-8192-ced']]
report=[]
for num,names in enumerate(models,1):
 dims=set()
 for name in names:
  f=R/f'calculations/results/{name}.json'
  if not f.exists():continue
  d=json.loads(f.read_text())
  def walk(x):
   if isinstance(x,dict):
    if 'input_width' in x and 'output_width' in x:dims.add((x['input_width'],x['output_width']))
    for key in ['weight_math','weight']:
     v=x.get(key)
     if isinstance(v,list) and len(v)==2 and all(isinstance(a,int) for a in v):dims.add(tuple(v))
    for v in x.values():walk(v)
   elif isinstance(x,list):
    for v in x:walk(v)
  walk(d)
 if num in (4,5):
  name='forward-deepseek-v4-flash-b1-t8192-s0' if num==4 else 'forward-kimi-k3-b1-t8192-s0-compact'
  total=json.loads((R/f'calculations/results/{name}.json').read_text())['parameter_components']['vocabulary_head']
  hidden=4096 if num==4 else 7168
  assert total%hidden==0
  dims.add((hidden,total//hidden))
  names=names+[name]
 if num==6:
  for op in d['operations']:
   if op['name'].endswith('attn.wo_a.weight'):
    width,total=op['weight_math'];dims.add((width,total//8))
 group_numeral=['一','二','三','四','五','六'][num-1]
 segment=md.split(f'**模型明细组{group_numeral}：')[1]
 segment=re.split(r'\n(?:\*\*模型明细组|### |<!-- MODEL-COMPARISON)',segment,maxsplit=1)[0]
 blocks=re.findall(r'\| 模块及作用.*?(?=\n\n)',segment,re.S)
 rows=[row for block in blocks for row in block.splitlines()[2:]]
 for row in rows:
  weight=row.split('|')[2 if num==6 else 3]
  for k,n in re.findall(r'\$(\d+)\\times(\d+)\$',weight):
   pair=(int(k),int(n));report.append({'table':num,'shape':pair,'matched':(pair in dims or ('词嵌入查行' in row and pair[::-1] in dims)),'sources':names})
(H/'table-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print('Table checks',len(report),'unmatched',[x for x in report if not x['matched']])

raise SystemExit(any(not x['matched'] for x in report))
