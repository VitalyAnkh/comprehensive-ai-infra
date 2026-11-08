#!/usr/bin/env python3
"""Check current chapter endings, references, figure indices and new derivations."""
from pathlib import Path
from urllib.parse import unquote,urlsplit
import re,json,hashlib
ROOT=Path(__file__).resolve().parents[1];M=ROOT/'manuscripts';R=ROOT/'archive/reviews/core-principles-revision-2026-09-10'
errors=[];rows=[]
def check(ok,msg):
 if not ok:errors.append(msg)
for n in range(1,25):
 p=next(M.glob(f'{n:02}-*.md'));s=p.read_text()
 headings=re.findall(r'^## (.+)$',s,re.M)
 check(headings[-1]=='本章小结' and headings.count('本章小结')==1,f'{n}: final summary')
 check(not any(any(w in h for w in ['参考资料','参考文献','资料说明','资料与','注释与','文献与','参考与']) for h in headings),f'{n}: reference sections removed')
 defined=set(re.findall(r'^\[\^([^\]]+)\]:',s,re.M));used=set(re.findall(r'\[\^([^\]]+)\](?!:)',s));check(used<=defined,f'{n}: unresolved footnotes {used-defined}')
 for url in re.findall(r'\]\(([^)]+)\)',s):
  u=urlsplit(url.strip('<>'))
  if not u.scheme and u.path:check((p.parent/unquote(u.path)).exists(),f'{n}: missing local link {url}')
 assets=re.findall(r'!\[[^\n]*\]\(([^)]+)\)',s)
 index=json.loads((M/f'ch{n:02}/figure-index.json').read_text())
 got=[r.get('asset') or f'ch{n:02}/'+(r.get('file') or r['name']+'.svg') for r in index]
 check(got==assets,f'{n}: active figure index')
 all_ex=re.findall(r'^(?:>\s*)?\*\*(?:(?:练习|实验|习题|拓展练习)\s+)?(\d+-\d+)\b(?!\s*解答)',s,re.M)
 check(len(all_ex)>0,f'{n}: has exercises')
 if n<=12:
  bp=R/'before'/p.relative_to(ROOT)
  before=(bp if bp.exists() else next((R/'before'/'manuscripts').glob(f'{n:02}-*.md'))).read_text()
  # Removing references must not remove exercises or alter their identifiers (adding new exercises is allowed).
  exercise=lambda x:re.findall(r'^> \*\*(?:练习|实验) (\d+-\d+)',x,re.M)
  check(set(exercise(before))<=set(exercise(s)),f'{n}: preserved exercise identifiers')
 rows.append(dict(chapter=n,last_section=headings[-1],figures=len(assets),exercises=len(all_ex),footnotes=len(defined)))
d=json.loads((ROOT/'calculations/results/core-principles.json').read_text())
check(d['old_static_capacity']==4 and d['new_static_capacity']==18,'capacity counterfactual')
check(abs(d['task_speedup']-3.5714285714285716)<1e-10,'task counterfactual')
for source in d['sources']:check(hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest()==source['sha256'],'counterfactual source hash')
report=dict(passed=not errors,chapters=rows,new_figures=9,total_figures=sum(r['figures'] for r in rows),total_exercises=sum(r['exercises'] for r in rows),errors=errors)
(R/'core-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2));raise SystemExit(bool(errors))
