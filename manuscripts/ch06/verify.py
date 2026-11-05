#!/usr/bin/env python3
"""Validate chapter coverage, source integrity, figure data and local references."""
from pathlib import Path
from fractions import Fraction
import hashlib, json, math, re
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
checks = []
def check(name, condition):
    checks.append({'check': name, 'passed': bool(condition)})
def read(path):
    return json.loads(path.read_text())
def close(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)
md = HERE.parent / '06-超节点.md'
s = md.read_text()
o = next(p for p in [ROOT / 'outlines/06-超节点.md', ROOT / 'archive/outlines/06-超节点.md'] if p.exists()).read_text()
headings = lambda t: re.findall(r'^#{2,3} (6\.\d+(?:\.\d+)?) ', t, re.M)
check('outline section and subsection coverage', [h for h in headings(o) if h in set(headings(s))] == [h for h in headings(s) if h in set(headings(o))])
ex = re.findall(r'^> \*\*实验 6-(\d+) · (核心|延伸)', s, re.M)
check('fifteen exercises with core 2, 3, 10', [int(x[0]) for x in ex] == list(range(1,16)) and [int(x[0]) for x in ex if x[1]=='核心'] == [2,3,10])
figs = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', s)
index=read(HERE/'figure-index.json')
check('active figures and sequential external captions', len(figs)==len(index) and re.findall(r'^\*图 6-(\d+)：',s,re.M)==[str(i) for i in range(1,len(index)+1)])
for f in figs:
    p = md.parent / f
    text = ''.join(ET.parse(p).getroot().itertext())
    check('no internal figure number: '+p.stem, re.search(r'图\s*\d+\s*[-－–]\s*\d+',text) is None)
    check('three figure formats: '+p.stem, all(p.with_suffix(ext).is_file() for ext in ['.svg','.png','.pdf']))
links = re.findall(r'\[[^\]]*\]\(([^)]+)\)', s)
missing = []
for link in links:
    u = urlsplit(link)
    if not u.scheme and u.path and not (md.parent/unquote(u.path)).exists(): missing.append(link)
check('all local manuscript link targets exist', not missing)
defs = re.findall(r'^\[\^([^\]]+)\]:',s,re.M)
refs = re.findall(r'\[\^([^\]]+)\](?!:)',s)
check('footnote references and definitions match', set(defs)==set(refs) and len(defs)==len(set(defs)))
for name,key in [('sources.json','sources'),('manifest.json','outputs')]:
    mismatches = [x['path'] for x in read(HERE/name)[key] if not (ROOT/x['path']).read_bytes().startswith(b'version https://git-lfs') and hashlib.sha256((ROOT/x['path']).read_bytes()).hexdigest()!=x['sha256']]
    check(name+' SHA256 integrity', not mismatches)
    if mismatches: checks[-1]['mismatches']=mismatches
check('formula rendering', read(HERE/'math-validation.json')['errors']==[] and read(HERE/'math-validation.json')['expressions']==len(re.findall(r'\$\$[\s\S]*?\$\$|\$[^$\n]+\$',s)))
layout=read(HERE/'teaching-layout-validation.json')+read(HERE/'parallel-layout-validation.json')+read(HERE/'ub-ep-layout-validation.json')+read(HERE/'legacy-layout-validation.json')
check('book-size figures and visible labels',len(layout)==len(figs) and all(x['width_pt']==420 and x['min_label_pt']>=11 and not x['text_extent_warnings'] for x in layout))
d = read(HERE/'figure-data.json')
check('single-card capacity arithmetic', sum(d['placement']['capacity_bytes'][0])==470187269120+192512*8192+2**31)
check('TP8 capacity matches saved placement', sum(d['placement']['capacity_bytes'][1])==d['placement']['tp8_saved_max_bytes'])
check('TP4 weight bytes', d['tp_pipeline']['tp4_weight_bytes']==187.5*2**20)
check('pipeline slots and utilization', d['tp_pipeline']['slots']==4+4-1 and close(d['tp_pipeline']['utilization'],4/7))
check('expert reuse expectation', close(d['expert_reuse']['expected_active_experts'][7],128*(1-(120/128)**8)))
routes = read(ROOT/'experiments/ch06/06-03/runs/routes-001/route-analysis.json')
rows = sorted([r for r in routes['counts'] if r['case_id']==d['expert_reuse']['route_case'] and r['phase']=='prefill'],key=lambda r:r['layer_id'])
check('heatmap uses actual 43-layer counts', len(rows)==43 and all(all(close(v,c/r['valid_tokens']) for v,c in zip(heat,r['expert_counts'])) and close(sum(heat),6) for heat,r in zip(d['expert_reuse']['route_heatmap'],rows)))
v=d['expert_reuse']['balanced_vs_concentrated']
check('equal 512 assignments and 16-fold ideal reads', all(e*r==64*8 for e,r in zip(v['active_experts'],v['rows_per_expert'])) and v['weight_read_bytes']==[4.5*2**30,288*2**20] and v['weight_read_bytes'][0]/v['weight_read_bytes'][1]==16)
check('1469 transactions are the minimum at 50 GB/s and 7.52 us', 1468*256/7.52e-6<50e9<=1469*256/7.52e-6)
check('capacity diagram includes every card workspace', all(close(x, y/1e9) for row,original in zip(d['capacity_plot']['segments_GB'],d['placement']['capacity_bytes']) for x,y in zip(row,original)))
check('expert placement preserves work and selected weights', d['expert_load']['tasks']==[[128]*4,[512,0,0,0],[128]*4] and d['expert_load']['reads_bytes'][1:]==[288*2**20]*2)
check('torus bisection grows more slowly than device count', all(n==k**3 and e==2*k*k for k,n,e in zip(d['torus_cut']['k'],d['torus_cut']['devices'],d['torus_cut']['cut_links'])))
check('memory borrowing fits physical nodes and preserves data', max(d['pool_placement']['after_assigned_GB'])<=80 and sum(d['pool_placement']['after_assigned_GB'])==240 and sum(d['pool_placement']['before_assigned_GB'])+d['pool_placement']['unassigned_GB']==240)
check('no unintended control characters', not any(ord(c)<32 and c not in '\n' for c in s))
c = d['collectives'];p=c['participants'];a=c['startup_seconds'];b=c['bandwidth_bytes_per_second']
check('ring and tree curves', all(close(r,2*(p-1)*a+2*(p-1)/p*m/b) and close(t,2*math.log2(p)*(a+m/b)) for m,r,t in zip(c['message_bytes'],c['ring_seconds'],c['tree_seconds'])))
check('10 KiB ring worked example on H100 NVLink', close((14*0.822e-6+1.75*10240/450e9)*1e6,d['continuous_execution']['collectives']['ring_s']*1e6))
check('remote window and access frequency', close(d['remote_memory']['window_bound_GBs'],128*256/7.52e-6/1e9) and all(close(v,20e9*f/1e9) for v,f in zip(d['remote_memory']['mean_payload_GBs'],d['remote_memory']['frequency_per_second'])))
continuous=read(HERE/'continuity-model.json')
check('figure uses current continuous execution model', d['continuous_execution']==continuous)
for c in continuous['candidates']:
    p=c['tp']
    # Independent formula in bytes from individual matrix shapes, not model functions.
    expected=[]
    for history in range(131064,131072):
        weights=64*2*(3*5120*25600+2*5120*8192+2*5120*1024)+151936*5120*2
        kv=2*64*8*128*2*(history+1)
        comm=128*(2*(p-1)*0.822e-6+2*(p-1)/p*10240/450e9)
        expected.append((weights+kv)/(p*3350e9)+comm)
    service=sum(expected)*1000
    ends=[(i//(8//p)+1)*service for i in range(4)]
    fault=[t+(60 if i%(8//p)==0 else 0) for i,t in enumerate(ends)]
    check('TP '+str(p)+' layer-to-step and eight-step service', all(close(x,y) for x,y in zip(c['step_times_s'],expected)) and close(c['service_ms'],service))
    check('TP '+str(p)+' independent round-robin and fault schedule', all(close(x,y) for x,y in zip(c['healthy_completion_ms'],ends)) and all(close(x,y) for x,y in zip(c['fault_completion_ms'],fault)))
    check('TP '+str(p)+' occupied GPU-seconds', close(c['healthy_gpu_seconds'],8*max(ends)/1000) and close(c['fault_gpu_seconds'],8*max(fault)/1000))
    check('TP '+str(p)+' capacity against 80 GB H100', c['capacity_fits']==(p>1))
for phase,curves in d['deadline_curves'].items():
    for curve in curves:
        c=next(c for c in continuous['candidates'] if c['tp']==curve['tp'])
        times=c[phase+'_completion_ms'];cost=c[phase+'_gpu_seconds'];valid=True
        for deadline,value in zip(curve['deadlines_ms'],curve['cost_per_valid']):
            n=sum(t<=deadline for t in times)
            valid &= (value is None) if (n<3 or not c['capacity_fits']) else (value is not None and close(value,cost/n))
        check(phase+' TP '+str(c['tp'])+' exact deadline curve',valid)
t16=continuous['two_servers']['tp16'];exp16=(63967068160/16+2*64*8*128*2*131065/8)/3350e9+128*32.74e-6
check('two-server TP16 (measured AllReduce) loses to local TP8', close(t16['total_s'],exp16) and t16['total_s']>continuous['first_steps'][3]['total_s'])
check('dense/MoE fit counts on H100', [m['requests_fit_h100']['1x131072'] for m in continuous['dense_moe']['models']]==[0,1,3] and [m['requests_fit_h100']['2x131072'] for m in continuous['dense_moe']['models']]==[2,7,31])
body=s.split('## 注释与资料')[0]
check('removed defensive paragraph endings', all(x not in body for x in ['不能只优化当前矩阵','不代表所有后端','上式只描述','不能据此','不提供故障概率']))
check('15 numbered equations in order', re.findall(r'\\tag\{6-(\d+)\}',body)==[str(i) for i in range(1,16)])
check('each exercise has progressive subquestions', len(re.findall(r'^> （a）',body,re.M))==15 and len(re.findall(r'^> （b）',body,re.M))==15 and len(re.findall(r'^> （c）',body,re.M))==15)
result={'passed':all(c['passed'] for c in checks),'checks':checks,'missing_local_links':missing,'scope':'Chapter 6 only; mathematical/data checks do not establish GPU performance.'}
(HERE/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'passed':result['passed'],'checks':len(checks),'failures':[c for c in checks if not c['passed']]},ensure_ascii=False,indent=2))
raise SystemExit(0 if result['passed'] else 1)
