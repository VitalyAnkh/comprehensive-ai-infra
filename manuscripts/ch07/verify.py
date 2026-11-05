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
md = HERE.parent / '07-数据中心网络.md'
s = md.read_text()
o = next(p for p in [ROOT / 'outlines/07-数据中心网络.md', ROOT / 'archive/outlines/07-数据中心网络.md'] if p.exists()).read_text()
headings = lambda t: re.findall(r'^#{2,3} (7\.\d+(?:\.\d+)?) ', t, re.M)
check('outline section and subsection coverage', headings(s) == headings(o))
ex = re.findall(r'^> \*\*实验 7-(\d+) · (核心|延伸|实战)', s, re.M)
check('eighteen exercises with core 5, 10, 15', [int(x[0]) for x in ex] == list(range(1,19)) and [int(x[0]) for x in ex if x[1]=='核心'] == [5,10,15])
figs = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', s)
index=read(HERE/'figure-index.json')
check('active figures and sequential external captions',len(figs)==len(index) and re.findall(r'^\*图 7-(\d+)[：\s　]',s,re.M)==[str(i) for i in range(1,len(index)+1)])
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
    mismatches = [x['path'] for x in read(HERE/name)[key] if hashlib.sha256((ROOT/x['path']).read_bytes()).hexdigest()!=x['sha256']]
    check(name+' SHA256 integrity', not mismatches)
    if mismatches: checks[-1]['mismatches']=mismatches
check('formula rendering', read(HERE/'math-validation.json')['errors']==[] and read(HERE/'math-validation.json')['expressions']==len(re.findall(r'\$\$[\s\S]*?\$\$|\$[^$\n]+\$',s)))
layout=read(HERE/'teaching-layout-validation.json')+read(HERE/'ub-ep-layout-validation.json')
check('book-size visible text',len(layout)==len(figs) and all(x['width_pt']==420 and x['min_label_pt']>=11 and not x['text_extent_warnings'] for x in layout))
d = read(HERE/'figure-data.json')

check('cut bound and strong scaling', close(d['7-2']['cut_ms'],360*2**20/50e9*1000) and all(close(t,20/n) for n,t in zip(d['7-2']['device_multipliers'],d['7-2']['compute_ms'])))
for i,name in enumerate(d['7-3']['sources']):
    z=read(ROOT/('calculations/results/'+name+'.json'))['summary']
    check('gradient ledger: '+name, close(d['7-3']['local_MiB'][i],z['local_send_bytes']/2**20) and close(d['7-3']['remote_MiB'][i],z['remote_send_bytes']/2**20) and close(d['7-3']['lower_ms'][i],float(Fraction(z['serial_barrier_lower_seconds_exact']))*1000) and close(d['7-3']['local_MiB'][i]+d['7-3']['remote_MiB'][i],5760))
check('state sharing retains endpoint and relation capacity', all(close(sum(row)*2**20,total) for row,total in zip(d['7-11']['state_MiB'],[64*256+64*128*(64+1024),64*256+64*128*64+128*1024,64*256+64*128*64+8*128*1024])))
order=read(ROOT/'calculations/results/operation-ordering-book.json')
check('ordering schedules match saved source',d['7-12']['schedules']=={k:order['request_schedules'][k] for k in ['strict','necessary_dependencies']})
check('ordering completion arithmetic',d['7-12']['schedules']['strict']['finish_ns']==(20+80+2+10)*1000 and d['7-12']['schedules']['necessary_dependencies']['finish_ns']==max(20+80+2,10)*1000)
check('stale read requires fresh sampling',not d['7-12']['stale_read_example']['response_order_only_valid'] and d['7-12']['stale_read_example']['repaired_delivery_ns']==6000)
for name,segs in d['7-15']['periodic_segments'].items():
    check('queue source: '+name,segs==read(ROOT/('calculations/results/'+name+'.json'))['queue_segments'])
    check('queue conservation: '+name,all(close(r['queue_end_bytes']-r['queue_start_bytes'],float(Fraction(r['arrived_exact_bytes'])-Fraction(r['served_exact_bytes'])-Fraction(r.get('dropped_exact_bytes','0')))) for r in segs))
q=d['7-15']['periodic_segments']
check('periodic queue maxima', all(close(max(r['queue_end_bytes'] for r in q[name]),target) for name,target in [('periodic-queue-aligned',1e9),('periodic-queue-staggered',0),('periodic-queue-drift',250e6)]))
fb=d['7-15']['feedback']
check('feedback saved source',fb==read(ROOT/'calculations/results/feedback-queue-overflow.json'))
check('feedback overflow and drain',close(fb['summary']['dropped_bytes'],(100e9-50e9)*20e-6+256*1024-512*1024) and close(float(Fraction(fb['summary']['last_drain_exact_ns'])),(20e-6+512*1024/(50e9-40e9))*1e9))
for name,events in d['7-15']['packets'].items():
    check('packet source: '+name,events==read(ROOT/('calculations/results/'+name+'.json'))['receive_events'])
    check('packet retention and full release: '+name,all(r['retained_bytes']==len(r['retained_sequences'])*4096 for r in events) and sorted(v for r in events for v in r['released'])==list(range(8)))
check('packet completion arithmetic', [float(Fraction(e[-1]['time_exact_ns'])) for e in d['7-15']['packets'].values()]==[1327.68,9327.68,21163.84])
check('window bound',math.ceil(50e9*2e-6/256)==391 and close(128*256/2e-6/1e9,16.384))
check('readiness critical path',close(max(d['7-19']['ready_ms'])+d['7-19']['exchange_ms'],2.4))
check('ring startup and serialization',all(close(value,(14*5e-6+1.75*m/b)*1e6) for row,b in zip(d['7-19']['ring_us'],[50e9,150e9]) for value,m in zip(row,[8192,8*2**20])))
measured=read(ROOT/'experiments/ch07/07-10/rank-readiness/results/summary.json')
check('CPU measurements copied without reinterpretation',d['7-19']['measured']==[r for r in measured if r['bytes']==4096])

# Independent checks for the integrated teaching cases and design boundaries.
x=read(HERE/'teaching-data.json');p=x['primary'];step=x['step']
check('six worked examples in order',re.findall(r'\*\*例题 7\.(\d+)：',s)==list('123456'))
prose=s.split('## 文献与数据说明')[0]
check('first-person design history', '笔者' in prose and '作者' not in prose)
check('removed editorial hedging',not any(t in prose for t in ['不代表所有','本章的明确基线','不能只优化','尚不能','仍需验证','但是否能实现']))
check('primary model reproduces saved round calculations',close(p['flat_s'],p['saved_s']['gradient-fp32-flat-contiguous-nic8']) and close(p['interleaved_s'],p['saved_s']['gradient-fp32-flat-interleaved-nic8']) and close(p['hier_s'],p['saved_s']['gradient-fp32-hierarchical-nic8']) and close(p['hier_one_nic_s'],p['saved_s']['gradient-fp32-hierarchical-nic1']))
m=2**20
check('book hierarchy wins with every NIC busy',p['hier_s']<p['flat_s'] and close(p['hier_s'],14*(24*m/450e9+833e-9)+2*(12*m/50e9+833e-9)) and close(p['flat_s'],30*(12*m/50e9+833e-9)) and p['local_crossover_Bps']<50e9)
c=x['contrast'];b=c['local_crossover_Bps']
check('contrast model reproduces saved round calculations',close(c['flat_s'],c['saved_s']['gradient-fp32-flat-contiguous-two-nic']) and close(c['hier_s'],c['saved_s']['gradient-fp32-hierarchical-two-nic']) and close(c['flat_one_nic_s'],c['saved_s']['gradient-fp32-flat-contiguous-one-nic']) and close(c['flat_three_nic_s'],c['saved_s']['gradient-fp32-flat-contiguous-three-nic']))
check('hierarchy crossover equates both schedules',close(6*48*m/b+192*m/32e9+8*833e-9,336*m/32e9+14*833e-9) and c['hier_s']>c['flat_s'] and c['hier_local_bridge_s']<c['flat_local_bridge_s'] and c['local_Bps']<b<c['local_bridge_Bps'])
check('scaling crossover',close(x['scaling']['crossover_device_multiplier'],.020/(360*m/50e9)))
check('message crossover balances startup and payload',close(x['small_messages']['single_message_crossover_bytes']/50e9,5e-6) and close(1.75*x['small_messages']['ring_crossover_bytes']/50e9,70e-6))
check('pipeline 90 percent integer boundary',26/29<.9 and 27/30>=.9 and x['pipeline']['microbatches_for_90pct']==27)
r=x['snapshot'];a=r['remote_per_read_s'];b=r['local_per_read_s'];c=r['setup_s']
check('snapshot integer choice boundaries',a<c+b and 2*a>c+2*b and 10*.1*a<c+10*.1*b and 11*.1*a>c+11*.1*b and r['full_first_winning_integer']==2 and r['ten_pct_first_winning_integer']==11)
i=x['isolation'];check('isolation integer capacity boundary',i['fixed_bytes']+3*i['per_class_bytes']<=i['budget_bytes']<i['fixed_bytes']+4*i['per_class_bytes'])
check('reclaim sustained rate',close(x['reclaim']['sustainable_Bps'],4*8192/20e-6) and close(x['reclaim']['required_interval_s'],8192/50e9))
check('feedback and phase budgets',close(x['queue']['max_overlap_s']*50e9,512*1024) and close(x['queue']['feedback_max_s']*50e9,256*1024))
check('multipath crossover',close(x['multipath']['balanced_s']+x['multipath']['delay_difference_boundary_s'],x['multipath']['single_s']))
check('full step critical paths',close(step['flat_serial_s'],.022+p['flat_s']) and close(step['flat_ready_17ms_s'],.019+p['flat_s']) and close(step['hier_ready_17ms_s'],max(.020,.017+p['hier_s'])+.002) and close(step['hier_ready_17ms_s'],.022) and close(step['hier_one_nic_ready_17ms_s'],.017+14*(24*m/450e9+833e-9)+2*(96*m/50e9+833e-9)+.002))
check('window limit propagates to whole step',close(step['hier_window_limited_serial_s'],.022+14*(24*m/450e9+833e-9)+2*(12*m/(128*256/2e-6)+833e-9)))
check('ring mechanism shows cross-server edges', [sum((a<8)!=(b<8) for a,b in zip(order,order[1:]+order[:1])) for order in d['7-3']['ring_orders']]==[2,16] and d['7-3']['nics_used_per_server']==[1,8,8])

# New mechanism figures must agree with independent arithmetic or saved events.
check('pipeline diagram schedule',d['7-4']['start_ms']==[[i+j for j in range(8)] for i in range(4)] and d['7-4']['finish_ms']==11)
check('expert ingress diagram bounds',all(close(t,32*2**20/b*1000) for t,b in zip(d['7-5']['lower_ms'],[50e9,400e9,32e9])))
check('relay diagram shared local bound',d['7-6']['pre_downstream_GBs']==50+min(450,100))
check('snapshot diagram crossings',all(close(v['crossover'],r['setup_s']/(v['fraction']*(r['remote_per_read_s']-r['local_per_read_s']))) for v in d['7-8']['cases'].values()))
check('window diagram idle intervals',close(d['7-9']['periods']['128']['send_batch_us'],.65536) and close(d['7-9']['periods']['128']['cycle_us'],2) and close(d['7-9']['periods']['391']['cycle_us'],2.00192))
check('buffer ownership diagram',d['7-10']['source_reusable_us']==5 and d['7-10']['consumer_start_us']==8 and d['7-10']['destination_reusable_us']==12)
check('stale sampling diagram events',d['7-13']['sample_us']<d['7-13']['update_us']<d['7-13']['flag_visible_us']<d['7-13']['flag_read_us']<d['7-13']['stale_delivery_us']<d['7-13']['repaired_delivery_us'])
check('reclaim diagrams preserve saved events',all(ops==read(ROOT/('calculations/results/'+name+'.json'))['completion_operations'] for name,ops in d['7-14']['operations'].items()))
check('feedback diagram preserves saved segments',d['7-16']['queue_segments']==read(ROOT/'calculations/results/feedback-queue-overflow.json')['queue_segments'])
check('multipath diagrams preserve arrival and delivery',all(v=={k:read(ROOT/('calculations/results/'+name+'.json'))[k] for k in ['transmissions','delivery']} for name,v in d['7-17']['cases'].items()))
check('step diagram starts update after both dependencies',all(close(v['update_start_ms'],max(20,v['ready_ms']+v['comm_ms'])) and close(v['finish_ms'],v['update_start_ms']+2) for v in d['7-20']['cases']))
check('message curves preserve startup and payload',all(close(t,70+1.75*m/50e9*1e6) for m,t in zip(d['7-21']['input_bytes'],d['7-21']['baseline_us'])))
# Switch-network additions (2026-09-11) must agree with saved results and the chapter's formulas.
cl=d['7-clos'];m=2**20
check('clos tiers, bisection and partition cut',cl['tiers']['nonblocking'][0]['endpoints']==64*32 and cl['tiers']['nonblocking'][1]['endpoints']==64**3//4 and float(Fraction(cl['tiers']['nonblocking'][0]['bisection_bytes_per_second_exact']))==1024*50e9 and cl['partition_cuts']['oversub3'][0]['cut_links']==22*16 and close(float(Fraction(cl['partition_cuts']['oversub3'][0]['cut_bytes_per_second_per_endpoint_exact'])),352*50e9/1024) and all(close(float(Fraction(r['oversubscribed_egress_bytes_per_second_exact'])),r['supernode_gpus']*50e9/3) for r in cl['supernode_egress']['oversub3']))
rl=d['7-rails']['summary']
check('rail pairing keeps aligned bytes off the spine',rl['aligned']['spine_crossing_bytes']==0 and rl['shifted']['spine_crossing_bytes']==8*24*m and close(float(Fraction(rl['aligned']['per_rail_seconds_exact'])),2*(12*m/50e9+833e-9)) and close(float(Fraction(rl['aligned']['single_nic_counterfactual_seconds_exact'])),2*(96*m/50e9+833e-9)))
check('in-network sweep arithmetic',all(close(r['ring_seconds'],2*(r['servers']-1)*(24*m/r['servers']/50e9+833e-9)) and close(r['switch_seconds'],24*m/50e9+833e-9) for r in d['7-in-network']['sweep']))
ec=d['7-ecmp']
check('ecmp effective cut and spraying',all(close(v['effective_fraction_of_nominal_cut'],(v['flows']/v['uplinks'])/v['expected_max_load']) for v in ec['cases'].values()) and close(float(Fraction(ec['spray']['completion_exact_ns'])),(m/50e9+8e-6)*1e9) and close(float(Fraction(ec['spray']['single_path_serial_counterfactual_exact_ns'])),(8*m/50e9+1e-6)*1e9))
check('incast feedback arithmetic',all(close(r['allowed_feedback_ns'],m/((r['N']-2)*50e9)*1e9) and close(r['required_buffer_bytes']['one_hop_pause']['bytes'],330e-9*(r['N']-2)*50e9) and close(r['required_buffer_bytes']['end_to_end']['bytes'],20e-6*(r['N']-2)*50e9) for r in d['7-incast']['rows']))
check('standalone captions and no editing placeholders',not re.search(r'^\*图 7-\d+：.*\*\S',s,re.M) and 'NEW' not in prose)
check('caption order agrees with active index',[Path(f).stem for f in figs]==[x['name'] for x in index])

result={'passed':all(c['passed'] for c in checks),'checks':checks,'missing_local_links':missing,'scope':'Chapter 7 only; source and arithmetic checks do not establish GPU performance.'}
(HERE/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'passed':result['passed'],'checks':len(checks),'failures':[c for c in checks if not c['passed']]},ensure_ascii=False,indent=2))
raise SystemExit(0 if result['passed'] else 1)
