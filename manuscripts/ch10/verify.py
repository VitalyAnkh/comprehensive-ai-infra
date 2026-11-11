#!/usr/bin/env python3
"""Verify chapter text, source-driven figures, formula rendering and local links."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from svg_labels import svg_labels
from fractions import Fraction
from urllib.parse import unquote,urlsplit
import hashlib,json,re,math,xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
from preview_output import preview_path
md=HERE.parent/'10-训练系统.md';s=md.read_text();page=preview_path(md).read_text();errors=[]
def check(ok,msg):
 if not ok:errors.append(msg)
def calc(n):return json.loads((ROOT/'calculations/results'/f'{n}.json').read_text())
figure_count=len(json.loads((HERE/'figure-index.json').read_text()))
heads=re.findall(r'^### (10\.\d+\.\d+)',s,re.M)
outline_path=next(p for p in [ROOT/'outlines'/md.name,ROOT/'archive/outlines'/md.name] if p.exists())
expected=re.findall(r'^### (10\.\d+\.\d+)',outline_path.read_text(),re.M)
check([h for h in heads if h in set(expected)]==expected,'outline subsection alignment')
check('训练一致性' not in s and '训练一致性' not in page, 'obsolete RL consistency terminology')
for marker in ['专家路由重放与训推一致性', '前缀分布偏移', 'OPD', '温度缩放', '权重长期不变']:
 check(marker in s and marker in page, 'RL consistency coverage: '+marker)
check(re.findall(r'^> \*\*习题 (10-\d+)',s,re.M)==[f'10-{i}' for i in range(1,21)],'exercise sequence')
check(re.findall(r'^> \*\*习题 (10-\d+) · 综合设计',s,re.M)==['10-3','10-7','10-10'],'integrative design exercises')
check(re.findall(r'^\*图 (10-\d+)(?:：|　)',s,re.M)==[f'10-{i}' for i in range(1,figure_count+1)],'external caption sequence')
check(len(re.findall(r'!\[',s))==figure_count,'active figures')
for rec in json.loads((HERE/'sources.json').read_text())['sources']:
 check(hashlib.sha256((ROOT/rec['path']).read_bytes()).hexdigest()==rec['sha256'],'source hash: '+rec['path'])
manifest=json.loads((HERE/'manifest.json').read_text())
for rec in manifest['outputs']:
 check(hashlib.sha256((ROOT/rec['path']).read_bytes()).hexdigest()==rec['sha256'],'output hash: '+rec['path'])
for rec in manifest['outputs']:
 p=ROOT/rec['path']
 if p.suffix!='.svg':continue
 texts=svg_labels(p)
 check(not any(re.search(r'图\s*\d+[-—]\d+',t) for t in texts),'figure number inside '+p.name)
 check(bool(texts),'SVG has text or outlined glyph labels: '+p.name)
for u in re.findall(r'\]\(([^)]+)\)',s):
 parts=urlsplit(u)
 if parts.scheme:continue
 target=(HERE.parent/unquote(parts.path)).resolve()
 check(target.exists(),'missing link '+u)
for marker in re.findall(r'\[\^([^\]]+)\]',s):check(f'[^{marker}]:' in s,'undefined footnote '+marker)
check('MATHPLACEHOLDER' not in page,'unrendered math')
check(page.count('src="data:image/png;base64,')==figure_count,'embedded image count')
check(len(re.findall(r'<nav[^>]*>.*?</nav>',page,re.S))==1,'navigation')
mathreport=json.loads((HERE/'math-validation.json').read_text());check(not mathreport['errors'],'math errors')
layout=json.loads((HERE/'teaching-layout-validation.json').read_text());check(len(layout)==figure_count and all(not r['text_extent_warnings'] and r['width_pt']==420 and r['min_label_pt']>=11 for r in layout),'book-size figure layout')
# Recompute the key decision examples independently of drawing code.
N=8190735360;check(16*N==calc('training-state-book')['summary']['unsharded_persistent_bytes'],'Adam total')
F=calc('training-deadline-book')['summary']['task_training_matrix_flops'];check(math.ceil(F/(165.2e12*30*86400))==13,'4090 peak count')
check(math.ceil(F/(165.2e12*.4*30*86400))==31,'4090 40 percent count')
G=96*2**20;cross=1/(3*(1/307.2e9-1/1008e9));check(abs(cross/1e9-147.2876712328767)<1e-8 and round(cross/1e9)==147,'cast crossover')
for B,cpu_ms,gpu_ms in [(32e9,4.13,6.59),(450e9,1.21,.75)]:check(round((G/B+3*G/307.2e9)*1e3,2)==cpu_ms and round((3*G/1008e9+2*G/B)*1e3,2)==gpu_ms,'cast path times')
check(8192*48*8*2==6*2**20,'routing bytes')
check(Fraction(8,5)/20+Fraction(24,5)/20==Fraction(8,25),'global token normalization')
check(2*4096*4097//2==16781312,'balanced pairs');check(7168*7169//2+1024*1025//2==26218496,'skew pairs')
check(Fraction(44,60)==Fraction(11,15),'async effective threshold')
MTBF=1024*7.9*3600;c=14*N/7e9;lam=1024/MTBF;check(abs(math.sqrt(2*c/lam)-965.2865142296353)<1e-6 and round(c,1)==16.4,'checkpoint interval')
d=json.loads((HERE/'figure-data.json').read_text());check(d['10-7']['填满排空']['summary']['step_makespan_seconds']<d['10-7']['1F1B']['summary']['step_makespan_seconds'],'pipeline counterexample')
# Check revised figure conclusions against source quantities and displayed precision.
for n,shown in [(8,25.3),(16,17.6)]:
 i=d['10-6']['participants'].index(n)
 check(math.isclose(d['10-6']['total_gib'][i],16*N/n/2**30+10),'sharded capacity source')
 check(round(d['10-6']['total_gib'][i],1)==shown,'sharded capacity displayed value')
for interval,expected in [(300,[5.5,.5,.4,6.4]),(900,[1.8,1.6,.4,3.8]),(1800,[.9,3.2,.4,4.5])]:
 losses=[100*c/interval,100*lam*interval/2,100*lam*120]
 check([round(x,1) for x in losses+[sum(losses)]]==expected,'checkpoint period comparison')
for label,rows in d['10-12']['timelines'].items():
 latest=max(row['seconds']['capture'] for row in rows if row['seconds']['durable']<=d['10-12']['failure_seconds'])
 check(latest==d['10-12']['recoverable_capture_seconds'][label],'recoverable checkpoint at failure')
v=calc('weight-handoff-qwen8')['summary']['phase_live_bytes']
check(max(d['10-15']['restore_bytes'])==v['restore_all_before_release'],'restore peak')
check(max(d['10-15']['staged_bytes'])==v['sync_weights_before_release'],'staged peak')
check(max(d['10-15']['restore_bytes'])-max(d['10-15']['staged_bytes'])==24*2**30,'handoff peak reduction')
check(round((2*3/4*72*2**20/25e9+.00002)*1000,2)==4.55 and round((2*3/4*72*2**20/50e9+.00002)*1000,2)==2.28,'whole bucket time')
check(round((2*3/4*24*2**20/25e9+.00002)*1000,2)==1.53 and round(3*(2*3/4*24*2**20/25e9+.00002)*1000,2)==4.59,'split bucket window')
check([min(8,.75*min(g,v)) for g,v in [(12,6),(24,6),(12,12)]]==[4.5,4.5,8],'RL supply bottleneck')
for name in ['填满排空','1F1B']:
 ev=d['10-7'][name]['events'];check(sum(x['kind']=='update' for x in ev)==4,'all optimizer events drawn');check(sum(x['kind']=='F' for x in ev)==32 and sum(x['kind']=='B' for x in ev)==32,'pipeline full event count')
# Independently evaluate the running design, using source FLOPs and declared inputs.
case=json.loads((HERE/'design-case.json').read_text())
check(case['tokens_per_update']==384*8192 and case['updates']==math.ceil(10**11/(384*8192)),'running task work')
EMBED=next(t['parameters'] for t in calc('training-state-book')['training_state_tensors'] if t['name']=='model.embed_tokens.weight')
for row in case['candidates']:
 p=row['devices'];local=F/10**11*(384*8192)/(p*165.2e12*.4)
 link=384//p*3*(p-1)/p*2*N/32e9;exposed=2*(p-1)/p*2*EMBED/32e9
 check(math.isclose(row['link_seconds_pcie'],link) and math.isclose(row['exposed_communication_seconds'],exposed),'ZeRO-3 link and exposed communication')
 check(link/(384//p)<local/(384//p),'per-microbatch link hidden by compute')
 loss=c/1800+p/MTBF*(900+120)
 finish=5+math.ceil(10**11/(384*8192))*(local+exposed+.5)*(1+loss)/86400
 check(math.isclose(row['finish_days'],finish,rel_tol=1e-12),'running deadline recompute')
 check(384%p==0 and row['microbatches_per_device']==384//p,'unchanged global batch')
 check(row['peak_budget_gib']<22,'running capacity feasibility')
 check((finish<30)==(p==48),'running design selection')
 check(round(finish,1)==(34.3 if p==32 else 24.6),'running displayed days')
check(round(case['candidates'][1]['max_base_step_seconds'],1)==67.2,'running acceptance step time')
r48=case['candidates'][1];check(round(r48['link_seconds_pcie'],1)==12.0 and round(r48['no_overlap_step_seconds'],1)==64.8<r48['max_base_step_seconds'],'no-overlap bound still meets deadline')
check(round(r48['link_seconds_single_nic_per_host'],1)==15.4>r48['max_communication_seconds'] and round(r48['single_nic_no_overlap_step_seconds'],1)==68.1,'single shared NIC needs overlap')
check(round(case['candidates'][0]['minimum_local_efficiency'],2)==.47,'32-card required efficiency')
for dev,curve in d['10-20'].items():
 source_row=curve['source_rows'][0]
 expected=source_row['training_days']*16384/90
 check(math.isclose(curve['continuous_required_devices'][-1],expected*10),'scale boundary')
 check(math.isclose(curve['parameters_at_16384_devices'],16384/expected*1e12),'scale intersection')
recompute=calc('pipeline-gemm-recompute-products')['reservation']
check(recompute['added_persistent_bytes_per_microbatch_stage'][0]==45*2**20,'recompute retained operands')
check(calc('pipeline-gemm-save-1f1b')['reservation']['added_persistent_bytes_per_microbatch_stage'][0]==135*2**20,'saved operands derivation')
check(recompute['additional_product_workspace_bytes_per_stage'][0]==6*2**20,'recompute workspace derivation')
ev=calc('training-pipeline-1f1b-m8')['events'];lookup={e['id']:e for e in ev}
check(math.isclose(lookup['F:3:2']['start']-lookup['B:3:1']['end'],.002),'annotated pipeline dependency gap')
# Pipeline schedule variants: makespans, peaks, message counts and bubble formulas.
sched={k:calc(f'training-pipeline-{k}-m8') for k in ['interleaved','zero-bubble','dualpipe']}
one8=calc('training-pipeline-1f1b-m8')
check(round(one8['summary']['step_makespan_seconds']*1000)==347,'1F1B m8 makespan')
check([round(sched[k]['summary']['step_makespan_seconds']*1000) for k in ['interleaved','zero-bubble','dualpipe']]==[298,283,306],'schedule variant m8 makespans')
check([round(calc(f'training-pipeline-{k}-m16')['summary']['step_makespan_seconds']*1000) for k in ['1f1b','interleaved','zero-bubble','dualpipe']]==[599,538,523,558],'schedule variant m16 makespans')
check([round(sched[k]['summary']['reserved_activation_scope_peak_bytes'][0]/2**20) for k in ['interleaved','zero-bubble','dualpipe']]==[1329,921,1228],'schedule stage-0 peaks')
check(round(one8['summary']['reserved_activation_scope_peak_bytes'][0]/2**20)==921,'1F1B stage-0 peak')
check(sched['interleaved']['transfers']['forward_messages']==56 and one8['transfers']['forward_messages']==24,'interleaved message count')
check(Fraction(1,2)*(4-1)/8==Fraction(3,16) and Fraction(4-1,8)==Fraction(3,8),'interleaved bubble fraction')
check(3*(10+10+10)==90 and 3*(10+10-10)==30 and 3*(10+10-20)==0,'1F1B/ZB-H1/ZB-H2 bubble ms with T_B=T_W=10')
check((4//2-1)*(30+20-3*10)==20,'DualPipe bubble ms')
for k in ['interleaved','zero-bubble','dualpipe']:
 summ=sched[k]['summary'];updates=sum(sched[k]['scenario']['optimizer_seconds'])
 check(math.isclose(summ.get('idle_device_seconds',4*summ['step_makespan_seconds']-summ['useful_forward_backward_device_seconds']-updates),4*summ['step_makespan_seconds']-summ['useful_forward_backward_device_seconds']-updates),'schedule idle accounting '+k)
check(round((4*one8['summary']['step_makespan_seconds']-one8['summary']['useful_forward_backward_device_seconds']-sum(one8['scenario']['optimizer_seconds']))*250)==106,'1F1B per-GPU idle ms')
gp8=calc('training-pipeline-gpipe-m8')
check(round(gp8['summary']['step_makespan_seconds']*1000)==337,'fill-drain m8 makespan')
check(round((4*gp8['summary']['step_makespan_seconds']-gp8['summary']['useful_forward_backward_device_seconds']-sum(gp8['scenario']['optimizer_seconds']))*250)==96,'fill-drain per-GPU idle ms')
check([round(b/2**20) for b in gp8['summary']['reserved_activation_scope_peak_bytes']]==[1840,1840,1840,2448],'fill-drain stage peaks')
# Critical batch size: noise-scale relation, scaling days and ceilings.
cb=calc('critical-batch-book');rows={r['cards']:r for r in cb['rows']}
check(round(cb['reference']['s_min'])==19434 and cb['reference']['steps']==31790,'critical batch reference')
check(abs(cb['summary']['weak_scaling_speedup_ceiling']-31790/19434.119549264942)<1e-9 and round(cb['summary']['weak_scaling_speedup_ceiling'],2)==1.64,'weak scaling ceiling')
check([round(rows[n]['weak_wall_days'],1) for n in [48,96,192,384,1536]]==[19.4,15.7,13.8,12.8,12.1],'weak scaling days')
check([round(rows[n]['strong_wall_days'],1) for n in [96,192,384,1536]]==[9.8,5.0,2.6,0.8],'strong scaling days')
check(384*8192==3145728 and abs(3145728/2e6-1.57)<0.005,'batch over noise scale')
check(round(calc('critical-batch-noise-20m')['summary']['weak_scaling_speedup_ceiling'],2)==7.36,'noise 20m ceiling')
# Stragglers: order statistics, responses and the spike-corrected checkpoint model.
st=calc('straggler-max-sigma-2pct');st5=calc('straggler-max-sigma-5pct')
check([round(st['rows'][i]['expected_standard_max'],2) for i in range(3)]==[1.42,2.23,3.25],'expected standard max')
check([round(st['rows'][i]['expected_step_seconds'],1) for i in range(3)]==[54.3,55.1,56.2],'straggler steps sigma 2pct')
check([round(st5['rows'][i]['expected_step_seconds'],1) for i in range(3)]==[56.5,58.6,61.3],'straggler steps sigma 5pct')
check(st['evict_response']['rows'][1]['evict_cost_seconds']==600/2+120==420,'evict cost at 600 s interval')
check(round(st['checkpoint_loss']['rows'][1]['hardware_only_loss'],4)==0.0280 and round(st['checkpoint_loss']['rows'][1]['with_spike_rollback_loss'],4)==0.0287,'spike-corrected loss at 600 s')
check(round(st['checkpoint_loss']['first_order_optimal_interval_hardware_only'])==4458 and round(st['checkpoint_loss']['first_order_optimal_interval_with_spike'])==3150,'spike-corrected optimal interval')
c48=14*N/7e9;lam48=48/MTBF;spike=1/604800
check(abs(c48/1800+lam48*(1800/2+120)-0.010782)<1e-5 and math.isclose(c48/1800+lam48*(1800/2+120),case['candidates'][1]['total_loss']),'design-case hardware loss at 1800 s')
check(abs(c48/1800+(lam48+spike)*(1800/2+120)-0.012468)<1e-5,'design-case loss with spike at 1800 s')
check(math.isclose(st['rows'][1]['expected_wait_for_mean_rank_seconds'],st['rows'][1]['expected_max_seconds']-52.2),'mean-rank wait is max minus mean')
# MoE capacity factor: chapter histogram and Qwen3-235B shape.
mcj=calc('moe-capacity-book');ch=mcj['chapter_example'];mo=mcj['model_example']
check([r['capacity_per_expert'] for r in ch['rows']]==[64,80,96,128] and [r['dropped_total'] for r in ch['rows']]==[32,16,0,0],'chapter capacity rows')
check([r['padded_total'] for r in ch['rows']]==[32,48,64,128],'chapter padding rows')
check(mo['experts']==128 and mo['top_k']==8 and mo['assignments']==8192*8,'model dispatch count')
check([r['capacity_per_expert'] for r in mo['rows']]==[512,640,768,1024],'model capacity rows')
check([float(Fraction(r['dropped_fraction_exact'])) for r in mo['rows']]==[.25,.125,0,0],'model dropped fractions')
check([float(Fraction(r['padded_fraction_of_executed_exact'])) for r in mo['rows']]==[.25,.3,1/3,.5],'model padded fractions')
check(8190735360*(1+4/16384)/1e9>8.1 and round(8190735360*(1+4/16384)/1e9,1)==8.2,'FP8 weight bytes')
d=json.loads((HERE/'figure-data.json').read_text())
for key,kinds in [('10-21',{'F','B','update'}),('10-22',{'F','X','W','update'}),('10-23',{'F','X','W','update'})]:
 ev=d[key]['events'];check(set(e['kind'] for e in ev)==kinds,'timeline event kinds '+key)
 for stage in range(4):check(math.isclose(sum(e['duration'] for e in ev if e['stage']==stage and e['kind']!='update'),.24),'timeline compute per stage '+key)
 check(max(e['start']+e['duration'] for e in ev)<=d[key]['summary']['step_makespan_seconds']+1e-9,'timeline within makespan '+key)
check(sum(e['chunk']==1 for e in d['10-21']['events'] if e['kind']!='update')==64,'interleaved second-chunk events');check(sum(e['direction']==1 for e in d['10-23']['events'] if e['kind']!='update')==48,'DualPipe reverse-direction events')
zb=sched['zero-bubble']
check(round(zb['stages'][0]['idle_during_training_seconds']*1000)==42,'zero-bubble per-GPU idle ms')
check([round(x*1000) for x in zb['summary']['idle_per_stage_zero_transfer_counterfactual_seconds']]==[30]*4,'zero-bubble zero-transfer idle equals ZB-H1 bound')
check(math.isclose(zb['summary']['bubble_bound_zb_h1_seconds'],.03) and zb['summary']['bubble_bound_zb_h2_seconds']==0,'ZB-H1/H2 bound fields')
check([round(b/2**20) for b in zb['summary']['reserved_activation_scope_peak_bytes']]==[921,921,921,1226] and round(one8['summary']['reserved_activation_scope_peak_bytes'][3]/2**20)==308,'zero-bubble stage peaks')
ps=d['10-24']
check([round(ps[k]['makespan_m8_ms']) for k in ps]==[347,298,283,306],'schedule figure m8 order')
check(ps['交错式 v=2']['boundary_crossings_per_microbatch']==7,'interleaved boundary crossings')
check(d['10-29']['model_dropped_fraction']==[.25,.125,0,0],'capacity figure data')
main=s.split('## 参考资料与进一步阅读')[0]
check(not re.search(r'不代表|不能只|尚需核验|不用于.*外推|上式只描述',main),'no defensive prose endings')
check(main.index('> **习题')>main.index('## 习题与实验'),'exercises collected after main exposition')
# New mechanism figures must depict the same conditions used in the derivations.
check(d['recompute_lifetime']['product_bytes']==6*2**20,'lifetime product shape')
check(d['cast_paths']['gpu_path_transfer_bytes']==2*d['cast_paths']['cpu_path_transfer_bytes'],'conversion path bytes')
w=d['communication_window'];check(w['queued_reduce_ms'][1]-w['compute_ms'][1]==w['wait_ms']==2,'communication waiting interval')
a=d['attention_area'];check([sum(n*(n+1)//2 for n in lens) for lens in a['lengths']]==a['pairs'],'attention triangles are true pairs')
q=d['input_queue'];check(q['pending_batches']==list(range(q['last_trained_batch']+1,q['last_dispatched_batch']+1)),'queue resumes at first untrained batch')
r=d['checkpoint_resharding'];check(r['old_rows']*r['old_parts']==r['new_rows']*r['new_parts']==r['shape'][0],'resharding preserves row coverage')
t=d['checkpoint_tradeoff'];check(math.isclose(t['optimal_seconds'],math.sqrt(2*t['checkpoint_seconds']/t['job_failure_rate_per_second'])),'checkpoint curve minimum')
rl=d['rl_supply_flow'];check(min(rl['learn_capacity'],rl['retained_fraction']*min(rl['generate_capacity'],rl['verify_capacity']))==rl['effective_rate'],'RL flow bottleneck')
cycle=d['rl_async_cycle'];check(cycle['overlapped_period_seconds']==max(cycle['generate_seconds'],cycle['learn_seconds'])+cycle['sync_seconds'],'async timeline')
for row in d['task_deadline_breakdown']['rows']:check(math.isclose(sum(row['days']),row['finish_days']),'deadline bars add to finish time')
figrefs=[int(n) for n in re.findall(r'图 10-(\d+)',s)]
check(all(1<=n<=figure_count for n in figrefs),'figure references within chapter sequence')
report={'status':'passed' if not errors else 'failed','sections':6,'subsections':len(heads),'exercises':len(re.findall(r'^(?:>\s*)?\*\*(?:(?:练习|实验|习题|拓展练习)\s+)?(10-\d+)\b(?!\s*解答)',s,re.M)),'figures':figure_count,'image_files':sum(Path(r['path']).suffix in ['.svg','.png','.pdf'] for r in manifest['outputs']),'formulas':mathreport['expressions'],'source_files':len(json.loads((HERE/'sources.json').read_text())['sources']),'chinese_characters':len(re.findall(r'[\u4e00-\u9fff]',s)),'errors':errors}
(HERE/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2));raise SystemExit(bool(errors))
