#!/usr/bin/env python3
"""Verify chapter coverage, source locks, math and figure data without GPU work."""
from pathlib import Path
from fractions import Fraction
from urllib.parse import unquote,urlsplit
import hashlib,json,re,xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
checks=[]
def check(condition,name):checks.append({'name':name,'passed':bool(condition)})
def read(n):return json.loads((ROOT/'calculations/results'/f'{n}.json').read_text())
def val(x):return float(Fraction(str(x)))
import sys;sys.path.insert(0,str(HERE.parent))
from preview_output import preview_path
md=HERE.parent/'09-分布式推理.md';s=md.read_text();html=preview_path(md).read_text();outline=next(p for p in [ROOT/'outlines/09-分布式推理.md',ROOT/'archive/outlines/09-分布式推理.md'] if p.exists()).read_text()
heads=re.findall(r'^#{2,3} (9\.\d+(?:\.\d+)?) ',s,re.M);expected=re.findall(r'^#{2,3} (9\.\d+(?:\.\d+)?) ',outline,re.M)
check([h for h in heads if h in set(expected)]==expected,'All seven sections and outline subsection numbers preserved')
check(re.findall(r'^> \*\*练习 9-(\d+)\s*·',s,re.M)==[str(i) for i in range(1,21)],'Twenty chapter exercises retained')
index=json.loads((HERE/'figure-index.json').read_text());count=len(index)
check(re.findall(r'^\*图 (9-\d+)[：　]',s,re.M)==[f'9-{i}' for i in range(1,count+1)],'Sequential external captions')
check(len(re.findall(r'!\[.*?\]\(ch09/(?:figure-|ch09-)',s))==count,'Active figures included')
ids=set(re.findall(r'\[\^([^\]]+)\]:',s));uses=set(re.findall(r'\[\^([^\]]+)\](?!:)',s));check(ids==uses,'All citations defined and used')
for match in re.findall(r'\]\(([^)]+)\)',s):
 parts=urlsplit(match)
 if parts.scheme:continue
 path=(md.parent/unquote(parts.path)).resolve();check(path.exists(),'Local reference '+match)
for source in json.loads((HERE/'sources.json').read_text())['sources']:
 path=ROOT/source['path'];check(path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256'],'Source lock '+source['path'])
for entry in index:
 p=HERE/(entry['name']+'.svg')
 root=ET.parse(p).getroot();text=' '.join(root.itertext());check(not re.search(r'图\s*\d+\s*[-–]\s*\d+',text),'No caption number inside '+p.name)
 for ext in ['png','pdf']:check(p.with_suffix('.'+ext).stat().st_size>1000,'Export '+p.with_suffix('.'+ext).name)
check(len(index)==count,'Active vector figure index')
check(html.count('src="data:image/png;base64,')==count,'Reading HTML embeds all active figures')
math=json.loads((HERE/'math-validation.json').read_text());check(math['expressions']>=80 and not math['errors'],'KaTeX parsed all inline and display formulas')
check('MATHPLACEHOLDER' not in html and 'katex-error' not in html,'No unrendered math')
layout=json.loads((HERE/'teaching-layout-validation.json').read_text())+json.loads((HERE/'ub-ep-layout-validation.json').read_text())+json.loads((HERE/'kv-tier-layout-validation.json').read_text())
check(len(layout)==count and all(x['width_pt']==420 and x['min_label_pt']>=11 and not x['text_extent_warnings'] for x in layout),'Book-size readable labels')
# Independently check the chapter's key arithmetic against frozen results.
V=2*36*8192*8*128*2;check(V==1207959552,'GQA KV bytes');check(V/2**30==1.125,'Binary capacity conversion')
handoff=read('pd-af-handoff-qwen8')['summary'];check(handoff['pd_snapshot_bytes']==V,'PD frozen payload');A=2*36*4096*2;check(A==589824==handoff['af_total_bytes'],'AF directional payload sum');check(36*2==handoff['af_directional_messages'],'AF message count')
af_ns=72*5000+Fraction(A*10**9,25*10**9);check(af_ns==Fraction(handoff['af_serialized_ns_exact']),'AF serialized time');check(abs(float(af_ns)*1024/1e6-392.79919104)<1e-8 and Fraction(read('pd-af-handoff-qwen8-1024')['summary']['af_serialized_ns_exact'])==1024*af_ns,'Full 1024-step decode handoff accumulation')
check(A*8192/2**30==4.5,'Full prefill AF payload in teaching placement')
z=read('pd-pool-book');grid=[]
# Stage rates recomputed from the forward-accounting FLOPs/bytes and datasheet peaks at 50% efficiency.
PF,SF,SW,SK=133594323353600,648659599360,15137073152,41075343360
def card(tf,gbs):
 c=Fraction(tf*10**12,2);b=Fraction(gbs*10**9,2)
 return Fraction(PF)/c, 1024*Fraction(SW+SK)/b/32, Fraction(SF)/c, Fraction(SW+SK)/b
tA,dA,scA,smA=card(312,2039);tH,dH,scH,smH=card(148,4096)
check(smA>scA and smH>scH,'Decode steps are memory-bound on both cards')
rates={d['device']:d for d in z['derived_stage_rates']}
check(Fraction(rates['a100-80gb-sxm']['prefill']['seconds_exact'])==tA and Fraction(rates['h20-sxm5-96gb']['decode']['seconds_per_request_exact'])==dH,'Stage rates follow datasheet peaks at 50%')
for row in z['pool_assignments']:
 pa=row['prefill_workers']['A100'];ph=row['prefill_workers']['H20'];rate=min(pa/tA+ph/tH,(4-pa)/dA+(4-ph)/dH,Fraction(25*10**9,V));check(rate==Fraction(row['bound_requests_per_second_exact']),f'Integer allocation A100={pa},H20={ph}');grid.append(rate)
check(max(grid)==4/dH and 4/(tA+dA)+4/(tH+dH)==Fraction(z['summary']['colocated_bound_requests_per_second_exact']),'PD and colocated upper bounds')
check(round(float(4/dH),2)==4.55 and round(float(4/(tA+dA)+4/(tH+dH)),2)==3.02 and round(float(4/tH),2)==2.22,'Heterogeneous PD, colocated and swapped-role rates')
h=read('pd-pool-homogeneous')['summary'];check(round(float(Fraction(h['best_pd_bound_requests_per_second_exact'])),2)==2.83 and round(float(Fraction(h['colocated_bound_requests_per_second_exact'])),2)==3.05,'Homogeneous A100 split loses throughput')
check(round(float((smA-scA)*Fraction(156*10**12)/(Fraction(PF)/8192)))==488 and round(float((smH-scH)*Fraction(74*10**12)/(Fraction(PF)/8192)))==85,'Prefill tokens hidden under one decode step')
ovA=max(tA+1024*scA/32,Fraction(rates['a100-80gb-sxm']['prefill']['memory_seconds_exact'])+dA);ovH=max(tH+1024*scH/32,Fraction(rates['h20-sxm5-96gb']['prefill']['memory_seconds_exact'])+dH)
check(round(float(8/ovA),2)==4.50 and round(float(4/ovA+4/ovH),2)==4.17,'Ideal chunked colocation bounds')
P=3*4096*1536;W=2*P
check(W==36*2**20 and 32*W*94/2**30==105.75,'Expert and all-layer resident capacity')
for name,m,C in [('expert-locality-avx512',1,1800),('expert-locality-avx512-128',128,1800),('expert-locality-amx',1,21300),('expert-locality-amx-128',128,21300)]:
 tasks=8*m
 cpu=Fraction(16*5000)+Fraction(tasks*16384,25)+max(Fraction(2*P*tasks,C),Fraction(W*8,220))
 gpu=Fraction(8*5000)+Fraction(W*8,25)+max(Fraction(2*P*tasks,156000),Fraction(W*8*10,7775))
 z=read(name)['summary'];check(cpu==Fraction(z['cpu_service_ns_exact']) and gpu==Fraction(z['weight_copy_service_ns_exact']),f'Independent expert paths {name}')
check([r['max_tokens_per_expert'] for r in read('expert-locality-avx512')['locality_reuse_regions']][0]==71 and read('expert-locality-amx')['locality_reuse_regions'][0]['max_tokens_per_expert']==688,'CPU/GPU crossovers at 71/72 and 688/689 tokens')
check(7*32==224 and 8*32==256,'Padding controls')
zn=read('replica-payback-hgx-h100-nvlink')['summary'];zc=read('replica-payback-hgx-h100-cx7')['summary']
for z,k,copy in [(zn,3,450),(zc,23,50)]:
 setup=Fraction(264241152,copy)+7*5000;check(Fraction(z['serialized_copy_setup_ns_exact'])==setup and Fraction(z['serialized_copy_setup_ns_exact'])//Fraction(z['per_batch_saving_ns_exact'])+1==k==z['algebraic_strict_payback_batches'],f'Strict replica payback {k} batches')
check(Fraction(zn['baseline_batch_ns_exact'])==Fraction(920649728*10**9,1675*10**9) and Fraction(zn['replicated_batch_ns_exact'])==Fraction(517865472,1675) and zn['capacity_feasible'],'H100 HBM-bound replica batch times')
dl=float(Fraction(zn['per_batch_saving_ns_exact']))/1e6;sn=float(Fraction(zn['serialized_copy_setup_ns_exact']))/1e6;sc=float(Fraction(zc['serialized_copy_setup_ns_exact']))/1e6
check([round(x,1) for x in (16*dl-sn,64*dl-sn,16*dl-sc,64*dl-sc)]==[3.2,14.8,-1.5,10.1] and round(sn,3)==.622 and round(sc,2)==5.32 and round(dl,3)==.240,'Replica payback windows on NVLink and ConnectX-7');check(94*W/2**30==3.3046875,'Per-layer replica capacity')
route=read('cache-route-a100-stale')['summary'];hit=80*10**6+Fraction(4813831012352*10**9,156*10**12);miss=80*10**6+Fraction(138406909706240*10**9,156*10**12)
check(Fraction(route['a_valid_hit_ns_exact'])==hit and Fraction(route['a_all_tiers_miss_ns_exact'])==miss and round(float(hit)/1e6,1)==110.9 and round(float(miss)/1e6,1)==967.2,'Stale cache two-point times on A100')
check(Fraction(route['a_expected_ns_exact'])==Fraction(9,10)*hit+Fraction(1,10)*miss and round(float(Fraction(route['a_expected_ns_exact']))/1e6,1)==196.5,'Stale cache expected time');check(Fraction(route['a_p99_ns_exact'])==miss,'Stale cache p99');check(round(float(Fraction(route['strict_mean_hit_probability_threshold_exact']))*100,1)==7.0,'Mean hit threshold')
cr=read('cache-route-a100-50gbe')['summary'];cf=read('cache-route-a100-200gbe')['summary']
check(round(float(Fraction(cr['full_compute_ns_exact']))/1e6)==887 and round(float(Fraction(cr['warm_compute_ns_exact']))/1e6,1)==30.9,'Route compute times follow A100 at 50%')
check(round(float(Fraction(cr['remote_equal_a_hit_bytes_per_second_exact']))/1e9,2)==6.30 and round(float(Fraction(cr['remote_equal_recompute_bytes_per_second_exact']))/1e9,2)==1.48,'Route bandwidth crossovers')
check(round(cf['remote_payload_demand_bytes_per_second']/25e9*100)==77 and cf['remote_payload_demand_strictly_below_bandwidth'],'Sixteen retrievals per second use 77% of 200 GbE')
z=read('cache-restart-book')['summary'];check(z['consumer_read_file_bytes']==144*2**20 and z['consumer_reusable_bytes']==Fraction(567,4)*2**20,'Read versus reusable bytes')
check(Fraction(10,1)/Fraction(2,10000)==50000,'Startup break-even execution steps')
check(round(10+35/(float(4/dH)-3.5))==43 and round(10+35/(float(4/ovA+4/ovH)-3.5))==63,'Queue drain after readiness')
# Added numerical arguments: use identical dimensions, explicit resource assumptions.
E,K,M=128,8,64;Pe=3*4096*1536;We=2*Pe
check(M*K==512 and M*K/E==4 and M==64,'Same 512 dispatches, four versus 64 rows per expert')
check(M*K*2*Pe==19327352832,'Expert work remains 19.3 GFLOPs')
check(E*We==int(4.5*2**30) and K*We==288*2**20 and E/K==16,'Distinct weight traffic differs by sixteen')
check(round(E*(1-(1-K/E)**M))==126,'Independent-routing expectation near 126 experts')
check(round(E*(1-(1-K/E)**M)*We/2**30,2)==4.43,'Expected distinct weight volume')
check((M*K*2*Pe)/(M*K*2*Pe/8)==8,'Most-loaded rank work differs eightfold')
zp=read('pd-pool-prefix')['summary'];check(zp['best_prefill_workers']=={'A100':2,'H20':0} and round(float(Fraction(zp['best_pd_bound_requests_per_second_exact'])),2)==5.69,'Prefix hit reallocates two A100 to decode')
zs=read('pd-pool-short-output')['summary'];check(zs['best_prefill_workers']=={'A100':4,'H20':3} and round(float(Fraction(zs['best_pd_bound_requests_per_second_exact'])),2)==6.33 and round(float(Fraction(zs['colocated_bound_requests_per_second_exact'])),2)==5.84,'Short output moves three H20 to prefill')
fig=json.loads((HERE/'figure-data.json').read_text())
check(round(fig['9-6']['cpu_avx512_ms'][127],1)==22.2 and round(fig['9-6']['cpu_amx_ms'][127],2)==2.57 and round(fig['9-6']['weight_copy_gpu_ms'][127],1)==12.5,'Instruction set decides the 128-token CPU/GPU choice')
ep=json.loads((ROOT/'calculations/results/ep-skew-book.json').read_text())['results']
check([[round(r['dispatch']['lower_s']*1e3,3),round(max(r['compute_s'])*1e3,3),round(r['combine']['lower_s']*1e3,3),round(r['barrier_lower_s']*1e3,3)] for r in ep]==[[.336,.156,.336,.827],[.839,.391,.839,2.068],[1.342,.156,1.342,2.841],[.419,.391,.839,1.649],[.093,.391,.093,.577]],'Large-EP table on two HGX H100 servers')
check(round((ep[0]['dispatch']['lower_s']+ep[0]['combine']['lower_s'])/ep[0]['barrier_lower_s'],1)==.8 and round(37748736/494.7e12*1e9)==76 and round(8192/50e9*1e9)==164,'Cross-server barrier is communication-dominated')
check(round(ep[0]['two_microbatch_pipeline_s']*1e3,3)==.581 and round((ep[0]['barrier_lower_s']-ep[0]['two_microbatch_pipeline_s'])*1e3,3)==.246,'Two-microbatch overlap example')
check(round((ep[0]['pipeline_break_even_comm_stretch']-1)*100)==49 and round(ep[0]['pipeline_break_even_compute_stretch'],1)==3.1,'Contention break-even for communication and compute')
check([round(x*2*18874368/494.7e12*1e6,1) for x in (64,512)]==[4.9,39.1],'Busiest-card compute on H100')
Trec=Fraction(326158016,380859375);Tr=Fraction(V,25*10**9)
check(round(float(Trec),3)==.856 and round(float(Tr)*1e3,1)==48.3 and 2*Tr<Trec/8 and round(V/float(Trec)/1e9,2)==1.41 and round(2*V/float(Trec)/1e9,2)==2.82 and round(V/1.25e9*1e3)==966,'Save-and-retrieve crossovers on A100 and 10 GbE')
check([round(V/12.5e9*1e3,1),round((16*V-12.5e9)/1e9,1),round(575668224/12.5e9*1e3,1)]==[96.6,6.8,46.1] and 8*V<12.5e9<16*V and 16*575668224<12.5e9,'PD handoff on one 100 GbE port')
mig={k:read(f'reconfiguration-ch09-{k}')['summary'] for k in ['50gbe','200g-nic','a100-nvlink']}
slow,fast,nvl=[float(Fraction(mig[k]['conditional_serial_switch_lower_bound_exact_seconds'])) for k in ['50gbe','200g-nic','a100-nvlink']]
check(mig['50gbe']['network_bytes']==16449646080 and round(slow,1)==11.6 and round(fast,1)==9.7 and round((slow-fast)/slow*100)==17 and round(nvl,1)==9.0,'Migration bandwidth yields 17 percent total improvement')
check(round(float(Fraction(mig['a100-nvlink']['transfer_lower_bound_exact_seconds']))*1e3,1)==15.7 and round(float(Fraction(mig['50gbe']['transfer_lower_bound_exact_seconds'])),2)==2.63,'Migration transfer bounds')
check([round(8/(3600*float(r))*1000,2) for r in [4/(tA+dA)+4/(tH+dH),4/dH,4/dH/4]]==[.74,.49,1.95],'Effective output changes cost ordering')
check(fig['9-12']['pages_read']==64 and fig['9-12']['pages_reused']==63,'Single relationship in page reuse figure')
check(fig['9-16']['pool_initial_payload_GiB']==2*fig['9-16']['direct_payload_GiB'],'Direct versus pool initial traffic')
check(set(fig['9-17'])=={'type','startup_s','arrival_rps','time_s','queues','drain_time_from_start_s'},'Startup figure contains only backlog argument')

# Complete deployment worked example and explanatory crossover.
check(round(32*9216*147456/2**30,1)==40.5 and (32*9216*147456+V+16381470720)/2**30 < 96e9/2**30 and round((32*9216*147456+V+16381470720)/2**30,1)==56.9,'Batch-32 final KV, receive buffer and weights fit one H20')
check(64//1.125==56,'Shared pool holds 56 complete prefixes')
check(round(25e9/(2*V),1)==10.3,'Shared-channel two-hop capacity')
check(round(35+(3.5-float(4/(tA+dA)+4/(tH+dH)))*50)==59,'Colocated backlog at 60 seconds')
check(Fraction(7,2)+Fraction(35,50)==Fraction(21,5) and round(10+35/(0.9*float(4/dH)-3.5))==68,'Recovery deadline requires 4.2 requests per second')
check(round(8/(3600*3.5)*1000,2)==.63,'Provisioned cost at actual arrival rate')
check([round(10+35/(float(Fraction(read('pd-pool-prefix')['summary'][k]))-3.5),1) for k in ['colocated_bound_requests_per_second_exact','best_pd_bound_requests_per_second_exact']]==[35.1,26.0],'Cached backlog drain times')
g4=Fraction(read('pd-pool-4k-output')['summary']['best_pd_bound_requests_per_second_exact']);check(2*g4<Fraction(7,2)<3*g4,'4097-output steady arrival needs three groups')
check(3*g4<Fraction(21,5)<4*g4,'4097-output recovery deadline needs four groups')
check([round(c/b,1) for c,b in [(1.8e12,220e9),(21.3e12,220e9),(1.8e12,125e9),(21.3e12,125e9)]]==[8.2,96.8,14.4,170.4],'Worked solution CPU resource crossover')

# Verify that each new mechanism diagram agrees with its teaching example.
check(fig['new-footprint']['active_experts'][0]*fig['new-footprint']['rows_per_expert'][0]==512 and fig['new-footprint']['active_experts'][1]*fig['new-footprint']['rows_per_expert'][1]==512,'Equal rectangle areas represent identical dispatch counts')
check(fig['new-footprint']['weight_MiB']==[128*36,8*36],'Expert footprint bytes')
check(all(sum(v)==512 for v in fig['new-balance']['assignments_per_card']),'Balanced and concentrated devices perform equal total work')
check([round(x,2) for x in fig['new-allocation']['rates']]==[4.55,5.69,6.33] and fig['new-allocation']['prefill_A100']==[4,2,4] and fig['new-allocation']['prefill_H20']==[0,0,3],'Allocation diagram uses three solved pool rates')
check([round(x) for x in fig['new-route']['first_token_ms']]==[281,907,282,137],'Route timelines reproduce four completion times')
check([round(x,2) for x in fig['new-migration']['catchup_s']]==[1.65,6.76] and round(fig['new-migration']['initial_GB'],1)==41.1 and round(fig['new-migration']['growth_GBs'],3)==.172,'Copy catches growing decode state')
check(fig['new-recovery']['input_positions']+fig['new-recovery']['replayed_outputs']==fig['new-recovery']['restored_KV_positions']==9216,'Recovery diagram preserves KV position boundary')
check(round(fig['new-overlap']['pipeline_ms'],3)==.581 and round(fig['new-overlap']['serial_ms'],3)==.827,'Overlap schedule completion times')

for record in json.loads((HERE/'manifest.json').read_text())['outputs']:
 p=ROOT/record['path'];check(hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256'],'Artifact hash '+p.name)
report={'status':'passed' if all(c['passed'] for c in checks) else 'failed','checks':len(checks),'errors':[c['name'] for c in checks if not c['passed']],'sections':7,'subsections':len(heads)-7,'figures':count,'exercises':20,'cjk_characters':sum('\u4e00'<=x<='\u9fff' for x in s),'new_gpu_measurements':False}
(HERE/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2));raise SystemExit(bool(report['errors']))
