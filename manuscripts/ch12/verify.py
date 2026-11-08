#!/usr/bin/env python3
"""Validate chapter structure, actual source data, links, artwork, and arithmetic."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from svg_labels import svg_labels
from preview_output import preview_path
from fractions import Fraction
from urllib.parse import unquote, urlsplit
import hashlib,json,re,math,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
md=ROOT/'manuscripts/12-端边云协同.md';text=md.read_text();page=preview_path(md).read_text();errors=[];checks={}
catalog=json.loads((HERE/'figure-catalog.json').read_text());active_catalog=json.loads((HERE/'figure-index.json').read_text());figure_count=len(active_catalog)
def check(name,condition):
 checks[name]=bool(condition)
 if not condition:errors.append(name)
def close(x,y,tol=1e-8):return abs(x-y)<tol
outline_path=next(p for p in [ROOT/'outlines'/md.name,ROOT/'archive/outlines'/md.name] if p.exists())
outline=outline_path.read_text()
check('five_sections',re.findall(r'^## (12\.\d+) ',text,re.M)==[f'12.{i}' for i in range(1,6)])
outline_subs=re.findall(r'^### (12\.\d+\.\d+) ',outline,re.M)
text_subs=re.findall(r'^### (12\.\d+\.\d+) ',text,re.M)
check('outline_subsection_coverage',text_subs[:len(outline_subs)]==outline_subs and text_subs[len(outline_subs):]==['12.5.5','12.5.6'])
check('ten_exercises',re.findall(r'^\*\*(12-\d+)\s',text,re.M)==[f'12-{i}' for i in range(1,11)])
check('three_core_exercises',re.findall(r'^\*\*(12-\d+)[^\n]*〔核心',text,re.M)==['12-2','12-3','12-7'])
check('external_captions',re.findall(r'^\*图 (12-\d+)：',text,re.M)==[f'12-{i}' for i in range(1,figure_count+1)])
check('html_figcaptions',page.count('<figcaption>')==figure_count)
check('embedded_images',page.count('src="data:image/png;base64,')==figure_count)
check('no_math_placeholders','MATHPLACEHOLDER' not in page)
check('no_katex_errors','class="katex-error"' not in page)
check('no_authoring_placeholders',not re.search('配图计划|待扩写|TODO|TBD',text))
# Source and artifact hashes are checked independently of the drawing code.
for record in json.loads((HERE/'sources.json').read_text())['sources']:
 p=ROOT/record['path']
 b=p.read_bytes() if p.exists() else b''
 h=b.decode().split('oid sha256:')[1].splitlines()[0].strip() if b.startswith(b'version https://git-lfs.github.com/spec/v1') else hashlib.sha256(b).hexdigest()
 check('source:'+record['path'],p.exists() and h==record['sha256'])
for record in json.loads((HERE/'manifest.json').read_text())['outputs']:
 p=ROOT/record['path'];check('artifact:'+record['path'],p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256'])
links=re.findall(r'\]\(([^)]+)\)',text)
for url in links:
 parts=urlsplit(url)
 if parts.scheme or not parts.path:continue
 p=(md.parent/unquote(parts.path)).resolve();check('link:'+url,p.exists())
for entry in active_catalog:
 p=HERE/entry['file']
 labels=svg_labels(p)
 check('artwork_has_no_caption_number:'+p.name,not any(re.search(r'图\s*\d+\s*[-－–]\s*\d+',s) for s in labels))
 check('artwork_has_labels:'+p.name,bool(labels))
check('book_size_layout',all(r['width_pt']==420 and r['min_label_pt']>=11 and not r['text_extent_warnings'] for r in json.loads((HERE/'teaching-layout-validation.json').read_text())))
rendered_data=json.loads((HERE/'figure-data.json').read_text());d={f"12-{e['source_id']}":rendered_data[f"12-{e['number']}"] for e in catalog};q=json.loads((ROOT/'calculations/results/queqiao-records-conditions.json').read_text());ec=json.loads((ROOT/'calculations/results/multimodal-cache-single.json').read_text())['summary']
check('full_ec_shape',400*10240*2==ec['complete_encoder_bytes_per_image']==d['12-3']['bytes'][1])
check('visual_kv',400*147456==ec['visual_kv_bytes_per_image']==d['12-3']['bytes'][2])
check('image_feature_transfer',d['12-3']['uplink_seconds_at_6_4Mbps'][:2]==[1,10.24])
check('raw_budget',close(30*8/20+.1+.3+5*8/100,12.8))
check('model_acceleration',close(12+.1+.03+.4,12.53))
check('compression_budget',close(15*8/20+.1+.3+.4+.15,6.95))
check('screenshot_round',close(.2*8/6.4+.03+2.5,2.78))
check('screenshot_30_round_saving',close((3.5-2.78)*30,21.6))
check('migration_break_even',19*.4 < 64*2**20*8/80e6+1 <20*.4)
check('raw_packet_wire',35000000+29966*(60+92)==39554832)
check('ack_difference_bytes',39555120-38177328==1377792)
check('ack_difference_time',close(14.564459796-14.52103724,.043422556))
check('air_exchange_totals',all(close(sum(a),b) for a,b in zip(d['12-5']['exchange_components_us'],[302,134])))
check('airtime_budget',close(30800*(302+134)/1e6,13.4288))
check('fixed_file_payload',d['12-6']['recorded_fixed_bytes']==354640)
check('queqiao_matched_rows',d['12-6']['matched_ASR_ms']==[[1185.3,301.6],[240.9,236.5]])
check('all_physical_rows_retained',d['12-6']['starvation_ms']==[[520,360],[700,1040],[460,560],[880,1440]])
edge=json.loads((ROOT/'calculations/results/edge-tiers-agent-book.json').read_text());tiers={x['tier']:x for x in edge['tiers']};up=edge['uplink'];cases={c['id']:{r['tier']:r for r in c['rows']} for c in edge['cases']}
hw={x['id']:x for x in json.loads((ROOT/'calculations/configs/hardware.json').read_text())['devices']}
step=15136819200*18//64+1207959552
check('tier_bandwidths',tiers['near']['bandwidth_bytes_per_second']==hw['rtx-pro6000-blackwell-ws']['memory']['bandwidth_bytes_per_second'] and tiers['cloud']['bandwidth_bytes_per_second']==hw['h100-sxm']['memory']['bandwidth_bytes_per_second'] and close(tiers['end']['bandwidth_bytes_per_second'],84.8e9,1))
check('tier_model_seconds',all(close(tiers[k]['model_seconds_per_round'],45*step/tiers[k]['bandwidth_bytes_per_second']) for k in tiers) and [round(tiers[k]['model_seconds_per_round'],3) for k in ('end','near','cloud')]==[2.9,0.137,0.073])
check('tier_prepare',close(tiers['near']['prepare_seconds'],133594323353600/503.8e12) and close(tiers['cloud']['prepare_seconds'],133594323353600/989.4e12) and tiers['end']['prepare_seconds']==0)
T=[cases['twenty-rounds'][k]['total_seconds'] for k in ('end','near','cloud')]
check('deployment_times',all(close(x,y) for x,y in zip(T,[20*(.3+tiers['end']['model_seconds_per_round']),tiers['near']['prepare_seconds']+20*(.3+tiers['near']['model_seconds_per_round']+.02+6.4/80),tiers['cloud']['prepare_seconds']+20*(.3+tiers['cloud']['model_seconds_per_round']+.2+6.4/6.4)])) and [round(x,1) for x in T]==[64.0,11.0,31.6] and d['12-7']['task_seconds']==T)
check('deployment_thresholds',close(up['cloud_fixed_seconds']+128/d['12-7']['feasible_threshold_Mbps'],45) and round(d['12-7']['feasible_threshold_Mbps'],1)==3.8 and up['cloud_fixed_seconds']>up['near_total_seconds'] and up['cloud_faster_than_near_uplink_bits_per_second'] is None)
check('deployment_energy',close(cases['twenty-rounds']['near']['energy_joules'][0],600*(tiers['near']['prepare_seconds']+20*tiers['near']['model_seconds_per_round'])) and close(cases['twenty-rounds']['cloud']['energy_joules'][0],700*(tiers['cloud']['prepare_seconds']+20*tiers['cloud']['model_seconds_per_round'])) and cases['twenty-rounds']['end']['energy_joules']==[900*.576,900*.756] and edge['cases'][0]['lowest_energy_feasible']=='cloud')
check('measured_ratio_rows',close(up['measured_ratio'],25.83/9.12) and round(up['measured_near_total_seconds'],1)==16.0 and round(up['measured_cloud_fixed_seconds'],1)==14.3 and round(up['measured_cloud_faster_uplink_bits_per_second']/1e6)==73)
check('encoder_placement',close(edge['encoder']['devices'][0]['encode_seconds'],1310300569600/165.2e12) and close(edge['encoder']['devices'][1]['encode_seconds'],1310300569600/989.4e12) and round(edge['encoder']['local_path_seconds'],2)==10.25 and round(edge['encoder']['remote_path_seconds'],2)==1.00 and close(edge['encoder']['fast_link_transfer_difference_seconds'],(8192000-800000)/50e9))
check('wifi_sync_startup',close(144*134e-6,.019296) and 144*134e-6>2*9.12e-3)
rp=json.loads((ROOT/'calculations/results/region-placement-agent-session.json').read_text());hb=rp['hold_boundary']
check('text_session_hold',rp['inputs']['device_flops_per_second']==989400000000000 and close(hb['hold_seconds']['numerator']/hb['hold_seconds']['denominator'],(296505803538432-60380764176384)/989.4e12/(465371136/80e9),1e-6) and round(hb['hold_seconds']['numerator']/hb['hold_seconds']['denominator'])==41)
for name,c in d['12-4']['comparisons'].items():
 for source,values in zip(c['sources'],c['image_audio_seconds']):
  b={x['id']:float(Fraction(x['complete'])) for x in json.loads((ROOT/'calculations/results'/f'{source}.json').read_text())['businesses']}
  check('media_saved_values:'+source,values==[b['image'],b['audio']])
check('no_control_characters',all(ord(c)>=32 or c=='\n' for c in text))
expected_subsets=['three_latency_curves','first_five_audio_chunks','image_and_complete_ec_only','delivery_order_only','exchange_components_only','matched_ASR_only','latency_thresholds_only']
for i,subset in enumerate(expected_subsets,1):
 check('single_relationship:'+str(i),d[f'12-{i}']['plotted_subset']==subset and d[f'12-{i}']['axes_count']==1)
check('screenshot_integer_boundary',37*2.78<105<38*2.78)
check('screenshot_baseline_components',close(.3+2+.2+.8*8/6.4,3.5))
check('connection_reuse_30_rounds',close((30-1)*2*.2,11.6))
check('buffer_depletion',.12<15360/(256000-130000)<.13)
check('deployment_examples',round(up['cloud_fixed_seconds']+128/4,1)==43.6 and round(up['cloud_fixed_seconds']+128/3.5,1)==48.2 and round(45-T[2],1)==13.4)
check('rounded_queqiao_prose','约 0.24 s | 约 0.24 s' in text)
check('migration_extra_restore_threshold',21*.4 < 64*2**20*8/80e6+2 <22*.4)
check('cloud_near_difference',close(20*((1-.08)+(.2-.02))-20*(tiers['near']['model_seconds_per_round']-tiers['cloud']['model_seconds_per_round'])-(tiers['near']['prepare_seconds']-tiers['cloud']['prepare_seconds']),T[2]-T[1]) and round(T[2]-T[1],1)==20.6)
check('variable_bandwidth_penalty',close(up['variable_uplink_seconds'],up['cloud_fixed_seconds']+10*6.4/6+10*6.4/10) and round(up['variable_uplink_seconds'],1)==28.7 and round(up['constant_uplink_seconds'],1)==27.6)
rc=tiers['cloud']['round_seconds']
check('recovery_retained_progress',close(cases['twenty-retain-nine']['cloud']['total_seconds'],T[2]+1+rc) and round(T[2]+1+rc,1)==34.2)
check('recovery_lost_progress',close(cases['twenty-lose-ten']['cloud']['total_seconds'],T[2]+1+10*rc) and round(T[2]+1+10*rc,1)==48.3 and not cases['twenty-lose-ten']['cloud']['meets_deadline'] and round(cases['twenty-lose-ten']['near']['total_seconds'],1)==17.4)
check('thirty_round_continuity','前十轮已完成，剩余二十轮' in text and '每轮 3.5 秒，继续执行需要 70 秒' in text)
check('mechanism_explanations',all(x in text for x in ['资源竞争','最大值','若连接准备与发送等待是主要原因','记录受其影响的事件发生时刻与请求总时间']))
check('figures_in_reading_order',[e['number'] for e in active_catalog]==list(range(1,figure_count+1)))
check('figure_catalog_paths',all((HERE/e['file']).exists() for e in catalog))
check('source_backed_overlap',close(d['12-8']['chunked_s'],12.36) and d['12-8']['chunk_return_s']==[.08,.16,.16])
check('buffer_double_duration',close(d['12-9']['empty_s'][1],2*d['12-9']['empty_s'][0]))
check('migration_figure',d['12-11']['minimum_rounds']==[20,22])
check('layer_sync_figure',d['12-12']['layers']*d['12-12']['reductions_per_layer']*d['12-12']['startup_stages_per_reduction']==144)
check('window_figure_timing',close(d['12-13']['batch_send_s'],.0256) and close(d['12-13']['cycle_s'],.1256))
check('multipath_figure',sum(d['12-14']['split_MB'])==30 and d['12-14']['shared_min_s']==10)
check('deployment_figure_components',all(close(x,y) for x,y in zip(d['12-15']['totals_s'],T)))
check('recovery_figure',all(close(1+rc*n,t) for n,t in zip(d['12-16']['redo_rounds'],d['12-16']['extra_s'])))
refs=[int(x) for x in re.findall(r'图 12-(\d+)',text)]
check('all_figure_references_resolve',all(1<=x<=figure_count for x in refs))
# 12.1.5 on-device execution resources; numbers come from the energy-ledger result and book formulas.
el=json.loads((ROOT/'calculations/results/energy-ledger-book.json').read_text())['summary']
w_bytes,kv_bytes=el['weight_read_bytes'],el['kv_read_bytes']
check('phone_channel_bandwidth',close(16*10.6/8,21.2) and close(4*21.2,el['phone_bus_gb_per_second']))
check('phone_weight_pass_seconds',close(w_bytes/84.8e9,.1785,1e-4) and close(1000/(w_bytes/84.8e9*1000),5.60,.01))
check('phone_step_with_kv',close((w_bytes+kv_bytes)/84.8e9*1000,192.75,.01) and close(84.8e9/(w_bytes+kv_bytes),5.188,.001))
check('q4_step',w_bytes*18//64==4257230400 and close((4257230400+kv_bytes)/84.8e9*1000,64.45,.01) and close(84.8e9/(4257230400+kv_bytes),15.52,.01))
check('q4_resident_weight',16381470720*18//64==4607288640)
check('q8_resident_weight',16381470720*34//64==8702656320)
check('kv_per_token',kv_bytes//8192==147456)
check('phone_12gb_capacity',(12-4)*10**9-4607288640==3392711360 and 3392711360//kv_bytes==2 and close(3392711360/147456,23008,1))
check('phone_16gb_capacity',(16-4)*10**9-4607288640==7392711360 and 7392711360//kv_bytes==6 and close(7392711360/147456,50135,1))
check('phone_6gb_too_small',(6-4)*10**9<4607288640)
check('phone_24gb_capacity',(24-4)*10**9-4607288640==15392711360 and 15392711360//kv_bytes==12 and close(15392711360/147456,104389,1) and ((24-4)*10**9-16381470720)//kv_bytes==2)
check('phone_16gb_q8',(16-4)*10**9-8702656320==3297343680 and 3297343680//kv_bytes==2)
check('melt_energy_joules',close(.16*3.6,.576) and close(.21*3.6,.756) and close(14.8165*.576,8.53,.01))
check('tier_bounds',close((w_bytes+kv_bytes)/819e9*1000,19.96,.01) and close((w_bytes+kv_bytes)/1792e9*1000,9.12,.005))
check('tier_token_rates',close(819e9/(w_bytes+kv_bytes),50.1,.05) and close(1792e9/(w_bytes+kv_bytes),109.6,.05))
check('local_tiers_figure',rendered_data['12-local-tiers']['step_bytes']==w_bytes+kv_bytes and rendered_data['12-local-tiers']['rows_ms'][2]==64.4)
# 12.3.3 loss is not congestion; numbers come from the two wan-loss-model results.
wan={p:json.loads((ROOT/'calculations/results'/f'wan-loss-model-{p}.json').read_text())['summary'] for p in ['book','p036']}
wb_,wp=wan['book'],wan['p036']
check('mathis_bound',close(wb_['mathis_mbit_per_second'],1448*8/.2/math.sqrt(.14)/1e6) and close(wb_['mathis_mbit_per_second'],.1548,1e-3))
check('mathis_send_only',close(354640*8/(wb_['mathis_mbit_per_second']*1e6),18.33,.01))
check('bbr_ideal',wb_['bdp_bytes']==333e6*.2/8 and 2*wb_['bdp_bytes']==16.65e6 and close(wb_['bbr_ideal_goodput_mbit_per_second'],.86*333))
check('bbr_vs_mathis',close(wb_['bbr_ideal_goodput_mbit_per_second']/wb_['mathis_mbit_per_second'],1850,1))
check('retransmit_tail',wb_['packets']==245 and close(wb_['expected_losses'],34.3) and close(wb_['serial_budget_seconds'],.2+.030+354640*8/333e6))
check('retransmit_completion',close(wb_['expected_completion_seconds'],.756,1e-3) and close(wb_['expected_completion_seconds']/wb_['serial_budget_seconds'],3.17,.01) and wb_['p99_rounds']==6 and close(wb_['p99_completion_seconds'],1.239,1e-3))
check('fec_repair',wb_['fec_repair_symbols']==63 and close(wb_['fec_overhead_ratio'],63/245) and wp['fec_repair_symbols']==20 and close(wp['fec_overhead_ratio'],20/245))
check('fec_completion',close(.230+(245+63)*1448*8/333e6,.2407,1e-3))
check('p036_tail',close(wp['p99_completion_seconds'],.8385,1e-3) and wp['p99_rounds']==4 and close(wp['mathis_mbit_per_second'],.3053,1e-3))
check('loss_repair_figure',rendered_data['12-loss-repair']['fec_repair']==63 and close(rendered_data['12-loss-repair']['serial_budget_s'],.2385,1e-3))
# A compact arithmetic answer key verifies that the newly specified exercises have determinate inputs.
answers={'12-1':{'compression_break_even_Mbps':120/.15,'only_0_2_s_accelerated_total_s':12.8-.2+.02},'12-2':{'compressed_38_rounds_s':38*2.78,'largest_faster_integer_rounds':37},'12-3':{'rtx4090_encode_s':1310300569600/165.2e12,'h100_encode_s':1310300569600/989.4e12,'extra_edge_encoding_s':1310300569600/165.2e12+10.24-(1+1310300569600/989.4e12),'migration_preparation_s':64*2**20*8/80e6+1},'12-5':{'ideal_airtime_saved_s':15400*134/1e6,'buffer_depletion_s':(.06*256000)/(256000-130000)},'12-6':{'ideal_split_fraction':2/3,'ideal_seconds':30*8/30,'shared_bottleneck_lower_s':30*8/24,'joint_failure':.02+.98*.1*.05},'12-8':{}}
answers['12-2']['RTT_plus_100ms_totals_s']=[30*3.6,38*2.88]
answers['12-3']['fast_link_extra_transfer_ms']=(8192000-800000)/50e9*1000
answers['12-3']['extra_restore_minimum_rounds']=22
answers['12-4']={'connection_new_each_s':12,'connection_reuse_s':.4,'saved_s':11.6}
answers['12-5']['24kHz_60ms_buffer_depletion_s']=.06*384000/(384000-130000)
answers['12-8']={'twenty_rounds_times_s':T,'twenty_rounds_energy_J':[cases['twenty-rounds'][k]['energy_joules'] for k in ('end','near','cloud')],'cloud_deadline_uplink_Mbps':up['cloud_deadline_uplink_bits_per_second']/1e6,'cloud_time_without_upload_s':up['cloud_fixed_seconds'],'near_time_s':up['near_total_seconds'],'ten_rounds_times_s':[cases['ten-rounds'][k]['total_seconds'] for k in ('end','near','cloud')],'ten_rounds_energy_J':[cases['ten-rounds'][k]['energy_joules'] for k in ('end','near','cloud')],'ten_rounds_23s_choice':edge['cases'][1]['lowest_energy_feasible'],'ten_rounds_15s_choice':edge['cases'][2]['lowest_energy_feasible'],'variable_bandwidth_twenty_rounds_s':up['variable_uplink_seconds'],'constant_8Mbps_s':up['constant_uplink_seconds']}
answers['12-9']={'bus_GBps':4*16*10.6/8,'bf16_step_s':(15136819200+1207959552)/84.8e9,'bf16_token_per_s':84.8e9/(15136819200+1207959552),'q4_step_s':(4257230400+1207959552)/84.8e9,'q4_token_per_s':84.8e9/(4257230400+1207959552),'phone_12gb_free_bytes':3392711360,'phone_12gb_8k_requests':2,'phone_12gb_single_context_tokens':3392711360//147456,'power_ceiling_token_per_s':[13.8/.756,13.8/.576]}
answers['12-10']={'mathis_Mbps':[1448*8/.2/math.sqrt(.14)/1e6,1448*8/.2/math.sqrt(.036)/1e6],'send_only_s':[354640*8/(1448*8/.2/math.sqrt(.14)),354640*8/(1448*8/.2/math.sqrt(.036))],'expected_losses':34.3,'rounds_cdf':'(1-p**n)**245','expected_completion_s':.7559545776801243,'p99_completion_s':1.2385198798798798,'fec_repair':[63,20],'fec_overhead':[63/245,20/245],'fec_completion_s':[.230+(245+63)*1448*8/333e6,.230+(245+20)*1448*8/333e6],'retransmit_p99_s':[1.2385198798798798,.8385198798798799]}
answers['12-8']['ten_rounds_recovery']={cid:{'times_s':[cases[cid][k]['total_seconds'] for k in ('end','near','cloud')],'energy_J':[cases[cid][k]['energy_joules'] for k in ('end','near','cloud')],'remaining_after_disconnect_s':[None]+[cases[cid][k]['extra_seconds']+2*tiers[k]['round_seconds'] for k in ('near','cloud')],'choice':next(c['lowest_energy_feasible'] for c in edge['cases'] if c['id']==cid)} for cid in ('ten-retain-seven','ten-lose-eight')}
answers['12-13']={'rand_s':1.60/0.40,'bundle_s':1.60/3.20,'speedup':8.0,'w_max':int(1.50//0.35),'reuse_at_w_max':1-1/math.sqrt(4),'win_s':1.60*(1-0.5)/3.20,'win_token_per_s':3.20/(1.60*0.5)}
answers['12-14']={'h_min_ms':250+90+260,'k_min':(250+90+260)//20,'disconnect_survival_ms':600-90}
check('ufs_flash_streaming',close(1.28/0.40,3.20) and close(1.28/3.20,0.40) and close(1.28*(1-0.75)/3.20,0.10))
check('vla_action_chunking',25*20==500 and 500-80==420 and 500-(200+80)==220)
check('case_study_links','../case-studies/macbook-rtx-pro6000.md' in text and '../case-studies/queqiao.md' in text)
check('extended_exercises',re.findall(r'^\*\*拓展练习 (12-\d+)\s',text,re.M)==[f'12-{i}' for i in range(11,20)])
(HERE/'exercise-check.json').write_text(json.dumps(answers,ensure_ascii=False,indent=2)+'\n')
report={'passed':not errors,'checks':len(checks),'sections':6,'subsections':len(text_subs),'figures':figure_count,'exercises':19,'sources':len(json.loads((HERE/'sources.json').read_text())['sources']),'local_links':len(links),'chinese_characters':len(re.findall(r'[\u4e00-\u9fff]',text)),'errors':errors}
(HERE/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2));raise SystemExit(bool(errors))
