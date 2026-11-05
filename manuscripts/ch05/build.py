#!/usr/bin/env python3
"""Rebuild chapter 5 illustrations and offline reading edition from locked evidence."""
from pathlib import Path
import argparse,base64,hashlib,html,json,re,subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle
import numpy as np
import markdown
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--font');args=parser.parse_args()
import sys
sys.path.insert(0,str(HERE.parent))
from figure_style.typography import configure_font
font,family=configure_font(args.font)
plt.rcParams.update({'font.family':[family,'DejaVu Sans'],'font.size':11,'axes.unicode_minus':False,'svg.fonttype':'path','svg.hashsalt':'ch05-operators-v1','axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white','text.color':'#193441','axes.labelcolor':'#193441'})
C={'ink':'#193441','blue':'#246f91','teal':'#138b83','orange':'#c9782b','pale':'#f0f6f8','light':'#e7f3ef','sand':'#fcf2e7','muted':'#55707d','line':'#cbd8df','red':'#b65757'}
for entry in json.loads((HERE/'sources.json').read_text())['sources']:
 b=(ROOT/entry['path']).read_bytes()
 if not b.startswith(b'version https://git-lfs.github.com/spec/v1') and hashlib.sha256(b).hexdigest()!=entry['sha256']:raise SystemExit('Review changed source: '+entry['path'])
def read(path):return json.loads((ROOT/path).read_text())
from teaching_revision import draw as draw_teaching_revision
outputs=[];data={};warnings=[]
def save(fig,name):
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 for t in fig.findobj(matplotlib.text.Text):
  if not t.get_visible() or not t.get_text():continue
  bb=t.get_window_extent(renderer)
  if bb.x0<0 or bb.y0<0 or bb.x1>fig.bbox.width or bb.y1>fig.bbox.height:warnings.append({'figure':name,'text':t.get_text()})
 for ext in ['svg','png']:
  p=HERE/f'{name}.{ext}';fig.savefig(p,dpi=170,bbox_inches='tight',pad_inches=.16);outputs.append(p)
 plt.close(fig)
def canvas(h=8):
 f,a=plt.subplots(figsize=(13,h));f.subplots_adjust(left=.025,right=.975,top=.97,bottom=.04);a.set(xlim=(0,1),ylim=(0,1));a.axis('off');return f,a
def box(a,x,y,w,h,title,body='',color='pale',size=12):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.005,rounding_size=0.01',edgecolor=C['line'],facecolor=C[color]));a.text(x+w/2,y+h*(.68 if body else .5),title,ha='center',va='center',fontsize=size,weight='bold')
 if body:a.text(x+w/2,y+h*.27,body,ha='center',va='center',fontsize=10,linespacing=1.5,color=C['muted'])
def arrow(a,p,q,col='teal',rad=0):a.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=15,lw=1.5,color=C[col],connectionstyle=f'arc3,rad={rad}'))
# All figure numbers and full titles are in the manuscript captions, never on the canvas.
data['5-2']={'kind':'teaching','relationship':'local capacity versus rereads','capacity_KiB':[8,24,80],'interface_MiB':[6168,3096,1560],'minimum_logical_MiB':128}

# One K tile of the 5.1.5 kernel on one H100 SM: HBM bandwidth and dense BF16 peak split evenly over 132 SMs.
H100_HBM,H100_BF16,H100_SMS=3.35e12,989.4e12,132
COPY_US=2*(128*64+64*128)*H100_SMS/H100_HBM*1e6;COMPUTE_US=2*128*128*64*H100_SMS/H100_BF16*1e6
serial=[];pipelined=[];copy_end=[];compute_end=[]
for i in range(4):
 serial.extend([{'kind':'copy','tile':i,'slot':i%2,'start':i*(COPY_US+COMPUTE_US),'duration':COPY_US},{'kind':'compute','tile':i,'slot':i%2,'start':i*(COPY_US+COMPUTE_US)+COPY_US,'duration':COMPUTE_US}])
 cs=max(copy_end[-1] if copy_end else 0,compute_end[i-2] if i>=2 else 0);copy_end.append(cs+COPY_US)
 ks=max(copy_end[i],compute_end[-1] if compute_end else 0);compute_end.append(ks+COMPUTE_US)
 pipelined.extend([{'kind':'copy','tile':i,'slot':i%2,'start':cs,'duration':COPY_US},{'kind':'compute','tile':i,'slot':i%2,'start':ks,'duration':COMPUTE_US}])
data['5-6']={'kind':'teaching_timeline','device':'h100-sxm, one of 132 SMs','copy_bytes':32768,'compute_flops':2097152,'copy_us':COPY_US,'compute_us':COMPUTE_US,'serial':serial,'double_buffer':pipelined,'completion_us':[4*(COPY_US+COMPUTE_US),compute_end[-1]]}

data['5-5']={'kind':'fixed_scale_teaching','required_MiB':[60,60,60],'T_write_read_MiB':[48,0,0],'Z_write_read_MiB':[48,48,0]}

# Show loop nesting and storage lifetimes, rather than a list of transformation names.
f,a=canvas(8.5)
a.text(.04,.965,'先保存完整矩阵',fontsize=17,weight='bold')
a.text(.54,.965,'逐块计算并激活',fontsize=17,weight='bold')
box(a,.035,.56,.405,.34,'',color='pale')
a.text(.06,.865,'for i, j:\n    acc = 0\n    for k:\n        acc += A[i,k] × W[k,j]\n    C[i,j] = BF16(acc)',va='top',fontsize=13,linespacing=1.6)
box(a,.035,.36,.405,.12,'完整 C：24 MiB','写出一次，再读入一次','sand',14)
arrow(a,(.24,.55),(.24,.49))
box(a,.035,.12,.405,.16,'',color='light')
a.text(.06,.24,'for i, j:\n    Y[i,j] = SiLU(C[i,j])',va='top',fontsize=13,linespacing=1.6)
arrow(a,(.24,.35),(.24,.29))
arrow(a,(.455,.64),(.515,.64))
a.text(.485,.77,'split / tile\nreorder',ha='center',va='center',fontsize=10,linespacing=1.7,color=C['muted'])
# The enclosing output-tile rectangle, the inner reduction, and the epilogue make scope visible.
a.add_patch(Rectangle((.53,.10),.43,.80,facecolor=C['pale'],edgecolor=C['blue'],lw=1.6))
a.text(.55,.865,'for io, jo:   16 × 192 个输出块',fontsize=13)
a.add_patch(Rectangle((.555,.72),.38,.09,facecolor=C['sand'],edgecolor=C['line']))
a.text(.575,.765,'acc：64 × 64，FP32，16 KiB',fontsize=12,va='center')
a.add_patch(Rectangle((.575,.36),.335,.31,facecolor='white',edgecolor=C['teal'],lw=1.6))
a.text(.595,.63,'for ko:   128 个归约块',fontsize=13)
a.text(.595,.55,'A 块 64 × 32：4 KiB\nW 块 32 × 64：4 KiB',fontsize=12,va='top',linespacing=1.7)
a.text(.595,.395,'acc += A 块 × W 块',fontsize=13)
a.add_patch(Rectangle((.555,.155),.38,.12,facecolor=C['light'],edgecolor=C['teal']))
a.text(.575,.235,'c = BF16(acc)\n写出 SiLU(c)',fontsize=13,va='top',linespacing=1.6)
arrow(a,(.745,.35),(.745,.28))
a.text(.76,.315,'归约结束',fontsize=10,color=C['teal'])
a.text(.04,.025,'C 的写出与读取：48 MiB',fontsize=13,color=C['orange'])
a.text(.54,.025,'输入槽随 ko 更新；acc 跨 ko 保留',fontsize=12,color=C['blue'])
save(f,'figure-5-9-polyhedral');data['5-9']={'kind':'loop_and_storage_scope','tile':[64,64,32],'output_tiles':3072,'k_tiles':128,'input_KiB':8,'accumulator_KiB':16,'eliminated_read_write_MiB':48}

f,a=plt.subplots(figsize=(10.5,5.6));f.subplots_adjust(left=.11,right=.97,top=.92,bottom=.16)
p=np.linspace(0,1,301);a.plot(p*100,np.full_like(p,10),lw=2.4,color=C['blue'],label='原实现：10 μs')
a.plot(p*100,20-15*p,lw=2.4,color=C['teal'],label='新实现：20 − 15p μs')
a.axvline(200/3,color=C['orange'],ls='--');a.text(200/3+2,19,'A 占比 2/3',color=C['orange'],fontsize=11)
for x,y,label,offset in [(50,12.5,'各出现一次：新实现慢 25%',(-160,65)),(10000/101,520/101,'A ×100、B ×1：新实现约快 1.9 倍',(-260,-30))]:
 a.plot(x,y,'o',color=C['teal']);a.annotate(label,(x,y),xytext=offset,textcoords='offset points',arrowprops={'arrowstyle':'->','color':C['muted']},fontsize=11)
a.set_xticks(range(0,101,20));a.set(xlim=(0,103),ylim=(0,22),xlabel='形状 A 的调用占比 / %',ylabel='每次调用平均时间 / μs');a.legend(frameon=False,loc='lower left')
save(f,'figure-5-11-feedback');data['5-11']={'kind':'teaching','relationship':'shape frequency versus average execution time','baseline_per_shape_us':[10,10],'candidate_per_shape_us':[5,20],'crossing_A_fraction':2/3,'frequencies':[[1,1],[100,1]]}

spec=read('calculations/results/specialization-medium.json');policies=spec['specialization_policies']
# Field names validated once against the saved computation; never infer an actual GPU time.

f,a=plt.subplots(figsize=(12,5.5));f.subplots_adjust(left=.10,bottom=.17,right=.95,top=.93);rs=np.arange(1,151);labels={'generic':'通用','bucket':'分桶','specialized':'逐形状特化'}
for z,col in zip(policies,['blue','orange','teal']):a.plot(rs,(z['prepare_ns']+rs*z['cohort_execution_ns'])/1e6,label=labels[z['policy']],lw=2.5,color=C[col])
for left,right in [(policies[0],policies[1]),(policies[1],policies[2])]:
 v=(right['prepare_ns']-left['prepare_ns'])/(left['cohort_execution_ns']-right['cohort_execution_ns']);a.axvline(v,color=C['line'],ls='--')
a.set_xticks(range(0,151,25));a.set(xlabel='指定十次调用组的重复次数',ylabel='准备 + 累计执行 / ms',xlim=(0,150),ylim=(0,2850));a.legend(frameon=False);a.text(14,2600,'1–64：通用',fontsize=11);a.text(66,2600,'65–89：分桶',fontsize=11);a.text(102,2600,'90 起：特化',fontsize=11)
save(f,'figure-5-14-specialization');data['5-14']={'kind':'teaching_from_calculations','source':'calculations/results/specialization-medium.json','policies':policies}
trace=read('experiments/ch05/05-08/results/trace-analysis.json');ranges=trace['ranges'][:4]
f,axs=plt.subplots(4,1,figsize=(13,9.5));f.subplots_adjust(left=.10,right=.98,top=.95,bottom=.09,hspace=.67)
for a,z,label in zip(axs,ranges,['普通提交','融合','图重放','融合 + 图重放']):
 origin=z['start_ns'];end=max([z['end_ns']]+[v['end_ns'] for v in z['kernels']]);duration=(end-origin)/1000
 for ev in z['apis']:
  if 'Launch' not in ev['name']:continue
  a.broken_barh([((ev['start_ns']-origin)/1000,(ev['end_ns']-ev['start_ns'])/1000)],(.66,.25),facecolors=C['orange'])
 for ev in z['kernels']:a.broken_barh([((ev['start_ns']-origin)/1000,(ev['end_ns']-ev['start_ns'])/1000)],(.16,.25),facecolors=C['blue'])
 a.set_yticks([.28,.78],['设备 kernel','主机 launch']);a.set_ylim(0,1.05);a.set_xlim(0,900);a.set_xticks(range(0,901,150));a.set_xlabel('相对各自采集起点 / μs',fontsize=10);a.set_title(f"{label}：主机 kernel launch {z['launch_api_count']-z['graph_launch_count']}，graph launch {z['graph_launch_count']}；设备 kernel {z['kernel_count']}",loc='left',fontsize=12)
save(f,'figure-5-12-runtime');data['5-12']={'kind':'measured_profile','source':'experiments/ch05/05-08/results/trace-analysis.json','ranges':ranges}
# A single dependency relationship before/after shortening branch A.
f,a=canvas(6.3)
for y,hot,total,critical,label in [(.58,60,80,'A','原始'),(.10,15,60,'B','A 加快四倍后')]:
 a.text(.035,y+.29,label+'：'+str(total)+' μs',fontsize=15,weight='bold')
 box(a,.035,y+.07,.145,.105,'准备 10 μs',color='pale',size=12)
 box(a,.32,y+.18,.25,.105,'A：'+str(hot)+' μs',color='sand' if critical=='A' else 'pale',size=14)
 box(a,.32,y-.005,.25,.105,'B：40 μs',color='sand' if critical=='B' else 'pale',size=14)
 box(a,.735,y+.07,.22,.105,'收尾 10 μs',color='pale',size=12)
 for branch,by in [('A',y+.2325),('B',y+.0475)]:
  color='orange' if critical==branch else 'line'
  arrow(a,(.185,y+.1225),(.315,by),color)
  arrow(a,(.575,by),(.73,y+.1225),color)
 a.text(.615,y+.26,'等待两支完成',fontsize=11,color=C['muted'])
a.text(.035,.02,'橙色标出决定完成时间的路径',fontsize=11,color=C['orange'])
save(f,'figure-5-16-critical-path');data['5-16']={'kind':'teaching_dependency_graph','prepare_us':10,'branch_A_us':[60,15],'branch_B_us':40,'finish_us':10,'completion_us':[80,60],'critical_branch':['A','B']}

summary=read('experiments/ch05/05-09/results/summary.json');pairs=summary['pairs'];v=[z['latency_saving_ms'] for z in pairs]
f,a=plt.subplots(figsize=(11,5.5));f.subplots_adjust(left=.12,right=.97,top=.93,bottom=.17)
x=np.arange(1,12);a.axhline(0,lw=1,color=C['ink']);a.vlines(x,0,v,color=[C['teal'] if z>0 else C['red'] for z in v],lw=2)
a.scatter(x,v,c=[C['teal'] if z>0 else C['red'] for z in v],s=65,zorder=3)
a.axhline(np.median(v),ls='--',color=C['orange'],label=f'配对时间差中位数：{np.median(v):.1f} ms')
a.set_yticks(range(-3,6));a.set_xticks(x);a.set(xlim=(.5,11.5),ylim=(-3.2,5.3),xlabel='配对轮次',ylabel='替换前 − 替换后的请求时间 / ms');a.legend(frameon=False,loc='upper left',fontsize=11)
a.text(7.3,4.6,'9 对更快，2 对更慢\n请求总时间约 817 ms',ha='right',va='top',fontsize=11,linespacing=1.7)
save(f,'figure-5-17-request');data['5-17']={'kind':'measured_paired_requests','relationship':'paired full-request time savings','source':'experiments/ch05/05-09/results/summary.json','pairs':pairs,'median_saving_ms':float(np.median(v)),'baseline_median_ms':summary['rows'][0]['median_latency_ms']}

# Added diagrams explain data movement and dependencies before comparing totals.
f,a=plt.subplots(figsize=(11,4.6));f.subplots_adjust(left=.18,right=.96,top=.9,bottom=.18)
for y,kernel,label in [(1,20,'原 kernel'),(0,5,'kernel 加快四倍')]:
 start=0
 for duration,name,col in [(3,'提交','orange'),(8,'H2D','teal'),(kernel,'kernel','blue'),(4,'D2H','muted')]:
  a.barh(y,duration,left=start,height=.42,color=C[col]);a.text(start+duration/2,y,name,ha='center',va='center',color='white',fontsize=10);start+=duration
 a.text(start+.5,y,f'{start} μs\n结果可用',va='center',fontsize=11)
a.set(xlim=(0,40),ylim=(-.65,1.65),xlabel='从 CPU 开始提交起计时 / μs');a.set_yticks([0,1],['kernel 加快四倍','原 kernel']);a.set_xticks([0,3,11,20,31,35]);a.grid(axis='x',alpha=.15)
save(f,'figure-5-1-execution');data['execution']={'kind':'teaching_timeline','submit_us':3,'h2d_us':8,'kernel_us':[20,5],'d2h_us':4,'completion_us':[35,20]}

data['banks']={'kind':'address_mapping','strides_words':[32,33],'banks':32,'column_banks':[[i*s%32 for i in range(32)] for s in [32,33]],'rounds':[32,1]}

data['reduction']={'kind':'dependency_and_reread','row_elements':4096,'segments':8,'partial_elements':512,'rows':1024,'groups':[1024,8192],'traffic_MiB':[24,33656832/2**20]}

data['softmax']={'kind':'online_recurrence','first':[0,1,1],'rescale':.5,'second':[float(np.log(2)),1.5,3.5],'output':7/3}

att_rows=read('calculations/results/attention-tiles-rtxpro6000.json')['attention_tile_rows']
data['attention_tradeoff']={'kind':'teaching_tradeoff','source':'calculations/results/attention-tiles-rtxpro6000.json','capacity_bytes':101376,'kv_rows':[r['kv_block'] for r in att_rows],'q_rows':[r['query_block'] for r in att_rows],'traffic_MiB':[r['interface_bytes']/2**20 for r in att_rows],'updates':[r['kv_block_pairs'] for r in att_rows]}

f,a=canvas(6.6)
for y,fused in [(.57,False),(.12,True)]:
 a.text(.025,y+.31,'在矩阵乘中重新量化' if fused else '保存一份 FP8 量化结果',fontsize=16,weight='bold')
 box(a,.03,y+.05,.18,.18,'FP16 输入','32 MiB',size=13)
 box(a,.34,y+.05,.24,.18,'求行 scale' if fused else '求行 scale，量化并保存','不保存 FP8 矩阵' if fused else '写出 FP8：16 MiB',color='sand' if fused else 'light',size=12)
 arrow(a,(.22,y+.14),(.33,y+.14))
 box(a,.74,y+.05,.23,.18,'12 个输出列块','各读 FP16 并量化' if fused else '各读同一份 FP8',size=12)
 if fused:
  arrow(a,(.59,y+.18),(.73,y+.18),col='line')
  a.text(.66,y+.235,'行 scale',ha='center',fontsize=10,color=C['muted'])
  arrow(a,(.21,y+.075),(.74,y+.075),col='orange',rad=.16)
  a.text(.45,y+.005,'原输入：12 × 32 MiB',ha='center',fontsize=11,color=C['orange'])
 else:arrow(a,(.59,y+.14),(.73,y+.14),col='teal')
 if not fused:a.text(.66,y+.235,'12 × 16 MiB',ha='center',fontsize=11,color=C['teal'])
 a.text(.34,y-.040,'读写：32 + 12 × 32 = 416 MiB' if fused else '读写：32 + 16 + 12 × 16 = 240 MiB',fontsize=12)
a.text(.025,.015,'重复读取的数据宽度翻倍，省下的一次写出不足以抵消它',fontsize=12,color=C['blue'])
save(f,'figure-5-10-quantization');data['quantization']={'kind':'data_reuse','column_tiles':12,'source_MiB':32,'quantized_MiB':16,'input_traffic_MiB':[240,416],'shared_other_MiB':204}

graph_small=read('calculations/results/graph-small-input-rtxpro6000.json');graph_large=read('calculations/results/graph-large-input-rtxpro6000.json')
COPY_BW=graph_small['scenario']['copy_bandwidth_bytes_per_second']
copy_small_us,copy_large_us=graph_small['summary']['extra_copy_ns']/1000,graph_large['summary']['extra_copy_ns']/1000
f,a=plt.subplots(figsize=(11,5));f.subplots_adjust(left=.24,right=.95,top=.85,bottom=.18)
for y,prep,copy in [(2,20,0),(1,5,copy_small_us),(0,5,copy_large_us)]:
 for start,dur,col,label in [(0,prep,'orange',f'{prep:g}'),(prep,copy,'teal',f'{copy:.1f}'),(prep+copy,20,'blue','20')]:
  if dur:a.barh(y,dur,left=start,height=.48,color=C[col]);a.text(start+dur/2,y,label,ha='center',va='center',color='white',fontsize=10)
 a.text(prep+copy+20+.6,y,f'{prep+copy+20:.1f} μs',va='center',fontsize=11)
a.set_yticks([2,1,0],['普通执行','图重放：2 MiB 输入','图重放：16 MiB 输入']);a.set(xlim=(0,52),xticks=range(0,51,10),ylim=(-.65,2.6),xlabel='完成时间 / μs');a.axvline(40,color=C['muted'],ls='--',lw=1)
a.legend(handles=[Rectangle((0,0),1,1,color=C[c],label=t) for c,t in [('orange','准备 / 提交等待'),('teal','复制输入'),('blue','设备计算')]],loc='lower left',bbox_to_anchor=(-.2,1),frameon=False,ncol=3,fontsize=11)
save(f,'figure-5-13-graph-copy');data['graph_copy']={'kind':'teaching_timeline','device':'rtx-pro6000-blackwell-ws','copy_bandwidth_bytes_per_second':COPY_BW,'payload_MiB':[2,16],'prepare_us':[20,5,5],'copy_us':[0,copy_small_us,copy_large_us],'compute_us':20,'completion_us':[40,25+copy_small_us,25+copy_large_us]}

persist=read('calculations/results/persistent-tiles-rtxpro6000.json');ps_=persist['scenario'];tiles_=persist['persistent_tiles']
PROJ_US=tiles_[0]['matrix_flops']/ps_['matrix_flops_per_second']*1e6;ACT_US=tiles_[0]['activation_elements']/ps_['activation_elements_per_second']*1e6
TASK_US=(ps_['task_dispatch_ns']+ps_['event_publish_ns'])/1000;LAUNCH_US=ps_['host_launch_ns']/1000
timelines={}
f,axs=plt.subplots(2,1,figsize=(12,6.5));f.subplots_adjust(left=.12,right=.97,top=.91,bottom=.12,hspace=.75)
for a,fine in zip(axs,[False,True]):
 if fine:
  bars=[(t['nanoseconds']['producer_start']/1000,t['nanoseconds']['producer_ready']/1000,t['nanoseconds']['consumer_start']/1000,t['nanoseconds']['consumer_done']/1000) for t in tiles_]
 else:
  first=2*LAUNCH_US+8*PROJ_US;bars=[(LAUNCH_US+i*PROJ_US,LAUNCH_US+(i+1)*PROJ_US,first+i*ACT_US,first+(i+1)*ACT_US) for i in range(8)]
 timelines['fine' if fine else 'coarse']=[dict(zip(['projection_start','projection_end','activation_start','activation_end'],b)) for b in bars]
 for i,(p0,p1,c0,c1) in enumerate(bars):
  for start,end,y in [(p0,p1,.64),(c0,c1,.13)]:
   a.broken_barh([(start,end-start)],(y,.28),facecolors=C['blue' if i%2==0 else 'teal'])
   if end-start>4:a.text((start+end)/2,y+.14,str(i),ha='center',va='center',color='white',fontsize=10)
 end=max(b[3] for b in bars);a.axvline(bars[0][2],ls='--',color=C['orange']);a.set_xticks(range(0,131,20));a.set(xlim=(0,135),ylim=(0,1.03),xlabel='时间 / μs');a.set_yticks([.27,.78],['激活','投影']);a.set_title(('逐块开始激活' if fine else '全部投影完成后开始激活')+f'：约 {end:.0f} μs',loc='left',fontsize=14)
save(f,'figure-5-15-persistent');data['persistent']={'kind':'teaching_timeline','device':'rtx-pro6000-blackwell-ws','source':'calculations/results/persistent-tiles-rtxpro6000.json','tiles':8,'projection_us':PROJ_US,'activation_us':ACT_US,'task_overhead_us':TASK_US,'launch_us':LAUNCH_US,'completion_us':[persist['summary']['barrier_finish_ns']/1000,persist['summary']['persistent_finish_ns']/1000],'timelines':timelines}

# Preserve existing numerical data identifiers; figure-index.json records current reading order.
new_keys={'execution':1,'banks':3,'reduction':4,'softmax':7,'attention_tradeoff':8,'quantization':10,'graph_copy':13,'persistent':15}
data={('5-'+str(new_keys[k]) if k in new_keys else k):v for k,v in data.items()}
data=dict(sorted(data.items(),key=lambda kv:int(kv[0].split('-')[1])))

revision_outputs,revision_checks=draw_teaching_revision(HERE,data)
import sys
sys.path.insert(0,str(HERE.parent))
from full_teaching_revision import draw as draw_full_revision
full_outputs,full_checks=draw_full_revision(HERE,data)
revision_outputs.extend(full_outputs)
revision_checks.extend(full_checks)
from structure_figures import draw as draw_structure
revision_outputs.extend(draw_structure(HERE,data))
revision_checks.extend(json.loads((HERE/'structure-layout-validation.json').read_text()))
(HERE/'teaching-layout-check.json').write_text(json.dumps(revision_checks,ensure_ascii=False,indent=2)+'\n')
outputs.extend(revision_outputs)
outputs=list(dict.fromkeys(outputs))
warnings.extend({'figure':c['figure'],'text':t} for c in revision_checks for t in c['text_extent_warnings'])
revision_names={p.stem for p in revision_outputs}

(HERE/'figure-data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');(HERE/'figure-layout-check.json').write_text(json.dumps({'text_extent_warnings':warnings},ensure_ascii=False,indent=2)+'\n')
# Offline HTML; KaTeX and its fonts are vendored with the chapter.
md=HERE.parent/'05-算子与运行时.md';raw=md.read_text();maths=[]
figure_refs=re.findall(r'^!\[图 (5-\d+) ([^\]]+)\]\(ch05/([^)]+)\)',raw,re.M)
figure_index=[{'number':n,'title':t,'asset':asset,'revised':Path(asset).stem in revision_names} for n,t,asset in figure_refs]
(HERE/'figure-index.json').write_text(json.dumps(figure_index,ensure_ascii=False,indent=2)+'\n')
def protect_math(match):
 text=match.group(0);display=text.startswith('$$');latex=text[2:-2] if display else text[1:-1];token=f'MATHPLACEHOLDER{len(maths)}END';maths.append({'latex':latex.strip(),'display':display,'token':token});return ('\n\n'+token+'\n\n') if display else token
protected=re.sub(r'\$\$[\s\S]*?\$\$|\$[^$\n]+\$',protect_math,raw)
body=markdown.markdown(protected,extensions=['tables','footnotes','fenced_code','toc'],output_format='html')
node_code="const fs=require('fs'),k=require(process.argv[1]);let a=JSON.parse(fs.readFileSync(0,'utf8'));process.stdout.write(JSON.stringify(a.map(x=>k.renderToString(x.latex,{displayMode:x.display,throwOnError:true,output:'htmlAndMathml'}))))"
result=subprocess.run(['node','-e',node_code,str(HERE/'vendor/katex/katex.js')],input=json.dumps(maths),text=True,capture_output=True)
if result.returncode:raise SystemExit(result.stderr)
for entry,rendered in zip(maths,json.loads(result.stdout)):body=body.replace('<p>'+entry['token']+'</p>',rendered) if entry['display'] else body.replace(entry['token'],rendered)
math_css=(HERE/'vendor/katex/katex.min.css').read_text()
def font_data(match):
 p=HERE/'vendor/katex'/match[1];return 'url(data:font/'+p.suffix[1:]+';base64,'+base64.b64encode(p.read_bytes()).decode()+')'
math_css=re.sub(r'url\((fonts/[^)]+)\)',font_data,math_css)
body=re.sub(r'<table>(.*?)</table>',r'<div class="table-scroll"><table>\1</table></div>',body,flags=re.S)
def wrap_figure(match):
 img,asset,caption=match[1],match[2],match[3]
 revised=Path(asset).stem in revision_names
 content=('<div class="diagram-scroll">'+img+'</div><p class="diagram-hint">左右滑动查看完整配图</p>') if revised else img
 return '<figure class="'+('teaching-figure' if revised else 'chapter-figure')+'" data-asset="'+asset+'">'+content+'<figcaption>'+caption+'</figcaption></figure>'
body=re.sub(r'<p>(<img [^>]*src="ch05/([^"]+)"[^>]*>)</p>\s*<p><em>([\s\S]*?)</em></p>',wrap_figure,body)
for p in outputs:
 if p.suffix=='.svg':body=body.replace('src="ch05/'+p.name+'"','src="data:image/png;base64,'+base64.b64encode(p.with_suffix('.png').read_bytes()).decode()+'"')
css='''body{margin:0;background:#fafaf8;color:#243640;font:18px/1.95 Georgia,"Songti SC",serif}main{max-width:960px;margin:auto;padding:50px 38px 90px;background:white}h1,h2,h3{font-family:Arial,"PingFang SC",sans-serif;line-height:1.45;color:#163747}h1{font-size:36px}h2{font-size:28px;border-top:1px solid #d5e1e4;margin-top:65px;padding-top:28px}h3{font-size:23px;margin-top:40px}p{margin:1em 0}a{color:#246f91;text-underline-offset:3px}img{display:block;width:100%;height:auto;margin:26px auto 10px}em{font-size:15px;color:#55707d}table{border-collapse:collapse;width:100%;font-size:15px;line-height:1.7;margin:24px 0}td,th{padding:10px 12px;border-bottom:1px solid #d5e1e4;text-align:left}th{background:#edf4f6}blockquote{margin:28px 0;padding:16px 24px;border-left:4px solid #138b83;background:#f1f8f6;font-size:16px}code{font:0.85em/1.65 Menlo,monospace;background:#f0f4f6;overflow-wrap:anywhere}pre{white-space:pre;overflow-x:auto;padding:16px;border:1px solid #d5e1e4;max-width:100%;box-sizing:border-box}nav{font:16px/1.9 Arial,"PingFang SC",sans-serif;background:#f0f6f8;padding:18px 24px}nav a{display:block}.footnote{font-size:14px;line-height:1.8}.footnote li{margin-bottom:12px}.table-scroll{overflow-x:auto;max-width:100%}.katex{font-size:1.04em}.katex-display{overflow-x:auto;overflow-y:hidden;padding:12px 0}@media(max-width:650px){main{padding:25px 18px}body{font-size:17px}h1{font-size:29px}h2{font-size:25px}}@media print{main{max-width:none;padding:0}h2,h3{break-after:avoid}img,blockquote{break-inside:avoid}body{font-size:11pt}}'''
css+=' main{max-width:760px}'
css+='''figure{margin:28px auto}figcaption{font:15px/1.8 Arial,"PingFang SC",sans-serif;color:#50616a;margin-top:12px}.teaching-figure{max-width:720px}.teaching-figure img{margin:0;width:100%;height:auto}.diagram-hint{display:none}.diagram-scroll{overflow-x:auto} @media(max-width:650px){.teaching-figure img{width:560px;min-width:560px;max-width:none}.diagram-hint{display:block;font:13px/1.6 Arial,"PingFang SC",sans-serif;color:#50616a;margin:6px 0}} @media print{.teaching-figure{width:148.167mm;max-width:100%;break-inside:avoid}.teaching-figure img{width:100%;min-width:0}.diagram-scroll{overflow:visible}.diagram-hint{display:none}figcaption{font-size:9pt}body{line-height:1.75}h1{font-size:20pt;margin:0 0 18pt}h2{font-size:16pt;margin:20pt 0 10pt;padding-top:10pt}h3{font-size:13pt;margin:16pt 0 8pt}p{margin:.75em 0;orphans:3;widows:3}figure{margin:14pt auto}.katex-display,pre{break-inside:avoid}.katex-display{padding:5pt 0}table{font-size:10pt}th,td{padding:6pt}.footnote{font-size:9pt;margin-top:20pt}}'''
nav=''.join('<a href="#'+ident+'">'+title+'</a>' for ident,title in re.findall(r'<h2 id="([^"]+)">(5\.\d+ [^<]+)</h2>',body))
page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第 5 章 算子与运行时</title><style>'+css+math_css+'</style><main><nav>'+nav+'</nav>'+body+'</main></html>'
import sys
sys.path.insert(0,str(HERE.parent))
from teaching_reading import readable_diagrams
from preview_output import preview_path
page=readable_diagrams(page)
html_path=preview_path(HERE.parent/'05-算子与运行时.html');html_path.write_text(page)
section_start=re.search(r'<h2 id="[^"]+">5\.2 ',body).start()
section_end=re.search(r'<h2 id="[^"]+">5\.4 ',body).start()
# Keep cited evidence available in the standalone reading sample.
section_body=body[section_start:section_end]
needed=set(re.findall(r'href="#(fn:[^"]+)"',section_body))
section_ids=set(re.findall(r'id="([^"]+)"',section_body))
footnote_items=[]
for i,(ident,content) in enumerate(re.findall(r'<li id="(fn:[^"]+)">([\s\S]*?)</li>',body),1):
 if ident in needed:
  content=re.sub(r'<a class="footnote-backref" href="#([^"]+)"[^>]*>[\s\S]*?</a>',lambda m:m[0] if m[1] in section_ids else '',content)
  footnote_items.append('<li id="'+ident+'" value="'+str(i)+'">'+content+'</li>')
footnotes='<div class="footnote"><hr><ol>'+''.join(footnote_items)+'</ol></div>'
excerpt=preview_path(HERE.parent/'05-算子与运行时-5.2-5.3.html')
excerpt.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第五章 5.2–5.3 图文试改</title><style>'+css+math_css+'</style><main><h1>第五章 5.2–5.3 图文试改</h1>'+body[section_start:section_end]+footnotes+'</main></html>')
(HERE/'math-validation.json').write_text(json.dumps({'renderer':'KaTeX 0.16.11','expressions':len(maths),'display_expressions':sum(x['display'] for x in maths),'errors':[]},indent=2)+'\n')
from book_assets import sync_figure_index
active_assets=sync_figure_index(HERE)
artifacts=outputs+[HERE/'figure-data.json',HERE/'figure-index.json',HERE/'teaching-layout-check.json',html_path,excerpt,md]+active_assets
(HERE/'manifest.json').write_text(json.dumps({'chapter':5,'generator':'manuscripts/ch05/build.py','figures':len(figure_refs),'font_family':family,'outputs':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in artifacts if not (p.parent==ROOT/'manuscripts' and re.match(r'^[01][0-9]-',p.name))]},ensure_ascii=False,indent=2)+'\n')
print(f'Built {len(outputs)} figure files, {len(maths)} equations and offline HTML; {len(warnings)} layout warnings.')
