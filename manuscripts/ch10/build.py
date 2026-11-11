#!/usr/bin/env python3
"""Generate chapter-ten vector figures, data, and an offline reading edition."""
from pathlib import Path
from fractions import Fraction
import argparse, base64, hashlib, html, json, re, subprocess, runpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import markdown
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--font');args=parser.parse_args()
import sys
sys.path.insert(0,str(HERE.parent))
from figure_style.typography import configure_font
font,family=configure_font(args.font)
plt.rcParams.update({'font.family':[family,'DejaVu Sans'],'font.size':11,'axes.unicode_minus':False,'svg.fonttype':'path','svg.hashsalt':'ch10-training-v1','axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white','text.color':'#203c48','axes.labelcolor':'#203c48','pdf.fonttype':42})
C={'ink':'#203c48','blue':'#286b98','teal':'#16857b','orange':'#bc722b','red':'#a94c52','pale':'#eef4f7','light':'#eaf5f1','sand':'#fbf0e5','line':'#c3d0d7','muted':'#546e7a'}
for x in json.loads((HERE/'sources.json').read_text())['sources']:
 if hashlib.sha256((ROOT/x['path']).read_bytes()).hexdigest()!=x['sha256']:raise SystemExit('Review changed source: '+x['path'])
def read(p):return json.loads((ROOT/p).read_text())
def calc(name):return read('calculations/results/'+name+'.json')
subprocess.run(['python3',str(HERE/'design-case.py')],check=True)
outputs=[];data={};layout=[]
FIGURE_RENUMBER = {'figure-10-1-state': 'figure-10-1-state', 'figure-10-2-budget': 'figure-10-2-budget', 'figure-10-3-candidates': 'figure-10-6-candidates', 'figure-10-4-pipeline': 'figure-10-7-pipeline', 'figure-10-5-recovery': 'figure-10-12-recovery', 'figure-10-6-rl': 'figure-10-15-rl', 'figure-10-7-replay': 'figure-10-17-replay', 'figure-10-8-hardware': 'figure-10-19-hardware', 'figure-10-9-scale': 'figure-10-20-scale'}
def save(f,name):
 name=FIGURE_RENUMBER.get(name,name)
 f.canvas.draw();renderer=f.canvas.get_renderer()
 for t in f.findobj(matplotlib.text.Text):
  if not t.get_visible() or not t.get_text():continue
  # Tick labels outside active limits are not painted by Matplotlib.
  if getattr(t,'axes',None) and not t.axes.get_visible():continue
  b=t.get_window_extent(renderer)
  if b.x0<0 or b.y0<0 or b.x1>f.bbox.width or b.y1>f.bbox.height:
   layout.append({'figure':name,'text':t.get_text(),'bbox':[round(v,2) for v in b.bounds]})
 for ext in ['svg','png','pdf']:
  p=HERE/(name+'.'+ext);f.savefig(p,dpi=180,bbox_inches='tight',pad_inches=.15);outputs.append(p)
 plt.close(f)
def canvas(height=8):
 f,a=plt.subplots(figsize=(13,height));f.subplots_adjust(left=.025,right=.975,top=.98,bottom=.035);a.set(xlim=(0,1),ylim=(0,1));a.axis('off');return f,a
def box(a,x,y,w,h,title,body='',col='pale',size=12):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.004,rounding_size=0.009',ec=C['line'],fc=C[col],lw=1.1))
 a.text(x+w/2,y+h*(.69 if body else .5),title,ha='center',va='center',fontsize=size,weight='bold')
 if body:a.text(x+w/2,y+h*.27,body,ha='center',va='center',fontsize=10,linespacing=1.4,color=C['muted'])
def arrow(a,p,q,col='teal',rad=0,lw=1.5):a.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=13,lw=lw,color=C[col],connectionstyle=f'arc3,rad={rad}'))
def panel(a,x,y,t):a.text(x,y,t,fontsize=14,weight='bold',va='top')
# All figures have their captions in the manuscript, not on the drawing canvas.
state=calc('training-state-book');deadline=calc('training-deadline-book')
N=state['summary']['parameters'];data['source_scope']={'state_parameters':N,'deadline_matrix_flops':deadline['summary']['task_training_matrix_flops']}
# 1. One relation: training persistent state versus inference weights.
f,ax=plt.subplots(figsize=(11.5,4.5));f.subplots_adjust(left=.14,right=.96,bottom=.20,top=.86)
vals=np.array([[2*N,0,0,0,0],[2*N,2*N,4*N,4*N,4*N]])/1e9;left=np.zeros(2)
for j,(label,col) in enumerate([('BF16 权重','blue'),('BF16 梯度','teal'),('FP32 主权重','orange'),('Adam 一阶矩','red'),('Adam 二阶矩','muted')]):
 ax.barh([1,0],vals[:,j],left=left,height=.42,label=label,color=C[col]);left+=vals[:,j]
for y,v in zip([1,0],left):ax.text(v+1.5,y,f'{v:.1f} GB',va='center',fontsize=11)
ax.set(yticks=[1,0],yticklabels=['推理权重','训练状态'],xlabel='容量 / GB',xlim=(0,153),xticks=[0,20,40,60,80,100,120,140],ylim=(-.65,1.75));ax.legend(ncol=3,fontsize=10,frameon=False,loc='upper left');ax.grid(axis='x',alpha=.15)
save(f,'figure-10-1-state');data['10-1']={'allocations_bytes':(vals*1e9).astype(int).tolist(),'training_to_inference_ratio':8}

# 2. Integer resource lower bounds, all from existing calculation rows.
f,ax=plt.subplots(figsize=(12,6));f.subplots_adjust(bottom=.17,top=.89,left=.09,right=.96)
devs=['rtx4090','a100-80gb-sxm','h100-sxm','b200-sxm'];labels=['RTX 4090','A100 80GB SXM','H100 SXM','B200'];x=np.arange(4);rows=deadline['training_deadline_rows'];series={}
for j,(ef,label,col) in enumerate([('3/10','计算：30%','blue'),('2/5','计算：40%','teal'),('1/2','计算：50%','orange')]):
 ys=[next(r['compute_count_bound'] for r in rows if r['device']==d and r['matrix_work_efficiency_exact']==ef) for d in devs];series[ef]=ys;ax.bar(x+(j-1)*.21,ys,.20,label=label,color=C[col]);
 for xx,yy in zip(x+(j-1)*.21,ys):ax.text(xx,yy+.65,str(yy),ha='center',fontsize=10)
caps=[next(r['persistent_capacity_count_bound'] for r in rows if r['device']==d) for d in devs];ax.scatter(x,caps,marker='D',s=55,facecolors='white',edgecolors=C['red'],linewidths=1.7,label='训练状态容量下限',zorder=5)
ax.set(xticks=x,xticklabels=labels,ylabel='必要设备数（向上取整）',ylim=(0,49));ax.legend(ncol=2,frameon=False,loc='upper right',fontsize=11);ax.grid(axis='y',alpha=.18)

save(f,'figure-10-2-budget');data['10-2']={'devices':devs,'compute':series,'capacity':caps}
# 3. One relation: shard count versus simultaneous memory requirement.
f,ax=plt.subplots(figsize=(11,6));f.subplots_adjust(left=.12,right=.94,bottom=.18,top=.91)
participants=np.arange(4,33);persist=16*N/participants/2**30;total=persist+10
ax.plot(participants,total,color=C['blue'],lw=2.3,label='训练状态分片 + 10 GiB 附加量')
ax.plot(participants,persist,color=C['teal'],lw=1.6,ls=':',label='训练状态分片')
ax.fill_between(participants,total,22,where=total<=22,interpolate=True,color=C['teal'],alpha=.17,label='剩余显存')
ax.axhline(22,color=C['red'],ls='--',label='RTX 4090 每卡 22 GiB 可用显存')
for d in [8,16]:
 v=16*N/d/2**30+10;ax.plot(d,v,'o',color=C['blue']);ax.annotate(f'{d} 卡：{v:.1f} GiB',(d,v),xytext=(15,16),textcoords='offset points',fontsize=12,arrowprops={'arrowstyle':'-','color':C['muted']})
ax.set(xlim=(4,32),ylim=(0,47),xticks=[4,8,12,16,24,32],xlabel='分片参与者数',ylabel='每卡容量 / GiB');ax.legend(frameon=False,fontsize=10,loc='upper right');ax.grid(alpha=.15)
save(f,'figure-10-3-candidates');data['10-3']={'participants':participants.tolist(),'persistent_gib':persist.tolist(),'total_gib':total.tolist(),'extra_live_gib':10,'net_budget_gib':22,'device':'rtx4090'}

# 4. Real generated event schedules, not invented pipeline rectangles.
f=plt.figure(figsize=(13,11));policies=[('training-pipeline-gpipe-m8','填满排空'),('training-pipeline-1f1b-m8','1F1B')];pipedata={}
for idx,(name,label) in enumerate(policies):
 d=calc(name);ax=f.add_axes([.09,.68-idx*.285,.86,.225]);events=[e for e in d['events'] if e['kind'] in ['F','B','update']]
 for ev in events:
  if ev['stage'] is None:continue
  col={'F':'blue','B':'orange','update':'teal'}[ev['kind']];start=ev['start']*1000;dur=ev['duration']*1000;y=3-ev['stage'];ax.broken_barh([(start,dur)],(y-.30,.6),facecolors=C[col],edgecolors='white',linewidth=.35)
  if dur>=9:ax.text(start+dur/2,y,str(ev['microbatch']+1),ha='center',va='center',fontsize=7,color='white')
 ax.set(yticks=[0,1,2,3],yticklabels=['阶段 3','阶段 2','阶段 1','阶段 0'],xlim=(0,355),xticks=[0,50,100,150,200,250,300,350],ylim=(-.65,3.65),xlabel='时间 / ms');ax.set_title(label+f"：{d['summary']['step_makespan_seconds']*1000:.0f} ms",loc='left',fontsize=13);ax.grid(axis='x',alpha=.15)
 if idx==1:
  ax.plot([93,95],[0,0],color=C['red'],lw=4,zorder=6)
  ax.annotate('输入未到：2 ms',xy=(94,0),xytext=(145,.43),fontsize=9,color=C['red'],arrowprops={'arrowstyle':'->','color':C['red']},bbox={'facecolor':'white','edgecolor':'none','alpha':.94})
 pipedata[label]={'events':events,'summary':d['summary'],'scenario':d['scenario']}
ax=f.add_axes([.09,.09,.57,.205]);x=np.arange(4)
for idx,(_,label) in enumerate(policies):ax.bar(x+(idx-.5)*.34,np.array(pipedata[label]['summary']['reserved_activation_scope_peak_bytes'])/1e9,.32,color=C[['blue','orange'][idx]],label=label)
ax.set(xticks=x,xticklabels=['阶段 0','阶段 1','阶段 2','阶段 3'],ylabel='中间结果与缓冲区峰值 / GB',ylim=(0,3.1));ax.legend(frameon=False,fontsize=10)
f.text(.71,.23,'蓝：前向；橙：反向\n绿：参数更新\n块内数字：micro-batch 编号',fontsize=11,va='top',linespacing=1.65)
save(f,'figure-10-4-pipeline');data['10-4']=pipedata
# 5. One relation: upload bandwidth determines recoverable progress at failure.
f,ax=plt.subplots(figsize=(12.5,5.7));f.subplots_adjust(left=.17,right=.80,bottom=.18,top=.88);checkpointdata={}
for j,(name,label,col) in enumerate([('checkpoint-async-rounded','7 GB/s','blue'),('checkpoint-async-fast','20 GB/s','teal')]):
 d=calc(name);checkpointdata[label]=d['checkpoint_save_rows']
 for row in d['checkpoint_save_rows']:
  t=row['seconds'];y=3-(j*2+row['snapshot'])
  ax.broken_barh([(t['capture'],t['staging_end']-t['capture'])],(y-.24,.48),facecolors=C['orange'])
  ax.broken_barh([(t['upload_start'],min(t['durable'],50)-t['upload_start'])],(y-.24,.48),facecolors=C[col])
  if t['durable']>50:ax.broken_barh([(50,t['durable']-50)],(y-.24,.48),facecolors='white',edgecolors=C[col],hatch='///')
  ax.plot(t['durable'],y,'o',markerfacecolor=C['red'] if t['durable']<=50 else 'white',markeredgecolor=C['red'])
  ax.text(t['durable']+.5,y,f"{t['durable']:g}",va='center',fontsize=10)
ax.axvline(50,ls='--',color=C['red']);ax.text(50,3.75,'故障：50 s',ha='center',color=C['red'],fontsize=12)
ax.set(xlim=(18,59),xticks=[20,30,40,50],ylim=(-.6,4.1),yticks=[3,2,1,0],yticklabels=['7 GB/s · 快照 1','7 GB/s · 快照 2','20 GB/s · 快照 1','20 GB/s · 快照 2'],xlabel='时间 / s');ax.grid(axis='x',alpha=.15)
f.text(.83,.73,'恢复到 20 s',fontsize=12,color=C['blue']);f.text(.83,.51,'恢复到 40 s',fontsize=12,color=C['teal']);f.text(.83,.28,'橙：复制到缓冲区\n实心点：已提交\n空心点：尚未提交',fontsize=10,linespacing=1.8)
save(f,'figure-10-5-recovery');data['10-5']={'timelines':checkpointdata,'failure_seconds':50,'recoverable_capture_seconds':{'7 GB/s':20,'20 GB/s':40}}

# 6. One relation: allocation order changes the handoff peak.
f,ax=plt.subplots(figsize=(12,6));f.subplots_adjust(left=.10,right=.94,bottom=.25,top=.90)
d=calc('weight-handoff-qwen8')['summary'];v=d['phase_live_bytes'];x=np.arange(4)
restore=[v['training'],v['restore_all_before_release'],v['rollout'],v['rollout']]
staged=[v['training'],v['sync_weights_before_release'],v['sync_weights_before_release']-40*2**30,v['rollout']]
for ys,label,col in [(restore,'先恢复权重和 KV，再释放训练状态','red'),(staged,'先同步权重，再释放训练状态、分配 KV','blue')]:
 ax.step(x,np.array(ys)/2**30,where='post',lw=2.3,color=C[col],label=label);ax.plot(x,np.array(ys)/2**30,'o',color=C[col])
CAP=80e9/2**30;ax.axhline(CAP,ls='--',color=C['muted']);ax.text(2.82,CAP+1.5,'H100 SXM：74.5 GiB',ha='right',fontsize=11)
for yy,dy in [(restore[1]/2**30,3),(staged[1]/2**30,-6)]:ax.annotate(f'{yy:.1f} GiB',(1,yy),xytext=(12,dy),textcoords='offset points',fontsize=12)
ax.annotate('',xy=(1.63,restore[1]/2**30),xytext=(1.63,staged[1]/2**30),arrowprops={'arrowstyle':'<->','color':C['orange'],'lw':1.5});ax.text(1.7,66,'24 GiB\n一份 KV 池',fontsize=11,color=C['orange'],va='center')
ax.set(xticks=x,xticklabels=['训练迭代结束','加载生成权重','释放训练状态','开始生成'],ylabel='显存占用 / GiB',ylim=(0,103),yticks=[0,20,40,60,80,100],xlim=(-.1,3.15));ax.legend(frameon=False,fontsize=11,loc='upper left');ax.grid(axis='y',alpha=.15)
save(f,'figure-10-6-rl');data['10-6']={'phase_live_bytes':v,'net_budget_gib':80e9/2**30,'device':'h100-sxm','restore_bytes':restore,'staged_bytes':staged,'x_unit':'ordered steps, not elapsed time'}

# 7. One relation: preserve discrete choices, recompute current values.
f,a=canvas(6.2)
box(a,.03,.65,.27,.20,'生成时选择','样本 A · token 17 · 层 3',col='pale',size=13)
box(a,.38,.65,.24,.20,'保存专家 ID','[2, 7]',col='sand',size=13)
box(a,.71,.65,.26,.20,'当前路由计算','当前 token 的隐藏状态与权重',size=13)
arrow(a,(.305,.75),(.37,.75))
box(a,.37,.19,.27,.22,'按记录选专家 2、7','离散选择保持一致',col='sand',size=13)
box(a,.71,.19,.26,.22,'计算输出和梯度','使用当前分数与专家权重',col='light',size=13)
a.annotate('',xy=(.5,.425),xytext=(.5,.64),arrowprops={'arrowstyle':'-|>','color':C['orange'],'linestyle':'--','lw':2});a.text(.46,.52,'重放',ha='right',fontsize=12,color=C['orange'])
arrow(a,(.645,.30),(.705,.30));arrow(a,(.84,.64),(.84,.425));a.text(.86,.53,'重算',fontsize=12,color=C['teal'])
a.text(.04,.33,'按样本、位置和层查找记录\n样本 A / token 17 / 层 3',fontsize=11,linespacing=1.8)
save(f,'figure-10-7-replay');data['10-7']={'illustration':{'sample':'A','token':17,'layer':3,'experts':[2,7]},'relation':'replay discrete IDs; recompute scores, outputs and gradients'}

# 8. Hypothetical sensitivity, no unsupported cross-hardware performance bars.
f,ax=plt.subplots(figsize=(11,6.5));f.subplots_adjust(left=.10,right=.95,bottom=.17,top=.91);rat=np.linspace(.25,1.5,200);curves={}
for frac,col in [(.05,'teal'),(.2,'blue'),(.5,'orange')]:
 y=1-frac+frac/rat;curves[str(frac)]=y.tolist();ax.plot(rat,y,color=C[col],lw=2.4,label=f'原方案通信等待占比 {frac:.0%}');ax.scatter([.5],[1-frac+frac/.5],color=C[col])
ax.axvline(1,ls='--',color=C['line']);ax.axhline(1,ls='--',color=C['line']);ax.set(xlabel='新方案通信带宽 / 原方案带宽',ylabel='新方案耗时 / 原方案耗时',xlim=(.25,1.5),xticks=[.25,.5,.75,1,1.25,1.5],ylim=(.8,2.65));ax.legend(frameon=False,fontsize=12);ax.grid(alpha=.15)

save(f,'figure-10-8-hardware');data['10-8']={'kind':'declared_sensitivity','formula':'1-f+f/r','relative_bandwidth':rat.tolist(),'curves':curves}
# 9. Required device count at a fixed deadline; derive from source work/peak rows.
f,ax=plt.subplots(figsize=(11.5,6.3));f.subplots_adjust(left=.12,right=.89,bottom=.18,top=.91)
rows=calc('dense-training-scale-book')['dense_scale_rows'];scaledata={}
for dev,label,col in [('a100-80gb-sxm','A100 80GB SXM','blue'),('h100-sxm','H100 SXM','teal'),('b200-sxm','B200','orange')]:
 rr=sorted([r for r in rows if r['device']==dev and r['efficiency_exact']=='2/5'],key=lambda r:r['parameters'])
 half=sorted([r for r in rows if r['device']==dev and r['efficiency_exact']=='1/2'],key=lambda r:r['parameters'])
 per_t=rr[0]['training_days']*16384/90;per_half=half[0]['training_days']*16384/90
 xx=np.linspace(.1,10,200);yy=xx*per_t
 ax.plot(xx,yy/1e4,color=C[col],lw=2,label=label);ax.plot(xx,xx*per_half/1e4,color=C[col],lw=1.2,ls='--')
 ax.annotate(f'{yy[-1]/1e4:.0f} 万卡',(10,yy[-1]/1e4),xytext=(7,0),textcoords='offset points',va='center',fontsize=10,color=C[col])
 crossing=16384/per_t;ax.plot(crossing,1.6384,'o',color=C[col])
 scaledata[dev]={'source_rows':rr,'efficiency':'2/5','deadline_days':90,'parameters':(xx*1e12).tolist(),'continuous_required_devices':yy.tolist(),'parameters_at_16384_devices':crossing*1e12,'continuous_required_devices_mfu50':(xx*per_half).tolist(),'source_rows_mfu50':half}
ax.axhline(1.6384,color=C['muted'],ls='--',lw=1);ax.text(6.5,1.76,'16,384 张卡',fontsize=11,color=C['muted'])
ax.text(5.6,112,'各曲线上方满足对应计算需求',fontsize=11,color=C['blue'])
ax.set(yscale='log',xlabel='Dense 参数量 / T',ylabel='90 天所需设备 / 万张（对数刻度）',xticks=[.1,1,5,10],xlim=(.1,10.6),ylim=(.1,150))
ax.set_yticks([.1,1,10,100],labels=['0.1','1','10','100']);ax.legend(frameon=False,fontsize=11,loc='upper left');ax.grid(axis='y',alpha=.15)
save(f,'figure-10-9-scale');data['10-9']=scaledata
# Gap-filling: schedule variants beyond 1F1B, and MoE capacity-factor rows.
sched={}
for key,stem in [('1F1B','training-pipeline-1f1b'),('交错式 v=2','training-pipeline-interleaved'),('零气泡','training-pipeline-zero-bubble'),('DualPipe','training-pipeline-dualpipe')]:
 d8=calc(stem+'-m8');d16=calc(stem+'-m16');s8=d8['summary']
 sched[key]={'makespan_m8_ms':s8['step_makespan_seconds']*1000,'makespan_m16_ms':d16['summary']['step_makespan_seconds']*1000,'idle_per_gpu_m8_ms':d8['stages'][0]['idle_during_training_seconds']*1000,'stage_peaks_m8_mib':[b/2**20 for b in s8['reserved_activation_scope_peak_bytes']],'forward_messages_m8':d8['transfers']['forward_messages'],'boundary_crossings_per_microbatch':d8['transfers'].get('boundary_crossings_per_microbatch',3),'bubble_bounds_ms':{k:v*1000 for k,v in (d8.get('bubble_reference') or {}).items() if isinstance(v,(int,float))}}
 # Per-stage timelines for the three bubble-compressing schedules, drawn from the recorded events only.
 if key!='1F1B':
  chunk={v['stage']:v['chunk'] for v in d8['partition']['virtual_stages']}
  events=[{'kind':e['kind'],'stage':e['stage'],'microbatch':e['microbatch'],'start':e['start'],'duration':e['duration'],'direction':e['direction'],'chunk':chunk.get(e['virtual_stage'],0)} for e in d8['events'] if e['stage'] is not None and e['kind'] in ['F','B','X','W','update']]
  data['pipeline_timeline_'+stem.removeprefix('training-pipeline-')]={'label':{'交错式 v=2':'交错式 1F1B（v=2）','零气泡':'零气泡（ZB-H1）','DualPipe':'DualPipe'}[key],'events':events,'summary':s8,'scenario':d8['scenario'],'declared_schedule_rule':d8.get('declared_schedule_rule')}
data['pipeline_schedules']=sched
mc=calc('moe-capacity-book')
data['moe_capacity']={'capacity_factors':[1,1.25,1.5,2],'model_experts':mc['model_example']['experts'],'model_top_k':mc['model_example']['top_k'],'model_tokens':mc['model_example']['tokens'],'model_assignments':mc['model_example']['assignments'],'model_capacity':[r['capacity_per_expert'] for r in mc['model_example']['rows']],'model_dropped_fraction':[float(Fraction(r['dropped_fraction_exact'])) for r in mc['model_example']['rows']],'model_padded_fraction':[float(Fraction(r['padded_fraction_of_executed_exact'])) for r in mc['model_example']['rows']],'chapter_capacity':[r['capacity_per_expert'] for r in mc['chapter_example']['rows']],'chapter_dropped':[r['dropped_total'] for r in mc['chapter_example']['rows']],'chapter_padded':[r['padded_total'] for r in mc['chapter_example']['rows']]}
runpy.run_path(str(HERE/'mechanism-figures.py'),init_globals=globals())
# Public figure-data keys follow the same order as the printed chapter.
old_numbers={1:1,2:2,3:6,4:7,5:12,6:15,7:17,8:19,9:20}
previous={f'10-{n}':data.pop(f'10-{n}') for n in old_numbers}
for old,new in old_numbers.items():data[f'10-{new}']=previous[f'10-{old}']
new_numbers={3:'sharding_lifetime',4:'recompute_lifetime',5:'cast_paths',8:'communication_window',9:'attention_area',10:'input_queue',11:'checkpoint_resharding',13:'checkpoint_tradeoff',14:'rl_supply_flow',16:'rl_async_cycle',18:'task_deadline_breakdown',21:'pipeline_timeline_interleaved',22:'pipeline_timeline_zero-bubble',23:'pipeline_timeline_dualpipe',24:'pipeline_schedules',29:'moe_capacity'}
for number,key in new_numbers.items():data[f'10-{number}']=data[key]

import sys
sys.path.insert(0,str(HERE.parent))
from teaching_revision import draw as draw_teaching
teaching_outputs,teaching_checks=draw_teaching(HERE,data)
mimo_outputs,mimo_check=runpy.run_path(str(HERE/'mimo-interruptions.py'))['draw']()
teaching_checks.append(mimo_check)
(HERE/'teaching-layout-validation.json').write_text(json.dumps(teaching_checks,ensure_ascii=False,indent=2)+'\n')
data['mimo_interruptions']=read('references/files/documents/mimo-v26-rl-run-log/recalculated.json')
outputs=list(dict.fromkeys(outputs+teaching_outputs+mimo_outputs))
(HERE/'figure-data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
(HERE/'figure-layout-check.json').write_text(json.dumps({'text_extent_warnings':layout},ensure_ascii=False,indent=2)+'\n')
# Render formulas on the build machine; bundle all image/font bytes into HTML.
md=HERE.parent/'10-训练系统.md';raw=md.read_text();maths=[]
def protect(match):
 text=match[0];display=text.startswith('$$');latex=text[2:-2] if display else text[1:-1];token=f'MATHPLACEHOLDER{len(maths)}END';maths.append({'latex':latex.strip(),'display':display,'token':token});return '\n\n'+token+'\n\n' if display else token
protected=re.sub(r'\$\$[\s\S]*?\$\$|\$[^$\n]+\$',protect,raw)
body=markdown.markdown(protected,extensions=['tables','footnotes','fenced_code','toc'],output_format='html')
node="const fs=require('fs'),k=require(process.argv[1]);const a=JSON.parse(fs.readFileSync(0,'utf8'));process.stdout.write(JSON.stringify(a.map(x=>k.renderToString(x.latex,{displayMode:x.display,throwOnError:true,output:'htmlAndMathml'}))))"
res=subprocess.run(['node','-e',node,str(HERE.parent/'ch06/vendor/katex/katex.js')],input=json.dumps(maths),text=True,capture_output=True)
if res.returncode:raise SystemExit(res.stderr)
for entry,rendered in zip(maths,json.loads(res.stdout)):
 body=body.replace('<p>'+entry['token']+'</p>',rendered) if entry['display'] else body.replace(entry['token'],rendered)
math_css=(HERE.parent/'ch06/vendor/katex/katex.min.css').read_text()
def font_url(m):
 p=HERE.parent/'ch06/vendor/katex'/m[1];return 'url(data:font/'+p.suffix[1:]+';base64,'+base64.b64encode(p.read_bytes()).decode()+')'
math_css=re.sub(r'url\((fonts/[^)]+)\)',font_url,math_css)
body=re.sub(r'<table>(.*?)</table>',r'<div class="table-scroll"><table>\1</table></div>',body,flags=re.S)
for p in outputs:
 if p.suffix=='.svg':body=body.replace('src="ch10/'+p.name+'"','src="data:image/png;base64,'+base64.b64encode(p.with_suffix('.png').read_bytes()).decode()+'"')
css='''*{box-sizing:border-box}body{margin:0;background:#f7f7f4;color:#243640;font:18px/1.95 Georgia,"Songti SC",serif}main{max-width:1020px;margin:auto;padding:48px 46px 85px;background:#fff}h1,h2,h3{font-family:Arial,"PingFang SC",sans-serif;line-height:1.45;color:#183949}h1{font-size:36px}h2{font-size:28px;border-top:1px solid #d5e1e4;margin-top:65px;padding-top:26px}h3{font-size:23px;margin-top:38px}a{color:#286b98;text-underline-offset:3px;overflow-wrap:anywhere}img{display:block;width:100%;height:auto;margin:28px auto 10px}em{font-size:15px;color:#546e7a}table{border-collapse:collapse;width:100%;font-size:15px;line-height:1.7;margin:22px 0}td,th{padding:10px 12px;border-bottom:1px solid #d5e1e4;text-align:left}th{background:#edf4f6}blockquote{margin:27px 0;padding:14px 24px;border-left:4px solid #16857b;background:#f0f7f4;font-size:16px}code{font:0.85em/1.6 Menlo,monospace;background:#f0f4f6;overflow-wrap:anywhere}pre{overflow-x:auto;padding:16px;max-width:100%}nav{font:16px/1.9 Arial,"PingFang SC",sans-serif;background:#eff5f7;padding:18px 24px}nav a{display:block}.footnote{font-size:14px;line-height:1.8}.footnote li{margin-bottom:13px}.table-scroll{overflow-x:auto;max-width:100%}.katex{font-size:1.04em}.katex-display{overflow-x:auto;overflow-y:hidden;padding:12px 0}@media(max-width:650px){main{padding:24px 18px}body{font-size:17px}h1{font-size:29px}h2{font-size:25px}h3{font-size:21px}}@media print{main{max-width:none;padding:0}h2,h3{break-after:avoid}img,blockquote{break-inside:avoid}body{font-size:11pt}}'''
css+='main{max-width:760px;padding-left:24px;padding-right:24px}img{max-width:720px}@media print{img{width:420pt;max-width:100%}}'
nav=''.join('<a href="#'+ident+'">'+title+'</a>' for ident,title in re.findall(r'<h2 id="([^"]+)">(10\.\d+ [^<]+)</h2>',body))
page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第 10 章 训练系统</title><style>'+css+math_css+'</style></head><body><main><nav aria-label="本章目录">'+nav+'</nav>'+body+'</main></body></html>'
import sys
sys.path.insert(0,str(HERE.parent))
from teaching_reading import readable_diagrams
from preview_output import preview_path
page=readable_diagrams(page)
hp=preview_path(md);hp.write_text(page)
(HERE/'math-validation.json').write_text(json.dumps({'renderer':'KaTeX 0.16.11','expressions':len(maths),'display_expressions':sum(x['display'] for x in maths),'errors':[]},indent=2)+'\n')
from book_assets import sync_figure_index
active_assets=sync_figure_index(HERE)
artifacts=outputs+[HERE/'teaching_revision.py',HERE/'h2_fp8_nvfp4_figures.py',HERE/'figure-index.json',HERE/'teaching-layout-validation.json',HERE/'figure-data.json',HERE/'design-case.json',HERE/'design-case.md',hp,md]+active_assets
(HERE/'manifest.json').write_text(json.dumps({'chapter':10,'generator':'manuscripts/ch10/build.py','figures':len(teaching_checks),'font_family':family,'outputs':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in artifacts if not (p.parent==ROOT/'manuscripts' and re.match(r'^[01][0-9]-',p.name))]},ensure_ascii=False,indent=2)+'\n')
print(f'Built {len(outputs)} image files, {len(maths)} formulas, offline HTML; {len(layout)} extent warnings.')
