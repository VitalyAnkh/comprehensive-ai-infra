#!/usr/bin/env python3
"""Generate chapter-six vector figures, data, and an offline reading edition."""
from pathlib import Path
from fractions import Fraction
import argparse, base64, hashlib, html, json, re, subprocess
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
plt.rcParams.update({'font.family':[family,'DejaVu Sans'],'font.size':11,'axes.unicode_minus':False,'svg.fonttype':'path','svg.hashsalt':'ch06-supernodes-v1','axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white','text.color':'#203c48','axes.labelcolor':'#203c48','pdf.fonttype':42})
C={'ink':'#203c48','blue':'#286b98','teal':'#16857b','orange':'#bc722b','red':'#a94c52','pale':'#eef4f7','light':'#eaf5f1','sand':'#fbf0e5','line':'#c3d0d7','muted':'#546e7a'}
for x in json.loads((HERE/'sources.json').read_text())['sources']:
 b=(ROOT/x['path']).read_bytes()
 if not b.startswith(b'version https://git-lfs.github.com/spec/v1') and hashlib.sha256(b).hexdigest()!=x['sha256']:raise SystemExit('Review changed source: '+x['path'])
def read(p):return json.loads((ROOT/p).read_text())
def calc(name):return read('calculations/results/'+name+'.json')
outputs=[];data={};layout=[]
figure_catalog=json.loads((HERE/'figure-catalog.json').read_text())
def save(f,name):
 name=figure_catalog['names'].get(name,name)
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
# Caption numbers are never drawn on any canvas.
# 1: Same cards, different inference-instance grouping; bytes from saved placement.
f,a=canvas(5)
for row,(groups,label) in enumerate([(8,'八个单卡实例'),(2,'两个四卡实例'),(1,'一个八卡实例')]):
 y=.77-row*.20;panel(a,.025,y+.14,label)
 for g in range(groups):
  count=8//groups;x=.22+g*(.75/groups);w=.75/groups-.012
  a.add_patch(FancyBboxPatch((x,y),w,.125,boxstyle='round,pad=0.003',fc=C['light'],ec=C['teal'],lw=1.2))
  for k in range(count):
   gx=x+.008+k*w/count;gw=w/count-.012
   box(a,gx,y+.033,gw,.061,str(g*count+k),size=10)
   if k<count-1:arrow(a,(gx+gw,y+.063),(gx+w/count-.001,y+.063),lw=1)
  if groups<=2:a.text(x+w/2,y+.013,'一个独立推理实例',ha='center',fontsize=9,color=C['teal'])
a.set_ylim(.32,1)
u=2**31;w1=470187269120;k1=1577058304
if (ROOT/'calculations/results/qwen235-placement-tp8-kv-replica.json').read_bytes().startswith(b'version https://git-lfs'):
 _p=json.loads((HERE/'figure-data.json').read_text())['placement']
 w8,k8=_p['capacity_bytes'][1][:2];max_sess=_p['max_sessions_per_server']
else:
 place=calc('qwen235-placement-tp8-kv-replica');fmt=place['ranks'][0]['formats'][0]
 w8=fmt['weight_bytes'];k8=fmt['kv_bytes_per_request'];max_sess=place['cohort'][0]['maximum_requests']
save(f,'figure-6-1-placement');data['6-1']={'kind':'declared_placement','model':'qwen3-235b-a22b','instance_groups':[8,2,1],'capacity_bytes':[[w1,k1,u],[w8,k8,u]],'tp8_saved_max_bytes':w8+k8+u,'max_sessions_per_server':max_sess}
# 2: Explicit TP split dimensions and partial-result flow.
f,a=canvas(8)
box(a,.015,.42,.17,.21,'完整输入 X','m × h',size=14)
for y,r,col in [(.69,0,'light'),(.30,1,'pale')]:
 box(a,.255,y,.265,.22,f'卡 {r}：上投影与激活',f'Wg,{r} / Wu,{r}：h × (f/2)\nZ{r}：m × (f/2)',col=col,size=12)
 box(a,.60,y,.16,.22,f'下投影 Wd,{r}','(f/2) × h',col=col,size=12)
 box(a,.835,y,.14,.22,f'部分和 Y{r}','m × h',col='sand',size=13)
 arrow(a,(.19,.53),(.25,y+.11));arrow(a,(.525,y+.11),(.595,y+.11));arrow(a,(.765,y+.11),(.83,y+.11))
box(a,.60,.04,.375,.13,'完整输出 Y = Y0 + Y1','AllReduce：求和后每卡取得 m × h',col='light',size=13)
arrow(a,(.98,.80),(.98,.105),'orange',rad=-.07)
arrow(a,(.905,.295),(.905,.175),'orange')
a.text(.03,.965,'上投影按列分片 → 激活留在本地 → 下投影按行分片 → 输出求和',fontsize=13,va='top')
save(f,'figure-6-2-tp');data['6-2']={'kind':'tp_partial_outputs_with_pipeline_text_data','ffn_dims':[4096,12288,4096],'tp4_weight_bytes':3*5120*6400*2,'pipeline_stages':4,'microbatches':4,'slots':7,'utilization':4/7}
# Independent PP diagram: one microbatch per color, one millisecond per cell.
f,ax=plt.subplots(figsize=(11,4.6));f.subplots_adjust(left=.11,right=.98,bottom=.18,top=.9)
for stage in range(4):
 for batch in range(4):
  start=stage+batch
  ax.add_patch(Rectangle((start,3-stage-.32),.94,.64,fc=C[['blue','teal','orange','red'][batch]]))
  ax.text(start+.47,3-stage,f'micro-batch {batch}',ha='center',va='center',color='white',fontsize=12)
ax.set(xlim=(0,7.1),ylim=(-.6,3.6),yticks=[3,2,1,0],yticklabels=['阶段 0','阶段 1','阶段 2','阶段 3'],xticks=range(8),xlabel='时间 / ms')
ax.axvline(4,color=C['muted'],ls=':',lw=1);ax.text(4.05,3.4,'首个结果',fontsize=11)
ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
save(f,'figure-6-3-pipeline')
# Independent token dispatch/combine ownership.
f,a=canvas(6.7)
box(a,.02,.29,.26,.29,'卡 0：token A','router 选择专家 1、6',col='light',size=14)
box(a,.40,.64,.25,.21,'本地专家 1','计算 y1',size=14)
box(a,.40,.12,.25,.21,'卡 3：专家 6','计算 y6',size=14)
box(a,.75,.29,.23,.29,'卡 0：加权合并','a1 y1 + a6 y6',col='sand',size=14)
arrow(a,(.285,.53),(.395,.74));arrow(a,(.285,.36),(.395,.225))
arrow(a,(.655,.74),(.745,.53));arrow(a,(.655,.225),(.745,.36),'orange')
a.text(.305,.20,'dispatch',fontsize=11,ha='right');a.text(.70,.18,'返回 y6',fontsize=11)
a.text(.05,.84,'保存输入的设备',fontsize=12);a.text(.76,.84,'合并结果的设备',fontsize=12)
save(f,'figure-6-4-dispatch')
# Ring: follow one block, then show completed block ownership.
f,a=canvas(8)
panel(a,.025,.965,'块 0：每经过一张卡，就加入该卡的贡献')
for r,val in enumerate([1,11,111,1111]):
 x=.025+r*.248
 box(a,x,.70,.21,.16,f'卡 {r}',f'块 0：{val}',col='light' if r==3 else 'pale',size=13)
 if r<3:
  arrow(a,(x+.215,.78),(x+.24,.78))
  a.text(x+.23,.64,f'第 {r+1} 轮',ha='center',fontsize=10)
panel(a,.025,.54,'ReduceScatter 完成：各卡持有一块完整归约结果')
for r,block in enumerate([1,2,3,0]):
 box(a,.025+r*.248,.34,.21,.13,f'卡 {r}：块 {block}',size=13)
 arrow(a,(.13+r*.248,.33),(.13+r*.248,.23))
 box(a,.025+r*.248,.045,.21,.17,f'卡 {r}','块 0、1、2、3',col='light',size=13)
a.text(.5,.265,'再经三轮 AllGather，交换已归约的数据块',ha='center',fontsize=12)
save(f,'figure-6-6-ring-rounds')
# 3: Equal assignments, different expert reuse.
m=np.arange(1,65);expected=128*(1-(1-8/128)**m)
f,a=canvas(6)
for y,title,experts,rows,col in [(.61,'均匀覆盖',128,4,'blue'),(.15,'集中到同八个专家',8,64,'orange')]:
 a.text(.02,y+.23,title,fontsize=15,weight='bold')
 box(a,.025,y,.23,.17,f'{experts} 个专家 × {rows} 行',f'共 {experts*rows} 次分派',size=13)
 arrow(a,(.27,y+.085),(.35,y+.085),col)
 width=.52*experts/128
 a.add_patch(Rectangle((.37,y+.035),width,.105,fc=C[col]))
 label='4.5 GiB' if experts==128 else '288 MiB'
 a.text(.37+width+.018,y+.09,label,va='center',fontsize=15,color=C[col])
 a.text(.37,y-.04,'128 × 36 MiB' if experts==128 else '8 × 36 MiB',fontsize=12)
a.text(.37,.96,'权重读取量',fontsize=15,weight='bold',va='top')
routes=read('experiments/ch06/06-03/runs/routes-001/route-analysis.json');case='retrieval-2048-A-early'
rows=sorted([x for x in routes['counts'] if x['case_id']==case and x['phase']=='prefill'],key=lambda x:x['layer_id'])
heat=np.array([np.array(x['expert_counts'])/x['valid_tokens'] for x in rows]);assert len(rows)==43 and np.allclose(heat.sum(axis=1),6)

save(f,'figure-6-5-reuse');data['6-3']={'kind':'expert_reuse_with_companion_observation','expected_active_experts':expected.tolist(),'tokens':m.tolist(),'route_case':case,'route_phase':'prefill','route_token_counts':[x['valid_tokens'] for x in rows],'route_heatmap':heat.tolist()}
data['6-3']['balanced_vs_concentrated']={'tokens':64,'top_k':8,'active_experts':[128,8],'rows_per_expert':[4,64],'weight_bytes_per_expert':36*2**20,'weight_read_bytes':[128*36*2**20,8*36*2**20]}
f,ax=plt.subplots(figsize=(10,6));f.subplots_adjust(left=.09,right=.87,bottom=.12,top=.94)
im=ax.imshow(heat,aspect='auto',origin='upper',cmap='YlGnBu',vmin=0,vmax=1,interpolation='nearest')
ax.set(xlabel='专家编号',ylabel='层编号',xticks=[0,64,128,192,255],yticks=[0,10,20,30,42]);f.colorbar(im,ax=ax,label='选择次数 / token')
save(f,'route-observation')
# 4: Message size changes algorithm choice.
f,ax=plt.subplots(figsize=(10,6));f.subplots_adjust(left=.10,right=.96,bottom=.15,top=.93)

from continuity_model import generate
continuous=generate();co=continuous['collectives'];A,BW=co['alpha_s'],co['bandwidth_Bps']
M=np.logspace(2,8,300);ring=14*A+1.75*M/BW;tree=6*(A+M/BW)
ax.loglog(M,ring*1e6,color=C['blue'],lw=2.2,label='Ring');ax.loglog(M,tree*1e6,color=C['orange'],lw=2.2,label='未分段二项树');ax.set(xlabel='每卡完整输入 / bytes',ylabel='模型时间 / μs',xlim=(100,1e8),ylim=(3,5000));ax.legend(frameon=False,loc='upper left');ax.grid(which='major',alpha=.2)
for y,col in [(co['ring_s']*1e6,'blue'),(co['tree_s']*1e6,'orange')]:ax.scatter([10240],[y],color=C[col],s=30)
ax.set_xticks([1e2,1e3,1e4,1e5,1e6,1e7,1e8]);ax.set_yticks([10,100,1000])
cross=co['crossover_bytes']
ax.axvline(cross,color=C['muted'],ls=':',lw=1)
ax.annotate('约 680 KiB：排序翻转',(cross,14*A*1e6+1.75*cross/BW*1e6),xytext=(3e6,40),fontsize=12,arrowprops={'arrowstyle':'->','color':C['muted']})
ax.axvline(10240,color=C['line'],ls='--');ax.text(10240,3.4,'10 KiB',ha='center',fontsize=11)
ax.axvline(80*2**20,color=C['line'],ls='--');ax.text(80*2**20,3.4,'80 MiB',ha='center',fontsize=11)
save(f,'figure-6-7-collectives');data['6-4']={'kind':'h100_nvlink_time_models','participants':8,'startup_seconds':A,'bandwidth_bytes_per_second':BW,'message_bytes':M.tolist(),'ring_seconds':ring.tolist(),'tree_seconds':tree.tolist(),'crossover_bytes':cross,'ring_reference':calc('ring-qwen3-32b-t1-p8-h100')['summary'],'tree_reference':calc('tree-qwen3-32b-t1-p8-h100')['summary']}
# 5: Faster isolated communication can delay the concurrent finish.
f,axs=plt.subplots(1,2,figsize=(12,5.5));f.subplots_adjust(left=.10,right=.95,bottom=.18,top=.87,wspace=.40)
ax=axs[0]
ax.barh([1,0],[2.674,36.919],height=.36,color=[C['blue'],C['orange']])
for y,t in [(1,2.674),(0,36.919)]:ax.text(t+.5,y,f'{t:.2f} ms',va='center',fontsize=12)
ax.set(yticks=[1,0],yticklabels=['4 MiB','64 MiB'],xlabel='时间 / ms',xlim=(0,50),ylim=(-.65,1.65));ax.set_title('通信单独运行',fontsize=14,loc='left')
ax=axs[1]
for base,comm,compute,col in [(1.1,6.141,11.523,'blue'),(0,45.209,12.885,'orange')]:
 ax.barh(base+.15,comm,height=.24,color=C[col],alpha=.45)
 ax.barh(base-.15,compute,height=.24,color=C[col])
 ax.text(.015,base+.15,'通信',va='center',fontsize=10)
 ax.text(.015,base-.15,'计算',va='center',fontsize=10,color='white')
 end=max(comm,compute);ax.plot([end,end],[base-.36,base+.36],ls='--',color=C[col])
 ax.text(end+.5,base,f'{end:.2f} ms',va='center',fontsize=11,color=C[col])
ax.set(yticks=[1.1,0],yticklabels=['4 MiB','64 MiB'],xlabel='从同时开始起的时间 / ms',xlim=(0,56),ylim=(-.65,1.75));ax.set_title('同时运行',fontsize=14,loc='left')

save(f,'figure-6-8-resources');data['6-5']={'kind':'measured_gloo_concurrency','message_MiB':[4,64],'measured_ms':{'independent_comm':[2.674,36.919],'independent_compute':[11.531,11.031],'shared_comm':[6.141,45.209],'shared_compute':[11.523,12.885],'shared_group':[11.582,45.242],'independent_group':[[2.726,11.538],[36.956,11.079]]},'measurement_source':'experiments/ch06/06-05/README.md'}
# 6: Same per-rank sends, different busiest-link traffic.
paths=calc('collective-paths-book')['collective_path_patterns']
f=plt.figure(figsize=(13,5.2))
for idx,z in enumerate(paths):
 ax=f.add_axes([.02+idx*.285,.16,.27,.69]);theta=np.arange(16)*2*np.pi/16+np.pi/2;xy=np.c_[np.cos(theta),np.sin(theta)]
 ax.plot(np.r_[xy[:,0],xy[0,0]],np.r_[xy[:,1],xy[0,1]],color=C['line'],lw=1.5)
 for i,(x,y) in enumerate(xy):
  ax.scatter([x],[y],s=28,c=C['blue']);ax.text(x*1.19,y*1.19,str(i),ha='center',va='center',fontsize=10)
 route=next(x for x in z['rounds'][2]['routes'] if x['sender']==0)
 for u,v in route['path']:arrow(ax,xy[u],xy[v],'orange',rad=.08,lw=2.5)
 ax.set(xlim=(-1.4,1.4),ylim=(-1.4,1.4),aspect='equal');ax.set_xticks([]);ax.set_yticks([]);ax.axis('off')
 ax.set_title(('递归' if idx==0 else 'Swing')+f'：第三轮 0 → {route["receiver"]}',fontsize=13)
ax=f.add_axes([.66,.21,.30,.61]);xx=np.arange(3)
for i,z in enumerate(paths):
 vals=[v['peak_link_bytes']/2**20 for v in z['rounds']]
 ax.bar(xx+(i-.5)*.34,vals,.32,color=C[['blue','orange'][i]],label=['递归','Swing'][i])
 for x,v in zip(xx+(i-.5)*.34,vals):ax.text(x,v+.12,f'{v:g}',ha='center',fontsize=11)
ax.set(xticks=xx,xticklabels=['1','2','3'],xlabel='轮次',ylabel='最大单向链路传输量 / MiB',ylim=(0,5.6));ax.legend(ncol=2,frameon=False,fontsize=11)
numa=[calc(name)['summary'] for name in ['staging-all-a-grouped','staging-local-grouped','staging-local-alternating']]

save(f,'figure-6-9-topology');data['6-6']={'kind':'sourced_topology','ports':{'radix':64,'leaf_options':[[32,32],[48,16]],'port_GBs':25},'collective_patterns':paths,'numa_summaries':numa}
# 7: Each system connects groups through an explicit communication layer.
f,a=canvas(8)
for y,title,left,right,middle,detail in [
 (.72,'NVIDIA：机柜级 NVLink','GPU / CPU 托盘','GPU / CPU 托盘','NVLink 交换','GB200 NVL72：72 GPU、36 CPU'),
 (.40,'TPU：连接电互联单元','64 芯片电互联单元','64 芯片电互联单元','OCS 光路交换','TPU v4：每单元 4 × 4 × 4 芯片'),
 (.08,'UB：连接跨主机资源','NPU / CPU 资源','NPU / CPU 资源','UB 交换','CloudMatrix384：384 NPU、192 CPU')]:
 panel(a,.025,y+.24,title)
 box(a,.025,y+.04,.28,.14,left,col='light',size=12)
 box(a,.385,y+.04,.23,.14,middle,col='sand',size=12)
 box(a,.695,y+.04,.28,.14,right,col='light',size=12)
 arrow(a,(.31,y+.11),(.38,y+.11));arrow(a,(.69,y+.11),(.62,y+.11))
 a.text(.5,y-.014,detail,ha='center',fontsize=11)
save(f,'figure-6-10-systems');data['6-7']={'kind':'sourced_organization_not_performance','nvidia_gpu_counts':[8,16,8,72],'tpu_v4_cube':[4,4,4],'cloudmatrix384':{'NPUs':384,'CPUs':192},'source_note':'Model generations are not normalized performance measurements or inferred design motives.'}
# 8: Remote read frequency versus attainable bandwidth.
from continuity_model import generate
pool=generate()['memory_pool'];demand=np.array(pool['demand_bytes'])/1e9
freq=np.array(pool['frequency_per_s']);demand_bw=np.array(pool['demand_Bps'])/1e9;win=pool['windows'][0]['window_Bps']/1e9
f,ax=plt.subplots(figsize=(10,6));f.subplots_adjust(left=.10,right=.96,bottom=.15,top=.94)
fs=np.logspace(-2,1.5,200);ax.loglog(fs,pool['borrowed_bytes']*fs/1e9,color=C['blue'],lw=2.5)
ax.axhline(50,color=C['red'],ls='--',label='路径：50 GB/s')
ax.axhline(win,color=C['orange'],ls=':',label='128 个在途事务：约 4.36 GB/s')
for i,v in enumerate(demand_bw):
 ax.scatter(freq[i],v,color=C['blue'],s=45,zorder=3)
 label=['每分钟一次\n约 0.33 GB/s','每秒一次\n20 GB/s','每秒二十次\n400 GB/s'][i]
 offset=[(12,-5),(12,-40),(-125,12)][i]
 ax.annotate(label,(freq[i],v),xytext=offset,textcoords='offset points',fontsize=11)
ax.set(xlabel='完整读取 20 GB 的频率 / 次每秒',ylabel='所需平均带宽 / GB/s',xlim=(.01,40),ylim=(.1,1500))
ax.set_xticks([.01,.1,1,10]);ax.set_yticks([.1,1,10,100,1000]);ax.legend(frameon=False,loc='upper left',fontsize=11);ax.grid(which='major',alpha=.15)

save(f,'figure-6-11-memory-pool');data['6-8']={'kind':'h100_pool_cx7_path','demand_GB':demand.tolist(),'physical_use_GB':(np.array(pool['physical_after_bytes'])/1e9).tolist(),'frequency_per_second':freq.tolist(),'mean_payload_GBs':demand_bw.tolist(),'window_bound_GBs':win,'path_GBs':pool['path_Bps']/1e9,'borrowed_GB':pool['borrowed_bytes']/1e9}
# Continuous model: derive service and cost from the same layer execution.
f=plt.figure(figsize=(12,5.8));colors=['muted','blue','teal','orange'];labels=['八个单卡实例','四个两卡实例','两个四卡实例','一个八卡实例']
curve_data={}
for pos,(phase,title) in enumerate([('healthy','无故障条件'),('fault','20 ms 故障，60 ms 开始重做')]):
 ax=f.add_axes([.08+pos*.49,.17,.38,.69]);deadlines=np.arange(20,241,dtype=float);curves=[]
 for i,c in enumerate(continuous['candidates']):
  times=np.array(c[phase+'_completion_ms']);cost=c[phase+'_gpu_seconds']
  # Include exact completions so each step occurs at the actual event, not a sampled deadline.
  ds=np.unique(np.r_[deadlines,times[(times>=20)&(times<=220)]])
  counts=(times[:,None]<=ds[None,:]).sum(axis=0);ys=np.full(len(ds),np.nan);ok=(counts>=3)&c['capacity_fits'];ys[ok]=cost/counts[ok]
  ax.step(ds,ys,where='post',color=C[colors[i]],lw=2,label=labels[i])
  curves.append({'tp':c['tp'],'capacity_fits':c['capacity_fits'],'deadlines_ms':ds.tolist(),'completion_ms':times.tolist(),'total_gpu_seconds':cost,'cost_per_valid':[None if np.isnan(x) else float(x) for x in ys]})
 ax.axvline(130,color=C['line'],ls='--');ax.set(xlim=(20,240),ylim=(0,1.9),xlabel='八步续写期限 / ms',ylabel='每个按时会话 / GPU·s')
 ax.set_title(title,loc='left',fontsize=13);ax.legend(fontsize=10,frameon=False,loc='upper right');curve_data[phase]=curves
save(f,'figure-6-12-scale-cost');data['continuous_execution']=continuous;data['deadline_curves']=curve_data
exec(compile((HERE/'visual_examples.py').read_text(),str(HERE/'visual_examples.py'),'exec'))
# Topic names identify data independently of the printed figure numbering.
keys={'6-1':'placement','6-2':'tp_pipeline','6-3':'expert_reuse','6-4':'collectives','6-5':'concurrency','6-6':'physical_paths','6-7':'systems','6-8':'remote_memory'}
data={keys.get(k,k):v for k,v in data.items()}
(HERE/'figure-data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
(HERE/'figure-layout-check.json').write_text(json.dumps({'text_extent_warnings':layout},ensure_ascii=False,indent=2)+'\n')
import sys
sys.path.insert(0,str(HERE.parent))
from teaching_revision import draw as draw_teaching
teaching_outputs,teaching_checks=draw_teaching(HERE,data)
outputs=list(dict.fromkeys(outputs+teaching_outputs))
figure_catalog["figures"]=json.loads((HERE/"figure-index.json").read_text())
# Render formulas on the build machine; bundle all image/font bytes into HTML.
md=HERE.parent/'06-超节点.md';raw=md.read_text();maths=[]
def protect(match):
 text=match[0];display=text.startswith('$$');latex=text[2:-2] if display else text[1:-1];token=f'MATHPLACEHOLDER{len(maths)}END';maths.append({'latex':latex.strip(),'display':display,'token':token});return '\n\n'+token+'\n\n' if display else token
protected=re.sub(r'\$\$[\s\S]*?\$\$|\$[^$\n]+\$',protect,raw)
body=markdown.markdown(protected,extensions=['tables','footnotes','fenced_code','toc'],output_format='html')
node="const fs=require('fs'),k=require(process.argv[1]);const a=JSON.parse(fs.readFileSync(0,'utf8'));process.stdout.write(JSON.stringify(a.map(x=>k.renderToString(x.latex,{displayMode:x.display,throwOnError:true,output:'htmlAndMathml'}))))"
res=subprocess.run(['node','-e',node,str(HERE/'vendor/katex/katex.js')],input=json.dumps(maths),text=True,capture_output=True)
if res.returncode:raise SystemExit(res.stderr)
for entry,rendered in zip(maths,json.loads(res.stdout)):
 body=body.replace('<p>'+entry['token']+'</p>',rendered) if entry['display'] else body.replace(entry['token'],rendered)
math_css=(HERE/'vendor/katex/katex.min.css').read_text()
def font_url(m):
 p=HERE/'vendor/katex'/m[1];return 'url(data:font/'+p.suffix[1:]+';base64,'+base64.b64encode(p.read_bytes()).decode()+')'
math_css=re.sub(r'url\((fonts/[^)]+)\)',font_url,math_css)
body=re.sub(r'<table>(.*?)</table>',r'<div class="table-scroll"><table>\1</table></div>',body,flags=re.S)
for p in outputs:
 if p.suffix=='.svg':body=body.replace('src="ch06/'+p.name+'"','src="data:image/png;base64,'+base64.b64encode(p.with_suffix('.png').read_bytes()).decode()+'"')
css='''*{box-sizing:border-box}body{margin:0;background:#f7f7f4;color:#243640;font:18px/1.95 Georgia,"Songti SC",serif}main{max-width:1020px;margin:auto;padding:48px 46px 85px;background:#fff}h1,h2,h3{font-family:Arial,"PingFang SC",sans-serif;line-height:1.45;color:#183949}h1{font-size:36px}h2{font-size:28px;border-top:1px solid #d5e1e4;margin-top:65px;padding-top:26px}h3{font-size:23px;margin-top:38px}a{color:#286b98;text-underline-offset:3px;overflow-wrap:anywhere}img{display:block;width:100%;height:auto;margin:28px auto 10px}em{font-size:15px;color:#546e7a}table{border-collapse:collapse;width:100%;font-size:15px;line-height:1.7;margin:22px 0}td,th{padding:10px 12px;border-bottom:1px solid #d5e1e4;text-align:left}th{background:#edf4f6}blockquote{margin:27px 0;padding:14px 24px;border-left:4px solid #16857b;background:#f0f7f4;font-size:16px}code{font:0.85em/1.6 Menlo,monospace;background:#f0f4f6;overflow-wrap:anywhere}pre{overflow-x:auto;padding:16px;max-width:100%}nav{font:16px/1.9 Arial,"PingFang SC",sans-serif;background:#eff5f7;padding:18px 24px}nav a{display:block}.footnote{font-size:14px;line-height:1.8}.footnote li{margin-bottom:13px}.table-scroll{overflow-x:auto;max-width:100%}.katex{font-size:1.04em}.katex-display{overflow-x:auto;overflow-y:hidden;padding:12px 0}@media(max-width:650px){table{min-width:600px}main{padding:24px 18px}body{font-size:17px}h1{font-size:29px}h2{font-size:25px}h3{font-size:21px}}@media print{main{max-width:none;padding:0}h2,h3{break-after:avoid}img,blockquote{break-inside:avoid}body{font-size:11pt}}'''
css+='main{max-width:760px;padding-left:24px;padding-right:24px}img{max-width:720px}@media print{img{width:420pt;max-width:100%}}'
nav=''.join('<a href="#'+ident+'">'+title+'</a>' for ident,title in re.findall(r'<h2 id="([^"]+)">(6\.\d+ [^<]+)</h2>',body))
page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第 6 章 超节点</title><style>'+css+math_css+'</style></head><body><main><nav aria-label="本章目录">'+nav+'</nav>'+body+'</main></body></html>'
import sys
sys.path.insert(0,str(HERE.parent))
from teaching_reading import readable_diagrams
from preview_output import preview_path
page=readable_diagrams(page)
hp=preview_path(md);hp.write_text(page)
(HERE/'math-validation.json').write_text(json.dumps({'renderer':'KaTeX 0.16.11','expressions':len(maths),'display_expressions':sum(x['display'] for x in maths),'errors':[]},indent=2)+'\n')
from ub_ep_figures import draw as draw_ub_ep
outputs += draw_ub_ep(6, HERE)
from parallel_figures import draw as draw_parallel
outputs += draw_parallel(HERE)
from book_assets import sync_figure_index
active_assets=sync_figure_index(HERE)
figure_catalog["figures"]=json.loads((HERE/"figure-index.json").read_text())
artifacts=outputs+[HERE.parent/'ub_ep_figures.py',HERE.parent/'parallel_figures.py',HERE/'parallel-layout-validation.json',HERE/'ub-ep-layout-validation.json',ROOT/'calculations/results/ep-skew-book.json',HERE/'teaching_revision.py',HERE/'figure-index.json',HERE/'teaching-layout-validation.json',HERE/'visual_examples.py',HERE/'figure-catalog.json',HERE/'figure-data.json',HERE/'continuity-model.json',HERE/'continuity_model.py',hp,md]+active_assets
(HERE/'manifest.json').write_text(json.dumps({'chapter':6,'generator':'manuscripts/ch06/build.py','figures':len(figure_catalog['figures']),'supplementary_figures':1,'font_family':family,'outputs':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in artifacts if not (p.parent==ROOT/'manuscripts' and re.match(r'^[01][0-9]-',p.name))]},ensure_ascii=False,indent=2)+'\n')
print(f'Built {len(outputs)} image files, {len(maths)} formulas, offline HTML; {len(layout)} extent warnings.')
