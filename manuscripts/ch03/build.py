#!/usr/bin/env python3
"""Build chapter 3 figures and offline HTML from locked local evidence."""
from pathlib import Path
import argparse,base64,hashlib,html,json,re,subprocess
from matplotlib.transforms import Bbox
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
import numpy as np
import markdown
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--font');args=parser.parse_args()
import sys
sys.path.insert(0,str(HERE.parent))
from figure_style.typography import configure_font
font,family=configure_font(args.font)
plt.rcParams.update({'font.family':[family,'DejaVu Sans'],'font.size':11,'axes.unicode_minus':False,'svg.fonttype':'path','svg.hashsalt':'ch03-workloads-v1','axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white','text.color':'#193441','axes.labelcolor':'#193441','axes.edgecolor':'#b8cbd2'})
C={'ink':'#193441','blue':'#246f91','teal':'#138b83','orange':'#c9782b','pale':'#f0f6f8','light':'#e7f3ef','sand':'#fcf2e7','muted':'#55707d','line':'#cbd8df','red':'#b65757'}
for z in json.loads((HERE/'sources.json').read_text())['sources']:
 b=(ROOT/z['path']).read_bytes()
 if not b.startswith(b'version https://git-lfs.github.com/spec/v1') and hashlib.sha256(b).hexdigest()!=z['sha256']:raise SystemExit('Source changed; review before relocking: '+z['path'])
def read(p):return json.loads((ROOT/p).read_text())
def calc(n):return read('calculations/results/'+n+'.json')
outputs=[];data={};extent_issues=[]
import sys
sys.path.insert(0,str(HERE.parent))
from math_style import normalize_figure
def save(fig,name):
 normalize_figure(fig)
 # Captions stay outside figures. Remove canvas headers and footer prose, then crop visible content.
 for artist in list(fig.texts):
  if artist.get_position()[1] > .85 or artist.get_position()[1] < .10: artist.remove()
 for ax in fig.axes:
  if not ax.axison and ax.get_position().width > .95:
   for artist in list(ax.texts):
    if artist.get_position()[1] > .85 or artist.get_position()[1] < .10: artist.remove()
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 bounds=[]
 for ax in fig.axes:
  if ax.axison: bounds.append(ax.get_tightbbox(renderer))
  else:
   for artist in [*ax.texts,*ax.patches,*ax.lines,*ax.collections]:
    if artist.get_visible():
     bb=artist.get_window_extent(renderer)
     if np.isfinite(bb.extents).all() and (bb.width or bb.height):bounds.append(bb)
 bounds.extend(t.get_window_extent(renderer) for t in fig.texts if t.get_visible())
 crop=Bbox.union(bounds).transformed(fig.dpi_scale_trans.inverted()).padded(.12)
 # Detect text escaping the canvas; interior placements are also visually reviewed.
 for t in fig.findobj(matplotlib.text.Text):
  if not t.get_visible() or not t.get_text():continue
  bb=t.get_window_extent(renderer)
  if bb.x0 < -2 or bb.y0 < -2 or bb.x1>fig.bbox.width+2 or bb.y1>fig.bbox.height+2:
   extent_issues.append({'figure':name,'text':t.get_text(),'bbox':list(bb.bounds)})
 for ext in ['svg','png']:
  p=HERE/f'{name}.{ext}';fig.savefig(p,dpi=160,bbox_inches=crop,pad_inches=0);outputs.append(p)
 plt.close(fig)
def canvas(title,subtitle,h=8):
 f=plt.figure(figsize=(14,h));a=f.add_axes([0,0,1,1]);a.set(xlim=(0,1),ylim=(0,1));a.axis('off');a.text(.045,.96,title,fontsize=23,weight='bold',va='top');a.text(.045,.895,subtitle,fontsize=11,color=C['muted'],va='top');return f,a
def box(a,x,y,w,h,title,body='',color='pale',size=12):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.005,rounding_size=0.008',edgecolor=C['line'],facecolor=C[color],lw=1));a.text(x+w/2,y+h*(.69 if body else .5),title,ha='center',va='center',weight='bold',fontsize=size)
 if body:a.text(x+w/2,y+h*.28,body,ha='center',va='center',fontsize=10,linespacing=1.55,color=C['muted'])
def arrow(a,start,end,col='teal',rad=0):a.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=13,color=C[col],lw=1.6,connectionstyle=f'arc3,rad={rad}'))
def footer(a,t):a.text(.045,.04,t,fontsize=10.5,color=C['muted'],va='bottom')
# 3-1: calls and logical state, not wall-clock performance.
f,a=canvas('图 3-1  同一模型的 Prefill 与 Decode','Qwen3-8B · S=6144、P=2048、G=4 · BF16 逻辑 KV，每 token 144 KiB。',8)
box(a,.06,.70,.20,.105,'已恢复前缀','6144 个 token',color='light')
box(a,.32,.70,.23,.105,'Prefill：处理新输入','2048 个 token → 首输出 $y_1$',color='sand');arrow(a,(.265,.75),(.31,.75))
a.text(.60,.75,'时间顺序 →',fontsize=12,color=C['muted']);a.text(.60,.70,'宽度不表示真实耗时',fontsize=10)
for i in range(3):
 x=.32+i*.215;box(a,x,.46,.175,.12,f'Decode {i+1}',f'输入 $y_{i+1}$ → 输出 $y_{i+2}$',color='pale')
 if i<2:arrow(a,(x+.18,.52),(x+.205,.52))
arrow(a,(.43,.69),(.405,.59))
a.text(.055,.515,'后续只需\n$G-1=3$ 次调用',fontsize=13,linespacing=1.8)
for i,(slots,label) in enumerate([(8192,'Prefill 后'),(8193,'Decode 1 后'),(8194,'Decode 2 后'),(8195,'Decode 3 后')]):
 x=.07+i*.23;box(a,x,.22,.19,.12,label,f'{slots} 个位置',color='light')
a.text(.055,.125,'基线：8192 位置 = 1.125 GiB；三次 decode 共追加 432 KiB。$y_4$ 返回后尚未再次入模。',fontsize=12)
footer(a,'前缀命中减少重算，但新输入与后续生成仍可读取已有上下文；V4 的状态使用第二章各自的结构账。')
save(f,'figure-3-1-stages');data['3-1']={'kind':'teaching_structure','S':6144,'P':2048,'G':4,'kv_bytes_per_position':147456,'slots_after_calls':[8192,8193,8194,8195]}
# 3-2: workload composition, observed metrics and explicit fluid queue.
arrival=read('experiments/ch03/03-02/results/summary.json')['reports']
f=plt.figure(figsize=(15,11));f.suptitle('图 3-2  对话长度与思考预算',x=.045,ha='left',y=.97,fontsize=24,weight='bold');f.text(.045,.916,'教学组成、实际重放与质量诊断分别呈现；两个时段各 60 秒，每组共 480 条请求。',fontsize=11,color=C['muted'])
axs=[f.add_axes(z) for z in [[.08,.59,.37,.25],[.57,.59,.36,.25],[.08,.17,.37,.27],[.57,.17,.36,.27]]]
a=axs[0];labels=['均匀\n前／后窗','变化\n前窗','变化\n后窗'];A=np.array([.5,.9,.1]);a.bar(range(3),A,color=C['blue'],label='A：8192 输入／256 输出');a.bar(range(3),1-A,bottom=A,color=C['orange'],label='B：1024 输入／2048 输出');a.set_xticks(range(3),labels);a.set_ylim(0,1.1);a.set_ylabel('请求比例');a.set_title('A  同总量，不同短窗组成〔教学〕',loc='left',fontsize=13);a.legend(fontsize=9,frameon=False,loc='upper left',bbox_to_anchor=(0,-.20))
a=axs[1];pref=np.array([18432,29900.8,6963.2]);dec=np.array([4604,1736.8,7471.2]);x=np.arange(3);a.bar(x-.18,pref,width=.35,label='输入 token/s',color=C['blue']);a.bar(x+.18,dec,width=.35,label='decode 步/s',color=C['teal']);a.set_xticks(x,labels);a.set_ylabel('阶段位置数 / s');a.set_title('B  输入与后续生成需求〔教学〕',loc='left',fontsize=13);a.legend(fontsize=9,frameon=False)
a=axs[2];x=np.arange(2)
for i,(field,lab,col) in enumerate([('ttft_p95_s','TTFT p95','blue'),('latency_p95_s','完整请求 p95','teal')]):a.bar(x+(i-.5)*.32,[z[field] for z in arrival],width=.30,label=lab,color=C[col])
a.set_xticks(x,['均匀','时段变化']);a.set_ylim(0,355);a.set_ylabel('秒');a.set_title('C  同一实例的实际回放',loc='left',fontsize=13);a.legend(frameon=False,fontsize=10)
for i,z in enumerate(arrival):a.text(i,320,f"抢占 {z['preemptions']} 次",ha='center',fontsize=10)
decode_card=[z for z in calc('batch-reuse-rtxpro6000-mix')['batch_reuse_rows'] if z['batch']==64][0];decode_cards=2;service=decode_cards*decode_card['throughput_upper_tokens_per_second'];peak_backlog=(dec[2]-service)*60
a=axs[3];seconds=np.array([0,60,120,120+peak_backlog/service]);backlog=np.array([0,0,peak_backlog,0]);a.plot(seconds,backlog,color=C['orange'],lw=2.5,marker='o');a.axvline(120,color=C['muted'],ls='--',lw=1);a.set_title('D  固定服务率下的积压〔教学〕',loc='left',fontsize=13);a.set_xlabel('从请求到达起点计时 / s');a.set_ylabel('待处理 decode 步');a.set_ylim(0,140000);a.annotate(f'{peak_backlog:,.0f} 步',xy=(120,peak_backlog),xytext=(40,127000),arrowprops={'arrowstyle':'->'},fontsize=11);a.grid(alpha=.15)
save(f,'figure-3-2-workload-budget');data['3-2']={'kind':'separate_teaching_and_measurement','A_fraction':A.tolist(),'prefill_positions_per_second':pref.tolist(),'decode_positions_per_second':dec.tolist(),'arrival_reports':arrival,'fluid_queue':{'device':'rtx-pro6000-blackwell-ws','cards':decode_cards,'batch_per_card':64,'mean_context_tokens':2742,'service_steps_per_second':service,'seconds':seconds.tolist(),'backlog_steps':backlog.tolist()}}
# 3-3: measured round durations, engine hits, and explicitly hypothetical retention.
ag=calc('agent-thinking-on');rounds=ag['agent_rounds'];f,a=canvas('图 3-3  Agent 轨迹与 KV 生命周期','开启 thinking 的四轮人工代码任务；模型／工具为墙钟观测，缓存大小为条件式逻辑预算。',10)
ax=f.add_axes([.12,.55,.79,.27]);y=np.arange(4);mods=[z['measured_model_seconds'] for z in rounds];tools=[z['measured_tool_seconds'] for z in rounds]
ax.barh(y,mods,color=C['blue'],label='模型墙钟');ax.barh(y,tools,left=mods,color=C['orange'],label='工具墙钟（本例很短）');ax.set_yticks(y,['第 1 轮：截断','第 2 轮：写文件','第 3 轮：测试','第 4 轮：结束']);ax.invert_yaxis();ax.set_xlim(0,44);ax.set_xlabel('秒（逐轮，非连续起点）');ax.legend(loc='lower right',fontsize=10,frameon=False)
for i,z in enumerate(rounds):ax.text(mods[i]+.5,i,f"{mods[i]:.3f} s",va='center',fontsize=10)
a.text(.06,.46,'输入 / 命中 / 输出',fontsize=12,weight='bold')
for i,z in enumerate(rounds):
 x=.065+i*.232;box(a,x,.315,.205,.105,f"{z['prompt_tokens']} / {z['cached_tokens']} / {z['output_tokens']}",f"工具段假设保留 {z['retained_logical_kv_bytes']/2**20:.2f} MiB",size=11)
box(a,.06,.12,.24,.105,'分支共享〔教学示意〕','一份前缀 + 各自尾部',color='light');box(a,.43,.19,.19,.075,'公共前缀',color='light');box(a,.76,.245,.18,.055,'分支 A 尾部',color='sand',size=11);box(a,.76,.14,.18,.055,'分支 B 尾部',color='sand',size=11);arrow(a,(.63,.23),(.75,.27));arrow(a,(.63,.22),(.75,.17))
footer(a,'实测轨迹是串行的；未记录真实 KV 块的保留／回收。只加速首轮两倍：76.510 → 58.323 s，假定其余行为与质量不变。')
save(f,'figure-3-3-agent');data['3-3']={'kind':'measured_wall_and_conditional_state','rounds':rounds,'summary':ag['summary'],'counterfactual':calc('agent-thinking-on-double-first')['summary']}
# 3-4: stage identity, teaching timing, observed reception.
aud=calc('audio-timing-base');large=calc('audio-timing-large-buffer');intr=calc('audio-timing-interrupt');real=read('experiments/ch03/03-05/historical-arrivals/summary.json')
f,a=canvas('图 3-4  实时交互的端到端时序','相同任务中，数据到达、开始播放、设备静音和后端停止计算是不同事件。',11)
for i,(title,body) in enumerate([('视觉编码 E','640² → 400 位置\nEC 7.8125 MiB'),('语言 P → D','视觉位置 KV\n56.25 MiB'),('语音生成与接收','声学码 → PCM\n到达后进入缓冲'),('设备播放','首播 / 连续供给\n打断 / 静音')]):
 x=.055+i*.235;box(a,x,.71,.195,.12,title,body,size=12)
 if i<3:arrow(a,(x+.20,.77),(x+.225,.77))
a.text(.055,.665,'上方是可组合的阶段示意；下方是独立的八块音频教学时序，不是该视觉模型的实测。',fontsize=10.5,color=C['muted'])
ax=f.add_axes([.12,.32,.80,.27]);ch=aud['audio_chunks']
for z in ch:
 i=z['chunk'];start=z['playback_start_ns']/1e6;ax.barh(i,20,left=start,height=.48,color=C['teal']);ax.plot(z['arrival_ns']/1e6,i,'o',color=C['blue']);ax.plot(z['deadline_ns']/1e6,i,'|',markersize=15,color=C['orange'])
ax.plot([],[],'o',color=C['blue'],label='块到达');ax.plot([],[],'|',markersize=13,color=C['orange'],label='原定播放时刻');ax.barh([],[],color=C['teal'],label='实际播放区间');ax.set_yticks(range(8),[f'块 {i+1}' for i in range(8)]);ax.invert_yaxis();ax.set_xlim(0,260);ax.set_xticks([0,50,100,150,200,250]);ax.set_xlabel('从采集起点计时 / ms');ax.legend(frameon=False,ncol=3,fontsize=10,loc='upper left',bbox_to_anchor=(0,1.20));ax.annotate('第三块晚到 5 ms',xy=(123,2),xytext=(167,1.3),arrowprops={'arrowstyle':'->','color':C['orange']},fontsize=10)
box(a,.06,.115,.27,.095,'缓冲对照〔教学〕','40 ms：首播 78，停顿 5 ms\n60 ms：首播 98，无停顿',size=12)
box(a,.365,.115,.27,.095,'打断投影〔教学〕','123 ms 发出 → 130 ms 静音\n未模拟后端取消',size=12)
box(a,.67,.115,.27,.095,'上下文接收〔真实记录〕','首块 399.919 / 370.459 ms\n首播、静音、取消未知',size=12)
footer(a,'每块 20 ms、24 kHz、单声道、2 bytes/sample：960 bytes；模型与网络时长是教学输入，设备播放并非上下文实测。')
save(f,'figure-3-4-realtime');data['3-4']={'kind':'separate_mechanism_teaching_and_historical_reception','audio_chunks':ch,'base_summary':aud['summary'],'large_buffer':large['summary'],'interrupt':intr['summary'],'historical_reception':real,'visual':{'positions':400,'ec_bytes':8192000,'kv_bytes':58982400}}
# 3-5: computational paths, matrices, parameter states.
t=calc('training-qwen3-8b-t8192');f,a=canvas('图 3-5  推理与各训练阶段的计算和状态','Qwen3-8B · B=1、T=8192；训练计算所有词表头行，无重计算；各面板使用独立单位。',9)
for i,(title,body) in enumerate([('前向','保存反向所需激活'),('反向','输入梯度 + 权重梯度'),('参数更新','权重 / 优化器状态')]):
 x=.07+i*.31;box(a,x,.69,.245,.12,title,body,color=['pale','light','sand'][i]);
 if i<2:arrow(a,(x+.25,.75),(x+.30,.75))
a.text(.07,.625,'推理使用前向；预训练、中期训练与 SFT 使用同一基本更新路径，各阶段采用不同的数据、标签和序列长度。',fontsize=11)
ax=f.add_axes([.10,.23,.37,.29]);vals=[t['summary'][k]/1e12 for k in ['forward_matrix_flops','backward_matrix_flops','training_matrix_flops','six_nd_flops']];ax.barh(range(4),vals,color=[C['blue'],C['teal'],C['ink'],C['orange']]);ax.set_yticks(range(4),['前向矩阵','反向矩阵','两者合计','总参数 $6ND$']);ax.invert_yaxis();ax.set_xlim(0,515);ax.set_xlabel('TFLOPs');ax.set_title('A  矩阵分项计算',loc='left',fontsize=13)
for i,v in enumerate(vals):ax.text(v+7,i,f'{v:.3f}',va='center',fontsize=10)
ax=f.add_axes([.64,.23,.29,.29]);states=t['parameter_state_bytes'];labels=['BF16 权重','FP32 梯度','FP32 master','Adam 一阶','Adam 二阶'];sv=[v/1e9 for v in states.values()];ax.barh(range(5),sv,color=[C['blue'],C['teal'],C['orange'],C['muted'],C['muted']]);ax.set_yticks(range(5),labels);ax.invert_yaxis();ax.set_xlim(0,42);ax.set_xlabel('十进制 GB');ax.set_title('B  未分片的参数相关状态',loc='left',fontsize=13)
for i,v in enumerate(sv):ax.text(v+.8,i,f'{v:.3f}',va='center',fontsize=10)
a.text(.06,.12,'状态合计 147.433 GB；激活和工作区另计。矩阵表未计非矩阵反向、优化器算术、重计算与通信。',fontsize=11)
footer(a,'“三倍前向”来自本例每个矩阵的两个梯度；不能用全部参数量或可训练参数占比推断任意训练路径。')
save(f,'figure-3-5-training');data['3-5']={'kind':'analytical_subaccounts','scenario':t['scenario'],'summary':t['summary'],'parameter_state_bytes':states,'training_matrix_rows':t['training_matrix_rows']}
# 3-6: RL/OPD dataflow and same accepted target.
rl=calc('rl-qwen8-base');low=calc('rl-qwen8-low-acceptance');f,a=canvas('图 3-6  RL／OPD 的工作与数据流','先统一同一批有效样本，再比较各阶段工作；数值使用 Qwen3-8B 教学配置，不是 V4 运行。',10)
for x,title,body in [(.06,'策略生成','生成的回答'),(.30,'反馈 / 验证','规则、模型或环境'),(.54,'筛选与组织','选择训练样本'),(.78,'策略更新','前向 / 反向 / 更新')]:box(a,x,.70,.17,.12,title,body,size=12)
for x in [.23,.47,.71]:arrow(a,(x+.004,.76),(x+.06,.76))
box(a,.32,.52,.20,.085,'可选教师前向','OPD：学生轨迹 → 教师分布',color='sand',size=11);arrow(a,(.15,.695),(.32,.56));arrow(a,(.52,.56),(.78,.715))
arrow(a,(.865,.695),(.865,.445),'orange');arrow(a,(.865,.445),(.145,.445),'orange');arrow(a,(.145,.445),(.145,.695),'orange');a.text(.54,.463,'权重版本就绪后用于下一批生成',ha='center',fontsize=11,color=C['orange'])
ax=f.add_axes([.12,.17,.77,.20]);fields=['rollout_prefill','rollout_decode','reference_scoring','policy_update'];colors=['blue','teal','orange','ink'];labels=['生成输入','后续生成','Reference','更新'];left=np.zeros(2)
for name,lab,col in zip(fields,labels,colors):
 vals=np.array([next(z['matrix_flops'] for z in d['rl_stages'] if z['name']==name)/1e12 for d in [rl,low]]);ax.barh([0,1],vals,left=left,label=lab,color=C[col],height=.45);left+=vals
ax.set_yticks([0,1],['生成 32 条 → 保留 16 条','生成 64 条 → 保留 16 条']);ax.invert_yaxis();ax.set_xlim(0,4050);ax.set_xlabel('已计矩阵 TFLOPs');ax.legend(ncol=4,frameon=False,fontsize=10,loc='upper center',bbox_to_anchor=(.5,1.37))
for i,v in enumerate(left):ax.text(v+45,i,f'{v:.3f}',va='center',fontsize=10)
footer(a,'本例不调用教师模型。一次 BF16 权重快照为 16.381 GB；优化器状态另行保存。')
save(f,'figure-3-6-rl');data['3-6']={'kind':'teaching_analytical_cycle','base_stages':rl['rl_stages'],'low_acceptance_stages':low['rl_stages'],'base':rl['summary'],'low_acceptance':low['summary'],'scenario':rl['scenario']}
# 3-7: real public points, independent holdout, conditional lifetime proxy.
fit=calc('datablations-real-c4-eight-point-fit');result=fit['primary']['result'];life=calc('real-c4-lifecycle-512-128')['variants'][0];law=result['law']
f=plt.figure(figsize=(15,10));f.suptitle('图 3-7  训练预算与生命周期成本',x=.045,y=.97,ha='left',fontsize=24,weight='bold');f.text(.045,.91,'六个 C4 点拟合、两个事前留出；成本按 40% MFU 折算为 H100 SXM GPU 时间。',fontsize=11,color=C['muted'])
a=f.add_axes([.09,.47,.34,.35]);a.plot([2,7.6],[2,7.6],color=C['line'],lw=1)
resids=[]
for split,col,marker,lab in [('fit','blue','o','拟合：6 点'),('holdout','orange','^','留出：2 点')]:
 rows=[z for z in fit['records'] if z['split']==split];obs=[z['loss'] for z in rows];pred=[law['E']+law['A']*(z['N']/law['N0'])**(-law['alpha'])+law['B']*(z['D']/law['D0'])**(-law['beta']) for z in rows];a.scatter(obs,pred,color=C[col],marker=marker,s=75,label=lab);resids += [{'id':z['id'],'split':split,'residual':p-z['loss']} for z,p in zip(rows,pred)]
a.set(xlim=(2,7.6),ylim=(2,7.6),xlabel='观测损失 / nats·token⁻¹',ylabel='预测损失 / nats·token⁻¹');a.set_title('A  公开点与预测',loc='left',fontsize=13);a.legend(frameon=False,fontsize=10)
a=f.add_axes([.09,.18,.34,.15]);a.bar(range(8),[z['residual'] for z in resids],color=[C['blue'] if z['split']=='fit' else C['orange'] for z in resids]);a.axhline(0,lw=.8,color=C['line']);a.set_xticks(range(8),['F1','F2','F3','F4','F5','F6','H1','H2']);a.set_ylabel('预测 − 观测');a.set_ylim(-.026,.033);a.set_title('留出 RMSE = 0.019345 nats/token',loc='left',fontsize=11)
a=f.add_axes([.58,.25,.36,.57]);calls=np.linspace(0,4e8,250)
for z,col in zip(life['lifecycle']['rows'],['orange','teal','blue','muted']):a.plot(calls/1e8,(z['upfront_cost']+calls*z['cost_per_call'])/3600,color=C[col],linestyle='--' if z['outside_fit_box'] else '-',lw=2,label=f"{z['N']/1e9:g}B"+('（外推）' if z['outside_fit_box'] else ''))
cross=(life['lifecycle']['rows'][0]['upfront_cost']-life['lifecycle']['rows'][1]['upfront_cost'])/(life['lifecycle']['rows'][1]['cost_per_call']-life['lifecycle']['rows'][0]['cost_per_call']);a.axvline(cross/1e8,color=C['line'],lw=1);a.text(cross/1e8+.06,1000,'0.1B / 0.5B\n约 2.048 亿次交叉',fontsize=10);a.set(xlabel='累计调用 / 亿次',ylabel='H100 SXM GPU 小时',xlim=(0,4),ylim=(0,1200));a.set_title('B  目标损失 2.9 的成本估算',loc='left',fontsize=13);a.legend(frameon=False,fontsize=10)
f.text(.53,.155,'$P=512$、$G=128$；每调用 $2N(P+G-1)$。\n0.1B 所需训练 D≈298.6B，超出拟合上界约 3.28 倍。',fontsize=11,linespacing=1.7)
f.text(.045,.055,'拟合系数依赖这组数据与预定网格；同损失不等于同任务质量。虚线表示模型参数量或训练数据量超出拟合范围。',fontsize=10.5,color=C['muted'])
save(f,'figure-3-7-scaling');data['3-7']={'kind':'public_observations_and_conditional_proxy','records':fit['records'],'law':law,'residuals':resids,'fit_sse':result['fit_sse'],'holdout_rmse':result['holdout_rmse'],'lifecycle':life['lifecycle'],'crossing_calls':cross}
# 3-8: historical scale/data and separate MoE parameter categories.
hist=calc('training-history-published');hr={z['input']['id']:z for z in hist['training_history_rows']};ids=['llama1-7b','llama2-7b','llama31-8b','qwen25-7b-proxy','qwen3-8b-proxy'];names=['Llama 1 6.7B\n2023 · 1T','Llama 2 ~7B\n2023 · 2T','Llama 3.1 ~8B\n2024 · ~15T','Qwen2.5 ~7B\n2024 · ~18T','Qwen3 ~8B\n2025 · ~36T']
f,a=canvas('图 3-8  Llama 与 Qwen 的模型—数据选择','上下文投入按报告中的模型规模估算；Qwen token 为 family 披露，产品点不是受控 Scaling Law 实验。',9)
ax=f.add_axes([.16,.26,.30,.51]);ratios=[hr[k]['input']['training_tokens']/hr[k]['input']['parameter_proxy'] for k in ids];ax.barh(range(5),ratios,color=[C['blue']]*3+[C['teal']]*2);ax.set_yticks(range(5),names);ax.invert_yaxis();ax.set_xlim(0,5400);ax.set_xlabel('训练 token / 参数');ax.set_title('A  相近规模 dense 模型',loc='left',fontsize=13)
for i,v in enumerate(ratios):ax.text(v+90,i,f'{v:,.0f}',va='center',fontsize=10)
ax=f.add_axes([.63,.26,.31,.51]);mids=['deepseek-v3-pretraining','deepseek-v4-flash','deepseek-v4-pro'];ctx=[hr[k]['parameter_context'] for k in mids];y=np.arange(3);tot=[z['total_reported']/1e9 for z in ctx];act=[z['active_reported']/1e9 for z in ctx];ax.barh(y-.17,tot,height=.30,color=C['blue'],label='总参数：估算容量');ax.barh(y+.17,act,height=.30,color=C['orange'],label='激活参数：估算计算量');ax.set_yticks(y,['V3 · 2024\n14.8T','V4-Flash · 2026\n32T','V4-Pro · 2026\n33T']);ax.invert_yaxis();ax.set_xlim(0,1900);ax.set_xlabel('十亿参数 B');ax.set_title('B  MoE 总参数与激活参数',loc='left',fontsize=13);ax.legend(frameon=False,fontsize=10,loc='upper center',bbox_to_anchor=(.5,-.17))
for i,(v,w) in enumerate(zip(tot,act)):ax.text(v+25,i-.17,f'{v:g}',va='center',fontsize=10);ax.text(w+25,i+.17,f'{w:g}',va='center',fontsize=10)
a.text(.06,.135,'D/N 从约 149 到约 4500，说明相近参数规模可投入更多训练；能力差异还涉及数据、训练方法和评测。',fontsize=11)
footer(a,'6ND 按给定参数量估算；MoE 激活量不能替代全部矩阵、状态更新与优化器工作，跨模型不据本图排名质量。')
save(f,'figure-3-8-history');data['3-8']={'kind':'reported_history_and_analytical_ratios','dense_rows':[hr[k] for k in ids],'ratios':ratios,'moe_rows':[hr[k] for k in mids]}
# 3-9: hardware-separated GPU-hour panels, missing values distinct from zero.
f,a=canvas('图 3-9  公开 GPU 小时与训练阶段','不同设备分面，横轴范围不同；GPU 小时不能直接跨硬件解释为算力或效率。',9)
for rect,keys,title,lim in [([.13,.40,.21,.37],['llama1-7b','llama1-65b','llama2-7b','llama2-70b'],'A  A100 80GB',2.1),([.48,.40,.20,.37],['llama31-8b','llama31-70b','llama31-405b'],'B  H100 80GB',36)]:
 ax=f.add_axes(rect);vals=[hr[k]['input']['gpu_hours']/1e6 for k in keys];ax.barh(range(len(keys)),vals,color=C['blue']);ax.set_yticks(range(len(keys)),[k.replace('llama31','Llama 3.1').replace('llama1','Llama 1').replace('llama2','Llama 2').replace('-',' ') for k in keys]);ax.invert_yaxis();ax.set_xlim(0,lim);ax.set_title(title,loc='left',fontsize=13);ax.set_xlabel('百万 GPU 小时')
 for i,v in enumerate(vals):ax.text(v+lim*.025,i,f'{v:.3f}',va='center',fontsize=10)
ax=f.add_axes([.82,.40,.13,.37]);st=hist['stage_reports'][0]['input']['parts'];vals=[st[k]/1e6 for k in ['pretraining','context_extension','posttraining']];ax.barh(range(3),vals,color=[C['blue'],C['teal'],C['orange']]);ax.set_yticks(range(3),['V3 预训练','上下文扩展','后训练']);ax.invert_yaxis();ax.set_xlim(0,3.5);ax.set_title('C  H800',loc='left',fontsize=13);ax.set_xlabel('百万 GPU 小时')
for i,v in enumerate(vals):ax.text(v+.07,i,f'{v:.3f}',va='center',fontsize=10)
box(a,.06,.155,.41,.115,'未披露完整预训练 GPU 小时','Qwen2.5 / Qwen3 / Qwen3.5 / V4\n未知值不画成零',color='pale',size=12)
box(a,.53,.155,.41,.115,'Qwen3 表 21：另一个统计范围','特定 8B 后训练：RL 17,920；OPD 1,800 小时\n两条替代分支，不相加，不填入预训练列',color='sand',size=12)
footer(a,'Llama 1 65B 在持续 2048 卡假设下约 20.80 天；405B 的 78.43 天需先对齐小时与最大卡数范围，且只是条件下界。')
save(f,'figure-3-9-gpu-hours');data['3-9']={'kind':'reported_gpu_hours_by_hardware_and_scope','history_rows':hist['training_history_rows'],'stage_reports':hist['stage_reports'],'missing_is_zero':False}
# Additional mechanism diagrams share this chapter's font and output handling.
import sys
sys.path.insert(0,str(HERE.parent))
from teaching_figures import draw as draw_teaching_figures
data['teaching_diagrams']=draw_teaching_figures(3,save,ROOT)

(HERE/'figure-data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
(HERE/'figure-layout-check.json').write_text(json.dumps({'outside_canvas_text':extent_issues},ensure_ascii=False,indent=2)+'\n')
from teaching_revision import draw as draw_revision
outputs.extend(draw_revision(HERE,data))
outputs=list(dict.fromkeys(outputs))
md=HERE.parent/'03-推理与训练负载.md';raw=md.read_text()
maths=[]
def protect_math(match):
 text=match.group(0);display=text.startswith('$$');latex=text[2:-2] if display else text[1:-1]
 token=f'MATHPLACEHOLDER{len(maths)}END';maths.append({'latex':latex.strip(),'display':display,'token':token})
 return ('\n\n'+token+'\n\n') if display else token
protected=re.sub(r'\$\$[\s\S]*?\$\$|\$[^$\n]+\$',protect_math,raw)
body=markdown.markdown(protected,extensions=['tables','footnotes','fenced_code','toc'],output_format='html')
node_code="const fs=require('fs'),k=require(process.argv[1]);let a=JSON.parse(fs.readFileSync(0,'utf8'));process.stdout.write(JSON.stringify(a.map(x=>k.renderToString(x.latex,{displayMode:x.display,throwOnError:true,output:'htmlAndMathml'}))))"
result=subprocess.run(['node','-e',node_code,str(HERE/'vendor/katex/katex.js')],input=json.dumps(maths),text=True,capture_output=True)
if result.returncode: raise SystemExit(result.stderr)
rendered=json.loads(result.stdout)
for entry,result in zip(maths,rendered):
 body=body.replace('<p>'+entry['token']+'</p>',result) if entry['display'] else body.replace(entry['token'],result)
math_css=(HERE/'vendor/katex/katex.min.css').read_text()
def font_data(match):
 path=HERE/'vendor/katex'/match[1];suffix=path.suffix[1:]
 return 'url(data:font/'+suffix+';base64,'+base64.b64encode(path.read_bytes()).decode()+')'
math_css=re.sub(r'url\((fonts/[^)]+)\)',font_data,math_css)
body=re.sub(r'<table>(.*?)</table>',r'<div class="table-scroll"><table>\1</table></div>',body,flags=re.S)
(HERE/'math-validation.json').write_text(json.dumps({'renderer':'KaTeX 0.16.11','expressions':len(maths),'display_expressions':sum(x['display'] for x in maths),'errors':[]},indent=2)+'\n')

for p in outputs:
 if p.suffix=='.svg':body=body.replace('src="ch03/'+p.name+'"','src="data:image/png;base64,'+base64.b64encode(p.with_suffix('.png').read_bytes()).decode()+'"')
css='''body{margin:0;background:#fafaf8;color:#243640;font:18px/1.95 Georgia,"Songti SC",serif}main{max-width:960px;margin:auto;padding:50px 38px 90px;background:white}h1,h2,h3{font-family:Arial,"PingFang SC",sans-serif;line-height:1.45;color:#163747}h1{font-size:36px}h2{font-size:28px;border-top:1px solid #d5e1e4;margin-top:65px;padding-top:28px}h3{font-size:23px;margin-top:40px}p{margin:1em 0}a{color:#246f91;text-underline-offset:3px}img{display:block;width:100%;height:auto;margin:26px auto 10px}em{font-size:15px;color:#55707d}table{border-collapse:collapse;width:100%;font-size:15px;line-height:1.7;margin:24px 0}td,th{padding:10px 12px;border-bottom:1px solid #d5e1e4;text-align:left}th{background:#edf4f6}blockquote{margin:28px 0;padding:16px 24px;border-left:4px solid #138b83;background:#f1f8f6;font-size:16px}code{font:0.85em/1.65 Menlo,monospace;background:#f0f4f6;overflow-wrap:anywhere}pre{white-space:pre-wrap}.equation{font:20px/1.8 Georgia,"Songti SC",serif;text-align:center;background:#f7f9fa;padding:18px 12px;margin:25px 0;overflow-wrap:anywhere}nav{font:16px/1.9 Arial,"PingFang SC",sans-serif;background:#f0f6f8;padding:18px 24px}nav a{display:block}.footnote{font-size:14px;line-height:1.8}.footnote li{margin-bottom:12px}@media(max-width:650px){main{padding:25px 18px}body{font-size:17px}table{display:block;overflow-x:auto}h1{font-size:29px}h2{font-size:25px}.equation{font-size:17px}}@media print{main{max-width:none;padding:0}h2,h3{break-after:avoid}img,blockquote{break-inside:avoid}body{font-size:11pt}}'''
css+=' main{max-width:760px}img{max-width:720px}@media print{img{width:420pt;max-width:100%}}'
css+=math_css+' .katex{font-size:1.04em}.katex-display{overflow-x:auto;overflow-y:hidden;padding:12px 0}.table-scroll{overflow-x:auto;max-width:100%}@media(max-width:650px){table{display:table}}'
nav=''.join('<a href="#'+ident+'">'+title+'</a>' for ident,title in re.findall(r'<h2 id="([^"]+)">(3\.\d+ [^<]+)</h2>',body))
page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第 3 章 推理与训练负载</title><style>'+css+'</style><main><nav>'+nav+'</nav>'+body+'</main></html>'
import sys
sys.path.insert(0,str(HERE.parent))
from teaching_reading import readable_diagrams
from preview_output import preview_path
page=readable_diagrams(page)
html_path=preview_path(HERE.parent/'03-推理与训练负载.html');html_path.write_text(page)
from book_assets import sync_figure_index
active_assets=sync_figure_index(HERE)
artifacts=outputs+[HERE/'figure-data.json',html_path,md]+active_assets
(HERE/'manifest.json').write_text(json.dumps({'chapter':3,'generator':'manuscripts/ch03/build.py','figures':len(re.findall(r'!\[',raw)),'font_family':family,'outputs':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in artifacts if not (p.parent==ROOT/'manuscripts' and re.match(r'^[01][0-9]-',p.name))]},ensure_ascii=False,indent=2)+'\n')
print(f'Built {len(outputs)} figure files and reading HTML; {len(extent_issues)} text extent warnings.')
