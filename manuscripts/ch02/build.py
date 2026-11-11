#!/usr/bin/env python3
"""Build chapter 2 vector/raster figures and offline reading HTML from locked evidence."""
from pathlib import Path
import base64, hashlib, html, json, re, argparse, subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
import numpy as np
import markdown
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--font');args=parser.parse_args()
import sys
sys.path.insert(0,str(HERE.parent))
from figure_style.typography import configure_font
font,family=configure_font(args.font)
plt.rcParams.update({'font.family':[family,'DejaVu Sans'],'font.size':12,'axes.unicode_minus':False,'svg.fonttype':'path','svg.hashsalt':'ch02-models-v1','axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white','text.color':'#193441','axes.labelcolor':'#193441','axes.edgecolor':'#bac6cd'})
C={'ink':'#193441','blue':'#246f91','teal':'#138b83','orange':'#c9782b','pale':'#f0f6f8','light':'#e7f3ef','sand':'#fcf2e7','muted':'#55707d','line':'#cbd8df'}
lock=json.loads((HERE/'sources.json').read_text())
for row in lock['sources']:
 raw_b=(ROOT/row['path']).read_bytes()
 h_raw=hashlib.sha256(raw_b).hexdigest()
 h_lfs=raw_b.decode('utf-8','ignore').split('oid sha256:')[1].split()[0] if raw_b.startswith(b'version https://git-lfs.github.com/spec/v1') else h_raw
 if row['sha256'] not in {h_raw,h_lfs}:raise SystemExit('Source changed: '+row['path'])
def load(n):
 p=ROOT/'calculations/results'/f'{n}.json'; txt=p.read_text()
 if txt.startswith('version https://git-lfs.github.com/spec/v1'):
  fd=json.loads((HERE/'figure-data.json').read_text())['figure_2_10']
  return {'contract':fd['contract'],'comparisons':[{'model':m,'summary':{'matrix_flops':v*1e12}} for m,v in zip(fd['model_order'],fd['matrix_tflops'])]}
 return json.loads(txt)
q=load('qwen3-8b-prefill-8192');v=load('state-deepseek-v4-flash-n8192-b1-native');k=load('state-kimi-k3-n8192-b1-compact')
r=load('request-four-models-book');cap=load('capacity-qwen3-8b-n8192');cap70=load('llama70-capacity-8k')
MiB=2**20;GiB=2**30;data={};out=[]
import sys
sys.path.insert(0,str(HERE.parent))
from math_style import normalize_figure
def save(fig,name):
 normalize_figure(fig)
 for artist in list(fig.texts): artist.remove()
 for ax in fig.axes:
  if not ax.axison and ax.get_position().width > .95:
   for artist in list(ax.texts):
    if artist.get_position()[1] >= .88 or artist.get_position()[1] <= .11: artist.remove()
   # Crop the drawing coordinate without changing shapes or the data axes.
   ax.set_ylim(.105,.855)
 if all(ax.axison or ax.get_position().width < .95 for ax in fig.axes):
  fig.subplots_adjust(top=.91,bottom=.20)
 for ext in ['svg','png']:
  p=HERE/f'{name}.{ext}';fig.savefig(p,dpi=150);out.append(p)
 plt.close(fig)
def canvas(title,subtitle,height=8):
 fig=plt.figure(figsize=(14,height));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
 # Figure title and explanatory scope belong to the external manuscript caption.
 return fig,ax
def box(ax,x,y,w,h,title,body='',color='pale',size=12):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.006,rounding_size=0.009',edgecolor=C['line'],facecolor=C[color],lw=1))
 ax.text(x+w/2,y+h*(.68 if body else .5),title,ha='center',va='center',fontsize=size,weight='bold')
 if body:ax.text(x+w/2,y+h*.27,body,ha='center',va='center',fontsize=10,color=C['muted'],linespacing=1.5)
def arrow(ax,a,b,col='teal',alpha=1,rad=0):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color=C[col],alpha=alpha,lw=1.5,connectionstyle=f'arc3,rad={rad}'))
# Figure 1: dependency graphs, separate known sequence and next unknown token.
f,a=canvas('图 2-1  序列模型的计算依赖','同一坐标：横轴为序列位置，纵轴为网络深度；颜色区分已完成状态与新增工作。',8.2)
for panel,x0,title in [(0,.065,'RNN：同层时间递推'),(1,.39,'因果 Transformer：跨层访问'),(2,.715,'缓存生成：仅追加新 token')]:
 a.text(x0+.11,.81,title,ha='center',fontsize=13,weight='bold')
 xs=[x0+i*.065 for i in range(4)];ys=[.30+i*.125 for i in range(4)]
 for l in range(1,4):
  for t in range(4):
   if panel==0:
    arrow(a,(xs[t],ys[l-1]+.02),(xs[t],ys[l]-.022),alpha=.55)
    if t:arrow(a,(xs[t-1]+.017,ys[l]),(xs[t]-.017,ys[l]),'orange')
   elif panel==1:
    for j in range(t+1):arrow(a,(xs[j],ys[l-1]+.02),(xs[t],ys[l]-.022),alpha=.27)
   elif t==3:
    for j in range(4):arrow(a,(xs[j],ys[l-1]+.02),(xs[t],ys[l]-.022),'orange',alpha=.55)
 for l,y in enumerate(ys):
  for t,x in enumerate(xs):
   color=C['orange'] if panel==2 and t==3 else C['teal'] if panel==2 else C['blue']
   a.scatter([x],[y],s=165,c=color,zorder=5)
  a.text(x0-.025,y,'输入' if l==0 else f'L{l}',ha='right',va='center',fontsize=9)
 for t,x in enumerate(xs):a.text(x,.255,f't{t+1}',ha='center',fontsize=10)
 a.text(x0+.10,.17,['固定宽度状态\n位置 t 等待位置 t−1','允许访问上一层的因果前缀\n同层已知位置可并行','绿色：可复用的上下文\n橙色：当前 token 前向'][panel],ha='center',fontsize=11,linespacing=1.7)
a.text(.05,.065,'参数共享方式和数据边不同，“旋转 90 度”仅是局部类比；未知输出之间仍有跨调用依赖。',fontsize=11,color=C['muted'])
save(f,'figure-2-1-dependencies');data['figure_2_1']={'positions':4,'layers':3,'evidence':'sequence-dependencies-book.json','note':'Dependency schematic; no measured time.'}
# Figure 2: two independent sublayers with residual, labeled dimensions.
f,a=canvas('图 2-2  Qwen3-8B 单层的尺寸与数据流','36 层 · d=4096 · 32 个 Q 头／8 个 KV 头 · 头宽 128；m=BP（prefill）或 B（decode）。',10)
box(a,.055,.75,.18,.075,'输入 X','$m\\times 4096$')
box(a,.30,.75,.18,.075,'RMSNorm',size=12);arrow(a,(.24,.785),(.29,.785))
box(a,.555,.72,.37,.13,'Q、K、V 投影',r'$Q:m\times4096;\ K,V:m\times1024$',color='light');arrow(a,(.485,.785),(.545,.785))
box(a,.555,.535,.37,.12,'QK Norm → RoPE → Attention',r'$QK^{\mathsf{T}}$ → Mask / Softmax → $AV$',color='light');arrow(a,(.74,.71),(.74,.665))
box(a,.30,.545,.18,.10,r'输出投影 $W_o$','$4096\\times 4096$');arrow(a,(.55,.59),(.49,.59))
box(a,.055,.545,.18,.10,'残差相加','返回 $m\\times 4096$');arrow(a,(.295,.59),(.24,.59));arrow(a,(.145,.745),(.145,.652),'orange')
box(a,.055,.365,.18,.095,'RMSNorm');arrow(a,(.145,.535),(.145,.468))
box(a,.30,.365,.27,.095,'gate / up 两路升维','各 $m\\times 4096$ × $4096\\times 12288$',color='sand');arrow(a,(.24,.415),(.29,.415))
box(a,.65,.365,.275,.095,r'$\operatorname{SiLU}(g)\odot u$','$m\\times 12288$',color='sand');arrow(a,(.58,.415),(.64,.415))
box(a,.65,.19,.275,.095,'down 降维','$m\\times 12288$ × $12288\\times 4096$',color='sand');arrow(a,(.787,.355),(.787,.292))
box(a,.30,.19,.27,.095,'残差相加 → 层输出','$m\\times 4096$');arrow(a,(.64,.24),(.58,.24));arrow(a,(.055,.59),(.025,.59),'orange');arrow(a,(.025,.59),(.025,.24),'orange');arrow(a,(.025,.24),(.29,.24),'orange')
a.text(.06,.11,'每层：QKV＋Wo 为 83,886,080m FLOPs；FFN 为 301,989,888m FLOPs。',fontsize=12)
a.text(.06,.065,'序列交互另计 16,384 × B[PS＋P(P＋1)/2] FLOPs；归一化、激活、残差与特殊函数另列。',fontsize=10.5,color=C['muted'])
save(f,'figure-2-2-layer');data['figure_2_2']={'dimensions':q['dimensions'],'summary':q['summary']}
# Figure 3: compare within model, not a shared ranking.
f,axs=plt.subplots(1,2,figsize=(14,7.7));f.subplots_adjust(left=.10,right=.97,bottom=.22,top=.75,wspace=.40)
f.suptitle('图 2-3  KV 共享与潜变量压缩',x=.05,ha='left',y=.96,fontsize=23,weight='bold')
f.text(.05,.875,'相同上下文长度 8192、BF16；两个面板分别比较自己的模型配置。',fontsize=12,color=C['muted'])
for ax,labels,vals,title in [(axs[0],['MHA\n32 KV 头','GQA\n8 KV 头','MQA\n1 KV 头'],[4608,1152,144],'A  固定 Qwen 层数与头宽'),(axs[1],['紧凑潜变量\n512＋64 维','参考展开 K/V'],[216,11520],'B  K3 的 24 个 MLA 层')]:
 ax.bar(range(len(vals)),vals,color=[C['blue'],C['teal'],C['orange']][:len(vals)],width=.55);ax.set_xticks(range(len(vals)),labels);ax.set_yscale('log');ax.set_ylim(50,30000);ax.set_ylabel('上下文状态（MiB，对数轴）');ax.set_title(title,loc='left',fontsize=14,pad=15)
 for i,vv in enumerate(vals):ax.text(i,vv*1.18,f'{vv:,}',ha='center',fontsize=12)
 ax.grid(axis='y',alpha=.15)
f.text(.05,.095,'A：仅 GQA 是真实基线，MHA／MQA 是机制变体；Q 头交互不按 KV 头数同比减少。',fontsize=11)
f.text(.05,.05,'B：未加 KDA／卷积／工作区；紧凑代数路径不代表固定参考代码的实际分配。',fontsize=11,color=C['muted'])
save(f,'figure-2-4-cache');data['figure_2_4']={'history':8192,'qwen_variants_mib':[4608,1152,144],'k3_mla_paths_mib':[216,11520]}
# Figure 4: qualitative pathways plus quantitative decomposition.
f,a=canvas('图 2-4  V4-Flash 的窗口、CSA 与 HCA','43 层主干：2 个纯窗口层＋21 个 CSA 层＋20 个 HCA 层；S=8192，B=1。',9)
for y,title,body in [(.71,'窗口 128','保存最近位置；每步访问局部上下文'),(.53,'CSA：压缩比 4','保存 2048 个压缩条目与索引；主注意力最多选择 512 条'),(.35,'HCA：压缩比 128','保存 64 个粗粒度条目；访问窗口与粗粒度上下文')]:
 box(a,.055,y,.27,.13,title,body.split('；')[0],color='light',size=13)
 arrow(a,(.335,y+.065),(.405,y+.065))
 box(a,.415,y,.52,.13,'访问与新增工作',('主注意力最多选择 512 条；索引扫描全部 2048 条' if title.startswith('CSA') else body.split('；')[-1] if '；' in body else '局部查询与滑动更新'),color='pale',size=12)
a.text(.065,.255,'整模型常驻参考状态（MiB）',fontsize=13,weight='bold')
com=v['components'];names=['窗口','压缩上下文','索引上下文','压缩缓冲'];vals=[com['window_history_bytes']/MiB,com['compressed_history_bytes']/MiB,com['index_history_bytes']/MiB,v['summary']['compressor_buffer_bytes']/MiB]
for i,(n,z) in enumerate(zip(names,vals)):box(a,.06+i*.23,.135,.20,.075,n,f'{z:.3f} MiB',size=11)
a.text(.06,.075,'最后查询：主注意力 17.125 MiB ＋ 索引扫描 10.500 MiB ＝ 27.625 MiB。',fontsize=11)
a.text(.06,.035,'索引仍扫描压缩索引上下文；未完成块与更新峰值单列。图中数据量不是 HBM 实测。',fontsize=10,color=C['muted'])
save(f,'figure-2-5-sparse');data['figure_2_5']=v['summary']|{'components':com}
# Figure 5: structure extrapolation curves and pinned whole-model prefill work.
lengths=np.array([8192,16384,32768,65536,131072,262144,524288,1048576],dtype=np.int64)
qr=lengths*147456
kr=k['components']['kda_recurrent_bytes']+k['components']['short_conv_slots_bytes']+lengths*24*576*2
vr=43*128*512*2+21*(lengths//4)*(512+128)*2+20*(lengths//128)*512*2+v['summary']['compressor_buffer_bytes']
qaccess=qr;kaccess=lengths*24*576*2+2*k['components']['kda_recurrent_bytes']
vaccess=43*128*512*2+21*np.minimum(lengths//4,512)*512*2+20*(lengths//128)*512*2+21*(lengths//4)*128*2
assert int(vr[0])==v['summary']['resident_bytes'];assert int(kr[0])==k['summary']['resident_bytes'];assert int(vaccess[0])==v['summary']['selected_history_payload_bytes']
pre=[q['summary']['matrix_flops']/1e12,load('forward-deepseek-v4-flash-b1-t8192-s0')['summary']['matrix_flops_effective_attention']/1e12,load('forward-kimi-k3-b1-t8192-s0-compact')['summary']['matrix_flops']/1e12]
f,axs=plt.subplots(1,3,figsize=(16,8));f.subplots_adjust(left=.065,right=.98,bottom=.29,top=.75,wspace=.35)
f.suptitle('图 2-5  状态容量、访问与输入工作',x=.05,ha='left',y=.96,fontsize=24,weight='bold');f.text(.05,.875,'B=1；容量曲线为结构推算，非长上下文质量验证；K3 采用紧凑 BF16 MLA＋FP32 递推状态。',fontsize=11)
for ax,series,title in [(axs[0],[qr,vr,kr],'A  常驻状态'),(axs[1],[qaccess,vaccess,kaccess],'B  每步已计状态访问')]:
 for vals,label,col in zip(series,['Qwen3-8B','V4-Flash','K3 compact'],['blue','teal','orange']):ax.plot(lengths/1024,vals/GiB,'o-',label=label,color=C[col],lw=2,ms=4)
 ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks([8,32,128,512,1024],['8K','32K','128K','512K','1M']);ax.set_xlabel('上下文长度');ax.set_ylabel('GiB（对数轴）');ax.set_title(title,loc='left',fontsize=14);ax.grid(alpha=.17);ax.legend(fontsize=8,frameon=False)
a=axs[2];a.bar(range(3),pre,color=[C['blue'],C['teal'],C['orange']],width=.55);a.set_xticks(range(3),['Qwen3\n8B','V4\nFlash','K3\ncompact']);a.set_yscale('log');a.set_ylim(50,4500);a.set_ylabel('矩阵 TFLOPs（对数轴）');a.set_title('C  8192-token prefill',loc='left',fontsize=14)
for i,z in enumerate(pre):a.text(i,z*1.15,f'{z:.3f}',ha='center',fontsize=10)
f.text(.055,.18,'A：K3 含卷积槽；V4 含压缩缓冲。B：Qwen 为旧 KV 读取，V4 为主注意力＋索引数据量，',fontsize=11)
f.text(.055,.135,'K3 为全局上下文读取＋递推矩阵理想读写；未覆盖全部状态操作、写入及物理 HBM。',fontsize=11)
f.text(.055,.09,'C：同为 8192 个新输入、最后位置输出头；采用各自已封存数学／参考路径，未统一所有算子实现。',fontsize=11)
f.text(.055,.045,'三面板分别表示容量、已计访问与计算量；不构成实测速度或同任务质量排名。',fontsize=11,color=C['muted'])
save(f,'figure-2-7-state-growth');data['figure_2_7']={'lengths':lengths.tolist(),'resident_bytes':dict(zip(['qwen','v4','k3'],[qr.tolist(),vr.tolist(),kr.tolist()])),'accounted_access_bytes':dict(zip(['qwen','v4','k3'],[qaccess.tolist(),vaccess.tolist(),kaccess.tolist()])),'prefill_matrix_tflops':pre,'scope':'Structural length extrapolation; partial state access; pinned compact K3 prefill, no measured runtime or quality.'}
# Figure 6: residual connection flow.
f,a=canvas('图 2-6  稠密层与 V4 层的数据流','同为 4096 维子层输入；连接状态、子层矩阵与专家路由分别标注。',10)
for x,title in [(.10,'Qwen3-8B'),(.60,'V4-Flash')]:a.text(x+.14,.82,title,ha='center',fontsize=17,weight='bold')
for y,title,body in [(.66,'GQA 注意力','QKV → 上下文访问 → Wo'),(.45,'残差相加／归一化','$m\\times 4096$'),(.24,'稠密 SwiGLU','所有 token 使用同组 FFN 权重')]:box(a,.10,y,.28,.12,title,body,color='light')
arrow(a,(.24,.65),(.24,.58));arrow(a,(.24,.44),(.24,.37));arrow(a,(.24,.23),(.24,.15));a.text(.24,.115,'残差相加 → 下一层',ha='center',fontsize=12)
for y,title,body in [(.66,'mHC 汇合 → 注意力','窗口／压缩／索引；输入 $m\\times 4096$'),(.45,'mHC 残差混合','保留四路状态；子层维度不乘 4'),(.24,'mHC 汇合 → MoE','256 选 6＋1 共享专家；每专家 $t_e$ 行')]:box(a,.60,y,.31,.12,title,body,color='sand')
arrow(a,(.755,.65),(.755,.58));arrow(a,(.755,.44),(.755,.37));arrow(a,(.755,.23),(.755,.15));a.text(.755,.115,'残差混合 → 下一层',ha='center',fontsize=12)
for x,center,right in [(.055,.24,.10),(.55,.755,.60)]:
 arrow(a,(center,.79),(x,.79),'orange');arrow(a,(x,.79),(x,.51),'orange');arrow(a,(x,.51),(right-.01,.51),'orange')
 arrow(a,(center,.435),(x,.435),'orange');arrow(a,(x,.435),(x,.115),'orange');arrow(a,(x,.115),(right+.03,.115),'orange')
a.text(.045,.05,'橙色侧路表示残差传递；完整归一化、混合与输出连接必须计量，不能仅累加注意力和专家矩阵。',fontsize=10.5,color=C['muted'])
save(f,'figure-2-8-residual');data['figure_2_8']={'qwen_hidden':4096,'v4_sublayer_hidden':4096,'v4_residual_streams':4,'v4_experts':256,'v4_top_k':6,'v4_shared':1}
# Figure 7: per-request capacity plus expert geometry.
f,axs=plt.subplots(1,2,figsize=(15,8.5),gridspec_kw={'width_ratios':[1.25,1]});f.subplots_adjust(left=.075,right=.96,bottom=.27,top=.77,wspace=.38)
f.suptitle('图 2-7  模型容量与架构形状',x=.05,ha='left',y=.96,fontsize=24,weight='bold');f.text(.05,.88,'容量使用十进制 GB；固定工作区 2 GiB，BF16 KV，8192 上下文，每柱包含一个请求。',fontsize=11)
rows=[(cap,16,24_000_000_000),(cap70,8,80_000_000_000),(cap70,4,80_000_000_000)];entries=[next(z for z in d['capacity_comparisons'] if z['matrix_bits']==bits and z['capacity_bytes']==capacity) for d,bits,capacity in rows]
a=axs[0];y=np.arange(3);left=np.zeros(3)
for field,label,col in [('weight_bytes','权重','blue'),('workspace_bytes','固定预留','orange'),('kv_bytes_per_request','1 条请求 KV','teal')]:
 vals=np.array([z[field]/1e9 for z in entries]);a.barh(y,vals,left=left,label=label,color=C[col],height=.5);left+=vals
for i,z in enumerate(entries):a.plot([z['capacity_bytes']/1e9]*2,[i-.32,i+.32],color=C['ink'],lw=2);a.text(2,i+.37,f"{ {24_000_000_000:'RTX 4090',80_000_000_000:'H100 SXM'}[z['capacity_bytes']]} {z['capacity_bytes']/1e9:.0f} GB：最多容纳 {z['maximum_requests']} 条",fontsize=10)
a.set_yticks(y,['Qwen8\nBF16','70B\n8-bit 方案','70B\n4-bit 方案']);a.invert_yaxis();a.set_xlim(0,87);a.set_xlabel('GB');a.set_title('A  权重、状态与容量',loc='left',fontsize=14);a.legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=3,fontsize=9,frameon=False)
a=axs[1];c32=load('llama70-capacity-32k')
y8=next(z['maximum_requests'] for z in cap70['capacity_comparisons'] if z['matrix_bits']==4 and z['capacity_bytes']==80_000_000_000)
y32=next(z['maximum_requests'] for z in c32['capacity_comparisons'] if z['matrix_bits']==4 and z['capacity_bytes']==80_000_000_000)
a.bar(['8K 上下文','32K 上下文'],[y8,y32],color=[C['blue'],C['orange']]);a.set_ylabel('容量允许的独立请求数');a.set_title('B  上下文增长挤占并发容量',loc='left',fontsize=14)
for i,z in enumerate([y8,y32]):a.text(i,z+.25,str(z),ha='center')
a.set_ylim(0,17);a.text(.5,.88,'70B · 4-bit 量化 · H100 SXM 80 GB\n固定预留 2 GiB',transform=a.transAxes,ha='center',fontsize=11)
save(f,'figure-2-9-capacity');data['figure_2_9']={'capacity_rows':entries,'history_capacity':{'history_tokens':[8192,32768],'maximum_requests':[y8,y32]}}
# Figure 8: request timeline and separate known resource / measured quality observations.
f,a=canvas('图 2-8  请求输入与计算状态映射','统一资源请求：S=0、P=128、G=4、B=1；首输出来自 prefill，decode 调用为 G−1=3。',9)
for i,(title,body) in enumerate([('Prefill','输入 128 → 输出 $y_1$\n保留 128 位置'),('Decode 1','输入 $y_1$ → 输出 $y_2$\n保留 129 位置'),('Decode 2','输入 $y_2$ → 输出 $y_3$\n保留 130 位置'),('Decode 3','输入 $y_3$ → 输出 $y_4$\n保留 131 位置')]):
 x=.055+i*.235;box(a,x,.66,.195,.15,title,body,color='light' if i==0 else 'pale',size=13)
 if i<3:arrow(a,(x+.203,.735),(x+.225,.735))
a.text(.055,.60,'$y_4$ 已返回但尚未再次进入模型；最终状态不是 132 个 token。',fontsize=12,color=C['orange'])
vals=[z['summary']['matrix_flops']/1e12 for z in r['comparisons']]
ax=f.add_axes([.095,.22,.44,.29]);ax.barh(range(4),vals,color=[C['blue'],C['teal'],C['teal'],C['orange']]);ax.set_yticks(range(4),['Qwen3-8B','V4-Flash','V4-Pro','K3 expanded']);ax.invert_yaxis();ax.set_xlim(0,34);ax.set_xlabel('完整请求矩阵 TFLOPs')
for i,z in enumerate(vals):ax.text(z+.5,i,f'{z:.3f}',va='center',fontsize=10)
box(a,.65,.235,.29,.27,'同一文本的检索实验','Qwen 与 V4-Flash 各 8/8\n输入 token 数、部署与精度不同\nPro／K3 未测',color='sand',size=13)
# Scope is stated in the external caption.
a.text(.055,.055,'左侧按给定 token 数估算运算量；右侧用同一文本检查回答，并记录各模型的输入长度。',fontsize=11,color=C['muted'])
save(f,'figure-2-10-request');data['figure_2_10']={'contract':r['contract'],'matrix_tflops':vals,'model_order':[z['model'] for z in r['comparisons']]}

# Query groups illustrate sharing before whole-model numerical comparisons.
f,a=canvas('','',8)
for x,title,groups in [(.05,'MHA',4),(.375,'GQA',2),(.70,'MQA',1)]:
 a.text(x+.125,.80,title,ha='center',fontsize=18,weight='bold')
 for j in range(4):
  qx=x+.01+j*.073
  box(a,qx,.62,.051,.065,rf'$Q_{j+1}$',color='light',size=13)
  g=j if groups==4 else j//2 if groups==2 else 0
  kx=x+.01+(g+.5)*.292/groups
  arrow(a,(qx+.0255,.61),(kx,.47))
 for g in range(groups):
  cx=x+.01+(g+.5)*.292/groups
  box(a,cx-.030,.37,.060,.09,rf'$K_{g+1},V_{g+1}$',size=11)
 a.text(x+.15,.27,f'{groups} 组上下文 K、V',ha='center',fontsize=13)
 a.text(x+.15,.19,'4 组查询评分与加权汇总',ha='center',fontsize=11,color=C['muted'])
save(f,'figure-2-3-sharing');data['figure_2_3']={'query_heads':4,'kv_groups':[4,2,1],'scope':'Teaching schematic; equal history and head width.'}
# Attention choice and expert choice are independent architectural dimensions.
f,a=canvas('','',8.8)
box(a,.35,.75,.30,.075,'当前层输入',r'$X\in\mathbb{R}^{m\times2048}$',size=13)
for x,title,body in [(.08,'线性注意力层（共 30 层）','固定递推矩阵＋短卷积'),(.57,'完整注意力层（共 10 层）','随上下文增长的全局 KV')]:
 box(a,x,.54,.35,.13,title,body,color='light',size=13)
 arrow(a,(.50,.74),(x+.175,.68))
 arrow(a,(x+.175,.53),(.50,.45))
box(a,.32,.345,.36,.095,'归一化与残差连接','执行模型配置指定的注意力分支',size=13)
arrow(a,(.50,.335),(.50,.285))
box(a,.10,.16,.37,.115,'路由专家：256 选 8',r'专家 $e$ 处理 $t_e$ 行',color='sand',size=13)
box(a,.59,.16,.31,.115,'共享专家',r'处理全部 $m$ 行',color='sand',size=13)
arrow(a,(.49,.30),(.285,.28));arrow(a,(.51,.30),(.745,.28))
save(f,'figure-2-6-hybrid');data['figure_2_6']={'linear_layers':30,'full_attention_layers':10,'experts':256,'top_k':8,'shared':1}

# Additional mechanism diagrams share this chapter's font and output handling.
import sys
sys.path.insert(0,str(HERE.parent))
from teaching_figures import draw as draw_teaching_figures
data['teaching_diagrams']=draw_teaching_figures(2,save,ROOT)

(HERE/'figure-data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
from teaching_revision import draw as draw_revision
out.extend(draw_revision(HERE,data))
out=list(dict.fromkeys(out))
md=HERE.parent/'02-模型架构.md';raw=md.read_text()
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
for p in out:
 if p.suffix=='.svg':body=body.replace('src="ch02/'+p.name+'"','src="data:image/png;base64,'+base64.b64encode(p.with_suffix('.png').read_bytes()).decode()+'"')
css='''body{margin:0;background:#fafaf8;color:#243640;font:18px/1.95 Georgia,"Songti SC",serif}main{max-width:960px;margin:auto;padding:50px 38px 90px;background:white}h1,h2,h3{font-family:Arial,"PingFang SC",sans-serif;line-height:1.45;color:#163747}h1{font-size:36px}h2{font-size:28px;border-top:1px solid #d5e1e4;margin-top:65px;padding-top:28px}h3{font-size:23px;margin-top:40px}p{margin:1em 0}a{color:#246f91;text-underline-offset:3px}img{display:block;width:100%;height:auto;margin:26px auto 10px}em{font-size:15px;color:#55707d}table{border-collapse:collapse;width:100%;font-size:15px;line-height:1.7;margin:24px 0}td,th{padding:10px 12px;border-bottom:1px solid #d5e1e4;text-align:left}th{background:#edf4f6}blockquote{margin:28px 0;padding:16px 24px;border-left:4px solid #138b83;background:#f1f8f6;font-size:16px}code{font:0.85em/1.65 Menlo,monospace;background:#f0f4f6;overflow-wrap:anywhere}pre{white-space:pre-wrap}.equation{font:20px/1.8 Georgia,"Songti SC",serif;text-align:center;background:#f7f9fa;padding:18px 12px;margin:25px 0;overflow-wrap:anywhere}nav{font:16px/1.9 Arial,"PingFang SC",sans-serif;background:#f0f6f8;padding:18px 24px}nav a{display:block}@media(max-width:650px){main{padding:25px 18px}body{font-size:17px}table{display:block;overflow-x:auto}h1{font-size:29px}h2{font-size:25px}.equation{font-size:17px}}@media print{main{max-width:none;padding:0}h2,h3{break-after:avoid}img,blockquote{break-inside:avoid}body{font-size:11pt}}'''
css+=' main{max-width:760px}img{max-width:720px}@media print{img{width:420pt;max-width:100%}}'
css+=math_css+' .katex{font-size:1.04em;position:relative}.katex .katex-mathml{contain:strict}.katex-display{overflow-x:auto;overflow-y:hidden;padding:14px 0}.table-scroll{overflow-x:auto;max-width:100%}td{min-width:120px}td .katex{white-space:nowrap}@media(max-width:650px){table{display:table}td{min-width:145px}}'
nav=''.join('<a href="#'+ident+'">'+title+'</a>' for ident,title in re.findall(r'<h2 id="([^"]+)">(2\.\d+ [^<]+)</h2>',body))
page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>第 2 章 模型架构</title><style>'+css+'</style><main><nav>'+nav+'</nav>'+body+'</main></html>'
import sys
sys.path.insert(0,str(HERE.parent))
from teaching_reading import readable_diagrams
from preview_output import preview_path
page=readable_diagrams(page)
html_path=preview_path(HERE.parent/'02-模型架构.html');html_path.write_text(page)
from book_assets import sync_figure_index
active_assets=sync_figure_index(HERE)
artifacts=out+[HERE/'figure-data.json',HERE/'model-comparison.json',HERE/'model-comparison.md',HERE/'comparison-v4-decode.json',HERE/'long-context-comparison.json',HERE/'compare_long_context.py',html_path,md]+active_assets
manifest={'chapter':2,'generator':'manuscripts/ch02/build.py','font_family':family,'figures':len(re.findall(r'!\[',raw)),'outputs':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in artifacts if not (p.parent==ROOT/'manuscripts' and p.suffix=='.md' and p.name[:2].isdigit())]}
(HERE/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(f'Built {len(out)} figure files and reading HTML.')
