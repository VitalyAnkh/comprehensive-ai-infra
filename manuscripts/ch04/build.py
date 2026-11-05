#!/usr/bin/env python3
"""Build chapter 4 publication figures and offline reading HTML from locked evidence."""
from pathlib import Path
import argparse,base64,hashlib,html,json,re,subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
import numpy as np
from derive import derive
import markdown
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--font');args=parser.parse_args()
import sys
sys.path.insert(0,str(HERE.parent))
from figure_style.typography import configure_font
font,family=configure_font(args.font)
plt.rcParams.update({'font.family':[family,'DejaVu Sans'],'font.size':11,'axes.unicode_minus':False,'svg.fonttype':'path','svg.hashsalt':'ch04-v1','axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white','text.color':'#203a47','axes.labelcolor':'#203a47','axes.edgecolor':'#aabbc3'})
C={'blue':'#286c90','teal':'#15867b','orange':'#bd742a','ink':'#203a47','pale':'#edf3f6','light':'#e5f3ed','sand':'#fff1df','line':'#b7c9d0','gray':'#6a7c87','red':'#aa4f4f'}
for z in json.loads((HERE/'sources.json').read_text())['sources']:
 b=(ROOT/z['path']).read_bytes()
 if not b.startswith(b'version https://git-lfs.github.com/spec/v1') and hashlib.sha256(b).hexdigest()!=z['sha256']:raise SystemExit('Input changed; review before relocking: '+z['path'])
def read(p):return json.loads((ROOT/p).read_text())
def calc(n):return read('calculations/results/'+n+'.json')
outputs=[];data={};extent_issues=[]
teaching=derive()
(HERE/'teaching-data.json').write_text(json.dumps(teaching,ensure_ascii=False,indent=2)+'\n')
def save(f,name):
 f.canvas.draw();renderer=f.canvas.get_renderer()
 for t in f.findobj(matplotlib.text.Text):
  if not t.get_visible() or not t.get_text():continue
  # Matplotlib retains off-axis tick artists; they are not drawn.
  ax=t.axes
  if ax is None:
   hidden_tick=False
   for axis in f.axes:
    for ticks,limits in [(axis.xaxis.get_major_ticks(),axis.get_xlim()),(axis.yaxis.get_major_ticks(),axis.get_ylim())]:
     for tick in ticks:
      if t in [tick.label1,tick.label2] and not min(limits)<=tick.get_loc()<=max(limits):hidden_tick=True
   if hidden_tick:continue
  b=t.get_window_extent(renderer)
  if b.x0<-2 or b.y0<-2 or b.x1>f.bbox.width+2 or b.y1>f.bbox.height+2:extent_issues.append({'figure':name,'text':t.get_text(),'bbox':list(b.bounds)})
 for ext in ['svg','png','pdf']:
  p=HERE/f'{name}.{ext}';f.savefig(p,dpi=180,bbox_inches='tight',pad_inches=.15);outputs.append(p)
 plt.close(f)
def canvas(h=7):
 f,a=plt.subplots(figsize=(13,h));f.subplots_adjust(left=.025,right=.975,bottom=.04,top=.97);a.set(xlim=(0,1),ylim=(0,1));a.axis('off');return f,a
def box(a,x,y,w,h,label,color='pale',size=12):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.006,rounding_size=.008',facecolor=C[color],edgecolor=C['line'],lw=1.2));a.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size,linespacing=1.65)
def arrow(a,p,q,dashed=False):a.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=14,lw=1.4,color=C['gray'] if dashed else C['teal'],linestyle='--' if dashed else '-'))
# Components: generic paths, no vendor-equivalence claim.
f,a=canvas(7)
box(a,.03,.78,.18,.13,'主机 CPU\n准备输入、提交任务')
box(a,.32,.78,.24,.13,'片外内存\n权重、输入、最终输出')
box(a,.71,.78,.23,.13,'设备控制与任务调度')
arrow(a,(.22,.845),(.31,.845));a.plot([.12,.12,.825],[.915,.96,.96],ls='--',color=C['gray'],lw=1.4);arrow(a,(.825,.96),(.825,.915),True);a.text(.46,.969,'提交命令',ha='center',fontsize=10);arrow(a,(.83,.77),(.58,.38),True)
box(a,.31,.57,.25,.12,'共享缓存 / 内存接口');arrow(a,(.435,.77),(.435,.70))
a.add_patch(FancyBboxPatch((.04,.05),.90,.43,boxstyle='round,pad=.012',facecolor='none',edgecolor=C['gray'],linestyle='--'))
a.text(.06,.445,'局部计算单元组（可有多组）',fontsize=11,color=C['gray'])
box(a,.08,.25,.20,.12,'局部输入缓冲\n分块与复用',color='light')
box(a,.38,.25,.20,.12,'矩阵计算单元\n乘加与局部累积',color='sand')
box(a,.69,.25,.20,.12,'累加存储\n寄存器 / 专用存储')
box(a,.69,.08,.20,.10,'向量与通用计算')
arrow(a,(.37,.56),(.18,.38));a.text(.12,.52,'加载 / 搬移',fontsize=11)
arrow(a,(.29,.31),(.37,.31));arrow(a,(.59,.31),(.68,.31));arrow(a,(.79,.24),(.79,.19));arrow(a,(.68,.12),(.48,.12));arrow(a,(.48,.12),(.48,.24));a.text(.35,.075,'中间结果继续计算',fontsize=10)
arrow(a,(.91,.37),(.91,.62));arrow(a,(.90,.63),(.57,.63));a.text(.70,.65,'按需要写回',fontsize=11)

save(f,'figure-4-2-components');data['4-2']={'type':'mechanism','scope':'generic hierarchy; not identical vendor topology'}
# Attention resource balance: each change moves the same bottleneck.
f,a=plt.subplots(figsize=(12,5.4));f.subplots_adjust(left=.23,right=.96,bottom=.16,top=.85)
fa=calc('fa4-qwen8-resource-balance');rows=fa['scenarios'][:4]
vals=np.array([[r['cycles'][k]['numerator']/r['cycles'][k]['denominator'] for k in ['matrix','smem','exp']] for r in rows])
for i,(label,col) in enumerate([('矩阵','blue'),('共享内存读取','teal'),('指数','orange')]):
 a.barh(np.arange(4)+(i-1)*.23,vals[:,i],height=.21,label=label,color=C[col])
 for j,v in enumerate(vals[:,i]):a.text(v+12,j+(i-1)*.23,f'{v:.0f}',va='center',fontsize=10)
a.set_yticks(range(4),['原配置','仅矩阵 ×2','矩阵与指数 ×2','三项同时 ×2']);a.invert_yaxis();a.set_xlabel('单个 SM 处理一块所需周期');a.set_xlim(0,1150);a.legend(ncol=3,frameon=False,loc='upper left',bbox_to_anchor=(0,1.18))
save(f,'figure-4-4-attention');data['4-4']={'source':'calculations/results/fa4-qwen8-resource-balance.json','service_cycles':vals.tolist(),'scope':'first 4 scenarios, M=N=d=128'}
# Precision paths.
f,a=canvas(7)
box(a,.03,.43,.20,.23,'压缩权重 + scale\n8.5 MiB','light')
for y in [.78,.22]:arrow(a,(.235,.55),(.32,y))
box(a,.33,.68,.22,.20,'展开 / 反量化\nBF16 副本 32 MiB')
box(a,.63,.68,.22,.20,'矩阵计算\n匹配累加精度','sand');arrow(a,(.56,.78),(.62,.78))
box(a,.33,.12,.22,.20,'低精度矩阵计算\n填充 / 部分和','sand')
box(a,.63,.12,.22,.20,'缩放、合并与转换\n规定输出格式');arrow(a,(.56,.22),(.62,.22))


save(f,'figure-4-5-precision');data['4-5']={'kind':'teaching_format','weights':4096**2,'bf16_bytes':33554432,'four_bit_bytes':8388608,'scale_bytes':524288}
# Concurrency limits bandwidth even when the physical interface is faster.
f,a=plt.subplots(figsize=(11,5.5));f.subplots_adjust(left=.10,right=.96,bottom=.16,top=.90)
n=np.arange(0,10001);transaction=128;latency=500e-9;windows=[calc('window-qwen3-8b-rtx4090-n4096'),calc('window-qwen3-8b-rtx5090-n4096')]
bandwidths=[w['scenario']['bandwidth_bytes_per_second'] for w in windows];thresholds=[w['summary']['required_transactions'] for w in windows]
for bandwidth,col,name in [(bandwidths[0],'blue','RTX 4090'),(bandwidths[1],'teal','RTX 5090')]:
 a.plot(n,np.minimum(bandwidth,n*transaction/latency)/1e12,color=C[col],lw=2.5,label=f'{name}：{bandwidth/1e9:.0f} GB/s')
for threshold,rate in zip(thresholds,[b/1e12 for b in bandwidths]):
 a.plot(threshold,rate,'o',color=C['orange']);a.annotate(f'至少 {threshold} 个',xy=(threshold,rate),xytext=(threshold+350,rate-.23),fontsize=11,arrowprops={'arrowstyle':'-','color':C['gray']})
a.axvline(4096,color=C['gray'],ls=':',lw=1);a.set(xlim=(0,10000),ylim=(0,2.25),xlabel='可同时未完成的请求数 N',ylabel='可达到的带宽上界 / TB/s');a.legend(frameon=False,loc='upper left');a.grid(axis='y',alpha=.15)
save(f,'figure-4-7-memory');data['4-7']={'kind':'named_device_concurrency','sources':['calculations/results/window-qwen3-8b-rtx4090-n4096.json','calculations/results/window-qwen3-8b-rtx5090-n4096.json'],'transaction_bytes':transaction,'latency_seconds':latency,'interface_bytes_per_second':bandwidths,'minimum_requests':thresholds,'requests':n.tolist(),'bandwidth_upper_bytes_per_second':[np.minimum(bw,n*transaction/latency).tolist() for bw in bandwidths]}
# One input pipeline, three buffer capacities.
p=calc('attention-input-base');f,axes=plt.subplots(3,1,figsize=(12,8.8),sharex=True);f.subplots_adjust(left=.12,right=.96,bottom=.09,top=.91,hspace=.58)
for a,slots in zip(axes,[1,2,3]):
 r=teaching['baseline'][slots-1]
 for t in r['chunks']:
  y=t['chunk'];a.barh(y,t['slot_released']-t['issue_start'],left=t['issue_start'],height=.64,color='#e8ecef');a.barh(y,t['transfer_end']-t['issue_start'],left=t['issue_start'],height=.44,color=C['blue']);a.barh(y,t['compute_end']-t['compute_start'],left=t['compute_start'],height=.44,color=C['teal']);a.plot(t['data_ready'],y,'|',color=C['orange'],markersize=11)
 a.set_yticks(range(4),['块 0','块 1','块 2','块 3']);a.invert_yaxis();a.set_xlim(0,1320);a.set_title(f'{slots} 个输入槽：完成 {r["finish_tick"]} tick',loc='left',fontsize=12);a.set_xticks([0,320,640,960,1280]);a.tick_params(labelbottom=True)
axes[-1].set_xlabel('时间 / tick（B200 SM 时钟周期）')
f.text(.12,.966,'蓝：输入传输　橙线：数据就绪　绿：计算　浅灰：槽位占用至计算结束',fontsize=11)
save(f,'figure-4-9-pipeline');data['4-9']={'input_scenario':p['scenario'],'input_rows':p['rows'],'displayed_schedules':teaching['baseline'][:3]}
# Same bandwidth change, two message sizes; independent axes show startup cost.
f,axes=plt.subplots(1,2,figsize=(12,4.7));f.subplots_adjust(left=.08,right=.96,bottom=.20,top=.77,wspace=.42)
payload=np.array([8192,2097152]);rates=np.array(teaching['interconnect']['rates'],dtype=float);alpha=teaching['interconnect']['alpha_us'];total=[]
for a,size,label in zip(axes,payload,['M=1：8 KiB','M=256：2 MiB']):
 ser=size/rates*1e6;total.append((ser+alpha).tolist());a.bar([0,1],[alpha,alpha],color=C['orange'],label='启动');a.bar([0,1],ser,bottom=alpha,color=C['blue'],label='传输');a.set_xticks([0,1],['A100 300 GB/s','H100 450 GB/s']);a.set_title(label,loc='left',fontsize=12);a.set_ylabel('时间 / μs');a.set_ylim(0,max(ser+alpha)*1.2)
 for i,v in enumerate(ser+alpha):a.text(i,v+max(ser+alpha)*.025,f'{v:.2f}',ha='center',fontsize=11)
axes[0].legend(frameon=False,ncol=2,loc='upper left',bbox_to_anchor=(0,1.38))
save(f,'figure-4-11-interconnect');data['4-11']={'kind':'nvlink_per_direction','devices':['A100 SXM','H100 SXM'],'payload_bytes':payload.tolist(),'one_way_bytes_per_second':rates.tolist(),'alpha_seconds':alpha/1e6,'total_us':total}
# Fixed weights and growing KV: one remaining resource constraint.
f,a=plt.subplots(figsize=(11,5.5));f.subplots_adjust(left=.11,right=.96,bottom=.16,top=.94)
b=np.arange(1,33);W=15136811008;K=1207959552;a.plot(b,np.full(len(b),W/1e9),color=C['blue'],lw=2,label='本步所用权重：每批读一次');a.plot(b,b*K/1e9,color=C['teal'],lw=2,label='KV：每请求读取一次');a.axvline(13,color=C['orange'],ls=':');a.annotate('从 batch 13 起，KV 超过权重',xy=(13,13*K/1e9),xytext=(16,9),arrowprops={'arrowstyle':'->','color':C['orange']},fontsize=11);a.set_xlabel('batch B');a.set_ylabel('每步读取 / GB');a.set_xlim(1,32);a.set_ylim(0,42);a.legend(frameon=False,fontsize=11,loc='upper left')
save(f,'figure-4-12-specialization');data['4-12']={'kind':'conditional_Qwen_OpenTallas_memory_only','active_weight_bytes':W,'kv_bytes_per_request':K,'batch':b.tolist(),'kv_read_bytes':(b*K).tolist()}
# Test the cache hypothesis with separately collected time and traffic.
f,axes=plt.subplots(1,2,figsize=(13,5.4));f.subplots_adjust(left=.08,right=.97,bottom=.20,top=.88,wspace=.28)
meas=read('experiments/ch04/04-06/results/projection-summary.json');traffic=read('experiments/ch04/04-06/results/projection-traffic.json')['rows']
conditions=[(1,'reused'),(1,'rotating'),(256,'reused'),(256,'rotating')]
labels=[f'M={n}\n'+('复用' if mode=='reused' else '轮换') for n,mode in conditions];colors=[C['blue'],C['teal']]*2
wall=[next(z['wall_median_us'] for z in meas if z['device']=='cuda' and z['m']==n and z['mode']==mode) for n,mode in conditions]
dram=[next(z['totals']['dram__bytes_op_read.sum'] for z in traffic if z['m']==n and z['mode']==mode)/2**20 for n,mode in conditions]
for a,values,ylabel,limit in [(axes[0],wall,'每次调用总耗时 / μs',62),(axes[1],dram,'单独测得的 DRAM 读取 / MiB',40)]:
 a.bar(range(4),values,color=colors,width=.65);a.set_xticks(range(4),labels);a.set_ylabel(ylabel);a.set_ylim(0,limit)
 for i,v in enumerate(values):a.text(i,v+limit*.025,'256 B' if v<.001 else f'{v:.1f}',ha='center',fontsize=11)
save(f,'figure-4-14-performance');data['4-14']={'measured_projection':meas,'traffic':traffic,'display_conditions':conditions,'wall_us':wall,'dram_read_mib':dram}
from illustrations import draw_additions
draw_additions(plt, C, canvas, box, arrow, save, data)
(HERE/'figure-data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
(HERE/'figure-layout-check.json').write_text(json.dumps({'outside_canvas_text':extent_issues},ensure_ascii=False,indent=2)+'\n')

import sys
sys.path.insert(0,str(HERE.parent))
from teaching_revision import draw as draw_revision
outputs.extend(draw_revision(HERE,data,teaching))
from architecture_evolution import draw as draw_evolution
outputs.extend(draw_evolution(HERE))
outputs=list(dict.fromkeys(outputs))
md=HERE.parent/'04-加速器架构.md';raw=md.read_text()
maths=[]
def protect_math(match):
 text=match.group(0);display=text.startswith('$$');latex=text[2:-2] if display else text[1:-1]
 token=f'MATHPLACEHOLDER{len(maths)}END';maths.append({'latex':latex.strip(),'display':display,'token':token})
 return ('\n\n'+token+'\n\n') if display else token
protected=re.sub(r'\$\$[\s\S]*?\$\$|\$[^$\n]+\$',protect_math,raw)
body=markdown.markdown(protected,extensions=['tables','footnotes','fenced_code','toc'],output_format='html')
node_code="const fs=require('fs'),k=require(process.argv[1]);let a=JSON.parse(fs.readFileSync(0,'utf8'));process.stdout.write(JSON.stringify(a.map(x=>k.renderToString(x.latex,{displayMode:x.display,throwOnError:true,output:'htmlAndMathml'}))))"
result=subprocess.run(['node','-e',node_code,str(HERE.parent/'ch03/vendor/katex/katex.js')],input=json.dumps(maths),text=True,capture_output=True)
if result.returncode: raise SystemExit(result.stderr)
rendered=json.loads(result.stdout)
for entry,result in zip(maths,rendered):
 body=body.replace('<p>'+entry['token']+'</p>',result) if entry['display'] else body.replace(entry['token'],result)
math_css=(HERE.parent/'ch03/vendor/katex/katex.min.css').read_text()
def font_data(match):
 path=HERE.parent/'ch03/vendor/katex'/match[1];suffix=path.suffix[1:]
 return 'url(data:font/'+suffix+';base64,'+base64.b64encode(path.read_bytes()).decode()+')'
math_css=re.sub(r'url\((fonts/[^)]+)\)',font_data,math_css)
body=re.sub(r'<table>(.*?)</table>',r'<div class="table-scroll"><table>\1</table></div>',body,flags=re.S)
(HERE/'math-validation.json').write_text(json.dumps({'renderer':'KaTeX 0.16.11','expressions':len(maths),'display_expressions':sum(x['display'] for x in maths),'errors':[]},indent=2)+'\n')

for p in outputs:
 if p.suffix=='.svg':body=body.replace('src="ch04/'+p.name+'"','src="data:image/png;base64,'+base64.b64encode(p.with_suffix('.png').read_bytes()).decode()+'"')
css='''body{margin:0;background:#fafaf8;color:#243640;font:18px/1.95 Georgia,"Songti SC",serif}main{max-width:960px;margin:auto;padding:50px 38px 90px;background:white}h1,h2,h3{font-family:Arial,"PingFang SC",sans-serif;line-height:1.45;color:#163747}h1{font-size:36px}h2{font-size:28px;border-top:1px solid #d5e1e4;margin-top:65px;padding-top:28px}h3{font-size:23px;margin-top:40px}p{margin:1em 0}a{color:#246f91;text-underline-offset:3px}img{display:block;width:100%;height:auto;margin:26px auto 10px}em{font-size:15px;color:#55707d}table{border-collapse:collapse;width:100%;font-size:15px;line-height:1.7;margin:24px 0}td,th{padding:10px 12px;border-bottom:1px solid #d5e1e4;text-align:left}th{background:#edf4f6}blockquote{margin:28px 0;padding:16px 24px;border-left:4px solid #138b83;background:#f1f8f6;font-size:16px}code{font:0.85em/1.65 Menlo,monospace;background:#f0f4f6;overflow-wrap:anywhere}pre{white-space:pre-wrap}.equation{font:20px/1.8 Georgia,"Songti SC",serif;text-align:center;background:#f7f9fa;padding:18px 12px;margin:25px 0;overflow-wrap:anywhere}nav{font:16px/1.9 Arial,"PingFang SC",sans-serif;background:#f0f6f8;padding:18px 24px}nav a{display:block}.footnote{font-size:14px;line-height:1.8}.footnote li{margin-bottom:12px}@media(max-width:650px){main{padding:25px 18px}body{font-size:17px}table{display:block;overflow-x:auto}h1{font-size:29px}h2{font-size:25px}.equation{font-size:17px}}@media print{main{max-width:none;padding:0}h2,h3{break-after:avoid}img,blockquote{break-inside:avoid}body{font-size:11pt}}'''
css+=' main{max-width:760px}img{max-width:720px}@media print{img{width:420pt;max-width:100%}}'
css+=math_css+' .katex{font-size:1.04em}.katex-display{overflow-x:auto;overflow-y:hidden;padding:12px 0}.table-scroll{overflow-x:auto;max-width:100%}@media(max-width:650px){table{display:table}}'
nav=''.join('<a href="#'+ident+'">'+title+'</a>' for ident,title in re.findall(r'<h2 id="([^"]+)">(4\.\d+ [^<]+)</h2>',body))
page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第 4 章 加速器架构</title><style>'+css+'</style><main><nav>'+nav+'</nav>'+body+'</main></html>'
import sys
sys.path.insert(0,str(HERE.parent))
from teaching_reading import readable_diagrams
from preview_output import preview_path
page=readable_diagrams(page)
html_path=preview_path(HERE.parent/'04-加速器架构.html');html_path.write_text(page)
from book_assets import sync_figure_index
active_assets=sync_figure_index(HERE)
artifacts=outputs+[HERE/'figure-data.json',HERE/'teaching-data.json',html_path,md]+active_assets
(HERE/'manifest.json').write_text(json.dumps({'chapter':4,'generator':'manuscripts/ch04/build.py','figures':len(re.findall(r'!\[',raw)),'font_family':family,'outputs':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in artifacts if not (p.parent==ROOT/'manuscripts' and re.match(r'^[01][0-9]-',p.name))]},ensure_ascii=False,indent=2)+'\n')
print(f'Built {len(outputs)} figure files and reading HTML; {len(extent_issues)} text extent warnings.')
