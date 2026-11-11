#!/usr/bin/env python3
"""Build chapter 11 figures and offline HTML; read-only access to locked evidence."""
from pathlib import Path
from fractions import Fraction
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
plt.rcParams.update({'font.family':[family,'DejaVu Sans'],'font.size':12,'axes.unicode_minus':False,'svg.fonttype':'path','svg.hashsalt':'chapter11-v1','axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white','text.color':'#233c48','axes.labelcolor':'#233c48','axes.edgecolor':'#9bacb4'})
C={'ink':'#233c48','blue':'#286b94','teal':'#20867e','orange':'#bb682d','red':'#aa4949','pale':'#edf3f7','green':'#e4f2ed','sand':'#fbf0df','muted':'#506975','line':'#bccdd5','gray':'#e1e6ea'}
for row in json.loads((HERE/'sources.json').read_text())['sources']:
 if hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()!=row['sha256']:raise SystemExit('Locked source changed: '+row['path'])
def load(name):return json.loads((ROOT/'calculations/results'/f'{name}.json').read_text())
lifecycle=load('environment-lifecycle-default');route=load('routing-cost-book');retry=load('retry-paths-book')
from platform_design import build_design
build_design(HERE / "platform-design.json")
outputs=[];data={};warnings=[]
def save(f,name):
 f.canvas.draw();renderer=f.canvas.get_renderer()
 for t in f.findobj(matplotlib.text.Text):
  if not t.get_visible() or not t.get_text():continue
  bb=t.get_window_extent(renderer)
  if bb.x0<0 or bb.y0<0 or bb.x1>f.bbox.width or bb.y1>f.bbox.height:warnings.append({'figure':name,'text':t.get_text()})
 for ext in ['svg','png','pdf']:
  path=HERE/f'{name}.{ext}';f.savefig(path,dpi=180,bbox_inches='tight',pad_inches=.15);outputs.append(path)
 plt.close(f)
def canvas(h=6.5):
 f,a=plt.subplots(figsize=(12,h));f.subplots_adjust(left=.025,right=.975,top=.97,bottom=.04);a.set(xlim=(0,1),ylim=(0,1));a.axis('off');return f,a
def box(a,x,y,w,h,title,body='',color='pale',size=13):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.005,rounding_size=0.008',edgecolor=C['line'],facecolor=C[color]))
 a.text(x+w/2,y+h*(.67 if body else .5),title,ha='center',va='center',fontsize=size,weight='bold')
 if body:a.text(x+w/2,y+h*.27,body,ha='center',va='center',fontsize=11,color=C['muted'],linespacing=1.5)
def arrow(a,p,q,color='teal',rad=0):a.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=15,lw=1.7,color=C[color],connectionstyle=f'arc3,rad={rad}'))
# Diagrams contain labels only. Figure numbering and explanatory captions live in Markdown.
f,a=plt.subplots(figsize=(12,4.8));f.subplots_adjust(left=.19,right=.96,bottom=.21,top=.89)
for k in range(3):
 a.broken_barh([(10*k,9)],(2.7,.5),facecolors=C['blue']);a.text(10*k+4.5,2.95,f'模型调用 {k+1}',color='white',ha='center',va='center',fontsize=12)
 a.broken_barh([(10*k+9,1)],(1.6,.5),facecolors=C['orange']);a.text(10*k+9.5,2.26,'1 s',ha='center',fontsize=10)
a.broken_barh([(0,30)],(.5,.5),facecolors=C['teal']);a.text(15,.75,'2 GiB × 30 s = 60 GiB·s',color='white',ha='center',va='center',fontsize=14)
a.set(yticks=[.75,1.85,2.95],yticklabels=['环境内存','工具 CPU','模型服务'],xlim=(0,30),ylim=(0,3.6),xticks=[0,9,10,19,20,29,30],xlabel='时间 / s')
a.spines['left'].set_visible(False);a.tick_params(axis='y',length=0);a.grid(axis='x',alpha=.16)
a.text(.0,1.04,'三轮串行任务：完成 30 s；累计 CPU 时间 3 CPU·秒',transform=a.transAxes,fontsize=15,weight='bold')
save(f,'figure-11-1-timeline');data['11-1']={'kind':'teaching','rounds':3,'model_seconds_per_round':9,'tool_core_seconds_per_round':1,'resident_gib':2,'resident_gib_seconds':60}
# Model and tool paths: one relationship, implementation comparison stays in prose.
f,a=canvas(5.3)
box(a,.04,.58,.23,.23,'Agent 控制器','任务、历史与测试结果')
box(a,.60,.58,.29,.23,'模型服务','返回工具参数')
arrow(a,(.27,.73),(.60,.73));arrow(a,(.60,.64),(.27,.64));a.text(.435,.79,'模型请求',ha='center',fontsize=12)
box(a,.04,.13,.23,.23,'环境平台','准入与节点分配');arrow(a,(.155,.58),(.155,.36));a.text(.20,.46,'工具调用请求',fontsize=12)
a.add_patch(Rectangle((.55,.06),.39,.39,fill=False,ls='--',lw=1.4,edgecolor=C['teal']))
a.text(.745,.40,'执行节点',ha='center',fontsize=13,weight='bold');box(a,.60,.13,.29,.21,'工具环境','文件、进程、工具','green')
arrow(a,(.27,.28),(.60,.28));arrow(a,(.60,.18),(.27,.18));a.text(.435,.10,'工具结果',ha='center',fontsize=12)
save(f,'figure-11-2-boundary');data['11-2']={'kind':'mechanism','focus':'model versus tool execution paths','source':'case-studies/platform-routing.md'}

# Preparation lead versus exposed wait, correct-prediction branch only.
f,a=plt.subplots(figsize=(11,5));f.subplots_adjust(left=.20,right=.84,top=.88,bottom=.17)
rows=[('按需创建',4,6),('提前 1 s',3,5),('提前 2 s',2,4),('提前 3 s',1,3)]
for i,(label,start,end) in enumerate(rows):
 y=3-i;a.broken_barh([(start,2)],(y-.23,.46),facecolors=C['blue'])
 if end<4:a.broken_barh([(end,4-end)],(y-.23,.46),facecolors=C['orange'])
 a.text(6.15,y,['等待 2 s','等待 1 s','无需等待','多驻留 1 s'][i],va='center',fontsize=12)
a.axvline(4,color=C['red'],ls='--',lw=1.4);a.text(4,3.55,'调用到达',ha='center',fontsize=12,color=C['red'])
a.set(yticks=[3,2,1,0],yticklabels=[x[0] for x in rows],xticks=list(range(7)),xlim=(0,6),ylim=(-.5,3.9),xlabel='时间 / s');a.spines['left'].set_visible(False);a.tick_params(axis='y',length=0);a.grid(axis='x',alpha=.15)
a.plot([],[],color=C['blue'],lw=8,label='准备');a.plot([],[],color=C['orange'],lw=8,label='就绪未使用');a.legend(frameon=False,loc='upper left',ncol=2,fontsize=11)
save(f,'figure-11-3-lifecycle');data['11-3']={'kind':'teaching','focus':'preparation lead versus exposed wait','preparation_seconds':2,'call_time':4,'lead_seconds':[0,1,2,3],'existing_budget':lifecycle['prewarm_budget']}

# Resource before/after diagram.
f,a=canvas(6)
a.text(.5,.92,'新作业：4 张 H100 + 16 核 CPU，同一节点',ha='center',fontsize=16,weight='bold')
for x,title in [(.04,'整理前'),(.55,'整理后')]:
 a.text(x+.19,.79,title,ha='center',fontsize=14,weight='bold')
 if x<.5:
  box(a,x,.48,.37,.22,'节点 1 DGX H100：4 张 H100 空闲','仅 8 核可用 → 无法启动','sand')
  box(a,x,.15,.37,.22,'节点 2 DGX A100：4 张 A100 空闲','32 核可用 → 卡型不符','pale')
 else:
  box(a,x,.48,.37,.22,'节点 1 DGX H100：4 张 H100 空闲','16 核可用 → 整组启动','green')
  box(a,x,.15,.37,.22,'节点 2 DGX A100：4 张 A100 空闲','24 核可用 + 接收迁移任务','pale')
arrow(a,(.43,.57),(.53,.57));a.text(.48,.39,'迁移\n8 核任务',ha='center',fontsize=11)
a.text(.5,.055,'状态传输与恢复 4 s；等待释放 12 s；新作业本地执行 20 s',ha='center',fontsize=12,color=C['muted'])
save(f,'figure-11-4-placement');data['11-4']={'kind':'named_nodes','required_gpu_type':'H100 SXM','required_gpus':4,'required_cpu':16,'nodes':[{'system':'DGX H100','gpu':'H100 SXM','host_cores':112,'free_gpus':4},{'system':'DGX A100','gpu':'A100','host_cores':128,'free_gpus':4}],'before_free_cpu':[8,32],'after_free_cpu':[16,24],'sources':['references/text/dgx-superpod-h100-ra.txt','references/text/nvidia-a100.txt']}
# Same verification work, different submission times.
f,a=plt.subplots(figsize=(11,4.8));f.subplots_adjust(left=.17,right=.96,top=.93,bottom=.18);cols=[C['blue'],C['teal'],C['orange']]
for y,starts in [(1.3,[0,10,20]),(.3,[20,30,40])]:
 for k,start in enumerate(starts):
  a.broken_barh([(start,10)],(y,.5),facecolors=cols[k]);a.text(start+5,y+.25,f'样本 {k+1}',ha='center',va='center',color='white',fontsize=12)
for k,t in enumerate([0,10,20]):a.annotate(f'样本 {k+1} 到达',(t,2.05),(t,2.55),arrowprops={'arrowstyle':'-|>','color':C['muted']},ha='center',fontsize=11)
a.set(yticks=[.55,1.55],yticklabels=['整批后提交','逐条提交'],xlim=(-1,51),ylim=(0,2.9),xticks=[0,10,20,30,40,50],xlabel='时间 / s');a.spines['left'].set_visible(False);a.tick_params(axis='y',length=0);a.grid(axis='x',alpha=.18)
save(f,'figure-11-5-stages');data['11-5']={'kind':'teaching','focus':'submission time versus batch completion','model':'Qwen3-8B BF16','weight_bytes':16_381_470_720,'receivers':6,'sender_bits_per_second':200_000_000_000,'receiver_bits_per_second':50_000_000_000,'all_transfer_lower_seconds':6*16_381_470_720*8/200e9,'single_transfer_lower_seconds':16_381_470_720*8/50e9,'verification_arrivals':[0,10,20],'worker_service_seconds':10,'stream_completion_seconds':30,'batch_completion_seconds':50}

# Model service pathways.
f,a=canvas(6)
box(a,.02,.51,.17,.22,'任务控制器','质量与期限')
box(a,.30,.51,.21,.22,'服务入口','鉴权、路由、限流')
box(a,.69,.69,.27,.21,'外部模型 API','队列、缓存、生成','pale')
box(a,.69,.26,.27,.21,'自建模型副本','加载、队列、执行','green')
arrow(a,(.19,.62),(.30,.62));arrow(a,(.51,.67),(.69,.79));arrow(a,(.51,.55),(.69,.37))
a.text(.56,.83,'调用账单',ha='center',fontsize=11);a.text(.57,.35,'设备预留成本',ha='center',fontsize=11)
box(a,.15,.045,.48,.21,'记录每项任务及各次尝试的成本','普通输入 / 创建 / 读取 / 生成 / 其他成本','sand',12)
arrow(a,(.40,.51),(.40,.255));arrow(a,(.81,.26),(.63,.16))
a.text(.07,.86,'请求与结果沿各自路径往返',fontsize=12,color=C['muted'])
save(f,'figure-11-6-service');data['11-6']={'kind':'mechanism','usage_categories':['uncached_input','cache_creation','cache_read','billed_generation','storage','tools']}
# Quantitative curves derived from locked routing results.
A,B=route['routing_cost_rows'];h=np.linspace(0,1,201);qB=float(Fraction(B['expected_quality_successes_exact'])/route['scenario']['tasks']);hit=float(Fraction(B['hit_attempt_cost_exact']));miss=float(Fraction(B['miss_attempt_cost_exact']));ca=float(Fraction(A['cost_per_quality_success_exact']));cb=(h*hit+(1-h)*miss)/qB
cross=float(Fraction(route['summary']['cost_crossover_b_hit_fraction_exact']));target=float(Fraction(route['summary']['minimum_b_hit_for_joint_target_exact']))
f,a=plt.subplots(figsize=(10.5,5.3));f.subplots_adjust(left=.12,right=.96,bottom=.18,top=.92)
a.plot(h*100,cb,color=C['blue'],label='B Sonnet 5：命中率变化',lw=2.3);a.axhline(ca,color=C['orange'],label='A Haiku 4.5：前缀全命中，10 s 超过期限',lw=2)
a.axvline(cross*100,color=C['muted'],ls='--',lw=1);a.axvline(target*100,color=C['teal'],ls='--',lw=1)
a.axvspan(target*100,100,color=C['green'],alpha=.9,zorder=0)
a.annotate(f'成本相同：约 {cross*100:.1f}%',(cross*100,ca),(37,.031),arrowprops={'arrowstyle':'->','color':C['muted']},fontsize=12)
a.annotate(f'B 达到按时通过率要求：约 {target*100:.1f}%',(target*100,.007),(12,.006),arrowprops={'arrowstyle':'->','color':C['teal']},fontsize=12)
a.set(xlabel='B 请求命中率 / %',ylabel='美元 / 通过测试的任务',xlim=(0,100),ylim=(0,.046));a.legend(frameon=False,fontsize=11,loc='upper right');a.grid(alpha=.15)
save(f,'figure-11-7-routing');data['11-7']={'kind':'teaching_from_locked_calculation','source':'calculations/results/routing-cost-book.json','focus':'cost comparison constrained by deadline','h':h.tolist(),'cost_A':ca,'cost_B':cb.tolist(),'cost_crossover':cross,'joint_target_hit':target}

# Recovery improves completion fraction at an additional cost.
rs=retry['summary'];initial=.01/.8;qc=float(Fraction(rs['cost_per_quality_success_exact']));dc=float(Fraction(rs['cost_per_quality_and_deadline_success_exact']))
f,a=plt.subplots(figsize=(10,5.3));f.subplots_adjust(left=.12,right=.96,bottom=.26,top=.86)
xx=np.array([0,1.4,2.4]);vals=[initial,qc,dc];a.bar(xx,vals,color=[C['blue'],C['teal'],C['orange']],width=.55)
for i,v in enumerate(vals):a.text(xx[i],v+.0003,f'{v:.4f}',ha='center',fontsize=13)
a.set(xticks=xx,xticklabels=['仅首次尝试\n成功率 80%','有限恢复\n通过测试约 99.7%','有限恢复\n按时通过测试约 95.0%'],ylim=(0,.018),ylabel='全部支出 / 符合条件的任务数');a.grid(axis='y',alpha=.15)
a.text(1.9,.0173,'同一有限恢复策略，分别统计成功与按时成功',ha='center',fontsize=11)
parts=[.8*.01+.072*.016+.0784*.04,.04704*.046,.00096*.046+.0016*.04]
save(f,'figure-11-8-retry');data['11-8']={'kind':'teaching_from_locked_calculation','source':'calculations/results/retry-paths-book.json','focus':'recovery cost versus completion criterion','policy_costs':[initial,qc,dc],'contributions':dict(zip(['按时成功路径','超时成功路径','失败路径'],parts)),'summary':rs}

from extra_figures import draw
draw(save,C,data,canvas,box,arrow,json.loads((HERE/'platform-design.json').read_text()),lifecycle)

import sys
sys.path.insert(0,str(HERE.parent))
from teaching_revision import draw as draw_teaching
teaching_outputs,teaching_checks=draw_teaching(HERE,data)
outputs=list(dict.fromkeys(outputs+teaching_outputs))
(HERE/'figure-data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
(HERE/'figure-layout-check.json').write_text(json.dumps({'text_extent_warnings':warnings},ensure_ascii=False,indent=2)+'\n')

# Offline HTML; KaTeX and its fonts are vendored with the chapter.
md=HERE.parent/'11-资源调度与运行环境.md';raw=md.read_text();maths=[]
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
for p in outputs:
 if p.suffix=='.svg':body=body.replace('src="ch11/'+p.name+'"','src="data:image/png;base64,'+base64.b64encode(p.with_suffix('.png').read_bytes()).decode()+'"')
body=re.sub(r'<p>(<img [^>]+>)</p>\s*<p><em>(图 11-[\s\S]*?)</em></p>',r'<figure>\1<figcaption>\2</figcaption></figure>',body)
css='''figure{margin:32px 0}figcaption{font:15px/1.8 Arial,"PingFang SC",sans-serif;color:#506975;padding:8px 0}body{margin:0;background:#fafaf8;color:#243640;font:18px/1.95 Georgia,"Songti SC",serif}main{max-width:960px;margin:auto;padding:50px 38px 90px;background:white}h1,h2,h3{font-family:Arial,"PingFang SC",sans-serif;line-height:1.45;color:#163747}h1{font-size:36px}h2{font-size:28px;border-top:1px solid #d5e1e4;margin-top:65px;padding-top:28px}h3{font-size:23px;margin-top:40px}p{margin:1em 0}a{color:#246f91;text-underline-offset:3px}img{display:block;width:100%;height:auto;margin:26px auto 10px}em{font-size:15px;color:#55707d}table{border-collapse:collapse;width:100%;font-size:15px;line-height:1.7;margin:24px 0}td,th{padding:10px 12px;border-bottom:1px solid #d5e1e4;text-align:left}th{background:#edf4f6}blockquote{margin:28px 0;padding:16px 24px;border-left:4px solid #138b83;background:#f1f8f6;font-size:16px}code{font:0.85em/1.65 Menlo,monospace;background:#f0f4f6;overflow-wrap:anywhere}pre{white-space:pre;overflow-x:auto;padding:16px;border:1px solid #d5e1e4;max-width:100%;box-sizing:border-box}nav{font:16px/1.9 Arial,"PingFang SC",sans-serif;background:#f0f6f8;padding:18px 24px}nav a{display:block}.footnote{font-size:14px;line-height:1.8}.footnote li{margin-bottom:12px}.table-scroll{overflow-x:auto;max-width:100%}.katex{font-size:1.04em}.katex-display{overflow-x:auto;overflow-y:hidden;padding:12px 0}@media(max-width:650px){main{padding:25px 18px}body{font-size:17px}h1{font-size:29px}h2{font-size:25px}}@media print{main{max-width:none;padding:0}h2,h3{break-after:avoid}img,blockquote{break-inside:avoid}body{font-size:11pt}}'''
css+='*{box-sizing:border-box}main{max-width:760px;padding-left:24px;padding-right:24px}img{max-width:720px}a{overflow-wrap:anywhere}@media print{img{width:420pt;max-width:100%}}'
nav=''.join('<a href="#'+ident+'">'+title+'</a>' for ident,title in re.findall(r'<h2 id="([^"]+)">(11\.\d+ [^<]+)</h2>',body))
page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第 11 章 资源调度与运行环境</title><style>'+css+math_css+'</style><main><nav>'+nav+'</nav>'+body+'</main></html>'
import sys
sys.path.insert(0,str(HERE.parent))
from teaching_reading import readable_diagrams
from preview_output import preview_path
page=readable_diagrams(page)
html_path=preview_path(HERE.parent/'11-资源调度与运行环境.html');html_path.write_text(page)
(HERE/'math-validation.json').write_text(json.dumps({'renderer':'KaTeX 0.16.11','expressions':len(maths),'display_expressions':sum(x['display'] for x in maths),'errors':[]},indent=2)+'\n')
from book_assets import sync_figure_index
active_assets=sync_figure_index(HERE)
artifacts=outputs+[HERE/'teaching_revision.py',HERE/'h2_cloudnative_finops_figures.py',HERE/'figure-index.json',HERE/'teaching-layout-validation.json',HERE/'figure-data.json',HERE/'platform-design.json',HERE/'platform_design.py',HERE/'extra_figures.py',HERE/'figure-order.json',html_path,md]+active_assets
(HERE/'manifest.json').write_text(json.dumps({'chapter':11,'generator':'manuscripts/ch11/build.py','figures':len(teaching_checks),'font_family':family,'outputs':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in artifacts if not (p.parent==ROOT/'manuscripts' and re.match(r'^[01][0-9]-',p.name))]},ensure_ascii=False,indent=2)+'\n')
print(f'Built {len(outputs)} figure files, {len(maths)} equations and offline HTML; {len(warnings)} layout warnings.')
