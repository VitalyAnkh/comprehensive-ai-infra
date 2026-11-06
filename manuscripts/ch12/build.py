#!/usr/bin/env python3
"""Build source-linked figures and a self-contained chapter 12 reading edition."""
from pathlib import Path
from fractions import Fraction
import argparse, base64, hashlib, html, json, re, subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import markdown
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--font');args=parser.parse_args()
import sys
sys.path.insert(0,str(HERE.parent))
from figure_style.typography import configure_font
font,family=configure_font(args.font)
plt.rcParams.update({'font.family':[family,'DejaVu Sans'],'font.size':11,'axes.unicode_minus':False,'svg.fonttype':'path','svg.hashsalt':'ch12-edge-v1','axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white','text.color':'#203c48','axes.labelcolor':'#203c48','pdf.fonttype':42})
C={'ink':'#203c48','blue':'#286b98','teal':'#16857b','orange':'#bc722b','red':'#a94c52','pale':'#eef4f7','light':'#eaf5f1','sand':'#fbf0e5','line':'#c3d0d7','muted':'#546e7a'}
for x in json.loads((HERE/'sources.json').read_text())['sources']:
 b=(ROOT/x['path']).read_bytes()
 h=b.decode().split('oid sha256:')[1].splitlines()[0].strip() if b.startswith(b'version https://git-lfs.github.com/spec/v1') else hashlib.sha256(b).hexdigest()
 if h!=x['sha256']:raise SystemExit('Review changed source: '+x['path'])
def calc(name):return json.loads((ROOT/'calculations/results'/f'{name}.json').read_text())
def num(v):
 if isinstance(v,dict) and 'numerator' in v:return v['numerator']/v['denominator']
 return float(Fraction(str(v)))
catalog=json.loads((HERE/'figure-catalog.json').read_text());number_map={x['source_id']:x['number'] for x in catalog}
outputs=[];data={};layout=[]
def save(f,name):
 name=re.sub(r'^figure-12-(\d+)-',lambda m:f"figure-12-{number_map[int(m[1])]}-",name)
 if len(f.axes)!=1:raise ValueError('Each main figure must use one axes')
 f.canvas.draw();renderer=f.canvas.get_renderer()
 for t in f.findobj(matplotlib.text.Text):
  if not t.get_visible() or not t.get_text():continue
  if re.search(r'图\s*\d+\s*[-－–]\s*\d+',t.get_text()):raise ValueError('Figure number inside artwork')
  b=t.get_window_extent(renderer)
  # Include only painted tick labels within the active coordinate range.
  if b.x0<0 or b.y0<0 or b.x1>f.bbox.width or b.y1>f.bbox.height:layout.append({'figure':name,'text':t.get_text(),'bbox':[round(v,2) for v in b.bounds]})
 for ext in ['svg','png','pdf']:
  p=HERE/(name+'.'+ext);f.savefig(p,dpi=180,bbox_inches='tight',pad_inches=.15);outputs.append(p)
 plt.close(f)
def canvas(height=8):
 f,a=plt.subplots(figsize=(13,height));f.subplots_adjust(left=.025,right=.975,top=.97,bottom=.04);a.set(xlim=(0,1),ylim=(0,1));a.axis('off');return f,a
def box(a,x,y,w,h,title,body='',col='pale',size=12):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.004,rounding_size=0.008',ec=C['line'],fc=C[col],lw=1.1))
 a.text(x+w/2,y+h*(.70 if body else .50),title,ha='center',va='center',fontsize=size,weight='bold')
 if body:a.text(x+w/2,y+h*.27,body,ha='center',va='center',fontsize=10,linespacing=1.4,color=C['muted'])
def arrow(a,p,q,col='teal',rad=0,lw=1.6):a.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=13,lw=lw,color=C[col],connectionstyle=f'arc3,rad={rad}'))
def panel(a,x,y,t):a.text(x,y,t,fontsize=14,weight='bold',va='top')
# Each main figure has one axes and one quantitative relationship.
def plot_canvas():
 f,a=plt.subplots(figsize=(10,5.5));f.subplots_adjust(left=.18,right=.95,bottom=.18,top=.88);return f,a
vals=np.array([[12,.1,.3,.4,0],[12,.1,.03,.4,0],[6,.1,.3,.4,.15]])
bw=np.geomspace(5,200,160);f,a=plot_canvas()
for label,y,c in [('原方案',240/bw+.8,'blue'),('处理十倍加速',240/bw+.53,'teal'),('输入减半，新增编解码',120/bw+.95,'red')]:a.plot(bw,y,color=C[c],lw=2,label=label)
a.set(xscale='log',xlabel='上行速率 / Mbit/s',ylabel='完整成片时间 / s',ylim=(0,51));a.set_xticks([5,10,20,50,100,200],['5','10','20','50','100','200']);a.set_yticks([0,10,20,30,40,50]);a.grid(alpha=.18);a.legend(frameon=False)
save(f,'figure-12-1-raw');data['12-1']={'kind':'teaching','input_MB':30,'output_MB':5,'down_Mbps':100,'rtt_s':.1,'stages_s':vals.tolist(),'uplink_Mbps':bw.tolist(),'base_s':(240/bw+.8).tolist(),'compressed_s':(120/bw+.95).tolist()}
chunks=calc('audio-timing-base')['audio_chunks'];f,a=plot_canvas()
for i,z in enumerate(chunks[:5]):
 y=4-i;a.broken_barh([(z['deadline_ns']/1e6,20)],(y-.30,.60),facecolors='none',edgecolors=C['line'],linewidth=2,label='原定播放' if i==0 else None)
 a.broken_barh([(z['playback_start_ns']/1e6,20)],(y-.13,.26),facecolors=C['teal'],label='实际播放' if i==0 else None)
 a.scatter(z['arrival_ns']/1e6,y,color=C['blue'],s=35,zorder=3,label='到达' if i==0 else None)
a.annotate('晚到 5 ms',(123,2),(140,2.6),arrowprops={'arrowstyle':'->','color':C['muted']},fontsize=11)
a.set(yticks=range(5),yticklabels=['块 5','块 4','块 3','块 2','块 1'],xlabel='从开始采集起的时间 / ms',xlim=(25,185),ylim=(-.6,4.7));a.set_xticks([40,60,80,100,120,140,160,180]);a.grid(axis='x',alpha=.18);a.legend(frameon=False,ncol=3,loc='lower left',bbox_to_anchor=(0,1.01))
save(f,'figure-12-2-paths');data['12-2']={'kind':'dependency_diagram_and_saved_teaching_events','audio_source':'calculations/results/audio-timing-base.json','audio_chunks':chunks}
ec=calc('multimodal-cache-single')['summary'];sizes=[800000,ec['complete_encoder_bytes_per_image'],ec['visual_kv_bytes_per_image']]
f,a=plot_canvas();times=np.array(sizes[:2])*8/6.4e6
a.barh([1,0],times,color=[C['blue'],C['teal']],height=.45)
for y,v in zip([1,0],times):a.text(v+.15,y,f'{v:.1f} s',va='center')
a.set(yticks=[1,0],yticklabels=['压缩图片','完整视觉特征'],xlabel='理想发送时间 / s',xlim=(0,12),ylim=(-.65,1.65));a.grid(axis='x',alpha=.15)
save(f,'figure-12-3-placement');data['12-3']={'kind':'fixed_model_shape_and_teaching_input','source':'calculations/results/multimodal-cache-single.json','bytes':sizes,'uplink_seconds_at_6_4Mbps':[x*8/6.4e6 for x in sizes],'complete_shape':[400,10240]}
comp={}
for files,title,labels in [(['shared-media-schedule-fifo','shared-media-schedule-priority'],'只改发送顺序',['FIFO','音频优先']),(['shared-media-hol-connection','shared-media-hol-per_stream'],'固定发送，只改交付顺序',['整体有序','逐流交付'])]:
 ys=[]
 for name in files:
  objects={z['id']:num(z['complete']) for z in calc(name)['businesses']};ys.append([objects['image'],objects['audio']])
 comp[title]={'sources':files,'rows':labels,'image_audio_seconds':ys}
f,a=plot_canvas();x=np.arange(2);v=np.array(comp['固定发送，只改交付顺序']['image_audio_seconds'])
for j,(lab,c) in enumerate([('图片','blue'),('音频','orange')]):
 bars=a.barh(x+(j-.5)*.30,v[:,j],height=.28,color=C[c],label=lab)
 for bar in bars:a.text(bar.get_width()+.1,bar.get_y()+.14,f'{bar.get_width():g} s',va='center')
a.set(yticks=x,yticklabels=['整体有序','逐流交付'],xlim=(0,8.5),xlabel='应用交付完成时间 / s');a.invert_yaxis();a.legend(frameon=False,ncol=2,loc='lower left',bbox_to_anchor=(0,1.01));a.grid(axis='x',alpha=.15)
save(f,'figure-12-4-transport');data['12-4']={'kind':'two_independent_saved_teaching_contrasts','comparisons':comp}
if (ROOT/'calculations/results/shared-airtime-image-baseline.json').read_bytes().startswith(b'version https://git-lfs'):
 services=json.loads((HERE/'figure-data.json').read_text())['12-11']['exchange_components_us']
else:
 air=calc('shared-airtime-image-baseline');services=[]
 for kind in ['data','ack']:
  z=next(x for x in air['wireless_attempts'] if x['kind']==kind)['service'];parts=[num(z['access_idle_seconds']),num(z['data_ppdu']['duration']),num(z['mac_ack_start_offset'])-num(z['data_end_offset']),num(z['mac_ack_ppdu']['duration'])];services.append([v*1e6 for v in parts])
 del air
f,a=plot_canvas();left=np.zeros(2)
for j,(lab,c) in enumerate([('接入等待','muted'),('数据 PPDU','blue'),('SIFS','orange'),('MAC ACK','teal')]):
 v=np.array(services)[:,j];a.barh([1,0],v,left=left,height=.45,color=C[c],label=lab)
 for y,w,l in zip([1,0],v,left):a.text(l+w/2,y,f'{w:g}',ha='center',va='center',fontsize=10,color='white')
 left+=v
for y,v in zip([1,0],left):a.text(v+5,y,f'{v:g} μs',va='center')
a.set(yticks=[1,0],yticklabels=['承载数据','承载端到端 ACK'],xlim=(0,350),ylim=(-.6,1.6),xlabel='一次成功交换的空口时间 / μs');a.legend(frameon=False,ncol=4,fontsize=10,loc='lower left',bbox_to_anchor=(-.08,1.01))
save(f,'figure-12-5-wireless');data['12-5']={'kind':'reference_airtime_and_separate_loopback_measurement','exchange_components_us':services,'components':['access_idle','data_ppdu','sifs','mac_ack'],'measured_each_strategy_successes':[5,5,0],'trials_each':5,'physical_wireless_measurement':False}
q=calc('queqiao-records-conditions');fixed=[x for x in q['same_condition_comparisons'] if x['condition'][:2]==['single-fixed-file','asr_upload']]
matched={x['condition'][2]:x for x in fixed};warm=matched['held_open'];cold=matched['new'];med=np.array([[num(cold['numerator_ms']),num(cold['denominator_ms'])],[num(warm['numerator_ms']),num(warm['denominator_ms'])]])
phys=(ROOT/'experiments/ch12/12-07/audio-physical/README.md').read_text();rows=re.findall(r'^\|([0-3])／(on|off)\|([\d.]+)\|([\d.]+)\|(\d+)\|(\d+)\|$',phys,re.M);assert len(rows)==4
starvation=np.array([[int(r[4]),int(r[5])] for r in rows]);f,a=plot_canvas();x=np.arange(2)
for j,(label,c) in enumerate([('直接路径','blue'),('Queqiao','teal')]):
 bars=a.bar(x+(j-.5)*.32,med[:,j]/1000,width=.29,color=C[c],label=label)
 for bar in bars:a.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.035,f'{bar.get_height():.2f} s',ha='center')
a.set(xticks=x,xticklabels=['新建连接','保持连接并调优'],ylabel='请求完成时间中位数 / s',ylim=(0,1.4));a.legend(frameon=False);a.grid(axis='y',alpha=.15)
save(f,'figure-12-6-queqiao');data['12-6']={'kind':'two_separate_recorded_experiments','matched_ASR_ms':med.tolist(),'recorded_fixed_bytes':q['generations']['single-fixed-file']['payload_low_bytes'],'starvation_ms':starvation.tolist(),'physical_rows':rows,'source_documents':['calculations/results/queqiao-records-conditions.json','experiments/ch12/12-07/audio-physical/README.md']}
edge=calc('edge-tiers-agent-book');tiers={x['tier']:x for x in edge['tiers']};up=edge['uplink'];base=edge['cases'][0]
b=np.array(up['scan_bits_per_second'])/1e6;cloud=np.array(up['scan_cloud_seconds']);near_s=up['near_total_seconds'];deadline=base['deadline_seconds'];feasible=up['cloud_deadline_uplink_bits_per_second']/1e6;f,a=plot_canvas()
a.plot(b,cloud,color=C['blue'],lw=2,label='云地域');a.axhline(near_s,color=C['teal'],lw=2,label='附近工作站');a.axhline(deadline,color=C['red'],ls='--',label='任务期限');a.scatter([feasible],[deadline],s=40,color=C['orange'],zorder=3)
a.annotate(f'满足期限：约 {feasible:.1f} Mbit/s',(feasible,deadline),(6,58),fontsize=11,arrowprops={'arrowstyle':'->','color':C['muted']})
a.set(xlabel='云路径上行速率 / Mbit/s',ylabel='剩余 20 轮完成时间 / s',xlim=(2,20),ylim=(0,80));a.legend(frameon=False,loc='upper right');a.grid(alpha=.16)
save(f,'figure-12-7-deployment');data['12-7']={'kind':'device_bound_deployment','source':'calculations/results/edge-tiers-agent-book.json','rounds':base['rounds'],'image_MB':.8,'local_other_s':tiers['end']['terminal_seconds'],'devices':[tiers[k]['device_name'] for k in ('end','near','cloud')],'model_s':[tiers[k]['model_seconds_per_round'] for k in ('end','near','cloud')],'prepare_s':[tiers[k]['prepare_seconds'] for k in ('end','near','cloud')],'RTT_s':[tiers[k]['rtt_seconds'] for k in ('end','near','cloud')],'uplink_Mbps':[None,80,6.4],'task_seconds':[r['total_seconds'] for r in base['rows']],'energy_joules':[r['energy_joules'] for r in base['rows']],'deadline_s':deadline,'feasible_threshold_Mbps':feasible,'cloud_fixed_s':up['cloud_fixed_seconds'],'near_s':near_s,'scan_Mbps':b.tolist(),'cloud_seconds':cloud.tolist()}
# Raw supporting values remain available, but are not additional figure panels.
relationships=['上行速率与完整成片时间','到达延迟与按序播放','数据表示与发送时间','交付依赖与完成时间','固定开销与空口交换时间','连接基线与请求时间差','上行速率与部署时间边界']
plotted=['three_latency_curves','first_five_audio_chunks','image_and_complete_ec_only','delivery_order_only','exchange_components_only','matched_ASR_only','latency_thresholds_only']
for i,(relationship,subset) in enumerate(zip(relationships,plotted),1):
 data[f'12-{i}']['relationship']=relationship;data[f'12-{i}']['plotted_subset']=subset;data[f'12-{i}']['axes_count']=1;data[f'12-{i}']['kind']='single_relationship';data[f'12-{i}']['supporting_values_note']='Only plotted_subset is drawn; other saved values support prose and evidence notes.'
exec(compile((HERE/'draw-concepts.py').read_text(),str(HERE/'draw-concepts.py'),'exec'),globals())
data={f'12-{number_map[int(k.split("-")[1])]}':v for k,v in data.items()}
import sys
sys.path.insert(0,str(HERE.parent))
from teaching_revision import draw as draw_teaching
teaching_outputs,teaching_checks=draw_teaching(HERE,data)
outputs=list(dict.fromkeys(outputs+teaching_outputs))
(HERE/'figure-data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
(HERE/'figure-layout-check.json').write_text(json.dumps({'text_extent_warnings':layout},ensure_ascii=False,indent=2)+'\n')
# HTML contains images, mathematical markup and font bytes; source links stay repository-relative.
md=HERE.parent/'12-端边云协同.md';raw=md.read_text();maths=[]
def protect(match):
 text=match[0];display=text.startswith('$$');latex=text[2:-2] if display else text[1:-1];token=f'MATHPLACEHOLDER{len(maths)}END';maths.append({'latex':latex.strip(),'display':display,'token':token});return '\n\n'+token+'\n\n' if display else token
protected=re.sub(r'\$\$[\s\S]*?\$\$|\$[^$\n]+\$',protect,raw)
body=markdown.markdown(protected,extensions=['tables','footnotes','fenced_code','toc'],output_format='html')
katex=ROOT/'manuscripts/ch06/vendor/katex'
node="const fs=require('fs'),k=require(process.argv[1]);const a=JSON.parse(fs.readFileSync(0,'utf8'));process.stdout.write(JSON.stringify(a.map(x=>k.renderToString(x.latex,{displayMode:x.display,throwOnError:true,output:'htmlAndMathml'}))))"
res=subprocess.run(['node','-e',node,str(katex/'katex.js')],input=json.dumps(maths),text=True,capture_output=True)
if res.returncode:raise SystemExit(res.stderr)
for entry,rendered in zip(maths,json.loads(res.stdout)):
 body=body.replace('<p>'+entry['token']+'</p>',rendered) if entry['display'] else body.replace(entry['token'],rendered)
math_css=(katex/'katex.min.css').read_text()
def font_url(m):
 p=katex/m[1];return 'url(data:font/'+p.suffix[1:]+';base64,'+base64.b64encode(p.read_bytes()).decode()+')'
math_css=re.sub(r'url\((fonts/[^)]+)\)',font_url,math_css)
body=re.sub(r'<table>(.*?)</table>',r'<div class="table-scroll"><table>\1</table></div>',body,flags=re.S)
body=re.sub(r'<p>(<img [^>]+>)</p>\s*<p><em>(图 12-[\s\S]*?)</em></p>',r'<figure>\1<figcaption>\2</figcaption></figure>',body)
from book_assets import sync_figure_index
active_assets=sync_figure_index(HERE)
for p in sorted(HERE.glob('*.svg')):
 if p.with_suffix('.png').exists():body=body.replace('src="ch12/'+p.name+'"','src="data:image/png;base64,'+base64.b64encode(p.with_suffix('.png').read_bytes()).decode()+'"')
css='''*{box-sizing:border-box}body{margin:0;background:#f7f7f4;color:#243640;font:18px/1.95 Georgia,"Songti SC",serif}main{max-width:1020px;margin:auto;padding:48px 46px 85px;background:#fff}h1,h2,h3{font-family:Arial,"PingFang SC",sans-serif;line-height:1.45;color:#183949}h1{font-size:36px}h2{font-size:28px;border-top:1px solid #d5e1e4;margin-top:65px;padding-top:26px}h3{font-size:23px;margin-top:38px}a{color:#286b98;text-underline-offset:3px;overflow-wrap:anywhere}figure{margin:30px 0}img{display:block;width:100%;height:auto;margin:0 auto 10px}figcaption{font:15px/1.85 Arial,"PingFang SC",sans-serif;color:#546e7a}table{border-collapse:collapse;width:100%;font-size:15px;line-height:1.7;margin:22px 0}td,th{padding:10px 12px;border-bottom:1px solid #d5e1e4;text-align:left}th{background:#edf4f6}code{font:0.85em/1.6 Menlo,monospace;background:#f0f4f6;overflow-wrap:anywhere}pre{overflow-x:auto;padding:16px;max-width:100%}nav{font:16px/1.9 Arial,"PingFang SC",sans-serif;background:#eff5f7;padding:18px 24px}nav a{display:block}.footnote{font-size:14px;line-height:1.8}.footnote li{margin-bottom:13px}.table-scroll{overflow-x:auto;max-width:100%}.katex{font-size:1.04em}.katex-display{overflow-x:auto;overflow-y:hidden;padding:12px 0}@media(max-width:650px){main{padding:24px 18px}body{font-size:17px}h1{font-size:29px}h2{font-size:25px}h3{font-size:21px}}@media print{main{max-width:none;padding:0}h2,h3{break-after:avoid}figure{break-inside:avoid}body{font-size:11pt}}'''
css+='figure img{cursor:zoom-in}figure img:focus{outline:2px solid #286b98}dialog{width:96vw;height:92vh;max-width:none;max-height:none;border:0;padding:18px;background:white}dialog::backdrop{background:#102a3bcc}.viewer-scroll{overflow:auto;height:calc(100% - 45px)}.viewer-scroll img{width:1100px;max-width:none;margin:0}.viewer-close{display:block;margin:0 0 10px auto;padding:7px 18px;font-size:16px} @media print{dialog{display:none}}'
viewer='<dialog id="figure-viewer" aria-label="放大插图"><button class="viewer-close" type="button">关闭</button><div class="viewer-scroll"></div></dialog><script>const viewer=document.getElementById("figure-viewer"),area=viewer.querySelector(".viewer-scroll");function openFigure(img){const copy=img.cloneNode();copy.removeAttribute("tabindex");copy.removeAttribute("role");area.replaceChildren(copy);viewer.showModal()}document.querySelectorAll("figure img").forEach(img=>{img.tabIndex=0;img.setAttribute("role","button");img.setAttribute("aria-label","放大插图："+img.alt);img.addEventListener("click",()=>openFigure(img));img.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();openFigure(img)}})});viewer.querySelector("button").onclick=()=>viewer.close();viewer.addEventListener("close",()=>area.replaceChildren());</script>'
css+='main{max-width:760px;padding-left:24px;padding-right:24px}img{max-width:720px}@media print{img{width:420pt;max-width:100%}}'
nav=''.join('<a href="#'+ident+'">'+title+'</a>' for ident,title in re.findall(r'<h2 id="([^"]+)">([^<]+)</h2>',body))
page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第 12 章 端边云协同</title><style>'+css+math_css+'</style></head><body><main><nav aria-label="本章目录">'+nav+'</nav>'+body+'</main>'+viewer+'</body></html>'
import sys
sys.path.insert(0,str(HERE.parent))
from teaching_reading import readable_diagrams
from preview_output import preview_path
page=readable_diagrams(page)
hp=preview_path(md);hp.write_text(page)
(HERE/'math-validation.json').write_text(json.dumps({'renderer':'KaTeX 0.16.11','expressions':len(maths),'display_expressions':sum(x['display'] for x in maths),'errors':[]},indent=2)+'\n')
from book_assets import sync_figure_index
active_assets=sync_figure_index(HERE)
artifacts=outputs+[HERE/'teaching_revision.py',HERE/'figure-index.json',HERE/'teaching-layout-validation.json',HERE/'figure-data.json',HERE/'figure-catalog.json',hp,md]+active_assets
(HERE/'manifest.json').write_text(json.dumps({'chapter':12,'generator':'manuscripts/ch12/build.py','figures':len(teaching_checks),'font_family':family,'outputs':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in artifacts if not (p.parent==ROOT/'manuscripts' and re.match(r'^[01][0-9]-',p.name))]},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'figures':len(teaching_checks),'maths':len(maths),'layout_warnings':len(layout),'html':str(hp)},ensure_ascii=False))
