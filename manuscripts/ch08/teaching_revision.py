"""Request iterations, physical KV blocks and service outcomes."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from figure_style import COL,STYLE,canvas,plot,text,box,arrow,Exporter

def draw(here,data):
    out=Exporter(here)
    def save(f,n):out.save(f,'figure-8-'+n)
    with plt.rc_context(STYLE):
        f,a=canvas(4.7);text(a,.04,.94,'请求保持身份，batch 每轮重新组成',14)
        for row,labels in enumerate([['请求 A','请求 B'],['请求 A','请求 C']]):
            y=.60-row*.36;text(a,.04,y+.17,f'迭代 {row}',12)
            for i,label in enumerate(labels):box(a,.25+i*.37,y,.30,.20,label,'blue' if label=='请求 A' else 'green',12)
        text(a,.5,.08,'B 已结束 → 空出的执行位置交给 C',12,ha='center');save(f,'request-iterations')
        f,a=plot(3.0,left=.12);start=0;life=data['lifecycle']
        for dur,l,c in [(life['queue_s'],'排队','gray'),(life['prefill_s'],'prefill','blue'),(life['output_intervals']*life['interval_s'],'decode','green')]:a.barh(0,dur,left=start,height=.35,color=COL[c],edgecolor=COL['line'],label=l);start+=dur
        a.scatter([life['queue_s']+life['prefill_s'],life['completion_s']],[0,0],color='#252525',zorder=5);a.axvline(life['deadline_s'],ls='--',color='#a56c28');a.set(yticks=[],xlim=(0,7.4),xlabel='从到达起计时（s）');f.subplots_adjust(top=.78);a.legend(ncol=3,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.28),columnspacing=1.5,handlelength=1.4);save(f,'1-lifecycle')
        f,a=plot(3.7);b=np.array(data['batch']['batch']);w=data['batch']['shared_weight_bytes']/2**30/b
        for L,c in [(2048,'#267398'),(8192,'#388768')]:a.plot(b,w+L*144/2**20,color=c,label=f'上下文 {L} token：总读取量');a.axhline(L*144/2**20,color=c,ls=':')
        a.plot(b,w,ls='--',color='#777777',label='每个输出 token 分摊的权重读取量');a.set(xscale='log',yscale='log',xlabel='批内请求数',ylabel='每生成一个 token 的读取量（GiB）');a.legend(frameon=False,fontsize=11);save(f,'2-batch')
        for i,key in enumerate(['fixed','continuous','chunked']):
            f,a=plot(3.8,left=.16)
            for step in data[key]:
                for plan in step['plans']:
                    lane=int(plan['request'][1:]);a.barh(lane,step['duration_ns']/1e6,left=step['start_ns']/1e6,height=.55,color=COL['blue' if plan['phase']=='prefill' else 'green'],edgecolor=COL['line'],linewidth=.45)
            a.scatter([0,0,20,30],range(4),marker='^',color='#252525',s=18,zorder=5);a.set(yticks=range(4),yticklabels=[f'请求 r{j}' for j in range(4)],xlim=(0,900),xlabel='时间（ms）');a.invert_yaxis();save(f,'3-scheduling' if i==0 else f'scheduling-{key}')
        for i,d in enumerate(data['attention_geometry']):
            f,a=canvas(3.5);h=d['history'];step=.065;start=.15
            for row in range(4):
                for col in range(h+4):box(a,start+col*step,.65-row*.13,step-.008,.105,'','blue' if col<h else ('green' if col-h<=row else 'white'))
            text(a,.04,.91,f'旧上下文 {h} token + 新输入 4 token',14);text(a,.5,.12,f'旧上下文配对 {4*h} + 块内配对 10 = {4*h+10}',12,ha='center');save(f,'4-attention' if i==0 else 'attention-history')
        f,a=canvas(4.5);text(a,.04,.94,'逻辑块按序排列，物理块分散存放',14)
        for i,target in enumerate([2,0,3]):
            x=.06+i*.31;box(a,x,.68,.24,.13,f'逻辑块 {i}','blue',11);text(a,x+.12,.54,f'块表：{i} → {target}',11,ha='center');arrow(a,(x+.12,.46),(.14+target*.235,.31))
        for i in range(4):box(a,.04+i*.235,.12,.20,.18,f'物理块 {i}','green' if i in [0,2,3] else 'gray',11)
        save(f,'page-map')
        for mode,name in [('reserved','5-pages'),('paged','pages-paged')]:
            f,a=canvas(4.6)
            for row,L in enumerate(data['pages']['lengths']):
                y=.70-row*.18;text(a,.03,y+.04,f'{"ABCD"[row]}：{L}',11)
                slots=16 if mode=='reserved' else (L+3)//4*4
                for j in range(slots):box(a,.19+j*.047,y,.039,.10,'','blue' if j<L else 'gray')
            text(a,.5,.05,f'KV 分配容量：{64 if mode=="reserved" else 52} token',13,ha='center');save(f,name)
        f,a=canvas(4.4);box(a,.04,.67,.27,.15,'A 的块表','blue');box(a,.69,.67,.27,.15,'B 的块表','green');box(a,.31,.34,.38,.16,'两个共同前缀块\n各有两个引用','purple',11)
        arrow(a,(.175,.67),(.40,.50));arrow(a,(.825,.67),(.60,.50));box(a,.03,.07,.28,.13,'A 私有后缀','blue',11);box(a,.69,.07,.28,.13,'B 私有后缀','green',11);arrow(a,(.175,.67),(.175,.20));arrow(a,(.825,.67),(.825,.20));save(f,'pages-shared')
        f,a=canvas(4.3);box(a,.25,.70,.50,.16,'共享尾块：[a, b, c, 空]','purple',12)
        for x,label,c in [(.03,'分支 A：[a, b, c, x]','blue'),(.54,'分支 B：[a, b, c, y]','green')]:box(a,x,.20,.43,.19,label,c,11);arrow(a,(.5,.70),(x+.215,.39))
        text(a,.5,.52,'写入分歧前，取得各自的尾块副本',12,ha='center');save(f,'copy-on-write')
        f,a=canvas(4.3)
        for row,(label,count,col) in enumerate([('A、B 使用共同块',2,'purple'),('A 结束，B 继续使用',1,'green'),('B 结束，加速器已不再访问',0,'gray')]):
            y=.69-row*.27;box(a,.04,y,.61,.17,label,col,11);box(a,.75,y,.21,.17,f'引用 {count}',col,11)
            if row<2:arrow(a,(.86,y),(.86,y-.10))
        save(f,'reference-release')
        f,a=canvas(4.2)
        for x,l,c in [(.03,'收到取消\n停止后续迭代','orange'),(.37,'等待已提交\n加速器操作完成','blue'),(.71,'释放私有块\n更新共享引用','green')]:box(a,x,.41,.26,.27,l,c,11)
        arrow(a,(.29,.54),(.37,.54));arrow(a,(.63,.54),(.71,.54));text(a,.5,.18,'取消调用返回 1.6 ms；块释放约 31 ms',11,ha='center');save(f,'cancel-lifetime')
        # Exact common-prefix lengths; tree topology is drawn by logical depth.
        d=data['prefix'];f,a=canvas(4.5);positions={0:(.12,.50),1:(.38,.83),2:(.38,.40),3:(.63,.63),4:(.63,.24),5:(.86,.39),6:(.86,.10)}
        for u,v,n in d['edges']:
            arrow(a,positions[u],positions[v]);x=(positions[u][0]+positions[v][0])/2;y=(positions[u][1]+positions[v][1])/2;text(a,x,y+.065,f'+{n}',11,ha='center')
        for i,node in enumerate(d['tree']):
            x,y=positions[i];a.plot(x,y,'o',color='#267398',ms=5)
            if len(node['ids'])==1:text(a,x,y+.09,f'轮 {node["ids"][0]+1}',11,ha='center')
        text(a,.04,.95,'前四轮：从共同 206 个 token 开始',14);text(a,.5,.03,'边上数字：新增 token 数',11,ha='center');save(f,'6-prefix')
        f,a=plot(3.8);v=np.array(d['adjacent_lcp']);a.bar(range(1,13),v,color=COL['blue'],label='与上一轮共同前缀');a.bar(range(1,13),np.array(d['input_lengths'])-v,bottom=v,color=COL['orange'],label='其余输入');a.set(xlabel='输入轮次',ylabel='token 数',xticks=[1,3,6,9,12]);a.legend(frameon=False);save(f,'prefix-lengths')
        f,a=canvas(3.8);text(a,.04,.94,'文本匹配到 10752，状态从 8192 恢复',13);a.plot([.05,.95],[.48,.48],color='#777777')
        for x,label,c in [(.10,'4096\n快照','blue'),(.48,'8192\n最近快照','green'),(.85,'10752\n匹配末端','orange')]:box(a,x-.07,.39,.17,.22,label,c,11)
        arrow(a,(.57,.49),(.77,.49));text(a,.68,.22,'重算 2560 个 token',12,ha='center');save(f,'prefix-restore')
        f,a=canvas(4.1)
        for row in range(2):
            y=.63-row*.40
            cc=data['cache_choice'];va=round(cc['A']['net_ms'],1);vb=round(cc['B']['each_net_ms'],1)
            if row==0:box(a,.04,y,.92,.19,f'A：864 MiB，净节省 {va} ms','blue')
            else:
                for i in range(3):box(a,.04+i*.31,y,.30,.19,f'B：288 MiB\n{vb} ms','green',11)
            text(a,.5,y-.09,f'合计节省 {va if row==0 else round(3*vb,1)} ms',12,ha='center')
        save(f,'7-cache-choice')
        f,a=plot(3.5,left=.20)
        for i,(payload,scale) in enumerate(zip(data['kv_layout']['payload_bytes'],data['kv_layout']['scale_bytes'])):a.barh(i,payload,height=.5,color=COL['blue'],edgecolor=COL['line']);a.barh(i,scale,left=payload,height=.5,color=COL['orange'],edgecolor=COL['line'])
        a.set(yticks=range(3),yticklabels=['BF16','q8_0','q4_0'],xlim=(0,68),xlabel='保存同一组 32 个值（bytes）');a.invert_yaxis();save(f,'8-kv-format')
        f,a=plot(3.6,left=.23)
        for i,buff in enumerate([288,576]):a.barh(i,buff,height=.48,color=COL['orange'],edgecolor=COL['line']);a.barh(i,2592-buff,left=buff,height=.48,color=COL['green'],edgecolor=COL['line'])
        a.set(yticks=[0,1],yticklabels=['一组预取缓冲','两组预取缓冲'],xlim=(0,2800),xlabel='卸载腾出的空间（MiB）');a.invert_yaxis();save(f,'9-offload')
        f,a=plot(3.5);v=data['offload']['copy_ms'];a.bar([0,1],v,color=[COL['blue'],COL['green']],edgecolor=COL['line']);a.set(xticks=[0,1],xticklabels=['PCIe Gen5 x16\n64 GB/s','NVLink-C2C\n450 GB/s'],ylabel='每轮复制下界（ms）',ylim=(0,50));save(f,'offload-copy')
        for i,m in enumerate(data['kv']['correct_by_task']):
            f,a=plot(4.7,left=.25);a.imshow(m,cmap=ListedColormap([COL['orange'],COL['green']]),vmin=0,vmax=1,aspect='auto')
            for y,row in enumerate(m):
                for x,val in enumerate(row):a.text(x,y,'○' if val else '×',ha='center',va='center',fontsize=14)
            a.set(yticks=range(8),yticklabels=[f'任务 {j+1}' for j in range(len(data['kv']['task_ids']))],xticks=range(4),xticklabels=['1 / 1','1 / 2','4 / 1','4 / 2'],xlabel='并发数 / 重复序号');save(f,'10-kv-quality' if i==0 else f'kv-quality-{i}')
        f,a=canvas(4.6)
        for row,(title,tokens) in enumerate([('草稿',['a','b','c','d']),('验证',['a','b','x','丢弃']),('保留',['a','b','x'])]):
            y=.71-row*.28;text(a,.03,y+.065,title,12)
            for j,t in enumerate(tokens):box(a,.21+j*.19,y,.16,.14,t,'green' if j<2 else ('orange' if row==0 or t=='丢弃' else 'blue'),12)
        save(f,'11-verification')
        for draft,n in [('A','sample-A'),('B','sample-B')]:
            p=.25 if draft=='A' else .75;f,a=canvas(4.0);box(a,.32,.71,.36,.17,f'草稿恒为 {draft}','gray')
            for x,label,prob,c in [(.03,'接受 '+draft,p,'green'),(.54,'拒绝，输出 '+('B' if draft=='A' else 'A'),1-p,'orange')]:box(a,x,.18,.43,.19,label,c,12);arrow(a,(.5,.71),(x+.215,.37));text(a,x+.215,.48,f'概率 {prob:g}',12,ha='center')
            text(a,.5,.06,'最终：P(A) = 1/4，P(B) = 3/4',12,ha='center');save(f,n)
        f,a=plot(3.6);q=np.linspace(0,60,121);sp=data['speculation']
        for d,c,x in zip(sp['drafts'],['#267398','#388768'],sp['query_break_even_ms']):a.plot(q,(sp['verification_ms']+q)/d['expected_output'],color=c,label=d['draft']);a.scatter([x],[sp['ordinary_ms']],color=c,zorder=5)
        a.axhline(sp['ordinary_ms'],color='#777777',ls='--',label='普通 decode');a.set(xlabel='每轮草稿查询（ms）',ylabel='每输出 token 耗时（ms）',xlim=(0,60),ylim=(0,70));a.legend(frameon=False);save(f,'12-speculation')
        for field,name in [('completed_per_s','13-service'),('slo_goodput_per_s','service-goodput')]:
            f,a=plot(4.0,bottom=.28);groups=data['service']['groups'];labels=[]
            for i,g in enumerate(groups):
                vals=[x[field] for x in g];a.bar(i,np.median(vals),color=COL['blue' if field=='completed_per_s' else 'green'],edgecolor=COL['line']);a.scatter([i]*len(vals),vals,color='#252525',s=16);labels.append(('逐个' if g[0]['service']=='serial' else '连续' if g[0]['service']=='continuous' else '突发')+('\n'+str(g[0]['rate']) if g[0]['rate'] is not None else '\n一次到达'))
            a.set(xticks=range(len(groups)),xticklabels=labels,ylim=(0,3.1),ylabel='完成吞吐（req/s）' if field=='completed_per_s' else '正确且按期完成（req/s）',xlabel='接纳策略；下行为到达率（req/s）');save(f,name)
        f,a=plot(3.8);dp=data['design_plane'];a.fill_between([0,dp['capacity_gib']],0,dp['deadline_s'],color=COL['green'],alpha=.65)
        for c in data['design_plane']['configurations']:a.scatter(c['memory_gib'],c['time_s'],color='#267398');a.annotate(c['name'],(c['memory_gib'],c['time_s']),xytext=(5,5),textcoords='offset points',fontsize=12)
        a.axvline(dp['capacity_gib'],ls='--',color='#777777');a.axhline(dp['deadline_s'],ls='--',color='#777777');a.set(xlim=(0,21),ylim=(0,16.5),xlabel='KV 与辅助缓冲（GiB）',ylabel='整组完成时间（s）');save(f,'14-design')
        f,a=plot(3.6);s=np.array(data['task']['local_speedups'])
        for frac,col in zip(data['task']['decode_fractions'],['#267398','#388768','#a56c28']):a.plot(s,1/(1-frac+frac/s),label=f'decode 占 {frac:.0%}',color=col)
        a.set(xlabel='decode 加速比',ylabel='完整任务加速比',xlim=(1,8));a.legend(frameon=False);save(f,'15-task')
        # Multi-LoRA batching diagram with proper math subscripts ($W_0$, $XW_0 + \Delta Y$)
        f,a=canvas(4.6)
        text(a,.03,.92,'(a) 计算路径：共享基础权重 + 分组低秩分支',12)
        text(a,.72,.92,'(b) 显存容量对比',12)
        box(a,.02,.40,.16,.44,'输入批次 $X$\n($b=16$ 行)','gray',11)
        box(a,.21,.65,.27,.20,'共享基础权重 $W_0$\n(14.1 GiB, 读 1 次)','blue',11)
        box(a,.21,.37,.27,.20,'分组低秩分支\n(按 Adapter 聚拢)','orange',11)
        box(a,.51,.48,.18,.28,'相加输出\n$XW_0+\\Delta Y$','green',11)
        arrow(a,(.18,.66),(.21,.74));arrow(a,(.18,.56),(.21,.48))
        arrow(a,(.48,.74),(.51,.65));arrow(a,(.48,.48),(.51,.57))
        box(a,.72,.68,.25,.18,'100 套 LoRA\n1.43 GiB','orange',11)
        box(a,.72,.46,.25,.18,'基础模型权重\n15.3 GiB','blue',11)
        box(a,.72,.24,.25,.18,'100 条 8K KV\n112.5 GiB (瓶颈)','purple',11)
        box(a,.03,.04,.94,.15,'可驻留 Adapter 数量取决于低秩小矩阵容量 (100 套仅 1.43 GiB)\n可同时服务请求数量仍受限于上下文 KV 缓存容量 (100 条需 112.5 GiB)','gray',11)
        save(f,'lora-batch')
        # Draft generation methods timeline
        f,a=canvas(4.2)
        rows=[('普通自回归解码',[(0,26.3,'gray','目标 26.3 ms')],'1 tok → 26.3 ms/tok'),
              ('串行草稿 ($E[N]=3$)',[(0,3.57,'orange',''),(3.57,3.57,'orange',''),(7.14,3.57,'orange',''),(10.71,3.57,'orange',''),(14.28,26.26,'blue','验证 26.3 ms')],'3 tok → 13.5 ms/tok'),
              ('并行 DFlash ($E[N]=3$)',[(0,3.57,'green',''),(3.57,26.26,'blue','验证 26.3 ms')],'3 tok → 9.9 ms/tok'),
              ('并行 DFlash ($E[N]=2$)',[(0,3.57,'orange',''),(3.57,26.26,'purple','验证 26.3 ms')],'2 tok → 14.9 ms/tok')]
        x0=.27;scale=.0105
        for idx,(label,segs,summary) in enumerate(rows):
            y=.76-idx*.19
            text(a,x0-.015,y+.06,label,11,ha='right')
            end_x=x0
            for st,dur,col,btxt in segs:
                bx=x0+st*scale;bw=dur*scale
                box(a,bx,y,bw,.12,btxt,col,11)
                end_x=bx+bw
            text(a,end_x+.015,y+.06,summary,11)
        a.plot([x0,x0+62*scale],[.14,.14],color=COL['ink'],lw=1)
        for tick in [0,10,20,30,40,50,60]:
            tx=x0+tick*scale
            a.plot([tx,tx],[.12,.14],color=COL['ink'],lw=1)
            text(a,tx,.07,str(tick),11,ha='center')
        text(a,x0+30*scale,.01,'单轮执行时间 / ms',11,ha='center')
        save(f,'draft-methods')
        # Path-A new 420pt 2-panel figure: XGrammar CPU-GPU overlap & Lossy KV compression reversal
        f,a=canvas(6.8)
        text(a,.50,.95,'A. 结构化生成（XGrammar）CPU-GPU 流水重叠与单步耗时（Qwen3-8B）',12,ha='center')
        box(a,.02,.78,.28,.12,'串行未重叠\n合计 5.2 ms','gray',11)
        box(a,.32,.78,.26,.12,'GPU 前向\n2.0 ms','blue',11)
        box(a,.60,.78,.38,.12,'CPU 单线程掩码 3.2 ms\n阻塞等待','orange',11)
        box(a,.02,.63,.28,.12,'单线程重叠\n合计 3.2 ms','gray',11)
        box(a,.32,.63,.38,.12,'CPU 掩码 3.2 ms\n掩盖 GPU 2.0 ms','purple',11)
        box(a,.72,.63,.26,.12,'受限于 CPU\n3.2 ms','orange',11)
        box(a,.02,.48,.28,.12,'多线程重叠\n合计 2.0 ms','gray',11)
        box(a,.32,.48,.38,.12,'GPU 前向 2.0 ms\n掩盖 4 线程 0.8 ms','green',11)
        box(a,.72,.48,.26,.12,'零掩码阻塞\n2.0 ms','green',11)
        text(a,.50,.42,'位图掩码 1.159 MiB（传输 48.6 μs）；融合采样省去 74.2 MiB Logits 读写',11,ha='center')
        a.plot([.02,.98],[.38,.38],color=COL['line'],ls='--',lw=1.0)
        text(a,.50,.33,'B. 有损 KV 压缩下“单步提速 20% 但输出膨胀 50%”引发耗时反转',12,ha='center')
        box(a,.02,.18,.28,.12,'BF16 无损基线\n50 tok/s · 200 tok','blue',11)
        box(a,.32,.18,.44,.12,'Prefill 0.5 s + Decode 4.0 s','green',11)
        box(a,.78,.18,.20,.12,'总耗时\n4.5 s','green',11)
        box(a,.02,.04,.28,.12,'有损 KV 压缩\n60 tok/s · 300 tok','orange',11)
        box(a,.32,.04,.44,.12,'Prefill 0.5 s + Decode 5.0 s','orange',11)
        box(a,.78,.04,.20,.12,'5.5 s\n(+22.2%)','orange',11)
        out.save(f,'ch08-xgrammar-overlap-and-kv-compression-reversal')
    from core_principles_figures import draw as draw_principles
    draw_principles(8, out)
    from v41_case_figures import draw as draw_v41
    draw_v41(8, out)
    out.finish();return out.outputs,out.checks
