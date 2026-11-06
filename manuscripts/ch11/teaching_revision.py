"""Keep the task, its environment and its resource occupancy visually distinct."""
import numpy as np
import matplotlib.pyplot as plt
from figure_style import COL,STYLE,canvas,plot,text,box,arrow,Exporter

def draw(here,data):
    out=Exporter(here)
    def save(f,n):out.save(f,'figure-11-'+n)
    def timeline(name,model=9,env=False,rebuild=False):
        f,a=plot(3.7,left=.21);period=model+1
        for k in range(3):
            if env:a.barh(0,3 if rebuild else period,left=k*period+model-2 if rebuild else k*period,height=.55,color=COL['green'],edgecolor=COL['line'])
            else:
                a.barh(1,model,left=k*period,height=.5,color=COL['blue'],edgecolor=COL['line']);a.barh(0,1,left=k*period+model,height=.5,color=COL['orange'],edgecolor=COL['line'])
        a.set(yticks=[0] if env else [1,0],yticklabels=['2 GiB 内存'] if env else ['模型调用','工具 CPU'],xlabel='从任务到达计时（s）',xlim=(0,31),ylim=(-.7,1.6 if not env else .7));save(f,name)
    with plt.rc_context(STYLE):
        timeline('1-timeline');timeline('memory-area',env=True)
        for i,key in enumerate(['demand_9','demand_12']):
            f,a=plot(3.6,left=.24);v=data['capacity'][key];cap=data['capacity']['capacity'];a.barh(range(3),np.array(v)/cap*100,color=[COL[c] for c in ['orange','blue','green']],edgecolor=COL['line']);a.axvline(100,ls='--',color='#666')
            for j,(x,c,l) in enumerate(zip(v,cap,data['capacity']['capacity_labels'])):a.text(x/c*100+2,j,f'{x}/{l}',fontsize=11,va='center')
            a.set(yticks=range(3),yticklabels=['CPU 核','模型并发','内存 GiB'],xlim=(0,250),xlabel='需求／容量（%）');a.invert_yaxis();save(f,'capacity' if i==0 else 'capacity-slow')
        f,a=canvas(4.5)
        for x,y,w,l,c in [(.04,.66,.32,'任务控制器','orange'),(.64,.66,.32,'模型服务','blue'),(.04,.19,.32,'环境平台','gray'),(.64,.19,.32,'工具环境','green')]:box(a,x,y,w,.22,l,c)
        for p,q in [((.36,.82),(.64,.82)),((.64,.71),(.36,.71)),((.2,.66),(.2,.41)),((.36,.35),(.64,.35)),((.64,.24),(.36,.24))]:arrow(a,p,q)
        text(a,.5,.94,'模型请求／完整工具参数',11,ha='center');text(a,.5,.09,'工具执行／保存结果／交回控制器',11,ha='center');save(f,'2-boundary')
        for i,l in enumerate(['进程','容器','microVM']):
            f,a=canvas(4.0);text(a,.04,.94,l+'的隔离边界',14);box(a,.04,.11,.92,.17,'宿主操作系统内核','gray')
            for j in range(2):
                x=.04+j*.48;box(a,x,.39,.44,.40,'','blue' if i<2 else 'green');text(a,x+.22,.67,'任务 '+str(j),12,ha='center');text(a,x+.22,.50,['私有地址空间','私有视图与配额','独立虚拟机内核'][i],11,ha='center');arrow(a,(x+.22,.39),(x+.22,.28))
            save(f,'isolation-'+str(i))
        f,a=canvas(4.2);box(a,.04,.62,.35,.24,'共享模板\n文件与依赖','blue');box(a,.61,.62,.35,.24,'环境私有内容\n修改页与管理数据','orange');box(a,.24,.13,.52,.24,'活跃环境内存\n进程、工作页与缓冲区','green')
        arrow(a,(.215,.62),(.40,.37));arrow(a,(.785,.62),(.60,.37));text(a,.5,.48,'加载执行所需内容',11,ha='center');save(f,'template-runtime')
        f,a=plot(4.0,left=.25);v=data['pages']['local_mib'];a.barh(range(4),v,color=COL['blue'],edgecolor=COL['line'])
        for i,x in enumerate(v):a.text(x+30,i,str(x),fontsize=11,va='center')
        a.set(yticks=range(4),yticklabels=['完整复制','按需加载','仅私有页','保留热点页'],xlim=(0,2500),xlabel='每环境本地内容（MiB）');a.invert_yaxis();save(f,'pages')
        for pause,name in [(False,'pause'),(True,'pause-release')]:
            f,a=plot(3.0,left=.21)
            for start,dur,c in ([(10,8,'orange'),(18,1,'blue')] if pause else [(10,9,'green')]):a.barh(0,dur,left=start,height=.5,color=COL[c],edgecolor=COL['line'])
            if pause:text(a,14,0,'保存 8 s',11,ha='center');text(a,18.5,0,'恢复',11,ha='center')
            a.set(yticks=[0],yticklabels=['2 GiB 内存'],xlim=(10,19),xticks=[10,14,18,19],xlabel='时间（s）');save(f,name)
        timeline('residency',env=True);timeline('residency-rebuild',env=True,rebuild=True)
        for lead,name in [(0,'3-lifecycle'),(1,'prewarm-1'),(2,'prewarm-2'),(3,'prewarm-3')]:
            f,a=plot(2.8,left=.22);start=4-lead;a.barh(0,2,left=start,height=.5,color=COL['blue'],edgecolor=COL['line'])
            if lead>2:a.barh(0,lead-2,left=start+2,height=.5,color=COL['orange'],edgecolor=COL['line'])
            a.axvline(4,ls='--',color='#666');a.set(yticks=[0],yticklabels=['准备环境'],xlim=(0,6.2),xlabel='时间（s）');save(f,name)
        f,a=plot(3.6,left=.25);a.barh(1,2,left=2,height=.5,color=COL['gray'],edgecolor=COL['line']);a.barh(0,2,left=4,height=.5,color=COL['blue'],edgecolor=COL['line']);a.axvline(4,ls='--',color='#666');a.set(yticks=[1,0],yticklabels=['误选环境','实际所需环境'],xlim=(0,6.2),xlabel='时间（s）');save(f,'prewarm-wrong')
        for after,name in [(False,'4-placement'),(True,'placement-after')]:
            f,a=canvas(4.1);text(a,.5,.93,'新作业需要：同节点四张 H100＋16 核',13,ha='center')
            for i,(node,gpu,cpu) in enumerate([('DGX H100','H100',16 if after else 8),('DGX A100','A100',24 if after else 32)]):
                x=.04+i*.49;box(a,x,.23,.43,.48,'','gray');text(a,x+.215,.62,f'节点 {i+1}：{node}',13,ha='center');text(a,x+.215,.47,f'4 张空闲 {gpu}',12,ha='center');text(a,x+.215,.33,f'{cpu} 个空闲 CPU 核',11,ha='center')
            save(f,name)
        for burst,name in [(False,'queue'),(True,'queue-burst')]:
            f,a=plot(5.0,left=.16)
            for i in range(10):
                arrive=0 if burst else i
                if burst:a.barh(i,i,left=0,height=.52,color=COL['gray'])
                a.barh(i,1,left=i,height=.52,color=COL['blue'],edgecolor=COL['line']);a.plot(arrive,i,'o',ms=4,color='#454545')
            a.set(yticks=range(10),yticklabels=[str(i+1) for i in range(10)],ylabel='工具调用',xlabel='时间（s）',xlim=(-.3,10.3));a.invert_yaxis();save(f,name)
        f,a=plot(3.8,left=.18)
        for i,row in enumerate(data['rl-stages']['stage_seconds']):
            start=0
            for j,(v,c,l) in enumerate(zip(row,['blue','orange','green','purple'],['生成','验证','更新','发布'])):a.barh(i,v,left=start,height=.5,color=COL[c],edgecolor=COL['line'],label=l if i==0 else None);start+=v
        a.set(yticks=[0,1],yticklabels=['原配置','生成加速'],xlim=(0,90),ylim=(-.5,2.1),xlabel='迭代时间（s）');a.legend(ncol=4,frameon=False,loc='upper left');save(f,'rl-stages')
        f,a=canvas(5.0);box(a,.27,.76,.46,.16,'发送端：200 Gbit/s','orange',12)
        for i in range(6):
            x=.04+(i%3)*.32;y=.41-(i//3)*.26;box(a,x,y,.28,.18,f'实例 {i+1}\n16.38 GB／50 Gbit/s','blue',11)
        arrow(a,(.5,.76),(.5,.63));text(a,.53,.695,'同一出口共传 98.3 GB',11,ha='left')
        for i in range(3):
            x=.18+i*.32;arrow(a,(x,.63),(x,.59));arrow(a,(x,.41),(x,.33))
        a.plot([.18,.82],[.63,.63],color=COL['line'],lw=1.3);text(a,.5,.05,'单份至少 2.62 s；全部至少 3.93 s',12,ha='center');save(f,'weights')
        for batch,name in [(False,'5-stages'),(True,'stages-batch')]:
            f,a=plot(3.1,left=.20)
            for i in range(3):a.barh(0,10,left=i*10+(20 if batch else 0),height=.5,color=COL[['blue','green','orange'][i]],edgecolor=COL['line']);a.plot(i*10,1,'o',color='#454545')
            a.set(yticks=[1,0],yticklabels=['样本到达','验证进程'],xlim=(-1,51),xlabel='时间（s）');save(f,name)
        f,a=plot(4.0,left=.20);a.barh(range(10),[1]*9+[100],height=.6,color=[COL['blue']]*9+[COL['orange']],edgecolor=COL['line']);a.axvline(10,ls='--',color='#666');a.set(yticks=[0,8,9],yticklabels=['样本 1','样本 9','样本 10'],xlim=(0,105),xlabel='验证时间（s）');a.invert_yaxis();save(f,'remaining-time')
        f,a=plot(3.9,left=.18)
        for i,row in enumerate(data['thinking']['cost_parts']):
            start=0
            for v,c,l in zip(row,['blue','orange','green'],['输入','思考','可见输出']):a.barh(i,v,left=start,height=.5,color=COL[c],edgecolor=COL['line'],label=l if i==0 else None);start+=v
        a.set(yticks=[0,1],yticklabels=['1000 思考','100 思考'],xlabel='单次调用成本（美元）',ylim=(-.5,2.0));a.legend(ncol=3,frameon=False,loc='upper left');save(f,'thinking')
        f,a=canvas(4.1);box(a,.04,.40,.23,.23,'任务控制器','orange',11);box(a,.38,.40,.23,.23,'服务入口','gray');box(a,.74,.68,.22,.19,'外部 API','blue',11);box(a,.74,.15,.22,.19,'自建副本','green',11);arrow(a,(.27,.515),(.38,.515));arrow(a,(.61,.56),(.74,.775));arrow(a,(.61,.46),(.74,.245));text(a,.50,.86,'按调用用量计费',11,ha='center');text(a,.5,.13,'按加速器与运行支出计费',11,ha='center');save(f,'6-service')
        for quality,name in [(False,'7-routing'),(True,'routing-deadline')]:
            d=data['11-7'];h=np.array(d['h']);f,a=plot(4.0)
            if quality:a.plot(h*100,.98*h*100,color='#267398');a.axhline(90,ls='--',color='#a56c28');a.axvline(d['joint_target_hit']*100,ls='--',color='#666');a.set(ylabel='按时成功的提交比例（%）',ylim=(0,105))
            else:a.plot(h*100,d['cost_B'],label='B：Sonnet 5',color='#267398');a.axhline(d['cost_A'],label='A：Haiku 4.5',color='#a56c28');a.axvline(d['cost_crossover']*100,ls='--',color='#666');a.set(ylabel='每个成功任务的成本（美元）');a.legend(frameon=False)
            a.set(xlabel='B 请求命中率（%）',xlim=(0,100));save(f,name)
        f,a=plot(3.8);d=data['purchase'];n=np.linspace(0,d['capacity'],100);m=np.linspace(0,3.2e6,100);a.plot(n/1e4,d['fixed']+n*d['self_per_task'],label='预留 4 张 B200',color='#267398');a.plot(m/1e4,m*d['api_per_task'],label='按量计费',color='#388768');a.axvline(d['crossover']/1e4,ls='--',color='#666');a.set(xlabel='每月提交任务数（万项）',ylabel='每月总成本（美元）',xlim=(0,320));a.legend(frameon=False);save(f,'purchase')
        f,a=canvas(4.6)
        for x,l,c in [(.04,'控制器','orange'),(.64,'外部系统','blue')]:box(a,x,.69,.32,.17,l,c)
        arrow(a,(.36,.77),(.64,.77));text(a,.5,.95,'操作 ID：K',12,ha='center');box(a,.64,.39,.32,.17,'操作已提交','green',11);arrow(a,(.80,.69),(.80,.56));arrow(a,(.64,.46),(.36,.46),'control');text(a,.17,.46,'确认丢失',12,ha='center');arrow(a,(.36,.18),(.64,.18),'control');text(a,.5,.08,'恢复后先按 K 查询结果',12,ha='center');save(f,'commit-ack')
        # Split the probability tree at the local-repair node to keep edge labels readable.
        f,a=canvas(5.3);box(a,.30,.75,.40,.17,'首次尝试：10 s','blue')
        for x,l,c in [(.02,'首次成功\n80%','green'),(.355,'局部修复\n12%','orange'),(.69,'直接升级\n8%','purple')]:box(a,x,.37,.29,.22,l,c);arrow(a,(.5,.75),(x+.145,.59))
        text(a,.5,.18,'下方比例均以全部提交为分母',11,ha='center');save(f,'retry-tree')
        f,a=canvas(5.4);box(a,.29,.74,.42,.18,'进入修复：全部的 12%','orange',11)
        box(a,.04,.37,.37,.22,'修复成功：7.2%\n累计 14 s','green',11);box(a,.59,.37,.37,.22,'修复后升级：4.8%\n累计 22 s','purple',11)
        arrow(a,(.40,.74),(.225,.59));arrow(a,(.60,.74),(.775,.59));text(a,.18,.68,'条件 60%',11,ha='center');text(a,.82,.68,'条件 40%',11,ha='center');text(a,.5,.19,'12% × 60% = 7.2%\n12% × 40% = 4.8%',12,ha='center');save(f,'retry-conditional')
        f,a=plot(4.0,bottom=.28);v=data['11-8']['policy_costs'];a.bar(range(3),v,color=[COL[c] for c in ['blue','green','orange']],edgecolor=COL['line'])
        for i,y in enumerate(v):a.text(i,y+.00035,f'{y:.4f}',ha='center',fontsize=12)
        a.set(xticks=range(3),xticklabels=['仅首次\n成功任务','有限恢复\n成功任务','有限恢复\n按时成功'],ylabel='全部支出（美元）／符合条件的任务数',ylim=(0,.018));save(f,'8-retry')
        f,a=plot(3.6,left=.20)
        for i,model in enumerate([9,6]):
            for k in range(3):a.barh(i,model,left=k*(model+1),height=.5,color=COL['blue'],edgecolor=COL['line']);a.barh(i,1,left=k*(model+1)+model,height=.5,color=COL['orange'],edgecolor=COL['line'])
        a.axvline(24,ls='--',color='#a56c28');a.set(yticks=[0,1],yticklabels=['普通模型','快速模型'],xlim=(0,31),xlabel='完成时间（s）');save(f,'decision')
        # Per-round cloud uploads may fail independently of local persistence.
        f,a=canvas(3.6)
        for x,label in [(.36,'运行测试\n第 1 轮'),(.59,'修改代码\n第 2 轮'),(.82,'再次测试\n第 3 轮')]:
            text(a,x,.89,label,11,ha='center')
        for y,label in [(.69,'本地记录'),(.43,'云端记录'),(.17,'云端进度')]:
            text(a,.02,y,label,12)
        for x in [.26,.49,.72]:box(a,x,.61,.20,.16,'已保存','blue',11)
        box(a,.26,.35,.20,.16,'已保存','green',11)
        gap=box(a,.49,.35,.20,.16,'上传失败','orange',11)
        gap.set_linestyle('--')
        box(a,.72,.35,.20,.16,'已保存','green',11)
        box(a,.26,.09,.20,.16,'到第 1 轮','green',11)
        a.plot([.475,.475],[.06,.56],color=COL['line'],ls='--',lw=1)
        text(a,.71,.17,'第 2 轮缺失，不能跳到第 3 轮',11,ha='center')
        save(f,'recovery-coverage')
        data['recovery-coverage']={'kind':'teaching','unit':'react_round','trusted_base':0,'local_saved':[1,2,3],'cloud_saved':[1,3],'cloud_gap':[2],'cloud_covered_through':1,'upload_policy':'per-round incremental async; failed upload does not block later uploads'}
        # Steady schedule timeline and capacity check (Figure 11-51).
        f,a=plot(3.8,left=.20)
        for k in range(3):
            st=k*7
            a.barh(1,6,left=st,height=.48,color=COL['blue'],edgecolor=COL['line'])
            a.text(st+3,1,f'第{k+1}轮 6 s',ha='center',va='center',fontsize=11)
            a.barh(0,2,left=st+4,height=.48,color=COL['green'],edgecolor=COL['line'])
            a.barh(0,1,left=st+6,height=.48,color=COL['orange'],edgecolor=COL['line'])
        a.axvline(21,ls='--',color='#207655',lw=1.4);a.axvline(24,ls='--',color='#b63c33',lw=1.4)
        a.text(20.6,.52,'21 s 完成',ha='right',va='center',fontsize=11,color='#207655')
        a.text(22.6,.98,'24 s\n期限',ha='center',va='center',fontsize=11,color='#b63c33')
        a.text(0.4,1.50,'稳态：180/192 槽，33/48 核，180/358 GiB',fontsize=11.5)
        a.set(yticks=[1,0],yticklabels=['快速模型','准备+工具'],xticks=[0,4,7,11,14,18,21,24],xlim=(0,25.5),ylim=(-.55,1.75),xlabel='单任务时间线（s，绿=准备 2 s，橙=工具 1 s）')
        save(f,'steady-schedule')
        # Inter-call waiting gap crossover (Figure 11-52).
        f,a=plot(3.9,left=.18)
        g=np.linspace(0,12,241)
        c_keep=0.9*g
        c_rebuild=np.full_like(g,1.94)
        c_pause=np.minimum(0.9*g,8.1)
        a.plot(g,c_keep,label='始终驻留（2G GiB·s）',color='#267398',lw=2.2)
        a.plot(g,c_rebuild,label='每轮重建（准备 2 s）',color='#2a7b58',ls='--',lw=2.2)
        a.plot(g,c_pause,label='暂停恢复（保存 8 s + 恢复 1 s）',color='#ba5b24',ls='-.',lw=2.2)
        a.axvline(97/45,ls=':',color='#666',lw=1.1);a.axvline(9.0,ls=':',color='#666',lw=1.1)
        a.plot([97/45,9.0],[1.94,8.1],'o',color='#b63c33',ms=6,zorder=5)
        a.text(2.4,1.0,'重建交点 G≈2.16 s',fontsize=11)
        a.text(9.2,6.8,'暂停门槛\nG=9.0 s',fontsize=11,va='center')
        a.set(xlim=(0,12),ylim=(0,11.8),xlabel='轮间模型等待时长 G（s）',ylabel='单次间隔成本（0.001 美分）')
        a.legend(frameon=False,loc='upper left',fontsize=11)
        save(f,'lifecycle-crossover')
        # Path-A new figure: Sandbox CXLfork memory density & Weave RL job interleaving Gantt.
        f,a=canvas(6.6)
        text(a,.03,.95,'A. 沙箱快照共享与按需缺页（100 实例内存与 64 GiB 并发上限）',12)
        cards=[
            (.03,'完整复制','200.39 GiB\n上限 31 个','orange'),
            (.27,'UFFD 缺页','50.39 GiB\n上限 127 个','blue'),
            (.51,'共享+热页','37.89 GiB\n上限 168 个','purple'),
            (.75,'CXLfork 脏页','12.89 GiB\n上限 496 个','green'),
        ]
        for x,title,sub,col in cards:
            box(a,x,.70,.22,.20,f'{title}\n{sub}',col,11)
        box(a,.03,.55,.94,.11,'4 KiB 按需缺页：64 页仅读 256 KiB（0.24 ms）\n2 MiB 大页读放大：读取 128 MiB（512× 放大，125 ms）','gray',11)
        text(a,.03,.46,'B. Weave 互补 RL 作业交错调度（A: 360/120 s，B: 120/360 s）',12)
        text(a,.03,.36,'Rollout 池',11)
        box(a,.18,.31,.51,.11,'作业 A Rollout（0–360 s）','blue',11)
        box(a,.71,.31,.26,.11,'作业 B Rollout\n360–480 s','green',11)
        text(a,.03,.21,'训练更新池',11)
        box(a,.18,.16,.51,.11,'作业 B 上轮更新（0–360 s）','green',11)
        box(a,.71,.16,.26,.11,'作业 A 更新\n360–480 s','blue',11)
        box(a,.03,.02,.94,.11,'热切换 5 s → 周期 490 s（≤528 s 达标）\n双作业驻留 400+500+150 = 1050 GiB > 1 TiB 主机上限','orange',11)
        out.save(f,'ch11-sandbox-cxlfork-and-weave-gantt')

        # Path-B new figure: ch11-api-pricing-vs-provider-cost-and-routing (Fig 11-53)
        f,a=canvas(6.4)
        text(a,.25,.95,'A. 算力波谷与 Batch API 填谷',12,ha='center')
        text(a,.745,.95,'B. 双模型缓存路由可行域',12,ha='center')
        a.plot([.495,.495],[.03,.97],color=COL['line'],ls='--',lw=1.0)

        box(a,.03,.75,.44,.15,'24×7 预留集群刚性折旧：\nB200 长租 6.79 美元/卡时\n夜间在线波谷大量卡时空转','orange',11)
        box(a,.03,.54,.44,.16,'异步 Batch API（24h 窗口）：\n零售牌价打 5 折（50% 折扣）\n剥离实时 TTFT / TPOT 约束','green',11)
        box(a,.03,.31,.44,.18,'物理填谷收益（TC-16）：\n夜间拼成最大批次满载运行\n仅付增量电费即可回收折旧','blue',11)
        box(a,.03,.05,.44,.20,'自建 vs 按量平衡点：\n预留实例需 >78.6% 利用率\n单会话（batch=1）宜用 API','gray',11)

        box(a,.52,.75,.45,.15,'小模型 A vs 旗舰模型 B：\nB 牌价高但思考精炼（100 词）\n通过率 98% vs 80%（门槛 75%）','purple',11)
        box(a,.52,.54,.45,.16,'成本反超交点 h_cost：\nClaude 牌价：h ≥ 79.52%\nI21 调研基线：h ≥ 84.93%','blue',11)
        box(a,.52,.31,.45,.18,'叠加 6 s 交付期限门槛：\nB 命中 4 s 达标、未中 12 s\n0.98h ≥ 0.90 => h ≥ 91.84%','green',11)
        box(a,.52,.05,.45,.20,'两级级联路由（TC-17）：\nE[C] = c_r + c_S + r·c_L(h_L)\n跨模型不可直接共享 KV 块','orange',11)
        out.save(f,'ch11-api-pricing-vs-provider-cost-and-routing')
    from dsec_figures import draw as draw_dsec
    draw_dsec(out)
    from core_principles_figures import draw as draw_principles
    draw_principles(11, out)
    out.finish();return out.outputs,out.checks
