"""Chapter six: follow ownership, then messages, then elapsed time."""
import numpy as np
import matplotlib.pyplot as plt
from figure_style import COL, STYLE, canvas, plot, text, box, arrow, Exporter

def draw(here,data):
    out=Exporter(here)
    def save(f,n): out.save(f,'figure-6-'+n)
    with plt.rc_context(STYLE):
        f,a=canvas(4.4)
        for row,group in enumerate([1,4,8]):
            y=.73-row*.29;text(a,.02,y+.19,f'每实例 {group} 张卡',12)
            for start in range(0,8,group):
                box(a,.03+start*.118,y,.118*group-.012,.13,'','gray')
                for i in range(start,start+group):box(a,.04+i*.118,y+.025,.09,.08,str(i),'blue',11)
        save(f,'1-placement')
        f,a=plot(3.6,left=.24);left=np.zeros(2)
        for i,label,col in [(0,'权重','blue'),(1,'KV','green'),(2,'工作区','orange')]:
            v=np.array(data['capacity_plot']['segments_GB'])[:,i];a.barh([0,1],v,left=left,height=.45,label=label,color=COL[col],edgecolor=COL['line']);left+=v
        a.axvline(80,ls='--',color='#777777');a.text(84,.5,'H100：80 GB',fontsize=11,va='center');[a.text(max(t+6,86),j,f'{t:.1f} GB' if t>100 else f'{t:.2f} GB',va='center',fontsize=11) for j,t in enumerate(left)];a.set(yticks=[0,1],yticklabels=['单卡实例','八卡中的每卡'],xlim=(0,560),xlabel='每卡内存占用（GB）');a.invert_yaxis();f.subplots_adjust(top=.84);a.legend(frameon=False,ncol=3,loc='lower center',bbox_to_anchor=(.5,1.0));save(f,'2-capacity')
        dm=data['continuous_execution']['dense_moe']['models'];names=['Qwen3-32B','Qwen3-30B-A3B','Qwen3.6-35B-A3B']
        f,axs=plt.subplots(1,2,figsize=(420/72,3.6));f.subplots_adjust(left=.12,right=.98,bottom=.12,top=.80,wspace=.45)
        a=axs[0];ctx=['32768','131072','262144'];x=np.arange(3)
        for i,(m,col) in enumerate(zip(dm,['blue','orange','green'])):
            vals=[m['state_per_request'][c]/1e9 if (c!='262144' or m['model']=='qwen3.6-35b-a3b') else np.nan for c in ctx]
            a.bar(x+(i-1)*.27,vals,width=.27,color=COL[col],edgecolor=COL['line'],label=names[i])
        a.set(xticks=x,xticklabels=['32K','128K','256K'],ylabel='每请求状态（GB）',ylim=(0,40));a.set_title('上下文与状态',fontsize=11)
        f.legend(*a.get_legend_handles_labels(),frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.0),fontsize=11,handlelength=1,columnspacing=.8)
        a=axs[1];w=np.array([m['b64_weight_read_s']*1e3 for m in dm]);st=np.array([m['b64_state_read_s']*1e3 for m in dm])
        a.bar(range(3),w,color=COL['gray'],edgecolor=COL['line'],label='权重');a.bar(range(3),st,bottom=w,color=COL['purple'],edgecolor=COL['line'],label='状态')
        for j in range(3):a.text(j,w[j]+st[j]+4,f'{w[j]+st[j]:.0f}',ha='center',fontsize=11)
        a.legend(frameon=False,loc='upper right',fontsize=11,handlelength=1)
        a.set(xticks=range(3),xticklabels=['32B','30B-A3B','3.6-35B'],ylabel='读取时间（ms）',ylim=(0,210));a.set_title('batch 64、32K',fontsize=11)
        save(f,'dense-moe')
        for rows,name in [(False,'tp-columns'),(True,'tp-rows')]:
            f,a=canvas(4.4);text(a,.04,.94,'同一乘法：[2, 3] × [[1, 4], [2, 5]]',12)
            for i in range(2):
                x=.04+i*.50;box(a,x,.37,.42,.43,'','green' if i else 'blue');text(a,x+.21,.72,f'卡 {i}',14,ha='center')
                expr=([ '2 × [1, 4]','3 × [2, 5]'] if rows else ['[2, 3] × [1, 2] T','[2, 3] × [4, 5] T'])[i]
                text(a,x+.21,.59,expr,12,ha='center');text(a,x+.21,.45,(['[2, 8]','[6, 15]'] if rows else ['8','23'])[i],14,ha='center');arrow(a,(x+.21,.37),(.50,.24))
            box(a,.16,.06,.68,.17,'逐元素求和 → [8, 23]' if rows else '并排拼接 → [8, 23]','orange');save(f,name)
        f,a=canvas(5.1);box(a,.29,.82,.42,.12,'完整输入 X：m × h','gray')
        for i in range(2):
            x=.04+.50*i;arrow(a,(.5,.82),(x+.21,.70));box(a,x,.52,.42,.18,f'上投影 → SiLU 与乘法\nZ{i}：m × (f/2)','blue' if i==0 else 'green',11);arrow(a,(x+.21,.52),(x+.21,.41));box(a,x,.26,.42,.15,f'下投影部分和\nm × h','blue' if i==0 else 'green',11);arrow(a,(x+.21,.26),(.5,.15))
        box(a,.23,.02,.54,.13,'相加 → 完整输出 m × h','orange');save(f,'3-tp')
        f,a=plot(3.8,left=.18)
        for stage in range(4):
            for batch in range(4):
                a.barh(stage,1,left=stage+batch,height=.72,color=COL[['blue','green','orange','purple'][batch]],edgecolor=COL['line']);a.text(stage+batch+.5,stage,str(batch),ha='center',va='center',fontsize=12)
        a.set(yticks=range(4),yticklabels=[f'阶段 {i}' for i in range(4)],xticks=range(8),xlim=(0,7),xlabel='时间（ms）；格内为 micro-batch 编号');a.invert_yaxis();save(f,'4-pipeline')
        f,a=canvas(4.7);box(a,.04,.72,.30,.18,'卡 0：输入 A','blue');box(a,.65,.72,.31,.18,'卡 3：输入 A','blue');arrow(a,(.34,.81),(.65,.81));text(a,.5,.94,'dispatch 输入',12,ha='center')
        for x,label in [(.04,'专家 1 → y1'),(.65,'专家 6 → y6')]:
            arrow(a,(x+.15,.72),(x+.15,.56));box(a,x,.39,.31,.17,label,'green');arrow(a,(x+.15,.39),(.5,.20))
        box(a,.12,.04,.76,.16,'卡 0：按路由权重合并 a1 y1 + a6 y6','orange',11);save(f,'5-dispatch')
        for reduction,name in [(False,'6-ep-layout'),(True,'ep-reduction')]:
            f,a=canvas(5.6);text(a,.5,.95,'专家贡献：先行内，再列内求和' if reduction else '每行分一组专家，同列重复保存 KV',14,ha='center')
            for row in range(4):
                y=.68-row*.18;text(a,.02,y+.055,f'组 {row}',11)
                for col in range(2):
                    x=.18+col*.43;label=f'卡 {2*row+col}\n'+(f'组贡献 {row+1}' if reduction else f'KV 头 {"0、1" if col==0 else "2、3"}')
                    box(a,x,y,.30,.13,label,'blue' if col==0 else 'green',11)
                    if reduction and row<3:arrow(a,(x+.15,y),(x+.15,y-.05))
                if reduction:arrow(a,(.48,y+.065),(.61,y+.065))
            text(a,.5,.06,'两列各得到完整输出：1 + 2 + 3 + 4 = 10' if reduction else '每行的专家中间维：左 768，右 768',11,ha='center');save(f,name)
        f,a=plot(3.4,left=.28);vals=data['expert_reuse']['balanced_vs_concentrated']['weight_read_bytes'];a.barh([0,1],np.array(vals)/2**20,height=.45,color=[COL['blue'],COL['orange']],edgecolor=COL['line']);a.set(yticks=[0,1],yticklabels=['选择 128 个专家','选择 8 个专家'],xlim=(0,5000),xlabel='一批读取的专家权重（MiB）');a.invert_yaxis();save(f,'7-reuse')
        for i,vals in enumerate(data['expert_load']['tasks']):
            f,a=plot(3.4);a.bar(range(4),vals,color=COL['blue'],edgecolor=COL['line']);a.set(xticks=range(4),xticklabels=[f'组 {j}' for j in range(4)],ylim=(0,580),ylabel='token—专家计算次数')
            for j,v in enumerate(vals):a.text(j,v+12,str(v),ha='center',fontsize=12)
            save(f,'8-expert-load' if i==0 else f'expert-load-{i}')
        f,a=canvas(4.3);text(a,.04,.94,'沿 0 → 1 → 2 → 3，累计块 0 的贡献',13)
        for i,v in enumerate([1,11,111,1111]):
            x=.025+i*.25;box(a,x,.47,.20,.23,f'卡 {i}\n{v}','green' if i==3 else 'blue');text(a,x+.10,.34,'起点' if i==0 else f'加 {10**i}',12,ha='center')
            if i<3:arrow(a,(x+.20,.585),(x+.25,.585))
        text(a,.5,.13,'三轮后，卡 3 持有块 0 的完整和',13,ha='center');save(f,'9-ring-rounds')
        f,a=canvas(5.0);text(a,.04,.94,'AllGather：每轮向下一卡转发一块',14)
        for r in range(4):
            y=.72-r*.20;text(a,.02,y+.06,'起点' if r==0 else f'{r} 轮',11)
            for card in range(4):
                owned=[(card+1-j)%4 for j in range(r+1)];box(a,.16+card*.21,y,.19,.13,','.join(map(str,owned)),'green' if r==3 else 'blue',11)
                if r==0:text(a,.255+card*.21,y+.20,f'卡 {card}',12,ha='center')
        text(a,.5,.03,'格内为已归约的数据块编号',11,ha='center');save(f,'ring-gather')
        f,a=plot(3.9,left=.19);left=np.zeros(4)
        for key,label,col in [('local_s','本地','blue'),('communication_s','归约','orange')]:
            v=np.array([x[key]*1000 for x in data['continuous_execution']['first_steps']]);a.barh(range(4),v,left=left,height=.5,color=COL[col],edgecolor=COL['line'],label=label);left+=v
        a.set(yticks=range(4),yticklabels=['TP 1','TP 2','TP 4','TP 8'],xlim=(0,32),xlabel='一次 decode 的时间（ms）');a.invert_yaxis();a.legend(ncol=2,frameon=False);save(f,'10-tp-time')
        d=data['collectives'];f,a=plot(3.7)
        for key,label,c in [('ring_seconds','环','#267398'),('tree_seconds','二项树','#a56c28')]:a.loglog(d['message_bytes'],np.array(d[key])*1e6,label=label,color=c)
        a.set(xlabel='每卡输入大小（bytes）',ylabel='通信时间（μs）');a.legend(frameon=False);save(f,'11-collectives')
        d=data['concurrency']['measured_ms']
        for concurrent,name in [(False,'12-resources'),(True,'resources-concurrent')]:
            f,a=plot(3.5,left=.24)
            comm=d['shared_comm' if concurrent else 'independent_comm'];comp=d['shared_compute' if concurrent else 'independent_compute']
            for i in range(2):
                a.barh(i-.16,comm[i],height=.28,color=COL['orange'],edgecolor=COL['line']);a.barh(i+.16,comp[i],height=.28,color=COL['blue'],edgecolor=COL['line'])
                a.text(comm[i]+.6,i-.16,f'{comm[i]+1e-9:.2f}',va='center',fontsize=11);a.text(comp[i]+.6,i+.16,f'{comp[i]+1e-9:.2f}',va='center',fontsize=11)
            from matplotlib.patches import Patch
            a.legend(handles=[Patch(facecolor=COL[c],edgecolor=COL['line'],label=l) for c,l in [('orange','通信'),('blue','矩阵乘')]],frameon=False,loc='lower right')
            a.set(yticks=[0,1],yticklabels=['4 MiB','64 MiB'],xlim=(0,56),xlabel='同时开始后的时间（ms）' if concurrent else '单独运行的时间（ms）');a.invert_yaxis();save(f,name)
        f,a=canvas(4.6)
        for row,(down,up) in enumerate([(32,32),(48,16)]):
            y=.62-row*.43;text(a,.04,y+.26,f'{down} 下联 + {up} 上联 = 64 个端口',13)
            for i in range(64):box(a,.045+(i%32)*.0292,y+(1-i//32)*.075,.022,.055,'','blue' if i<down else 'orange')
            text(a,.5,y-.075,f'下联 {down*25} GB/s → 上联 {up*25} GB/s',12,ha='center')
        save(f,'13-ports')
        patterns=data['physical_paths']['collective_patterns']
        for i,p in enumerate(patterns):
            f,a=canvas(4.8);angles=np.linspace(np.pi/2,np.pi/2-2*np.pi,16,endpoint=False);coords=np.array([[.5+.34*np.cos(t),.49+.35*np.sin(t)] for t in angles]);a.plot(*np.vstack([coords,coords[:1]]).T,color='#999999')
            for j,(x,y) in enumerate(coords):a.plot(x,y,'o',color='#267398',ms=4);text(a,.5+(x-.5)*1.17,.49+(y-.49)*1.17,str(j),11,ha='center')
            route=p['rounds'][2]['routes'][0]
            for u,v in route['path']:arrow(a,coords[u],coords[v])
            text(a,.5,.96,('递归' if i==0 else 'Swing')+f'：第三轮 0 → {route["receiver"]}',14,ha='center');save(f,'14-topology' if i==0 else 'topology-swing')
        f,a=plot(3.5)
        for i,p in enumerate(patterns):a.bar(np.arange(3)+(i-.5)*.32,[r['peak_link_bytes']/2**20 for r in p['rounds'][:3]],width=.32,color=COL['blue' if i==0 else 'orange'],edgecolor=COL['line'],label='递归' if i==0 else 'Swing')
        a.set(xticks=range(3),xticklabels=['第 1 轮','第 2 轮','第 3 轮'],ylim=(0,5),ylabel='最忙单向链路传输量（MiB）');f.subplots_adjust(top=.84);a.legend(frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.45,1.17),columnspacing=1.5,handlelength=1.4);save(f,'topology-load')
        f,a=canvas(4.2);box(a,.04,.36,.32,.33,'左半部','blue');box(a,.64,.36,.32,.33,'右半部','green');arrow(a,(.36,.52),(.64,.52));text(a,.5,.76,'中间切面：k² 条链路',12,ha='center');a.plot([.20,.20,.80,.80],[.36,.18,.18,.36],color='#454545');text(a,.5,.07,'首尾连接再切一次：另有 k² 条',12,ha='center');save(f,'15-torus')
        f,a=plot(3.4)
        for shift,vals,label,col in [(-.17,[1,8],'加速器数','blue'),(.17,[1,4],'二分链路数','orange')]:a.bar(np.arange(2)+shift,vals,width=.32,label=label,color=COL[col],edgecolor=COL['line'])
        a.set(xticks=[0,1],xticklabels=['k = 4','k = 8'],ylim=(0,10),ylabel='相对 k = 4 的倍数');a.legend(frameon=False);save(f,'torus-growth')
        f,a=canvas(5.4)
        for row,(title,left,middle,right) in enumerate([('GB200 NVL72','计算托盘','NVLink\n交换','计算托盘'),('TPU v4','64 芯片\n电互联单元','光电路\n交换机','64 芯片\n电互联单元'),('Unified Bus','主机与\n计算资源','UB\n互联','主机与\n内存资源')]):
            y=.65-row*.29;text(a,.03,y+.23,title,14)
            for x,label,col in [(.03,left,'blue'),(.38,middle,'gray'),(.73,right,'green')]:box(a,x,y,.24,.17,label,col,11)
            arrow(a,(.27,y+.085),(.38,y+.085));arrow(a,(.62,y+.085),(.73,y+.085))
        save(f,'16-systems')
        for after,name in [(False,'17-pool-placement'),(True,'pool-after')]:
            f,a=plot(3.8)
            for i,v in enumerate([80,60,40,40]):a.bar(i,v,color=COL[['blue','green','orange','purple'][i]],edgecolor=COL['line'],width=.6)
            if after:a.bar(1,20,bottom=60,color=COL['blue'],edgecolor=COL['line'],width=.6);a.text(1,70,'20',ha='center',va='center',fontsize=12)
            else:a.text(.05,95,'任务 0 另需 20 GB，尚未分配',fontsize=12)
            a.axhline(80,ls='--',color='#777777');a.set(xticks=range(4),xticklabels=[f'节点 {i}' for i in range(4)],ylim=(0,105),ylabel='已占用物理内存（GB）');save(f,name)
        f,a=canvas(4.6);text(a,.04,.94,'在途请求：已经发出，尚未收到结果',13)
        for i in range(4):
            y=.70-i*.15;box(a,.04,y,.21,.10,f'请求 {i}','blue',11);arrow(a,(.25,y+.05),(.72,y+.05));box(a,.73,y,.23,.10,'256 字节','green',11)
        text(a,.5,.09,'往返 7.52 μs；允许 128 个请求同时在途',12,ha='center');save(f,'18-read-window')
        d=data['remote_memory'];f,a=plot(3.7);a.loglog(d['frequency_per_second'],d['mean_payload_GBs'],color='#267398',label='20 GB × 读取频率');a.axhline(d['path_GBs'],color='#a56c28',label='路径：50 GB/s');a.axhline(d['window_bound_GBs'],ls='--',color='#388768',label='在途窗口：4.36 GB/s');a.set(xlabel='每秒完整读取次数',ylabel='平均带宽需求（GB/s）');a.legend(frameon=False,loc='upper left');save(f,'19-memory-pool')
        cands=data['continuous_execution']['candidates'];cap=data['continuous_execution']['capacity']
        f,a=plot(3.6,left=.20);left=np.zeros(4)
        for key,label,col in [('weights','权重','blue'),('kv','KV','green'),('workspace','工作区','orange')]:
            v=np.array([cap[str(c['tp'])][key]*(c['sessions_per_instance'] if key=='kv' else 1)/1e9 for c in cands]);a.barh(range(4),v,left=left,height=.5,color=COL[col],edgecolor=COL['line'],label=label);left+=v
        for j,t in enumerate(left):a.text(t+1.5,j,f'{t:.2f}',va='center',fontsize=11)
        a.axvline(80,ls='--',color='#777777');a.set(yticks=range(4),yticklabels=['TP 1 × 8','TP 2 × 4','TP 4 × 2','TP 8 × 1'],xlim=(0,118),xlabel='每卡内存需求（GB）');a.invert_yaxis();f.subplots_adjust(top=.84);a.legend(ncol=3,frameon=False,loc='lower center',bbox_to_anchor=(.5,1.0));save(f,'session-capacity')
        for i,c in enumerate(cands):
            if not c['capacity_fits']:continue
            f,a=plot(3.6,left=.20);instances=c['instances'];service=c['service_ms'];shown=min(instances,4)
            for req in range(4):
                lane=req%instances;start=req//instances*service;a.barh(lane,service,left=start,height=.52,color=COL[['blue','green','orange','purple'][req]],edgecolor=COL['line']);a.text(start+service/2,lane,f'会话 {req}',fontsize=11,ha='center',va='center')
            a.axvline(130,ls='--',color='#777777');a.set(yticks=range(shown),yticklabels=[f'实例 {j}' for j in range(shown)],xlim=(0,180),ylim=(-.6,shown-.4),xlabel='从四会话到达起计时（ms）');a.invert_yaxis();save(f,f'session-tp{c["tp"]}')
        for phase,name in [('healthy','21-scale-cost'),('fault','scale-cost-fault')]:
            f,a=plot(3.8)
            for c,col in zip(data['deadline_curves'][phase],['#267398','#388768','#a56c28','#86649b']):
                if c['capacity_fits']:a.step(c['deadlines_ms'],[np.nan if v is None else v for v in c['cost_per_valid']],where='post',label=f'TP{c["tp"]}',color=col)
            a.axvline(130,ls='--',color='#777777');a.set(xlim=(80,240),ylim=(0,.8),xlabel='完成期限（ms）',ylabel='每个按时会话（GPU·s）');a.legend(ncol=3,frameon=False);save(f,name)
        # 6.7.4: V4.1 Flash decode throughput per card against supernode size, and the token time once weights leave HBM.
        import json
        from pathlib import Path
        si=json.loads((Path(here).parents[1]/'calculations/results/supernode-inference-book.json').read_text())
        rows=si['results'];rdma=si['rdma']
        f,a=plot(3.7,left=.17);f.subplots_adjust(top=.84)
        a.plot(range(4),[r['tokens_per_s_per_gpu']/1e3 for r in rows],marker='o',color='#267398',label='实例在一个超节点内')
        a.plot([1],[rdma['tokens_per_s_per_gpu']/1e3],marker='s',linestyle='none',color='#a56c28',label='64 卡实例跨八台服务器')
        for i,r in enumerate(rows[:2]):a.annotate(f"每卡 {r['sessions']} 个会话",(i,r['tokens_per_s_per_gpu']/1e3),xytext=(8,-14 if i==0 else 6),textcoords='offset points',fontsize=11)
        a.set(xticks=range(4),xticklabels=['8','64','128','256'],xlabel='每超节点卡数',ylabel='每卡 decode 吞吐（千 token/s）',ylim=(0,24))
        a.legend(frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.5,1.0));a.grid(axis='y',alpha=.15);save(f,'supernode-inference')
        # Three placements of weights and KV: shared HBM, ROM + HBM, ROM + on-chip SRAM.
        f,a=canvas(3.9)
        cols=[('GPU 超节点',.03,.22,[('HBM\n权重 + KV','blue',.34)]),('ROM 晶圆 + HBM',.29,.33,[('ROM\n权重','gray',.34),('HBM\nKV','green',.34)]),('ROM 晶圆 + SRAM',.66,.33,[('ROM\n权重','gray',.34),('片上 SRAM\nKV','green',.18)])]
        for title,x,w,stores in cols:
            text(a,x+w/2,.95,title,13,ha='center')
            box(a,x+w/2-.08,.64,.16,.15,'计算','orange')
            n=len(stores);sw=(w-.03*(n-1))/n
            for j,(label,col,h) in enumerate(stores):
                sx=x+j*(sw+.03);box(a,sx,.48-h,sw,h,label,col,11);arrow(a,(sx+sw/2,.48),(x+w/2,.64))
        text(a,.5,.03,'箭头：每一步都要读取的数据；方框高度示意容量',11,ha='center')
        save(f,'weight-placement')
        # Where the Engram tables live: host memory, sharded HBM, or mask ROM; the lookup address is known before layer 0.
        f,a=canvas(4.2)
        box(a,.05,.86,.90,.11,'token 序列 → 哈希 → 每 token 48 行的地址，在第 0 层计算前就已确定','gray',11)
        cols=[('主机内存',.05,'DDR\n两张表 203 GB','gray',.26,'PCIe／RDMA 往返'),('超节点各卡 HBM',.375,'HBM 分片\n每卡 203 GB / S','green',.18,'NVLink 交换一轮'),('掩模 ROM',.70,'ROM\n约半片晶圆','gray',.34,'片内读取')]
        for title,x,label,col,h,path in cols:
            w=.25
            text(a,x+w/2,.78,title,12,ha='center')
            box(a,x+.01,.55,w-.02,.12,'第 1 层 Engram','orange',11)
            box(a,x,.46-h,w,h,label,col,11)
            arrow(a,(x+.05,.46),(x+.05,.55));text(a,x+.085,.505,path,11)
        save(f,'engram-placement')
        bars=[si['gpu_row'],si['rom']['rows'][2],si['rom']['rows'][0]];labels=['8 张 H100\n权重在 HBM','58 张 B200\n权重在 HBM','两片 ROM 晶圆\n权重在 ROM']
        f,a=plot(3.5,left=.26);f.subplots_adjust(top=.84)
        for j,(seg,col,name) in enumerate([('storage_compute_s','blue','存储与计算'),('link_s','orange','集合通信'),('fixed_s','gray','固定延迟')]):
            left=np.array([sum(b[k] for k in ['storage_compute_s','link_s','fixed_s'][:j]) for b in bars])*1e6
            a.barh(range(3),[b[seg]*1e6 for b in bars],left=left,height=.55,color=COL[col],edgecolor=COL['line'],label=name)
        for i,b in enumerate(bars):a.text(b['token_s']*1e6+60,i,f"{b['per_user_tokens_s']:,.0f} token/s",va='center',fontsize=11)
        a.set(yticks=range(3),yticklabels=labels,xlim=(0,3900),xlabel='单用户每 token 时间（μs）');a.invert_yaxis()
        a.legend(frameon=False,ncol=3,loc='lower center',bbox_to_anchor=(.5,1.0));save(f,'rom-token-time')
        # 6.6.3 & 6.5.2: Tiered memory prefetch/promotion break-even & 800 VDC + CPO rack scaling.
        f,a=canvas(6.4)
        text(a,.03,.965,'A. 分层内存两级流水瓶颈与预取暴露停顿（288 MiB 权重块）',12)
        box(a,.02,.75,.45,.16,'源端读取 + 24 GiB/s H2D\n100 / 40 GiB/s：11.72 ms\n8 GiB/s 冷层：35.16 ms','blue',11)
        box(a,.53,.75,.45,.16,'冷层 → 温层页迁移代价\n读出 8 GiB/s 需 35.16 ms\n占温层 288 MiB 与带宽','purple',11)
        arrow(a,(.47,.83),(.53,.83))
        box(a,.02,.54,.45,.16,'预取提前量与暴露停顿\n16 ms 窗：40 GiB/s 停顿 0\n8 GiB/s 暴露 19.16 ms','orange',11)
        box(a,.53,.54,.45,.16,'回本阈值与 PACT / Camp\n每次省 19.16 ms → 需 ≥ 2 次\n按 1/MLP 关键度触发迁移','green',11)
        arrow(a,(.245,.75),(.245,.70))
        arrow(a,(.755,.75),(.755,.70))
        arrow(a,(.47,.62),(.53,.62))
        a.plot([.02,.98],[.49,.49],color=COL['line'],ls='--',lw=1.0)
        text(a,.03,.445,'B. 500 kW 超节点机柜：800 VDC 高压配电与光电互连能耗边界',12)
        box(a,.02,.24,.45,.16,'传统 50 V 直流母排（500 kW）\n母排电流 I = 10,000 A\n0.5 mΩ 铜损 50 kW（10%）','orange',11)
        box(a,.53,.24,.45,.16,'800 VDC 高压母排（电压 ×16）\n母排电流 I = 625 A（÷16）\n铜损 195.3 W（降 256 倍）','green',11)
        arrow(a,(.47,.32),(.53,.32))
        box(a,.02,.03,.45,.16,'224G 铜缆 vs 可插拔光模块\n柜内铜缆（<2 m）：~5 pJ/bit\n可插拔光模块：15–20 pJ/bit','blue',11)
        box(a,.53,.03,.45,.16,'共封装光学 CPO（跨柜超节点）\n硅光引擎入封装：5–9 pJ/bit\n单机柜省下数十 kW 功耗','purple',11)
        arrow(a,(.47,.11),(.53,.11))
        out.save(f,'ch06-memory-tiering-prefetch-and-800vdc-cpo')
    out.finish();return out.outputs,out.checks
