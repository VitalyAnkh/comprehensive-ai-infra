"""Placement first, state handoff second, resource rates last."""
from fractions import Fraction
import numpy as np
import matplotlib.pyplot as plt
from figure_style import COL,STYLE,canvas,plot,text,box,arrow,Exporter

def draw(here,data):
    out=Exporter(here)
    def save(f,n):out.save(f,'figure-9-'+n)
    with plt.rc_context(STYLE):
        for i,title in enumerate(['多卡共同执行一个完整模型','完整副本分别接收请求','PD：按输入处理与生成阶段分工','AF：按每层的算子分工']):
            f,a=canvas(3.7);text(a,.04,.94,title,14)
            if i==0:
                box(a,.04,.22,.92,.48,'','gray');text(a,.5,.61,'一个调度器，一组模型权重与状态',12,ha='center')
                for j in range(8):box(a,.06+j*.112,.32,.09,.12,str(j),'blue',11)
            elif i==1:
                for x in [.04,.56]:box(a,x,.25,.40,.26,'完整模型副本','blue');text(a,x+.2,.72,'独立请求',12,ha='center');arrow(a,(x+.2,.65),(x+.2,.51))
            else:
                for x,label,c in [(.04,'P：处理输入' if i==2 else '注意力','blue'),(.61,'D：逐步生成' if i==2 else 'FFN／专家','green')]:box(a,x,.28,.35,.27,label,c)
                arrow(a,(.39,.46),(.61,.46));text(a,.5,.68,'传输上下文 KV' if i==2 else '逐层传递激活',12,ha='center')
                if i==3:arrow(a,(.61,.35),(.39,.35))
            save(f,'1-organization' if i==0 else f'organization-{i}')
        f,a=canvas(4.3)
        for j,l in enumerate(['P','D','工具等待','下一轮 P']):box(a,.20+j*.19,.73,.18,.15,l,'gray',11)
        for row,(l,start,c) in enumerate([('权重',.20,'blue'),('KV',.20,'green'),('可复用 EC',.20,'purple')]):
            y=.48-row*.17;text(a,.02,y+.05,l,11);box(a,start,y,.75,.10,'',c)
        text(a,.5,.04,'阶段宽度示意顺序；等待期间状态仍保留',11,ha='center');save(f,'2-state')
        for stage,name in [(0,'kv-residency'),(1,'kv-publish'),(2,'kv-release')]:
            if stage==0:
                # Two states of the same 8K context; box widths follow the byte counts (1.125 GiB versus 549 MiB).
                f,a=canvas(4.6);text(a,.04,.95,'传输开始：两端都占用完整空间',13)
                for row,(model,size,w) in enumerate([('Qwen3-8B，GQA','1.125 GiB',.35),('DeepSeek-V3，紧凑 MLA','549 MiB',.17)]):
                    y=.52-row*.38;text(a,.04,y+.30,model,12)
                    box(a,.04,y,w,.22,'源 P\n'+size,'blue');box(a,.96-w,y,w,.22,'目的 D\n'+size,'orange');arrow(a,(.04+w,y+.11),(.96-w,y+.11),'data')
                text(a,.5,.05,'数据复制中；方框宽度按字节数比例',12,ha='center');save(f,name);continue
            f,a=canvas(3.9);text(a,.04,.94,['传输开始：两端都占用完整空间','传输完成：通知目的端可以使用 KV','源端释放缓冲区，D 继续生成'][stage],13)
            box(a,.04,.35,.35,.29,'源 P\n1.125 GiB' if stage<2 else '源 P\n已释放','blue' if stage<2 else 'gray');box(a,.61,.35,.35,.29,'目的 D\n1.125 GiB','orange' if stage==0 else 'green');arrow(a,(.39,.5),(.61,.5),'data' if stage==0 else 'control');text(a,.5,.15,'数据复制中' if stage==0 else '完成标记建立使用顺序' if stage==1 else '下一请求可使用源端空间',12,ha='center');save(f,name)
        f,a=canvas(4.4)
        r=data['9-3']['per_card_rates']
        for row,(title,per,c) in enumerate([('P 池：四张 A100',r['A100']['prefill'],'blue'),('D 池：四张 H20',r['H20']['decode'],'green')]):
            y=.61-row*.40;text(a,.04,y+.23,title,14)
            for i in range(4):box(a,.04+i*.235,y,.21,.16,f'{per:.2f} 请求/s',c,11)
            text(a,.5,y-.09,f'池能力：{4*per:.2f} 请求/s',12,ha='center')
        save(f,'pd-layout')
        f,a=plot(4.5,left=.17);matrix=np.zeros((5,5))
        for row in data['9-3']['assignments']:
            x=row['prefill_workers']['A100'];y=row['prefill_workers']['H20'];matrix[y,x]=float(Fraction(row['bound_requests_per_second_exact']))
        a.imshow(matrix,origin='lower',cmap='Blues',vmin=0,vmax=10,aspect='equal')
        for y in range(5):
            for x in range(5):a.text(x,y,f'{matrix[y,x]:.2f}',fontsize=11,ha='center',va='center')
        a.add_patch(plt.Rectangle((3.5,-.5),1,1,fill=False,edgecolor='#a56c28',lw=2.2))
        a.set(xticks=range(5),yticks=range(5),xlabel='分给 P 的 A100 数量',ylabel='分给 P 的 H20 数量');save(f,'3-pd')
        f,a=canvas(4.9)
        for row,(na,nh) in enumerate(zip(data['new-allocation']['prefill_A100'],data['new-allocation']['prefill_H20'])):
            y=.70-row*.31;text(a,.04,y+.18,['推理请求：1025 个输出','前缀命中 6144 个 token','输出减至 129 个 token'][row],13)
            for i in range(8):
                isp=i<na if i<4 else i-4<nh
                box(a,.035+i*.117,y,.10,.12,('A100' if i<4 else 'H20')+'\n'+('P' if isp else 'D'),'blue' if isp else 'green',11)
            text(a,.5,y-.045,f'请求率上限 {data["new-allocation"]["rates"][row]:.2f}/s',11,ha='center')
        save(f,'4-allocation')
        for local,name in [(False,'5-local'),(True,'local-cpu')]:
            f,a=canvas(4.4);box(a,.04,.67,.35,.18,'CPU 主存：专家权重','blue',11);box(a,.61,.67,.35,.18,'GPU：输入激活','green',11)
            if local:
                arrow(a,(.61,.76),(.39,.76));box(a,.04,.31,.35,.18,'CPU 专家计算','blue');arrow(a,(.215,.67),(.215,.49));arrow(a,(.39,.40),(.61,.40));box(a,.61,.31,.35,.18,'GPU 汇合结果','orange',11)
            else:
                arrow(a,(.39,.76),(.61,.76));box(a,.61,.31,.35,.18,'GPU 专家计算','green');arrow(a,(.785,.67),(.785,.49))
            text(a,.5,.13,'激活往返：每行共 16 KiB' if local else '搬移一份专家权重：36 MiB',12,ha='center');save(f,name)
        d=data['9-6'];f,a=plot(3.9)
        for key,label,c,ls in [('cpu_avx512_ms','CPU，AVX-512','#267398','-'),('cpu_amx_ms','CPU，AMX','#267398','--'),('weight_copy_gpu_ms','搬权重到 GPU','#a56c28','-')]:a.plot(d['tokens_per_expert'],d[key],label=label,color=c,ls=ls)
        a.set_xscale('log',base=2);a.set(xlabel='每个专家收到的 token 数（对数刻度）',ylabel='八个专家的路径时间（ms）',xlim=(1,1024),ylim=(0,30),xticks=[1,4,16,64,256,1024],xticklabels=['1','4','16','64','256','1024']);a.minorticks_off();a.legend(frameon=False);save(f,'6-reuse')
        # The two expert-footprint rectangles repeated figure 6-19, so section 9.3.3 keeps only the prose recall.
        d=data['9-mla-handoff'];f,a=plot(3.8)
        for k,l,c in [('pd_gqa_ms','PD 一次交接：GQA 1.125 GiB','#267398'),('pd_mla_ms','PD 一次交接：紧凑 MLA 549 MiB','#388768'),('af_step_ms','AF 一步交接：72 次','#a56c28')]:a.plot(d['startup_us'],d[k],label=l,color=c)
        for key,c in [('gqa25','#267398'),('mla25','#388768')]:
            x=d['equal_time_startup_us'][key];y=589824/25e9*1e3+72*x/1e3
            a.scatter([x],[y],color=c,zorder=3);a.annotate(f'{x:.0f} μs',(x,y),xytext=(x+25,y-9),fontsize=11)
        a.set(xlabel='每次启动开销（μs）',ylabel='串行交接时间（ms）',xlim=(0,800),ylim=(0,80),yticks=[0,20,40,60,80]);a.legend(frameon=False,loc='upper left');save(f,'mla-handoff')
        for i,tasks in enumerate(data['new-balance']['assignments_per_card']):
            f,a=plot(4,left=.17);times=np.array(tasks)*2*18874368/(data['new-balance']['effective_TFLOPs']*1e12)*1e6;a.barh(range(8),times,color=COL['blue'],edgecolor=COL['line']);a.axvline(max(times),ls='--',color='#a56c28');a.set(yticks=range(8),yticklabels=[f'卡 {j}' for j in range(8)],xlim=(0,42),xlabel='H100 上的专家矩阵计算（μs）');a.invert_yaxis();save(f,'9-balance' if i==0 else 'balance-hotspot')
        d=data['9-10'];f,a=plot(3.6)
        for key,label,c in [('nvlink','同一台 HGX 内经 NVLink','#388768'),('cx7','跨服务器经 ConnectX-7','#a56c28')]:a.plot(d['batches'],d['net_saving_ms'][key],color=c,label=label)
        a.axhline(0,color='#777777');a.set(xlabel='热点持续的批数',ylabel='累计净节省（ms）');a.legend(frameon=False);save(f,'10-experts')
        for i,items in enumerate([data['new-overlap']['serial_items'],data['new-overlap']['pipeline_items']]):
            f,a=plot(3.5,left=.23)
            for row,start,dur,_ in items:a.barh(row,dur,left=start,height=.5,color=COL[['blue','green','orange'][row]],edgecolor=COL['line'])
            a.set(yticks=[0,1,2],yticklabels=['分派','专家计算','结果合并'],xlim=(0,.88),xlabel='时间（ms）');a.invert_yaxis();save(f,'11-overlap' if i==0 else 'overlap-pipeline')
        f,a=canvas(3.8);box(a,.04,.50,.34,.24,'目录记录\n标识 → 存储位置','orange',11);box(a,.62,.50,.34,.24,'KV 数据对象\n实际上下文状态','blue',11);arrow(a,(.38,.62),(.62,.62),'control');text(a,.5,.22,'先按目录定位，再确认对象可用并取回',12,ha='center');save(f,'cache-directory')
        f,a=canvas(4.5)
        for i in range(64):
            x=.04+i%16*.059;y=.72-i//16*.14;box(a,x,y,.05,.10,str(i+1),'orange' if i==63 else 'green',11)
        text(a,.5,.94,'读入 64 页，前 63 页可连续复用',14,ha='center');text(a,.5,.10,'绿色：1008 token；橙色：16 token 仍需处理',11,ha='center');save(f,'12-cache')
        d=data['new-route']
        for i,(queue,segs,compute) in enumerate(zip(d['queues_ms'],d['retrieval_segments_ms'],d['compute_ms'])):
            f,a=plot(3.0,left=.17);a.barh(0,queue,height=.45,color=COL['gray']);ready=0
            for dur,c in zip(segs,['gray','orange','blue']):a.barh(1,dur,left=ready,height=.45,color=COL[c],edgecolor=COL['line']);ready+=dur
            a.barh(0,compute,left=max(queue,ready),height=.45,color=COL['green']);a.set(yticks=[0,1],yticklabels=['GPU','取回'],xlim=(0,950),xlabel='从请求到达起计时（ms）');a.invert_yaxis();save(f,'13-route' if i==0 else f'route-{i}')
        d=data['new-migration'];f,a=plot(3.7);t=np.linspace(0,8,321);src=d['initial_GB']+d['growth_GBs']*t;a.plot(t,src,color='#a56c28',label='源端状态')
        for B,c,label in zip(d['copy_GBs'],['#388768','#267398'],['已复制：25 GB/s 网卡','已复制：50 GbE']):a.plot(t,np.minimum(B*t,src),color=c,label=label)
        a.set(xlabel='后台复制时间（s）',ylabel='累计状态（GB）',xlim=(0,8),ylim=(0,48));a.legend(frameon=False,loc='lower right');save(f,'14-migration')
        f,a=canvas(4.8)
        for row,title in enumerate(['已可靠记录的序列','故障前保存的 KV','恢复后继续生成']):
            y=.68-row*.27;text(a,.04,y+.20,title,13);box(a,.04,y,.36,.13,'输入 8192 token','blue',11)
            if row!=1:box(a,.43,y,.32,.13,'输出 1—1024','green',11);box(a,.78,y,.19,.13,'输出 1025','orange',11)
            else:text(a,.68,y+.065,'生成部分尚未保存',11,ha='center')
        text(a,.5,.03,'补算 1024 token → KV 到 9216 → 处理输出 1025',11,ha='center');save(f,'15-recovery')
        for pooled,name in [(False,'16-composition'),(True,'composition-pool')]:
            f,a=canvas(3.6);nodes=['P','共享池','D'] if pooled else ['P','D'];xs=[.04,.40,.76] if pooled else [.04,.76]
            for x,n in zip(xs,nodes):box(a,x,.39,.20,.22,n,'orange' if n=='共享池' else 'blue')
            for x1,x2 in zip(xs,xs[1:]):arrow(a,(x1+.20,.5),(x2,.5));text(a,(x1+.20+x2)/2,.71,'1.125 GiB',11,ha='center')
            text(a,.5,.18,'完整写入并通知 D 后，D 再读取' if pooled else 'P 直接将完整状态交给 D',12,ha='center');save(f,name)
        d=data['9-17'];f,a=plot(4.8,bottom=.36)
        for (key,q),c,label in zip(d['queues'].items(),['#388768','#267398','#a95159'],['直接 PD','理想分块共置','不分块共置']):a.plot(d['time_s'],q,color=c,label=f'{label} {key} 请求/s')
        a.axvline(10,ls='--',color='#777777');a.axvline(60,ls=':',color='#a95159');a.set(xlabel='从启动开始计时（s）',ylabel='积压请求数',xlim=(0,80),ylim=(0,70));a.legend(frameon=False,loc='upper center',bbox_to_anchor=(.42,-.2));save(f,'17-service')

        # Overhauled figure-9-chunked-vs-pd (Fig 9-9)
        f,a=canvas(5.8)
        text(a,.04,.94,'① 先算完再生成：长 Prefill 阻塞在途 Decode，单步算力闲置',12)
        box(a,.04,.69,.38,.17,'长 Prefill 独占 856 ms\n阻塞全部在途 Decode','orange',11)
        box(a,.45,.69,.24,.17,'访存 55.1 ms\n计算仅 4.2 ms','blue',11)
        box(a,.72,.69,.24,.17,'访存 55.1 ms\n约 92% 算力空转','gray',11)
        text(a,.04,.61,'② 分块 Prefill 捎带：在 55.1 ms 访存窗口内顺带算约 488 token',12)
        box(a,.04,.37,.44,.17,'合并批次 1：访存 55.1 ms\n32 条 Decode + 488 Prefill','green',11)
        box(a,.52,.37,.44,.17,'合并批次 2：访存 55.1 ms\n填满原本闲置的矩阵算力','green',11)
        text(a,.04,.29,'③ 严苛延迟下的 PD 分离：两池解耦，各司其职互不阻塞',12)
        box(a,.04,.05,.36,.17,'P 池（高算力卡）\n专注长输入 Prefill','blue',11)
        arrow(a,(.40,.135),(.56,.135),'data')
        text(a,.48,.22,'KV 交接',11,ha='center')
        box(a,.56,.05,.40,.17,'D 池（高带宽卡）\n稳定 55.1 ms/步生成','green',11)
        save(f,'chunked-vs-pd')

        # Overhauled figure-9-expert-footprint (Fig 9-17) with proper math italics
        f,(ax1,ax2)=plt.subplots(1,2,figsize=(420/72,4.1))
        f.subplots_adjust(left=.12,right=.97,bottom=.22,top=.83,wspace=.42)
        bars=[0.28125,4.43,4.50]
        ax1.bar([0,1,2],bars,color=[COL['green'],COL['orange'],COL['blue']],edgecolor=COL['line'],width=.56)
        for idx_b,val_b in enumerate(bars):
            ax1.text(idx_b,val_b+.18,f'{val_b:.2f} GiB',ha='center',fontsize=11)
        ax1.set(xticks=[0,1,2],xticklabels=['8 热点\n(64 行)','均匀路由\n(126 专家)','全触达\n(128 专家)'],ylim=(0,5.4),ylabel='单层权重读取量（GiB）')
        ax1.set_title('$M=64$（512 次分派）',fontsize=12,pad=8)
        m_vals=np.arange(1,129)
        exp_gib=128*(1-(1-8/128)**m_vals)*36/1024
        ax2.plot(m_vals,exp_gib,color='#267398',lw=2.2)
        ax2.axhline(4.50,ls='--',color='#a56c28',lw=1.2)
        for m_pt,dx,dy in [(1,10,.15),(16,10,-.70),(64,-6,-1.25)]:
            y_pt=128*(1-(1-8/128)**m_pt)*36/1024
            ax2.scatter([m_pt],[y_pt],color='#a56c28',zorder=3)
            ax2.annotate(f'$M={m_pt}$\n{y_pt:.2f} GiB',(m_pt,y_pt),xytext=(m_pt+dx,y_pt+dy),fontsize=11)
        ax2.set(xlabel='Decode 批大小 $M$',ylabel='期望读取权重（GiB）',xlim=(-6,136),ylim=(0,5.4),xticks=[1,32,64,96,128])
        ax2.set_title('独立路由期望触达体积',fontsize=12,pad=8)
        save(f,'expert-footprint')

        # New Path-A figure: ch09-pd-a100-h20-vs-af-ktransformers (Fig 9-49)
        f,a=canvas(5.8)
        text(a,.04,.95,'异构推理的两种切分：跨池 PD 分离 vs. 单机层内 AF 分离',13)
        box(a,.03,.76,.45,.13,'阶段级 PD 切分（A100 + H20）\n按请求阶段跨机解耦','blue',11)
        box(a,.03,.52,.45,.17,'A100 80GB 池（Prefill）\n算力密集：全部 Attention + FFN','blue',11)
        arrow(a,(.10,.52),(.10,.41),'data')
        text(a,.285,.465,'单次 1～1.125 GiB KV',11,ha='center')
        box(a,.03,.24,.45,.17,'H20 96GB 池（Decode）\n带宽密集：并发读取权重与 KV','green',11)
        box(a,.03,.04,.45,.15,'木桶：$\\min(n_P\\mu_P, n_D\\mu_D, B/V_{KV})$\n按层流水压至 1.34 ms 暴露项','gray',11)

        box(a,.52,.76,.45,.13,'层内 AF 切分（KTransformers）\nRTX 4090 + 双路 Xeon 6454S','orange',11)
        box(a,.52,.52,.45,.17,'RTX 4090 24GB（GPU 侧）\nAttention + KV + 共享/热点专家','green',11)
        arrow(a,(.58,.52),(.58,.41),'data')
        arrow(a,(.91,.41),(.91,.52),'data')
        text(a,.745,.465,'每步 64～72 次激活',11,ha='center')
        box(a,.52,.24,.45,.17,'双路 Xeon 1TB DDR5（CPU 侧）\nAMX 就地计算常驻稀疏 MoE 专家','orange',11)
        box(a,.52,.04,.45,.15,'免搬 36 MiB 权重，达 13.99 tok/s\n受限于启动开销 $\\alpha$ 与 NUMA 带宽','gray',11)
        out.save(f,'ch09-pd-a100-h20-vs-af-ktransformers')

        # New Path-B figure: ch09-tenant-cache-salt-and-ep-fault-radius (Fig 9-50, D11-07 / D14-04)
        f,a=canvas(6.4)
        text(a,.25,.95,'A. 多租户 cache_salt 隔离',12,ha='center')
        text(a,.745,.95,'B. 64 卡单 Rank 故障恢复',12,ha='center')
        a.plot([.495,.495],[.03,.97],color=COL['line'],ls='--',lw=1.0)

        box(a,.03,.76,.44,.14,'网关注入租户盐值 s_tenant\nh_0 = Hash(s_tenant || T_0)','blue',11)
        arrow(a,(.135,.76),(.135,.65),'data')
        arrow(a,(.365,.76),(.365,.65),'data')
        box(a,.03,.47,.215,.17,'租户 A\nh_0^A->h_1^A\n独立 KV 页','green',11)
        box(a,.255,.47,.215,.17,'租户 B\nh_0^B->h_1^B\n隔离 KV 页','orange',11)
        box(a,.03,.24,.44,.17,'阻断 TTFT 侧信道(856/30.9ms)\n防跨租户越权复用与探测','purple',11)
        box(a,.03,.04,.44,.15,'代价：公共前缀占 K·V_shared\n留存窗口缩窄，命中率下降','gray',11)

        box(a,.52,.76,.45,.14,'独立副本（8 × TP8，64 卡）\n单卡故障半径 12.5%（7 副本续服）','green',11)
        box(a,.52,.55,.45,.16,'紧耦合大 EP（EP64 All-to-All）\n单卡掉线阻塞全组：爆炸半径 100%','orange',11)
        box(a,.52,.31,.45,.18,'冗余专家热切换（EP64 -> EP63）\n路由切向热备副本，免冷加载权重\n停机 T_outage 12 s -> 0.8 s','blue',11)
        box(a,.52,.04,.45,.21,'四阶段恢复与排空时间线：\n检测 -> 重建 -> WAL/KV 重放\n排空 T_drain = λT_out/(μ-λ)','gray',11)
        out.save(f,'ch09-tenant-cache-salt-and-ep-fault-radius')

        # New Path-D figure 1: ch09-deepep-asymmetric-forwarding-and-bgda (Fig 9-51)
        f,a=canvas(6.4)
        text(a,.25,.95,'A. 高吞吐：RDMA->NVLink 转发',12,ha='center')
        text(a,.745,.95,'B. 低延迟：IBGDA 与 0-SM 轮询',12,ha='center')
        a.plot([.495,.495],[.03,.97],color=COL['line'],ls='--',lw=1.0)

        box(a,.03,.75,.44,.15,'源节点 GPU i：跨机专家 Token\n先经 RDMA 发往对等 GPU i','blue',11)
        arrow(a,(.25,.75),(.25,.66),'data')
        box(a,.03,.49,.44,.16,'目标机 GPU i：经机内 NVLink\n二次转发至本机专家 GPU j','green',11)
        box(a,.03,.27,.44,.17,'跨机 QP 扇出缩减 G=8 倍\n消除多卡争抢网卡与队头阻塞','orange',11)
        box(a,.03,.05,.44,.17,'DeepEP V2（SM90/SM100）：\nCX7 8×2 达 90/81 GB/s (12 SM)','gray',11)

        box(a,.52,.75,.45,.15,'GPU 线程直驱网卡门铃 (IBGDA)\n绕过 CPU 代理，单跳直达目标卡','blue',11)
        arrow(a,(.745,.75),(.745,.66),'data')
        box(a,.52,.49,.45,.16,'接收钩子 (Hook) 后台轮询完成\n零 SM 占用，让出算力给 MLA','green',11)
        box(a,.52,.27,.45,.17,'V2 ElasticBuffer + NCCL Gin：\n峰值 1.3×，SM 压降 4× (4-6 SM)','purple',11)
        box(a,.52,.05,.45,.17,'SwiftEP 整形 + cached_handle：\n消除微突发排队与解码期同步','gray',11)
        out.save(f,'ch09-deepep-asymmetric-forwarding-and-bgda')

        # New Path-D figure 2: ch09-mooncake-kv-centric-and-early-rejection (Fig 9-52)
        f,a=canvas(6.4)
        text(a,.25,.95,'A. Mooncake KV 中心解耦池',12,ha='center')
        text(a,.745,.95,'B. 预测性早期拒绝机制',12,ha='center')
        a.plot([.495,.495],[.03,.97],color=COL['line'],ls='--',lw=1.0)

        box(a,.03,.75,.44,.15,'Conductor 全局调度器：\n维护全局块哈希目录与水位','blue',11)
        arrow(a,(.25,.75),(.25,.66),'data')
        box(a,.03,.49,.44,.16,'多级 KV 池 (HBM+DRAM+NVMe)\nRDMA 异步存取与前缀路由','green',11)
        box(a,.03,.27,.44,.17,'vLLM 0.24 / Dynamo 1.3 边界：\n仅 HBM 发事件，防混合脏读','orange',11)
        box(a,.03,.05,.44,.17,'长上下文吞吐提升最高 525%\nSLO 下承载力显著优于共置','gray',11)

        box(a,.52,.75,.45,.15,'朴素准入陷阱：仅看 P 池空闲\n盲目接纳并耗费算力跑 Prefill','orange',11)
        arrow(a,(.745,.75),(.745,.66),'data')
        box(a,.52,.49,.45,.16,'交接至 D 池触发未来 KV 耗尽：\n在途生成增长致 OOM 驱逐重算','purple',11)
        box(a,.52,.27,.45,.17,'Mooncake 预测性准入判据：\n预估未来窗 [t, t+T] 的 KV 水位','green',11)
        box(a,.52,.05,.45,.17,'超上限 M_D_max 则 P 前拒绝\n或延后准入，零无效 Prefill','blue',11)
        out.save(f,'ch09-mooncake-kv-centric-and-early-rejection')

        # New Path-E3 figure 1: ch09-eplb-shadow-expert-warmup-atomic-swap (Fig 9-53)
        f,a=canvas(6.4)
        text(a,.25,.95,'A. 影子专家后台预热与权重复制',12,ha='center')
        text(a,.745,.95,'B. 路由表双缓冲原子切换与排空',12,ha='center')
        a.plot([.495,.495],[.03,.97],color=COL['line'],ls='--',lw=1.0)

        box(a,.03,.75,.44,.15,'滑窗统计专家命中直方图\n识别超阈值热点专家 (4× 负载)','orange',11)
        arrow(a,(.25,.75),(.25,.66),'data')
        box(a,.03,.49,.44,.16,'在空闲卡冗余槽分配影子缓冲\n异步 RDMA/NVLink 复制 36 MiB','blue',11)
        box(a,.03,.27,.44,.17,'复制期 (0.62~5.32 ms) 零阻塞\n前台仍按旧路由表 v_t 分发','green',11)
        box(a,.03,.05,.44,.17,'影子副本校验与算子预热就绪\n标记 ShadowSlot 状态 READY','gray',11)

        box(a,.52,.75,.45,.15,'构造新路由表 v_{t+1}：\n热点专家按权重轮询多副本','blue',11)
        arrow(a,(.745,.75),(.745,.66),'data')
        box(a,.52,.49,.45,.16,'微批边界原子交换路由表指针\n新批次立即读取 v_{t+1} 分流','green',11)
        box(a,.52,.27,.45,.17,'旧版本在途 Combine 引用计数\n等待在途微批归并 (ref=0)','purple',11)
        box(a,.52,.05,.45,.17,'安全回收旧冷门副本冗余槽位\n全程零停机、零丢包、等价','gray',11)
        out.save(f,'ch09-eplb-shadow-expert-warmup-atomic-swap')

        # New Path-E3 figure 2: ch09-layerwise-pipelined-kv-transfer-ttft-hiding (Fig 9-54)
        f,a=canvas(6.4)
        text(a,.25,.95,'A. 整份串行 vs 按层流水 KV 交接',12,ha='center')
        text(a,.745,.95,'B. 暴露延迟压缩与双轨拥塞回退',12,ha='center')
        a.plot([.495,.495],[.03,.97],color=COL['line'],ls='--',lw=1.0)

        box(a,.03,.75,.44,.15,'传统整份串行交接：\n32/36 层算完再发 1~1.125 GiB','orange',11)
        box(a,.03,.53,.44,.16,'串行暴露交接耗时：\n25 GB/s 下等待 42.95~48.32 ms','orange',11)
        box(a,.03,.30,.44,.17,'按层流水异步 RDMA 交接：\n算完第 l 层即发 32 MiB KV','blue',11)
        box(a,.03,.05,.44,.19,'前 L-1 层传输隐藏于计算中\n仅末层 1.35 ms 暴露于 TTFT','green',11)

        box(a,.52,.75,.45,.15,'计算掩盖判据：\n单层计算 t_P,l >= 传输 t_X,l','blue',11)
        box(a,.52,.53,.45,.16,'双轨 RDMA 负载分担：\n奇偶层交错走 Rail-0 / Rail-1','green',11)
        box(a,.52,.30,.45,.17,'网络拥塞或接收槽满回退：\n合并 k 层聚批发送防队头阻塞','purple',11)
        box(a,.52,.05,.45,.19,'细粒度完成位图 (Bitmask)：\n全层置 1 后 D 池即启动首步','gray',11)
        out.save(f,'ch09-layerwise-pipelined-kv-transfer-ttft-hiding')
    from v41_case_figures import draw as draw_v41
    draw_v41(9, out)
    out.finish();return out.outputs,out.checks
