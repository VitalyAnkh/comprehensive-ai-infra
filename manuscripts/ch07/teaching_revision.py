"""Follow a shard across servers, request slots and consumer dependencies."""
import json, math
import numpy as np
from fractions import Fraction
import matplotlib.pyplot as plt
from figure_style import COL,STYLE,canvas,plot,text,box,arrow,Exporter

def draw(here,data):
    out=Exporter(here)
    def save(f,n):out.save(f,'figure-7-'+n)
    with plt.rc_context(STYLE):
        # 1: two eight-GPU servers, NVSwitch inside, one NIC per GPU, rail switches between.
        f,a=canvas(5.6)
        for row in range(2):
            top=row==0
            base=.615 if top else .03
            box(a,.02,base,.96,.355,'','gray')
            text(a,.05,.945 if top else .055,f'服务器 {"AB"[row]}',13)
            sw_y=base+.24 if top else base+.055
            gpu_y=base+.14 if top else base+.155
            nic_y=base+.04 if top else base+.255
            box(a,.05,sw_y,.90,.06,'NVSwitch：每卡 NVLink 每方向 450 GB/s','green',11)
            for i in range(8):
                x=.05+i*.1125
                box(a,x,gpu_y,.095,.08,str(row*8+i),'blue',11)
                box(a,x,nic_y,.095,.06,'NIC','orange',11)
                if top:arrow(a,(x+.0475,nic_y),(x+.0475,.545))
                else:arrow(a,(x+.0475,.455),(x+.0475,nic_y+.06))
        box(a,.02,.455,.96,.09,'交换网络：每条 rail 一台交换机，每张网卡每方向 50 GB/s','gray',11)
        save(f,'1-boundaries')
        d=data['7-2'];f,a=plot(3.6);n=np.array(d['device_multipliers']);c=np.array(d['compute_ms']);t=d['cut_ms'];a.plot(n,c+t,label='串行',color='#267398');a.plot(n,np.maximum(c,t),label='完全重叠',color='#388768');a.plot(n,c,ls=':',label='计算',color='#a56c28');a.set(xlabel='加速器数量／基准数量',ylabel='每步时间（ms）',xticks=n,ylim=(0,32));a.legend(frameon=False);save(f,'2-cut')
        for idx,order in enumerate(data['7-3']['ring_orders']):
            f,a=canvas(4.4);text(a,.04,.93,'连续环：十六条边只有两条跨服务器' if idx==0 else '交错环：十六条边都跨服务器',14)
            xs=[.03+i*.118 for i in range(8)];pos={}
            for i,rank in enumerate(order):
                col=i if i<8 else 15-i;y=.60 if i<8 else .28;pos[i]=(xs[col],y)
                box(a,xs[col],y,.082,.14,str(rank),'blue' if rank<8 else 'orange',12)
            for i in range(16):
                (x1,y1),(x2,y2)=pos[i],pos[(i+1)%16]
                if i<7:arrow(a,(x1+.082,y1+.07),(x2,y2+.07))
                elif i==7:arrow(a,(x1+.041,y1),(x2+.041,y2+.14))
                elif i<15:arrow(a,(x1,y1+.07),(x2+.082,y2+.07))
                else:arrow(a,(x1+.041,y1+.14),(x2+.041,y2))
            if idx==0:text(a,xs[7]+.026,.51,'跨服务器',11,ha='right');text(a,xs[0]+.056,.51,'跨服务器',11)
            text(a,.5,.12,'蓝：服务器 A；橙：服务器 B',12,ha='center');save(f,'3-hierarchy' if idx==0 else 'ring-interleaved')
        f,a=canvas(5.2)
        for col in range(2):
            x=.04+.51*col;text(a,x+.20,.93,f'服务器 {"AB"[col]}',14,ha='center');box(a,x,.64,.41,.17,'本地 ReduceScatter\n得到八个 24 MiB 分片','blue',11);box(a,x,.35,.41,.16,'对应分片跨服务器\n两卡 AllReduce，各走一条 rail','orange',11);box(a,x,.06,.41,.16,'本地 AllGather\n得到完整归约结果','green',11);arrow(a,(x+.20,.64),(x+.20,.51));arrow(a,(x+.20,.35),(x+.20,.22))
        arrow(a,(.45,.46),(.55,.46));arrow(a,(.55,.39),(.45,.39));save(f,'hierarchy-stages')
        f,a=plot(3.6,left=.22);left=np.zeros(3)
        for k,l,c in [('local_MiB','本地','blue'),('remote_MiB','跨服务器','orange')]:v=data['7-3'][k];a.barh(range(3),v,left=left,color=COL[c],edgecolor=COL['line'],label=l);left+=v
        a.set(yticks=range(3),yticklabels=['连续环','交错环','分层归约'],xlabel='逻辑发送量，两方向合计（MiB）');a.invert_yaxis();f.subplots_adjust(top=.84);a.legend(frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.45,1.17),columnspacing=1.5,handlelength=1.4);save(f,'hierarchy-bytes')
        f,a=plot(3.6,left=.18)
        for stage in range(4):
            for batch in range(8):a.barh(stage,1,left=stage+batch,height=.7,color=COL[['blue','green','orange','purple'][batch%4]],edgecolor=COL['line']);a.text(stage+batch+.5,stage,str(batch),ha='center',va='center',fontsize=11)
        a.set(yticks=range(4),yticklabels=[f'阶段 {i}' for i in range(4)],xlim=(0,11),xticks=[0,2,4,6,8,10,11],xlabel='时间（ms）；格内为 micro-batch 编号');a.invert_yaxis();save(f,'4-pipeline')
        for i in range(3):
            f,a=canvas(3.9);text(a,.04,.94,['集中到一张网卡','均分到八张独立网卡','对照配置：双端口网卡共用一个 PCIe 插槽'][i],14);box(a,.04,.42,.24,.24,'32 MiB','blue')
            if i==0:box(a,.64,.42,.30,.24,'网卡\n50 GB/s','orange');arrow(a,(.28,.54),(.64,.54))
            elif i==1:
                for j in range(8):
                    y=.10+j*.10;box(a,.70,y,.25,.08,'4 MiB','orange',11);arrow(a,(.28,.54),(.70,y+.04))
            else:
                box(a,.36,.32,.24,.42,'PCIe Gen4\nx16 插槽\n32 GB/s','gray',11);arrow(a,(.28,.54),(.36,.54))
                for j in range(2):
                    y=.29+j*.28;box(a,.72,y,.25,.16,'16 MiB','orange',11);arrow(a,(.60,.54),(.72,y+.08))
            text(a,.5,.04,f'接收阶段至少 {data["7-5"]["lower_ms"][i]:.2f} ms',13,ha='center');save(f,'5-expert' if i==0 else f'expert-path-{i}')
        f,a=canvas(4.8);box(a,.03,.44,.21,.18,'源 GPU','blue');box(a,.39,.72,.27,.15,'直连 50','green');box(a,.36,.17,.31,.19,'NVLink 450\n中继网卡 100','orange',11);box(a,.78,.40,.20,.25,'下游\n可用带宽','gray');arrow(a,(.24,.53),(.39,.79));arrow(a,(.24,.53),(.36,.26));arrow(a,(.66,.79),(.78,.52));arrow(a,(.67,.26),(.78,.52));text(a,.5,.05,'单位 GB/s；中继受借用网卡合计 100 限制',12,ha='center');save(f,'6-relay')
        paths=[('主机 RPC',['CPU A','网卡 A','网卡 B','CPU B']),('CPU 提交 GPUDirect RDMA',['GPU A','网卡 A','网卡 B','GPU B']),('GPU 经 NVLink 访问',['GPU A','NVLink','GPU B']),('设备发起 URMA 异步访问',['设备 A','UB 互联','设备 B'])]
        for i,(title,nodes) in enumerate(paths):
            f,a=canvas(3.8);text(a,.04,.94,title,14);step=.94/len(nodes)
            for j,node in enumerate(nodes):
                x=.03+j*step;box(a,x,.36,step-.045,.18,node,'blue' if j in [0,len(nodes)-1] else 'gray',11)
                if j<len(nodes)-1:arrow(a,(x+step-.045,.45),(x+step,.45))
            if i==1:box(a,.20,.69,.28,.12,'CPU 提交请求','orange',11);arrow(a,(.34,.69),(.38,.54),'control')
            else:text(a,.5,.73,'发起者提交操作，接收方按完成条件使用结果',11,ha='center')
            text(a,.5,.13,'实线：数据；虚线：控制提交',11,ha='center');save(f,'7-access' if i==0 else f'access-{i}')
        f,a=canvas(4.1)
        for i in range(2):
            y=.61-i*.43;box(a,.04,y,.31,.19,'远端快照','blue');box(a,.65,y,.31,.19,'消费者' if i==0 else '本地副本','green');arrow(a,(.35,y+.095),(.65,y+.095));text(a,.5,y+.25,'每次使用都跨网络读取' if i==0 else '搬回一次，再读取本地副本',13,ha='center')
        save(f,'snapshot-paths')
        for i,(label,c) in enumerate(data['7-8']['cases'].items()):
            f,a=plot(3.5);r=np.arange(1,6 if i==0 else 26);a.plot(r,c['remote_per_read_ms']*r,label='每次远程读取',color='#267398');a.plot(r,c['setup_ms']+c['local_per_read_ms']*r,label='先搬回本地',color='#388768');a.set(xlabel='复用次数',ylabel='累计时间（ms）');a.legend(frameon=False);save(f,'8-snapshot' if i==0 else 'snapshot-partial')
        f,a=canvas(4.0);text(a,.04,.94,'槽位保存一项尚未完成的请求',14)
        for x,label,c in [(.03,'分配槽位','blue'),(.37,'传输与等待','orange'),(.71,'处理通知\n释放槽位','green')]:box(a,x,.39,.26,.23,label,c,12)
        arrow(a,(.29,.5),(.37,.5));arrow(a,(.63,.5),(.71,.5));text(a,.5,.18,'占用时间：从分配到释放，共 2 μs',12,ha='center');save(f,'slot-lifetime')
        f,a=plot(3.5,left=.24)
        for y,(key,p) in enumerate(data['7-9']['periods'].items()):
            a.barh(y,4,height=.5,color=COL['gray'])
            for start,dur in p['segments_us']:a.barh(y,dur,left=start,height=.5,color=COL['blue'],edgecolor=COL['line'])
        a.set(yticks=[0,1],yticklabels=['128 个槽位','391 个槽位'],xlim=(0,4),xlabel='时间（μs）；蓝色为载荷发送');a.invert_yaxis();save(f,'9-window')
        f,a=plot(3.5);n=np.arange(1,501);a.plot(n,np.minimum(50,n*256/2000),label='路径与槽位限制',color='#267398');a.plot(n,np.minimum(data['7-9']['submission_cap_GBs'],n*256/2000),label='再加 18.6 ns 启动间隔',color='#a56c28');a.set(xlabel='活跃请求槽位数',ylabel='吞吐上限（GB/s）');a.legend(frameon=False);save(f,'window-rate')
        f,a=plot(3.8,left=.23);a.barh(0,5,color=COL['blue'],height=.48);a.barh(1,12,color=COL['orange'],height=.48);a.barh(1,4,left=8,color=COL['green'],height=.48);a.axvline(6,ls=':',color='#a54b43');a.text(6.2,-.45,'6 μs 提前覆盖',fontsize=11);a.set(yticks=[0,1],yticklabels=['源缓冲占用','目的缓冲占用'],xlim=(0,14),ylim=(-.7,1.6),xlabel='时间（μs）；绿色为消费者读取');a.invert_yaxis();save(f,'10-lifetime')
        f,a=canvas(4.5)
        for i in range(2):
            y=.63-i*.39;box(a,.04,y,.23,.18,f'端点 {i}','blue');box(a,.38,y,.24,.18,'关系绑定','orange');arrow(a,(.27,y+.09),(.38,y+.09));arrow(a,(.62,y+.09),(.77,.52))
        box(a,.77,.35,.20,.34,'目标\n传输状态','green',11);text(a,.5,.08,'端点身份分别保存，目标传输状态共享',12,ha='center');save(f,'11-state')
        f,a=plot(3.7,left=.26);left=np.zeros(3)
        for j,label,col in [(0,'端点','blue'),(1,'绑定','orange'),(2,'传输','green')]:v=np.array(data['7-11']['state_MiB'])[:,j];a.barh(range(3),v,left=left,height=.5,color=COL[col],edgecolor=COL['line'],label=label);left+=v
        a.axvline(1,ls='--',color='#777777');a.set(yticks=range(3),yticklabels=['逐关系独占','按目标共享','八类隔离'],xlabel='状态占用（MiB）',xlim=(0,10));a.invert_yaxis();a.legend(frameon=False);save(f,'state-capacity')
        for i,(label,schedule) in enumerate(data['7-12']['schedules'].items()):
            f,a=plot(3.6,left=.21)
            for t in schedule['tasks']:
                lane=1 if t['id']=='independent_transfer' else 0;color={'write_data':'blue','recover_and_make_visible':'orange','publish_notification':'green','independent_transfer':'purple'}[t['id']];a.barh(lane,t['duration_ns']/1000,left=t['start_ns']/1000,height=.45,color=COL[color],edgecolor=COL['line'])
            a.set(yticks=[0,1],yticklabels=['写入与发布','独立传输'],xlim=(0,115),xlabel='时间（μs）');a.invert_yaxis();save(f,'12-ordering' if i==0 else 'ordering-independent')
        f,a=canvas(5.0)
        for row,(time,label,c) in enumerate([(1,'提前读取：得到旧值','orange'),(2,'生产者写入新值','blue'),(3,'就绪标志可见','green'),(4,'消费者读到就绪标志','green'),(5,'返回先前读到的旧值','orange'),(6,'若在 4 μs 重读，此时得到新值','blue')]):
            y=.78-row*.14;text(a,.03,y+.04,f'{time} μs',11);box(a,.19,y,.76,.10,label,c,11)
        save(f,'13-stale')
        for i,(name,ops) in enumerate(data['7-14']['operations'].items()):
            f,a=plot(5,left=.17)
            for op in ops:
                y=op['operation'];start=op['submit_ns']/1000;end=op['transfer_complete_ns']/1000;done=op['completion_consumed_ns']/1000;a.barh(y,end-start,left=start,height=.65,color=COL['blue']);a.barh(y,done-end,left=end,height=.65,color=COL['orange'])
            a.set(xlim=(0,82),yticks=[0,4,8,12,15],ylabel='请求编号',xlabel='时间（μs）\n蓝：传输；橙：等待槽位释放');a.invert_yaxis();save(f,'14-reclaim' if i==0 else 'reclaim-more')
        for kind,n in [('arrival','15-congestion'),('queue','congestion-queue')]:
            f,a=plot(3.8)
            for (name,segs),label,c in zip(data['7-15']['periodic_segments'].items(),['重叠 20 ms','重叠 5 ms','不重叠'],['#267398','#a56c28','#388768']):
                xs=[];ys=[]
                for seg in segs:
                    xs.extend([seg['start_ns']/1e6,seg['end_ns']/1e6]);ys.extend([seg['arrival_bytes_per_second']/1e9]*2 if kind=='arrival' else [seg['queue_start_bytes']/1e6,seg['queue_end_bytes']/1e6])
                a.plot(xs,ys,label=label,color=c)
            a.set(xlabel='时间（ms）',ylabel='到达速率（GB/s）' if kind=='arrival' else '积压（MB）',xlim=(0,100));a.legend(frameon=False);save(f,n)
        f,a=canvas(3.8);box(a,.04,.42,.24,.20,'发送者\n100 GB/s','blue');box(a,.43,.42,.24,.20,'缓冲队列','orange');box(a,.78,.42,.19,.20,'出口\n50 GB/s','green',11);arrow(a,(.28,.52),(.43,.52));arrow(a,(.67,.52),(.78,.52));arrow(a,(.55,.42),(.17,.18),'control');text(a,.46,.12,'反馈生效后，发送降至 40 GB/s',12,ha='center');save(f,'feedback-loop')
        f,a=plot(3.8)
        for seg in data['7-16']['queue_segments']:a.plot([seg['start_ns']/1000,seg['end_ns']/1000],[seg['queue_start_bytes']/1024,seg['queue_end_bytes']/1024],color='#267398')
        a.axvline(20,ls='--',color='#a56c28');a.set(xlabel='时间（μs）',ylabel='队列占用（KiB）',xlim=(0,120),ylim=(0,580));save(f,'16-feedback')
        packet_cases=dict(data['7-17']['cases'])
        loss=json.loads((here.parents[1]/'calculations/results/packet-reorder-loss.json').read_text())
        packet_cases['packet-reorder-loss']={k:loss[k] for k in ['transmissions','delivery']}
        for i,(name,c) in enumerate(packet_cases.items()):
            f,a=plot(3.7);delivery={x['sequence']:float(Fraction(x['delivery_exact_ns']))/1000 for x in c['delivery']}
            for t in c['transmissions']:
                if t['lost']:continue
                arrival=float(Fraction(t['arrival_exact_ns']))/1000;seq=t['sequence'];a.plot([arrival,delivery[seq]],[seq,seq],color='#a56c28');a.scatter(arrival,seq,color='#267398',s=22)
            a.set(xlim=(0,[1.6,10,22][i]),yticks=range(8),ylabel='报文序号',xlabel='到达与交给应用的时刻（μs）');a.invert_yaxis();save(f,'17-packets' if i==0 else f'packets-{i}')
        f,a=canvas(3.6);box(a,.04,.40,.34,.25,'请求 0 持有 A\n等待 B','blue');box(a,.62,.40,.34,.25,'请求 1 持有 B\n等待 A','orange');arrow(a,(.38,.58),(.62,.58));arrow(a,(.62,.46),(.38,.46));text(a,.5,.16,'双方都需要对方先释放',13,ha='center');save(f,'18-deadlock')
        f,a=canvas(3.7)
        for x,label,col in [(.03,'请求资源','blue'),(.37,'执行资源','orange'),(.71,'独立响应\n资源','green')]:box(a,x,.42,.26,.24,label,col,12)
        arrow(a,(.29,.54),(.37,.54));arrow(a,(.63,.54),(.71,.54));text(a,.5,.18,'响应有预留通路，可返回并释放原请求',12,ha='center');save(f,'response-reserve')
        for i,(ready,ex) in enumerate([([0,0,0,2],.4),([0,0,0,2],.2),([0]*4,.4)]):
            f,a=plot(3.5,left=.18)
            for lane,t in enumerate(ready):
                a.barh(lane,t,height=.5,color=COL['gray']);a.barh(lane,max(ready)-t,left=t,height=.5,color=COL['orange']);a.barh(lane,ex,left=max(ready),height=.5,color=COL['blue']);a.scatter(t,lane,color='#252525',s=16)
            a.set(yticks=range(4),yticklabels=[f'参与者 {j}' for j in range(4)],xlim=(0,2.6),xlabel='时间（ms）；圆点为就绪');a.invert_yaxis();save(f,'19-progress' if i==0 else f'progress-{i}')
        for i,c in enumerate(data['7-20']['cases']):
            f,a=plot(2.9,left=.17);a.barh(0,20,height=.45,color=COL['gray']);a.barh(1,c['comm_ms'],left=c['ready_ms'],height=.45,color=COL['blue']);a.barh(0,2,left=c['update_start_ms'],height=.45,color=COL['green']);a.scatter(c['ready_ms'],1,color='#252525',s=16);a.set(yticks=[0,1],yticklabels=['计算／更新','通信'],xlim=(0,37),xlabel='时间（ms）');a.invert_yaxis();save(f,'20-step' if i==0 else f'step-{i}')
        f,a=plot(3.7);m=np.logspace(3,8,200)
        for alpha,b,label,col in [(5e-6,50e9,'原配置','#267398'),(5e-6,150e9,'带宽三倍','#388768'),(2e-6,50e9,'启动缩短','#a56c28')]:a.loglog(m,(14*alpha+1.75*m/b)*1e6,label=label,color=col)
        a.set(xlabel='每参与者输入（bytes）',ylabel='归约时间（μs）');a.legend(frameon=False);save(f,'21-message')
        # Switch network (2026-09-11): Clos cut, rail pairing, in-network sweep, ECMP collision, incast feedback.
        f,a=canvas(4.3);text(a,.04,.95,'两层 Clos：叶交换机接端点，脊交换机连接所有叶交换机',13)
        spx=[.24,.56];lx=[.03+i*.245 for i in range(4)]
        for j,x in enumerate(spx):box(a,x,.70,.20,.11,f'脊交换机 {j}','green',11)
        for i,x in enumerate(lx):
            box(a,x,.40,.20,.11,f'叶交换机 {i}','blue',11);box(a,x,.10,.20,.11,'32 个端点','gray',11)
            a.plot([x+.10,x+.10],[.21,.40],color=COL['line'],lw=1)
            for sx in spx:a.plot([x+.10,sx+.10],[.51,.70],color=COL['line'],lw=1)
        a.plot([.5,.5],[.06,.67],ls='--',color='#a54b43',lw=1.2);text(a,.505,.30,'半分割集',11,color='#a54b43')
        text(a,.5,.03,'64 端口：每叶 32 下行、32 上联；共 64 叶、32 脊、2048 个端点',11,ha='center');save(f,'clos-cut')
        f,a=canvas(6.2)
        from matplotlib.path import Path as MPath
        from matplotlib.patches import FancyArrowPatch
        for panel,(title,shifted) in enumerate([('对齐配对 rank i ↔ i+8：每一对留在自己的 rail 内',False),('错位配对 rank i ↔ i+9：每一对经脊交换机换到下一条 rail',True)]):
            y0=.52-panel*.50;text(a,.03,y0+.455,title,12);xs=[.03+i*.1175 for i in range(8)]
            box(a,.03,y0+.35,.94,.07,'脊交换机' if not shifted else '','green',11)
            if shifted:text(a,.5,y0+.385,'脊交换机',11,ha='center')
            for i,x in enumerate(xs):
                box(a,x,y0+.22,.105,.08,f'rail {i}','blue',11);box(a,x,y0+.05,.048,.07,str(i),'orange',11);box(a,x+.057,y0+.05,.048,.07,str(i+8),'purple',11)
                arrow(a,(x+.024,y0+.12),(x+.03,y0+.22))
                if shifted:
                    x2=xs[(i+1)%8];arrow(a,(x+.03,y0+.30),(x+.03,y0+.35));arrow(a,(x2+.075,y0+.35),(x2+.075,y0+.30));arrow(a,(x2+.075,y0+.22),(x2+.081,y0+.12))
                else:arrow(a,(x+.075,y0+.22),(x+.081,y0+.12))
            if shifted:
                # Two representative cross-rail paths drawn end to end: NIC i -> leaf i -> spine -> leaf i+1 -> NIC i+9.
                for i,col,label in [(0,'#a54b43','0 → 脊 → 9'),(5,'#267398','5 → 脊 → 14')]:
                    x=xs[i];x2=xs[(i+1)%8];xa=x+.012;xb=x2+.093
                    verts=[(xa,y0+.12),(xa,y0+.365),(xb,y0+.365),(xb,y0+.12)]
                    a.add_patch(FancyArrowPatch(path=MPath(verts),arrowstyle='-|>',mutation_scale=11,linewidth=1.6,color=col,shrinkA=0,shrinkB=0,zorder=5))
                    text(a,(xa+xb)/2,y0+.403,label,11,ha='center',color=col)
            text(a,.5,y0+.015,'橙：服务器 A 的网卡 0–7；紫：服务器 B 的网卡 8–15；每条 rail 一台叶交换机',11,ha='center')
        save(f,'rail-pairing')
        sw=data['7-in-network']['sweep'];f,a=plot(3.4,left=.15);xs=[r['servers'] for r in sw]
        a.plot(xs,[r['ring_seconds']*1000 for r in sw],marker='o',label='环形 AllReduce',color='#a56c28');a.plot(xs,[r['switch_seconds']*1000 for r in sw],marker='s',label='在网归约',color='#267398')
        a.set_xscale('log',base=2);a.set_xticks(xs,[str(x) for x in xs]);a.set(xlabel='持有同一分片的服务器数',ylabel='跨服务器阶段（ms）',ylim=(0,1.35));a.legend(frameon=False,loc='upper left')
        a.text(32,sw[-1]['ring_seconds']*1000+.06,f"{sw[-1]['ring_seconds']*1000:.2f} ms",ha='right',fontsize=11);a.text(32,sw[-1]['switch_seconds']*1000-.12,f"{sw[-1]['switch_seconds']*1000:.2f} ms",ha='right',fontsize=11);save(f,'in-network-sweep')
        ec=data['7-ecmp']['cases'];f,a=plot(3.5,left=.14,bottom=.26);ns=[8,32,128];x=np.arange(3);w=.36
        for j,(m,label,c) in enumerate([(32,'32 条上联（无阻塞叶）',COL['blue']),(16,'16 条上联（3:1 超售叶）',COL['orange'])]):
            ks=[f'{n}-on-{m}' for n in ns];vals=[ec[k]['expected_max_load']/math.ceil(ec[k]['flows']/ec[k]['uplinks']) if k in ec else None for k in ks]
            pos=[i+(j-.5)*w for i,v in enumerate(vals) if v is not None];a.bar(pos,[v for v in vals if v is not None],color=c,edgecolor=COL['line'],width=w*.92,label=label)
            for p,v in zip(pos,vals):
                if v is not None:a.text(p,v+.12,f'{v:.2f}',ha='center',fontsize=11)
        a.axhline(1,ls=':',color='#777777');a.set(xticks=x,xticklabels=[f'{n} 条流' for n in ns],ylabel='最忙上联流数／无冲突时',ylim=(0,4.3));a.legend(frameon=False,loc='upper right');save(f,'ecmp-collision')
        ic=data['7-incast'];f,a=plot(3.6,left=.16);t=np.linspace(0,4,401);free=ic['free_buffer_bytes']
        for row,c in zip(ic['rows'],['#267398','#388768','#a56c28']):
            ex=float(Fraction(row['excess_bytes_per_second_exact']));a.plot(t,np.minimum(ex*t*1e-6,free)/1024,color=c,label=f"N = {row['N']}");ta=row['allowed_feedback_ns']/1000;a.scatter([ta],[free/1024],color=c,s=28,zorder=4);a.text(ta+.06,free/1024-140,f'{ta:.2f} μs',fontsize=11,color=c)
        a.axhline(free/1024,ls='--',color='#777777');a.text(2.6,free/1024+40,'1 MiB 剩余缓冲',fontsize=11);a.axvline(.33,ls=':',color='#a54b43');a.text(.40,1150,'PFC 一跳 0.33 μs',fontsize=11,color='#a54b43')
        a.set(xlabel='时间（μs）；端到端反馈 20 μs 在图外',ylabel='队列占用（KiB）',xlim=(0,4),ylim=(0,1260));a.legend(frameon=False,loc='lower right');save(f,'incast-feedback')
        # Checkpoint recovery figure (Fig 7-63)
        f,a=canvas(4.6)
        text(a,.03,.94,'周期性保存与故障回滚恢复：开销比例 ε ≈ C/I + λ(I/2 + R)',12.5)
        box(a,.03,.64,.22,.18,'正常计算 I\n600 s','blue',11)
        box(a,.28,.64,.14,.18,'保存 C\n10 s','green',11)
        box(a,.45,.64,.23,.18,'丢失进度\n平均 I/2 = 300 s','orange',11)
        box(a,.71,.64,.26,.18,'回滚恢复 R\n60 s + 读回状态','purple',11)
        arrow(a,(.25,.73),(.28,.73));arrow(a,(.42,.73),(.45,.73));arrow(a,(.68,.73),(.71,.73))
        arrow(a,(.66,.64),(.43,.52),'control')
        text(a,.54,.47,'故障中断，回退至最近一致检查点',11,ha='center')
        box(a,.03,.14,.44,.26,'128 卡超节点：R = 188 s\n附加开销 ε ≈ 3.4%\n有效吞吐 150.9 万 token/s','blue',11)
        box(a,.53,.14,.44,.26,'256 卡超节点：R = 316 s\n附加开销 ε ≈ 3.8%\n有效吞吐 152.3 万 token/s','green',11)
        text(a,.5,.05,'平均中断间隔 7.9 小时；恢复通道 64 GB/s，每卡读回 64 GB 状态',11,ha='center')
        save(f,'checkpoint-recovery')
        # Path-A new figure: NVLS + Multi-rail AllReduce and UEC SACK vs GBN (Fig 7-64)
        f,a=canvas(6.4)
        text(a,.02,.955,'A. 1,024 卡集群同步 16 GiB 梯度的分阶段耗时账本（算例 7-A）',12)
        box(a,.015,.75,.46,.15,'扁平连续环（1,024 卡）\n单轨网卡串行 2,046 轮\n每步 16 MiB → 耗时 688.2 ms','orange',11)
        box(a,.525,.75,.46,.15,'节点内 Ring + 跨机 8 轨\n本地 66.8 ms + 跨机 85.4 ms\n三阶段合计 152.3 ms（4.5×）','blue',11)
        arrow(a,(.475,.825),(.525,.825))
        box(a,.015,.55,.46,.15,'8 卡 NVLS + 跨机 8 轨\nNVSwitch 网内规约 38.2 ms\n省 42.9% 本地耗时 → 123.6 ms','purple',11)
        box(a,.525,.55,.46,.15,'64 卡超节点 NVLS + 64 轨\n机架内 38.2 ms + 跨柜 10.1 ms\n总耗时 48.3 ms（提速 14.3×）','green',11)
        arrow(a,(.245,.75),(.245,.70));arrow(a,(.755,.75),(.755,.70));arrow(a,(.475,.625),(.525,.625))
        a.plot([.015,.985],[.49,.49],color=COL['line'],lw=1.0,ls='--')
        text(a,.02,.445,'B. 万卡链路微闪断（万分之一丢包）下 Go-Back-N 与 UEC SACK 对比',12)
        box(a,.015,.24,.46,.15,'光模块温漂微闪断\n毫秒级 BER 突增越过 FEC 门限\n万分之一突发丢包（监控易漏检）','orange',11)
        box(a,.525,.24,.46,.15,'传统 RoCEv2 回退 N 步（GBN）\n单包丢失回退整窗 N = 256 包\n200 μs 超时使带宽跌至 40.2 GB/s','orange',11)
        arrow(a,(.475,.315),(.525,.315))
        box(a,.015,.04,.46,.15,'UEC 逐包喷洒 + 选择性重传\n64–256 熵值轮转 + 乱序位图\n仅精准补发 4 KiB 缺失单包','blue',11)
        box(a,.525,.04,.46,.15,'有效带宽保持率（50 GB/s 线速）\nSACK 保持 49.995 GB/s（99.99%）\n千分之一丢包 GBN 跌至 14.5 GB/s','green',11)
        arrow(a,(.245,.24),(.245,.19));arrow(a,(.475,.115),(.525,.115))
        out.save(f,'ch07-nvls-multirail-allreduce-and-sack')
        # Path-B new figure: TPOT budget back-calculation to per-collective tail latency (Fig 7-65, D11-06)
        f,a=canvas(6.4)
        text(a,.02,.955,'A. 从 TPOT 预算倒推单次集合通信上限（L=61, k=2, f=20%）',12)
        box(a,.015,.75,.46,.15,'宽松交互：TPOT = 50 ms\n总通信预算 f·TPOT = 10.0 ms\n122 次均摊上限 τ ≈ 82.0 μs','green',11)
        box(a,.525,.75,.46,.15,'标准对话：TPOT = 25 ms\n总通信预算 f·TPOT = 5.0 ms\n122 次均摊上限 τ ≈ 41.0 μs','blue',11)
        arrow(a,(.475,.825),(.525,.825))
        box(a,.015,.55,.46,.15,'严苛实时：TPOT = 20 ms\n总通信预算 f·TPOT = 4.0 ms\n122 次均摊上限 τ ≈ 32.8 μs','orange',11)
        box(a,.525,.55,.46,.15,'极速解码：TPOT = 5 ms\n总通信预算 f·TPOT = 1.0 ms\n122 次均摊上限 τ ≈ 8.2 μs','purple',11)
        arrow(a,(.755,.75),(.245,.70));arrow(a,(.475,.625),(.525,.625))
        a.plot([.015,.985],[.49,.49],color=COL['line'],lw=1.0,ls='--')
        text(a,.02,.445,'B. 集合通信尾延迟四类根因解耦与联合验证矩阵（D11-06）',12)
        box(a,.015,.24,.46,.15,'1. 拓扑路径与哈希碰撞\n跨叶脊跳数 + ECMP 重叠流\n需对照静态流表与队列水线','blue',11)
        box(a,.525,.24,.46,.15,'2. MoE 专家负载偏斜\n热门专家接收 Token 激增\nD×D 为潜在配对非实际消息数','orange',11)
        box(a,.015,.04,.46,.15,'3. Rank 到达不齐（Straggler）\n前序算子或内核抢占致晚到\n必须剥离同步等待与网络传输','purple',11)
        box(a,.525,.04,.46,.15,'4. 竞争流量与联合验证规范\n隔离检查点/存储突发背景流\n同报 p50/p99/max 与微秒时钟','green',11)
        arrow(a,(.475,.315),(.525,.315));arrow(a,(.245,.24),(.245,.19));arrow(a,(.475,.115),(.525,.115))
        out.save(f,'ch07-tpot-budget-to-collective-tail')
        # Path-D Fig 7-66: RoCEv2 (ECMP + PFC + GBN) vs UEC 1.0 (Packet Spraying + CBFC + SACK) & MoE hot expert splitting
        f,a=canvas(6.8)
        text(a,.02,.965,'A. RoCEv2 与 UEC 1.0 四大支柱协议栈及 UCCL-Tran 多路径对比',11.5)
        box(a,.015,.76,.46,.15,'传统 RoCEv2 瓶颈\nECMP 流级碰撞 + PFC 蔓延\nGBN 丢包回退整窗 N=256','orange',11)
        box(a,.525,.76,.46,.15,'UEC 1.0 传输层（PRI+UET）\n64–256 熵值逐包多路径喷洒\n硬件乱序位图 + SACK 单包重传','green',11)
        arrow(a,(.475,.835),(.525,.835))
        box(a,.015,.56,.46,.15,'UEC 1.0 链路层（CBFC+LLR）\n信用流控替代 PFC 防死锁风暴\n链路级毫秒/微秒局部重传','blue',11)
        box(a,.525,.56,.46,.15,'UCCL-Tran（OSDI \'26）\n解耦主机驱动与多队列喷洒\n屏蔽底层异构 RDMA 网卡差异','purple',11)
        arrow(a,(.755,.76),(.245,.71));arrow(a,(.475,.635),(.525,.635))
        a.plot([.015,.985],[.50,.50],color=COL['line'],lw=1.0,ls='--')
        text(a,.02,.455,'B. MoE 热门专家负载偏斜（640 vs 256 tok）与零拷贝逻辑专家切分',11.5)
        box(a,.015,.25,.46,.15,'64 专家 / 8 卡（B=2048, k=2）\n理想均衡每专家 256 tok\n热门专家激增至 640 tok（2.5×）','orange',11)
        box(a,.525,.25,.46,.15,'本地分块 Padding 开销\n8 专家 [32,16,8,4,2,1,1,0]=64 行\n按 32 补齐 224 行 / 最大值 256 行','blue',11)
        box(a,.015,.04,.46,.15,'CRAFT 选择性专家副本\n全层增 1 副本需 3.30 GiB/卡\n仅热点层增 8 副本占 0.28 GiB','purple',11)
        box(a,.525,.04,.46,.15,'零拷贝逻辑切分与负载重排\n结合 EPLB 将 640 tok 拆散分流\n消除 All-to-All 最忙卡尾延迟','green',11)
        arrow(a,(.475,.325),(.525,.325));arrow(a,(.245,.25),(.245,.19));arrow(a,(.475,.115),(.525,.115))
        out.save(f,'ch07-rocev2-pfc-vs-uec10-packet-spraying')
        # Path-D Fig 7-67: Broadcom Tomahawk 6 102.4 Tbps two-tier Clos scaling & AutoCCL / NCCL 2.28-2.31 tuning
        f,a=canvas(6.8)
        text(a,.02,.965,'A. Tomahawk 6（102.4 Tbps）高基数两层扁平 Clos 万卡组网革命',11.5)
        box(a,.015,.76,.46,.15,'芯片代际跃迁：102.4 Tbps\n512×200G SerDes + CPO 选项\n单芯片 128×800G 或 256×400G','blue',11)
        box(a,.525,.76,.46,.15,'两层无阻塞 Clos 容量翻四倍\nk=128 容纳 8,192 张 800G 卡\nk=256 容纳 32,768 张 400G 卡','green',11)
        arrow(a,(.475,.835),(.525,.835))
        box(a,.015,.56,.46,.15,'对比传统三层 Fat-Tree\n砍掉整层 Core 交换机与跳数\n节省 1/3（33.3%）光模块与光纤','purple',11)
        box(a,.525,.56,.46,.15,'2:1 超售两层万卡扩展\nk=128（85 下行 + 43 上联）\n两层直连 ~1.4 万张 800G 卡','orange',11)
        arrow(a,(.755,.76),(.245,.71));arrow(a,(.475,.635),(.525,.635))
        a.plot([.015,.985],[.50,.50],color=COL['line'],lw=1.0,ls='--')
        text(a,.02,.455,'B. 计算通信并发争用反转（Qwen3-8B）与 AutoCCL / NCCL Zero-CTA',11.5)
        box(a,.015,.25,.46,.15,'孤立通信基准假象（8 MiB）\n配置 A：0.24 ms vs B：0.18 ms\n孤立测试下 B 快 33.3%','orange',11)
        box(a,.525,.25,.46,.15,'并发重叠下 SM/HBM 争用反转\n配置 A：0.44 ms（计算0.44）\n配置 B：0.62 ms（慢40.9%）','purple',11)
        box(a,.015,.04,.46,.15,'在线搜索回本与 AutoCCL\n12 ms 搜索 / 省 0.18 ms → 67 步\nAutoCCL 提速 1.07x–1.32x','blue',11)
        box(a,.525,.04,.46,.15,'NCCL 2.28–2.31.2 演进\ntuner_v6 回调动态调优分块\nZero-CTA Copy Engine 零 SM 占用','green',11)
        arrow(a,(.475,.325),(.525,.325));arrow(a,(.245,.25),(.245,.19));arrow(a,(.475,.115),(.525,.115))
        out.save(f,'ch07-tomahawk6-102t-two-tier-clos-scaling')
        # Path-E3 Fig 7-68: Pluggable DSP vs LPO vs CPO photonics power, BER & 16k-GPU TCO ledger
        f,a=canvas(6.8)
        text(a,.02,.965,'A. 800G / 1.6T 数据中心互连三代光电架构对比：DSP 可插拔 vs. LPO vs. CPO',11.5)
        box(a,.015,.76,.46,.15,'传统 DSP 可插拔光模块\n每只含 5nm/3nm DSP 重定时器\n800G 约 16 W / 1.6T 约 28 W','orange',11)
        box(a,.525,.76,.46,.15,'线性直驱光模块（LPO / LRO）\n移除模块 DSP 由交换 ASIC 直驱\n800G 约 8 W（省 50% 功耗）','blue',11)
        arrow(a,(.475,.835),(.525,.835))
        box(a,.015,.56,.46,.15,'共封装光学（CPO / Quantum-X）\n硅光引擎紧邻 ASIC（<15 mm）\n每端口等效光学功耗仅 4.5–5 W','green',11)
        box(a,.525,.56,.46,.15,'插损预算与可维护性权衡\nLPO 受限主机通道插损 ≤15 dB\nCPO 用外置激光源 ELS 热插拔','purple',11)
        arrow(a,(.755,.76),(.245,.71));arrow(a,(.475,.635),(.525,.635))
        a.plot([.015,.985],[.50,.50],color=COL['line'],lw=1.0,ls='--')
        text(a,.02,.455,'B. 16,384 卡（800GbE 两层 Clos，65,536 端口等效）光电功耗与可用性账本',11.5)
        box(a,.015,.25,.46,.15,'全网 65,536 只 DSP 可插拔\n光模块总功耗 1,048.6 kW\n占交换网络总功耗 >55%','orange',11)
        box(a,.525,.25,.46,.15,'Leaf 下行 LPO + Spine DSP\n下行 32,768 端用 LPO（8 W）\n光学功耗 786.4 kW（省 262 kW）','blue',11)
        box(a,.015,.04,.46,.15,'交换侧全面 CPO（Spectrum-X）\n49,152 交换端口用 4.8 W CPO\n总功耗 498.1 kW（省 52.5%）','green',11)
        box(a,.525,.04,.46,.15,'链路闪断率与集群 MTBF\n消除数万个可插拔电接触面\n万卡光学闪断率降低 3.5×–10×','purple',11)
        arrow(a,(.475,.325),(.525,.325));arrow(a,(.245,.25),(.245,.19));arrow(a,(.475,.115),(.525,.115))
        out.save(f,'ch07-pluggable-dsp-vs-lpo-vs-cpo-photonics-ledger')
        # Path-E3 Fig 7-69: SHARP v4 In-Network Aggregation & Multi-Tenant Victim Flow Isolation
        f,a=canvas(6.8)
        text(a,.02,.965,'A. 网内计算规约（SHARP v3/v4）树形聚合流水线与跨机流量减半机理',11.5)
        box(a,.015,.76,.46,.15,'主机侧 Ring AllReduce（N 机）\n每机跨网收发各 2(N-1)/N·M\n流量渐近 2M，步数随 N 增长','orange',11)
        box(a,.525,.76,.46,.15,'SHARP 网内树形规约（ALU）\n叶脊交换机线速硬件浮点累加\n每机仅上送 1M、下收 1M（省半）','green',11)
        arrow(a,(.475,.835),(.525,.835))
        box(a,.015,.56,.46,.15,'SHARP v4（Quantum-X / CX-8）\n原生支持 BF16/FP16/FP8/FP32\nAllReduce 跨机有效带宽 ~2×','blue',11)
        box(a,.525,.56,.46,.15,'规约树配额与消息尺寸交叉点\n交换机片上 SHARP 树槽位受限\n大消息省 50% 流量、小消息省跳','purple',11)
        arrow(a,(.755,.76),(.245,.71));arrow(a,(.475,.635),(.525,.635))
        a.plot([.015,.985],[.50,.50],color=COL['line'],lw=1.0,ls='--')
        text(a,.02,.455,'B. 多租户万卡集群网络性能隔离：受害流尾延迟劣化与硬件流量整形',11.5)
        box(a,.015,.25,.46,.15,'多租户共享 Spine 拥塞扩散\n租户 B 突发高强度 All-to-All\nPFC 暂停与队列积压波及租户 A','orange',11)
        box(a,.525,.25,.46,.15,'受害作业（Victim）尾延迟暴增\n租户 A 正常 AllReduce p99 ×3.4\n引起计算流大面积同步空转','purple',11)
        box(a,.015,.04,.46,.15,'拓扑感知分区与虚通道隔离\n优先按整台 Leaf / Pod 切分\n跨租户设独立 QoS 队列与缓冲','blue',11)
        box(a,.525,.04,.46,.15,'源端硬件速率整形（Pacing）\n网卡按租户配额控制注入节拍\n将跨租户 p99 干扰压回 <8%','green',11)
        arrow(a,(.475,.325),(.525,.325));arrow(a,(.245,.25),(.245,.19));arrow(a,(.475,.115),(.525,.115))
        out.save(f,'ch07-sharp-in-network-aggregation-and-tenant-isolation')
    out.finish();return out.outputs,out.checks
