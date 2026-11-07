"""A physical path before its budget; an event clock before its recurrence."""
import numpy as np
import matplotlib.pyplot as plt
from figure_style import COL,STYLE,canvas,plot,text,box,arrow,Exporter

def draw_encoder_paths(out):
    """Show the same vision stages with the network boundary in two places."""
    with plt.rc_context(STYLE):
        for local, name in [(False, 'encoder-remote'), (True, 'encoder-local')]:
            f, a = canvas(6.5)
            # Two device lanes; vertical order follows the execution order.
            for x, title in [(.02, '端侧设备'), (.59, '远端服务器')]:
                box(a, x, .32, .39, .63, '', 'gray')
                text(a, x + .195, .918, title, 13, ha='center', weight='medium')
            if not local:
                a.plot([.5, .5], [.34, .85], color=COL['line'], lw=.8, ls=':')
                a.plot([.5, .5], [.90, .95], color=COL['line'], lw=.8, ls=':')
            else:
                a.plot([.5, .5], [.34, .51], color=COL['line'], lw=.8, ls=':')
                a.plot([.5, .5], [.56, .95], color=COL['line'], lw=.8, ls=':')
            text(a, .5, .983, '部署边界：上行 6.4 Mbit/s', 12, ha='center')
            box(a, .0375, .80, .355, .072, '压缩截图  0.8 MB', 'blue', 11)
            x = .0375 if local else .6075
            cx = x + .1775
            if not local:
                box(a, .6075, .80, .355, .072, '接收压缩图片', 'blue', 11)
                arrow(a, (.3925, .836), (.6075, .836))
                text(a, .50, .875, '图片', 11, ha='center')
                text(a, .215, .68, '上传图片', 12, ha='center')
                text(a, .215, .61, '0.8 MB ÷ 0.8 MB/s\n= 1.00 s', 11, ha='center')
                text(a, .215, .45, '视觉计算在远端完成\n特征留在服务器内', 11, ha='center')
            box(a, x, .684, .355, .076, '解码与预处理\n缩放至 640 × 640', 'blue', 11)
            arrow(a, (cx, .80), (cx, .76))
            box(a, x, .562, .355, .080, '视觉编码与图像块合并\n形成 400 个视觉 token', 'green', 11)
            arrow(a, (cx, .684), (cx, .642))
            box(a, x, .430, .355, .090, '最终投影 + 3 组 DeepStack\n完整 BF16 特征  8.192 MB', 'green', 11)
            arrow(a, (cx, .562), (cx, .520))
            if local:
                box(a, .6075, .430, .355, .090, '接收并还原四组特征\n形状、精度与模型匹配', 'green', 11)
                arrow(a, (.3925, .475), (.6075, .475))
                text(a, .5, .536, '完整特征', 11, ha='center')
                text(a, .785, .77, '上传完整数值特征', 12, ha='center')
                text(a, .785, .68, '8.192 MB ÷ 0.8 MB/s\n= 10.24 s', 11, ha='center')
                text(a, .215, .363, '本地编码后序列化发送', 11, ha='center')
            arrow(a, (.785, .430), (.785, .397))
            arrow(a, (.785, .343), (.785, .307))
            text(a, .785, .367, '送入远端语言模型', 11, ha='center')
            # Both placements use identical downstream feature interfaces.
            box(a, .02, .025, .96, .277, '', 'white')
            text(a, .50, .272, '四组特征如何进入语言模型', 12, ha='center', weight='medium')
            box(a, .045, .162, .39, .060, '最终投影  [400, 2560]', 'green', 11)
            box(a, .565, .162, .39, .060, '与文本嵌入组成输入', 'purple', 11)
            arrow(a, (.435, .192), (.565, .192))
            box(a, .045, .072, .39, .060, 'DeepStack  3 × [400, 2560]', 'green', 11)
            box(a, .565, .072, .39, .060, '分别注入对应的中间层', 'purple', 11)
            arrow(a, (.435, .102), (.565, .102))
            out.save(f, 'figure-12-' + name)

def draw_local_tiers(out,data):
    """Three local device tiers reading the same 16.345 GB decode step."""
    rows=[('RTX PRO 6000',9.12,'109.6 token/s','green'),
          ('M3 Ultra',19.96,'50.1 token/s','green'),
          ('手机，q4_0 权重',64.4,'15.5 token/s','blue'),
          ('手机，BF16 权重',192.7,'5.2 token/s','blue')]
    with plt.rc_context(STYLE):
        f,a=plot(3.4,left=.28)
        for y,(label,ms,rate,c) in enumerate(rows):
            a.barh(y,ms,height=.55,color=COL[c],edgecolor=COL['line'])
            text(a,ms+6,y,f'{ms:g} ms，{rate}',11)
        a.set(yticks=range(4),yticklabels=[r[0] for r in rows],xlabel='每步读取时间下界（ms）',xlim=(0,330))
        out.save(f,'figure-12-local-tiers')
    data['12-local-tiers']={'kind':'teaching_bound','step_bytes':16344778752,'phone_channels_x16_declared':4,
        'phone_bus_GBps':84.8,'q4_weight_bytes':4257230400,'rows_ms':[r[1] for r in rows],
        'token_per_s':[109.6,50.1,15.5,5.2],'relationship':'带宽决定每步读取下界'}

def draw_loss_repair(out,data):
    """Completion time on the 14 percent loss path under three repair schemes."""
    rows=[('Mathis 上界下仅发送',18.33,'gray'),
          ('逐轮重传 p99（6 轮）',1.239,'orange'),
          ('逐轮重传期望',0.756,'orange'),
          ('FEC 308 符号，99.9% 免重传',0.2407,'blue'),
          ('串行预算：RTT＋模型＋发送',0.2385,'green')]
    with plt.rc_context(STYLE):
        f,a=plot(3.6,left=.42)
        for y,(label,t,c) in enumerate(rows):
            a.barh(y,t,height=.55,color=COL[c],edgecolor=COL['line'])
            text(a,t*1.12,y,f'{t:g} s',11)
        a.set(xscale='log',xlim=(0.1,60),yticks=range(5),yticklabels=[r[0] for r in rows],
              xlabel='完成时间（s，对数坐标）',xticks=[.1,1,10],xticklabels=['0.1','1','10'])
        out.save(f,'figure-12-loss-repair')
    data['12-loss-repair']={'kind':'teaching_bound','path':{'rtt_s':.2,'knee_Mbps':333,'loss':.14,'mss_bytes':1448,
        'request_bytes':354640},'packets':245,'serial_budget_s':.2385,'mathis_Mbps':.1548,'mathis_send_s':18.33,
        'retransmit_expected_s':.756,'retransmit_p99_s':1.239,'fec_repair':63,'fec_overhead':.257,'fec_s':.2407,
        'relationship':'修复方式决定完成时间'}

def draw(here,data):
    out=Exporter(here)
    def save(f,n):out.save(f,'figure-12-'+n)
    with plt.rc_context(STYLE):
        f,a=canvas(4.8);box(a,.04,.67,.32,.20,'终端：30 MB 原图','blue',11);box(a,.64,.67,.32,.20,'服务器：处理 0.3 s','green',11);arrow(a,(.36,.77),(.64,.77));text(a,.5,.95,'上行：20 Mbit/s，发送 12 s',12,ha='center')
        box(a,.64,.22,.32,.20,'生成 5 MB 成片','green',11);box(a,.04,.22,.32,.20,'终端收齐：任务完成','orange',11);arrow(a,(.8,.67),(.8,.42));arrow(a,(.64,.32),(.36,.32));text(a,.5,.09,'下行：100 Mbit/s，发送 0.4 s',12,ha='center');text(a,.27,.52,'双向传播合计 0.1 s',11,ha='center');save(f,'image-path')
        d=data['12-1'];f,a=plot(4.0);b=np.array(d['uplink_Mbps'])
        for y,l,c in [(240/b+.8,'原方案','#267398'),(240/b+.53,'处理快十倍','#388768'),(120/b+.95,'输入减半','#a56c28')]:a.plot(b,y,label=l,color=c)
        a.set(xscale='log',xlabel='上行速率（Mbit/s）',ylabel='完整成片返回时间（s）');a.legend(frameon=False);save(f,'1-raw')
        for chunked,name in [(False,'2-overlap'),(True,'overlap-chunks')]:
            f,a=plot(4.0,left=.20)
            if chunked:
                for i,ret in enumerate([.08,.16,.16]):
                    u=4*(i+1);a.barh(2,4,left=4*i,height=.5,color=COL['blue'],edgecolor=COL['line']);a.barh(1,.1,left=u+.05,height=.5,color=COL['green'],edgecolor=COL['line']);a.barh(0,ret,left=u+.15,height=.5,color=COL['orange'],edgecolor=COL['line'])
            else:
                for row,start,dur,c in [(2,0,12,'blue'),(1,12.05,.3,'green'),(0,12.35,.4,'orange')]:a.barh(row,dur,left=start,height=.5,color=COL[c],edgecolor=COL['line'])
            a.set(yticks=[2,1,0],yticklabels=['上传','处理','回传'],xlim=(0,13),xlabel='时间（s）');save(f,name)
        f,a=plot(4.1,left=.23)
        for row,start,dur,c in [(3,0,20,'blue'),(2,20,12,'green'),(1,32,1,'orange'),(0,78,20,'purple')]:a.barh(row,dur,left=start,height=.5,color=COL[c],edgecolor=COL['line'])
        a.plot(38,1,'o',color='#454545');a.annotate('到达 38 ms',(38,1),(45,1.55),fontsize=11,arrowprops={'arrowstyle':'->'});a.set(yticks=[3,2,1,0],yticklabels=['采集','模型处理','发送／到达','播放'],xlim=(0,105),xlabel='时间（ms）');save(f,'audio-clocks')
        f,a=plot(4.3,left=.20);chunks=data['12-3']['audio_chunks'][:5]
        for i,r in enumerate(chunks):
            a.barh(i,20,left=r['deadline_ns']/1e6,height=.6,color='white',edgecolor=COL['line']);a.barh(i,20,left=r['playback_start_ns']/1e6,height=.36,color=COL['blue'],edgecolor=COL['line']);a.plot(r['arrival_ns']/1e6,i,'o',color='#a56c28')
        a.set(yticks=range(5),yticklabels=[f'块 {i+1}' for i in range(5)],xlabel='时间（ms）',xlim=(25,190));a.invert_yaxis();save(f,'3-paths')
        d=data['12-4'];f,a=plot(3.8)
        for bits,c in zip(d['initial_bits'],['#267398','#a56c28']):t=np.linspace(0,bits/(256000-130000),100);a.plot(t*1000,(bits-t*126000)/1000,color=c,label=f'初始 {bits/256000*1000:g} ms 音频')
        a.set(xlabel='播放开始后的时间（ms）',ylabel='尚未播放的数据（kbit）',ylim=(0,34));a.legend(frameon=False);save(f,'4-buffer')
        f,a=canvas(4.1)
        for x,y,l,c in [(.04,.63,'截图 v','blue'),(.62,.63,'上传与模型判断','green'),(.62,.18,'执行操作','orange'),(.04,.18,'更新界面 v+1','blue')]:box(a,x,y,.34,.21,l,c)
        for p,q in [((.38,.735),(.62,.735)),((.79,.63),(.79,.39)),((.62,.285),(.38,.285)),((.21,.39),(.21,.63))]:arrow(a,p,q)
        save(f,'5-agent')
        draw_encoder_paths(out)
        f,a=plot(3.3,left=.23);a.barh([1,0],[1,10.24],height=.5,color=[COL['blue'],COL['green']],edgecolor=COL['line']);a.set(yticks=[1,0],yticklabels=['压缩图片','完整特征'],xlabel='6.4 Mbit/s 上行发送时间（s）',xlim=(0,11.5));save(f,'6-placement')
        f,a=canvas(5.4)
        for row,(l,c) in enumerate([('图片：0.8 MB','blue'),('编码缓存 EC：7.8 MiB','green'),('视觉 token KV：56.3 MiB','purple')]):
            y=.73-row*.28;box(a,.04,y,.49,.17,l,c,11);text(a,.57,y+.085,['重做视觉编码\n再处理语言前缀','从语言前缀处理开始','从匹配前缀之后继续'][row],11)
        save(f,'cache-restart')
        f,a=plot(3.9);n=np.arange(0,41)
        for extra,c in [(0,'#267398'),(1,'#a56c28')]:a.plot(n,n*.4-data['12-7']['preparation_s']-extra,label=f'恢复 {1+extra} s',color=c)
        a.axhline(0,color='#666');a.set(xlabel='迁移后完成轮数',ylabel='累计净节省（s）');a.legend(frameon=False);save(f,'7-migration')
        # A layer's reductions are separate handoffs with two matching participants.
        for ffn,name in [(False,'8-sync'),(True,'sync-ffn')]:
            f,a=canvas(4.1);text(a,.5,.94,'前馈输出汇总后，进入下一层' if ffn else '注意力输出汇总后，进入前馈',13,ha='center')
            for row in range(2):
                y=.62-row*.36;box(a,.04,y,.29,.20,('FFN' if ffn else '注意力')+f'分片 {row}','blue',11);box(a,.68,y,.28,.20,'下一层' if ffn else '前馈计算','green',11);arrow(a,(.33,y+.10),(.68,y+.10))
            arrow(a,(.50,.72),(.50,.36));arrow(a,(.54,.36),(.54,.72));text(a,.54,.11,'两卡归约输出',11,ha='center');save(f,name)
        f,a=plot(3.8,left=.18);cycle=data['12-9']['cycle_s']*1000
        for i in range(3):
            a.barh(0,25.6,left=i*cycle,height=.5,color=COL['blue'],edgecolor=COL['line']);a.barh(0,100,left=i*cycle+25.6,height=.5,color=COL['gray'],edgecolor=COL['line'])
        a.set(yticks=[0],yticklabels=['发送端'],xlabel='时间（ms）',xlim=(0,380));text(a,20,.6,'发 25.6 ms',11);text(a,60,-.6,'等整批确认 100 ms',11);a.set_ylim(-1,1);save(f,'9-window')
        for order,name in [(False,'10-transport'),(True,'transport-per-stream')]:
            f,a=plot(3.2,left=.20)
            for row,ready,done,c in [(1,3,3 if order else 7,'blue'),(0,7,7,'green')]:
                a.barh(row,ready,height=.5,color=COL[c],edgecolor=COL['line'])
                if done>ready:a.barh(row,done-ready,left=ready,height=.5,color=COL['gray'],edgecolor=COL['line'])
                a.plot(done,row,'o',color='#454545')
            a.set(yticks=[1,0],yticklabels=['音频','图片'],xlim=(0,8),xlabel='数据交给应用的时刻（s）');save(f,name)
        f,a=plot(4.0,left=.20)
        for row,parts in enumerate(data['12-11']['exchange_components_us']):
            start=0
            for v,c,l in zip(parts,['gray','blue','orange','green'],['接入等待','数据帧','SIFS','MAC ACK']):a.barh(row,v,left=start,height=.5,color=COL[c],edgecolor=COL['line'],label=l if row==0 else None);start+=v
        a.set(yticks=[0,1],yticklabels=['数据交换','反馈交换'],xlabel='空口占用（μs）',ylim=(-.5,2.4));a.legend(ncol=2,frameon=False,loc='upper left');save(f,'11-wireless')
        f,a=canvas(4.4);box(a,.30,.73,.40,.18,'用户发出取消','orange')
        for x,l,c in [(.04,'本地：清空播放缓冲','blue'),(.57,'远端：停止生成与发送','green')]:box(a,x,.30,.39,.23,l,c,11);arrow(a,(.5,.73),(x+.195,.53),'control')
        text(a,.235,.15,'立即停止播放',11,ha='center');text(a,.765,.15,'控制消息到达后执行',11,ha='center');save(f,'cancel-paths')
        for shared,name in [(False,'multipath-independent'),(True,'12-multipath')]:
            f,a=canvas(4.6);box(a,.04,.39,.23,.21,'30 MB 图片','blue',11)
            for y,l in [(.72,'20 MB／20 Mbit/s'),(.15,'10 MB／10 Mbit/s')]:box(a,.38,y,.40,.17,l,'green',11);arrow(a,(.27,.5),(.38,y+.085))
            if shared:box(a,.82,.37,.15,.25,'共同\n出口','orange',11);arrow(a,(.78,.805),(.89,.62));arrow(a,(.78,.235),(.89,.37));text(a,.5,.05,'共同出口 24 Mbit/s → 至少 10 s',11,ha='center')
            else:text(a,.5,.05,'两路独立：均在 8 s 发完自己的部分',11,ha='center')
            save(f,name)
        for i,row in enumerate(data['12-13']['matched_ASR_ms']):
            f,a=plot(3.8);a.bar([0,1],np.array(row)/1000,color=[COL['blue'],COL['green']],edgecolor=COL['line'])
            for j,v in enumerate(row):a.text(j,v/1000+.03,f'{v:.1f} ms',fontsize=12,ha='center')
            a.set(xticks=[0,1],xticklabels=['直接路径','Queqiao'],ylabel='请求中位数（s）',ylim=(0,1.4));save(f,'13-queqiao' if i==0 else 'queqiao-tuned')
        f,a=canvas(4.7);text(a,.04,.94,'单因素实验设计：每步只改变一个设置',13)
        for row,(l,c) in enumerate([('连接复用 → 握手与首字节时刻','blue'),('窗口设置 → 在途量与确认时刻','green'),('发送节奏 → 出口空闲与完成时刻','orange')]):box(a,.04,.68-row*.26,.92,.18,l,c,12)
        save(f,'experiment-design')
        for i,row in enumerate(data['12-14']['seconds']):
            f,a=plot(3.6,left=.18);start=0
            for v,c,l in zip(row,['gray','orange','green','purple','blue'],['准备','终端','模型','传播','上传']):
                if v:a.barh(0,v,left=start,height=.5,color=COL[c],edgecolor=COL['line'],label=l)
                start+=v
            a.axvline(45,ls='--',color='#a56c28');a.set(yticks=[0],yticklabels=[['端侧','附近','云端'][i]],xlabel='剩余二十轮完成时间（s）',xlim=(0,68),ylim=(-.6,1.6));a.legend(ncol=3,frameon=False,loc='upper left');save(f,'14-budgets' if i==0 else 'budgets-'+str(i))
        d=data['12-15'];f,a=plot(4.0);b=np.array(d['scan_Mbps']);a.plot(b,d['cloud_seconds'],color='#267398',label='云端 H100');a.axhline(d['near_s'],color='#388768',label=f"附近 RTX PRO 6000：{d['near_s']:.1f} s");a.axhline(d['deadline_s'],ls='--',color='#a56c28',label=f"期限：{d['deadline_s']:g} s");a.set(xlabel='云上行（Mbit/s）',ylabel='二十轮完成时间（s）',ylim=(0,80));a.legend(frameon=False);save(f,'15-deployment')
        for retained,name in [(True,'16-recovery'),(False,'recovery-all')]:
            rc=data['12-16']['round_s'];f,a=canvas(4.2);text(a,.04,.94,f'云端已执行十轮，每轮 {rc:.2f} s；恢复连接需 1 s',14)
            for i in range(10):box(a,.04+i%5*.187,.66-(i//5)*.22,.17,.15,str(i+1),'green' if retained and i<9 else 'orange',12)
            text(a,.5,.22,'保留前九轮，只重做第十轮' if retained else '进度丢失，重做十轮',12,ha='center');text(a,.5,.08,f'额外 1 + {rc:.2f} = {1+rc:.2f} s' if retained else f'额外 1 + 10 × {rc:.2f} ≈ {1+10*rc:.1f} s',12,ha='center');save(f,name)
    from core_principles_figures import draw as draw_principles
    draw_principles(12, out)
    draw_local_tiers(out,data);draw_loss_repair(out,data)
    with plt.rc_context(STYLE):
        f,a=canvas(6.4)
        box(a,.02,.52,.96,.44,'','gray')
        text(a,.50,.92,'A. 移动端 UFS 4.0 闪存捆绑读取（1.28 GB／步）',12,ha='center',weight='medium')
        ufs_cards=[
            (.035,'2 KiB 随机读\n400 MB/s · 3.20 s\n0.31 token/s','orange'),
            (.3525,'64 KiB 块捆绑\n3.2 GB/s · 0.40 s\n2.50 token/s（8×）','blue'),
            (.67,'捆绑 + 5 步滑窗\n75% 复用 · 0.10 s\n10.0 token/s（32×）','green'),
        ]
        for x,lbl,c in ufs_cards:
            box(a,x,.56,.295,.30,lbl,c,11)
        arrow(a,(.33,.71),(.3525,.71))
        arrow(a,(.6475,.71),(.67,.71))
        box(a,.02,.03,.96,.45,'','gray')
        text(a,.50,.44,'B. 具身智能 VLA 双环路与动作块容错（K=25 步 · 500 ms）',12,ha='center',weight='medium')
        vla_cards=[
            (.035,'云端重规划 5 Hz\n200 ms 发 25 步','blue'),
            (.3525,'基准传输 80 ms\n余量 420 ms','green'),
            (.67,'最大抖动 220 ms\n延迟上限 300 ms','orange'),
        ]
        for x,lbl,c in vla_cards:
            box(a,x,.22,.295,.17,lbl,c,11)
        arrow(a,(.33,.305),(.3525,.305))
        arrow(a,(.6475,.305),(.67,.305))
        box(a,.035,.055,.93,.135,'端侧 Orin/Thor 快环：50 Hz 轨迹插值（20 ms/步）驱动 1000 Hz 伺服\n单包 700 B；超 300 ms 缓冲耗尽时触发看门狗安全制动','purple',11)
        out.save(f,'ch12-ufs-flash-streaming-and-vla-chunking')

        # Path-B new figure: ch12-m2max-rtxpro6000-cloud-three-tier (Fig 12-38)
        f,a=canvas(6.4)
        text(a,.25,.95,'A. 96 GB 统一内存 vs 独立显存',12,ha='center')
        text(a,.745,.95,'B. 端—边—云三维部署边界',12,ha='center')
        a.plot([.495,.495],[.03,.97],color=COL['line'],ls='--',lw=1.0)

        box(a,.03,.75,.44,.15,'M2 Max（96 GB 统一内存）：\n400 GB/s，4 GiB 读 10.74 ms\nM2 Pro（200 GB/s）：21.47 ms','blue',11)
        box(a,.03,.54,.44,.16,'RTX PRO 6000（96GB 显存）：\n1,792 GB/s，4 GiB 读 2.40 ms\n带宽达 M2 Max 的 4.48 倍','green',11)
        box(a,.03,.31,.44,.18,'统一内存工作集边界：\nCPU/GPU/OS 争抢同一物理池\n受 Metal 安全工作集上限约束','orange',11)
        box(a,.03,.05,.44,.20,'独立显存边界（600 W）：\nPCIe 5.0 x16 负责初始加载\n96 GB ECC 显存由 GPU 独占','gray',11)

        box(a,.52,.75,.45,.15,'端侧便携（M2 Max 笔记本）：\n断网可用、最高隐私、数十瓦\n适合单流低频交互与外出开发','blue',11)
        box(a,.52,.54,.45,.16,'附近工作站（RTX PRO 6000）：\n局域网低 RTT + 4.48× 解码带宽\n20 轮截图 11.0 s（快于云端）','green',11)
        box(a,.52,.31,.45,.18,'远端云集群（H100 / B200）：\n3.35–8.0 TB/s 支撑超长上下文\n受广域网上行与 200 ms RTT 制约','purple',11)
        box(a,.52,.05,.45,.20,'Ollama 配对评测铁律：\n剥离 load_duration 冷启动\n严防 KV 缓存命中造成虚假加速','orange',11)
        out.save(f,'ch12-m2max-rtxpro6000-cloud-three-tier')

        # Path-D new figure 1: ch12-npu-static-chunked-graph-and-flexsp (Fig 12-39)
        f,a=canvas(6.4)
        text(a,.25,.95,'A. NPU 静态分块图与填充开销',12,ha='center')
        text(a,.745,.95,'B. FlexSP 变长序列与通信',12,ha='center')
        a.plot([.495,.495],[.03,.97],color=COL['line'],ls='--',lw=1.0)

        box(a,.03,.76,.44,.14,'Hexagon NPU / ExecuTorch：\n算子图须按固定块长 B 编译','blue',11)
        box(a,.03,.53,.44,.17,'填充浪费：L_p=513 按 B=256\n切 3 块实算 768 词元 (+49.7%)','orange',11)
        box(a,.03,.30,.44,.17,'细粒度 B=128：切为 5 块\n实算 640 词元（多算 127 个）','green',11)
        box(a,.03,.04,.44,.20,'llm.npu / MLLM 异构协同：\nNPU 跑 INT8 块，CPU 跑补偿','gray',11)

        box(a,.52,.76,.45,.14,'同为 8192 词元的不同分箱：\n[4096,4096] vs [7168,1024]','blue',11)
        box(a,.52,.53,.45,.17,'因果注意力对数差 1.56×~2.0×\n长文档配更高 SP 度数降显存','purple',11)
        box(a,.52,.30,.45,.17,'Ulysses GQA 陷阱 (Qwen3)：\n先扩 KV 使载荷 12 增至 24KiB','orange',11)
        box(a,.52,.04,.45,.20,'梯度归约：按有效词元加权\n(token_mean 2.75 vs 均值 2.0)','green',11)
        out.save(f,'ch12-npu-static-chunked-graph-and-flexsp')

        # Path-D new figure 2: ch12-mobile-thermal-throttling-and-mlx-mtp (Fig 12-40)
        f,a=canvas(6.4)
        text(a,.25,.95,'A. 手机散热墙与热降频相变',12,ha='center')
        text(a,.745,.95,'B. Apple MLX 与 MTP 投机',12,ha='center')
        a.plot([.495,.495],[.03,.97],color=COL['line'],ls='--',lw=1.0)

        box(a,.03,.76,.44,.14,'瞬时冲刺 (0~t_th)：18 W 跑满\n84.8 GB/s 带宽 (15.5 token/s)','blue',11)
        arrow(a,(.25,.76),(.25,.67),'data')
        box(a,.03,.50,.44,.16,'触温上限触发 DVFS 降频：\n稳态功率墙降至 8.5~13.8 W','orange',11)
        arrow(a,(.25,.50),(.25,.41),'data')
        box(a,.03,.24,.44,.16,'LPDDR5X 与 SoC 频率下调：\n持续吞吐跌落 30%~45%','purple',11)
        box(a,.03,.04,.44,.15,'评测铁律：禁冷启动短测冒充\n须测 10 分钟热平衡吞吐与能耗','gray',11)

        box(a,.52,.76,.45,.14,'Ollama v0.31 集成 MLX：\nUMA 统一内存零拷贝直驱 GPU','blue',11)
        box(a,.52,.53,.45,.17,'多词元预测 (MTP) 投机收益：\n单步验 γ+1 词元，权重读 1 次','green',11)
        box(a,.52,.30,.45,.17,'算术强度跃升：接受率 α=0.8\n期望产 2.44 词元，提速 2.1×','purple',11)
        box(a,.52,.04,.45,.20,'端侧能效红利：摊薄词元访存\n翻转能耗，推迟热降频相变','orange',11)
        out.save(f,'ch12-mobile-thermal-throttling-and-mlx-mtp')
    data['12-ufs-vla']={'kind':'teaching_bound','sparse_ffn_gb':1.28,'rand_bw_gbps':0.4,'bundle_bw_gbps':3.2,
        'win_reuse':0.75,'step_s':[3.20,0.40,0.10],'vla_horizon_ms':500,'vla_plan_ms':200,'vla_base_ms':80,'vla_jitter_max_ms':220}
    out.finish();return out.outputs,out.checks
