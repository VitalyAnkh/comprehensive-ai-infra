"""Follow state ownership, then execution dependencies, then recoverable progress."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from figure_style import COL,BORDER_COL,TEXT_COL,STYLE,canvas,plot,text,box,card,badge,arrow,Exporter

def draw(here,data):
    out=Exporter(here)
    def save(f,n):out.save(f,'figure-10-'+n)
    with plt.rc_context(STYLE):
        f,a=canvas(5.0)
        for x,y,w,l,c in [(.04,.70,.25,'输入样本','blue'),(.38,.70,.25,'前向计算','blue'),(.72,.70,.24,'损失','orange'),(.38,.39,.25,'反向计算','green'),(.04,.08,.25,'参数梯度','green'),(.38,.08,.25,'Adam 更新','purple'),(.72,.08,.24,'新权重','blue')]:box(a,x,y,w,.17,l,c)
        for p,q in [((.29,.785),(.38,.785)),((.63,.785),(.72,.785)),((.84,.70),(.63,.475)),((.38,.475),(.165,.25)),((.29,.165),(.38,.165)),((.63,.165),(.72,.165))]:arrow(a,p,q)
        text(a,.04,.95,'一次迭代：由预测误差得到下一份权重',13);text(a,.50,.63,'保留激活供反向使用',11,ha='center');text(a,.75,.39,'更新时读取\n主权重与两份矩状态',11);save(f,'update-cycle')
        f,a=plot(4.7,left=.20,bottom=.16);vals=np.array(data['10-1']['allocations_bytes'])/1e9;left=np.zeros(2)
        for j,(l,c) in enumerate([('BF16 权重','blue'),('BF16 梯度','green'),('FP32 主权重','orange'),('一阶矩','purple'),('二阶矩','gray')]):a.barh([1,0],vals[:,j],left=left,height=.45,color=COL[c],edgecolor=COL['line'],label=l);left+=vals[:,j]
        a.set(yticks=[1,0],yticklabels=['推理权重','训练状态'],xlabel='容量（GB）',xlim=(0,145),ylim=(-.5,2.8));a.legend(frameon=False,ncol=2,loc='upper left');save(f,'1-state')
        for j,key in enumerate(['3/10','2/5','1/2']):
            f,a=plot(3.7);v=data['10-2']['compute'][key];a.bar(range(4),v,color=COL['blue'],edgecolor=COL['line']);a.scatter(range(4),data['10-2']['capacity'],marker='D',facecolors=COL['orange'],edgecolors=COL['line'],label='显存约束卡数下限',zorder=3)
            for x,y in enumerate(v):a.text(x,y+.8,str(y),fontsize=12,ha='center')
            a.set(xticks=range(4),xticklabels=['4090','A100','H100','B200'],ylabel='所需加速器数（张）',ylim=(0,50));a.legend(frameon=False);save(f,'2-budget' if j==0 else f'budget-{j}')
        for stage in range(4):
            f,a=canvas(4.6);text(a,.04,.94,['普通数据并行','ZeRO-1：优化器相关状态分片','ZeRO-2：再将梯度分片','ZeRO-3：再将模型权重分片'][stage],13)
            for row,(l,c,split) in enumerate([('权重','blue',stage>=3),('梯度','green',stage>=2),('主权重＋矩','purple',stage>=1)]):
                y=.64-row*.22;text(a,.02,y+.07,l,11)
                for i in range(4):box(a,.29+i*.17,y,.15,.15,('片 '+str(i)) if split else '完整',c,11)
            for i in range(4):text(a,.365+i*.17,.84,'卡 '+str(i),11,ha='center')
            save(f,f'zero-{stage}')
        f,a=canvas(4.7)
        for i in range(4):
            x=.04+i*.235;box(a,x,.70,.21,.15,f'卡 {i}：片 {i}',['blue','green','orange','purple'][i],11);arrow(a,(x+.105,.70),(x+.105,.45))
        box(a,.04,.18,.92,.27,'','gray');text(a,.5,.12,'GPU 0 的执行缓冲区',12,ha='center')
        for i in range(4):box(a,.06+i*.225,.23,.205,.15,f'片 {i}',['blue','green','orange','purple'][i],11)
        text(a,.5,.03,'用完释放完整缓冲，原始分片继续保留',11,ha='center');save(f,'3-sharding')
        f,a=canvas(4.1);box(a,.04,.59,.29,.20,'上游梯度 dY','green');box(a,.63,.59,.33,.20,'本层：Y = XW','blue');arrow(a,(.33,.69),(.63,.69))
        box(a,.04,.18,.39,.20,'dX：传给前一层','green');box(a,.57,.18,.39,.20,'dW：更新本层参数','purple')
        arrow(a,(.70,.59),(.235,.38));arrow(a,(.83,.59),(.765,.38));save(f,'gradient-branches')
        for rebuild,name in [(False,'4-recompute'),(True,'recompute-rebuild')]:
            f,a=canvas(3.8);text(a,.04,.92,'反向前重建乘积' if rebuild else '保留前向计算的乘积',14)
            for i,l in enumerate(['前向相乘','等待反向','下投影反向']):text(a,.17+i*.33,.71,l,11,ha='center')
            box(a,.04,.41,.92,.14,'一直保留 a、u','blue')
            box(a,.71 if rebuild else .04,.18,.25 if rebuild else .92,.14,'h：6 MiB','orange',11)
            if rebuild:arrow(a,(.83,.41),(.83,.32))
            save(f,name)
        for gpu,name in [(False,'5-casting'),(True,'casting-gpu')]:
            f,a=canvas(4.5);box(a,.04,.65,.35,.20,'GPU：BF16\n96 MiB','blue');box(a,.61,.23,.35,.20,'CPU：FP32\n192 MiB','green')
            if gpu:box(a,.04,.23,.35,.20,'GPU：FP32\n192 MiB','orange');arrow(a,(.215,.65),(.215,.43));arrow(a,(.39,.33),(.61,.33));text(a,.5,.52,'链路传 192 MiB',11,ha='center')
            else:box(a,.61,.65,.35,.20,'CPU：BF16\n96 MiB','blue');arrow(a,(.39,.75),(.61,.75));arrow(a,(.785,.65),(.785,.43));text(a,.5,.52,'链路传 96 MiB',11,ha='center')
            text(a,.5,.08,'GPU 转换时同时占用 288 MiB' if gpu else '格式转换在 CPU 完成',12,ha='center');save(f,name)
        d=data['10-6'];f,a=plot(4.2)
        for k,l,c in [('total_gib','分片＋10 GiB 临时缓冲','#267398'),('persistent_gib','训练状态分片','#388768')]:a.plot(d['participants'],d[k],label=l,color=c)
        a.axhline(22,ls='--',color='#a56c28',label='RTX 4090 可用 22 GiB');a.set(xlabel='分片参与者数',ylabel='每卡显存需求（GiB）',ylim=(0,47));a.legend(frameon=False);save(f,'6-candidates')
        for i,(label,d) in enumerate(data['10-7'].items()):
            f,a=plot(4.1,left=.18)
            for e in d['events']:
                if e['stage'] is None:continue
                a.barh(e['stage'],e['duration']*1000,left=e['start']*1000,height=.58,color=COL[{'F':'blue','B':'orange','update':'green'}[e['kind']]],edgecolor=COL['line'],linewidth=.5)
            a.set(yticks=range(4),yticklabels=[f'阶段 {j}' for j in range(4)],xlim=(0,355),xlabel='时间（ms）');a.invert_yaxis();a.set_title(f'{label}：{d["summary"]["step_makespan_seconds"]*1000:.0f} ms',loc='left');save(f,'7-pipeline' if i==0 else 'pipeline-1f1b')
        f,a=plot(3.4,left=.22)
        for e in data['10-7']['1F1B']['events']:
            if e['stage']==3 and e['end']*1000>65 and e['start']*1000<110:a.barh(0,min(e['end']*1000,110)-max(e['start']*1000,65),left=max(e['start']*1000,65),height=.5,color=COL[{'F':'blue','B':'orange','update':'green'}[e['kind']]],edgecolor=COL['line'])
        a.axvspan(93,95,color=COL['gray']);a.annotate('等输入：2 ms',(94,0),xytext=(82,.7),fontsize=12,arrowprops={'arrowstyle':'->'});a.set(xlim=(65,110),ylim=(-.6,1.2),yticks=[0],yticklabels=['阶段 3'],xlabel='时间（ms）',xticks=[65,75,85,95,105]);save(f,'pipeline-gap')
        f,a=plot(3.7)
        for i,(l,d) in enumerate(data['10-7'].items()):a.bar(np.arange(4)+(i-.5)*.34,np.array(d['summary']['reserved_activation_scope_peak_bytes'])/1e9,.32,color=COL[['blue','orange'][i]],edgecolor=COL['line'],label=l)
        a.set(xticks=range(4),xticklabels=[f'阶段 {i}' for i in range(4)],ylabel='激活与收发缓冲峰值（GB）',ylim=(0,3.4));a.legend(frameon=False);save(f,'pipeline-memory')
        for busy,name in [(False,'8-overlap'),(True,'overlap-busy')]:
            f,a=plot(3.2,left=.20);a.barh(1,5,color=COL['blue'],height=.5,edgecolor=COL['line'])
            if busy:a.barh(0,4,color=COL['gray'],height=.5,edgecolor=COL['line'])
            a.barh(0,3,left=4 if busy else 0,color=COL['orange'],height=.5,edgecolor=COL['line']);a.axvline(5,ls='--',color='#666');a.set(xlim=(0,7.5),yticks=[1,0],yticklabels=['独立计算','链路'],xlabel='时间（ms）');save(f,name)
        for i,lengths in enumerate(data['10-9']['lengths']):
            f,a=plot(4.8,left=.20);offset=0
            for n,c in zip(lengths,['blue','green']):a.add_patch(Polygon([(offset,offset),(offset,offset+n),(offset+n,offset+n)],facecolor=COL[c],edgecolor=COL['line']));offset+=n
            a.set(xlim=(0,8192),ylim=(8192,0),xticks=[0,lengths[0],8192],yticks=[0,lengths[0],8192],xlabel='键的 token 索引',ylabel='查询 token 索引');a.set_aspect('equal');save(f,'9-attention-area' if i==0 else 'attention-unequal')
        f,a=canvas(4.1);box(a,.04,.65,.92,.17,'已训练：到 batch 100','green')
        for i in range(8):box(a,.04+(i%4)*.235,.37-(i//4)*.20,.21,.14,str(101+i),'blue' if i<4 else 'gray',12)
        text(a,.5,.05,'101—108：已安排准备，仍等待训练',12,ha='center');save(f,'10-input-queue')
        f,a=canvas(4.0)
        for i in range(4):
            x=.04+i*.235;box(a,x,.66,.215,.18,f'旧 {i}',['blue','green','orange','purple'][i])
            for j in range(2):box(a,x+j*.1175,.22,.10,.18,str(i*2+j),['blue','green','orange','purple'][i]);arrow(a,(x+.1075,.66),(x+j*.1175+.05,.40))
        text(a,.5,.07,'新分片：每片 1536 行；旧分片：3072 行',11,ha='center');save(f,'11-resharding')
        f,a=canvas(4.4)
        for i,(l,c) in enumerate([('捕获一致状态','blue'),('复制到独立缓冲','orange'),('后台写入数据','blue'),('提交完整快照','green')]):
            y=.75-i*.21;box(a,.23,y,.54,.15,l,c)
            if i<3:arrow(a,(.5,y),(.5,y-.06))
        text(a,.80,.60,'训练\n可继续',11);save(f,'checkpoint-commit')
        for i,(l,rows) in enumerate(data['10-12']['timelines'].items()):
            f,a=plot(3.9,left=.19)
            for j,r in enumerate(rows):
                t=r['seconds'];a.barh(j,.5,left=t['capture'],color=COL['orange'],height=.5,edgecolor=COL['line']);a.barh(j,min(t['durable'],50)-t['upload_start'],left=t['upload_start'],color=COL['blue'],height=.5,edgecolor=COL['line'])
                if t['durable']>50:a.barh(j,t['durable']-50,left=50,color='white',hatch='///',height=.5,edgecolor=COL['line'])
                a.plot(t['durable'],j,'o',mfc=COL['green'] if t['durable']<=50 else 'white',mec=COL['line']);a.text(t['durable'],j+.4,str(t['durable'])+' s',fontsize=11,ha='center')
            a.axvline(50,ls='--',color='#a56c28');a.set(xlim=(18,59),ylim=(-.5,1.8),yticks=[0,1],yticklabels=['快照 1','快照 2'],xlabel='时间（s）');a.invert_yaxis();a.set_title(l+'：故障发生在 50 s',loc='left');save(f,'12-recovery' if i==0 else 'recovery-fast')
        d=data['10-13'];t=np.array(d['interval_seconds']);c=d['checkpoint_seconds'];lam=d['job_failure_rate_per_second'];r=d['restore_seconds'];f,a=plot(4.1)
        for y,l,color in [(c/t,'保存','#267398'),(lam*t/2,'重做','#a56c28'),(c/t+lam*t/2+lam*r,'总附加时间','#388768')]:a.plot(t/60,y*100,label=l,color=color)
        a.set(xlabel='保存间隔（分钟）',ylabel='每单位有用训练的附加时间（%）');a.legend(frameon=False);save(f,'13-save-interval')
        f,a=canvas(4.2)
        for i,(l,c) in enumerate([('生成\n12 条/s','blue'),('验证\n6 条/s','orange'),('学习\n8 条/s','green')]):box(a,.04+i*.34,.48,.24,.24,l,c)
        for x in [.28,.62]:arrow(a,(x,.60),(x+.10,.60))
        text(a,.70,.34,'保留 75% → 4.5 条/s',11,ha='center');arrow(a,(.84,.48),(.84,.16),'control');arrow(a,(.84,.16),(.16,.16),'control');arrow(a,(.16,.16),(.16,.48),'control');text(a,.5,.08,'新权重用于后续生成',12,ha='center');save(f,'14-rl-flow')
        for i,key in enumerate(['restore_bytes','staged_bytes']):
            f,a=plot(4.1,bottom=.26);v=np.array(data['10-15'][key])/2**30;a.step(range(4),v,where='post',color='#267398');a.plot(range(4),v,'o',color='#267398');a.axhline(80e9/2**30,ls='--',color='#a56c28',label='H100 SXM 74.5 GiB');a.legend(frameon=False,loc='upper right');a.set(xticks=range(4),xticklabels=['训练\n结束','加载权重\n分配 KV' if i==0 else '只加载\n权重','释放\n训练状态','开始\n生成'],ylabel='显存占用（GiB）',ylim=(0,100));a.text(1,v[1]+4,f'峰值 {v[1]:.1f} GiB',fontsize=12,ha='center');save(f,'15-rl' if i==0 else 'rl-staged')
        f,a=canvas(4.5)
        for row,(l,c) in enumerate([('$\\mu$：实际生成样本的策略','blue'),('$\\pi_{\\mathrm{old}}$：本轮优化开始时的策略','orange'),('$\\pi_\\theta$：本轮更新中的当前策略','green')]):
            y=.70-row*.28;box(a,.04,y,.92,.18,l,c)
            if row<2:arrow(a,(.5,y),(.5,y-.10),'control')
        text(a,.5,.04,'对同一前缀与 token 分别记录概率',12,ha='center');save(f,'policy-versions')
        for async_,name in [(False,'16-async-cycle'),(True,'async-overlap')]:
            f,a=plot(3.5,left=.20);items=[(2,0,40,'blue'),(1,0 if async_ else 40,16,'green'),(0,40 if async_ else 56,4,'orange')]
            for row,s,d,c in items:a.barh(row,d,left=s,height=.5,color=COL[c],edgecolor=COL['line'])
            a.set(yticks=[2,1,0],yticklabels=['生成','学习','权重同步'],xlim=(0,62),xlabel='稳态周期内时间（s）');save(f,name)
        f,a=canvas(4.9);box(a,.04,.69,.40,.20,'生成时记录\n样本 A／token 17／层 3','blue',11);box(a,.60,.69,.36,.20,'专家 ID：[2, 7]','orange',11);arrow(a,(.44,.79),(.60,.79))
        box(a,.60,.28,.36,.20,'训练时仍选 2、7','orange',11);arrow(a,(.78,.69),(.78,.48),'control');box(a,.04,.28,.40,.20,'用当前权重计算\n分数、输出与梯度','green',11);arrow(a,(.60,.38),(.44,.38));text(a,.5,.08,'记录固定离散选择；当前权重参与数值计算',11,ha='center');save(f,'17-replay')
        f,a=plot(3.8,left=.18);rows=data['10-18']['rows']
        for i,r in enumerate(rows):
            left=0
            for j,(v,c) in enumerate(zip(r['days'],['blue','orange','gray'])):a.barh(i,v,left=left,height=.45,color=COL[c],edgecolor=COL['line'],label=['基础训练','保存恢复','计划停顿'][j] if i==0 else None);left+=v
            a.text(left+.4,i,f'{left:.1f}',fontsize=12,va='center')
        a.axvline(30,ls='--',color='#a56c28');a.set(yticks=[0,1],yticklabels=['32 卡','48 卡'],xlabel='完成时间（天）',xlim=(0,41),ylim=(-.5,2.0));a.legend(frameon=False,ncol=2,loc='upper left');save(f,'18-deadline')
        d=data['10-19'];f,a=plot(3.9)
        for k,c in zip(d['curves'],['#388768','#267398','#a56c28']):a.plot(d['relative_bandwidth'],d['curves'][k],color=c,label=f'原通信占比 {float(k):.0%}')
        a.set(xlabel='新带宽／原带宽',ylabel='新耗时／原耗时');a.legend(frameon=False);save(f,'19-hardware')
        f,a=plot(4.2)
        for (key,d),l,c in zip(data['10-20'].items(),['A100','H100','B200'],['#267398','#388768','#a56c28']):a.plot(np.array(d['parameters'])/1e12,np.array(d['continuous_required_devices'])/1e4,color=c,label=l+'（40%）');a.plot(np.array(d['parameters'])/1e12,np.array(d['continuous_required_devices_mfu50'])/1e4,color=c,ls='--',lw=1)
        a.plot([],[],color='#666',ls='--',lw=1,label='虚线：50%');a.axhline(1.6384,ls='--',color='#666');a.set(yscale='log',xlabel='稠密模型参数量（万亿）',ylabel='90 天所需加速器（万张）');a.legend(frameon=False);save(f,'20-scale')
        # 21-23: where each bubble-compressing schedule fills the 1F1B gaps; same scale and colours as 17/18.
        fill={'F':'blue','B':'orange','X':'orange','W':'purple','update':'green'}
        for key,name in [('10-21','pipeline-interleaved'),('10-22','pipeline-zero-bubble'),('10-23','pipeline-dualpipe')]:
            d=data[key];f,a=plot(4.1,left=.18)
            for e in d['events']:
                second=e['chunk']==1 or e['direction']==1
                a.barh(e['stage'],e['duration']*1000,left=e['start']*1000,height=.58,color=COL[fill[e['kind']]],edgecolor=COL['line'],linewidth=.5,hatch='////' if second else None)
            a.set(yticks=range(4),yticklabels=[f'阶段 {j}' for j in range(4)],xlim=(0,355),xlabel='时间（ms）');a.invert_yaxis();a.set_title(f"{d['label']}：{d['summary']['step_makespan_seconds']*1000:.0f} ms",loc='left');save(f,name)
        # 24: smaller pipeline bubbles trade against transfers, activation peaks and parameters.
        d=data['10-24'];names=list(d);x=np.arange(4)
        f,(a1,a2)=plt.subplots(1,2,figsize=(420/72,3.9));f.subplots_adjust(left=.125,right=.94,bottom=.24,top=.80,wspace=.42)
        a1.bar(x-.17,[d[k]['makespan_m8_ms'] for k in names],.32,color=COL['blue'],edgecolor=COL['line'],label='8 个 micro-batch')
        a1.bar(x+.17,[d[k]['makespan_m16_ms'] for k in names],.32,color=COL['orange'],edgecolor=COL['line'],label='16 个 micro-batch')
        for xx,k in zip(x,names):a1.text(xx-.17,d[k]['makespan_m8_ms']+8,f"{d[k]['makespan_m8_ms']:.0f}",ha='center',fontsize=11);a1.text(xx+.17,d[k]['makespan_m16_ms']+8,f"{d[k]['makespan_m16_ms']:.0f}",ha='center',fontsize=11)
        a1.set(xticks=x,ylabel='完成时间（ms）',ylim=(0,700));a1.set_xticklabels(['1F1B','交错 v=2','零气泡','DualPipe'],rotation=22,ha='right');a1.legend(frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.5,1.01),fontsize=11)
        peaks=[max(d[k]['stage_peaks_m8_mib']) for k in names]
        a2.bar(x,peaks,.55,color=COL['green'],edgecolor=COL['line'])
        for xx,p in zip(x,peaks):a2.text(xx,p+30,f'{p:,.0f}',ha='center',fontsize=11)
        a2.set(xticks=x,ylabel='最大单阶段峰值（MiB）',ylim=(0,1900));a2.set_xticklabels(['1F1B','交错 v=2','零气泡','DualPipe'],rotation=22,ha='right')
        out.save(f,'figure-10-pipeline-schedules')
        # 29: capacity factor trades dropped assignments against padded rows.
        d=data['10-29'];f,a=plot(3.9)
        a.bar(x-.17,[v*100 for v in d['model_dropped_fraction']],.32,color=COL['orange'],edgecolor=COL['line'],label='丢弃的 dispatch 占比')
        a.bar(x+.17,[v*100 for v in d['model_padded_fraction']],.32,color=COL['blue'],edgecolor=COL['line'],label='补零行占执行量')
        for xx,dr,pa in zip(x,d['model_dropped_fraction'],d['model_padded_fraction']):
            a.text(xx-.17,dr*100+1.2,f'{dr*100:.1f}',ha='center',fontsize=11);a.text(xx+.17,pa*100+1.2,f'{pa*100:.0f}',ha='center',fontsize=11)
        a.set(xticks=x,xticklabels=['1.0','1.25','1.5','2.0'],xlabel='容量因子 c',ylabel='占比（%）',ylim=(0,58));a.legend(frameon=False,loc='upper left')
        out.save(f,'figure-10-moe-capacity')

        # Overhauled figure-10-critical-batch (Fig 10-7) with clean label offsets
        f,a=plot(4.4,bottom=.21,top=.86)
        x_cb=np.arange(5)
        weak_days=[19.4,15.7,13.8,12.8,12.1]
        strong_days=[19.4,9.8,5.0,2.6,0.8]
        a.plot(x_cb,weak_days,'o-',color='#a56c28',lw=2.2,label='弱扩展（每卡微批固定，全局批随卡数放大）')
        a.plot(x_cb,strong_days,'s-',color='#267398',lw=2.2,label='强扩展（全局批固定 3.15M token，约 $1.57 B_{\\mathrm{noise}}$）')
        a.axhline(11.9,ls='--',color='#a56c28',lw=1.3)
        a.text(2.85,9.7,'弱扩展渐近下界 11.9 天（上限 $1.64\\times$）',fontsize=11,color='#a56c28',ha='center')
        for idx_cb,val_w in enumerate(weak_days):
            a.text(idx_cb-(0.08 if idx_cb==0 else 0.0),val_w+0.9,f'{val_w:.1f} 天',ha='center',fontsize=11,color='#a56c28')
        for idx_cb,val_s,dx_s,dy_s in [(1,9.8,-.32,-2.1),(2,5.0,-.28,-2.2),(3,2.6,.0,1.05),(4,0.8,.0,1.05)]:
            a.text(idx_cb+dx_s,val_s+dy_s,f'{val_s:.1f} 天',ha='center',fontsize=11,color='#267398')
        a.set(xticks=x_cb,xticklabels=['48 卡\n(3.15M)','96 卡\n(6.29M)','192 卡\n(12.6M)','384 卡\n(25.2M)','1536 卡\n(101M)'],ylabel='纯训练完成时间（天）',xlim=(-.42,4.45),ylim=(-1,28.5))
        a.set_title('临界批大小约束下弱扩展与强扩展的完成时间对比',fontsize=12,pad=8)
        a.legend(frameon=False,loc='upper right',fontsize=11)
        out.save(f,'figure-10-critical-batch')

        # Overhauled figure-10-zero3-timeline (Fig 10-12)
        f,a=canvas(4.5)
        text(a,.03,.93,'ZeRO-3 单微批逐层预取与梯度归约重叠（仅首尾词嵌入层通信暴露）',12)
        text(a,.03,.67,'GPU 计算',11)
        box(a,.17,.54,.13,.25,'等待\n首层','gray',11)
        box(a,.31,.54,.26,.25,'前向传播 L0–L35\n(逐层计算)','blue',11)
        box(a,.58,.54,.26,.25,'反向传播 L35–L0\n(求梯度 dX, dW)','green',11)
        box(a,.85,.54,.12,.25,'等待\n末层','gray',11)
        text(a,.03,.36,'集合通信',11)
        box(a,.17,.24,.13,.24,'首层 AG\n(0.04 s)','orange',11)
        box(a,.31,.24,.26,.24,'预取后续层 AG\n(单层 ~12 ms 隐藏)','purple',11)
        box(a,.58,.24,.26,.24,'反向 AG + 梯度 RS\n(与反向计算重叠)','purple',11)
        box(a,.85,.24,.12,.24,'末层 RS\n(0.04 s)','orange',11)
        box(a,.17,.03,.80,.16,'每步 385 GB 通信（链路 12.0 s）中 99.3% 被计算掩盖\n仅首尾词嵌入层合计暴露约 0.08 s','gray',11)
        out.save(f,'figure-10-zero3-timeline')

        # New Path-A figure: ch10-rl-weight-handoff-and-stream-trainer (Fig 10-46)
        f,(ax1,ax2)=plt.subplots(2,1,figsize=(420/72,5.8))
        f.subplots_adjust(left=.23,right=.96,bottom=.10,top=.92,hspace=.64)
        stages_x=[0,1,2,3]
        restore_gib=[44.0,83.26,43.26,43.26]
        staged_gib=[44.0,59.26,19.26,43.26]
        ax1.plot(stages_x,restore_gib,'s--',color='#a95159',lw=2.0,label='全量恢复（83.3 GiB OOM）')
        ax1.plot(stages_x,staged_gib,'o-',color='#388768',lw=2.2,label='分段唤醒（59.3 GiB）')
        ax1.axhline(74.5,ls=':',color='#a56c28',lw=1.4,label='H100 上限 74.5 GiB')
        ax1.text(0.56,86.5,'83.3 GiB (OOM)',ha='center',fontsize=11,color='#a95159')
        ax1.text(0.58,28.5,'59.3 GiB (省 24 GiB)',ha='center',fontsize=11,color='#388768')
        ax1.set(xticks=stages_x,xticklabels=['训练结束\n(44.0 GiB)','同步/加载\n推理权重','释放训练\n状态','分配 24 GiB KV\n开始生成'],ylabel='显存 (GiB)',xlim=(-.25,3.35),ylim=(0,126))
        ax1.set_title('① Qwen3-8B 共置 RL 显存切换：全量恢复 vs. 分段唤醒',fontsize=12,pad=6)
        ax1.legend(frameon=False,loc='upper right',fontsize=11)

        # Bottom panel: RollPacker Stream Trainer 8-GPU timeline (matching 算例 10-B: t=10s split, 29s/43s/55s vs 40s/60s)
        ax2.barh(2,40,left=0,height=.48,color=COL['blue'],edgecolor=COL['line'],label='Rollout 采样')
        ax2.barh(2,20,left=40,height=.48,color=COL['green'],edgecolor=COL['line'],label='梯度累积与更新')
        ax2.text(60.8,2,'60 s',va='center',fontsize=11)

        ax2.barh(1,10,left=0,height=.48,color=COL['blue'],edgecolor=COL['line'])
        ax2.barh(1,33,left=10,height=.48,color=COL['orange'],edgecolor=COL['line'],label='长尾续生/流训切换')
        ax2.barh(1,12,left=43,height=.48,color=COL['green'],edgecolor=COL['line'])
        ax2.text(55.8,1,'55 s',va='center',fontsize=11,color='#388768')

        ax2.barh(0,10,left=0,height=.48,color=COL['blue'],edgecolor=COL['line'])
        ax2.barh(0,19,left=10,height=.48,color=COL['green'],edgecolor=COL['line'])
        ax2.barh(0,14,left=29,height=.48,color=COL['gray'],edgecolor=COL['line'])
        ax2.barh(0,12,left=43,height=.48,color=COL['green'],edgecolor=COL['line'])
        ax2.text(55.8,0,'55 s (-8.3%)',va='center',fontsize=11,color='#388768')

        for xv in [10,29,43]:
            ax2.axvline(xv,ls='--',color='#777777',lw=1.0)
        ax2.set(yticks=[2,1,0],yticklabels=['朴素共置 (8 卡)','生成组 (4 卡)','流训组 (4 卡)'],xlabel='单轮迭代时间（s）',xlim=(0,68),ylim=(-.55,2.75),xticks=[0,10,20,29,40,43,55,60])
        ax2.invert_yaxis()
        ax2.set_ylim(2.65,-1.18)
        ax2.set_title('② RollPacker 流式训练器（留守单卡 KV 9 GiB ≤ 12 GiB）',fontsize=12,pad=6)
        ax2.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(0.46,1.02),columnspacing=0.9,handletextpad=0.4,fontsize=11)
        out.save(f,'ch10-rl-weight-handoff-and-stream-trainer')

        # New Path-B figure 1: ch10-dualpipe-vs-1f1b-schedule (Fig 10-51, D14-02)
        f,a=canvas(6.4)
        text(a,.25,.95,'A. 单向 1F1B 流水线',12,ha='center')
        text(a,.745,.95,'B. DualPipe 双向流水线',12,ha='center')
        a.plot([.495,.495],[.03,.97],color=COL['line'],ls='--',lw=1.0)

        box(a,.03,.75,.44,.15,'单向微批注入（Stage 0 -> PP-1）\n首尾气泡：(PP-1)(F+B)','orange',11)
        box(a,.03,.54,.44,.16,'参数驻留：1× 模型分片\n在途激活上限：PP 份','blue',11)
        box(a,.03,.31,.44,.18,'单流串行瓶颈：\n跨机 EP All-to-All 难与计算全重叠\n首尾预热与排空暴露明显','gray',11)
        box(a,.03,.05,.44,.20,'ZB1P（零气泡变体）对比：\n剥离权重梯度 W 填入空隙\n气泡：(PP-1)(F+B-2W)，1× 参数','purple',11)

        box(a,.52,.75,.45,.15,'两端相向注入（Stage 0 <-> PP-1）\n气泡减半：(PP/2-1)(F&B+B-3W)','green',11)
        box(a,.52,.54,.45,.16,'配对块 F&B 细粒度重叠：\n正向 EP 通信掩盖于反向计算下','blue',11)
        box(a,.52,.31,.45,.18,'显存交换代价（双向驻留）：\n每设备需驻留 2× 参数副本\n在途激活上限增至 PP+1 份','orange',11)
        box(a,.52,.05,.45,.20,'DualPipeV（V 型折叠变体）：\n单卡承载 2 个对称流水阶段\n设备数从 PP 减半至 PP/2','gray',11)
        out.save(f,'ch10-dualpipe-vs-1f1b-schedule')

        # New Path-B figure 2: ch10-training-bias-balance-vs-inference-eplb (Fig 10-52, D14-03)
        f,a=canvas(6.4)
        text(a,.25,.95,'A. 训练期：无辅助损失动态偏置',12,ha='center')
        text(a,.745,.95,'B. 推理期：EPLB 物理副本均衡',12,ha='center')
        a.plot([.495,.495],[.03,.97],color=COL['line'],ls='--',lw=1.0)

        box(a,.03,.75,.44,.15,'路由判据加入动态偏置 b_i：\nTopk(s+b)，过载减γ、欠载加γ','blue',11)
        box(a,.03,.54,.44,.16,'乘入专家输出的权重保持无偏：\ng\'_it = s_it（不干扰主损失梯度）','green',11)
        box(a,.03,.31,.44,.18,'改变每个 Token 选中的逻辑专家：\n边缘 Token 改道流向欠载专家\n后续反向更新使专家参数自适应','orange',11)
        box(a,.03,.05,.44,.20,'辅助机制与资源开销：\n叠加极小系数序列平衡损失 L_Bal\n每专家仅 1 个标量 b_i，零显存','gray',11)

        box(a,.52,.75,.45,.15,'模型权重与偏置 b_i* 已冻结定型：\n严禁在推理期篡改逻辑 Topk 选择','purple',11)
        box(a,.52,.54,.45,.16,'严格保持数学输出等价：\n每个 Token 命中的逻辑专家不变','green',11)
        box(a,.52,.31,.45,.18,'物理层冗余复制与重排（EPLB）：\n为热点逻辑专家复制多份物理权重\n均匀分流跨卡 All-to-All 请求','blue',11)
        box(a,.52,.05,.45,.20,'系统资源代价（如 EP144）：\n需预留冗余专家 HBM（如 32 槽）\n周期性按观测窗口跨网迁移副本','gray',11)
        out.save(f,'ch10-training-bias-balance-vs-inference-eplb')

        # New Path-D figure 1: ch10-sglang-weight-update-six-stage-boundary (Fig 10-53)
        f,a=canvas(6.4)
        text(a,.25,.95,'A. SGLang 热更新六阶段职责边界',12,ha='center')
        text(a,.745,.95,'B. [成功,成功,失败,成功] 失败边界',12,ha='center')
        a.plot([.495,.495],[.03,.97],color=COL['line'],ls='--',lw=1.0)

        box(a,.03,.76,.44,.14,'① 请求进入与 ② 暂停生成\n读锁放行；is_pause 挂起/撤回','blue',11)
        arrow(a,(.25,.76),(.25,.67),'data')
        box(a,.03,.50,.44,.16,'③ 提交更新与 ④ 汇总 all(success)\n磁盘/张量写锁，IPC/分布式无锁','orange',11)
        arrow(a,(.25,.50),(.25,.41),'data')
        box(a,.03,.24,.44,.16,'⑤ 发布管理信息（仅全成功生效）\n更新 model_path 并递增版本号','green',11)
        box(a,.03,.04,.44,.15,'⑥ 恢复生成（不查前序成败）\ncontinue_generation 须显式调用','gray',11)

        box(a,.52,.76,.45,.14,'四分片返回 [T, T, F, T]：\nall(success)=False，拒发新版本号','purple',11)
        box(a,.52,.53,.45,.17,'就地修改 (load_weights) 无回滚：\n分片 1/2/4 新权重，分片 3 半成品','orange',11)
        box(a,.52,.30,.45,.17,'投机解码 (EAGLE) 非原子更新：\ntarget 成功而 draft 失败致失配','orange',11)
        box(a,.52,.04,.45,.20,'恢复正道：禁直接恢复生成\n须统一下发重载旧权重并清空 KV','green',11)
        out.save(f,'ch10-sglang-weight-update-six-stage-boundary')

        # New Path-D figure 2: ch10-bytecheckpoint-resharding-and-checkmate (Fig 10-54)
        f,a=canvas(6.4)
        text(a,.25,.95,'A. ByteCheckpoint 异步与重分片',12,ha='center')
        text(a,.745,.95,'B. Checkmate 无盘内存对等恢复',12,ha='center')
        a.plot([.495,.495],[.03,.97],color=COL['line'],ls='--',lw=1.0)

        box(a,.03,.76,.44,.14,'进程级 Stager + 计划缓存 (2.14)\nstaging 后即恢复训练 (0.59 s)','blue',11)
        box(a,.03,.53,.44,.17,'1856 卡 H200 Llama3-70B 实测：\n后台落盘 436 s -> 67 s (6.5×)','green',11)
        box(a,.03,.30,.44,.17,'2400 -> 4800 卡重分片：\nGPU 62.10 s；全状态 401.21 s','orange',11)
        box(a,.03,.04,.44,.20,'木桶瓶颈：CPU Dataloader 长尾\n需异步双缓冲与流式游标解耦','gray',11)

        box(a,.52,.76,.45,.14,'远端 PFS 带宽瓶颈：\n万卡并发写数 TB 易触发严重拥塞','orange',11)
        box(a,.52,.53,.45,.17,'Checkmate (NSDI 26) 内存冗余：\n经 RDMA 镜像至邻近健康节点','blue',11)
        box(a,.52,.30,.45,.17,'单节点故障恢复路径：\n备用节点从对等端内存拉取状态','green',11)
        box(a,.52,.04,.45,.20,'结合张量重计算消除磁盘往返：\n恢复耗时较传统 PFS 缩短十倍','purple',11)
        out.save(f,'ch10-bytecheckpoint-resharding-and-checkmate')

        # New Path-E3 figure 1: ch10-i20-double-buffer-dma-staging-timeline (Fig 10-55, Locked Case I20)
        f,a=canvas(6.4)
        text(a,.25,.95,'A. 串行 vs 单设备槽 vs 双缓冲流水',12,ha='center')
        text(a,.745,.95,'B. 缓冲区复用约束与链路降级翻转',12,ha='center')
        a.plot([.495,.495],[.03,.97],color=COL['line'],ls='--',lw=1.0)

        box(a,.03,.75,.44,.15,'8 块 × 64 MiB BF16 激活张量\n准备 1ms / H2D 2.604ms / 算 4ms','gray',11)
        box(a,.03,.53,.44,.16,'① 同步串行 (1 主机+1 设备槽)\n8 × 7.604 = 60.833 ms','orange',11)
        box(a,.03,.30,.44,.17,'② 单设备槽 (2 主机+1 设备槽)\nDMA 等计算完：53.833 ms','purple',11)
        box(a,.03,.05,.44,.19,'③ 乒乓双槽 (2 主机+2 设备槽)\n7.604 + 7×4 = 35.604 ms (1.71×)','green',11)

        box(a,.52,.75,.45,.15,'事件同步与安全复用边界：\n主机槽等 DMA；设备槽等计算','blue',11)
        box(a,.52,.53,.45,.16,'暂存空间代价：\n双侧各 2×64 = 128 MiB 缓冲','blue',11)
        box(a,.52,.30,.45,.17,'④ PCIe 降级至 12 GiB/s：\n单块 H2D 升至 5.208 ms (>4ms)','orange',11)
        box(a,.52,.05,.45,.19,'瓶颈由计算翻转回传输：\n双缓冲耗时升至 46.667 ms','gray',11)
        out.save(f,'ch10-i20-double-buffer-dma-staging-timeline')

        # New Path-E3 figure 2: ch10-distributed-muon-newton-schulz-and-rl-readmission (Fig 10-56)
        f,a=canvas(6.4)
        text(a,.25,.95,'A. 分布式 Muon 正交化与重叠',12,ha='center')
        text(a,.745,.95,'B. RL 节点掉线弹性重准入状态机',12,ha='center')
        a.plot([.495,.495],[.03,.97],color=COL['line'],ls='--',lw=1.0)

        box(a,.03,.75,.44,.15,'动量归约：跨 DP ReduceScatter\n分片动量 M_i，通信量减半','blue',11)
        arrow(a,(.25,.75),(.25,.66),'data')
        box(a,.03,.49,.44,.16,'5 步 BF16 Newton-Schulz 迭代\nX\' = aX + bAX + cA^2X (A=XX^T)','green',11)
        box(a,.03,.27,.44,.17,'分片正交化与 AllGather 重叠\n第 l 层正交化掩盖 l-1 层广播','purple',11)
        box(a,.03,.05,.44,.17,'省去二阶矩 v，状态降 33%\n额外正交化 FLOPs 占比 < 1%','gray',11)

        box(a,.52,.75,.45,.15,'Rollout 掉线隔离 (DRAINING)\n健康节点接管未完成采样','orange',11)
        arrow(a,(.745,.75),(.745,.66),'data')
        box(a,.52,.49,.45,.16,'重启并建通信域 (SYNCING)\n禁旧版本权重直接入生成池','purple',11)
        box(a,.52,.27,.45,.17,'定向上传最新策略切片 v_curr\n校验哈希并清空残留 KV 池','blue',11)
        box(a,.52,.05,.45,.17,'版本屏障对齐后重入 (ACTIVE)\n严防跨代差样本污染策略梯度','green',11)
        out.save(f,'ch10-distributed-muon-newton-schulz-and-rl-readmission')

        # Path-F3 Fig 10-57: Selective Activation Recomputation (Core Attention + MLA Up-Proj + SwiGLU)
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(420 / 72, 6.8))
        f.subplots_adjust(left=0.16, right=0.95, bottom=0.10, top=0.92, hspace=0.52)
        for ax in (ax1, ax2):
            ax.spines[['top', 'right']].set_visible(False)

        recomp_modes = ['全量保存\n(无重算)', '仅重算注意力\n(Core Attn)', '细粒度重算\n(三模块联合)', '整层全重算\n(Full Ckpt)']
        act_mem_mib = [1424.0, 784.0, 240.0, 64.0]
        cols_rc = ['orange', 'blue', 'green', 'purple']
        x_rc = np.arange(len(recomp_modes))
        for xi, val, cn in zip(x_rc, act_mem_mib, cols_rc):
            ax1.bar(xi, val, width=0.44, color=COL[cn], edgecolor=BORDER_COL[cn], lw=1.2)
        lbls_rc = ['1424 MiB\n(+0% 算力)', '784 MiB\n(+4.2% 算力)', '240 MiB\n(+11.8% 算力)', '64 MiB\n(+33.3% 算力)']
        for xi, val, lr, cn in zip(x_rc, act_mem_mib, lbls_rc, cols_rc):
            ax1.text(xi, val + 45.0, lr, ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL[cn])
        ax1.set_xticks(x_rc, recomp_modes)
        ax1.set_xlim(-0.55, 3.55)
        ax1.set_ylim(0, 1820.0)
        ax1.set_yticks([0, 400, 800, 1200, 1600])
        ax1.set_ylabel('单层每微批保留激活（MiB）', fontsize=11)
        ax1.set_title('A  算例 10-6：四种重计算策略下单层保留激活与算力开销', loc='left', fontsize=12, weight='bold', pad=8)
        ax1.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E1')

        stage_act_gib = np.array([89.0, 49.0, 15.0, 4.0])
        eff_mfu_pct = np.array([0.0, 46.5, 51.4, 41.2])  # 0.0 represents OOM for No Recompute on 80GB H100
        x_s2 = np.arange(len(recomp_modes))
        w_s2 = 0.34

        ax2.bar(x_s2 - w_s2 / 2, stage_act_gib, width=w_s2, color=COL['blue'], edgecolor=BORDER_COL['blue'], lw=1.2, label='8 微批在途激活 (GiB)')
        ax2.bar(x_s2 + w_s2 / 2, eff_mfu_pct, width=w_s2, color=COL['green'], edgecolor=BORDER_COL['green'], lw=1.2, label='端到端有效 MFU (%)')
        for xi, vg, vm in zip(x_s2, stage_act_gib, eff_mfu_pct):
            ax2.text(xi - w_s2 * 0.62, vg + 2.5, f'{vg:.0f}G', ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL['blue'])
            m_lbl = 'OOM' if vm == 0.0 else f'{vm:.1f}%'
            m_col = TEXT_COL['orange'] if vm == 0.0 else TEXT_COL['green']
            ax2.text(xi + w_s2 * 0.62, vm + 2.5, m_lbl, ha='center', va='bottom', fontsize=11, weight='bold', color=m_col)
        ax2.set_xticks(x_s2, recomp_modes)
        ax2.set_xlim(-0.55, 3.55)
        ax2.set_ylim(0, 118.0)
        ax2.set_ylabel('激活显存 (GiB) / MFU (%)', fontsize=11)
        ax2.set_title('B  PP=8 首阶段在途激活显存与 80GB H100 有效 MFU 对比', loc='left', fontsize=12, weight='bold', pad=8)
        ax2.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E1')
        ax2.legend(frameon=True, loc='upper right', fontsize=11)
        out.save(f, 'ch10-selective-activation-recomputation-mla-and-swiglu')

        # Path-F3 Fig 10-58: Megatron-Core / TorchTitan 5D Hybrid Parallel Memory Breakdown
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(420 / 72, 6.8))
        f.subplots_adjust(left=0.16, right=0.95, bottom=0.10, top=0.92, hspace=0.52)
        for ax in (ax1, ax2):
            ax.spines[['top', 'right']].set_visible(False)

        p5_configs = ['3D 基线 (无重算)\nTP8·PP8', '4D + 整层重算\nTP4·EP16·PP8', '5D + 细粒度重算\nTP2·CP2·EP32']
        m_wt_grad = np.array([18.4, 14.2, 14.2])
        m_opt_shard = np.array([11.0, 21.0, 21.0])
        m_act_inflight = np.array([78.0, 18.0, 15.0])
        m_comm_buf = np.array([11.0, 11.0, 8.4])
        m_totals = m_wt_grad + m_opt_shard + m_act_inflight + m_comm_buf
        x_p5 = np.arange(len(p5_configs))
        w_p5 = 0.42

        ax1.bar(x_p5, m_wt_grad, width=w_p5, color=COL['blue'], edgecolor=BORDER_COL['blue'], lw=1.1, label='权重+梯度分片')
        ax1.bar(x_p5, m_opt_shard, width=w_p5, bottom=m_wt_grad, color=COL['purple'], edgecolor=BORDER_COL['purple'], lw=1.1, label='ZeRO-1 优化器分片')
        ax1.bar(x_p5, m_act_inflight, width=w_p5, bottom=m_wt_grad + m_opt_shard, color=COL['orange'], edgecolor=BORDER_COL['orange'], lw=1.1, label='在途微批激活')
        ax1.bar(x_p5, m_comm_buf, width=w_p5, bottom=m_wt_grad + m_opt_shard + m_act_inflight, color=COL['teal'], edgecolor=BORDER_COL['teal'], lw=1.1, label='通信与临时缓冲')
        ax1.axhline(79.1, color=BORDER_COL['orange'], ls='--', lw=1.4)
        ax1.text(2.48, 83.0, '上限 79.1 GiB', ha='right', va='bottom', fontsize=11, weight='bold', color=TEXT_COL['orange'])
        ax1.text(0.0, 122.0, '118.4 GiB (OOM)', ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL['orange'])
        ax1.text(1.0, 83.0, '64.2 GiB (41.2%)', ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL['blue'])
        ax1.text(2.0, 61.0, '58.6 GiB (51.4%)', ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL['green'])
        ax1.set_xticks(x_p5, p5_configs)
        ax1.set_xlim(-0.55, 2.55)
        ax1.set_ylim(0, 195.0)
        ax1.set_yticks([0, 40, 80, 120])
        ax1.set_ylabel('单卡峰值显存（GiB）', fontsize=11)
        ax1.set_title('A  算例 10-7：235B MoE 在三种并行方案下的单卡显存拆解', loc='left', fontsize=12, weight='bold', pad=8)
        ax1.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E1')
        ax1.legend(frameon=True, loc='upper right', ncol=2, fontsize=11)

        seq_lens = ['8K', '32K', '64K', '128K']
        mfu_tp8_nocp = np.array([46.2, 38.5, 31.0, 24.5])
        mfu_5d_hybrid = np.array([51.4, 50.2, 48.6, 46.8])
        x_sq = np.arange(len(seq_lens))

        ax2.plot(x_sq, mfu_5d_hybrid, color=BORDER_COL['green'], lw=2.4, marker='o', ms=5.5, label='5D 混合并行 (TP2×CP×EP32×PP8)')
        ax2.plot(x_sq, mfu_tp8_nocp, color=BORDER_COL['orange'], lw=2.2, marker='s', ms=5.5, label='传统重 TP 并行 (TP8 无 CP)')
        for xi, v5, vt in zip(x_sq, mfu_5d_hybrid, mfu_tp8_nocp):
            ax2.text(xi, v5 + 2.2, f'{v5:.1f}%', ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL['green'])
            ax2.text(xi, vt - 3.0, f'{vt:.1f}%', ha='center', va='top', fontsize=11, weight='bold', color=TEXT_COL['orange'])
        ax2.set_xticks(x_sq, seq_lens)
        ax2.set_xlim(-0.4, 3.4)
        ax2.set_ylim(12.0, 74.0)
        ax2.set_yticks([20, 30, 40, 50, 60])
        ax2.set_xlabel('训练上下文序列长度 S', fontsize=11)
        ax2.set_ylabel('集群有效算力利用率 MFU（%）', fontsize=11)
        ax2.set_title('B  长序列扩展下 5D 混合并行与重 TP 方案的 MFU 保持率对比', loc='left', fontsize=12, weight='bold', pad=8)
        ax2.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1')
        ax2.legend(frameon=True, loc='upper right', fontsize=11)
        out.save(f, 'ch10-megatron-core-torchtitan-5d-parallel-memory-breakdown')
        f, a = canvas(4.3)
        text(a, 0.04, 0.94, 'A. 固定权重版本的在线推理服务', 13.5)
        card(a, 0.04, 0.65, 0.41, 0.19, '只读权重存储（ROM/HBM）', '固定模型参数 v0', 'blue')
        card(a, 0.64, 0.65, 0.32, 0.19, '解码生成引擎', '维护可写请求 KV', 'green')
        arrow(a, (0.45, 0.745), (0.64, 0.745), label='只读复用', color='blue')
        text(a, 0.04, 0.48, 'B. 强化学习（RL）持续更新的策略回路', 13.5)
        card(a, 0.04, 0.18, 0.26, 0.20, '训练引擎', '反向传播更新参数', 'orange')
        card(a, 0.38, 0.18, 0.26, 0.20, '可写权重缓冲', '版本演进 v0 → v1', 'blue')
        card(a, 0.72, 0.18, 0.24, 0.20, '采样引擎', '切换至新权重 v1', 'green')
        arrow(a, (0.30, 0.28), (0.38, 0.28), color='orange')
        arrow(a, (0.64, 0.28), (0.72, 0.28), color='blue')
        badge(a, 0.50, 0.065, '每次策略更新后同步发布新权重，保证采样与训练策略一致', 'purple', size=11.0)
        out.save(f, 'figure-10-weight-update')
    from h2_fp8_nvfp4_figures import draw as draw_h2_fp8_nvfp4
    data.update(draw_h2_fp8_nvfp4(out))
    out.finish();return out.outputs,out.checks
