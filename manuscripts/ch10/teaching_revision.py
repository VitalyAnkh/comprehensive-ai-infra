"""Follow state ownership, then execution dependencies, then recoverable progress."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from figure_style import COL,STYLE,canvas,plot,text,box,arrow,Exporter

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
    from core_principles_figures import draw as draw_principles
    draw_principles(10, out)
    out.finish();return out.outputs,out.checks
