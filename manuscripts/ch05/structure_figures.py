"""Chapter-five mechanism figures added in the structure revision (book figure style)."""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Patch
import numpy as np
from figure_style import COL, STYLE, Exporter, canvas, plot, box, text, arrow
from figure_style.typography import configure_font

ROOT = Path(__file__).resolve().parents[2]


def cut(ax, p, q):
    ax.plot([p[0], q[0]], [p[1], q[1]], ls=(0, (5, 3)), lw=1.6, color=COL['ink'])


def draw(here, data=None):
    here = Path(here); out = Exporter(here); data = {} if data is None else data
    with plt.rc_context(STYLE):
        # 5.1.2 Copy paths between host memory and device memory.
        f, a = canvas(3.6)
        for x, title in [(.02, '主机内存（CPU）'), (.56, '显存（GPU）')]:
            a.add_patch(Rectangle((x, .06), .42, .82, facecolor=COL['white'], edgecolor=COL['line'], lw=.9, ls=(0, (1, 2))))
            text(a, x + .21, .93, title, 12, ha='center')
        box(a, .06, .66, .34, .16, '普通内存\n页面可被换出', 'gray', 11)
        box(a, .06, .42, .34, .16, '锁页缓冲\n驻留，可直接搬移', 'blue', 11)
        box(a, .06, .14, .34, .16, '结果缓冲', 'green', 11)
        box(a, .60, .66, .34, .16, '权重\n加载一次，长期驻留', 'gray', 11)
        box(a, .60, .42, .34, .16, '输入缓冲', 'blue', 11)
        box(a, .60, .14, .34, .16, '输出', 'green', 11)
        arrow(a, (.23, .66), (.23, .58)); text(a, .26, .625, '主机内复制', 11)
        arrow(a, (.40, .50), (.60, .50)); text(a, .50, .55, 'H2D', 11, ha='center')
        arrow(a, (.60, .22), (.40, .22)); text(a, .50, .27, 'D2H', 11, ha='center')
        box(a, .68, .335, .18, .07, 'kernel', 'white', 11)
        arrow(a, (.77, .42), (.77, .405)); arrow(a, (.77, .335), (.77, .30))
        out.save(f, 'figure-5-copy-paths')

        # 5.1.3 Two streams ordered by events; a slot is reused only after its reader finishes.
        f, a = canvas(3.4)
        text(a, .0, .70, '复制流', 12); text(a, .0, .32, '计算流', 12)
        box(a, .14, .62, .20, .16, 'H2D 批 0\n写入槽 A', 'blue', 11)
        box(a, .38, .62, .20, .16, 'H2D 批 1\n写入槽 B', 'green', 11)
        box(a, .76, .62, .20, .16, 'H2D 批 2\n写入槽 A', 'blue', 11)
        box(a, .38, .24, .30, .16, '计算 批 0\n读取槽 A', 'blue', 11)
        box(a, .72, .24, .24, .16, '计算 批 1\n读取槽 B', 'green', 11)
        arrow(a, (.34, .62), (.40, .40), 'control'); text(a, .10, .49, '事件：批 0 已传完', 11)
        arrow(a, (.68, .40), (.76, .62), 'control'); text(a, .44, .53, '事件：批 0 已用完', 11)
        a.plot([.14, .96], [.10, .10], color=COL['line'], lw=.8); text(a, .96, .05, '时间', 11, ha='right')
        out.save(f, 'figure-5-stream-event')

        # 5.5.1 Host preparation and accelerator compute: serial, pipelined, faster accelerator.
        f, a = canvas(4.6)
        scale = .80 / 160; x0 = .16
        panels = [(.70, '串行：每段准备 20 μs、计算 20 μs，四段共 160 μs', 40, 20, 20, 0),
                  (.38, '流水：准备下一段与计算当前段同时进行，共 100 μs', 20, 20, 20, 20),
                  (.06, '加速器提速到 5 μs：主机仍每 20 μs 才准备好一段，共 85 μs', 20, 20, 5, 20)]
        for y, title, period, host, acc, lag in panels:
            text(a, .0, y + .27, title, 11)
            text(a, .0, y + .175, '主机准备', 11); text(a, .0, y + .045, '加速器计算', 11)
            for i in range(4):
                hx = x0 + scale * period * i
                box(a, hx, y + .13, scale * host - .004, .09, str(i), 'orange', 11)
                ax_ = x0 + scale * (period * i + (host if lag == 0 else lag))
                if acc >= 10:
                    box(a, ax_, y, scale * acc - .004, .09, str(i), 'blue', 11)
                else:
                    box(a, ax_, y, scale * acc - .004, .09, '', 'blue', 11)
                    text(a, ax_ + scale * acc + .004, y + .045, str(i), 11)
        a.plot([x0, .96], [.015, .015], color=COL['line'], lw=.8)
        for t in (0, 40, 80, 120, 160):
            text(a, x0 + scale * t, -.01, f'{t}', 11, ha='center')
        out.save(f, 'figure-5-host-pipeline')

        # 5.5.2 Ordinary submission versus graph replay: launches change, kernels do not.
        f, a = canvas(3.2)
        for x, title, host_label in [(.03, '普通提交', '主机：六次 kernel launch'), (.53, '图重放', '主机：一次 graph launch')]:
            text(a, x, .93, title, 12); text(a, x, .80, host_label, 11); text(a, x, .08, '加速器：六个 kernel', 11)
            for i in range(6):
                bx = x + .072 * i
                box(a, bx, .18, .06, .14, str(i + 1), 'blue', 11)
                if x < .5:
                    box(a, bx, .56, .06, .14, str(i + 1), 'orange', 11); arrow(a, (bx + .03, .56), (bx + .03, .32))
        box(a, .53, .56, .16, .14, '图', 'orange', 11)
        a.plot([.56, .95], [.38, .38], color=COL['line'], lw=1)
        for i in range(6): a.plot([.56 + .072 * i, .56 + .072 * i], [.38, .34], color=COL['line'], lw=1)
        arrow(a, (.61, .56), (.61, .38))
        out.save(f, 'figure-5-launch-vs-graph')

        # 5.5.3 Real row counts padded into two buckets.
        f, a = plot(3.0, left=.30, bottom=.25); f.subplots_adjust(top=.80)
        rows = [256, 1536, 2048]; buckets = [512, 2048, 2048]; labels = ['256 行，8 次', '1536 行，1 次', '2048 行，1 次']
        a.barh(range(3), rows, color=COL['blue'], edgecolor=COL['line'], height=.55, label='实际行数')
        a.barh(range(3), [b - r for b, r in zip(buckets, rows)], left=rows, color=COL['gray'], edgecolor=COL['line'], height=.55, label='补齐到桶')
        for i, b in enumerate(buckets): a.text(b + 40, i, f'桶 {b}', va='center', fontsize=11)
        a.set(yticks=range(3), yticklabels=labels, xlim=(0, 2500), xlabel='每次调用执行的行数'); a.invert_yaxis()
        a.legend(frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(.5, 1.0))
        out.save(f, 'figure-5-shape-buckets')

        # 5.2.4 Three ways to split one matrix multiply (moved here from chapter six).
        f, a = canvas(4.8)
        for row, (title, left, right, result, c) in enumerate([
                ('切 M：输出行', '输入行分片', '权重复制', '不同输出行', 'blue'),
                ('切 N：输出列', '输入复制', '权重列分片', '不同输出列', 'green'),
                ('切 K：归约维', '输入列分片', '权重行分片', '同一输出的部分和', 'orange')]):
            y = .71 - row * .30
            text(a, .04, y + .19, title, 13)
            box(a, .04, y, .26, .13, left, c, 11); box(a, .36, y, .26, .13, right, c, 11); box(a, .70, y, .27, .13, result, c, 11)
            text(a, .33, y + .065, '×', 13, ha='center'); arrow(a, (.62, y + .065), (.70, y + .065))
        text(a, .50, .04, '输出分片按需收集；归约分片必须求和', 12, ha='center')
        out.save(f, 'figure-5-split-axes')
        # 5.1.5 Two resident blocks against the SM's three resource limits (calculations/results/sm-occupancy-book-tile.json).
        limits = dict(threads=2048, registers=65536, shared_bytes=233472, blocks=32)
        block = dict(threads=256, registers_per_thread=128, shared_bytes=98304, reserved_shared_bytes=1024)
        by = dict(threads=limits['threads'] // block['threads'],
                  registers=limits['registers'] // (block['threads'] * block['registers_per_thread']),
                  shared_memory=limits['shared_bytes'] // (block['shared_bytes'] + block['reserved_shared_bytes']),
                  blocks=limits['blocks'])
        resident = min(by.values())
        f, a = canvas(3.6)
        text(a, .0, .95, '两个驻留线程块（蓝：块 0，绿：块 1）', 12)
        x0, w = .30, .62
        rows = [('寄存器 64K 个', block['threads'] * block['registers_per_thread'] / limits['registers'], '两块填满，第三块放不下', True),
                ('共享内存 228 KB', (block['shared_bytes'] + block['reserved_shared_bytes']) / limits['shared_bytes'], '剩 34 KiB，第三块放不下', True),
                ('线程槽 2048 个', block['threads'] / limits['threads'], '用了 512 个，还能放 6 块', False)]
        for i, (label, frac, note, binding) in enumerate(rows):
            y = .73 - i * .27
            text(a, .0, y + .06, label, 12)
            a.add_patch(Rectangle((x0, y), w, .12, facecolor=COL['white'], edgecolor=COL['ink' if binding else 'line'], lw=1.8 if binding else .9))
            for j in range(resident):
                a.add_patch(Rectangle((x0 + j * w * frac, y), w * frac, .12, facecolor=COL['blue' if j == 0 else 'green'], edgecolor=COL['line'], lw=.9))
                text(a, x0 + (j + .5) * w * frac, y + .06, str(j), 11, ha='center')
            text(a, x0, y - .065, ('限制驻留：' if binding else '') + note, 11)
        text(a, .0, .03, f'驻留 {resident} 块 = {resident * block["threads"] // 32} 个 warp，占用率 {resident * block["threads"] // 32}/64 = {resident * block["threads"] // 32 * 100 // 64}%', 12)
        out.save(f, 'figure-5-sm-residency')
        data['sm_residency'] = dict(kind='resource_residency', source='calculations/results/sm-occupancy-book-tile.json', limits=limits, block=block,
                                    blocks_by_limit=by, resident_blocks=resident, resident_warps=resident * block['threads'] // 32,
                                    occupancy=resident * block['threads'] / 32 / 64, binding_limits=[k for k, v in by.items() if v == resident],
                                    shared_left_bytes=limits['shared_bytes'] - resident * (block['shared_bytes'] + block['reserved_shared_bytes']))

        # 5.1.5 Producer warp copies, consumer warps compute; barriers hand over two slots (same timing as figure 5-16).
        # One K tile (A and W, 32 KiB; 2,097,152 FLOPs) on one H100 SM: 3.35 TB/s and 989.4 TFLOP/s split over 132 SMs.
        copy_us = 32768 * 132 / 3.35e12 * 1e6; compute_us = 2097152 * 132 / 989.4e12 * 1e6; tiles = 4
        copies, computes = [], []
        for t in range(tiles):
            earliest = copies[-1]['start'] + copy_us if copies else 0
            freed = computes[t - 2]['start'] + compute_us if t >= 2 else 0
            copies.append(dict(tile=t, slot=t % 2, start=max(earliest, freed), duration=copy_us))
            ready = copies[t]['start'] + copy_us
            previous = computes[-1]['start'] + compute_us if computes else 0
            computes.append(dict(tile=t, slot=t % 2, start=max(ready, previous), duration=compute_us))
        f, a = plot(3.4, left=.25, bottom=.19); f.subplots_adjust(top=.80)
        for e in copies + computes:
            y = 1 if e in copies else 0
            a.broken_barh([(e['start'], e['duration'])], (y - .19, .38), facecolors=COL['blue' if e['slot'] == 0 else 'green'], edgecolors=COL['line'], lw=.8)
            a.text(e['start'] + e['duration'] / 2, y, str(e['tile']), ha='center', va='center', fontsize=11)
        full = [(c['start'] + c['duration'], k['start']) for c, k in zip(copies, computes)]
        empty = [(computes[t - 2]['start'] + compute_us, copies[t]['start']) for t in range(2, tiles)]
        for done, begin in full:
            a.annotate('', xy=(begin, .19), xytext=(done, .81), arrowprops=dict(arrowstyle='-|>', lw=1, color=COL['line'], shrinkA=0, shrinkB=0))
        for done, begin in empty:
            a.annotate('', xy=(begin, .81), xytext=(done, .19), arrowprops=dict(arrowstyle='-|>', lw=1, color=COL['line'], linestyle='--', shrinkA=0, shrinkB=0))
        a.text(full[0][0] + .05, .5, '满', fontsize=11, va='center'); a.text(empty[0][1] + .05, .62, '空', fontsize=11, va='center')
        a.text((computes[0]['start'] + compute_us + computes[1]['start']) / 2, -.42, '等数据', fontsize=11, ha='center', va='center')
        a.set(yticks=[0, 1], yticklabels=['消费者 warp\n计算', '生产者 warp\n复制'], xlim=(0, 5.8), ylim=(-.6, 1.6),
              xticks=[0] + [round(c['start'] + copy_us, 2) for c in copies], xlabel='时间（μs）')
        a.set_xticklabels(['0'] + [f"{c['start'] + copy_us:.2f}" for c in copies])
        f.legend(handles=[Patch(fc=COL[c], ec=COL['line'], label=l) for c, l in [('blue', '槽 A：块 0、2'), ('green', '槽 B：块 1、3')]],
                 loc='upper center', ncol=2, frameon=False)
        out.save(f, 'figure-5-warp-pipeline')
        data['warp_pipeline'] = dict(kind='teaching_timeline', copy_us=copy_us, compute_us=compute_us, copies=copies, computes=computes,
                                     full_barrier_us=[d for d, _ in full], empty_barrier_us=[d for d, _ in empty],
                                     completion_us=computes[-1]['start'] + compute_us)

        # 5.5.2 FlashInfer plan/run cross-layer reuse vs. RL Sleep Mode virtual address reservation.
        f, a = canvas(7.4)
        a.add_patch(FancyBboxPatch((.02, .51), .96, .46, boxstyle='round,pad=0.003,rounding_size=0.012',
                                   facecolor='#FAFCFE', edgecolor='#CBD5E1', lw=1.0))
        text(a, .04, .94, 'A. FlashInfer plan/run 分离与跨层计划复用', 12, weight='bold')
        box(a, .04, .79, .43, .11, '逐层独立 plan（36 层）\n主机 36×20 = 720 μs\n144 次 H2D · 2.0~2.4 ms', 'orange', 11)
        box(a, .04, .66, .43, .11, '同构共享 1 次 plan\n主机 1×20 = 20 μs（省 700 μs）\n4 次 H2D · 0.72~0.75 ms', 'green', 11)
        box(a, .04, .53, .43, .11, '异构模型按族 plan\nMLA + SWA/KDA 分 2 族\n每步 2×20 = 40 μs', 'blue', 11)

        box(a, .51, .79, .20, .11, '1. CPU plan\n序列长+页表\n切分均衡任务', 'blue', 11)
        box(a, .75, .79, .21, .11, '2. 显存计划表\n单次 H2D 写入\n工作区常驻', 'purple', 11)
        arrow(a, (.71, .845), (.75, .845), 'control')
        box(a, .51, .66, .45, .10, '3. GPU run：第 0..35 层注意力\n复读同一计划表（开销降 ~3×）', 'green', 11)
        arrow(a, (.855, .79), (.855, .76), 'control')
        box(a, .51, .53, .45, .10, '失效边界：下一步长度+1 或批增减\n必须重跑 plan 刷新页表与切分', 'gray', 11)

        a.add_patch(FancyBboxPatch((.02, .02), .96, .46, boxstyle='round,pad=0.003,rounding_size=0.012',
                                   facecolor='#FAFCFE', edgecolor='#CBD5E1', lw=1.0))
        text(a, .04, .45, 'B. RL 训推切换：虚拟地址保留与两级休眠权衡', 12, weight='bold')
        box(a, .04, .30, .44, .11, 'cuMemAddressReserve 锁地址\nCUDA Graph 指针始终合法', 'blue', 11)
        box(a, .04, .16, .205, .11, '休眠 Unmap\n释放物理 HBM\n39.26 GiB', 'orange', 11)
        box(a, .275, .16, .205, .11, '唤醒 Map\n重挂物理 HBM\n原虚拟址恢复', 'green', 11)
        arrow(a, (.245, .215), (.275, .215), 'line')
        box(a, .04, .04, .44, .095, '收益：释放 39.26 GiB 供训练\n零重录 100% 保活 CUDA Graph', 'gray', 11)

        box(a, .52, .30, .44, .11, 'Level 1 休眠（卸载至 CPU 内存）\n15.26 GiB 迁出+迁回（25 GB/s）\n双向搬移耗时 ≈ 1.31 s', 'orange', 11)
        box(a, .52, .16, .44, .11, 'Level 2 休眠（直接丢弃权重/KV）\n省去 CPU 往返备份（0 s 搬移）\n唤醒后训练器原地覆写新权重', 'green', 11)
        box(a, .52, .04, .44, .095, '工程边界：Level 2 省 1.31 s/轮\n未训 MTP/LoRA 需单独保留', 'purple', 11)
        out.save(f, 'ch05-flashinfer-plan-run-and-sleep-vaddr')
        data['flashinfer_and_sleep'] = dict(
            kind='plan_reuse_and_sleep_vaddr',
            sources=['case-studies/cache-and-reconfiguration.md', 'case-studies/weight-handoff.md', 'experiments/ch05/05-08/flashinfer-plan/README.md'],
            qwen3_8b_layers=36, plan_us_per_call=20.0, per_layer_plan_us=720.0,
            shared_plan_us=20.0, saved_host_plan_us=700.0, hybrid_two_group_plan_us=40.0,
            weight_gib=15.2564, kv_gib=24.0, level1_roundtrip_s=1.31
        )

        # 5.3.3 & 5.6.3 Fig 5-45: FlashAttention-4 Blackwell B200 cycle ledger & TMEM ping-pong pipeline.
        f, a = canvas(7.4)
        a.add_patch(FancyBboxPatch((.02, .51), .96, .46, boxstyle='round,pad=0.003,rounding_size=0.012',
                                   facecolor='#FAFCFE', edgecolor='#CBD5E1', lw=1.0))
        text(a, .04, .94, 'A. Blackwell B200（128×128 块）周期账本与 SFU 瓶颈', 11.5, weight='bold')
        box(a, .04, .79, .44, .11, 'Tensor Core（tcgen05）\nQK^T 512 + PV 512 = 1,024c\n8,192 FLOP/cycle/SM', 'blue', 11)
        box(a, .04, .66, .44, .11, 'Shared Memory（SMEM）\nK/V 读取 128 B/c/SM\n单块稳态耗时 = 768 周期', 'gray', 11)
        box(a, .04, .53, .44, .11, 'SFU 指数（MUFU.EX2）\n朴素 16,384 次 = 1,024c\n与矩阵核 1:1 暴露空泡', 'orange', 11)

        box(a, .52, .79, .44, .11, 'Hopper H100（FA-3）\n矩阵 2,048c > SFU 1,024c\n天然掩盖 Softmax', 'gray', 11)
        box(a, .52, .66, .44, .11, 'FA-4 多项式模拟 exp2\nFMA 分担 75% 指数计算\nSFU 1,024c → 256c', 'green', 11)
        arrow(a, (.48, .585), (.52, .715), 'control')
        box(a, .52, .53, .44, .11, '条件重缩放（阈值 τ）\n跳过非必要输出重缩放\n达 71% 峰值（1.6 PF）', 'purple', 11)

        a.add_patch(FancyBboxPatch((.02, .02), .96, .46, boxstyle='round,pad=0.003,rounding_size=0.012',
                                   facecolor='#FAFCFE', edgecolor='#CBD5E1', lw=1.0))
        text(a, .04, .45, 'B. 256 KB TMEM 双缓冲与双 Warp-Group 乒乓流水', 11.5, weight='bold')
        box(a, .04, .31, .44, .10, 'TMEM 槽 0（128 KB）\n驻留块 j 分数 S_j 与 O_j', 'blue', 11)
        box(a, .52, .31, .44, .10, 'TMEM 槽 1（128 KB）\n驻留块 j+1 分数与累加器', 'purple', 11)
        box(a, .04, .17, .29, .11, '1. TMA+UMMA\nHBM→SMEM\n发射 mma', 'orange', 11)
        box(a, .355, .17, .29, .11, '2. WG-0\n模拟 exp2\n处理槽 0', 'green', 11)
        box(a, .67, .17, .29, .11, '3. WG-1\n乒乓槽 1\n与 PV 重叠', 'green', 11)
        arrow(a, (.33, .225), (.355, .225), 'control')
        arrow(a, (.645, .225), (.67, .225), 'control')
        box(a, .04, .04, .92, .095, '净收益：累加器驻留 TMEM 免除寄存器争用\n256 周期 SFU+FMA 隐藏于 1,024 周期矩阵乘内', 'gray', 11)
        out.save(f, 'ch05-flashattention4-blackwell-sfu-pipeline')

        # 5.3.3 & 5.5.2 Fig 5-46: DeepGEMM fine-grained FP8 JIT & GraCE compiler-assisted whole-graph capture.
        f, a = canvas(7.4)
        a.add_patch(FancyBboxPatch((.02, .51), .96, .46, boxstyle='round,pad=0.003,rounding_size=0.012',
                                   facecolor='#FAFCFE', edgecolor='#CBD5E1', lw=1.0))
        text(a, .04, .94, 'A. DeepGEMM 细粒度 FP8 两级累加与运行时 JIT', 11.5, weight='bold')
        box(a, .04, .78, .28, .12, '细粒度 FP8\n激活 1×128\n权重 128×128', 'blue', 11)
        box(a, .36, .78, .28, .12, '1. 矩阵核\nK_b=128 步\n免尾数截断', 'orange', 11)
        box(a, .68, .78, .28, .12, '2. CUDA 核\nFP32 累加\n每块恢复精度', 'green', 11)
        arrow(a, (.32, .84), (.36, .84))
        arrow(a, (.64, .84), (.68, .84))
        box(a, .04, .65, .44, .10, 'JIT 特化（~300 行模板）\n动态 M 编码为立即数常量', 'purple', 11)
        box(a, .52, .65, .44, .10, '尾块展开+TMA 多播择优\n省体积，提速 1.3x–2.7x', 'green', 11)
        box(a, .04, .53, .92, .09, '块级原语：ThunderKittens 2.0（16×16）与 TileLang\n将双缓冲与两级累加浓缩至数十行可移植代码', 'gray', 11)

        a.add_patch(FancyBboxPatch((.02, .02), .96, .46, boxstyle='round,pad=0.003,rounding_size=0.012',
                                   facecolor='#FAFCFE', edgecolor='#CBD5E1', lw=1.0))
        text(a, .04, .45, 'B. GraCE（ASPLOS \'26）跨分支整图捕获与权衡', 11.5, weight='bold')
        box(a, .04, .30, .44, .11, '传统分段图（Graph Break）\n动态分支回退主机 CPU\n暴露数十 μs 发射空泡', 'orange', 11)
        box(a, .52, .30, .44, .11, 'GraCE 编译器重写（谓词）\n分支转 GPU 条件节点\n跨分支拼接端到端整图', 'green', 11)
        arrow(a, (.48, .355), (.52, .355))
        box(a, .04, .16, .44, .11, '运行时净收益\n消除子图间主机发射空泡\n小批量与 Agent 尾延迟收敛', 'blue', 11)
        box(a, .52, .16, .44, .11, '显式工程代价（实测）\n图编译构建耗时 +66%\n静态并集使峰值显存 +20%', 'purple', 11)
        box(a, .04, .04, .92, .09, '准则：显存充裕、延迟敏感的解码开启整图合并\n显存受限或形状跳变的预填充用分段图+计划复用', 'gray', 11)
        out.save(f, 'ch05-deepgemm-fp8-jit-and-grace-graph')

        # 5.6.4 Fig 5-47: TileLang & ThunderKittens abstraction spectrum and TMEM double-buffered pipeline.
        f, a = canvas(7.4)
        a.add_patch(FancyBboxPatch((.02, .51), .96, .46, boxstyle='round,pad=0.003,rounding_size=0.012',
                                   facecolor='#FAFCFE', edgecolor='#CBD5E1', lw=1.0))
        text(a, .04, .94, 'A. 四代算子编程抽象的硬件控制粒度与开发效率定位', 11.5, weight='bold')
        box(a, .04, .78, .44, .12, 'CUDA C++ / CUTLASS 3.x\n~1,800 行 · 100% 硬件控制\n手写 mbarrier 与寄存器调度', 'gray', 11)
        box(a, .52, .78, .44, .12, 'Triton 3.x（块级隐式编译）\n~90 行 · B200 利用率 ~76%\n难控 Warp-Group 与 TMEM', 'orange', 11)
        box(a, .04, .63, .44, .12, 'ThunderKittens 2.0\n~220 行 C++ · 96% 硬件峰值\n16×16 瓦片 + NVLink 重叠', 'blue', 11)
        box(a, .52, .63, .44, .12, 'TileLang（三级显式流水）\n~80 行 Python · 98% 峰值\nKernel + pipelined_for + copy', 'green', 11)
        arrow(a, (.26, .78), (.26, .75), 'control')
        arrow(a, (.74, .78), (.74, .75), 'control')
        box(a, .04, .53, .92, .08, '演进主线：将 TMEM/寄存器布局与异步 DMA 显式暴露给高层原语', 'purple', 11)

        a.add_patch(FancyBboxPatch((.02, .02), .96, .46, boxstyle='round,pad=0.003,rounding_size=0.012',
                                   facecolor='#FAFCFE', edgecolor='#CBD5E1', lw=1.0))
        text(a, .04, .45, 'B. 算例 5-4 / 实验 5-17：Blackwell TMEM 双缓冲 TMA 流水对比', 11.5, weight='bold')
        box(a, .04, .30, .44, .11, '朴素单缓冲（累加器在寄存器）\nTMA 0.75μs + MMA 1.00μs 串行\n16 步 28.00 μs（利用率 57.1%）', 'orange', 11)
        box(a, .52, .30, .44, .11, 'TileLang 双缓冲 + TMEM\n预取 0.75μs + 16 步 16.00μs\n总计 16.75 μs（提速 1.67×）', 'green', 11)
        arrow(a, (.48, .355), (.52, .355))
        box(a, .04, .16, .44, .11, '单缓冲：128 regs/线程\n256 线程占 32,768 寄存器', 'gray', 11)
        box(a, .52, .16, .44, .11, 'TMEM 卸载：56 regs/线程\n省 56.2% 寄存器供预取', 'blue', 11)
        box(a, .04, .04, .92, .09, '结论：双缓冲流水 + 256 KB TMEM 驻留同时打破带宽空泡与寄存器墙', 'purple', 11)
        out.save(f, 'ch05-tilelang-thunderkittens-abstraction-spectrum')

        # 5.6.5 Fig 5-48: Mirage algebraic superoptimizer & Persistent Mega-Kernel (MPK) execution.
        f, a = canvas(7.4)
        a.add_patch(FancyBboxPatch((.02, .51), .96, .46, boxstyle='round,pad=0.003,rounding_size=0.012',
                                   facecolor='#FAFCFE', edgecolor='#CBD5E1', lw=1.0))
        text(a, .04, .94, 'A. Mirage 三级代数张量超优化与有限域 F_p 等价验证', 11.5, weight='bold')
        box(a, .04, .78, .28, .12, '1. 内核图 KGraph\n跨算子代数重写\n融合归约与投影', 'blue', 11)
        box(a, .36, .78, .28, .12, '2. 块图 TBGraph\n共享内存分块\n免全局显存落地', 'purple', 11)
        box(a, .68, .78, .28, .12, '3. 线程图 TGraph\n寄存器指令合成\nF_p 等价验证', 'green', 11)
        arrow(a, (.32, .84), (.36, .84))
        arrow(a, (.64, .84), (.68, .84))
        box(a, .04, .64, .92, .10, '代数融合：将 RMSNorm 归约折叠进前驱投影尾声，免去 HBM 往返', 'orange', 11)
        box(a, .04, .53, .92, .08, '验证完备性：有限域 F_p 随机指纹测试，毫秒级判定代数等价', 'gray', 11)

        a.add_patch(FancyBboxPatch((.02, .02), .96, .46, boxstyle='round,pad=0.003,rounding_size=0.012',
                                   facecolor='#FAFCFE', edgecolor='#CBD5E1', lw=1.0))
        text(a, .04, .45, 'B. 算例 5-5 / 实验 5-16：80 层模型（720 算子）单步解码耗时对比', 11.5, weight='bold')
        box(a, .04, .30, .28, .12, 'Eager 逐算子\n计算 4.80+启动 2.52\n总计 8.80 ms', 'orange', 11)
        box(a, .36, .30, .28, .12, 'CUDA Graph\n启动压至 0.08ms\n总计 6.30 ms', 'blue', 11)
        box(a, .68, .30, .28, .12, 'Mirage MPK\n计算 4.80+队列 0.18\n总计 4.98 ms', 'green', 11)
        arrow(a, (.32, .36), (.36, .36))
        arrow(a, (.64, .36), (.68, .36))
        box(a, .04, .16, .44, .11, 'SM 角色特化（132 SM）\n112 计算 SM + 20 通信 SM', 'purple', 11)
        box(a, .52, .16, .44, .11, '片上 L2 / SMEM 任务队列\n消除 720 道全局栅栏与尾波', 'green', 11)
        box(a, .04, .04, .92, .09, '净收益：MPK 较 Eager 快 1.77×（省 3.82ms），较 Graph 省 1.32ms', 'gray', 11)
        out.save(f, 'ch05-mirage-mpk-persistent-megakernel-execution')
    (here / 'structure-layout-validation.json').write_text(json.dumps(out.checks, ensure_ascii=False, indent=2) + '\n')
    return out.outputs


if __name__ == '__main__':
    _, family = configure_font()
    plt.rcParams.update({'font.family': [family, 'DejaVu Sans'], 'axes.unicode_minus': False, 'svg.hashsalt': 'ch05-structure'})
    draw(ROOT / 'manuscripts/ch05')
