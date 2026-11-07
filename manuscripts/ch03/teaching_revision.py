"""Work arrives, waits, and releases state: publication-grade 420pt diagrams for chapter 3."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from figure_style import COL, BORDER_COL, TEXT_COL, STYLE, canvas, plot, text, box, card, badge, arrow, Exporter


def draw(here, data):
    out = Exporter(here)

    def save(f, name):
        out.save(f, 'figure-3-' + name)

    def bars(name, labels, values, xlabel, height=3.6, colors=None):
        f, a = plot(height, left=0.31, bottom=0.18, right=0.95, top=0.92)
        if colors is None:
            colors = ['blue'] * len(values)
        for i, (v, c) in enumerate(zip(values, colors)):
            a.barh(i, v, color=COL[c], edgecolor=BORDER_COL[c], linewidth=1.2, height=0.52)
        a.set(yticks=range(len(values)), yticklabels=labels, xlim=(0, max(values) * 1.32), xlabel=xlabel)
        a.invert_yaxis()
        a.grid(axis='x', linestyle=':', alpha=0.45, color='#CBD5E1')
        for i, v in enumerate(values):
            a.text(v + max(values) * 0.025, i, f'{v:,.3f}', fontsize=11, va='center', color=COL['ink'], weight='bold')
        save(f, name)

    with plt.rc_context(STYLE):
        # 1. figure-3-1-stages
        f, a = canvas(4.6)
        text(a, 0.04, 0.94, '复用 6144 token，处理 2048 新 token', 13, weight='bold')
        for i, n in enumerate(data['3-1']['slots_after_calls']):
            y = 0.70 - i * 0.205
            c_left = 'orange' if i == 0 else 'blue'
            box(a, 0.04, y, 0.43, 0.13, 'prefill（首个输出 token）' if i == 0 else f'decode {i}', c_left, 11, weight='bold')
            box(a, 0.58, y, 0.38, 0.13, f'KV 覆盖 {n} token', 'green', 11, weight='bold')
            arrow(a, (0.475, y + 0.065), (0.575, y + 0.065), c_left)
            text(a, 0.05, y - 0.038, f'产生第 {i+1} 个输出 token', 11, color=COL['muted'])
        save(f, '1-stages')

        # 2. figure-3-request-clocks
        f, a = canvas(3.5)
        text(a, 0.04, 0.93, '同一条请求的三个观察时刻与计时区间', 13, weight='bold')
        arrow(a, (0.06, 0.62), (0.95, 0.62), 'ink')
        for x, label in [(0.12, '请求到达'), (0.43, '首个输出 token'), (0.85, '最后输出 token')]:
            a.plot([x, x], [0.58, 0.66], color=COL['ink'], lw=1.6)
            text(a, x, 0.74, label, 11, ha='center', weight='bold')
        box(a, 0.12, 0.36, 0.31, 0.13, '首响应 TTFT', 'blue', 11, weight='bold')
        box(a, 0.43, 0.36, 0.42, 0.13, '后续输出间隔 TPOT', 'green', 11, weight='bold')
        box(a, 0.12, 0.13, 0.73, 0.13, '完整请求端到端时间', 'orange', 11, weight='bold')
        save(f, 'request-clocks')

        # 3. figure-3-state-time-area
        f, a = plot(3.2, left=0.18, bottom=0.19)
        a.fill_between([0, 10], [1, 1], color=COL['blue'], edgecolor=BORDER_COL['blue'], linewidth=1.4, alpha=0.85)
        a.set(xlim=(0, 12), ylim=(0, 1.45), xlabel='工具等待时长（s）', ylabel='状态占用（GiB）', yticks=[0, 0.5, 1.0])
        a.grid(True, linestyle=':', alpha=0.45, color='#CBD5E1')
        a.text(5, 0.5, '1 GiB × 10 s = 10 GiB·s\n（等待期间状态时空面积）', ha='center', va='center', fontsize=12, weight='bold', color=TEXT_COL['blue'])
        save(f, 'state-time-area')

        # 4. figure-3-2-workload-budget
        d = data['3-2']
        f, a = plot(3.6, left=0.18, bottom=0.16)
        A = np.array(d['A_fraction'])
        a.bar(range(3), A, width=0.52, color=COL['blue'], edgecolor=BORDER_COL['blue'], linewidth=1.2, label='长输入类（8192/256）')
        a.bar(range(3), 1 - A, bottom=A, width=0.52, color=COL['orange'], edgecolor=BORDER_COL['orange'], linewidth=1.2, label='长输出类（1024/2048）')
        a.set(xticks=range(3), xticklabels=['均匀混合', '变化前窗', '变化后窗'], ylim=(0, 1.32), ylabel='请求比例')
        a.grid(axis='y', linestyle=':', alpha=0.45, color='#CBD5E1')
        a.legend(ncol=2, frameon=True, loc='upper center', fontsize=11)
        save(f, '2-workload-budget')

        # 5. figure-3-stage-demand
        f, a = plot(3.6, left=0.19, bottom=0.16)
        x = np.arange(3)
        for dx, key, label, col in [
            (-0.18, 'prefill_positions_per_second', '输入 token/s', 'blue'),
            (0.18, 'decode_positions_per_second', '后续 decode 步/s', 'orange'),
        ]:
            a.bar(x + dx, d[key], width=0.34, label=label, color=COL[col], edgecolor=BORDER_COL[col], linewidth=1.2)
        a.set(xticks=x, xticklabels=['均匀混合', '变化前窗', '变化后窗'], ylabel='阶段工作量（token/s）', ylim=(0, 36000))
        a.grid(axis='y', linestyle=':', alpha=0.45, color='#CBD5E1')
        a.legend(frameon=True, loc='upper right', fontsize=11)
        save(f, 'stage-demand')

        # 6. figure-3-queue-mechanism
        f, a = canvas(2.9)
        card(a, 0.03, 0.42, 0.25, 0.36, '新到工作', ['到达率 λ(t)'], 'orange', 11)
        card(a, 0.37, 0.42, 0.26, 0.36, '待处理队列', ['积压 Q(t)'], 'blue', 11)
        card(a, 0.72, 0.42, 0.25, 0.36, '完成工作', ['服务率 μ(t)'], 'green', 11)
        arrow(a, (0.285, 0.60), (0.365, 0.60), 'orange')
        arrow(a, (0.635, 0.60), (0.715, 0.60), 'blue')
        text(a, 0.50, 0.18, '到来速度超过处理速度时，差额在队列中累积', 11, ha='center', weight='bold', color=COL['muted'])
        save(f, 'queue-mechanism')

        # 7. figure-3-queue-backlog
        q = d['fluid_queue']
        f, a = plot(3.6, left=0.20, bottom=0.18)
        a.fill_between(q['seconds'], q['backlog_steps'], color=COL['orange'], alpha=0.55)
        a.plot(q['seconds'], q['backlog_steps'], 'o-', color=BORDER_COL['orange'], lw=2.2)
        a.axvline(120, ls='--', lw=1.2, color='#64748B')
        a.annotate(
            f"峰值积压 {q['backlog_steps'][2]:,.0f} 步",
            (120, q['backlog_steps'][2]),
            xytext=(34, 118000),
            arrowprops={'arrowstyle': '-|>', 'color': BORDER_COL['orange'], 'lw': 1.3},
            fontsize=11,
            weight='bold',
            color=TEXT_COL['orange'],
        )
        a.set(xlim=(0, 150), ylim=(0, 138000), xlabel='从开始到达计时（s）', ylabel='待处理 decode 步')
        a.grid(True, linestyle=':', alpha=0.45, color='#CBD5E1')
        save(f, 'queue-backlog')

        # 8. figure-3-arrival-measured
        f, a = plot(3.6, left=0.18, bottom=0.16)
        rows = d['arrival_reports']
        x = np.arange(2)
        for dx, key, label, col in [
            (-0.18, 'ttft_p95_s', '首响应 p95', 'blue'),
            (0.18, 'latency_p95_s', '完整请求 p95', 'green'),
        ]:
            vals = [r[key] for r in rows]
            a.bar(x + dx, vals, width=0.34, label=label, color=COL[col], edgecolor=BORDER_COL[col], linewidth=1.2)
            for xi, v in zip(x + dx, vals):
                a.text(xi, v + 8, f'{v:.1f} s', ha='center', va='bottom', fontsize=11, weight='bold', color=COL['ink'])
        a.set(xticks=x, xticklabels=['均匀混合', '时段变化'], ylabel='客户端时间（s）', ylim=(0, 455))
        a.grid(axis='y', linestyle=':', alpha=0.45, color='#CBD5E1')
        a.legend(frameon=True, loc='upper left', fontsize=11)
        save(f, 'arrival-measured')

        # 9. figure-3-agent-prefix-cache
        f, a = plot(4.2, left=0.34, bottom=0.17, right=0.96, top=0.80)
        y = np.arange(4)
        nocache = [2.875, 20.661, 23.815, 28.844]
        cached = [2.875, 0.752, 3.199, 5.149]
        labels = [
            '第 1 轮 (0 / 206)',
            '第 2 轮 (1,392 / 51)',
            '第 3 轮 (1,440 / 216)',
            '第 4 轮 (1,648 / 344)',
        ]
        a.barh(y - 0.17, nocache, height=0.32, color=COL['orange'], edgecolor=BORDER_COL['orange'], linewidth=1.2, label='无缓存重算完整输入')
        a.barh(y + 0.17, cached, height=0.32, color=COL['blue'], edgecolor=BORDER_COL['blue'], linewidth=1.2, label='命中前缀缓存仅算新增')
        for yi, v_no, v_ca in zip(y, nocache, cached):
            a.text(v_no + 0.6, yi - 0.17, f'{v_no:.3f}', va='center', fontsize=11, weight='bold', color=TEXT_COL['orange'])
            a.text(v_ca + 0.6, yi + 0.17, f'{v_ca:.3f}', va='center', fontsize=11, weight='bold', color=TEXT_COL['blue'])
        a.set(yticks=y, yticklabels=labels, xlim=(0, 36.5), xlabel='单轮 prefill 矩阵运算量（TFLOPs）')
        a.invert_yaxis()
        a.grid(axis='x', linestyle=':', alpha=0.45, color='#CBD5E1')
        a.legend(ncol=1, frameon=True, loc='upper center', bbox_to_anchor=(0.45, 1.26), fontsize=11)
        save(f, 'agent-prefix-cache')

        # 10. figure-3-success-cost
        f, a = canvas(3.6)
        text(a, 0.50, 0.93, '两种策略各尝试 100 次的单次成功任务成本对比', 13, ha='center', weight='bold')
        for y, title, cost, success, c in [(0.54, '原策略（50% 成功）', 100, 50, 'blue'), (0.14, '新策略（80% 成功）', 200, 80, 'orange')]:
            box(a, 0.03, y, 0.31, 0.22, title, c, 11, weight='bold')
            box(a, 0.42, y, 0.55, 0.22, f'{cost} 单位成本 ÷ {success} 次成功\n= {cost/success:g} 单位／成功任务', c, 11, weight='bold')
            arrow(a, (0.345, y + 0.11), (0.415, y + 0.11), c)
        save(f, 'success-cost')

        # 11. figure-3-3-agent
        rounds = data['3-3']['rounds']
        bars(
            '3-agent',
            ['第 1 轮：截断', '第 2 轮：写文件', '第 3 轮：测试', '第 4 轮：结束'],
            [r['measured_model_seconds'] for r in rounds],
            '每轮模型墙钟时间（s）',
            colors=['orange', 'blue', 'green', 'blue'],
        )

        # 12 & 13. figure-3-tool-dependency & figure-3-tool-parallel
        for parallel, name in [(False, 'tool-dependency'), (True, 'tool-parallel')]:
            f, a = plot(3.4, left=0.17, bottom=0.19)
            stages = [
                (0, 2, 0, '2 s', 'blue'),
                (2, 6, 1, '6 s', 'orange'),
                (2 if parallel else 8, 10, 2, '10 s', 'green'),
                (12 if parallel else 18, 3, 0, '3 s', 'blue'),
            ]
            for start, dur, y, label, col in stages:
                a.barh(y, dur, left=start, height=0.52, color=COL[col], edgecolor=BORDER_COL[col], linewidth=1.2)
                a.text(start + dur / 2, y, label, ha='center', va='center', fontsize=11, weight='bold', color=COL['ink'])
            a.set(yticks=[0, 1, 2], yticklabels=['模型', '工具 A', '工具 B'], xlim=(0, 22), ylim=(-0.6, 2.6), xlabel='任务时间（s）')
            a.grid(axis='x', linestyle=':', alpha=0.45, color='#CBD5E1')
            a.invert_yaxis()
            save(f, name)

        # 14. figure-3-branch-state
        f, a = canvas(3.4)
        text(a, 0.04, 0.93, '公共前缀在显存中只存一份，各分支独立追加尾部', 13, weight='bold')
        card(a, 0.04, 0.30, 0.40, 0.38, '共享公共前缀', [r'$H_0$ 个 token', '1 份物理副本'], 'blue', 11)
        for y, label in [(0.62, r'分支 A 私有尾部 ($h_A$)'), (0.14, r'分支 B 私有尾部 ($h_B$)')]:
            box(a, 0.56, y, 0.39, 0.18, label, 'orange', 11, weight='bold')
            arrow(a, (0.445, 0.49), (0.555, y + 0.09), 'blue')
        save(f, 'branch-state')

        # 16. figure-3-decision-request
        dr = json.loads((Path(here).parents[1] / 'calculations/results/decision-request-book.json').read_text())
        dd = dr['decision_path_dense']
        dl = dr['llm_path_dense'][0]
        q = dr['scenario']['questions']
        pre = dd['latency_seconds'] * 1000
        step = dl['step_seconds_batch_1'] * 1000
        steps = dl['decode_steps']
        total = dl['latency_seconds'] * 1000
        f, axs = plt.subplots(2, 1, figsize=(420 / 72, 4.3), sharex=True)
        f.subplots_adjust(left=0.20, right=0.96, bottom=0.14, top=0.91, hspace=0.55)
        for a in axs:
            a.spines[['top', 'right']].set_visible(False)
            a.grid(axis='x', linestyle=':', alpha=0.45, color='#CBD5E1')
        a = axs[0]
        a.set_title(f'决策路径：一次前向，{pre:.1f} ms 后同时得到 {q} 个答案', loc='left', fontsize=12, weight='bold')
        a.barh(0, pre, height=0.5, color=COL['blue'], edgecolor=BORDER_COL['blue'], linewidth=1.2)
        a.text(pre + 14, 0, f'prefill {dr["scenario"]["input_tokens"]} token（受算力限制）', va='center', fontsize=11, color=COL['ink'])
        a.scatter([pre + 4] * q, np.linspace(0.78, 1.22, q), s=18, color=COL['green'], edgecolor=BORDER_COL['green'], linewidth=0.9, zorder=3)
        a.text(pre + 14, 1, f'{q} 个位置的输出分布一次读出', va='center', fontsize=11, color=TEXT_COL['green'], weight='bold')
        a.set(yticks=[0, 1], yticklabels=['prefill', '读取分布'], ylim=(-0.6, 1.6))
        a.invert_yaxis()
        a = axs[1]
        a.set_title(f'LLM 路径：同样的 prefill 之后逐 token 生成 {dl["output_tokens"]} 个 token', loc='left', fontsize=12, weight='bold')
        a.barh(0, pre, height=0.5, color=COL['blue'], edgecolor=BORDER_COL['blue'], linewidth=1.2)
        a.barh([1] * steps, [step * 0.72] * steps, left=[pre + i * step for i in range(steps)], height=0.5, color=COL['orange'], edgecolor=BORDER_COL['orange'], linewidth=0.4)
        a.text(pre + steps * step / 2, 0.35, f'{steps} 步 decode，每步 {step:.1f} ms（受带宽限制）', ha='center', va='center', fontsize=11, color=TEXT_COL['orange'], weight='bold')
        a.set(yticks=[0, 1], yticklabels=['prefill', '逐 token 生成'], ylim=(-0.6, 1.6), xlim=(0, total * 1.05), xlabel='请求开始后的时间（ms）')
        a.invert_yaxis()
        a.set_xticks([0, round(pre), 200, 400, round(total)])
        save(f, 'decision-request')

        # 17. figure-3-vision-shapes
        f, a = canvas(4.0)
        text(a, 0.04, 0.94, '图像编码为视觉 token，再送入语言模型主干', 13, weight='bold')
        for y, label, c in [
            (0.68, '原始缩放图像：640 × 640 像素', 'gray'),
            (0.41, '16 × 16 像素切块 → 40 × 40 = 1600 个图像块', 'blue'),
            (0.14, '相邻 2 × 2 块空间合并 → 20 × 20 = 400 个视觉 token', 'green'),
        ]:
            box(a, 0.06, y, 0.88, 0.16, label, c, 11, weight='bold')
            if y > 0.2:
                arrow(a, (0.50, y), (0.50, y - 0.105), 'ink')
        save(f, 'vision-shapes')

        # 18. figure-3-vision-state
        f, a = canvas(4.0)
        text(a, 0.04, 0.94, '400 个视觉 token 的特征宽度与跨阶段状态容量', 13, weight='bold')
        box(a, 0.05, 0.66, 0.90, 0.17, 'DeepStack 四组视觉特征：每组宽 2560（合计 10240 维）', 'blue', 11, weight='bold')
        box(a, 0.05, 0.38, 0.90, 0.17, '编码缓存 EC：400 × 10240 × 2 B = 7.8125 MiB', 'green', 11, weight='bold')
        arrow(a, (0.50, 0.66), (0.50, 0.555), 'blue')
        box(a, 0.05, 0.10, 0.90, 0.17, '进入语言主干后：400 × 144 KiB = 56.25 MiB 语言 KV', 'orange', 11, weight='bold')
        arrow(a, (0.50, 0.38), (0.50, 0.275), 'green')
        save(f, 'vision-state')

        # 19. figure-3-4-realtime
        f, a = canvas(4.4)
        text(a, 0.04, 0.94, '从多模态输入表征到终端可听见声音的四阶段流水线', 13, weight='bold')
        for i, (label, c) in enumerate([
            ('视觉或音频编码：提取特征形成模型输入', 'blue'),
            ('语言模型（Thinker）：跨模态理解并生成回复', 'green'),
            ('声学模型与编解码器（Talker / Codec）：合成音频采样', 'orange'),
            ('客户端抖动缓冲与播放设备：按声卡时钟输出声音', 'purple'),
        ]):
            y = 0.72 - i * 0.205
            box(a, 0.05, y, 0.90, 0.135, label, c, 11, weight='bold')
            if i < 3:
                arrow(a, (0.50, y), (0.50, y - 0.068), c)
        save(f, '4-realtime')

        # 20. figure-3-audio-timing
        d = data['3-4']
        f, a = plot(4.8, left=0.17, bottom=0.15)
        for z in d['audio_chunks']:
            i = z['chunk']
            a.barh(i, 20, left=z['playback_start_ns'] / 1e6, height=0.48, color=COL['green'], edgecolor=BORDER_COL['green'], linewidth=1.1)
            a.plot(z['arrival_ns'] / 1e6, i, 'o', color=BORDER_COL['blue'], markersize=6)
            a.plot(z['deadline_ns'] / 1e6, i, '|', markersize=14, color=BORDER_COL['orange'], markeredgewidth=2.0)
        a.plot([], [], 'o', color=BORDER_COL['blue'], label='网络到达')
        a.plot([], [], '|', color=BORDER_COL['orange'], markersize=12, markeredgewidth=2.0, label='原定播放时刻')
        a.barh([], [], color=COL['green'], edgecolor=BORDER_COL['green'], label='实际播放区间')
        a.set(yticks=range(8), yticklabels=[f'块 {i+1}' for i in range(8)], xlim=(0, 260), ylim=(-1.5, 7.6), xlabel='从采集起点计时（ms）')
        a.grid(axis='x', linestyle=':', alpha=0.45, color='#CBD5E1')
        a.invert_yaxis()
        a.legend(ncol=3, fontsize=11, frameon=True, loc='upper left')
        save(f, 'audio-timing')

        # 21. figure-3-audio-interrupt
        f, a = canvas(2.9)
        text(a, 0.04, 0.92, '发出打断指令与本地设备物理静音是两个不同时刻', 13, weight='bold')
        card(a, 0.04, 0.34, 0.38, 0.42, '123 ms：触发打断', ['用户开口说话', '发出取消指令'], 'orange', 11)
        card(a, 0.58, 0.34, 0.38, 0.42, '130 ms：设备静音', ['声卡轮询生效', '清空播放缓冲'], 'green', 11)
        arrow(a, (0.425, 0.55), (0.575, 0.55), 'orange')
        text(a, 0.50, 0.14, '本地静音延迟 7 ms；远端推理停止与 KV 释放走独立网络控制路径', 11, ha='center', color=COL['muted'], weight='bold')
        save(f, 'audio-interrupt')

        # NEW 22. ch03-rag-prefill-tradeoff-and-vla-loop (420pt, 2-panel vertical stack)
        f = plt.figure(figsize=(420 / 72, 5.5))
        ax1 = f.add_axes([0.22, 0.58, 0.73, 0.32])
        ax1.spines[['top', 'right']].set_visible(False)
        ax1.set_title('A  RAG 精检索与粗检索耗时反转（R* = 268.0 TFLOP/s）', loc='left', fontsize=11.5, weight='bold', pad=8)
        y_pos = [0, 1]
        ret_ms = [200.0, 20.0]
        pre_ms = [110.168, 351.378]
        ax1.barh(y_pos, ret_ms, height=0.46, color=COL['blue'], edgecolor=BORDER_COL['blue'], linewidth=1.2, label='检索耗时')
        ax1.barh(y_pos, pre_ms, left=ret_ms, height=0.46, color=COL['orange'], edgecolor=BORDER_COL['orange'], linewidth=1.2, label='GPU Prefill 耗时')
        ax1.text(100, 0, '200.0 ms', ha='center', va='center', fontsize=11, weight='bold', color=COL['ink'])
        ax1.text(200 + 55, 0, '110.2 ms', ha='center', va='center', fontsize=11, weight='bold', color=COL['ink'])
        ax1.text(316, 0, '310.2 ms\n(KV 216 MiB)', va='center', fontsize=11, weight='bold', color=TEXT_COL['green'])
        ax1.text(20 + 175, 1, 'Prefill 351.4 ms', ha='center', va='center', fontsize=11, weight='bold', color=COL['ink'])
        ax1.text(376, 1, '371.4 ms\n(净慢 61.2 ms)', va='center', fontsize=11, weight='bold', color=TEXT_COL['orange'])
        ax1.set(yticks=y_pos, yticklabels=['候选 A\n(精检索 8 篇)', '候选 B\n(粗检索 32 篇)'], xlim=(0, 540), ylim=(-0.55, 1.65), xlabel='首 token 前累计耗时（ms，200 TFLOP/s 吞吐）')
        ax1.invert_yaxis()
        ax1.grid(axis='x', linestyle=':', alpha=0.45, color='#CBD5E1')
        ax1.legend(ncol=2, loc='lower right', frameon=True, fontsize=11)

        ax2 = f.add_axes([0.22, 0.11, 0.73, 0.33])
        ax2.spines[['top', 'right']].set_visible(False)
        ax2.set_title('B  具身智能 VLA 快慢双环路与动作块缓冲时序', loc='left', fontsize=11.5, weight='bold', pad=8)
        # Row 0: Slow VLM Planner (10 Hz, refresh at 0 and 100 ms)
        ax2.barh(0, 45, left=0, height=0.42, color=COL['blue'], edgecolor=BORDER_COL['blue'], linewidth=1.2)
        ax2.text(22.5, 0, '45 ms', ha='center', va='center', fontsize=11, weight='bold', color=COL['ink'])
        ax2.barh(0, 140, left=100, height=0.42, color=COL['orange'], edgecolor=BORDER_COL['orange'], linewidth=1.2)
        ax2.text(170, 0, '突发抖动推理 140 ms', ha='center', va='center', fontsize=11, weight='bold', color=COL['ink'])
        # Row 1: Action Chunk Buffer (K=20 steps = 200 ms)
        ax2.barh(1, 100, left=0, height=0.42, color=COL['green'], edgecolor=BORDER_COL['green'], linewidth=1.2)
        ax2.text(50, 1, '步 1–10', ha='center', va='center', fontsize=11, weight='bold', color=COL['ink'])
        ax2.barh(1, 40, left=100, height=0.42, color=COL['purple'], edgecolor=BORDER_COL['purple'], linewidth=1.2)
        ax2.text(120, 1, '11-14', ha='center', va='center', fontsize=11, weight='bold', color=COL['ink'])
        ax2.barh(1, 120, left=140, height=0.42, color=COL['green'], edgecolor=BORDER_COL['green'], linewidth=1.2)
        ax2.text(200, 1, '新动作块接管', ha='center', va='center', fontsize=11, weight='bold', color=COL['ink'])
        # Row 2: Fast 1000 Hz Servo Loop
        for t_tick in range(0, 260, 10):
            ax2.barh(2, 8, left=t_tick, height=0.36, color=COL['teal'], edgecolor=BORDER_COL['teal'], linewidth=0.5)
        ax2.text(135, 2.42, '1000 Hz 力矩伺服连续执行（余量 60 ms，零断供）', ha='center', va='center', fontsize=11, weight='bold', color=TEXT_COL['teal'])
        ax2.axvline(100, ls='--', lw=1.1, color='#64748B')
        ax2.axvline(140, ls=':', lw=1.2, color=BORDER_COL['purple'])
        ax2.set(yticks=[0, 1, 2], yticklabels=['慢环规划\n(10 Hz)', '动作块缓冲\n(K=20)', '快环伺服\n(1000 Hz)'], xlim=(0, 275), ylim=(-0.55, 2.85), xlabel='物理控制时间轴（ms）')
        ax2.invert_yaxis()
        ax2.grid(axis='x', linestyle=':', alpha=0.45, color='#CBD5E1')
        out.save(f, 'ch03-rag-prefill-tradeoff-and-vla-loop')

        # 23. figure-3-activation-lifetime
        f, a = plot(3.2, left=0.20, bottom=0.19)
        for y, start, duration, label, c in [
            (0, 0, 1, '前向', 'blue'),
            (0, 3, 1, '反向', 'orange'),
            (1, 1, 3, '激活张量的驻留生命周期', 'green'),
        ]:
            a.barh(y, duration, left=start, height=0.52, color=COL[c], edgecolor=BORDER_COL[c], linewidth=1.2)
            a.text(start + duration / 2, y, label, fontsize=11, ha='center', va='center', weight='bold', color=COL['ink'])
        a.set(
            yticks=[0, 1],
            yticklabels=['该层计算', '该层激活'],
            xlim=(-0.1, 4.3),
            ylim=(-0.7, 1.7),
            xticks=[0, 1, 3, 4],
            xticklabels=['开始', '前向完成', '反向开始', '反向完成'],
            xlabel='事件次序（间距仅作示意）',
        )
        a.grid(axis='x', linestyle=':', alpha=0.45, color='#CBD5E1')
        a.invert_yaxis()
        save(f, 'activation-lifetime')

        # 24. figure-3-5-training
        f, a = canvas(4.1)
        text(a, 0.04, 0.94, '前向计算输出并留存激活，反向沿依赖链传递梯度', 13, weight='bold')
        for x, label, c in [(0.04, '前层', 'blue'), (0.38, '当前层', 'orange'), (0.72, '后层', 'green')]:
            box(a, x, 0.58, 0.24, 0.18, label, c, 12, weight='bold')
        arrow(a, (0.285, 0.71), (0.375, 0.71), 'blue')
        arrow(a, (0.625, 0.71), (0.715, 0.71), 'orange')
        text(a, 0.50, 0.83, '前向传播：输入激活 → 输出激活', 11, ha='center', weight='bold', color=TEXT_COL['blue'])
        arrow(a, (0.715, 0.62), (0.625, 0.62), 'green')
        arrow(a, (0.375, 0.62), (0.285, 0.62), 'orange')
        text(a, 0.50, 0.45, '反向传播：后一层梯度 → 前一层输入梯度 dX', 11, ha='center', weight='bold', color=TEXT_COL['orange'])
        box(a, 0.06, 0.10, 0.88, 0.16, '当前层结合前向激活算权重梯度 dW，交优化器更新', 'purple', 11, weight='bold')
        arrow(a, (0.50, 0.41), (0.50, 0.265), 'purple')
        save(f, '5-training')

        # 25 & 26. figure-3-training-states & figure-3-training-flops
        t = data['3-5']
        bars(
            'training-flops',
            ['前向矩阵', '反向矩阵', '前向加反向', '按总参数估算：6ND'],
            [t['summary'][k] / 1e12 for k in ['forward_matrix_flops', 'backward_matrix_flops', 'training_matrix_flops', 'six_nd_flops']],
            '矩阵运算量（TFLOPs）',
            colors=['blue', 'teal', 'green', 'orange'],
        )
        bars(
            'training-states',
            ['BF16 权重', 'FP32 梯度', 'FP32 主权重', 'Adam 一阶矩', 'Adam 二阶矩'],
            [v / 1e9 for v in t['parameter_state_bytes'].values()],
            '参数相关常驻状态（十进制 GB）',
            4.0,
            colors=['blue', 'orange', 'green', 'purple', 'purple'],
        )

        # 27. figure-3-training-breakdown
        f, a = plot(4.4, left=0.33, bottom=0.17, right=0.96, top=0.79)
        y = np.arange(4)
        row_labels = [
            '8192 全量词表头',
            '4096 有效词表头',
            '打包 7168 + 1024',
            '打包 4096 + 4096',
        ]
        trunk = np.array([341.398, 341.398, 341.398, 341.398])
        attn = np.array([59.381, 59.381, 46.393, 29.694])
        head = np.array([30.589, 15.295, 30.589, 30.589])
        a.barh(y, trunk, height=0.52, color=COL['blue'], edgecolor=BORDER_COL['blue'], linewidth=1.2, label='主干线性与 FFN')
        a.barh(y, attn, left=trunk, height=0.52, color=COL['green'], edgecolor=BORDER_COL['green'], linewidth=1.2, label='因果注意力 QK^T 与 AV')
        a.barh(y, head, left=trunk + attn, height=0.52, color=COL['orange'], edgecolor=BORDER_COL['orange'], linewidth=1.2, label='词表头 LM head')
        totals = [431.368, 416.074, 418.380, 401.681]
        for yi, tot in zip(y, totals):
            a.text(tot + 6, yi, f'{tot:.3f}', va='center', fontsize=11, weight='bold', color=COL['ink'])
        a.axvline(402.591, color='#64748B', ls='--', lw=1.3)
        a.text(394, 3.62, '6ND = 402.6', ha='right', va='bottom', fontsize=11, weight='bold', color=COL['muted'])
        a.set(yticks=y, yticklabels=row_labels, xlim=(0, 515), ylim=(-0.55, 3.95), xlabel='单步前反向矩阵总运算量（TFLOPs）')
        a.invert_yaxis()
        a.grid(axis='x', linestyle=':', alpha=0.45, color='#CBD5E1')
        a.legend(ncol=2, frameon=True, loc='upper center', bbox_to_anchor=(0.36, 1.27), fontsize=11)
        save(f, 'training-breakdown')

        # 28. figure-3-6-rl
        f, a = canvas(4.6)
        text(a, 0.04, 0.94, '强化学习（RL）中生成、环境反馈与策略更新的数据流回路', 13, weight='bold')
        for i, (label, c) in enumerate([
            ('策略模型（Actor）采样生成候选回答', 'blue'),
            ('规则、奖励模型或沙箱环境给出反馈评分', 'orange'),
            ('过滤筛选合格样本，组织训练微批次', 'green'),
            ('学习器（Learner）计算梯度并更新策略权重', 'purple'),
        ]):
            y = 0.73 - i * 0.20
            box(a, 0.16, y, 0.79, 0.135, label, c, 11, weight='bold')
            if i < 3:
                arrow(a, (0.55, y), (0.55, y - 0.062), c)
        a.plot([0.16, 0.05, 0.05, 0.155], [0.195, 0.195, 0.795, 0.795], color=BORDER_COL['orange'], lw=1.6)
        arrow(a, (0.05, 0.795), (0.155, 0.795), 'orange')
        text(a, 0.05, 0.05, '左侧权重同步回路：更新后的新权重分发给下一轮 rollout 生成端', 11, weight='bold', color=TEXT_COL['orange'])
        save(f, '6-rl')

        # 29. figure-3-rl-stage-work
        d = data['3-6']
        f, a = plot(3.8, left=0.20, bottom=0.16)
        names = ['rollout_prefill', 'rollout_decode', 'reference_scoring', 'policy_update']
        labs = ['生成输入', '后续生成', '参考评分', '策略更新']
        x = np.arange(4)
        for dx, key, col, label in [
            (-0.18, 'base_stages', 'blue', '生成 32 条，保留 16 条'),
            (0.18, 'low_acceptance_stages', 'orange', '生成 64 条，保留 16 条'),
        ]:
            vals = [next(z['matrix_flops'] / 1e12 for z in d[key] if z['name'] == name) for name in names]
            a.bar(x + dx, vals, width=0.34, color=COL[col], edgecolor=BORDER_COL[col], linewidth=1.2, label=label)
        a.set(xticks=x, xticklabels=labs, ylabel='矩阵运算量（TFLOPs）', ylim=(0, 1650))
        a.grid(axis='y', linestyle=':', alpha=0.45, color='#CBD5E1')
        a.legend(frameon=True, loc='upper left', fontsize=11)
        save(f, 'rl-stage-work')

        # 30. figure-3-7-scaling
        d = data['3-7']
        f, a = plot(3.8, left=0.18, bottom=0.18)
        a.plot([2, 7.6], [2, 7.6], color='#64748B', lw=1.3, ls='--')
        for split, col, marker, lab in [('fit', 'blue', 'o', '拟合集（6 点）'), ('holdout', 'orange', '^', '留出检验集（2 点）')]:
            rows = [z for z in d['records'] if z['split'] == split]
            law = d['law']
            pred = [law['E'] + law['A'] * (z['N'] / law['N0']) ** (-law['alpha']) + law['B'] * (z['D'] / law['D0']) ** (-law['beta']) for z in rows]
            a.scatter([z['loss'] for z in rows], pred, color=BORDER_COL[col], facecolor=COL[col], marker=marker, s=58, lw=1.4, label=lab, zorder=4)
        a.set(xlim=(2, 7.6), ylim=(2, 7.6), xlabel='实测验证损失（nats/token）', ylabel='幂律预测损失（nats/token）')
        a.grid(True, linestyle=':', alpha=0.45, color='#CBD5E1')
        a.legend(frameon=True, loc='upper left', fontsize=11)
        save(f, '7-scaling')

        # 31. figure-3-scaling-residual
        f, a = plot(3.2, left=0.19, bottom=0.19)
        res = d['residuals']
        for i, z in enumerate(res):
            c = 'blue' if z['split'] == 'fit' else 'orange'
            a.bar(i, z['residual'], color=COL[c], edgecolor=BORDER_COL[c], linewidth=1.2, width=0.56)
        a.axhline(0, lw=1.0, color='#64748B')
        a.set(
            xticks=range(8),
            xticklabels=['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'H1', 'H2'],
            xlabel='F1–F6：拟合点；H1–H2：留出检验点（RMSE = 0.0193）',
            ylabel='预测 − 观测（nats/token）',
        )
        a.grid(axis='y', linestyle=':', alpha=0.45, color='#CBD5E1')
        save(f, 'scaling-residual')

        # 32. figure-3-lifecycle-cost
        f, a = plot(3.8, left=0.19, bottom=0.18)
        calls = np.linspace(0, 4e8, 200)
        for r, col in zip(d['lifecycle']['rows'], ['orange', 'green', 'blue', 'gray']):
            a.plot(
                calls / 1e8,
                (r['upfront_cost'] + calls * r['cost_per_call']) / 3600,
                color=BORDER_COL[col],
                ls='--' if r['outside_fit_box'] else '-',
                lw=2.1,
                label=f"{r['N']/1e9:g}B" + ('（外推）' if r['outside_fit_box'] else ''),
            )
        a.axvline(d['crossing_calls'] / 1e8, color='#64748B', lw=1.2, ls=':')
        a.text(d['crossing_calls'] / 1e8 + 0.08, 960, '0.1B 与 0.5B 交点\n≈ 2.048 亿次调用', fontsize=11, weight='bold', color=COL['ink'])
        a.set(xlim=(0, 4), ylim=(0, 1200), xlabel='累计服务调用量（亿次）', ylabel='累计成本（H100 SXM GPU 小时）')
        a.grid(True, linestyle=':', alpha=0.45, color='#CBD5E1')
        a.legend(frameon=True, loc='upper left', fontsize=11)
        save(f, 'lifecycle-cost')

        # 33. figure-3-8-history
        d = data['3-8']
        bars(
            '8-history',
            ['Llama 1 6.7B', 'Llama 2 7B', 'Llama 3.1 8B', 'Qwen2.5 7B', 'Qwen3 8B'],
            d['ratios'],
            '报告训练 token 数／参数量（D/N）',
            4.0,
            colors=['blue', 'blue', 'blue', 'green', 'green'],
        )

        # 34. figure-3-9-gpu-hours
        hist = {r['input']['id']: r for r in data['3-9']['history_rows']}
        hopper = 989.4 / 312
        rows = [
            ('llama1-7b', 'Llama 1 6.7B（A100）', 1, 'blue'),
            ('llama1-65b', 'Llama 1 65B（A100）', 1, 'blue'),
            ('llama2-7b', 'Llama 2 7B（A100）', 1, 'blue'),
            ('llama2-70b', 'Llama 2 70B（A100）', 1, 'blue'),
            ('llama31-8b', 'Llama 3.1 8B（H100）', hopper, 'green'),
            ('llama31-70b', 'Llama 3.1 70B（H100）', hopper, 'green'),
            ('llama31-405b', 'Llama 3.1 405B（H100）', hopper, 'green'),
            ('deepseek-v3-pretraining', 'DeepSeek-V3 预训练（H800）', hopper, 'orange'),
        ]
        vals = [hist[k]['input']['gpu_hours'] * ratio / 1e6 for k, _, ratio, _ in rows]
        f, a = plot(5.0, left=0.40, bottom=0.22, right=0.95, top=0.93)
        for i, (v, (_, _, _, c)) in enumerate(zip(vals, rows)):
            a.barh(i, v, color=COL[c], edgecolor=BORDER_COL[c], linewidth=1.2, height=0.55)
        a.set(
            yticks=range(len(vals)),
            yticklabels=[label for _, label, _, _ in rows],
            xscale='log',
            xlim=(0.05, 360),
            xticks=[0.1, 1, 10, 100],
            xlabel='A100 80GB 等效训练用量\n（百万 GPU 小时，对数尺度）',
        )
        a.xaxis.set_minor_formatter(plt.NullFormatter())
        a.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
        a.invert_yaxis()
        a.set_ylim(10.8, -0.6)
        a.grid(axis='x', linestyle=':', alpha=0.45, color='#CBD5E1')
        for i, v in enumerate(vals):
            a.text(v * 1.12, i, f'{v:,.2f}', fontsize=11, va='center', color=COL['ink'], weight='bold')
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COL[c], edgecolor=BORDER_COL[c], linewidth=1.2) for c in ['blue', 'green', 'orange']]
        a.legend(handles, ['原设备 A100（实测）', '原设备 H100（×3.17）', '原设备 H800（×3.17）'], frameon=True, loc='lower right', fontsize=11)
        save(f, '9-gpu-hours')

        # 35. ch03-reasoning-cost-reversal-and-breakeven (Path B Fig 3-35)
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(420 / 72, 6.2))
        f.subplots_adjust(left=0.16, right=0.95, bottom=0.11, top=0.92, hspace=0.48)
        for ax in (ax1, ax2):
            ax.spines[['top', 'right']].set_visible(False)

        gamma = np.linspace(1, 24, 200)
        rel_cost = gamma * (0.10 / 1.60)
        ax1.plot(gamma, rel_cost, color=BORDER_COL['blue'], lw=2.2, label='单价 ×0.1, 成功率 50%→80%')
        ax1.axhline(1.0, ls='--', lw=1.3, color='#64748B')
        ax1.axvline(16.0, ls=':', lw=1.3, color=BORDER_COL['purple'])
        ax1.scatter([16.0, 20.0], [1.0, 1.25], s=55, color=[BORDER_COL['green'], BORDER_COL['orange']], facecolor=[COL['green'], COL['orange']], linewidth=1.5, zorder=5)
        ax1.text(16.4, 0.32, '临界膨胀上限 γ = 16×', ha='left', va='bottom', fontsize=11, weight='bold', color=TEXT_COL['purple'])
        ax1.text(20.0, 1.06, '20× → 1.25×\n(+25% 反升)', ha='center', va='top', fontsize=11, weight='bold', color=TEXT_COL['orange'])
        ax1.set(xlim=(0, 25), ylim=(0, 1.68), xlabel='单任务 Token 膨胀倍数 γ', ylabel='相对成功任务成本')
        ax1.set_title('A  长思考 Token 膨胀导致的单任务成本反转 (TC-05)', loc='left', fontsize=12, weight='bold', pad=8)
        ax1.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1')
        ax1.legend(frameon=True, loc='upper left', fontsize=11)

        q_e8 = np.linspace(0, 8.0, 200)
        saving_musd = 0.20 * q_e8
        ax2.plot(q_e8, saving_musd * 100, color=BORDER_COL['green'], lw=2.2, label='累计推理节省 (0.20 美元/百次)')
        ax2.axhline(100.0, color=BORDER_COL['orange'], lw=2.0, ls='--', label='超训额外投入 (100 万美元)')
        ax2.axvline(5.0, color='#64748B', lw=1.3, ls=':')
        ax2.scatter([5.0], [100.0], s=60, color=BORDER_COL['blue'], facecolor=COL['blue'], linewidth=1.6, zorder=5)
        ax2.text(5.15, 52, '盈亏平衡点 Q* = 5 亿次', ha='left', va='bottom', fontsize=11, weight='bold', color=COL['ink'])
        ax2.set(xlim=(0, 8.0), ylim=(0, 175), xlabel='累计成功任务量 Q（亿次）', ylabel='成本或节省（万美元）')
        ax2.set_title('B  超训投入与推理节省的盈亏平衡线 (TC-06)', loc='left', fontsize=12, weight='bold', pad=8)
        ax2.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1')
        ax2.legend(frameon=True, loc='upper left', fontsize=11)
        out.save(f, 'ch03-reasoning-cost-reversal-and-breakeven')

        # 36. ch03-servegen-bursty-and-bimodal-workload (Path D Fig 3-36)
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(420 / 72, 6.4))
        f.subplots_adjust(left=0.16, right=0.95, bottom=0.10, top=0.92, hspace=0.48)
        for ax in (ax1, ax2):
            ax.spines[['top', 'right']].set_visible(False)

        rho = np.linspace(0.10, 0.88, 160)
        lq_poisson = (rho ** 2 / (1.0 - rho)) * ((1.0 + 1.0) / 2.0)
        lq_servegen = (rho ** 2 / (1.0 - rho)) * ((4.0 + 1.0) / 2.0)
        ax1.plot(rho * 100, lq_poisson, color=BORDER_COL['blue'], lw=2.2, ls='--', label='泊松到达 (CV_a² = 1.0, NAIVE 预测 12 实例)')
        ax1.plot(rho * 100, lq_servegen, color=BORDER_COL['orange'], lw=2.4, label='ServeGen 突发到达 (CV_a² = 4.0, 实测 24 实例)')
        ax1.scatter([80.0, 80.0], [3.20, 8.00], s=55, color=[BORDER_COL['blue'], BORDER_COL['orange']], facecolor=[COL['blue'], COL['orange']], linewidth=1.5, zorder=5)
        ax1.annotate('ρ = 80%: 队列 3.2 → 8.0\n(2.5× 队列膨胀)', xy=(80.0, 8.0), xytext=(38.0, 8.5),
                     arrowprops=dict(arrowstyle='->', color=BORDER_COL['orange'], lw=1.2),
                     fontsize=11, weight='bold', color=TEXT_COL['orange'])
        ax1.set(xlim=(10, 90), ylim=(0, 22.0), xlabel='系统平均利用率 ρ（%）', ylabel='平均排队请求数 L_q')
        ax1.set_title('A  ServeGen 重尾突发 (CV_a²=4.0) 引发的 2.5× 队列膨胀', loc='left', fontsize=12, weight='bold', pad=8)
        ax1.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1')
        ax1.legend(frameon=True, loc='upper left', fontsize=11)

        out_bins = ['短答模式\n(128–512)', '常规生成\n(512–2K)', '深度推理模式\n(4K–16K)', '极端长尾\n(>16K)']
        share_pct = [42.0, 23.0, 28.0, 7.0]
        kv_weight = [8.0, 14.0, 52.0, 26.0]
        x_b = np.arange(len(out_bins))
        w_b = 0.34
        ax2.bar(x_b - w_b / 2, share_pct, width=w_b, color=COL['blue'], edgecolor=BORDER_COL['blue'], linewidth=1.2, label='请求数量占比 (%)')
        ax2.bar(x_b + w_b / 2, kv_weight, width=w_b, color=COL['purple'], edgecolor=BORDER_COL['purple'], linewidth=1.2, label='TokenScale 词元/KV 负载占比 (%)')
        for xi, s_v, k_v in zip(x_b, share_pct, kv_weight):
            ax2.text(xi - w_b / 2, s_v + 1.5, f'{s_v:.0f}%', ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL['blue'])
            ax2.text(xi + w_b / 2, k_v + 1.5, f'{k_v:.0f}%', ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL['purple'])
        ax2.set_xticks(x_b, out_bins)
        ax2.set_xlim(-0.6, 3.6)
        ax2.set_ylim(0, 82)
        ax2.set_ylabel('占总流量比例（%）', fontsize=11)
        ax2.set_title('B  双峰重尾输出分布与 TokenScale 滑窗词元速度权重对比', loc='left', fontsize=12, weight='bold', pad=8)
        ax2.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E1')
        ax2.legend(frameon=True, loc='upper left', fontsize=11)
        out.save(f, 'ch03-servegen-bursty-and-bimodal-workload')

        # 37. ch03-osworld-computer-use-agent-timeline (Path D Fig 3-37)
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(420 / 72, 6.4))
        f.subplots_adjust(left=0.22, right=0.95, bottom=0.10, top=0.92, hspace=0.48)
        for ax in (ax1, ax2):
            ax.spines[['top', 'right']].set_visible(False)

        modes = ['朴素 Agent\n(18 步冗余)', '动作晶化+滑窗\n(10 步紧凑)']
        t_env = np.array([10.1, 4.2])
        t_vis = np.array([5.6, 2.4])
        t_llm = np.array([12.2, 4.8])
        t_act = np.array([8.1, 3.8])
        y_m = np.arange(len(modes))
        ax1.barh(y_m, t_env, height=0.42, color=COL['gray'], edgecolor=BORDER_COL['gray'], linewidth=1.2, label='截图+DOM解析')
        ax1.barh(y_m, t_vis, left=t_env, height=0.42, color=COL['teal'], edgecolor=BORDER_COL['teal'], linewidth=1.2, label='高分视觉编码')
        ax1.barh(y_m, t_llm, left=t_env + t_vis, height=0.42, color=COL['blue'], edgecolor=BORDER_COL['blue'], linewidth=1.2, label='LLM 预填充+解码')
        ax1.barh(y_m, t_act, left=t_env + t_vis + t_llm, height=0.42, color=COL['orange'], edgecolor=BORDER_COL['orange'], linewidth=1.2, label='GUI 动作执行等待')
        totals = t_env + t_vis + t_llm + t_act
        for yi, tot in zip(y_m, totals):
            ax1.text(tot + 0.6, yi, f'{tot:.1f} s', va='center', fontsize=11, weight='bold', color=COL['ink'])
        ax1.set_yticks(y_m, modes)
        ax1.invert_yaxis()
        ax1.set_ylim(2.25, -0.45)
        ax1.set_xlim(0, 43)
        ax1.set_xlabel('单次桌面任务端到端总耗时（s）', fontsize=11)
        ax1.set_title('A  Computer-Use 视觉 Agent 耗时分解 (OSWorld-Human)', loc='left', fontsize=12, weight='bold', pad=8)
        ax1.grid(axis='x', linestyle=':', alpha=0.5, color='#CBD5E1')
        ax1.legend(frameon=True, loc='lower right', ncol=2, fontsize=11)

        steps_k = np.arange(1, 16)
        s_img = 1.2  # 1,200 visual tokens per screenshot in K tokens
        s_txt = 0.08
        cum_full = np.cumsum(steps_k * s_img + steps_k * s_txt)
        win_w = 3
        per_step_win = np.minimum(steps_k, win_w) * s_img + steps_k * s_txt
        cum_win = np.cumsum(per_step_win)
        ax2.plot(steps_k, cum_full, color=BORDER_COL['orange'], lw=2.3, marker='o', ms=4.5, label='全量保留历史截图 (O(K²) 增长: 153.6K)')
        ax2.plot(steps_k, cum_win, color=BORDER_COL['green'], lw=2.3, marker='s', ms=4.5, label='滑窗保留近 3 帧截图 + 动作摘要 (58.8K, -61.7%)')
        ax2.set(xlim=(1, 15.5), ylim=(0, 195), xlabel='多轮 GUI 交互步数 K', ylabel='累计输入 Token 数（千）')
        ax2.set_title('B  多轮历史截图累积的二次方膨胀与滑窗截断收益', loc='left', fontsize=12, weight='bold', pad=8)
        ax2.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1')
        ax2.legend(frameon=True, loc='upper left', fontsize=11)
        out.save(f, 'ch03-osworld-computer-use-agent-timeline')

    from v41_case_figures import draw as draw_v41
    draw_v41(3, out)
    from core_principles_figures import draw as draw_principles
    draw_principles(3, out)
    return out.finish()
