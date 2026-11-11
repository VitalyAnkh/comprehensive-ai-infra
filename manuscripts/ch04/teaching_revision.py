"""Physical resources and finite buffers at final book dimensions (420pt publication grade)."""
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from figure_style import COL, BORDER_COL, TEXT_COL, STYLE, canvas, plot, text, box, card, badge, arrow, Exporter


def draw(here, data, teaching):
    out = Exporter(here)
    root = Path(here).resolve().parents[1]

    def save(f, n):
        out.save(f, 'figure-4-' + n)

    with plt.rc_context(STYLE):
        # 1. figure-4-1-reuse
        f, a = canvas(4.7)
        for row, m in enumerate([1, 256]):
            y = 0.58 - row * 0.44
            text(a, 0.04, y + 0.30, f'{m} 行共用 32 MiB 权重', 13, weight='bold')
            box(a, 0.34, y + 0.04, 0.32, 0.20, '权重 W\n(32 MiB)', 'orange', 11, weight='bold')
            for j in range(1 if m == 1 else 4):
                yy = y + 0.12 + (j - 1.5) * 0.06 if m > 1 else y + 0.12
                box(a, 0.04, yy, 0.17, 0.045, '', 'blue')
                box(a, 0.80, yy, 0.16, 0.045, '', 'green')
                arrow(a, (0.215, yy + 0.023), (0.335, y + 0.14), 'blue')
                arrow(a, (0.665, y + 0.14), (0.795, yy + 0.023), 'orange')
            text(a, 0.50, y - 0.065, '每行分摊：' + ('32 MiB' if m == 1 else '128 KiB'), 11.5, ha='center', weight='bold', color=TEXT_COL['blue' if m == 1 else 'green'])
        save(f, '1-reuse')

        # 2. figure-4-2-components
        f, a = canvas(4.9)
        box(a, 0.04, 0.79, 0.36, 0.14, '主机 CPU', 'gray', 11.5, weight='bold')
        box(a, 0.60, 0.79, 0.36, 0.14, '片外显存（HBM / GDDR）', 'blue', 11.5, weight='bold')
        box(a, 0.03, 0.06, 0.94, 0.60, '', 'gray')
        text(a, 0.06, 0.615, '加速器芯片内部', 12.5, weight='bold')
        box(a, 0.10, 0.41, 0.80, 0.12, '共享缓存与数据搬移引擎', 'blue', 11.5, weight='bold')
        arrow(a, (0.78, 0.785), (0.68, 0.535), 'blue')
        arrow(a, (0.22, 0.785), (0.24, 0.535), kind='control', color='purple')
        box(a, 0.08, 0.14, 0.25, 0.15, '局部缓冲\n(SRAM)', 'green', 11, weight='bold')
        box(a, 0.38, 0.14, 0.24, 0.15, '矩阵单元\n(Tensor / Cube)', 'orange', 11, weight='bold')
        box(a, 0.67, 0.14, 0.25, 0.15, '累加存储\n(寄存器 / TMEM)', 'purple', 11, weight='bold')
        arrow(a, (0.24, 0.405), (0.21, 0.295), 'blue')
        arrow(a, (0.335, 0.215), (0.375, 0.215), 'green')
        arrow(a, (0.625, 0.215), (0.665, 0.215), 'orange')
        save(f, '2-components')

        # 3. figure-4-davinci-attention-path
        f, a = canvas(4.5)
        card(a, 0.02, 0.68, 0.22, 0.22, '全局内存', ['权重、Q/K/V'], 'orange', 11.5, 11)
        card(a, 0.27, 0.68, 0.22, 0.22, 'MTE / NDDMA', ['搬移与地址'], 'green', 11.5, 11)
        card(a, 0.52, 0.68, 0.22, 0.22, 'L1 / UB', ['行组、缓冲槽'], 'blue', 11.5, 11)
        card(a, 0.77, 0.68, 0.21, 0.22, 'Scalar', ['循环、同步'], 'gray', 11.5, 11)
        arrow(a, (0.242, 0.79), (0.268, 0.79), 'orange')
        arrow(a, (0.492, 0.79), (0.518, 0.79), 'green')
        arrow(a, (0.742, 0.79), (0.768, 0.79), kind='control', color='gray')

        card(a, 0.10, 0.36, 0.24, 0.20, 'L0A / L0B', ['输入 tile'], 'orange', 11.5, 11)
        card(a, 0.39, 0.36, 0.24, 0.20, 'Cube / AIC', ['QK、PV、L0C'], 'orange', 11.5, 11)
        card(a, 0.68, 0.36, 0.24, 0.20, 'Vector / AIV', ['Softmax、归约'], 'blue', 11.5, 11)
        arrow(a, (0.63, 0.675), (0.22, 0.565), 'blue')
        arrow(a, (0.345, 0.46), (0.385, 0.46), 'orange')
        arrow(a, (0.635, 0.46), (0.675, 0.46), 'orange')

        card(a, 0.24, 0.06, 0.52, 0.20, 'CV 直通或 UB / L1 交接', ['QK → Softmax → PV'], 'green', 11.5, 11)
        arrow(a, (0.51, 0.355), (0.50, 0.265), 'orange')
        arrow(a, (0.78, 0.355), (0.58, 0.265), kind='control', color='blue')
        save(f, 'davinci-attention-path')

        # 4. figure-4-matrix-array
        f, a = canvas(3.9)
        text(a, 0.04, 0.94, '小阵列反复复用输入与权重', 13, weight='bold')
        for i in range(3):
            text(a, 0.12, 0.66 - i * 0.18, f'$A_{i}$', 11.5, ha='center', weight='bold', color=TEXT_COL['blue'])
            for j in range(3):
                x = 0.30 + j * 0.22
                y = 0.59 - i * 0.18
                box(a, x, y, 0.16, 0.14, '乘加', 'green', 11, weight='bold')
                arrow(a, (0.18 if j == 0 else x - 0.06, y + 0.07), (x, y + 0.07), 'blue')
                if i == 0:
                    text(a, x + 0.08, 0.84, f'$W_{j}$', 11.5, ha='center', weight='bold', color=TEXT_COL['orange'])
                arrow(a, (x + 0.08, 0.79 if i == 0 else y + 0.18), (x + 0.08, y + 0.14), 'orange')
        text(a, 0.50, 0.08, '每个乘加单元保留部分和，继续累加下一对输入', 11, ha='center', weight='bold', color=COL['muted'])
        save(f, 'matrix-array')

        # 5. figure-4-3-expert-rows
        f, a = canvas(4.0)
        text(a, 0.04, 0.94, '每块固定 16 行，灰色补零行也消耗阵列周期', 13, weight='bold')
        for x, count, label in [(0.08, 2, '每专家 2 行\n(14 行补零)'), (0.57, 16, '每专家 64 行\n(由 4 个满块组成)')]:
            for i in range(16):
                c_fill = COL['green'] if i < count else COL['gray']
                c_edge = BORDER_COL['green'] if i < count else BORDER_COL['gray']
                a.add_patch(Rectangle((x, 0.74 - i * 0.034), 0.30, 0.028, facecolor=c_fill, edgecolor=c_edge, lw=0.5))
            text(a, x + 0.15, 0.10, label, 11.5, ha='center', weight='bold')
        save(f, '3-expert-rows')

        # 6. figure-4-expert-padding-total
        f, a = plot(3.3, left=0.28, bottom=0.19)
        a.barh([0, 1], [512, 512], color=COL['green'], edgecolor=BORDER_COL['green'], linewidth=1.2, height=0.52, label='有效行')
        a.barh([0, 1], [3584, 0], left=[512, 512], color=COL['gray'], edgecolor=BORDER_COL['gray'], linewidth=1.2, height=0.52, label='补零行')
        a.text(4096 + 60, 0, '4,096', va='center', fontsize=11, weight='bold', color=COL['ink'])
        a.text(512 + 60, 1, '512', va='center', fontsize=11, weight='bold', color=COL['ink'])
        a.set(yticks=[0, 1], yticklabels=['256 个专家', '8 个专家'], xlim=(0, 4800), ylim=(-0.65, 1.65), xlabel='执行的矩阵行数（含补零）')
        a.invert_yaxis()
        a.grid(axis='x', linestyle=':', alpha=0.45, color='#CBD5E1')
        a.legend(frameon=True, loc='lower right', fontsize=11)
        save(f, 'expert-padding-total')

        # 7. figure-4-4-attention
        vals = np.array(data['4-4']['service_cycles'])
        f, a = plot(4.1, left=0.28, bottom=0.17, top=0.84)
        for i, (label, c) in enumerate([('矩阵', 'orange'), ('共享内存', 'blue'), ('指数', 'green')]):
            a.barh(np.arange(4) + (i - 1) * 0.23, vals[:, i], height=0.21, color=COL[c], edgecolor=BORDER_COL[c], linewidth=1.1, label=label)
        a.set(yticks=range(4), yticklabels=['原配置', '矩阵 ×2', '矩阵、指数均 ×2', '三项 ×2'], xlabel='单计算组耗时（周期）', xlim=(0, 1150), ylim=(-0.7, 3.6))
        a.invert_yaxis()
        a.grid(axis='x', linestyle=':', alpha=0.45, color='#CBD5E1')
        a.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.45, 1.18), frameon=True, fontsize=11)
        save(f, '4-attention')

        # 8. figure-4-5-precision
        f, a = canvas(4.3)
        for y, title, stages in [
            (0.58, '先展开为高精度副本', [('压缩 8.5 MiB', 'blue'), ('展开 32 MiB', 'orange'), ('BF16 计算', 'green')]),
            (0.12, '在低精度路径中直接计算', [('压缩 8.5 MiB', 'blue'), ('低精度计算', 'orange'), ('缩放与合并', 'green')]),
        ]:
            text(a, 0.04, y + 0.28, title, 12.5, weight='bold')
            for i, (label, c) in enumerate(stages):
                x = 0.03 + i * 0.335
                box(a, x, y, 0.27, 0.20, label, c, 11, weight='bold')
                if i < 2:
                    arrow(a, (x + 0.275, y + 0.10), (x + 0.33, y + 0.10), c)
        save(f, '5-precision')

        # 9. figure-4-6-capacity
        d = data['4-6']
        f, a = plot(3.9, left=0.29, bottom=0.24, top=0.88)
        for y, row in enumerate(d['cases']):
            left = 0
            for n, c, label in [
                (d['weight_bytes'], 'blue', '权重'),
                (d['workspace_bytes'], 'orange', '工作区'),
                (row['requests'] * row['context_multiplier'] * d['kv_bytes_per_request'], 'green', 'KV'),
            ]:
                a.barh(y, n / 1e9, left=left, height=0.52, color=COL[c], edgecolor=BORDER_COL[c], linewidth=1.2, label=label if y == 0 else None)
                left += n / 1e9
        a.axvline(24, ls='--', color='#B91C1C', lw=1.4)
        a.text(24, -0.52, '24 GB 上限', ha='center', va='bottom', fontsize=11, weight='bold', color='#B91C1C', bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='none'))
        a.set(yticks=range(3), yticklabels=['8K token × 4 请求', '8K token × 5 请求', '16K token × 2 请求'], xlim=(0, 27.5), ylim=(-0.6, 2.6), xlabel='RTX 4090 显存占用（GB）')
        a.invert_yaxis()
        a.grid(axis='x', linestyle=':', alpha=0.45, color='#CBD5E1')
        a.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.45, -0.22), frameon=True, fontsize=11)
        save(f, '6-capacity')

        # 10. figure-4-memory-inflight
        f, a = canvas(3.3)
        text(a, 0.04, 0.92, '同时等待返回的访问持续占用请求槽', 13, weight='bold')
        for i in range(4):
            box(a, 0.05 + i * 0.235, 0.48, 0.20, 0.20, f'请求 {i+1}', 'blue', 11, weight='bold')
        text(a, 0.50, 0.29, '每请求 128 bytes，发出后 500 ns 返回', 11.5, ha='center', weight='bold')
        text(a, 0.50, 0.11, '持续带宽同时受限于物理接口速率与在途请求槽上限', 11, ha='center', weight='bold', color=COL['muted'])
        save(f, 'memory-inflight')

        # 11. figure-4-7-memory
        d = data['4-7']
        f, a = plot(3.7, left=0.18, bottom=0.18)
        for b, ys, c in zip(['RTX 4090：1008 GB/s', 'RTX 5090：1792 GB/s'], d['bandwidth_upper_bytes_per_second'], ['blue', 'green']):
            a.plot(d['requests'], np.array(ys) / 1e12, label=b, color=BORDER_COL[c], lw=2.2)
        a.set(xlim=(0, 10000), ylim=(0, 2.4), xlabel='同时未完成请求数 N', ylabel='带宽上界（TB/s）')
        a.grid(True, linestyle=':', alpha=0.45, color='#CBD5E1')
        a.legend(frameon=True, loc='upper left', fontsize=11)
        save(f, '7-memory')

        # 12. figure-4-8-layout
        f, a = plot(3.5, left=0.22, bottom=0.19)
        for i in range(4):
            a.barh(i, 8192, color=COL['gray'], edgecolor=BORDER_COL['gray'], linewidth=1.0, height=0.52)
            a.barh(i, 256, color=COL['blue'], edgecolor=BORDER_COL['blue'], linewidth=1.2, height=0.52)
        a.set(yticks=range(4), yticklabels=['第 0 行', '第 1 行', '第 2 行', '第 127 行'], xlim=(0, 8600), xlabel='相对每行起点的字节偏移', xticks=[0, 4096, 8192])
        a.invert_yaxis()
        a.grid(axis='x', linestyle=':', alpha=0.45, color='#CBD5E1')
        a.annotate('实际读取 256 bytes', (256, 0), xytext=(2000, 0.65), arrowprops={'arrowstyle': '-|>', 'color': BORDER_COL['blue'], 'lw': 1.3}, fontsize=11, weight='bold', color=TEXT_COL['blue'])
        save(f, '8-layout')

        # 13. figure-4-slot-lifetime
        f, a = plot(3.2, left=0.12, bottom=0.20)
        a.barh(0, 320, left=0, color=COL['gray'], edgecolor=BORDER_COL['gray'], linewidth=1.1, height=0.62, label='槽位占用')
        a.barh(0, 64, left=0, color=COL['blue'], edgecolor=BORDER_COL['blue'], linewidth=1.2, height=0.40)
        a.barh(0, 128, left=192, color=COL['green'], edgecolor=BORDER_COL['green'], linewidth=1.2, height=0.40)
        a.axvline(192, color=BORDER_COL['orange'], lw=1.4)
        a.set(xlim=(0, 350), ylim=(-0.7, 0.7), yticks=[], xticks=[0, 64, 192, 320], xlabel='时间（tick）')
        for x, label in [(32, '传输'), (128, '等待返回'), (256, '计算')]:
            a.text(x, 0.42, label, fontsize=11, ha='center', weight='bold', color=COL['ink'])
        a.text(175, -0.44, '320 tick 后计算完成，输入槽才能释放复用', fontsize=11, ha='center', weight='bold', color=COL['muted'])
        save(f, 'slot-lifetime')

        # 14, 15, 16. figure-4-9-pipeline, pipeline-two, pipeline-three
        for slots, name in [(1, '9-pipeline'), (2, 'pipeline-two'), (3, 'pipeline-three')]:
            r = teaching['baseline'][slots - 1]
            f, a = plot(3.5, left=0.17, bottom=0.18)
            for t in r['chunks']:
                y = t['chunk']
                a.barh(y, t['slot_released'] - t['issue_start'], left=t['issue_start'], height=0.62, color=COL['gray'], edgecolor=BORDER_COL['gray'], linewidth=0.9)
                a.barh(y, t['transfer_end'] - t['issue_start'], left=t['issue_start'], height=0.40, color=COL['blue'], edgecolor=BORDER_COL['blue'], linewidth=1.1)
                a.barh(y, t['compute_end'] - t['compute_start'], left=t['compute_start'], height=0.40, color=COL['green'], edgecolor=BORDER_COL['green'], linewidth=1.1)
                a.plot(t['data_ready'], y, '|', color=BORDER_COL['orange'], markersize=13, markeredgewidth=2.0)
            a.set(yticks=range(4), yticklabels=[f'块 {i}' for i in range(4)], xlim=(0, 1320), xticks=[0, 320, 640, 960, 1280], xlabel='时间（tick）')
            a.grid(axis='x', linestyle=':', alpha=0.45, color='#CBD5E1')
            a.invert_yaxis()
            save(f, name)

        # 17. figure-4-matrix-vector-handoff
        f, a = canvas(4.0)
        text(a, 0.04, 0.93, '传递完整行，两组交替使用缓冲区', 13, weight='bold')
        box(a, 0.04, 0.61, 0.26, 0.17, '矩阵单元\n计算 QK', 'orange', 11, weight='bold')
        box(a, 0.70, 0.61, 0.26, 0.17, '向量单元\n计算 Softmax', 'green', 11, weight='bold')
        box(a, 0.38, 0.66, 0.23, 0.11, '槽 A', 'blue', 11, weight='bold')
        box(a, 0.38, 0.42, 0.23, 0.11, '槽 B', 'purple', 11, weight='bold')
        arrow(a, (0.305, 0.695), (0.375, 0.715), 'orange')
        arrow(a, (0.615, 0.715), (0.695, 0.695), 'blue')
        arrow(a, (0.17, 0.605), (0.375, 0.475), 'orange')
        box(a, 0.26, 0.09, 0.48, 0.17, '矩阵单元计算 PV\n用完后释放缓冲区', 'orange', 11, weight='bold')
        arrow(a, (0.83, 0.605), (0.74, 0.18), 'green')
        save(f, 'matrix-vector-handoff')

        # 18. figure-4-davinci-attention-timeline (2 panels: Staged 3200 tick vs Direct 3008 tick)
        staged = json.loads((root / 'calculations/results/matrix-vector-staged-rows32-slots2.json').read_text())
        direct = json.loads((root / 'calculations/results/matrix-vector-direct-rows32-slots2.json').read_text())
        f, axs = plt.subplots(2, 1, figsize=(420 / 72, 5.4), sharex=True)
        f.subplots_adjust(left=0.22, right=0.96, bottom=0.12, top=0.91, hspace=0.45)
        for ax, doc, title, fin in [
            (axs[0], staged, 'A  写出再读入（两槽、32 行分组，完成于 3200 tick）', 3200),
            (axs[1], direct, 'B  片上直接传递（两槽、32 行分组，完成于 3008 tick）', 3008),
        ]:
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_title(title, loc='left', fontsize=11.5, weight='bold', pad=6)
            for item in doc['timeline']:
                res = item['resource']
                st, dur = item['start_tick'], item['duration_ticks']
                if res == 'matrix':
                    ax.barh(0, dur, left=st, height=0.48, color=COL['orange'], edgecolor=BORDER_COL['orange'], linewidth=0.9)
                elif res == 'handoff':
                    ax.barh(1, dur, left=st, height=0.48, color=COL['green'], edgecolor=BORDER_COL['green'], linewidth=0.9)
                elif res == 'vector':
                    ax.barh(2, dur, left=st, height=0.48, color=COL['blue'], edgecolor=BORDER_COL['blue'], linewidth=0.9)
                    grp = item['id'].split('.')[0].upper()
                    ax.text(st + dur / 2, 2, grp, ha='center', va='center', fontsize=11, weight='bold', color=COL['ink'])
            for own in doc['slot_ownership']:
                y_slot = 3 + own['slot'] * 0.45
                st, rel = own['start_tick'], own['release_tick']
                c_slot = 'gray' if own['slot'] == 0 else 'purple'
                ax.barh(y_slot, rel - st, left=st, height=0.36, color=COL[c_slot], edgecolor=BORDER_COL[c_slot], linewidth=0.9)
                ax.text(st + (rel - st) / 2, y_slot, f"G{own['group']} (槽{own['slot']})", ha='center', va='center', fontsize=11, weight='bold', color=COL['ink'])
            ax.axvline(fin, ls='--', lw=1.2, color=BORDER_COL['orange'])
            ax.set(yticks=[0, 1, 2, 3.22], yticklabels=['Cube / AIC', '交接 / CV', 'Vector / AIV', '槽位锁定'], xlim=(0, 3420), ylim=(-0.55, 3.85))
            ax.set_xticks([0, 512, 1024, 1536, 2048, 2560, 3200])
            ax.invert_yaxis()
            ax.grid(axis='x', linestyle=':', alpha=0.45, color='#CBD5E1')
        axs[1].text(3008 + 25, 0.95, '3008', ha='left', va='center', fontsize=11, weight='bold', color=TEXT_COL['orange'])
        axs[1].set_xlabel('教学时间（tick）')
        save(f, 'davinci-attention-timeline')

        # 19 & 20. figure-4-10-locality & locality-compute
        cases = teaching['die_locality']['cases']
        for move, name in [(False, '10-locality'), (True, 'locality-compute')]:
            f, a = canvas(4.8)
            text(a, 0.03, 0.96, '计算都放在 die 0' if not move else '计算放到权重所在的 die', 13, weight='bold')
            for row, c in enumerate(cases):
                y = 0.56 - row * 0.46
                text(a, 0.03, y + 0.30, 'HGX B200' if row == 0 else '昇腾 910C', 12, weight='bold')
                box(a, 0.03, y, 0.27, 0.25, 'die 0\n32 GiB 权重\n计算', 'blue', 11, weight='bold')
                box(a, 0.70, y, 0.27, 0.25, 'die 1\n32 GiB 权重' + ('\n计算' if move else ''), 'orange', 11, weight='bold')
                if move:
                    arrow(a, (0.305, y + 0.17), (0.695, y + 0.17), 'blue')
                    arrow(a, (0.695, y + 0.08), (0.305, y + 0.08), 'orange')
                    act_txt = f"输入与结果 64 MiB：{c['activation_us']:.1f} μs" if c['activation_us'] < 100 else f"输入与结果 64 MiB：{c['activation_us']/1000:.2f} ms"
                    text(a, 0.50, y + 0.215, act_txt, 11, ha='center', weight='bold')
                    text(a, 0.50, y - 0.045, f"两侧各读本地权重：{c['split_ms']:.1f} ms", 11.5, ha='center', weight='bold', color=TEXT_COL['green'])
                else:
                    arrow(a, (0.695, y + 0.125), (0.305, y + 0.125), 'orange')
                    limit = '受 die 1 的 HBM 限制' if c['link_to_hbm_ratio'] >= 1 else '受 die 间链路限制'
                    text(a, 0.50, y + 0.20, f"跨 die 读 32 GiB：{c['remote_ms']:.1f} ms", 11, ha='center', weight='bold')
                    text(a, 0.50, y + 0.05, limit, 11, ha='center', color=COL['muted'], weight='bold')
                    text(a, 0.50, y - 0.045, f"阶段读取时间：{c['overlapped_ms']:.1f} ms", 11.5, ha='center', weight='bold', color=TEXT_COL['orange'])
            save(f, name)

        # NEW 21. ch04-unified-memory-vs-gddr7-and-gpu-tee (420pt, 2-panel vertical stack)
        f = plt.figure(figsize=(420 / 72, 5.7))
        ax1 = f.add_axes([0.28, 0.60, 0.67, 0.30])
        ax1.spines[['top', 'right']].set_visible(False)
        ax1.set_title('A  4 GiB 活跃权重单步读取下界（统一内存 vs GDDR7）', loc='left', fontsize=11.5, weight='bold', pad=8)
        devs = ['Apple M2 Pro\n(200 GB/s)', 'Apple M2 Max\n(400 GB/s)', 'RTX PRO 6000\n(1,792 GB/s)']
        ms_vals = [21.47, 10.74, 2.40]
        cols_a = ['orange', 'blue', 'green']
        for i, (v, c) in enumerate(zip(ms_vals, cols_a)):
            ax1.barh(i, v, height=0.48, color=COL[c], edgecolor=BORDER_COL[c], linewidth=1.2)
            ratio_txt = ' (8.96×)' if i == 0 else (' (4.48×)' if i == 1 else ' (1.00× 基准)')
            ax1.text(v + 0.5, i, f'{v:.2f} ms{ratio_txt}', va='center', fontsize=11, weight='bold', color=COL['ink'])
        ax1.set(yticks=range(3), yticklabels=devs, xlim=(0, 31.5), ylim=(-0.55, 2.55), xlabel='4 GiB (4.295 GB) 单步读取时间下界（ms）')
        ax1.invert_yaxis()
        ax1.grid(axis='x', linestyle=':', alpha=0.45, color='#CBD5E1')

        ax2 = f.add_axes([0.28, 0.11, 0.67, 0.35])
        ax2.spines[['top', 'right']].set_visible(False)
        ax2.set_title('B  GPU 机密计算（TEE）跨 PCIe 64 MiB 传输拆解', loc='left', fontsize=11.5, weight='bold', pad=8)
        tee_labels = ['非 TEE 明文直传', '软 TEE 串行弹跳', '软 TEE 16 块流水', '硬件 TEE-I/O (IDE)']
        p_host = [0.0, 5.59, 5.59, 0.0]
        p_dma = [1.20, 1.20, 0.08, 1.29]
        tot_tee = [1.20, 6.79, 5.67, 1.29]
        y_t = np.arange(4)
        ax2.barh(y_t, p_host, height=0.48, color=COL['orange'], edgecolor=BORDER_COL['orange'], linewidth=1.1, label='CPU 弹跳加密')
        ax2.barh(y_t, p_dma, left=p_host, height=0.48, color=COL['blue'], edgecolor=BORDER_COL['blue'], linewidth=1.1, label='暴露 PCIe 传输')
        for yi, tot in enumerate(tot_tee):
            ax2.text(tot + 0.15, yi, f'{tot:.2f} ms', va='center', fontsize=11, weight='bold', color=COL['ink'])
        ax2.set(yticks=y_t, yticklabels=tee_labels, xlim=(0, 8.6), ylim=(-0.55, 4.75), xlabel='64 MiB 跨 PCIe 5.0 x16 端到端就绪时间（ms）')
        ax2.invert_yaxis()
        ax2.grid(axis='x', linestyle=':', alpha=0.45, color='#CBD5E1')
        ax2.legend(ncol=2, frameon=True, loc='lower right', fontsize=11, columnspacing=1.0, handlelength=1.2)
        out.save(f, 'ch04-unified-memory-vs-gddr7-and-gpu-tee')

        # 22 & 23. figure-4-11-interconnect & large-message
        d = data['4-11']
        for i, name in enumerate(['11-interconnect', 'large-message']):
            f, a = plot(3.3, left=0.20, bottom=0.18)
            vals = np.array(d['total_us'][i])
            alpha = d['alpha_seconds'] * 1e6
            a.bar([0, 1], [alpha, alpha], width=0.52, color=COL['orange'], edgecolor=BORDER_COL['orange'], linewidth=1.2, label='启动')
            a.bar([0, 1], vals - alpha, bottom=alpha, width=0.52, color=COL['blue'], edgecolor=BORDER_COL['blue'], linewidth=1.2, label='传输')
            a.set(xticks=[0, 1], xticklabels=['A100 NVLink\n300 GB/s', 'H100 NVLink\n450 GB/s'], ylabel='传输时间（μs）', ylim=(0, max(vals) * 1.45))
            a.grid(axis='y', linestyle=':', alpha=0.45, color='#CBD5E1')
            a.legend(ncol=2, frameon=True, loc='upper right', fontsize=11)
            for j, v in enumerate(vals):
                a.text(j, v + max(vals) * 0.035, f'{v:.3f}', ha='center', fontsize=11, weight='bold', color=COL['ink'])
            save(f, name)

        # 24. figure-4-nvidia-product-tiers
        f = plt.figure(figsize=(420 / 72, 3.9))
        ax1 = f.add_axes([0.15, 0.19, 0.34, 0.68])
        ax2 = f.add_axes([0.66, 0.19, 0.30, 0.68])
        for ax in (ax1, ax2):
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(True, linestyle=':', alpha=0.45, color='#CBD5E1')
        tiers = [
            ('RTX 3090', 936, 71.0, 75.9, 'blue', (220, -10), 'left'),
            ('RTX 4090', 1008, 165.2, 163.9, 'blue', (-80, 75), 'center'),
            ('RTX 5090', 1792, 209.5, 116.9, 'blue', (220, -35), 'left'),
            ('PRO 6000', 1792, 503.8, 281.1, 'purple', (0, 80), 'center'),
            ('A100 80GB', 2039, 312.0, 153.0, 'orange', (240, 20), 'left'),
            ('H100 SXM', 3350, 989.4, 295.3, 'orange', (0, 85), 'center'),
            ('H200 SXM', 4800, 989.4, 206.1, 'green', (-120, -115), 'center'),
        ]
        ax1.set_title('A  带宽—算力坐标', loc='left', fontsize=11.5, weight='bold', pad=8)
        for name, bw, fp, _, col, (dx, dy), ha_mode in tiers:
            ax1.scatter([bw], [fp], s=62, facecolor=COL[col], edgecolor=BORDER_COL[col], linewidth=1.4, zorder=4)
            short = name.replace('RTX ', '').replace(' 80GB', '').replace(' SXM', '')
            ax1.text(bw + dx, fp + dy, short, ha=ha_mode, va='center', fontsize=11, weight='bold', color=COL['ink'])
        ax1.set(xlim=(100, 5500), ylim=(-40, 1180), xlabel='显存带宽（GB/s）', ylabel='稠密 BF16 算力（TFLOP/s）')

        ax2.set_title('B  转折点 I* = P/R', loc='left', fontsize=11.5, weight='bold', pad=8)
        for idx, (name, _, _, istar, col, _, _) in enumerate(tiers):
            ax2.barh(idx, istar, height=0.55, color=COL[col], edgecolor=BORDER_COL[col], linewidth=1.2)
            ax2.text(istar + 8, idx, f'{istar:.1f}', va='center', fontsize=11, weight='bold', color=COL['ink'])
        ax2.set(yticks=range(len(tiers)), yticklabels=[t[0] for t in tiers], xlim=(0, 395), xlabel='转折点 I*（FLOP/B）')
        ax2.invert_yaxis()
        save(f, 'nvidia-product-tiers')

        # 25. figure-4-sram-kv-scaling
        f, a = plot(4.0, left=0.17, bottom=0.20, right=0.96, top=0.86)
        a.set_title('纯片上 SRAM 架构（220 MiB/颗）下的级联芯片规模', loc='left', fontsize=11.5, weight='bold', pad=10)
        x_sram = np.arange(4)
        w_chips = np.array([72, 72, 72, 72])
        kv_chips = np.array([0, 6, 42, 168])
        a.bar(x_sram, w_chips, width=0.48, color=COL['blue'], edgecolor=BORDER_COL['blue'], linewidth=1.2, label='静态权重芯片数 (15,623 MiB → 72 颗)')
        a.bar(x_sram, kv_chips, bottom=w_chips, width=0.48, color=COL['orange'], edgecolor=BORDER_COL['orange'], linewidth=1.2, label='动态 KV 缓存芯片数 (220 MiB SRAM/颗)')
        for xi, wc, kc in zip(x_sram, w_chips, kv_chips):
            a.text(xi, wc / 2, '72 颗', ha='center', va='center', fontsize=11, weight='bold', color=COL['ink'])
            a.text(xi, wc + kc + 6, f'共 {wc + kc} 颗', ha='center', va='bottom', fontsize=11, weight='bold', color=COL['ink'])
            if kc >= 30:
                a.text(xi, wc + kc / 2, f'+{kc} 颗', ha='center', va='center', fontsize=11, weight='bold', color=TEXT_COL['orange'])
            elif kc > 0:
                a.text(xi + 0.28, wc + kc / 2, f'+{kc} 颗', ha='left', va='center', fontsize=11, weight='bold', color=TEXT_COL['orange'])
        a.set(
            xticks=x_sram,
            xticklabels=['仅静态权重\n(B=0)', '单请求 8K\n(B=1, 8K)', '8 并发 8K\n(B=8, 8K)', '8 并发 32K\n(B=8, 32K)'],
            ylim=(0, 285),
            ylabel='级联 TSP 芯片数量（颗）',
        )
        a.grid(axis='y', linestyle=':', alpha=0.45, color='#CBD5E1')
        a.legend(frameon=True, loc='upper left', fontsize=11)
        save(f, 'sram-kv-scaling')

        # 26. figure-4-12-specialization
        d = data['4-12']
        f, a = plot(3.5, left=0.18, bottom=0.18)
        a.axhline(d['active_weight_bytes'] / 1e9, color=BORDER_COL['blue'], lw=2.2, label='每批权重')
        a.plot(d['batch'], np.array(d['kv_read_bytes']) / 1e9, color=BORDER_COL['green'], lw=2.2, label='各请求 KV')
        a.axvline(13, ls=':', color=BORDER_COL['orange'], lw=1.3)
        a.set(xlim=(1, 32), ylim=(0, 45), xlabel='批内请求数 B', ylabel='每步读取（GB）')
        a.grid(True, linestyle=':', alpha=0.45, color='#CBD5E1')
        a.legend(frameon=True, loc='upper left', fontsize=11)
        save(f, '12-specialization')

        # 27. figure-4-13-roofline
        d = data['4-13']
        f, a = plot(3.6, left=0.18, bottom=0.18)
        for key, label, c in [('compute_us', '矩阵计算', 'orange'), ('memory_us', '片外读取', 'blue')]:
            a.plot(d['rows'], d[key], label=label, color=BORDER_COL[c], lw=2.2)
        a.axvline(179, ls=':', color='#64748B', lw=1.3)
        a.text(179 - 6, 52, 'M = 179', ha='right', fontsize=11, weight='bold', color=COL['muted'])
        a.set(xlim=(1, 256), ylim=(0, 65), xlabel='本次 token 数 M', ylabel='资源时间（μs）')
        a.grid(True, linestyle=':', alpha=0.45, color='#CBD5E1')
        a.legend(frameon=True, loc='upper left', fontsize=11)
        save(f, '13-roofline')

        # 28 & 29. figure-4-14-performance & performance-traffic
        d = data['4-14']
        for key, name, label in [('wall_us', '14-performance', '每次调用总耗时（μs）'), ('dram_read_mib', 'performance-traffic', 'DRAM 读取（MiB）')]:
            f, a = plot(3.5, left=0.19, bottom=0.19)
            vals = d[key]
            cols = ['blue', 'green', 'blue', 'green']
            for idx, (v, c) in enumerate(zip(vals, cols)):
                a.bar(idx, v, color=COL[c], edgecolor=BORDER_COL[c], linewidth=1.2, width=0.56)
            a.set(xticks=range(4), xticklabels=['1 行\n复用', '1 行\n轮换', '256 行\n复用', '256 行\n轮换'], ylabel=label, ylim=(0, max(vals) * 1.32))
            a.grid(axis='y', linestyle=':', alpha=0.45, color='#CBD5E1')
            for i, v in enumerate(vals):
                a.text(i, v + max(vals) * 0.03, '256 B' if v < 0.001 else f'{v:.1f}', ha='center', fontsize=11, weight='bold', color=COL['ink'])
            save(f, name)

        # 30. ch04-hopper-blackwell-rubin-scaling (Path B Fig 4-46)
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(420 / 72, 6.4))
        f.subplots_adjust(left=0.16, right=0.95, bottom=0.10, top=0.92, hspace=0.46)
        for ax in (ax1, ax2):
            ax.spines[['top', 'right']].set_visible(False)

        gens = ['H100 SXM', 'H200 SXM', 'HGX B200', 'Rubin GPU']
        hbm_bw = [3.35, 4.80, 8.00, 22.00]
        hbm_cap = [80, 141, 180, 288]
        x_g = np.arange(len(gens))
        ax1.bar(x_g, hbm_bw, width=0.46, color=[COL['blue'], COL['teal'], COL['orange'], COL['green']], edgecolor=[BORDER_COL['blue'], BORDER_COL['teal'], BORDER_COL['orange'], BORDER_COL['green']], linewidth=1.2)
        for xi, bw_v, cap_v in zip(x_g, hbm_bw, hbm_cap):
            ax1.text(xi, bw_v + 0.6, f'{bw_v:.2f} TB/s\n({cap_v} GB)' if bw_v < 4 else f'{bw_v:.1f} TB/s\n({cap_v} GB)', ha='center', va='bottom', fontsize=11, weight='bold', color=COL['ink'])
        ax1.set_xticks(x_g, gens)
        ax1.set_xlim(-0.55, 3.55)
        ax1.set_ylim(0, 28.5)
        ax1.set_ylabel('HBM 带宽（TB/s）', fontsize=11)
        ax1.set_title('A  四代数据中心 GPU 的 HBM 容量与带宽跃迁 (TC-07)', loc='left', fontsize=12, weight='bold', pad=8)
        ax1.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E1')

        gens_c = ['H100 (FP8)', 'B200 (FP4)', 'Rubin (NVFP4)']
        pflops_low = [1.98, 9.0, 50.0]
        nvlink_tb = [0.9, 1.8, 3.6]
        x_c = np.arange(len(gens_c))
        ax2.bar(x_c, pflops_low, width=0.42, color=[COL['blue'], COL['orange'], COL['green']], edgecolor=[BORDER_COL['blue'], BORDER_COL['orange'], BORDER_COL['green']], linewidth=1.2)
        for xi, pf_v, nv_v in zip(x_c, pflops_low, nvlink_tb):
            pf_str = f'{pf_v:.2f} PFLOPS' if pf_v < 5 else f'{pf_v:.0f} PFLOPS'
            ax2.text(xi, pf_v + 1.5, f'{pf_str}\n(NVLink {nv_v} TB/s)', ha='center', va='bottom', fontsize=11, weight='bold', color=COL['ink'])
        ax2.set_xticks(x_c, gens_c)
        ax2.set_xlim(-0.55, 2.55)
        ax2.set_ylim(0, 64)
        ax2.set_ylabel('低比特算力（PFLOPS）', fontsize=11)
        ax2.set_title('B  主推低比特算力与双向 NVLink 带宽演进 (TC-08)', loc='left', fontsize=12, weight='bold', pad=8)
        ax2.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E1')
        out.save(f, 'ch04-hopper-blackwell-rubin-scaling')

        # 31. ch04-tpu7x-dual-chiplet-vs-sram-asic (Path B Fig 4-47)
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(420 / 72, 6.4))
        f.subplots_adjust(left=0.22, right=0.95, bottom=0.10, top=0.92, hspace=0.46)
        for ax in (ax1, ax2):
            ax.spines[['top', 'right']].set_visible(False)

        tpu_gens = ['TPU v5p', 'TPU v6e (Trillium)', 'TPU7x (Ironwood)']
        tpu_bf16 = [0.459, 0.918, 2.307]
        tpu_bw = [2.765, 1.640, 7.380]
        tpu_hbm = [95, 32, 192]
        x_t = np.arange(len(tpu_gens))
        ax1.bar(x_t, tpu_bw, width=0.42, color=[COL['blue'], COL['purple'], COL['green']], edgecolor=[BORDER_COL['blue'], BORDER_COL['purple'], BORDER_COL['green']], linewidth=1.2)
        for xi, bw_v, cap_v, bf_v in zip(x_t, tpu_bw, tpu_hbm, tpu_bf16):
            ax1.text(xi, bw_v + 0.25, f'{bw_v:.2f} TB/s ({cap_v} GiB)\nBF16 {bf_v*1000:.0f} TFLOPs', ha='center', va='bottom', fontsize=11, weight='bold', color=COL['ink'])
        ax1.set_xticks(x_t, tpu_gens)
        ax1.set_xlim(-0.55, 2.55)
        ax1.set_ylim(0, 9.8)
        ax1.set_ylabel('HBM 带宽（TB/s）', fontsize=11)
        ax1.set_title('A  Google TPU 近三代 HBM 容量、带宽与算力演进', loc='left', fontsize=12, weight='bold', pad=8)
        ax1.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E1')

        devs_4g = ['M2 Pro\n(200 GB/s)', 'M2 Max\n(400 GB/s)', 'PRO 6000\n(1.79 TB/s)', 'TPU7x\n(7.38 TB/s)']
        t_4gib_ms = [21.47, 10.74, 2.40, 0.58]
        y_d = np.arange(len(devs_4g))
        ax2.barh(y_d, t_4gib_ms, height=0.52, color=[COL['gray'], COL['blue'], COL['purple'], COL['green']], edgecolor=[BORDER_COL['gray'], BORDER_COL['blue'], BORDER_COL['purple'], BORDER_COL['green']], linewidth=1.2)
        for yi, tv in zip(y_d, t_4gib_ms):
            ax2.text(tv + 0.5, yi, f'{tv:.2f} ms', va='center', fontsize=11, weight='bold', color=COL['ink'])
        ax2.set_yticks(y_d, devs_4g)
        ax2.invert_yaxis()
        ax2.set_xlim(0, 27.5)
        ax2.set_xlabel('读取 4 GiB 常驻权重时间下界（ms）', fontsize=11)
        ax2.set_title('B  跨平台读取 4 GiB 常驻权重的理想时间下界对比', loc='left', fontsize=12, weight='bold', pad=8)
        ax2.grid(axis='x', linestyle=':', alpha=0.5, color='#CBD5E1')
        out.save(f, 'ch04-tpu7x-dual-chiplet-vs-sram-asic')

        # 32. ch04-nvfp4-vs-mxfp4-microscaling-layout (Path D Fig 4-48)
        f, a = canvas(4.7)
        text(a, 0.50, 0.945, '微缩放低精度格式：OCP MXFP4 vs. NVIDIA NVFP4', 12.5, ha='center', weight='bold')
        card(a, 0.04, 0.53, 0.44, 0.34, 'OCP MXFP4 (4.25 bits)', [
            '32 个 FP4 (E2M1) = 16 B',
            '共享 1 个 E8M0 缩放 = 1 B',
            '有效位宽: 4 + 8/32 = 4.25b',
            '2 的幂次移位对齐，硬件极简',
        ], 'blue', body_size=11)
        card(a, 0.52, 0.53, 0.44, 0.34, 'NVIDIA NVFP4 (4.50 bits)', [
            '16 个 FP4 (E2M1) = 8 B',
            '共享 1 个 E4M3 缩放 = 1 B',
            '外加张量级 FP32 二级缩放',
            '有效位宽: 4.50b，信噪比更高',
        ], 'orange', body_size=11)
        box(a, 0.04, 0.28, 0.28, 0.16, 'HBM 紧凑块读取\n(17B/32 或 9B/16)', 'teal', 11, weight='bold')
        box(a, 0.36, 0.28, 0.28, 0.16, 'Tensor Core 前端\n硬件解包与块缩放', 'purple', 11, weight='bold')
        box(a, 0.68, 0.28, 0.28, 0.16, 'FP32 累加器合并\n+ 二级缩放输出', 'green', 11, weight='bold')
        arrow(a, (0.322, 0.36), (0.356, 0.36), 'teal')
        arrow(a, (0.642, 0.36), (0.676, 0.36), 'purple')
        box(a, 0.04, 0.05, 0.92, 0.16,
            '70B 权重: BF16 140 GB → NVFP4 39.38 GB → MXFP4 37.19 GB\n'
            '671B 权重: BF16 1,342 GB → NVFP4 377.44 GB → MXFP4 356.47 GB',
            'gray', 11, weight='bold')
        out.save(f, 'ch04-nvfp4-vs-mxfp4-microscaling-layout')

        # 33. ch04-ascend950-davinci-and-trainium-sbuf (Path D Fig 4-49)
        f, a = canvas(4.8)
        text(a, 0.50, 0.945, '昇腾 950 第三代达芬奇架构 vs. AWS Trainium2/3 片上存储', 12.5, ha='center', weight='bold')
        card(a, 0.04, 0.44, 0.44, 0.43, '昇腾 950 AI Core (Table 4-2)', [
            'L1 Buffer: 512 KB | UB: 512 KB',
            'L0A: 64 KB | L0B: 64 KB',
            'L0C 矩阵累加器: 256 KB',
            '全芯片 L2: 128 MB (512B 行)',
            'CV 直连 + BufferID + NDDMA',
        ], 'orange', body_size=11)
        card(a, 0.52, 0.44, 0.44, 0.43, 'Trainium2 / 3 NeuronCore', [
            'Trn2 (v3): 28 MiB SBUF/核',
            'Trn3 (v4): 32 MiB SBUF/核',
            '每核独立 2 MiB PSUM 累加存',
            '全芯片 224 / 256 MiB SBUF',
            'NKI 显式 DMA，零硬件标签税',
        ], 'blue', body_size=11)
        box(a, 0.04, 0.22, 0.44, 0.15, '950DT: 144GB HiZQ (4 TB/s)\nMXFP4 2.0 / HiF8 1.0 PFLOPS', 'green', 11, weight='bold')
        box(a, 0.52, 0.22, 0.44, 0.15, 'Trn3: 144GiB HBM3e (4.9 TB/s)\nMXFP8/4 2.52 PFLOPS (8核)', 'teal', 11, weight='bold')
        box(a, 0.04, 0.04, 0.92, 0.13,
            '昇腾 950: 多级专用缓冲 (L1/L0/UB) + BufferID 细粒度流水\n'
            'Trainium2/3: 每核 28–32 MiB 软件管理 SBUF 驻留大块状态',
            'gray', 11, weight='bold')
        out.save(f, 'ch04-ascend950-davinci-and-trainium-sbuf')

        # 34. ch04-hbm4-2048bit-logic-base-die-and-vera-rubin (Path E3 Fig 4-50)
        f, a = canvas(5.2)
        text(a, 0.50, 0.95, 'HBM3e (1024-bit) → HBM4 (2048-bit) 与 Vera Rubin 架构', 12.2, ha='center', weight='bold')
        card(a, 0.04, 0.52, 0.44, 0.37, 'HBM3e (B200: 192 GB)', [
            '单堆位宽: 1024-bit (16 通道)',
            'Base Die: DRAM 无源直通',
            '8 堆总带宽: 8.0 TB/s',
            'FP8/4 嵴点: 562.5 / 1,125',
        ], 'blue', body_size=11)
        card(a, 0.52, 0.52, 0.44, 0.37, 'HBM4 (Rubin: 288 GB)', [
            '单堆位宽: 2048-bit (32 通道)',
            'Base Die: TSMC 逻辑工艺',
            '8 堆总带宽: 22.0 TB/s (2.75×)',
            'FP8/4 嵴点: 1,591 / 2,273',
        ], 'orange', body_size=11)
        box(a, 0.04, 0.27, 0.28, 0.17, 'Vera CPU (88 核)\n1.5TB LPDDR5X', 'teal', 11, weight='bold')
        box(a, 0.36, 0.27, 0.28, 0.17, 'NVLink-C2C\n双向 1.8 TB/s', 'purple', 11, weight='bold')
        box(a, 0.68, 0.27, 0.28, 0.17, 'Rubin GPU\n50 PFLOPS NVFP4', 'green', 11, weight='bold')
        arrow(a, (0.322, 0.355), (0.356, 0.355), 'teal')
        arrow(a, (0.642, 0.355), (0.676, 0.355), 'purple')
        box(a, 0.04, 0.05, 0.92, 0.15,
            '235B-A22B (NVFP4 激活 12.38 GB): B200 1.55 ms → Rubin 0.56 ms\n'
            'VR200 NVL72: 72 GPU | 3.6 EFLOPS NVFP4 | 1.58 PB/s HBM4',
            'gray', 11, weight='bold')
        out.save(f, 'ch04-hbm4-2048bit-logic-base-die-and-vera-rubin')

        # 35. ch04-sram-asic-vs-hbm-gpu-heterogeneous-pd-serving (Path E3 Fig 4-51)
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(420 / 72, 6.6))
        f.subplots_adjust(left=0.16, right=0.95, bottom=0.10, top=0.92, hspace=0.52)
        for ax in (ax1, ax2):
            ax.spines[['top', 'right']].set_visible(False)

        chips = [
            ('HGX B200 (HBM3e)', 192.0, 8.0, 'blue', (0.72, 0.65), 'right'),
            ('Rubin GPU (HBM4)', 288.0, 22.0, 'teal', (0.72, 1.25), 'right'),
            ('Groq LPU (SRAM)', 0.23, 80.0, 'orange', (1.35, 0.55), 'left'),
            ('d-Matrix Corsair', 2.0, 150.0, 'purple', (1.35, 1.35), 'left'),
            ('Cerebras WSE-3', 44.0, 21000.0, 'green', (0.72, 0.58), 'right'),
        ]
        for label, cap_gb, bw_tbs, col, (fx, fy), ha_align in chips:
            ax1.scatter([cap_gb], [bw_tbs], s=70, facecolor=COL[col], edgecolor=BORDER_COL[col], linewidth=1.6, zorder=5)
            ax1.text(cap_gb * fx, bw_tbs * fy, f'{label}\n({cap_gb:g} GB, {bw_tbs:g} TB/s)',
                     ha=ha_align, va='center', fontsize=11, weight='bold', color=TEXT_COL[col])
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set(xlim=(0.12, 850.0), ylim=(2.5, 68000.0),
                xlabel='单芯片/单晶圆高速存储容量（GB，对数坐标）',
                ylabel='片内/显存带宽（TB/s）')
        ax1.set_title('A  HBM GPU 与纯 SRAM 推理 ASIC 的容量—带宽帕累托边界', loc='left', fontsize=12, weight='bold', pad=8)
        ax1.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1')

        # Exact numbers locked to 算例 4-5 & 实验 4-18 (120 req/s, Lin=4096, Lout=512, 61,440 tok/s)
        configs = [
            '同构 HBM GPU\n(TPOT≤2.5ms, 15台)',
            '异构 GPU+SRAM\n(3 GPU + 4 SRAM)',
            '同构 HBM GPU\n(TPOT≤10ms, 5台)',
        ]
        cost_per_m = [1.63, 0.98, 0.54]
        sub_labels = ['360美元/h (B=16)', '216美元/h (省40%)', '120美元/h (B=496)']
        x_c = np.arange(len(configs))
        cols_c = ['orange', 'green', 'blue']
        for xi, cv, sl, c_name in zip(x_c, cost_per_m, sub_labels, cols_c):
            ax2.bar(xi, cv, width=0.44, color=COL[c_name], edgecolor=BORDER_COL[c_name], linewidth=1.3)
            ax2.text(xi, cv + 0.06, f'{cv:.2f} 美元/M\n{sl}', ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL[c_name])
        ax2.set_xticks(x_c, configs)
        ax2.set_xlim(-0.55, 2.55)
        ax2.set_ylim(0, 2.25)
        ax2.set_ylabel('成本（美元/M tokens）', fontsize=11)
        ax2.set_title('B  算例 4-5：120 req/s 下同构与异构 PD 分离成本对比', loc='left', fontsize=12, weight='bold', pad=8)
        ax2.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E1')
        out.save(f, 'ch04-sram-asic-vs-hbm-gpu-heterogeneous-pd-serving')

        # 36. ch04-nvfp4-mxfp4-two-level-microscaling-tensor-core-datapath (Path F3 Fig 4-52)
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(420 / 72, 6.8))
        f.subplots_adjust(left=0.16, right=0.95, bottom=0.10, top=0.92, hspace=0.52)
        for ax in (ax1, ax2):
            ax.spines[['top', 'right']].set_visible(False)

        mult_labels = ['1.0× (高斯)', '2.0× 离群', '4.0× 离群', '6.0× 离群', '8.0× 离群']
        sqnr_i4 = [18.80, 17.63, 13.46, 11.58, 10.66]
        sqnr_mx = [18.93, 18.64, 17.00, 15.63, 15.00]
        sqnr_nv = [20.46, 20.45, 20.84, 20.70, 20.74]
        x_m = np.arange(len(mult_labels))

        ax1.plot(x_m, sqnr_nv, color=BORDER_COL['green'], lw=2.4, marker='o', ms=5.5, label='NVFP4 (g=16, E4M3+FP32)')
        ax1.plot(x_m, sqnr_mx, color=BORDER_COL['blue'], lw=2.2, marker='s', ms=5.2, label='OCP MXFP4 (g=32, E8M0)')
        ax1.plot(x_m, sqnr_i4, color=BORDER_COL['orange'], lw=2.1, ls='--', marker='^', ms=5.0, label='INT4 对称块量化 (g=128)')
        for xi, v_nv, v_mx in zip(x_m, sqnr_nv, sqnr_mx):
            ax1.text(xi, v_nv + 0.55, f'{v_nv:.2f} dB', ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL['green'])
            mx_dy = -1.35 if xi == 0 else 0.50
            mx_va = 'top' if xi == 0 else 'bottom'
            ax1.text(xi, v_mx + mx_dy, f'{v_mx:.2f} dB', ha='center', va=mx_va, fontsize=11, weight='bold', color=TEXT_COL['blue'])
        ax1.set_xticks(x_m, mult_labels)
        ax1.set_xlim(-0.45, 4.45)
        ax1.set_ylim(5.5, 25.5)
        ax1.set_ylabel('量化信噪比 SQNR（dB）', fontsize=11)
        ax1.set_title('A  算例 4-6：离群通道下 INT4、MXFP4 与 NVFP4 信噪比对比', loc='left', fontsize=12, weight='bold', pad=8)
        ax1.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1')
        ax1.legend(frameon=True, loc='lower left', fontsize=11)

        stages = ['1. 16/32元\nE2M1内积树', '2. 微块缩放\n(E8M0/E4M3)', '3. TMEM/FP32\n主累加器', '4. 张量级FP32\n二级重缩放']
        eff_precision = [14.0, 18.9, 20.7, 23.5]
        x_s = np.arange(len(stages))
        cols_s = ['blue', 'teal', 'green', 'purple']
        sub_s = ['4b内积树', '微块缩放', 'FP32累加', '二级缩放']
        for xi, ep, sl, cn in zip(x_s, eff_precision, sub_s, cols_s):
            ax2.bar(xi, ep, width=0.45, color=COL[cn], edgecolor=BORDER_COL[cn], lw=1.3)
            ax2.text(xi, ep + 0.6, f'{ep:.1f} dB\n{sl}', ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL[cn])
        ax2.set_xticks(x_s, stages)
        ax2.set_xlim(-0.55, 3.55)
        ax2.set_ylim(0, 31.0)
        ax2.set_ylabel('等效动态范围（dB）', fontsize=11)
        ax2.set_title('B  第五代/第六代 Tensor Core 双级微缩放点积硬件数据通路', loc='left', fontsize=12, weight='bold', pad=8)
        ax2.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E1')
        out.save(f, 'ch04-nvfp4-mxfp4-two-level-microscaling-tensor-core-datapath')

        # 37. ch04-cxl31-memory-pooling-fabric-queueing-latency-model (Path F3 Fig 4-53)
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(420 / 72, 6.8))
        f.subplots_adjust(left=0.16, right=0.95, bottom=0.10, top=0.92, hspace=0.52)
        for ax in (ax1, ax2):
            ax.spines[['top', 'right']].set_visible(False)

        rho_grid = np.linspace(0.05, 0.95, 180)
        t_cxl_switch = 245.0 + 45.0 * (rho_grid / (1.0 - rho_grid))
        t_cxl_direct = 175.0 + 35.0 * (rho_grid / (1.0 - rho_grid))
        ax1.plot(rho_grid * 100.0, t_cxl_switch, color=BORDER_COL['orange'], lw=2.4, label='CXL 3.1 交换机池 (245ns)')
        ax1.plot(rho_grid * 100.0, t_cxl_direct, color=BORDER_COL['blue'], lw=2.2, ls='--', label='CXL 3.1 直连卡 (175ns)')
        ax1.axhline(85.0, color=BORDER_COL['green'], lw=1.8, ls=':', label='本地 DDR5 基线 (85ns)')
        ax1.scatter([70.0, 90.0], [350.0, 650.0], s=58,
                    color=[BORDER_COL['teal'], BORDER_COL['orange']],
                    facecolor=[COL['teal'], COL['orange']], linewidth=1.6, zorder=5)
        ax1.annotate('甜点区 ρ=70%:\nT = 350.0 ns',
                     xy=(70.0, 350.0), xytext=(32.0, 440.0),
                     arrowprops=dict(arrowstyle='->', color=BORDER_COL['teal'], lw=1.3),
                     fontsize=11, weight='bold', color=TEXT_COL['teal'])
        ax1.annotate('排队发散 ρ=90%:\nT = 650.0 ns',
                     xy=(90.0, 650.0), xytext=(55.0, 810.0),
                     arrowprops=dict(arrowstyle='->', color=BORDER_COL['orange'], lw=1.3),
                     fontsize=11, weight='bold', color=TEXT_COL['orange'])
        ax1.set(xlim=(0, 100), ylim=(0, 1180),
                xlabel='CXL 3.1 端口带宽利用率 ρ（%）',
                ylabel='端到端访存延迟（ns）')
        ax1.set_title('A  CXL 3.1 交换机内存池在突发 KV 访存下的 M/G/1 排队延迟曲线', loc='left', fontsize=12, weight='bold', pad=8)
        ax1.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1')
        ax1.legend(frameon=True, loc='upper left', fontsize=11)

        tiers = ['全量 HBM\n(64.4 GB)', '2%冷页召回\n(HBM掩盖)', '5%冷页召回\n(CXL受限)', '全量 CXL\n(无热过滤)']
        t_step_vals = [8.05, 3.55, 7.65, 153.06]
        x_t = np.arange(len(tiers))
        cols_t = ['blue', 'green', 'teal', 'orange']
        for xi, tv, cn in zip(x_t, t_step_vals, cols_t):
            ax2.bar(xi, tv, width=0.45, color=COL[cn], edgecolor=BORDER_COL[cn], lw=1.3)
            ax2.text(xi, tv * 1.15, f'{tv:.2f} ms', ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL[cn])
        ax2.set_yscale('log')
        ax2.set_xticks(x_t, tiers)
        ax2.set_xlim(-0.55, 3.55)
        ax2.set_ylim(1.5, 420.0)
        ax2.set_ylabel('单步 Decode 耗时（ms，对数）', fontsize=11)
        ax2.set_title('B  算例 4-7：128K 上下文 HBM + CXL 3.1 分层解码耗时', loc='left', fontsize=12, weight='bold', pad=8)
        ax2.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1')
        out.save(f, 'ch04-cxl31-memory-pooling-fabric-queueing-latency-model')

    from sync_figures import draw as draw_sync
    data['4-sync'] = draw_sync(out, root)
    from core_principles_figures import draw as draw_principles
    draw_principles(4, out)
    from energy_physics import draw as draw_energy
    data['4-energy'] = draw_energy(out)
    from h2_accelerators_figures import draw as draw_h2
    data['4-h2'] = draw_h2(out)
    return out.finish()
