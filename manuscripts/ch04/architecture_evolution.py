"""Model-led cross-generation diagrams; all quantitative labels use calculate.py."""
from pathlib import Path
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure_style import STYLE, COL, BORDER_COL, TEXT_COL, canvas, plot, text, box, card, badge, arrow, Exporter
from figure_style.typography import configure_font

ROOT = Path(__file__).resolve().parents[2]


def draw(here):
    r = json.loads((ROOT / 'calculations/research/architecture-evolution-quantitative/result.json').read_text())
    out = Exporter(here)
    with plt.rc_context(STYLE):
        # 1. figure-4-evolution-tensor-budget
        f, axs = plt.subplots(2, 1, figsize=(420 / 72, 4.7))
        f.subplots_adjust(left=0.26, right=0.88, hspace=0.92, top=0.90, bottom=0.13)
        for a, m, unit, factor in [(axs[0], 4096, 'ms', 1), (axs[1], 1, 'μs', 1000)]:
            rows = [v for v in r['volta'] if v['rows'] == m]
            vals = [v['compute_ms'] * factor for v in rows] + [rows[0]['memory_ms'] * factor]
            cols = ['orange', 'green', 'blue']
            for idx, (v, c) in enumerate(zip(vals, cols)):
                a.barh(idx, v, color=COL[c], edgecolor=BORDER_COL[c], linewidth=1.2, height=0.55)
            a.set_yticks(range(3), ['普通 FP32', 'Tensor Core', '数据传输'])
            a.invert_yaxis()
            a.set_xlim(0, max(vals) * 1.25)
            a.set_xlabel('资源服务时间（' + unit + '）')
            a.set_title(f'{m:,} 行输入', fontsize=12.5, weight='bold', loc='left')
            a.grid(axis='x', linestyle=':', alpha=0.45, color='#CBD5E1')
            a.spines[['top', 'right']].set_visible(False)
            for i, v in enumerate(vals):
                a.text(v + max(vals) * 0.035, i, f'{v:.3g}', va='center', fontsize=11, weight='bold', color=COL['ink'])
        out.save(f, 'figure-4-evolution-tensor-budget')

        # 2. figure-4-evolution-precision
        f, a = plot(3.8, left=0.25, bottom=0.20)
        for i, key, label, c in [(0, 'read_us_4090', 'RTX 4090', 'blue'), (1, 'read_us_5090', 'RTX 5090', 'orange')]:
            vals = [v[key] for v in r['formats']]
            ys = [j + (i - 0.5) * 0.34 for j in range(3)]
            a.barh(ys, vals, height=0.30, color=COL[c], edgecolor=BORDER_COL[c], linewidth=1.2, label=label)
            for y, v in zip(ys, vals):
                a.text(v + 0.6, y, f'{v:.2f}', va='center', fontsize=11, weight='bold', color=COL['ink'])
        a.set_yticks(range(3), [f"{z['format']}\n{z['MiB']:g} MiB" for z in r['formats']])
        a.invert_yaxis()
        a.set_xlim(0, 42)
        a.set_xlabel('同一权重读取时间（μs）')
        a.grid(axis='x', linestyle=':', alpha=0.45, color='#CBD5E1')
        a.legend(frameon=True, loc='lower right', fontsize=11)
        out.save(f, 'figure-4-evolution-precision')

        # 3. figure-4-evolution-nvidia-path
        f, a = canvas(4.5)
        for y, label, loader, acc in [
            (0.72, 'Ampere', '异步拷贝', '寄存器'),
            (0.42, 'Hopper', 'TMA 搬移', '寄存器'),
            (0.12, 'Blackwell SM100', 'TMA 搬移', 'TMEM'),
        ]:
            text(a, 0.03, y + 0.20, label, 12.5, weight='bold')
            for x, w, t, c in [
                (0.03, 0.20, loader, 'blue'),
                (0.29, 0.20, '共享内存', 'green'),
                (0.55, 0.19, '矩阵单元', 'orange'),
                (0.80, 0.18, acc, 'purple'),
            ]:
                box(a, x, y, w, 0.14, t, c, 11, weight='bold')
            for x1, x2 in [(0.235, 0.285), (0.495, 0.545), (0.745, 0.795)]:
                arrow(a, (x1, y + 0.07), (x2, y + 0.07), 'blue')
        text(a, 0.50, 0.02, '输入块 32 KiB；输出累加状态 64 KiB', 11, ha='center', weight='bold', color=COL['muted'])
        out.save(f, 'figure-4-evolution-nvidia-path')

        # 4. figure-4-evolution-img2col
        c = r['conv']
        f, a = canvas(4.4)
        text(a, 0.03, 0.95, '56 × 56 × 64 输入，3 × 3 卷积', 12.5, weight='bold')
        box(a, 0.03, 0.73, 0.30, 0.14, f"输入\n{c['input_bytes']/2**20:.3f} MiB", 'blue', 11, weight='bold')
        box(a, 0.45, 0.70, 0.52, 0.20, f"HBM 展开矩阵\n{c['expanded_bytes']/2**20:.3f} MiB（9 倍）", 'purple', 11, weight='bold')
        arrow(a, (0.335, 0.80), (0.445, 0.80), 'blue')
        text(a, 0.39, 0.88, '展开', 11, ha='center', weight='bold')
        box(a, 0.65, 0.49, 0.32, 0.12, '矩阵乘法', 'orange', 11, weight='bold')
        arrow(a, (0.81, 0.695), (0.81, 0.615), 'purple')
        text(a, 0.03, 0.58, '显式展开：先写出至 HBM，再读入计算', 11, weight='bold', color=COL['muted'])
        text(a, 0.03, 0.34, 'MTE 在片上实时组织窗口', 12.5, weight='bold')
        for x, label, color in [(0.03, '原始输入', 'blue'), (0.37, 'img2col', 'green'), (0.71, 'Cube', 'orange')]:
            box(a, x, 0.16, 0.26, 0.12, label, color, 11, weight='bold')
        arrow(a, (0.295, 0.22), (0.365, 0.22), 'blue')
        arrow(a, (0.635, 0.22), (0.705, 0.22), 'green')
        text(a, 0.50, 0.045, f"省去 HBM 写读 {c['explicit_expansion_roundtrip_bytes']/2**20:.3f} MiB ≈ {c['saved_hbm_service_us_at_1_2TBs']:.2f} μs", 11, ha='center', weight='bold', color=TEXT_COL['green'])
        out.save(f, 'figure-4-evolution-img2col')

        # 5. figure-4-evolution-ascend
        f, a = canvas(4.6)
        text(a, 0.03, 0.95, '910A：同核分工，MTE 支持 img2col', 12.5, weight='bold')
        box(a, 0.03, 0.73, 0.94, 0.15, '', 'gray')
        for x, t, c in [(0.06, 'Cube', 'orange'), (0.40, '局部缓冲', 'blue'), (0.74, 'Vector', 'green')]:
            box(a, x, 0.75, 0.20, 0.11, t, c, 11, weight='bold')
        arrow(a, (0.265, 0.805), (0.395, 0.805), 'orange')
        arrow(a, (0.605, 0.805), (0.735, 0.805), 'blue')
        text(a, 0.03, 0.65, '910B / 910C：矩阵、向量独立控制', 12.5, weight='bold')
        for x, w, t, c in [(0.03, 0.23, 'AIC', 'orange'), (0.34, 0.32, '全局地址交换\n经过缓存层次', 'blue'), (0.74, 0.23, 'AIV', 'green')]:
            box(a, x, 0.43, w, 0.15, t, c, 11, weight='bold')
        arrow(a, (0.265, 0.505), (0.335, 0.505), 'orange')
        arrow(a, (0.665, 0.505), (0.735, 0.505), 'blue')
        text(a, 0.03, 0.34, '950：增强 Vector，增加 CV 直连', 12.5, weight='bold')
        box(a, 0.03, 0.11, 0.34, 0.15, 'Cube Core\nL1 Buffer', 'orange', 11, weight='bold')
        box(a, 0.63, 0.11, 0.34, 0.15, 'Vector Core\nUB + 寄存器', 'green', 11, weight='bold')
        arrow(a, (0.375, 0.215), (0.625, 0.215), 'orange')
        arrow(a, (0.625, 0.155), (0.375, 0.155), 'green')
        text(a, 0.50, 0.28, 'CV 直连', 11, ha='center', weight='bold', color=TEXT_COL['orange'])
        text(a, 0.50, 0.035, '矩阵与向量之间的交接留在更近的片上通路', 11, ha='center', weight='bold', color=COL['muted'])
        out.save(f, 'figure-4-evolution-ascend')

        # 6. figure-4-evolution-cv-budget
        f, a = canvas(4.4)
        text(a, 0.03, 0.94, 'QK → Softmax → PV：两次交接', 12.5, weight='bold')
        box(a, 0.03, 0.70, 0.26, 0.15, 'Cube\n64 KiB 分数', 'orange', 11, weight='bold')
        box(a, 0.71, 0.70, 0.26, 0.15, 'Vector\n64 KiB 概率', 'green', 11, weight='bold')
        box(a, 0.35, 0.66, 0.30, 0.22, '外层接口\n写 + 读 × 2\n256 KiB', 'blue', 11, weight='bold')
        arrow(a, (0.295, 0.79), (0.345, 0.79), 'orange')
        arrow(a, (0.655, 0.79), (0.705, 0.79), 'blue')
        arrow(a, (0.705, 0.73), (0.655, 0.70), 'green')
        arrow(a, (0.345, 0.70), (0.295, 0.73), 'blue')
        text(a, 0.50, 0.56, '94 GB/s 时占用外层接口约 2.79 μs', 11, ha='center', weight='bold', color=TEXT_COL['blue'])
        box(a, 0.03, 0.28, 0.29, 0.14, 'Cube L1', 'orange', 11.5, weight='bold')
        box(a, 0.68, 0.28, 0.29, 0.14, 'Vector UB', 'green', 11.5, weight='bold')
        arrow(a, (0.325, 0.385), (0.675, 0.385), 'orange')
        arrow(a, (0.675, 0.31), (0.325, 0.31), 'green')
        text(a, 0.50, 0.465, 'CV 直通：64 KiB + 64 KiB', 11, ha='center', weight='bold', color=TEXT_COL['orange'])
        text(a, 0.50, 0.19, '要在 1.024 μs 内交接，CV 需 128 GB/s', 11, ha='center', weight='bold')
        text(a, 0.50, 0.065, '外层接口释放 256 KiB；直接通路承担 128 KiB', 11, ha='center', weight='bold', color=COL['muted'])
        out.save(f, 'figure-4-evolution-cv-budget')

        # 7. figure-4-evolution-apple
        f, a = canvas(4.7)
        box(a, 0.03, 0.85, 0.94, 0.12, '统一内存：权重、KV 与中间数据共享物理池', 'blue', 11.5, weight='bold')
        for x, w, label, c in [
            (0.03, 0.19, 'CPU', 'gray'),
            (0.28, 0.39, 'GPU', 'green'),
            (0.73, 0.24, '独立 Neural\nEngine', 'orange'),
        ]:
            box(a, x, 0.62, w, 0.15, label, c, 11, weight='bold')
            arrow(a, (x + w / 2, 0.845), (x + w / 2, 0.775), 'blue')
        text(a, 0.03, 0.53, 'GPU 内部的代际演进', 12.5, weight='bold')
        for y, gen, left_txt, right_txt in [
            (0.32, 'M3 / M4', '通用计算阵列', 'Dynamic\nCaching'),
            (0.09, 'M5', '通用计算 +\nNeural Accelerator', '第二代\nDynamic Caching'),
        ]:
            text(a, 0.03, y + 0.085, gen, 11.5, weight='bold')
            box(a, 0.24, y, 0.42, 0.17, left_txt, 'orange' if gen == 'M5' else 'green', 11, weight='bold')
            box(a, 0.71, y, 0.26, 0.17, right_txt, 'blue', 11, weight='bold')
            arrow(a, (0.705, y + 0.085), (0.665, y + 0.085), 'blue')
        out.save(f, 'figure-4-evolution-apple')

        # 8. figure-4-evolution-convergence
        f, axs = plt.subplots(2, 1, figsize=(420 / 72, 5.3))
        f.subplots_adjust(left=0.25, right=0.90, top=0.91, bottom=0.10, hspace=0.88)
        ids = ['rtx3090', 'rtx4090', 'rtx5090', 'a100-80gb-sxm', 'h100-sxm', 'm3-ultra-80gpu-256gb']
        labels = ['RTX 3090', 'RTX 4090', 'RTX 5090', 'A100 SXM', 'H100 SXM', 'M3 Ultra']
        vals = [next(x['read_ms'] for x in r['decode'] if x['device'] == id and x['workload'] == 'qwen3-8b-b1-h8191-w16-balanced') for id in ids]
        for a, names, values, title, xlabel, col in [
            (axs[0], labels, vals, '8K 单请求 decode', '主要载荷读取下界（ms）', 'blue'),
            (axs[1], labels[:-1], [next(x['prefill_matrix_ms'] for x in r['qwen_compute'] if x['device'] == id) for id in ids[:-1]], '4K 输入 prefill', '矩阵计算下界（ms）', 'orange'),
        ]:
            a.barh(range(len(values)), values, color=COL[col], edgecolor=BORDER_COL[col], linewidth=1.2, height=0.55)
            a.set_yticks(range(len(values)), names)
            a.invert_yaxis()
            a.set_xlim(0, max(values) * 1.25)
            a.set_xlabel(xlabel)
            a.set_title(title, fontsize=12.5, weight='bold', loc='left')
            a.grid(axis='x', linestyle=':', alpha=0.45, color='#CBD5E1')
            a.spines[['top', 'right']].set_visible(False)
            for y, v in enumerate(values):
                a.text(v + max(values) * 0.03, y, f'{v:.1f}', va='center', fontsize=11, weight='bold', color=COL['ink'])
        out.save(f, 'figure-4-evolution-convergence')
    (Path(here) / 'evolution-layout-validation.json').write_text(json.dumps(out.checks, ensure_ascii=False, indent=2) + '\n')
    return out.outputs


if __name__ == '__main__':
    _, family = configure_font()
    plt.rcParams['font.family'] = [family, 'DejaVu Sans']
    draw(Path(__file__).resolve().parent)
