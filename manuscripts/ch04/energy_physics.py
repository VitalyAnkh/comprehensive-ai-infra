"""Section 4.1.3 chip physics and energy: four derivation figures at final book width."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from figure_style import COL, BORDER_COL, TEXT_COL, STYLE, canvas, plot, text, box, card, badge, arrow

ROOT = Path(__file__).resolve().parents[2]
H100_PEAK_BF16_FLOPS = 989.4e12
H100_HBM_BYTES_PER_S = 3350e9
H100_TDP_W = 700.0


def draw(out):
    led = json.loads((ROOT / 'calculations/results/energy-ledger-book.json').read_text())
    epb = led['energy_per_byte']
    ref = led['energy_per_byte_references']
    ladder = led['joules_per_token']['by_level_if_all_bytes_came_from_that_level']
    split = led['joules_per_token']['per_token_split_weights_kv_compute']
    pj_flop = led['joules_per_token']['compute']['pj_per_flop']
    v_ratio = led['voltage_scaling']['dynamic_power_ratio']
    p_ratio = float(led['power_density']['power_ratio'].split('/')[0]) / float(led['power_density']['power_ratio'].split('/')[1])
    a_ratio = float(led['power_density']['projected_area_ratio'].split('/')[0]) / float(led['power_density']['projected_area_ratio'].split('/')[1])
    d_ratio = led['power_density']['density_ratio']

    with plt.rc_context(STYLE):
        # Figure A: wire capacitance mechanism plus the two LogicFolding ratios.
        f = plt.figure(figsize=(420 / 72, 5.3))
        a = f.add_axes([0.03, 0.50, 0.94, 0.48])
        a.set(xlim=(0, 1), ylim=(0, 1))
        a.axis('off')
        text(a, 0.02, 0.95, '横向连线：数百 μm，电容沿长度累积', 12.5, weight='bold')
        box(a, 0.02, 0.60, 0.16, 0.16, '驱动门', 'green', 11, weight='bold')
        box(a, 0.82, 0.60, 0.16, 0.16, '接收门', 'green', 11, weight='bold')
        a.plot([0.18, 0.82], [0.68, 0.68], color='#475569', lw=1.8)
        for x in np.linspace(0.26, 0.74, 5):
            a.plot([x, x], [0.68, 0.56], color='#64748B', lw=1.1)
            a.plot([x - 0.03, x + 0.03], [0.56, 0.56], color='#475569', lw=1.5)
            a.plot([x - 0.03, x + 0.03], [0.51, 0.51], color='#475569', lw=1.5)
            a.plot([x, x], [0.51, 0.45], color='#64748B', lw=1.1)
            a.plot([x - 0.02, x + 0.02], [0.45, 0.45], color='#475569', lw=1.3)
        text(a, 0.50, 0.36, '每次 0→1 充电从电源取走 C·V² 的能量', 11.5, ha='center', weight='bold', color=COL['muted'])
        text(a, 0.02, 0.19, '折叠后：上下两层之间的垂直连接只有数 μm', 12.5, weight='bold')
        box(a, 0.28, 0.00, 0.19, 0.11, '上层', 'blue', 11, weight='bold')
        box(a, 0.53, 0.00, 0.19, 0.11, '下层', 'blue', 11, weight='bold')
        arrow(a, (0.475, 0.055), (0.525, 0.055), 'blue')
        b = f.add_axes([0.14, 0.12, 0.82, 0.32])
        labels = ['基线', '降压后\n动态功耗', '折叠后\n功耗', '折叠后\n投影面积', '折叠后\n功率密度']
        vals = [1.0, v_ratio, p_ratio, a_ratio, d_ratio]
        cols = ['gray', 'blue', 'green', 'orange', 'purple']
        for i, (v, c) in enumerate(zip(vals, cols)):
            b.bar(i, v, color=COL[c], edgecolor=BORDER_COL[c], linewidth=1.2, width=0.58)
            b.text(i, v + 0.04, f'{v:.2f}', ha='center', fontsize=11, weight='bold', color=COL['ink'])
        b.axhline(1.0, ls=':', color='#64748B', lw=1.1)
        b.set(xticks=range(5), xticklabels=labels, ylim=(0, 1.55), ylabel='相对基线')
        b.grid(axis='y', linestyle=':', alpha=0.45, color='#CBD5E1')
        b.spines[['top', 'right']].set_visible(False)
        out.save(f, 'figure-4-energy-wire')

        # Figure B: pJ/byte ladder (log scale) and one decode step in joules.
        f = plt.figure(figsize=(420 / 72, 5.7))
        a = f.add_axes([0.32, 0.60, 0.64, 0.35])
        rows = [
            ('KB 级局部 SRAM', epb['shared_l1']['pj_per_byte'], 'green'),
            ('NVLink-C2C 链路', epb['nvlink']['pj_per_byte'], 'purple'),
            ('MB 级片上 SRAM', epb['l2']['pj_per_byte'], 'teal'),
            ('HBM2', epb['hbm']['pj_per_byte'], 'blue'),
            ('LPDDR DRAM', ref['dram_lpddr_45nm']['pj_per_byte'], 'orange'),
        ]
        for y, (_, v, col) in enumerate(rows):
            a.barh(y, v, color=COL[col], edgecolor=BORDER_COL[col], linewidth=1.2, height=0.58)
            a.text(v * 1.15, y, f'{v:g}', va='center', fontsize=11, weight='bold', color=COL['ink'])
        a.set_xscale('log')
        a.set(xlim=(1, 420), yticks=range(len(rows)), yticklabels=[r[0] for r in rows], xlabel='每 byte 能耗（pJ，对数坐标）')
        a.invert_yaxis()
        a.grid(axis='x', linestyle=':', alpha=0.45, color='#CBD5E1')
        a.spines[['top', 'right']].set_visible(False)

        b = f.add_axes([0.32, 0.10, 0.64, 0.36])
        names = ['实际分账（HBM）', '若全部来自 KB 级 SRAM', '若全部来自 NVLink-C2C', '若全部来自 MB 级 SRAM', '若全部来自 HBM2']
        w, k, c = split['weight_joules'], split['kv_joules'], split['compute_joules']
        b.barh(0, w, color=COL['blue'], edgecolor=BORDER_COL['blue'], linewidth=1.2, height=0.58, label='权重读取')
        b.barh(0, k, left=w, color=COL['green'], edgecolor=BORDER_COL['green'], linewidth=1.2, height=0.58, label='KV 读取')
        b.barh(0, c, left=w + k, color=COL['orange'], edgecolor=BORDER_COL['orange'], linewidth=1.2, height=0.58, label='矩阵计算')
        b.text(w + k + c + 0.015, 0, f'{w + k + c:.3f} J', va='center', fontsize=11, weight='bold', color=COL['ink'])
        lv = [
            ladder['shared_l1']['weight_plus_kv_joules'],
            ladder['nvlink']['weight_plus_kv_joules'],
            ladder['l2']['weight_plus_kv_joules'],
            ladder['hbm']['weight_plus_kv_joules'],
        ]
        for y, v in zip(range(1, 5), lv):
            b.barh(y, v, color=COL['gray'], edgecolor=BORDER_COL['gray'], linewidth=1.2, height=0.58)
            b.text(v + 0.015, y, f'{v:.3f} J', va='center', fontsize=11, weight='bold', color=COL['ink'])
        b.set(xlim=(0, 0.74), yticks=range(5), yticklabels=names, xlabel='一步 decode 的能量（J，单请求、8K 上下文）')
        b.invert_yaxis()
        b.grid(axis='x', linestyle=':', alpha=0.45, color='#CBD5E1')
        b.spines[['top', 'right']].set_visible(False)
        b.legend(frameon=True, fontsize=11, loc='center right', bbox_to_anchor=(0.98, 0.55))
        out.save(f, 'figure-4-energy-ladder')

        # Figure C: package top view with two reticle-limited dies and eight HBM stacks on an interposer.
        f, a = canvas(4.6)
        text(a, 0.50, 0.95, '每颗 die 的面积受光罩单次曝光上限约束（H100：814 mm²）', 12, ha='center', weight='bold')
        box(a, 0.05, 0.13, 0.90, 0.74, '', 'gray')
        text(a, 0.07, 0.17, '中介层（interposer）', 11, weight='bold', color=COL['muted'])
        box(a, 0.31, 0.30, 0.18, 0.46, 'die 0', 'orange', 12, weight='bold')
        box(a, 0.51, 0.30, 0.18, 0.46, 'die 1', 'orange', 12, weight='bold')
        a.plot([0.49, 0.51], [0.53, 0.53], color=BORDER_COL['orange'], lw=2.4)
        text(a, 0.50, 0.81, 'die 间高速链路', 11, ha='center', weight='bold', color=TEXT_COL['orange'])
        for i in range(4):
            y = 0.24 + i * 0.14
            box(a, 0.08, y, 0.17, 0.11, 'HBM', 'blue', 11, weight='bold')
            box(a, 0.75, y, 0.17, 0.11, 'HBM', 'blue', 11, weight='bold')
            a.plot([0.25, 0.31], [y + 0.055, y + 0.055], color=BORDER_COL['blue'], lw=1.4)
            a.plot([0.69, 0.75], [y + 0.055, y + 0.055], color=BORDER_COL['blue'], lw=1.4)
        text(a, 0.50, 0.05, '每堆带宽 = 1024 引脚 × 引脚速率；每堆容量 = 层数 × 每层容量', 11, ha='center', color=COL['muted'], weight='bold')
        out.save(f, 'figure-4-energy-package')

        # Figure D: sustained-frequency ratio versus e/b.
        p_hbm = H100_HBM_BYTES_PER_S * epb['hbm']['pj_per_byte'] * 1e-12
        budget = H100_TDP_W - p_hbm
        b_pj = budget / H100_PEAK_BF16_FLOPS * 1e12
        x = np.linspace(1.0, 3.0, 200)
        example = 1.5
        f, a = plot(3.9, left=0.17, bottom=0.20)
        a.plot(x, 1 / x, color=BORDER_COL['blue'], lw=2.2, label='电压不变：P ∝ f，持续频率比 = b/e')
        a.plot(x, x ** (-1 / 3), color=BORDER_COL['orange'], lw=2.2, label='电压随频率下降：P ∝ f³，持续频率比 = (b/e)^(1/3)')
        a.axhline(1.0, ls=':', color='#64748B', lw=1.1)
        a.axvline(example, ls=':', color='#64748B', lw=1.1)
        for r, dy, col in [(1 / example, -16, 'blue'), (example ** (-1 / 3), 6, 'orange')]:
            a.plot(example, r, 'o', color=BORDER_COL[col], markersize=6)
            a.annotate(f'{r:.2f}', (example, r), xytext=(6, dy), textcoords='offset points', fontsize=11, weight='bold', color=TEXT_COL[col])
        a.text(1.85, 0.955, 'e ≤ b：峰值频率可持续', fontsize=11, weight='bold', color=COL['muted'])
        a.set(xlim=(1.0, 3.0), ylim=(0.3, 1.06), xlabel='实际每 FLOP 能耗 e ÷ 预算 b', ylabel='持续频率 ÷ 峰值频率')
        a.grid(True, linestyle=':', alpha=0.45, color='#CBD5E1')
        a.legend(frameon=True, fontsize=11, loc='lower left')
        out.save(f, 'figure-4-energy-power-cap')

    record = dict(
        dynamic_power_ratio=v_ratio,
        density_ratio=d_ratio,
        step_joules=w + k + c,
        ladder_joules=lv,
        hbm_full_bandwidth_w=p_hbm,
        compute_budget_w=budget,
        pj_per_flop_budget_tdp=H100_TDP_W / H100_PEAK_BF16_FLOPS * 1e12,
        pj_per_flop_budget_after_hbm=b_pj,
        reference_pj_per_flop_45nm=pj_flop,
        example_e_over_b=example,
        sustained_ratio_v_fixed_at_example=1 / example,
        sustained_ratio_v_scaled_at_example=example ** (-1 / 3),
        inputs=dict(
            ledger='calculations/results/energy-ledger-book.json',
            h100_peak_bf16_flops=H100_PEAK_BF16_FLOPS,
            h100_hbm_bytes_per_s=H100_HBM_BYTES_PER_S,
            h100_tdp_w=H100_TDP_W,
        ),
    )
    (Path(__file__).resolve().parent / 'energy-figure-data.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    return record
