"""Path H2 Ch08 figures: cleaned Fig 8-43 + QuaRot/SpinQuant Hadamard rotation (Fig 8-47) & PTQ Pareto taxonomy (Fig 8-48)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from figure_style import BORDER_COL, COL, STYLE, TEXT_COL, Exporter, arrow, box, canvas, text
from figure_style.typography import configure_font


def draw_clean_i10(out: Exporter) -> None:
    """Re-export Fig 8-43 with internal label standardized to Worked Example 8-4."""
    with plt.rc_context(STYLE):
        f, a = canvas(6.8)
        text(a, .02, .965, 'A. 算例 8-4：有损 KV 压缩“单步提速 20% 但输出膨胀 50%”端到端反转', 11.5)
        box(a, .015, .76, .46, .15, '无损 BF16 基线（TTFT=0.50 s）\n步速 v0 = 50 tok/s（20 ms/步）\n一步生成 L0 = 200 tok（199 步）', 'blue', 11)
        box(a, .525, .76, .46, .15, '无损基线端到端耗时 T0\nT0 = 0.50 + 199/50 = 4.480 s\n（L0/v0 连续近似 4.500 s）', 'green', 11)
        arrow(a, (.475, .835), (.525, .835))
        box(a, .015, .56, .46, .15, '有损 KV 压缩 / Token 驱逐\n步速 v_c = 60 tok/s（+20%）\n重复验算致 L_c = 300 tok（+50%）', 'orange', 11)
        box(a, .525, .56, .46, .15, '压缩后端到端延迟反转劣化\nT_c = 0.50 + 299/60 = 5.483 s\n单步快 20%，总耗时反慢 22.4%', 'purple', 11)
        arrow(a, (.755, .76), (.245, .71))
        arrow(a, (.475, .635), (.525, .635))
        a.plot([.015, .985], [.50, .50], color=COL['line'], lw=1.0, ls='--')
        text(a, .02, .455, 'B. 不发生延迟反转的临界输出长度上界与注意力流形保持校准准则', 11.5)
        box(a, .015, .25, .46, .15, '临界不退化不等式（无反转）\n(L_c - 1)/(L0 - 1) ≤ v_c / v0\n60 tok/s 下上限 L_max = 239.8', 'blue', 11)
        box(a, .525, .25, .46, .15, '为何有损压缩诱发思维链膨胀？\n驱逐推导锚点或破坏 Key 相位\n丢失前文结论被迫反复重算', 'orange', 11)
        box(a, .015, .04, .46, .15, '注意力 Sink 与滑窗强保护\n保留首部 Sink（前 4–16 tok）\n加最近滑窗（256 tok）防驱逐', 'purple', 11)
        box(a, .525, .04, .46, .15, 'RoPE 旋转前通道离群校准\n对 Key 离群通道做旋转前缩放\n长思维链输出膨胀率压至 <3%', 'green', 11)
        arrow(a, (.475, .325), (.525, .325))
        arrow(a, (.245, .25), (.245, .19))
        arrow(a, (.475, .115), (.525, .115))
        out.save(f, 'ch08-i10-kv-compression-length-inflation-paradox')


def draw(out: Exporter) -> dict:
    with plt.rc_context(STYLE):
        # Fig 8-47: ch08-smoothquant-vs-quarot-spinquant-hadamard-outlier-elimination
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(420 / 72, 7.6))
        f.subplots_adjust(left=0.14, right=0.96, bottom=0.09, top=0.93, hspace=0.56)
        for ax in (ax1, ax2):
            ax.spines[['top', 'right']].set_visible(False)

        # Panel A: Worked Example 8-8 (d=4096, x1=64.0, 4095 semantic channels N(0, 0.5^2))
        schemes_q = [
            '直接 INT4\n(Δ=9.14)',
            'SmoothQuant\n对角 W4A4',
            'QuaRot\nHadamard',
            'QuaRot\n+裁剪优化',
            'SpinQuant\n学习旋转',
        ]
        sqnr_sem = [0.0, 10.6, 19.7, 20.2, 21.5]
        sqnr_tot = [6.99, 11.8, 19.7, 20.2, 21.5]
        x_q = np.arange(len(schemes_q))
        w_q = 0.30

        ax1.bar(x_q - 0.17, sqnr_sem, width=w_q, color=COL['orange'], edgecolor=BORDER_COL['orange'], lw=1.2,
                label='4,095 个语义通道 SQNR (dB)')
        ax1.bar(x_q + 0.17, sqnr_tot, width=w_q, color=COL['green'], edgecolor=BORDER_COL['green'], lw=1.2,
                label='全向量 4,096 维总 SQNR (dB)')
        for xi, vs, vt in zip(x_q, sqnr_sem, sqnr_tot):
            vs_str = f'{vs:.1f}'
            vt_str = '6.99' if xi == 0 else f'{vt:.1f}'
            ax1.text(xi - 0.21, vs + 0.7, vs_str, ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL['orange'])
            ax1.text(xi + 0.21, vt + 0.7, vt_str, ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL['green'])

        ax1.annotate(
            '语义通道 100% 归零\n(x1=64 独占步长)',
            xy=(-0.17, 2.2),
            xytext=(-0.38, 23.6),
            arrowprops=dict(arrowstyle='->', color=BORDER_COL['orange'], lw=1.3),
            bbox=dict(boxstyle='round,pad=0.14', facecolor='white', edgecolor='none', alpha=0.90),
            fontsize=11,
            weight='bold',
            color=TEXT_COL['orange'],
        )
        ax1.set_xticks(x_q, schemes_q)
        ax1.set_xlim(-0.62, 4.62)
        ax1.set_ylim(0, 41.5)
        ax1.set_yticks([0, 6, 12, 18, 24])
        ax1.set_ylabel('量化信噪比 SQNR（dB）', fontsize=11)
        ax1.set_title('A  算例 8-8：单通道离群 (x1=64) 下 INT4 量化信噪比', loc='left', fontsize=12, weight='bold', pad=8)
        ax1.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E1')
        ax1.legend(frameon=True, loc='upper right', fontsize=11)

        # Panel B: Worked Example 8-9 (70B on single B200 192GB, S=8192)
        deploy_cfgs = [
            'BF16 基线\n(B=14 上限)',
            'FP8 基线\n(B=64)',
            'QuaRot 4bit\n(同批 B=64)',
            'QuaRot 4bit\n(并发 B=128)',
        ]
        t_weight = np.array([21.88, 11.02, 5.75, 5.75])
        t_kv = np.array([5.87, 13.76, 6.92, 13.84])
        t_other = np.array([1.45, 1.82, 1.42, 1.62])
        t_fwht = np.array([0.00, 0.00, 0.75, 1.82])
        t_sum = t_weight + t_kv + t_other + t_fwht
        x_d = np.arange(len(deploy_cfgs))
        w_d = 0.44

        ax2.bar(x_d, t_weight, width=w_d, color=COL['blue'], edgecolor=BORDER_COL['blue'], lw=1.1, label='权重读取')
        ax2.bar(x_d, t_kv, width=w_d, bottom=t_weight, color=COL['green'], edgecolor=BORDER_COL['green'], lw=1.1, label='8K KV 读取')
        ax2.bar(x_d, t_other, width=w_d, bottom=t_weight + t_kv, color=COL['gray'], edgecolor=BORDER_COL['gray'], lw=1.1, label='常规算子')
        mask_fwht = t_fwht > 0
        ax2.bar(
            x_d[mask_fwht],
            t_fwht[mask_fwht],
            width=w_d,
            bottom=(t_weight + t_kv + t_other)[mask_fwht],
            color=COL['purple'],
            edgecolor=BORDER_COL['purple'],
            lw=1.1,
            label='在线 FWHT (2×/层)',
        )

        tags = [
            '29.2 ms\n480 tok/s',
            '26.6 ms\n2,406 tok/s',
            '14.8 ms\n4,313 tok/s',
            '23.0 ms\n5,558 tok/s (2.31×)',
        ]
        tag_cols = ['gray', 'blue', 'teal', 'green']
        for xi, tot, tg, tc in zip(x_d, t_sum, tags, tag_cols):
            dx = -0.14 if xi == 3 else 0.0
            ax2.text(xi + dx, tot + 1.1, tg, ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL[tc])

        ax2.set_xticks(x_d, deploy_cfgs)
        ax2.set_xlim(-0.56, 3.68)
        ax2.set_ylim(0, 49.5)
        ax2.set_yticks([0, 10, 20, 30])
        ax2.set_ylabel('单步解码耗时（ms）', fontsize=11)
        ax2.set_title('B  算例 8-9：B200 部署 70B 的在线 FWHT 开销与净吞吐', loc='left', fontsize=12, weight='bold', pad=8)
        ax2.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E1')
        ax2.legend(frameon=True, loc='upper right', ncol=2, fontsize=11)
        out.save(f, 'ch08-smoothquant-vs-quarot-spinquant-hadamard-outlier-elimination')

        # Fig 8-48: ch08-ptq-taxonomy-awq-gptq-fp8-quarot-nvfp4-pareto-frontier
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(420 / 72, 7.6))
        f.subplots_adjust(left=0.14, right=0.96, bottom=0.09, top=0.93, hspace=0.56)
        for ax in (ax1, ax2):
            ax.spines[['top', 'right']].set_visible(False)

        # Panel A: PTQ Pareto Frontier (Throughput speedup vs FP8 on X, WikiText-2 delta PPL on Y)
        pts = [
            ('BF16 基线', 0.20, 0.00, 'gray', (0.14, 0.65)),
            ('FP8 (W8A8KV8)', 1.00, 0.06, 'blue', (0.58, 0.88)),
            ('AWQ / GPTQ\n(W4A16)', 1.38, 0.14, 'teal', (1.08, 1.52)),
            ('SmoothQuant\n(W4A4 退化)', 2.15, 2.85, 'orange', (1.76, 2.02)),
            ('QuaRot\n(W4A4KV4)', 2.31, 0.42, 'purple', (1.62, 1.08)),
            ('SpinQuant\n(W4A4KV4)', 2.31, 0.24, 'green', (2.34, 1.28)),
            ('NVFP4+旋转', 2.68, 0.12, 'green', (2.36, 0.66)),
        ]
        ax1.plot([0.20, 1.00, 2.68], [0.00, 0.06, 0.12], ls='--', lw=1.4, color='#64748B', label='硬件—算法联合帕累托前沿')

        for name, px, py, ck, (tx, ty) in pts:
            ax1.scatter([px], [py], s=72, facecolor=COL[ck], edgecolor=BORDER_COL[ck], lw=1.7, zorder=5)
            ax1.annotate(
                name,
                xy=(px, py),
                xytext=(tx, ty),
                arrowprops=dict(arrowstyle='->', color=BORDER_COL[ck], lw=1.2),
                bbox=dict(boxstyle='round,pad=0.10', facecolor='white', edgecolor='none', alpha=0.88),
                fontsize=11,
                weight='bold',
                color=TEXT_COL[ck],
            )
        ax1.set_xlim(0.05, 3.18)
        ax1.set_ylim(-0.22, 3.55)
        ax1.set_xlabel('70B 相对 FP8 基线的端到端解码吞吐倍数（×）', fontsize=11)
        ax1.set_ylabel('WikiText-2 困惑度增量 ΔPPL', fontsize=11)
        ax1.set_title('A  后训练量化（PTQ）统一谱系的精度—吞吐帕累托前沿', loc='left', fontsize=12, weight='bold', pad=8)
        ax1.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1')
        ax1.legend(frameon=True, loc='upper left', fontsize=11)

        # Panel B: Memory footprint & Max 8K concurrent batch size on single B200 (180 GB usable pool)
        tax_names = [
            'BF16\n16-bit',
            'FP8\nW8A8KV8',
            'AWQ/GPTQ\nW4A16KV16',
            'AWQ+FP8\nW4A16KV8',
            'QuaRot\nW4A4KV4',
        ]
        w_gb = np.array([140.0, 70.5, 36.8, 36.8, 36.8])
        kv_pool_gb = 180.0 - w_gb
        max_reqs = [14, 79, 53, 104, 206]
        x_t = np.arange(len(tax_names))
        w_t = 0.44

        ax2.bar(x_t, w_gb, width=w_t, color=COL['orange'], edgecolor=BORDER_COL['orange'], lw=1.1, label='70B 权重常驻显存 (GB)')
        ax2.bar(x_t, kv_pool_gb, width=w_t, bottom=w_gb, color=COL['green'], edgecolor=BORDER_COL['green'], lw=1.1, label='可用 KV 缓存池 (GB)')
        for xi, mr in zip(x_t, max_reqs):
            ck = 'green' if mr >= 100 else ('blue' if mr >= 70 else 'orange')
            tok_m = mr * 8192 / 1e6
            ax2.text(xi, 185.0, f'{mr} 并发\n({tok_m:.2f}M)', ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL[ck])

        ax2.set_xticks(x_t, tax_names)
        ax2.set_xlim(-0.58, 4.58)
        ax2.set_ylim(0, 298.0)
        ax2.set_yticks([0, 60, 120, 180])
        ax2.set_ylabel('单卡可用显存切分（GB）', fontsize=11)
        ax2.set_title('B  单卡 B200 (180 GB 可用池) 下 8K 请求最大并发容量', loc='left', fontsize=12, weight='bold', pad=8)
        ax2.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E1')
        ax2.legend(frameon=True, loc='upper left', ncol=2, fontsize=11)
        out.save(f, 'ch08-ptq-taxonomy-awq-gptq-fp8-quarot-nvfp4-pareto-frontier')

    return {
        'figures': [
            'ch08-smoothquant-vs-quarot-spinquant-hadamard-outlier-elimination',
            'ch08-ptq-taxonomy-awq-gptq-fp8-quarot-nvfp4-pareto-frontier',
        ]
    }


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    _, family = configure_font()
    plt.rcParams.update({'font.family': [family, 'DejaVu Sans'], 'axes.unicode_minus': False, 'svg.hashsalt': 'ch08-h2'})
    layout_path = here / 'teaching-layout-validation.json'
    existing = json.loads(layout_path.read_text(encoding='utf-8')) if layout_path.exists() else []
    out = Exporter(here)
    draw_clean_i10(out)
    draw(out)
    out.finish()
    by_name = {c['figure']: c for c in out.checks}
    merged = [by_name.get(c.get('figure'), c) for c in existing]
    existing_names = {c.get('figure') for c in existing}
    for c in out.checks:
        if c['figure'] not in existing_names:
            merged.append(c)
    layout_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out.checks, ensure_ascii=False, indent=2))
