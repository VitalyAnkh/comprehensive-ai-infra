"""Path H2 Ch09 figures: Punica/S-LoRA unified paging & BGMV (Fig 9-57) and ServerlessLLM/BlitzScale multicast (Fig 9-58)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from figure_style import BORDER_COL, COL, STYLE, TEXT_COL, Exporter
from figure_style.typography import configure_font


def draw(out: Exporter) -> dict:
    with plt.rc_context(STYLE):
        # Fig 9-57: ch09-slora-unified-paging-and-bgmv-heterogeneous-batching
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(420 / 72, 7.6))
        f.subplots_adjust(left=0.14, right=0.96, bottom=0.09, top=0.93, hspace=0.56)
        for ax in (ax1, ax2):
            ax.spines[['top', 'right']].set_visible(False)

        # Panel A: Worked Example 9-8 (Qwen3-32B, 2,000 LoRAs, 149 warm-window tenants, B=128)
        lora_archs = [
            '权重合并独占实例\n(149 张 H100)',
            '未合并串行小矩阵\n(8 张 H100)',
            'S-LoRA 统一分页\n+ BGMV (8 张 H100)',
        ]
        rel_gpu_cost = [100.0, 5.37, 5.37]
        step_lat_ms = [14.30, 89.40, 15.50]
        x_a = np.arange(len(lora_archs))
        w_a = 0.30

        ax1.bar(x_a - 0.17, rel_gpu_cost, width=w_a, color=COL['orange'], edgecolor=BORDER_COL['orange'], lw=1.2,
                label='归一化硬件卡时成本（%）')
        ax1.bar(x_a + 0.17, step_lat_ms, width=w_a, color=COL['green'], edgecolor=BORDER_COL['green'], lw=1.2,
                label='B=128 单步解码耗时（ms）')

        cost_lbls = ['100%\n(149 卡)', '5.4%\n(8 卡)', '5.4%\n降本18.6×']
        lat_lbls = ['14.3 ms', '89.4 ms\n(违反 SLO)', '15.5 ms\n(+8.4%)']
        lat_cols = ['blue', 'orange', 'green']
        for xi, vc, vl, cl, ll, lc in zip(x_a, rel_gpu_cost, step_lat_ms, cost_lbls, lat_lbls, lat_cols):
            if xi == 2:
                ax1.text(xi - 0.04, vc + 3.5, cl, ha='right', va='bottom', fontsize=11, weight='bold', color=TEXT_COL['orange'])
                ax1.text(xi + 0.04, vl + 4.0, ll, ha='left', va='bottom', fontsize=11, weight='bold', color=TEXT_COL[lc])
            else:
                ax1.text(xi - 0.18, vc + 3.5, cl, ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL['orange'])
                ax1.text(xi + 0.19, vl + 3.5, ll, ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL[lc])

        ax1.set_xticks(x_a, lora_archs)
        ax1.set_xlim(-0.58, 2.72)
        ax1.set_ylim(0, 168.0)
        ax1.set_yticks([0, 30, 60, 90, 120])
        ax1.set_ylabel('成本占比（%）/ 步耗时（ms）', fontsize=11)
        ax1.set_title('A  算例 9-8：托管 2,000 个 Qwen3-32B LoRA 的成本与步耗时', loc='left', fontsize=12, weight='bold', pad=8)
        ax1.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E1')
        ax1.legend(frameon=True, loc='upper right', fontsize=11)

        # Panel B: BGMV step latency vs distinct active adapters U in B=128 step
        # Note: GB / (TB/s) = 10^9 B / (10^12 B/s) = 10^-3 s = 1 ms directly!
        u_vals = np.array([1, 8, 16, 32, 48, 96, 128])
        x_u = np.arange(len(u_vals))
        t_r16 = 14.30 + 0.18 + u_vals * (0.25585 / 21.44)
        t_mix = 14.30 + 0.20 + u_vals * (0.44774 / 21.44)
        t_r64 = 14.30 + 0.22 + u_vals * (1.02341 / 21.44)

        ax2.plot(x_u, t_r64, marker='s', ms=5.5, lw=2.2, ls='-.', color=BORDER_COL['orange'], label='全高秩 r=64 (1.02 GB/个)')
        ax2.plot(x_u, t_mix, marker='o', ms=6.0, lw=2.5, color=BORDER_COL['green'], label='混合秩 75% r=16 + 25% r=64')
        ax2.plot(x_u, t_r16, marker='^', ms=5.5, lw=2.2, ls='--', color=BORDER_COL['blue'], label='全低秩 r=16 (256 MB/个)')
        ax2.axhline(14.30, ls=':', lw=1.3, color='#475569', label='共享底座无 LoRA 基线 (14.3 ms)')

        ax2.annotate(
            '算例 9-8 (U=48):\n15.50 ms (+8.4%)',
            xy=(4, t_mix[4]),
            xytext=(3.95, 21.6),
            arrowprops=dict(arrowstyle='->', color=BORDER_COL['green'], lw=1.3),
            bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.90),
            fontsize=11,
            weight='bold',
            color=TEXT_COL['green'],
        )
        ax2.set_xticks(x_u, [f'U={u}' for u in u_vals])
        ax2.set_xlim(-0.45, 6.45)
        ax2.set_ylim(13.5, 25.5)
        ax2.set_yticks([14, 17, 20, 23])
        ax2.set_xlabel('同批 B=128 内去重命中的活跃租户适配器数 U', fontsize=11)
        ax2.set_ylabel('TP8 单步解码耗时（ms）', fontsize=11)
        ax2.set_title('B  统一分页 BGMV 算子在不同异构租户分散度下的延迟增量', loc='left', fontsize=12, weight='bold', pad=8)
        ax2.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1')
        ax2.legend(frameon=True, loc='upper left', fontsize=11)
        out.save(f, 'ch09-slora-unified-paging-and-bgmv-heterogeneous-batching')

        # Fig 9-58: ch09-serverlessllm-blitzscale-live-weight-multicast-autoscaling
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(420 / 72, 7.6))
        f.subplots_adjust(left=0.14, right=0.96, bottom=0.09, top=0.93, hspace=0.56)
        for ax in (ax1, ax2):
            ax.spines[['top', 'right']].set_visible(False)

        # Panel A: Worked Example 9-9 (1 -> 16 replicas of 140 GB 70B model)
        scale_methods = [
            '远端对象存储\n并发拉取',
            '源实例一对多\n串行单播',
            'ServerlessLLM\n本地内存池',
            'BlitzScale\n链式热转发',
        ]
        ready_s = [18.40, 6.04, 1.85, 0.62]
        m_cols = ['orange', 'purple', 'blue', 'green']
        x_m = np.arange(len(scale_methods))

        for xi, rv, mc in zip(x_m, ready_s, m_cols):
            ax1.bar(xi, rv, width=0.45, color=COL[mc], edgecolor=BORDER_COL[mc], lw=1.2)
            lbl = f'{rv:.2f} s\n(快 29.7×)' if xi == 3 else f'{rv:.2f} s'
            ax1.text(xi, rv + 0.55, lbl, ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL[mc])

        ax1.set_xticks(x_m, scale_methods)
        ax1.set_xlim(-0.56, 3.56)
        ax1.set_ylim(0, 24.5)
        ax1.set_yticks([0, 5, 10, 15, 20])
        ax1.set_ylabel('1→16 副本扩容就绪耗时（s）', fontsize=11)
        ax1.set_title('A  算例 9-9：70B 模型 (140 GB) 扩容至 16 副本就绪耗时', loc='left', fontsize=12, weight='bold', pad=8)
        ax1.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E1')

        # Panel B: Ready latency scaling vs target replica count N (1 -> N)
        n_targets = np.array([2, 4, 8, 16, 32])
        x_n = np.arange(len(n_targets))
        new_cnt = n_targets - 1
        t_store_curve = (new_cnt * 140.0) / 120.0 + 0.90
        t_unicast_curve = new_cnt * (17.5 / 45.0) + 0.21
        t_sllm_curve = np.array([1.82, 1.83, 1.84, 1.85, 1.88])
        t_blitz_curve = (17.5 / 45.0) + np.maximum(0, new_cnt - 1) * 0.0015 + 0.21

        ax2.plot(x_n, t_store_curve, marker='s', ms=5.5, lw=2.3, ls=':', color=BORDER_COL['orange'], label='共享存储并发拉取 O(N)')
        ax2.plot(x_n, t_unicast_curve, marker='d', ms=5.5, lw=2.2, ls='-.', color=BORDER_COL['purple'], label='源实例串行单播 O(N)')
        ax2.plot(x_n, t_sllm_curve, marker='^', ms=5.5, lw=2.2, ls='--', color=BORDER_COL['blue'], label='本地主机内存池预驻留')
        ax2.plot(x_n, t_blitz_curve, marker='o', ms=6.0, lw=2.6, color=BORDER_COL['green'], label='BlitzScale 64 MiB 流水链式多播')

        ax2.annotate(
            'N=32 仍稳定在 0.64 s\n(每增 1 跳仅 +1.5 ms)',
            xy=(4, t_blitz_curve[4]),
            xytext=(0.15, 11.2),
            arrowprops=dict(arrowstyle='->', color=BORDER_COL['green'], lw=1.3),
            bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.90),
            fontsize=11,
            weight='bold',
            color=TEXT_COL['green'],
        )
        ax2.set_xticks(x_n, [f'1→{n} 副本' for n in n_targets])
        ax2.set_xlim(-0.42, 4.42)
        ax2.set_ylim(0, 44.0)
        ax2.set_yticks([0, 10, 20, 30, 40])
        ax2.set_xlabel('突发弹性扩容目标副本规模', fontsize=11)
        ax2.set_ylabel('全组副本全部就绪耗时（s）', fontsize=11)
        ax2.set_title('B  不同扩容跨度下四类权重分发路径的尾副本就绪延迟', loc='left', fontsize=12, weight='bold', pad=8)
        ax2.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1')
        ax2.legend(frameon=True, loc='upper left', fontsize=11)
        out.save(f, 'ch09-serverlessllm-blitzscale-live-weight-multicast-autoscaling')

    return {
        'figures': [
            'ch09-slora-unified-paging-and-bgmv-heterogeneous-batching',
            'ch09-serverlessllm-blitzscale-live-weight-multicast-autoscaling',
        ]
    }


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    _, family = configure_font()
    plt.rcParams.update({'font.family': [family, 'DejaVu Sans'], 'axes.unicode_minus': False, 'svg.hashsalt': 'ch09-h2'})
    layout_path = here / 'teaching-layout-validation.json'
    existing = json.loads(layout_path.read_text(encoding='utf-8')) if layout_path.exists() else []
    out = Exporter(here)
    draw(out)
    out.finish()
    new_names = {c['figure'] for c in out.checks}
    merged = [c for c in existing if c.get('figure') not in new_names] + out.checks
    layout_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out.checks, ensure_ascii=False, indent=2))
