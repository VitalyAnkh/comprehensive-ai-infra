"""Path H2 Ch06 figures: Five global supernode architectures & OCS vs switch fault resilience."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import numpy as np

from figure_style import BORDER_COL, COL, STYLE, TEXT_COL, Exporter
from figure_style.typography import configure_font


def draw(out: Exporter) -> dict:
    with plt.rc_context(STYLE):
        # Fig 6-68: ch06-five-global-supernode-architectures-topology-and-bandwidth-comparison
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(420 / 72, 7.5))
        f.subplots_adjust(left=0.15, right=0.96, bottom=0.09, top=0.93, hspace=0.56)
        for ax in (ax1, ax2):
            ax.spines[['top', 'right']].set_visible(False)

        # Panel A: Scale N vs Half-Cluster Per-Card Unidirectional Bisection BW (GB/s)
        pts = [
            (65.0, 1800.0, 'teal', 'o', 'Rubin NVL72\n(72 卡, 20.7 TB, 铜缆)', (36.0, 195.0), 'left'),
            (80.0, 1800.0, 'purple', 's', 'AMD Helios 72\n(72 卡, 31.1 TB, UALink)', (84.0, 5000.0), 'left'),
            (576.0, 1800.0, 'gray', 'D', 'Kyber NVL576\n(576 卡, CPO 正交中板)', (980.0, 4300.0), 'left'),
            (144.0, 500.0, 'orange', '^', 'AWS Trn3 144\n(20.7 TB, 两级电交换)', (205.0, 82.0), 'left'),
            (384.0, 392.0, 'blue', 'o', 'CloudMatrix 384\n(49.2 TB, LPO 光互连)', (860.0, 430.0), 'left'),
            (9216.0, 16.67, 'green', 'o', 'TPU v7 Pod 9,216\n(1.77 PB, 3D Torus+OCS)', (420.0, 22.0), 'left'),
        ]

        ax1.axhline(1800.0, ls=':', lw=1.1, color='#94A3B8', zorder=2)
        for xp, bw_uni, col, mk, label_txt, (tx, ty), ha_a in pts:
            ax1.scatter(
                [xp],
                [bw_uni],
                s=58,
                marker=mk,
                facecolor=COL[col],
                edgecolor=BORDER_COL[col],
                linewidth=1.6,
                zorder=5,
            )
            ax1.annotate(
                label_txt,
                xy=(xp, bw_uni),
                xytext=(tx, ty),
                ha=ha_a,
                va='center',
                fontsize=11,
                weight='bold',
                color=TEXT_COL[col],
                arrowprops=dict(arrowstyle='->', color=BORDER_COL[col], lw=1.2, shrinkA=2, shrinkB=4),
                bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.88),
                zorder=6,
            )

        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set(
            xlim=(34.0, 18500.0),
            ylim=(8.0, 10800.0),
            xlabel='超节点单域互连卡数 N（对数坐标）',
            ylabel='半区单卡单向对分带宽（GB/s）',
        )
        ax1.set_xticks([72, 144, 384, 9216])
        ax1.set_yticks([15, 50, 150, 500, 1800])
        ax1.xaxis.set_major_formatter(ScalarFormatter())
        ax1.yaxis.set_major_formatter(ScalarFormatter())
        ax1.xaxis.set_minor_formatter(NullFormatter())
        ax1.yaxis.set_minor_formatter(NullFormatter())
        ax1.minorticks_off()
        ax1.set_title('A  五大全球超节点的规模、单卡对分带宽与内存池坐标', loc='left', fontsize=12, weight='bold', pad=8)
        ax1.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1')

        # Panel B: Worked Example 6-8 (671B MoE at EP=64, 128, 384)
        ep_labels = ['EP = 64\n(单立方体/单柜)', 'EP = 128\n(跨 2 单元)', 'EP = 384\n(跨 6 单元/16 柜)']
        tps_nvl = [49.9, 49.4, 49.1]
        tps_cm384 = [24.6, 24.4, 24.3]
        tps_torus = [14.3, 7.6, 5.2]
        x_e = np.arange(len(ep_labels))
        w_b = 0.25

        ax2.bar(x_e - w_b, tps_nvl, width=w_b, color=COL['teal'], edgecolor=BORDER_COL['teal'], lw=1.2,
                label='Rubin NVL72/576 全连接交换')
        ax2.bar(x_e, tps_cm384, width=w_b, color=COL['blue'], edgecolor=BORDER_COL['blue'], lw=1.2,
                label='CloudMatrix 384 两级全光交换')
        ax2.bar(x_e + w_b, tps_torus, width=w_b, color=COL['green'], edgecolor=BORDER_COL['green'], lw=1.2,
                label='TPU v7 3D Torus (全域均匀路由)')

        for xi, vn, vc, vt in zip(x_e, tps_nvl, tps_cm384, tps_torus):
            ax2.text(xi - w_b, vn + 1.6, f'{vn:.1f}k', ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL['teal'])
            ax2.text(xi, vc + 1.6, f'{vc:.1f}k', ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL['blue'])
            ax2.text(xi + w_b, vt + 1.6, f'{vt:.1f}k', ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL['green'])

        ax2.set_xticks(x_e, ep_labels)
        ax2.set_xlim(-0.55, 2.55)
        ax2.set_ylim(0, 98.0)
        ax2.set_yticks([0, 15, 30, 45, 60])
        ax2.set_ylabel('单卡有效吞吐（k tok/s）', fontsize=11)
        ax2.set_title('B  算例 6-8：671B MoE 随专家并行扩展的单卡吞吐对比', loc='left', fontsize=12, weight='bold', pad=8)
        ax2.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E1')
        ax2.legend(frameon=True, loc='upper right', fontsize=11)
        out.save(f, 'ch06-five-global-supernode-architectures-topology-and-bandwidth-comparison')

        # Fig 6-69: ch06-ocs-reconfigurable-3d-torus-vs-all-to-all-switch-fault-resilience
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(420 / 72, 7.4))
        f.subplots_adjust(left=0.15, right=0.96, bottom=0.09, top=0.93, hspace=0.55)
        for ax in (ax1, ax2):
            ax.spines[['top', 'right']].set_visible(False)

        p_fault = np.linspace(0.0, 5.0, 201)
        avail_ocs = 100.0 / (1.0 + 0.0071 * p_fault)
        avail_sw = 100.0 / (1.0 + 0.0938 * p_fault)
        avail_static_torus = 100.0 / (1.0 + 0.3865 * p_fault)

        ax1.plot(p_fault, avail_ocs, color=BORDER_COL['green'], lw=2.5, label='Palomar OCS 光路热替换 (10 ms 切镜 / 18 s 恢复)')
        ax1.plot(p_fault, avail_sw, color=BORDER_COL['blue'], lw=2.3, ls='--', label='固定铜背板电交换 (整柜隔离重编排 420 s)')
        ax1.plot(p_fault, avail_static_torus, color=BORDER_COL['orange'], lw=2.3, ls='-.', label='无 OCS 静态直连 3D Torus (破洞拥塞绕行)')

        ax1.scatter([2.0, 2.0, 2.0], [98.6, 84.2, 56.4], s=60,
                    facecolor=[COL['green'], COL['blue'], COL['orange']],
                    edgecolor=[BORDER_COL['green'], BORDER_COL['blue'], BORDER_COL['orange']],
                    lw=1.6, zorder=5)

        ax1.annotate(
            'OCS 热替换: 98.6%',
            xy=(2.0, 98.6),
            xytext=(2.65, 89.0),
            arrowprops=dict(arrowstyle='->', color=BORDER_COL['green'], lw=1.3),
            fontsize=11,
            weight='bold',
            color=TEXT_COL['green'],
        )
        ax1.annotate(
            '整柜弹性切换: 84.2%',
            xy=(2.0, 84.2),
            xytext=(2.35, 63.0),
            arrowprops=dict(arrowstyle='->', color=BORDER_COL['blue'], lw=1.3),
            fontsize=11,
            weight='bold',
            color=TEXT_COL['blue'],
        )
        ax1.annotate(
            '静态环面绕行: 56.4%',
            xy=(2.0, 56.4),
            xytext=(0.25, 39.5),
            arrowprops=dict(arrowstyle='->', color=BORDER_COL['orange'], lw=1.3),
            fontsize=11,
            weight='bold',
            color=TEXT_COL['orange'],
        )

        ax1.set(
            xlim=(0.0, 5.0),
            ylim=(32.0, 152.0),
            xlabel='24 小时累计节点/立方体故障率 p（%）',
            ylabel='集群有效算力可用率（%）',
        )
        ax1.set_yticks([40, 60, 80, 100])
        ax1.set_title('A  算例 6-9：OCS 动态重配置与固定拓扑在故障下的可用率', loc='left', fontsize=12, weight='bold', pad=8)
        ax1.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1')
        ax1.legend(frameon=True, loc='upper right', fontsize=11)

        modes = [
            '静态 Torus\n邻环绕行',
            '固定铜背板\n整柜重编排',
            '电交换平面\n单平面降级',
            'OCS 镜面切换\n热备拼接',
        ]
        bw_keep = [50.0, 100.0, 94.4, 100.0]
        rec_speed_score = [56.4, 84.2, 93.8, 98.6]
        x_m = np.arange(len(modes))
        w_m = 0.28
        off_m = 0.19

        ax2.bar(x_m - off_m, bw_keep, width=w_m, color=COL['teal'], edgecolor=BORDER_COL['teal'], lw=1.2,
                label='重整后稳态有效带宽保持率（%）')
        ax2.bar(x_m + off_m, rec_speed_score, width=w_m, color=COL['purple'], edgecolor=BORDER_COL['purple'], lw=1.2,
                label='2% 故障率下全天有效吞吐保持率（%）')

        for xi, vb, vr in zip(x_m, bw_keep, rec_speed_score):
            vb_str = '100%' if vb >= 99.95 else f'{vb:.1f}%'
            dy_b = 9.5 if abs(vb - vr) < 6.0 else 2.8
            dy_r = 2.2 if abs(vb - vr) < 6.0 else 2.8
            ax2.text(xi - 0.24, vb + dy_b, vb_str, ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL['teal'])
            ax2.text(xi + 0.24, vr + dy_r, f'{vr:.1f}%', ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL['purple'])

        ax2.set_xticks(x_m, modes)
        ax2.set_xlim(-0.62, 3.62)
        ax2.set_ylim(0, 162.0)
        ax2.set_yticks([0, 25, 50, 75, 100])
        ax2.set_ylabel('保持率指标（%）', fontsize=11)
        ax2.set_title('B  四类容错机制的稳态带宽保持率与全天有效吞吐对比', loc='left', fontsize=12, weight='bold', pad=8)
        ax2.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E1')
        ax2.legend(frameon=True, loc='upper left', fontsize=11)
        out.save(f, 'ch06-ocs-reconfigurable-3d-torus-vs-all-to-all-switch-fault-resilience')

    return {
        'figures': [
            'ch06-five-global-supernode-architectures-topology-and-bandwidth-comparison',
            'ch06-ocs-reconfigurable-3d-torus-vs-all-to-all-switch-fault-resilience',
        ]
    }


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    _, family = configure_font()
    plt.rcParams.update({'font.family': [family, 'DejaVu Sans'], 'axes.unicode_minus': False, 'svg.hashsalt': 'ch06-h2'})
    layout_path = here / 'teaching-layout-validation.json'
    existing = json.loads(layout_path.read_text(encoding='utf-8')) if layout_path.exists() else []
    out = Exporter(here)
    draw(out)
    new_names = {c['figure'] for c in out.checks}
    merged = [c for c in existing if c.get('figure') not in new_names] + out.checks
    layout_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out.checks, ensure_ascii=False, indent=2))
