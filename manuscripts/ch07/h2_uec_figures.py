"""Path H2 Ch07 figures: UEC packet spraying & selective repeat vs RoCEv2, and four-fabric comparison."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

from figure_style import BORDER_COL, COL, STYLE, TEXT_COL, Exporter
from figure_style.typography import configure_font


def draw(out: Exporter) -> dict:
    with plt.rc_context(STYLE):
        # Fig 7-72: ch07-rocev2-pfc-gobackn-vs-uec-packet-spraying-direct-placement
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(420 / 72, 7.5))
        f.subplots_adjust(left=0.15, right=0.96, bottom=0.09, top=0.93, hspace=0.56)
        for ax in (ax1, ax2):
            ax.spines[['top', 'right']].set_visible(False)

        # Panel A: Worked Example 7-8 (800 Gbps link, RTT=20 us, MTU=4 KiB)
        loss_pct = np.logspace(-4, -1, 200)
        p_prob = loss_pct / 100.0
        t_pkt_us = 0.04096
        t_good_us = t_pkt_us / p_prob

        eta_roce_rto = (t_good_us / (t_good_us + 683.0)) * 98.2
        eta_roce_nak = (t_good_us / (t_good_us + 280.0)) * 98.2
        eta_uec = 98.9 / (1.0 + 1.2 * p_prob)

        ax1.plot(loss_pct, eta_uec, color=BORDER_COL['green'], lw=2.5,
                 label='UEC 1.0 (选择性重传 SACK + 乱序零拷贝直写)')
        ax1.plot(loss_pct, eta_roce_nak, color=BORDER_COL['blue'], lw=2.2, ls='--',
                 label='RoCEv2 硬件快速 NAK (Go-Back-N 冲刷)')
        ax1.plot(loss_pct, eta_roce_rto, color=BORDER_COL['orange'], lw=2.4, ls='-.',
                 label='标准 RoCEv2 (Go-Back-N + 超时降窗恢复)')

        ax1.scatter([0.01, 0.01], [98.9, 36.8], s=64,
                    facecolor=[COL['green'], COL['orange']],
                    edgecolor=[BORDER_COL['green'], BORDER_COL['orange']],
                    lw=1.6, zorder=5)
        ax1.annotate(
            'UEC 保持 98.9%',
            xy=(0.01, 98.9),
            xytext=(0.012, 78.0),
            arrowprops=dict(arrowstyle='->', color=BORDER_COL['green'], lw=1.3),
            bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.88),
            fontsize=11,
            weight='bold',
            color=TEXT_COL['green'],
        )
        ax1.annotate(
            '0.01% 丢包: 标准 RoCEv2\n跌至 36.8% (-63.2%)',
            xy=(0.01, 36.8),
            xytext=(0.00016, 22.0),
            arrowprops=dict(arrowstyle='->', color=BORDER_COL['orange'], lw=1.3),
            bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.88),
            fontsize=11,
            weight='bold',
            color=TEXT_COL['orange'],
        )

        ax1.set_xscale('log')
        ax1.set(
            xlim=(1e-4, 0.12),
            ylim=(0.0, 162.0),
            xlabel='800 Gbps 链路随机丢包率 p（%，对数坐标）',
            ylabel='单流有效带宽利用率（%）',
        )
        ax1.set_xticks([1e-4, 1e-3, 1e-2, 1e-1])
        ax1.set_xticklabels(['0.0001%', '0.001%', '0.01%', '0.1%'])
        ax1.xaxis.set_minor_formatter(NullFormatter())
        ax1.minorticks_off()
        ax1.set_yticks([0, 20, 40, 60, 80, 100])
        ax1.set_title('A  算例 7-8：800 Gbps 随机丢包率对有效带宽的影响', loc='left', fontsize=12, weight='bold', pad=8)
        ax1.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1')
        ax1.legend(frameon=True, loc='upper left', fontsize=11)

        # Panel B: Worked Example 7-9 (2,048-GPU 2-tier Fat-Tree CCT & overload)
        schemes = [
            '静态 ECMP\n(1 QP/卡)',
            '多流 ECMP\n(4 QP/卡)',
            '多流 ECMP\n(8 QP/卡)',
            '多流 ECMP\n(23 QP/卡)',
            'UEC 喷洒\n+权重微调',
        ]
        cct_us = [3868.3, 2088.3, 1632.9, 1217.4, 719.2]
        eff_pct = [17.9, 33.1, 42.3, 56.8, 96.1]
        x_s = np.arange(len(schemes))
        w_s = 0.48

        ax2.bar(
            x_s,
            cct_us,
            width=w_s,
            color=[COL['orange'], COL['orange'], COL['blue'], COL['purple'], COL['green']],
            edgecolor=[BORDER_COL['orange'], BORDER_COL['orange'], BORDER_COL['blue'], BORDER_COL['purple'], BORDER_COL['green']],
            lw=1.3,
        )
        col_keys = ['orange', 'orange', 'blue', 'purple', 'green']
        for xi, vc, ve, ck in zip(x_s, cct_us, eff_pct, col_keys):
            ax2.text(
                xi,
                vc + 100.0,
                f'{vc:.0f} μs\n({ve:.1f}%)',
                ha='center',
                va='bottom',
                fontsize=11,
                weight='bold',
                color=TEXT_COL[ck],
            )
        ax2.axhline(691.1, ls='--', lw=1.3, color='#475569', label='无碰撞理想下界 691 μs (100%)')
        ax2.set_xticks(x_s, schemes)
        ax2.set_xlim(-0.58, 4.58)
        ax2.set_ylim(0, 5150.0)
        ax2.set_yticks([0, 1000, 2000, 3000, 4000])
        ax2.set_ylabel('64 MiB 全域 CCT（μs）', fontsize=11)
        ax2.set_title('B  算例 7-9：2,048 卡 Fat-Tree 多路径方案 CCT 对比', loc='left', fontsize=12, weight='bold', pad=8)
        ax2.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E1')
        ax2.legend(frameon=True, loc='upper right', fontsize=11)
        out.save(f, 'ch07-rocev2-pfc-gobackn-vs-uec-packet-spraying-direct-placement')

        # Fig 7-73: ch07-ib-rocev2-spectrumx-uec-ai-fabric-quantitative-radar
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(420 / 72, 7.5))
        f.subplots_adjust(left=0.15, right=0.96, bottom=0.09, top=0.93, hspace=0.56)
        for ax in (ax1, ax2):
            ax.spines[['top', 'right']].set_visible(False)

        dims = [
            '多径喷洒\n割集利用',
            '有损容忍\n免死锁',
            'Incast\n队列抑制',
            '微秒级\n故障自愈',
            '开放互通\n成本弹性',
            '线速包级\nPSP 加密',
        ]
        s_ib = [95, 86, 94, 92, 38, 65]
        s_roce = [36, 28, 42, 35, 88, 55]
        s_specx = [93, 78, 90, 88, 48, 82]
        s_uec = [96, 98, 93, 94, 95, 96]

        x_d = np.arange(len(dims))
        ax1.plot(x_d, s_uec, marker='o', ms=6, lw=2.5, color=BORDER_COL['green'], label='UEC 1.0 (超以太网 UET)')
        ax1.plot(x_d, s_ib, marker='s', ms=5.5, lw=2.2, ls='--', color=BORDER_COL['teal'], label='InfiniBand NDR/XDR')
        ax1.plot(x_d, s_specx, marker='^', ms=6, lw=2.2, ls='-.', color=BORDER_COL['purple'], label='NVIDIA Spectrum-X')
        ax1.plot(x_d, s_roce, marker='d', ms=5.5, lw=2.1, ls=':', color=BORDER_COL['orange'], label='传统 RoCEv2 (PFC+ECMP)')

        ax1.set_xticks(x_d, dims)
        ax1.set_xlim(-0.42, 5.42)
        ax1.set_ylim(15.0, 145.0)
        ax1.set_yticks([25, 50, 75, 100])
        ax1.set_ylabel('综合工程能力评分（0–100）', fontsize=11)
        ax1.set_title('A  四大 AI 数据中心网络技术栈的六维核心能力谱系', loc='left', fontsize=12, weight='bold', pad=8)
        ax1.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1')
        ax1.legend(frameon=True, loc='upper center', ncol=2, fontsize=11)

        fabrics = [
            '传统 RoCEv2\n(PFC+ECMP)',
            'InfiniBand\nXDR 800G',
            'Spectrum-X\n800G 以太网',
            'UEC 1.0\n800G 超以太网',
        ]
        a2a_eff = [54.0, 94.5, 92.8, 95.6]
        norm_perf_per_dollar = [63.5, 67.5, 84.4, 95.6]
        x_f = np.arange(len(fabrics))
        w_f = 0.30
        off_f = 0.18

        ax2.bar(x_f - off_f, a2a_eff, width=w_f, color=COL['blue'], edgecolor=BORDER_COL['blue'], lw=1.2,
                label='万卡 MoE All-to-All 有效带宽率（%）')
        ax2.bar(x_f + off_f, norm_perf_per_dollar, width=w_f, color=COL['green'], edgecolor=BORDER_COL['green'], lw=1.2,
                label='归一化单位网络 CapEx 有效吞吐指数')

        for xi, va, vp in zip(x_f, a2a_eff, norm_perf_per_dollar):
            ax2.text(xi - 0.22, va + 2.8, f'{va:.1f}%', ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL['blue'])
            ax2.text(xi + 0.22, vp + 2.8, f'{vp:.1f}', ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL['green'])

        ax2.set_xticks(x_f, fabrics)
        ax2.set_xlim(-0.62, 3.62)
        ax2.set_ylim(0, 158.0)
        ax2.set_yticks([0, 25, 50, 75, 100])
        ax2.set_ylabel('百分比 / 性价比指数', fontsize=11)
        ax2.set_title('B  万卡集群 All-to-All 有效带宽率与单位网络投资回报对比', loc='left', fontsize=12, weight='bold', pad=8)
        ax2.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E1')
        ax2.legend(frameon=True, loc='upper left', fontsize=11)
        out.save(f, 'ch07-ib-rocev2-spectrumx-uec-ai-fabric-quantitative-radar')

    return {
        'figures': [
            'ch07-rocev2-pfc-gobackn-vs-uec-packet-spraying-direct-placement',
            'ch07-ib-rocev2-spectrumx-uec-ai-fabric-quantitative-radar',
        ]
    }


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    _, family = configure_font()
    plt.rcParams.update({'font.family': [family, 'DejaVu Sans'], 'axes.unicode_minus': False, 'svg.hashsalt': 'ch07-h2'})
    layout_path = here / 'teaching-layout-validation.json'
    existing = json.loads(layout_path.read_text(encoding='utf-8')) if layout_path.exists() else []
    out = Exporter(here)
    draw(out)
    new_names = {c['figure'] for c in out.checks}
    merged = [c for c in existing if c.get('figure') not in new_names] + out.checks
    layout_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out.checks, ensure_ascii=False, indent=2))
