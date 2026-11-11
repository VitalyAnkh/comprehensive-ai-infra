"""Path H2 Ch11 figures: Cloud-native Kueue/Volcano/llm-d control plane (Fig 11-60) and VTC + FinOps Tokenomics (Fig 11-61)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from figure_style import BORDER_COL, COL, STYLE, TEXT_COL, Exporter, arrow, box, canvas, text


def draw(out: Exporter) -> dict:
    with plt.rc_context(STYLE):
        # Fig 11-60: ch11-kueue-volcano-gang-topology-and-llmd-kv-gateway-architecture
        f, a = canvas(6.6)
        text(a, 0.24, 0.955, 'A. 批训练控制面 (Kueue/Volcano)', 12, ha='center', weight='bold')
        text(a, 0.745, 0.955, 'B. 推理控制面 (llm-d/AIBrix)', 12, ha='center', weight='bold')
        a.plot([0.495, 0.495], [0.03, 0.97], color=COL['line'], ls='--', lw=1.0)

        box(a, 0.03, 0.75, 0.44, 0.15,
            '成组准入 (minAvailable)\n全员就绪分配，防半数死锁', 'blue', 11)
        arrow(a, (0.25, 0.75), (0.25, 0.67), 'data')
        box(a, 0.03, 0.50, 0.44, 0.16,
            '拓扑亲和放置 (TAS)\n优先装箱 NVL72 与同叶脊', 'green', 11)
        arrow(a, (0.25, 0.50), (0.25, 0.42), 'data')
        box(a, 0.03, 0.26, 0.44, 0.16,
            '层级配额借用 (Cohort)\n闲置训练配额出借离线评测', 'purple', 11)
        box(a, 0.03, 0.04, 0.44, 0.17,
            'SkyPilot 跨云异构经纪\n联合单价/抢占/出网费寻优', 'orange', 11)

        box(a, 0.52, 0.75, 0.45, 0.15,
            'Envoy + ExtProc 推理网关\n替代轮询，微秒级状态路由', 'blue', 11)
        arrow(a, (0.745, 0.75), (0.745, 0.67), 'data')
        box(a, 0.52, 0.50, 0.45, 0.16,
            'KV 前缀树命中感知打分\nα·Hit_KV - β·Queue_Load', 'green', 11)
        arrow(a, (0.745, 0.50), (0.745, 0.42), 'data')
        box(a, 0.52, 0.26, 0.45, 0.16,
            'PD 分离与 VTC 时钟准入\nw_in L_in + w_out L_out', 'purple', 11)
        box(a, 0.52, 0.04, 0.45, 0.17,
            '潮汐弹性与合成数据填谷\n闲时回收算力生产蒸馏语料', 'gray', 11)

        out.save(f, 'ch11-kueue-volcano-gang-topology-and-llmd-kv-gateway-architecture')

        # Fig 11-61: ch11-vtc-fair-queuing-and-finops-capacity-portfolio-tokenomics
        f = plt.figure(figsize=(420 / 72, 7.8))
        ax1 = f.add_axes([0.15, 0.56, 0.81, 0.36])
        ax2 = f.add_axes([0.27, 0.08, 0.69, 0.35])
        for ax in (ax1, ax2):
            ax.spines[['top', 'right']].set_visible(False)

        # Panel A: Worked Example 11-8 (Tenant A 256/64 vs Tenant B 1024/4096, w_out/w_in = 5)
        sched_policies = [
            '请求数公平轮询\n(1:1 交替出队)',
            'VTC 虚拟词元公平\n(w_out/w_in = 5)',
        ]
        share_a = np.array([2.61, 50.00])
        share_b = np.array([97.39, 50.00])
        x_v = np.arange(len(sched_policies))
        w_v = 0.28

        ax1.bar(x_v - 0.16, share_a, width=w_v, color=COL['green'], edgecolor=BORDER_COL['green'], lw=1.2,
                label='短问答租户 A 算力占比')
        ax1.bar(x_v + 0.16, share_b, width=w_v, color=COL['orange'], edgecolor=BORDER_COL['orange'], lw=1.2,
                label='长推理租户 B 算力占比')

        ax1.text(0 - 0.16, share_a[0] + 3.0, '2.6%\n(饥饿)', ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL['orange'])
        ax1.text(0 + 0.16, share_b[0] + 3.0, '97.4%\n(重37.3×)', ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL['orange'])
        ax1.text(1 - 0.16, share_a[1] + 3.0, '50.0%\n(37.3:1)', ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL['green'])
        ax1.text(1 + 0.16, share_b[1] + 3.0, '50.0%\n(对等)', ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL['blue'])

        ax1.text(0.0, 144.0, 'A P99 TTFT: 4.80 s', ha='center', va='center', fontsize=11, weight='bold',
                 color=TEXT_COL['orange'], bbox=dict(boxstyle='round,pad=0.25', fc=COL['orange'], ec=BORDER_COL['orange'], lw=1.0))
        ax1.text(1.0, 144.0, 'A P99 TTFT: 0.31 s (-93.5%)', ha='center', va='center', fontsize=11, weight='bold',
                 color=TEXT_COL['green'], bbox=dict(boxstyle='round,pad=0.25', fc=COL['green'], ec=BORDER_COL['green'], lw=1.0))

        ax1.set_xticks(x_v, sched_policies)
        ax1.set_xlim(-0.55, 1.55)
        ax1.set_ylim(0, 206.0)
        ax1.set_yticks([0, 25, 50, 75, 100])
        ax1.set_ylabel('GPU 服务时间占比（%）', fontsize=11)
        ax1.set_title('A  算例 11-8：VTC 虚拟词元公平消除长输出挤占', loc='left', fontsize=12, weight='bold', pad=8)
        ax1.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E1')
        ax1.legend(frameon=True, loc='upper right', fontsize=11)

        # Panel B: Worked Example 11-9 (80B tok/day, 2.4:1 peak-to-avg, USD / 1M Tokens)
        portfolios = [
            '纯峰值预留\n(利用率 41.7%)',
            '纯按需弹性\n(On-Demand)',
            '三层容量组合\n(65/25/10%)',
        ]
        res_part = np.array([0.9053, 0.0000, 0.30555])
        od_spot_part = np.array([0.0000, 0.6000, 0.07005])
        totals_usd = res_part + od_spot_part
        y_f = np.arange(len(portfolios))[::-1]
        h_f = 0.48

        ax2.barh(y_f, res_part, height=h_f, color=COL['blue'], edgecolor=BORDER_COL['blue'], lw=1.1,
                 label='预留基荷与填谷')
        ax2.barh(y_f, od_spot_part, left=res_part, height=h_f, color=COL['green'], edgecolor=BORDER_COL['green'], lw=1.1,
                 label='按需 / 跨云 Spot')

        cost_notes = [
            '0.9053 USD (72.4k/天)',
            '0.6000 USD (48.0k/天)',
            '0.3756 USD (-37.4%)',
        ]
        note_cols = ['orange', 'blue', 'green']
        for yp, tot, note, ck in zip(y_f, totals_usd, cost_notes, note_cols):
            ax2.text(tot + 0.025, yp, note, va='center', ha='left', fontsize=11, weight='bold', color=TEXT_COL[ck])

        ax2.set_yticks(y_f, portfolios)
        ax2.set_xlim(0, 1.66)
        ax2.set_ylim(-0.55, 2.95)
        ax2.set_xticks([0.0, 0.4, 0.8, 1.2])
        ax2.set_xlabel('日均 80B 词元成本（USD / 1M Tokens）', fontsize=11)
        ax2.set_title('B  算例 11-9：三层容量组合单位词元成本账本', loc='left', fontsize=12, weight='bold', pad=8)
        ax2.grid(axis='x', linestyle=':', alpha=0.5, color='#CBD5E1')
        ax2.legend(frameon=True, loc='upper right', fontsize=11)

        out.save(f, 'ch11-vtc-fair-queuing-and-finops-capacity-portfolio-tokenomics')

    return {
        '11-60': {'gang_admission': 'minAvailable', 'gateway': 'Envoy+ExtProc KV-aware'},
        '11-61': {
            'vtc_weight_ratio': 5,
            'req_fair_share_a_pct': 2.61,
            'vtc_share_a_pct': 50.0,
            'p99_ttft_before_s': 4.80,
            'p99_ttft_after_s': 0.31,
            'ondemand_usd_per_1m': 0.6000,
            'portfolio_usd_per_1m': 0.3756,
            'cost_reduction_pct': 37.4,
        },
    }
