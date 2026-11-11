"""Path H2 Ch10 figures: FP8/NVFP4 3-GEMM training datapath (Fig 10-59) and Stochastic Rounding & Cooldown (Fig 10-60)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from figure_style import BORDER_COL, COL, STYLE, TEXT_COL, Exporter, arrow, box, text


def draw(out: Exporter) -> dict:
    with plt.rc_context(STYLE):
        # Fig 10-59: ch10-fp8-blockwise-three-gemm-training-and-transpose-datapath
        f = plt.figure(figsize=(420 / 72, 7.8))
        ax_top = f.add_axes([0.02, 0.50, 0.96, 0.46])
        ax_top.set(xlim=(0, 1), ylim=(0, 1))
        ax_top.axis('off')

        text(ax_top, 0.01, 0.96, 'A  原生低比特三组 GEMM 双轴分块缩放通路', 12, weight='bold')

        box(ax_top, 0.01, 0.52, 0.30, 0.35,
            '① 前向 Fprop\n激活 1×128 缩放\n权重 128×128 块', 'blue', 11)
        box(ax_top, 0.35, 0.52, 0.30, 0.35,
            '② 梯度 Dgrad\n梯度 1×128 缩放\nTMA 块转置读 W', 'green', 11)
        box(ax_top, 0.69, 0.52, 0.30, 0.35,
            '③ 权重梯度 Wgrad\n序列轴 128×1 量化\n累加至 FP32 主梯度', 'purple', 11)
        arrow(ax_top, (0.31, 0.695), (0.35, 0.695), 'data')
        arrow(ax_top, (0.65, 0.695), (0.69, 0.695), 'data')

        box(ax_top, 0.01, 0.06, 0.47, 0.34,
            '每 128 元素二级累加防下溢\nMMA 局部和 → CUDA Core FP32', 'orange', 11)
        box(ax_top, 0.52, 0.06, 0.47, 0.34,
            '高精度保底模块 (BF16/FP32)\n主权重 · 词嵌入 · 路由 · Adam', 'gray', 11)
        arrow(ax_top, (0.16, 0.52), (0.20, 0.40), 'data')
        arrow(ax_top, (0.50, 0.52), (0.36, 0.40), 'data')
        arrow(ax_top, (0.84, 0.52), (0.76, 0.40), 'data')

        ax_bot = f.add_axes([0.25, 0.08, 0.71, 0.35])
        ax_bot.spines[['top', 'right']].set_visible(False)

        recipes = [
            'BF16 混合精度\n(1.00× 基线)',
            'FP8 块缩放\n(1.44× 实测)',
            'NVFP4+RHT+SR\n(2.18× 实测)',
        ]
        gemm_ms = np.array([130.80, 72.00, 34.00])
        quant_ms = np.array([0.00, 10.84, 11.84])
        tail_ms = np.array([26.16, 26.16, 26.16])
        totals = gemm_ms + quant_ms + tail_ms
        y_pos = np.arange(len(recipes))[::-1]
        h_bar = 0.48

        ax_bot.barh(y_pos, gemm_ms, height=h_bar, color=COL['blue'], edgecolor=BORDER_COL['blue'], lw=1.1,
                    label='三组 GEMM 核')
        ax_bot.barh(y_pos, quant_ms, left=gemm_ms, height=h_bar, color=COL['orange'], edgecolor=BORDER_COL['orange'], lw=1.1,
                    label='量化/转置/RHT')
        ax_bot.barh(y_pos, tail_ms, left=gemm_ms + quant_ms, height=h_bar, color=COL['purple'], edgecolor=BORDER_COL['purple'], lw=1.1,
                    label='非矩阵与通信')

        tot_lbls = ['156.96 ms', '109.00 ms (1.44×)', '72.00 ms (2.18×)']
        tot_cols = ['blue', 'green', 'green']
        for yp, tot, lbl, col_k in zip(y_pos, totals, tot_lbls, tot_cols):
            ax_bot.text(tot + 3.0, yp, lbl, va='center', ha='left', fontsize=11, weight='bold', color=TEXT_COL[col_k])

        ax_bot.set_yticks(y_pos, recipes)
        ax_bot.set_xlim(0, 245.0)
        ax_bot.set_ylim(-0.55, 3.05)
        ax_bot.set_xticks([0, 50, 100, 150, 200])
        ax_bot.set_xlabel('B200 单步端到端训练耗时拆解（ms）', fontsize=11)
        ax_bot.set_title('B  算例 10-8：非矩阵尾项塑造的真实提速比', loc='left', fontsize=12, weight='bold', pad=8)
        ax_bot.grid(axis='x', linestyle=':', alpha=0.5, color='#CBD5E1')
        ax_bot.legend(frameon=True, loc='upper right', ncol=2, fontsize=11)

        out.save(f, 'ch10-fp8-blockwise-three-gemm-training-and-transpose-datapath')

        # Fig 10-60: ch10-nvfp4-stochastic-rounding-hadamard-and-precision-cooldown
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(420 / 72, 7.6))
        f.subplots_adjust(left=0.16, right=0.95, bottom=0.09, top=0.93, hspace=0.56)
        for ax in (ax1, ax2):
            ax.spines[['top', 'right']].set_visible(False)

        # Panel A: Worked Example 10-9 (|eta * g| = 0.18 * Delta_q over K = 0..10,000 steps)
        rng = np.random.default_rng(202610)
        k_steps = np.arange(0, 10001)
        p_sr = 0.18
        jumps = rng.binomial(1, p_sr, size=10000)
        sr_traj = np.concatenate([[0.0], np.cumsum(jumps)])
        true_traj = p_sr * k_steps
        rne_traj = np.zeros_like(true_traj)
        std_band = np.sqrt(k_steps * p_sr * (1.0 - p_sr))

        ax1.fill_between(k_steps, true_traj - 2.0 * std_band, true_traj + 2.0 * std_band,
                         color=COL['green'], alpha=0.55, label='随机舍入 ±2σ 置信带')
        ax1.plot(k_steps, true_traj, color=BORDER_COL['blue'], lw=2.0, ls='--', label='FP32 精确累积 (0.18K)')
        ax1.plot(k_steps, sr_traj, color=BORDER_COL['green'], lw=2.1, label='硬件随机舍入 SR 轨迹')
        ax1.plot(k_steps, rne_traj, color=BORDER_COL['orange'], lw=2.4, label='确定性 RNE (停滞为 0)')

        ax1.scatter([100, 10000], [18.0, 1800.0], color=BORDER_COL['green'], s=34, zorder=5)
        ax1.annotate('K=100: 误差 21.34%', xy=(100, 18.0), xytext=(450, 930.0),
                     fontsize=11, weight='bold', color=TEXT_COL['blue'],
                     arrowprops=dict(arrowstyle='->', color=BORDER_COL['blue'], lw=1.2))
        ax1.annotate('K=10,000: 1,800 Δq\n相对 RMSE = 2.13%', xy=(10000, 1800.0), xytext=(5650, 2080.0),
                     fontsize=11, weight='bold', color=TEXT_COL['green'],
                     arrowprops=dict(arrowstyle='->', color=BORDER_COL['green'], lw=1.2))
        ax1.text(6500, 150.0, 'RNE: |ηg|=0.18Δq < 0.5Δq → 零更新', ha='center', va='bottom',
                 fontsize=11, weight='bold', color=TEXT_COL['orange'])

        ax1.set_xlim(0, 10400)
        ax1.set_ylim(-140, 2920)
        ax1.set_xticks([0, 2000, 4000, 6000, 8000, 10000], ['0', '2k', '4k', '6k', '8k', '10k'])
        ax1.set_yticks([0, 600, 1200, 1800, 2400])
        ax1.set_xlabel('参数累积更新步数 K', fontsize=11)
        ax1.set_ylabel('累积更新（Δq 倍数）', fontsize=11)
        ax1.set_title('A  算例 10-9：RNE 停滞与随机舍入收敛证明', loc='left', fontsize=12, weight='bold', pad=8)
        ax1.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1')
        ax1.legend(frameon=True, loc='upper left', fontsize=11)

        # Panel B: Validation loss across 1T tokens with RHT + SR + last 5% BF16 precision cooldown
        tok_pct = np.linspace(5, 100, 191)
        loss_bf16 = 2.18 + 0.92 * np.exp(-tok_pct / 26.0)
        loss_naive_fp4 = 2.415 + 0.95 * np.exp(-tok_pct / 24.0)
        loss_quartet = 2.206 + 0.92 * np.exp(-tok_pct / 26.0)
        cooldown_mask = tok_pct >= 95.0
        loss_quartet[cooldown_mask] = loss_quartet[cooldown_mask] - 0.024 * ((tok_pct[cooldown_mask] - 95.0) / 5.0) ** 0.75

        ax2.axvspan(95.0, 100.0, color=COL['purple'], alpha=0.55, label='末期 5% BF16 退火窗')
        ax2.plot(tok_pct, loss_naive_fp4, color=BORDER_COL['orange'], lw=2.2, ls='-.',
                 label='朴素 FP4 (RNE 无旋转: 2.430)')
        ax2.plot(tok_pct, loss_bf16, color=BORDER_COL['blue'], lw=2.1, ls='--',
                 label='BF16 基准线 (终值 2.200)')
        ax2.plot(tok_pct, loss_quartet, color=BORDER_COL['green'], lw=2.3,
                 label='NVFP4 (RHT+SR+退火: 2.202)')

        ax2.annotate('95% 切回高精度退火\n消除噪声底 (ΔL=0.002)', xy=(95.0, float(loss_quartet[tok_pct == 95.0][0])),
                     xytext=(43.0, 2.58), fontsize=11, weight='bold', color=TEXT_COL['purple'],
                     bbox=dict(boxstyle='round,pad=0.22', fc='white', ec=BORDER_COL['purple'], alpha=0.92, lw=0.9),
                     arrowprops=dict(arrowstyle='->', color=BORDER_COL['purple'], lw=1.2))

        ax2.set_xlim(5, 102)
        ax2.set_ylim(2.14, 3.52)
        ax2.set_xticks([10, 30, 50, 70, 85, 100], ['10%', '30%', '50%', '70%', '85%', '100%'])
        ax2.set_yticks([2.2, 2.5, 2.8, 3.1])
        ax2.set_xlabel('训练词元进度（占 1T 总预算百分比）', fontsize=11)
        ax2.set_ylabel('验证集损失（nats）', fontsize=11)
        ax2.set_title('B  随机 Hadamard 与末期 5% 退火收敛曲线', loc='left', fontsize=12, weight='bold', pad=8)
        ax2.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1')
        ax2.legend(frameon=True, loc='upper right', fontsize=11)

        out.save(f, 'ch10-nvfp4-stochastic-rounding-hadamard-and-precision-cooldown')

    return {
        '10-59': {'bf16_ms': 156.96, 'fp8_ms': 109.00, 'nvfp4_ms': 72.00, 'fp8_speedup': 1.44, 'nvfp4_speedup': 2.18},
        '10-60': {'p_sr': 0.18, 'rrmse_100': 0.213437, 'rrmse_10000': 0.021344, 'cooldown_start_pct': 95.0},
    }
