"""Path H2 Ch05 figures: torch.compile/OpenXLA min-cut, NanoFlow 3-domain overlap, Helion/CuTe & e-graphs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from figure_style import BORDER_COL, COL, STYLE, TEXT_COL, Exporter, arrow, box, canvas, text
from figure_style.typography import configure_font


def draw(out: Exporter) -> dict:
    with plt.rc_context(STYLE):
        # Fig 5-51: ch05-torch-compile-dynamo-aotautograd-mincut-inductor-pipeline
        f = plt.figure(figsize=(420 / 72, 7.5))
        ax_top = f.add_axes([0.02, 0.48, 0.96, 0.49])
        ax_top.set_xlim(0, 1)
        ax_top.set_ylim(0, 1)
        ax_top.axis('off')

        ax_top.add_patch(
            FancyBboxPatch(
                (0.01, 0.03),
                0.98,
                0.94,
                boxstyle='round,pad=0.003,rounding_size=0.014',
                facecolor='#FAFCFE',
                edgecolor='#CBD5E1',
                lw=1.0,
            )
        )
        text(ax_top, 0.03, 0.91, 'A  torch.compile 四阶段编译栈与 OpenXLA 双轨映射', 12, weight='bold')
        box(ax_top, 0.04, 0.60, 0.44, 0.25, '1. TorchDynamo\n字节码守卫生成\nSymInt 动态形状', 'blue', 11)
        box(ax_top, 0.52, 0.60, 0.44, 0.25, '2. AOTAutograd\n前反向联合捕获\n最小割激活切分', 'purple', 11)
        box(ax_top, 0.52, 0.28, 0.44, 0.24, '3. PrimTorch\n约 250 核心原语\n算子语义降维', 'teal', 11)
        box(ax_top, 0.04, 0.28, 0.44, 0.24, '4. Inductor / XLA\n循环 IR 算子融合\nTriton / StableHLO', 'green', 11)
        arrow(ax_top, (0.48, 0.725), (0.52, 0.725))
        arrow(ax_top, (0.74, 0.60), (0.74, 0.52))
        arrow(ax_top, (0.52, 0.40), (0.48, 0.40))
        box(ax_top, 0.04, 0.07, 0.92, 0.16, '最小割准则：保存 GEMM 输出激活，反向重算逐元素链', 'orange', 11)

        ax_b1 = f.add_axes([0.12, 0.08, 0.37, 0.32])
        ax_b2 = f.add_axes([0.61, 0.08, 0.35, 0.32])
        for ax in (ax_b1, ax_b2):
            ax.spines[['top', 'right']].set_visible(False)

        # Panel B-left: 算例 5-8 Min-Cut activation memory vs recompute overhead (Qwen3-8B single layer, T=4096)
        policies = ['朴素\n全存', 'AOT\n最小割', '整层\n重算']
        mem_mib = [632.0, 336.0, 32.0]
        cols_p = ['orange', 'green', 'blue']
        x_p = np.arange(len(policies))
        for xi, mv, cn in zip(x_p, mem_mib, cols_p):
            ax_b1.bar(xi, mv, width=0.46, color=COL[cn], edgecolor=BORDER_COL[cn], lw=1.3)
            ax_b1.text(xi, mv + 16.0, f'{mv:.0f}', ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL[cn])
        ax_b1.set_xticks(x_p, policies)
        ax_b1.set_xlim(-0.55, 2.55)
        ax_b1.set_ylim(0, 770.0)
        ax_b1.set_ylabel('保存激活（MiB）', fontsize=11)
        ax_b1.set_title('B1  算例 5-8：激活显存', loc='left', fontsize=11.5, weight='bold', pad=7)
        ax_b1.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E1')

        # Panel B-right: 算例 5-10 Static recompile storm (64x = 192s) vs SymInt (3.2s)
        comp_modes = ['静态64次', 'SymInt动态']
        comp_sec = [192.0, 3.2]
        cols_c = ['orange', 'green']
        x_c = np.arange(len(comp_modes))
        for xi, sv, cn in zip(x_c, comp_sec, cols_c):
            ax_b2.bar(xi, sv, width=0.42, color=COL[cn], edgecolor=BORDER_COL[cn], lw=1.3)
            ax_b2.text(xi, sv + 6.0, f'{sv:.1f} s', ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL[cn])
        ax_b2.set_xticks(x_c, comp_modes)
        ax_b2.set_xlim(-0.55, 1.55)
        ax_b2.set_ylim(0, 245.0)
        ax_b2.set_ylabel('编译耗时（s）', fontsize=11)
        ax_b2.set_title('B2  算例 5-10：编译风暴', loc='left', fontsize=11.5, weight='bold', pad=7)
        ax_b2.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E1')
        out.save(f, 'ch05-torch-compile-dynamo-aotautograd-mincut-inductor-pipeline')

        # Fig 5-52: ch05-nanoflow-pod-attention-intra-gpu-three-pipeline-overlap
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(420 / 72, 6.9))
        f.subplots_adjust(left=0.20, right=0.95, bottom=0.10, top=0.92, hspace=0.54)
        for ax in (ax1, ax2):
            ax.spines[['top', 'right']].set_visible(False)

        y_rows = [3.2, 2.0, 1.2, 0.4]
        ax1.barh(3.2, 1.80, left=0.0, height=0.48, color=COL['blue'], edgecolor=BORDER_COL['blue'], lw=1.3)
        ax1.barh(3.2, 1.50, left=1.80, height=0.48, color=COL['teal'], edgecolor=BORDER_COL['teal'], lw=1.3)
        ax1.barh(3.2, 1.20, left=3.30, height=0.48, color=COL['purple'], edgecolor=BORDER_COL['purple'], lw=1.3)
        ax1.text(0.90, 3.2, '计算 1.80', ha='center', va='center', fontsize=11, weight='bold', color=TEXT_COL['blue'])
        ax1.text(2.55, 3.2, '访存 1.50', ha='center', va='center', fontsize=11, weight='bold', color=TEXT_COL['teal'])
        ax1.text(3.90, 3.2, '通信 1.20', ha='center', va='center', fontsize=11, weight='bold', color=TEXT_COL['purple'])
        ax1.text(4.56, 3.2, '4.50 ms', ha='left', va='center', fontsize=11, weight='bold', color=TEXT_COL['orange'])

        ax1.barh(2.0, 2.38, left=0.0, height=0.44, color=COL['blue'], edgecolor=BORDER_COL['blue'], lw=1.3)
        ax1.barh(1.2, 2.34, left=0.0, height=0.44, color=COL['teal'], edgecolor=BORDER_COL['teal'], lw=1.3)
        ax1.barh(0.4, 2.31, left=0.0, height=0.44, color=COL['purple'], edgecolor=BORDER_COL['purple'], lw=1.3)
        ax1.text(1.19, 2.0, '96 SM: 2.38 ms', ha='center', va='center', fontsize=11, weight='bold', color=TEXT_COL['blue'])
        ax1.text(1.17, 1.2, '32 SM: 2.34 ms', ha='center', va='center', fontsize=11, weight='bold', color=TEXT_COL['teal'])
        ax1.text(1.15, 0.4, '20 SM: 2.31 ms', ha='center', va='center', fontsize=11, weight='bold', color=TEXT_COL['purple'])

        ax1.axvline(2.38, color=BORDER_COL['green'], lw=1.8, ls='--')
        ax1.text(2.46, 1.2, '三域并发完成:\n2.38 ms (1.89×)', ha='left', va='center', fontsize=11, weight='bold', color=TEXT_COL['green'])
        ax1.set_yticks(y_rows, ['传统顺序阶段', 'NanoFlow计算', 'NanoFlow访存', 'NanoFlow通信'])
        ax1.set_xlim(0.0, 5.45)
        ax1.set_ylim(-0.15, 3.85)
        ax1.set_xlabel('混合批次单轮执行时间（ms）', fontsize=11)
        ax1.set_title('A  算例 5-9：B200 顺序阶段与算子内三域并发重叠时间线', loc='left', fontsize=12, weight='bold', pad=8)
        ax1.grid(axis='x', linestyle=':', alpha=0.5, color='#CBD5E1')

        s_comp = np.arange(64, 118, 2)
        s_net = 20.0
        s_mem = 148.0 - s_net - s_comp
        # Calibrated to match 算例 5-9: T_seq=(1.80, 1.50, 1.20) at 148 SM; at (96, 32, 20) -> (2.38, 2.34, 2.31) ms
        tau_c, tau_m, tau_n = 146.03, 31.75, 27.47
        eta_c = (1.0 - np.exp(-s_comp / tau_c)) / (1.0 - np.exp(-148.0 / tau_c))
        eta_m = (1.0 - np.exp(-s_mem / tau_m)) / (1.0 - np.exp(-148.0 / tau_m))
        eta_n = (1.0 - np.exp(-s_net / tau_n)) / (1.0 - np.exp(-148.0 / tau_n))
        t_comp = 1.80 / eta_c
        t_mem = 1.50 / eta_m
        t_net = np.full_like(s_comp, 1.20 / eta_n, dtype=float)
        t_max = np.maximum(np.maximum(t_comp, t_mem), t_net)

        ax2.plot(s_comp, t_comp, color=BORDER_COL['blue'], lw=2.0, ls='--', label=r'计算域耗时 $T_{\mathrm{comp}}(s_c)$')
        ax2.plot(s_comp, t_mem, color=BORDER_COL['teal'], lw=2.0, ls='-.', label=r'HBM 访存域耗时 $T_{\mathrm{mem}}(128-s_c)$')
        ax2.plot(s_comp, t_max, color=BORDER_COL['green'], lw=2.6, label=r'三域并发上界 $T_{\mathrm{nanoflow}}$')
        ax2.scatter([96.0], [2.38], s=64, facecolor=COL['green'], edgecolor=BORDER_COL['green'], lw=1.7, zorder=5)
        ax2.annotate(
            '最优配额 $(96, 32, 20)$ SM:\n$T^* = 2.38$ ms (提速 $1.89\\times$)',
            xy=(96.0, 2.38),
            xytext=(64.5, 3.35),
            arrowprops=dict(arrowstyle='->', color=BORDER_COL['green'], lw=1.3),
            fontsize=11,
            weight='bold',
            color=TEXT_COL['green'],
        )
        ax2.set(
            xlim=(62, 118),
            ylim=(1.5, 5.6),
            xlabel=r'计算密集微片 SM 配额 $s_{\mathrm{comp}}$（固定通信 $s_{\mathrm{net}} = 20$ SM）',
            ylabel='域耗时（ms）',
        )
        ax2.set_yticks([2.0, 3.0, 4.0, 5.0])
        ax2.set_title('B  SM 子分区配额博弈曲线与木桶均衡点', loc='left', fontsize=12, weight='bold', pad=8)
        ax2.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1')
        ax2.legend(frameon=True, loc='upper left', fontsize=11)
        out.save(f, 'ch05-nanoflow-pod-attention-intra-gpu-three-pipeline-overlap')

        # Fig 5-53: ch05-helion-cute-dsl-and-egraph-equivalence-verification
        f, a = canvas(7.4)
        a.add_patch(
            FancyBboxPatch(
                (0.02, 0.51),
                0.96,
                0.46,
                boxstyle='round,pad=0.003,rounding_size=0.012',
                facecolor='#FAFCFE',
                edgecolor='#CBD5E1',
                lw=1.0,
            )
        )
        text(a, 0.04, 0.94, 'A. 新一代块级 DSL 双路径：PyTorch Helion 与 NVIDIA CuTe DSL', 11.5, weight='bold')
        box(a, 0.04, 0.76, 0.44, 0.14, 'PyTorch Helion\nhl.tile 多维索引\n自动搜流水与分块', 'blue', 11)
        box(a, 0.52, 0.76, 0.44, 0.14, 'NVIDIA CuTe DSL\n分层 Layout 代数\n显式控 TMA/TMEM', 'green', 11)
        box(a, 0.04, 0.60, 0.44, 0.12, '代码量缩减 8×\n直连 torch.compile', 'teal', 11)
        box(a, 0.52, 0.60, 0.44, 0.12, '双 CTA + Warp 特化\nFA4 逼近 98% 峰值', 'purple', 11)
        arrow(a, (0.26, 0.76), (0.26, 0.72), 'control')
        arrow(a, (0.74, 0.76), (0.74, 0.72), 'control')
        box(a, 0.04, 0.525, 0.92, 0.06, '定位：Helion 覆盖快速融合，CuTe DSL 攻坚极致核心库', 'gray', 11)

        a.add_patch(
            FancyBboxPatch(
                (0.02, 0.02),
                0.96,
                0.46,
                boxstyle='round,pad=0.003,rounding_size=0.012',
                facecolor='#FAFCFE',
                edgecolor='#CBD5E1',
                lw=1.0,
            )
        )
        text(a, 0.04, 0.45, 'B. 等式饱和（SmoothE）与并行图符号等价验证（TrainVerify）', 11.5, weight='bold')
        box(a, 0.04, 0.29, 0.28, 0.13, '1. 注入 e-graph\n并发代数重写\n保留等价类', 'blue', 11)
        box(a, 0.36, 0.29, 0.28, 0.13, '2. SmoothE 提取\n平滑代价寻优\n跳出局部极小', 'purple', 11)
        box(a, 0.68, 0.29, 0.28, 0.13, '3. TrainVerify\n血缘切分+SMT\n证跨卡等价', 'green', 11)
        arrow(a, (0.32, 0.355), (0.36, 0.355))
        arrow(a, (0.64, 0.355), (0.68, 0.355))
        box(a, 0.04, 0.15, 0.44, 0.10, '拦截算子 Bug：漏乘缩放因子 r', 'orange', 11)
        box(a, 0.52, 0.15, 0.44, 0.10, '拦截并行 Bug：TP/CP 漏加归约', 'orange', 11)
        box(a, 0.04, 0.04, 0.92, 0.08, '双层协同：等式饱和发掘融合空间 + 形式验证守住数学等价', 'teal', 11)
        out.save(f, 'ch05-helion-cute-dsl-and-egraph-equivalence-verification')

    return {
        'figures': [
            'ch05-torch-compile-dynamo-aotautograd-mincut-inductor-pipeline',
            'ch05-nanoflow-pod-attention-intra-gpu-three-pipeline-overlap',
            'ch05-helion-cute-dsl-and-egraph-equivalence-verification',
        ]
    }


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    _, family = configure_font()
    plt.rcParams.update({'font.family': [family, 'DejaVu Sans'], 'axes.unicode_minus': False, 'svg.hashsalt': 'ch05-h2'})
    layout_path = here / 'structure-layout-validation.json'
    existing = json.loads(layout_path.read_text(encoding='utf-8')) if layout_path.exists() else []
    out = Exporter(here)
    draw(out)
    new_names = {c['figure'] for c in out.checks}
    merged = [c for c in existing if c.get('figure') not in new_names] + out.checks
    layout_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out.checks, ensure_ascii=False, indent=2))
