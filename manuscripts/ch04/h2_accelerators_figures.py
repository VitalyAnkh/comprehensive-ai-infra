"""Path H2 Ch04 figures: Big-Five heterogeneous AI accelerators & software stack portability."""
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
        # Fig 4-54: ch04-big-five-accelerators-roofline-and-microarchitecture-matrix
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(420 / 72, 7.2))
        f.subplots_adjust(left=0.15, right=0.96, bottom=0.09, top=0.93, hspace=0.54)
        for ax in (ax1, ax2):
            ax.spines[['top', 'right']].set_visible(False)

        chips = [
            ('Rubin GPU (NVFP4)', 22.0, 50.0, 2273, 'teal', (-4, 9), 'center', 'bottom'),
            ('MI450 (FP4)', 19.6, 40.0, 2041, 'purple', (-8, -7), 'right', 'top'),
            ('MI355X (MXFP4)', 8.0, 10.0, 1250, 'blue', (-7, 8), 'right', 'bottom'),
            ('TPU v7 (FP8)', 7.38, 4.614, 625, 'green', (9, 2), 'left', 'center'),
            ('Trainium3 (MXFP4)', 4.9, 2.52, 514, 'orange', (9, -4), 'left', 'top'),
            ('Ascend 950DT', 4.0, 2.007, 502, 'gray', (-10, 13), 'center', 'bottom'),
        ]
        bw_ref = np.logspace(np.log10(2.4), np.log10(38.0), 100)
        ax1.plot(bw_ref, bw_ref * 0.6, ls=':', lw=1.2, color='#94A3B8', zorder=2, label=r'等嵴点 $\beta = 600$ FLOP/B')
        ax1.plot(bw_ref, bw_ref * 2.0, ls='--', lw=1.2, color='#64748B', zorder=2, label=r'等嵴点 $\beta = 2{,}000$ FLOP/B')

        for name, bw_tbs, pflops, ridge, col, (dx_pt, dy_pt), ha_align, va_align in chips:
            ax1.scatter(
                [bw_tbs],
                [pflops],
                s=70,
                facecolor=COL[col],
                edgecolor=BORDER_COL[col],
                linewidth=1.7,
                zorder=5,
            )
            ax1.annotate(
                f'{name}\n$\\beta$={ridge} FLOP/B',
                xy=(bw_tbs, pflops),
                xytext=(dx_pt, dy_pt),
                textcoords='offset points',
                ha=ha_align,
                va=va_align,
                fontsize=11,
                weight='bold',
                color=TEXT_COL[col],
                zorder=6,
                bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.88),
            )
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set(
            xlim=(2.3, 42.0),
            ylim=(0.85, 165.0),
            xlabel='片外 HBM3e / HBM4 峰值带宽（TB/s，对数坐标）',
            ylabel='低比特算力（PFLOPS）',
        )
        ax1.set_xticks([3, 4, 6, 8, 12, 20, 30])
        ax1.set_yticks([1, 2, 5, 10, 20, 50, 100])
        ax1.xaxis.set_major_formatter(ScalarFormatter())
        ax1.yaxis.set_major_formatter(ScalarFormatter())
        ax1.xaxis.set_minor_formatter(NullFormatter())
        ax1.yaxis.set_minor_formatter(NullFormatter())
        ax1.minorticks_off()
        ax1.set_title('A  算例 4-8：五大异构加速器谱系的带宽—算力—机脊点坐标', loc='left', fontsize=12, weight='bold', pad=8)
        ax1.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1')
        ax1.legend(frameon=True, loc='upper left', fontsize=11)

        workloads = [
            '规则大矩阵\n稠密 GEMM',
            '长序列因果\nFlashAttention',
            '动态偏斜路由\nMoE 分组乘',
            '变长分页/跳层\n稀疏算子',
        ]
        util_systolic = [91, 78, 54, 48]
        util_simt = [89, 82, 79, 76]
        x_w = np.arange(len(workloads))
        w_bar = 0.36
        ax2.bar(
            x_w - w_bar / 2,
            util_systolic,
            width=w_bar,
            color=COL['green'],
            edgecolor=BORDER_COL['green'],
            lw=1.3,
            label='脉动阵列 + 静态 VMEM/SBUF (TPU/Ascend/Trn)',
        )
        ax2.bar(
            x_w + w_bar / 2,
            util_simt,
            width=w_bar,
            color=COL['blue'],
            edgecolor=BORDER_COL['blue'],
            lw=1.3,
            label='SIMT 张量核 + TMA/TMEM (NVIDIA/AMD)',
        )
        for xi, vs, vm in zip(x_w, util_systolic, util_simt):
            ax2.text(
                xi - w_bar / 2,
                vs + 2.5,
                f'{vs}%',
                ha='center',
                va='bottom',
                fontsize=11,
                weight='bold',
                color=TEXT_COL['green'],
            )
            ax2.text(
                xi + w_bar / 2,
                vm + 2.5,
                f'{vm}%',
                ha='center',
                va='bottom',
                fontsize=11,
                weight='bold',
                color=TEXT_COL['blue'],
            )
        ax2.set_xticks(x_w, workloads)
        ax2.set_xlim(-0.58, 3.58)
        ax2.set_ylim(0, 146.0)
        ax2.set_yticks([0, 25, 50, 75, 100])
        ax2.set_ylabel('有效算力实现率（%）', fontsize=11)
        ax2.set_title('B  脉动阵列与 SIMT 张量核在规则及动态负载下的有效利用率', loc='left', fontsize=12, weight='bold', pad=8)
        ax2.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E1')
        ax2.legend(frameon=True, loc='upper right', fontsize=11)
        out.save(f, 'ch04-big-five-accelerators-roofline-and-microarchitecture-matrix')

        # Fig 4-55: ch04-heterogeneous-software-stack-portability-and-amdahl-tail-trap
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(420 / 72, 7.2))
        f.subplots_adjust(left=0.15, right=0.96, bottom=0.09, top=0.93, hspace=0.54)
        for ax in (ax1, ax2):
            ax.spines[['top', 'right']].set_visible(False)

        alpha_pct = np.linspace(0.0, 20.0, 201)
        alpha = alpha_pct / 100.0
        eta_gemm = 95.0
        eta_r3 = eta_gemm / (1.0 + alpha * (3.0 - 1.0))
        eta_r65 = eta_gemm / (1.0 + alpha * (6.5 - 1.0))
        eta_r15 = eta_gemm / (1.0 + alpha * (15.0 - 1.0))

        ax1.plot(alpha_pct, eta_r3, color=BORDER_COL['green'], lw=2.2, ls='--', label=r'轻度未融合回退 ($r = 3.0\times$)')
        ax1.plot(alpha_pct, eta_r65, color=BORDER_COL['blue'], lw=2.5, label=r'算例 4-9 访存膨胀回退 ($r = 6.5\times$)')
        ax1.plot(alpha_pct, eta_r15, color=BORDER_COL['orange'], lw=2.2, ls='-.', label=r'主机同步/CPU 回退 ($r = 15.0\times$)')

        ax1.scatter([8.0, 8.0], [65.97, 44.81], s=62, facecolor=[COL['blue'], COL['orange']],
                    edgecolor=[BORDER_COL['blue'], BORDER_COL['orange']], lw=1.6, zorder=5)
        ax1.annotate(
            '92% 覆盖率陷阱 ($\\alpha=8\\%$):\n95.0% → 66.0% (-29.0 pp)',
            xy=(8.0, 65.97),
            xytext=(10.2, 60.5),
            arrowprops=dict(arrowstyle='->', color=BORDER_COL['blue'], lw=1.3),
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.88),
            fontsize=11,
            weight='bold',
            color=TEXT_COL['blue'],
        )
        ax1.annotate(
            '含主机同步 ($r=15\\times$):\n跌至 44.8%',
            xy=(8.0, 44.81),
            xytext=(1.2, 24.5),
            arrowprops=dict(arrowstyle='->', color=BORDER_COL['orange'], lw=1.3),
            fontsize=11,
            weight='bold',
            color=TEXT_COL['orange'],
        )
        ax1.set(
            xlim=(0.0, 20.0),
            ylim=(18.0, 118.0),
            xlabel=r'未融合长尾回退算子理想时间占比 $\alpha_{\mathrm{tail}}$（%）',
            ylabel=r'端到端实现率 $\eta_{\mathrm{eff}}$（%）',
        )
        ax1.set_yticks([20, 40, 60, 80, 100])
        ax1.set_title('A  算例 4-9：Amdahl 长尾放大与 92% 算子覆盖率陷阱', loc='left', fontsize=12, weight='bold', pad=8)
        ax1.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1')
        ax1.legend(frameon=True, loc='upper right', fontsize=11)

        stacks = [
            'CUDA +\nCUTLASS',
            'OpenXLA\n+ Pallas',
            'ROCm +\nHIP/CK',
            'CANN +\nAscend C',
            'Triton /\nTileLang',
        ]
        gemm_pct = [95, 93, 92, 91, 94]
        e2e_pct = [90, 85, 79, 77, 86]
        x_s = np.arange(len(stacks))
        w_s = 0.30
        off_s = 0.18
        ax2.bar(x_s - off_s, gemm_pct, width=w_s, color=COL['teal'], edgecolor=BORDER_COL['teal'], lw=1.2, label=r'核心矩阵乘峰值实现率 $\eta_{\mathrm{gemm}}$')
        ax2.bar(x_s + off_s, e2e_pct, width=w_s, color=COL['purple'], edgecolor=BORDER_COL['purple'], lw=1.2, label=r'全模型端到端实现率 $\eta_{\mathrm{eff}}$（含长尾）')
        for xi, vg, ve in zip(x_s, gemm_pct, e2e_pct):
            ax2.text(xi - 0.21, vg + 2.5, f'{vg}%', ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL['teal'])
            ax2.text(xi + 0.22, ve + 2.5, f'{ve}%', ha='center', va='bottom', fontsize=11, weight='bold', color=TEXT_COL['purple'])
        ax2.set_xticks(x_s, stacks)
        ax2.set_xlim(-0.6, 4.6)
        ax2.set_ylim(0, 146.0)
        ax2.set_yticks([0, 25, 50, 75, 100])
        ax2.set_ylabel('算力实现率（%）', fontsize=11)
        ax2.set_title('B  五大软件栈的局部 GEMM 效率与全模型端到端实现率对比', loc='left', fontsize=12, weight='bold', pad=8)
        ax2.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E1')
        ax2.legend(frameon=True, loc='upper right', fontsize=11)
        out.save(f, 'ch04-heterogeneous-software-stack-portability-and-amdahl-tail-trap')

    return {'figures': ['ch04-big-five-accelerators-roofline-and-microarchitecture-matrix',
                        'ch04-heterogeneous-software-stack-portability-and-amdahl-tail-trap']}


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    _, family = configure_font()
    plt.rcParams.update({'font.family': [family, 'DejaVu Sans'], 'axes.unicode_minus': False, 'svg.hashsalt': 'ch04-h2'})
    layout_path = here / 'teaching-layout-validation.json'
    existing = json.loads(layout_path.read_text(encoding='utf-8')) if layout_path.exists() else []
    out = Exporter(here)
    draw(out)
    new_names = {c['figure'] for c in out.checks}
    merged = [c for c in existing if c.get('figure') not in new_names] + out.checks
    layout_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out.checks, ensure_ascii=False, indent=2))
