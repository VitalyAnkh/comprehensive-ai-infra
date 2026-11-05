"""Figures for the book's three principles; use the existing book-size exporter."""
import matplotlib.pyplot as plt
import numpy as np
from figure_style import COL, BORDER_COL, STYLE, canvas, plot, text, box, card, badge, arrow

W = 15136811008
WEIGHTS = 16381470720
K = 2 * 36 * 8 * 128 * 2 * 8192
WORKSPACE = 2 * 2**30
CAPACITY = 24 * 10**9


def draw(chapter, out):
    with plt.rc_context(STYLE):
        if chapter == 1:
            f, a = canvas(5.2)
            text(a, 0.245, 0.94, '传统应用栈', 13.5, ha='center')
            text(a, 0.755, 0.94, '模型驱动的应用栈', 13.5, ha='center')
            left_stack = [
                ('代码与结构化数据', '确定性控制流与数据结构', 'orange'),
                ('编译器与操作系统', '系统调用、进程与文件抽象', 'purple'),
                ('程序指令顺序执行', 'CPU 流水线、分支预测与缓存', 'green'),
                ('处理器、内存与网络', '通用计算核、DRAM 与以太网', 'blue'),
            ]
            right_stack = [
                ('Agent 与任务编排', '提示词、工具调用与多轮工作流', 'orange'),
                ('模型服务接口', '批量调度、流式输出与语法约束', 'purple'),
                ('前向生成与上下文状态', '算子核执行、KV 缓存与前缀复用', 'green'),
                ('加速器、HBM 与超节点互联', '张量核心、高带宽显存与高速互联', 'blue'),
            ]
            for x, stack in [(0.03, left_stack), (0.54, right_stack)]:
                for i, (title, sub, c) in enumerate(stack):
                    y = 0.72 - i * 0.185
                    card(a, x, y, 0.43, 0.135, title, sub, c, title_size=11.5, body_size=11.0)
                    if i < 3:
                        arrow(a, (x + 0.215, y), (x + 0.215, y - 0.05), color=c)
            arrow(a, (0.54, 0.79), (0.46, 0.61), kind='control', color='purple')
            badge(a, 0.50, 0.065, '外部工具与沙箱程序仍由宿主操作系统执行', 'gray', size=11.0)
            out.save(f, 'figure-1-programmability')
        if chapter == 2:
            f, a = plot(3.8, left=0.08, bottom=0.22)
            vals = [4 * K / 1e9, WORKSPACE / 1e9, (CAPACITY - 4 * K - WORKSPACE) / 1e9]
            left = 0
            for v, c in zip(vals, ['green', 'gray', 'blue']):
                a.barh(0.5, v, left=left, height=0.26, color=COL[c], edgecolor=BORDER_COL[c], linewidth=1.2)
                left += v
            a.set(
                xlim=(0, 24.5),
                ylim=(0, 1.48),
                yticks=[],
                xticks=[0, 4, 8, 12, 16, 20, 24],
                xlabel='RTX 4090 的 24 GB 容量分配（GB）',
            )
            for x, y, label, target, c in [
                (2.4, 1.20, '4 条请求 KV\n4.83 GB', 2.4, 'green'),
                (7.2, 0.96, '工作区\n2.15 GB', 5.9, 'gray'),
                (16.2, 1.20, '权重可用余量\n17.02 GB', 15.5, 'blue'),
            ]:
                a.annotate(
                    label,
                    xy=(target, 0.64),
                    xytext=(x, y),
                    ha='center',
                    va='center',
                    fontsize=11.5,
                    weight='medium',
                    color=COL['ink'],
                    bbox=dict(boxstyle='round,pad=0.22', facecolor=COL[c], edgecolor=BORDER_COL[c], lw=0.9),
                    arrowprops=dict(arrowstyle='-|>', color=BORDER_COL[c], lw=1.1),
                )
            a.text(12, 0.14, 'BF16 参数上界 ≈ 85.1 亿', ha='center', fontsize=12.5, weight='bold', color=COL['ink'])
            out.save(f, 'figure-2-reverse-budget')
        if chapter == 4:
            f, a = canvas(4.9)
            rows = [
                ('1. 既有加速器硬件边界', 'HBM 容量、访存带宽、片上 SRAM 与互联拓扑上限', 'blue'),
                ('2. 模型结构与系统软件适配', 'KV 压缩（GQA/MLA）、算子分块融合与多维并行切分', 'green'),
                ('3. 暴露持续的执行瓶颈', '长上下文访存墙、跨节点通信停顿与尾延迟毛刺', 'orange'),
                ('4. 定义下一代硬件与新模型候选', '微缩低比特（MXFP4）、专用通路与软硬件协同重设计', 'purple'),
            ]
            for i, (title, desc, c) in enumerate(rows):
                y = 0.77 - i * 0.225
                card(a, 0.06, y, 0.84, 0.165, title, desc, c, title_size=11.5, body_size=11.0)
                if i < 3:
                    arrow(a, (0.48, y), (0.48, y - 0.06), color=c)
            arrow(a, (0.90, 0.175), (0.90, 0.855), kind='control', rad=0.22, color='purple')
            out.save(f, 'figure-4-codesign-loop')

            f, a = canvas(4.6)
            text(a, 0.04, 0.94, 'A. 权重与状态共用 HBM 接口', 13.5)
            card(a, 0.04, 0.64, 0.39, 0.20, 'HBM 统一显存', '模型权重 W + 动态 KV', 'blue')
            card(a, 0.66, 0.64, 0.30, 0.20, '张量计算阵列', '矩阵乘加与注意力', 'orange')
            arrow(a, (0.43, 0.74), (0.66, 0.74), label='W + B·K', color='blue')
            text(a, 0.04, 0.47, 'B. 独立的只读权重通路（ROM + HBM）', 13.5)
            card(a, 0.04, 0.25, 0.39, 0.15, '片上/专用 ROM', '只读模型权重 W', 'blue')
            card(a, 0.04, 0.04, 0.39, 0.15, 'HBM 显存', '可写 KV 状态 B·K', 'green')
            card(a, 0.66, 0.11, 0.30, 0.23, '张量计算阵列', '双路并发取数执行', 'orange')
            arrow(a, (0.43, 0.325), (0.66, 0.27), color='blue')
            arrow(a, (0.43, 0.115), (0.66, 0.18), color='green')
            out.save(f, 'figure-4-rom-paths')
        if chapter == 8:
            f, a = plot(3.8, left=0.16, bottom=0.19)
            b = np.arange(1, 33)
            a.plot(b, (W / b + K) / 1e9, color='#2B6CB0', lw=2.2, label='传统 HBM：W/B + K')
            a.plot(b, np.full_like(b, K, dtype=float) / 1e9, color='#2F855A', lw=2.2, label='独立 ROM：仅读取 K')
            for batch in [1, 16]:
                v = (W / batch + K) / 1e9
                a.scatter([batch], [v], color='#2B6CB0', s=28, zorder=4)
                a.annotate(f'{v:.2f} GB', (batch, v), xytext=(8, 3), textcoords='offset points', fontsize=11, weight='medium')
            a.set(
                xlim=(0, 35),
                ylim=(0, 20.5),
                xticks=[1, 8, 16, 24, 32],
                xlabel='批大小 B',
                ylabel='每输出 token 的 HBM 读取（GB）',
            )
            a.grid(alpha=0.35)
            a.legend(loc='upper right')
            out.save(f, 'figure-8-batch-counterfactual')

            f, a = canvas(4.4)
            rows = [
                (0.72, '末尾追加', [(8, '8K 历史前缀直接复用', 'blue'), (1, '', 'orange')]),
                (0.45, '改写开头', [(9, '9K 全量重新计算 KV', 'orange')]),
                (0.18, '压缩总结', [(2, '2K 摘要', 'green'), (1, '', 'orange')]),
            ]
            for y, label, segments in rows:
                text(a, 0.03, y + 0.065, label, 12, weight='bold')
                x = 0.25
                for length, lab, col in segments:
                    width = length * 0.074
                    box(a, x, y, width, 0.13, lab, col, 11)
                    x += width
                if label in ('末尾追加', '压缩总结'):
                    badge(a, x - 0.037, y - 0.065, '1K 新增', 'orange', size=11)
                if label == '压缩总结':
                    badge(a, 0.68, y + 0.065, '另计一次摘要生成开销', 'gray', size=11)
            text(a, 0.5, 0.94, '同一段多轮历史在三种上下文维护策略下的计算范围', 13.5, ha='center')
            out.save(f, 'figure-8-context-edits')
        if chapter == 10:
            f, a = canvas(4.3)
            text(a, 0.04, 0.94, 'A. 固定权重版本的在线推理服务', 13.5)
            card(a, 0.04, 0.65, 0.38, 0.19, '只读权重存储（ROM/HBM）', '固定模型参数 v0', 'blue')
            card(a, 0.64, 0.65, 0.32, 0.19, '解码生成引擎', '维护可写请求 KV', 'green')
            arrow(a, (0.42, 0.745), (0.64, 0.745), label='只读复用', color='blue')
            text(a, 0.04, 0.48, 'B. 强化学习（RL）持续更新的策略闭环', 13.5)
            card(a, 0.04, 0.18, 0.26, 0.20, '训练引擎', '反向传播更新参数', 'orange')
            card(a, 0.38, 0.18, 0.26, 0.20, '可写权重缓冲', '版本演进 v0 → v1', 'blue')
            card(a, 0.72, 0.18, 0.24, 0.20, '采样引擎', '切换至新权重 v1', 'green')
            arrow(a, (0.30, 0.28), (0.38, 0.28), color='orange')
            arrow(a, (0.64, 0.28), (0.72, 0.28), color='blue')
            badge(a, 0.50, 0.065, '每次策略更新后同步发布新权重，保证采样与训练策略一致', 'purple', size=11.0)
            out.save(f, 'figure-10-weight-update')
        if chapter == 11:
            f, a = plot(4.4, left=0.24, bottom=0.17)
            cold = 2 + 2**31 / (25e9 / 8)
            for y, v, c in [(4, 6, 'blue'), (3, cold, 'green'), (1, 1, 'blue'), (0, cold, 'green')]:
                a.barh(y, v, height=0.55, color=COL[c], edgecolor=BORDER_COL[c], linewidth=1.15)
                a.text(v + 0.15, y, f'{v:.1f} 秒' if v != int(v) else f'{int(v)} 秒', va='center', fontsize=11, weight='medium')
            a.axhline(2.2, color=COL['line'], lw=0.8, ls='--')
            a.annotate('', xy=(cold, 1.5), xytext=(1, 1.5), arrowprops=dict(arrowstyle='<->', color=BORDER_COL['orange'], lw=1.3))
            a.text(3.6, 1.66, f'暴露等待约 {cold - 1:.1f} 秒', ha='left', fontsize=11, weight='bold', color='#7C2D12')
            a.set(
                yticks=[4, 3, 1, 0],
                yticklabels=['常速模型思考', '沙箱冷启动', '提速后模型', '沙箱冷启动'],
                xlim=(0, 8.2),
                ylim=(-0.65, 4.7),
                xlabel='从模型开始输出工具调用起计时（秒）',
            )
            a.grid(axis='x', alpha=0.35)
            out.save(f, 'figure-11-environment-overlap')
        if chapter == 12:
            f, a = plot(3.6, left=0.24, bottom=0.21)
            for y, v in [(2, 8), (1, 0.8), (0, 0)]:
                a.barh(y, v, height=0.55, color=COL['blue'], edgecolor=BORDER_COL['blue'], linewidth=1.15, label='模型推理阶段' if y == 2 else None)
                a.barh(y, 2, left=v, height=0.55, color=COL['orange'], edgecolor=BORDER_COL['orange'], linewidth=1.15, label='外部串行阶段' if y == 2 else None)
                a.text(v + 2 + 0.16, y, f'{v + 2:g} 秒', va='center', fontsize=11.5, weight='medium')
            a.set(
                yticks=[2, 1, 0],
                yticklabels=['原任务', '模型快 10 倍', '理想下界'],
                xlim=(0, 12.5),
                ylim=(-0.7, 3.0),
                xlabel='完整任务端到端时间（秒）',
            )
            f.subplots_adjust(top=0.83)
            a.grid(axis='x', alpha=0.35)
            a.legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.45, 1.18), columnspacing=1.5, handlelength=1.4)
            out.save(f, 'figure-12-task-counterfactual')
