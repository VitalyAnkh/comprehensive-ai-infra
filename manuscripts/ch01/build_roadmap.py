#!/usr/bin/env python3
"""Book-size 24-chapter five-part roadmap using the shared figure style."""
from pathlib import Path
import sys
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figure_style import COL, BORDER_COL, TEXT_COL, STYLE, Exporter
from figure_style.typography import configure_font

_, family = configure_font()
plt.rcParams.update(STYLE)
plt.rcParams.update({'font.family': family, 'svg.hashsalt': 'ai-infra-book-roadmap'})

# Coordinates are points (420pt width x 596pt height) so label size and space are checked at print size.
fig = plt.figure(figsize=(420 / 72, 596 / 72))
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, 420), ylim=(0, 596))
ax.axis('off')

rows = [
    (512, 76, 'blue', '第一部分  模型与负载（第 1–3 章）', '收束：第 3 章',
     ['1  初识 AI Infra', '2  模型架构', '3  推理与训练负载']),
    (418, 76, 'green', '第二部分  芯片与硬件系统（第 4–7 章）', '收束：第 7 章',
     ['4  加速器架构', '5  算子与运行时', '6  超节点', '7  数据中心网络']),
    (304, 96, 'teal', '第三部分  推理、训练与调度系统（第 8–12 章）', '收束：第 12 章',
     ['8  推理优化', '9  分布式推理', '10  训练系统', '11  资源调度与运行环境',
      '12  端边云协同']),
    (190, 96, 'orange', '第四部分  数据、集群底座与后训练系统（第 13–17 章）', '收束：第 17 章',
     ['13  存储系统与数据工程', '14  智算中心与电力热学系统',
      '15  集群可靠性与基准评测', '16  机密计算与可信安全系统',
      '17  后训练与强化学习系统']),
    (56, 116, 'purple', '第五部分  多模态、领域负载与前沿架构（第 18–24 章）', '收束：第 24 章',
     ['18  自主智能体系统', '19  多模态生成与实时语音',
      '20  具身智能与物理仿真', '21  推荐检索与向量数据库',
      '22  科学计算智能系统', '23  软硬件协同与新型架构',
      '24  全栈跨层优化与自演进系统']),
]

for y, h, color, title, cap_tag, chapters in rows:
    ax.add_patch(FancyBboxPatch(
        (4, y), 412, h, boxstyle='round,pad=0,rounding_size=5',
        facecolor=COL[color], edgecolor=BORDER_COL[color], linewidth=1.1
    ))
    ax.text(14, y + h - 18, title, fontsize=12.0, weight='bold', va='center', color=COL['ink'])
    ax.text(406, y + h - 18, cap_tag, fontsize=11.0, weight='bold', va='center', ha='right', color=TEXT_COL[color])
    for i, label in enumerate(chapters):
        ax.text(16 + (i % 2) * 198, y + h - 41 - (i // 2) * 20, label, fontsize=11.0, va='center', color=COL['ink'])

for top, bottom in [(512, 494), (418, 400), (304, 286), (190, 172)]:
    ax.add_patch(FancyArrowPatch(
        (210, top - 1), (210, bottom + 1), arrowstyle='-|>',
        mutation_scale=10, linewidth=1.2, color=COL['ink']
    ))

ax.add_patch(FancyBboxPatch(
    (4, 8), 412, 34, boxstyle='round,pad=0,rounding_size=5',
    facecolor=COL['gray'], edgecolor=BORDER_COL['gray'], linewidth=1.0
))
ax.text(
    210, 25,
    '全栈定量主线：Qwen3-8B + DeepSeek-V3/V4.1 · calc.py 24 章复算',
    fontsize=11.0, weight='bold', ha='center', va='center', color=COL['ink']
)

out = Exporter(HERE)
out.save(fig, 'figure-1-book-roadmap')
(HERE / 'roadmap-layout-validation.json').write_text(
    json.dumps(out.checks, ensure_ascii=False, indent=2) + '\n'
)
