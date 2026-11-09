#!/usr/bin/env python3
"""Book-size 30-chapter five-part roadmap using the shared figure style."""
from pathlib import Path
import sys, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figure_style import COL, STYLE, Exporter
from figure_style.typography import configure_font

_, family = configure_font()
plt.rcParams.update(STYLE)
plt.rcParams.update({'font.family': family, 'svg.hashsalt': 'ai-infra-book-roadmap'})

# Coordinates are points (420pt width x 564pt height) so label size and space can be checked at print size.
fig = plt.figure(figsize=(420 / 72, 564 / 72))
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, 420), ylim=(0, 564))
ax.axis('off')

rows = [
    (488, 72, 'blue', '第一部分  模型与负载（第 1–3 章）',
     ['1  初识 AI Infra', '2  模型架构', '3  推理与训练负载']),
    (400, 72, 'green', '第二部分  芯片与系统（第 4–7 章）',
     ['4  加速器架构', '5  算子与运行时', '6  超节点', '7  数据中心网络']),
    (291, 93, 'teal', '第三部分  推理、训练与端边云调度（第 8–12 章）',
     ['8  推理优化', '9  分布式推理', '10  训练系统', '11  资源调度与运行环境',
      '12  端边云协同']),
    (119, 156, 'orange', '第四部分  跨层协同、超大规模集群与科学负载（第 13–24 章）',
     ['13  AI 原生存储与数据底座', '14  模型与系统硬件协同设计',
      '15  吉瓦级数据中心与电力热学', '16  机密 AI 计算与安全沙箱',
      '17  多模态、视频 DiT 与世界模型', '18  大规模后训练与推理时扩展',
      '19  具身智能（VLA）与物理仿真', '20  十万卡集群可靠性与基准科学',
      '21  自主智能体与分层 KV 记忆', '22  全双工实时语音与原生全模态',
      '23  百亿亿次数据工程与合成工厂', '24  科学智能（AI4Science）底座']),
    (10, 93, 'purple', '第五部分  前沿生成范式、源码解剖与自演进系统（第 25–30 章）',
     ['25  生成式推荐与 GPU 向量库', '26  扩散语言模型、无分词与记忆网',
      '27  晶圆级 SRAM、微秒 AI 与三值化', '28  广域网去中心化训练与安全水印',
      '29  DeepSeek 开源全栈源码级解剖', '30  自演进 AI Infra 与三十章大收束']),
]

for y, h, color, title, chapters in rows:
    ax.add_patch(FancyBboxPatch(
        (4, y), 412, h, boxstyle='round,pad=0,rounding_size=4',
        facecolor=COL[color], edgecolor=COL['line'], linewidth=0.9
    ))
    ax.text(16, y + h - 17, title, fontsize=13.0, weight='medium', va='center')
    for i, label in enumerate(chapters):
        ax.text(18 + (i % 2) * 200, y + h - 40 - (i // 2) * 21, label, fontsize=11.0, va='center')

for top, bottom in [(488, 472), (400, 384), (291, 275), (119, 103)]:
    ax.add_patch(FancyArrowPatch(
        (210, top - 1), (210, bottom + 1), arrowstyle='-|>',
        mutation_scale=10, linewidth=1.0, color=COL['line']
    ))

out = Exporter(HERE)
out.save(fig, 'figure-1-book-roadmap')
(HERE / 'roadmap-layout-validation.json').write_text(json.dumps(out.checks, ensure_ascii=False, indent=2) + '\n')
