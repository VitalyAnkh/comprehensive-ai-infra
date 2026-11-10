#!/usr/bin/env python3
"""Build all 16 publication-grade figures (420pt) and validation manifests for Chapter 26."""
from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figure_style import COL, STYLE, Exporter, arrow, box, canvas, card, text
from preview_output import preview_path

plt.rcParams.update(STYLE)


def sha256(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b'version https://git-lfs.github.com/spec/v1'):
        m = re.search(rb'oid sha256:([0-9a-f]{64})', raw)
        if m:
            return m.group(1).decode('ascii')
    return hashlib.sha256(raw).hexdigest()


def draw_dual_band(
    ex: Exporter,
    stem: str,
    top_title: str,
    top_color: str,
    top_boxes: list[tuple[str, str]],
    bot_title: str,
    bot_color: str,
    bot_boxes: list[tuple[str, str]],
    footer: str,
) -> None:
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, top_title, color=top_color)
    if len(top_boxes) == 3:
        xs = [(0.028, 0.296), (0.352, 0.296), (0.676, 0.296)]
        for (x, w), (lbl, role) in zip(xs, top_boxes):
            box(ax, x, 0.58, w, 0.24, lbl, role, size=11)
        arrow(ax, (0.324, 0.70), (0.352, 0.70))
        arrow(ax, (0.648, 0.70), (0.676, 0.70))
    else:
        xs = [(0.03, 0.45), (0.52, 0.45)]
        for (x, w), (lbl, role) in zip(xs, top_boxes):
            box(ax, x, 0.58, w, 0.24, lbl, role, size=11)
        arrow(ax, (0.48, 0.70), (0.52, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, bot_title, color=bot_color)
    if len(bot_boxes) == 3:
        xs = [(0.028, 0.296), (0.352, 0.296), (0.676, 0.296)]
        for (x, w), (lbl, role) in zip(xs, bot_boxes):
            box(ax, x, 0.13, w, 0.20, lbl, role, size=11)
        arrow(ax, (0.324, 0.23), (0.352, 0.23))
        arrow(ax, (0.648, 0.23), (0.676, 0.23))
    else:
        xs = [(0.03, 0.45), (0.52, 0.45)]
        for (x, w), (lbl, role) in zip(xs, bot_boxes):
            box(ax, x, 0.13, w, 0.20, lbl, role, size=11)
        arrow(ax, (0.48, 0.23), (0.52, 0.23))

    text(ax, 0.50, 0.06, footer, size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, stem)


def build_chapter() -> None:
    ex = Exporter(HERE)

    draw_dual_band(
        ex,
        'ch26-autoregressive-vs-diffusion-llm-roofline-phase-transition',
        '32B 模型自回归解码（100 tok/s）vs. 块扩散去噪（592.6 tok/s）',
        'blue',
        [
            ('① 自回归逐词元解码\n算术强度 1 FLOP/B\n512 步耗 5.12 s', 'orange'),
            ('② 块扩散 K=64 并行\n强度升至 64 FLOP/B\n仅 80 次前向传播', 'blue'),
            ('③ 权重读取减 6.4x\n总耗时 0.864 s\n提速 5.93 倍', 'teal'),
        ],
        '为什么扩散大语言模型（Mercury / LLaDA）能突破内存带宽墙？',
        'teal',
        [
            ('逐词元自回归的内存带宽死锁：\n生成 512 词元需读 512 次权重\nTensor Core 利用率不足 0.6%', 'orange'),
            ('块掩码并行去噪（K=64, T=10）：\n每次前向并行炼制 64 个词元\n8B 模型实测飙升至 1,420 tok/s', 'teal'),
        ],
        '将 512 次串行访存折算为 80 次高算术强度矩阵乘，实现 5.93 倍解码跃迁',
    )

    draw_dual_band(
        ex,
        'ch26-block-diffusion-semi-autoregressive-kv-cache-and-mask',
        '半自回归块扩散（Block Diffusion）块间因果与块内双向注意力',
        'purple',
        [
            ('① 历史块 KV 固化\n已完成块冻结缓存\n免除全序列重算', 'blue'),
            ('② 当前块 K=64 掩码\n块内开启双向注意\n10 步迭代去噪', 'purple'),
            ('③ 混合阶梯掩码核\n因果前缀+双向当前\n兼顾流式与并行', 'teal'),
        ],
        '全序列双向扩散（LLaDA）vs. 半自回归块扩散（Block Diffusion）',
        'blue',
        [
            ('纯全序列双向扩散的局限：\n每步去噪均需重算全部前缀\n长上下文下二次方算力爆炸', 'orange'),
            ('块间自回归 + 块内并行去噪：\n完美复用 PagedAttention 缓存\n支持任意长度流式低延迟输出', 'teal'),
        ],
        '融合自回归 KV 缓存复用与扩散块内并行去噪，打通长文本生产部署',
    )

    draw_dual_band(
        ex,
        'ch26-dynamic-confidence-threshold-remasking-scheduler',
        '动态置信度重掩码（gamma=0.92）与块级提前终止（算例 26-2）',
        'teal',
        [
            ('① 词元后验概率评估\np_max >= 0.92 锁定\n低置信位重回 MASK', 'blue'),
            ('② 代码模板快速收敛\n仅需 4~6 步全锁定\n平均 6.5 步/块', 'purple'),
            ('③ 前向由 80 降 52\n节省 35.0% 去噪步\n吞吐达 911.7 tok/s', 'teal'),
        ],
        '固定去噪步数（T=10）vs. 自适应置信度提前退出对比',
        'blue',
        [
            ('固定步数调度的算力浪费：\n简单语法词元在第 3 步已收敛\n后续 7 步仍在重复无效重算', 'orange'),
            ('置信度驱动动态早退控制器：\n全块锁定即刻转入下一块生成\n较自回归基线提速达 9.12 倍', 'teal'),
        ],
        '按词元确定性动态分配去噪步数，削减 35% 冗余前向并守住无损生成质量',
    )

    draw_dual_band(
        ex,
        'ch26-dllm-speculative-parallel-verification-hybrid-engine',
        '扩散草稿批量提议 + 自回归单步精确校验混合推理引擎',
        'blue',
        [
            ('① 扩散模型块提议\n3 步并行出 32 词元\n摆脱串行草稿延迟', 'blue'),
            ('② 自回归树状验证\n1 次前向并行打分\n精准守住分布无偏', 'purple'),
            ('③ 拒绝点重掩码修补\n保留后部高置信段\n突破首错截断瓶颈', 'teal'),
        ],
        '传统小模型串行投机采样 vs. 扩散并行草稿混合引擎对比',
        'teal',
        [
            ('传统自回归小模型草稿瓶颈：\n草稿长度 K>8 时串行耗时过长\n且首错之后词元被全部丢弃', 'orange'),
            ('dLLM 并行提议 + 局部修复：\nO(1) 步生成长草稿块供主模验证\n有效接受长度提升至 14.5 词元', 'teal'),
        ],
        '用扩散并行去噪消除投机采样草稿端串行停顿，实现跨范式协同加速',
    )

    draw_dual_band(
        ex,
        'ch26-bpe-tokenizer-bottleneck-vs-blt-byte-entropy-patching',
        'Byte Latent Transformer（BLT）字节条件熵在线动态分块',
        'purple',
        [
            ('① 轻量字节熵探针\n100M 模型算 H(b_t)\n阈值 tau=1.35 bits', 'blue'),
            ('② 熵峰处切分新块\n常规 4.5 B/Patch\n模板拉长至 12.0 B', 'purple'),
            ('③ 砍掉 256k 大词表\n省 4.194 GB 显存\n端到端省 52.4% 算力', 'teal'),
        ],
        '固定 BPE 静态分词（256k 词表）vs. BLT 字节级无分词架构对比',
        'blue',
        [
            ('BPE 大词表的显存与算力税：\n输入输出投影占 4.194 GB 显存\n对可预测代码空白浪费同等算力', 'orange'),
            ('BLT 按信息熵动态分配算力：\n仅在不确定性跃升处触发重算\n骨干序列长度压缩 2.67 倍', 'teal'),
        ],
        '告别固定词表偏见与 4.19 GB 投影税，将每个 FLOP 精准投向高信息熵边界',
    )

    draw_dual_band(
        ex,
        'ch26-blt-local-byte-encoder-global-latent-cross-attention',
        'BLT 三段式架构：局部字节编码—全局潜空间—局部字节解码',
        'teal',
        [
            ('① 局部字节编码器\n滑动窗口聚合字节\n交叉注意汇入块表征', 'blue'),
            ('② 全局潜空间骨干\n仅在稀疏 Patch 运行\n承接 85% 参数容量', 'purple'),
            ('③ 局部字节解码器\n结合潜向量恢复字节\n输出仅 256 维投影', 'teal'),
        ],
        '为什么 BLT 能同时兼顾字节级细粒度鲁棒性与词元级长上下文吞吐？',
        'purple',
        [
            ('纯字节级自回归（ByT5）死穴：\n序列长度膨胀 4.5 倍触发 O(L^2)\n长文本下注意力延迟彻底失控', 'orange'),
            ('局部轻模块 + 全局重骨干解耦：\n字节层仅做短距窗口特征提取\n昂贵长程注意力运行在短 Patch', 'teal'),
        ],
        '用非对称交叉注意力桥接原始字节流与高维潜空间，兼得无分词与高吞吐',
    )

    draw_dual_band(
        ex,
        'ch26-blt-ragged-patch-packing-and-load-balancing',
        'BLT 变长动态分块（1~16 B）非规则交叉注意力（Ragged Kernel）',
        'blue',
        [
            ('① 前缀和偏移数组\ncu_seqlens 零填充\n紧凑拼接变长字节块', 'blue'),
            ('② 32-Byte 块对齐\n分块跨度调度 Warp\n消除尾块空转气泡', 'purple'),
            ('③ 批内长度重排均衡\n平衡各 SM 字节负载\n核利用率达 91.5%', 'teal'),
        ],
        '矩形最大长度填充（Padding 浪费 62%）vs. Ragged 紧凑打包内核',
        'teal',
        [
            ('朴素按最长 Patch 矩形填充：\n对齐至 16 B 产生逾 60% 零填充\n动态分块的算力收益被完全抵消', 'orange'),
            ('FlashAttention 式变长索引：\n按 cu_seqlens_patch 精确寻址\n零无效计算完成跨粒度池化', 'teal'),
        ],
        '以一维紧凑内存布局与前缀和间接寻址驯服动态字节分块的硬件不规则性',
    )

    draw_dual_band(
        ex,
        'ch26-titans-short-term-attention-and-long-term-neural-memory',
        'Google Titans 双系统架构：短期精确注意力 + 长期神经记忆',
        'purple',
        [
            ('① 短期滑动窗口注意\n精确捕捉局部语法\n窗口 W=4,096 词元', 'blue'),
            ('② 神经长期记忆模块\n片上快速权重 W_t\n惊奇度动量在线更新', 'purple'),
            ('③ 持久记忆前缀门控\n三支路融合注入残差\n支持 200 万上下文', 'teal'),
        ],
        '将推理期长上下文记忆定义为“测试时在线梯度下降（TTT）”',
        'blue',
        [
            ('传统线性 RNN / SSM 表达瓶颈：\n固定衰减线性状态容量受限\n超长文本大海捞针召回率退化', 'orange'),
            ('惊奇度驱动多层感知机记忆：\n高梯度反常事实获得强动量写入\n权重衰减门自适应腾出记忆容量', 'teal'),
        ],
        '梯度即惊奇度，把百万词元历史动态压入恒定体积的片上神经连接权重',
    )

    draw_dual_band(
        ex,
        'ch26-ttt-token-by-token-vs-chunkwise-dual-form-tensor-core',
        'TTT 快速权重更新：逐词元串行（1.5）vs. 分块对偶（48 FLOP/B）',
        'teal',
        [
            ('① 逐词元串行更新\n每步读写 128x128\n吞吐仅 42k tok/s', 'orange'),
            ('② 分块对偶 b=64\n聚合成矩阵乘 MMA\n强度升至 48 FLOP/B', 'blue'),
            ('③ 吞吐 1,033k tok/s\n算子提速 24.60 倍\n吃满 Tensor Core', 'teal'),
        ],
        '如何将测试时串行梯度下降转化为 Tensor Core 分块矩阵乘？',
        'purple',
        [
            ('逐词元在线 SGD 的硬件困局：\n64 步产生 64 次寄存器写回依赖\n无法调用矩阵乘单元陷入算力饥饿', 'orange'),
            ('Chunk-wise Dual Form 对偶化：\n块内梯度聚合为 [128,64]x[64,128]\n单次片上归约跑出 24.6 倍加速', 'teal'),
        ],
        '利用块内线性可加性将逐词元外积折叠为分块 GEMM，打通测试时训练算力',
    )

    draw_dual_band(
        ex,
        'ch26-titans-2m-context-constant-memory-vs-kv-cache-growth',
        '200 万词元显存对比：Softmax KV Cache vs. Titans 恒定记忆',
        'blue',
        [
            ('① 标准 MHA KV 缓存\n2M 词元占 1.02 TB\n撑爆单机 8 卡 HBM', 'orange'),
            ('② GQA/MLA 线性增长\n随长度仍涨至百 GB\n解码带宽持续恶化', 'purple'),
            ('③ Titans 快速权重\n恒定 16.0 MB/层\nO(1) 显存零膨胀', 'teal'),
        ],
        'O(L) 键值缓存线性膨胀墙 vs. O(1) 测试时快速权重恒定内存',
        'teal',
        [
            ('追加式 KV 缓存的物理极限：\n每增 1 词元即永久占用显存带宽\n百万上下文下单请求需多机分片', 'orange'),
            ('参数化状态记忆（16 MB/层）：\n80 层总记忆状态仅占 1.28 GB\n第 200 万词元解码延迟与首词相同', 'teal'),
        ],
        '将历史序列压缩为固定维度的可微分参数矩阵，彻底终结长上下文显存墙',
    )

    draw_dual_band(
        ex,
        'ch26-coconut-continuous-latent-thought-zero-copy-loop',
        'Coconut 连续潜空间思维链（c=3）零拷贝回环反馈（算例 26-5）',
        'purple',
        [
            ('① 省去 LM Head 投影\n免除 256k Softmax\n消除词表采样同步', 'blue'),
            ('② 末层隐藏态直回\nGPU 显存内指针传递\n1 潜步抵 3 离散步', 'purple'),
            ('③ 300 步压至 100 步\n耗时 3000->924 ms\n端到端提速 3.25 倍', 'teal'),
        ],
        '显式离散思维链（CoT 采样）vs. 连续潜空间思维链（Coconut）',
        'blue',
        [
            ('显式离散 CoT 的信息与系统损耗：\n4096 维丰富表征被坍缩为单词 ID\n300 步耗费 3,000 ms 带宽时间', 'orange'),
            ('连续潜向量原位回环推理：\n全维度保留多解叠加态概率假设\n推理延迟削减 69.2% (924 ms)', 'teal'),
        ],
        '不在中间思考步强行坍缩为人类自然语言词元，在连续高维流形中全速推理',
    )

    draw_dual_band(
        ex,
        'ch26-coconut-latent-tree-breadth-first-parallel-expansion',
        '连续潜空间隐式波束搜索与多分支树状并行验证',
        'teal',
        [
            ('① 潜向量叠加态表征\n单状态编码多条路径\n隐式执行广度优先', 'blue'),
            ('② 分叉点微扰扩展\n生成 B=4 潜空间分支\n共享前缀 KV 缓存', 'purple'),
            ('③ 价值头在线剪枝\n剔除低分推理死胡同\n转换率升至 92%', 'teal'),
        ],
        '离散词元级思维树（ToT）vs. 潜空间叠加态并行展开对比',
        'purple',
        [
            ('离散思维树搜索的显存风暴：\n每条分支需展开数十个词元序列\nKV 缓存随树深呈指数级爆炸', 'orange'),
            ('潜空间紧凑节点树状批处理：\n每个逻辑推理步仅占 1 个槽位\n同批并行探索 4 倍以上候选路径', 'teal'),
        ],
        '依托连续潜向量的高维叠加特性，以极小 KV 开销实现树状并行推演',
    )

    draw_dual_band(
        ex,
        'ch26-mixture-of-depths-token-routing-and-warp-divergence',
        'Mixture-of-Depths（MoD，rho=0.25）动态跳层与 Warp 发散',
        'blue',
        [
            ('① 轻量路由打分器\n线性投影选 Top-25%\n75% 词元残差直通', 'blue'),
            ('② 未紧缩掩码陷阱\n32 线程混杂跳层点\nWarp 效率仅 58.1%', 'orange'),
            ('③ 硬件算力未减反增\nSIMT 空槽浪费周期\n理论省 75% 实省 18%', 'purple'),
        ],
        '为什么直接用布尔掩码跳过词元无法在 GPU 上兑现理论加速比？',
        'teal',
        [
            ('SIMT 锁步执行的物理铁律：\n只要 Warp 内有 1 个词元需计算\n其余 31 个跳过线程也必须陪跑', 'orange'),
            ('破局之道：物理流紧缩重组\n在进入重算子前剔除直通词元\n将活跃词元紧密拼成满载矩阵块', 'teal'),
        ],
        '揭示算法级稀疏（FLOPs 削减）与硬件级规整（SIMT 锁步）的核心矛盾',
    )

    draw_dual_band(
        ex,
        'ch26-stream-compaction-gather-scatter-for-early-exit-layers',
        'MoD 硬件流紧缩（Stream Compaction）：Gather-GEMM-Scatter',
        'purple',
        [
            ('① Warp 投票前缀和\n位掩码硬件指令计数\n纳秒级算紧缩索引', 'blue'),
            ('② Gather 紧凑张量\n[N,d] 压为 [0.25N,d]\nWarp 满载达 96.4%', 'purple'),
            ('③ Scatter 残差归位\n单层实测提速 3.32x\n重排开销仅占 3.8%', 'teal'),
        ],
        '原地布尔掩码执行（58.1%）vs. 流紧缩 Gather/Scatter（96.4%）',
        'blue',
        [
            ('无紧缩稀疏执行的带宽浪费：\n矩阵乘仍按原长 N 启动线程块\n大量线程空读零权重浪费带宽', 'orange'),
            ('前缀和流紧缩连续内存重排：\nTensor Core 仅处理 25% 稠密行\n兑现 3.32 倍端到端单层加速', 'teal'),
        ],
        '用微秒级片上前缀和 Gather/Scatter 换取 Tensor Core 96.4% 满载率',
    )

    draw_dual_band(
        ex,
        'ch26-post-autoregressive-architectures-throughput-latency-pareto',
        '后自回归四大范式（dLLM / BLT / Titans / Coconut）帕累托前沿',
        'teal',
        [
            ('① 解码吞吐维突破\ndLLM 块并行 5.93x\n置信早退达 9.12x', 'blue'),
            ('② 粒度与深度维突破\nBLT 省 52.4% 算力\nCoconut 提速 3.25x', 'purple'),
            ('③ 上下文长度维突破\nTitans 24.6x 加速\n2M 序列 16 MB/层', 'teal'),
        ],
        '打破“逐词元、固定词表、全层等深、O(L) 缓存”四大传统枷锁',
        'purple',
        [
            ('传统自回归 Transformer 困局：\n空间（词表/层深）与时间（逐词）\n均按最坏均匀网格静态分配算力', 'orange'),
            ('后自回归自适应基础设施：\n沿时间步、字节熵、上下文与层深\n实现四维算力与显存按需分配', 'teal'),
        ],
        '从静态均匀网格走向多维动态自适应，开启千词元每秒与百万上下文时代',
    )

    draw_dual_band(
        ex,
        'ch26-next-gen-foundation-model-infrastructure-synthesis',
        '第 26 章全景：扩散语言模型与字节级动态分块（上）',
        'blue',
        [
            ('26.1 扩散大语言模型 dLLM：\nBlock Diffusion 提速 5.93 倍\n置信度重掩码早退达 911.7 tok/s', 'blue'),
            ('26.2 字节无分词架构 BLT：\n条件熵动态切分省 4.194 GB 词表\nRagged 核削减 52.4% 端到端算力', 'purple'),
        ],
        '第 26 章全景：测试时记忆网络与连续潜空间推理（下）',
        'teal',
        [
            ('26.3 测试时记忆 Titans/TTT：\n分块对偶矩阵乘提速 24.60 倍\n2M 上下文保持 16.0 MB/层恒定', 'blue'),
            ('26.4 潜空间与动态深度路由：\nCoconut 零拷贝回环提速 3.25 倍\nMoD 流紧缩恢复 96.4% Warp 效率', 'teal'),
        ],
        '以块并行、熵分块、分块对偶与流紧缩四大算子原语支撑后自回归模型革命',
    )

    outputs = ex.finish()

    fig_index = [
        {
            "figure_id": "图 26-1",
            "number": 1,
            "file": "ch26-autoregressive-vs-diffusion-llm-roofline-phase-transition.svg",
            "asset": "ch26/ch26-autoregressive-vs-diffusion-llm-roofline-phase-transition.svg",
            "section": "26.1.1",
            "caption": "32B FP8 模型在 H100 上的解码 Roofline 相变：从逐词元自回归（I=1 FLOP/B，100.0 tok/s）到半自回归块扩散（K=64, T=10，592.6 tok/s，提速 5.93 倍）（算例 26-1）",
            "source_refs": ["experiments/ch23/23-01/run.py", "manuscripts/ch26/figure-data.json"],
            "TeachingRole": "揭示扩散语言模型如何通过块内 64 词元并行去噪将权重读取次数削减 6.40 倍并突破内存带宽墙"
        },
        {
            "figure_id": "图 26-2",
            "number": 2,
            "file": "ch26-block-diffusion-semi-autoregressive-kv-cache-and-mask.svg",
            "asset": "ch26/ch26-block-diffusion-semi-autoregressive-kv-cache-and-mask.svg",
            "section": "26.1.2",
            "caption": "半自回归块扩散（Block Diffusion）的块间因果 KV 缓存固化与块内 K=64 双向全注意力混合阶梯掩码架构",
            "source_refs": ["experiments/ch23/23-01/run.py", "manuscripts/ch26/figure-data.json"],
            "TeachingRole": "展示如何兼顾自回归模型的历史 KV 缓存复用与扩散模型的块内并行迭代生成"
        },
        {
            "figure_id": "图 26-3",
            "number": 3,
            "file": "ch26-dynamic-confidence-threshold-remasking-scheduler.svg",
            "asset": "ch26/ch26-dynamic-confidence-threshold-remasking-scheduler.svg",
            "section": "26.1.2",
            "caption": "动态置信度阈值重掩码调度器（gamma=0.92）：提前锁定高置信词元并省去 35.0% 冗余去噪步，单流吞吐跃升至 911.7 tok/s（提速 9.12 倍）（算例 26-2）",
            "source_refs": ["experiments/ch23/23-01/run.py", "manuscripts/ch26/figure-data.json"],
            "TeachingRole": "量化自适应去噪步数早退机制对代码与结构化文本生成的额外加速收益"
        },
        {
            "figure_id": "图 26-4",
            "number": 4,
            "file": "ch26-dllm-speculative-parallel-verification-hybrid-engine.svg",
            "asset": "ch26/ch26-dllm-speculative-parallel-verification-hybrid-engine.svg",
            "section": "26.1.2",
            "caption": "扩散模型 O(1) 步并行草稿提议与自回归主模型单步精确校验的跨范式混合投机推理引擎",
            "source_refs": ["manuscripts/ch26/figure-data.json"],
            "TeachingRole": "解决传统投机解码在草稿过长时串行提议耗时急剧攀升的瓶颈"
        },
        {
            "figure_id": "图 26-5",
            "number": 5,
            "file": "ch26-bpe-tokenizer-bottleneck-vs-blt-byte-entropy-patching.svg",
            "asset": "ch26/ch26-bpe-tokenizer-bottleneck-vs-blt-byte-entropy-patching.svg",
            "section": "26.2.1",
            "caption": "Byte Latent Transformer（BLT）基于字节流条件熵阈值（tau_H=1.35 bits）的在线动态分块（Entropy Patching），省去 4.194 GB 词表显存并降低 52.4% 算力（算例 26-3）",
            "source_refs": ["experiments/ch23/23-02/run.py", "manuscripts/ch26/figure-data.json"],
            "TeachingRole": "对比静态 BPE 分词与按信息熵动态合并字节块在显存与计算分配效率上的本质差异"
        },
        {
            "figure_id": "图 26-6",
            "number": 6,
            "file": "ch26-blt-local-byte-encoder-global-latent-cross-attention.svg",
            "asset": "ch26/ch26-blt-local-byte-encoder-global-latent-cross-attention.svg",
            "section": "26.2.2",
            "caption": "BLT 局部字节编码器、全局潜空间 Transformer 与局部字节解码器的非对称交叉注意力流水线",
            "source_refs": ["experiments/ch23/23-02/run.py", "manuscripts/ch26/figure-data.json"],
            "TeachingRole": "阐明如何将 85% 以上的骨干参数集中在缩短 2.67 倍的潜空间 Patch 序列上"
        },
        {
            "figure_id": "图 26-7",
            "number": 7,
            "file": "ch26-blt-ragged-patch-packing-and-load-balancing.svg",
            "asset": "ch26/ch26-blt-ragged-patch-packing-and-load-balancing.svg",
            "section": "26.2.2",
            "caption": "面向变长字节块（1~16 Bytes）的前缀和紧凑打包（cu_seqlens）与无填充非规则交叉注意力（Ragged Cross-Attention）内核",
            "source_refs": ["manuscripts/ch26/figure-data.json"],
            "TeachingRole": "消除动态分块长度不齐带来的 60% 矩形 Padding 算力浪费"
        },
        {
            "figure_id": "图 26-8",
            "number": 8,
            "file": "ch26-titans-short-term-attention-and-long-term-neural-memory.svg",
            "asset": "ch26/ch26-titans-short-term-attention-and-long-term-neural-memory.svg",
            "section": "26.3.1",
            "caption": "Google Titans 双系统记忆架构：短期精确滑动窗口注意力与惊奇度（梯度范数）驱动动量更新的长期神经记忆模块",
            "source_refs": ["experiments/ch23/23-02/run.py", "manuscripts/ch26/figure-data.json"],
            "TeachingRole": "解释如何将长上下文记忆建模为测试时在线梯度下降（Test-Time Training）"
        },
        {
            "figure_id": "图 26-9",
            "number": 9,
            "file": "ch26-ttt-token-by-token-vs-chunkwise-dual-form-tensor-core.svg",
            "asset": "ch26/ch26-ttt-token-by-token-vs-chunkwise-dual-form-tensor-core.svg",
            "section": "26.3.2",
            "caption": "TTT-Linear / Titans 快速权重更新对比：逐词元串行外积（1.5 FLOP/B，42k tok/s）vs. 分块对偶形式（b=64，48.0 FLOP/B，1,033.2k tok/s，提速 24.60 倍）（算例 26-4）",
            "source_refs": ["experiments/ch23/23-02/run.py", "manuscripts/ch26/figure-data.json"],
            "TeachingRole": "推导如何将 64 步串行梯度更新折叠为 Tensor Core 矩阵乘指令"
        },
        {
            "figure_id": "图 26-10",
            "number": 10,
            "file": "ch26-titans-2m-context-constant-memory-vs-kv-cache-growth.svg",
            "asset": "ch26/ch26-titans-2m-context-constant-memory-vs-kv-cache-growth.svg",
            "section": "26.3.2",
            "caption": "200 万（2M）词元超长上下文下的显存对比：线性膨胀的 Softmax KV 缓存 vs. 恒定 16.0 MB/层（O(1) 零膨胀）的 Titans 快速权重（算例 26-4）",
            "source_refs": ["experiments/ch23/23-02/run.py", "manuscripts/ch26/figure-data.json"],
            "TeachingRole": "直观展示参数化测试时记忆如何彻底打破超长上下文的显存容量墙"
        },
        {
            "figure_id": "图 26-11",
            "number": 11,
            "file": "ch26-coconut-continuous-latent-thought-zero-copy-loop.svg",
            "asset": "ch26/ch26-coconut-continuous-latent-thought-zero-copy-loop.svg",
            "section": "26.4.1",
            "caption": "连续思维链（Coconut）隐藏状态显存内零拷贝回环反馈架构：以 c=3 潜空间压缩比将 3,000 ms 离散推理缩减至 924 ms（提速 3.25 倍）（算例 26-5）",
            "source_refs": ["experiments/ch23/23-03/run.py", "manuscripts/ch26/figure-data.json"],
            "TeachingRole": "剖析省去中间思考步 LM Head 词表投影、采样同步与离散坍缩的系统与信息论双重收益"
        },
        {
            "figure_id": "图 26-12",
            "number": 12,
            "file": "ch26-coconut-latent-tree-breadth-first-parallel-expansion.svg",
            "asset": "ch26/ch26-coconut-latent-tree-breadth-first-parallel-expansion.svg",
            "section": "26.4.1",
            "caption": "Coconut 连续潜向量叠加态（Superposition）驱动的隐式广度优先波束搜索与共享前缀分支验证",
            "source_refs": ["manuscripts/ch26/figure-data.json"],
            "TeachingRole": "展示单潜向量同时编码多条候选推理轨迹以规避离散思维树 KV 爆炸的机制"
        },
        {
            "figure_id": "图 26-13",
            "number": 13,
            "file": "ch26-mixture-of-depths-token-routing-and-warp-divergence.svg",
            "asset": "ch26/ch26-mixture-of-depths-token-routing-and-warp-divergence.svg",
            "section": "26.4.2",
            "caption": "词元级动态跳层（Mixture-of-Depths，保留率 rho=0.25）在未紧缩掩码执行下引发的 GPU SIMT Warp 分支发散（有效利用率跌至 58.1%）（算例 26-6）",
            "source_refs": ["experiments/ch23/23-03/run.py", "manuscripts/ch26/figure-data.json"],
            "TeachingRole": "通过泊松分布概率推导解释为何细粒度条件跳层会严重破坏 GPU 32 线程锁步效率"
        },
        {
            "figure_id": "图 26-14",
            "number": 14,
            "file": "ch26-stream-compaction-gather-scatter-for-early-exit-layers.svg",
            "asset": "ch26/ch26-stream-compaction-gather-scatter-for-early-exit-layers.svg",
            "section": "26.4.2",
            "caption": "基于硬件 Warp 投票（__ballot_sync）前缀和的流紧缩（Stream Compaction）Gather-GEMM-Scatter 管线，将 Warp 满载率恢复至 96.4% 并实现 3.32 倍单层提速（算例 26-6）",
            "source_refs": ["experiments/ch23/23-03/run.py", "manuscripts/ch26/figure-data.json"],
            "TeachingRole": "展示如何用极低开销的片上连续重排将动态稀疏词元转化为满载 Tensor Core 矩阵块"
        },
        {
            "figure_id": "图 26-15",
            "number": 15,
            "file": "ch26-post-autoregressive-architectures-throughput-latency-pareto.svg",
            "asset": "ch26/ch26-post-autoregressive-architectures-throughput-latency-pareto.svg",
            "section": "26.4.2",
            "caption": "后自回归生成架构（Block Diffusion、BLT、Titans/TTT、Coconut 与 MoD）在吞吐、延迟与上下文长度三维空间的帕累托前沿",
            "source_refs": ["manuscripts/ch26/figure-data.json"],
            "TeachingRole": "横向对比四大新兴范式各自攻克的硬件瓶颈与协同组合路径"
        },
        {
            "figure_id": "图 26-16",
            "number": 16,
            "file": "ch26-next-gen-foundation-model-infrastructure-synthesis.svg",
            "asset": "ch26/ch26-next-gen-foundation-model-infrastructure-synthesis.svg",
            "section": "26.4.2",
            "caption": "第 26 章系统工程全景图：从扩散并行去噪、字节熵动态分块、测试时分块对偶记忆到连续潜空间与流紧缩运行时",
            "source_refs": ["manuscripts/ch26/figure-data.json"],
            "TeachingRole": "收束全章四大模块，凝练把动态非规则算法还原为高算术强度分块矩阵原语的统一方法论"
        }
    ]
    for row in fig_index:
        row['figure'] = f"26-{row['number']}"
        row['name'] = Path(row['file']).stem
    (HERE / 'figure-index.json').write_text(json.dumps(fig_index, ensure_ascii=False, indent=2) + '\n')

    md = HERE.parent / '26-后自回归生成范式基础设施：扩散大语言模型（dLLM）、字节级无分词架构（BLT）与测试时记忆网络（Titans-TTT）.md'
    if md.exists():
        text_md = md.read_text()
        display = re.findall(r'\$\$(.+?)\$\$', text_md, re.S)
        inline = re.findall(r'(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)', text_md)
        formulas = [f.strip() for f in display + inline]
        katex_js = ROOT / 'manuscripts/ch06/vendor/katex/katex.js'
        js = (
            'const katex=require(process.argv[1]);const fs=require("fs");'
            'const a=JSON.parse(fs.readFileSync(0,"utf8"));const e=[];'
            'a.forEach((s,i)=>{try{katex.renderToString(s,{throwOnError:true,displayMode:i<'
            + str(len(display))
            + '})}catch(err){e.push({i,s,err:err.message})}});'
            'console.log(JSON.stringify({total:a.length,display:'
            + str(len(display))
            + ',inline:'
            + str(len(inline))
            + ',errors:e}));'
        )
        res = subprocess.run(
            ['node', '-e', js, str(katex_js)],
            input=json.dumps(formulas),
            text=True,
            capture_output=True,
            check=True,
        )
        math_val = json.loads(res.stdout)
        assert not math_val['errors'], math_val['errors']
        (HERE / 'math-validation.json').write_text(json.dumps(math_val, ensure_ascii=False, indent=2) + '\n')

        import markdown

        def repl_img(m):
            alt, rel = m.group(1), m.group(2)
            p = (md.parent / rel).with_suffix('.png')
            b64 = base64.b64encode(p.read_bytes()).decode()
            return f'<figure><img alt="{alt}" src="data:image/png;base64,{b64}"><figcaption>{alt}</figcaption></figure>'

        work = re.sub(r'!\[([^\]]*)\]\((ch26/[^)]+)\)', repl_img, text_md)
        body = markdown.markdown(work, extensions=['tables', 'fenced_code', 'footnotes', 'toc'])
        html = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>第 26 章：后自回归生成范式基础设施：扩散大语言模型（dLLM）、字节级无分词架构（BLT）与测试时记忆网络（Titans-TTT）</title></head><body>{body}</body></html>\n'
        out_html = preview_path(md)
        out_html.write_text(html)

    sources = [
        HERE / 'build.py',
        HERE / 'figure-data.json',
        HERE / 'figure-index.json',
        ROOT / 'experiments/ch23/23-01/run.py',
        ROOT / 'experiments/ch23/23-02/run.py',
        ROOT / 'experiments/ch23/23-03/run.py',
    ]
    if md.exists():
        sources.append(md)
    (HERE / 'sources.json').write_text(
        json.dumps(
            {"sources": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in sources]},
            ensure_ascii=False,
            indent=2,
        )
        + '\n'
    )
    all_outputs = sorted(outputs) + [
        HERE / 'figure-data.json',
        HERE / 'figure-index.json',
        HERE / 'teaching-layout-validation.json',
    ]
    if (HERE / 'math-validation.json').exists():
        all_outputs.append(HERE / 'math-validation.json')
    (HERE / 'manifest.json').write_text(
        json.dumps(
            {"outputs": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in all_outputs]},
            ensure_ascii=False,
            indent=2,
        )
        + '\n'
    )
    print(f'Built {len(outputs)} figure files for Chapter 26; 0 layout warnings.')


if __name__ == '__main__':
    build_chapter()
