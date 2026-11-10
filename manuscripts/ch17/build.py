#!/usr/bin/env python3
"""Build Chapter 17 figures, locked calculations, and offline HTML preview."""
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
from figure_style import COL, BORDER_COL, STYLE, Exporter, arrow, box, canvas, card, text
from preview_output import preview_path

plt.rcParams.update(STYLE)


def sha256(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b'version https://git-lfs.github.com/spec/v1'):
        m = re.search(rb'oid sha256:([0-9a-f]{64})', raw)
        if m:
            return m.group(1).decode('ascii')
    return hashlib.sha256(raw).hexdigest()


def fig_17_1_1d_vs_3d_tokens(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, '一维语言模型（1D LLM）', color='blue', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '一维离散词元序列 L\n典型窗口 4K–128K 词元', 'blue', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '因果下三角掩码（Causal）\nPrefill 1 次 + Decode 逐词元', 'teal', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, 'Decode 阶段受限于 HBM 带宽\n算术强度 ~1–4 FLOP/B', 'purple', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '一维自回归：访存墙主导', size=11, ha='center', weight='bold', color=BORDER_COL['blue'])

    card(ax, 0.52, 0.04, 0.46, 0.92, '三维视频生成（3D DiT）', color='orange', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '10s·24fps·1080p 视频潜空间\n60×68×120 = 489,600 词元', 'orange', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '三维全双向自注意力 O(N²d)\n单步计算量超文本百倍', 'orange', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '还需迭代去噪 K = 20–50 步\n尾端叠加 3D VAE 显存洪峰', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '三维迭代去噪：算力与通信墙', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])
    ex.save(fig, 'ch17-1d-llm-vs-3d-spatiotemporal-dit-tokens')


def fig_17_2_anyres_navit_qwen3vl(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'AnyRes 动态切片与 Qwen3-VL-4B 视觉词元生成（算例 17-1）', color='blue')
    stages = [
        (0.04, 0.58, 0.28, 0.24, '① 图像网格切片\n640×640 像素\n切成 1600 块', 'blue'),
        (0.36, 0.58, 0.28, 0.24, '② 2×2 空间合并\n1600 / 4 = 400 词元\nNaViT 变长打包', 'teal'),
        (0.68, 0.58, 0.28, 0.24, '③ DeepStack 特征\n末层+3 中间层\n[400, 10240] BF16', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.36, 0.70))
    arrow(ax, (0.64, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '跨阶段传输账本：编码器输出（EC）vs. 36 层主干 KV 缓存', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '视觉编码器输出（EC 缓存）\n400×10240×2 B = 8,192,000 B\n精确体积仅 7.8125 MiB', 'teal', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '36 层 LLM 主干 KV 缓存\n400×147,456 B = 58,982,400 B\n体积膨胀至 56.25 MiB（7.2x）', 'orange', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '在 E→P 边界传输或缓存 EC 比在 P→D 边界搬移 KV 省 7.2 倍容量', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-anyres-navit-dynamic-image-tiling')


def fig_17_3_3d_vae_tiling(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, '未分块全量 3D VAE 解码', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '潜变量 [60, 135, 240] 经\n4×8×8 3D 因果卷积上采样', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '全分辨率 240×1080×1920\n128 通道 FP16 达 118.65 GiB', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '远超单张 80 GB 显存上限\n在去噪最后一步触发 OOM', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '像素还原阶段激活洪峰致命', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, '时空分块解码（Tiled VAE）', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '空间 4×4 分块 + 16 帧滑窗\n保留因果时间卷积历史缓存', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '分块边界加 12.5% 重叠 Halo\n余弦羽化融合消除接缝伪影', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '单缓冲激活降至 0.63 GiB\n显存峰值压缩超 90 倍', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '以 Halo 重算换百倍省显存', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch17-3d-causal-vae-memory-spike-and-tiling')


def fig_17_4_tp_ulysses_ring(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '长序列 DiT 三类并行范式的通信量与扩展度边界（算例 17-4）', color='blue')
    modes = [
        (0.04, 0.58, 0.28, 0.24, '① 张量并行 TP\n切分权重列/行\n通信 O(Nd) 不降', 'orange'),
        (0.36, 0.58, 0.28, 0.24, '② Ulysses 并行\nAll-to-All 转置头\n每层 4Nd / P', 'teal'),
        (0.68, 0.58, 0.28, 0.24, '③ Ring 注意力\n环路传 KV 分块\n全环流量约 2Nd', 'blue'),
    ]
    for x, y, w, h, label, role in modes:
        box(ax, x, y, w, h, label, role, size=11)

    card(ax, 0.02, 0.03, 0.96, 0.42, 'N = 489,600, d = 4,096, P = 64 卡下的定量对比', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, 'Ulysses：每层 239.06 MiB\n通信随 1/P 下降，但受限于\n并行度 P ≤ 注意力头数', 'teal', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, 'Ring：全环 63 步共 7,530 MiB\n突破头数上限，但纯跨机环路\n易受慢链路与分块延迟拖累', 'blue', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, 'Ulysses 省总流量但受限于头数；Ring 无头数上限但总流量不随 P 衰减', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-tp-vs-ulysses-vs-ring-attention-comm')


def fig_17_5_2d_usp(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '2D 统一序列并行（USP：P_Ulysses × P_Ring = 8 × 8 = 64 卡）', color='teal')
    stages = [
        (0.04, 0.58, 0.28, 0.24, '① 节点内 NVLink\nP_Ulysses = 8 卡\nAll-to-All 切头', 'teal'),
        (0.36, 0.58, 0.28, 0.24, '② 节点间 RoCE\nP_Ring = 8 节点环\n传 1/8 头维 KV 块', 'blue'),
        (0.68, 0.58, 0.28, 0.24, '③ 算通双重叠\nFlashAttn 分块核\n掩盖 RDMA P2P', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.36, 0.70))
    arrow(ax, (0.64, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '纯 64 卡 Ring vs. 8×8 2D-USP 的跨节点 RDMA 流量对比', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, '纯 64 卡 Ring Attention\n63 步均跨越节点间网络\n单卡跨机流量 7,530.47 MiB', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '8×8 2D-USP 混合拓扑\n跨机 Ring 仅 7 步×119.53 MiB\n跨机 RDMA 降至 836.72 MiB', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '节点内 NVLink 吸收 All-to-All，将跨节点 RDMA 流量压缩 9.0 倍', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-2d-usp-hybrid-ulysses-ring-topology')


def fig_17_6_dbsp(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, '朴素均匀序列切分（负载失衡）', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '多模态交错流含双向图像块\n与单向因果掩码文本/视频帧', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '按位置顺序等长切分序列\n前卡仅算短前缀·末卡算全长', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '最快卡与最慢卡 FLOPs 差 2x\n同步屏障产生严重木桶空泡', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '掩码稀疏与变长导致算力倾斜', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'db-SP（MLSys 26）双均衡切分', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '第一级：模态内首尾对称折返\n（ZigZag 配对首块与末块）', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '第二级：跨样本与跨模态重排\n对齐每张 GPU 有效注意力面积', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '同时消除计算倾斜与通信偏斜\n长序列多模态吞吐提升 1.4x+', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '有效非零注意力块数跨卡严格对齐', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch17-dbsp-dual-balanced-sequence-parallelism')


def fig_17_7_distrifusion(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, '同步序列并行（通信阻塞）', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '第 t 去噪步计算局部 QKV 后\n必须等待全量远端 KV 到达', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '高分辨率图像/短视频 Batch=1\n单步计算短，难掩盖跨卡通信', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '50 步迭代累加数百次同步等待\n多卡扩展效率随卡数骤降', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '当前步强同步屏障卡死低延迟', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'DistriFusion 异步位移并行', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '利用相邻去噪步激活缓变特性\n||x_t - x_{t-1}|| << ||x_t||', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '本地 Patch 用第 t 步最新 KV\n远端 Patch 直接读 t-1 步缓存', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '第 t 步通信与第 t 步计算全重叠\n零同步暴露实现近线性降延迟', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '用一阶时间步平滑性换零通信暴露', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch17-distrifusion-displaced-patch-parallelism')


def fig_17_8_pipefusion(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'PipeFusion：空间 Patch 分块 × 去噪时间步二维交错流水', color='blue')
    stages = [
        (0.04, 0.58, 0.28, 0.24, '① 模型层切分\nL 层 DiT 切入 P 卡\n免除全量权重冗余', 'blue'),
        (0.36, 0.58, 0.28, 0.24, '② 空间切 M 个块\n单步内 M 个 Patch\n充当流水微批次', 'teal'),
        (0.68, 0.58, 0.28, 0.24, '③ 跨步无缝衔接\n第 t-1 步末块后\n紧接第 t 步首块', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.36, 0.70))
    arrow(ax, (0.64, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '传统每步排空流水线 vs. PipeFusion 跨步连续流水的气泡率', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '传统步内同步流水线\n每个去噪步末尾均需排空重填\n50 步累计产生 50 × (P-1) 气泡', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, 'PipeFusion 异步跨步流水\n仅在第 1 步启动与第 50 步结束\n产生 1 次排空，气泡压缩 50 倍', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '仅传输层边界 Patch 激活（远小于全层 QKV），适配 PCIe 与普通以太网', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-pipefusion-step-patch-2d-pipeline')


def fig_17_9_teacache(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'TeaCache / AdaCache：免训练时间步特征缓存与自适应跳步', color='teal')
    stages = [
        (0.04, 0.58, 0.28, 0.24, '① 时间步嵌入探测\n计算调制输入差分\n多项式标定变化', 'teal'),
        (0.36, 0.58, 0.28, 0.24, '② 累积误差门限\n若漂移 δ_acc < ε\n跳过整座 DiT', 'blue'),
        (0.68, 0.58, 0.28, 0.24, '③ 残差特征复用\nx_{t-1} = x_t + ΔF\n越过平缓去噪步', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.36, 0.70))
    arrow(ax, (0.64, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '算例 17-5：50 步视频 DiT 去噪跳过 28 步的端到端加速账本', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, '50 步全量计算基线\n50 × 1.20 s + 3.60 s（编码解码）\n端到端生成耗时 = 63.60 s', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, 'TeaCache 动态保留 22 个关键步\n22 × 1.20 s + 3.60 s = 30.00 s\n视觉无损下端到端加速 2.12x', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '首尾剧变期密集计算、中段平缓期跳步复用，零训练成本砍半计算量', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-teacache-timestep-residual-reuse')


def fig_17_10_sageattention(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, '直接低比特量化注意力失真', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, 'DiT 键张量 K 存在通道离群值\n部分通道偏置高达均值 10–50x', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '直接做 INT8/FP4 对称块量化\n有效尾数位被离群通道吞噬', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, 'Softmax 概率分布严重畸变\n生成视频出现雪花伪影与崩坏', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '通道直流偏置摧毁低位宽精度', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'SageAttention-3 平滑微缩放', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '利用 Q·(K - μ_K)^T 等价变形\n先减去序列均值平滑 K 离群值', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, 'QK^T 用 FP8/FP4 硬件块乘\nPV 采用两级缩放防下溢累加', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, 'Blackwell 获 3–4x 算子加速\n生成视频 PSNR/VBench 零退化', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '均值平滑离群值，释放 FP4 算力', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch17-sageattention3-fp4-quantized-attention')


def fig_17_11_world_model_30fps(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '交互式世界模型（StreamDiffusionV2 / CausVid）30 FPS 实时流水', color='blue')
    stages = [
        (0.04, 0.58, 0.28, 0.24, '① 因果自回归改造\n块级因果注意力\n按帧块流式吐出', 'blue'),
        (0.36, 0.58, 0.28, 0.24, '② 分布匹配蒸馏\n50 步浓缩为 1–4 步\n砍掉 CFG 双倍分支', 'teal'),
        (0.68, 0.58, 0.28, 0.24, '③ 滚动 KV 缓存池\n固定窗口复用历史帧\n淘汰过期帧防溢出', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.36, 0.70))
    arrow(ax, (0.64, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '步间小批次流水（Stream Batching）：在 33.3 ms 红线内响应动作', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '传统整段双向扩散\n必须等 240 帧全部去噪 50 步\n首帧响应延迟 > 30,000 ms', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '4 步因果流式 + 跨帧去噪步拼批\n4 个相邻帧分处去噪步 s=4,3,2,1\n每 ≤ 33.3 ms 稳定输出 1 帧新画面', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '用因果块注意力 + 滚动 KV + 4 步流式拼批跨越实时交互生死线', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-interactive-world-model-rolling-kv-30fps')


def fig_17_12_epdg_disaggregation(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '原生全模态服务四池解耦架构（E-P-D-G：vLLM-Omni / ModServe）', color='blue')
    pools = [
        (0.03, 0.58, 0.21, 0.24, '① Encode 池\nViT/音频编码\n计算密·小显存', 'teal'),
        (0.27, 0.58, 0.21, 0.24, '② Prefill 池\n多模态预填充\n注意力算力密', 'blue'),
        (0.51, 0.58, 0.21, 0.24, '③ Decode 池\n自回归解码\nHBM 带宽受限', 'purple'),
        (0.75, 0.58, 0.22, 0.24, '④ Generate 池\n图像/视频 DiT\n高算力密度', 'orange'),
    ]
    for x, y, w, h, label, role in pools:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.24, 0.70), (0.27, 0.70))
    arrow(ax, (0.48, 0.70), (0.51, 0.70))
    arrow(ax, (0.72, 0.70), (0.75, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, 'TriInfer（MLSys 26 #127）动态合并 vs. 独立建池的临界准则', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '何时分离 E 与 PD？（算例 17-6）\n4 卡下 (1 E, 3 PD) 吞吐 12 req/s\n75% EC 缓存命中使 E 需求降 4x', 'teal', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, 'TriInfer 路由（c_attn / c_ffn）\n长图像序列走独立 E+P 防干扰\n短图低负载走融合节点省传输', 'blue', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '按各阶段 Roofline 瓶颈与缓存命中率 h 动态配比四类异构 GPU 池', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-epdg-four-stage-disaggregated-serving')


def fig_17_13_ec_vs_kv_cache(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '多轮视觉对话的三级缓存形态对比（Qwen3-VL-4B 单图 400 词元）', color='teal')
    tiers = [
        (0.04, 0.58, 0.28, 0.24, '① 仅存原始图像\n约 0.2–0.5 MiB\n每轮重算 ViT', 'orange'),
        (0.36, 0.58, 0.28, 0.24, '② 缓存编码特征 EC\n体积 7.8125 MiB\n免除 ViT 重算', 'teal'),
        (0.68, 0.58, 0.28, 0.24, '③ 缓存全层 KV\n体积 56.25 MiB\n膨胀达 7.2 倍', 'blue'),
    ]
    for x, y, w, h, label, role in tiers:
        box(ax, x, y, w, h, label, role, size=11)

    card(ax, 0.02, 0.03, 0.96, 0.42, '分层多模态缓存策略：热轮次留 KV，温会话退回 EC（省 86.1% 容量）', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, '单层投影特征 [400, 2560]\nEC 体积可压至 1.953125 MiB\n相比 56.25 MiB KV 省 28.8 倍', 'teal', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '含 3 层 DeepStack（7.8125 MiB）\n将温冷会话从 KV 降级为 EC 缓存\n释放 86.1% 内存并省 7.2x 带宽', 'purple', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, 'EC 缓存是多模态服务独有的“高压缩比中间态检查点”', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-vit-embedding-cache-vs-llm-kv-cache')


def fig_17_14_interleaved_batching(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, '串行锁步执行（异构气泡）', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '同一批次内混杂纯文本请求、\n多图请求与长音频片段请求', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '主干 LLM 停下等待 ViT 与\nWhisper 编码器串行跑完', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '纯文本请求被重型视觉编码拖累\n首词元延迟 TTFT 暴涨 3–8 倍', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '模态耦合阻塞纯文本快路径', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, '模态微批聚合与异步对齐', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '按模态拆分图像/音频子张量\n在专属 CUDA Stream 攒批编码', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '纯文本请求零等待直入主干\n多模态请求待 EC 就绪后注入', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '通过散射索引将 EC 嵌回占位符\n消除跨模态气泡并吃满 Tensor 核', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '异构编码流与主干连续批处理解耦', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch17-audio-video-text-interleaved-batching')


def fig_17_15_gpu_decode_dataloader(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, '传统 CPU 软解码（GPU 饥饿）', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, 'H.264/HEVC 视频与高分 JPEG\n由宿主机 CPU FFmpeg/PIL 解码', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '解压后 RGB 体积暴涨 20–50 倍\n撑爆主机内存带宽与 PCIe 通道', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '128 核 CPU 满载仍供不上 8 卡\nGPU 算力利用率跌至 < 45%', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '在 PCIe 上游解压是灾难性反模式', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'DALI + NVDEC/nvJPEG 硬件卸载', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '压缩码流直接经 PCIe DMA\n传入 GPU 显存（省 25x 带宽）', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '调用 GPU 独立硬件引擎 NVDEC\n与 nvJPEG 零 SM 占用硬解码', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '显存内原位完成缩放与归一化\n直供 ViT/VAE，GPU 满载 > 92%', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '贴近算力侧解压 + 专用硅片硬解码', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch17-multimodal-dataloader-gpu-decode-nvjpeg')


def fig_17_16_dit_cfg(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '分类器自由引导（Classifier-Free Guidance, CFG）的算力翻倍与优化', color='blue')
    modes = [
        (0.04, 0.58, 0.28, 0.24, '① 标准双分支 CFG\n每步跑条件+无条件\n计算与显存翻 2 倍', 'orange'),
        (0.36, 0.58, 0.28, 0.24, '② CFG 并行 (CFG-P)\n2 组 GPU 分跑两路\n步末仅同步潜变量', 'teal'),
        (0.68, 0.58, 0.28, 0.24, '③ 引导蒸馏与截断\n后半程关停无条件\n或蒸馏入单路嵌入', 'blue'),
    ]
    for x, y, w, h, label, role in modes:
        box(ax, x, y, w, h, label, role, size=11)

    card(ax, 0.02, 0.03, 0.96, 0.42, 'CFG 双路组合与跨组通信优化（ε = ε_u + w·(ε_c - ε_u)）', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, 'CFG-Parallel 跨组通信极轻\n每去噪步仅 AllGather 1 次输出\n比扩 2 倍 Ulysses 省 L 层通信', 'teal', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '动态 CFG 截断（后 40% 步停跑）\n构图已定型后省去无条件分支\n直接削减 20% 总去噪 FLOPs', 'purple', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '优先将首级并行度 2 分配给 CFG-Parallel，再叠加 2D-USP', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-dit-cfg-classifier-free-guidance-parallel')


def fig_17_17_end_to_end_breakdown(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '1080p 10s 视频生成端到端三阶段延迟与显存水位解剖', color='blue')
    stages = [
        (0.04, 0.58, 0.28, 0.24, '① 文本与图编码\nT5-XXL + ViT\n耗时 0.4s (<2%)', 'blue'),
        (0.36, 0.58, 0.28, 0.24, '② 3D DiT 去噪\n48.96 万词元×50步\n耗时 60s (94%)', 'orange'),
        (0.68, 0.58, 0.28, 0.24, '③ 3D VAE 解码\n耗时 3.2s (~5%)\n触发显存洪峰', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.36, 0.70))
    arrow(ax, (0.64, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '阶段错位规律：去噪期霸占 94% 时间，解码期触发 118 GiB 显存洪峰', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '优化时间轴盯紧阶段 ②（DiT）\n用 2D-USP + TeaCache + FP4\n将 60 s 去噪压缩至 8–15 s', 'teal', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '优化显存峰值盯紧阶段 ③（VAE）\n卸载 T5 权重 + 时空分块 Tiled VAE\n将 118.65 GiB 峰值压至 < 2 GiB', 'blue', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '时间瓶颈在 DiT 二次方注意力，空间峰值在 3D VAE 像素上采样', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-video-dit-end-to-end-latency-breakdown')


def fig_17_18_synthesis(ex: Exporter):
    fig, ax = canvas(4.15)
    card(ax, 0.02, 0.03, 0.96, 0.94, '第 17 章多模态、视频 DiT 与世界模型基础设施工程总账本', color='blue')
    grid = [
        (0.05, 0.66, 0.42, 0.20, '① 时空表示与分块（17.1）\n489,600 视频词元·EC 7.81 MiB\nTiled VAE 消解 118 GiB 洪峰', 'blue'),
        (0.53, 0.66, 0.42, 0.20, '② 序列并行体系（17.2）\nUlysses 4Nd/P 与 Ring 2Nd\n2D-USP 降跨机流量 9.0x', 'teal'),
        (0.53, 0.37, 0.42, 0.20, '③ 时间步冗余与蒸馏（17.3）\nDistriFusion 异步掩盖通信\nTeaCache 2.12x 与 30 FPS 流', 'purple'),
        (0.05, 0.37, 0.42, 0.20, '④ 四池解耦服务（17.4）\nTriInfer 扩展至 E-P-D-G 四池\n缓存 EC 省 7.2x 带宽与容量', 'orange'),
    ]
    for x, y, w, h, label, role in grid:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.47, 0.76), (0.53, 0.76))
    arrow(ax, (0.74, 0.66), (0.74, 0.57))
    arrow(ax, (0.53, 0.47), (0.47, 0.47))
    arrow(ax, (0.26, 0.57), (0.26, 0.66))
    box(ax, 0.05, 0.08, 0.90, 0.21, '实验验证：experiments/ch17/17-01 & 17-02（Qwen3-VL 7.2x 膨胀、\n2D-USP 9.0x 减流、TeaCache 2.12x、Transfusion 62.5% 混合掩码）', 'teal', size=11)
    ex.save(fig, 'ch17-multimodal-world-model-infra-synthesis')


def fig_17_19_transfusion_hybrid_mask(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'Transfusion 统一主干：分段因果—双向混合掩码（算例 17-7）', color='blue')
    stages = [
        (0.04, 0.58, 0.28, 0.24, '① 文本前缀 T1\n1024 词元因果掩码\n36 个活跃块', 'blue'),
        (0.36, 0.58, 0.28, 0.24, '② 图像潜块 I1\n2048 词元块内双向\n384 个满载块', 'teal'),
        (0.68, 0.58, 0.28, 0.24, '③ 文本续写 T2\n1024 词元见全前缀\n228 个活跃块', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.36, 0.70))
    arrow(ax, (0.64, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '块稀疏跳过与 <boi> 触发 D→G 跨池零阻塞切换', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, 'FlashAttn 分块掩码跳过\n仅算 648/1024 块（省 36.7%）\n执行块内有效利用率 98.77%', 'teal', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '遇 <boi> 算术强度升 2048x\n2.88 ms 迁出 144 MiB KV\n移交 G 池跑 20 步扩散去噪', 'orange', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '有效注意力面积 10,485,760 对（占全矩阵 62.5%），模式切换须跨池隔离', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-transfusion-hybrid-causal-bidirectional-mask')


def fig_17_20_3dgs_world_model(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '3DGS 显式几何脚手架 + 视频 DiT 混合世界模型管线', color='teal')
    stages = [
        (0.04, 0.58, 0.28, 0.24, '① 显式 3DGS 光栅\n120 万三维高斯点\n2.4 ms 渲染深度图', 'teal'),
        (0.36, 0.58, 0.28, 0.24, '② 4 步受控视频 DiT\n注入几何防视角漂移\n流式拼批 24.0 ms', 'blue'),
        (0.68, 0.58, 0.28, 0.24, '③ 分块因果 VAE\n还原光影与非刚体\n耗时 4.5 ms 吐帧', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.36, 0.70))
    arrow(ax, (0.64, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '纯隐式视频 DiT vs. 3DGS—DiT 混合管线的几何一致性与实时帧率', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, '纯隐式视频 DiT 世界模型\n视角大角度旋转后三维坍塌\n闭环重投影误差 > 15 cm', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '3DGS 刚性几何 + DiT 光影\n总耗时 2.4+24.0+4.5 = 30.9 ms\n稳定达成 32.36 FPS 零漂移', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '用 3DGS 锁定欧氏空间几何守恒，用少步 DiT 补齐动态物理交互与材质', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-3dgs-scaffold-guided-world-model-dit')


def fig_17_21_causal_streaming_20fps(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'CausVid / Matrix-Game 块因果滚动 KV 窗口架构（算例 17-8）', color='blue')
    stages = [
        (0.04, 0.58, 0.28, 0.24, '① 首块注意力锚点\n1 个 Sink 块（3帧）\n3600 词元常驻不驱逐', 'blue'),
        (0.36, 0.58, 0.28, 0.24, '② 环形滚动 KV 窗\n5 个近期块（15帧）\n18000 词元 FIFO 覆写', 'teal'),
        (0.68, 0.58, 0.28, 0.24, '③ 当前去噪生成块\n块内双向 + 块间因果\n4 步 DMD 吐出 3 帧', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.36, 0.70))
    arrow(ax, (0.64, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '全历史线性增长 vs. 6 块滚动 KV 池的显存与延迟边界', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '无界历史 KV 累积\n10 分钟达 1440 万词元\n显存爆表 > 878 GiB 且掉帧', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '1 Sink + 5 滑窗（21600 词元）\n28 层 KV 恒定 1.3184 GiB\n无限时长稳定 > 20 FPS', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '块因果掩码将离线双向扩散转化为 O(1) 显存复杂度的无限流式世界模型', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-causal-streaming-world-model-20fps-kv-rolling')


def fig_17_22_teacache_pab_reuse(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '金字塔注意力广播（PAB）与 TeaCache 跨帧/跨步特征复用', color='teal')
    stages = [
        (0.04, 0.58, 0.28, 0.24, '① 空间自注意力\n高频细节变化最快\n间隔 2 步广播复用', 'blue'),
        (0.36, 0.58, 0.28, 0.24, '② 时间因果注意力\n中频运动轨迹平滑\n间隔 3 步跨步复用', 'teal'),
        (0.68, 0.58, 0.28, 0.24, '③ 条件交叉注意力\n低频控制语义最稳\n4 步仅首步算 1 次', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.36, 0.70))
    arrow(ax, (0.64, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '4 步块因果去噪下未复用 vs. PAB + TeaCache 复用延迟账本', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, '4 步全量块因果计算\nDiT 168 ms + VAE 18 ms\n单帧 62.0 ms（仅 16.13 FPS）', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, 'PAB + TeaCache 跨步复用\nDiT 压至 96 ms + VAE 18 ms\n单帧 38.0 ms（达 26.32 FPS）', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '按空间/时间/条件特征频谱分层复用，单块耗时从 186 ms 降至 114 ms', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-teacache-timestep-embedding-aware-activation-reuse')


def main():
    ex = Exporter(HERE)
    fig_17_1_1d_vs_3d_tokens(ex)
    fig_17_2_anyres_navit_qwen3vl(ex)
    fig_17_3_3d_vae_tiling(ex)
    fig_17_4_tp_ulysses_ring(ex)
    fig_17_5_2d_usp(ex)
    fig_17_6_dbsp(ex)
    fig_17_7_distrifusion(ex)
    fig_17_8_pipefusion(ex)
    fig_17_9_teacache(ex)
    fig_17_10_sageattention(ex)
    fig_17_11_world_model_30fps(ex)
    fig_17_12_epdg_disaggregation(ex)
    fig_17_13_ec_vs_kv_cache(ex)
    fig_17_14_interleaved_batching(ex)
    fig_17_15_gpu_decode_dataloader(ex)
    fig_17_16_dit_cfg(ex)
    fig_17_17_end_to_end_breakdown(ex)
    fig_17_18_synthesis(ex)
    fig_17_19_transfusion_hybrid_mask(ex)
    fig_17_20_3dgs_world_model(ex)
    fig_17_21_causal_streaming_20fps(ex)
    fig_17_22_teacache_pab_reuse(ex)
    outputs = ex.finish()

    fig_index = [
        {
            "figure_id": "图 17-1",
            "number": 1,
            "file": "ch17-1d-llm-vs-3d-spatiotemporal-dit-tokens.svg",
            "asset": "ch17/ch17-1d-llm-vs-3d-spatiotemporal-dit-tokens.svg",
            "section": "17.1.1",
            "caption": "一维文本 Token 序列与 3D 视频扩散模型（T×H×W 潜空间分块，489,600 Tokens）的规模与算术强度对比",
            "source_refs": ["manuscripts/ch17/figure-data.json", "experiments/ch17/17-01/run.py"],
            "TeachingRole": "对比 1D LLM 自回归访存受限与 3D DiT 489,600 时空词元双向注意力算力墙"
        },
        {
            "figure_id": "图 17-2",
            "number": 2,
            "file": "ch17-anyres-navit-dynamic-image-tiling.svg",
            "asset": "ch17/ch17-anyres-navit-dynamic-image-tiling.svg",
            "section": "17.1.1",
            "caption": "AnyRes 动态高分辨率切片、NaViT 序列打包与 Qwen3-VL-4B DeepStack 特征（7.8125 MiB EC → 56.25 MiB KV，7.2x 放大）演化（算例 17-1）",
            "source_refs": ["case-studies/multimodal-stage-placement.md", "experiments/ch17/17-01/run.py"],
            "TeachingRole": "复算 Qwen3-VL-4B 单图 400 词元从 7.8125 MiB 编码特征膨胀为 56.25 MiB KV 缓存的 7.2 倍比值"
        },
        {
            "figure_id": "图 17-3",
            "number": 3,
            "file": "ch17-3d-causal-vae-memory-spike-and-tiling.svg",
            "asset": "ch17/ch17-3d-causal-vae-memory-spike-and-tiling.svg",
            "section": "17.1.3",
            "caption": "3D 因果 VAE 解码阶段的特征图显存洪峰（突破 80 GB）与带重叠 Halo 的时空分块（Spatial-Temporal Tiling）解码（算例 17-3）",
            "source_refs": ["manuscripts/ch17/figure-data.json", "experiments/ch17/17-01/run.py"],
            "TeachingRole": "揭示视频去噪尾端 3D VAE 118.65 GiB 特征图洪峰及带 Halo 时空分块的百倍显存压缩"
        },
        {
            "figure_id": "图 17-4",
            "number": 4,
            "file": "ch17-tp-vs-ulysses-vs-ring-attention-comm.svg",
            "asset": "ch17/ch17-tp-vs-ulysses-vs-ring-attention-comm.svg",
            "section": "17.2.2",
            "caption": "张量并行（TP）、DeepSpeed-Ulysses（每层 4Nd/P，P ≤ n_head）与 Ring Attention（单步 2Nd/P、全环 2(P-1)Nd/P ≈ 2Nd，无头数限制）对比（算例 17-4）",
            "source_refs": ["manuscripts/ch17/figure-data.json", "experiments/ch17/17-01/run.py"],
            "TeachingRole": "精确对比 Ulysses 与 Ring Attention 的单步/全环通信公式及注意力头数上限约束"
        },
        {
            "figure_id": "图 17-5",
            "number": 5,
            "file": "ch17-2d-usp-hybrid-ulysses-ring-topology.svg",
            "asset": "ch17/ch17-2d-usp-hybrid-ulysses-ring-topology.svg",
            "section": "17.2.3",
            "caption": "2D 统一序列并行（USP）：节点内 NVLink 跑 Ulysses All-to-All + 跨节点 RoCE 跑 Ring Attention P2P 重叠流水",
            "source_refs": ["manuscripts/ch17/figure-data.json", "experiments/ch17/17-01/run.py"],
            "TeachingRole": "证明 8×8 2D-USP 可将跨节点 RDMA 流量从 7,530.47 MiB 压缩 9.0 倍至 836.72 MiB"
        },
        {
            "figure_id": "图 17-6",
            "number": 6,
            "file": "ch17-dbsp-dual-balanced-sequence-parallelism.svg",
            "asset": "ch17/ch17-dbsp-dual-balanced-sequence-parallelism.svg",
            "section": "17.2.3",
            "caption": "db-SP（MLSys '26）面向变长多模态与因果掩码的双均衡序列切分",
            "source_refs": ["research/2026-infra-survey/reading-mlsys-2026.md"],
            "TeachingRole": "解析模态内折返配对与跨样本重排如何消除因果/双向混合掩码导致的算力倾斜"
        },
        {
            "figure_id": "图 17-7",
            "number": 7,
            "file": "ch17-distrifusion-displaced-patch-parallelism.svg",
            "asset": "ch17/ch17-distrifusion-displaced-patch-parallelism.svg",
            "section": "17.3.1",
            "caption": "DistriFusion 异步位移 Patch 并行——利用第 t-1 步陈旧激活掩盖第 t 步跨卡通信",
            "source_refs": ["manuscripts/ch17/figure-data.json"],
            "TeachingRole": "利用相邻扩散步激活缓变性将跨卡 KV 传输彻底移出当前步关键路径"
        },
        {
            "figure_id": "图 17-8",
            "number": 8,
            "file": "ch17-pipefusion-step-patch-2d-pipeline.svg",
            "asset": "ch17/ch17-pipefusion-step-patch-2d-pipeline.svg",
            "section": "17.3.1",
            "caption": "PipeFusion 将空间分块与去噪时间步交错流水化的气泡消除时序图",
            "source_refs": ["manuscripts/ch17/figure-data.json"],
            "TeachingRole": "展示空间分块与时间步二维交错流水如何将 50 次流水线排空气泡压缩为 1 次"
        },
        {
            "figure_id": "图 17-9",
            "number": 9,
            "file": "ch17-teacache-timestep-residual-reuse.svg",
            "asset": "ch17/ch17-teacache-timestep-residual-reuse.svg",
            "section": "17.3.2",
            "caption": "TeaCache / AdaCache 基于时间步特征差分的自适应跳步复用与质量—延迟帕累托曲线（算例 17-5）",
            "source_refs": ["manuscripts/ch17/figure-data.json", "experiments/ch17/17-01/run.py"],
            "TeachingRole": "量化免训练时间步残差复用跳过 28/50 步实现的 2.12x 端到端视频生成加速"
        },
        {
            "figure_id": "图 17-10",
            "number": 10,
            "file": "ch17-sageattention3-fp4-quantized-attention.svg",
            "asset": "ch17/ch17-sageattention3-fp4-quantized-attention.svg",
            "section": "17.3.2",
            "caption": "SageAttention 平滑均值抑制离群值与 FP8/FP4 块量化注意力的硬件加速路径",
            "source_refs": ["manuscripts/ch17/figure-data.json"],
            "TeachingRole": "阐明减去 K 通道序列均值如何消除离群偏置并释放 FP8/FP4 Tensor Core 吞吐"
        },
        {
            "figure_id": "图 17-11",
            "number": 11,
            "file": "ch17-interactive-world-model-rolling-kv-30fps.svg",
            "asset": "ch17/ch17-interactive-world-model-rolling-kv-30fps.svg",
            "section": "17.3.3",
            "caption": "交互式世界模型（StreamDiffusionV2 / CausVid）的因果少步蒸馏与 <33.3 ms 滚动 KV 帧生成流水线",
            "source_refs": ["manuscripts/ch17/figure-data.json"],
            "TeachingRole": "剖析块因果扩散、1–4 步蒸馏、滚动 KV 缓存与步间流式拼批如何达成 30 FPS 实时交互"
        },
        {
            "figure_id": "图 17-12",
            "number": 12,
            "file": "ch17-epdg-four-stage-disaggregated-serving.svg",
            "asset": "ch17/ch17-epdg-four-stage-disaggregated-serving.svg",
            "section": "17.4.1",
            "caption": "从 TriInfer（MLSys '26）混合 EPD 解耦到 vLLM-Omni 原生全模态 E-P-D-G 四阶段异构池化架构（算例 17-6）",
            "source_refs": ["case-studies/multimodal-stage-placement.md", "research/2026-infra-survey/reading-mlsys-2026.md"],
            "TeachingRole": "建立 Encode、Prefill、Decode、Generate 四池分离与 TriInfer 动态放置准则"
        },
        {
            "figure_id": "图 17-13",
            "number": 13,
            "file": "ch17-vit-embedding-cache-vs-llm-kv-cache.svg",
            "asset": "ch17/ch17-vit-embedding-cache-vs-llm-kv-cache.svg",
            "section": "17.4.2",
            "caption": "多轮多模态对话中缓存原始图片、缓存 ViT 视觉特征（7.8125 MiB）与缓存 LLM 全层 KV（56.25 MiB）的体积与重算延迟折中",
            "source_refs": ["case-studies/multimodal-stage-placement.md", "experiments/ch17/17-01/run.py"],
            "TeachingRole": "证明在多轮对话温冷层缓存 EC 而非 KV 可节省 86.1% 内存并消除 ViT 重算"
        },
        {
            "figure_id": "图 17-14",
            "number": 14,
            "file": "ch17-audio-video-text-interleaved-batching.svg",
            "asset": "ch17/ch17-audio-video-text-interleaved-batching.svg",
            "section": "17.4.2",
            "caption": "图文音交错流（Omni-modal）中模态专属编码器微批聚合与主干对齐调度",
            "source_refs": ["case-studies/multimodal-stage-placement.md"],
            "TeachingRole": "解决图文音混合批次中重型模态编码阻塞纯文本快路径的异构气泡难题"
        },
        {
            "figure_id": "图 17-15",
            "number": 15,
            "file": "ch17-multimodal-dataloader-gpu-decode-nvjpeg.svg",
            "asset": "ch17/ch17-multimodal-dataloader-gpu-decode-nvjpeg.svg",
            "section": "17.1.3",
            "caption": "多模态预训练中 CPU 解码饥饿瓶颈与 DALI / nvJPEG / NVDEC 硬件视频解码卸载流水线",
            "source_refs": ["manuscripts/ch17/figure-data.json"],
            "TeachingRole": "阐明为何必须在 PCIe 下游利用 GPU NVDEC/nvJPEG 硬解码消除 CPU 与总线瓶颈"
        },
        {
            "figure_id": "图 17-16",
            "number": 16,
            "file": "ch17-dit-cfg-classifier-free-guidance-parallel.svg",
            "asset": "ch17/ch17-dit-cfg-classifier-free-guidance-parallel.svg",
            "section": "17.2.3",
            "caption": "分类器自由引导（CFG 双分支）与条件/无条件批量合并及尾部蒸馏对比",
            "source_refs": ["manuscripts/ch17/figure-data.json"],
            "TeachingRole": "解析 CFG-Parallel 跨组单次通信优势及后段去噪关停无条件分支的算力节省"
        },
        {
            "figure_id": "图 17-17",
            "number": 17,
            "file": "ch17-video-dit-end-to-end-latency-breakdown.svg",
            "asset": "ch17/ch17-video-dit-end-to-end-latency-breakdown.svg",
            "section": "17.3.2",
            "caption": "1080p 视频生成从文本编码、50 步 DiT 去噪到 3D VAE 解码的端到端延迟与显存水位全貌",
            "source_refs": ["manuscripts/ch17/figure-data.json", "experiments/ch17/17-01/run.py"],
            "TeachingRole": "建立时间瓶颈在 DiT 去噪（94% 耗时）而显存尖峰在 3D VAE 解码（118 GiB）的正交视图"
        },
        {
            "figure_id": "图 17-18",
            "number": 18,
            "file": "ch17-multimodal-world-model-infra-synthesis.svg",
            "asset": "ch17/ch17-multimodal-world-model-infra-synthesis.svg",
            "section": "17.4.2",
            "caption": "多模态与世界模型基础设施的算法—并行—缓存—解耦四维总架构图",
            "source_refs": ["case-studies/multimodal-stage-placement.md", "experiments/ch17/17-01/run.py"],
            "TeachingRole": "收束第 17 章三维潜空间、2D-USP、时间步复用与 E-P-D-G 四池解耦的工程总账本"
        },
        {
            "figure_id": "图 17-19",
            "number": 19,
            "file": "ch17-transfusion-hybrid-causal-bidirectional-mask.svg",
            "asset": "ch17/ch17-transfusion-hybrid-causal-bidirectional-mask.svg",
            "section": "17.4.3",
            "caption": "统一多模态模型（Transfusion / Show-o / Bagel）中文本因果掩码与图像块双向扩散掩码的混合排布及模式切换状态机（算例 17-7）",
            "source_refs": ["manuscripts/ch17/figure-data.json", "experiments/ch17/17-02/run.py"],
            "TeachingRole": "推导混合掩码 62.5% 有效面积、648/1024 分块跳过及 <boi> 触发 D→G 跨池迁移的定量准则"
        },
        {
            "figure_id": "图 17-20",
            "number": 20,
            "file": "ch17-3dgs-scaffold-guided-world-model-dit.svg",
            "asset": "ch17/ch17-3dgs-scaffold-guided-world-model-dit.svg",
            "section": "17.4.3",
            "caption": "显式 3DGS 几何光栅化提供深度/法线控制条件与视频 DiT 生成逼真物理交互画面的混合世界模型管线",
            "source_refs": ["manuscripts/ch17/figure-data.json", "experiments/ch17/17-02/run.py"],
            "TeachingRole": "剖析 2.4 ms 3DGS 光栅化 + 24.0 ms 4 步视频 DiT + 4.5 ms VAE 达成 32.36 FPS 零几何漂移的管线账本"
        },
        {
            "figure_id": "图 17-21",
            "number": 21,
            "file": "ch17-causal-streaming-world-model-20fps-kv-rolling.svg",
            "asset": "ch17/ch17-causal-streaming-world-model-20fps-kv-rolling.svg",
            "section": "17.4.3",
            "caption": "自回归流式因果视频世界模型（CausVid / Oasis / Matrix-Game）的块因果滚动 KV 缓存池（1 Sink + 5 滑窗，恒定 1.3184 GiB）架构（算例 17-8）",
            "source_refs": ["manuscripts/ch17/figure-data.json", "experiments/ch17/17-03/run.py"],
            "TeachingRole": "推导 1 Sink + 5 滑窗（21,600 词元）将无限时长世界模型 KV 显存锁定在 1.3184 GiB 的定量账本"
        },
        {
            "figure_id": "图 17-22",
            "number": 22,
            "file": "ch17-teacache-timestep-embedding-aware-activation-reuse.svg",
            "asset": "ch17/ch17-teacache-timestep-embedding-aware-activation-reuse.svg",
            "section": "17.4.3",
            "caption": "金字塔注意力广播（PAB）与 TeaCache 时间步嵌入感知特征复用在流式世界模型中的跨步/跨帧加速路径",
            "source_refs": ["manuscripts/ch17/figure-data.json", "experiments/ch17/17-03/run.py"],
            "TeachingRole": "解析空间/时间/交叉注意力金字塔广播与 TeaCache 如何将单帧延迟从 62.0 ms 压至 38.0 ms（26.32 FPS）"
        }
    ]
    for row in fig_index:
        row['figure'] = f"17-{row['number']}"
        row['name'] = Path(row['file']).stem
    (HERE / 'figure-index.json').write_text(json.dumps(fig_index, ensure_ascii=False, indent=2) + '\n')

    md = HERE.parent / '17-多模态、视频生成（DiT）与世界模型基础设施.md'
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
            + '})}catch(err){e.push({index:i,formula:s,error:String(err)})}});'
            'console.log(JSON.stringify({checked:a.length,display:'
            + str(len(display))
            + ',inline:'
            + str(len(inline))
            + ',errors:e}));if(e.length)process.exit(1);'
        )
        res = subprocess.run(
            ['node', '-e', js, str(katex_js)],
            input=json.dumps(formulas),
            text=True,
            capture_output=True,
            cwd=ROOT,
            check=True,
        )
        (HERE / 'math-validation.json').write_text(res.stdout)

        import markdown

        def repl_img(m):
            alt, rel = m.group(1), m.group(2)
            p = (md.parent / rel).with_suffix('.png')
            b64 = base64.b64encode(p.read_bytes()).decode()
            return f'<figure><img alt="{alt}" src="data:image/png;base64,{b64}"><figcaption>{alt}</figcaption></figure>'

        work = re.sub(r'!\[([^\]]*)\]\((ch17/[^)]+)\)', repl_img, text_md)
        body = markdown.markdown(work, extensions=['tables', 'fenced_code', 'footnotes', 'toc'])
        html = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>第 17 章：多模态、视频生成（DiT）与世界模型基础设施</title></head><body>{body}</body></html>\n'
        out_html = preview_path(md)
        out_html.write_text(html)

    sources = [HERE / 'build.py', HERE / 'figure-data.json', HERE / 'figure-index.json']
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
    print(f'Built {len(outputs)} figure files for Chapter 17; 0 layout warnings.')


if __name__ == '__main__':
    main()
