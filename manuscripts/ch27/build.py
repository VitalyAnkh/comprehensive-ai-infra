#!/usr/bin/env python3
"""Build all 16 publication-grade figures (420pt) and validation manifests for Chapter 27."""
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
        'ch27-sram-wafer-scale-vs-hbm-gpu-memory-wall-comparison',
        '70B 模型解码：4xH100 HBM（164.7）vs. 片上 SRAM（1,850 tok/s）',
        'blue',
        [
            ('① 4xH100 HBM 架构\n带宽 13.4 TB/s\n单步读权重 5.22 ms', 'orange'),
            ('② 全片上 SRAM 阵列\n聚合带宽 21 PB/s\n单步读仅 0.35 ms', 'blue'),
            ('③ 单流 1,850 tok/s\n较 H100 快 11.23x\n彻底击碎内存墙', 'teal'),
        ],
        '为什么 HBM 在 Batch=1 下必然撞墙而片上 SRAM 能十倍破局？',
        'teal',
        [
            ('片外 HBM 硅中介层物理极限：\n70 GB 权重每词元跨封装搬运\n加 AllReduce 耗时达 6.07 ms', 'orange'),
            ('全片上 SRAM 零片外访存架构：\n计算核心紧邻本地分布式 SRAM\n实现 21 PB/s 级芯片内超宽读取', 'teal'),
        ],
        '用 21 PB/s 全片上分布式 SRAM 替代片外 HBM，实现 11.23 倍单流提速',
    )

    draw_dual_band(
        ex,
        'ch27-cerebras-wse3-wafer-scale-noc-and-weight-streaming',
        'Cerebras WSE-3 晶圆级芯片（44 GB SRAM / 90 万核心）网格',
        'purple',
        [
            ('① 整片 300mm 晶圆\n不切割 84 曝光区\n90 万 AI 计算核心', 'blue'),
            ('② 44 GB 片上 SRAM\n均匀分布于各核心\n21 PB/s 本地带宽', 'purple'),
            ('③ 2D 晶圆级片上网\n跨曝光区无缝直连\n7 PB/s 织物带宽', 'teal'),
        ],
        '传统切割封装多裸片互连 vs. 晶圆级光刻跨区互连（Wafer-Scale）',
        'blue',
        [
            ('传统晶圆切割与 SerDes 税：\n裸片边缘 I/O 引脚密度受限\n跨芯片通信功耗与延迟飙升', 'orange'),
            ('跨划线光刻金属层直连：\n核心间延迟低至纳秒级单跳\n两台 WSE-3 全量驻留 70B 权重', 'teal'),
        ],
        '不切割晶圆保留原生金属互连，单张硅片兑现 44 GB SRAM 与 21 PB/s',
    )

    draw_dual_band(
        ex,
        'ch27-groq-lpu-320-chip-deterministic-daisy-chain-pipeline',
        'Groq LPU 320 颗芯片（230 MB/颗）全片上 SRAM 菊花链流水线',
        'teal',
        [
            ('① 单颗 230 MB SRAM\n片内带宽 80 TB/s\n零片外 DRAM 控制器', 'blue'),
            ('② 320 颗串联成环\n聚合容量 73.6 GB\n全量装入 70B FP8', 'purple'),
            ('③ 激活值沿链流转\n权重原地零搬移\n单步仅耗 0.54 ms', 'teal'),
        ],
        '单芯片大容量 HBM 路线 vs. 百颗小容量 SRAM 芯片串联路线',
        'purple',
        [
            ('单颗 14nm 成熟制程芯片边界：\n230 MB SRAM 良率高且成本可控\n规避 CoWoS 与 HBM 供应链瓶颈', 'blue'),
            ('空间展开式张量—流水混合切分：\n每组芯片固定驻留 1 层模型权重\n仅几 KB 激活向量跨芯片接力', 'teal'),
        ],
        '以 320 颗 SRAM 芯片空间展开 70B 模型，实现 80 TB/s/chip 吞吐',
    )

    draw_dual_band(
        ex,
        'ch27-3d-stacked-sram-soic-tsv-thermal-and-bandwidth-model',
        '3D 垂直堆叠 SRAM（TSMC SoIC 8 层 1.5 GB）与热阻耦合模型',
        'blue',
        [
            ('① 8 层 SRAM 堆叠\n单芯片达 1.5 GB\nTSV 间距 9.0 um', 'blue'),
            ('② 垂直硅穿孔互连\n带宽 12.8 TB/s/颗\n功耗仅 0.12 pJ/bit', 'purple'),
            ('③ 热阻耦合温升\n逻辑裸片贴液冷板\n温升锁定 6.4 摄氏度', 'teal'),
        ],
        '正装堆叠（SRAM 在上）vs. 倒装把高发热计算裸片直接贴合冷板',
        'teal',
        [
            ('传统计算裸片在底部的热瓶颈：\n热量需穿透 8 层 SRAM 界面\n热阻叠加导致结温超标降频', 'orange'),
            ('计算裸片朝上直触微通道冷板：\n500 W 算力热量直接向上排走\n下方 8 层低功耗 SRAM 温升 6.4℃', 'teal'),
        ],
        '利用 9 um 混合键合与倒装散热拓扑，将单芯片 SRAM 容量推向 GB 级',
    )

    draw_dual_band(
        ex,
        'ch27-sram-asic-vs-hbm-gpu-tco-crossover-by-batch-size',
        '片上 SRAM 阵列与 HBM GPU 的并发度—成本交叉点（B*=14.2）',
        'purple',
        [
            ('① 低并发 B <= 14\nSRAM 快 11.23 倍\n主导实时 Agent 交互', 'blue'),
            ('② 盈亏平衡点 B*\nB_cross = 14.2\n单位词元成本相等', 'purple'),
            ('③ 高并发 B = 128\nHBM 吞吐 6,400/s\n成本优势达 6.8 倍', 'teal'),
        ],
        '为什么云厂商需要“低并发 SRAM 加速池 + 高并发 HBM 批处理池”？',
        'blue',
        [
            ('SRAM 架构的容量资本开支制约：\n70 GB SRAM 需百颗芯片分摊\n扩大 Batch 无法进一步提速', 'orange'),
            ('HBM 架构的高并发算力复用：\n4 张 H100 在 B=128 下算力满载\n单词元摊销硬件成本降低 6.8 倍', 'teal'),
        ],
        '定量确立 B*=14.2 交叉律：交互式串行推理用 SRAM，高吞吐批处理用 HBM',
    )

    draw_dual_band(
        ex,
        'ch27-hardware-dynamic-arbitration-vs-cycle-accurate-compiler',
        '动态硬件缓存/路由仲裁 vs. 静态编译期周期精确确定性执行',
        'teal',
        [
            ('① 剥离缓存控制器\n零 L1/L2 未命中\nSRAM 地址编译期定', 'blue'),
            ('② 剥离分支预测器\n全静态展开指令流\n时钟周期严格可算', 'purple'),
            ('③ 软件调度片上网\n无包头无路由仲裁\n排队抖动 0.0 us', 'teal'),
        ],
        '传统 GPU 硬件反应式调度（Reactive）vs. LPU 编译器预知式编排',
        'purple',
        [
            ('GPU 动态仲裁的面积与抖动税：\n缓存标签与仲裁器占大量晶体管\n运行期抢占引发百微秒尾延迟', 'orange'),
            ('编译期静态排定每周期数据流：\n硅片面积全投向算力与 SRAM\n跨千次运行的耗时方差严格为零', 'teal'),
        ],
        '把运行时硬件动态仲裁前移至编译期静态规划，根除硬件层面的延迟抖动',
    )

    draw_dual_band(
        ex,
        'ch27-software-scheduled-tensor-stream-instruction-alignment',
        '跨 320 颗 SRAM 芯片的周期精确流寄存器对齐（偏差 <= 2 ns）',
        'blue',
        [
            ('① 矢量加载流指令\nSRAM 读入流寄存器\n按固定节拍注入', 'blue'),
            ('② 矩阵乘加流水线\n权重流与激活流相遇\n零停顿乘加归约', 'purple'),
            ('③ 硬件同步屏障\n每层校准晶振频偏\n跨芯片偏差 <= 2 ns', 'teal'),
        ],
        '异步包交换集合通信（NCCL）vs. 周期锁步波前传输（Wavefront）',
        'teal',
        [
            ('异步包交换网络的握手开销：\n发送端写描述符、网卡轮询、ACK\n仅协议握手即耗费数微秒时间', 'orange'),
            ('预定周期直接从端口读数据：\n编译器精确算出飞行时间 T_flight\n接收端在到达周期直接采样引脚', 'teal'),
        ],
        '用编译期飞行时间预计算与 ±2 ns 硬件锁相屏障替代异步网络握手协议',
    )

    draw_dual_band(
        ex,
        'ch27-sub-millisecond-hft-ai-kernel-bypass-nic-to-gpu-p2p',
        '亚毫秒级（16.5 us）高频量化 AI：网卡内核旁路 P2P 直写显存',
        'purple',
        [
            ('① 网卡硬件流过滤\nef_vi 旁路内核栈\n直收行情以太网帧', 'blue'),
            ('② PCIe P2P 直写\nDMA 写入 GPU BAR1\n耗时仅 2.8 us', 'purple'),
            ('③ GDR 网卡直发单\nGPU 写完成门铃\n发单仅耗 2.3 us', 'teal'),
        ],
        '传统 Linux Socket 拷贝栈（78 us）vs. 零拷贝 P2P 通路（16.5 us）',
        'blue',
        [
            ('传统内核协议栈与 CPU 中转瓶颈：\nSocket 软中断 18 us + 拷贝 6.5 us\n还没进 GPU 已耗掉 36.5 us', 'orange'),
            ('NIC-to-GPU 零 CPU 介入直通：\n行情包经 PCIe 交换机直落显存\n收发两端 I/O 总耗时仅 5.1 us', 'teal'),
        ],
        '消灭 OS 内核协议栈与主机内存双缓冲，打通 2.8 us 网卡至显存直写通道',
    )

    draw_dual_band(
        ex,
        'ch27-cuda-launch-overhead-vs-persistent-polling-megakernel',
        '持久化轮询超级内核（Persistent Megakernel）：0 us 启动开销',
        'teal',
        [
            ('① 传统逐层 Launch\n6 层耗 27.0 us\n驱动下发比算子还慢', 'orange'),
            ('② 常驻轮询超级核\nGPU 自旋读标志位\n启动开销 0.0 us', 'blue'),
            ('③ 全模型驻留 L2\n计算仅 11.4 us\n中位延迟 16.5 us', 'teal'),
        ],
        '为什么在小批量微秒级推理中必须抛弃 cudaLaunchKernel？',
        'purple',
        [
            ('主机驱动命令队列下发瓶颈：\n每次 Launch 需约 4.5 us 穿透 PCIe\n6 层小模型启动开销占总时延 35%', 'orange'),
            ('单超级内核融合全图与 L2 锁页：\n内核启动一次后全天候在 SM 轮询\n标志翻转后纳秒级开始首层计算', 'teal'),
        ],
        '让 GPU 内核反转为主动轮询者，彻底清零 27.0 us 主机启动与多层同步税',
    )

    draw_dual_band(
        ex,
        'ch27-linux-isolcpus-rt-throttling-and-p9999-tail-latency',
        'Linux 核心物理隔离（isolcpus）与 P99.99 尾延迟猎杀（24.8 us）',
        'blue',
        [
            ('① 物理核心隔离\nisolcpus + nohz_full\n剥离 OS 调度时钟', 'blue'),
            ('② 屏蔽核间中断\n网卡 IRQ 绑定专核\n关停 C-State 休眠', 'purple'),
            ('③ P99.99 锁 24.8 us\n较 420 us 降 16.94x\n收窄 P50-P9999 距', 'teal'),
        ],
        '未隔离标准 Linux 环境（P99.99=420 us）vs. 实时核隔离环境对比',
        'teal',
        [
            ('标准操作系统后台噪声源：\n1kHz 时钟中断、RCU 回调与缺页\n打断推理线程造成百微秒尖峰', 'orange'),
            ('无滴答独占核 + 显存持久驻留：\n推理全链路不经过任何 OS 调度器\nP99.99 尾延迟被死死钉在 24.8 us', 'teal'),
        ],
        '结合 GPU 持久轮询与 CPU 核隔离，实现 16.94 倍 P99.99 尾延迟削减',
    )

    draw_dual_band(
        ex,
        'ch27-bitnet-b158-ternary-weight-packing-5trits-per-byte',
        'BitNet b1.58 三值权重（{-1, 0, +1}）5-Trits-per-Byte 编码',
        'purple',
        [
            ('① 三值信息论极限\nlog2(3)=1.585 bits\n原生训练零精度坍塌', 'blue'),
            ('② 5 权重压入 1 Byte\n3^5 = 243 <= 256\n实测 1.60 bits/w', 'purple'),
            ('③ 70B 仅占 14.0 GB\n较 FP16 压缩 10.0x\n编码效率 99.06%', 'teal'),
        ],
        '简单 2-Bit 独立对齐编码（2.0 bits）vs. 基数 3 紧凑打包（1.60 bits）',
        'blue',
        [
            ('朴素 2-bit 编码的 25% 空间浪费：\n4 种状态仅用 3 种，70B 需 17.5 GB\n白白多耗 3.5 GB 宝贵片上缓存', 'orange'),
            ('5-Trits 进 1-Byte 无损进制打包：\n利用 243/256 状态逼近香农熵极限\n70B 模型缩至 14.0 GB 装入端侧内存', 'teal'),
        ],
        '以 99.06% 编码效率实现 10 倍模型压缩，让 70B 大模型仅占 14.0 GB',
    )

    draw_dual_band(
        ex,
        'ch27-fp16-mac-unit-vs-ternary-adder-tree-silicon-area-energy',
        '硬件乘加器革命：从 FP16 浮点乘法器到 BitNet 纯整数加法树',
        'teal',
        [
            ('① 传统 FP16 乘加器\n尾数乘法阵列庞大\n单次乘加 1.10 pJ', 'orange'),
            ('② 三值无乘法运算\nw in {-1, 0, +1}\n仅需符号取反与选择', 'blue'),
            ('③ 纯 INT8 加法树\n面积缩减逾 12 倍\n动态乘法功耗省 88.5%', 'teal'),
        ],
        '为什么三值量化能从物理底层改写 AI 芯片的硅片面积与能效极限？',
        'purple',
        [
            ('浮点乘法器的晶体管与功耗霸权：\n16 位乘法树占算术单元 80% 功耗\n限制了端侧与 3D 堆叠芯片的热密度', 'orange'),
            ('多路选择器 + 纯整数加法器：\n权重为 0 直接门控关断零翻转\n单位硅片面积算力密度跃升十倍', 'teal'),
        ],
        '彻底消灭矩阵乘中的硬件乘法器，将计算简化为符号多路选择与纯加法归约',
    )

    draw_dual_band(
        ex,
        'ch27-tmac-precomputed-activation-lut-simd-pshufb-kernel',
        'T-MAC 预计算激活查找表（g=4）与 SIMD 寄存器洗牌（pshufb）',
        'blue',
        [
            ('① 激活分组 g=4\n预计算 3^4=81 组合\n对称折半存 40 项', 'blue'),
            ('② 装入 SIMD 寄存器\n128-bit 向量驻留\n零 L1 缓存查表延迟', 'purple'),
            ('③ pshufb/tbl 查表\n1 条指令抵 4 次乘加\n端侧实测快 6.4 倍', 'teal'),
        ],
        '传统逐权重解包反量化（Dequant）vs. T-MAC 寄存器查找表内循环',
        'teal',
        [
            ('传统 W2A8 解包反量化开销：\n在内循环执行位掩码与类型转换\n算术指令被解包指令严重稀释', 'orange'),
            ('预计算激活表 + 字节索引查表：\n预计算开销沿输出维摊销至 0.1%\n主循环化为纯寄存器洗牌与累加', 'teal'),
        ],
        '用激活预计算换取主循环零乘法寄存器查表，实现 6.4 倍端侧提速',
    )

    draw_dual_band(
        ex,
        'ch27-bitblas-mixed-precision-w158a8-tensor-core-dequant',
        'BitBLAS 面向 GPU Tensor Core 的 W1.58A8 寄存器级软件流水',
        'purple',
        [
            ('① 紧凑权重加载\nHBM 仅读 1.6 bit/w\n访存带宽省 5~10 倍', 'blue'),
            ('② LOP3 位运算解包\n寄存器内无分支解码\n2 周期转 INT8 格式', 'purple'),
            ('③ INT8 Tensor Core\n送入 mma.sync.s8\n兼得高带宽与高算力', 'teal'),
        ],
        'CPU/NPU 端侧查找表（T-MAC）vs. GPU Tensor Core（BitBLAS）',
        'blue',
        [
            ('在通用 GPU 上运行三值网的挑战：\n现代 GPU 90% 算力在 Tensor Core\n不支持原生 1.58-bit 输入格式', 'orange'),
            ('HBM 省带宽 + 寄存器内即时解包：\n跨 HBM 搬运 14 GB 紧凑三值权重\n进寄存器后用 LOP3 送入 INT8 核', 'teal'),
        ],
        '在 HBM 侧享受 1.6-bit 压缩，在寄存器侧无缝对接 INT8 Tensor Core',
    )

    draw_dual_band(
        ex,
        'ch27-ultra-low-latency-and-1bit-inference-pareto-frontier',
        '超低延迟与三值量化协同：14 GB 70B 模型装入单颗 3D-SRAM',
        'teal',
        [
            ('① FP8 70B 需 70 GB\n需 320 颗 LPU 串联\n硬件 CapEx 昂贵', 'orange'),
            ('② BitNet 70B 14 GB\n仅需 10 颗 3D-SRAM\n芯片数锐减 32 倍', 'blue'),
            ('③ 算力再省 88.5%\n单流突破 4,000 tok/s\n成本与延迟双王冠', 'teal'),
        ],
        '当全片上 SRAM 架构遇上 BitNet b1.58：化解 SRAM 容量成本魔咒',
        'purple',
        [
            ('纯 SRAM 架构的最大商业软肋：\n每 GB SRAM 造价高昂导致机器昂贵\nFP8 下 70B 模型需庞大机柜阵列', 'orange'),
            ('1.58-bit 将 SRAM 需求砍掉 80%：\n14 GB 权重轻松塞入单机 3D-SRAM\n让千词元每秒极速推理走向平民化', 'teal'),
        ],
        '用三值量化治愈片上 SRAM 的容量昂贵症，开启单卡全片上 70B 极速推理',
    )

    draw_dual_band(
        ex,
        'ch27-extreme-latency-and-ternary-computing-synthesis',
        '第 27 章全景：晶圆级/3D-SRAM 架构与静态确定性编译（上）',
        'blue',
        [
            ('27.1 全片上 SRAM 架构：\n21 PB/s 带宽跑出 1,850 tok/s\n并发交叉点 B*=14.2 与 3D-SRAM', 'blue'),
            ('27.2 静态确定性编译器：\n零缓存未命中与零路由包仲裁\n320 芯片 ±2 ns 锁步零排队抖动', 'purple'),
        ],
        '第 27 章全景：微秒级高频 AI 系统与 BitNet 三值量化（下）',
        'teal',
        [
            ('27.3 微秒级高频 AI 推理：\nef_vi P2P + 持久轮询 Megakernel\nP50 16.5 us 且 P99.99 锁 24.8 us', 'blue'),
            ('27.4 BitNet b1.58 与 T-MAC：\n5 权重/Byte 将 70B 压至 14.0 GB\nSIMD 查找表实现 6.4x 提速省电', 'teal'),
        ],
        '从片上 SRAM 驻留、静态确定性编译、持久轮询到三值量化全栈贯通',
    )

    outputs = ex.finish()

    fig_index = [
        {
            "figure_id": "图 27-1",
            "number": 1,
            "file": "ch27-sram-wafer-scale-vs-hbm-gpu-memory-wall-comparison.svg",
            "asset": "ch27/ch27-sram-wafer-scale-vs-hbm-gpu-memory-wall-comparison.svg",
            "section": "27.1.1",
            "caption": "Llama-3.3 70B（FP8 70.0 GB）在 Batch=1 下的第一性原理对决：4xH100 HBM 架构（164.7 tok/s）vs. 全片上 SRAM 架构（21 PB/s，单步读取 0.35 ms，1,850.0 tok/s，提速 11.23 倍）（算例 27-1）",
            "source_refs": ["experiments/ch27/27-01/run.py", "manuscripts/ch27/figure-data.json"],
            "TeachingRole": "从物理带宽差两个数量级解释为什么全片上 SRAM 能将单流生成速度推至 1,850 tok/s"
        },
        {
            "figure_id": "图 27-2",
            "number": 2,
            "file": "ch27-cerebras-wse3-wafer-scale-noc-and-weight-streaming.svg",
            "asset": "ch27/ch27-cerebras-wse3-wafer-scale-noc-and-weight-streaming.svg",
            "section": "27.1.1",
            "caption": "Cerebras WSE-3 晶圆级引擎（Wafer-Scale Engine）架构：整片 300mm 硅晶圆跨曝光区互连、90 万 AI 核心与 44 GB 分布式片上 SRAM（21 PB/s）",
            "source_refs": ["experiments/ch27/27-01/run.py", "manuscripts/ch27/figure-data.json"],
            "TeachingRole": "展示不切割晶圆如何消除 SerDes 封装瓶颈并在单晶圆内提供 44 GB 超高速 SRAM"
        },
        {
            "figure_id": "图 27-3",
            "number": 3,
            "file": "ch27-groq-lpu-320-chip-deterministic-daisy-chain-pipeline.svg",
            "asset": "ch27/ch27-groq-lpu-320-chip-deterministic-daisy-chain-pipeline.svg",
            "section": "27.1.1",
            "caption": "Groq LPU 320 颗芯片（每颗 230 MB SRAM，片内带宽 80 TB/s）串联菊花链架构：全量驻留 70B 权重并让激活向量沿芯片环高速流转",
            "source_refs": ["experiments/ch27/27-01/run.py", "manuscripts/ch27/figure-data.json"],
            "TeachingRole": "剖析成熟制程小容量 SRAM 芯片如何通过百芯片空间展开承载大模型"
        },
        {
            "figure_id": "图 27-4",
            "number": 4,
            "file": "ch27-3d-stacked-sram-soic-tsv-thermal-and-bandwidth-model.svg",
            "asset": "ch27/ch27-3d-stacked-sram-soic-tsv-thermal-and-bandwidth-model.svg",
            "section": "27.1.2",
            "caption": "TSMC SoIC 8 层 3D 垂直堆叠 SRAM（1.5 GB/chip，TSV 带宽 12.8 TB/s）与计算裸片直触液冷板的倒装热阻耦合模型（温升仅 6.4 摄氏度）（算例 27-2）",
            "source_refs": ["experiments/ch27/27-01/run.py", "manuscripts/ch27/figure-data.json"],
            "TeachingRole": "推导 3D 堆叠 SRAM 的硅穿孔带宽密度与一维傅里叶热传导温升控制"
        },
        {
            "figure_id": "图 27-5",
            "number": 5,
            "file": "ch27-sram-asic-vs-hbm-gpu-tco-crossover-by-batch-size.svg",
            "asset": "ch27/ch27-sram-asic-vs-hbm-gpu-tco-crossover-by-batch-size.svg",
            "section": "27.1.2",
            "caption": "全片上 SRAM 阵列与 HBM GPU 超节点的每百万词元成本（USD/MTok）随并发批大小 B 变化的交叉曲线（盈亏平衡并发度 B_cross = 14.2）（算例 27-1）",
            "source_refs": ["experiments/ch27/27-01/run.py", "manuscripts/ch27/figure-data.json"],
            "TeachingRole": "定量界定片上 SRAM 架构（B <= 14 交互式场景）与 HBM 架构（B >= 32 高吞吐场景）的经济统治边界"
        },
        {
            "figure_id": "图 27-6",
            "number": 6,
            "file": "ch27-hardware-dynamic-arbitration-vs-cycle-accurate-compiler.svg",
            "asset": "ch27/ch27-hardware-dynamic-arbitration-vs-cycle-accurate-compiler.svg",
            "section": "27.2.1",
            "caption": "传统 GPU 动态硬件缓存/分支/路由仲裁架构 vs. 软件静态确定性编译器（Cycle-Accurate Deterministic Execution，排队抖动 0.0 us）",
            "source_refs": ["experiments/ch27/27-01/run.py", "manuscripts/ch27/figure-data.json"],
            "TeachingRole": "解释剥离硬件动态调度器后如何由编译器精确排定每一时钟周期的数据移动"
        },
        {
            "figure_id": "图 27-7",
            "number": 7,
            "file": "ch27-software-scheduled-tensor-stream-instruction-alignment.svg",
            "asset": "ch27/ch27-software-scheduled-tensor-stream-instruction-alignment.svg",
            "section": "27.2.2",
            "caption": "跨 320 颗 SRAM 芯片的编译期张量流发射时刻表与 ±2 ns 硬件锁相屏障（Zero-Jitter Hardware Barrier）（算例 27-3）",
            "source_refs": ["experiments/ch27/27-01/run.py", "manuscripts/ch27/figure-data.json"],
            "TeachingRole": "证明周期精确飞行时间预计算与微秒级晶振频偏校准如何实现跨百芯片零排队通信"
        },
        {
            "figure_id": "图 27-8",
            "number": 8,
            "file": "ch27-sub-millisecond-hft-ai-kernel-bypass-nic-to-gpu-p2p.svg",
            "asset": "ch27/ch27-sub-millisecond-hft-ai-kernel-bypass-nic-to-gpu-p2p.svg",
            "section": "27.3.1",
            "caption": "亚毫秒级高频量化与实时控制 AI 的内核旁路（ef_vi / DPDK）+ PCIe P2P 网卡直写 GPU BAR1 显存零拷贝数据通路（收包 2.8 us，发包 2.3 us）（算例 27-4）",
            "source_refs": ["experiments/ch27/27-02/run.py", "manuscripts/ch27/figure-data.json"],
            "TeachingRole": "展示如何绕过 Linux 内核协议栈与主机内存双缓冲实现微秒级数据入卡"
        },
        {
            "figure_id": "图 27-9",
            "number": 9,
            "file": "ch27-cuda-launch-overhead-vs-persistent-polling-megakernel.svg",
            "asset": "ch27/ch27-cuda-launch-overhead-vs-persistent-polling-megakernel.svg",
            "section": "27.3.2",
            "caption": "传统逐层 cudaLaunchKernel 驱动下发开销（27.0 us）vs. 持久化自旋轮询超级内核（Persistent Polling Megakernel，0.0 us 启动开销，计算 11.4 us）（算例 27-4）",
            "source_refs": ["experiments/ch27/27-02/run.py", "manuscripts/ch27/figure-data.json"],
            "TeachingRole": "揭示在小批量微秒级推理中用单持久内核融合全图并轮询显存标志位的关键机制"
        },
        {
            "figure_id": "图 27-10",
            "number": 10,
            "file": "ch27-linux-isolcpus-rt-throttling-and-p9999-tail-latency.svg",
            "asset": "ch27/ch27-linux-isolcpus-rt-throttling-and-p9999-tail-latency.svg",
            "section": "27.3.2",
            "caption": "结合 Linux 核心物理隔离（isolcpus + nohz_full）与持久轮询内核将端到端中位延迟降至 16.5 us、P99.99 尾延迟从 420.0 us 压至 24.8 us（削减 16.94 倍）（算例 27-4）",
            "source_refs": ["experiments/ch27/27-02/run.py", "manuscripts/ch27/figure-data.json"],
            "TeachingRole": "剖析操作系统调度中断对长尾延迟的破坏及实时核隔离工程方案"
        },
        {
            "figure_id": "图 27-11",
            "number": 11,
            "file": "ch27-bitnet-b158-ternary-weight-packing-5trits-per-byte.svg",
            "asset": "ch27/ch27-bitnet-b158-ternary-weight-packing-5trits-per-byte.svg",
            "section": "27.4.1",
            "caption": "BitNet b1.58 三值权重（{-1, 0, +1}，香农熵 1.585 bits）的 5-Trits-per-Byte（3^5=243 <= 256）无损打包编码：实测 1.60 bits/weight，70B 模型压缩 10 倍至 14.0 GB（算例 27-5）",
            "source_refs": ["experiments/ch27/27-03/run.py", "manuscripts/ch27/figure-data.json"],
            "TeachingRole": "推导三值编码的信息论效率（99.06%）与 5 权重进 1 字节的进制打包算法"
        },
        {
            "figure_id": "图 27-12",
            "number": 12,
            "file": "ch27-fp16-mac-unit-vs-ternary-adder-tree-silicon-area-energy.svg",
            "asset": "ch27/ch27-fp16-mac-unit-vs-ternary-adder-tree-silicon-area-energy.svg",
            "section": "27.4.1",
            "caption": "传统 FP16 浮点乘加器（MAC）与 BitNet b1.58 无乘法器纯整数加法树（消除 88.5% 动态乘法功耗）的硅片面积及能耗对比（算例 27-6）",
            "source_refs": ["experiments/ch27/27-03/run.py", "manuscripts/ch27/figure-data.json"],
            "TeachingRole": "从数字电路门级结构解释三值权重如何将矩阵乘简化为多路选择与纯加法"
        },
        {
            "figure_id": "图 27-13",
            "number": 13,
            "file": "ch27-tmac-precomputed-activation-lut-simd-pshufb-kernel.svg",
            "asset": "ch27/ch27-tmac-precomputed-activation-lut-simd-pshufb-kernel.svg",
            "section": "27.4.2",
            "caption": "T-MAC 预计算激活查找表（g=4，81 种组合对称折半）与 128-bit SIMD 寄存器洗牌查表指令（pshufb / vtbl），实现 6.4 倍端侧加速（算例 27-6）",
            "source_refs": ["experiments/ch27/27-03/run.py", "manuscripts/ch27/figure-data.json"],
            "TeachingRole": "展示如何利用 SIMD 寄存器查表指令在不具备原生三值硬件的 CPU/NPU 上实现高速推理"
        },
        {
            "figure_id": "图 27-14",
            "number": 14,
            "file": "ch27-bitblas-mixed-precision-w158a8-tensor-core-dequant.svg",
            "asset": "ch27/ch27-bitblas-mixed-precision-w158a8-tensor-core-dequant.svg",
            "section": "27.4.2",
            "caption": "BitBLAS 面向 GPU Tensor Core 的 W1.58A8 混合精度流水线：HBM 紧凑读取 + 寄存器 LOP3 位运算无分支解包 + INT8 MMA 矩阵乘",
            "source_refs": ["manuscripts/ch27/figure-data.json"],
            "TeachingRole": "打通 1.58-bit 极低显存带宽占用与现代 GPU INT8 Tensor Core 超高算力之间的桥梁"
        },
        {
            "figure_id": "图 27-15",
            "number": 15,
            "file": "ch27-ultra-low-latency-and-1bit-inference-pareto-frontier.svg",
            "asset": "ch27/ch27-ultra-low-latency-and-1bit-inference-pareto-frontier.svg",
            "section": "27.4.2",
            "caption": "全片上 3D-SRAM 架构与 BitNet b1.58 三值量化的化学反应：将 70B 模型压缩至 14.0 GB 装入 10 颗 3D-SRAM 芯片，兼得超低延迟与低硬件成本",
            "source_refs": ["manuscripts/ch27/figure-data.json"],
            "TeachingRole": "揭示三值量化如何将片上 SRAM 芯片需求量削减一个数量级并重塑 TCO 交叉点"
        },
        {
            "figure_id": "图 27-16",
            "number": 16,
            "file": "ch27-extreme-latency-and-ternary-computing-synthesis.svg",
            "asset": "ch27/ch27-extreme-latency-and-ternary-computing-synthesis.svg",
            "section": "27.4.2",
            "caption": "第 27 章系统工程全景图：从晶圆级/3D-SRAM 全片上驻留、静态确定性编译、微秒级内核旁路持久轮询到 BitNet b1.58 查找表计算",
            "source_refs": ["manuscripts/ch27/figure-data.json"],
            "TeachingRole": "收束全章四大支柱，总结突破内存带宽墙、调度抖动墙与乘法器能耗墙的统一工程法则"
        }
    ]
    for row in fig_index:
        row['figure'] = f"27-{row['number']}"
        row['name'] = Path(row['file']).stem
    (HERE / 'figure-index.json').write_text(json.dumps(fig_index, ensure_ascii=False, indent=2) + '\n')

    md = HERE.parent / '27-晶圆级与 3D-SRAM 专用推理芯片（Cerebras-Groq）、微秒级极低延迟 AI 与三值量化（BitNet b1.58）系统.md'
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

        work = re.sub(r'!\[([^\]]*)\]\((ch27/[^)]+)\)', repl_img, text_md)
        body = markdown.markdown(work, extensions=['tables', 'fenced_code', 'footnotes', 'toc'])
        html = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>第 27 章：晶圆级与 3D-SRAM 专用推理芯片（Cerebras-Groq）、微秒级极低延迟 AI 与三值量化（BitNet b1.58）系统</title></head><body>{body}</body></html>\n'
        out_html = preview_path(md)
        out_html.write_text(html)

    sources = [
        HERE / 'build.py',
        HERE / 'figure-data.json',
        HERE / 'figure-index.json',
        ROOT / 'experiments/ch27/27-01/run.py',
        ROOT / 'experiments/ch27/27-02/run.py',
        ROOT / 'experiments/ch27/27-03/run.py',
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
    print(f'Built {len(outputs)} figure files for Chapter 27; 0 layout warnings.')


if __name__ == '__main__':
    build_chapter()
