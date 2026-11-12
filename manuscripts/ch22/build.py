#!/usr/bin/env python3
"""Build all 22 publication-grade figures (420pt) and validation manifests for Chapter 22."""
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

    text(ax, 0.50, 0.072, footer, size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, stem)


def build_chapter() -> None:
    ex = Exporter(HERE)

    # Fig 1
    draw_dual_band(
        ex,
        'ch22-llm-1d-tokens-vs-ai4science-four-structural-domains',
        '一维自然语言大模型 vs. 生物复合物与百万碱基基因计算图',
        'blue',
        [
            ('自然语言大模型\n1D 离散词元序列\n受制于权重带宽墙', 'blue'),
            ('生物大分子预测\n2D 残基对 Z_ij\nO(N^3) 三角注意', 'purple'),
            ('百万碱基基因组\n1M 单核苷酸序列\n频域长卷积主干', 'teal'),
        ],
        '全球球面偏微分方程网格与亿级原子分子动力学图',
        'teal',
        [
            ('全球气象物理场（SFNO）：\n3600x7200 经纬度高分球面网格\n球谐变换 SHT + 2D 光环同步', 'blue'),
            ('分子动力学势函数（MACE）：\n三维空间截断半径图 + 球谐张量积\n飞秒步长受限于微秒级单步延迟墙', 'teal'),
        ],
        '科学模型参数量适中，但高阶几何激活张量与异构 I/O 成为核心瓶颈',
    )

    # Fig 2
    draw_dual_band(
        ex,
        'ch22-alphafold3-boltz1-two-stage-disaggregated-pipeline',
        '传统同机串行部署：45 分钟 CPU 检索闲置 GPU（算例 22-1）',
        'orange',
        [
            ('Stage 1 CPU 序列比对：\nJackHMMER 扫描 TB 级库\n耗时 2,700.0 s（45 分钟）', 'blue'),
            ('Stage 2 GPU 结构生成：\nPairformer + 扩散 48.0 s\nGPU 空转率高达 98.25%', 'orange'),
        ],
        '两阶段物理池解耦 + MMseqs2-GPU 加速管线（提速 55.85 倍）',
        'teal',
        [
            ('Stage 1 独立 GPU 检索池：\nMMseqs2-GPU 仅需 1.20 s\n单序列 MSA 加速 2,250.0 倍', 'blue'),
            ('Stage 2 异步队列供给 GPU：\n总耗时降至 49.20 s（快 55.85x）\n主干 GPU 利用率达 97.56%', 'teal'),
        ],
        '解耦 MSA 检索与几何扩散，是生物大分子预测系统的首要架构准则',
    )

    # Fig 3
    draw_dual_band(
        ex,
        'ch22-mmseqs2-gpu-kmer-prefilter-and-smith-waterman',
        'MMseqs2-GPU 第一阶段：片上共享内存 k-mer 查表预过滤',
        'blue',
        [
            ('目标库常驻显存\n紧凑氨基酸字节流\n消除磁盘随机读取', 'blue'),
            ('查询序列 6-mer 表\n驻留高速共享内存\n单拍并发查表匹配', 'purple'),
            ('无间隙对角线累加\n筛除 >99.5% 序列\n大幅压缩候选集合', 'teal'),
        ],
        'MMseqs2-GPU 第二阶段：Hopper/Blackwell 硬件 DPX 动态规划精排',
        'teal',
        [
            ('向量化带状动态规划比对：\n直接调用硬件 DPX 动态规划指令\n单拍完成加法、三路求最大与截断', 'blue'),
            ('单卡比对吞吐超越万核 CPU：\n将 2,700 秒检索压缩至 1.20 秒\n消除蛋白质结构预测的排头等待', 'teal'),
        ],
        '片上 SRAM k-mer 过滤结合硬件 DPX 指令，重塑了生物序列检索底座',
    )

    # Fig 4
    draw_dual_band(
        ex,
        'ch22-pairformer-triangle-multiplication-and-attention-o-n3',
        'Pairformer 二维残基对表示 Z_ij 与三维几何三角不等式约束',
        'purple',
        [
            ('残基对张量 Z_ij\nN_res x N_res x 128\nN=4096 占 4.0 GiB', 'blue'),
            ('三角乘法更新算子\n聚合中间残基 k 边\nO(N_res^3 * c_z)', 'purple'),
            ('三角自注意力算子\n对第 3 维 k 做注意\n传递空间几何约束', 'orange'),
        ],
        'N_res = 4,096 复合物的 O(N_res^3) 激活显存膨胀（算例 22-2）',
        'orange',
        [
            ('为何权重不足 1 GB 却溢出：\n4 头三角注意力形状为 H x N_res^3\n元素总数达 4 x 4096^3 = 2.75e11', 'blue'),
            ('朴素物化中间激活显存账本：\n单层 BF16 注意力达 512.00 GiB\n远超单张 H100（80 GB）显存上限', 'orange'),
        ],
        'Pairformer 是典型的“小参数量、巨型三次方激活张量”科学负载',
    )

    # Fig 5
    draw_dual_band(
        ex,
        'ch22-chunked-triangle-attention-and-dynamic-axial-parallelism',
        '分块三角注意力核融合（Chunked Triangle Attn，算例 22-2）',
        'blue',
        [
            ('外维分块 B = 64\n每次仅加载 64 行\n驻留片上 SRAM', 'blue'),
            ('在线 Softmax 融合\n寄存器广播加 b_jk\n零 HBM 物化 N^3', 'purple'),
            ('工作区缩减 512x\n从 512.00 GiB 压至\n仅需 1.00 GiB 显存', 'teal'),
        ],
        '4 卡动态轴向并行（DAP）：行列正交切分与 All-to-All 转置',
        'teal',
        [
            ('行注意按行切分，列注意转置：\n4 卡分摊后单卡 Z_ij 仅 1.0 GiB\n工作区压至 0.25 GiB（降 2048x）', 'blue'),
            ('NVLink 全互换通信开销极低：\n4 卡 All-to-All 转置仅 2.85 ms\n占三角算子计算（34.5ms）8.26%', 'teal'),
        ],
        '分块核融合消除三次方激活物化，动态轴向并行突破万残基显存上限',
    )

    # Fig 6
    draw_dual_band(
        ex,
        'ch22-all-atom-diffusion-module-multi-sample-batching',
        '全原子三维坐标扩散模块（All-Atom Diffusion）多构象批处理',
        'teal',
        [
            ('Pairformer 躯干\n单序列仅前向 1 次\n提取 z_ij 驻留显存', 'blue'),
            ('K=16 路噪声构象\n打包为扩散 Batch\n共享权重与条件', 'purple'),
            ('置信度打分排序\n并行输出 16 个结构\n筛选自由能最优解', 'teal'),
        ],
        '共享躯干多样本扩散的 Roofline 算术强度倍增效应',
        'blue',
        [
            ('单构象扩散困于小批量带宽墙：\n仅生成 K=1 个构象时矩阵过窄\nTensor Core 利用率不足 15%', 'orange'),
            ('K=16 多构象并批摊薄权重读取：\n算术强度升 16 倍，生成 16 构象\n仅比生成单构象增加 14% 耗时', 'teal'),
        ],
        '一躯干对多扩散头的非对称批处理，将结构采样吞吐提升一个数量级',
    )

    # Fig 7
    draw_dual_band(
        ex,
        'ch22-evo2-stripedhyena2-multi-scale-hyena-attention-backbone',
        '百万碱基基因组模型（Evo 2 40B）StripedHyena 2 多尺度主干',
        'blue',
        [
            ('单核苷酸分辨率\nL=1,048,576 (1M)\n捕获 1bp 剪接突变', 'blue'),
            ('短/中程门控卷积\nHyena-SE / MR\n提取密码子语法', 'purple'),
            ('FlashFFTConv\n配合稀疏自注意力\n跨百万碱基建模', 'teal'),
        ],
        'L = 1M bp 下二次自注意力与长卷积的算力账本对比（算例 22-3）',
        'teal',
        [
            ('纯 Transformer 二次注意力：\n4*L^2*d = 36,028.8 TFLOPs\n单卡单层数十秒，难以训练', 'orange'),
            ('FlashFFTConv 频域长卷积：\n单层 1.74 TFLOPs（降 20,764x）\n含门控投影单层实测提速 28.5x', 'teal'),
        ],
        '亚二次多尺度长卷积结合稀疏注意力，是支撑百万碱基基因组的关键',
    )

    # Fig 8
    draw_dual_band(
        ex,
        'ch22-flashfftconv-monarch-decomposition-tensor-core',
        'FlashFFTConv：利用 Monarch 矩阵分解将 FFT 映射至张量核',
        'purple',
        [
            ('1. 序列二维矩阵化\n将 L=N1xN2 序列\n重排为 1024x1024', 'blue'),
            ('2. Monarch 矩阵乘\n调 Tensor Core 执行\n子块傅里叶变换', 'purple'),
            ('3. 片上频域乘融合\nSRAM 内滤波门控\n零 HBM 中间落盘', 'teal'),
        ],
        '标准 cuFFT 蝶形访存 vs. FlashFFTConv 核融合硬件效率对比',
        'teal',
        [
            ('标准 cuFFT 的内存带宽瓶颈：\n三次往返读写 HBM 且仅用 ALU\n算术强度极低，受限于带宽墙', 'orange'),
            ('Monarch 分解唤醒矩阵引擎：\n将长卷积转为矩阵乘并片上融合\n使长卷积发挥矩阵引擎高吞吐', 'teal'),
        ],
        '用 Monarch 矩阵分解将 FFT 改写为 Tensor Core 原语并驻留 SRAM',
    )

    # Fig 9
    draw_dual_band(
        ex,
        'ch22-sfno-spherical-harmonics-and-2d-grid-decomposition',
        '全球气象大模型（FourCastNet 3 / Aurora）球面傅里叶算子',
        'blue',
        [
            ('0.05° 全球气象网格\n3600x7200x128 通道\n消除极区投影畸变', 'blue'),
            ('经度维快速 FFT\n纬度高斯勒让德求积\n伴随勒让德矩阵乘', 'purple'),
            ('谱域滤波+局部卷积\n兼顾全球罗斯贝波\n与局部台风锋面', 'teal'),
        ],
        '64 卡（8 x 8）二维空间域网格分解与边界光环账本（算例 22-4）',
        'teal',
        [
            ('单卡负责 450 x 900 经纬度子域：\n比一维切分具有更短二维分块周长\n光环 w_h=4 仅含 10,800 个格点', 'blue'),
            ('轻量级跨卡边界同步流量：\n单卡每步四向 Halo 仅 5.53 MB\nNVLink/RDMA 传输仅需 0.18 ms', 'teal'),
        ],
        '二维空间域分解兼顾球谐子群全互换与局部差分最小周长',
    )

    # Fig 10
    draw_dual_band(
        ex,
        'ch22-spatial-domain-halo-exchange-computation-overlap',
        '二维/三维物理场网格并行：内部计算与边界光环（Halo）双流掩盖',
        'teal',
        [
            ('优先算边界壳层\n打包 10,800 格点\n触发异步 P2P 发送', 'blue'),
            ('主流算内部子域\n442x892 核心网格\n耗时 14.20 ms', 'purple'),
            ('汇合 Halo 更新边界\n暴露等待 0.00 ms\n100% 通信掩盖', 'teal'),
        ],
        '表面积/体积比定律（Surface-to-Volume Ratio）与强扩展极限',
        'blue',
        [
            ('光环通信占比仅 1.27%：\n通信 0.18ms 远小于计算 14.20ms\n通信完全隐藏于内部场更新下', 'blue'),
            ('何时会出现强扩展退化：\n若子域缩至 <32x32，边界超 35%\n需用 NVSHMEM 单边直写降延迟', 'orange'),
        ],
        '先算边界、双流异步交换光环并推进内部子域实现零暴露通信',
    )

    # Fig 11
    draw_dual_band(
        ex,
        'ch22-era5-petabyte-zarr-gds-direct-streaming-dataloader',
        'PB 级多维气象场（ERA5 Zarr/HDF5）GPUDirect Storage 零拷贝流',
        'blue',
        [
            ('Zarr 四维分块对齐\n按时间与气压切块\n消除随机寻道开销', 'blue'),
            ('GDS cuFile 直通\nNVMe 直写 GPU 显存\n旁路主机页缓存', 'purple'),
            ('GPU nvCOMP 解压\n流处理器并发解压\n吞吐 >120 GB/s', 'teal'),
        ],
        '为何传统数据加载器会在气象与流体训练中引发严重 I/O 饥饿',
        'teal',
        [
            ('高分辨率科学数据的体积冲击：\n单个 0.05° 全球气象场帧达数 GB\nCPU 解压使 GPU 利用率跌破 20%', 'orange'),
            ('存储—显存零拷贝硬直通：\n用 GDS 与 nvCOMP 卸载至 GPU\n持续以百 GB/s 带宽供给训练', 'teal'),
        ],
        '科学大模型数据供给须升级为 GDS 零拷贝直通与 GPU 原位解压',
    )

    # Fig 12
    draw_dual_band(
        ex,
        'ch22-mace-equivariant-message-passing-and-verlet-neighbor-list',
        '等变神经原子势（MACE）亿级原子分子动力学三维空间分解',
        'purple',
        [
            ('三维空间块分解\n切分 3D 物理子域\n同步幽灵壳层原子', 'blue'),
            ('Verlet 缓冲邻居表\nr_list=r_cut+r_skin\n每 24 步重构 1 次', 'purple'),
            ('cuEquivariance\n融合球谐张量积\nCUDA Graph 固化', 'teal'),
        ],
        '化解分子动力学“百万飞秒顺序步长”的微秒级延迟瓶颈',
        'teal',
        [
            ('时间维不可并行的严苛约束：\n需顺序积分百万至十亿个飞秒步\n单步若慢 10ms，算 1 纳秒需数小时', 'orange'),
            ('邻居表摊销 + 内核全图固化：\n消除 93% 空间哈希与发射开销\n单步亚毫秒，日产数十纳秒轨迹', 'teal'),
        ],
        '三维块分解突破规模上限，Verlet 缓冲与球谐核融合压低步长延迟',
    )

    # Fig 13
    draw_dual_band(
        ex,
        'ch22-ai-in-the-loop-hybrid-hpc-pde-solver-and-surrogate',
        'AI-in-the-Loop：传统超算 PDE 求解器与神经代理模型耦合',
        'blue',
        [
            ('传统 MPI 求解器\n严格求解守恒方程\n推进粗网格主干', 'blue'),
            ('DLPack / IPC 桥\n零拷贝共享显存指针\n交接延迟 < 5 us', 'purple'),
            ('Tensor Core 代理\n高效加速湍流参数化\n与微观物理源项', 'teal'),
        ],
        '告别“写磁盘文件交换数据”的传统超算松耦合反模式',
        'teal',
        [
            ('物理守恒律与非线性拟合互补：\n纯 AI 外推易偏离质量/能量守恒\n纯数值求解器在微观项算力受限', 'blue'),
            ('同节点显存原位交接（In-Situ）：\nFortran 与 PyTorch 共享显存\n数值超算与深度学习同作业协同', 'teal'),
        ],
        '数值方程守住守恒底线，神经代理加速微观项，零拷贝共享粘合二者',
    )

    # Fig 14
    draw_dual_band(
        ex,
        'ch22-autonomous-wet-lab-active-learning-closed-loop',
        '自动化科学发现工厂：从数字生成仿真到机器人湿实验回流',
        'teal',
        [
            ('1. 生成式结构设计\nMatterGen/Boltz\n日产百万候选结构', 'blue'),
            ('2. 多级仿真筛选\nMACE 弛豫+DFT\n筛出百个潜力候选', 'purple'),
            ('3. 机器人湿实验\n自动合成与表征\n主动学习回流修正', 'teal'),
        ],
        '主动学习（Active Learning）跨越“静态历史数据库边界”',
        'blue',
        [
            ('仅靠公开静态数据集的局限：\nPDB 与材料库仅覆盖已知稳态样本\n缺乏失败负样本与极端工况数据', 'orange'),
            ('不确定度驱动的迭代探索：\n挑方差最大的分子送机械臂实测\n实验回流使靶点亲和力持续跃升', 'teal'),
        ],
        '数字生成仿真与自动化湿实验协同迭代，驱动科学发现持续演进',
    )

    # Fig 15
    draw_dual_band(
        ex,
        'ch22-tokamak-plasma-tearing-mode-sub-ms-rl-control',
        '托卡马克等离子体撕裂模 4,000 Hz（200 us 预算）实时控制流水',
        'blue',
        [
            ('1. 磁探针 RDMA 直写\n14.0 us 零拷贝入显存\n旁路主机内核协议栈', 'blue'),
            ('2. L2 锁存融合推理\n31.0+17.0=48.0 us\n提前 28.5 ms 预警', 'purple'),
            ('3. 安全联锁与下发\n联锁投影 6.5 us\n总计 68.5 us (34.3%)', 'teal'),
        ],
        '主机中转动态调度（435.0 us 超时）vs. 静态图 L2 驻留（68.5 us）',
        'teal',
        [
            ('传统主机中转 + 逐算子 HBM：\nSocket 85 + 发射 208 + 访存 142\n总耗时 435.0 us 超出 200 us 预算', 'orange'),
            ('GPUDirect + CUDA Graph 锁频：\n权重常驻片上 L2 SRAM 零抖动\n提速 6.35 倍，P99.9 仅 74.0 us', 'teal'),
        ],
        '消除主机中转与 HBM 往返，以 68.5 us 确定性控制抑制撕裂模',
    )

    # Fig 16
    draw_dual_band(
        ex,
        'ch22-single-cell-atlas-sparse-gene-expression-dataloader',
        '亿级单细胞图谱（scGPT / Geneformer）CSR 稀疏加载与非零打包',
        'purple',
        [
            ('1. CSR 稀疏块直读\n4,096->615.2 GB\n存储体积省 6.66 倍', 'blue'),
            ('2. 仅发射非零基因\n20,480->1,536 基因\n序列压缩 13.33 倍', 'purple'),
            ('3. 变长拼包自注意力\n平方计算降 177.8 倍\n训练吞吐升 20.00 倍', 'teal'),
        ],
        '稠密零填充矩阵（92.5% 零元素浪费）vs. 动态非零基因打包流水线',
        'blue',
        [
            ('传统稠密 20,480 基因全长输入：\n92.5% 表达为零耗费无效注意力\n单细胞整网高达 68.40 GFLOPs', 'orange'),
            ('CSR 紧缩直读 + Flash-Varlen：\n仅对活跃表达基因计算自注意力\n降至 3.42 GFLOPs（提速 20.0x）', 'teal'),
        ],
        '顺应转录组 92.5% 天然稀疏性，消除 177.8 倍冗余二次方注意力',
    )

    # Fig 17
    draw_dual_band(
        ex,
        'ch22-five-domain-scientific-ai-workload-atlas',
        '五大科学领域跨尺度统一账本（上）：高阶几何与超长序列',
        'blue',
        [
            ('生物复合物 AF3\n4 卡 DAP 通信 8.26%\n工作区缩减 2,048x', 'blue'),
            ('基因组 Evo 2 40B\n1M 碱基 Monarch FFT\n混合算子省 20,764x', 'purple'),
            ('全球气象 SFNO\n8x8 域 Halo 5.53MB\n通信比 1.27% 零暴露', 'teal'),
        ],
        '五大科学领域跨尺度统一账本（下）：双精度重构与微秒级磁控',
        'teal',
        [
            ('量子化学 DFT / CCSD(T)：\nOzaki S=7 有符号 INT8 重构\n3.68x 提速达 147.2 TFLOPS', 'blue'),
            ('聚变等离子体 4,000 Hz 磁控：\nRDMA + L2 静态图 68.5 us\n守住 200 us 截止期（余 131.5 us）', 'teal'),
        ],
        '五大科学领域沿显存、带宽、通信周界、双精度与微秒截止期映射',
    )

    # Fig 18
    draw_dual_band(
        ex,
        'ch22-unified-laws-across-all-24-chapters-synthesis',
        '贯穿全书的四大物理守恒准则（上）：算术强度与状态局部性',
        'purple',
        [
            ('准则一：算术强度与带宽边界\nAI = FLOPs / Bytes\n跨过拐点发挥矩阵引擎吞吐', 'blue'),
            ('准则二：状态生命周期局部性\nSRAM -> HBM -> DRAM -> SSD\n分块融合与异构拆池消除空转', 'purple'),
        ],
        '贯穿全书的四大物理守恒准则（下）：关键路径掩盖与有效产出',
        'teal',
        [
            ('准则三：通信掩盖与尾延迟\n暴露延迟 = max(0, 通信 - 计算)\n等周切分与双流重叠隐藏传输', 'blue'),
            ('准则四：可验证有效科学产出\n兼顾机器精度、守恒律与截止期\n将每焦耳电能转化为可信发现', 'teal'),
        ],
        '算术强度、状态局部性、通信掩盖与物理守恒律贯穿跨尺度科学底座',
    )

    # Fig 19
    draw_dual_band(
        ex,
        'ch22-ozaki-scheme-fp64-mantissa-slicing-on-int8-tensor-cores',
        '有符号中心 Ozaki 尾数切片与 INT8 张量核零溢出内积（算例 22-8）',
        'blue',
        [
            ('1. 行列中心化切片\n[-127, 127] 就近舍入\n每片推进约 7.4 bits', 'blue'),
            ('2. 上三角交叉乘积\np+q <= S+1 裁剪\nS=7 仅算 28 次 GEMM', 'purple'),
            ('3. INT32 精确累加\nK <= 133,144 零溢出\nFP64 尾声升序重构', 'teal'),
        ],
        '双精度裁剪架构（INT8 4500 TOPS vs. FP64 40 TFLOPS）吞吐跃迁',
        'teal',
        [
            ('原生 FP64 双精度单元受限：\n硅片面积向低比特矩阵核倾斜\n原生双精度峰值仅 40.0 TFLOPS', 'orange'),
            ('Ozaki S=7 重构 51.8 bits 精度：\n唤醒 112.5 倍吞吐 INT8 张量核\n实测提速 3.68x（147.2 TFLOPS）', 'teal'),
        ],
        '有符号中心切片结合上三角裁剪，在低比特张量核重建双精度矩阵乘',
    )

    # Fig 20
    draw_dual_band(
        ex,
        'ch22-ozaki-slice-count-accuracy-vs-speedup-crossover',
        'Ozaki 切片档位 S = 5..8 的有效尾数精度与 GEMM 次数递进',
        'purple',
        [
            ('S=5..6 中高精档\n15 / 21 次 INT8 乘\n37.2 / 44.6 bits', 'blue'),
            ('S=7 主力甜点档\n28 次上三角 GEMM\n51.8 bits 准双精度', 'purple'),
            ('S=8 满精度等价档\n36 次上三角 GEMM\n53.0 bits 满双精度', 'teal'),
        ],
        '跨芯片峰值比盈亏阈值与新架构端到端加速比曲线（算例 22-8）',
        'blue',
        [
            ('硬件峰值比盈亏翻转条件：\nS=7 需 INT8/FP64 > 30.57x\nH100 (29.5x) 宜用原生 FP64', 'orange'),
            ('新一代 112.5x 架构全面胜出：\nS=6/7/8 实测快 4.85x/3.68x/2.88x\n兼顾 DFT 收敛精度与百 TFLOPS', 'teal'),
        ],
        '按矩阵条件数与芯片算力比自适应选切片数 S，兼顾精度与高吞吐',
    )

    # Fig 21
    draw_dual_band(
        ex,
        'ch22-insitu-mpi-dlpack-gpu-coupling-vs-filesystem-io-wall',
        '传统松耦合反模式：千卡 384 GB 耦合场落盘读写停顿 58.2 秒',
        'orange',
        [
            ('MPI 求解器写磁盘：\n384 GB 经 PCIe 落 Lustre\n元数据锁争用与全局屏障', 'orange'),
            ('Python 代理读回显存：\n重复主机内存与 H2D 拷贝\n单步耦合停顿达 58.2 s', 'orange'),
        ],
        '同卡 DLPack / CUDA IPC 零拷贝显存原位耦合（0.41 s，快 142.0x）',
        'teal',
        [
            ('共享物理 HBM 页表零拷贝：\n单卡 375 MB 封装 DLPack 视图\n设备指针移交耗时低于 5 us', 'blue'),
            ('原位布局转换 + 张量核推理：\n零磁盘 I/O 与零 PCIe 往返\n单步仅 0.41 s（提速 142.0 倍）', 'teal'),
        ],
        '以 DLPack 显存零拷贝替代文件落盘，消除混合仿真 I/O 停顿墙',
    )

    # Fig 22
    draw_dual_band(
        ex,
        'ch22-conservation-law-projection-layer-long-rollout-stability',
        '硬约束拉格朗日守恒正交投影算子三步融合流水（算例 22-10）',
        'blue',
        [
            ('1. 局部 FP64 点积\n算子尾声乘求积权重\n零额外显存遍历开销', 'blue'),
            ('2. 五标量全局归约\n1024 卡 AllReduce\n通信耗时 < 18 us', 'purple'),
            ('3. 超平面正交投影\n拉格朗日原位修正\n附加开销仅 +3.1%', 'teal'),
        ],
        '500 步长程自回归推演：无约束质量漂移崩溃 vs. 机器精度守恒',
        'teal',
        [
            ('无硬约束神经代理自回归推演：\n单步 2.16e-4 微小偏差复利累积\n500 步总质量漂移 11.4% 致发散', 'orange'),
            ('融合拉格朗日正交投影层：\n每步严格拉回守恒积分超平面\n500 步相对误差仅 4.2e-13', 'teal'),
        ],
        '以 +3.1% 尾声正交投影开销，保障长程神经 PDE 推演严格物理守恒',
    )

    outputs = ex.finish()

    fig_index = [
        {
            "figure_id": "图 22-1",
            "number": 1,
            "file": "ch22-llm-1d-tokens-vs-ai4science-four-structural-domains.svg",
            "asset": "ch22/ch22-llm-1d-tokens-vs-ai4science-four-structural-domains.svg",
            "section": "22.1.1",
            "caption": "一维文本 Token 与四大科学智能领域（生物大分子 2D 对表示、基因组 1M 核苷酸序列、地球球面 PDE 网格、三维原子截断半径图）的计算图与复杂度对比",
            "source_refs": ["manuscripts/ch22/figure-data.json"],
            "TeachingRole": "揭示科学基础模型参数量适中但高阶几何激活张量呈二次/三次方膨胀且伴随显著异构 I/O 失配的共性特征"
        },
        {
            "figure_id": "图 22-2",
            "number": 2,
            "file": "ch22-alphafold3-boltz1-two-stage-disaggregated-pipeline.svg",
            "asset": "ch22/ch22-alphafold3-boltz1-two-stage-disaggregated-pipeline.svg",
            "section": "22.2.1",
            "caption": "生物大分子预测两阶段异构解耦架构：Stage 1（MMseqs2-GPU 序列检索池，1.20 s）与 Stage 2（Pairformer + 扩散生成池，48.0 s）独立扩缩容实现 55.85 倍端到端提速（算例 22-1）",
            "source_refs": ["experiments/ch22/22-01/run.py", "manuscripts/ch22/figure-data.json"],
            "TeachingRole": "解决传统同机串行部署中 45 分钟 CPU MSA 搜索导致 H100/B200 空转 98.25% 的资源错配问题"
        },
        {
            "figure_id": "图 22-3",
            "number": 3,
            "file": "ch22-mmseqs2-gpu-kmer-prefilter-and-smith-waterman.svg",
            "asset": "ch22/ch22-mmseqs2-gpu-kmer-prefilter-and-smith-waterman.svg",
            "section": "22.2.1",
            "caption": "MMseqs2-GPU 利用片上共享内存 k-mer 查表与硬件 DPX 动态规划指令将 MSA 搜索从 2,700 秒压降至 1.20 秒（2,250.0 倍加速）的核内流水（算例 22-1）",
            "source_refs": ["experiments/ch22/22-01/run.py", "manuscripts/ch22/figure-data.json"],
            "TeachingRole": "拆解片上 SRAM 预过滤与 Hopper/Blackwell 原生 DPX 动态规划指令在序列同源检索中的硬件加速机理"
        },
        {
            "figure_id": "图 22-4",
            "number": 4,
            "file": "ch22-pairformer-triangle-multiplication-and-attention-o-n3.svg",
            "asset": "ch22/ch22-pairformer-triangle-multiplication-and-attention-o-n3.svg",
            "section": "22.2.2",
            "caption": "AlphaFold 3 / Boltz-1 Pairformer 中残基对表示 Z_ij（4.00 GiB）的三角乘法更新与 O(N_res^3) 三角自注意力（朴素物化达 512.00 GiB）几何依赖图（算例 22-2）",
            "source_refs": ["experiments/ch22/22-01/run.py", "manuscripts/ch22/figure-data.json"],
            "TeachingRole": "阐明三维空间三角不等式约束如何导致 N_res=4,096 时单层激活张量突破 512 GiB 物理显存上限"
        },
        {
            "figure_id": "图 22-5",
            "number": 5,
            "file": "ch22-chunked-triangle-attention-and-dynamic-axial-parallelism.svg",
            "asset": "ch22/ch22-chunked-triangle-attention-and-dynamic-axial-parallelism.svg",
            "section": "22.2.2",
            "caption": "分块三角注意力核融合（B=64 压至 1.00 GiB）与 4 卡动态轴向并行（DAP 行列正交切分 + All-to-All 转置，工作区压至 0.25 GiB，缩减 2,048 倍）（算例 22-2）",
            "source_refs": ["experiments/ch22/22-01/run.py", "manuscripts/ch22/figure-data.json"],
            "TeachingRole": "通过内核级在线 Softmax 融合与正交轴向切分化解万残基生物复合物的显存瓶颈"
        },
        {
            "figure_id": "图 22-6",
            "number": 6,
            "file": "ch22-all-atom-diffusion-module-multi-sample-batching.svg",
            "asset": "ch22/ch22-all-atom-diffusion-module-multi-sample-batching.svg",
            "section": "22.2.2",
            "caption": "全原子三维坐标扩散模块（All-Atom Diffusion）在共享同一躯干 Pairformer 特征下并行生成 K=16 个候选构象的算术强度倍增",
            "source_refs": ["manuscripts/ch22/figure-data.json"],
            "TeachingRole": "利用一躯干对多扩散头的非对称批处理将结构去噪推入 Tensor Core 计算密集区"
        },
        {
            "figure_id": "图 22-7",
            "number": 7,
            "file": "ch22-evo2-stripedhyena2-multi-scale-hyena-attention-backbone.svg",
            "asset": "ch22/ch22-evo2-stripedhyena2-multi-scale-hyena-attention-backbone.svg",
            "section": "22.3.1",
            "caption": "百万碱基基因组模型（Evo 2 40B）的 StripedHyena 2（短/中/长三级门控卷积 + 稀疏旋转注意力）混合主干架构（算例 22-3）",
            "source_refs": ["experiments/ch22/22-02/run.py", "manuscripts/ch22/figure-data.json"],
            "TeachingRole": "解释单核苷酸分辨率（L=1,048,576 bp）下为何必须用亚二次长卷积替代 36,028.8 TFLOPs 的二次自注意力"
        },
        {
            "figure_id": "图 22-8",
            "number": 8,
            "file": "ch22-flashfftconv-monarch-decomposition-tensor-core.svg",
            "asset": "ch22/ch22-flashfftconv-monarch-decomposition-tensor-core.svg",
            "section": "22.3.2",
            "caption": "FlashFFTConv 利用 Monarch 矩阵分解将长序列快速傅里叶变换（FFT）映射为 Tensor Core 矩阵乘并驻留 SRAM（单层提速 28.5 倍）（算例 22-3）",
            "source_refs": ["experiments/ch22/22-02/run.py", "manuscripts/ch22/figure-data.json"],
            "TeachingRole": "揭示如何通过二维矩阵重排与 Monarch 分解将访存受限的 FFT 转化为片上高密矩阵乘流水"
        },
        {
            "figure_id": "图 22-9",
            "number": 9,
            "file": "ch22-sfno-spherical-harmonics-and-2d-grid-decomposition.svg",
            "asset": "ch22/ch22-sfno-spherical-harmonics-and-2d-grid-decomposition.svg",
            "section": "22.4.1",
            "caption": "全球气象大模型（FourCastNet 3 / Aurora）的球面傅里叶神经算子（SFNO）与 64 卡（8x8）经纬度二维空间域切分（算例 22-4）",
            "source_refs": ["experiments/ch22/22-02/run.py", "manuscripts/ch22/figure-data.json"],
            "TeachingRole": "在 3,600x7,200 全球高分网格上实现极区无畸变球谐变换与最小周长二维分块"
        },
        {
            "figure_id": "图 22-10",
            "number": 10,
            "file": "ch22-spatial-domain-halo-exchange-computation-overlap.svg",
            "asset": "ch22/ch22-spatial-domain-halo-exchange-computation-overlap.svg",
            "section": "22.4.1",
            "caption": "物理网格并行中内部格点计算（14.20 ms）与边界幽灵区光环交换（Halo Exchange，5.53 MB / 0.18 ms，占比 1.27%）的双流完全掩盖时序（算例 22-4）",
            "source_refs": ["experiments/ch22/22-02/run.py", "manuscripts/ch22/figure-data.json"],
            "TeachingRole": "证明表面积/体积比定律下优先计算边界壳层可使跨卡 Halo 通信暴露延迟归零"
        },
        {
            "figure_id": "图 22-11",
            "number": 11,
            "file": "ch22-era5-petabyte-zarr-gds-direct-streaming-dataloader.svg",
            "asset": "ch22/ch22-era5-petabyte-zarr-gds-direct-streaming-dataloader.svg",
            "section": "22.4.1",
            "caption": "面向 PB 级多维气象场（ERA5 Zarr/HDF5 分块存储）的 GPUDirect Storage 零拷贝按需切片与 nvCOMP 硬件解压流水线",
            "source_refs": ["manuscripts/ch22/figure-data.json"],
            "TeachingRole": "消除高分辨率科学场训练中的 CPU 页缓存拷贝与单线程解压瓶颈"
        },
        {
            "figure_id": "图 22-12",
            "number": 12,
            "file": "ch22-mace-equivariant-message-passing-and-verlet-neighbor-list.svg",
            "asset": "ch22/ch22-mace-equivariant-message-passing-and-verlet-neighbor-list.svg",
            "section": "22.4.2",
            "caption": "等变神经原子势函数（MACE）在亿级原子分子动力学中的三维空间块分解、动态 Verlet 皮肤缓冲邻居表与球谐核融合",
            "source_refs": ["manuscripts/ch22/figure-data.json"],
            "TeachingRole": "化解分子动力学百万飞秒顺序步长下的微秒级单步启动与邻居重构开销"
        },
        {
            "figure_id": "图 22-13",
            "number": 13,
            "file": "ch22-ai-in-the-loop-hybrid-hpc-pde-solver-and-surrogate.svg",
            "asset": "ch22/ch22-ai-in-the-loop-hybrid-hpc-pde-solver-and-surrogate.svg",
            "section": "22.4.2",
            "caption": "传统 HPC 偏微分方程求解器与 AI 神经代理模型（Neural Surrogate）在同一个超算作业内经 DLPack/IPC 零拷贝原位耦合的混合仿真架构",
            "source_refs": ["manuscripts/ch22/figure-data.json"],
            "TeachingRole": "结合数值求解器的离散守恒性与深度学习代理模型在亚网格微观过程上的加速能力"
        },
        {
            "figure_id": "图 22-14",
            "number": 14,
            "file": "ch22-autonomous-wet-lab-active-learning-closed-loop.svg",
            "asset": "ch22/ch22-autonomous-wet-lab-active-learning-closed-loop.svg",
            "section": "22.4.2",
            "caption": "自动化科学发现工厂：生成式分子/材料设计（MatterGen/Boltz-1）-> 多级物理仿真筛选 -> 自动化机器人湿实验验证 -> 主动学习回流",
            "source_refs": ["manuscripts/ch22/figure-data.json"],
            "TeachingRole": "构建从数字生成仿真到真实物理实验室机器人验证的主动学习迭代管线"
        },
        {
            "figure_id": "图 22-15",
            "number": 15,
            "file": "ch22-tokamak-plasma-tearing-mode-sub-ms-rl-control.svg",
            "asset": "ch22/ch22-tokamak-plasma-tearing-mode-sub-ms-rl-control.svg",
            "section": "22.4.2",
            "caption": "托卡马克核聚变等离子体磁流体（MHD）撕裂模 4,000 Hz（200.0 us 预算）实时强化学习控制推理底座：GPUDirect RDMA（14.0 us）+ L2 锁存静态图（48.0 us）+ 联锁投影（6.5 us）将延迟从 435.0 us 压至 68.5 us（提速 6.35 倍）（算例 22-5）",
            "source_refs": ["experiments/ch22/22-03/run.py", "manuscripts/ch22/figure-data.json"],
            "TeachingRole": "阐明硬实时科学控制回路如何消除主机 PCIe 往返与 HBM 读取抖动，以 68.5 us 确定性延迟抑制撕裂模破裂"
        },
        {
            "figure_id": "图 22-16",
            "number": 16,
            "file": "ch22-single-cell-atlas-sparse-gene-expression-dataloader.svg",
            "asset": "ch22/ch22-single-cell-atlas-sparse-gene-expression-dataloader.svg",
            "section": "22.4.2",
            "caption": "亿级单细胞转录组图谱（scGPT / Geneformer，1 亿细胞 x 20,480 基因，非零率 7.5%）的 CSR 稀疏压缩存储（4,096 -> 615.20 GB，6.66 倍）与动态非零序列打包（二次方注意力降低 177.78 倍，端到端训练提速 20.00 倍）（算例 22-6）",
            "source_refs": ["experiments/ch22/22-03/run.py", "manuscripts/ch22/figure-data.json"],
            "TeachingRole": "利用 CSR 稀疏块直读与变长非零基因 Token 拼包消除 92.5% 零表达基因的二次方算力开销"
        },
        {
            "figure_id": "图 22-17",
            "number": 17,
            "file": "ch22-five-domain-scientific-ai-workload-atlas.svg",
            "asset": "ch22/ch22-five-domain-scientific-ai-workload-atlas.svg",
            "section": "22.5.1",
            "caption": "五大科学计算智能领域（生物大分子、百万碱基基因组、全球气象流场、量子化学双精度求解与托卡马克硬实时控制）的算术强度、通信拓扑与实时性统一图谱（算例 22-7）",
            "source_refs": ["calculations/results/ch22-science-ozaki-coupling.json", "experiments/ch22/22-02/run.py", "manuscripts/ch22/figure-data.json"],
            "TeachingRole": "横向拉通五大科学计算领域的显存压缩比、通信计算比、双精度重构吞吐与硬实时截止期裕度"
        },
        {
            "figure_id": "图 22-18",
            "number": 18,
            "file": "ch22-unified-laws-across-all-24-chapters-synthesis.svg",
            "asset": "ch22/ch22-unified-laws-across-all-24-chapters-synthesis.svg",
            "section": "22.5.2",
            "caption": "贯穿全书的统一物理底座——从单个矩阵乘法的算术强度，到驾驭语言、全模态、机器人与科学发现的四大智能计算准则",
            "source_refs": ["experiments/ch22/22-02/run.py", "manuscripts/ch22/figure-data.json"],
            "TeachingRole": "以算术强度、状态局部性、通信周界掩盖与可验证有效科学产出统一跨尺度基础设施设计准则"
        },
        {
            "figure_id": "图 22-19",
            "number": 19,
            "file": "ch22-ozaki-scheme-fp64-mantissa-slicing-on-int8-tensor-cores.svg",
            "asset": "ch22/ch22-ozaki-scheme-fp64-mantissa-slicing-on-int8-tensor-cores.svg",
            "section": "22.6.1",
            "caption": "基于 INT8 张量核重构 FP64 双精度矩阵乘法的有符号中心 Ozaki 尾数切片、上三角交叉乘积裁剪与 FP64 升序归约流水线（算例 22-8）",
            "source_refs": ["calculations/results/ch22-science-ozaki-coupling.json", "experiments/ch22/22-04/run.py", "manuscripts/ch22/figure-data.json"],
            "TeachingRole": "阐明 [-127, 127] 有符号就近舍入如何在 K <= 133,144 下保证硬件 INT32 累加器零溢出并实现 3.68 倍端到端双精度加速"
        },
        {
            "figure_id": "图 22-20",
            "number": 20,
            "file": "ch22-ozaki-slice-count-accuracy-vs-speedup-crossover.svg",
            "asset": "ch22/ch22-ozaki-slice-count-accuracy-vs-speedup-crossover.svg",
            "section": "22.6.1",
            "caption": "有符号中心 Ozaki 方案在不同切片数 S=5..8 下的有效尾数精度比特、INT8 GEMM 次数与端到端加速比交叉曲线（算例 22-8）",
            "source_refs": ["calculations/results/ch22-science-ozaki-coupling.json", "experiments/ch22/22-04/run.py", "manuscripts/ch22/figure-data.json"],
            "TeachingRole": "量化切片数 S、芯片峰值比临界阈值与目标矩阵条件数之间的工程选型边界"
        },
        {
            "figure_id": "图 22-21",
            "number": 21,
            "file": "ch22-insitu-mpi-dlpack-gpu-coupling-vs-filesystem-io-wall.svg",
            "asset": "ch22/ch22-insitu-mpi-dlpack-gpu-coupling-vs-filesystem-io-wall.svg",
            "section": "22.6.2",
            "caption": "千卡三维大气海洋混合模式中并行文件系统落盘耦合（58.2 s）对比同卡 DLPack/CUDA IPC 零拷贝显存原位耦合（0.41 s，提速 142.0 倍）的时序与数据通路（算例 22-9）",
            "source_refs": ["calculations/results/ch22-science-ozaki-coupling.json", "experiments/ch22/22-05/run.py", "manuscripts/ch22/figure-data.json"],
            "TeachingRole": "展示如何用 DLManagedTensor 显存零拷贝共享替代数百 GB 级并行文件读写以拔除混合仿真 I/O 墙"
        },
        {
            "figure_id": "图 22-22",
            "number": 22,
            "file": "ch22-conservation-law-projection-layer-long-rollout-stability.svg",
            "asset": "ch22/ch22-conservation-law-projection-layer-long-rollout-stability.svg",
            "section": "22.6.2",
            "caption": "500 步长程神经偏微分方程自回归推演中无约束预测引发 11.4% 质量漂移发散对比融合拉格朗日正交投影层（附加开销 +3.1%，相对误差压制在 < 1e-12）的稳定性对比（算例 22-10）",
            "source_refs": ["calculations/results/ch22-science-ozaki-coupling.json", "experiments/ch22/22-05/run.py", "manuscripts/ch22/figure-data.json"],
            "TeachingRole": "证明算子尾声融合拉格朗日最小二乘超平面正交投影可在极低开销下保证长程推演严格物理守恒"
        },
    ]
    for row in fig_index:
        row['figure'] = f"22-{row['number']}"
        row['name'] = Path(row['file']).stem
    (HERE / 'figure-index.json').write_text(json.dumps(fig_index, ensure_ascii=False, indent=2) + '\n')

    md = HERE.parent / '22-科学计算智能系统.md'
    if md.exists():
        text_md = md.read_text(encoding='utf-8')
        work_math = re.sub(r'^```.*?^```', '', text_md, flags=re.M | re.S)
        work_math = re.sub(r'`[^`\n]+`', '', work_math)
        display = re.findall(r'\$\$(.+?)\$\$', work_math, re.S)
        work_no_disp = re.sub(r'\$\$.+?\$\$', '', work_math, flags=re.S)
        inline = re.findall(r'(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)', work_no_disp)
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

        work = re.sub(r'!\[([^\]]*)\]\((ch22/[^)]+)\)', repl_img, text_md)
        body = markdown.markdown(work, extensions=['tables', 'fenced_code', 'footnotes', 'toc'])
        html = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>第 22 章：科学计算智能系统</title></head><body>{body}</body></html>\n'
        out_html = preview_path(md)
        out_html.write_text(html, encoding='utf-8')

    sources = [
        HERE / 'build.py',
        HERE / 'figure-data.json',
        HERE / 'figure-index.json',
        ROOT / 'calculations/results/ch22-science-ozaki-coupling.json',
        ROOT / 'experiments/ch22/22-01/run.py',
        ROOT / 'experiments/ch22/22-02/run.py',
        ROOT / 'experiments/ch22/22-03/run.py',
        ROOT / 'experiments/ch22/22-04/run.py',
        ROOT / 'experiments/ch22/22-05/run.py',
        md,
    ]
    (HERE / 'sources.json').write_text(
        json.dumps(
            {"sources": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in sources if p.exists()]},
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
    print(f'Built {len(outputs)} figure files for Chapter 22; 0 layout warnings.')


if __name__ == '__main__':
    build_chapter()
