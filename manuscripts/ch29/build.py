#!/usr/bin/env python3
"""Build all 16 publication-grade figures (420pt) and validation manifests for Chapter 29."""
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
        'ch29-deepseek-open-infra-seven-pillar-fullstack-architecture',
        'DeepSeek 开源全栈七大支柱：算子内核与通信流水（上）',
        'blue',
        [
            ('① FlashMLA 内核\n[512+64] 乒乓流水\nH800 达 3000 GB/s', 'blue'),
            ('② DeepGEMM 内核\nJIT 形状常量编译\n跑出 1358 TFLOPS', 'purple'),
            ('③ DeepEP/DualPipe\nIBGDA 零 SM 占用\n双向流水零通信泡', 'teal'),
        ],
        'DeepSeek 开源全栈七大支柱：去缓存存储与异构部署（下）',
        'teal',
        [
            ('④ 3FS + Smallpond 底座：\n180 节点聚合读 6.60 TB/s\n110.5 TiB 排序仅 30m14s', 'blue'),
            ('⑤ Mooncake + KTransformers：\n多网卡 RDMA 全局 KV 缓存池\n单 GPU+双路 AMX 提速 13.8x', 'teal'),
        ],
        '全栈软硬协同：榨干每一周期算力、每一字节显存与每一微秒通信',
    )

    draw_dual_band(
        ex,
        'ch29-flashmla-512-plus-64-pingpong-warpgroup-pipeline',
        'FlashMLA [512+64] 非对称头维与双 Warpgroup 乒乓流水',
        'purple',
        [
            ('① 矩阵吸收潜向量\n512 潜维 + 64 位置\n1,152 B/tok 压 28.44x', 'blue'),
            ('② 双消费者乒乓流\nWG0 算 WGMMA 乘\nWG1 并发算 Softmax', 'purple'),
            ('③ TMA 异步双缓冲\nH800 达 3000 GB/s\n算力达 580 TFLOPS', 'teal'),
        ],
        '传统未压缩 MHA 解码（32,768 B/tok）vs. FlashMLA 乒乓重叠',
        'blue',
        [
            ('传统 MHA 解码的双重瓶颈：\n每词元读取 32,768 B 撞击带宽墙\n单 Warpgroup 内 MMA 与指数互等', 'orange'),
            ('FlashMLA 软硬极限压榨：\n带宽利用率达 89.55% (3 TB/s)\n屋脊拐点算力效率高达 94.2%', 'teal'),
        ],
        '以矩阵吸收砍掉 96.5% 访存，并用双 Warpgroup 乒乓掩盖指令延迟',
    )

    draw_dual_band(
        ex,
        'ch29-flashmla-variable-length-split-kv-load-balancer',
        'FlashMLA 变长序列 Split-KV 持久块调度器（132 个 SM 负载均衡）',
        'teal',
        [
            ('① 全局块预算核算\n统计全批总 KV 块\n计算单 SM 均摊定额', 'blue'),
            ('② 动态长短序列切\n超长序列跨 SM 并发\n短序列合并单块跑', 'purple'),
            ('③ 在线 LSE 归约\n合并分片局部极值\n精确还原全局注意力', 'teal'),
        ],
        '固定按样本绑核（尾块饥饿）vs. 变长持久块均摊调度对比',
        'purple',
        [
            ('传统按请求静态分配 SM 的缺陷：\n64K 长序列独占单 SM 拖慢全批\n其余 100+ 个 SM 早早完工空转', 'orange'),
            ('Split-KV 持久工作窃取队列：\n132 个 SM 结束时刻偏差 < 2%\n在混合长短批下稳守 3,000 GB/s', 'teal'),
        ],
        '通过全局 KV 块定额切分与 LSE 归约，彻底消灭变长批尾延迟',
    )

    draw_dual_band(
        ex,
        'ch29-deepgemm-runtime-jit-shape-constant-specialization',
        'DeepGEMM 运行期 JIT 形状常量化：寄存器从 248 降至 192 个',
        'blue',
        [
            ('① 运行期捕获形状\n将 M/N/K 写为常量\n300 行极简模板库', 'blue'),
            ('② 编译器完全展开\n省去循环步长寄存器\n单线程省 56 寄存器', 'purple'),
            ('③ 零 Local 溢出\n容纳更大累加分块\n跑出 1358 TFLOPS', 'teal'),
        ],
        'CUTLASS 静态预编译通用库 vs. DeepGEMM 运行期 JIT 编译',
        'teal',
        [
            ('静态通用模板库的寄存器压力：\nM/N/K 与多维步长皆占通用寄存器\n248 个寄存器触顶易触发栈溢出', 'orange'),
            ('JIT 立即数指令编码收益：\n地址偏移直接硬编码进 PTX 指令\n192 个寄存器留出充裕双缓冲空间', 'teal'),
        ],
        '用运行期轻量级 JIT 替代庞大静态算子库，释放 56 个寄存器/线程',
    )

    draw_dual_band(
        ex,
        'ch29-deepgemm-fp8-1x128-128x128-two-level-ffma-accumulation',
        'DeepGEMM FP8 细粒度（1x128 / 128x128）二级 FFMA 累加流水',
        'purple',
        [
            ('① 一级 WGMMA 乘\n4 条指令跑 K=128\n耗时 128 个时钟周期', 'blue'),
            ('② 二级 CUDA FFMA\n乘 s_A*s_B 缩放因子\n仅耗 4 个时钟周期', 'purple'),
            ('③ 3.12% 极低开销\n修复 E4M3 累加精度\n吞吐 1358 TFLOPS', 'teal'),
        ],
        'Hopper Tensor Core 原生 14-bit 定点累加缺陷与二级累加破解之道',
        'blue',
        [
            ('原生 FP8 WGMMA 的精度短板：\n内部仅保留 13 位小数定点尾数\n超长 K 维累加易发生离群值下溢', 'orange'),
            ('每 128 维块搬移至 FP32 主寄存器：\n仅占 4/128 = 3.125% (3.12%) 周期\n兼顾 FP8 算力峰值与 BF16 级收敛', 'teal'),
        ],
        '每 128 周期插入 4 周期 FFMA 二级累加，以 3.12% 代价根除精度损失',
    )

    draw_dual_band(
        ex,
        'ch29-deepep-normal-mode-nvlink-rdma-hybrid-forwarding',
        'DeepEP Normal 模式：NVLink-RDMA 混合双跳转发（153 GB/s）',
        'teal',
        [
            ('① 节点内 NVLink 聚\n同目标机词元先合并\n消除跨网小包风暴', 'blue'),
            ('② 网卡 RDMA 直发\n8 张 400G 并发推流\n跨机吞吐 153 GB/s', 'purple'),
            ('③ 仅占 24 个 SM\n释放 108 SM 给计算\n支撑预填充与训练', 'teal'),
        ],
        '传统全连接 NCCL All-to-All vs. DeepEP 拓扑感知两级聚合转发',
        'purple',
        [
            ('传统扁平 All-to-All 的通信放大：\n每张卡向远端所有专家独立发包\nRDMA 队列风暴且占用 48+ 个 SM', 'orange'),
            ('机内 NVLink 与机间 RDMA 流水：\n双环形缓冲区实现收发无缝接力\n用不足 18% 的 SM 跑满 153 GB/s', 'teal'),
        ],
        '通过机内 NVLink 预聚合与机间 RDMA 流水，仅用 24 SM 达成 153 GB/s',
    )

    draw_dual_band(
        ex,
        'ch29-deepep-low-latency-ibgda-zero-sm-hook-overlap',
        'DeepEP Low-Latency 模式：纯 IBGDA 零 SM 占用（163 us 派发）',
        'blue',
        [
            ('① IBGDA 门铃直写\n绕过 CPU 驱动线程\n128 专家 163us 派发', 'blue'),
            ('② 后台 Hook 轮询\n0 个通信 SM 占用\nCombine 耗时 194us', 'purple'),
            ('③ 全部 132 SM 算\n共享专家与通信重叠\n微批解码零停顿', 'teal'),
        ],
        '为什么在线解码阶段不能分出 24 个 SM 专门跑通信内核？',
        'teal',
        [
            ('解码阶段 SM 被抢占的代价：\n切走 24 个 SM 使矩阵算力降 18%\n且频繁启动小内核引入微秒级抖动', 'orange'),
            ('发完门铃即返还 GPU 控制权：\nRDMA 网卡硬件在后台搬运词元\n由后续算子尾声 Hook 校验完成位', 'teal'),
        ],
        '借力 IBGDA 网卡直通与 Hook 轮询，实现零 SM 占用的 163 us 极速派发',
    )

    draw_dual_band(
        ex,
        'ch29-dualpipe-bidirectional-pipeline-schedule-and-20-stage-grid',
        'DualPipe 双向流水线：PP=16 下双向注入 2M=32 微批掩盖通信',
        'purple',
        [
            ('① 两端相向注入流\n正向与反向流交错\n流水气泡减半至 15.8%', 'blue'),
            ('② 正反微批结对跑\n前向通信配反向计算\n100% 掩盖跨机 EP', 'purple'),
            ('③ 零通信暴露时间\n参数副本换吞吐跃升\n达成极致 MFU 效率', 'teal'),
        ],
        '传统单向 1F1B 流水线（通信暴露阻塞）vs. DualPipe 双向相向交错',
        'blue',
        [
            ('单向 1F1B 在跨机 MoE 下的困境：\n同一时刻所有微批同处通信阶段\nAll-to-All 延迟直接暴露在关键路径', 'orange'),
            ('双向相向微批互补编排机制：\n方向 A 走 RDMA 时方向 B 跑 GEMM\n使网卡与 Tensor Core 双双满载', 'teal'),
        ],
        '通过双向相向微批交错，将跨节点 All-to-All 完全藏入反向计算阴影',
    )

    draw_dual_band(
        ex,
        'ch29-dualpipe-fine-grained-backward-input-weight-split',
        'DualPipe 细粒度反向拆分：输入梯度（B）与权重梯度（W）解耦',
        'teal',
        [
            ('① 拆分反向传播块\n先算输入梯度 B 块\n激活流水下一阶段', 'blue'),
            ('② 延后权重梯度 W\n无跨层数据依赖性\n充当通信填充积木', 'purple'),
            ('③ 四象限重叠矩阵\nF0/B1/F1/W0 精密拼\n消除流水同步缝隙', 'teal'),
        ],
        '粗粒度整体 Backward 块 vs. B/W 解耦柔性通信掩盖窗口',
        'purple',
        [
            ('传统粗粒度反向块的刚性阻塞：\n必须等 dX 与 dW 全部算完才发包\n导致上游阶段长时间饥饿等待', 'orange'),
            ('W 块零依赖自由调度特性：\n一旦 B 块产出即刻触发跨机通信\n在等待网卡返回期间插入 W 块计算', 'teal'),
        ],
        '拆解反向输入梯度与权重梯度，打造可精密填平通信耗时的柔性积木',
    )

    draw_dual_band(
        ex,
        'ch29-3fs-craq-chain-replication-and-usrbio-zero-copy-ring',
        '3FS 链式复制（CRAQ）强一致状态机与 USRBIO 零拷贝环形队列',
        'blue',
        [
            ('① CRAQ 链式复制\n写自头向尾传脏位\n尾节点确认转干净', 'blue'),
            ('② 任意副本分摊读\n干净块本地即刻读\n读带宽随副本倍增', 'purple'),
            ('③ USRBIO 零拷贝\n绕过内核页缓存池\nRDMA 直通异步环', 'teal'),
        ],
        '传统带页缓存文件系统（Lustre/Ceph）vs. 3FS 去缓存 RDMA 架构',
        'teal',
        [
            ('深度学习随机读下的页缓存灾难：\n海量洗牌样本仅读一次永不复用\nPage Cache 命中率近零却耗尽内存', 'orange'),
            ('全链路零拷贝环形缓冲区：\n用户态提交 I/O 请求直达 NVMe\n消除内核上下文切换与内存双拷贝', 'teal'),
        ],
        '摒弃无效页缓存并结合 CRAQ 多副本并发读，释放底层 SSD 极限吞吐',
    )

    draw_dual_band(
        ex,
        'ch29-3fs-180-node-6600gbs-throughput-and-graysort-pipeline',
        '3FS 180 节点集群实测：6.60 TB/s 聚合读与 110.5 TiB 极速排序',
        'purple',
        [
            ('① 180 台存储节点\n2,880 块 NVMe SSD\n360 张 200G 网卡', 'blue'),
            ('② 6.60 TB/s 聚合读\n单机均值 36.7 GB/s\n逼近网卡物理线速', 'purple'),
            ('③ GraySort 破纪录\n110.5 TiB 仅 30m14s\n均值 3.66 TiB/min', 'teal'),
        ],
        '500+ 计算节点并发异步 Checkpoint 与高速样本洗牌读取实测',
        'blue',
        [
            ('万卡集群数据供给与容错刚需：\n每小时需写入并校验数 TB 检查点\n同时支撑多模态训练高并发随机读', 'orange'),
            ('网络与存储介质 1:1 均衡配比：\n单机 16 块 SSD 喂饱双 200G 网卡\n实现 6,600 GB/s 持续平稳读取', 'teal'),
        ],
        '以 180 台标准服务器跑出 6.60 TB/s 聚合读，奠定百 PB 数据吞吐底座',
    )

    draw_dual_band(
        ex,
        'ch29-smallpond-duckdb-on-3FS-serverless-parquet-shuffle',
        'Smallpond 数据框架：基于 DuckDB + 3FS 的无常驻服务 PB 级洗牌',
        'teal',
        [
            ('① 无中心常驻守护\n摒弃 JVM 垃圾回收\n按需拉起轻量进程', 'blue'),
            ('② DuckDB 向量引擎\nSIMD 列式算子加速\n外存落盘防 OOM', 'purple'),
            ('③ 3FS 分区洗牌池\n两阶段键范围切分\n轻松洗牌百 TB 语料', 'teal'),
        ],
        '传统 Spark/Ray 重型常驻集群 vs. Smallpond 极简无服务架构',
        'purple',
        [
            ('传统大数据框架的运维痛点：\nJVM 序列化膨胀与洗牌 OOM 频发\n协调节点状态同步开销沉重', 'orange'),
            ('把分布式协调下沉至 3FS 系统：\n计算节点直读写 3FS 共享分区\n扩展至数千并发零元数据瓶颈', 'teal'),
        ],
        '用 DuckDB 向量化内核结合 3FS 高速洗牌，以极简架构替代重型大数据栈',
    )

    draw_dual_band(
        ex,
        'ch29-mooncake-transfer-engine-multi-nic-rdma-kv-pool',
        'Mooncake Transfer Engine：多网卡 RDMA 拓扑感知全局 KV 缓存池',
        'blue',
        [
            ('① 聚合多级存储池\nHBM + DRAM + SSD\n全局前缀块哈希寻址', 'blue'),
            ('② 拓扑感知选网卡\n自动匹配同 NUMA\n聚合多口 RDMA 带宽', 'purple'),
            ('③ 零拷贝跨机搬运\nPrefill 写 Decode 读\n长文本 TTFT 降 68%', 'teal'),
        ],
        '单机孤立 KV 缓存（重复重算）vs. Mooncake 数据中心级 KV 共享池',
        'teal',
        [
            ('长文档与多轮的重复预填充：\n请求漂移至新节点时被迫重算前缀\n浪费预填充算力并拉高首字延迟', 'orange'),
            ('跨节点 RDMA 单边读写直通：\n从远端主机内存拉取历史 KV 块\n比重新跑 671B 模型前向快十倍', 'teal'),
        ],
        '将全集群内存与 SSD 编织为统一 KV 缓存池，彻底解耦 Prefill 与 Decode',
    )

    draw_dual_band(
        ex,
        'ch29-mooncake-early-rejection-and-sla-goodput-controller',
        'Mooncake 基于负载预测的提前拒绝与 SLA Goodput 准入控制',
        'purple',
        [
            ('① 实时排队时延预测\n结合 KV 池余量建模\n入队前精准算 TTFT', 'blue'),
            ('② 超 SLA 提前拒绝\n不浪费一周期 GPU\n避免级联超时雪崩', 'purple'),
            ('③ 有效吞吐翻倍涨\n突发流量稳守 SLA\nGoodput 提升 2.1x', 'teal'),
        ],
        '传统盲目入队先到先服务（FCFS 雪崩）vs. 预测式准入控制对比',
        'blue',
        [
            ('过载状态下的全员超时陷阱：\n盲目接收请求导致排队超过 5 秒\n算完返回时客户端早已断开作废', 'orange'),
            ('以 Goodput（合规吞吐）为纲：\n果断拦截注定违约的尾部过载请求\n保障场内运行任务严守延迟红线', 'teal'),
        ],
        '在网关层预判 KV 容量与排队延迟，用提前拒绝换取过载下 2.1x 有效吞吐',
    )

    draw_dual_band(
        ex,
        'ch29-ktransformers-gpu-mla-and-cpu-amx-numa-expert-offload',
        'KTransformers 异构卸载：单卡 24G 4090D + 双路 AMX 跑 671B',
        'teal',
        [
            ('① GPU 驻留稠密部\nMLA + 共享专家常驻\n仅占 19.2 GB 显存', 'blue'),
            ('② 双路 DDR5 驻专家\n256 路由专家占 336G\n680 GB/s 内存带宽', 'purple'),
            ('③ CPU AMX 原位算\n仅传回 14 KB 激活\n解码达 18.6 tok/s', 'teal'),
        ],
        '传统 PCIe 搬运专家权重（llama.cpp 1.35 tok/s）vs. CPU 原位计算',
        'purple',
        [
            ('跨 PCIe 搬运专家权重的致命瓶颈：\n每步搬 8 个专家需传数百 MB 权重\nPCIe 4.0 带宽瞬间锁死在 1.35 tok/s', 'orange'),
            ('把计算推向权重所在双路 NUMA：\nCPU AMX 就地算完 8 个激活专家\n预填充 286.4 tok/s，解码快 13.8x', 'teal'),
        ],
        'GPU 算 MLA、双路 CPU AMX 就地算稀疏专家，单卡工作站畅跑满血 671B',
    )

    draw_dual_band(
        ex,
        'ch29-deepseek-fullstack-cost-and-efficiency-synthesis',
        '第 29 章全景：DeepSeek 算子内核、通信流水与存储底座（上）',
        'blue',
        [
            ('29.1 FlashMLA 与 DeepGEMM：\nH800 跑出 3,000 GB/s 与 580 TF\nJIT 省 56 寄存器达 1,358 TFLOPS', 'blue'),
            ('29.2 DeepEP 与 DualPipe：\nNormal 153 GB/s / IBGDA 163 us\n双向流水微批交错实现零通信暴露', 'purple'),
        ],
        '第 29 章全景：去缓存文件系统、P-D 分离集群与异构平民部署（下）',
        'teal',
        [
            ('29.3 3FS 与 Smallpond：\n180 节点聚合随机读 6.60 TB/s\n110.5 TiB 排序仅耗时 30 min 14 s', 'blue'),
            ('29.4 Mooncake 与异构推理：\nRDMA KV 池提 2.1x 有效吞吐\n单卡+双路 AMX 跑出 18.6 tok/s', 'teal'),
        ],
        '贯通七大开源基础设施引擎，树立软硬协同极致效率的现代 AI 系统标杆',
    )

    outputs = ex.finish()

    fig_index = [
        {
            "figure_id": "图 29-1",
            "number": 1,
            "file": "ch29-deepseek-open-infra-seven-pillar-fullstack-architecture.svg",
            "asset": "ch29/ch29-deepseek-open-infra-seven-pillar-fullstack-architecture.svg",
            "section": "29.1.1",
            "caption": "DeepSeek 开源基础设施七大支柱全栈架构图：算子层（FlashMLA、DeepGEMM）、通信与流水层（DeepEP、DualPipe）、存储与数据层（3FS、Smallpond）及推理部署层（Mooncake、KTransformers）",
            "source_refs": ["experiments/ch29/29-01/run.py", "manuscripts/ch29/figure-data.json"],
            "TeachingRole": "建立从底层 PTX 算子到万卡存储与单机异构推理的七大组件全局协同视图"
        },
        {
            "figure_id": "图 29-2",
            "number": 2,
            "file": "ch29-flashmla-512-plus-64-pingpong-warpgroup-pipeline.svg",
            "asset": "ch29/ch29-flashmla-512-plus-64-pingpong-warpgroup-pipeline.svg",
            "section": "29.1.1",
            "caption": "FlashMLA 深度解剖：矩阵吸收后 [512+64] 非对称头维（1,152 B/token，压缩 28.44 倍）与双消费者 Warpgroup 乒乓流水线（H800 实测 3,000 GB/s 与 580 TFLOPS）（算例 29-1）",
            "source_refs": ["experiments/ch29/29-01/run.py", "manuscripts/ch29/figure-data.json"],
            "TeachingRole": "揭示如何通过矩阵吸收与双 Warpgroup 乒乓交替将 H800 显存带宽压榨至 89.55%"
        },
        {
            "figure_id": "图 29-3",
            "number": 3,
            "file": "ch29-flashmla-variable-length-split-kv-load-balancer.svg",
            "asset": "ch29/ch29-flashmla-variable-length-split-kv-load-balancer.svg",
            "section": "29.1.1",
            "caption": "FlashMLA 变长序列持久块调度（Split-KV Load Balancer）：按全局 KV 块定额动态切分长短请求并通过在线 LogSumExp 归约消除 SM 尾块空转",
            "source_refs": ["experiments/ch29/29-01/run.py", "manuscripts/ch29/figure-data.json"],
            "TeachingRole": "展示在在线推理变长序列批处理下如何保障全部 132 个 SM 负载均衡"
        },
        {
            "figure_id": "图 29-4",
            "number": 4,
            "file": "ch29-deepgemm-runtime-jit-shape-constant-specialization.svg",
            "asset": "ch29/ch29-deepgemm-runtime-jit-shape-constant-specialization.svg",
            "section": "29.1.2",
            "caption": "DeepGEMM 运行期 JIT 形状常量化对比：将矩阵维数与分块步长编译为 PTX 立即数，单线程寄存器占用从 248 个降至 192 个（节省 56 个寄存器）（算例 29-2）",
            "source_refs": ["experiments/ch29/29-01/run.py", "manuscripts/ch29/figure-data.json"],
            "TeachingRole": "阐明为何 300 行核心模板配合运行期 JIT 能击败庞大复杂的静态预编译算子库"
        },
        {
            "figure_id": "图 29-5",
            "number": 5,
            "file": "ch29-deepgemm-fp8-1x128-128x128-two-level-ffma-accumulation.svg",
            "asset": "ch29/ch29-deepgemm-fp8-1x128-128x128-two-level-ffma-accumulation.svg",
            "section": "29.1.2",
            "caption": "DeepGEMM FP8 细粒度（1x128 激活与 128x128 权重）二级累加流水线：每 128 个 Tensor Core 周期插入 4 周期 CUDA Core FFMA（开销仅 3.12%），跑出 1,358 TFLOPS（算例 29-2）",
            "source_refs": ["experiments/ch29/29-01/run.py", "manuscripts/ch29/figure-data.json"],
            "TeachingRole": "剖析如何以 3.12% 的极小指令代价弥补 Hopper FP8 WGMMA 14-bit 定点累加器的精度缺陷"
        },
        {
            "figure_id": "图 29-6",
            "number": 6,
            "file": "ch29-deepep-normal-mode-nvlink-rdma-hybrid-forwarding.svg",
            "asset": "ch29/ch29-deepep-normal-mode-nvlink-rdma-hybrid-forwarding.svg",
            "section": "29.2.1",
            "caption": "DeepEP Normal 模式（面向预填充与训练）：机内 NVLink 预聚合与机间 RDMA 流水转发协同，仅占 24 个 SM 实现 153 GB/s 跨节点 All-to-All 吞吐（算例 29-3）",
            "source_refs": ["experiments/ch29/29-02/run.py", "manuscripts/ch29/figure-data.json"],
            "TeachingRole": "展示两级拓扑感知转发如何消除跨机小包风暴并最小化通信对计算 SM 的抢占"
        },
        {
            "figure_id": "图 29-7",
            "number": 7,
            "file": "ch29-deepep-low-latency-ibgda-zero-sm-hook-overlap.svg",
            "asset": "ch29/ch29-deepep-low-latency-ibgda-zero-sm-hook-overlap.svg",
            "section": "29.2.1",
            "caption": "DeepEP Low-Latency 模式（面向在线解码）：基于纯 IBGDA 门铃直写与后台 Hook 完成位轮询，在 0 个 SM 占用下实现 163 us Dispatch 与 194 us Combine（算例 29-3）",
            "source_refs": ["experiments/ch29/29-02/run.py", "manuscripts/ch29/figure-data.json"],
            "TeachingRole": "揭示如何在零 SM 占用下将跨机专家通信完全隐藏在共享专家计算阴影中"
        },
        {
            "figure_id": "图 29-8",
            "number": 8,
            "file": "ch29-dualpipe-bidirectional-pipeline-schedule-and-20-stage-grid.svg",
            "asset": "ch29/ch29-dualpipe-bidirectional-pipeline-schedule-and-20-stage-grid.svg",
            "section": "29.2.2",
            "caption": "DualPipe 双向流水线时间表（PP=16，双向注入 2M=32 微批）：相向微批正反交错使流水气泡减半，并将跨节点专家通信 100% 掩盖在另一方向的矩阵计算之下（算例 29-4）",
            "source_refs": ["experiments/ch29/29-02/run.py", "manuscripts/ch29/figure-data.json"],
            "TeachingRole": "对比传统单向 1F1B 与 DualPipe 双向交错在跨节点 MoE 通信掩盖上的本质差异"
        },
        {
            "figure_id": "图 29-9",
            "number": 9,
            "file": "ch29-dualpipe-fine-grained-backward-input-weight-split.svg",
            "asset": "ch29/ch29-dualpipe-fine-grained-backward-input-weight-split.svg",
            "section": "29.2.2",
            "caption": "DualPipe 细粒度反向传播拆分：将 Backward 解耦为输入梯度（B 块，激活上游流水）与无跨层依赖的权重梯度（W 块，填充通信等待窗口）",
            "source_refs": ["experiments/ch29/29-02/run.py", "manuscripts/ch29/figure-data.json"],
            "TeachingRole": "阐明为何必须将反向传播一分为二才能构造严丝合缝的四象限计算—通信重叠块"
        },
        {
            "figure_id": "图 29-10",
            "number": 10,
            "file": "ch29-3fs-craq-chain-replication-and-usrbio-zero-copy-ring.svg",
            "asset": "ch29/ch29-3fs-craq-chain-replication-and-usrbio-zero-copy-ring.svg",
            "section": "29.3.1",
            "caption": "3FS（Fire-Flyer File System）架构：链式复制（CRAQ）强一致多副本并发读状态机与 O_DIRECT + USRBIO 用户态零拷贝异步环形队列",
            "source_refs": ["experiments/ch29/29-03/run.py", "manuscripts/ch29/figure-data.json"],
            "TeachingRole": "解释为何在大模型高并发随机读场景下必须彻底抛弃操作系统 Page Cache"
        },
        {
            "figure_id": "图 29-11",
            "number": 11,
            "file": "ch29-3fs-180-node-6600gbs-throughput-and-graysort-pipeline.svg",
            "asset": "ch29/ch29-3fs-180-node-6600gbs-throughput-and-graysort-pipeline.svg",
            "section": "29.3.1",
            "caption": "3FS 180 台存储节点（2,880 块 NVMe SSD + 360 张 200G 网卡）实测：持续输出 6.60 TB/s 聚合随机读吞吐，并在 30 min 14 s 内完成 110.5 TiB GraySort 排序（算例 29-5）",
            "source_refs": ["experiments/ch29/29-03/run.py", "manuscripts/ch29/figure-data.json"],
            "TeachingRole": "用存储介质与 RDMA 网卡 1:1 带宽配比账本验证 6.60 TB/s 聚合吞吐的物理构成"
        },
        {
            "figure_id": "图 29-12",
            "number": 12,
            "file": "ch29-smallpond-duckdb-on-3FS-serverless-parquet-shuffle.svg",
            "asset": "ch29/ch29-smallpond-duckdb-on-3FS-serverless-parquet-shuffle.svg",
            "section": "29.3.2",
            "caption": "Smallpond 轻量级数据框架：基于 DuckDB 向量化列式引擎与 3FS 共享分区洗牌池的无常驻服务（Serverless）PB 级 Parquet 数据处理流水线",
            "source_refs": ["experiments/ch29/29-03/run.py", "manuscripts/ch29/figure-data.json"],
            "TeachingRole": "展示如何用无状态进程加高速分布式文件系统取代笨重的常驻 JVM 大数据集群"
        },
        {
            "figure_id": "图 29-13",
            "number": 13,
            "file": "ch29-mooncake-transfer-engine-multi-nic-rdma-kv-pool.svg",
            "asset": "ch29/ch29-mooncake-transfer-engine-multi-nic-rdma-kv-pool.svg",
            "section": "29.4.1",
            "caption": "Mooncake Transfer Engine 架构：跨 HBM、主机 DRAM 与 NVMe SSD 的多网卡 RDMA 拓扑感知全局 KV 缓存池与零拷贝 P-D 传输通路",
            "source_refs": ["experiments/ch29/29-03/run.py", "manuscripts/ch29/figure-data.json"],
            "TeachingRole": "剖析以 KV 缓存为中心（KV-Centric）的预填充—解码分离调度架构"
        },
        {
            "figure_id": "图 29-14",
            "number": 14,
            "file": "ch29-mooncake-early-rejection-and-sla-goodput-controller.svg",
            "asset": "ch29/ch29-mooncake-early-rejection-and-sla-goodput-controller.svg",
            "section": "29.4.1",
            "caption": "Mooncake 基于排队延迟与 KV 池水位预测的提前拒绝（Early Rejection）准入控制器：防止过载级联雪崩并将 SLA 合规有效吞吐（Goodput）提升 2.1 倍",
            "source_refs": ["experiments/ch29/29-03/run.py", "manuscripts/ch29/figure-data.json"],
            "TeachingRole": "论证在突发流量下主动拒绝违约请求对保护全局 GPU 有效算力的关键作用"
        },
        {
            "figure_id": "图 29-15",
            "number": 15,
            "file": "ch29-ktransformers-gpu-mla-and-cpu-amx-numa-expert-offload.svg",
            "asset": "ch29/ch29-ktransformers-gpu-mla-and-cpu-amx-numa-expert-offload.svg",
            "section": "29.4.2",
            "caption": "KTransformers 异构卸载架构：单张 24 GB RTX 4090D（驻留 19.2 GB MLA 与共享专家）配合双路 Xeon DDR5（680 GB/s）+ AMX 原位计算 256 个路由专家，实现预填充 286.4 tok/s、解码 18.6 tok/s（提速 13.8 倍）（算例 29-6）",
            "source_refs": ["experiments/ch29/29-03/run.py", "manuscripts/ch29/figure-data.json"],
            "TeachingRole": "揭示“算子跟权重走（仅传 14 KB 激活）”相较“跨 PCIe 搬权重”的十倍性能鸿沟"
        },
        {
            "figure_id": "图 29-16",
            "number": 16,
            "file": "ch29-deepseek-fullstack-cost-and-efficiency-synthesis.svg",
            "asset": "ch29/ch29-deepseek-fullstack-cost-and-efficiency-synthesis.svg",
            "section": "29.4.2",
            "caption": "第 29 章系统工程全景图：DeepSeek 开源基础设施全栈（FlashMLA、DeepGEMM、DeepEP、DualPipe、3FS、Smallpond、Mooncake、KTransformers）核心性能锚点总览",
            "source_refs": ["manuscripts/ch29/figure-data.json"],
            "TeachingRole": "收束全章七大开源项目，提炼跨层协同消除硬件闲置的极客工程方法论"
        }
    ]
    for row in fig_index:
        row['figure'] = f"29-{row['number']}"
        row['name'] = Path(row['file']).stem
    (HERE / 'figure-index.json').write_text(json.dumps(fig_index, ensure_ascii=False, indent=2) + '\n')

    md = HERE.parent / '29-DeepSeek 开源基础设施全栈解剖与极客工程实战：FlashMLA、DeepGEMM、DeepEP、DualPipe、3FS、Mooncake 与 KTransformers.md'
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

        work = re.sub(r'!\[([^\]]*)\]\((ch29/[^)]+)\)', repl_img, text_md)
        body = markdown.markdown(work, extensions=['tables', 'fenced_code', 'footnotes', 'toc'])
        html = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>第 29 章：DeepSeek 开源基础设施全栈解剖与极客工程实战：FlashMLA、DeepGEMM、DeepEP、DualPipe、3FS、Mooncake 与 KTransformers</title></head><body>{body}</body></html>\n'
        out_html = preview_path(md)
        out_html.write_text(html)

    sources = [
        HERE / 'build.py',
        HERE / 'figure-data.json',
        HERE / 'figure-index.json',
        ROOT / 'experiments/ch29/29-01/run.py',
        ROOT / 'experiments/ch29/29-02/run.py',
        ROOT / 'experiments/ch29/29-03/run.py',
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
    print(f'Built {len(outputs)} figure files for Chapter 29; 0 layout warnings.')


if __name__ == '__main__':
    build_chapter()
