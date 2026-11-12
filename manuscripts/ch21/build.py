#!/usr/bin/env python3
"""Build all 22 publication-grade figures (420pt) and validation manifests for Chapter 21."""
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
        'ch21-trillion-param-dlrm-three-tier-hbm-cxl-ssd-embedding',
        '10.0 TB 万亿稀疏嵌入表三级分层存储架构（HBM-CXL-NVMe）',
        'blue',
        [
            ('① HBM 热表 320 GB\n拦截 82.0% 查表\n0.35 us (24 TB/s)', 'blue'),
            ('② CXL 温表 2.0 TB\n拦截 15.5% 查表\n1.80 us (1.2 TB/s)', 'purple'),
            ('③ NVMe 冷表 10 TB\n穿透 2.5% 长尾\nGDS 延迟 65.0 us', 'teal'),
        ],
        '纯 SSD 查表（65.0 us）vs. 三级缓存（加权 2.19 us，提速 29.7 倍）',
        'teal',
        [
            ('无分层纯 NVMe 直读瓶颈：\n每次查表均受限于 65.0 us\n矩阵引擎空转率超过 95%', 'orange'),
            ('LFU 热表 + 无锁哈希缓存：\n97.5% 请求在微秒内完成命中\n加权延迟 2.19 us（29.7 倍）', 'teal'),
        ],
        '利用 3.2% HBM 容量捕获 82.0% 幂律流量，突破 10 TB 稀疏表容量墙',
    )

    draw_dual_band(
        ex,
        'ch21-zipf-power-law-hot-embedding-cache-hit-curve',
        '搜广推特征幂律分布（alpha=1.12）与时间漂移（eta=0.038）',
        'purple',
        [
            ('① 静态 Zipf 幂律\n前 3.2% 热门 ID\n理论聚焦 85.8%', 'blue'),
            ('② 15 min 窗口漂移\n突发热点轮动 3.8%\n实测 HBM 达 82.0%', 'purple'),
            ('③ 异步预取与晋升\n后台流式替换温冷行\n消除同步换页停顿', 'teal'),
        ],
        '静态固定缓存 vs. 滑动窗口衰减 LFU 动态晋升机制对比',
        'blue',
        [
            ('静态词频表固定驻留退化：\n短视频实时热点在数分钟漂移\n数小时后命中率跌破 60%', 'orange'),
            ('指数衰减计数动态换入换出：\n每步后台仅交换 0.15% 边界行\n长期稳态维持 82.0% 命中率', 'teal'),
        ],
        '兼顾长周期幂律聚集与短周期突发漂移，守住微秒级查表有效带宽',
    )

    draw_dual_band(
        ex,
        'ch21-torchrec-all-to-all-sparse-lookup-and-bucket-sharding',
        '64 卡 TorchRec 稀疏嵌入表热度分桶切分与 All-to-All 通信',
        'teal',
        [
            ('① 热度感知分桶切分\n倾斜 2.85->1.06\n均衡各卡网卡流量', 'blue'),
            ('② 稀疏 All-to-All\n单卡发 165.15 MB\n通信耗时 1.53 ms', 'purple'),
            ('③ 底端 MLP 双流掩盖\n计算耗时 1.85 ms\n暴露通信 0.00 ms', 'teal'),
        ],
        '随机哈希分片（阻塞 2.26 ms）vs. 热度均衡双流掩盖（0.00 ms）',
        'blue',
        [
            ('朴素随机取模分片（倾斜 2.85）：\n超热门特征集中耗尽单卡带宽\nAll-to-All 达 4.11 ms 暴露', 'orange'),
            ('贪心装箱 + 热门表局部复制：\n64 卡通信负载分布严密对齐\n1.53 ms 通信藏入 1.85 ms 计算', 'teal'),
        ],
        '消除跨卡热度木桶效应，让稀疏全互换通信完全隐藏于稠密矩阵计算之下',
    )

    draw_dual_band(
        ex,
        'ch21-cascade-dlrm-vs-generative-recommender-hstu-paradigm',
        '传统多级级联漏斗（DLRM）vs. 生成式推荐（Meta HSTU / OneRec）',
        'blue',
        [
            ('① 传统级联漏斗\n召回->粗排->精排\n特征占 90% 耗时', 'orange'),
            ('② 算力碎片化困局\n上百个异构小算子\nMFU 长期低于 12%', 'purple'),
            ('③ 生成式序列转导\n统一自回归 Token\nMFU 跃升至 48%+', 'teal'),
        ],
        '推荐系统从“带宽受限特征工程”向“算力扩展定律”的范式跃迁',
        'teal',
        [
            ('级联 DLRM 的性能天花板：\n增加参数仅增大稀疏表体积\n难以带来长序列推理增益', 'orange'),
            ('HSTU 统一生成式序列架构：\n将物品与动作交织为万级序列\n遵循 Scaling Law 持续提效', 'teal'),
        ],
        '把碎片化搜广推漏斗收敛为高算术强度序列转导器，充分释放 Tensor Core',
    )

    draw_dual_band(
        ex,
        'ch21-hstu-pointwise-aggregated-attention-and-stochastic-length',
        'Meta HSTU 非归一化点积聚合注意力与随机序列长度压缩',
        'purple',
        [
            ('① 摒弃 Softmax 归一\nSiLU 点积聚合\n保留活跃度幅值', 'blue'),
            ('② 随机序列长度压缩\n8,192->5,410 词元\n近期全留+远期采样', 'purple'),
            ('③ 二次方算力减半\n注意力省 56.2%\n精度损 < 0.04%', 'teal'),
        ],
        '为什么推荐序列必须用非归一化点积注意力替代标准 Softmax？',
        'blue',
        [
            ('标准 Softmax 概率归一缺陷：\n强行将总权重归一为 1.0\n抹平重度与冷启动用户强度差', 'orange'),
            ('SiLU 点积 + 融合内核优势：\n省去行级最大值与指数和同步\n单核吞吐较标准提升 1.45 倍', 'teal'),
        ],
        '非归一化点积兼顾推荐偏好幅值表达与片上核融合效率，SL 削减过半算力',
    )

    draw_dual_band(
        ex,
        'ch21-m-falcon-candidate-microbatch-prefix-amortization',
        'M-FALCON 千级候选物品微批前缀摊销架构（算例 21-4）',
        'teal',
        [
            ('① 历史前缀编码\nL=4,096 仅算 1 次\n耗 1.376 TFLOPs', 'blue'),
            ('② 千级候选掩码\nM=1,024 共享 KV\n互不可见并行打分', 'purple'),
            ('③ 算力降 44.80 倍\n共 31.46 TFLOPs\n32卡达 485 QPS', 'teal'),
        ],
        '逐物品打分（1,409.3 TFLOPs）vs. M-FALCON 摊销（31.46 TFLOPs）',
        'blue',
        [
            ('朴素逐候选拼接（M=1,024）：\n重复编码 1,024 次相同历史\n算力达 1,409.3 TFLOPs 难上线', 'orange'),
            ('M-FALCON 前缀固化 + 微批：\n候选阶段仅需 30.084 TFLOPs\n每秒打分 49.7 万物品 (11.8 ms)', 'teal'),
        ],
        '一次历史编码摊销千个候选打分，将生成式推荐单请求算力缩减 44.80 倍',
    )

    draw_dual_band(
        ex,
        'ch21-hstu-online-event-streaming-and-delta-kv-update',
        '在线实时行为流（Kafka/Flink）与用户历史 KV 缓存增量追加',
        'blue',
        [
            ('① 毫秒级点击流\n事件产生 < 50 ms\n实时触发特征编码', 'blue'),
            ('② 仅算 Delta KV\n追加 4~8 个新事件\n免除 4,096 重算', 'purple'),
            ('③ 分层 KV 原位更新\n跨请求复用前缀块\n重算率 < 3.0%', 'teal'),
        ],
        '全量历史重编码 vs. 增量 Delta KV 追加的在线推理算力对比',
        'teal',
        [
            ('无状态全量重算模式：\n每次刷屏均重算 4,096 词元\n75% 预填充算力被重复消耗', 'orange'),
            ('有状态会话 KV 增量追加：\n仅对最新产生的行为计算 K/V\n预填充开销降低 18.5 倍', 'teal'),
        ],
        '利用因果注意力的单向依赖性，以增量 KV 追加实现秒级兴趣实时捕获',
    )

    draw_dual_band(
        ex,
        'ch21-cpu-hnsw-vs-gpu-cuvs-cagra-graph-build-pipeline',
        '十亿级 768 维向量（1.536 TB）建图：CPU HNSW vs. GPU CAGRA',
        'purple',
        [
            ('① 10 亿向量分片\n768 维 1.536 TB\nIVF-PQ 粗筛候选', 'blue'),
            ('② GPU 并行建图\n无跳表锁竞争\n两阶段剪枝定度', 'purple'),
            ('③ 40.8 min 收敛\n提速 16.91 倍\n小时级全量刷新', 'teal'),
        ],
        '双路 128 核 CPU HNSW（11.50 h）vs. 8 卡 H100 CAGRA（0.68 h）',
        'blue',
        [
            ('CPU HNSW 增量插入瓶颈：\n逐点贪心搜索触发亿次随机访存\n多层图细粒度锁使扩展受阻', 'orange'),
            ('GPU cuVS CAGRA 原生构建：\n批量距离矩阵乘 + 并行拓扑优化\n40.8 分钟完成十亿级高召回图', 'teal'),
        ],
        '抛弃面向 CPU 设计的分层跳表，以扁平固定度图释放 GPU 并行吞吐',
    )

    draw_dual_band(
        ex,
        'ch21-cagra-warp-splitting-distance-and-two-stage-pruning',
        'cuVS CAGRA 核心算法：Warp 切分距离计算与两阶段图剪枝',
        'teal',
        [
            ('① Warp-Split 距离\n8 线程协算 1 向量\n合并访存消除发散', 'blue'),
            ('② 两阶段绕行剪枝\n剔除二跳可达冗余\n度数 128 减至 64', 'purple'),
            ('③ 反向边注入连通\n修补有向图不可达点\nRecall 达 98.2%', 'teal'),
        ],
        '为什么两阶段剪枝 + 反向边注入能兼顾低度数与高召回率？',
        'purple',
        [
            ('单纯截断最近邻的连通退化：\n簇边缘节点极易沦为无入边孤岛\n贪心波束搜索陷入局部极小', 'orange'),
            ('三角绕行剪枝 + 反向边修复：\n用远程桥接边替代同向冗余近邻\n搜索跳数省 35% 且召回不降', 'teal'),
        ],
        '细粒度 Warp 距离核与两阶段图拓扑整形协同，铸就 GPU 原生图检索引擎',
    )

    draw_dual_band(
        ex,
        'ch21-ivf-pq-shared-memory-lut-asymmetric-distance-kernel',
        '乘积量化（IVF-PQ）非对称距离查找表（LUT）片上共享内存融合',
        'blue',
        [
            ('① 768 维切 64 子空间\n每空间 256 中心\n1,536 B 缩为 64 B', 'blue'),
            ('② 距离表驻 SRAM\n64x256 占 32 KB\n32-Bank 交错无冲突', 'purple'),
            ('③ 寄存器查表累加\n省去高维浮点乘加\n扫描超 1.8 TB/s', 'teal'),
        ],
        'HBM 向量解压计算 vs. 片上 Shared Memory LUT 查表对比',
        'teal',
        [
            ('显式反量化回 FP16 再算内积：\n产生 24 倍中间显存带宽膨胀\n受制于 HBM 物理带宽上限', 'orange'),
            ('ADC 距离表常驻片上 SRAM：\nHBM 仅流式读取 64 B 紧凑码\n距离估算化为纳秒级片上查表', 'teal'),
        ],
        '将高维内积压缩为 32 KB 片上查找表累加，实现 24 倍带宽放大',
    )

    draw_dual_band(
        ex,
        'ch21-diskann-4kb-sector-aligned-vector-and-neighbor-layout',
        'DiskANN 4,096 B 物理扇区对齐布局（算例 21-6：每扇区 2 节点）',
        'purple',
        [
            ('① 内存常驻 PQ 码\n10 亿 x 64 B=64 GB\n零磁盘 I/O 导航', 'blue'),
            ('② 同扇区紧凑打包\n1536+256=1792 B\n单 4 KB 装 2 节点', 'purple'),
            ('③ 512 B 对齐填充\n消除跨页双倍读取\n单跳对应 1 次 I/O', 'teal'),
        ],
        '向量与邻接表分离存储（2x IOPS）vs. 同扇区物理对齐（1x IOPS）',
        'blue',
        [
            ('分离存储或跨扇区未对齐：\n读向量与读邻居触发 2~3 次读\nNVMe IOPS 被无效读放大消耗', 'orange'),
            ('4,096 B 扇区耦合打包布局：\n1 次 4 KB 读取回向量与出边\n单次查询仅 W*S = 24 次随机读', 'teal'),
        ],
        '顺应 NVMe 4 KB 最小读写粒度，以同页耦合布局消除磁盘随机读放大',
    )

    draw_dual_band(
        ex,
        'ch21-diskann-io-uring-gds-beam-search-pipeline',
        'DiskANN 基于 io_uring / GDS 的异步波束搜索（W=4, S=6）',
        'teal',
        [
            ('① 内存 PQ 贪心选点\n每轮选 W=4 候选\n耗时仅 1.5 us', 'blue'),
            ('② io_uring 并发读\n4 个 4 KB 扇区齐发\n隐藏 85 us 延迟', 'purple'),
            ('③ 6 跳收敛精确重排\n24 次读达 21 万 QPS\nRecall 达 95.4%', 'teal'),
        ],
        '同步阻塞 pread（延迟叠加）vs. 异步波束并发 I/O（延迟掩盖）',
        'blue',
        [
            ('单路同步阻塞磁盘读取：\n24 次串行读耗时 2.04 ms\n单队列难以利用 PCIe 5.0 并发', 'orange'),
            ('波束宽度 W=4 异步并发提交：\n每跳 4 次 I/O 合并为 1 周期\n4 盘达 21 万 QPS (P99 2.85 ms)', 'teal'),
        ],
        '用内存 PQ 距离导航规避盲目读盘，用异步波束并发释放 6.0M NVMe IOPS',
    )

    draw_dual_band(
        ex,
        'ch21-freshdiskann-streaming-insert-delete-consolidation',
        'FreshDiskANN 实时流式写入、墓碑删除与后台无锁邻居重连',
        'blue',
        [
            ('① 写入内存增量图\nFreshVamana 接收\n秒级可见 5 万写/s', 'blue'),
            ('② 墓碑位图标记\n查询动态过滤失效点\n避免阻塞 SSD 图', 'purple'),
            ('③ 后台流式合并\n修补被删节点邻居\nI/O 占用 < 15%', 'teal'),
        ],
        '全量重建停机反模式 vs. LSM 式流式增量合并（FreshDiskANN）',
        'teal',
        [
            ('直接删除图节点的拓扑断裂：\n枢纽节点被删导致近邻不可达\n数轮更新后 Recall 下跌 12%', 'orange'),
            ('Vamana 鲁棒重连合并（Stitch）：\n将被删节点出边嫁接给入边前驱\n7x24 小时持续写入召回零退化', 'teal'),
        ],
        '借鉴 LSM-Tree 增量合并与图拓扑缝合，解决十亿向量索引实时更新难题',
    )

    draw_dual_band(
        ex,
        'ch21-graphrag-hierarchical-community-and-vector-hybrid-retrieval',
        'GraphRAG 多跳实体关系社区摘要与 GPU 向量混合检索流水线',
        'purple',
        [
            ('① 稠密向量锚点\nTop-32 种子文档块\nCAGRA 耗 1.2 ms', 'blue'),
            ('② 二跳图谱遍历\nCSR 邻接子图扩展\n提 Leiden 社区摘要', 'purple'),
            ('③ RRF 倒数秩融合\n去重合并注入上下文\n总检索仅 8.4 ms', 'teal'),
        ],
        '纯扁平向量分块检索 vs. GraphRAG 图谱—向量混合检索对比',
        'blue',
        [
            ('纯 Top-K 扁平向量检索盲区：\n难以回答跨文档全局归纳推理\n碎片化段落丢失跨实体因果链', 'orange'),
            ('局部向量锚定 + 社区摘要协同：\n兼顾细粒度事实与宏观脉络完整\n复杂多跳问答准确率升 34.0%', 'teal'),
        ],
        '向量检索提供亚毫秒级局部语义锚点，图谱遍历补齐跨文档多跳拓扑关系',
    )

    draw_dual_band(
        ex,
        'ch21-search-ads-rec-gpu-fleet-tco-and-sla-roofline',
        '搜广推与向量检索 GPU 机队全链路 SLA（50 ms）预算拆解',
        'teal',
        [
            ('① 多路向量召回\nCAGRA + DiskANN\n耗时 3.5 ms (7%)', 'blue'),
            ('② 三级稀疏查表\nHBM+CXL+GDS\n耗时 2.5 ms (5%)', 'purple'),
            ('③ HSTU 微批精排\n千级候选并行打分\n11.8 ms (共 17.8 ms)', 'teal'),
        ],
        '搜广推机队从 CPU+级联小模型向“全栈 GPU 生成式底座”演进',
        'purple',
        [
            ('传统万台 CPU 检索 + 级联卡池：\n跨机房 RPC 序列化耗去 25 ms\n机柜功耗高且利用率不足 15%', 'orange'),
            ('同机柜 GPU 检索 + HSTU 部署：\n端到端 P99 仅 17.8 ms (< 50 ms)\n单瓦特 GMV 产出提升 3.4 倍', 'teal'),
        ],
        '协同向量召回、三级稀疏查表与生成式精排，提升搜广推机队能效产出',
    )

    draw_dual_band(
        ex,
        'ch21-generative-recommendation-and-vector-engine-synthesis',
        '在线搜广推前四层协同模块（上）：稀疏状态缓存与生成式精排',
        'blue',
        [
            ('21.1 万亿稀疏表底座：\nHBM-CXL-SSD 缓存 (2.19 us)\nTorchRec 分桶与双流掩盖', 'blue'),
            ('21.2 生成式推荐 HSTU：\nSiLU 点积 + 随机长度压缩\nM-FALCON 微批减负 44.80 倍', 'purple'),
        ],
        '在线搜广推前四层协同模块（下）：GPU 图索引与 SSD 混合检索',
        'teal',
        [
            ('21.3 GPU 向量图引擎：\ncuVS CAGRA 提速 16.91 倍\nIVF-PQ 32 KB 片上 LUT 查表', 'blue'),
            ('21.4 百亿 SSD 与 GraphRAG：\nDiskANN 4 KB 扇区双节点对齐\nFreshDiskANN 与社区摘要融合', 'teal'),
        ],
        '以分级局部性管理 10 TB 稀疏状态，以前缀共享摊销释放生成式精排算力',
    )

    draw_dual_band(
        ex,
        'ch21-colpali-visual-patch-late-interaction-maxsim-architecture',
        'ColPali 视觉文档晚交互架构：1,030 视觉词元与 MaxSim 对齐',
        'purple',
        [
            ('① 页面整图直编\n免除脆弱 OCR 切块\n保留表格双栏结构', 'blue'),
            ('② 1,030 视觉词元\n投影至 d=128 维\nBF16 占 263.68 KB', 'purple'),
            ('③ 迟交互 MaxSim\n逐查询词元局部寻优\nnDCG@5 升至 0.816', 'teal'),
        ],
        '传统 OCR 单向量池化（nDCG 0.542）vs. ColPali 多向量晚交互',
        'blue',
        [
            ('PDF -> OCR -> 单向量瓶颈：\n跨栏报表与连线框图空间失真\n需取 32 页拖慢下游 LLM 预填充', 'orange'),
            ('ColPali 词元级视觉细粒度对齐：\n精准命中局部单元格与公式区域\n仅送 8 页节省 432 MiB KV 显存', 'teal'),
        ],
        '绕开 OCR 级联失真，以词元级 MaxSim 精准对齐降低大模型预填充负担',
    )

    draw_dual_band(
        ex,
        'ch21-plaid-centroid-pruning-and-2bit-residual-compression',
        'PLAID 聚类中心残差量化与融合 MaxSim 三级漏斗（算例 21-7）',
        'teal',
        [
            ('① 质心倒排粗筛\nQ*C^T 筛 4,096 页\n免读残差省 92% I/O', 'blue'),
            ('② 2-bit / 1-bit 残差\n34 B (7.53x) / 17.5 B\n14.6x 压至 901.3 GB', 'purple'),
            ('③ 寄存器融合解压\n片上还原算 MaxSim\n单查询 P99 14.2 ms', 'teal'),
        ],
        '朴素 ColPali BF16（50M 页 13.18 TB）vs. PLAID 残差量化压缩',
        'purple',
        [
            ('全精度 BF16 裸存（256 B/vec）：\n5,000 万页高达 13.18 TB\n十亿页 263.7 TB 远超显存上限', 'orange'),
            ('PLAID 1-bit 二值残差（17.5 B）：\n5,000 万页仅 901.3 GB (14.6x)\n十亿页 18.0 TB 维持 0.811 精度', 'teal'),
        ],
        '剥离聚类质心与低比特残差并在寄存器内融合解包，兼顾高密度与低延迟',
    )

    draw_dual_band(
        ex,
        'ch21-three-way-hybrid-retrieval-and-cross-encoder-rerank-funnel',
        '三路混合检索与紧凑装箱 0.6B 交叉编码器精排漏斗（算例 21-8）',
        'blue',
        [
            ('① 词项+稠密并发\nBM25 6.4+cuVS 9.8ms\nRRF 融合得 K1=1000', 'blue'),
            ('② 紧凑序列装箱\ncu_seqlens 拼 K2=64\n零填充耗时 14.6 ms', 'purple'),
            ('③ 全链路预算达标\n含网关共 27.6 ms\n严守 30 ms SLA', 'teal'),
        ],
        '三维填充重排超时风险 vs. 拉格朗日宽召回窄精排紧凑装箱',
        'teal',
        [
            ('盲目扩大 K2 或三维填充：\n45%+ 算力浪费于无效填充\n精排超 35 ms 破坏总预算', 'orange'),
            ('KKT 最优 K1=1000, K2=64：\n20,480 词元对齐 128 行块\nTop-8 证据召回率达 97.8%', 'teal'),
        ],
        '以低边际成本宽召回配合紧凑装箱交叉编码器精排，兑现 30 ms 混合检索',
    )

    draw_dual_band(
        ex,
        'ch21-streaming-feature-pipeline-flink-to-online-embedding-sync',
        'Kafka/Flink 实时样本拼接与 5 秒微窗脏键增量多播同步',
        'purple',
        [
            ('① 双流窗口拼接\n曝光流 + 行为流\nFlink 乱序窗口归因', 'blue'),
            ('② 脏键位图聚合\n5s 微窗提取非零梯度\n带宽仅 660 MB/s', 'purple'),
            ('③ 单调版本校验\n携带 64 位逻辑步号\n故障重放幂等防回退', 'teal'),
        ],
        'T+1 离线全量换表冷启动折损 vs. 分钟/秒级流式增量同步',
        'blue',
        [
            ('每日凌晨全量静态切换：\n突发热点数小时处于随机态\n完播率与 eCPM 损失 3%~6%', 'orange'),
            ('Monolith 流式增量同步：\n每秒推送 250 万活跃向量\n新内容数分钟获精准表征', 'teal'),
        ],
        '仅多播非零梯度脏键并附带单调版本号，打通秒级实时兴趣反馈主干链路',
    )

    draw_dual_band(
        ex,
        'ch21-rcu-epoch-lockfree-embedding-hot-update-vs-rwlock-tail',
        '每秒 250 万向量更新：分段读写锁 vs. Epoch 双槽翻转（算例 21-10）',
        'teal',
        [
            ('① 影子槽异地写\n后台填妥新版本向量\nRelease 原子发指针', 'blue'),
            ('② 读路径零锁直读\n320 路扇出 Acquire\n无互斥阻塞与自旋', 'purple'),
            ('③ 3-Epoch 宽限回收\n15 ms 窗口安全复用\nP99 稳定在 1.15 ms', 'teal'),
        ],
        '分段原地读写锁（P99 18.4 ms）vs. Epoch 无锁发布（1.15 ms）',
        'purple',
        [
            ('原地分段读写锁扇出放大：\n单段 0.5% 阻塞经 320 路\n碰撞率近 80% 致 18.4 ms', 'orange'),
            ('Epoch 双槽指针原子翻转：\nP99 收敛 16 倍至 1.15 ms\n内存仅增 6.2% 省 709 GB', 'teal'),
        ],
        '以代际延迟回收消除百路扇出锁争用，兼顾亚毫秒尾延迟与低内存开销',
    )

    draw_dual_band(
        ex,
        'ch21-cuckoo-admission-filter-and-ttl-eviction-memory-curve',
        '万亿稀疏表容量治理：布谷鸟准入（tau=5）与过期淘汰（算例 21-9）',
        'blue',
        [
            ('① 日增 40 亿事件\n无过滤达 2.40 TB\n超六成属偶发噪声', 'orange'),
            ('② 布谷鸟频次准入\ntau=5 保留 36.5%\n拦截 63.5% 低频键', 'blue'),
            ('③ 30 天过期淘汰\n净留 31.5% 缩 68.5%\n活跃表仅 756 GB', 'teal'),
        ],
        '无门槛全量建行（2.40 TB）vs. 频次准入 + TTL 治理（756.0 GB）',
        'teal',
        [
            ('单次曝光即建 500 B 行：\n长尾噪声迅速耗尽 HBM\n降级取模哈希致特征污染', 'orange'),
            ('5 B 指纹计数加过期驱逐：\n活跃表容量净缩减 68.5%\n单柜 8 卡无碰撞全量承载', 'teal'),
        ],
        '在入口拦截低频长尾噪声并定期清理僵尸键，将活跃稀疏表收敛至 756 GB',
    )

    outputs = ex.finish()

    fig_index = [
        {
            "figure_id": "图 21-1",
            "number": 1,
            "file": "ch21-trillion-param-dlrm-three-tier-hbm-cxl-ssd-embedding.svg",
            "asset": "ch21/ch21-trillion-param-dlrm-three-tier-hbm-cxl-ssd-embedding.svg",
            "section": "21.1.1",
            "caption": "10.0 TB 万亿参数 DLRM 稀疏嵌入表三级分层存储架构（HBM 320 GB 热表 + CXL 2.0 TB 温表 + NVMe 10.0 TB 冷表），加权查表延迟降至 2.19 us（算例 21-1）",
            "source_refs": ["experiments/ch21/21-01/run.py", "manuscripts/ch21/figure-data.json", "calculations/results/ch21-recsys-vector-colpali.json"],
            "TeachingRole": "展示如何利用 3.2% HBM 热缓存捕获 82.0% 幂律访问，将加权延迟较纯 SSD 提升 29.7 倍"
        },
        {
            "figure_id": "图 21-2",
            "number": 2,
            "file": "ch21-zipf-power-law-hot-embedding-cache-hit-curve.svg",
            "asset": "ch21/ch21-zipf-power-law-hot-embedding-cache-hit-curve.svg",
            "section": "21.1.1",
            "caption": "搜广推稀疏特征 Zipf 幂律分布（alpha=1.12）与 15 分钟滑动窗口时间漂移（eta=0.038）下的三级缓存命中率曲线与无锁 Cuckoo 哈希晋升机制（算例 21-1）",
            "source_refs": ["experiments/ch21/21-01/run.py", "manuscripts/ch21/figure-data.json", "case-studies/memory-criticality-and-tiering.md"],
            "TeachingRole": "量化静态幂律聚集与在线突发热点漂移对多级缓存未命中率的耦合影响"
        },
        {
            "figure_id": "图 21-3",
            "number": 3,
            "file": "ch21-torchrec-all-to-all-sparse-lookup-and-bucket-sharding.svg",
            "asset": "ch21/ch21-torchrec-all-to-all-sparse-lookup-and-bucket-sharding.svg",
            "section": "21.1.2",
            "caption": "64 卡 TorchRec 稀疏嵌入表热度感知分桶切分（倾斜度从 2.85 降至 1.06）与底端 MLP 计算完全掩盖 All-to-All 通信（暴露延迟 0.00 ms）（算例 21-2）",
            "source_refs": ["experiments/ch21/21-01/run.py", "manuscripts/ch21/figure-data.json"],
            "TeachingRole": "阐明如何通过按访问热度分桶消除 All-to-All 木桶效应并实现双流零暴露重叠"
        },
        {
            "figure_id": "图 21-4",
            "number": 4,
            "file": "ch21-cascade-dlrm-vs-generative-recommender-hstu-paradigm.svg",
            "asset": "ch21/ch21-cascade-dlrm-vs-generative-recommender-hstu-paradigm.svg",
            "section": "21.2",
            "caption": "传统多阶段级联推荐漏斗（DLRM 碎片化特征工程，MFU < 12%）向统一生成式推荐序列转导器（Meta HSTU / OneRec，MFU > 48%）的范式跃迁",
            "source_refs": ["manuscripts/ch21/figure-data.json"],
            "TeachingRole": "揭示搜广推如何从访存受限的浅层交叉网络演进为遵循扩展定律的自回归序列生成模型"
        },
        {
            "figure_id": "图 21-5",
            "number": 5,
            "file": "ch21-hstu-pointwise-aggregated-attention-and-stochastic-length.svg",
            "asset": "ch21/ch21-hstu-pointwise-aggregated-attention-and-stochastic-length.svg",
            "section": "21.2.1",
            "caption": "层次化序列转导单元（HSTU）的非归一化 SiLU 点积聚合注意力内核与随机序列长度压缩（Stochastic Length，beta=0.75 削减 56.2% 注意力算力）（算例 21-3）",
            "source_refs": ["experiments/ch21/21-02/run.py", "manuscripts/ch21/figure-data.json", "calculations/results/ch21-recsys-vector-colpali.json"],
            "TeachingRole": "解释为何推荐系统必须摒弃概率归一 Softmax 以保留用户偏好强度并利用幂律稀疏采样压缩长序列"
        },
        {
            "figure_id": "图 21-6",
            "number": 6,
            "file": "ch21-m-falcon-candidate-microbatch-prefix-amortization.svg",
            "asset": "ch21/ch21-m-falcon-candidate-microbatch-prefix-amortization.svg",
            "section": "21.2.2",
            "caption": "M-FALCON 候选集微批前缀摊销算法：单次编码 L=4,096 用户历史前缀并共享 KV 状态并行打分 M=1,024 个候选物品，算力缩减 44.80 倍（1,409.3 -> 31.46 TFLOPs）（算例 21-4）",
            "source_refs": ["experiments/ch21/21-02/run.py", "manuscripts/ch21/figure-data.json", "calculations/results/ch21-recsys-vector-colpali.json"],
            "TeachingRole": "通过前缀 KV 共享与候选对角注意力掩码突破生成式推荐在线打分千级候选物品的算力瓶颈"
        },
        {
            "figure_id": "图 21-7",
            "number": 7,
            "file": "ch21-hstu-online-event-streaming-and-delta-kv-update.svg",
            "asset": "ch21/ch21-hstu-online-event-streaming-and-delta-kv-update.svg",
            "section": "21.2.2",
            "caption": "在线实时行为事件流（Kafka / Flink）驱动的用户历史序列增量 Delta KV 缓存追加与跨请求前缀复用流水线",
            "source_refs": ["manuscripts/ch21/figure-data.json"],
            "TeachingRole": "消除同一用户连续刷屏时的 4,096 历史序列重复预填充开销"
        },
        {
            "figure_id": "图 21-8",
            "number": 8,
            "file": "ch21-cpu-hnsw-vs-gpu-cuvs-cagra-graph-build-pipeline.svg",
            "asset": "ch21/ch21-cpu-hnsw-vs-gpu-cuvs-cagra-graph-build-pipeline.svg",
            "section": "21.3.1",
            "caption": "十亿级 768 维向量（1.536 TB）图索引构建对比：双路 128 核 CPU HNSW（11.50 小时）vs. 8 卡 H100 cuVS CAGRA（40.8 分钟，提速 16.91 倍）（算例 21-5）",
            "source_refs": ["experiments/ch21/21-03/run.py", "manuscripts/ch21/figure-data.json"],
            "TeachingRole": "剖析 CPU HNSW 层级锁竞争与随机访存瓶颈，展示 GPU 原生扁平固定度图的并行建图优势"
        },
        {
            "figure_id": "图 21-9",
            "number": 9,
            "file": "ch21-cagra-warp-splitting-distance-and-two-stage-pruning.svg",
            "asset": "ch21/ch21-cagra-warp-splitting-distance-and-two-stage-pruning.svg",
            "section": "21.3.1",
            "caption": "NVIDIA cuVS CAGRA 的 Warp-Split 协同距离计算内核、两阶段二跳绕行路径剪枝与反向边注入拓扑优化",
            "source_refs": ["experiments/ch21/21-03/run.py", "manuscripts/ch21/figure-data.json"],
            "TeachingRole": "拆解 GPU 图索引如何在精简一半出边度数的同时维持 98.2% 以上的全局强连通召回率"
        },
        {
            "figure_id": "图 21-10",
            "number": 10,
            "file": "ch21-ivf-pq-shared-memory-lut-asymmetric-distance-kernel.svg",
            "asset": "ch21/ch21-ivf-pq-shared-memory-lut-asymmetric-distance-kernel.svg",
            "section": "21.3.2",
            "caption": "乘积量化（IVF-PQ）非对称距离查找表（32 KB ADC-LUT）在 GPU 片上共享内存中的无 Bank 冲突交错布局与寄存器级累加内核",
            "source_refs": ["manuscripts/ch21/figure-data.json"],
            "TeachingRole": "利用 32 KB 片上 SRAM 查找表消除 24 倍显存解压带宽膨胀"
        },
        {
            "figure_id": "图 21-11",
            "number": 11,
            "file": "ch21-diskann-4kb-sector-aligned-vector-and-neighbor-layout.svg",
            "asset": "ch21/ch21-diskann-4kb-sector-aligned-vector-and-neighbor-layout.svg",
            "section": "21.4.1",
            "caption": "DiskANN 物理扇区对齐存储布局：主机 DRAM 常驻 64.0 GB PQ 码 + NVMe 4,096 B 扇区紧凑封装 2 个完整图节点（1,792 B/节点）（算例 21-6）",
            "source_refs": ["experiments/ch21/21-03/run.py", "manuscripts/ch21/figure-data.json"],
            "TeachingRole": "通过将全精度向量与 64 个邻居 ID 同页打包，确保每跳遍历严格只消耗 1 次 4 KB 随机读"
        },
        {
            "figure_id": "图 21-12",
            "number": 12,
            "file": "ch21-diskann-io-uring-gds-beam-search-pipeline.svg",
            "asset": "ch21/ch21-diskann-io-uring-gds-beam-search-pipeline.svg",
            "section": "21.4.1",
            "caption": "DiskANN 基于 io_uring / GDS 的异步波束搜索（W=4, S=6，单查询 24 次随机读）在 4 块 PCIe 5.0 NVMe SSD 上实现 210,000 QPS 与 2.85 ms P99 延迟（算例 21-6）",
            "source_refs": ["experiments/ch21/21-03/run.py", "manuscripts/ch21/figure-data.json"],
            "TeachingRole": "用内存 PQ 粗排导航结合多路异步 I/O 并发隐藏闪存读取延迟"
        },
        {
            "figure_id": "图 21-13",
            "number": 13,
            "file": "ch21-freshdiskann-streaming-insert-delete-consolidation.svg",
            "asset": "ch21/ch21-freshdiskann-streaming-insert-delete-consolidation.svg",
            "section": "21.4.2",
            "caption": "FreshDiskANN 在线流式增量图写入、墓碑删除标记与后台鲁棒重连合并（Robust Stitch Consolidation）双缓冲架构",
            "source_refs": ["manuscripts/ch21/figure-data.json"],
            "TeachingRole": "解决静态图索引无法支持每秒数万次实时增删而导致拓扑断裂的工程问题"
        },
        {
            "figure_id": "图 21-14",
            "number": 14,
            "file": "ch21-graphrag-hierarchical-community-and-vector-hybrid-retrieval.svg",
            "asset": "ch21/ch21-graphrag-hierarchical-community-and-vector-hybrid-retrieval.svg",
            "section": "21.4.2",
            "caption": "GraphRAG 多跳知识图谱社区摘要（Leiden 层次聚类）与 GPU 稠密向量检索的并发执行及倒数秩融合（RRF）流水线",
            "source_refs": ["manuscripts/ch21/figure-data.json"],
            "TeachingRole": "在 8.4 ms 预算内协同完成细粒度向量语义命中与跨文档全局关系图谱召回"
        },
        {
            "figure_id": "图 21-15",
            "number": 15,
            "file": "ch21-search-ads-rec-gpu-fleet-tco-and-sla-roofline.svg",
            "asset": "ch21/ch21-search-ads-rec-gpu-fleet-tco-and-sla-roofline.svg",
            "section": "21.4.2",
            "caption": "互联网搜广推与混合检索全链路 50 ms SLA 延迟分解（向量召回 3.5 ms + 三级查表 2.5 ms + HSTU 精排 11.8 ms = 17.8 ms）与 GPU 机队 TCO 收益",
            "source_refs": ["manuscripts/ch21/figure-data.json"],
            "TeachingRole": "从数据中心机队视角算清全栈 GPU 化替代传统 CPU 级联集群的延迟裕度与投资回报率"
        },
        {
            "figure_id": "图 21-16",
            "number": 16,
            "file": "ch21-generative-recommendation-and-vector-engine-synthesis.svg",
            "asset": "ch21/ch21-generative-recommendation-and-vector-engine-synthesis.svg",
            "section": "21.4.2",
            "caption": "在线搜广推前四层核心计算与单向量检索协同架构：从 10 TB 三级稀疏嵌入表、HSTU + M-FALCON 生成式精排到 cuVS CAGRA / DiskANN 十亿级向量召回底座",
            "source_refs": ["manuscripts/ch21/figure-data.json"],
            "TeachingRole": "归纳单向量召回、三级稀疏缓存与生成式精排模块的协同机制，衔接后续多向量晚交互与流式热更新"
        },
        {
            "figure_id": "图 21-17",
            "number": 17,
            "file": "ch21-colpali-visual-patch-late-interaction-maxsim-architecture.svg",
            "asset": "ch21/ch21-colpali-visual-patch-late-interaction-maxsim-architecture.svg",
            "section": "21.5.1",
            "caption": "ColPali 视觉文档晚交互架构：绕过脆弱 OCR 版面解析，将每页 PDF 映射为 1,030 个 128 维视觉词元并通过 MaxSim 算子精确对齐跨栏表格与图表（算例 21-7）",
            "source_refs": [
                "experiments/ch21/21-04/run.py",
                "manuscripts/ch21/figure-data.json",
                "calculations/results/ch21-recsys-vector-colpali.json",
                "case-studies/retrieval-and-generation.md"
            ],
            "TeachingRole": "阐明页面视觉 Patch 多向量晚交互如何消除 OCR 版面失真并削减下游 LLM 预填充上下文"
        },
        {
            "figure_id": "图 21-18",
            "number": 18,
            "file": "ch21-plaid-centroid-pruning-and-2bit-residual-compression.svg",
            "asset": "ch21/ch21-plaid-centroid-pruning-and-2bit-residual-compression.svg",
            "section": "21.5.1",
            "caption": "PLAID 聚类中心倒排剪枝与 2-bit（34 B/vec，7.53x）/ 1-bit（17.5 B/vec，14.6x）残差压缩及 Tensor Core 融合 MaxSim 三级流水线（算例 21-7）",
            "source_refs": [
                "experiments/ch21/21-04/run.py",
                "manuscripts/ch21/figure-data.json",
                "calculations/results/ch21-recsys-vector-colpali.json",
                "case-studies/generative-recsys-and-colpali-rag.md"
            ],
            "TeachingRole": "定量拆解 13.18 TB 朴素 BF16 索引压缩至 1.75 TB / 901.3 GB 并在 14.2 ms 内融合精排的机理"
        },
        {
            "figure_id": "图 21-19",
            "number": 19,
            "file": "ch21-three-way-hybrid-retrieval-and-cross-encoder-rerank-funnel.svg",
            "asset": "ch21/ch21-three-way-hybrid-retrieval-and-cross-encoder-rerank-funnel.svg",
            "section": "21.5.2",
            "caption": "三路混合检索与重排漏斗：BM25 稀疏倒排（6.4 ms）与 GPU cuVS 稠密图索引（9.8 ms）并行召回 K1=1,000，经 RRF 融合后送入紧凑装箱的 0.6B 交叉编码器重排 K2=64，端到端 P99 耗时 27.6 ms（算例 21-8）",
            "source_refs": [
                "experiments/ch21/21-04/run.py",
                "manuscripts/ch21/figure-data.json",
                "calculations/results/ch21-recsys-vector-colpali.json"
            ],
            "TeachingRole": "展示拉格朗日约束下宽召回（K1=1,000）与紧凑装箱窄精排（K2=64）在 30 ms SLA 内的最优分配"
        },
        {
            "figure_id": "图 21-20",
            "number": 20,
            "file": "ch21-streaming-feature-pipeline-flink-to-online-embedding-sync.svg",
            "asset": "ch21/ch21-streaming-feature-pipeline-flink-to-online-embedding-sync.svg",
            "section": "21.6.1",
            "caption": "实时样本流拼接与秒级增量稀疏参数同步架构：Kafka 曝光/行为双流经 Flink 窗口连接送入在线训练，每 5 秒仅多播非零更新的脏键向量至线上推理池",
            "source_refs": [
                "experiments/ch21/21-05/run.py",
                "manuscripts/ch21/figure-data.json",
                "case-studies/generative-recsys-and-colpali-rag.md"
            ],
            "TeachingRole": "剖析实时曝光—行为双流连接、脏键位图增量抽取与单调版本号幂等保护链路"
        },
        {
            "figure_id": "图 21-21",
            "number": 21,
            "file": "ch21-rcu-epoch-lockfree-embedding-hot-update-vs-rwlock-tail.svg",
            "asset": "ch21/ch21-rcu-epoch-lockfree-embedding-hot-update-vs-rwlock-tail.svg",
            "section": "21.6.1",
            "caption": "每秒 250 万向量热更新下的读写同步机制对比：分段读写锁因扇出碰撞导致查表 P99 升至 18.4 ms vs. 无锁 Epoch 双槽指针翻转维持 P99 = 1.15 ms（算例 21-10）",
            "source_refs": [
                "experiments/ch21/21-05/run.py",
                "manuscripts/ch21/figure-data.json",
                "calculations/results/ch21-recsys-vector-colpali.json",
                "case-studies/generative-recsys-and-colpali-rag.md"
            ],
            "TeachingRole": "揭示百路扇出下分段读写锁的指数碰撞放大效应与 3-Epoch 代际无锁回收 16.0 倍尾延迟收敛"
        },
        {
            "figure_id": "图 21-22",
            "number": 22,
            "file": "ch21-cuckoo-admission-filter-and-ttl-eviction-memory-curve.svg",
            "asset": "ch21/ch21-cuckoo-admission-filter-and-ttl-eviction-memory-curve.svg",
            "section": "21.6.2",
            "caption": "万亿稀疏表容量治理防线：前置紧凑布谷鸟频次准入过滤器（阈值 tau=5）叠加 30 天不活跃过期淘汰，将 2.40 TB 原始稀疏表缩减 68.5% 至 756.0 GB（算例 21-9）",
            "source_refs": [
                "experiments/ch21/21-05/run.py",
                "manuscripts/ch21/figure-data.json",
                "calculations/results/ch21-recsys-vector-colpali.json",
                "case-studies/generative-recsys-and-colpali-rag.md"
            ],
            "TeachingRole": "定量证明频次阈值过滤与指数衰减淘汰如何消除 68.5% 的长尾噪声内存占用"
        }
    ]
    for row in fig_index:
        row['figure'] = f"21-{row['number']}"
        row['title'] = row['caption']
        row['name'] = Path(row['file']).stem
    (HERE / 'figure-index.json').write_text(json.dumps(fig_index, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    md = HERE.parent / '21-推荐检索与向量数据库.md'
    if md.exists():
        text_md = md.read_text(encoding='utf-8')
        no_code = re.sub(r'^```.*?^```', '', text_md, flags=re.M | re.S)
        display = re.findall(r'\$\$(.+?)\$\$', no_code, re.S)
        no_disp = re.sub(r'\$\$.+?\$\$', '', no_code, flags=re.S)
        no_bt = re.sub(r'`[^`\n]+`', '', no_disp)
        inline = re.findall(r'(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)', no_bt)
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
        (HERE / 'math-validation.json').write_text(json.dumps(math_val, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

        import markdown

        def repl_img(m):
            alt, rel = m.group(1), m.group(2)
            p = (md.parent / rel).with_suffix('.png')
            b64 = base64.b64encode(p.read_bytes()).decode()
            return f'<figure><img alt="{alt}" src="data:image/png;base64,{b64}"><figcaption>{alt}</figcaption></figure>'

        work = re.sub(r'!\[([^\]]*)\]\((ch21/[^)]+)\)', repl_img, text_md)
        body = markdown.markdown(work, extensions=['tables', 'fenced_code', 'footnotes', 'toc'])
        html = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>第 21 章：推荐检索与向量数据库</title></head><body>{body}</body></html>\n'
        out_html = preview_path(md)
        out_html.write_text(html, encoding='utf-8')

    sources = [
        HERE / 'build.py',
        HERE / 'figure-data.json',
        HERE / 'figure-index.json',
        ROOT / 'calculations/results/ch21-recsys-vector-colpali.json',
        ROOT / 'case-studies/generative-recsys-and-colpali-rag.md',
        ROOT / 'case-studies/retrieval-and-generation.md',
        ROOT / 'case-studies/memory-criticality-and-tiering.md',
        ROOT / 'experiments/ch21/21-01/run.py',
        ROOT / 'experiments/ch21/21-02/run.py',
        ROOT / 'experiments/ch21/21-03/run.py',
        ROOT / 'experiments/ch21/21-04/run.py',
        ROOT / 'experiments/ch21/21-05/run.py',
    ]
    if md.exists():
        sources.append(md)
    (HERE / 'sources.json').write_text(
        json.dumps(
            {"sources": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in sources]},
            ensure_ascii=False,
            indent=2,
        )
        + '\n',
        encoding='utf-8',
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
        + '\n',
        encoding='utf-8',
    )
    print(f'Built {len(outputs)} figure files for Chapter 21; 0 layout warnings.')


if __name__ == '__main__':
    build_chapter()
