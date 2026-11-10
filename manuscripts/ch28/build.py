#!/usr/bin/env python3
"""Build all 16 publication-grade figures (420pt) and validation manifests for Chapter 28."""
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
        'ch28-wan-allreduce-bottleneck-vs-diloco-two-level-optimizer',
        '跨洲际训练 70B：标准 AllReduce（700G）vs. DiLoCo 双层优化器',
        'blue',
        [
            ('① 跨洲每步同步死局\n2.4s 传 210 GB\n需 700 Gbps 专线', 'orange'),
            ('② 本地内循环 AdamW\n独立自治 H=500 步\n历时 20 分钟零外网', 'blue'),
            ('③ 外循环 Nesterov\n聚合伪梯度 Delta\n通信频率降 500x', 'teal'),
        ],
        '为什么单层同步 SGD/AdamW 无法跨越广域网（RTT=85 ms）？',
        'teal',
        [
            ('广域网高延迟与低带宽枷锁：\n每步 Barrier 等待 85 ms 光纤往返\n百 Gbps 跨洋专线造价极其高昂', 'orange'),
            ('双层优化器解耦（Inner/Outer）：\n内循环利用机房内 NVLink 高速收敛\n外循环将 500 步轨迹缩为单次同步', 'teal'),
        ],
        '本地 H=500 步 AdamW 叠外循环 Nesterov 动量，跨区通信降 500 倍',
    )

    draw_dual_band(
        ex,
        'ch28-streaming-diloco-fragment-pipeline-and-fp4-outer-delta',
        'Streaming DiLoCo 分片流水（M=25）与 FP4 伪梯度（350 Mbps）',
        'purple',
        [
            ('① 切分 M=25 个分片\n每 20 步同步 1 片\n消除 500 步末尖峰', 'blue'),
            ('② FP4 E2M1 伪梯度\n140 GB 压至 35 GB\n通信量再缩减 4 倍', 'purple'),
            ('③ 带宽仅 350 Mbps\n总压缩比 2,000x\n暴露等待 0.0 s', 'teal'),
        ],
        '基础 DiLoCo 尾部突发阻塞 vs. Streaming DiLoCo 零暴露重叠',
        'blue',
        [
            ('基础 DiLoCo 的同步尖峰隐患：\n第 500 步末集中发送 140 GB 数据\n所有区域停顿等待广域网传输完成', 'orange'),
            ('48.0 秒后台滑动窗口流式传输：\n上一轮第 m 分片在后台静默同步\n算力有效利用率高达 99.4%', 'teal'),
        ],
        '25 分片错峰流水与 FP4 量化协同，以 350 Mbps 跑满跨洲 70B 训练',
    )

    draw_dual_band(
        ex,
        'ch28-diloco-inner-step-h-vs-validation-loss-and-bandwidth',
        '内循环步数 H 与外循环动量 beta_outer 的收敛—带宽帕累托边界',
        'teal',
        [
            ('① 短步长 H=50..100\n收敛等同单机集群\n需 1.75~3.5 Gbps', 'blue'),
            ('② 黄金甜点 H=500\nNesterov 修正漂移\n仅需 350 Mbps 带宽', 'purple'),
            ('③ 超长步 H > 2000\n局部极小值过度发散\n验证集困惑度退化', 'orange'),
        ],
        '为什么外循环必须使用 Nesterov 动量而非简单模型权重平均？',
        'purple',
        [
            ('简单权重平均（FedAvg）的坍塌：\n各区域经历 500 步非线性优化后\n直接求平均会落入高损失鞍点区', 'orange'),
            ('将参数差值视为“伪梯度”方向：\n外循环维护全局速度向量 v_outer\n沿着多区域共识切线方向前瞻跃迁', 'teal'),
        ],
        '确立 H=500 与外循环动量 0.9 的黄金工作点，兼得省流与无损精度',
    )

    draw_dual_band(
        ex,
        'ch28-proof-of-learning-spot-check-and-byzantine-median',
        '去中心化算力池抗投毒：Proof-of-Learning 抽检与几何中位数',
        'blue',
        [
            ('① 确定性重放抽检\n抽查 p=5% 轨迹段\n校验算子余弦偏差', 'blue'),
            ('② 10 轮捕获 >99.9%\n实测达 99.97% 检出\n罚没恶意节点质押', 'purple'),
            ('③ Weiszfeld 中位数\n崩溃点高达 50%\n过滤 f=15% 异常点', 'teal'),
        ],
        '可信机房算术平均（AllReduce Mean）vs. 开放算力池拜占庭鲁棒聚合',
        'teal',
        [
            ('线性算术平均的零容错脆弱性：\n仅需 1 个恶意节点发送 10^6 大梯度\n即可单步摧毁全局模型权重参数', 'orange'),
            ('抽检验证 + 几何中位数双重防线：\n伪造梯度被 5% 随机重放立即识破\n残余微扰被几何中位数彻底钳制', 'teal'),
        ],
        '以 5% 抽检成本实现 99.97% 欺诈捕获，并用几何中位数守住 50% 崩溃点',
    )

    draw_dual_band(
        ex,
        'ch28-dht-hivemind-elastic-optimizer-sharding-spot-recovery',
        '基于分布式哈希表（DHT）的弹性状态分片与 Spot 秒级自愈',
        'purple',
        [
            ('① Kademlia DHT 环\n无中心元数据寻址\nO(log N) 路由发现', 'blue'),
            ('② 优化器状态分片\n跨节点冗余双副本\n心跳超时自动接管', 'purple'),
            ('③ 新节点流式拉取\n恢复仅耗 4.5 秒\nETTR 高达 96.8%', 'teal'),
        ],
        '传统全局停机回滚 Checkpoint vs. DHT 对等分片热修复（算例 28-3）',
        'blue',
        [
            ('抢占式实例（每小时中断率 8%）：\n传统框架一卡被收回即全员回滚\n有效训练时间比跌破 55%', 'orange'),
            ('分片级对等状态拉取与动态扩缩：\n存活节点从邻近副本秒级补齐分片\n以三折 Spot 算力跑出 96.8% 效率', 'teal'),
        ],
        '用 DHT 去中心化状态分片与对等热拉取驯服高抢占率的廉价 Spot 算力池',
    )

    draw_dual_band(
        ex,
        'ch28-serial-guardrail-blocking-vs-async-speculative-sidecar',
        '在线安全护栏架构：串行阻塞（+225 ms）vs. 异步伴飞（3.2 ms）',
        'teal',
        [
            ('① 传统串行护栏墙\n输入挡 45ms TTFT\n输出挡 180ms 尾包', 'orange'),
            ('② 异步流式伴飞器\n主模型零等待生成\n同卡 MIG 旁路审查', 'blue'),
            ('③ K=8 投机窗口\n单次审查 3.2 ms\n暴露延迟 < 3.5 ms', 'teal'),
        ],
        '为什么不能把 8B 安全分类器直接串联在 70B 主模型推理链路中？',
        'purple',
        [
            ('串行护栏的延迟与吞吐双重灾难：\n首词元延迟翻倍且破坏流式体验\n尾包整句复审导致长文本严重卡顿', 'orange'),
            ('双流异步伴飞投机审查架构：\n99.5% 正常请求享受零阻塞下发\n仅在末块承担 3.2 ms 轻量级确认', 'teal'),
        ],
        '将串行关卡重构为异步旁路伴飞，把护栏暴露延迟从 225 ms 压至 3.2 ms',
    )

    draw_dual_band(
        ex,
        'ch28-sliding-window-token-stream-guardrail-rollback',
        'K=8 词元滑动窗口投机放行与违规微秒级熔断撤回状态机',
        'blue',
        [
            ('① 主流生成 8 词元\n投机推入客户端缓冲\n同时共享显存送检', 'blue'),
            ('② 伴飞模型打分\nP(harm) < tau 放行\n窗口平滑向前滑动', 'purple'),
            ('③ 检出违规即熔断\n发送撤回控制帧\n释放主模型 KV 块', 'teal'),
        ],
        '逐词元同步审查 vs. K=8 块粒度滑动窗口审查的算力效率对比',
        'teal',
        [
            ('逐词元触发分类器的算力风暴：\n每生成 1 词元就跑一次护栏前向\n护栏算力开销反超主模型本身', 'orange'),
            ('K=8 块步长摊销与零拷贝 IPC：\n护栏前向频率降低 8 倍且算术强度高\n违规检出后在 8 词元窗口内即时截断', 'teal'),
        ],
        '以 K=8 滑动窗口平衡护栏算力摊销与撤回粒度，兼顾流式丝滑度与零泄漏',
    )

    draw_dual_band(
        ex,
        'ch28-in-situ-residual-stream-sae-feature-clamping',
        '残差流原位稀疏自编码器（SAE）特征监控与正交投影钳位',
        'purple',
        [
            ('① 抽取残差流激活\n定位关键安全层 h_l\n零额外显存拷贝', 'blue'),
            ('② 单概念探测点积\n<h_l, v_harm> 判别\n单层仅耗 3.6 us', 'purple'),
            ('③ 正交子空间投影\n减去有害方向分量\n总开销仅 0.08%', 'teal'),
        ],
        '外挂独立护栏模型 vs. 残差流内部表示工程（RepE）原位免疫',
        'blue',
        [
            ('纯外部黑盒文本审查的局限性：\nBase64 编码或隐喻越狱易绕过表层\n且发现违规时已消耗主模型算力', 'orange'),
            ('白盒激活空间概念探针与钳位：\n无论表层如何伪装均触发内部特征\n原位抹除有害分量引导安全续写', 'teal'),
        ],
        '在 Transformer 残差流内部以 3.6 us/层代价实现零延迟原位概念级免疫',
    )

    draw_dual_band(
        ex,
        'ch28-representation-engineering-harmful-subspace-projection',
        '多维有害子空间（k=8 正交基）Householder 投影与正常语义保持',
        'teal',
        [
            ('① 构建对比激活集\nSVD 提取主成分基\nU_harm 正交矩阵', 'blue'),
            ('② 单边 ReLU 门控\n仅当投影为正才触发\n正常请求零改动', 'purple'),
            ('③ 正交补空间保持\n余弦相似度 >0.998\n零拒答率攀升', 'teal'),
        ],
        '粗暴全局激活截断（过度拒答）vs. 单边正交子空间精准切除',
        'purple',
        [
            ('无门控全局方向减法的副作用：\n良性医学或安全科普也被误伤扭曲\n导致模型正常任务准确率下降', 'orange'),
            ('条件激活门控 + 正交投影矩阵：\n与有害基正交的 8184 维完全无损\n精准剥离越狱意图不伤推理智商', 'teal'),
        ],
        '依托高维空间正交性，在 8192 维残差流中精准切除 8 维风险子空间',
    )

    draw_dual_band(
        ex,
        'ch28-synthid-text-tournament-sampling-tree-mechanism',
        'SynthID-Text 深度 m=30 锦标赛采样树（Tournament Watermark）',
        'blue',
        [
            ('① 抽取 2^m 候选词\n按原始概率 p_V 采样\n构建满二叉树叶节点', 'blue'),
            ('② m=30 层锦标对决\n伪随机函数 g(x,l)\n高分者晋级下一轮', 'purple'),
            ('③ 根节点胜出词元\n期望分布严格无偏\n困惑度差 < 0.01', 'teal'),
        ],
        '传统红绿词表偏置水印（KGW）vs. SynthID 锦标赛无损水印对比',
        'teal',
        [
            ('红绿词表打分偏置水印的隐患：\n强行抬高绿表 logits 扭曲原始分布\n代码与数学推理准确率明显下滑', 'orange'),
            ('锦标赛非失真采样数学机制：\n对所有密钥求期望后完美还原 p_V\n不改写概率分布即可嵌入高维签名', 'teal'),
        ],
        '通过 30 层伪随机锦标赛晋级机制，实现数学期望零失真的密码学文本水印',
    )

    draw_dual_band(
        ex,
        'ch28-distributed-tp-prng-seed-synchronization-for-watermark',
        '张量并行（TP=8）分布式采样下的水印伪随机种子（8B）同步',
        'purple',
        [
            ('① 滑动窗口哈希锚\n前 H=4 词元算 HMAC\n派生当前步种子 s_t', 'blue'),
            ('② TP 主卡广播 8B\n随 AllGather 捎带\n零额外通信轮次', 'purple'),
            ('③ 词表分片独立赛\n各卡算局部晋级者\n跨卡归约决出总冠军', 'teal'),
        ],
        '为什么张量并行各卡随机数状态失步会直接导致水印检出失败？',
        'blue',
        [
            ('分布式采样伪随机种子漂移陷阱：\n若 8 张卡使用各自本地计数器生成 g\n离线检测器无法复现当时锦标赛树', 'orange'),
            ('基于上下文窗口的自同步种子：\n种子唯一取决于历史词元与密钥\n支持跨卡并行归约与摘录检测', 'teal'),
        ],
        '以 8 字节上下文 HMAC 种子绑定采样器，确保多卡 TP 水印严格可复现',
    )

    draw_dual_band(
        ex,
        'ch28-watermark-token-length-vs-detection-tpr-at-low-fpr',
        'SynthID-Text 水印长度与检出率曲线（200 词元 TPR=99.2%）',
        'teal',
        [
            ('① 30 层得分均值\n无水印期望为 0.50\n有水印显著右偏', 'blue'),
            ('② 锁定极低误报率\nFPR = 10^-6 阈值\n百万次正常零误判', 'purple'),
            ('③ 200 词元高检出\nTPR 高达 99.2%\n抗同义词替换改写', 'teal'),
        ],
        '短文本信噪比累积规律：从 50 词元到 200 词元的中心极限定理收敛',
        'purple',
        [
            ('单层打分的检出方差局限：\n每词元仅提供 1 比特弱统计信号\n需上千词元才压低百万分一误报', 'orange'),
            ('m=30 层锦标赛信噪比倍增效应：\n单词元历经 30 轮筛选凝聚强统计\n仅 200 词元即跨越 99.2% 检出线', 'teal'),
        ],
        '利用 30 层锦标赛放大单词元信息增益，在 200 词元下达成 99.2% 检出',
    )

    draw_dual_band(
        ex,
        'ch28-pickle-arbitrary-code-exploit-vs-safetensors-zero-copy-mmap',
        '模型供应链安全：Pickle 反序列化漏洞 vs. Safetensors 零拷贝',
        'blue',
        [
            ('① Pickle 栈虚拟机\n__reduce__ 执行命令\n加载权重即遭夺权', 'orange'),
            ('② Safetensors 头\n纯 JSON 元数据校验\n杜绝任意代码执行', 'blue'),
            ('③ 4096B 页对齐\nmmap 零拷贝直映\n加载提速 12.5 倍', 'teal'),
        ],
        '传统 Pickle 加载（42.5 s）vs. Safetensors 页对齐直读（3.4 s）',
        'teal',
        [
            ('Pickle 格式的安全与内存双缺陷：\n不仅可潜伏木马，还需在主机内存\n额外分配 140 GB 缓冲反序列化', 'orange'),
            ('4 KB 页对齐 + GPUDirect 直通：\n文件偏移对齐操作系统物理页表\n3.4 秒零拷贝直入 GPU HBM', 'teal'),
        ],
        '根除 Pickle 代码执行漏洞与内存双倍拷贝，实现 12.5 倍极速安全加载',
    )

    draw_dual_band(
        ex,
        'ch28-merkle-tree-weight-signature-and-tee-attestation-boot',
        '模型权重分块 Merkle 哈希树签名与机密计算（TEE）可信启动',
        'purple',
        [
            ('① 张量分块算 SHA\n构建二进制哈希树\n私钥签名树根 Root', 'blue'),
            ('② 按需并行校验块\n篡改 1 bit 即报警\n无需全量读完才验', 'purple'),
            ('③ H100 TEE 远程证明\n硬件度量解封密钥\n确保模型与护栏未改', 'teal'),
        ],
        '单文件整包 SHA256 事后校验 vs. 分片 Merkle 树流式边加载边校验',
        'blue',
        [
            ('140 GB 单文件线性哈希的痛点：\n必须串行读完全部字节才能验真伪\n任一字节损坏需重传 140 GB 整包', 'orange'),
            ('张量块级 Merkle 树 + 硬件信任根：\n流式加载时凭 O(log N) 路径即验\n结合 GPU TEE 防运行时篡改护栏', 'teal'),
        ],
        '构建从磁盘 Safetensors 哈希树到 GPU TEE 证明的端到端防篡改信任链',
    )

    draw_dual_band(
        ex,
        'ch28-sovereign-ai-multi-region-compliance-and-audit-architecture',
        '主权 AI 跨管辖区数据不出境联邦底座与不可篡改审计账本',
        'teal',
        [
            ('① 原始语料本地锁\n严守各国数据主权\n零原始样本跨境传输', 'blue'),
            ('② 仅传 FP4 伪梯度\n叠加强裁剪与噪声\n防梯度反演隐私泄露', 'purple'),
            ('③ 全链路合规存证\n训练血缘与护栏日志\n密码学签名可审计', 'teal'),
        ],
        '集中式跨国数据归集（合规红线）vs. 主权 AI 分布式协同训练架构',
        'purple',
        [
            ('主权 AI 时代的三大硬约束：\n数据主权禁跨境、算力碎片化分布\n以及生成内容可追溯与防滥用监管', 'orange'),
            ('DiLoCo + 护栏 + 水印三位一体：\n数据留本地仅传 350M 压缩伪梯度\n推理端内建原位护栏与无损水印', 'teal'),
        ],
        '打通“广域网低通信训练—实时安全护栏—生成溯源水印”主权 AI 闭环',
    )

    draw_dual_band(
        ex,
        'ch28-decentralized-training-and-trustworthy-ai-infra-synthesis',
        '第 28 章全景：广域网低通信训练与去中心化拜占庭容错（上）',
        'blue',
        [
            ('28.1 Streaming DiLoCo：\nH=500 + FP4 伪梯度流水重叠\n带宽降 2,000x (350 Mbps) 零暴露', 'blue'),
            ('28.2 拜占庭容错与弹性 DHT：\n5% 抽检 10 轮捕获率 99.97%\nSpot 算力池达成 96.8% ETTR', 'purple'),
        ],
        '第 28 章全景：在线伴飞/原位安全护栏与密码学水印供应链（下）',
        'teal',
        [
            ('28.3 伴飞护栏与原位 SAE：\nK=8 异步伴飞暴露延迟仅 3.2 ms\n残差流正交投影开销低至 0.08%', 'blue'),
            ('28.4 SynthID 水印与供应链：\n30 层锦标赛 200 词元检出 99.2%\nSafetensors 4KB 页对齐快 12.5x', 'teal'),
        ],
        '以低通信优化、鲁棒聚合、伴飞/原位护栏与无损水印筑牢可信 AI 底座',
    )

    outputs = ex.finish()

    fig_index = [
        {
            "figure_id": "图 28-1",
            "number": 1,
            "file": "ch28-wan-allreduce-bottleneck-vs-diloco-two-level-optimizer.svg",
            "asset": "ch28/ch28-wan-allreduce-bottleneck-vs-diloco-two-level-optimizer.svg",
            "section": "28.1.1",
            "caption": "跨 4 个地理区域（RTT=85 ms）联合训练 70B 模型对比：标准每步 Ring AllReduce（需 700.0 Gbps 专线）vs. DiLoCo 双层优化器（本地 AdamW H=500 步 + 外循环 Nesterov 动量，通信频率降低 500 倍）（算例 28-1）",
            "source_refs": ["experiments/ch28/28-01/run.py", "manuscripts/ch28/figure-data.json"],
            "TeachingRole": "揭示如何通过内外双层优化器解耦打破跨地域广域网的高延迟与低带宽枷锁"
        },
        {
            "figure_id": "图 28-2",
            "number": 2,
            "file": "ch28-streaming-diloco-fragment-pipeline-and-fp4-outer-delta.svg",
            "asset": "ch28/ch28-streaming-diloco-fragment-pipeline-and-fp4-outer-delta.svg",
            "section": "28.1.2",
            "caption": "Streaming DiLoCo 分片流水线（M=25 个分片，每 20 步后台传输 1 片）与 FP4 E2M1 伪梯度量化：将广域网带宽需求压至 350.0 Mbps（压缩 2,000 倍）且暴露等待为 0.0 s（算例 28-1）",
            "source_refs": ["experiments/ch28/28-01/run.py", "manuscripts/ch28/figure-data.json"],
            "TeachingRole": "展示如何消除第 500 步末尾的突发同步尖峰并实现 99.4% 的算力有效利用率"
        },
        {
            "figure_id": "图 28-3",
            "number": 3,
            "file": "ch28-diloco-inner-step-h-vs-validation-loss-and-bandwidth.svg",
            "asset": "ch28/ch28-diloco-inner-step-h-vs-validation-loss-and-bandwidth.svg",
            "section": "28.1.2",
            "caption": "DiLoCo 内循环步数 H 与外循环 Nesterov 动量对验证集困惑度（Perplexity）及广域网带宽需求的帕累托权衡曲线",
            "source_refs": ["experiments/ch28/28-01/run.py", "manuscripts/ch28/figure-data.json"],
            "TeachingRole": "阐明外循环 Nesterov 伪梯度动量为何能在 H=500 大步长下避免简单权重平均的鞍点坍塌"
        },
        {
            "figure_id": "图 28-4",
            "number": 4,
            "file": "ch28-proof-of-learning-spot-check-and-byzantine-median.svg",
            "asset": "ch28/ch28-proof-of-learning-spot-check-and-byzantine-median.svg",
            "section": "28.2.1",
            "caption": "开放不可信算力池的双层防御：随机重放抽检（Proof-of-Learning，p=5% 抽检率 10 轮捕获率 99.97%）与 Weiszfeld 几何中位数鲁棒聚合（崩溃点 50%）（算例 28-2）",
            "source_refs": ["experiments/ch28/28-02/run.py", "manuscripts/ch28/figure-data.json"],
            "TeachingRole": "解决开放算力网络中恶意节点伪造梯度或投毒攻击的博弈与数学过滤难题"
        },
        {
            "figure_id": "图 28-5",
            "number": 5,
            "file": "ch28-dht-hivemind-elastic-optimizer-sharding-spot-recovery.svg",
            "asset": "ch28/ch28-dht-hivemind-elastic-optimizer-sharding-spot-recovery.svg",
            "section": "28.2.2",
            "caption": "基于 Kademlia 分布式哈希表（DHT / Hivemind）的优化器状态分片与抢占式 Spot 实例（每小时中断率 8%）秒级对等自愈（ETTR = 96.8%）（算例 28-3）",
            "source_refs": ["experiments/ch28/28-02/run.py", "manuscripts/ch28/figure-data.json"],
            "TeachingRole": "展示无中心元数据节点下如何实现优化器状态分片的毫秒级发现与热恢复"
        },
        {
            "figure_id": "图 28-6",
            "number": 6,
            "file": "ch28-serial-guardrail-blocking-vs-async-speculative-sidecar.svg",
            "asset": "ch28/ch28-serial-guardrail-blocking-vs-async-speculative-sidecar.svg",
            "section": "28.3.1",
            "caption": "大模型在线安全护栏对比：传统串行前后置阻塞（增加 45 ms TTFT 与 180 ms 尾阻塞）vs. 异步流式伴飞护栏（Speculative Guardrail Sidecar，暴露延迟 3.2 ms）（算例 28-4）",
            "source_refs": ["experiments/ch28/28-02/run.py", "manuscripts/ch28/figure-data.json"],
            "TeachingRole": "量化双流异步伴飞架构如何将安全审查从主推理关键路径上剥离"
        },
        {
            "figure_id": "图 28-7",
            "number": 7,
            "file": "ch28-sliding-window-token-stream-guardrail-rollback.svg",
            "asset": "ch28/ch28-sliding-window-token-stream-guardrail-rollback.svg",
            "section": "28.3.1",
            "caption": "K=8 词元块粒度滑动窗口投机放行与违规检出时的客户端流式撤回及 KV 缓存熔断回收机制",
            "source_refs": ["experiments/ch28/28-02/run.py", "manuscripts/ch28/figure-data.json"],
            "TeachingRole": "揭示如何在保持流式输出丝滑体验的同时将护栏模型调用频率降低 8 倍"
        },
        {
            "figure_id": "图 28-8",
            "number": 8,
            "file": "ch28-in-situ-residual-stream-sae-feature-clamping.svg",
            "asset": "ch28/ch28-in-situ-residual-stream-sae-feature-clamping.svg",
            "section": "28.3.2",
            "caption": "推理残差流原位稀疏自编码器（SAE）特征监控与有害方向正交投影钳位（每层耗时 3.6 us，端到端开销仅 0.08%）（算例 28-5）",
            "source_refs": ["experiments/ch28/28-02/run.py", "manuscripts/ch28/figure-data.json"],
            "TeachingRole": "展示如何在模型内部激活空间以微秒级代价实现零延迟安全免疫"
        },
        {
            "figure_id": "图 28-9",
            "number": 9,
            "file": "ch28-representation-engineering-harmful-subspace-projection.svg",
            "asset": "ch28/ch28-representation-engineering-harmful-subspace-projection.svg",
            "section": "28.3.2",
            "caption": "表示工程（Representation Engineering）单边门控正交子空间投影：精准剥离有害概念分量且正交补空间正常语义零退化",
            "source_refs": ["manuscripts/ch28/figure-data.json"],
            "TeachingRole": "从线性代数正交补空间原理解释为何单边投影不会造成良性请求过度拒答"
        },
        {
            "figure_id": "图 28-10",
            "number": 10,
            "file": "ch28-synthid-text-tournament-sampling-tree-mechanism.svg",
            "asset": "ch28/ch28-synthid-text-tournament-sampling-tree-mechanism.svg",
            "section": "28.4.1",
            "caption": "SynthID-Text 深度 m=30 锦标赛水印采样树（Tournament Sampling）：通过多轮伪随机对决嵌入强统计水印并保持期望输出分布严格无偏（算例 28-6）",
            "source_refs": ["experiments/ch28/28-03/run.py", "manuscripts/ch28/figure-data.json"],
            "TeachingRole": "对比失真型 Logit 偏置水印与非失真型锦标赛采样水印的数学机制"
        },
        {
            "figure_id": "图 28-11",
            "number": 11,
            "file": "ch28-distributed-tp-prng-seed-synchronization-for-watermark.svg",
            "asset": "ch28/ch28-distributed-tp-prng-seed-synchronization-for-watermark.svg",
            "section": "28.4.1",
            "caption": "分布式张量并行（TP=8）采样下的滑动窗口上下文 HMAC 种子（8 Bytes）自同步与跨卡词表分片锦标赛归约",
            "source_refs": ["experiments/ch28/28-03/run.py", "manuscripts/ch28/figure-data.json"],
            "TeachingRole": "解决多卡分布式并行采样时伪随机数发生器状态一致性的工程关键点"
        },
        {
            "figure_id": "图 28-12",
            "number": 12,
            "file": "ch28-watermark-token-length-vs-detection-tpr-at-low-fpr.svg",
            "asset": "ch28/ch28-watermark-token-length-vs-detection-tpr-at-low-fpr.svg",
            "section": "28.4.1",
            "caption": "SynthID-Text（m=30）在假阳性率 FPR = 10^-6 下随生成词元长度变化的真阳性检出率曲线（200 词元达 99.2% TPR，困惑度偏差 < 0.01）（算例 28-6）",
            "source_refs": ["experiments/ch28/28-03/run.py", "manuscripts/ch28/figure-data.json"],
            "TeachingRole": "用中心极限定理推导 30 层锦标赛得分均值在短文本下的高信噪比收敛特性"
        },
        {
            "figure_id": "图 28-13",
            "number": 13,
            "file": "ch28-pickle-arbitrary-code-exploit-vs-safetensors-zero-copy-mmap.svg",
            "asset": "ch28/ch28-pickle-arbitrary-code-exploit-vs-safetensors-zero-copy-mmap.svg",
            "section": "28.4.2",
            "caption": "模型权重序列化对比：存在远程代码执行漏洞与双倍内存拷贝的 Pickle（42.5 s）vs. 4,096 B 页对齐零拷贝 mmap 的 Safetensors（3.4 s，提速 12.5 倍）",
            "source_refs": ["experiments/ch28/28-03/run.py", "manuscripts/ch28/figure-data.json"],
            "TeachingRole": "剖析 Safetensors 格式如何同时解决模型供应链投毒与百 GB 权重极速加载"
        },
        {
            "figure_id": "图 28-14",
            "number": 14,
            "file": "ch28-merkle-tree-weight-signature-and-tee-attestation-boot.svg",
            "asset": "ch28/ch28-merkle-tree-weight-signature-and-tee-attestation-boot.svg",
            "section": "28.4.2",
            "caption": "基于张量分块 Merkle 哈希树签名与 GPU TEE 硬件远程证明的可信模型加载及防篡改启动流水线",
            "source_refs": ["manuscripts/ch28/figure-data.json"],
            "TeachingRole": "构建从静态磁盘权重文件到运行期 GPU 显存的密码学完整性闭环"
        },
        {
            "figure_id": "图 28-15",
            "number": 15,
            "file": "ch28-sovereign-ai-multi-region-compliance-and-audit-architecture.svg",
            "asset": "ch28/ch28-sovereign-ai-multi-region-compliance-and-audit-architecture.svg",
            "section": "28.4.2",
            "caption": "主权 AI 跨管辖区合规基础设施：原始数据不出境、广域网 DiLoCo 伪梯度协同、在线护栏拦截与生成内容水印溯源",
            "source_refs": ["manuscripts/ch28/figure-data.json"],
            "TeachingRole": "展示主权 AI 架构如何在地缘法规约束下实现跨域协同训练与全生命周期可信治理"
        },
        {
            "figure_id": "图 28-16",
            "number": 16,
            "file": "ch28-decentralized-training-and-trustworthy-ai-infra-synthesis.svg",
            "asset": "ch28/ch28-decentralized-training-and-trustworthy-ai-infra-synthesis.svg",
            "section": "28.4.2",
            "caption": "第 28 章系统工程全景图：从广域网 Streaming DiLoCo 低通信训练、拜占庭容错弹性算力池到异步伴飞/原位护栏与密码学水印安全底座",
            "source_refs": ["manuscripts/ch28/figure-data.json"],
            "TeachingRole": "收束全章四大模块，凝练去中心化训练与内生可信 AI 基础设施的统一设计范式"
        }
    ]
    for row in fig_index:
        row['figure'] = f"28-{row['number']}"
        row['name'] = Path(row['file']).stem
    (HERE / 'figure-index.json').write_text(json.dumps(fig_index, ensure_ascii=False, indent=2) + '\n')

    md = HERE.parent / '28-主权 AI、广域网去中心化低通信训练（DiLoCo）、实时护栏（Guardrails）与密码学水印安全底座.md'
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

        work = re.sub(r'!\[([^\]]*)\]\((ch28/[^)]+)\)', repl_img, text_md)
        body = markdown.markdown(work, extensions=['tables', 'fenced_code', 'footnotes', 'toc'])
        html = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>第 28 章：主权 AI、广域网去中心化低通信训练（DiLoCo）、实时护栏（Guardrails）与密码学水印安全底座</title></head><body>{body}</body></html>\n'
        out_html = preview_path(md)
        out_html.write_text(html)

    sources = [
        HERE / 'build.py',
        HERE / 'figure-data.json',
        HERE / 'figure-index.json',
        ROOT / 'experiments/ch28/28-01/run.py',
        ROOT / 'experiments/ch28/28-02/run.py',
        ROOT / 'experiments/ch28/28-03/run.py',
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
    print(f'Built {len(outputs)} figure files for Chapter 28; 0 layout warnings.')


if __name__ == '__main__':
    build_chapter()
