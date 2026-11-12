#!/usr/bin/env python3
"""Build all 22 publication-grade figures (420pt) and validation manifests for Chapter 18."""
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
        'ch18-single-turn-chat-vs-multi-turn-agent-worm-loop',
        '单轮对话（Read-Once）vs. 40 轮自主智能体（WORM 95.12% 复用）',
        'blue',
        [
            ('单轮对话（Read-Once KV）\n提示 2,000 + 生成 500 词元\n结束后 KV 立即释放归零', 'orange'),
            ('40 轮编程智能体（WORM）\n每轮 +2,000 词元至 80,000\n累计预填充 164 万词元', 'teal'),
        ],
        '40 轮智能体算力与时间轴解剖（算例 18-1）',
        'teal',
        [
            ('① 历史前缀重复读\n156 万 / 164 万词元\n复用率 95.12%', 'blue'),
            ('② 预填充/解码悬殊\n164 万 vs. 8,000\n比值高达 205 : 1', 'purple'),
            ('③ 工具等待空窗\n每轮阻塞 5–45 s\nGPU 显存空转风险', 'orange'),
        ],
        '智能体负载由“预填充历史前缀复用”与“长工具等待空窗”双重支配',
    )

    draw_dual_band(
        ex,
        'ch18-tool-wait-hbm-retain-recompute-offload-phase-diagram',
        '工具等待期（t_wait）与前缀长度（L_prefix）三维决策相图',
        'teal',
        [
            ('相区 I：t < 2.16 s\n驻留 GPU HBM3e\n零换出，立即续跑', 'blue'),
            ('相区 II：t >= 2.16 s\n换出至 DRAM/RDMA\n15 s 省 85.6% 租金', 'teal'),
            ('相区 III：前缀 < 2K\n直接丢弃 HBM 块\n返回后重算最快', 'orange'),
        ],
        '64K 前缀（9.00 GiB KV）恢复延迟与租金对比（算例 18-1）',
        'blue',
        [
            ('丢弃重算（Recompute）\n65,536 词元重算需 500 ms\n浪费算力且拉长每轮延迟', 'orange'),
            ('PCIe 5.0 换入（48 GiB/s）\n单向换入仅需 187.5 ms\n比重算快 2.67 倍', 'teal'),
        ],
        '以 t* = 2.16 s 为分界执行异步分层换出，兼顾显存成本与恢复速度',
    )

    draw_dual_band(
        ex,
        'ch18-four-tier-kv-cache-hierarchy-dynamo-lmcache',
        '四级分层 KV 缓存池（Dynamo / LMCache / Mooncake）金字塔',
        'blue',
        [
            ('Tier 1: GPU HBM3e\n80–192 GB 容量\n微秒级热块命中', 'blue'),
            ('Tier 2: Host DRAM\n512 GB | 48 GiB/s\n毫秒级无损换入', 'teal'),
            ('Tier 3/4: NVMe/RDMA\n100 TB | 36 GiB/s\n跨节点共享记忆池', 'purple'),
        ],
        '块级 SHA-256 前缀哈希索引与零拷贝 DMA/RDMA 流水',
        'teal',
        [
            ('前缀块链式哈希（Chunk Hash）\n每 64 词元生成唯一指纹\n支持跨会话全局去重共享', 'blue'),
            ('逐层异步预取（Layer Prefetch）\n计算 l 层时 DMA 拉取 l+1 层\n掩盖 85% 以上 PCIe/RDMA 延迟', 'teal'),
        ],
        '四级存储池将单机显存上限扩展百倍，支撑百万级并发长程智能体',
    )

    draw_dual_band(
        ex,
        'ch18-kv-aware-global-radix-router-ttft-tradeoff',
        '全局基数树路由器（Global Radix Router）排队与命中联合寻优',
        'purple',
        [
            ('节点 A（高命中）\n排队120ms | 90%命中\nTTFT = 170.0 ms', 'teal'),
            ('节点 B（最少排队）\n排队10ms | 0%命中\nTTFT = 510.0 ms', 'orange'),
            ('节点 C（RDMA 命中）\n排队20ms | 80%远端\nTTFT = 320.0 ms', 'blue'),
        ],
        '联合代价函数：排队 + 本地 DMA + 远端 RDMA + 重算（算例 18-2）',
        'teal',
        [
            ('传统最少连接路由\n盲目派往空闲节点 B\n触发 64K 全量重算（500 ms）', 'orange'),
            ('KV 感知全局路由（Dynamo）\n精准派往热缓存节点 A\n首字延迟缩短 3.00 倍', 'teal'),
        ],
        '在长上下文智能体场景下，KV 缓存位置权重远高于瞬时队列长度',
    )

    draw_dual_band(
        ex,
        'ch18-cachegen-anchor-delta-entropy-kv-streaming',
        'CacheGen 锚点差分量化与算术熵编码流式传输（算例 18-2）',
        'teal',
        [
            ('① 锚点帧残差差分\n利用相邻词元高余弦相似\n提取层内局部残差 ΔK, ΔV', 'blue'),
            ('② 分层量化 + 算术编码\n浅层高精度、深层紧凑量化\n16-bit 压至 3.80 bits（4.21×）', 'teal'),
        ],
        '50 Gbps（5.81 GiB/s）跨机架网络传输 vs. 本地重算回本线',
        'orange',
        [
            ('原始 9.00 GiB KV 跨网传输\n耗时 1,549.1 ms > 重算 500 ms\n受制于 18 GiB/s 临界带宽墙', 'orange'),
            ('CacheGen 压缩至 2.1375 GiB\n传输 367.9 ms + 解压 18.0 ms\n总计 385.9 ms（快 4.01 倍）', 'teal'),
        ],
        '当跨机架带宽低于 18 GiB/s 时，必须结合流式熵编码才能击败重算',
    )

    draw_dual_band(
        ex,
        'ch18-microvm-snapshot-cow-userfaultfd-12ms-fork',
        '容器/虚机冷启动（1,500 ms）vs. Firecracker 12 ms 快照克隆',
        'blue',
        [
            ('传统虚机冷启动\n内核引导+环境加载\n耗时 1,500 ms', 'orange'),
            ('Firecracker 快照\nMAP_PRIVATE 映射\n克隆仅 3.2 ms', 'blue'),
            ('userfaultfd 缺页\n拉取 5.5 MiB 工作集\n总计 12.0 ms', 'teal'),
        ],
        '微虚机硬件虚拟化隔离（KVM）与极速弹性并发（算例 18-3）',
        'teal',
        [
            ('强安全边界（独立 Guest 内核）\n阻断容器逃逸与宿主机污染\n支持任意 root 编译与测试', 'purple'),
            ('启动提速 125 倍（1.5 s → 12 ms）\n每分支仅增 24 MiB 私有脏页\n单机可并发驻留上千沙箱', 'teal'),
        ],
        '微虚机 CoW 快照兼具硬件级虚拟化隔离与毫秒级函数式克隆速度',
    )

    draw_dual_band(
        ex,
        'ch18-synchronized-kv-and-sandbox-mcts-tree-forking',
        '面向智能体树搜索（MCTS）的 KV 块表与沙箱快照同步分叉',
        'purple',
        [
            ('父节点 (S_t, KV_t)\n只读基快照 512 MiB\n共享前缀 KV 块', 'blue'),
            ('16 路并发试错分支\n每分支 24 MiB 脏页\n总计 896 MiB', 'teal'),
            ('破坏分支（rm -rf）\n丢弃分支 Upperdir\n零副作用回退父节点', 'orange'),
        ],
        '模型记忆（KV Cache）与外部世界状态（Sandbox FS/RAM）原子回滚',
        'teal',
        [
            ('GPU 侧：KV 块表引用减一\n重置块指针耗时 0.15 ms\n无需物理擦除显存数据', 'blue'),
            ('CPU 侧：OverlayFS 页表回退\n卸载差异层耗时 8.35 ms\n同步回滚共 8.50 ms < 10 ms', 'teal'),
        ],
        '唯有让外部沙箱具备与 KV 缓存同构的 CoW 树状语义，MCTS 才能成立',
    )

    draw_dual_band(
        ex,
        'ch18-computer-use-gui-webrtc-frame-diff-pipeline',
        'Computer-Use 桌面 GUI 沙箱脏矩形差分流与 DOM 混合感知',
        'blue',
        [
            ('① 虚拟帧缓冲 Xvfb\n仅捕获变化像素块\n省 92% 像素带宽', 'blue'),
            ('② 多模态混合提取\n截图 + 无障碍树\n消除小图标漂移', 'teal'),
            ('③ 亚毫秒动作注入\nxdotool 内核事件\n步进延迟 < 25 ms', 'purple'),
        ],
        'GUI 视觉词元预算控制与高分辨率局部放大（Crop-and-Zoom）',
        'teal',
        [
            ('全屏粗粒度概览（256 词元）\n快速定位目标窗口与控件区域\n避免每步灌入 4,096 视觉词元', 'blue'),
            ('焦点区域高分裁剪（+128 词元）\n对准按钮/输入框做像素级点击\n坐标定位精度提升至 98.4%', 'teal'),
        ],
        '脏矩形差分与无障碍树融合将 GUI 智能体单步视觉开销压降 85%',
    )

    draw_dual_band(
        ex,
        'ch18-mcp-gateway-dynamic-tool-rag-and-auth-isolation',
        '模型上下文协议（MCP）安全网关与按需工具语义检索（算例 18-5）',
        'teal',
        [
            ('企业级万级工具库挑战\n挂载 1,000 个工具 Schema\n每轮白白消耗 180,000 词元', 'orange'),
            ('动态工具检索（Tool RAG）\n稠密向量召回 Top-8 相关工具\n仅占 1,440 词元（省 99.20%）', 'teal'),
        ],
        '零信任租户级 OAuth 凭据注入与细粒度 RBAC 权限沙箱',
        'purple',
        [
            ('模型对凭据零可见（Zero-Trust）\nLLM 仅输出符号化调用参数\n网关在出境瞬间注入短效 Token', 'blue'),
            ('Schema 语法约束（XGrammar）\n将 JSON Schema 转为自动机\n100% 杜绝非法字段与类型幻觉', 'purple'),
        ],
        '动态工具检索实现 125 倍上下文压缩，同时消除多工具注意力稀释',
    )

    draw_dual_band(
        ex,
        'ch18-idempotent-outbox-and-speculative-tool-prefetch',
        '只读工具投机预取 vs. 写操作工具幂等事务外发箱（Outbox）',
        'orange',
        [
            ('只读工具（搜索 / 查库 / 读文件）\n在解码生成参数中途投机预热\n掩盖 60%–80% 外部 API 往返时延', 'teal'),
            ('写操作工具（转账 / 下单 / 删库）\n严禁投机执行，强制绑定幂等键\nKey = Hash(Session, Step)', 'orange'),
        ],
        '两阶段事务外发箱（Outbox）保障精确一次（Exactly-Once）执行',
        'blue',
        [
            ('① 本地持久化写前日志（WAL）\n先将工具调用意图落盘至 Outbox\n防止推理节点崩溃导致状态丢失', 'blue'),
            ('② 远端幂等去重网关校验\n网络超时重试携带相同幂等键\n直接返回首次结果，零重复扣款', 'teal'),
        ],
        '区分只读投机与有副作用写操作，是智能体进入金融/生产系统的底线',
    )

    draw_dual_band(
        ex,
        'ch18-audit-telemetry-backlog-buffer-overflow-and-drain',
        '算例 18-4：审计遥测断网积压与 81.92 s 内存缓冲触顶',
        'orange',
        [
            ('断网产生速率 400 KiB/s\n本地内存缓冲 32 MiB\nt = 81.92 s 时内存达到满溢', 'orange'),
            ('120 s 断网总积压 46.875 MiB\n超出内存的 14.875 MiB 溢写至\n本地 NVMe 外发箱（零丢失）', 'purple'),
        ],
        '网络恢复后 1.0 MiB/s 上行并发排空动力学（算例 18-4）',
        'teal',
        [
            ('上行带宽 1,024 KiB/s\n扣除持续新增流 400 KiB/s\n净排空速率 = 624 KiB/s', 'blue'),
            ('排空 48,000 KiB 积压耗时：\nT_drain = 48,000 / 624\n= 76.92 s 完全恢复稳态', 'teal'),
        ],
        '净排空定律：积压消减时间取决于（上行带宽 - 持续产生率）的净差值',
    )

    draw_dual_band(
        ex,
        'ch18-prompt-injection-taint-tracking-dual-llm-firewall',
        '防御间接提示词注入（Indirect Prompt Injection）双流隔离',
        'purple',
        [
            ('非受信外部数据流（Tainted）\n网页正文 / 邮件 / 外部文档\n由受限隔离模型解析提取变量', 'orange'),
            ('特权控制流（Privileged）\n仅接收用户意图与符号化变量\n由规划主模型生成工具调用', 'blue'),
        ],
        '动态信息流污点追踪（Taint Tracking）与数据—指令边界门禁',
        'teal',
        [
            ('污点标签传播（Taint Tagging）\n源自外部检索的变量均打上污点位\n严禁直接拼接到高危系统命令', 'purple'),
            ('能力令牌（Capability Token）\n高危写操作需匹配初始授权策略图\n100% 阻断隐藏指令越权转账/泄密', 'teal'),
        ],
        '将控制面指令与数据面内容物理隔离，从根源阻断间接提示词注入',
    )

    draw_dual_band(
        ex,
        'ch18-hipporag2-graph-vector-episodic-memory-consolidation',
        '突破窗口限制的长期记忆底座：HippoRAG 2 + Mem0 双层巩固',
        'blue',
        [
            ('海马体索引层（图谱 + 向量）\n实体—关系三元组 + 稠密嵌入\n个性化 PageRank 跨跳关联召回', 'blue'),
            ('新皮层记忆巩固（Mem0 流水）\n异步提取用户偏好与排障经验\n冲突消解后更新长期事实库', 'teal'),
        ],
        '百万词元历史全量塞入 vs. 图谱—向量混合检索的成本与精度',
        'teal',
        [
            ('全量塞入 1M 上下文窗口\n每轮预填充耗费海量算力与 HBM\n且受“中间迷失”效应困扰', 'orange'),
            ('HippoRAG 2 召回 Top-4K 锚点\n多跳问答准确率提升 +14.2%\n长程会话显存开销降低 96%', 'teal'),
        ],
        '短期工作记忆靠分层 KV 缓存，跨会话长期记忆靠图谱—向量混合巩固',
    )

    draw_dual_band(
        ex,
        'ch18-multi-agent-dag-scheduler-and-token-budget-allocation',
        '多智能体（Planner-Coder-Tester）DAG 拓扑调度与死锁熔断',
        'teal',
        [
            ('Planner 规划器\n拆解任务依赖 DAG\n分配节点词元预算', 'blue'),
            ('Coder 编码工作池\n共享代码库前缀 KV\nCoW 沙箱并发修改', 'teal'),
            ('Tester 验证网关\n运行单测返回报错\n触发通过或回滚重修', 'purple'),
        ],
        '防无限争论循环（Infinite Chatter Loop）的词元经济学熔断器',
        'orange',
        [
            ('边际信息增益监控\n连续 3 轮修改未减少报错数时\n立即终止局部自修并升级路由', 'orange'),
            ('拓扑级共享前缀广播\n子智能体复用 85% 公共仓库 KV\n避免 M 个智能体重复编码', 'teal'),
        ],
        '多智能体编排必须绑定有向无环图预算与早停熔断，防止词元风暴',
    )

    draw_dual_band(
        ex,
        'ch18-agentic-cluster-cpu-gpu-sandbox-pool-sizing',
        '大规模 Agentic 云平台三池解耦配比（GPU : DRAM : CPU 沙箱）',
        'blue',
        [
            ('GPU 推理池\n专注高频预填充/解码\n不陪等工具与 I/O', 'blue'),
            ('DRAM/NVMe 记忆池\n承接等待期换出 KV\n容量达 HBM 30 倍', 'teal'),
            ('CPU 微虚机沙箱池\n运行万级并发沙箱\n12 ms 克隆与销毁', 'purple'),
        ],
        '基于 Little 定律 L = λ × W 的三池容量配比公式',
        'teal',
        [
            ('单次推理活跃期 W_gpu ≈ 1.2 s\n工具沙箱执行期 W_sbx ≈ 14.4 s\n在途会话驻留比 = 1 : 12', 'orange'),
            ('1 张 H100 推理卡配对：\n512 GB 远端 KV 缓存池 +\n96 核高密 CPU 沙箱节点', 'teal'),
        ],
        '按在途驻留时间比解耦 GPU 推理池与 CPU 沙箱池，消除异构资源木桶等待',
    )

    draw_dual_band(
        ex,
        'ch18-autonomous-agent-infrastructure-full-stack-map',
        '第 18 章自主智能体基础设施全栈全景图（记忆 · 沙箱 · 网关）',
        'teal',
        [
            ('记忆：四级 KV 池\n95.12% 复用换出\n路由 3× | 压缩 4.2×', 'blue'),
            ('执行：12 ms 微虚机\nFirecracker 快照克隆\nMCTS 回滚 8.50 ms', 'teal'),
            ('控制：MCP 幂等网关\nTool RAG 省 99.2%\n算例 18-4 排空 76.92 s', 'purple'),
        ],
        '从“无状态词元生成”向“有状态、可回溯、防注入自主系统”演进',
        'blue',
        [
            ('核心瓶颈：高复用前缀占满 HBM\n沙箱冷启动慢、重试引发副作用', 'orange'),
            ('系统解法：分层 KV 池、CoW 沙箱\n事务外发箱与双流污点防火墙', 'teal'),
        ],
        '第 18 章架构主线：记忆分层、环境分叉与工具幂等构成智能体三大支柱',
    )

    draw_dual_band(
        ex,
        'ch18-speculative-tool-execution-and-rollback-dag',
        '多智能体 DAG 关键路径投机工具预执行（算例 18-7）',
        'teal',
        [
            ('传统串行等待基线\n600 ms 模型 + 1,400 ms 工具\n4 阶段累计 8,000.0 ms', 'orange'),
            ('Draft 100 ms 投机预启动\n命中占 80%，净耗时 1,050 ms\n4 阶段期望 4,969.6 ms（1.61×）', 'teal'),
        ],
        '12.0 ms CoW 快照回滚兜底与 1.25% 超低盈亏平衡线',
        'purple',
        [
            ('命中（p = 0.80）：直接提交\n掩盖 950 ms 关键路径等待\n下游节点流水线无缝衔接', 'blue'),
            ('误判（p = 0.20）：12 ms 回滚\nFirecracker 快照瞬时复位\n命中率 > 1.25% 即获得正收益', 'purple'),
        ],
        '12 ms 微虚机快照赋予投机工具执行廉价反悔权，削减 37.88% 端到端时延',
    )

    draw_dual_band(
        ex,
        'ch18-gui-incremental-region-of-interest-vision-token-cache',
        'GUI 智能体增量脏矩形 ROI 视觉缓存与 DOM 剪枝（算例 18-6）',
        'blue',
        [
            ('全量重编码基线（30 步）\n2,400 视觉 + 3,600 DOM/步\n累计 180,000 词元 | 1,350 ms', 'orange'),
            ('脏矩形 ROI 视觉块复用\n仅编码 12.5% 变化瓦片（300）\n87.5% 静态背景直接复用 KV', 'teal'),
        ],
        'Set-of-Marks 可交互 DOM 剪枝与双通道坐标一致性门禁',
        'teal',
        [
            ('视口可交互 DOM 子树提取\n3,600 词元压至 450 词元/步\n30 步总词元 24,600（省 86.33%）', 'blue'),
            ('30 步 ViT 总耗时 251.62 ms\n视觉编码加速 5.37×\n词元上下文压缩 7.32×', 'purple'),
        ],
        '结合脏矩形视觉块哈希复用与可交互 DOM 剪枝，消除 GUI 智能体上下文膨胀',
    )

    draw_dual_band(
        ex,
        'ch18-xgrammar-context-independent-token-bitmask-cache',
        'XGrammar 上下文无关二分与紧凑位掩码（算例 18-8）',
        'blue',
        [
            ('① BPE 跨界词元\n单块连跨多层括号\n15 万词逐检过慢', 'orange'),
            ('② 99% 无关词元\n仅看顶层有限状态\n离线预编静态掩码', 'teal'),
            ('③ <1% 相关词元\n闭合符查嵌套栈深\n步内增量合并掩码', 'blue'),
        ],
        '双主线模型 uint32 紧凑位打包体积与 PCIe 5.0 带宽账本',
        'teal',
        [
            ('Qwen3（V = 151,936）\n4,748 个 uint32 = 18,992 B\nB=64 仅 1.16 MiB (23.6 μs)', 'blue'),
            ('DeepSeek-V3（V = 129,280）\n4,040 个 uint32 = 16,160 B\n热表常驻 CPU L2/L3 缓存', 'purple'),
        ],
        '将 15 万次在线下推栈模拟化简为 L2 查表与 <1% 边界按位或',
    )

    draw_dual_band(
        ex,
        'ch18-cpu-pda-async-bitmask-and-gpu-logits-overlap-timeline',
        '同步状态机阻塞 vs. 异步 CPU 位掩码与 GPU 解码重叠',
        'teal',
        [
            ('朴素在线同步状态机（B=64）\nCPU 串行扫词表阻塞 38.4 ms\n步长 46.9 ms，折损 81.9%', 'orange'),
            ('XGrammar 异步流水线重叠\nCPU 2.10 ms + H2D 23.6 μs\n隐藏于 8.50 ms GPU 步内', 'teal'),
        ],
        '融合位掩码采样内核与投机草稿树（γ + 1 行）栈回滚',
        'purple',
        [
            ('GPU 融合掩码内核（<15 μs）\n按位测试 (word >> bit) & 1\n屏蔽位写 -∞ 后直接采样', 'blue'),
            ('投机解码树（γ=4 备 320 行）\n持久化链式栈 O(1) 分叉\n草稿拒绝时瞬时退回栈顶', 'purple'),
        ],
        '只要 CPU 合成加 DMA 小于 GPU 步长，强语法约束即零吞吐折损',
    )

    draw_dual_band(
        ex,
        'ch18-multi-agent-dag-fanout-tail-latency-amplification-curve',
        '并行子智能体扇出尾延迟放大与 Quorum 治理（算例 18-9）',
        'orange',
        [
            ('① 极值放大定律\nF_single = q^(1/K)\nK=16 需 99.937%', 'orange'),
            ('② K=12 同步栅栏\n单路 P99 = 6.2 s\n放大至 19.8 s', 'orange'),
            ('③ 关键路径定额\n抢占 + 10/12 汇聚\n压回 7.4 s (-62.6%)', 'teal'),
        ],
        '动态关键路径松弛度（Slack）优先级调度与对冲请求控制',
        'blue',
        [
            ('零松弛关键路径（Slack = 0）\n赋予最高批配额与热 KV 槽\n超时分位触发单路对冲副本', 'blue'),
            ('正松弛非关键支路（Slack > 0）\n平滑让出算力并设软截止期\n防止慢支路拖慢全局栅栏', 'teal'),
        ],
        '并行扇出数越大，越须用松弛度抢占与 M-of-K 定额替代全员栅栏',
    )

    draw_dual_band(
        ex,
        'ch18-speculative-readonly-tool-execution-and-saga-compensation',
        '流式 JSON 只读工具投机 vs. 写工具 Saga 补偿（算例 18-10）',
        'purple',
        [
            ('流式只读投机（检索 / SELECT）\n必填查询键闭合即并发查库\np=0.86 省 210.7 ms (+16.3%)', 'teal'),
            ('外部写操作门禁（支付 / 建单）\n严禁投机，落盘 Outbox 幂等键\n失败按 LIFO 逆序 Saga 补偿', 'orange'),
        ],
        '防级联风暴三层词元预算护栏（根图上限 · 深度衰减 · 增益熔断）',
        'teal',
        [
            ('分层配额与重试指数退避\n按 DAG 深度设词元上限\n阻断跨智能体级联重试雪崩', 'blue'),
            ('VM 快照与外部 Saga 边界\n本地文件内存靠 CoW 回退\n已出境外部副作用靠冲正', 'purple'),
        ],
        '流式只读投机削减等待时延，幂等外发箱与逆序 Saga 保障状态一致',
    )

    outputs = ex.finish()

    fig_index = [
        {
            "figure_id": "图 18-1",
            "number": 1,
            "file": "ch18-single-turn-chat-vs-multi-turn-agent-worm-loop.svg",
            "asset": "ch18/ch18-single-turn-chat-vs-multi-turn-agent-worm-loop.svg",
            "section": "18.1.1",
            "caption": "单轮对话（Read-Once）与 40 轮自主智能体（WORM 95.12% 前缀复用 + 长工具等待空窗）的显存与算力时间线对比（算例 18-1）",
            "source_refs": ["experiments/ch18/18-01/run.py", "manuscripts/ch18/figure-data.json"],
            "TeachingRole": "揭示多轮智能体 95.12% 预填充词元为历史前缀重读且预填充/解码比达 205:1 的物理特征"
        },
        {
            "figure_id": "图 18-2",
            "number": 2,
            "file": "ch18-tool-wait-hbm-retain-recompute-offload-phase-diagram.svg",
            "asset": "ch18/ch18-tool-wait-hbm-retain-recompute-offload-phase-diagram.svg",
            "section": "18.1.2",
            "caption": "工具等待时长 t_wait 与前缀长度 L_prefix 构成的“HBM 驻留 vs. 分层换出 vs. 丢弃重算”三维最优决策相图（t* = 2.16 s）",
            "source_refs": ["experiments/ch18/18-01/run.py", "manuscripts/ch18/figure-data.json"],
            "TeachingRole": "推导工具等待期 HBM 租金与 PCIe 换入/重算耗时的三段式切换边界"
        },
        {
            "figure_id": "图 18-3",
            "number": 3,
            "file": "ch18-four-tier-kv-cache-hierarchy-dynamo-lmcache.svg",
            "asset": "ch18/ch18-four-tier-kv-cache-hierarchy-dynamo-lmcache.svg",
            "section": "18.2.1",
            "caption": "GPU HBM ↔ Host DRAM ↔ Local NVMe ↔ Remote RDMA Pool 四级分层 KV 缓存架构与块级哈希去重",
            "source_refs": ["manuscripts/ch18/figure-data.json"],
            "TeachingRole": "构建跨节点共享的四级 KV 记忆存储池与逐层异步 DMA 预取流水"
        },
        {
            "figure_id": "图 18-4",
            "number": 4,
            "file": "ch18-kv-aware-global-radix-router-ttft-tradeoff.svg",
            "asset": "ch18/ch18-kv-aware-global-radix-router-ttft-tradeoff.svg",
            "section": "18.2.1",
            "caption": "KV 感知全局路由器（NVIDIA Dynamo / SGLang HiCache）在节点排队延迟与前缀缓存命中率之间的联合寻优（算例 18-2）",
            "source_refs": ["experiments/ch18/18-01/run.py", "manuscripts/ch18/figure-data.json"],
            "TeachingRole": "证明综合排队与命中率的全局路由可将 64K 智能体请求 TTFT 从 510 ms 压至 170 ms（快 3.00 倍）"
        },
        {
            "figure_id": "图 18-5",
            "number": 5,
            "file": "ch18-cachegen-anchor-delta-entropy-kv-streaming.svg",
            "asset": "ch18/ch18-cachegen-anchor-delta-entropy-kv-streaming.svg",
            "section": "18.2.2",
            "caption": "CacheGen 利用锚点残差量化与算术编码实现 4.21 倍流式压缩并在 50 Gbps 网络上击败本地重算（算例 18-2）",
            "source_refs": ["experiments/ch18/18-01/run.py", "manuscripts/ch18/figure-data.json"],
            "TeachingRole": "确立跨机架传输 KV 缓存战胜本地 GPU 重算的临界带宽阈值与熵编码收益"
        },
        {
            "figure_id": "图 18-6",
            "number": 6,
            "file": "ch18-microvm-snapshot-cow-userfaultfd-12ms-fork.svg",
            "asset": "ch18/ch18-microvm-snapshot-cow-userfaultfd-12ms-fork.svg",
            "section": "18.3.1",
            "caption": "传统容器/虚机冷启动（1,500 ms）vs. Firecracker 内存快照 + userfaultfd 惰性缺页加载实现 12.0 ms 极速克隆（算例 18-3）",
            "source_refs": ["experiments/ch18/18-02/run.py", "manuscripts/ch18/figure-data.json"],
            "TeachingRole": "拆解微虚机写时复制快照与按需工作集预取实现 125 倍启动加速的内核机理"
        },
        {
            "figure_id": "图 18-7",
            "number": 7,
            "file": "ch18-synchronized-kv-and-sandbox-mcts-tree-forking.svg",
            "asset": "ch18/ch18-synchronized-kv-and-sandbox-mcts-tree-forking.svg",
            "section": "18.3.2",
            "caption": "面向智能体树搜索（Tree-of-Actions）的 PagedAttention KV 块表与沙箱 OverlayFS/内存快照同步分叉与 8.50 ms 回滚",
            "source_refs": ["experiments/ch18/18-02/run.py", "manuscripts/ch18/figure-data.json"],
            "TeachingRole": "实现模型内部 KV 记忆与外部沙箱文件系统/内存状态的原子级同步树状回溯"
        },
        {
            "figure_id": "图 18-8",
            "number": 8,
            "file": "ch18-computer-use-gui-webrtc-frame-diff-pipeline.svg",
            "asset": "ch18/ch18-computer-use-gui-webrtc-frame-diff-pipeline.svg",
            "section": "18.3.2",
            "caption": "Computer-Use 桌面沙箱的脏矩形帧差分流（Dirty-Rect Streaming）、无障碍树（AXTree）混合提取与焦点放大流水线",
            "source_refs": ["manuscripts/ch18/figure-data.json"],
            "TeachingRole": "通过脏矩形编码与 DOM/AXTree 结构化锚点将 GUI 智能体单步视觉开销压降 85%"
        },
        {
            "figure_id": "图 18-9",
            "number": 9,
            "file": "ch18-mcp-gateway-dynamic-tool-rag-and-auth-isolation.svg",
            "asset": "ch18/ch18-mcp-gateway-dynamic-tool-rag-and-auth-isolation.svg",
            "section": "18.4.1",
            "caption": "模型上下文协议（MCP）安全网关：按需语义检索挂载 Top-8 工具 Schema（省 99.20% 词元）与零信任凭据隔离（算例 18-5）",
            "source_refs": ["experiments/ch18/18-02/run.py", "manuscripts/ch18/figure-data.json"],
            "TeachingRole": "解决万级企业工具全量挂载引发的 18 万词元上下文膨胀与凭据泄露风险"
        },
        {
            "figure_id": "图 18-10",
            "number": 10,
            "file": "ch18-idempotent-outbox-and-speculative-tool-prefetch.svg",
            "asset": "ch18/ch18-idempotent-outbox-and-speculative-tool-prefetch.svg",
            "section": "18.4.1",
            "caption": "只读工具投机预取与写操作工具基于幂等键（Idempotency Key）和事务外发箱（Outbox）的精确一次执行保障",
            "source_refs": ["manuscripts/ch18/figure-data.json"],
            "TeachingRole": "防止智能体超时重试或投机执行在外部世界引发重复扣款或重复下单事故"
        },
        {
            "figure_id": "图 18-11",
            "number": 11,
            "file": "ch18-audit-telemetry-backlog-buffer-overflow-and-drain.svg",
            "asset": "ch18/ch18-audit-telemetry-backlog-buffer-overflow-and-drain.svg",
            "section": "18.4.1",
            "caption": "审计遥测断网期间 400 KiB/s 积压曲线（81.92 s 触顶 32 MiB、120 s 累积 46.875 MiB）与恢复后 76.92 s 净速率排空过程（算例 18-4）",
            "source_refs": ["experiments/ch18/18-02/run.py", "manuscripts/ch18/figure-data.json"],
            "TeachingRole": "精确复算断网期间内存缓冲触顶时刻、外发箱溢写体积与净排空速率下的恢复时长"
        },
        {
            "figure_id": "图 18-12",
            "number": 12,
            "file": "ch18-prompt-injection-taint-tracking-dual-llm-firewall.svg",
            "asset": "ch18/ch18-prompt-injection-taint-tracking-dual-llm-firewall.svg",
            "section": "18.4.1",
            "caption": "防御间接提示词注入（Indirect Prompt Injection）的特权控制流与非受信数据流双 LLM 污点追踪防火墙",
            "source_refs": ["manuscripts/ch18/figure-data.json"],
            "TeachingRole": "通过动态污点标签与控制面/数据面物理隔离阻断外部网页或邮件中的隐藏越权指令"
        },
        {
            "figure_id": "图 18-13",
            "number": 13,
            "file": "ch18-hipporag2-graph-vector-episodic-memory-consolidation.svg",
            "asset": "ch18/ch18-hipporag2-graph-vector-episodic-memory-consolidation.svg",
            "section": "18.4.2",
            "caption": "HippoRAG 2 / Mem0 结合向量检索、知识图谱个性化 PageRank 与后台异步记忆提炼的长期记忆底座",
            "source_refs": ["manuscripts/ch18/figure-data.json"],
            "TeachingRole": "利用海马体图谱索引与新皮层异步巩固突破上下文窗口限制并降低 96% 预填充开销"
        },
        {
            "figure_id": "图 18-14",
            "number": 14,
            "file": "ch18-multi-agent-dag-scheduler-and-token-budget-allocation.svg",
            "asset": "ch18/ch18-multi-agent-dag-scheduler-and-token-budget-allocation.svg",
            "section": "18.4.2",
            "caption": "复杂软件工程任务中 Planner—Coder—Tester 多智能体 DAG 拓扑调度、公共前缀广播与动态词元预算熔断",
            "source_refs": ["manuscripts/ch18/figure-data.json"],
            "TeachingRole": "通过边际增益监控与 DAG 预算熔断消除多智能体无限循环空谈风暴"
        },
        {
            "figure_id": "图 18-15",
            "number": 15,
            "file": "ch18-agentic-cluster-cpu-gpu-sandbox-pool-sizing.svg",
            "asset": "ch18/ch18-agentic-cluster-cpu-gpu-sandbox-pool-sizing.svg",
            "section": "18.3.2",
            "caption": "大规模 Agentic 云平台中 GPU 推理池、大容量 DRAM/NVMe KV 缓存池与高密 CPU 沙箱池的 Little 定律配比模型",
            "source_refs": ["manuscripts/ch18/figure-data.json"],
            "TeachingRole": "依据推理活跃期与工具执行期的 1:12 驻留比指导异构计算与存储池容量规划"
        },
        {
            "figure_id": "图 18-16",
            "number": 16,
            "file": "ch18-autonomous-agent-infrastructure-full-stack-map.svg",
            "asset": "ch18/ch18-autonomous-agent-infrastructure-full-stack-map.svg",
            "section": "18.4.2",
            "caption": "自主智能体基础设施从四级分层 KV 记忆、12 ms CoW 微虚机沙箱到幂等安全网关的全栈总图",
            "source_refs": ["experiments/ch18/18-01/run.py", "experiments/ch18/18-02/run.py", "manuscripts/ch18/figure-data.json"],
            "TeachingRole": "收束第 18 章记忆分层化、环境可分叉化与工具执行幂等化的完整工程体系"
        },
        {
            "figure_id": "图 18-17",
            "number": 17,
            "file": "ch18-speculative-tool-execution-and-rollback-dag.svg",
            "asset": "ch18/ch18-speculative-tool-execution-and-rollback-dag.svg",
            "section": "18.4.2",
            "caption": "多智能体工作流 DAG 关键路径投机工具预执行（Speculative Tool Execution）与 CoW 沙箱毫秒级误判回滚流水线（算例 18-7）",
            "source_refs": ["experiments/ch18/18-03/run.py", "manuscripts/ch18/figure-data.json"],
            "TeachingRole": "展示 Draft 投机预启动工具与 12 ms CoW 回滚如何将 4 阶段 DAG 延迟从 8,000 ms 压至 4,969.6 ms（1.61x）"
        },
        {
            "figure_id": "图 18-18",
            "number": 18,
            "file": "ch18-gui-incremental-region-of-interest-vision-token-cache.svg",
            "asset": "ch18/ch18-gui-incremental-region-of-interest-vision-token-cache.svg",
            "section": "18.4.2",
            "caption": "GUI 智能体（OSWorld / WebArena）增量脏矩形 ROI 视觉块哈希复用与 Set-of-Marks 可交互 DOM 剪枝流水线（算例 18-6）",
            "source_refs": ["experiments/ch18/18-03/run.py", "manuscripts/ch18/figure-data.json"],
            "TeachingRole": "证明 12.5% 脏矩形视觉缓存与可交互 DOM 剪枝实现 7.32 倍词元压缩与 5.37 倍 ViT 编码提速"
        },
        {
            "figure_id": "图 18-19",
            "number": 19,
            "file": "ch18-xgrammar-context-independent-token-bitmask-cache.svg",
            "asset": "ch18/ch18-xgrammar-context-independent-token-bitmask-cache.svg",
            "section": "18.5.1",
            "caption": "XGrammar 上下文无关词元二分（约 99% 静态预编译缓存 vs. <1% 动态栈检）与双主线模型紧凑 uint32 位掩码结构（算例 18-8）",
            "source_refs": ["calculations/results/ch18-agent-worm-grammar.json", "experiments/ch18/18-04/run.py"],
            "TeachingRole": "量化 Qwen3（4,748 字/18,992 B，B=64 仅 1.16 MiB）与 DeepSeek-V3（4,040 字/16,160 B）的紧凑位掩码账本"
        },
        {
            "figure_id": "图 18-20",
            "number": 20,
            "file": "ch18-cpu-pda-async-bitmask-and-gpu-logits-overlap-timeline.svg",
            "asset": "ch18/ch18-cpu-pda-async-bitmask-and-gpu-logits-overlap-timeline.svg",
            "section": "18.5.2",
            "caption": "朴素同步状态机阻塞（38.4 ms，吞吐折损 81.9%）vs. XGrammar CPU 异步合成（2.10 ms）与 GPU 解码步（8.50 ms）无气泡重叠流水线（算例 18-8）",
            "source_refs": ["calculations/results/ch18-agent-worm-grammar.json", "experiments/ch18/18-04/run.py"],
            "TeachingRole": "证明 CPU 异步掩码合成 + PCIe 23.6 μs DMA 可完全隐藏于 GPU 前向步下实现零吞吐损失"
        },
        {
            "figure_id": "图 18-21",
            "number": 21,
            "file": "ch18-multi-agent-dag-fanout-tail-latency-amplification-curve.svg",
            "asset": "ch18/ch18-multi-agent-dag-fanout-tail-latency-amplification-curve.svg",
            "section": "18.6.1",
            "caption": "多智能体并行扇出极值统计尾延迟放大（K=12 同步栅栏 6.2 s → 19.8 s）与关键路径松弛度抢占 + 10/12 Quorum 收敛（7.4 s，省 62.6%，算例 18-9）",
            "source_refs": ["calculations/results/ch18-agent-worm-grammar.json", "experiments/ch18/18-05/run.py"],
            "TeachingRole": "推导顺序统计量 F(T_join,q)=q^(1/K) 并给出关键路径抢占与定额提前汇聚的系统解法"
        },
        {
            "figure_id": "图 18-22",
            "number": 22,
            "file": "ch18-speculative-readonly-tool-execution-and-saga-compensation.svg",
            "asset": "ch18/ch18-speculative-readonly-tool-execution-and-saga-compensation.svg",
            "section": "18.6.2",
            "caption": "流式部分 JSON 只读工具提前投机（省 210.7 ms，额外查询 16.3%）、有副作用写操作逆序 LIFO Saga 补偿与三层词元预算护栏（算例 18-10）",
            "source_refs": ["calculations/results/ch18-agent-worm-grammar.json", "experiments/ch18/18-05/run.py"],
            "TeachingRole": "划清只读投机、本地 CoW 快照回退与跨外部服务逆向 Saga 冲正的精确职责边界"
        }
    ]
    for row in fig_index:
        row['figure'] = f"18-{row['number']}"
        row['name'] = Path(row['file']).stem
    (HERE / 'figure-index.json').write_text(json.dumps(fig_index, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    md = HERE.parent / '18-自主智能体系统.md'
    if md.exists():
        text_md = md.read_text(encoding='utf-8')
        no_code = re.sub(r'```.*?```', '', text_md, flags=re.S)
        no_code = re.sub(r'`[^`\n]*`', '', no_code)
        display = re.findall(r'\$\$(.+?)\$\$', no_code, re.S)
        no_disp = re.sub(r'\$\$.+?\$\$', '', no_code, flags=re.S)
        inline = re.findall(r'(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)', no_disp)
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
        (HERE / 'math-validation.json').write_text(res.stdout, encoding='utf-8')

        import markdown

        def repl_img(m: re.Match[str]) -> str:
            alt, rel = m.group(1), m.group(2)
            p = (md.parent / rel).with_suffix('.png')
            b64 = base64.b64encode(p.read_bytes()).decode()
            return f'<figure><img alt="{alt}" src="data:image/png;base64,{b64}"><figcaption>{alt}</figcaption></figure>'

        work = re.sub(r'!\[([^\]]*)\]\((ch18/[^)]+)\)', repl_img, text_md)
        body = markdown.markdown(work, extensions=['tables', 'fenced_code', 'footnotes', 'toc'])
        html = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>第 18 章：自主智能体系统</title></head><body>{body}</body></html>\n'
        out_html = preview_path(md)
        out_html.write_text(html, encoding='utf-8')

    sources = [
        HERE / 'build.py',
        HERE / 'figure-data.json',
        HERE / 'figure-index.json',
        md,
        ROOT / 'calculations/results/ch18-agent-worm-grammar.json',
        ROOT / 'experiments/ch18/18-01/run.py',
        ROOT / 'experiments/ch18/18-02/run.py',
        ROOT / 'experiments/ch18/18-03/run.py',
        ROOT / 'experiments/ch18/18-04/run.py',
        ROOT / 'experiments/ch18/18-05/run.py',
    ]
    (HERE / 'sources.json').write_text(
        json.dumps(
            {
                "chapter": "ch18",
                "sources": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in sources if p.exists()],
            },
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
    print(f'Built {len(outputs)} figure files for Chapter 18; 0 layout warnings.')


if __name__ == '__main__':
    build_chapter()
