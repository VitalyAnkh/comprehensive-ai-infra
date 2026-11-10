#!/usr/bin/env python3
"""Build all 16 publication-grade figures (420pt) and validation manifests for Chapter 30."""
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
        'ch30-llm-rlvr-gpu-kernel-synthesis-closed-loop-architecture',
        '大模型驱动的 GPU 内核自动合成（Kevin / KernelBench 闭环）',
        'blue',
        [
            ('① 候选内核生成\n输出 CUDA/Triton\nNVCC 语法编译校验', 'blue'),
            ('② 数值等价门控\n多形状边界随机测\n无穷范数 <= eps', 'purple'),
            ('③ Nsight 剖析反馈\n回传硬件计数器\n8 轮 RLVR 自我进化', 'teal'),
        ],
        '单轮零样本生成（34.0% 正确）vs. 8 轮硬件剖析闭环（94.8% 正确）',
        'teal',
        [
            ('单轮盲目生成内核的瓶颈：\n容易出现共享内存越界与死锁\n超越 torch.compile 比例仅 18.0%', 'orange'),
            ('Profile-in-the-Loop 强化学习：\n将实测加速比与瓶颈作为奖励信号\n长尾融合算子较 Eager 提速 3.42x', 'teal'),
        ],
        '通过“编译校验—数值测试—硬件剖析”8 轮闭环实现 94.8% 正确率与 1.58x 超越',
    )

    draw_dual_band(
        ex,
        'ch30-nsight-compute-roofline-and-warp-stall-feedback-prompt',
        'Nsight Compute 硬件计数器结构化反馈与瓶颈靶向改写',
        'purple',
        [
            ('① 采集核心计数器\nSM 占用率与带宽\nBank 冲突与寄存器', 'blue'),
            ('② 自动诊断瓶颈类\n判定访存或延迟墙\n定位 Top1 停顿源', 'purple'),
            ('③ 靶向注入提示词\n指导向量化或填充\n下一轮精准修复', 'teal'),
        ],
        '只给标量运行耗时（盲人摸象）vs. 注入细粒度硬件性能计数器',
        'blue',
        [
            ('纯标量耗时反馈的探索低效性：\n智能体不知为何变慢，只能随机改\n陷入局部次优分块难以收敛', 'orange'),
            ('硬件白盒遥测引导精准手术：\n见 Bank 冲突即加 Padding 偏移\n见 MIO 停顿即改双缓冲流水', 'teal'),
        ],
        '把 Nsight 硬件遥测翻译为结构化提示词，让大模型像首席架构师般调优',
    )

    draw_dual_band(
        ex,
        'ch30-kernelbench-multi-turn-evolution-accuracy-and-speedup',
        'KernelBench 250 算子随进化轮次（T=1..8）的正确率与加速跃迁',
        'teal',
        [
            ('① 第 1 轮零样本\n编译正确率 34.0%\n仅 18% 超越编译器', 'orange'),
            ('② 第 4 轮中程进化\n修复语法与越界错\n正确率升至 82.4%', 'blue'),
            ('③ 第 8 轮收敛稳态\n正确率达 94.8%\n较编译器快 1.58x', 'teal'),
        ],
        '奖励函数设计：如何防范算子合成中的作弊行为（Reward Hacking）？',
        'purple',
        [
            ('常见内核奖励作弊陷阱：\n直接调原库函数、硬编码特定形状\n或忽略尾块边界换取虚假高分', 'orange'),
            ('AST 禁调检查与质数形状测试：\n严禁调用外部算子库并随机抽检\n确保每一分加速均来自真实内核', 'teal'),
        ],
        '8 轮迭代将正确率从 34.0% 推升至 94.8%，较 PyTorch Eager 提速 3.42 倍',
    )

    draw_dual_band(
        ex,
        'ch30-auto-discovered-register-fusion-vs-human-handwritten-kernel',
        '自动突破人类手写直觉：跨算子寄存器融合与非对称分块发现',
        'blue',
        [
            ('① 跨 4 算子全融合\nLN+RoPE+GQA+FP8\n省 38.4 GB/s 落盘', 'blue'),
            ('② 发现非对称分块\nBM=128, BN=192\n打破 2 的幂次教条', 'purple'),
            ('③ 精确卡满寄存器\n占用 192 个零溢出\nSM 占用率提 33%', 'teal'),
        ],
        '人类工程师受限于代码维护成本 vs. AI 探索非规则代数等价变换',
        'teal',
        [
            ('人类手写算子的工程边界：\n习惯采用 64/128/256 规整分块\n且难以手写跨四个算子的巨型内核', 'orange'),
            ('AI 无惧复杂度的极致特化：\n利用矩阵结合律交换计算与缩放序\n针对特定头维定制非对称最优块', 'teal'),
        ],
        '省去 38.4 GB/s 中间 HBM 读写并发现 BM=128, BN=192 非对称甜点',
    )

    draw_dual_band(
        ex,
        'ch30-ilp-mcts-5d-auto-parallelism-and-collective-tree-search',
        '5D 混合并行与自定义集合通信树的 ILP + MCTS 联合自动寻优',
        'purple',
        [
            ('① 算子级 ILP 建模\n切分 TP/PP/DP/EP/CP\n满足显存硬约束', 'blue'),
            ('② MCTS 搜通信树\n适配非对称拓扑图\n绕开拥塞慢速链路', 'purple'),
            ('③ 自动生成调度表\n端到端吞吐升 24%\n胜过人工专家启发', 'teal'),
        ],
        '手工固定 3D/4D 网格配置 vs. 异构拓扑感知的自动并行编译器',
        'blue',
        [
            ('手工配置在异构模型下的失效：\n视觉编码器、稠密层与稀疏 MoE 层\n最优并行切分策略截然不同', 'orange'),
            ('分层解耦双引擎搜索架构：\n外层动态规划切分阶段，内层 ILP\n结合 MCTS 定制跨层异步重叠流', 'teal'),
        ],
        '用整数线性规划与蒙特卡洛树搜索自动化解万卡 5D 并行组合爆炸难题',
    )

    draw_dual_band(
        ex,
        'ch30-alphachip-rl-macro-placement-and-wirelength-optimization',
        '芯片物理布局强化学习（AlphaChip）：宏单元布局与线长缩减',
        'teal',
        [
            ('① 顺序落子马尔可夫\n将 SRAM 宏视为棋子\n图神经网络编码网表', 'blue'),
            ('② 奖励函数多维约束\n最小化半周长线长\n消除布线拥塞热点', 'purple'),
            ('③ 跨代迁移预训练\n布局周期从月缩至天\n关键线长降 14.2%', 'teal'),
        ],
        'AI 辅助设计下一代 AI 加速器（TPU / 自研 ASIC）的自举正循环',
        'purple',
        [
            ('百亿晶体管芯片物理设计瓶颈：\n人类物理设计团队需数月手工微调\n宏单元互连线长直接限制主频上限', 'orange'),
            ('强化学习智能体越用越聪明：\n每设计一代芯片即积累先验经验\n为下一代算力底座赢得更高能效', 'teal'),
        ],
        '以深度强化学习自动布局百个 SRAM 与矩阵宏单元，互连线长缩减 14.2%',
    )

    draw_dual_band(
        ex,
        'ch30-per-kernel-dvfs-frequency-scaling-energy-delay-product',
        '算子级动态电压频率调节（Per-Kernel DVFS）三次方能效方程',
        'blue',
        [
            ('① 识别算子屋脊属性\nGEMM 属算力受限\nAllReduce 属访存受限', 'blue'),
            ('② 访存算子降频 18%\nHBM 带宽完全不变\n耗时仅增 0.72%', 'purple'),
            ('③ 三次方功耗骤降\n动态功耗降 26.5%\n显著优化能效积', 'teal'),
        ],
        '全程锁死最高 Boost 频率（高功耗发热）vs. 算子感知精细调频',
        'teal',
        [
            ('盲目锁死最高核心频率的浪费：\n跑集合通信或解码注意力时 SM 空等\n高电压高频白白转化成废热', 'orange'),
            ('微秒级硬件 PLL 调频闭环：\n在访存与通信阶段顺势降频 18%\n计算性能无损而功耗立降 26.5%', 'teal'),
        ],
        '利用 P_dyn 正比于 V^2*f 物理定律，降频 18% 换取 26.5% 动态功耗节省',
    )

    draw_dual_band(
        ex,
        'ch30-autonomous-aiops-straggler-and-pfc-root-cause-graph',
        '十万卡自治运维（Autonomous AIOps）：慢节点与 PFC 根因诊断图',
        'purple',
        [
            ('① 多源遥测流汇聚\nNCCL 环路延迟偏度\n光模块 SNR 与温度', 'blue'),
            ('② 时空因果图推理\n区分真慢卡与受害者\n锁定故障交换机口', 'purple'),
            ('③ 亚分钟自动处置\n42 秒隔离并热替换\n零人工介入自愈', 'teal'),
        ],
        '为什么 Ring AllReduce 中 1 张慢卡会让全环 32,768 张卡同时报警？',
        'blue',
        [
            ('集合通信故障传染与定位迷雾：\n单光模块误码或单口 PFC 风暴\n会导致整条通信环路全员超时阻塞', 'orange'),
            ('图差分偏度定位算法：\n比对通信域延迟张量的非对称拐点\n秒级揪出唯一物理始作俑者节点', 'teal'),
        ],
        '融合 NCCL 飞行记录仪与光模块遥测，在 32,768 卡中秒级锁定罪魁祸首',
    )

    draw_dual_band(
        ex,
        'ch30-sub-minute-fault-isolation-mttr-and-ettr-recovery',
        '自治 AIOps 收益账本：MTTR 压缩 360 倍与半年挽回 1,350 万美元',
        'teal',
        [
            ('① 人工排障 4.20h\n日志海捞与二分测\nETTR 跌至 88.4%', 'orange'),
            ('② 自治闭环 42.0s\n自动隔离+热备顶替\nMTTR 压缩 360 倍', 'blue'),
            ('③ ETTR 跃升 97.8%\n多抢回 9.4% 算力\n半年省 13.5M USD', 'teal'),
        ],
        '算例 30-4：32,768 卡集群隐性故障修复耗时对有效训练时间比的影响',
        'purple',
        [
            ('十万卡时代的阿姆达尔容错铁律：\n当平均无故障间隔缩短至数小时级\n4.2 小时人工排障将吞噬逾一成算力', 'orange'),
            ('被动救火转向亚分钟免疫：\n42 秒完成故障驱逐与快照接管\n将集群有效训练比推上 97.8%', 'teal'),
        ],
        '将隐性故障 MTTR 从 4.2 小时压至 42 秒，半年直接挽回 1,350 万美元算力',
    )

    draw_dual_band(
        ex,
        'ch30-datacenter-digital-twin-workload-network-thermal-cosim',
        '智算中心数字孪生（Digital Twin）：负载—网络—热力学三域联合仿真',
        'blue',
        [
            ('① 计算图轨迹注入\n提取五维并行算子\n与集合通信依赖图', 'blue'),
            ('② 包级网络仿真器\n模拟胖树 ECMP 哈希\n检出 21.4% 拥塞降速', 'purple'),
            ('③ 液冷热力学仿真\n推演机柜进出水温度\n上线前清零架构隐患', 'teal'),
        ],
        '斥资百亿盲目建群试错 vs. 数字孪生零成本预演十万卡架构（算例 30-5）',
        'teal',
        [
            ('物理集群上线后的改造代价：\n布线拓扑不合理或局部超额认购\n建成后整改需停机数周耗资巨大', 'orange'),
            ('软件定义的高保真预演沙盒：\n提前发现 2:1 超额认购致 MoE 降速\n并优化任务放置消除机房局部热点', 'teal'),
        ],
        '在购置第一根光纤前，用三域耦合数字孪生预演并根除 21.4% 潜在性能劣化',
    )

    draw_dual_band(
        ex,
        'ch30-thirty-chapter-five-part-encyclopedia-master-atlas',
        '《AI 基础设施系统工程》30 章皇冠全栈终极百科版：五大架构部类',
        'purple',
        [
            ('第一~二部(Ch01-08)\n硬件微架构与高速互连\n分布式并行训练工程', 'blue'),
            ('第三~四部(Ch09-24)\n吞吐推理与生产运维\n多模态/智能体/科学', 'purple'),
            ('第五部类(Ch25-30)\n推荐/dLLM/晶圆芯片\n主权/极客栈/自演进', 'teal'),
        ],
        '全书 30 章核心交付物全景数字资产总账本（算例 30-6）',
        'blue',
        [
            ('手稿与视觉工程规模：\n43,548+ 行正文 | 979+1=980 幅配图\n2,940 个出版级 SVG/PNG/PDF 文件', 'blue'),
            ('代码、教学与可复现实验规模：\n201+1=202 个代码块 | 228 个算例\n341 张表 | 492 块习题 | 165 个实验', 'teal'),
        ],
        '三十章五大部类浑然一体，铸就从硅片物理到自演进系统的终极工程百科',
    )

    draw_dual_band(
        ex,
        'ch30-pillar-1-to-3-hardware-networking-training-inference-map',
        '第一至第三部类（第 1–12 章）核心知识图谱与定量性能锚点',
        'teal',
        [
            ('第 1–4 章：单机算力\nGPU/TPU微架构定律\nRoofline与CXL内存池', 'blue'),
            ('第 5–8 章：通信与训练\nNVL72/576 与 RoCEv2\n5D 并行、FP8 与容错', 'purple'),
            ('第 9–12 章：推理与云\n分页注意力与P-D分离\nK8s 调度、观测与端侧', 'teal'),
        ],
        '支撑现代万亿参数大模型的三大基础底座（算力、互连、训练与服务）',
        'purple',
        [
            ('向上承托全场景模型演进：\n将单卡微架构极限转化为万卡扩展\n消除显存墙、通信墙与长尾延迟墙', 'blue'),
            ('向下扎根第一性原理核算：\n每章均配备零依赖可运行实验套件\n从字节与时钟周期推导系统最优解', 'teal'),
        ],
        '夯实“硬件微架构—超节点互连—分布式训练—高吞吐推理”十二章主干底座',
    )

    draw_dual_band(
        ex,
        'ch30-pillar-4-to-5-multimodal-agent-science-frontier-map',
        '第四至第五部类（第 13–30 章）全场景扩展与前沿架构疆域图谱',
        'blue',
        [
            ('第13–18章：架构演进\n长序列/MoE/绿色智算\n机密/多模态/RLVR', 'blue'),
            ('第19–24章：行业疆域\n具身智驾/基准/Agent\n实时语音/数据栈/AI4S', 'purple'),
            ('第25–30章：极客前沿\n推荐/dLLM/晶圆芯片\n主权/极客栈/自演进', 'teal'),
        ],
        '从大语言模型走向全模态、全行业、去中心化与自主演化的广阔天地',
        'teal',
        [
            ('打破“AI Infra 仅限 LLM”局限：\n全量覆盖搜广推、视频模型与智驾\n核聚变控制、高频金融与科学智能', 'blue'),
            ('深潜极客开源真经与未来范式：\n完整拆解 DeepSeek 七大开源引擎\n并收束于 AI 自我演化基础设施闭环', 'teal'),
        ],
        '十八章纵深拓展，全面囊括当代与下一代 AI 基础设施的每一个关键战场',
    )

    draw_dual_band(
        ex,
        'ch30-four-fundamental-conservation-laws-across-30-chapters',
        '统治全书 30 章的四大普适物理守恒律（上：算术强度与状态局部性）',
        'purple',
        [
            ('定律一：算术强度守恒\nI = FLOPs / Bytes\n破局：融合/分块/低比特\n对应：FlashMLA/HSTU/dLLM', 'blue'),
            ('定律二：状态局部性守恒\n最小化跨层级搬运熵\n破局：片上驻留/原位算\n对应：WSE-3/KTransformers', 'purple'),
        ],
        '统治全书 30 章的四大普适物理守恒律（下：关键路径与能量转换）',
        'teal',
        [
            ('定律三：关键路径掩盖律\nT_exp = max(0, T_cm-T_cp)\n破局：双向流水/伴飞重叠\n对应：DualPipe/DiLoCo/护栏', 'blue'),
            ('定律四：信息—能量转换律\n最小化每有效词元焦耳\n破局：三值无乘/算子DVFS\n对应：BitNet/液冷微网/AIOps', 'teal'),
        ],
        '万变不离其宗：一切架构创新皆是对四大底层物理守恒律的精妙求解',
    )

    draw_dual_band(
        ex,
        'ch30-ten-year-ai-infrastructure-roadmap-2026-to-2035',
        '2026–2035 未来十年 AI 基础设施演进路线图：硬件与互连跃迁',
        'teal',
        [
            ('2026–2028：近期\nRubin NVFP4与HBM4\nCPO 共封装光学商用\n800V 高压直流园区', 'blue'),
            ('2029–2031：中期\n3D 晶圆级硅光超节点\n片内微流体冷却普及\n跨洲际亿卡联邦网络', 'purple'),
            ('2032–2035：远期\n光电混合与存算一体\n零人工自演进底座\n行星级绿色智能电网', 'teal'),
        ],
        '从“人工堆叠算力”向“物理极限压榨与智能体自主设计”的代际跨越',
        'purple',
        [
            ('硬件物理层的三大突破方向：\n用硅光互连打破铜缆距离损耗墙\n用 3D 堆叠与晶圆级集成打破访存墙', 'blue'),
            ('软件系统层的自治形态：\n大模型接管芯片布局与算子合成\n并行编译与十万卡亚秒级自愈', 'teal'),
        ],
        '洞察未来十年技术坐标，在剧烈演进的浪潮中把握不朽的系统第一性原理',
    )

    draw_dual_band(
        ex,
        'ch30-grand-finale-ai-infrastructure-system-engineering-crown',
        '第 30 章暨全书终极收束：AI 基础设施系统工程的皇冠闭环',
        'blue',
        [
            ('① 人类奠定第一性原理\n构建数学模型与守恒律\n打造初代算力与系统栈', 'blue'),
            ('② 系统孕育超级智能\n支撑万亿参数基座训练\n涌现深度代码与推理力', 'purple'),
            ('③ 智能反哺进化底座\n自动合成内核与芯片\n开启螺旋上升自举纪元', 'teal'),
        ],
        '致敬每一位在硅片、光纤、内核与分布式战壕中锻造智能底座的工程师',
        'teal',
        [
            ('全书三十章的工程信仰：\n拒绝黑盒调参与经验主义玄学\n坚持用公式、代码与实测数据说话', 'blue'),
            ('迈向自演进 AI 基础设施新纪元：\n当 AI 开始设计更强大的 AI 底座\n系统工程将成为撬动奇点的支点', 'teal'),
        ],
        '全书收束：从第一性原理出发，跨越三十章全栈疆域，抵达自演进智能未来',
    )

    outputs = ex.finish()

    fig_index = [
        {
            "figure_id": "图 30-1",
            "number": 1,
            "file": "ch30-llm-rlvr-gpu-kernel-synthesis-closed-loop-architecture.svg",
            "asset": "ch30/ch30-llm-rlvr-gpu-kernel-synthesis-closed-loop-architecture.svg",
            "section": "30.1.1",
            "caption": "大模型与强化学习驱动的 GPU 内核自动合成闭环（Kevin / KernelBench）：融合编译校验、数值等价门控与 Nsight Compute 硬件剖析反馈的 8 轮自我进化架构（算例 30-1）",
            "source_refs": ["experiments/ch30/30-01/run.py", "manuscripts/ch30/figure-data.json"],
            "TeachingRole": "揭示如何将硬件性能剖析器接入强化学习验证奖励回路，实现算子自动进化"
        },
        {
            "figure_id": "图 30-2",
            "number": 2,
            "file": "ch30-nsight-compute-roofline-and-warp-stall-feedback-prompt.svg",
            "asset": "ch30/ch30-nsight-compute-roofline-and-warp-stall-feedback-prompt.svg",
            "section": "30.1.1",
            "caption": "Nsight Compute 硬件计数器（SM 占用率、Shared Memory Bank 冲突、DRAM 带宽与 Warp Stall 原因）转化为结构化诊断提示词的反馈机制",
            "source_refs": ["experiments/ch30/30-01/run.py", "manuscripts/ch30/figure-data.json"],
            "TeachingRole": "阐明为何白盒硬件瓶颈分类反馈远比单一标量耗时更能引导大模型跳出局部次优"
        },
        {
            "figure_id": "图 30-3",
            "number": 3,
            "file": "ch30-kernelbench-multi-turn-evolution-accuracy-and-speedup.svg",
            "asset": "ch30/ch30-kernelbench-multi-turn-evolution-accuracy-and-speedup.svg",
            "section": "30.1.1",
            "caption": "KernelBench 250 个算子基准在 8 轮硬件剖析反馈下的进化轨迹：编译与数值正确率从 34.0% 跃升至 94.8%，平均较 PyTorch Eager 提速 3.42 倍、较 torch.compile 提速 1.58 倍（算例 30-1）",
            "source_refs": ["experiments/ch30/30-01/run.py", "manuscripts/ch30/figure-data.json"],
            "TeachingRole": "用实测收敛曲线量化多轮硬件反馈对消除算子编译错误与压榨硬件潜能的决定性作用"
        },
        {
            "figure_id": "图 30-4",
            "number": 4,
            "file": "ch30-auto-discovered-register-fusion-vs-human-handwritten-kernel.svg",
            "asset": "ch30/ch30-auto-discovered-register-fusion-vs-human-handwritten-kernel.svg",
            "section": "30.1.2",
            "caption": "AI 自动合成内核突破人类手写直觉：跨四算子寄存器级持久化融合（省去 38.4 GB/s 中间激活落盘）与非对称分块（BM=128, BN=192）自主发现（算例 30-2）",
            "source_refs": ["experiments/ch30/30-01/run.py", "manuscripts/ch30/figure-data.json"],
            "TeachingRole": "展示 AI 如何通过非2幂次分块与代数重结合找到人类工程师极少涉足的性能甜点"
        },
        {
            "figure_id": "图 30-5",
            "number": 5,
            "file": "ch30-ilp-mcts-5d-auto-parallelism-and-collective-tree-search.svg",
            "asset": "ch30/ch30-ilp-mcts-5d-auto-parallelism-and-collective-tree-search.svg",
            "section": "30.2.1",
            "caption": "基于整数线性规划与蒙特卡洛树搜索（ILP + MCTS）的 5D 混合并行切分与非对称拓扑自定义集合通信树联合寻优器",
            "source_refs": ["experiments/ch30/30-02/run.py", "manuscripts/ch30/figure-data.json"],
            "TeachingRole": "展示如何将算子级并行策略搜索与底层物理网络拓扑路由联合建模"
        },
        {
            "figure_id": "图 30-6",
            "number": 6,
            "file": "ch30-alphachip-rl-macro-placement-and-wirelength-optimization.svg",
            "asset": "ch30/ch30-alphachip-rl-macro-placement-and-wirelength-optimization.svg",
            "section": "30.2.2",
            "caption": "芯片物理布局强化学习（AlphaChip）：基于边特征图神经网络（Edge-GNN）的 SRAM 与脉动阵列宏单元顺序落子策略，互连线长缩减 14.2%",
            "source_refs": ["experiments/ch30/30-02/run.py", "manuscripts/ch30/figure-data.json"],
            "TeachingRole": "剖析 AI 如何直接参与下一代 TPU 与 AI 芯片的物理版图设计并形成自举闭环"
        },
        {
            "figure_id": "图 30-7",
            "number": 7,
            "file": "ch30-per-kernel-dvfs-frequency-scaling-energy-delay-product.svg",
            "asset": "ch30/ch30-per-kernel-dvfs-frequency-scaling-energy-delay-product.svg",
            "section": "30.2.2",
            "caption": "算子级动态电压频率调节（Per-Kernel DVFS）：在访存/通信受限算子自动下调核心频率 18%，耗时仅增 0.72% 而动态功耗立降 26.5%（算例 30-3）",
            "source_refs": ["experiments/ch30/30-02/run.py", "manuscripts/ch30/figure-data.json"],
            "TeachingRole": "运用 CMOS 动态功耗三次方物理定律推导算子粒度调频的显著能效收益"
        },
        {
            "figure_id": "图 30-8",
            "number": 8,
            "file": "ch30-autonomous-aiops-straggler-and-pfc-root-cause-graph.svg",
            "asset": "ch30/ch30-autonomous-aiops-straggler-and-pfc-root-cause-graph.svg",
            "section": "30.3.1",
            "caption": "十万卡智算中心自治运维（Autonomous AIOps）：融合 NCCL 环路延迟偏度与光模块信噪比斜率的时空因果图慢节点根因诊断（RCA）引擎",
            "source_refs": ["experiments/ch30/30-02/run.py", "manuscripts/ch30/figure-data.json"],
            "TeachingRole": "解决集合通信“一卡降速、全环陪绑”假象下的亚分钟级精准故障定位难题"
        },
        {
            "figure_id": "图 30-9",
            "number": 9,
            "file": "ch30-sub-minute-fault-isolation-mttr-and-ettr-recovery.svg",
            "asset": "ch30/ch30-sub-minute-fault-isolation-mttr-and-ettr-recovery.svg",
            "section": "30.3.1",
            "caption": "32,768 卡集群自治故障自愈账本：将隐性故障 MTTR 从人工 4.20 小时压缩 360 倍至 42.0 秒，集群 ETTR 从 88.4% 跃升至 97.8%，半年挽回 1,350 万美元算力（算例 30-4）",
            "source_refs": ["experiments/ch30/30-02/run.py", "manuscripts/ch30/figure-data.json"],
            "TeachingRole": "用硬核财务与可用性公式证明自治 AIOps 在超大规模集群中的经济价值"
        },
        {
            "figure_id": "图 30-10",
            "number": 10,
            "file": "ch30-datacenter-digital-twin-workload-network-thermal-cosim.svg",
            "asset": "ch30/ch30-datacenter-digital-twin-workload-network-thermal-cosim.svg",
            "section": "30.3.2",
            "caption": "数据中心级高保真数字孪生仿真器（Workload-Network-Thermal Co-Simulator）：上线前提前识别 2:1 超额认购导致的 21.4% MoE 通信降速与局部热岛（算例 30-5）",
            "source_refs": ["experiments/ch30/30-02/run.py", "manuscripts/ch30/figure-data.json"],
            "TeachingRole": "展示如何通过离散事件网络仿真与热力学耦合在建群前消除百亿级投资架构隐患"
        },
        {
            "figure_id": "图 30-11",
            "number": 11,
            "file": "ch30-thirty-chapter-five-part-encyclopedia-master-atlas.svg",
            "asset": "ch30/ch30-thirty-chapter-five-part-encyclopedia-master-atlas.svg",
            "section": "30.4.1",
            "caption": "《AI 基础设施系统工程》30 章皇冠全栈终极百科版全景总图：五大架构部类、43,548+ 行手稿、979+1=980 幅配图、201+1=202 个代码块、228 个算例、492 块习题与 165 个实验总账本（算例 30-6）",
            "source_refs": ["experiments/ch30/30-03/run.py", "manuscripts/ch30/figure-data.json"],
            "TeachingRole": "全景呈现全书 30 章五大部类的结构版图与机器可校验的数字资产账本"
        },
        {
            "figure_id": "图 30-12",
            "number": 12,
            "file": "ch30-pillar-1-to-3-hardware-networking-training-inference-map.svg",
            "asset": "ch30/ch30-pillar-1-to-3-hardware-networking-training-inference-map.svg",
            "section": "30.4.1",
            "caption": "第一至第三部类（第 1–12 章：硬件与集群基础设施、分布式训练系统工程、推理服务与生产运维）核心技术链路与跨章因果映射图",
            "source_refs": ["experiments/ch30/30-03/run.py", "manuscripts/ch30/figure-data.json"],
            "TeachingRole": "串联前十二章主干底座，梳理从单芯片微架构到万卡训练与高吞吐推理的演进脉络"
        },
        {
            "figure_id": "图 30-13",
            "number": 13,
            "file": "ch30-pillar-4-to-5-multimodal-agent-science-frontier-map.svg",
            "asset": "ch30/ch30-pillar-4-to-5-multimodal-agent-science-frontier-map.svg",
            "section": "30.4.1",
            "caption": "第四至第五部类（第 13–30 章：前沿架构演进与全场景系统工程、新一代范式与DeepSeek及自演进闭环）十八章专题全景映射图",
            "source_refs": ["experiments/ch30/30-03/run.py", "manuscripts/ch30/figure-data.json"],
            "TeachingRole": "串联后十八章前沿疆域，展示基础架构原理在多模态、智驾、科学计算与极客系统中的延展"
        },
        {
            "figure_id": "图 30-14",
            "number": 14,
            "file": "ch30-four-fundamental-conservation-laws-across-30-chapters.svg",
            "asset": "ch30/ch30-four-fundamental-conservation-laws-across-30-chapters.svg",
            "section": "30.4.2",
            "caption": "统治 AI 基础设施的四大普适物理守恒律（算术强度守恒、状态局部性守恒、关键路径掩盖律、信息—能量转换效率律）及其在 30 章中的具体化身",
            "source_refs": ["experiments/ch30/30-03/run.py", "manuscripts/ch30/figure-data.json"],
            "TeachingRole": "将全书数百项工程技术在第一性原理层面高度凝练为四条普适守恒方程"
        },
        {
            "figure_id": "图 30-15",
            "number": 15,
            "file": "ch30-ten-year-ai-infrastructure-roadmap-2026-to-2035.svg",
            "asset": "ch30/ch30-ten-year-ai-infrastructure-roadmap-2026-to-2035.svg",
            "section": "30.4.2",
            "caption": "2026–2035 未来十年 AI 基础设施演进路线预测图：从 Rubin/CPO 光互连、3D 晶圆级与片内微流体冷却到行星级自演进智能底座",
            "source_refs": ["experiments/ch30/30-03/run.py", "manuscripts/ch30/figure-data.json"],
            "TeachingRole": "提供未来十年软硬件协同演进的坐标轴，帮助读者建立超越具体硬件代际的远见"
        },
        {
            "figure_id": "图 30-16",
            "number": 16,
            "file": "ch30-grand-finale-ai-infrastructure-system-engineering-crown.svg",
            "asset": "ch30/ch30-grand-finale-ai-infrastructure-system-engineering-crown.svg",
            "section": "30.4.2",
            "caption": "全书终极收束图：从人类工程第一性原理筑造 AI 基础设施，到大模型智能体反哺演化下一代芯片、算子与集群的自举上升闭环",
            "source_refs": ["experiments/ch30/30-03/run.py", "manuscripts/ch30/figure-data.json"],
            "TeachingRole": "完成全书 30 章的思想升华，定格“系统孕育智能、智能演进系统”的宏大闭环"
        }
    ]
    for row in fig_index:
        row['figure'] = f"30-{row['number']}"
        row['name'] = Path(row['file']).stem
    (HERE / 'figure-index.json').write_text(json.dumps(fig_index, ensure_ascii=False, indent=2) + '\n')

    md = HERE.parent / '30-自演进 AI 基础设施（AI for AI Infra）：大模型驱动的算子自动合成、强化学习编译寻优与三十章全栈终极收束.md'
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

        work = re.sub(r'!\[([^\]]*)\]\((ch30/[^)]+)\)', repl_img, text_md)
        body = markdown.markdown(work, extensions=['tables', 'fenced_code', 'footnotes', 'toc'])
        html = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>第 30 章：自演进 AI 基础设施（AI for AI Infra）：大模型驱动的算子自动合成、强化学习编译寻优与三十章全栈终极收束</title></head><body>{body}</body></html>\n'
        out_html = preview_path(md)
        out_html.write_text(html)

    sources = [
        HERE / 'build.py',
        HERE / 'figure-data.json',
        HERE / 'figure-index.json',
        ROOT / 'experiments/ch30/30-01/run.py',
        ROOT / 'experiments/ch30/30-02/run.py',
        ROOT / 'experiments/ch30/30-03/run.py',
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
    print(f'Built {len(outputs)} figure files for Chapter 30; 0 layout warnings.')


if __name__ == '__main__':
    build_chapter()
