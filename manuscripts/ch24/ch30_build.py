#!/usr/bin/env python3
"""Build publication-grade figures (420pt) for Chapter 24 (Sections 24.5–24.8)."""
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


def build_part2(ex: Exporter) -> None:
    draw_dual_band(
        ex,
        'ch24-llm-rlvr-gpu-kernel-synthesis-closed-loop-architecture',
        '大模型驱动的 GPU 内核自动合成（Kevin / KernelBench 多轮反馈）',
        'blue',
        [
            ('① 候选内核生成\n输出 CUDA/Triton\nNVCC 语法编译校验', 'blue'),
            ('② 数值等价门控\n多形状边界随机测\n无穷范数 <= eps', 'purple'),
            ('③ Nsight 剖析反馈\n回传硬件计数器\n8 轮 RLVR 自我进化', 'teal'),
        ],
        '单轮零样本生成（34.0% 正确）vs. 8 轮硬件剖析迭代（94.8% 正确）',
        'teal',
        [
            ('单轮盲目生成内核的瓶颈：\n易出现共享内存越界与同步停滞\n超越 torch.compile 比例仅 18.0%', 'orange'),
            ('Profile-in-the-Loop 强化学习：\n将实测加速比与瓶颈作为奖励信号\n长尾融合算子较 Eager 提速 3.42x', 'teal'),
        ],
        '经“编译校验—数值测试—硬件剖析”8 轮迭代实现 94.8% 正确率与 1.58x 超越',
    )

    draw_dual_band(
        ex,
        'ch24-nsight-compute-roofline-and-warp-stall-feedback-prompt',
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
        'ch24-kernelbench-multi-turn-evolution-accuracy-and-speedup',
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
        'ch24-auto-discovered-register-fusion-vs-human-handwritten-kernel',
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
        'ch24-ilp-mcts-5d-auto-parallelism-and-collective-tree-search',
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
        'ch24-alphachip-rl-macro-placement-and-wirelength-optimization',
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
        'ch24-per-kernel-dvfs-frequency-scaling-energy-delay-product',
        '算子级动态电压频率调节（Per-Kernel DVFS）三次方能效方程',
        'blue',
        [
            ('① 识别算子屋脊属性\nGEMM 属算力受限\nAllReduce 属访存受限', 'blue'),
            ('② 访存算子降频 18%\nHBM 带宽完全不变\n耗时仅增 0.72%', 'purple'),
            ('③ 三次方功耗下降\n动态功耗降 26.5%\n显著优化能效积', 'teal'),
        ],
        '全程固定最高 Boost 频率（高功耗发热）vs. 算子感知精细调频',
        'teal',
        [
            ('盲目固定最高核心频率的浪费：\n跑集合通信或解码注意力时 SM 空等\n高电压高频白白转化成废热', 'orange'),
            ('微秒级硬件 PLL 动态调频：\n在访存与通信阶段顺势降频 18%\n计算性能无损而功耗立降 26.5%', 'teal'),
        ],
        '利用 P_dyn 正比于 V^2*f 物理定律，降频 18% 换取 26.5% 动态功耗节省',
    )

    draw_dual_band(
        ex,
        'ch24-autonomous-aiops-straggler-and-pfc-root-cause-graph',
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
            ('图差分偏度定位算法：\n比对通信域延迟张量的非对称拐点\n秒级定位唯一物理始作俑者节点', 'teal'),
        ],
        '融合 NCCL 飞行记录仪与光模块遥测，在 32,768 卡中秒级定位根因节点',
    )

    draw_dual_band(
        ex,
        'ch24-sub-minute-fault-isolation-mttr-and-ettr-recovery',
        '自治 AIOps 收益账本：MTTR 压缩 360 倍与半年挽回 1,350 万美元',
        'teal',
        [
            ('① 人工排障 4.20h\n日志海捞与二分测\nETTR 跌至 88.4%', 'orange'),
            ('② 自治自愈 42.0s\n自动隔离+热备顶替\nMTTR 压缩 360 倍', 'blue'),
            ('③ ETTR 跃升 97.8%\n多抢回 9.4% 算力\n半年省 13.5M USD', 'teal'),
        ],
        '算例 24-10：32,768 卡集群隐性故障修复耗时对有效训练时间比的影响',
        'purple',
        [
            ('十万卡时代的阿姆达尔容错铁律：\n当平均无故障间隔缩短至数小时级\n4.2 小时人工排障将吞噬逾一成算力', 'orange'),
            ('被动救火转向亚分钟免疫：\n42 秒完成故障驱逐与快照接管\n将集群有效训练比推上 97.8%', 'teal'),
        ],
        '将隐性故障 MTTR 从 4.2 小时压至 42 秒，半年直接挽回 1,350 万美元算力',
    )

    draw_dual_band(
        ex,
        'ch24-datacenter-digital-twin-workload-network-thermal-cosim',
        '智算中心数字孪生（Digital Twin）：负载—网络—热力学三域联合仿真',
        'blue',
        [
            ('① 计算图轨迹注入\n提取五维并行算子\n与集合通信依赖图', 'blue'),
            ('② 包级网络仿真器\n模拟胖树 ECMP 哈希\n检出 21.4% 拥塞降速', 'purple'),
            ('③ 液冷热力学仿真\n推演机柜进出水温度\n上线前清零架构隐患', 'teal'),
        ],
        '百亿盲目建群试错 vs. 数字孪生零成本预演十万卡架构（算例 24-11）',
        'teal',
        [
            ('物理集群上线后的改造代价：\n布线拓扑不合理或局部超额认购\n建成后整改需停机数周耗资巨大', 'orange'),
            ('软件定义的高保真预演沙盒：\n提前发现 2:1 超额认购致 MoE 降速\n并优化任务放置消除机房局部热点', 'teal'),
        ],
        '在购置第一根光纤前，用三域耦合数字孪生预演并根除 21.4% 潜在性能劣化',
    )

    draw_dual_band(
        ex,
        'ch24-hundred-mw-ai-factory-five-part-master-atlas',
        '100 MW 智算工厂全栈容量切分：从变电站红线到 35,712 张 GPU',
        'purple',
        [
            ('① 园区电热切分\n100 MW / PUE 1.12\n89.29 MW IT 预算', 'blue'),
            ('② 496 台 72 卡柜\n81.84 MW 超节点池\n7.45 MW 光网+3FS', 'purple'),
            ('③ 三池精确配比\n400 训练 + 82 推理\n14 备柜 (97.6% 有效)', 'teal'),
        ],
        '算例 24-12：双轨预训练工期与三年期单位词元成本联立账本',
        'blue',
        [
            ('持续有效算力 27.81 EFLOPS：\n10T/380B (60T) 预训 62.4d\n叠 18.5d RLVR 共 80.9d 交付', 'blue'),
            ('三年总成本 2,030.82M USD：\n日产 216.5B (1.41 USD/1M)\n潮汐共享 RL 后净 0.61 USD/1M', 'teal'),
        ],
        '五大部类联立闭合：在 100 MW 与季度窗口内兑现 0.61 USD/1M 生产力',
    )

    draw_dual_band(
        ex,
        'ch24-pillar-1-to-3-hardware-networking-training-inference-map',
        '第一至第三部分（第 1–12 章）跨层耦合方程与百兆瓦工厂参数锚点',
        'teal',
        [
            ('第一部分 (Ch01-03)\n模型缩放与状态体积\n10T/380B 与 MLA 账本', 'blue'),
            ('第二部分 (Ch04-07)\n微架构算存比与光网\n72 卡超节点与 UEC', 'purple'),
            ('第三部分 (Ch08-12)\n量化解耦与云边协同\nFP8/NVFP4 与 PD 池', 'teal'),
        ],
        '支撑 27.81 EFLOPS 训练与 216.5B 日产词元的三大核心工程底座',
        'purple',
        [
            ('向上决定单卡算力与显存效率：\nQuaRot 压缩与算子融合流水\n将单卡持续算力稳在 970 TFLOPS', 'blue'),
            ('向下决定跨柜扩展与成本摊销：\nDualPipe 重叠与昼夜弹性混部\n将三年期全池摊薄至 2.16 USD/卡时', 'teal'),
        ],
        '贯通第 1–12 章：从架构缩放律、超节点互连到训推一体资源调度',
    )

    draw_dual_band(
        ex,
        'ch24-pillar-4-to-5-multimodal-agent-science-frontier-map',
        '第四至第五部分（第 13–24 章）集群底座与自演进应用全栈映射',
        'blue',
        [
            ('第四部分 (Ch13-17)\n3FS/800V 液冷/容错\nTEE 安全与异步 RLVR', 'blue'),
            ('第五部分上 (Ch18-22)\nAgent/多模态/具身\n生成式搜推与科学智能', 'purple'),
            ('第五部分下 (Ch23-24)\n后自回归与片上 SRAM\n开源全栈与自治演进', 'teal'),
        ],
        '从百 PB 语料供给、温水液冷微网到多模态智能体与内核自演进',
        'teal',
        [
            ('物理与数据底座刚性护航：\n800V+W45 液冷守住 PUE 1.12\n3FS 与亚分钟自愈稳住 97.6%', 'blue'),
            ('后训练与前沿负载价值变现：\n波谷闲置算力反哺 18.5d RLVR\nSRAM 草稿与 HBM 主池协同', 'teal'),
        ],
        '贯通第 13–24 章：连接物理能源、安全容错、垂直场景与自演进软件栈',
    )

    draw_dual_band(
        ex,
        'ch24-four-fundamental-conservation-laws-across-24-chapters',
        '贯穿全书二十四章的四大普适物理守恒律（上：算术强度与状态局部性）',
        'purple',
        [
            ('定律一：算术强度守恒\nI = FLOPs/Bytes 融合分块\nFlashMLA / HSTU / dLLM', 'blue'),
            ('定律二：状态局部性守恒\n最小化跨层级状态搬运量\nWSE-3 / KTransformers', 'purple'),
        ],
        '贯穿全书二十四章的四大普适物理守恒律（下：关键路径与能量转换）',
        'teal',
        [
            ('定律三：关键路径掩盖律\nT_exp = max(0, T_cm-T_cp)\nDualPipe / DiLoCo / 护栏', 'blue'),
            ('定律四：信息—能量转换律\n电网每焦耳 377.89 GFLOP\nBitNet / 温水液冷 / DVFS', 'teal'),
        ],
        '万变不离其宗：百兆瓦工厂的每一层效率皆源于四大底层物理守恒律',
    )

    draw_dual_band(
        ex,
        'ch24-ten-year-ai-infrastructure-roadmap-2026-to-2035',
        '2026–2035 未来十年 AI 基础设施演进路线图：硬件与互连跃迁',
        'teal',
        [
            ('2026–2028：近期\nRubin NVFP4与HBM4\nCPO 共封装光学商用\n800V 高压直流园区', 'blue'),
            ('2029–2031：中期\n3D 晶圆级硅光超节点\n片内微流体冷却普及\n跨区域广域协同训练网', 'purple'),
            ('2032–2035：远期\n光电混合与存算一体\n零人工自演进软件栈\n吉瓦级绿色智算微电网', 'teal'),
        ],
        '从“人工堆叠算力”向“物理效率演进与智能体自主设计”的代际跨越',
        'purple',
        [
            ('硬件物理层的三大突破方向：\n用硅光互连突破铜缆距离损耗墙\n用 3D 堆叠与晶圆级集成跨越访存墙', 'blue'),
            ('软件系统层的自治形态：\n大模型接管芯片布局与算子合成\n并行编译与十万卡亚分钟级自愈', 'teal'),
        ],
        '洞察未来十年技术坐标，在快速演进的浪潮中把握稳固的系统第一性原理',
    )

    draw_dual_band(
        ex,
        'ch24-grand-finale-ai-infrastructure-system-engineering-crown',
        '《深入理解 AI Infra：量化分析与系统设计》二十四章全栈演进螺旋',
        'blue',
        [
            ('① 人类奠定第一性原理\n建立四资源与守恒方程\n打造超节点并行软件栈', 'blue'),
            ('② 系统孕育前沿大模型\n百兆瓦工厂训推一体池\n涌现代码与长链条推理', 'purple'),
            ('③ 智能反哺基础设施栈\n自动合成内核与布局图\n开启软硬正向演进周期', 'teal'),
        ],
        '致敬每一位在硅片、光纤、内核与集群一线锻造智能底座的系统工程师',
        'teal',
        [
            ('全书二十四章的工程准则：\n拒绝黑盒调参与经验主义猜测\n坚持用公式、代码与实测数据说话', 'blue'),
            ('迈向自演进 AI 基础设施时代：\n当 AI 开始协同设计下一代算力栈\n定量系统工程始终是稳固的支点', 'teal'),
        ],
        '全书收束：从第一性原理出发，跨越二十四章五大部类，迈向自演进系统',
    )


def build_chapter() -> None:
    ex = Exporter(HERE)
    build_part2(ex)
    outputs = ex.finish()
    print("Exported", len(outputs), "from b30")

if __name__ == "__main__":
    build_chapter()
