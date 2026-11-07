#!/usr/bin/env python3
"""Build Chapter 19 figures, locked calculations, and offline HTML preview."""
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


def fig_19_1_cloud_vs_embodied(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, '云端对话（软实时 SLO）', color='blue', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, 'TTFT≤500 ms，TPOT≤50 ms\n偶发抖动仅影响打字体验', 'blue', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '数字环境可重放、可重试\n输出错误可由沙箱纠正', 'teal', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '目标：最大化批处理吞吐\n允许微批排队 20–100 ms', 'gray', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '软实时容忍百毫秒尾部延迟', size=11, ha='center', weight='bold', color=BORDER_COL['blue'])

    card(ax, 0.52, 0.04, 0.46, 0.92, '具身控制（硬实时截止期）', color='orange', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '控制周期 5–20 ms（100 Hz）\n抖动 < 1 ms，单样本 B=1', 'orange', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '盲飞位移 Δx = v·Δt（19-1）\n0.5 m/s × 200 ms = 100 mm', 'orange', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '插孔公差 5 mm → Δt ≤ 10 ms\n物理碰撞不可回滚', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '时间延迟直接化为空间碰撞误差', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])
    ex.save(fig, 'ch19-cloud-llm-slo-vs-embodied-hard-realtime')


def fig_19_2_vlalagbench(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'VLALagBench：推理延迟对静态与动态操作成功率的冲击', color='orange')
    stages = [
        (0.03, 0.58, 0.29, 0.24, 'Δt_inf = 10 ms\n盲飞漂移 5 mm\n动态成功率 88%', 'teal'),
        (0.355, 0.58, 0.29, 0.24, 'Δt_inf = 100 ms\n盲飞漂移 50 mm\n成功率跌至 56%', 'orange'),
        (0.68, 0.58, 0.29, 0.24, 'Δt_inf = 200 ms\n盲飞漂移 100 mm\n成功率崩至 18%', 'orange'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '为什么静态抓取能容忍延迟而动态追踪在 100 ms 后崩溃？', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, '静态桌面抓取（v_obj = 0）\n延迟仅拉长任务总耗时\n成功率保持 > 75% 缓降', 'teal', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '动态接触操作（状态过时）\n动作基于 Δt 前旧图像算出\n闭合时目标已移出接触区', 'orange', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '观测过时是单环重型 VLA 无法用于高频动态控制的主因', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch19-vlalagbench-latency-vs-dynamic-grasp-success')


def fig_19_3_action_chunking(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '动作块预测（Action Chunking, H=16）与时间集成平滑', color='teal')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '① 一次吐 H=16 步\na_t ... a_{t+15}\n分摊单次前向开销', 'blue'),
        (0.355, 0.58, 0.29, 0.24, '② 多周期重叠缓冲\n汇聚历史与当前周期\n对时刻 t 的预测', 'teal'),
        (0.68, 0.58, 0.29, 0.24, '③ 指数时间集成\nw_k = exp(-m·k)\n抑制块间切换跳变', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '动作块长度 H 的工程折中：平滑性 vs. 开环盲区', color='orange')
    box(ax, 0.04, 0.13, 0.44, 0.20, 'H 过长（开环窗口过大）\n中途环境突变无法即时修正\n时间集成还会滞后避障指令', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '解耦方案：System 2 低频潜特征\n+ System 1 融合实时力觉反馈\n每 10 ms 闭环修正短动作块', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '动作块解决高频抖动，但需配合高频本体反馈消除开环盲区', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch19-action-chunking-and-temporal-ensembling')


def fig_19_4_dual_loop(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'System 2（慢思考 VLM）+ System 1（快反射动作头）双频架构', color='blue')
    stages = [
        (0.03, 0.58, 0.29, 0.24, 'System 2：7B VLM\n频率 2 Hz（500 ms）\n融合多目图像与指令', 'blue'),
        (0.355, 0.58, 0.29, 0.24, '无锁潜特征环形缓冲\n发布语义潜向量 z_t\n读写零阻塞最旧覆写', 'purple'),
        (0.68, 0.58, 0.29, 0.24, 'System 1：150M 专家\n频率 100 Hz（10 ms）\n融合关节与六维力觉', 'teal'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '双系统频率比 50:1（π0 / Helix / GR00T 主流范式）', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, 'System 2 单次 500 ms 推理内\nSystem 1 读取最新潜特征 z_t\n连续完成 50 次 10 ms 闭环', 'teal', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '6 步欧拉 ODE 积分耗时 4.2 ms\n严格小于 10 ms 硬实时上限\n兼顾语义泛化与柔顺控制', 'blue', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '慢系统提供语义目标，快系统负责高频抗扰与接触闭环', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch19-system2-vlm-system1-flow-matching-dual-loop')


def fig_19_5_jetson_thor(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '单颗机载 SoC（Jetson Thor）上的硬件级确定性隔离', color='teal')
    stages = [
        (0.03, 0.58, 0.29, 0.24, 'MIG 0（占 75% SM）\n跑 System 2 VLM\n大吞吐视觉预填充', 'blue'),
        (0.355, 0.58, 0.29, 0.24, '共享内存 SeqLock\n双缓冲原子换指针\n零系统调用零锁', 'purple'),
        (0.68, 0.58, 0.29, 0.24, 'MIG 1（占 25% SM）\n专供 System 1\n独立 L2 与带宽', 'teal'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '共置竞争抖动 vs. MIG + PREEMPT_RT 核绑定确定性保障', color='orange')
    box(ax, 0.04, 0.13, 0.44, 0.20, '无隔离同卡混跑（MPS/共享流）\n视觉矩阵乘挤占 L2 与带宽\nSystem 1 P99.9 飙至 > 35 ms', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '硬件 MIG + PREEMPT_RT 绑核\nCUDA Graph 固化动作头算子\nSystem 1 WCET 锁在 ≤ 4.2 ms', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '硬实时控制看最坏执行时间（WCET），必须切断缓存与带宽干扰', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch19-jetson-thor-mig-rtos-deterministic-isolation')


def fig_19_6_flow_matching_ode(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, '传统 DDPM 扩散（50 步）', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '弯曲随机轨迹需 50–100 步\n串行调用动作网络 50 次', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '单次生成耗时 40–120 ms\n无法满足 100 Hz 闭环周期', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '每步注入随机高斯噪声\n引发力矩抖动与机械谐振', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '步数多且含噪声，控制抖动大', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, '流匹配（Flow-Matching 6 步）', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '最优传输直线速度场 v_θ\n沿直线仅需 4–6 步欧拉积分', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '静态展开为单个 CUDA Graph\n6 步总耗时 4.2 ms（零启动税）', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '确定性常微分方程积分\n轨迹平滑，抖动 < 0.1 ms', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '少步确定性积分契合硬实时控制', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch19-flow-matching-ode-steps-vs-control-jitter')


def fig_19_7_gpu_native_sim(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, 'CPU 仿真 + PCIe 往返瓶颈', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, 'CPU 进程池跑物理步进\n受核数限制仅 64–256 环境', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '每步跨 PCIe 传图像与状态\n4,096 环境需 43.39 GB/s', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '打满 PCIe 5.0 x16 带宽上限\nGPU 策略 85% 时间空等 I/O', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '跨总线搬移锁死大规模仿真吞吐', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'GPU 原生万级仿真（Isaac）', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '碰撞求解全编译为 GPU 核\n单卡并发 4,096–16,384 环境', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '批量光追渲染直写 HBM 显存\n零拷贝映射为 PyTorch 张量', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '物理→渲染→训练全在 HBM\nPCIe 流量=0，吞吐升 200x', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '数据零出卡成就万级并发训练', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch19-cpu-sim-pcie-bottleneck-vs-gpu-native-sim')


def fig_19_8_three_way_budget(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '单卡 4,096 环境三合一显存账本（共 25.52 GiB，算例 19-4）', color='blue')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '物理与接触缓冲\n状态 2,048 MiB\nBVH 资产 4,096 MiB', 'teal'),
        (0.355, 0.58, 0.29, 0.24, '批量多目渲染缓冲\n4,096 × 双目相机\n占 3,600 MiB', 'blue'),
        (0.68, 0.58, 0.29, 0.24, '策略与经验轨迹池\n权重+激活+Rollout\n占 16,384 MiB', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '单张 H100（132 SM）上的三方计算单元与 SM 分区协同', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '异构硬件单元流水交错\n物理吃 FP32 核（44 SM）\n渲染 48 SM + 策略 40 SM', 'teal', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '双缓冲环境批次（2,048×2）\n半数渲染时另半数跑策略推理\nRT 与 Tensor 核双双拉满', 'blue', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '物理、渲染与策略恰好对应 SIMT、RT Core 与 Tensor Core', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch19-physics-renderer-policy-three-way-sm-budget')


def fig_19_9_contact_divergence(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '4,096 并行环境中稀疏碰撞引发的 SIMT Warp 分支发散', color='orange')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '按环境静态映射线程\n1 个 Warp（32 线程）\n绑 32 个相邻环境', 'gray'),
        (0.355, 0.58, 0.29, 0.24, '29 个环境无碰撞\n0 接触点 1 次迭代\n早早完成空转等待', 'teal'),
        (0.68, 0.58, 0.29, 0.24, '3 个环境多点碰撞\n32 接触点迭代 20 轮\n拖慢全 Warp 跌 8x', 'orange'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, 'MuJoCo Warp / Genesis 接触点流式紧缩（Stream Compaction）', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '碰撞检测后执行前缀和紧缩\n仅提取活跃接触对建任务队列\n跳过无碰撞环境的空转分支', 'teal', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '按活跃接触对分配线程束\nSIMT 占用率从 25% 升至 88%\n支撑灵巧手多指接触高速仿真', 'blue', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '从“一环境一线程”改为“一活跃接触对一线程”消除分支发散', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch19-contact-divergence-warp-inefficiency')


def fig_19_10_tiled_rendering(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '多环境相机画布拼接渲染（Tiled Rendering）与零拷贝映射', color='teal')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '① 实例化场景图\n4,096 环境享底模\n仅更新位姿矩阵', 'blue'),
        (0.355, 0.58, 0.29, 0.24, '② 大画布拼接光追\n64×64 网格大纹理\n1 次 Kernel 全渲染', 'teal'),
        (0.68, 0.58, 0.29, 0.24, '③ 显存零拷贝视图\n经 DLPack 映射为\n[4096,C,H,W] 张量', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '消除逐环境 DrawCall 风暴与 Vulkan-CUDA 双份拷贝', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, '朴素逐环境循环渲染\n每步触发 8,192 次微光追调用\nCPU 驱动提交耗达 35 ms/步', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '大画布批量光追 + 零拷贝视图\n4,096 双目渲染耗时压至 1.8 ms\n显存减半且零显存间复制', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '将数千微型相机合批为单次全屏光追是突破万帧每秒的前提', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch19-multi-camera-tiled-rendering-zero-copy-tensor')


def fig_19_11_ptp_alignment(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '机载多传感器硬件脉冲触发与 PTP 微秒级时间同步', color='blue')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '硬件主时钟 PTP\n30 Hz 硬同步脉冲\n统一相机曝光中心', 'blue'),
        (0.355, 0.58, 0.29, 0.24, '高频 IMU 1,000 Hz\n与关节编码器 500 Hz\n打硬件单调时钟戳', 'teal'),
        (0.68, 0.58, 0.29, 0.24, 'SE(3) 运动插值\n将点云与关节角对齐\n至相机曝光时刻', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '为什么软件接收时间戳（ROS arrival time）会毒化模仿学习？', color='orange')
    box(ax, 0.04, 0.13, 0.44, 0.20, '软件时间戳受内核调度抖动\n图像与关节角偏移 15–40 ms\n模型学到错误的因果滞后映射', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '硬件曝光触发 + 环形缓冲对齐\n跨模态时间同步精度 < 100 μs\n精细操作学习成功率升 22%', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '时空未对齐的演示数据比噪音更危险，硬件同步是数据第一门禁', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch19-multi-sensor-ptp-hardware-trigger-alignment')


def fig_19_12_shadow_mode(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '真机车队影子模式（Shadow Mode）与难例触发（算例 19-5）', color='teal')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '机载 60 s 环形黑匣子\n原始速率 180 MB/s\n全量达 2.592 TB/天', 'gray'),
        (0.355, 0.58, 0.29, 0.24, '影子策略并行旁路\n实时算动作分歧 D_t\n或接管/超限触发', 'orange'),
        (0.68, 0.58, 0.29, 0.24, '冻结前后 15 s 窗口\n0.5% 触发 12.96 GB\n回传带宽压 200 倍', 'teal'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '三大高价值边缘触发器（Edge Triage Triggers）', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, '① 策略分歧与流匹配方差触发\n在役与影子策略预测轨迹偏离\n或动作分布出现双峰高熵', 'blue', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '② 物理反常与人类接管触发\n遥操作紧急接管、力冲击超限\n或世界模型预测画面失配', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '丢弃 99.5% 平庸数据、捕获 0.5% 边界难例可百倍降低闭环成本', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch19-shadow-mode-disagreement-edge-trigger')


def fig_19_13_3dgs_sim2real(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '3D 高斯泼溅（3DGS）神经重建与可微闭环重仿真流水线', color='purple')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '① 难例日志云端重建\nSLAM 位姿精化\n训 3DGS 辐射场', 'blue'),
        (0.355, 0.58, 0.29, 0.24, '② 物理属性对齐导入\n碰撞网格入 Isaac\n前景物体独立受控', 'teal'),
        (0.68, 0.58, 0.29, 0.24, '③ 闭环新视角合成\n策略偏离原采集路线\n3DGS 渲染对应视角', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '跨越 Sim-to-Real：开环日志回放 vs. 3DGS + 物理闭环重仿真', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '传统开环视频回放（Log Replay）\n新策略动作偏离原采集轨迹时\n相机画面无法改变，分布偏移', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '3DGS 渲染 + 刚体接触物理引擎\n兼具真实光照纹理与反事实交互\n1 个难例扩增 1,000 个变体', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '3DGS 消除视觉外观差距，GPU 物理引擎保证接触力学闭环', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch19-3dgs-neural-reconstruction-sim2real-loop')


def fig_19_14_edge_offload_safety(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '机载端侧计算、边缘卸载（MEC）与本地安全护盾（CBF）', color='blue')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '局域 5G/Wi-Fi 7 边缘\n卸载 30B+ System 2\n网络 P99 抖动 50 ms', 'blue'),
        (0.355, 0.58, 0.29, 0.24, '机载 System 1 100 Hz\n无线闪断 200 ms 时\n凭潜特征维持控制', 'teal'),
        (0.68, 0.58, 0.29, 0.24, '控制屏障护盾（CBF）\n运行于 1,000 Hz 核\n硬裁剪越界力矩', 'orange'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '具身安全铁律：绝不允许无线网络处于高频闭环控制关键路径', color='orange')
    box(ax, 0.04, 0.13, 0.44, 0.20, '错误设计：边缘直接下发力矩\n无线重传抖动致机械臂剧烈抽搐\n断网瞬间处于无保护盲飞状态', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '正确设计：仅卸载慢规划 VLM\n端侧闭环 System 1 + CBF 护盾\n超时未获新目标自动平滑驻停', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '无论大模型多强，最后一层必须是确定性物理安全包络', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch19-onboard-vs-mec-edge-offload-safety-fallback')


def fig_19_15_vla_token_pruning(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '面向硬实时截止期的视觉词元剪枝（VLA-Pruner）与早退', color='teal')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '① 4 路高清相机输入\n共 1,024 视觉词元\n全算耗尽 150 ms', 'orange'),
        (0.355, 0.58, 0.29, 0.24, '② 时空差分剪枝\n复用静态背景 KV\n仅留夹爪周 256 词元', 'teal'),
        (0.68, 0.58, 0.29, 0.24, '③ 截止期感知早退\n充裕跑 28 层，紧急时\n第 18 层提前输出', 'blue'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '将可变长视觉推理改造为硬实时确定性算子预算', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, 'Top-K 固定配额（非动态阈值）\n强制保留词元数恒为 K = 256\n后续层张量形状 100% 静态', 'teal', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '静态形状兼容 CUDA Graph\n视觉 Prefill 从 145 压至 38 ms\n抖动 < 0.3 ms，成功率无损', 'blue', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '用固定配额 Top-K 剪枝替代动态阈值，兼得加速与静态图确定性', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch19-vla-early-exit-and-token-pruning-wcet')


def fig_19_16_full_stack_map(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '具身智能（VLA）全栈底座：万级仿真、机载硬实时与影子飞轮', color='blue')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '云端万级 GPU 仿真\n4,096 环境零拷贝\n25.52 GiB（19.3 节）', 'blue'),
        (0.355, 0.58, 0.29, 0.24, '机载双频确定性推理\n2 Hz VLM + 100 Hz 流\nWCET ≤ 4.2 ms', 'teal'),
        (0.68, 0.58, 0.29, 0.24, '车队影子与 3DGS\nPTP + 0.5% 触发\n省带宽 200 倍', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '第 19 章三大核心定量结论汇总', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '盲飞位移法则：Δx = v·Δt_inf\n0.5 m/s 下 200 ms 漂移 100 mm\n5 mm 插孔要求周期 ≤ 10 ms', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '仿真与飞轮：破 43.39 GB/s 墙\n单卡 HBM 融合求解光追与训练\n影子触发将 2.59 TB 压至 13 GB', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '具身底座的本质是跨越时间确定性（WCET）与空间多样性（Sim2Real）', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch19-embodied-ai-full-stack-infrastructure-map')


def build_chapter() -> None:
    ex = Exporter(HERE)
    fig_19_1_cloud_vs_embodied(ex)
    fig_19_2_vlalagbench(ex)
    fig_19_3_action_chunking(ex)
    fig_19_4_dual_loop(ex)
    fig_19_5_jetson_thor(ex)
    fig_19_6_flow_matching_ode(ex)
    fig_19_7_gpu_native_sim(ex)
    fig_19_8_three_way_budget(ex)
    fig_19_9_contact_divergence(ex)
    fig_19_10_tiled_rendering(ex)
    fig_19_11_ptp_alignment(ex)
    fig_19_12_shadow_mode(ex)
    fig_19_13_3dgs_sim2real(ex)
    fig_19_14_edge_offload_safety(ex)
    fig_19_15_vla_token_pruning(ex)
    fig_19_16_full_stack_map(ex)
    outputs = ex.finish()

    fig_index = [
        {
            "figure_id": "图 19-1",
            "number": 1,
            "file": "ch19-cloud-llm-slo-vs-embodied-hard-realtime.svg",
            "asset": "ch19/ch19-cloud-llm-slo-vs-embodied-hard-realtime.svg",
            "section": "19.1.1",
            "caption": "云端对话软实时 SLO 与物理世界闭环控制硬实时截止期（盲飞位移 Δx = v·Δt_inf，算例 19-1）对比",
            "source_refs": ["experiments/ch19/19-01/run.py", "manuscripts/ch19/figure-data.json"],
            "TeachingRole": "建立时间延迟向物理空间碰撞误差转化的第一性原理公式"
        },
        {
            "figure_id": "图 19-2",
            "number": 2,
            "file": "ch19-vlalagbench-latency-vs-dynamic-grasp-success.svg",
            "asset": "ch19/ch19-vlalagbench-latency-vs-dynamic-grasp-success.svg",
            "section": "19.1.2",
            "caption": "VLALagBench 实证规律：静态操作 vs. 动态操作随推理延迟 Δt_inf 增大的成功率退化曲线",
            "source_refs": ["experiments/ch19/19-01/run.py", "manuscripts/ch19/figure-data.json"],
            "TeachingRole": "揭示观测过时（Observation Staleness）导致 100 ms 以上动态抓取成功率崩塌的机理"
        },
        {
            "figure_id": "图 19-3",
            "number": 3,
            "file": "ch19-action-chunking-and-temporal-ensembling.svg",
            "asset": "ch19/ch19-action-chunking-and-temporal-ensembling.svg",
            "section": "19.1.3",
            "caption": "动作块预测（Action Chunking H 步）与滑动窗口指数加权时间集成（Temporal Ensembling）时序图",
            "source_refs": ["manuscripts/ch19/figure-data.json"],
            "TeachingRole": "剖析动作块长度在抑制高频抖动与引入开环盲区之间的工程权衡"
        },
        {
            "figure_id": "图 19-4",
            "number": 4,
            "file": "ch19-system2-vlm-system1-flow-matching-dual-loop.svg",
            "asset": "ch19/ch19-system2-vlm-system1-flow-matching-dual-loop.svg",
            "section": "19.2.1",
            "caption": "System 2（2 Hz 多模态 VLM 慢思考）+ System 1（100 Hz Flow-Matching 动作头快反射）双频异步架构",
            "source_refs": ["experiments/ch19/19-01/run.py", "manuscripts/ch19/figure-data.json"],
            "TeachingRole": "展示 50:1 双频异步协同如何兼顾大模型语义泛化与 10 ms 硬实时闭环控制"
        },
        {
            "figure_id": "图 19-5",
            "number": 5,
            "file": "ch19-jetson-thor-mig-rtos-deterministic-isolation.svg",
            "asset": "ch19/ch19-jetson-thor-mig-rtos-deterministic-isolation.svg",
            "section": "19.2.2",
            "caption": "NVIDIA Jetson Thor / Drive Thor 上利用 GPU MIG 分区与无锁共享内存隔离 System 2 干扰的 WCET 保障",
            "source_refs": ["experiments/ch19/19-01/run.py"],
            "TeachingRole": "解析硬件级缓存/带宽隔离与 PREEMPT_RT 核绑定如何锁定 WCET ≤ 4.2 ms"
        },
        {
            "figure_id": "图 19-6",
            "number": 6,
            "file": "ch19-flow-matching-ode-steps-vs-control-jitter.svg",
            "asset": "ch19/ch19-flow-matching-ode-steps-vs-control-jitter.svg",
            "section": "19.2.1",
            "caption": "动作生成头中 DDPM 多步采样 vs. Flow-Matching 少步确定性 ODE 积分的周期稳定性对比",
            "source_refs": ["experiments/ch19/19-01/run.py"],
            "TeachingRole": "说明最优传输直线流匹配与静态 CUDA Graph 如何消除扩散采样抖动"
        },
        {
            "figure_id": "图 19-7",
            "number": 7,
            "file": "ch19-cpu-sim-pcie-bottleneck-vs-gpu-native-sim.svg",
            "asset": "ch19/ch19-cpu-sim-pcie-bottleneck-vs-gpu-native-sim.svg",
            "section": "19.3.1",
            "caption": "传统 CPU 物理仿真 + PCIe 往返（43.39 GB/s 瓶颈）vs. GPU 原生万级并行仿真的数据零搬移闭环",
            "source_refs": ["experiments/ch19/19-01/run.py", "manuscripts/ch19/figure-data.json"],
            "TeachingRole": "定量证明 4,096 环境跨 PCIe 传输观测必撞带宽墙，必须全驻留 GPU 显存"
        },
        {
            "figure_id": "图 19-8",
            "number": 8,
            "file": "ch19-physics-renderer-policy-three-way-sm-budget.svg",
            "asset": "ch19/ch19-physics-renderer-policy-three-way-sm-budget.svg",
            "section": "19.3.2",
            "caption": "单张 GPU 内刚体接触求解器、批量光线追踪渲染器与 VLA 策略网络的三方显存（25.52 GiB）与 SM 分配",
            "source_refs": ["experiments/ch19/19-01/run.py", "manuscripts/ch19/figure-data.json"],
            "TeachingRole": "核算 4,096 并行环境在 HBM 与 132 个 SM（CUDA/RT/Tensor Core）上的协同账本"
        },
        {
            "figure_id": "图 19-9",
            "number": 9,
            "file": "ch19-contact-divergence-warp-inefficiency.svg",
            "asset": "ch19/ch19-contact-divergence-warp-inefficiency.svg",
            "section": "19.3.2",
            "caption": "4,096 个并行仿真环境中稀疏多点碰撞导致的 SIMT Warp 分支发散与流式紧缩重排序",
            "source_refs": ["manuscripts/ch19/figure-data.json"],
            "TeachingRole": "展示按活跃接触对流式紧缩（Stream Compaction）如何解决 GPU 物理求解分支发散"
        },
        {
            "figure_id": "图 19-10",
            "number": 10,
            "file": "ch19-multi-camera-tiled-rendering-zero-copy-tensor.svg",
            "asset": "ch19/ch19-multi-camera-tiled-rendering-zero-copy-tensor.svg",
            "section": "19.3.1",
            "caption": "多环境相机画布拼接渲染（Tiled Rendering）直接映射为 PyTorch 张量避免逐环境小 Kernel 启动",
            "source_refs": ["manuscripts/ch19/figure-data.json"],
            "TeachingRole": "通过单次大画布光追与 DLPack 零拷贝映射消除 8,192 次微调用开销"
        },
        {
            "figure_id": "图 19-11",
            "number": 11,
            "file": "ch19-multi-sensor-ptp-hardware-trigger-alignment.svg",
            "asset": "ch19/ch19-multi-sensor-ptp-hardware-trigger-alignment.svg",
            "section": "19.4.1",
            "caption": "多目相机曝光、激光雷达扫描、IMU 与关节力矩传感器的微秒级硬件 PTP 时间戳对齐",
            "source_refs": ["manuscripts/ch19/figure-data.json"],
            "TeachingRole": "对比软件接收时间戳抖动与硬件曝光触发 + SE(3) 插值对齐的精度差异"
        },
        {
            "figure_id": "图 19-12",
            "number": 12,
            "file": "ch19-shadow-mode-disagreement-edge-trigger.svg",
            "asset": "ch19/ch19-shadow-mode-disagreement-edge-trigger.svg",
            "section": "19.4.2",
            "caption": "真机影子模式（Shadow Mode）：利用策略动作分歧度触发 0.5% 难例切片回传（省带宽 200 倍，算例 19-5）",
            "source_refs": ["experiments/ch19/19-01/run.py", "manuscripts/ch19/figure-data.json"],
            "TeachingRole": "核算机载环形黑匣子与边缘难例触发将单机日传数据从 2.592 TB 压至 12.96 GB"
        },
        {
            "figure_id": "图 19-13",
            "number": 13,
            "file": "ch19-3dgs-neural-reconstruction-sim2real-loop.svg",
            "asset": "ch19/ch19-3dgs-neural-reconstruction-sim2real-loop.svg",
            "section": "19.4.2",
            "caption": "利用真机难例日志构建 3D 高斯泼溅（3DGS）可微数字孪生场景并用于闭环重仿真的完整流水线",
            "source_refs": ["manuscripts/ch19/figure-data.json"],
            "TeachingRole": "结合 3DGS 新视角合成与物理碰撞网格实现照片级反事实闭环重仿真"
        },
        {
            "figure_id": "图 19-14",
            "number": 14,
            "file": "ch19-onboard-vs-mec-edge-offload-safety-fallback.svg",
            "asset": "ch19/ch19-onboard-vs-mec-edge-offload-safety-fallback.svg",
            "section": "19.2.2",
            "caption": "机载端侧计算与局域 5G/Wi-Fi 7 边缘卸载的抖动边界及本地控制屏障安全护盾（CBF）",
            "source_refs": ["manuscripts/ch19/figure-data.json"],
            "TeachingRole": "确立仅卸载慢速语义规划、端侧保留高频闭环与 1,000 Hz 安全护盾的容错边界"
        },
        {
            "figure_id": "图 19-15",
            "number": 15,
            "file": "ch19-vla-early-exit-and-token-pruning-wcet.svg",
            "asset": "ch19/ch19-vla-early-exit-and-token-pruning-wcet.svg",
            "section": "19.2.2",
            "caption": "面向硬实时截止期的多视角视觉词元固定配额剪枝（VLA-Pruner）与层早退机制",
            "source_refs": ["manuscripts/ch19/figure-data.json"],
            "TeachingRole": "说明为何必须采用固定 Top-K 词元配额才能保持静态张量形状与 CUDA Graph 兼容性"
        },
        {
            "figure_id": "图 19-16",
            "number": 16,
            "file": "ch19-embodied-ai-full-stack-infrastructure-map.svg",
            "asset": "ch19/ch19-embodied-ai-full-stack-infrastructure-map.svg",
            "section": "19.4.2",
            "caption": "具身智能从万级 GPU 物理仿真、双频实时机载推理到车队影子数据飞轮的全栈基础设施总图",
            "source_refs": ["experiments/ch19/19-01/run.py"],
            "TeachingRole": "收束第 19 章硬实时控制、GPU 原生仿真与 Sim-to-Real 数据闭环的工程总账本"
        }
    ]
    for row in fig_index:
        row['figure'] = f"19-{row['number']}"
        row['name'] = Path(row['file']).stem
    (HERE / 'figure-index.json').write_text(json.dumps(fig_index, ensure_ascii=False, indent=2) + '\n')

    md = HERE.parent / '19-具身智能（VLA）、实时控制与大规模物理仿真底座.md'
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

        work = re.sub(r'!\[([^\]]*)\]\((ch19/[^)]+)\)', repl_img, text_md)
        body = markdown.markdown(work, extensions=['tables', 'fenced_code', 'footnotes', 'toc'])
        html = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>第 19 章：具身智能（VLA）、实时控制与大规模物理仿真底座</title></head><body>{body}</body></html>\n'
        out_html = preview_path(md)
        out_html.write_text(html)

    sources = [HERE / 'build.py', HERE / 'figure-data.json', HERE / 'figure-index.json', ROOT / 'experiments/ch19/19-01/run.py']
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
    print(f'Built {len(outputs)} figure files for Chapter 19; 0 layout warnings.')


if __name__ == '__main__':
    build_chapter()
