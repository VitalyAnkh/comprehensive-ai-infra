#!/usr/bin/env python3
"""Build Chapter 15 figures, locked calculations, and offline HTML preview."""
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


def fig_15_1_power_chain(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.52, 0.96, 0.44, '吉瓦级 AI 工厂（1 GW）四级供电链路与损耗分解', color='blue')
    stages = [
        (0.04, 0.60, 0.21, 0.22, '① 高压变电\n110–220 kV\n损耗 ~0.8%', 'blue'),
        (0.28, 0.60, 0.21, 0.22, '② 中压配电\n10–35 kV\n损耗 ~1.2%', 'teal'),
        (0.52, 0.60, 0.21, 0.22, '③ 变压/UPS\nSST 固态变压\n损耗 ~2.5%', 'purple'),
        (0.76, 0.60, 0.20, 0.22, '④ 机柜母排\n800 VDC\n直供 NVL72', 'orange'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.25, 0.71), (0.28, 0.71))
    arrow(ax, (0.49, 0.71), (0.52, 0.71))
    arrow(ax, (0.73, 0.71), (0.76, 0.71))

    card(ax, 0.02, 0.03, 0.96, 0.43, '算例 15-1：132 kW 机柜 48 VDC vs. 800 VDC 母排铜损', color='teal')
    box(ax, 0.04, 0.14, 0.44, 0.21, '传统 48 VDC 机柜铜母排\nI = 132 kW / 48 V = 2,750 A\n母排厚重且 I²R 发热剧烈', 'orange', size=11)
    box(ax, 0.52, 0.14, 0.44, 0.21, '新一代 800 VDC 高压母排\nI = 165 A（电流降 16.7x）\n同阻值 I²R 铜损骤降 277.8x', 'teal', size=11)
    arrow(ax, (0.48, 0.245), (0.52, 0.245))
    text(ax, 0.50, 0.07, '电压升 16.67 倍 → 电流降为 1/16.7 → 同阻值铜损降为 1/278', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch15-gigawatt-ai-factory-power-chain')


def fig_15_2_feasibility(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.04, 0.46, 0.92, '口径错配陷阱（高估容量）', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '分子分母错位（D11-08）\n整柜功率直乘单卡 GB/W', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '漏扣非 GPU 功耗 28%\nCPU·交换机·PSU·水泵', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '漏扣显存运行预留 20%\n激活·通信缓冲·页碎片', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '纸面高估显存 40%–70%', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, '物理与经济可行双边界', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '正确物理容量方程\nM = P·η_gpu·(GB/W)·u', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '物理可行：装得下供得上\n结温与机柜功率不越界', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '经济可行：SLO 下最优\n单词元成本低于商业线', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '经济域严格内嵌于物理域', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch15-physical-vs-economic-feasibility')


def fig_15_3_site_tco(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.50, 0.96, 0.46, '站点条件作为一等设计输入（D11-09）：四大核心变量', color='blue')
    vars_box = [
        (0.04, 0.58, 0.21, 0.24, '① 工业电价\n0.04–0.12\n美元/kWh', 'blue'),
        (0.28, 0.58, 0.21, 0.24, '② 供电配额\n变电站上限\nPUE 1.08+', 'teal'),
        (0.52, 0.58, 0.21, 0.24, '③ 并网工期\n等 6–18 月\n技术折旧', 'orange'),
        (0.76, 0.58, 0.20, 0.24, '④ 机柜热限\n风冷 20 kW\n液冷 130 kW', 'purple'),
    ]
    for x, y, w, h, label, role in vars_box:
        box(ax, x, y, w, h, label, role, size=11)

    card(ax, 0.02, 0.03, 0.96, 0.42, '算例 15-3：100 MW 园区电价差与并网延期 6 个月折旧', color='orange')
    box(ax, 0.04, 0.13, 0.44, 0.20, '电价差 0.04 vs 0.12 美元\n100 MW 三年差 2.10 亿美元\n高电价区须优先压每瓦吞吐', 'blue', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '36 个月寿命下延期 6 个月\n20 亿美元硬件折旧 3.33 亿\n远超跨园区分建或自备电费', 'orange', size=11)
    text(ax, 0.50, 0.06, '并网工期（Time-to-Power）的折旧代价往往压倒单纯电价差', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch15-site-conditions-tco-sensitivity')


def fig_15_4_power_swing(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.50, 0.96, 0.46, '十万卡同步训练 Step 边界的周期性功率阶跃（Power Swing）', color='orange')
    phases = [
        (0.04, 0.58, 0.28, 0.24, '前反向 GEMM\n10 万卡满载\n功率 100 MW', 'orange'),
        (0.36, 0.58, 0.28, 0.24, 'AllReduce 屏障\n矩阵单元停顿\n跌至 35 MW', 'blue'),
        (0.68, 0.58, 0.28, 0.24, '下一 Step 拉满\n毫秒级重满载\n爬升 +65 MW', 'orange'),
    ]
    for x, y, w, h, label, role in phases:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.36, 0.70))
    arrow(ax, (0.64, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '65 MW / 10 ms 阶跃斜率（6,500 MW/s）对电网的冲击', color='purple')
    box(ax, 0.04, 0.13, 0.44, 0.20, '50 ms–1 s 周期次同步谐振\n超出燃气轮机与变压器调速\n引发母线电压闪变与跳闸', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '训练故障急停（Job Crash）\n100 MW 负载数毫秒内归零\n导致园区母线过压与频升', 'purple', size=11)
    text(ax, 0.50, 0.06, '同步训练呈现出传统互联网机房从未有过的强相干方波冲击', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch15-synchronous-training-power-swing')


def fig_15_5_rubin_smoothing(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.04, 0.46, 0.92, 'Rubin 机架智能功率平滑', color='teal', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '电源架内置储能单元\n按荷电状态 SoC 反馈控制', 'teal', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '高峰放电 · 通信低谷充电\n削平约 20% 的 50 ms 脉冲', 'blue', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '消除防尖峰预留裕量浪费\n降低约 10% 机架平均功耗', 'purple', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '将尖锐方波过滤为平滑负载', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'DSX MaxLPS 功率超配', color='blue', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '传统按铭牌峰值刚性配电\n100 MW 配额仅装 750 柜', 'orange', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '软硬件封顶与储能缓冲\n将持续功率包络压低 28%', 'teal', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '同一 100 MW 变电配额下\n安全超配最高 +40% GPU', 'blue', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '固定电力内多产 40% 算力', size=11, ha='center', weight='bold', color=BORDER_COL['blue'])
    ex.save(fig, 'ch15-rubin-power-smoothing-and-maxlps')


def fig_15_6_runtime_stagger(ex: Exporter):
    fig, ax = canvas(3.60)
    card(ax, 0.02, 0.04, 0.46, 0.92, '未协同基线：全群相位锁死', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '所有并行组在同一时刻\n齐步进出 GEMM 或屏障', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '峰值叠加至 N · P_max\n谷值跌落至 N · P_idle', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, 'dP/dt 极大且储能电容\n须按全群满幅跌落设计', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '同步屏障成电网谐振放大器', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, '运行时软件协同抑峰三机制', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '① 流水微批次交错相位\n子群通信窗口错峰互补', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '② 功率爬升斜率限制器\n启停过程按 5 MW/s 缓变', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '③ 通信期动态 DVFS 降频\n或注入低功耗校验填谷', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '零硬件改动消减 60%+ 摆幅', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch15-runtime-staggered-microbatch-power')


def fig_15_7_ashrae_w45(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.52, 0.96, 0.44, 'ASHRAE W45（45 °C 进水）芯片冷板与无冷机自然冷却回路', color='teal')
    loop = [
        (0.04, 0.60, 0.21, 0.22, '① 芯片冷板\n1.0–1.5 kW\n45→57 °C', 'orange'),
        (0.28, 0.60, 0.21, 0.22, '② 分集水器\nManifold\n汇集 132 kW', 'purple'),
        (0.52, 0.60, 0.21, 0.22, '③ 液冷 CDU\n板式换热器\n一二次隔离', 'blue'),
        (0.76, 0.60, 0.20, 0.22, '④ 室外干冷\n免除压缩机\n全年自然冷', 'teal'),
    ]
    for x, y, w, h, label, role in loop:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.25, 0.71), (0.28, 0.71))
    arrow(ax, (0.49, 0.71), (0.52, 0.71))
    arrow(ax, (0.73, 0.71), (0.76, 0.71))

    card(ax, 0.02, 0.03, 0.96, 0.43, 'ASHRAE 风冷极限（1,900 CFM/砖）vs. 算例 15-4 液冷流量', color='blue')
    box(ax, 0.04, 0.14, 0.44, 0.21, '传统高密风冷极限（TC 9.9）\n单块地板砖 1,900 CFM（19U）\n风扇功耗反扑至整机 10%–20%', 'orange', size=11)
    box(ax, 0.52, 0.14, 0.44, 0.21, '算例 15-4：132 kW 液冷柜\n温差 12 °C 仅需 2.63 L/s 流量\n折合 157.8 L/min 冷却水流量', 'teal', size=11)
    text(ax, 0.50, 0.07, '水体积热容约为空气 3,500 倍 · W45 温水直冷将 PUE 压至 ≤1.08', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch15-ashrae-w45-cdu-liquid-cooling-loop')


def fig_15_8_logicfolding(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.04, 0.46, 0.92, '降压与三维折叠的功耗收益', color='teal', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '动态功耗 P ∝ C · V² · f\n电压 0.85 V 降至 0.55 V', 'teal', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '电压平方项降至 0.419\n节省约 58% 动态功耗', 'blue', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '三维堆叠缩短互连电容\n折叠 DSP 总功耗 0.75x', 'purple', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '总功耗降 25%（NPU 降 66%）', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])

    card(ax, 0.52, 0.04, 0.46, 0.92, '功率密度悖论（热流 +25%）', color='orange', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '三维折叠缩减投影面积 40%\n散热底面积降至 0.60x', 'orange', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '面积缩减快于功耗下降：\n热流密度 0.75/0.60 = 1.25x', 'orange', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '结温 T_j = T_f + P · Ψ_ca\n层间热阻叠加易局部过热', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '更省电芯片反而需更强液冷', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])
    ex.save(fig, 'ch15-logicfolding-power-density-paradox')


def fig_15_9_straggler(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.50, 0.96, 0.46, '热节流（Thermal Throttling）在同步训练中的木桶放大效应', color='orange')
    nodes = [
        (0.04, 0.58, 0.28, 0.24, '1,023 张正常卡\n结温 72 °C 达标\n耗时 100 ms', 'teal'),
        (0.36, 0.58, 0.28, 0.24, '1 张微堵过热卡\n触碰 88 °C 节流\n延至 117.6 ms', 'orange'),
        (0.68, 0.58, 0.28, 0.24, 'AllReduce 屏障\n千卡空等 17.6 ms\n算力跌 15%', 'purple'),
    ]
    for x, y, w, h, label, role in nodes:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.36, 0.70))
    arrow(ax, (0.64, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '单卡过热代价 = 单卡降频比例 × 同步并行域总卡数 N', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, '浪费算力 = 1,023 卡 × 17.6 ms\n相当于白白空转 153 张满载卡\n却不产生任何额外训练进度', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '工程根治：监测 Ψ_ca 热阻漂移\n主动将高热卡切出同步环替换\n或缩减该卡微批次计算配额', 'teal', size=11)
    text(ax, 0.50, 0.06, '在万卡同步集群中，液冷均匀性直接决定有效算力利用率（MFU）', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch15-thermal-throttling-straggler-amplification')


def fig_15_10_tapas(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.52, 0.96, 0.44, '算例 15-6：TAPAS 功率封顶（100 kW）与延迟（1.0 s）双重约束', color='orange')
    box(ax, 0.04, 0.59, 0.44, 0.24, '候选排 Row 0（功率违约）\n96 + 6 = 102 kW > 100 kW\n延迟 0.8 s 达标但功率越界', 'orange', size=11)
    box(ax, 0.52, 0.59, 0.44, 0.24, '候选排 Row 1（SLO 违约）\n80 + 6 = 86 kW ≤ 100 kW\n功率达标但延迟 1.3 s > 1.0 s', 'purple', size=11)

    card(ax, 0.02, 0.03, 0.96, 0.44, '热感知重配置（耗时 3.0 s，单请求省 0.1 s）的 30 请求回本线', color='teal')
    box(ax, 0.04, 0.14, 0.44, 0.21, '静态：feasible=[1], admitted=[]\n两排均无法同时满足双约束\n被迫拒绝请求或触发断路器', 'orange', size=11)
    box(ax, 0.52, 0.14, 0.44, 0.21, '动态重配置：付 3.0 s 串行迁移\n将单请求延迟压缩 Δt = 0.1 s\nN* = 3.0 / 0.1 = 30 请求回本', 'teal', size=11)
    arrow(ax, (0.48, 0.245), (0.52, 0.245))
    text(ax, 0.50, 0.07, '重配置须跨过 N ≥ 30 请求的稳态窗口方能摊平 3.0 s 迁移停顿', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch15-tapas-thermal-power-routing-breakeven')


def fig_15_11_dci_latency(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.50, 0.96, 0.46, '跨 1,000 km 广域光网（DCI）的物理延迟与带宽时延积（BDP）', color='blue')
    dci = [
        (0.04, 0.58, 0.28, 0.24, '光纤折射率极限\n光速约 20 万 km/s\n每公里 ~5 μs', 'blue'),
        (0.36, 0.58, 0.28, 0.24, '1,000 km 双园区\nRTT ≈ 10 ms\n慢于 NVLink 万倍', 'orange'),
        (0.68, 0.58, 0.28, 0.24, '1.6 Tbps 专线\n200 GB/s × 10 ms\n在途 BDP 2.0 GB', 'purple'),
    ]
    for x, y, w, h, label, role in dci:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.36, 0.70))
    arrow(ax, (0.64, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '五维并行策略对广域网延迟的容忍度阶梯', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '严禁跨城：TP 与 EP 全对全\n80 层每步需 160 次跨城 RTT\n仅光速延迟即空耗 1.6 s/Step', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '可跨城：Cross-DC PP / 分层 DP\n及低频外循环同步（DiLoCo）\n每微批传激活或 500 步传伪梯度', 'teal', size=11)
    text(ax, 0.50, 0.06, '跨城铁律：高敏通信留园区内，长延迟光纤只走低频大块传输', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch15-multi-datacenter-dci-optical-latency')


def fig_15_12_cross_dc_pp(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.04, 0.46, 0.92, '跨园区流水线（Cross-DC PP）', color='blue', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '园区 A 驻留第 1–40 层\n园区 B 驻留第 41–80 层', 'blue', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '跨城仅传边界激活张量\n单次仅几十 MiB 数据量', 'teal', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '掩盖条件：微批计算耗时\n≥ 跨城传输 + 5 ms 延迟', 'purple', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '用充足微批数 m 摊薄气泡', size=11, ha='center', weight='bold', color=BORDER_COL['blue'])

    card(ax, 0.52, 0.04, 0.46, 0.92, '两级分层数据并行（分层 DP）', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '第一级：园区内高速网络\n先完成园内万卡规约散射', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '第二级：跨城 DWDM 光纤\n仅在园区代表间交换分片', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '配合梯度累积步数 ≥ 32\n将跨城规约与反向相重叠', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '跨城流量与园内卡数解耦', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch15-cross-dc-pipeline-and-hierarchical-dp')


def fig_15_13_diloco(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'DiLoCo 双循环优化：内循环 H=500 步独立更新 + 外循环伪梯度', color='teal')
    steps = [
        (0.04, 0.58, 0.28, 0.24, '① 园内 H=500 步\nAdamW 独立训练\n跨城通信为 0', 'teal'),
        (0.36, 0.58, 0.28, 0.24, '② 算伪梯度 Δθ\nθ^(0) - θ^(H)\n仅跨城规约一次', 'blue'),
        (0.68, 0.58, 0.28, 0.24, '③ 外循环动量\n更新全局主权重\n带宽直降 500x', 'purple'),
    ]
    for x, y, w, h, label, role in steps:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.36, 0.70))
    arrow(ax, (0.64, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '算例 15-7：100B 模型跨城同步带宽压缩与外循环显存代价', color='orange')
    box(ax, 0.04, 0.13, 0.44, 0.20, '传统每步跨城 DP（2 s/Step）\n每步传 400 GB 需 1,600 Gbps\nDiLoCo (H=500) 均值 3.2 Gbps', 'teal', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '外循环显存代价（非免费午餐）\n需保存初始快照与外动量状态\n每参数多占 4–8 B（可分片）', 'orange', size=11)
    text(ax, 0.50, 0.06, '用园区内充裕的存储空间置换稀缺昂贵的跨地域广域网带宽', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch15-diloco-inner-outer-optimization')


def fig_15_14_streaming_diloco(ex: Exporter):
    fig, ax = canvas(3.60)
    card(ax, 0.02, 0.04, 0.46, 0.92, '标准 DiLoCo（第 H 步阻塞）', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '前 499 步跨城光纤闲置\n广域网链路利用率 = 0%', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '第 500 步突发全量传输\n400 GB 伪梯度阻塞训练', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '在 10 Gbps 慢速链路上\n单次同步停顿长达 320 s', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '均值虽低但峰值仍阻塞', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'Streaming DiLoCo（分片流水）', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '将模型切为 M 个参数分片\n每分片错开 H/M 步同步', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '分片 p 在后台跨城传输时\nGPU 正常跑其余分片训练', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '叠加 FP4 伪梯度压缩 4x\n跨城通信暴露趋近于 0', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '将尖峰抹平为恒定背景流', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch15-streaming-diloco-chunked-sync')


def fig_15_15_decision_tree(ex: Exporter):
    fig, ax = canvas(3.95)
    card(ax, 0.02, 0.03, 0.96, 0.94, '单园区等待变电站扩容 vs. 多园区跨城协同训练决策树', color='blue')
    box(ax, 0.05, 0.72, 0.90, 0.14, '目标：部署 300 MW 训练集群（单站点现成可用电力仅 100 MW）', 'blue', size=11)
    box(ax, 0.05, 0.40, 0.42, 0.24, '方案 A：单园区等变电扩容\n等待 9 个月 · 训练零跨城损耗\n承担 9 个月硬件闲置折旧', 'orange', size=11)
    box(ax, 0.53, 0.40, 0.42, 0.24, '方案 B：3 座 100 MW 即刻并网\n零等待 · 分层 DP 与 DiLoCo\n仅付 4%–7% 跨城效率折损', 'teal', size=11)
    arrow(ax, (0.26, 0.72), (0.26, 0.64))
    arrow(ax, (0.74, 0.72), (0.74, 0.64))
    box(ax, 0.05, 0.09, 0.90, 0.23, '工期翻转判据：当并网延期 ΔT_wait > T_job · (1/η - 1) 时多园区胜出\n对 90 天训练任务，即使跨城折损 10%，电网延期超 10 天即应选多园区', 'purple', size=11)
    ex.save(fig, 'ch15-single-campus-vs-multi-campus-decision')


def fig_15_16_summary(ex: Exporter):
    fig, ax = canvas(4.10)
    card(ax, 0.02, 0.03, 0.96, 0.94, '吉瓦级 AI 基础设施全景：结温—液冷—电网平滑—广域光网协同', color='teal')
    layers = [
        (0.05, 0.66, 0.42, 0.20, '① 硅片热流层（mm² 尺度）\nLogicFolding 1.25x 密度悖论\n控热阻防热节流木桶效应', 'orange'),
        (0.53, 0.66, 0.42, 0.20, '② 机柜液冷母排层（m 尺度）\n800 VDC 母排降铜损 278x\nASHRAE W45 温水免冷机', 'teal'),
        (0.53, 0.37, 0.42, 0.20, '③ 园区变电储能层（km 尺度）\nRubin 削平 20% 峰值脉冲\nDSX MaxLPS 超配 +40% 卡', 'blue'),
        (0.05, 0.37, 0.42, 0.20, '④ 跨城光网层（千公里尺度）\nTAPAS 路由 30 请求回本\nDiLoCo 压缩跨城带宽 500x', 'purple'),
    ]
    for x, y, w, h, label, role in layers:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.47, 0.76), (0.53, 0.76))
    arrow(ax, (0.74, 0.66), (0.74, 0.57))
    arrow(ax, (0.53, 0.47), (0.47, 0.47))
    arrow(ax, (0.26, 0.57), (0.26, 0.66))
    box(ax, 0.05, 0.08, 0.90, 0.21, '吉瓦时代恒等式：Token/s/MW = (单瓦有效算力 ÷ 每词元算力) ÷ PUE\n供电拓扑、液冷热学与广域光网已内化为 AI 体系结构的一等公民', 'blue', size=11)
    ex.save(fig, 'ch15-power-thermal-wan-co-design-summary')


def fig_15_17_microfluidics(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '单相冷板、两相相变冷却与硅背面微通道直冷的热阻对比', color='blue')
    modes = [
        (0.04, 0.58, 0.28, 0.24, '① 单相水冷板\n显热温升受限\n含 R_TIM 界面', 'orange'),
        (0.36, 0.58, 0.28, 0.24, '② 两相沸腾冷\n汽化潜热恒温\n仍受 R_TIM 限', 'blue'),
        (0.68, 0.58, 0.28, 0.24, '③ 硅背微流道\n蚀刻晶圆背面\n消除 R_TIM 层', 'teal'),
    ]
    for x, y, w, h, label, role in modes:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.36, 0.70))
    arrow(ax, (0.64, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '2,000 W 芯片消除 R_TIM=0.02 °C/W 的进水温度红利', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '传统冷板（R_TIM=0.02 °C/W）\n仅界面层即产生 40 °C 温差\n2 kW 下最高进水压至 21 °C', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '硅背面微通道直冷（R_TIM=0）\n消除 40 °C 界面温差瓶颈\n允许进液温度提升 +40.0 °C', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '当芯片迈向 2–3 kW 时，消除 R_TIM 是维持 W45 免冷机散热的关键', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch15-single-phase-vs-two-phase-microfluidics')


def fig_15_18_campus_bess(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '百兆瓦时园区储能（100 MWh BESS）的三重时间尺度调度', color='teal')
    tiers = [
        (0.04, 0.58, 0.28, 0.24, '① 毫秒级平抑\n削平 50 ms 方波\n抑制功率斜率', 'teal'),
        (0.36, 0.58, 0.28, 0.24, '② 分钟级调频\n电网一次调频\n虚拟转动惯量', 'blue'),
        (0.68, 0.58, 0.28, 0.24, '③ 小时级套利\n谷电 0.04 充入\n峰电 0.12 放出', 'purple'),
    ]
    for x, y, w, h, label, role in tiers:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.36, 0.70))
    arrow(ax, (0.64, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '100 MWh 储能系统峰谷套利与调频收益账本（练习 15-13）', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, '日内一充一放（η_rt = 88%）\n有效放电 88 MWh · 价差 0.08\n年峰谷套利 239–257 万美元', 'teal', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '叠加调频与需量削峰（+350 万）\n综合年收益 589–607 万美元\n静态投资回收期缩至 3.6–3.7 年', 'blue', size=11)
    text(ax, 0.50, 0.06, '将单用途备电 UPS 升级为跨时间尺度复用的收益型电网资产', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch15-campus-bess-peak-shaving-and-arbitrage')


def fig_15_19_kareus_energy(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, '均匀分摊（高静态漏电）', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '集群负载率 u = 40% 时\n所有 GPU 均保持唤醒态', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '单卡静态漏电 P_s = 180 W\n动态功耗 0.4×520 = 208 W', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '单卡总功耗 = 388 W\n静态漏电占比达 46.4%', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '浅度降频难消静态漏电', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'Kareus 动静态能耗合并', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, 'Kareus（OSDI 26 #63）\n频率调控与实例紧缩合并', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '40% 活跃卡满载（700 W）\n60% 空闲卡休眠（15 W）', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '等效单卡均功降至 289 W\n整群节电 25.5%–27.8%', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '压缩活跃卡数扑灭漏电', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch15-kareus-dynamic-static-energy-consolidation')


def fig_15_20_dips_wan(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'Di-PS（NSDI 26 #21）异构跨集群异步参数服务器训练', color='blue')
    stages = [
        (0.04, 0.58, 0.28, 0.24, '① 异构算力园\n不同代 NPU/GPU\n按算力定步频', 'blue'),
        (0.36, 0.58, 0.28, 0.24, '② 分片参数服\n算完第 l 层后\n异步推广域网', 'teal'),
        (0.68, 0.58, 0.28, 0.24, '③ 陈旧度阻尼\n伪梯度加修正\n防慢卡拖收敛', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.36, 0.70))
    arrow(ax, (0.64, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '同步跨城 AllReduce vs. Di-PS 层间流式参数服务吞吐', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '同步跨城集合通信（木桶受限）\n最快园区须等最慢园区与 RTT\n异构代差使快集群闲置 >30%', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, 'Di-PS 异步分片推送 + 阻尼对齐\n通信与前反向计算全时段重叠\n百毫秒 WAN 下保持 >90% 效率', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '解耦异构园区的硬件步调，用算法阻尼吸收广域网抖动', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch15-dips-wan-backbone-parameter-sync')


def fig_15_21_carbon_lifecycle(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, '隐含碳（Embodied）', color='purple', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '3nm/2nm 极紫外光刻晶圆\nCoWoS 封装与 HBM4 制造', 'purple', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '变电站·液冷管网·储能柜\n混凝土与铜母排基建碳排', 'blue', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '通电前已一次性锁定支出\n需靠高利用率满载摊薄', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '闲置等待推高单词元隐含碳', size=11, ha='center', weight='bold', color=BORDER_COL['purple'])

    card(ax, 0.52, 0.04, 0.46, 0.92, '运行碳（Operational）', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '取决于电网碳强度 CI(t)\n煤电 ~800 vs 水核风 <50', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '3 年满载耗电约 26.3 亿度\n火电区运行碳占比 > 75%', 'orange', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '绿电直供 + BESS 错峰消纳\n跨园区追逐风光富余电力', 'teal', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '绿电区隐含碳与运行碳并重', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch15-embodied-carbon-vs-operational-carbon-lifecycle')


def fig_15_22_master_ledger(ex: Exporter):
    fig, ax = canvas(4.15)
    card(ax, 0.02, 0.03, 0.96, 0.94, '吉瓦级 AI 算力园区工程总账本（从并网到跨洲协同）', color='blue')
    grid = [
        (0.05, 0.66, 0.42, 0.20, '① 供电与母排（15.1–2）\n800 VDC 母排降铜损 278x\nRubin 平滑 +40% 超配', 'blue'),
        (0.53, 0.66, 0.42, 0.20, '② 热学与微流道（15.3/6）\nW45 温水免冷机 PUE≤1.08\n硅背面微流道消除 R_TIM', 'teal'),
        (0.53, 0.37, 0.42, 0.20, '③ 储能与能效（15.6.2–3）\n100 MWh BESS 峰谷套利\nKareus 合并休眠省电 25%', 'purple'),
        (0.05, 0.37, 0.42, 0.20, '④ 跨地域训练（15.4 & 6）\nStreaming DiLoCo 零暴露\nDi-PS 异步容忍百毫秒 WAN', 'orange'),
    ]
    for x, y, w, h, label, role in grid:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.47, 0.76), (0.53, 0.76))
    arrow(ax, (0.74, 0.66), (0.74, 0.57))
    arrow(ax, (0.53, 0.47), (0.47, 0.47))
    arrow(ax, (0.26, 0.57), (0.26, 0.66))
    box(ax, 0.05, 0.08, 0.90, 0.21, '实验验证：ch15/15-01（微通道热阻·LogicFolding·TAPAS）\n与 ch15/15-02（DiLoCo·Kareus·100 MWh BESS 峰谷套利）', 'teal', size=11)
    ex.save(fig, 'ch15-gigawatt-campus-master-engineering-ledger')


def main():
    ex = Exporter(HERE)
    fig_15_1_power_chain(ex)
    fig_15_2_feasibility(ex)
    fig_15_3_site_tco(ex)
    fig_15_4_power_swing(ex)
    fig_15_5_rubin_smoothing(ex)
    fig_15_6_runtime_stagger(ex)
    fig_15_7_ashrae_w45(ex)
    fig_15_8_logicfolding(ex)
    fig_15_9_straggler(ex)
    fig_15_10_tapas(ex)
    fig_15_11_dci_latency(ex)
    fig_15_12_cross_dc_pp(ex)
    fig_15_13_diloco(ex)
    fig_15_14_streaming_diloco(ex)
    fig_15_15_decision_tree(ex)
    fig_15_16_summary(ex)
    fig_15_17_microfluidics(ex)
    fig_15_18_campus_bess(ex)
    fig_15_19_kareus_energy(ex)
    fig_15_20_dips_wan(ex)
    fig_15_21_carbon_lifecycle(ex)
    fig_15_22_master_ledger(ex)
    outputs = ex.finish()

    fig_index = [
        {
            "figure_id": "图 15-1",
            "number": 1,
            "file": "ch15-gigawatt-ai-factory-power-chain.svg",
            "asset": "ch15/ch15-gigawatt-ai-factory-power-chain.svg",
            "section": "15.1.1",
            "caption": "吉瓦级 AI 工厂从 220 kV 变电站到机柜 800 VDC 母排与液冷 CDU 的全链路供电及铜损分解（算例 15-1）",
            "source_refs": ["references/files/specs/nvidia-rubin-system.md", "manuscripts/ch15/figure-data.json"],
            "TeachingRole": "建立从高压电网到机柜 800 VDC 直流母排的四级转换链路与铜损缩减 277.8 倍账本"
        },
        {
            "figure_id": "图 15-2",
            "number": 2,
            "file": "ch15-physical-vs-economic-feasibility.svg",
            "asset": "ch15/ch15-physical-vs-economic-feasibility.svg",
            "section": "15.1.2",
            "caption": "物理可行域与经济可行域双边界及 P_domain × GB/W 口径错配高估警示图（算例 15-2，D11-08）",
            "source_refs": ["research/draft11-outline-audit/report.md", "manuscripts/ch15/figure-data.json"],
            "TeachingRole": "揭示漏扣 28% 非 GPU 固定功耗与 20% 显存运行余量导致的容量虚胖"
        },
        {
            "figure_id": "图 15-3",
            "number": 3,
            "file": "ch15-site-conditions-tco-sensitivity.svg",
            "asset": "ch15/ch15-site-conditions-tco-sensitivity.svg",
            "section": "15.1.3",
            "caption": "站点条件（电价、供电上限、并网延期折旧与散热阈值）作为一等设计输入的 TCO 敏感性矩阵（算例 15-3，D11-09）",
            "source_refs": ["research/draft11-outline-audit/report.md", "manuscripts/ch15/figure-data.json"],
            "TeachingRole": "量化电价差（2.10 亿美元）与并网延期 6 个月（3.33 亿美元折旧）对集群选址的决定性影响"
        },
        {
            "figure_id": "图 15-4",
            "number": 4,
            "file": "ch15-synchronous-training-power-swing.svg",
            "asset": "ch15/ch15-synchronous-training-power-swing.svg",
            "section": "15.2.1",
            "caption": "十万卡同步训练在 GEMM 计算期与 AllReduce 通信期周期性切换引发的 85 MW 电网阶跃冲击波形",
            "source_refs": ["references/files/specs/nvidia-rubin-system.md", "manuscripts/ch15/figure-data.json"],
            "TeachingRole": "展示毫秒级同步屏障如何将微观算子切换放大为宏观电网频率冲击"
        },
        {
            "figure_id": "图 15-5",
            "number": 5,
            "file": "ch15-rubin-power-smoothing-and-maxlps.svg",
            "asset": "ch15/ch15-rubin-power-smoothing-and-maxlps.svg",
            "section": "15.2.2",
            "caption": "Vera Rubin 机架级储能 SoC 功率平滑（削峰 20%、均功降 10%）与 DSX MaxLPS 同配额 +40% GPU 超配机制",
            "source_refs": ["references/files/specs/nvidia-rubin-system.md", "references/text/nvidia-rubin-arch.txt"],
            "TeachingRole": "解析机架级储能反馈控制与 DSX MaxLPS 如何在固定兆瓦配额下多塞入 40% 算力"
        },
        {
            "figure_id": "图 15-6",
            "number": 6,
            "file": "ch15-runtime-staggered-microbatch-power.svg",
            "asset": "ch15/ch15-runtime-staggered-microbatch-power.svg",
            "section": "15.2.3",
            "caption": "运行时软件协同抑峰：流水线微批次错峰注入、斜坡限幅器与空闲期受控保底核时序对比",
            "source_refs": ["references/files/specs/nvidia-rubin-system.md", "manuscripts/ch15/figure-data.json"],
            "TeachingRole": "展示软件调度如何与硬件储能协同平抑集群级 dP/dt 斜率"
        },
        {
            "figure_id": "图 15-7",
            "number": 7,
            "file": "ch15-ashrae-w45-cdu-liquid-cooling-loop.svg",
            "asset": "ch15/ch15-ashrae-w45-cdu-liquid-cooling-loop.svg",
            "section": "15.3.1",
            "caption": "ASHRAE TC 9.9 W45（45°C 温水进液）冷板—分集水器—CDU—干冷器无冷机换热回路与流量核算（算例 15-4）",
            "source_refs": ["references/text/ashrae-liquid-cooling.txt", "references/files/specs/nvidia-rubin-system.md"],
            "TeachingRole": "建立风冷物理极限（1,900 CFM/砖）与 132 kW 液冷机柜 Q = m_dot·c_p·ΔT 流量账本"
        },
        {
            "figure_id": "图 15-8",
            "number": 8,
            "file": "ch15-logicfolding-power-density-paradox.svg",
            "asset": "ch15/ch15-logicfolding-power-density-paradox.svg",
            "section": "15.3.2",
            "caption": "LogicFolding 降压节能（0.85 V → 0.55 V）与面积缩减更快导致的功率密度反升（0.75/0.60 = 1.25x，算例 15-5）",
            "source_refs": ["case-studies/logicfolding-energy.md", "manuscripts/ch15/figure-data.json"],
            "TeachingRole": "阐明总功耗下降不等于散热变易：投影面积压缩 40% 会将单位面积热流密度推高 25%"
        },
        {
            "figure_id": "图 15-9",
            "number": 9,
            "file": "ch15-thermal-throttling-straggler-amplification.svg",
            "asset": "ch15/ch15-thermal-throttling-straggler-amplification.svg",
            "section": "15.3.3",
            "caption": "单颗过热 GPU 触发降频 15% 如何通过同步 AllReduce 屏障拖慢整组 1,024 张卡的有效吞吐",
            "source_refs": ["research/2026-infra-survey/verify_tapas_reading.py", "manuscripts/ch15/figure-data.json"],
            "TeachingRole": "揭示局部冷板微水道堵塞或进液偏温如何经集合通信屏障放大为全局 15% 算力损失"
        },
        {
            "figure_id": "图 15-10",
            "number": 10,
            "file": "ch15-tapas-thermal-power-routing-breakeven.svg",
            "asset": "ch15/ch15-tapas-thermal-power-routing-breakeven.svg",
            "section": "15.3.3",
            "caption": "TAPAS（ASPLOS '25）热感知与功率封顶重配置（[102, 86] kW 削峰至 ≤ 100 kW，3.0 s 停顿在 30 请求处回本，算例 15-6）",
            "source_refs": ["research/2026-infra-survey/verify_tapas_reading.py", "manuscripts/ch15/figure-data.json"],
            "TeachingRole": "用受控算例验证功率/温度与延迟双约束下的路由可行集及重配置回本阈值"
        },
        {
            "figure_id": "图 15-11",
            "number": 11,
            "file": "ch15-multi-datacenter-dci-optical-latency.svg",
            "asset": "ch15/ch15-multi-datacenter-dci-optical-latency.svg",
            "section": "15.4.1",
            "caption": "跨 1,000 km 广域光网（RTT = 10 ms，BDP = 2.0 GB）的传播延迟与四类并行维度的跨城可迁移性边界",
            "source_refs": ["manuscripts/ch15/figure-data.json"],
            "TeachingRole": "从光纤折射率物理极限推导为何 TP/EP 严禁跨城而 PP/低频 DP 可以跨城"
        },
        {
            "figure_id": "图 15-12",
            "number": 12,
            "file": "ch15-cross-dc-pipeline-and-hierarchical-dp.svg",
            "asset": "ch15/ch15-cross-dc-pipeline-and-hierarchical-dp.svg",
            "section": "15.4.2",
            "caption": "跨园区流水线并行（Cross-DC PP）与园区内 NVLink/RoCE + 园区际 DWDM 分层数据并行时序",
            "source_refs": ["manuscripts/ch15/figure-data.json"],
            "TeachingRole": "推导用微批次计算掩盖 10 ms 广域网 RTT 的临界条件与两级规约流量缩减比"
        },
        {
            "figure_id": "图 15-13",
            "number": 13,
            "file": "ch15-diloco-inner-outer-optimization.svg",
            "asset": "ch15/ch15-diloco-inner-outer-optimization.svg",
            "section": "15.4.3",
            "caption": "DiLoCo 内循环 H = 500 步独立更新与外循环 Nesterov 伪梯度同步将跨城带宽需求压缩 500 倍的机制（算例 15-7）",
            "source_refs": ["manuscripts/ch15/figure-data.json"],
            "TeachingRole": "剖析外循环伪梯度同步如何换取 500 倍广域网带宽缩减及其额外显存代价"
        },
        {
            "figure_id": "图 15-14",
            "number": 14,
            "file": "ch15-streaming-diloco-chunked-sync.svg",
            "asset": "ch15/ch15-streaming-diloco-chunked-sync.svg",
            "section": "15.4.3",
            "caption": "全量阻塞式外循环同步与 Streaming DiLoCo 分片流水重叠同步的广域网链路利用率对比",
            "source_refs": ["manuscripts/ch15/figure-data.json"],
            "TeachingRole": "展示如何将第 H 步的突发全量伪梯度通信摊平至整个内循环窗口以消除 DCI 拥塞"
        },
        {
            "figure_id": "图 15-15",
            "number": 15,
            "file": "ch15-single-campus-vs-multi-campus-decision.svg",
            "asset": "ch15/ch15-single-campus-vs-multi-campus-decision.svg",
            "section": "15.4.3",
            "caption": "单园区等待变电站扩容 vs. 多园区跨城协同训练的完成工期与成本交点决策树",
            "source_refs": ["research/draft11-outline-audit/report.md", "manuscripts/ch15/figure-data.json"],
            "TeachingRole": "联合并网延期折旧与广域网专线租金给出单园区与多园区方案的定量切换准则"
        },
        {
            "figure_id": "图 15-16",
            "number": 16,
            "file": "ch15-power-thermal-wan-co-design-summary.svg",
            "asset": "ch15/ch15-power-thermal-wan-co-design-summary.svg",
            "section": "15.5.2",
            "caption": "从芯片结温、机柜液冷、变电站功率平滑到跨地域光网的吉瓦级 AI 基础设施全景图",
            "source_refs": ["references/files/specs/nvidia-rubin-system.md", "manuscripts/ch15/figure-data.json"],
            "TeachingRole": "收束第 15 章供电、散热、瞬态功率与跨城光网的四层物理协同体系"
        },
        {
            "figure_id": "图 15-17",
            "number": 17,
            "file": "ch15-single-phase-vs-two-phase-microfluidics.svg",
            "asset": "ch15/ch15-single-phase-vs-two-phase-microfluidics.svg",
            "section": "15.6.1",
            "caption": "单相冷板、两相相变浸没/冷板与硅背面微通道直冷的热阻网络（R_TIM, R_conv）与千瓦级芯片结温边界",
            "source_refs": ["references/text/ashrae-liquid-cooling.txt", "experiments/ch15/15-01/run.py"],
            "TeachingRole": "对比显热、汽化潜热与硅背面微流道消除 R_TIM=0.02 °C/W 带来的 +40 °C 进液温度提升"
        },
        {
            "figure_id": "图 15-18",
            "number": 18,
            "file": "ch15-campus-bess-peak-shaving-and-arbitrage.svg",
            "asset": "ch15/ch15-campus-bess-peak-shaving-and-arbitrage.svg",
            "section": "15.6.2",
            "caption": "百兆瓦级园区储能（BESS）在 50 ms 训练脉冲平滑、日内峰谷电价套利与电网掉电备电中的三重能量流",
            "source_refs": ["references/files/specs/nvidia-rubin-system.md", "experiments/ch15/15-02/run.py"],
            "TeachingRole": "量化 100 MWh 园区储能在毫秒平抑、分钟调频与小时峰谷套利中的年收益与投资回收期"
        },
        {
            "figure_id": "图 15-19",
            "number": 19,
            "file": "ch15-kareus-dynamic-static-energy-consolidation.svg",
            "asset": "ch15/ch15-kareus-dynamic-static-energy-consolidation.svg",
            "section": "15.6.3",
            "caption": "Kareus（OSDI '26）动静态能耗分解：低负载下全卡浅度降频的高静态漏电税 vs. 负载紧缩合并 + 空闲卡深度休眠",
            "source_refs": ["research/2026-infra-survey/reading-osdi-2026.md", "experiments/ch15/15-02/run.py"],
            "TeachingRole": "证明在 40% 负载下紧缩合并休眠可将单卡等效功耗从 388 W 压至 280–289 W"
        },
        {
            "figure_id": "图 15-20",
            "number": 20,
            "file": "ch15-dips-wan-backbone-parameter-sync.svg",
            "asset": "ch15/ch15-dips-wan-backbone-parameter-sync.svg",
            "section": "15.6.3",
            "caption": "Di-PS（NSDI '26）跨地域训练的广域网链路段带宽调度与层间异步梯度传输重叠",
            "source_refs": ["research/2026-infra-survey/reading-nsdi-2026.md", "experiments/ch15/15-02/run.py"],
            "TeachingRole": "展示异构跨集群训练如何通过异步参数服务器与伪梯度阻尼容忍百毫秒级 WAN RTT"
        },
        {
            "figure_id": "图 15-21",
            "number": 21,
            "file": "ch15-embodied-carbon-vs-operational-carbon-lifecycle.svg",
            "asset": "ch15/ch15-embodied-carbon-vs-operational-carbon-lifecycle.svg",
            "section": "15.6.3",
            "caption": "AI 数据中心 3–5 年生命周期内隐含碳（Embodied Carbon：晶圆制造与基建）与运行碳（Operational Carbon：发电结构）的权衡曲线",
            "source_refs": ["manuscripts/ch15/figure-data.json"],
            "TeachingRole": "阐明先进制程晶圆隐含碳与电网碳强度 CI(t) 在园区全生命周期中的权衡"
        },
        {
            "figure_id": "图 15-22",
            "number": 22,
            "file": "ch15-gigawatt-campus-master-engineering-ledger.svg",
            "asset": "ch15/ch15-gigawatt-campus-master-engineering-ledger.svg",
            "section": "15.6.3",
            "caption": "吉瓦级 AI 园区从电网并网、储能平滑、微通道液冷到跨城光网协同的完整工程总账本",
            "source_refs": ["experiments/ch15/15-01/run.py", "experiments/ch15/15-02/run.py"],
            "TeachingRole": "汇总结温、液冷、储能套利、动静态能耗与跨城协同训练的工程总账本"
        }
    ]
    (HERE / 'figure-index.json').write_text(json.dumps(fig_index, ensure_ascii=False, indent=2) + '\n')

    md = HERE.parent / '15-吉瓦级 AI 数据中心、电力热学与跨地域协同训练.md'
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

        work = re.sub(r'!\[([^\]]*)\]\((ch15/[^)]+)\)', repl_img, text_md)
        body = markdown.markdown(work, extensions=['tables', 'fenced_code', 'footnotes', 'toc'])
        html = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>第 15 章：吉瓦级 AI 数据中心、电力热学与跨地域协同训练</title></head><body>{body}</body></html>\n'
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
    print(f'Built {len(outputs)} figure files for Chapter 15; 0 layout warnings.')


if __name__ == '__main__':
    main()
