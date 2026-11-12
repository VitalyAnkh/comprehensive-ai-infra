#!/usr/bin/env python3
"""Build all 26 publication-grade figures (420pt) and validation manifests for Chapter 17."""
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


def fig_17_1_ppo_vs_grpo(ex: Exporter) -> None:
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, '经典 PPO 四模型架构', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, 'Actor（16Ψ）+ Critic（16Ψ）\n双训练模型均含优化器状态', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, 'Ref（2Ψ）+ Reward（2Ψ）\n双前向打分模型常驻显存', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '总状态 36Ψ（32B 占 1,152 GB）\n跨四套异构并行图频繁通信', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, 'Critic 占据近半训练显存与算力', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'GRPO / DAPO 组相对优势', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '省去 Critic 与神经 RM\n仅留 Actor（16Ψ）+ Ref（2Ψ）', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '每提示词组采样 G=8–64 条轨迹\n用组内均值/标准差算相对优势', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '模型状态省 50%（降至 576 GB）\n但自回归 Rollout 膨胀 G 倍', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '省去训练态显存，压力转向 Rollout', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch17-ppo-four-model-vs-grpo-memory-ledger')


def fig_17_2_rollout_dominance(ex: Exporter) -> None:
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'RLVR（可验证奖励强化学习）单步迭代时间线分解', color='blue')
    stages = [
        (0.03, 0.58, 0.36, 0.24, '① 自回归 Rollout 生成\n长思维链 16K–64K 词元\n占单步墙钟 70%–90%', 'orange'),
        (0.42, 0.58, 0.26, 0.24, '② 沙箱规则验证\n代码单测 / Lean4\n占单步 5%–15%', 'teal'),
        (0.71, 0.58, 0.26, 0.24, '③ Ref 前向 + 训练\n计算密集高 MFU\n占单步 10%–20%', 'blue'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.39, 0.70), (0.42, 0.70))
    arrow(ax, (0.68, 0.70), (0.71, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '为何 Rollout 主导 RLVR 墙钟时间？（算术强度反差）', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '阶段 ①：Decode 逐词元生成\n每吐 1 词元需搬移全量权重与 KV\n算术强度极低，受限于 HBM 带宽', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '阶段 ③：Prefill 与反向传播\n全序列并行矩阵乘充盈 Tensor 核\n同词元量吞吐比 Decode 快 10–30x', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '加速 RLVR 的核心在于削减长尾 Rollout 气泡与提升生成吞吐', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-rlvr-iteration-timeline-rollout-dominance')


def fig_17_3_verl_hybrid_engine(ex: Exporter) -> None:
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'veRL（HybridFlow）3D-HybridEngine 训推共置零冗余重分片', color='teal')
    stages = [
        (0.04, 0.58, 0.28, 0.24, '① 训练态 3D 切分\nTP×PP×DP 并行\nZeRO 分片省显存', 'blue'),
        (0.36, 0.58, 0.28, 0.24, '② 微分组 AllGather\n仅在小重组域 g 内\n35 ms 重组推理权', 'teal'),
        (0.68, 0.58, 0.28, 0.24, '③ 生成态并发引擎\n小 TP 跑多路实例\n生成完释放完整权', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.36, 0.70))
    arrow(ax, (0.64, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '消除训练态与推理态的并行拓扑冲突（零双份权重常驻）', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, '朴素共置或全网广播重分片\n要么常驻两份权重挤占显存\n要么跨全集群广播阻塞 1.575 s', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '3D-HybridEngine 正交子网格对齐\n重分片通信局限在节点内 NVLink\n切换降至 0.035 s（提速 45.0x）', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '用瞬态 NVLink 组内 AllGather 取代常驻冗余副本', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-verl-3d-hybrid-engine-zero-redundancy-reshard')


def fig_17_4_colocated_vs_disagg(ex: Exporter) -> None:
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, '训推共置（veRL 同池切换）', color='blue', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '同一批同构 GPU 分时复用\n先跑 Rollout 再切回 Learner', 'blue', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '优势：无跨池全量权重搬移\n适合同构算力与中等长度生成', 'teal', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '局限：长尾生成期全集群陪等\n且推理/训练阶段互斥无法重叠', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '时分复用：受制于同步长尾气泡', size=11, ha='center', weight='bold', color=BORDER_COL['blue'])

    card(ax, 0.52, 0.04, 0.46, 0.92, '训推分离与错峰池化（Weave）', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '推理池(H20)与训练池(H800)\n按算术强度物理分池扩展', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, 'Weave (OSDI 26) 互补错峰\nA(360/120s) 配 B(120/360s)', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '峰值仅 216 H20 + 152 H800\n异步流水消除跨阶段空等', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '空分异步 + 多作业互补压平波峰', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch17-colocated-vs-disaggregated-rl-architecture')


def fig_17_5_long_cot_bubble(ex: Exporter) -> None:
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, '长思维链（Long-CoT）幂律分布', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '简单数学/代码题 1K–2K 词元\n在首段 15% 时间内快速结束', 'teal', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '困难反思回溯轨迹达 32K–64K\n仅占批次 5% 却拖延 10 倍时长', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '批次完成时间由最慢轨迹决定\nT_batch = max T_decode(L_i)', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '5% 极长轨迹主导 80% 墙钟耗时', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, '同步批次尾部的 GPU 算力坍塌', color='purple', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '随 90% 短轨迹陆续吐出 EOS\n各 DP 实例活跃 Batch 跌至 1–2', 'orange', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '已完成实例受同步屏障阻塞\n数千张 GPU 空转等待尾轨迹', 'orange', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '全集群有效 MFU 跌破 10%–15%\n成为长思维链 RL 核心瓶颈', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '世代尾气泡吞噬逾半集群算力', size=11, ha='center', weight='bold', color=BORDER_COL['purple'])
    ex.save(fig, 'ch17-long-cot-power-law-straggler-bubble')


def fig_17_6_rollpacker(ex: Exporter) -> None:
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'RollPacker（NSDI 26）：尾部轨迹重分批、流式训练与无偏归一化', color='teal')
    stages = [
        (0.04, 0.58, 0.28, 0.24, '① 尾部拼箱与扩并行\n跨实例归拢长尾轨迹\n闲置卡转 TP/CP 加速', 'teal'),
        (0.36, 0.58, 0.28, 0.24, '② 严防投机长度偏置\n双槽竞争使长解率\n从 1/2 跌至 1/4', 'orange'),
        (0.68, 0.58, 0.28, 0.24, '③ 流式训练重叠\n就绪块提前送训\n60s 周期缩至 55s', 'blue'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.36, 0.70))
    arrow(ax, (0.64, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '算例 17-6：不均等微批下“按微批均值平均”vs.“按总词元归一化”', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, '错误：按微批独立求均值再平均\n微批 1(1项:1) · 微批 2(3项:3)\n(1 + 3)/2 = 2.0（放大微批 1）', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '正确：按全局总词元数加权归一化\n汇总 4 项梯度和 (1 + 9) 除以 4\n(1 + 9)/4 = 2.5（严格无偏真值）', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '动态拼箱改变微批词元数时，必须用全局有效总词元数做梯度分母', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-rollpacker-tail-migration-and-cp-scaling')


def fig_17_7_seer_speculative(ex: Exporter) -> None:
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'Seer（OSDI 26）：分而治之上下文缓存与组内同源投机解码', color='blue')
    stages = [
        (0.04, 0.58, 0.28, 0.24, '① 细粒度分段生成\n长轨迹按固定块切分\n均衡各卡 KV 占用', 'blue'),
        (0.36, 0.58, 0.28, 0.24, '② 组内后缀字典树\n同提示词已完短轨迹\n构建免模型草稿树', 'teal'),
        (0.68, 0.58, 0.28, 0.24, '③ 尾部自适应投机\n对剩余长尾轨迹开启\n多步树验证加速吐词', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.36, 0.70))
    arrow(ax, (0.64, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '利用 GRPO 组采样（G=8–64）天然同源性实现零额外显存投机加速', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '传统独立草稿模型投机\n需额外部署并同步训练 Draft 模型\n初期大 Batch 下反而拖慢吞吐', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, 'Seer / DAS 自适应尾部投机\n仅在尾部小 Batch 期启用组内草稿\n长尾生成延迟削减 40%–60%', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '利用同组兄弟轨迹的高重合步骤，为尾部 5% 慢轨迹提供免模型草稿', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-seer-divided-and-conquered-speculative-rollout')


def fig_17_8_partial_rollout(ex: Exporter) -> None:
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, '全同步屏障 Rollout', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '第 v 步必须集齐当前批次全部\nN 条完整轨迹才可启动训练', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '长尾未完轨迹阻塞整轮迭代\n或被强行截断沦为无效样本', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '训练池与推理池交替空置\n端到端硬件利用率 < 45%', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '全同步屏障与长思维链天然冲突', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, '陈旧度有界异步部分采样', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '攒够 B 条完成轨迹即刻送训\n未完长轨迹保留 KV 跨步续生', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '轨迹携带生成策略版本标签\n控制偏移 τ = v - v_old ≤ τ_max', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '完整保留 64K 长轨迹且零气泡\n集群有效利用率跃升至 > 88%', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '跨步续生 + 有界陈旧度消除尾气泡', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch17-staleness-bounded-partial-rollout-pipeline')


def fig_17_9_importance_sampling(ex: Exporter) -> None:
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '异步偏离策略（Off-Policy）下的重要性采样比率与近端截断', color='blue')
    stages = [
        (0.04, 0.58, 0.28, 0.24, '① 策略版本漂移\n旧策略采样轨迹\n送交新策略训练', 'orange'),
        (0.36, 0.58, 0.28, 0.24, '② 逐词元重要性比\nr_t = π_θ / π_old\n修正行为分布偏移', 'blue'),
        (0.68, 0.58, 0.28, 0.24, '③ 双重安全护栏\nclip(r_t, 1±ε)\n滤除 τ > τ_max 样本', 'teal'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.36, 0.70))
    arrow(ax, (0.64, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '分段跨版本轨迹（Partial Rollout）的逐段旧概率记录铁律', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '跨步续生轨迹由多版本拼接而成\n前段由 v-1 生成，后段由 v 生成\n严禁用单一旧模型统一重算 π_old', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '必须在生成每个词元当下同步记录\n真实行为对数概率 log π_behav\n确保重要性比率 r_t 分母绝对真实', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '随词元流持久化真实行为 logprob 是异步 Partial Rollout 收敛的根基', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-off-policy-importance-sampling-clipping')


def fig_17_10_robustrl(ex: Exporter) -> None:
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'RobustRL（OSDI 26）：面向训推异构角色的非对称容错恢复', color='teal')
    roles = [
        (0.04, 0.58, 0.44, 0.24, '① Rollout 节点故障（无状态）\n无需回滚检查点，仅替换故障机\n从保存的工具轮次轨迹续接', 'teal'),
        (0.52, 0.58, 0.44, 0.24, '② Learner 节点故障（有状态）\n严禁丢弃本轮已耗时 300s 轨迹\n仅检测恢复并重算 60s 更新步', 'blue'),
    ]
    for x, y, w, h, label, role in roles:
        box(ax, x, y, w, h, label, role, size=11)

    card(ax, 0.02, 0.03, 0.96, 0.42, '算例 17-7：训练侧故障下整步重做 vs. 保留轨迹重做更新', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, '整步重做（丢弃已生成 Rollout）\n300s 采样 + 30s 失败 + 100s 检复\n+ 300s 重采 + 60s 更新 = 790 s', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, 'RobustRL 保留已完成 300s 轨迹\n300s 采样 + 30s 失败 + 100s 检复\n+ 60s 更新 = 490 s（净省 300 s）', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '将耗时的 Rollout 轨迹持久化为事务边界，阻断训练侧故障向上游蔓延', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-robustrl-role-based-fault-isolation')


def fig_17_11_distrs(ex: Exporter) -> None:
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'DistRS（NSDI 26）：解耦奖励验证流式重叠与条件剩余时间调度', color='blue')
    stages = [
        (0.04, 0.58, 0.28, 0.24, '① 生成验证流式重叠\n短轨迹一吐出 EOS\n即刻异步送沙箱判题', 'teal'),
        (0.36, 0.58, 0.28, 0.24, '② 条件剩余时间预测\n用 E[S - e | S > e]\n替代朴素 E[S] - e', 'blue'),
        (0.68, 0.58, 0.28, 0.24, '③ 超期死循环裁剪\n守住 216,000 词元\n沉没工作量红线', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.36, 0.70))
    arrow(ax, (0.64, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '算例 17-8：9 个 1s 任务 + 1 个 100s 任务在 e=10s 时的剩余时间反差', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '错误朴素估计 max(E[S] - e, 0)\n无条件均值 E[S] = 10.9 s\n在 e=10 s 时误判仅剩 0.9 s', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '真实条件剩余 E[S - 10 | S > 10]\n存活过 10s 必为 100s 重尾任务\n真实剩余高达 90.0 s（差 100x）', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '重尾验证分布具有“越老越慢”特性，必须按条件生存分布调度与抢占', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-distrs-disaggregated-reward-sandbox-streaming')


def fig_17_12_prm_tree_cow(ex: Exporter) -> None:
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, '独立复制分支 KV（显存膨胀）', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '推理时树搜索（MCTS / Beam）\n从父节点派生 B 个候选推理步', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '为每条子分支物理深拷贝整段\n历史前缀与中间步骤 KV 缓存', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '64 分支搜索使显存占用膨胀 8x+\n且大量带宽消耗于 Memcpy', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '物理深拷贝制约宽分支搜索', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'PagedAttention 写时复制树', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '树节点分支时仅复制逻辑页表\n物理 KV 块引用计数 ref_cnt += B', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '新生成词元写入独立新页帧\n剪枝淘汰分支直接递减 ref_cnt', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '叠加树注意力单次扫过公共祖先\n显存与带宽开销双双降低 70%+', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '零拷贝分支派发 + 引用计数瞬时回收', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch17-test-time-scaling-prm-tree-search-cow-kv')


def fig_17_13_reward_barrier(ex: Exporter) -> None:
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'Breaking the Reward Barrier（OSDI 26）：投机思维树探索', color='blue')
    stages = [
        (0.04, 0.58, 0.28, 0.24, '① 过程奖励屏障\n每步扩展后需等待\n重型 PRM 评分返回', 'orange'),
        (0.36, 0.58, 0.28, 0.24, '② 策略先验预判\n用 Actor 置信度\n预挑高潜子节点', 'teal'),
        (0.68, 0.58, 0.28, 0.24, '③ 投机并行展开\n与 PRM 打分并发\n分出后原地剪枝', 'blue'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.36, 0.70))
    arrow(ax, (0.64, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '串行“生成—等分—再生成”vs. 投机重叠树搜索的端到端延迟', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '传统同步 PRM 树搜索\n每层思维步均被 PRM 前向打断\n搜索深度 D 层累加 D 次评分停顿', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '投机思维树（Speculative ToT）\n将 PRM 评分隐藏在下层生成后\n树搜索端到端延迟降 1.8x–2.6x', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '把算子级的投机解码思想上升到思维树节点级的异步超前展开', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-speculative-tot-reward-barrier-bypass')


def fig_17_14_multi_turn_tool_rl(ex: Exporter) -> None:
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, '阻塞占卡等待外部工具', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, 'SWE-bench / 搜索智能体在生成\n工具调用后等待沙箱编译或网页', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '外部环境响应长达 2–30 秒\n期间数万词元 KV 滞留 GPU 显存', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, 'GPU 因显存满载无法接纳新轨迹\n算力在 I/O 等待中白白空转', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '慢速环境交互挤占宝贵 HBM', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, '状态挂起换出与协程复用', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '触发工具调用瞬间挂起会话\n经 PCIe 异步将 KV 卸至 DRAM', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '立即释放 HBM 页帧给就绪轨迹\n单卡并发活跃环境数扩大 5–10x', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '沙箱返回结果时预取回填 KV\n追加环境观测无缝续跑下一轮', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '将多轮工具交互改造为无阻塞协程', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch17-multi-turn-tool-env-rl-state-suspension')


def fig_17_15_hetrl(ex: Exporter) -> None:
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'HetRL：面向异构推理卡与训练卡的最优算力配比方程', color='blue')
    stages = [
        (0.04, 0.58, 0.28, 0.24, '① 带宽型推理池\nH20 / L40S 等卡\n高容量比跑生成', 'teal'),
        (0.36, 0.58, 0.28, 0.24, '② 算力型训练池\nH800 / B200 超节点\n高 FLOPS 跑反向更新', 'blue'),
        (0.68, 0.58, 0.28, 0.24, '③ 流水节拍精准匹配\nT_rollout(N_inf) ≈\nT_train(N_tr) 零互等', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.36, 0.70))
    arrow(ax, (0.64, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '异构配比准则：N_inf / N_tr = (W_inf / η_inf) / (W_tr / η_tr)', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '全用旗舰训练卡跑同步 RL\n80% 时间在跑低算术强度 Decode\n高算力 Tensor Core 长期闲置', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '按吞吐速率比配置 3:1–5:1 异构池\n训练卡持续高负载跑密集矩阵乘\n单次 RL 迭代硬件 TCO 降低 35%+', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '让高带宽比卡做生成、高算力比卡做更新，消除硬件特性的错配税', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-hetrl-heterogeneous-inference-training-matching')


def fig_17_16_dapo_filter(ex: Exporter) -> None:
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'DAPO 动态采样过滤：消除全对（r=1）与全错（r=0）零梯度浪费', color='teal')
    groups = [
        (0.04, 0.58, 0.28, 0.24, '① 过易提示词(全对)\nG 条轨迹全 r=1\n方差 σ=0 → A=0', 'orange'),
        (0.36, 0.58, 0.28, 0.24, '② 过难提示词(全错)\nG 条轨迹全 r=0\n方差 σ=0 → A=0', 'orange'),
        (0.68, 0.58, 0.28, 0.24, '③ 有效区分提示词\n0 < 正确数 < G\n提供有效策略梯度', 'teal'),
    ]
    for x, y, w, h, label, role in groups:
        box(ax, x, y, w, h, label, role, size=11)

    card(ax, 0.02, 0.03, 0.96, 0.42, 'RollPacker 投机超采（160×10 = 1,600 选 1,024）保障有效微批充盈', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, '静态采样 128 提示词 × 8 = 1,024\n剔除 30% 零方差组后仅剩 ~716 条\n训练微批欠载且梯度方差剧增', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '投机超采 160×10 = 1,600 (+56%)\n在线滤除零方差组并凑齐 1,024 条\n每步训练 100% 皆为有效非零梯度', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '结合投机超额采样与在线方差过滤，但须控制槽位竞争避免长度偏置', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-dapo-dynamic-sampling-zero-variance-filter')


def fig_17_17_throughput_waterfall(ex: Exporter) -> None:
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '大规模 RLVR 系统四阶有效吞吐跃迁瀑布图', color='blue')
    steps = [
        (0.03, 0.58, 0.21, 0.24, '① 朴素同步基线\n双份权+长尾等\n1.0x 基准吞吐', 'orange'),
        (0.27, 0.58, 0.21, 0.24, '② 零冗余重分片\nveRL 混合引擎\n跃升至 ~1.8x', 'blue'),
        (0.51, 0.58, 0.21, 0.24, '③ 尾部拼箱投机\n拼箱+同源投机\n跃升至 ~3.2x', 'teal'),
        (0.75, 0.58, 0.22, 0.24, '④ 异步部分采样\nPartial+DistRS\n达 5.0x–6.5x', 'purple'),
    ]
    for x, y, w, h, label, role in steps:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.24, 0.70), (0.27, 0.70))
    arrow(ax, (0.48, 0.70), (0.51, 0.70))
    arrow(ax, (0.72, 0.70), (0.75, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '系统加速与数学等价性的双重铁律', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '系统优化潜藏的统计偏置陷阱\n• 槽位竞争偏向短解 (1/2→1/4)\n• 微批均值直接平均偏置 (2.0)', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '无偏收敛保障三件套\n• 全局总词元归一化（真值 2.5）\n• 逐词元行为概率 + 陈旧门限', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '脱离无偏梯度与重要性采样校准的工程提速会导致策略退化', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-rl-post-training-throughput-roofline')


def fig_17_18_closed_loop(ex: Exporter) -> None:
    fig, ax = canvas(4.15)
    card(ax, 0.02, 0.03, 0.96, 0.94, '第 17 章 RLVR 后训练与推理时计算扩展工程总账本', color='blue')
    grid = [
        (0.05, 0.66, 0.42, 0.20, '① 算法与显存账本（17.1）\nGRPO 省 50% 状态(576 GB)\nRollout 占 70%–90% 耗时', 'blue'),
        (0.53, 0.66, 0.42, 0.20, '② 训推拓扑与池化（17.2）\nveRL 零冗余重分片引擎\nWeave 错峰(216 H20/152 H800)', 'teal'),
        (0.53, 0.37, 0.42, 0.20, '③ 攻克长尾木桶（17.3）\nRollPacker 归一化(2.5 vs 2.0)\nSeer 投机·RobustRL 省 300s', 'purple'),
        (0.05, 0.37, 0.42, 0.20, '④ 验证与推理扩展（17.4）\nDistRS 条件剩余(90s vs 0.9s)\nCoW 树搜索与投机越障', 'orange'),
    ]
    for x, y, w, h, label, role in grid:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.47, 0.76), (0.53, 0.76))
    arrow(ax, (0.74, 0.66), (0.74, 0.57))
    arrow(ax, (0.53, 0.47), (0.47, 0.47))
    arrow(ax, (0.26, 0.57), (0.26, 0.66))
    box(ax, 0.05, 0.08, 0.90, 0.21, '实验：experiments/ch17/17-01 & 17-02（GRPO 省 50% 状态、\nRobustRL 省 300s、DistRS 216K 词元、跨洲 14,745x 压流）', 'teal', size=11)
    ex.save(fig, 'ch17-rlvr-and-test-time-compute-closed-loop')


def fig_17_19_geo_async_rollout(ex: Exporter) -> None:
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'Prime-RL：跨地域闲置算力异步 Rollout 收割（算例 17-9）', color='blue')
    stages = [
        (0.04, 0.58, 0.28, 0.24, '① 跨洲异构推理池\n收割闲置 H20/4090\n异步跑长尾生成', 'teal'),
        (0.36, 0.58, 0.28, 0.24, '② 10 B/tok 窄带回传\n传词元 ID 与概率\n全批仅 160.0 MiB', 'blue'),
        (0.68, 0.58, 0.28, 0.24, '③ 中心训练池过滤\n校验陈旧度 τ ≤ 2\n与序列几何 IS 比', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.36, 0.70))
    arrow(ax, (0.64, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '10 Gbps 广域网下传激活/KV vs. 传紧凑轨迹元数据的百倍反差', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '错误：跨机房回传 36 层 KV\n1,024×16K 词元达 2,473.9 GB\n10 Gbps 专线阻塞 1,979.1 s', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '正确：回传 10 B/tok 紧凑三元组\n全批 160.0 MiB（134.2 ms 传完）\n配 FP8 差分权重广播零暴露', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '用 14,745.6x 上行压缩 + 序列几何均值 IS 过滤打通广域异构算力 RL', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-geo-distributed-async-rollout-harvester')


def fig_17_20_reward_hacking_breaker(ex: Exporter) -> None:
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'RLVR 奖励黑客（Reward Hacking）的三维异常症状相图', color='orange')
    symptoms = [
        (0.04, 0.58, 0.28, 0.24, '① 长度失控膨胀\n堆砌无意义伪代码\n均长破 28K 词元', 'orange'),
        (0.36, 0.58, 0.28, 0.24, '② 策略熵急剧坍缩\n退化为高频复读机\n熵跌破 0.05 nats', 'orange'),
        (0.68, 0.58, 0.28, 0.24, '③ KL 散度突增\n偏离参考模型流形\n突破 0.25 熔断线', 'orange'),
    ]
    for x, y, w, h, label, role in symptoms:
        box(ax, x, y, w, h, label, role, size=11)

    card(ax, 0.02, 0.03, 0.96, 0.42, '双级在线统计制动器：软边界超长惩罚 + 硬熔断回滚机制', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '一级软制动（KL > 0.06 或超长）\n倍增 β_KL 并加分段长度罚项\n抑制无信息量的凑字数行为', 'teal', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '二级硬熔断（KL > 0.25 且低熵）\n中止当前步并回滚健康检查点\n隔离污染提示词并调高温度', 'blue', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '监控 (KL, 长度, 策略熵) 联合轨迹是长周期无人值守 RLVR 防崩溃底线', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-rlvr-reward-hacking-kl-divergence-breaker')


def fig_17_21_multi_turn_tree_fork(ex: Exporter) -> None:
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'SWE-RL 多轮环境状态与 KV 树状检查点分叉（算例 17-10）', color='blue')
    stages = [
        (0.04, 0.58, 0.28, 0.24, '① 根容器镜像快照\nOverlayFS + CRIU\n18 s 初始化仅跑1次', 'blue'),
        (0.36, 0.58, 0.28, 0.24, '② 轮次级状态分叉\n4 个中间工具节点\n共享 CoW 内存与页', 'teal'),
        (0.68, 0.58, 0.28, 0.24, '③ 叶节点并行验证\n16 条叶轨迹跑单测\n分段计算轮次优势', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.36, 0.70))
    arrow(ax, (0.64, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '16 条独立线性重跑 vs. 容器+KV 树分叉的沙箱与显存收益', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '16 条独立线性重跑\n沙箱耗时 672.0 s（重复克隆）\n前缀 KV 占 131,072 槽位', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, 'OverlayFS + CoW 树分叉\n沙箱耗时 114.0 s（快 5.89x）\n前缀 KV 降为 20,480（省84.4%）', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '将容器文件系统快照与 Paged KV 页表同步分叉，消除多轮环境重复执行', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-multi-turn-agentic-rlvr-trajectory-tree-forking')


def fig_17_22_async_vtrace_tis(ex: Exporter) -> None:
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '异步离策略 V-trace 与截断重要性采样（TIS）修正', color='teal')
    stages = [
        (0.04, 0.58, 0.28, 0.24, '① 词元级连乘爆炸\n10 步 r=1.25 连乘\n膨胀至 9.31x 失稳', 'orange'),
        (0.36, 0.58, 0.28, 0.24, '② V-trace 双重截断\nρ_t ≤ 1.20 控更新幅\nc_t ≤ 1.00 控回溯迹', 'teal'),
        (0.68, 0.58, 0.28, 0.24, '③ 轮次双层优势\n环境观测掩码置零\n动作段无偏信用分配', 'blue'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.36, 0.70))
    arrow(ax, (0.64, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '无截断连乘 IS vs. V-trace / TIS 在陈旧度 τ=3 下的方差控制', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, '未截断轨迹连乘 IS\n随序列长度指数级发散\n单条离群样本破坏梯度', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, 'V-trace (ρ̄=1.2, c̄=1.0) + 掩码\n严格有界递推价值目标\n支撑 τ ≤ 4 异步多轮收敛', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '外部环境返回词元必须从损失与 IS 比率中掩码剔除，仅对策略动作算梯度', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-asynchronous-off-policy-importance-sampling-clipping')


def fig_17_23_sft_varlen_zigzag(ex: Exporter) -> None:
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '长序列 SFT 无填充变长装箱与二次方计算量均衡（算例 17-11）', color='blue')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '① Pad-to-64K 基线\n均值 4.2K 填至 64K\n有效 MFU 仅 13.6%', 'orange'),
        (0.355, 0.58, 0.29, 0.24, '② cu_seqlens 装箱\n块对角隔离零填充\n受平方偏斜限 41.2%', 'blue'),
        (0.68, 0.58, 0.29, 0.24, '③ LPT + ZigZag 环\n按 ΣL_i² 均衡切块\nMFU 达 56.4% (4.15x)', 'teal'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '跨文档位置重置隔离 + DPO 参考对数概率离线预缓存', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, 'cu_seqlens + RoPE 起点归零\n跳过跨文档空白注意力瓦片\n杜绝多样本拼接信息串扰', 'blue', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, 'DPO 参考概率离线预缓存\n每偏好对仅存 8 B 双标量落盘\n省去 64 GB Ref 显存与 25% 算力', 'purple', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '消除填充词元后，还须按 ΣL_i² 均衡跨 DP 注意力复杂度以消除同步空等', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-sft-varlen-binpacking-cuseqlens-and-quadratic-load-balance')


def fig_17_24_minitron_distill(ex: Exporter) -> None:
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'Minitron 结构化剪枝与 32-bit 位打包 Top-128 蒸馏（算例 17-12）', color='teal')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '① 全词表 BF16 落盘\nQwen3 V=151,936\n200B 词元达 60.8 PB', 'orange'),
        (0.355, 0.58, 0.29, 0.24, '② 32-bit 位打包缓存\n18b ID + 14b FP14\n压至 102.4 TB (593x)', 'teal'),
        (0.68, 0.58, 0.29, 0.24, '③ 在线 NVFP4 教师\n伴飞免百 TB 落盘\n翻转点 E* = 1.88 轮', 'blue'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, 'Tensor Core 128 对齐剪枝 + 稀疏 Top-K 残差桶单纯形恢复', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, '结构化宽度—深度硬件对齐剪枝\n保留维度强制满足 mod 128 = 0\n防 uint16 溢出（18-bit 覆 262K）', 'purple', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '单轮学生选在线 NVFP4 省 46.7%\n多尺寸家族复用 (E=3) 选离线缓存\n净省 10,800 GPU 小时算力账本', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '18-bit ID + 14-bit FP14 单字打包兼顾大词表无损索引与百倍存储压缩', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-minitron-structured-pruning-and-online-vs-offline-distillation')


def fig_17_25_agentic_rl_pool(ex: Exporter) -> None:
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '集群级多轮 Agentic RL 异构沙箱池配比与前缀树 KV 复用', color='blue')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '① GPU 推理生成池\n单轮吐动作 0.8 s\n触发工具即挂起卸出', 'blue'),
        (0.355, 0.58, 0.29, 0.24, '② 微虚机沙箱集群\n编译单测耗时 12 s\n利特尔法则 15:1 配槽', 'teal'),
        (0.68, 0.58, 0.29, 0.24, '③ Radix 前缀树追加\n32 轮仅算新增观测\n省去平方级重复预填', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '同步绑卡苦等慢容器 vs. 异步协程挂起 + 增量树缓存', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '同步阻塞等待最慢容器单测\n尾部 60–90 s 编译拖住整批卡\nGPU 推理与训练池空转达 68%', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '按 L = λ·W 解耦沙箱并发槽位\n配合 DSec/Firecracker 快照池\n让百卡 GPU 始终服务就绪轮次', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '用高密 CPU 微虚机并发槽位吸收环境长尾，释放高成本 GPU 张量算力', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-multi-turn-agentic-rl-prefix-tree-and-sandbox-pool')


def fig_17_26_bounded_staleness(ex: Exporter) -> None:
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '有界陈旧度（Δ_max = 3）离策略校准与第四部分收束（算例 17-13）', color='purple')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '① 有界陈旧度窗口\n允许滞后 Δ ≤ 3 步\n消除 68% 同步气泡', 'teal'),
        (0.355, 0.58, 0.29, 0.24, '② Rényi 方差护栏\nVar(r)=exp(D2)-1\n超界陈旧组自动淘汰', 'blue'),
        (0.68, 0.58, 0.29, 0.24, '③ 有效更新步增益\n单位小时有效步数\n净提速达 2.64x', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '第四部分（第 13–17 章）数据—供电—容错—TEE—后训练全栈咬合', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, '底座支撑：Ch13 3FS 高速供给\nCh14 800V 液冷 · Ch15 持续剖析\nCh16 DSec 微虚机与 TEE-I/O', 'blue', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '进化管线：Ch17 变长 SFT (4.15x)\n32-bit 剪枝蒸馏 (102.4 TB)\n异步 RLVR / Agentic RL (2.64x)', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '有界陈旧度与截断重要性采样在系统吞吐和策略统计效率之间取得最优平衡', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch17-bounded-staleness-off-policy-importance-sampling-stability')


def main() -> None:
    ex = Exporter(HERE)
    fig_17_1_ppo_vs_grpo(ex)
    fig_17_2_rollout_dominance(ex)
    fig_17_3_verl_hybrid_engine(ex)
    fig_17_4_colocated_vs_disagg(ex)
    fig_17_5_long_cot_bubble(ex)
    fig_17_6_rollpacker(ex)
    fig_17_7_seer_speculative(ex)
    fig_17_8_partial_rollout(ex)
    fig_17_9_importance_sampling(ex)
    fig_17_10_robustrl(ex)
    fig_17_11_distrs(ex)
    fig_17_12_prm_tree_cow(ex)
    fig_17_13_reward_barrier(ex)
    fig_17_14_multi_turn_tool_rl(ex)
    fig_17_15_hetrl(ex)
    fig_17_16_dapo_filter(ex)
    fig_17_17_throughput_waterfall(ex)
    fig_17_18_closed_loop(ex)
    fig_17_19_geo_async_rollout(ex)
    fig_17_20_reward_hacking_breaker(ex)
    fig_17_21_multi_turn_tree_fork(ex)
    fig_17_22_async_vtrace_tis(ex)
    fig_17_23_sft_varlen_zigzag(ex)
    fig_17_24_minitron_distill(ex)
    fig_17_25_agentic_rl_pool(ex)
    fig_17_26_bounded_staleness(ex)
    outputs = ex.finish()

    fig_index = [
        {
            "figure_id": "图 17-1",
            "number": 1,
            "file": "ch17-ppo-four-model-vs-grpo-memory-ledger.svg",
            "asset": "ch17/ch17-ppo-four-model-vs-grpo-memory-ledger.svg",
            "section": "17.1.1",
            "caption": "PPO 四模型（Actor/Critic/Ref/RM）与 GRPO/DAPO（消去 Critic + 组采样 G 条轨迹）的显存与计算量对比（算例 17-1）",
            "source_refs": ["manuscripts/ch17/figure-data.json", "experiments/ch17/17-01/run.py"],
            "TeachingRole": "量化 32B 模型下 PPO 1,152 GB（36Ψ）与 GRPO 576 GB（18Ψ）的 50% 状态显存削减"
        },
        {
            "figure_id": "图 17-2",
            "number": 2,
            "file": "ch17-rlvr-iteration-timeline-rollout-dominance.svg",
            "asset": "ch17/ch17-rlvr-iteration-timeline-rollout-dominance.svg",
            "section": "17.1.2",
            "caption": "RLVR 单步训练时间线分解——为何自回归 Rollout 占据 70%–90% 墙钟时间且算术强度极低",
            "source_refs": ["manuscripts/ch17/figure-data.json"],
            "TeachingRole": "从 Decode 访存带宽受限与 Prefill/Backward 算力密集的反差解释 Rollout 主导现象"
        },
        {
            "figure_id": "图 17-3",
            "number": 3,
            "file": "ch17-verl-3d-hybrid-engine-zero-redundancy-reshard.svg",
            "asset": "ch17/ch17-verl-3d-hybrid-engine-zero-redundancy-reshard.svg",
            "section": "17.2.1",
            "caption": "veRL（HybridFlow）3D-HybridEngine 训练态（ZeRO+TP+PP）与生成态（微张量并行）的零冗余内存重分片机制（算例 17-2）",
            "source_refs": ["manuscripts/ch17/figure-data.json", "calculations/results/ch17-posttrain-rlvr-distill.json"],
            "TeachingRole": "解析子网格微分组 AllGather 如何将切换耗时从 1.575 s 压至 0.035 s（提速 45.0x）"
        },
        {
            "figure_id": "图 17-4",
            "number": 4,
            "file": "ch17-colocated-vs-disaggregated-rl-architecture.svg",
            "asset": "ch17/ch17-colocated-vs-disaggregated-rl-architecture.svg",
            "section": "17.2.2",
            "caption": "训推共置（veRL）与物理训推分离及互补错峰池化（Weave OSDI '26，算例 17-3）对比",
            "source_refs": ["case-studies/rl-scheduling-and-recovery.md", "experiments/ch17/17-01/run.py"],
            "TeachingRole": "复算 Weave 互补作业（360/120 s 与 120/360 s）共享 216 H20 与 152 H800 的削峰收益"
        },
        {
            "figure_id": "图 17-5",
            "number": 5,
            "file": "ch17-long-cot-power-law-straggler-bubble.svg",
            "asset": "ch17/ch17-long-cot-power-law-straggler-bubble.svg",
            "section": "17.3.1",
            "caption": "长思维链（Long-CoT）输出长度幂律分布引发的同步 Rollout 尾部 GPU 空泡",
            "source_refs": ["case-studies/rollout-tail-and-sampling.md"],
            "TeachingRole": "揭示 5% 超长推理轨迹如何通过同步屏障迫使 90% 已完成 GPU 空转等待"
        },
        {
            "figure_id": "图 17-6",
            "number": 6,
            "file": "ch17-rollpacker-tail-migration-and-cp-scaling.svg",
            "asset": "ch17/ch17-rollpacker-tail-migration-and-cp-scaling.svg",
            "section": "17.3.2",
            "caption": "RollPacker（NSDI '26）尾部存活请求紧缩迁移、双槽位长度偏差（1/2 → 1/4）与微批均值平均（2.0）vs. 全局总词元归一化（2.5）对比（算例 17-5 & 17-6）",
            "source_refs": ["case-studies/rollout-tail-and-sampling.md", "experiments/ch17/17-01/run.py"],
            "TeachingRole": "严格证明不均等微批下按微批均值平均会产生 2.0 vs 2.5 的梯度偏置"
        },
        {
            "figure_id": "图 17-7",
            "number": 7,
            "file": "ch17-seer-divided-and-conquered-speculative-rollout.svg",
            "asset": "ch17/ch17-seer-divided-and-conquered-speculative-rollout.svg",
            "section": "17.3.2",
            "caption": "Seer（OSDI '26）分段治理长尾 Rollout 与组内共享前缀投机解码加速",
            "source_refs": ["research/2026-infra-survey/reading-osdi-2026.md", "case-studies/rollout-tail-and-sampling.md"],
            "TeachingRole": "展示如何利用同组已完成短轨迹构建免模型草稿树加速剩余长尾轨迹"
        },
        {
            "figure_id": "图 17-8",
            "number": 8,
            "file": "ch17-staleness-bounded-partial-rollout-pipeline.svg",
            "asset": "ch17/ch17-staleness-bounded-partial-rollout-pipeline.svg",
            "section": "17.3.3",
            "caption": "同步屏障 Rollout vs. 陈旧度有界（τ ≤ τ_max）异步部分采样（Partial Rollout）的流水线吞吐跃升",
            "source_refs": ["case-studies/rollout-tail-and-sampling.md"],
            "TeachingRole": "允许未完长轨迹保留 KV 跨训练步续生，将集群有效利用率提升至 >88%"
        },
        {
            "figure_id": "图 17-9",
            "number": 9,
            "file": "ch17-off-policy-importance-sampling-clipping.svg",
            "asset": "ch17/ch17-off-policy-importance-sampling-clipping.svg",
            "section": "17.3.3",
            "caption": "异步 RL 中策略版本偏移 Δv = τ 下的重要性采样比率波动与逐词元真实行为概率记录准则",
            "source_refs": ["case-studies/rollout-tail-and-sampling.md"],
            "TeachingRole": "阐明跨版本拼接轨迹必须随词元流持久化生成当下的真实 logprob"
        },
        {
            "figure_id": "图 17-10",
            "number": 10,
            "file": "ch17-robustrl-role-based-fault-isolation.svg",
            "asset": "ch17/ch17-robustrl-role-based-fault-isolation.svg",
            "section": "17.3.3",
            "caption": "RobustRL（OSDI '26）区分无状态 Rollout 故障与有状态 Learner 故障的非对称容错（790 s vs. 490 s，净省 300 s，算例 17-7）",
            "source_refs": ["case-studies/rl-scheduling-and-recovery.md", "experiments/ch17/17-01/run.py"],
            "TeachingRole": "复算保留已完成 Rollout 轨迹将训练侧故障恢复耗时从 790 s 压至 490 s 的定量账本"
        },
        {
            "figure_id": "图 17-11",
            "number": 11,
            "file": "ch17-distrs-disaggregated-reward-sandbox-streaming.svg",
            "asset": "ch17/ch17-distrs-disaggregated-reward-sandbox-streaming.svg",
            "section": "17.4.1",
            "caption": "DistRS（NSDI '26）将沙箱验证与长尾生成流式重叠及基于条件剩余时间 E[S-e | S>e]（90 s vs. 0.9 s）与 216,000 词元阈值的超期裁剪（算例 17-8）",
            "source_refs": ["case-studies/reward-deadlines-and-feedback.md", "experiments/ch17/17-01/run.py"],
            "TeachingRole": "证明重尾验证分布下朴素剩余时间估计（0.9 s）与真实条件期望（90.0 s）的百倍误差"
        },
        {
            "figure_id": "图 17-12",
            "number": 12,
            "file": "ch17-test-time-scaling-prm-tree-search-cow-kv.svg",
            "asset": "ch17/ch17-test-time-scaling-prm-tree-search-cow-kv.svg",
            "section": "17.4.2",
            "caption": "推理时扩展（Best-of-N / PRM 树搜索）利用 PagedAttention 写时复制（CoW）共享父节点 KV 缓存",
            "source_refs": ["manuscripts/ch17/figure-data.json"],
            "TeachingRole": "解析页表引用计数 CoW 与树注意力如何消除树状搜索的物理拷贝与显存膨胀"
        },
        {
            "figure_id": "图 17-13",
            "number": 13,
            "file": "ch17-speculative-tot-reward-barrier-bypass.svg",
            "asset": "ch17/ch17-speculative-tot-reward-barrier-bypass.svg",
            "section": "17.4.2",
            "caption": "Breaking the Reward Barrier（OSDI '26）在等待 PRM 评分期间投机性预展开高置信分支的时序对比",
            "source_refs": ["research/2026-infra-survey/reading-osdi-2026.md", "case-studies/reward-deadlines-and-feedback.md"],
            "TeachingRole": "通过超前投机展开子节点将重型 PRM 评分延迟隐藏在树搜索关键路径之外"
        },
        {
            "figure_id": "图 17-14",
            "number": 14,
            "file": "ch17-multi-turn-tool-env-rl-state-suspension.svg",
            "asset": "ch17/ch17-multi-turn-tool-env-rl-state-suspension.svg",
            "section": "17.4.1",
            "caption": "多轮工具调用 / SWE-bench 强化学习中等待外部环境返回时的 KV 换出挂起与高并发复用",
            "source_refs": ["case-studies/rl-scheduling-and-recovery.md"],
            "TeachingRole": "消除慢速外部沙箱执行期间巨量 KV 缓存滞留 HBM 导致的算力饥饿"
        },
        {
            "figure_id": "图 17-15",
            "number": 15,
            "file": "ch17-hetrl-heterogeneous-inference-training-matching.svg",
            "asset": "ch17/ch17-hetrl-heterogeneous-inference-training-matching.svg",
            "section": "17.2.2",
            "caption": "HetRL 在带宽型推理卡与算力型训练卡之间的最优配比曲线",
            "source_refs": ["case-studies/rl-scheduling-and-recovery.md"],
            "TeachingRole": "建立按生成吞吐与训练吞吐匹配异构卡数的定量公式"
        },
        {
            "figure_id": "图 17-16",
            "number": 16,
            "file": "ch17-dapo-dynamic-sampling-zero-variance-filter.svg",
            "asset": "ch17/ch17-dapo-dynamic-sampling-zero-variance-filter.svg",
            "section": "17.3.2",
            "caption": "DAPO 动态过滤全对（r=1）与全错（r=0）零梯度提示词及 RollPacker 1,600 选 1,024 投机池",
            "source_refs": ["case-studies/rollout-tail-and-sampling.md", "experiments/ch17/17-01/run.py"],
            "TeachingRole": "结合 +56.25% 投机超采池与在线零方差过滤确保训练微批 100% 充盈有效梯度"
        },
        {
            "figure_id": "图 17-17",
            "number": 17,
            "file": "ch17-rl-post-training-throughput-roofline.svg",
            "asset": "ch17/ch17-rl-post-training-throughput-roofline.svg",
            "section": "17.3.3",
            "caption": "大规模 RLVR 系统从朴素同步共置到异步分池+尾部拼箱的四阶有效吞吐跃迁瀑布图",
            "source_refs": ["case-studies/rollout-tail-and-sampling.md"],
            "TeachingRole": "汇总零冗余重分片、尾部拼箱投机与异步部分采样的阶梯收益及无偏性条件"
        },
        {
            "figure_id": "图 17-18",
            "number": 18,
            "file": "ch17-rlvr-and-test-time-compute-closed-loop.svg",
            "asset": "ch17/ch17-rlvr-and-test-time-compute-closed-loop.svg",
            "section": "17.4.2",
            "caption": "测试时搜索合成高价值轨迹 → 沙箱验证 → 异步 RLVR 更新 → 蒸馏回直觉模型的演进总账本",
            "source_refs": ["experiments/ch17/17-01/run.py"],
            "TeachingRole": "收束第 17 章 RLVR 显存账本、训推池化、长尾治理与推理时搜索的工程总图"
        },
        {
            "figure_id": "图 17-19",
            "number": 19,
            "file": "ch17-geo-distributed-async-rollout-harvester.svg",
            "asset": "ch17/ch17-geo-distributed-async-rollout-harvester.svg",
            "section": "17.4.3",
            "caption": "跨地域闲置推理算力异步收割 Rollout 轨迹、Token ID 极窄带宽回传与中心 Learner 重要性采样过滤架构（算例 17-9）",
            "source_refs": ["manuscripts/ch17/figure-data.json", "experiments/ch17/17-02/run.py"],
            "TeachingRole": "复算 10 B/tok 紧凑回传将 1,024×16K 轨迹从 2,473.9 GB 压缩 14,745.6x 至 160.0 MiB 的广域网账本"
        },
        {
            "figure_id": "图 17-20",
            "number": 20,
            "file": "ch17-rlvr-reward-hacking-kl-divergence-breaker.svg",
            "asset": "ch17/ch17-rlvr-reward-hacking-kl-divergence-breaker.svg",
            "section": "17.4.3",
            "caption": "RLVR 训练中奖励黑客（长度失控与熵坍缩）的在线监控指标相图与自适应 KL 制动熔断机制",
            "source_refs": ["manuscripts/ch17/figure-data.json", "experiments/ch17/17-02/run.py"],
            "TeachingRole": "建立基于 (KL 散度, 输出长度, 策略熵) 的两级在线奖励黑客制动与检查点回滚状态机"
        },
        {
            "figure_id": "图 17-21",
            "number": 21,
            "file": "ch17-multi-turn-agentic-rlvr-trajectory-tree-forking.svg",
            "asset": "ch17/ch17-multi-turn-agentic-rlvr-trajectory-tree-forking.svg",
            "section": "17.4.3",
            "caption": "多轮工具交互强化学习（SWE-RL / DeepResearch-RL）中容器环境快照（OverlayFS + CRIU）与 Paged KV 树状分叉架构（算例 17-10）",
            "source_refs": ["manuscripts/ch17/figure-data.json", "experiments/ch17/17-03/run.py"],
            "TeachingRole": "推导 16 条叶分支下容器+KV 树分叉将沙箱 CPU 耗时从 672.0 s 压至 114.0 s（5.89x）并省 84.38% 前缀 KV 的账本"
        },
        {
            "figure_id": "图 17-22",
            "number": 22,
            "file": "ch17-asynchronous-off-policy-importance-sampling-clipping.svg",
            "asset": "ch17/ch17-asynchronous-off-policy-importance-sampling-clipping.svg",
            "section": "17.4.3",
            "caption": "异步离策略强化学习中连乘重要性采样爆炸与 V-trace（ρ̄, c̄ 双重截断）/ TIS 轨迹修正及环境观测掩码机制",
            "source_refs": ["manuscripts/ch17/figure-data.json", "experiments/ch17/17-03/run.py"],
            "TeachingRole": "对比 10 步未截断连乘 IS（9.31x 膨胀）与 V-trace 双重截断递推及环境观测掩码置零的收敛保障"
        },
        {
            "figure_id": "图 17-23",
            "number": 23,
            "file": "ch17-sft-varlen-binpacking-cuseqlens-and-quadratic-load-balance.svg",
            "asset": "ch17/ch17-sft-varlen-binpacking-cuseqlens-and-quadratic-load-balance.svg",
            "section": "17.5.1",
            "caption": "64K 长序列 SFT 从 Pad-to-Max（MFU 13.6%）、cu_seqlens 变长装箱（41.2%）到二次计算量均衡 + ZigZag Ring（56.4%，提速 4.15x）及预缓存 DPO 流水线（算例 17-11）",
            "source_refs": ["calculations/results/ch17-posttrain-rlvr-distill.json", "experiments/ch17/17-04/run.py"],
            "TeachingRole": "证明消除填充后还须按 ΣL_i² 均衡跨 DP 注意力工作量并对称切块才能释放 4.15x 有效吞吐"
        },
        {
            "figure_id": "图 17-24",
            "number": 24,
            "file": "ch17-minitron-structured-pruning-and-online-vs-offline-distillation.svg",
            "asset": "ch17/ch17-minitron-structured-pruning-and-online-vs-offline-distillation.svg",
            "section": "17.5.2",
            "caption": "Minitron 硬件对齐结构化剪枝与 18-bit ID + 14-bit FP14 单字 32-bit 位打包 Top-128 缓存（102.4 TB，压缩 593.5x）对比在线 NVFP4 教师的 1.88 轮翻转点（算例 17-12）",
            "source_refs": ["calculations/results/ch17-posttrain-rlvr-distill.json", "experiments/ch17/17-05/run.py"],
            "TeachingRole": "揭示大词表 V>65,535 下 uint16 溢出风险，确立 32-bit 紧凑打包与在线低比特伴飞的经济边界"
        },
        {
            "figure_id": "图 17-25",
            "number": 25,
            "file": "ch17-multi-turn-agentic-rl-prefix-tree-and-sandbox-pool.svg",
            "asset": "ch17/ch17-multi-turn-agentic-rl-prefix-tree-and-sandbox-pool.svg",
            "section": "17.6.1",
            "caption": "集群级多轮 Agentic RL 的异构 GPU—微虚机沙箱池利特尔法则配比与跨轮次前缀树增量 KV 复用架构",
            "source_refs": ["calculations/results/ch17-posttrain-rlvr-distill.json", "experiments/ch17/17-05/run.py"],
            "TeachingRole": "以高密微虚机槽位承接 8–90 s 容器编译长尾，消除同步绑卡造成的 68% GPU 空转"
        },
        {
            "figure_id": "图 17-26",
            "number": 26,
            "file": "ch17-bounded-staleness-off-policy-importance-sampling-stability.svg",
            "asset": "ch17/ch17-bounded-staleness-off-policy-importance-sampling-stability.svg",
            "section": "17.6.2",
            "caption": "有界陈旧度（Δ_max = 3）异步离策略 Rényi 方差护栏、2.64x 有效更新步增益与第四部分全栈底座咬合总图（算例 17-13）",
            "source_refs": ["calculations/results/ch17-posttrain-rlvr-distill.json", "experiments/ch17/17-05/run.py"],
            "TeachingRole": "收束第 13–17 章存储、供电液冷、可观测容错、机密沙箱与后训练进化管线的跨层协同"
        }
    ]
    for row in fig_index:
        row['figure'] = f"17-{row['number']}"
        row['name'] = Path(row['file']).stem
    (HERE / 'figure-index.json').write_text(json.dumps(fig_index, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    md = HERE.parent / '17-后训练与强化学习系统.md'
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

        work = re.sub(r'!\[([^\]]*)\]\((ch17/[^)]+)\)', repl_img, text_md)
        body = markdown.markdown(work, extensions=['tables', 'fenced_code', 'footnotes', 'toc'])
        html = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>第 17 章：后训练与强化学习系统</title></head><body>{body}</body></html>\n'
        out_html = preview_path(md)
        out_html.write_text(html, encoding='utf-8')

    sources = [
        HERE / 'build.py',
        HERE / 'figure-data.json',
        HERE / 'figure-index.json',
        md,
        ROOT / 'calculations/results/ch17-posttrain-rlvr-distill.json',
        ROOT / 'experiments/ch17/17-01/run.py',
        ROOT / 'experiments/ch17/17-02/run.py',
        ROOT / 'experiments/ch17/17-03/run.py',
        ROOT / 'experiments/ch17/17-04/run.py',
        ROOT / 'experiments/ch17/17-05/run.py',
    ]
    (HERE / 'sources.json').write_text(
        json.dumps(
            {
                "chapter": "ch17",
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
    print(f'Built {len(outputs)} figure files for Chapter 17; 0 layout warnings.')


if __name__ == '__main__':
    main()
