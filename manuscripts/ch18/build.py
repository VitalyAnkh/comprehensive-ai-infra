#!/usr/bin/env python3
"""Build Chapter 18 figures, locked calculations, and offline HTML preview."""
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


def fig_18_1_ppo_vs_grpo(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, '经典 PPO 四模型架构', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, 'Actor（16Ψ）+ Critic（16Ψ）\n双训练模型均含优化器状态', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, 'Ref（2Ψ）+ Reward（2Ψ）\n双前向打分模型常驻显存', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '总状态 36Ψ（32B 占 1,152 GB）\n跨四套异构并行图频繁通信', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, 'Critic 吃掉近半训练显存与算力', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'GRPO / DAPO 组相对优势', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '彻底砍掉 Critic 与神经 RM\n仅留 Actor（16Ψ）+ Ref（2Ψ）', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '每提示词组采样 G=8–64 条轨迹\n用组内均值/标准差算相对优势', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '模型状态省 50%（降至 576 GB）\n但自回归 Rollout 膨胀 G 倍', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '省掉训练态显存，压力转嫁 Rollout', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch18-ppo-four-model-vs-grpo-memory-ledger')


def fig_18_2_rollout_dominance(ex: Exporter):
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
    box(ax, 0.04, 0.13, 0.44, 0.20, '阶段 ①：Decode 逐词元生成\n每吐 1 词元需搬移全量权重与 KV\n算术强度极低，卡在 HBM 带宽', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '阶段 ③：Prefill 与反向传播\n全序列并行矩阵乘吃满 Tensor 核\n同词元量吞吐比 Decode 快 10–30x', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '加速 RLVR 的核心在于削减长尾 Rollout 气泡与提升生成吞吐', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch18-rlvr-iteration-timeline-rollout-dominance')


def fig_18_3_verl_hybrid_engine(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'veRL（HybridFlow）3D-HybridEngine 训推共置零冗余重分片', color='teal')
    stages = [
        (0.04, 0.58, 0.28, 0.24, '① 训练态 3D 切分\nTP×PP×DP 并行\nZeRO 分片省显存', 'blue'),
        (0.36, 0.58, 0.28, 0.24, '② 微分组 AllGather\n仅在小重组域 g 内\n微秒级重组推理权', 'teal'),
        (0.68, 0.58, 0.28, 0.24, '③ 生成态并发引擎\n小 TP 跑多路实例\n生成完释放完整权', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.36, 0.70))
    arrow(ax, (0.64, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '消除训练态与推理态的并行拓扑冲突（零双份权重常驻）', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, '朴素共置或全网广播重分片\n要么常驻两份权重撑爆显存\n要么跨全集群广播阻塞数秒', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '3D-HybridEngine 正交子网格对齐\n重分片通信局限在节点内 NVLink\n切换耗时压低 70%–90%', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '用瞬态 NVLink 组内 AllGather 取代常驻冗余副本', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch18-verl-3d-hybrid-engine-zero-redundancy-reshard')


def fig_18_4_colocated_vs_disagg(ex: Exporter):
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
    ex.save(fig, 'ch18-colocated-vs-disaggregated-rl-architecture')


def fig_18_5_long_cot_bubble(ex: Exporter):
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
    box(ax, 0.54, 0.20, 0.42, 0.16, '全集群有效 MFU 跌破 10%–15%\n成为长思维链 RL 头号杀手', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '世代尾气泡吞噬逾半集群算力', size=11, ha='center', weight='bold', color=BORDER_COL['purple'])
    ex.save(fig, 'ch18-long-cot-power-law-straggler-bubble')


def fig_18_6_rollpacker(ex: Exporter):
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

    card(ax, 0.02, 0.03, 0.96, 0.42, '算例 18-6：不均等微批下“按微批均值平均”vs.“按总词元归一化”', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, '错误：按微批独立求均值再平均\n微批 1(1项:1) · 微批 2(3项:3)\n(1 + 3)/2 = 2.0（放大微批 1）', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '正确：按全局总词元数加权归一化\n汇总 4 项梯度和 (1 + 9) 除以 4\n(1 + 9)/4 = 2.5（严格无偏真值）', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '动态拼箱改变微批词元数时，必须用全局有效总词元数做梯度分母', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch18-rollpacker-tail-migration-and-cp-scaling')


def fig_18_7_seer_speculative(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'Seer（OSDI 26）：分而治之上下文缓存与组内同源投机解码', color='blue')
    stages = [
        (0.04, 0.58, 0.28, 0.24, '① 细粒度分段生成\n长轨迹按固定块切分\n均衡各卡 KV 水位', 'blue'),
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
    ex.save(fig, 'ch18-seer-divided-and-conquered-speculative-rollout')


def fig_18_8_partial_rollout(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, '全同步屏障 Rollout', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '第 v 步必须集齐当前批次全部\nN 条完整轨迹才可启动训练', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '长尾未完轨迹阻塞整轮迭代\n或被强行截断沦为无效废样本', 'orange', size=11)
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
    ex.save(fig, 'ch18-staleness-bounded-partial-rollout-pipeline')


def fig_18_9_importance_sampling(ex: Exporter):
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
    text(ax, 0.50, 0.06, '随词元流持久化真实行为 logprob 是异步 Partial Rollout 收敛的命脉', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch18-off-policy-importance-sampling-clipping')


def fig_18_10_robustrl(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'RobustRL（OSDI 26）：面向训推异构角色的非对称容错恢复', color='teal')
    roles = [
        (0.04, 0.58, 0.44, 0.24, '① Rollout 节点故障（无状态）\n无需回滚检查点，仅替换故障机\n从保存的工具轮次轨迹续接', 'teal'),
        (0.52, 0.58, 0.44, 0.24, '② Learner 节点故障（有状态）\n严禁丢弃本轮已耗时 300s 轨迹\n仅检测恢复并重算 60s 更新步', 'blue'),
    ]
    for x, y, w, h, label, role in roles:
        box(ax, x, y, w, h, label, role, size=11)

    card(ax, 0.02, 0.03, 0.96, 0.42, '算例 18-7（锁定真值）：训练侧故障下整步重做 vs. 保留轨迹重做更新', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, '整步重做（丢弃已生成 Rollout）\n300s 采样 + 30s 失败 + 100s 检复\n+ 300s 重采 + 60s 更新 = 790 s', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, 'RobustRL 保留已完成 300s 轨迹\n300s 采样 + 30s 失败 + 100s 检复\n+ 60s 更新 = 490 s（净省 300 s）', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '将耗时的 Rollout 轨迹持久化为事务边界，阻断训练侧故障向上游蔓延', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch18-robustrl-role-based-fault-isolation')


def fig_18_11_distrs(ex: Exporter):
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

    card(ax, 0.02, 0.03, 0.96, 0.42, '算例 18-8：9 个 1s 任务 + 1 个 100s 任务在 e=10s 时的剩余时间反差', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '错误朴素估计 max(E[S] - e, 0)\n无条件均值 E[S] = 10.9 s\n在 e=10 s 时误判仅剩 0.9 s', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '真实条件剩余 E[S - 10 | S > 10]\n存活过 10s 必为 100s 重尾任务\n真实剩余高达 90.0 s（差 100x）', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '重尾验证分布具有“越老越慢”特性，必须按条件生存分布调度与抢占', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch18-distrs-disaggregated-reward-sandbox-streaming')


def fig_18_12_prm_tree_cow(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, '独立复制分支 KV（显存爆炸）', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '推理时树搜索（MCTS / Beam）\n从父节点派生 B 个候选推理步', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '为每条子分支物理深拷贝整段\n历史前缀与中间步骤 KV 缓存', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '64 分支搜索使显存占用膨胀 8x+\n且大量带宽浪费于 Memcpy', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '物理深拷贝扼杀宽分支搜索', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'PagedAttention 写时复制树', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '树节点分支时仅复制逻辑页表\n物理 KV 块引用计数 ref_cnt += B', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '新生成词元写入独立新页帧\n剪枝淘汰分支直接递减 ref_cnt', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '叠加树注意力单次扫过公共祖先\n显存与带宽开销双双降低 70%+', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '零拷贝分支派发 + 引用计数瞬时回收', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch18-test-time-scaling-prm-tree-search-cow-kv')


def fig_18_13_reward_barrier(ex: Exporter):
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
    ex.save(fig, 'ch18-speculative-tot-reward-barrier-bypass')


def fig_18_14_multi_turn_tool_rl(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, '阻塞占卡等待外部工具', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, 'SWE-bench / 搜索智能体在生成\n工具调用后等待沙箱编译或网页', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '外部环境响应长达 2–30 秒\n期间数万词元 KV 死占 GPU 显存', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, 'GPU 因显存满载无法接纳新轨迹\n算力在 I/O 等待中白白空转', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '慢速环境交互冻死昂贵 HBM', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, '状态挂起换出与协程复用', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '触发工具调用瞬间挂起会话\n经 PCIe 异步将 KV 卸至 DRAM', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '立即释放 HBM 页帧给就绪轨迹\n单卡并发活跃环境数扩大 5–10x', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '沙箱返回结果时预取回填 KV\n追加环境观测无缝续跑下一轮', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '将多轮工具交互改造为无阻塞协程', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch18-multi-turn-tool-env-rl-state-suspension')


def fig_18_15_hetrl(ex: Exporter):
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
    box(ax, 0.04, 0.13, 0.44, 0.20, '全用旗舰训练卡跑同步 RL\n80% 时间在跑低算术强度 Decode\n昂贵 Tensor Core 长期闲置', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '按吞吐速率比配置 3:1–5:1 异构池\n训练卡 100% 满载跑密集矩阵乘\n单次 RL 迭代硬件 TCO 降低 35%+', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '让高带宽比卡做生成、高算力比卡做更新，消除硬件特性的错配税', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch18-hetrl-heterogeneous-inference-training-matching')


def fig_18_16_dapo_filter(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'DAPO 动态采样过滤：消除全对（r=1）与全错（r=0）零梯度浪费', color='teal')
    groups = [
        (0.04, 0.58, 0.28, 0.24, '① 过易提示词(全对)\nG 条轨迹全 r=1\n方差 σ=0 → A=0', 'orange'),
        (0.36, 0.58, 0.28, 0.24, '② 过难提示词(全错)\nG 条轨迹全 r=0\n方差 σ=0 → A=0', 'orange'),
        (0.68, 0.58, 0.28, 0.24, '③ 有效区分提示词\n0 < 正确数 < G\n提供有效策略梯度', 'teal'),
    ]
    for x, y, w, h, label, role in groups:
        box(ax, x, y, w, h, label, role, size=11)

    card(ax, 0.02, 0.03, 0.96, 0.42, 'RollPacker 投机超采（160×10 = 1,600 选 1,024）保障有效微批满载', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, '静态采样 128 提示词 × 8 = 1,024\n剔除 30% 零方差组后仅剩 ~716 条\n训练微批不满载且梯度方差剧增', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '投机超采 160×10 = 1,600 (+56%)\n在线滤除零方差组并凑齐 1,024 条\n每步训练 100% 皆为有效非零梯度', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '结合投机超额采样与在线方差过滤，但须控制槽位竞争避免长度偏置', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch18-dapo-dynamic-sampling-zero-variance-filter')


def fig_18_17_throughput_waterfall(ex: Exporter):
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
    text(ax, 0.50, 0.06, '脱离无偏梯度与重要性采样校准的“工程提速”只会加速策略崩溃', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch18-rl-post-training-throughput-roofline')


def fig_18_18_closed_loop(ex: Exporter):
    fig, ax = canvas(4.15)
    card(ax, 0.02, 0.03, 0.96, 0.94, '第 18 章 RLVR 后训练与推理时计算扩展工程总账本', color='blue')
    grid = [
        (0.05, 0.66, 0.42, 0.20, '① 算法与显存账本（18.1）\nGRPO 省 50% 状态(576 GB)\nRollout 占 70%–90% 耗时', 'blue'),
        (0.53, 0.66, 0.42, 0.20, '② 训推拓扑与池化（18.2）\nveRL 零冗余重分片引擎\nWeave 错峰(216 H20/152 H800)', 'teal'),
        (0.53, 0.37, 0.42, 0.20, '③ 攻克长尾木桶（18.3）\nRollPacker 归一化(2.5 vs 2.0)\nSeer 投机·RobustRL 省 300s', 'purple'),
        (0.05, 0.37, 0.42, 0.20, '④ 验证与推理扩展（18.4）\nDistRS 条件剩余(90s vs 0.9s)\nCoW 树搜索与投机越障', 'orange'),
    ]
    for x, y, w, h, label, role in grid:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.47, 0.76), (0.53, 0.76))
    arrow(ax, (0.74, 0.66), (0.74, 0.57))
    arrow(ax, (0.53, 0.47), (0.47, 0.47))
    arrow(ax, (0.26, 0.57), (0.26, 0.66))
    box(ax, 0.05, 0.08, 0.90, 0.21, '实验：experiments/ch18/18-01 & 18-02（GRPO 省 50% 状态、\nRobustRL 省 300s、DistRS 216K 词元、跨洲 14,745x 压流）', 'teal', size=11)
    ex.save(fig, 'ch18-rlvr-and-test-time-compute-closed-loop')


def fig_18_19_geo_async_rollout(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'Prime-RL：跨地域闲置算力异步 Rollout 收割（算例 18-9）', color='blue')
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
    text(ax, 0.50, 0.06, '用 14,745.6x 上行压缩 + 序列几何均值 IS 过滤打通全球异构算力 RL', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch18-geo-distributed-async-rollout-harvester')


def fig_18_20_reward_hacking_breaker(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'RLVR 奖励黑客（Reward Hacking）的三维异常症状相图', color='orange')
    symptoms = [
        (0.04, 0.58, 0.28, 0.24, '① 长度失控膨胀\n堆砌无意义伪代码\n均长破 28K 词元', 'orange'),
        (0.36, 0.58, 0.28, 0.24, '② 策略熵断崖坍缩\n退化为高频复读机\n熵跌破 0.05 nats', 'orange'),
        (0.68, 0.58, 0.28, 0.24, '③ KL 散度突增\n偏离参考模型流形\n突破 0.25 熔断线', 'orange'),
    ]
    for x, y, w, h, label, role in symptoms:
        box(ax, x, y, w, h, label, role, size=11)

    card(ax, 0.02, 0.03, 0.96, 0.42, '双级在线统计制动器：软边界超长惩罚 + 硬熔断回滚机制', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '一级软制动（KL > 0.06 或超长）\n倍增 β_KL 并加分段长度罚项\n抑制无信息量的凑字数行为', 'teal', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '二级硬熔断（KL > 0.25 且低熵）\n中止当前步并回滚健康检查点\n隔离污染提示词并调高温度', 'blue', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '监控 (KL, 长度, 策略熵) 联合轨迹是长周期无人值守 RLVR 防崩溃底线', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch18-rlvr-reward-hacking-kl-divergence-breaker')


def main():
    ex = Exporter(HERE)
    fig_18_1_ppo_vs_grpo(ex)
    fig_18_2_rollout_dominance(ex)
    fig_18_3_verl_hybrid_engine(ex)
    fig_18_4_colocated_vs_disagg(ex)
    fig_18_5_long_cot_bubble(ex)
    fig_18_6_rollpacker(ex)
    fig_18_7_seer_speculative(ex)
    fig_18_8_partial_rollout(ex)
    fig_18_9_importance_sampling(ex)
    fig_18_10_robustrl(ex)
    fig_18_11_distrs(ex)
    fig_18_12_prm_tree_cow(ex)
    fig_18_13_reward_barrier(ex)
    fig_18_14_multi_turn_tool_rl(ex)
    fig_18_15_hetrl(ex)
    fig_18_16_dapo_filter(ex)
    fig_18_17_throughput_waterfall(ex)
    fig_18_18_closed_loop(ex)
    fig_18_19_geo_async_rollout(ex)
    fig_18_20_reward_hacking_breaker(ex)
    outputs = ex.finish()

    fig_index = [
        {
            "figure_id": "图 18-1",
            "number": 1,
            "file": "ch18-ppo-four-model-vs-grpo-memory-ledger.svg",
            "asset": "ch18/ch18-ppo-four-model-vs-grpo-memory-ledger.svg",
            "section": "18.1.1",
            "caption": "PPO 四模型（Actor/Critic/Ref/RM）与 GRPO/DAPO（消去 Critic + 组采样 G 条轨迹）的显存与计算量对比（算例 18-1）",
            "source_refs": ["manuscripts/ch18/figure-data.json", "experiments/ch18/18-01/run.py"],
            "TeachingRole": "量化 32B 模型下 PPO 1,152 GB（36Ψ）与 GRPO 576 GB（18Ψ）的 50% 状态显存削减"
        },
        {
            "figure_id": "图 18-2",
            "number": 2,
            "file": "ch18-rlvr-iteration-timeline-rollout-dominance.svg",
            "asset": "ch18/ch18-rlvr-iteration-timeline-rollout-dominance.svg",
            "section": "18.1.2",
            "caption": "RLVR 单步训练时间线分解——为何自回归 Rollout 占据 70%–90% 墙钟时间且算术强度极低",
            "source_refs": ["manuscripts/ch18/figure-data.json"],
            "TeachingRole": "从 Decode 访存带宽受限与 Prefill/Backward 算力密集的反差解释 Rollout 主导现象"
        },
        {
            "figure_id": "图 18-3",
            "number": 3,
            "file": "ch18-verl-3d-hybrid-engine-zero-redundancy-reshard.svg",
            "asset": "ch18/ch18-verl-3d-hybrid-engine-zero-redundancy-reshard.svg",
            "section": "18.2.1",
            "caption": "veRL（HybridFlow）3D-HybridEngine 训练态（ZeRO+TP+PP）与生成态（微张量并行）的零冗余内存重分片机制",
            "source_refs": ["manuscripts/ch18/figure-data.json"],
            "TeachingRole": "解析子网格微分组 AllGather 如何消除训练态与推理态的并行拓扑冲突"
        },
        {
            "figure_id": "图 18-4",
            "number": 4,
            "file": "ch18-colocated-vs-disaggregated-rl-architecture.svg",
            "asset": "ch18/ch18-colocated-vs-disaggregated-rl-architecture.svg",
            "section": "18.2.2",
            "caption": "训推共置（veRL）与物理训推分离及互补错峰池化（Weave OSDI '26，算例 18-3）对比",
            "source_refs": ["case-studies/rl-scheduling-and-recovery.md", "experiments/ch18/18-01/run.py"],
            "TeachingRole": "复算 Weave 互补作业（360/120 s 与 120/360 s）共享 216 H20 与 152 H800 的削峰收益"
        },
        {
            "figure_id": "图 18-5",
            "number": 5,
            "file": "ch18-long-cot-power-law-straggler-bubble.svg",
            "asset": "ch18/ch18-long-cot-power-law-straggler-bubble.svg",
            "section": "18.3.1",
            "caption": "长思维链（Long-CoT）输出长度幂律分布引发的同步 Rollout 尾部巨大 GPU 空泡",
            "source_refs": ["case-studies/rollout-tail-and-sampling.md"],
            "TeachingRole": "揭示 5% 超长推理轨迹如何通过同步屏障迫使 90% 已完成 GPU 空转等待"
        },
        {
            "figure_id": "图 18-6",
            "number": 6,
            "file": "ch18-rollpacker-tail-migration-and-cp-scaling.svg",
            "asset": "ch18/ch18-rollpacker-tail-migration-and-cp-scaling.svg",
            "section": "18.3.2",
            "caption": "RollPacker（NSDI '26）尾部存活请求紧缩迁移、双槽位长度偏差（1/2 → 1/4）与微批均值平均（2.0）vs. 全局总词元归一化（2.5）对比（算例 18-5 & 18-6）",
            "source_refs": ["case-studies/rollout-tail-and-sampling.md", "experiments/ch18/18-01/run.py"],
            "TeachingRole": "严格证明不均等微批下按微批均值平均会产生 2.0 vs 2.5 的梯度偏置"
        },
        {
            "figure_id": "图 18-7",
            "number": 7,
            "file": "ch18-seer-divided-and-conquered-speculative-rollout.svg",
            "asset": "ch18/ch18-seer-divided-and-conquered-speculative-rollout.svg",
            "section": "18.3.2",
            "caption": "Seer（OSDI '26）分段征服长尾 Rollout 与组内共享前缀投机解码加速",
            "source_refs": ["research/2026-infra-survey/reading-osdi-2026.md", "case-studies/rollout-tail-and-sampling.md"],
            "TeachingRole": "展示如何利用同组已完成短轨迹构建免模型草稿树加速剩余长尾轨迹"
        },
        {
            "figure_id": "图 18-8",
            "number": 8,
            "file": "ch18-staleness-bounded-partial-rollout-pipeline.svg",
            "asset": "ch18/ch18-staleness-bounded-partial-rollout-pipeline.svg",
            "section": "18.3.3",
            "caption": "同步屏障 Rollout vs. 陈旧度有界（τ ≤ τ_max）异步部分采样（Partial Rollout）的流水线吞吐跃升",
            "source_refs": ["case-studies/rollout-tail-and-sampling.md"],
            "TeachingRole": "允许未完长轨迹保留 KV 跨训练步续生，将集群有效利用率推高至 >88%"
        },
        {
            "figure_id": "图 18-9",
            "number": 9,
            "file": "ch18-off-policy-importance-sampling-clipping.svg",
            "asset": "ch18/ch18-off-policy-importance-sampling-clipping.svg",
            "section": "18.3.3",
            "caption": "异步 RL 中策略版本偏移 Δv = τ 下的重要性采样比率波动与逐词元真实行为概率记录准则",
            "source_refs": ["case-studies/rollout-tail-and-sampling.md"],
            "TeachingRole": "阐明跨版本拼接轨迹必须随词元流持久化生成当下的真实 logprob"
        },
        {
            "figure_id": "图 18-10",
            "number": 10,
            "file": "ch18-robustrl-role-based-fault-isolation.svg",
            "asset": "ch18/ch18-robustrl-role-based-fault-isolation.svg",
            "section": "18.3.3",
            "caption": "RobustRL（OSDI '26）区分无状态 Rollout 故障与有状态 Learner 故障的非对称容错（790 s vs. 490 s，净省 300 s，算例 18-7）",
            "source_refs": ["case-studies/rl-scheduling-and-recovery.md", "experiments/ch18/18-01/run.py"],
            "TeachingRole": "复算保留已完成 Rollout 轨迹将训练侧故障恢复耗时从 790 s 压至 490 s 的锁定账本"
        },
        {
            "figure_id": "图 18-11",
            "number": 11,
            "file": "ch18-distrs-disaggregated-reward-sandbox-streaming.svg",
            "asset": "ch18/ch18-distrs-disaggregated-reward-sandbox-streaming.svg",
            "section": "18.4.1",
            "caption": "DistRS（NSDI '26）将沙箱验证与长尾生成流式重叠及基于条件剩余时间 E[S-e | S>e]（90 s vs. 0.9 s）与 216,000 词元阈值的超期裁剪（算例 18-8）",
            "source_refs": ["case-studies/reward-deadlines-and-feedback.md", "experiments/ch18/18-01/run.py"],
            "TeachingRole": "证明重尾验证分布下朴素剩余时间估计（0.9 s）与真实条件期望（90.0 s）的百倍误差"
        },
        {
            "figure_id": "图 18-12",
            "number": 12,
            "file": "ch18-test-time-scaling-prm-tree-search-cow-kv.svg",
            "asset": "ch18/ch18-test-time-scaling-prm-tree-search-cow-kv.svg",
            "section": "18.4.2",
            "caption": "推理时扩展（Best-of-N / PRM 树搜索）利用 PagedAttention 写时复制（CoW）共享父节点 KV 缓存",
            "source_refs": ["manuscripts/ch18/figure-data.json"],
            "TeachingRole": "解析页表引用计数 CoW 与树注意力如何消除树状搜索的物理拷贝与显存膨胀"
        },
        {
            "figure_id": "图 18-13",
            "number": 13,
            "file": "ch18-speculative-tot-reward-barrier-bypass.svg",
            "asset": "ch18/ch18-speculative-tot-reward-barrier-bypass.svg",
            "section": "18.4.2",
            "caption": "Breaking the Reward Barrier（OSDI '26）在等待 PRM 评分期间投机性预展开高置信分支的时序对比",
            "source_refs": ["research/2026-infra-survey/reading-osdi-2026.md", "case-studies/reward-deadlines-and-feedback.md"],
            "TeachingRole": "通过超前投机展开子节点将重型 PRM 评分延迟隐藏在树搜索关键路径之外"
        },
        {
            "figure_id": "图 18-14",
            "number": 14,
            "file": "ch18-multi-turn-tool-env-rl-state-suspension.svg",
            "asset": "ch18/ch18-multi-turn-tool-env-rl-state-suspension.svg",
            "section": "18.4.1",
            "caption": "多轮工具调用 / SWE-bench 强化学习中等待外部环境返回时的 KV 换出挂起与高并发复用",
            "source_refs": ["case-studies/rl-scheduling-and-recovery.md"],
            "TeachingRole": "消除慢速外部沙箱执行期间巨量 KV 缓存死占 HBM 导致的算力饥饿"
        },
        {
            "figure_id": "图 18-15",
            "number": 15,
            "file": "ch18-hetrl-heterogeneous-inference-training-matching.svg",
            "asset": "ch18/ch18-hetrl-heterogeneous-inference-training-matching.svg",
            "section": "18.2.2",
            "caption": "HetRL 在带宽型推理卡与算力型训练卡之间的最优配比曲线",
            "source_refs": ["case-studies/rl-scheduling-and-recovery.md"],
            "TeachingRole": "建立按生成吞吐与训练吞吐匹配异构卡数的定量公式"
        },
        {
            "figure_id": "图 18-16",
            "number": 16,
            "file": "ch18-dapo-dynamic-sampling-zero-variance-filter.svg",
            "asset": "ch18/ch18-dapo-dynamic-sampling-zero-variance-filter.svg",
            "section": "18.3.2",
            "caption": "DAPO 动态过滤全对（r=1）与全错（r=0）零梯度提示词及 RollPacker 1,600 选 1,024 投机池",
            "source_refs": ["case-studies/rollout-tail-and-sampling.md", "experiments/ch18/18-01/run.py"],
            "TeachingRole": "结合 +56.25% 投机超采池与在线零方差过滤确保训练微批 100% 满载有效梯度"
        },
        {
            "figure_id": "图 18-17",
            "number": 17,
            "file": "ch18-rl-post-training-throughput-roofline.svg",
            "asset": "ch18/ch18-rl-post-training-throughput-roofline.svg",
            "section": "18.3.3",
            "caption": "大规模 RLVR 系统从朴素同步共置到异步分池+尾部拼箱的四阶有效吞吐跃迁瀑布图",
            "source_refs": ["case-studies/rollout-tail-and-sampling.md"],
            "TeachingRole": "汇总零冗余重分片、尾部拼箱投机与异步部分采样的阶梯收益及无偏性条件"
        },
        {
            "figure_id": "图 18-18",
            "number": 18,
            "file": "ch18-rlvr-and-test-time-compute-closed-loop.svg",
            "asset": "ch18/ch18-rlvr-and-test-time-compute-closed-loop.svg",
            "section": "18.4.2",
            "caption": "测试时搜索合成高价值轨迹 → 沙箱验证 → 异步 RLVR 更新 → 蒸馏回直觉模型的自我演进总账本",
            "source_refs": ["experiments/ch18/18-01/run.py"],
            "TeachingRole": "收束第 18 章 RLVR 显存账本、训推池化、长尾治理与推理时搜索的工程总图"
        },
        {
            "figure_id": "图 18-19",
            "number": 19,
            "file": "ch18-geo-distributed-async-rollout-harvester.svg",
            "asset": "ch18/ch18-geo-distributed-async-rollout-harvester.svg",
            "section": "18.4.3",
            "caption": "跨地域闲置推理算力异步收割 Rollout 轨迹、Token ID 极窄带宽回传与中心 Learner 重要性采样过滤架构（算例 18-9）",
            "source_refs": ["manuscripts/ch18/figure-data.json", "experiments/ch18/18-02/run.py"],
            "TeachingRole": "复算 10 B/tok 紧凑回传将 1,024×16K 轨迹从 2,473.9 GB 压缩 14,745.6x 至 160.0 MiB 的广域网账本"
        },
        {
            "figure_id": "图 18-20",
            "number": 20,
            "file": "ch18-rlvr-reward-hacking-kl-divergence-breaker.svg",
            "asset": "ch18/ch18-rlvr-reward-hacking-kl-divergence-breaker.svg",
            "section": "18.4.3",
            "caption": "RLVR 训练中奖励黑客（长度失控与熵坍缩）的在线监控指标相图与自适应 KL 制动熔断机制",
            "source_refs": ["manuscripts/ch18/figure-data.json", "experiments/ch18/18-02/run.py"],
            "TeachingRole": "建立基于 (KL 散度, 输出长度, 策略熵) 的两级在线奖励黑客制动与检查点回滚状态机"
        }
    ]
    for row in fig_index:
        row['figure'] = f"18-{row['number']}"
        row['name'] = Path(row['file']).stem
    (HERE / 'figure-index.json').write_text(json.dumps(fig_index, ensure_ascii=False, indent=2) + '\n')

    md = HERE.parent / '18-大规模后训练（RLVR）、异步强化学习与推理时扩展系统.md'
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

        work = re.sub(r'!\[([^\]]*)\]\((ch18/[^)]+)\)', repl_img, text_md)
        body = markdown.markdown(work, extensions=['tables', 'fenced_code', 'footnotes', 'toc'])
        html = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>第 18 章：大规模后训练（RLVR）、异步强化学习与推理时扩展系统</title></head><body>{body}</body></html>\n'
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
    print(f'Built {len(outputs)} figure files for Chapter 18; 0 layout warnings.')


if __name__ == '__main__':
    main()
