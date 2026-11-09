#!/usr/bin/env python3
"""Build Chapter 14 figures, locked co-design calculations, and offline HTML preview."""
from pathlib import Path
import base64
import hashlib
import json
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figure_style import COL, BORDER_COL, STYLE, Exporter, arrow, box, card, canvas, plot, text
from preview_output import preview_path

plt.rcParams.update(STYLE)


def sha256(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b'version https://git-lfs.github.com/spec/v1'):
        m = re.search(rb'oid sha256:([0-9a-f]{64})', raw)
        if m:
            return m.group(1).decode('ascii')
    return hashlib.sha256(raw).hexdigest()


def build_figure_data() -> dict:
    return {
        "example_14_1_v41_dual_track": {
            "mla_40layer_b_per_tok": 23040,
            "v4_global_b_per_tok": 3598.72,
            "v4_persistent_b_per_tok": 3514.25,
            "v41_global_b_per_tok": 890.0,
            "32k_mla_mib": 720.0,
            "32k_v4_global_mib": 112.46,
            "32k_v4_total_mib": 115.52,
            "32k_v41_global_mib": 27.8125,
            "32k_v41_total_mib": 30.875,
            "1m_v4_resident_gib": 3.351,
            "1m_v41_resident_gib": 0.849,
            "resident_ratio": 3.95,
            "8k_v4_read_mib": 25.06,
            "8k_v41_read_mib": 22.84,
            "8k_read_ratio": 1.10,
            "1m_v4_read_gib": 3.679,
            "1m_v41_read_gib": 1.466,
            "1m_read_ratio": 2.51,
            "persistent_cache_reduction": 8
        },
        "example_14_2_digital_worker": {
            "v41_state_200k_gib": 1.30,
            "v41_state_1m_gib": 6.42,
            "k3_state_200k_gib": 5.57,
            "k3_state_1m_gib": 26.17,
            "v41_200k_b200_usd_per_m": 5.97,
            "v41_200k_api_usd_per_m": 4.69,
            "v41_1m_b200_usd_per_m": 13.34,
            "v41_1m_api_usd_per_m": 6.41,
            "k3_200k_b300_usd_per_m": 145.13,
            "k3_200k_api_usd_per_m": 65.91,
            "k3_1m_b200_usd_per_m": 496.08,
            "k3_1m_api_usd_per_m": 125.19,
            "k3_fixed_layer_overhead_ms": 13.95,
            "k3_rebuild_200k_ms": 9.18,
            "k3_rebuild_1m_ms": 9.26
        },
        "example_14_3_opentallas": {
            "target_tokens_per_s": 8050,
            "step_budget_us": 124.221,
            "rom_weight_read_us": 7.828,
            "hbm_kv_read_us": 41.979,
            "tensor_compute_us": 25.932,
            "serial_comm_us": 69.821,
            "unpriced_aux_budget_us": 14.568,
            "wafer_hops": 15,
            "hop_ns": 100,
            "layers": 120,
            "wafer_comm_floor_us": 180.0,
            "incremental_nre_usd": 20000000,
            "breakeven_tokens": 40000000000000,
            "machines_1yr": 254,
            "machines_6mo": 508
        },
        "example_14_4_counterfactual": {
            "swa_layers": 40,
            "swa_window_tokens": 128,
            "swa_state_mib": 2.578,
            "candidate_pool_tokens": 16384,
            "reindex_k": 2048
        },
        "example_14_4a_beyond_chinchilla": {
            "densing_doubling_months": 3.3,
            "extra_train_cost_usd": 1000000,
            "saving_per_req_usd": 0.002,
            "breakeven_requests": 500000000
        },
        "example_14_5_design_review": {
            "requests_per_candidate": 32,
            "distinct_tasks": 8,
            "bf16_correct": 28,
            "fp8_correct": 26,
            "fp8_qbf16_correct": 28,
            "unique_kv_storage_bytes": 12884115456,
            "proceed_to_validation": ["bf16", "fp8_qbf16"],
            "approved_for_deployment": []
        },
        "example_14_7_exp13_07": {
            "k96_preserve_s": 12.800002,
            "k96_restart_s": 11.618788,
            "k96_ratio": 1.101664,
            "k192_predicted_interval": [0.70, 1.00],
            "k192_subjective_prob": 0.65,
            "k192_observed_min_ratio": 0.373470,
            "k192_observed_max_ratio": 0.950270,
            "k192_observed_median_ratio": 0.693141,
            "k192_verdict": "refuted_in_scope",
            "mechanism_checks_passed": 19516,
            "sampled_tokens_with_eos": 9420,
            "unique_committed_positions": 7500,
            "duplicate_deliveries_intercepted": 1920,
            "total_wall_s": 522.648,
            "rss_peak_bytes": 4714250240
        },
        "example_14_8_cost_telescoping": {
            "gpt4_mmlu86_ratio": 208,
            "gpt4_gpqa33_ratio": 313,
            "gpt4_humaneval67_ratio": 375,
            "weight_read_70b_int8_h100_ms": 20.90,
            "weight_read_8b_int8_h100_ms": 2.39,
            "weight_read_8b_fp4_h100_ms": 1.19,
            "weight_read_8b_fp4_h200_ms": 0.83,
            "weight_read_total_ratio": 25.1
        },
        "example_14_11_hwnas_hybrid_layers": {
            "total_layers": 64,
            "concurrent_sessions": 16,
            "context_tokens": 131072,
            "kv_budget_gib": 36.0,
            "tpot_budget_ms": 25.0,
            "full_layer_mib": 4096.0,
            "linear_layer_mib": 64.0,
            "optimal_full_layers": 8,
            "optimal_linear_layers": 56,
            "resident_state_gib": 35.50,
            "state_compression_x": 7.21,
            "step_tpot_ms": 24.08,
            "tpot_speedup_x": 3.60
        },
        "example_14_12_sparse_upcycling": {
            "dense_params_b": 8.0,
            "num_experts": 8,
            "active_experts": 2,
            "active_params_b": 13.0,
            "total_moe_params_b": 47.0,
            "scratch_tokens_t": 15.0,
            "align_tokens_t": 0.2,
            "upcycle_tokens_t": 3.0,
            "scratch_flops_e24": 1.17,
            "migrate_flops_e24": 0.2374,
            "compute_saving_pct": 79.71
        },
        "experiment_14_03_moe_pareto": {
            "optimal_G": 8,
            "optimal_expert_dim": 512,
            "optimal_layer_ms": 0.74,
            "optimal_utility_score": 3.87,
            "rubin_ridge_flop_per_b": 2272.73,
            "tpuv7_ridge_flop_per_b": 626.05,
            "rubin_step_ms": 11.86,
            "tpuv7_step_ms": 33.64,
            "rubin_speedup_vs_tpuv7": 2.84,
            "hwnas_best_tpot_ms": 12.91,
            "hwnas_best_state_gib": 28.25,
            "hwnas_best_quality": 96.80
        }
    }


def fig_14_1_simultaneous(ex: Exporter):
    fig, ax = canvas(3.80)
    card(ax, 0.02, 0.50, 0.96, 0.47, '硬件物理边界内化为模型架构正则项的跨层耦合', color='gray')
    hw = [
        (0.04, 0.69, 0.44, 0.16, '微缩块量化（MXFP4 32元素）\n约束激活离群值与谱范数', 'blue'),
        (0.52, 0.69, 0.44, 0.16, 'SFU 与 Tensor Core 配比\n约束 Softmax 频次与能耗', 'teal'),
        (0.04, 0.52, 0.44, 0.14, 'HBM 容量与 SSD 块粒度\n约束持久 KV 与跨层共享', 'purple'),
        (0.52, 0.52, 0.44, 0.14, '超节点互连半径与跳数\n约束专家度与状态重放', 'orange'),
    ]
    for x, y, w, h, label, role in hw:
        box(ax, x, y, w, h, label, role, size=11)

    card(ax, 0.02, 0.03, 0.96, 0.43, '协同设计的四大定量杠杆（Four Co-Design Levers）', color='blue')
    levers = [
        (0.04, 0.20, 0.44, 0.14, '① 数值分布整形\nRoPE-post 与正交旋转', 'blue'),
        (0.52, 0.20, 0.44, 0.14, '② 跨层状态锚点化\nCSA2 4-Owner 共享 KV', 'teal'),
        (0.04, 0.05, 0.44, 0.12, '③ 有界重算代传输\n128-token SWA 局部重放', 'purple'),
        (0.52, 0.05, 0.44, 0.12, '④ 前向计算重分布\nCED 早期丢弃与深编码', 'orange'),
    ]
    for x, y, w, h, label, role in levers:
        box(ax, x, y, w, h, label, role, size=11)
    ex.save(fig, 'ch14-codesign-simultaneous-equations')


def fig_14_2_dual_track(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.04, 0.46, 0.92, '轨道一：驻留容量（恒定 3.95×）', color='teal', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, 'V4 Flash（40 层独立 KV）\n3514.25 B/tok · 32K 112.46M', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, 'V4.1 Flash（4-Owner 共享）\n890 B/tok · 32K 27.8125M', 'teal', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '1M 上下文驻留容量对比\n3.351 GiB → 0.849 GiB', 'blue', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '压缩比 3.95× 与长度 H 无关', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])

    card(ax, 0.52, 0.04, 0.46, 0.92, '轨道二：单步读取（1.10→2.51×）', color='blue', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '8K 短上下文：Top-k 主导\n25.06 → 22.84 MiB (1.10×)', 'purple', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '128K–200K：索引占比抬升\n36 追随层免扫索引显效', 'teal', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '1M 超长上下文：索引主导\n3.679 → 1.466 GiB (2.51×)', 'blue', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '随 H 拉长逼近渐近上限 ~2.60×', size=11, ha='center', weight='bold', color=BORDER_COL['blue'])
    ex.save(fig, 'ch14-v41-dual-track-context-scaling')


def fig_14_3_four_stage(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'V4.1 Flash 四阶段全局注意力流水线（4 Owner + 36 Follower）', color='purple')
    stages = [
        (0.04, 0.58, 0.20, 0.24, '① Full 建库\n4 Owner 层\n写 KV+索引', 'blue'),
        (0.28, 0.58, 0.21, 0.24, '② 16K 候选池\nOwner 粗筛\n共享候选集', 'purple'),
        (0.53, 0.58, 0.20, 0.24, '③ Reindex\n16K 池内精排\n选 2,048 块', 'teal'),
        (0.77, 0.58, 0.19, 0.24, '④ Reuse\n追随层直取\n2,048 索引', 'orange'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.24, 0.70), (0.28, 0.70))
    arrow(ax, (0.49, 0.70), (0.53, 0.70))
    arrow(ax, (0.73, 0.70), (0.77, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.43, '持久化前缀缓存压缩至 1/8 的正交拆解（1/2 × 1/4 = 1/8）', color='teal')
    box(ax, 0.04, 0.16, 0.43, 0.19, '因子一 (1/2)：免存 40 层 SWA\n128-token 重算代 2.578 MiB 块', 'blue', size=11)
    box(ax, 0.53, 0.16, 0.43, 0.19, '因子二 (1/4)：CSA2 跨层共享\n全局 KV 3514.25→890 B/tok', 'teal', size=11)
    arrow(ax, (0.47, 0.255), (0.53, 0.255))
    text(ax, 0.50, 0.08, '两项正交机制叠加，使持久化前缀缓存体积降至原架构的约 1/8', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch14-v41-four-stage-attention-pipeline')


def fig_14_4_digital_worker(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.50, 0.96, 0.46, '算例 14-2：24×7 数字员工（≥30 tok/s）未优化基线成本对比', color='blue')
    box(ax, 0.04, 0.56, 0.44, 0.25, 'V4.1 Flash（1.30→6.42 GiB）\n200K：B200 5.97 vs API 4.69\n1M：B200 13.34 vs API 6.41 ($/M)', 'teal', size=11)
    box(ax, 0.52, 0.56, 0.44, 0.25, 'Kimi K3 基线（5.57→26.17 GiB）\n200K：B300 145.13 vs 65.91\n1M：B200 496.08 vs 125.19 ($/M)', 'orange', size=11)

    card(ax, 0.02, 0.03, 0.96, 0.43, '性能归因审计（OPTIMIZATION-AUDIT）：区分架构代价与运行时伪影', color='orange')
    box(ax, 0.04, 0.16, 0.44, 0.19, '伪影一：逐层固定调度 150 μs\n93 层累加 13.95 ms/tok', 'orange', size=11)
    box(ax, 0.52, 0.16, 0.44, 0.19, '伪影二：10% 重建 + 漏算 DSpark\n摊入 9.18–9.26 ms/tok（占七成）', 'purple', size=11)
    text(ax, 0.50, 0.08, '审计结论：未启用 MTP/DSpark 与算子融合的基线不能外推为架构劣势', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch14-digital-worker-v41-vs-k3-cost')


def fig_14_5_opentallas(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.50, 0.96, 0.46, '算例 14-3（Case B）：OpenTallas 单步 124.221 μs 四资源账本', color='teal')
    res_boxes = [
        (0.04, 0.58, 0.21, 0.24, 'ROM 权重读\n7.828 μs\n并行隐藏', 'teal'),
        (0.28, 0.58, 0.21, 0.24, 'HBM KV 读\n41.979 μs\n记忆主导', 'blue'),
        (0.52, 0.58, 0.21, 0.24, 'Tensor 计算\n25.932 μs\n重叠隐藏', 'purple'),
        (0.76, 0.58, 0.20, 0.24, '通信+余量\n69.82+14.57\n=84.389 μs', 'orange'),
    ]
    for x, y, w, h, label, role in res_boxes:
        box(ax, x, y, w, h, label, role, size=11)

    card(ax, 0.02, 0.03, 0.96, 0.43, 'Case D 晶圆级跳数延迟越界与 Case E/F 2000 万美元 NRE 回本边界', color='orange')
    box(ax, 0.04, 0.15, 0.44, 0.20, '晶圆级扩展直径反超（Case D）\n15 跳 × 100 ns × 120 层 = 180 μs\n单凭通信即击穿 124.221 μs 预算', 'orange', size=11)
    box(ax, 0.52, 0.15, 0.44, 0.20, '2000 万美元 NRE 回本（Case F）\n需产 40 万亿 tok（省 $0.50/M）\n1 年需 254 台 · 半年需 508 台', 'blue', size=11)
    text(ax, 0.50, 0.08, '专用 ASIC 必须在物理跳数延迟与模型半衰期之间同时闭合方程', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch14-opentallas-four-resource-and-wafer-hop')


def fig_14_6_decision_tree(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.03, 0.96, 0.94, '全书总结：AI Infra 架构师的五步定量诊断与反事实协同决策树', color='blue')
    steps = [
        (0.05, 0.72, 0.42, 0.14, '第一步：锁定量纲与负载边界\n区分十进制/二进制 · 明确 SLO', 'blue'),
        (0.53, 0.72, 0.42, 0.14, '第二步：建立四资源下界方程\n计算 / 访存 / 通信 / 存储 I/O', 'teal'),
        (0.53, 0.50, 0.42, 0.15, '第三步：定位瓶颈与翻转点\nRoofline · 利特尔 · Amdahl', 'purple'),
        (0.05, 0.50, 0.42, 0.15, '第四步：剥离架构与软件伪影\n审计逐层固定开销与双重拷贝', 'orange'),
        (0.05, 0.26, 0.90, 0.16, '第五步：跨层反事实推演与 TCO 均衡（Counterfactual Co-Design）\n当 HBM 免费 / 互连翻倍 / 半衰期减半时，重验算法—系统—硬件最优解', 'teal'),
    ]
    for x, y, w, h, label, role in steps:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.47, 0.79), (0.53, 0.79))
    arrow(ax, (0.74, 0.72), (0.74, 0.65))
    arrow(ax, (0.53, 0.575), (0.47, 0.575))
    arrow(ax, (0.26, 0.50), (0.26, 0.42))
    box(ax, 0.05, 0.08, 0.90, 0.12, '终极准则：以物理下界指导架构取舍，让每焦耳能量转化为有效智能', 'blue', size=11)
    ex.save(fig, 'ch14-five-step-diagnostic-decision-tree')


def fig_14_7_dflash(ex: Exporter):
    fig, ax = canvas(3.60)
    card(ax, 0.02, 0.04, 0.46, 0.92, '传统自回归草稿（串行受限）', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '小模型逐词元串行自回归\n提出 γ 个草稿需 γ 次启动', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '单步算术强度极低（~1 FLOP/B）\n受限于草稿权重读取与延迟', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '长草稿链误差逐位置累积\n远端位置接受率呈指数衰减', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '草稿步数 γ 难以突破 4–6', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'DFlash 块扩散草稿（角色反转）', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '双向注意力块扩散去噪器\n单步并行生成 b=16 草稿词元', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '把串行小矩阵转为并行块矩阵\n直接对齐 Tensor Core 高吞吐', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '自回归主干一次前向并行验证\n严格保持目标分布无损输出', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '扩散做草稿 + 自回归做验证', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch14-dflash-block-diffusion-role-reversal')


def fig_14_8_beyond_chinchilla(ex: Exporter):
    fig, ax = canvas(3.60)
    card(ax, 0.02, 0.04, 0.46, 0.92, 'Chinchilla 训练算力最优', color='purple', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '仅最小化一次性预训练成本\nC_train ≈ 6 · P · D_train', 'purple', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '按 D_train ≈ 20P 均衡放大\n倾向训练更大参数量模型', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '高频在线服务期推理账单\n2 · P · D_inf 随调用量爆炸', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '忽略服务期万亿级推理需求', size=11, ha='center', weight='bold', color=BORDER_COL['purple'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'Beyond-Chinchilla 全周期最优', color='blue', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '最小化训练+推理全周期总成本\nC_tot = 6PD_tr + 2PD_inf', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '能力密度约每 3.3 个月翻倍\n用 15T+ 词元超配训练 8B 模型', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '多投 100 万美元 · 单次省 0.002\n累计 5 亿次请求即完全回本', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '以一次性训练换长期服务降本', size=11, ha='center', weight='bold', color=BORDER_COL['blue'])
    ex.save(fig, 'ch14-beyond-chinchilla-lifecycle-compute-allocation')


def fig_14_9_design_review(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.04, 0.46, 0.92, '三候选 32 请求质量门控审计', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '候选 A（bf16）：28/32 正确\n4 次失败均在 n512-r0 长表', 'blue', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '候选 B（fp8）：跌至 26/32\n新增 2 次 n512-r1 退化被拒', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '候选 C（fp8_qbf16）：28/32\n恢复精度但失败集偏移至 r1', 'purple', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, 'approved_for_deployment = []', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'vLLM 固定 KV 池与双边界真理', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '三候选固定预分配 KV 池大小\n恒为 12,884,115,456 B', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, 'FP8 仅使每块容纳翻倍词元\n绝不自动退还预分配显存池', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '物理可行（装得下）≠ 经济可行\n须通过留出集与 P99 联合门控', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '仅 A 与 C 获准进入下一轮验证', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch14-design-review-3candidate-kv-audit')


def fig_14_10_gpu_crossover(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.04, 0.46, 0.92, '工作站/消费级卡优势区间', color='teal', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, 'RTX 4090（24G · 1.0 TB/s）\n单流/离线批处理极低成本', 'teal', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, 'RTX PRO 6000（96G GDDR7）\n单卡免切分装 70B/8B 长池', 'blue', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '硬伤：无 NVLink（仅 PCIe）\nGDDR 带宽 1.0–1.8 TB/s 低', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '适合单卡驻留与离线吞吐任务', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])

    card(ax, 0.52, 0.04, 0.46, 0.92, '数据中心旗舰卡优势区间', color='blue', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, 'H100 / B200（3.35–8 TB/s）\n高并发在线低 TPOT 占优', 'blue', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '900–1800 GB/s NVLink 域\n支撑 TP8/EP 全对全同步', 'purple', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '站点电价、配电容量与交期\n共同决定 TCO 最优机型切换', 'teal', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '严苛 SLO 与强通信不可替代', size=11, ha='center', weight='bold', color=BORDER_COL['blue'])
    ex.save(fig, 'ch14-datacenter-vs-consumer-gpu-crossover')


def fig_14_11_preregistration(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.50, 0.96, 0.46, '预注册预测协议（D11-SP04）：必填字段与 SHA-256 冻结', color='blue')
    fields = [
        (0.04, 0.58, 0.21, 0.24, '① 时间对象\ncreated_utc\ndeadline_utc', 'blue'),
        (0.28, 0.58, 0.21, 0.24, '② 冻结哈希\n脚本与协议\nSHA-256 锁定', 'teal'),
        (0.52, 0.58, 0.21, 0.24, '③ 指标区间\n目标统计量\n[L, U] 置信', 'purple'),
        (0.76, 0.58, 0.20, 0.24, '④ 证伪判据\n质量+数值\n联立门槛', 'orange'),
    ]
    for x, y, w, h, label, role in fields:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.25, 0.70), (0.28, 0.70))
    arrow(ax, (0.49, 0.70), (0.52, 0.70))
    arrow(ax, (0.73, 0.70), (0.76, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.43, '不可篡改的复核终态（严禁事后改 v1 区间或偷换分母）', color='teal')
    box(ax, 0.04, 0.15, 0.44, 0.20, 'supported_in_scope\n质量全过且落入 [L, U]\n仅限当前条件，严禁外推', 'teal', size=11)
    box(ax, 0.52, 0.15, 0.44, 0.20, 'refuted_in_scope\n13-07 观测 0.6931 < 0.70\n保留 v1 记录，另立 v2', 'orange', size=11)
    text(ax, 0.50, 0.08, '科学诚实准则：预测落空与反例记录比事后拼凑成功更具诊断价值', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch14-preregistered-prediction-lifecycle')


def fig_14_12_exp13_07_ratios(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.04, 0.46, 0.92, 'K96 历史反例（保留反慢）', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '实验 11-4 sequence（K96）\n1 次抢占恢复 · 128 词元', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, 'preserve 耗时 12.800 s\nrestart 重算仅 11.619 s', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '比值 R = 1.101664 > 1.0\n短前缀省算不敌事务开销', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '证伪“保留前缀永远更快”直觉', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'K192 预注册（0.6931 证伪）', color='purple', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '预注册区间 R ∈ [0.70, 1.00]\n5 轮 × 2 任务 = 10 对配对', 'blue', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '实测 10 对全部 < 1.0\n跨度 0.3735 至 0.9503', 'teal', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '10 对中位数 R = 0.693141\n以 0.0069 突破 0.70 下界', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '严守规则判定 refuted_in_scope', size=11, ha='center', weight='bold', color=BORDER_COL['purple'])
    ex.save(fig, 'ch14-exp13-07-k96-counterexample-and-k192-ratios')


def fig_14_13_exp13_07_buffer(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.50, 0.96, 0.46, '实验 13-07（K192）：30 路径与 20 次抢占的幂等恢复流水线', color='teal')
    stages = [
        (0.04, 0.58, 0.21, 0.24, '① 分段生成\n3×64=192 词\n64 处抢占', 'blue'),
        (0.28, 0.58, 0.21, 0.24, '② 采样缓冲\n采 9,420 词\n含 EOS 符', 'purple'),
        (0.52, 0.58, 0.21, 0.24, '③ 幂等去重\n拦 1,920 次\n重复投递词', 'orange'),
        (0.76, 0.58, 0.20, 0.24, '④ 唯一提交\n7,500 唯一位\n30/30 零差', 'teal'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.25, 0.70), (0.28, 0.70))
    arrow(ax, (0.49, 0.70), (0.52, 0.70))
    arrow(ax, (0.73, 0.70), (0.76, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.43, '19,516 项机制守卫全通过与 Trial 2 极端慢路径解剖', color='blue')
    box(ax, 0.04, 0.15, 0.44, 0.20, '机制账本（19,516/19,516）\n总墙钟 522.648 s · 零孤儿\n进程组 RSS 峰值 4.714 GB', 'teal', size=11)
    box(ax, 0.52, 0.15, 0.44, 0.20, 'Trial 2 extract 比率 0.3735\n17.619s vs 重启 47.176s\n第 3 段重算遇 24.4s 停顿', 'blue', size=11)
    text(ax, 0.50, 0.08, '教训：重启策略将前缀重算暴露了 3 次，放大操作系统与加载抖动风险', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch14-exp13-07-commit-buffer-idempotence')


def fig_14_14_double_counting(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.50, 0.96, 0.46, '跨层协同收益归因的四大重复计数陷阱（TC-18）', color='orange')
    traps = [
        (0.04, 0.69, 0.44, 0.14, '陷阱一：KV 压缩 × 扩批吞吐\n扩批已含 KV 腾出的显存', 'orange'),
        (0.52, 0.69, 0.44, 0.14, '陷阱二：FP4 字宽 × 算力翻倍\n端到端 FP4 已同时含两者', 'orange'),
        (0.04, 0.52, 0.44, 0.14, '陷阱三：整机跑分 × 软件倍率\n系统已启用 Dynamo/TRT', 'purple'),
        (0.52, 0.52, 0.44, 0.14, '陷阱四：MTP × EAGLE 叠加\n同类推测路线共享接受上限', 'purple'),
    ]
    for x, y, w, h, label, role in traps:
        box(ax, x, y, w, h, label, role, size=11)

    card(ax, 0.02, 0.03, 0.96, 0.43, '正确归因：固定顺序连乘（Telescoping）与交叉消融', color='teal')
    box(ax, 0.04, 0.15, 0.44, 0.20, '顺序连乘恒等式\nC0/CK = (C0/C1)·(C1/C2)\n单项归因依赖替换次序', 'teal', size=11)
    box(ax, 0.52, 0.15, 0.44, 0.20, '2×2 交叉消融实验\n分离主效应与软硬件耦合\n严禁虚构不支持格式基线', 'blue', size=11)
    text(ax, 0.50, 0.08, '纪律（TC-18）：只有共享同一任务、质量与分母的相邻状态才可连乘', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch14-four-double-counting-traps-matrix')


def fig_14_15_cost_telescoping(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.04, 0.46, 0.92, '公开门槛历史降价（Epoch）', color='blue', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, 'MMLU ≥86（GPT-4 门槛）\n37.5→0.18 美元（208×）', 'blue', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, 'GPQA ≥33（科学推理门槛）\n37.5→0.12 美元（313×）', 'teal', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, 'HumanEval ≥67（代码门槛）\n37.5→0.10 美元（375×）', 'purple', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '幅度取决于起点与评测门槛', size=11, ha='center', weight='bold', color=BORDER_COL['blue'])

    card(ax, 0.52, 0.04, 0.46, 0.92, '权重读取下界拆解（25.1×）', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '70B INT8 H100（20.9 ms）\n→ 8B INT8（2.39 ms，8.75×）', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '→ 8B FP4（1.19 ms，2.0×）\n→ 换 H200（0.83 ms，1.43×）', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '读取下界连乘改善 25.1×\n叠批处理与推测达百倍降本', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '模型密度 × 低精度 × 带宽 × 调度', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch14-2023-2026-cost-reduction-telescoping')


def fig_14_16_dual_track_book(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.04, 0.46, 0.92, '轨道 A：Qwen3-8B Dense', color='blue', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '第 2–5 章：144 KiB/tok\nRoofline 拐点与 FA 内核', 'blue', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '第 8–12 章：分页 KV 池\n双卡调度、端侧与抢占恢复', 'teal', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '第 13–14 章：RAG 23/77 块\nFP8 复审与 K192 预注册证伪', 'purple', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '小而密模型的单卡/端云压榨', size=11, ha='center', weight='bold', color=BORDER_COL['blue'])

    card(ax, 0.52, 0.04, 0.46, 0.92, '轨道 B：DeepSeek MoE', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '第 2–7 章：671B/37B + MLA\nNVL72 超节点与 EP 通信', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '第 9–11 章：PD/AF 与 3FS\nDualPipe 训练与 RL 推演', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '第 13–14 章：CSA2 890 B\n1/8 持久缓存与员工 TCO', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '超大规模稀疏系统的跨层求解', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch14-dual-track-book-throughline-summary')


def fig_14_17_rubin_v41(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.50, 0.96, 0.46, '下一代 Vera Rubin NVL72 硬件跃迁（相对 Blackwell）', color='blue')
    hw = [
        (0.04, 0.58, 0.21, 0.24, 'NVFP4 算力\n50 PFLOPS\n单卡增 5×', 'blue'),
        (0.28, 0.58, 0.21, 0.24, 'HBM4 带宽\n22 TB/s\n带宽 2.75×', 'teal'),
        (0.52, 0.58, 0.21, 0.24, 'NVLink 6\n3.6 TB/s\n柜 260 TB/s', 'purple'),
        (0.76, 0.58, 0.20, 0.24, 'Vera CPU\n88 核 1.5T\n1.8 TB/s', 'orange'),
    ]
    for x, y, w, h, label, role in hw:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.25, 0.70), (0.28, 0.70))
    arrow(ax, (0.49, 0.70), (0.52, 0.70))
    arrow(ax, (0.73, 0.70), (0.76, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.43, '硬件跃迁对 V4.1 Flash / Kimi K3 架构正则约束的反向重塑', color='teal')
    box(ax, 0.04, 0.15, 0.44, 0.20, '算存比从 1250 升至 2273\n奖励 RoPE-post 与 DFlash\n吸收 50 PFLOPS 矩阵算力', 'teal', size=11)
    box(ax, 0.52, 0.15, 0.44, 0.20, '22 TB/s HBM4 + 3.6 TB/s\n支撑百万上下文 16K 精排\n与跨卡 AF/EP 无气泡流水', 'blue', size=11)
    text(ax, 0.50, 0.08, '结论：算力增速快于带宽增速的剪刀差，使跨层状态压缩成为永恒主线', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch14-rubin-v41-codesign-mapping-matrix')


def fig_14_18_four_dim_loop(ex: Exporter):
    fig, ax = canvas(3.80)
    card(ax, 0.02, 0.50, 0.96, 0.47, 'AI 基础设施四维协同设计循环（Four-Dimensional Loop）', color='blue')
    dims = [
        (0.04, 0.69, 0.44, 0.16, '第一维：模型结构（Ch02–03）\nCSA2 · KDA · RoPE-post', 'blue'),
        (0.52, 0.69, 0.44, 0.16, '第二维：运行时（Ch05, 08–11）\nFlashMLA · 图重放 · PD 分离', 'teal'),
        (0.04, 0.52, 0.44, 0.14, '第三维：超节点（Ch04, 06–07）\nMXFP4 · HBM4 · NVLink 6', 'purple'),
        (0.52, 0.52, 0.44, 0.14, '第四维：电力与安全（Ch15–16）\n液冷热学 · 跨城训练 · TEE', 'orange'),
    ]
    for x, y, w, h, label, role in dims:
        box(ax, x, y, w, h, label, role, size=11)

    card(ax, 0.02, 0.03, 0.96, 0.43, '承前启后：迈向第 15–16 章的吉瓦级能源与机密安全底座', color='teal')
    box(ax, 0.04, 0.15, 0.44, 0.20, '向下扎入物理极限（第 15 章）\n单柜 130–600 kW、集群 GW\n电力热学与跨城光网成核心', 'orange', size=11)
    box(ax, 0.52, 0.15, 0.44, 0.20, '向外筑牢可信防线（第 16 章）\n万亿权重与百万 KV 驻云端\nGPU TEE 与安全沙箱成刚需', 'purple', size=11)
    text(ax, 0.50, 0.08, '全书通式：用可复算的物理账本与可证伪的实验纪律驾驭智能基础设施', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch14-four-dimensional-codesign-closed-loop')


def fig_14_19_chakra(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.04, 0.46, 0.92, 'MLCommons Chakra 执行轨迹图', color='blue', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '计算节点 + 通信节点 + 数据边\n构建跨流有向无环图 ETG', 'blue', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '解耦训练框架与网络仿真器\n支持拓扑扫描与瓶颈推演', 'teal', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '关键局限：静态轨迹不反映\n运行时融合与动态路由变化', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '流片与建群前的标准轨迹底座', size=11, ha='center', weight='bold', color=BORDER_COL['blue'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'Phantora 与 Charon 细粒度仿真', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, 'NSDI 26 Phantora：截获框架\n算子与通信调用，免重写代码', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, 'MLSys 26 Charon：细粒度模拟\n对比不同 TP/PP/EP 训推配置', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '审计铁律：单卡拦截≠万卡验证\n须先过容量与关键路径下界', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '模拟器是假设生成器而非真值本身', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch14-chakra-execution-trace-graph-simulation')


def fig_14_20_neusight(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.04, 0.46, 0.92, 'NeuSight 瓦片级预测与适用边界', color='purple', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, 'ASPLOS 25：算子拆为输出 Tile\n估算波数并在 Roofline 校准', 'purple', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '源码 6945927d：依赖旧卡近邻\n查表与 cu121 固定特征', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '单 microbatch 互斥并行分支\n不可盲目外推 FP8 融合内核', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '校准须包含真实 Tile 与库版本', size=11, ha='center', weight='bold', color=BORDER_COL['purple'])

    card(ax, 0.52, 0.04, 0.46, 0.92, '算例：总误差为 0 的归因陷阱', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '两串行阶段：预测 [8, 2] ms\n实测 [6, 4] ms（总长 10 ms）', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '整段误差 0%，但逐段误差达\n33.3% 与 50%（二段翻倍）', 'orange', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '若将阶段一提速减半：\n预测总长 6 ms vs 实测 7 ms', 'blue', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '严禁仅用端到端总和验证预测器', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch14-neusight-tile-roofline-calibration')


def fig_14_21_farskip(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.04, 0.46, 0.92, '标准 MoE 逐层阻塞通信', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '第 l 层专家计算完成后\n第 l+1 层立即依赖其输出', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '跨节点 All-to-All Combine\n直接暴露在关键路径上', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '当 T_comm > T_shared 时\nTensor Core 陷入通信停顿', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '受制于逐层串行数据依赖边', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'FarSkip 跨层跳步通信解耦', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, 'MLSys 26：重构残差连接图\n第 l 层输出延至 l+2 层消费', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '第 l+1 层仅依赖本地激活\n整层 T_layer 全额掩盖通信', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '工程边界：改变前向计算图\n须经重训练或蒸馏对齐精度', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '用模型结构换取一整层通信窗', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch14-farskip-collective-delayed-dependency')


def fig_14_22_moe_granularity(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.04, 0.46, 0.92, '专家细粒度切分 G 的算法收益', color='blue', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '固定激活参数切细专家 G 倍\nTop-2 大专家 → Top-8 细专家', 'blue', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '组合表达空间呈指数级扩展\n降低同等算力下训练困惑度', 'teal', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '配合 1–2 个常驻共享专家\n吸收高频通用语法与梯度', 'purple', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, 'DeepSeekMoE / Qwen3-MoE 路线', size=11, ha='center', weight='bold', color=BORDER_COL['blue'])

    card(ax, 0.52, 0.04, 0.46, 0.92, '网络小包通信税与分块拐点', color='orange', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '代价一：All-to-All 扇出乘 G\n单包缩为 1/G，触发小包延迟', 'orange', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '代价二：每专家 Token 数缩减\n跌破 Tensor Core M=64 下界', 'orange', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '协同拐点：节点分组受限路由\n限制跨机跳数 + 守住分块下界', 'teal', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '专家粒度 G 须与网络基数联立', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])
    ex.save(fig, 'ch14-moe-granularity-vs-network-radix-co-design')


def fig_14_23_kv_head_tp(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.04, 0.46, 0.92, 'GQA 无冗余并行区（TP≤n_kv）', color='teal', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '设模型 n_q=32、n_kv=8\nTP≤8 时每卡分整数个 KV 头', 'teal', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '每卡 KV 体积随 1/TP 递减\n全组总 KV 容量等于单卡基准', 'blue', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '单步 Decode 读取量除以 TP\n算力与访存按卡数线性扩展', 'purple', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, 'n_kv=8 匹配 8 卡 NVLink 节点', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])

    card(ax, 0.52, 0.04, 0.46, 0.92, '激进压缩触发复制税（TP>n_kv）', color='orange', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '若 n_kv=2 却按 TP=8 部署\n每 KV 头被迫复制 R=8/2=4 份', 'orange', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '总 KV 显存膨胀 4 倍抵消收益\n或改走跨卡交换引入额外通信', 'orange', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '协同法则：n_kv 取 TP 约数\n或采用 MLA 潜向量 + DP 注意力', 'teal', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '架构超参数锁死部署并行上限', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])
    ex.save(fig, 'ch14-kv-head-ratio-vs-tensor-parallel-limit')


def fig_14_24_codesign_loop(ex: Exporter):
    fig, ax = canvas(3.80)
    card(ax, 0.02, 0.03, 0.96, 0.94, '从流片前架构探索到生产集群预注册证伪的完整协同工程飞轮', color='blue')
    steps = [
        (0.04, 0.68, 0.44, 0.18, '① 第一性原理四资源下界筛选\n核算计算 / HBM / 网络 / SSD\n直接淘汰违反物理守恒候选', 'blue'),
        (0.52, 0.68, 0.44, 0.18, '② 瓦片级与轨迹图细粒度仿真\nNeuSight 校准 + Chakra ETG\nPhantora / Charon 预估重叠', 'teal'),
        (0.52, 0.44, 0.44, 0.18, '③ 算法—系统—硬件联立改造\nCSA2 压 KV · FarSkip 掩通信\nDFlash 块扩散对齐矩阵单元', 'purple'),
        (0.04, 0.44, 0.44, 0.18, '④ 预注册预测与留出集门控实验\n实验 14-01（K192 R=0.693141）\n实验 14-02（条件比率瀑布）', 'orange'),
        (0.04, 0.20, 0.92, 0.18, '⑤ 全生命周期 TCO 与代际半衰期反馈（Beyond-Chinchilla + NRE 摊销）\n将线上实测阶段归因（如 70% 主机固定开销）反馈至下一代芯片定义', 'teal'),
    ]
    for x, y, w, h, label, role in steps:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.48, 0.77), (0.52, 0.77))
    arrow(ax, (0.74, 0.68), (0.74, 0.62))
    arrow(ax, (0.52, 0.53), (0.48, 0.53))
    arrow(ax, (0.26, 0.44), (0.26, 0.38))
    box(ax, 0.04, 0.06, 0.92, 0.10, '第 14 章定理：唯有经受住逐阶段归因与预注册证伪的协同设计才具价值', 'blue', size=11)
    ex.save(fig, 'ch14-codesign-pre-silicon-to-production-loop')


def fig_14_25_hwnas_pareto(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, 'A. 算例 14-11：混合层整数规划', color='blue', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '64 层 · 16 并发 · 预算 ≤36 GiB\n全注意力 4 GiB vs 线性 64 MiB', 'blue', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '规划最优解：1:7 混合装配\n8 层全注意力 + 56 层线性层', 'teal', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '状态 35.50 GiB (省 7.21×)\nTPOT 24.08 ms (快 3.60×)', 'purple', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '1:7 全局/线性配比达帕累托最优', size=11, ha='center', weight='bold', color=BORDER_COL['blue'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'B. 硬件感知架构搜索 (HW-NAS)', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '摒弃纯 FLOPs 代理指标\n以芯片 TPOT 与 HBM 为目标', 'orange', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '异构装配：KDA/Mamba 压状态\nGQA/MLA 守检索 · MoE 扩容量', 'teal', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '随芯片算存比动态迁移：\n高算存比倾向高密度细专家', 'blue', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '将芯片物理边界编译为层间配比', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch14-hardware-aware-hybrid-layer-allocation-pareto')


def fig_14_26_sparse_upcycling(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, 'A. 算例 14-12：Sparse Upcycling', color='purple', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '起点：8B 稠密底座 (15T 词元)\n复制 FFN 为 8 专家 + 微扰', 'purple', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '0.2T 对齐 + 3.0T 稀疏退火\n总耗 0.2374×10²⁴ FLOPs', 'teal', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '较从头训 15T (1.17×10²⁴)\n净省 79.71% 预训练算力', 'blue', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '复用稠密底座沉没算力的捷径', size=11, ha='center', weight='bold', color=BORDER_COL['purple'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'B. 跨架构注意力线性化三阶蒸馏', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '阶段一：注意力矩阵对齐\n用教师 QK^T 监督线性核', 'blue', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '阶段二：逐层隐藏态对齐\n冻结主干最小化输出 L2 距离', 'teal', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '阶段三：端到端 KL 蒸馏微调\n约 20% 词元完成架构跃迁', 'orange', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, 'MOHAWK 低成本跨架构迁移范式', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch14-sparse-upcycling-and-attention-distillation-pipeline')


def fig_14_27_moe_bw_pareto(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, 'A. 细粒度专家切分 G 拐点', color='purple', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, 'G=1..4：d_exp ≥ 1024\n瓦片满载但表达收益未饱', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '最优 G*=8：d_exp=512\n耗时 0.74 ms · 效用 3.87', 'teal', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '过切 G=16：d_exp=256\n瓦片折半 · 耗时 2.02 ms', 'purple', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, 'd_exp 严丝合缝对齐 N_tile=512', size=11, ha='center', weight='bold', color=BORDER_COL['purple'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'B. Rubin vs TPU v7 单步账本', color='blue', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, 'Vera Rubin (22 TB/s HBM4)\n算存比 2,272.73 FLOP/B', 'blue', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, 'TPU v7 (7.37 TB/s HBM3e)\n算存比 626.05 FLOP/B', 'teal', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '单步 11.86 vs 33.64 ms\nRubin 解码吞吐快 2.84×', 'blue', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, 'HBM4 与 NVLink 6 协同提速', size=11, ha='center', weight='bold', color=BORDER_COL['blue'])
    ex.save(fig, 'ch14-moe-granularity-vs-memory-bandwidth-pareto')


def fig_14_28_hwnas_loop(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, 'A. 三维超参数候选空间', color='teal', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '粒度 G ∈ {2, 4, 8, 16}\n激活比 ρ_act 三档扫描', 'teal', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '全注意力比 r_attn 三档\n正交组合 36 组候选架构', 'blue', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '硬件红线：TPOT ≤ 15.0 ms\n并发状态显存 ≤ 40.0 GiB', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '剔除 r_attn=50% (54.83 GiB)', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'B. 闭环帕累托筛选最优解', color='purple', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '36 组候选筛出 6 组可行解\n按质量函数 Q 降序排列', 'purple', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '锁定 (G*=8, 3.125%, 25%)\n16/512 专家 + 1:3 注意力', 'teal', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '单步 12.91 ms · 状态 28.25G\n综合质量得分 96.80', 'blue', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '实验 14-03 全自动闭环寻优', size=11, ha='center', weight='bold', color=BORDER_COL['purple'])
    ex.save(fig, 'ch14-hardware-aware-neural-architecture-search-loop')


def main():
    ex = Exporter(HERE)
    fig_14_1_simultaneous(ex)
    fig_14_2_dual_track(ex)
    fig_14_3_four_stage(ex)
    fig_14_4_digital_worker(ex)
    fig_14_5_opentallas(ex)
    fig_14_6_decision_tree(ex)
    fig_14_7_dflash(ex)
    fig_14_8_beyond_chinchilla(ex)
    fig_14_9_design_review(ex)
    fig_14_10_gpu_crossover(ex)
    fig_14_11_preregistration(ex)
    fig_14_12_exp13_07_ratios(ex)
    fig_14_13_exp13_07_buffer(ex)
    fig_14_14_double_counting(ex)
    fig_14_15_cost_telescoping(ex)
    fig_14_16_dual_track_book(ex)
    fig_14_17_rubin_v41(ex)
    fig_14_18_four_dim_loop(ex)
    fig_14_19_chakra(ex)
    fig_14_20_neusight(ex)
    fig_14_21_farskip(ex)
    fig_14_22_moe_granularity(ex)
    fig_14_23_kv_head_tp(ex)
    fig_14_24_codesign_loop(ex)
    fig_14_25_hwnas_pareto(ex)
    fig_14_26_sparse_upcycling(ex)
    fig_14_27_moe_bw_pareto(ex)
    fig_14_28_hwnas_loop(ex)
    outputs = ex.finish()

    fig_data = build_figure_data()
    (HERE / 'figure-data.json').write_text(json.dumps(fig_data, ensure_ascii=False, indent=2) + '\n')

    fig_index = [
        {
            "figure_id": "图 14-1",
            "number": 1,
            "file": "ch14-codesign-simultaneous-equations.svg",
            "asset": "ch14/ch14-codesign-simultaneous-equations.svg",
            "section": "14.1.2",
            "caption": "模型—系统—硬件协同设计的跨层耦合约束（MXFP4 块粒度、SFU 配比、HBM 容量与超节点半径）及四大定量设计杠杆",
            "source_refs": ["manuscripts/ch14/figure-data.json"],
            "TeachingRole": "建立第 14 章跨层联立方程组与四大协同设计杠杆的总览框架"
        },
        {
            "figure_id": "图 14-2",
            "number": 2,
            "file": "ch14-v41-dual-track-context-scaling.svg",
            "asset": "ch14/ch14-v41-dual-track-context-scaling.svg",
            "section": "14.2.1",
            "caption": "算例 14-1：在 8K 至 1M 上下文扩展过程中，DeepSeek V4 Flash 与 V4.1 Flash 的驻留 KV 容量（恒定缩小 3.95x）与单步 Decode 理论读取量（从 1.10x 演进至 2.51x）的双轨分离规律",
            "source_refs": ["manuscripts/ch14/figure-data.json"],
            "TeachingRole": "直观区分驻留显存容量压缩比与单步注意力读取带宽压缩比的不同演进轨迹"
        },
        {
            "figure_id": "图 14-3",
            "number": 3,
            "file": "ch14-v41-four-stage-attention-pipeline.svg",
            "asset": "ch14/ch14-v41-four-stage-attention-pipeline.svg",
            "section": "14.2.3",
            "caption": "V4.1 Flash 的四阶段全局注意力执行流水线（Full -> 16,384 候选池 -> Reindex -> Reuse）以及持久化前缀缓存缩减至 1/8（1/2 SWA 免存 × 1/4 CSA2 跨层共享）的正交拆解",
            "source_refs": ["manuscripts/ch14/figure-data.json"],
            "TeachingRole": "剖析 4 个 Owner 层与 36 个 Follower 层如何协同消除重复索引扫描并实现 1/8 持久化缓存"
        },
        {
            "figure_id": "图 14-4",
            "number": 4,
            "file": "ch14-digital-worker-v41-vs-k3-cost.svg",
            "asset": "ch14/ch14-digital-worker-v41-vs-k3-cost.svg",
            "section": "14.3.2",
            "caption": "算例 14-2：24×7 数字员工在 200K/1M 上下文下 V4.1 Flash 与 Kimi K3 的跨代 GPU 选卡及 API 成本对比，以及揭示约七成耗时来自逐层固定开销与频繁重建假设的性能归因审计",
            "source_refs": ["manuscripts/ch14/figure-data.json"],
            "TeachingRole": "演示如何在真实工程评估中剥离未优化运行时伪影与模型架构内生差异"
        },
        {
            "figure_id": "图 14-5",
            "number": 5,
            "file": "ch14-opentallas-four-resource-and-wafer-hop.svg",
            "asset": "ch14/ch14-opentallas-four-resource-and-wafer-hop.svg",
            "section": "14.4.2",
            "caption": "算例 14-3：OpenTallas 专用 ASIC 单步 124.221 μs（8,050 tok/s）的四资源时间账本（Case B）、晶圆级 15 跳 180 μs 通信直径越界（Case D）以及 2,000 万美元 NRE 的 40 万亿 token 回本边界（Case F）",
            "source_refs": ["manuscripts/ch14/figure-data.json"],
            "TeachingRole": "用严密的四资源时间线与跨芯片跳数下界审计专用推理芯片的物理与商业可行性"
        },
        {
            "figure_id": "图 14-6",
            "number": 6,
            "file": "ch14-five-step-diagnostic-decision-tree.svg",
            "asset": "ch14/ch14-five-step-diagnostic-decision-tree.svg",
            "section": "14.5.2",
            "caption": "全书收束：AI 基础设施架构师的五步定量诊断法与跨层反事实协同设计决策树",
            "source_refs": ["manuscripts/ch14/figure-data.json"],
            "TeachingRole": "收束全书十四章知识体系，提供可复用的一线 AI Infra 性能诊断与架构设计方法"
        },
        {
            "figure_id": "图 14-7",
            "number": 7,
            "file": "ch14-dflash-block-diffusion-role-reversal.svg",
            "asset": "ch14/ch14-dflash-block-diffusion-role-reversal.svg",
            "section": "14.5.3",
            "caption": "DFlash 块扩散草稿（16-token 并行去噪）与自回归验证的角色反转架构及接受长度增益",
            "source_refs": ["research/token-cost-2023-2026/report.md"],
            "TeachingRole": "展示如何用双向扩散块并行生成替代串行自回归草稿以对齐 Tensor Core 算力"
        },
        {
            "figure_id": "图 14-8",
            "number": 8,
            "file": "ch14-beyond-chinchilla-lifecycle-compute-allocation.svg",
            "asset": "ch14/ch14-beyond-chinchilla-lifecycle-compute-allocation.svg",
            "section": "14.5.3",
            "caption": "算例 14-4a：推理感知的生命周期总计算最小化（Beyond-Chinchilla 3.3 个月能力密度翻倍与超配训练回本点）",
            "source_refs": ["research/token-cost-2023-2026/report.md"],
            "TeachingRole": "对比仅优化预训练成本的 Chinchilla 最优与纳入服务期推理需求的全生命周期最优"
        },
        {
            "figure_id": "图 14-9",
            "number": 9,
            "file": "ch14-design-review-3candidate-kv-audit.svg",
            "asset": "ch14/ch14-design-review-3candidate-kv-audit.svg",
            "section": "14.6.1",
            "caption": "实验 13-01（design-review）：Qwen3-8B 三种 KV 表示候选在 32 请求上的质量门控与 12.88 GB 固定池真相",
            "source_refs": ["experiments/ch13/13-01/design-review/README.md"],
            "TeachingRole": "揭示 vLLM 固定预分配 KV 池机制与物理可行/经济可行双边界的真实门控纪律"
        },
        {
            "figure_id": "图 14-10",
            "number": 10,
            "file": "ch14-datacenter-vs-consumer-gpu-crossover.svg",
            "asset": "ch14/ch14-datacenter-vs-consumer-gpu-crossover.svg",
            "section": "14.6.2",
            "caption": "算例 14-6：数据中心卡（H100/B200）与工作站/消费级卡（RTX PRO 6000 / 4090）的四场景翻转边界",
            "source_refs": ["research/draft14-15-outline-audit/snapshots/13-架构协同设计.md"],
            "TeachingRole": "定量标定不同并发、通信敏感度与站点供电条件下跨市场 GPU 的性价比交点"
        },
        {
            "figure_id": "图 14-11",
            "number": 11,
            "file": "ch14-preregistered-prediction-lifecycle.svg",
            "asset": "ch14/ch14-preregistered-prediction-lifecycle.svg",
            "section": "14.7.1",
            "caption": "可证伪系统科学：预注册预测协议的 10 项必填字段与不可篡改状态机（D11-SP04）",
            "source_refs": ["research/draft11-outline-audit/report.md", "experiments/ch13/13-07/prediction.json"],
            "TeachingRole": "建立面向 AI 系统性能工程的预注册预测、SHA-256 冻结与诚实证伪全流程"
        },
        {
            "figure_id": "图 14-12",
            "number": 12,
            "file": "ch14-exp13-07-k96-counterexample-and-k192-ratios.svg",
            "asset": "ch14/ch14-exp13-07-k96-counterexample-and-k192-ratios.svg",
            "section": "14.7.2",
            "caption": "实验 13-07：K96 保留前缀反慢反例（1.1017x）与 K192 预注册 10 对中位数（0.6931 < 0.70）范围外证伪",
            "source_refs": ["experiments/ch13/13-07/README.md"],
            "TeachingRole": "复盘真实系统实验中为何保留前缀在 K96 反慢而在 K192 超预期变快导致区间失配"
        },
        {
            "figure_id": "图 14-13",
            "number": 13,
            "file": "ch14-exp13-07-commit-buffer-idempotence.svg",
            "asset": "ch14/ch14-exp13-07-commit-buffer-idempotence.svg",
            "section": "14.7.2",
            "caption": "实验 13-07：30 路径全通过下 7,500 唯一提交位置与 1,920 次重复投递幂等拦截的机制解剖",
            "source_refs": ["experiments/ch13/13-07/README.md"],
            "TeachingRole": "解剖抢占恢复流式推理中的采样与提交分离架构及操作系统长尾停顿放大效应"
        },
        {
            "figure_id": "图 14-14",
            "number": 14,
            "file": "ch14-four-double-counting-traps-matrix.svg",
            "asset": "ch14/ch14-four-double-counting-traps-matrix.svg",
            "section": "14.8.1",
            "caption": "跨层协同收益归因的四大重复计数陷阱（Double-Counting Traps）与正交消融实验纪律（TC-18）",
            "source_refs": ["research/token-cost-2023-2026/report.md"],
            "TeachingRole": "防止读者将相互耦合的容量、精度、系统基准与推测倍率错误连乘"
        },
        {
            "figure_id": "图 14-15",
            "number": 15,
            "file": "ch14-2023-2026-cost-reduction-telescoping.svg",
            "asset": "ch14/ch14-2023-2026-cost-reduction-telescoping.svg",
            "section": "14.8.2",
            "caption": "2023–2026 固定能力门槛下百倍（208x–375x）价格下降的顺序连乘（Telescoping）与 25.1x 读取下界拆解",
            "source_refs": ["research/token-cost-2023-2026/report.md"],
            "TeachingRole": "用公开价格序列与可复算权重读取下界收束全书的同等能力降本主线"
        },
        {
            "figure_id": "图 14-16",
            "number": 16,
            "file": "ch14-dual-track-book-throughline-summary.svg",
            "asset": "ch14/ch14-dual-track-book-throughline-summary.svg",
            "section": "14.8.2",
            "caption": "全书双主线（Qwen3-8B Dense 与 DeepSeek-V3/V4.1 MoE）跨章逐步修正推算全景图（D11-SP01）",
            "source_refs": ["research/draft11-outline-audit/report.md"],
            "TeachingRole": "串联第 2 章至第 14 章的两条核心模型推算轨道，展现约束逐章具体化的演进脉络"
        },
        {
            "figure_id": "图 14-17",
            "number": 17,
            "file": "ch14-rubin-v41-codesign-mapping-matrix.svg",
            "asset": "ch14/ch14-rubin-v41-codesign-mapping-matrix.svg",
            "section": "14.8.2",
            "caption": "下一代 Vera Rubin NVL72（50 PFLOPS NVFP4 / 22 TB/s HBM4 / 3.6 TB/s NVLink 6）与 V4.1/K3 协同映射矩阵",
            "source_refs": ["references/files/specs/nvidia-rubin-system.md"],
            "TeachingRole": "展望 Rubin 世代硬件参数如何继续强化跨层状态压缩、微缩量化与块扩散草稿"
        },
        {
            "figure_id": "图 14-18",
            "number": 18,
            "file": "ch14-four-dimensional-codesign-closed-loop.svg",
            "asset": "ch14/ch14-four-dimensional-codesign-closed-loop.svg",
            "section": "14.8.2",
            "caption": "全书总结：从算法结构、运行时调度、硅片微架构到吉瓦级能源站点的四维协同设计循环",
            "source_refs": ["manuscripts/ch14/figure-data.json"],
            "TeachingRole": "完成第 1–14 章技术收束并自然引出第 15–16 章的吉瓦级能源与机密计算底座"
        },
        {
            "figure_id": "图 14-19",
            "number": 19,
            "file": "ch14-chakra-execution-trace-graph-simulation.svg",
            "asset": "ch14/ch14-chakra-execution-trace-graph-simulation.svg",
            "section": "14.9.1",
            "caption": "MLCommons Chakra 计算—通信依赖执行轨迹图（ETG）与 Phantora（NSDI '26）、Charon（MLSys '26）集群级细粒度模拟框架及适用边界对比",
            "source_refs": ["research/2026-infra-survey/reading-nsdi-2026.md", "research/2026-infra-survey/reading-mlsys-2026.md"],
            "TeachingRole": "建立从静态算子图向耦合计算—通信依赖轨迹图演进的仿真方法论，并明确单卡拦截模拟不能替代解析下界与真实集群验证"
        },
        {
            "figure_id": "图 14-20",
            "number": 20,
            "file": "ch14-neusight-tile-roofline-calibration.svg",
            "asset": "ch14/ch14-neusight-tile-roofline-calibration.svg",
            "section": "14.9.1",
            "caption": "NeuSight（ASPLOS '25，源码 6945927d）瓦片级 Roofline 利用率校准架构，以及展示“整段总误差为 0% 但逐段误差达 33.3%/50%”的阶段归因陷阱算例",
            "source_refs": ["references/framework-history/2026-09-09/neusight-calibration/NOTES.md", "references/framework-history/2026-09-09/neusight-calibration/arithmetic.json"],
            "TeachingRole": "揭示瓦片级性能预测器对历史数据库与软件栈版本的依赖，并证明必须核对逐阶段误差而非仅看端到端总和"
        },
        {
            "figure_id": "图 14-21",
            "number": 21,
            "file": "ch14-farskip-collective-delayed-dependency.svg",
            "asset": "ch14/ch14-farskip-collective-delayed-dependency.svg",
            "section": "14.9.2",
            "caption": "标准 MoE 逐层阻塞式 All-to-All 通信与 FarSkip-Collective（MLSys '26）通过跨层残差跳步将专家输出延后至第 l+2 层消费、从而用整层计算时间 T_layer 掩盖通信的对比",
            "source_refs": ["research/2026-infra-survey/reading-mlsys-2026.md"],
            "TeachingRole": "演示如何通过修改模型层间依赖图打破集合通信的串行阻塞，并强调其对重训练/知识蒸馏的精度对齐要求"
        },
        {
            "figure_id": "图 14-22",
            "number": 22,
            "file": "ch14-moe-granularity-vs-network-radix-co-design.svg",
            "asset": "ch14/ch14-moe-granularity-vs-network-radix-co-design.svg",
            "section": "14.9.2",
            "caption": "MoE 专家细粒度切分倍数 G 的组合表达收益与 All-to-All 小包扇出膨胀、Tensor Core 最小分组块 M_tile=64 限制的跨层帕累托拐点",
            "source_refs": ["manuscripts/ch14/figure-data.json"],
            "TeachingRole": "推导专家切分粒度 G 与超节点网络基数、矩阵乘瓦片尺寸联立求解的定量边界"
        },
        {
            "figure_id": "图 14-23",
            "number": 23,
            "file": "ch14-kv-head-ratio-vs-tensor-parallel-limit.svg",
            "asset": "ch14/ch14-kv-head-ratio-vs-tensor-parallel-limit.svg",
            "section": "14.9.2",
            "caption": "GQA 查询头与 KV 头数量配比在无冗余张量并行区（TP ≤ n_kv）与过激进压缩导致 KV 头复制膨胀区（TP > n_kv）的硬约束对比",
            "source_refs": ["manuscripts/ch14/figure-data.json"],
            "TeachingRole": "说明模型注意力头超参数如何直接决定推理部署时的最大无冗余张量并行度"
        },
        {
            "figure_id": "图 14-24",
            "number": 24,
            "file": "ch14-codesign-pre-silicon-to-production-loop.svg",
            "asset": "ch14/ch14-codesign-pre-silicon-to-production-loop.svg",
            "section": "14.9.2",
            "caption": "从四资源下界筛选、Chakra/NeuSight 细粒度仿真、算法—系统—硬件联立改造到预注册证伪实验（实验 14-01 / 14-02）与全生命周期 TCO 摊销的协同工程飞轮",
            "source_refs": ["manuscripts/ch14/figure-data.json"],
            "TeachingRole": "收束第 14 章全章协同设计方法论，串联流片前探索与生产集群实测验证"
        },
        {
            "figure_id": "图 14-25",
            "number": 25,
            "file": "ch14-hardware-aware-hybrid-layer-allocation-pareto.svg",
            "asset": "ch14/ch14-hardware-aware-hybrid-layer-allocation-pareto.svg",
            "section": "14.9.3",
            "caption": "固定硬件显存与延迟预算下，混合层配比（线性递推层比例 ρ_lin vs. 全局注意力层 1-ρ_lin）的质量—吞吐帕累托寻优曲面",
            "source_refs": ["manuscripts/ch14/figure-data.json"],
            "TeachingRole": "建立以真实硬件 KV 预算与延迟为目标的异构混合层整数规划模型"
        },
        {
            "figure_id": "图 14-26",
            "number": 26,
            "file": "ch14-sparse-upcycling-and-attention-distillation-pipeline.svg",
            "asset": "ch14/ch14-sparse-upcycling-and-attention-distillation-pipeline.svg",
            "section": "14.9.3",
            "caption": "稠密全注意力模型通过 Sparse Upcycling 与跨层注意力蒸馏低成本转化为系统友好型混合 MoE 架构的训练流水线",
            "source_refs": ["manuscripts/ch14/figure-data.json"],
            "TeachingRole": "展示如何仅用从头预训练约 20% 的算力（省 79.71%）将稠密模型热启动迁移为混合线性注意力与稀疏 MoE 架构"
        },
        {
            "figure_id": "图 14-27",
            "number": 27,
            "file": "ch14-moe-granularity-vs-memory-bandwidth-pareto.svg",
            "asset": "ch14/ch14-moe-granularity-vs-memory-bandwidth-pareto.svg",
            "section": "14.9.3",
            "caption": "MoE 细粒度专家切分因子 G 的表达收益与矩阵瓦片填充率拐点（G*=8 时 d_exp=512 严丝合缝对齐瓦片下界，单层耗时 0.74 ms），以及 Vera Rubin NVL72 与 TPU v7 在 671B/37B MoE 上的单步吞吐对比",
            "source_refs": ["manuscripts/ch14/figure-data.json", "experiments/ch14/14-03/README.md"],
            "TeachingRole": "建立 MoE 专家切分粒度 G 与加速器矩阵核心瓦片尺寸、HBM4/NVLink 6 带宽比联立寻优的定量边界"
        },
        {
            "figure_id": "图 14-28",
            "number": 28,
            "file": "ch14-hardware-aware-neural-architecture-search-loop.svg",
            "asset": "ch14/ch14-hardware-aware-neural-architecture-search-loop.svg",
            "section": "14.9.3",
            "caption": "面向下一代加速器的三维超参数（专家粒度 G、激活比 ρ_act、全注意力混合比 r_attn）硬件感知神经架构搜索（HW-NAS）闭环：从 36 组候选中筛出 6 组可行解并锁定 (8, 3.125%, 25%) 最优工作点",
            "source_refs": ["manuscripts/ch14/figure-data.json", "experiments/ch14/14-03/README.md"],
            "TeachingRole": "展示三维架构超参数在硬件 TPOT 与状态显存红线约束下的闭环帕累托寻优流程"
        }
    ]
    (HERE / 'figure-index.json').write_text(json.dumps(fig_index, ensure_ascii=False, indent=2) + '\n')

    md = HERE.parent / '14-模型与系统硬件协同设计.md'
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

        work = re.sub(r'!\[([^\]]*)\]\((ch14/[^)]+)\)', repl_img, text_md)
        body = markdown.markdown(work, extensions=['tables', 'fenced_code', 'footnotes', 'toc'])
        html = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>第 14 章：模型与系统硬件协同设计</title></head><body>{body}</body></html>\n'
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
    print(f'Built {len(outputs)} figure files for Chapter 14; 0 layout warnings.')


if __name__ == '__main__':
    main()
