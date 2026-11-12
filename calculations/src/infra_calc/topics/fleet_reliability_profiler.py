"""Ch15: Fleet MTBF, Poisson hot-spare queue, Young-Daly ETTR, Oobleck/Kingman, eBPF/Kineto, and Monarch."""
from __future__ import annotations

import math
from .. import hardware
from ..sources import model_config


def _poisson_cdf(lam: float, k_max: int) -> float:
    cum = 0.0
    term = math.exp(-lam)
    for k in range(0, k_max + 1):
        if k > 0:
            term *= lam / k
        cum += term
    return cum


def calculate(
    mega_fleet_gpus: int = 100_000,
    per_gpu_mtbf_hours: float = 50_677.0,
    m_cluster_sec: float = 1800.0,
    ckpt_traditional_s: float = 180.0,
    recovery_traditional_s: float = 300.0,
    ckpt_fast_s: float = 5.0,
    recovery_fast_s: float = 5.0,
    node_failure_rate_per_h: float = 2.0,
    node_repair_mttr_h: float = 12.0,
    profiler_fleet_gpus: int = 16_384,
    cupti_overhead_frac: float = 0.08,
    ebpf_l0_overhead_frac: float = 0.0035,
    dynolog_l1_amortized_frac: float = 0.0007,
) -> dict:
    dsv3 = model_config("deepseek-v3")
    hw_map = {d["id"]: d for d in hardware.catalog()["devices"]}

    # 1. 算例 15-1: 100k-GPU fleet MTBF (30.41 min) & Plan A vs Plan B Young-Daly ETTR (locked to experiments/ch15/15-01)
    fleet_mtbf_min = round((per_gpu_mtbf_hours * 60.0) / mega_fleet_gpus, 2)  # 30.41 min
    tau_opt_plan_a_s = round(math.sqrt(2.0 * ckpt_traditional_s * m_cluster_sec), 2)  # 804.98 s
    tau_opt_plan_b_s = round(math.sqrt(2.0 * ckpt_fast_s * m_cluster_sec), 2)         # 134.16 s
    ettr_plan_a_pct = 63.22
    ettr_plan_b_pct = 92.34
    gpus_saved_by_plan_b = int(round(mega_fleet_gpus * (ettr_plan_b_pct - ettr_plan_a_pct) / 100.0))  # 29,120 GPUs

    # 2. 算例 15-2: Hot-spare node pool Poisson M/G/inf sizing (lambda=2.0/h, MTTR=12.0h -> L=24.0, S_min=40 for >=99.9%)
    offered_load_l = round(node_failure_rate_per_h * node_repair_mttr_h, 2)           # 24.0 nodes
    prob_no_exhaust_at_mean_pct = round(_poisson_cdf(offered_load_l, int(offered_load_l)) * 100.0, 2)  # 55.40%
    min_spares_999 = next(s for s in range(1, 100) if _poisson_cdf(offered_load_l, s) >= 0.999)        # 40 nodes
    prob_at_s_min_pct = round(_poisson_cdf(offered_load_l, min_spares_999) * 100.0, 3)                 # 99.901%
    spare_pool_ratio_pct = round(min_spares_999 / (mega_fleet_gpus / 8.0) * 100.0, 2)                  # 0.32%

    # 3. 算例 15-3: ABFT checksum overhead on 4096x4096 GEMM: 6*N^2 / (2*N^3) = 3/4096 = 0.0732%
    abft_overhead_pct = round((3.0 / 4096.0) * 100.0, 4)  # 0.0732%

    # 4. 算例 15-7: Oobleck 1.45 s template failover & Kingman bursty Poisson Goodput collapse (experiments/ch15/15-03)
    oobleck_reconfig_s = round(0.25 + 0.85 + 0.35, 2)                           # 1.45 s
    oobleck_speedup_vs_restart_x = round(300.0 / oobleck_reconfig_s, 2)         # 206.90x
    oobleck_retained_pct = round((30 * 1.00 + 2 * 0.80) / 32.0 * 100.0, 2)      # 98.75%
    parachute_retained_pct = round(30.0 / 32.0 * 100.0, 2)                      # 93.75%

    rho = 0.85
    service_ms = 120.0
    prefill_ms = 80.0
    var_factor = 0.5 * (1.0**2 + 1.2**2)                                        # 1.22
    wq_bursty_ms = round((rho / (1.0 - rho)) * var_factor * service_ms, 2)      # 829.60 ms
    ttft_bursty_ms = round(prefill_ms + wq_bursty_ms, 2)                        # 909.60 ms
    slo_attain_ratio = 1.0 - rho * math.exp(-(500.0 - prefill_ms) / wq_bursty_ms)
    slo_attainment_bursty_pct = round(slo_attain_ratio * 100.0, 2)              # 48.77%
    goodput_bursty_req_s = round((rho * 1000.0 / service_ms) * slo_attain_ratio, 2)  # 3.45 req/s
    odds_safe = (300.0 - prefill_ms) / (var_factor * service_ms)
    max_safe_utilization_pct = round((odds_safe / (1.0 + odds_safe)) * 100.0, 2)     # 60.04%

    # 5. Ch15.6 算例 15-8: 16,384-GPU CUPTI always-on vs eBPF L0 + Dynolog/Kineto on-demand L1
    cupti_equiv_gpus_burned = round(profiler_fleet_gpus * cupti_overhead_frac, 2)  # 1310.72 -> ~1311 GPUs
    cupti_equiv_gpus_int = round(cupti_equiv_gpus_burned)
    cupti_daily_trace_pb = 1.42
    tiered_total_overhead_pct = round((ebpf_l0_overhead_frac + dynolog_l1_amortized_frac) * 100.0, 2)  # 0.42%
    tiered_equiv_gpus_burned = round(profiler_fleet_gpus * (ebpf_l0_overhead_frac + dynolog_l1_amortized_frac), 2)
    tiered_daily_trace_tb = 1.80
    storage_reduction_x = round((cupti_daily_trace_pb * 1000.0) / tiered_daily_trace_tb, 1)

    # 6. Ch15.6 算例 15-9: 8-Rank (TP=2, PP=2, DP=2) wait-chain DAG straggler attribution
    ranks_timeline = []
    for r in range(8):
        enter_ms = 200.0 if r == 5 else (62.0 if r in (4, 7) else 60.0)
        collective_finish_ms = 204.0
        ranks_timeline.append({
            "rank": r,
            "enter_collective_ms": enter_ms,
            "exit_collective_ms": collective_finish_ms,
            "apparent_nccl_kernel_ms": round(collective_finish_ms - enter_ms, 1),
            "role": "root_cause_straggler" if r == 5 else "victim_waiter",
        })
    root_cause_rank = max(ranks_timeline, key=lambda x: x["enter_collective_ms"])["rank"]
    victim_count = sum(1 for x in ranks_timeline if x["role"] == "victim_waiter")

    # 7. Ch15.6 算例 15-10: PyTorch Monarch actor-mesh local fault isolation vs SPMD torchrun global restart
    monarch_recovery_s = 0.85
    spmd_global_restart_s = 142.0
    monarch_speedup_x = round(spmd_global_restart_s / monarch_recovery_s, 1)

    return {
        "schema": "h2-topic-v1",
        "calculation": "fleet-reliability-profiler",
        "chapter": 15,
        "title": "第 15 章：十万卡 MTBF/热备池排队、eBPF+Dynolog 三级遥测、等待链归因与 Monarch 局部自愈账本",
        "scenario": {
            "mega_fleet_gpus": mega_fleet_gpus,
            "per_gpu_mtbf_hours": per_gpu_mtbf_hours,
            "node_failure_rate_per_h": node_failure_rate_per_h,
            "node_repair_mttr_h": node_repair_mttr_h,
            "profiler_fleet_gpus": profiler_fleet_gpus,
            "cupti_overhead_frac": cupti_overhead_frac,
            "ebpf_l0_overhead_frac": ebpf_l0_overhead_frac,
            "dynolog_l1_amortized_frac": dynolog_l1_amortized_frac,
            "deepseek_v3_hidden_size": dsv3["hidden_size"],
            "b200_hbm_gb": hw_map["b200-sxm"]["memory"]["nominal_capacity"],
        },
        "summary": {
            "fleet_100k_mtbf_minutes": fleet_mtbf_min,
            "young_daly_plan_a_opt_ckpt_s": tau_opt_plan_a_s,
            "young_daly_plan_b_opt_ckpt_s": tau_opt_plan_b_s,
            "fleet_100k_ettr_plan_a_pct": ettr_plan_a_pct,
            "fleet_100k_ettr_plan_b_pct": ettr_plan_b_pct,
            "fleet_100k_gpus_saved_plan_b": gpus_saved_by_plan_b,
            "hot_spare_offered_load_nodes": offered_load_l,
            "hot_spare_prob_at_mean24_pct": prob_no_exhaust_at_mean_pct,
            "hot_spare_min_nodes_99_9pct": min_spares_999,
            "hot_spare_prob_at_smin40_pct": prob_at_s_min_pct,
            "hot_spare_pool_ratio_pct": spare_pool_ratio_pct,
            "abft_4096_overhead_pct": abft_overhead_pct,
            "oobleck_reconfig_seconds": oobleck_reconfig_s,
            "oobleck_vs_restart_speedup_x": oobleck_speedup_vs_restart_x,
            "oobleck_retained_throughput_pct": oobleck_retained_pct,
            "parachute_retained_throughput_pct": parachute_retained_pct,
            "kingman_bursty_wq_ms": wq_bursty_ms,
            "kingman_bursty_ttft_ms": ttft_bursty_ms,
            "kingman_bursty_slo_attainment_pct": slo_attainment_bursty_pct,
            "kingman_bursty_goodput_req_s": goodput_bursty_req_s,
            "kingman_max_safe_utilization_pct": max_safe_utilization_pct,
            "cupti_16k_burned_gpus_exact": cupti_equiv_gpus_burned,
            "cupti_16k_burned_gpus_rounded": cupti_equiv_gpus_int,
            "cupti_daily_trace_pb": cupti_daily_trace_pb,
            "ebpf_dynolog_total_overhead_pct": tiered_total_overhead_pct,
            "ebpf_dynolog_burned_gpus": tiered_equiv_gpus_burned,
            "ebpf_dynolog_daily_trace_tb": tiered_daily_trace_tb,
            "trace_storage_reduction_x": storage_reduction_x,
            "wait_chain_root_cause_rank": root_cause_rank,
            "wait_chain_victim_ranks_stripped": victim_count,
            "monarch_isolation_seconds": monarch_recovery_s,
            "spmd_global_restart_seconds": spmd_global_restart_s,
            "monarch_vs_spmd_speedup_x": monarch_speedup_x,
        },
        "wait_chain_8rank_timeline": ranks_timeline,
        "assumptions": [
            "算例 15-1 与 15-2：单卡 MTBF=50,677 h 外推至十万卡 MTBF=30.41 min，异步快照+热迁移将 ETTR 从 63.22% 提至 92.34%（净挽回 29,120 卡）；泊松维修队列 L=24.0 台需配置 S_min=40 台（0.32%）达 99.901% 零降级可用率。",
            "算例 15-7：Oobleck 预实例化模板 1.45 s 重配置（快 206.90x，保留 98.75% 吞吐）；泊松突发流量（Ca=1.0, Cs=1.2）在 rho=0.85 下排队膨胀至 829.60 ms，安全利用率上限为 60.04%。",
            "算例 15-8 至 15-10：eBPF L0（0.35%）+ Dynolog/Kineto 按需 L1（0.07%）将 16,384 卡遥测损耗控制在 0.42%（对比全量 CUPTI 8% 烧毁 1,310.72 卡），等待链逆向 arg max t_enter 精准锁定 Rank 5。",
        ],
    }
