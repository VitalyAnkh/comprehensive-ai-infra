#!/usr/bin/env python3
"""Experiment 20-05: Heterogeneous Robot SoC SLC/QoS Isolation & 64-GPU Closed-Loop World Model Regression."""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"
SUMMARY_PATH = HERE / "results" / "summary.json"


def compute_metrics() -> dict:
    # 1. Worked Example 20-9: Edge heterogeneous SoC SLC way-partitioning & memory QoS for 1 kHz MPC
    slc_total_mib = 64
    slc_total_ways = 16
    mib_per_way = slc_total_mib // slc_total_ways  # 4 MiB/way
    mpc_working_set_mib = 3.2
    mpc_deadline_ms = 1.00

    reserved_rt_ways = math.ceil(mpc_working_set_mib / mib_per_way) + 1  # 2 ways = 8 MiB (with double-buffer margin)
    reserved_pva_sys1_ways = 4  # 16 MiB
    reserved_gpu_sys2_ways = slc_total_ways - reserved_rt_ways - reserved_pva_sys1_ways  # 10 ways = 40 MiB

    peak_dram_gbps = 273.0
    non_rt_bw_cap_gbps = 205.0
    non_rt_bw_cap_pct = round(non_rt_bw_cap_gbps / peak_dram_gbps * 100.0, 1)  # 75.1%

    unisolated_wcet_ms = 1.82
    isolated_wcet_ms = 0.41
    wcet_speedup_x = round(unisolated_wcet_ms / isolated_wcet_ms, 2)  # 4.44x
    wcet_margin_ms = round(mpc_deadline_ms - isolated_wcet_ms, 2)  # 0.59 ms

    # 2. Worked Example 20-10: 64-GPU Closed-Loop Neural World Model regression vs physical robot cells
    scenarios = 1000
    rollouts_per_scenario = 100
    total_rollouts = scenarios * rollouts_per_scenario  # 100,000
    world_model_gpus = 64
    per_gpu_rollouts_per_hour = 3906.25
    cluster_rollouts_per_hour = world_model_gpus * per_gpu_rollouts_per_hour  # 250,000
    world_model_eval_hours = round(total_rollouts / cluster_rollouts_per_hour, 2)  # 0.40 h (24.0 min)
    world_model_gpu_hours = round(world_model_gpus * world_model_eval_hours, 2)  # 25.60 GPU-h
    gpu_hour_rate_usd = 30.0
    world_model_cost_usd = round(world_model_gpu_hours * gpu_hour_rate_usd, 2)  # $768.00

    sec_per_physical_trial = 120.0
    physical_cell_count = 10
    physical_cell_hours = round((total_rollouts * sec_per_physical_trial) / 3600.0, 2)  # 3333.33 cell-h
    physical_wall_hours = round(physical_cell_hours / physical_cell_count, 2)  # 333.33 h
    physical_cell_rate_usd = 33.408
    physical_eval_cost_usd = round((total_rollouts * sec_per_physical_trial / 3600.0) * physical_cell_rate_usd, 2)  # $111,360.00
    cost_compression_x = round(physical_eval_cost_usd / world_model_cost_usd, 1)  # 145.0x
    wall_time_speedup_x = round(physical_wall_hours / world_model_eval_hours, 2)  # 833.33x

    policy_comparison = {
        "policy_a_conservative_flow": {
            "open_loop_action_mse": 0.0048,
            "closed_loop_collision_violation_pct": 1.1,
            "closed_loop_task_success_pct": 89.5,
            "release_gate_passed": True,
        },
        "policy_b_overfit_mse": {
            "open_loop_action_mse": 0.0041,
            "closed_loop_collision_violation_pct": 14.2,
            "closed_loop_task_success_pct": 54.0,
            "release_gate_passed": False,
        },
    }

    # 3. Exercise 20-17 & Experiment 20-18
    ex17_slc_mib, ex17_ways = 32, 16
    ex17_mib_per_way = ex17_slc_mib / ex17_ways  # 2.0 MiB
    ex17_rt_ws_mib = 5.4
    ex17_min_rt_ways = math.ceil(ex17_rt_ws_mib / ex17_mib_per_way)  # 3 ways (6 MiB)
    ex17_peak_bw_gbps = 204.8
    ex17_rt_reserved_bw_gbps = 12.8
    ex17_non_rt_cap_gbps = round(ex17_peak_bw_gbps * 0.75 - ex17_rt_reserved_bw_gbps, 2)  # 140.80 GB/s
    ex17_unisolated_wcet_ms = 2.15
    ex17_isolated_wcet_ms = 0.48

    ex18_gpus = 128
    ex18_rollouts = 250_000
    ex18_base_hours = round(ex18_rollouts / (ex18_gpus * per_gpu_rollouts_per_hour), 2)  # 0.50 h
    ex18_base_cost_usd = round(ex18_gpus * ex18_base_hours * gpu_hour_rate_usd, 2)  # $1,920.00
    ex18_phys_cost_usd = round((ex18_rollouts * sec_per_physical_trial / 3600.0) * physical_cell_rate_usd, 2)  # $278,400.00
    ex18_early_stop_saving_pct = 21.50
    ex18_early_stop_hours = round(ex18_base_hours * (1.0 - ex18_early_stop_saving_pct / 100.0), 4)  # 0.3925 h
    ex18_early_stop_cost_usd = round(ex18_gpus * ex18_early_stop_hours * gpu_hour_rate_usd, 2)  # $1,507.20

    return {
        "experiment_id": "20-05",
        "soc_slc_qos_isolation_ex20_9": {
            "slc_total_mib": slc_total_mib,
            "slc_total_ways": slc_total_ways,
            "reserved_rt_ways": reserved_rt_ways,
            "reserved_pva_sys1_ways": reserved_pva_sys1_ways,
            "reserved_gpu_sys2_ways": reserved_gpu_sys2_ways,
            "peak_dram_gbps": peak_dram_gbps,
            "non_rt_bw_cap_gbps": non_rt_bw_cap_gbps,
            "non_rt_bw_cap_pct": non_rt_bw_cap_pct,
            "reflex_1khz_deadline_ms": mpc_deadline_ms,
            "reflex_wcet_unisolated_ms": unisolated_wcet_ms,
            "reflex_wcet_slc_qos_isolated_ms": isolated_wcet_ms,
            "wcet_speedup_x": wcet_speedup_x,
            "wcet_margin_ms": wcet_margin_ms,
        },
        "world_model_regression_ex20_10": {
            "regression_scenarios": scenarios,
            "rollouts_per_scenario": rollouts_per_scenario,
            "total_closed_loop_rollouts": total_rollouts,
            "world_model_gpus": world_model_gpus,
            "world_model_64gpu_eval_hours": world_model_eval_hours,
            "world_model_gpu_hours": world_model_gpu_hours,
            "world_model_64gpu_cost_usd": world_model_cost_usd,
            "physical_robot_eval_cost_usd": physical_eval_cost_usd,
            "world_model_cost_compression_x": cost_compression_x,
            "wall_time_speedup_x": wall_time_speedup_x,
            "policy_comparison": policy_comparison,
        },
        "exercises_20_17_and_20_18": {
            "ex17_min_rt_ways": ex17_min_rt_ways,
            "ex17_non_rt_cap_gbps": ex17_non_rt_cap_gbps,
            "ex17_isolated_wcet_ms": ex17_isolated_wcet_ms,
            "ex18_base_hours": ex18_base_hours,
            "ex18_base_cost_usd": ex18_base_cost_usd,
            "ex18_phys_cost_usd": ex18_phys_cost_usd,
            "ex18_early_stop_saving_pct": ex18_early_stop_saving_pct,
            "ex18_early_stop_hours": ex18_early_stop_hours,
            "ex18_early_stop_cost_usd": ex18_early_stop_cost_usd,
        },
    }


def main() -> None:
    metrics = compute_metrics()
    payload = json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    for p in (EXPECTED_PATH, SUMMARY_PATH):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(payload, encoding="utf-8")
    print(payload.strip())


if __name__ == "__main__":
    main()
