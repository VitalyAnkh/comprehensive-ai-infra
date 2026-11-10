#!/usr/bin/env python3
"""Experiment 15-03: Oobleck Pre-Instantiated Pipeline Failover & Bursty Poisson Traffic Goodput Benchmark."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_PATH = ROOT / "expected" / "metrics.json"
RESULTS_PATH = ROOT / "results" / "summary.json"


def compute_metrics() -> dict:
    # 1. Worked Example 20-7(1): Oobleck 32 pipelines x 8 stages, F = 2 node failures
    num_pipelines = 32
    stages_per_pipeline = 8
    failed_nodes = 2
    t_detect_s = 0.25
    t_shard_s = 0.85
    t_sync_s = 0.35
    oobleck_reconfig_sec = round(t_detect_s + t_shard_s + t_sync_s, 2)
    full_restart_sec = 300.00
    reconfig_speedup_x = round(full_restart_sec / oobleck_reconfig_sec, 2)

    healthy_pipes = num_pipelines - failed_nodes
    oobleck_retained_throughput_pct = round(
        100.0 * (healthy_pipes * 1.00 + failed_nodes * 0.80) / num_pipelines, 2
    )
    parachute_retained_throughput_pct = round(100.0 * healthy_pipes / num_pipelines, 2)

    # Exercise 20-13: 64 pipelines x 8 stages (512 nodes), F = 4 failed nodes, degraded eff = 0.82
    ex13_para_pct = round(100.0 * (64 - 4) / 64, 2)
    ex13_oob_exact = (60 * 1.00 + 4 * 0.82) / 64
    ex13_oob_pct = round(100.0 * ex13_oob_exact, 2)
    ex13_saved_node_hours_per_day = round(512 * (ex13_oob_exact - (60 / 64)) * 24.0, 2)

    # 2. Worked Example 20-7(2): Bursty Poisson (Ca=1.0, Cs=1.2) vs. Uniform (Ca=0, Cs=0) at rho = 0.85
    service_ms = 120.0
    prefill_ms = 80.0
    slo_ttft_ms = 500.0
    rho = 0.85
    mu_req_s = 1000.0 / service_ms
    arrival_req_s = rho * mu_req_s
    w_max_ms = slo_ttft_ms - prefill_ms

    ttft_uniform_ms = round(prefill_ms, 2)
    goodput_uniform_req_s = round(arrival_req_s, 2)

    var_factor = 0.5 * (1.0 ** 2 + 1.2 ** 2)
    wq_bursty_ms = round((rho / (1.0 - rho)) * var_factor * service_ms, 2)
    ttft_bursty_ms = round(prefill_ms + wq_bursty_ms, 2)
    ttft_inflation_x = round(ttft_bursty_ms / ttft_uniform_ms, 2)

    slo_attain_ratio = 1.0 - rho * math.exp(-w_max_ms / wq_bursty_ms)
    slo_attainment_bursty_pct = round(100.0 * slo_attain_ratio, 2)
    goodput_bursty_req_s = round(arrival_req_s * slo_attain_ratio, 2)
    goodput_drop_pct = round(100.0 * (goodput_uniform_req_s - goodput_bursty_req_s) / goodput_uniform_req_s, 2)

    target_wq_ms = 300.0 - prefill_ms
    odds = target_wq_ms / (var_factor * service_ms)
    max_safe_utilization_pct = round(100.0 * (odds / (1.0 + odds)), 2)

    # Exercise 20-14: rho = 0.80, Ts = 100.0 ms, T_prefill = 60.0 ms, Ca = Cs = 1.5
    ex14_wq_bursty_ms = round((0.80 / 0.20) * (0.5 * (1.5 ** 2 + 1.5 ** 2)) * 100.0, 2)
    ex14_ttft_bursty_ms = round(60.0 + ex14_wq_bursty_ms, 2)
    ex14_inflation_x = round(ex14_ttft_bursty_ms / 60.0, 2)

    return {
        "oobleck_elastic_pipeline_failover": {
            "num_pipelines": num_pipelines,
            "stages_per_pipeline": stages_per_pipeline,
            "failed_nodes": failed_nodes,
            "t_detect_s": t_detect_s,
            "t_shard_s": t_shard_s,
            "t_sync_s": t_sync_s,
            "oobleck_reconfig_sec": oobleck_reconfig_sec,
            "full_restart_sec": full_restart_sec,
            "reconfig_speedup_x": reconfig_speedup_x,
            "oobleck_retained_throughput_pct": oobleck_retained_throughput_pct,
            "parachute_retained_throughput_pct": parachute_retained_throughput_pct,
            "ex20_13_parachute_pct": ex13_para_pct,
            "ex20_13_oobleck_pct": ex13_oob_pct,
            "ex20_13_saved_node_hours_per_day": ex13_saved_node_hours_per_day,
        },
        "bursty_goodput_vs_uniform": {
            "rho": rho,
            "service_ms": service_ms,
            "prefill_ms": prefill_ms,
            "slo_ttft_ms": slo_ttft_ms,
            "ttft_uniform_ms": ttft_uniform_ms,
            "goodput_uniform_req_s": goodput_uniform_req_s,
            "wq_bursty_ms": wq_bursty_ms,
            "ttft_bursty_ms": ttft_bursty_ms,
            "ttft_inflation_x": ttft_inflation_x,
            "slo_attainment_bursty_pct": slo_attainment_bursty_pct,
            "goodput_bursty_req_s": goodput_bursty_req_s,
            "goodput_drop_pct": goodput_drop_pct,
            "max_safe_utilization_pct": max_safe_utilization_pct,
            "ex20_14_wq_bursty_ms": ex14_wq_bursty_ms,
            "ex20_14_ttft_bursty_ms": ex14_ttft_bursty_ms,
            "ex20_14_inflation_x": ex14_inflation_x,
        },
    }


def main() -> None:
    metrics = compute_metrics()
    EXPECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    for p in (EXPECTED_PATH, RESULTS_PATH):
        p.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
