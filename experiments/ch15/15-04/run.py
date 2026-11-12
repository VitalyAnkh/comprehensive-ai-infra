#!/usr/bin/env python3
"""Experiment 15-04: Three-Tier eBPF + Dynolog/Kineto Telemetry Overhead & Blink Sampling Skid Calibration."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_PATH = ROOT / "expected" / "metrics.json"
RESULTS_PATH = ROOT / "results" / "summary.json"


def solve_3x3_lower_triangular(m: list[list[float]], y: list[float]) -> list[float]:
    x0 = y[0] / m[0][0]
    x1 = (y[1] - m[1][0] * x0) / m[1][1]
    x2 = (y[2] - m[2][0] * x0 - m[2][1] * x1) / m[2][2]
    return [x0, x1, x2]


def compute_metrics() -> dict:
    # 1. Worked Example 15-8: 16,384-GPU fleet always-on CUPTI vs eBPF L0 + Dynolog/Kineto on-demand L1
    fleet_gpus = 16_384
    cupti_overhead_frac = 0.08
    cupti_burned_gpus_exact = round(fleet_gpus * cupti_overhead_frac, 2)  # 1310.72
    cupti_burned_gpus_rounded = round(cupti_burned_gpus_exact)            # 1311
    cupti_daily_trace_pb = 1.42                                           # 1,420 TB/day

    ebpf_l0_overhead_frac = 0.0035                                        # 0.35%
    kineto_active_overhead_frac = 0.08                                    # 8.0% during triggered burst
    dynolog_duty_cycle_frac = 0.00875                                     # 0.875% fleet-amortized burst duty cycle
    dynolog_l1_amortized_frac = round(kineto_active_overhead_frac * dynolog_duty_cycle_frac, 6)  # 0.0007 (0.07%)

    total_tiered_frac = ebpf_l0_overhead_frac + dynolog_l1_amortized_frac  # 0.0042
    total_tiered_overhead_pct = round(total_tiered_frac * 100.0, 2)        # 0.42%
    tiered_burned_gpus = round(fleet_gpus * total_tiered_frac, 2)          # 68.81
    net_saved_gpus = round(cupti_burned_gpus_exact - tiered_burned_gpus, 2)  # 1241.91

    ebpf_l0_daily_tb = 1.70
    dynolog_l1_daily_tb = 0.10
    tiered_daily_trace_tb = round(ebpf_l0_daily_tb + dynolog_l1_daily_tb, 2)  # 1.80 TB/day
    trace_storage_reduction_x = round((cupti_daily_trace_pb * 1000.0) / tiered_daily_trace_tb, 1)  # 788.9x

    # 2. Exercise 15-15: Blink flat-workload PMU skid transition matrix inversion
    skid_matrix = [
        [0.60, 0.00, 0.00],
        [0.40, 0.75, 0.00],
        [0.00, 0.25, 1.00],
    ]
    raw_samples = [300.0, 500.0, 200.0]
    true_counts = solve_3x3_lower_triangular(skid_matrix, raw_samples)
    total_true = sum(true_counts)
    raw_total = sum(raw_samples)
    func_names = ["tokenize_dispatch", "launch_stub", "post_hook"]
    distorted_share_pct = {k: round(100.0 * v / raw_total, 2) for k, v in zip(func_names, raw_samples)}
    debiased_share_pct = {k: round(100.0 * v / total_true, 2) for k, v in zip(func_names, true_counts)}

    return {
        "experiment_id": "15-04",
        "three_tier_telemetry_ledger": {
            "fleet_gpus": fleet_gpus,
            "cupti_overhead_pct": round(cupti_overhead_frac * 100.0, 2),
            "cupti_burned_gpus_exact": cupti_burned_gpus_exact,
            "cupti_burned_gpus_rounded": cupti_burned_gpus_rounded,
            "cupti_daily_trace_pb": cupti_daily_trace_pb,
            "ebpf_l0_overhead_pct": round(ebpf_l0_overhead_frac * 100.0, 2),
            "dynolog_l1_amortized_pct": round(dynolog_l1_amortized_frac * 100.0, 2),
            "ebpf_dynolog_total_overhead_pct": total_tiered_overhead_pct,
            "ebpf_dynolog_burned_gpus": tiered_burned_gpus,
            "net_saved_gpus": net_saved_gpus,
            "ebpf_dynolog_daily_trace_tb": tiered_daily_trace_tb,
            "trace_storage_reduction_x": trace_storage_reduction_x,
        },
        "blink_pmu_skid_calibration": {
            "raw_samples": [int(v) for v in raw_samples],
            "true_counts": [int(round(v)) for v in true_counts],
            "distorted_share_pct": distorted_share_pct,
            "debiased_share_pct": debiased_share_pct,
        },
    }


def main() -> None:
    metrics = compute_metrics()
    EXPECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    for p in (EXPECTED_PATH, RESULTS_PATH):
        p.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
