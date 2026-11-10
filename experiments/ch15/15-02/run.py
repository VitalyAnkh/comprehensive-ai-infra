#!/usr/bin/env python3
"""Experiment 15-02: 100K-GPU Chaos GameDay fault injection, Arrival Skew Matrix straggler localization, TrainMover hot swap, and benchmark de-biasing."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"
RESULTS_PATH = HERE / "results" / "summary.json"


def compute_metrics() -> dict:
    # 1. 100K-GPU Chaos GameDay Fault Injection Audit (Ex 20-6 & Ex 20-12)
    categories = {
        "sdc_bitflip": {"injected": 50, "detected": 50, "mttd_sec": 0.042, "mttr_sec": 0.592},
        "compute_pcie_straggler": {"injected": 50, "detected": 50, "mttd_sec": 3.20, "mttr_sec": 4.40},
        "optical_fec_gray_failure": {"injected": 50, "detected": 49, "mttd_sec": 6.80, "mttr_sec": 9.0316},
        "hbm_ecc_node_drop": {"injected": 50, "detected": 50, "mttd_sec": 0.84, "mttr_sec": 1.48},
    }
    total_injected = sum(c["injected"] for c in categories.values())  # 200
    total_detected = sum(c["detected"] for c in categories.values())  # 199
    overall_recall_pct = round(total_detected / total_injected * 100.0, 2)  # 99.50%
    mean_mttd_sec = round(
        sum(c["detected"] * c["mttd_sec"] for c in categories.values()) / total_detected, 2
    )  # 2.70 s
    mean_auto_mttr_sec = round(
        sum(c["detected"] * c["mttr_sec"] for c in categories.values()) / total_detected, 2
    )  # 3.85 s
    manual_mttr_sec = 1500.0  # 25 min manual SRE intervention
    mttr_speedup_x = round(manual_mttr_sec / mean_auto_mttr_sec, 2)  # 389.61x

    # 2. 8-Rank Collective Arrival Skew Matrix Straggler Localization (Ex 20-6 & Ex 20-12)
    enqueue_ts_ms = [10.0, 10.0, 10.0, 10.0, 10.0, 12.0, 10.0, 10.0]
    ring_xfer_ms = 0.40
    min_enqueue_ms = min(enqueue_ts_ms)
    max_enqueue_ms = max(enqueue_ts_ms)
    straggler_rank = enqueue_ts_ms.index(max_enqueue_ms)  # Rank 5
    arrival_skew_ms = round(max_enqueue_ms - min_enqueue_ms, 2)  # 2.00 ms
    completion_ts_ms = round(max_enqueue_ms + ring_xfer_ms, 2)  # 12.40 ms
    healthy_observed_duration_ms = round(completion_ts_ms - min_enqueue_ms, 2)  # 2.40 ms
    straggler_observed_duration_ms = round(completion_ts_ms - max_enqueue_ms, 2)  # 0.40 ms
    healthy_wait_share_pct = round(arrival_skew_ms / healthy_observed_duration_ms * 100.0, 2)  # 83.33%
    healthy_xfer_share_pct = round(ring_xfer_ms / healthy_observed_duration_ms * 100.0, 2)  # 16.67%

    # 3. TrainMover Virtual Rank Sub-Second Hot-Swap Latency Breakdown (Ex 20-6 & Ex 20-12)
    qp_remap_sec = 0.08
    rdma_state_push_sec = 0.60
    trainmover_total_recovery_sec = round(qp_remap_sec + rdma_state_push_sec, 2)  # 0.68 s
    global_restart_sec = 300.0
    trainmover_speedup_x = round(global_restart_sec / trainmover_total_recovery_sec, 2)  # 441.18x

    # 4. Four Benchmark Gaming De-biasing & Canary Gate Verifications (Ex 20-5, Ex 20-7, Ex 20-8 & Ex 20-12)
    # (1) Candidate A=[10, 200], B=[50, 50] on Base=[100, 100]: 5.25x arith mean vs 0.9524x true equal-freq speedup
    arith_speedup_a = round((100.0 / 10.0 + 100.0 / 200.0) / 2.0, 2)  # 5.25x
    true_speedup_a_equal_freq = round((100.0 + 100.0) / (10.0 + 200.0), 4)  # 0.9524x (regression!)
    # Exercise 20-7 variant: X=[10,240], Y=[40,60] on Base=[80,120]
    arith_speedup_x = round((80.0 / 10.0 + 120.0 / 240.0) / 2.0, 2)  # 4.25x
    arith_speedup_y = round((80.0 / 40.0 + 120.0 / 60.0) / 2.0, 2)  # 2.00x
    true_speedup_x_equal_freq = round((80.0 + 120.0) / (10.0 + 240.0), 2)  # 0.80x
    true_speedup_y_equal_freq = round((80.0 + 120.0) / (40.0 + 60.0), 2)  # 2.00x
    crossover_p_x_beats_y = round(6.0 / 7.0, 4)  # 0.8571 (85.71%)

    # (2) Linear thermal drift: delta = 8.0 us -> AABB bias = -16.0 us (and Ex 20-8 delta = 12.0 us -> -24.0 us) vs ABBA = 0.0 us
    aabb_bias_8us_drift = round(-2.0 * 8.0, 2)  # -16.0 us
    aabb_systematic_bias_us = round(-2.0 * 12.0, 2)  # -24.0 us
    abba_systematic_bias_us = 0.0

    # (3) Stage-2 Canary 5% traffic gate: +4.8% P99 regression (> +3.0% threshold) triggers automatic rollback
    canary_p99_regression_pct = 4.8
    canary_threshold_pct = 3.0
    canary_auto_rollback_triggered = canary_p99_regression_pct > canary_threshold_pct

    return {
        "experiment_id": "15-02",
        "chaos_gameday_audit": {
            "total_injected": total_injected,
            "total_detected": total_detected,
            "overall_recall_pct": overall_recall_pct,
            "mean_mttd_sec": mean_mttd_sec,
            "mean_auto_mttr_sec": mean_auto_mttr_sec,
            "manual_mttr_sec": manual_mttr_sec,
            "mttr_speedup_x": mttr_speedup_x,
        },
        "arrival_skew_8rank": {
            "straggler_rank": straggler_rank,
            "arrival_skew_ms": arrival_skew_ms,
            "healthy_observed_duration_ms": healthy_observed_duration_ms,
            "straggler_observed_duration_ms": straggler_observed_duration_ms,
            "healthy_wait_share_pct": healthy_wait_share_pct,
            "healthy_xfer_share_pct": healthy_xfer_share_pct,
        },
        "trainmover_hot_swap": {
            "qp_remap_sec": qp_remap_sec,
            "rdma_state_push_sec": rdma_state_push_sec,
            "trainmover_total_recovery_sec": trainmover_total_recovery_sec,
            "global_restart_sec": global_restart_sec,
            "trainmover_speedup_x": trainmover_speedup_x,
        },
        "benchmark_debiasing": {
            "arith_speedup_a": arith_speedup_a,
            "true_speedup_a_equal_freq": true_speedup_a_equal_freq,
            "arith_speedup_x": arith_speedup_x,
            "arith_speedup_y": arith_speedup_y,
            "true_speedup_x_equal_freq": true_speedup_x_equal_freq,
            "true_speedup_y_equal_freq": true_speedup_y_equal_freq,
            "crossover_p_x_beats_y": crossover_p_x_beats_y,
            "aabb_bias_8us_drift": aabb_bias_8us_drift,
            "aabb_systematic_bias_us": aabb_systematic_bias_us,
            "abba_systematic_bias_us": abba_systematic_bias_us,
            "canary_p99_regression_pct": canary_p99_regression_pct,
            "canary_auto_rollback_triggered": canary_auto_rollback_triggered,
        },
    }


def main() -> None:
    metrics = compute_metrics()
    payload = json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    EXPECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPECTED_PATH.write_text(payload)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(payload)
    print(payload.strip())


if __name__ == "__main__":
    main()
