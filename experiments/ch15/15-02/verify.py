#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 15-02."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH, RESULTS_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    results = json.loads(Path(RESULTS_PATH).read_text())
    assert actual == expected == results, f"Mismatch:\nactual={actual}\nexpected={expected}\nresults={results}"
    assert actual["chaos_gameday_audit"]["total_injected"] == 200
    assert actual["chaos_gameday_audit"]["total_detected"] == 199
    assert actual["chaos_gameday_audit"]["overall_recall_pct"] == 99.5
    assert actual["chaos_gameday_audit"]["mean_mttd_sec"] == 2.7
    assert actual["chaos_gameday_audit"]["mean_auto_mttr_sec"] == 3.85
    assert actual["chaos_gameday_audit"]["mttr_speedup_x"] == 389.61
    assert actual["arrival_skew_8rank"]["straggler_rank"] == 5
    assert actual["arrival_skew_8rank"]["arrival_skew_ms"] == 2.0
    assert actual["arrival_skew_8rank"]["healthy_observed_duration_ms"] == 2.4
    assert actual["arrival_skew_8rank"]["healthy_wait_share_pct"] == 83.33
    assert actual["trainmover_hot_swap"]["trainmover_total_recovery_sec"] == 0.68
    assert actual["trainmover_hot_swap"]["trainmover_speedup_x"] == 441.18
    assert actual["benchmark_debiasing"]["arith_speedup_a"] == 5.25
    assert actual["benchmark_debiasing"]["true_speedup_a_equal_freq"] == 0.9524
    assert actual["benchmark_debiasing"]["true_speedup_x_equal_freq"] == 0.8
    assert actual["benchmark_debiasing"]["crossover_p_x_beats_y"] == 0.8571
    assert actual["benchmark_debiasing"]["aabb_bias_8us_drift"] == -16.0
    assert actual["benchmark_debiasing"]["aabb_systematic_bias_us"] == -24.0
    assert actual["benchmark_debiasing"]["abba_systematic_bias_us"] == 0.0
    assert actual["benchmark_debiasing"]["canary_auto_rollback_triggered"] is True
    print("Experiment 15-02 verified successfully.")


if __name__ == "__main__":
    main()
