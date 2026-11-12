#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 15-01."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH, RESULTS_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    results = json.loads(Path(RESULTS_PATH).read_text())
    assert actual == expected == results, f"Mismatch:\nactual={actual}\nexpected={expected}\nresults={results}"
    assert actual["cluster_reliability_100k"]["llama3_16k_interruptions_54d"] == 419
    assert actual["cluster_reliability_100k"]["mtbf_100k_minutes"] == 30.41
    assert actual["cluster_reliability_100k"]["ettr_traditional_pct"] == 63.22
    assert actual["cluster_reliability_100k"]["ettr_fast_pct"] == 92.34
    assert actual["cluster_reliability_100k"]["ettr_traditional_linear_pct"] == 49.83
    assert actual["cluster_reliability_100k"]["ettr_fast_linear_pct"] == 92.55
    assert actual["cluster_reliability_100k"]["ettr_traditional_exact_pct"] == 58.78
    assert actual["cluster_reliability_100k"]["ettr_fast_exact_pct"] == 92.70
    assert actual["cluster_reliability_100k"]["min_spares_999"] == 40
    assert actual["abft_sdc_defense"]["abft_overhead_4096_pct"] < 0.5
    assert actual["abft_sdc_defense"]["detected_sdc_coord"] == [2, 3]
    assert actual["straggler_and_benchmark_science"]["stripe_mean_ms"] == 1.25
    assert actual["straggler_and_benchmark_science"]["stripe_max_ms"] == 2.0
    assert actual["straggler_and_benchmark_science"]["deadlock_cycle_detected"] is True
    assert actual["straggler_and_benchmark_science"]["mean_speedup_cand_a"] == 5.25
    assert actual["straggler_and_benchmark_science"]["equal_freq_time_cand_a_us"] == 210.0
    assert actual["straggler_and_benchmark_science"]["aabb_means_us"] == [105.0, 115.0]
    assert actual["straggler_and_benchmark_science"]["abba_means_us"] == [115.0, 105.0]
    assert actual["straggler_and_benchmark_science"]["plan_a_at_1_3_ms"] == 109.0
    assert actual["straggler_and_benchmark_science"]["plan_b_at_1_3_ms"] == 105.6
    assert actual["straggler_and_benchmark_science"]["telemetry_drain_sec"] == 76.92
    assert actual["straggler_and_benchmark_science"]["canary_rollback_on_190ms"] is True
    print("Experiment 15-01 verified successfully.")


if __name__ == "__main__":
    main()
