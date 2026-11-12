#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 15-04."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH, RESULTS_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text(encoding="utf-8"))
    results = json.loads(Path(RESULTS_PATH).read_text(encoding="utf-8"))
    assert actual == expected == results, f"Mismatch:\nactual={actual}\nexpected={expected}\nresults={results}"

    tel = actual["three_tier_telemetry_ledger"]
    assert tel["cupti_burned_gpus_exact"] == 1310.72
    assert tel["cupti_burned_gpus_rounded"] == 1311
    assert tel["cupti_daily_trace_pb"] == 1.42
    assert tel["ebpf_dynolog_total_overhead_pct"] == 0.42
    assert tel["ebpf_dynolog_burned_gpus"] == 68.81
    assert tel["net_saved_gpus"] == 1241.91
    assert tel["ebpf_dynolog_daily_trace_tb"] == 1.80
    assert tel["trace_storage_reduction_x"] == 788.9

    cal = actual["blink_pmu_skid_calibration"]
    assert cal["true_counts"] == [500, 400, 100]
    assert cal["debiased_share_pct"] == {
        "tokenize_dispatch": 50.0,
        "launch_stub": 40.0,
        "post_hook": 10.0,
    }
    print("Experiment 15-04 verified successfully.")


if __name__ == "__main__":
    main()
