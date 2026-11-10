#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 21-03."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH, RESULTS_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    results = json.loads(Path(RESULTS_PATH).read_text())
    assert actual == expected == results, f"Mismatch:\nactual={actual}\nexpected={expected}\nresults={results}"
    assert actual["gui_incremental_roi_cache_ex21_6"]["opt_total_tokens"] == 24600
    assert actual["gui_incremental_roi_cache_ex21_6"]["token_compression_x"] == 7.32
    assert actual["gui_incremental_roi_cache_ex21_6"]["token_saving_pct"] == 86.33
    assert actual["gui_incremental_roi_cache_ex21_6"]["opt_vit_total_ms"] == 251.62
    assert actual["gui_incremental_roi_cache_ex21_6"]["vit_speedup_x"] == 5.37
    assert actual["speculative_dag_execution_ex21_7"]["expected_total_ms"] == 4969.60
    assert actual["speculative_dag_execution_ex21_7"]["dag_speedup_x"] == 1.61
    assert actual["speculative_dag_execution_ex21_7"]["latency_reduction_pct"] == 37.88
    assert actual["speculative_dag_execution_ex21_7"]["breakeven_hit_prob_pct"] == 1.25
    assert actual["exercises_21_11_and_21_12"]["p11_opt_tokens"] == 35680
    assert actual["exercises_21_11_and_21_12"]["p12_expected_total_ms"] == 6643.75
    print("Experiment 21-03 verified successfully.")


if __name__ == "__main__":
    main()
