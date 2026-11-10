#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 22-03."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH, RESULTS_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    results = json.loads(Path(RESULTS_PATH).read_text())
    assert actual == expected == results, f"Mismatch:\nactual={actual}\nexpected={expected}\nresults={results}"
    assert actual["tokamak_sub_ms_rl_control"]["unopt_total_us"] == 435.0
    assert actual["tokamak_sub_ms_rl_control"]["opt_total_us"] == 68.5
    assert actual["tokamak_sub_ms_rl_control"]["speedup_x"] == 6.35
    assert actual["tokamak_sub_ms_rl_control"]["budget_util_pct"] == 34.25
    assert actual["tokamak_sub_ms_rl_control"]["p999_latency_us"] == 74.0
    assert actual["tokamak_sub_ms_rl_control"]["actuation_margin_ms"] == 16.5
    assert actual["single_cell_csr_sparse_atlas"]["dense_storage_gb"] == 4096.0
    assert actual["single_cell_csr_sparse_atlas"]["csr_storage_gb"] == 615.2
    assert actual["single_cell_csr_sparse_atlas"]["storage_compression_x"] == 6.66
    assert actual["single_cell_csr_sparse_atlas"]["attn_flop_reduction_x"] == 177.78
    assert actual["single_cell_csr_sparse_atlas"]["e2e_training_speedup_x"] == 20.0
    print("Experiment 22-03 verified successfully.")


if __name__ == "__main__":
    main()
