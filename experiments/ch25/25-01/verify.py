#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 25-01."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH, RESULTS_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    results = json.loads(Path(RESULTS_PATH).read_text())
    assert actual == expected == results, f"Mismatch:\nactual={actual}\nexpected={expected}\nresults={results}"
    assert actual["three_tier_embedding_cache"]["total_table_tb"] == 10.0
    assert actual["three_tier_embedding_cache"]["weighted_latency_us"] == 2.19
    assert actual["three_tier_embedding_cache"]["pure_ssd_speedup_x"] == 29.7
    assert actual["torchrec_all_to_all_overlap"]["balanced_a2a_ms"] == 1.53
    assert actual["torchrec_all_to_all_overlap"]["exposed_balanced_comm_ms"] == 0.0
    print("Experiment 25-01 verified successfully.")


if __name__ == "__main__":
    main()
