#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 25-03."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH, RESULTS_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    results = json.loads(Path(RESULTS_PATH).read_text())
    assert actual == expected == results, f"Mismatch:\nactual={actual}\nexpected={expected}\nresults={results}"
    assert actual["cuvs_cagra_vs_hnsw"]["raw_vectors_tb"] == 1.536
    assert actual["cuvs_cagra_vs_hnsw"]["gpu_cagra_minutes"] == 40.8
    assert actual["cuvs_cagra_vs_hnsw"]["build_speedup_x"] == 16.91
    assert actual["diskann_sector_aligned_search"]["dram_pq_gb"] == 64.0
    assert actual["diskann_sector_aligned_search"]["node_bytes"] == 1792
    assert actual["diskann_sector_aligned_search"]["nodes_per_sector"] == 2
    assert actual["diskann_sector_aligned_search"]["random_reads_per_query"] == 24
    assert actual["diskann_sector_aligned_search"]["sustainable_qps"] == 210000
    print("Experiment 25-03 verified successfully.")


if __name__ == "__main__":
    main()
