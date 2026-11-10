#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 16-03."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    assert actual == expected, f"Mismatch:\nactual={actual}\nexpected={expected}"
    assert actual["traffic_shaping"]["raw_total_tokens"] == 506
    assert actual["traffic_shaping"]["padded_total_tokens"] == 704
    assert actual["traffic_shaping"]["bw_tax_pct"] == 39.13
    assert actual["traffic_shaping"]["latency_tax_pct"] == 12.11
    assert actual["freivalds_and_zkml"]["gemm_gflops"] == 481.036
    assert actual["freivalds_and_zkml"]["freivalds_overhead_pct"] == 2.2321
    assert actual["freivalds_and_zkml"]["sumcheck_speedup_vs_naive"] == 64.86
    assert actual["hybrid_zk_fault_proof"]["bisect_rounds"] == 9
    assert actual["hybrid_zk_fault_proof"]["zk_fault_proof_s"] == 1.85
    assert actual["hybrid_zk_fault_proof"]["latency_speedup"] == 24.32
    print("Experiment 16-03 verified successfully.")


if __name__ == "__main__":
    main()
