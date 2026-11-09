#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 13-04."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH, SUMMARY_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    summary = json.loads(Path(SUMMARY_PATH).read_text())
    assert actual == expected == summary, f"Mismatch:\nactual={actual}\nexpected={expected}"
    assert actual["lmcache_pipeline"]["compressed_pipelined_ms"] == 273.02
    assert actual["lmcache_pipeline"]["ttft_speedup_x"] == 2.61
    assert actual["three_tier_lifecycle"]["avg_fetch_ms"] == 230.11
    assert actual["three_tier_lifecycle"]["cost_saving_pct"] == 87.98
    assert actual["nvme_wrr_qos"]["fifo_p99_read_ms"] == 13.80
    assert actual["nvme_wrr_qos"]["wrr_p99_read_ms"] == 1.35
    assert actual["nvme_wrr_qos"]["tail_latency_reduction_x"] == 10.22
    print("Experiment 13-04 verified successfully.")


if __name__ == "__main__":
    main()
