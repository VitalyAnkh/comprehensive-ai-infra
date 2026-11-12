#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 21-05."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH, RESULTS_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text(encoding="utf-8"))
    results = json.loads(Path(RESULTS_PATH).read_text(encoding="utf-8"))
    assert actual == expected == results, f"Mismatch:\nactual={actual}\nexpected={expected}\nresults={results}"
    ca = actual["cuckoo_admission_and_ttl"]
    ep = actual["epoch_lockfree_hot_update"]
    assert ca["cuckoo_tau5_embedding_shrink_pct"] == 68.5
    assert ca["cuckoo_embedding_active_gb"] == 756.0
    assert ep["hot_update_rwlock_p99_ms"] == 18.4
    assert ep["hot_update_epoch_lockfree_p99_ms"] == 1.15
    assert ep["hot_update_tail_reduction_x"] == 16.0
    assert ep["hot_update_epoch_memory_overhead_pct"] == 6.2
    print("Experiment 21-05 verified successfully.")


if __name__ == "__main__":
    main()
