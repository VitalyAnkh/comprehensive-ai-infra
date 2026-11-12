#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 15-05."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH, RESULTS_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text(encoding="utf-8"))
    results = json.loads(Path(RESULTS_PATH).read_text(encoding="utf-8"))
    assert actual == expected == results, f"Mismatch:\nactual={actual}\nexpected={expected}\nresults={results}"

    wc = actual["wait_chain_8rank"]
    assert wc["root_cause_rank"] == 5
    assert wc["victim_ranks_stripped"] == 7
    assert wc["root_enter_ms"] == 200.0
    assert wc["collective_exit_ms"] == 204.0
    assert wc["root_apparent_nccl_ms"] == 4.0
    assert wc["max_victim_apparent_nccl_ms"] == 144.0

    mon = actual["monarch_vs_spmd"]
    assert mon["monarch_isolation_seconds"] == 0.85
    assert mon["spmd_global_restart_seconds"] == 142.0
    assert mon["monarch_vs_spmd_speedup_x"] == 167.1

    cap = actual["forensics_capstones"]
    assert cap["ex15_17_kineto_multihop"]["root_cause_rank"] == 2
    assert cap["ex15_17_kineto_multihop"]["healthy_dp_victim_wait_ms"] == 85.0
    assert cap["ex15_18_flight_recorder"]["incident_a_diagnosis"] == "COLL_TYPE_MISMATCH_DEADLOCK"
    assert cap["ex15_18_flight_recorder"]["incident_b_ring_slot"] == 51
    assert cap["ex15_18_flight_recorder"]["ring_retention_window_sec"] == 6.40
    print("Experiment 15-05 verified successfully.")


if __name__ == "__main__":
    main()
