#!/usr/bin/env python3
"""Verifier for Experiment 18-03."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"
SUMMARY_PATH = HERE / "results" / "summary.json"


def main() -> None:
    if not EXPECTED_PATH.exists() or not SUMMARY_PATH.exists():
        print("Missing expected/metrics.json or results/summary.json; run run.py first.", file=sys.stderr)
        sys.exit(1)
    expected = json.loads(EXPECTED_PATH.read_text())
    actual = json.loads(SUMMARY_PATH.read_text())
    assert expected == actual, f"Mismatch:\nexpected={expected}\nactual={actual}"

    cow = actual["cow_trajectory_tree_forking"]
    assert cow["num_leaves"] == 16
    assert cow["linear_sandbox_cpu_sec"] == 672.0
    assert cow["forked_sandbox_cpu_sec"] == 114.0
    assert cow["sandbox_cpu_speedup"] == 5.89
    assert cow["linear_total_cpu_sec"] == 1056.0
    assert cow["forked_total_cpu_sec"] == 498.0
    assert cow["total_cpu_saving_pct"] == 52.84
    assert cow["linear_prefix_kv_slots"] == 131072
    assert cow["forked_prefix_kv_slots"] == 20480
    assert cow["prefix_kv_saving_pct"] == 84.38

    vt = actual["vtrace_and_tis_off_policy"]
    assert vt["step_is_ratio"] == 1.25
    assert vt["horizon_steps"] == 10
    assert vt["unclipped_prod_is"] == 9.3132
    assert vt["unclipped_var_amp"] == 86.74
    assert vt["vtrace_rho_bar"] == 1.2
    assert vt["vtrace_c_bar"] == 1.0
    assert vt["vtrace_prod_c"] == 1.0
    assert vt["vtrace_first_step_target"] == 1.2
    assert vt["vtrace_last_step_target"] == 1.2
    assert vt["variance_amplification_reduction"] == 7.76
    print("Experiment 18-03 verified: all locked metrics match.")


if __name__ == "__main__":
    main()
