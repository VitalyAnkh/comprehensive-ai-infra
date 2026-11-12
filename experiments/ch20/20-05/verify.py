#!/usr/bin/env python3
"""Verifier for Experiment 20-05."""
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
    expected = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
    actual = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    assert expected == actual, f"Mismatch:\nexpected={expected}\nactual={actual}"

    soc = actual["soc_slc_qos_isolation_ex20_9"]
    assert soc["reserved_rt_ways"] == 2
    assert soc["non_rt_bw_cap_gbps"] == 205.0
    assert soc["reflex_1khz_deadline_ms"] == 1.00
    assert soc["reflex_wcet_unisolated_ms"] == 1.82
    assert soc["reflex_wcet_slc_qos_isolated_ms"] == 0.41
    assert soc["wcet_speedup_x"] == 4.44

    wm = actual["world_model_regression_ex20_10"]
    assert wm["total_closed_loop_rollouts"] == 100_000
    assert wm["world_model_64gpu_eval_hours"] == 0.40
    assert wm["world_model_64gpu_cost_usd"] == 768.00
    assert wm["physical_robot_eval_cost_usd"] == 111_360.00
    assert wm["world_model_cost_compression_x"] == 145.0
    assert wm["policy_comparison"]["policy_a_conservative_flow"]["release_gate_passed"] is True
    assert wm["policy_comparison"]["policy_b_overfit_mse"]["release_gate_passed"] is False

    ex = actual["exercises_20_17_and_20_18"]
    assert ex["ex17_min_rt_ways"] == 3
    assert ex["ex17_non_rt_cap_gbps"] == 140.80
    assert ex["ex17_isolated_wcet_ms"] == 0.48
    assert ex["ex18_base_hours"] == 0.50
    assert ex["ex18_base_cost_usd"] == 1920.00
    assert ex["ex18_phys_cost_usd"] == 278_400.00
    assert ex["ex18_early_stop_hours"] == 0.3925
    assert ex["ex18_early_stop_cost_usd"] == 1507.20
    print("Experiment 20-05 verified: all locked metrics match.")


if __name__ == "__main__":
    main()
