#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 15-03."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH, RESULTS_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    results = json.loads(Path(RESULTS_PATH).read_text())
    assert actual == expected == results, f"Mismatch:\nactual={actual}\nexpected={expected}\nresults={results}"

    oob = actual["oobleck_elastic_pipeline_failover"]
    assert oob["oobleck_reconfig_sec"] == 1.45
    assert oob["full_restart_sec"] == 300.00
    assert oob["reconfig_speedup_x"] == 206.90
    assert oob["oobleck_retained_throughput_pct"] == 98.75
    assert oob["parachute_retained_throughput_pct"] == 93.75
    assert oob["ex20_13_parachute_pct"] == 93.75
    assert oob["ex20_13_oobleck_pct"] == 98.88
    assert oob["ex20_13_saved_node_hours_per_day"] == 629.76

    gp = actual["bursty_goodput_vs_uniform"]
    assert gp["ttft_uniform_ms"] == 80.00
    assert gp["goodput_uniform_req_s"] == 7.08
    assert gp["wq_bursty_ms"] == 829.60
    assert gp["ttft_bursty_ms"] == 909.60
    assert gp["ttft_inflation_x"] == 11.37
    assert gp["slo_attainment_bursty_pct"] == 48.77
    assert gp["goodput_bursty_req_s"] == 3.45
    assert gp["goodput_drop_pct"] == 51.27
    assert gp["max_safe_utilization_pct"] == 60.04
    assert gp["ex20_14_wq_bursty_ms"] == 900.00
    assert gp["ex20_14_ttft_bursty_ms"] == 960.00
    assert gp["ex20_14_inflation_x"] == 16.00
    print("Experiment 15-03 verified successfully.")


if __name__ == "__main__":
    main()
