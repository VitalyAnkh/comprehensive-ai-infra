#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 19-03."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH, RESULTS_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    results = json.loads(Path(RESULTS_PATH).read_text())
    assert actual == expected == results, f"Mismatch:\nactual={actual}\nexpected={expected}\nresults={results}"

    sim = actual["closed_loop_3dgs_sensor_sim"]
    assert sim["active_gaussians"] == 700_000
    assert sim["naive_total_ms"] == 75.00
    assert sim["naive_fps"] == 13.33
    assert sim["opt_total_ms"] == 16.80
    assert sim["opt_fps"] == 59.52
    assert sim["render_speedup_x"] == 4.46
    assert sim["meets_30hz_hil_sla"] is True
    assert sim["ex19_13_8cam_2lidar_naive_ms"] == 104.00
    assert sim["ex19_13_8cam_2lidar_naive_fps"] == 9.62
    assert sim["ex19_13_8cam_2lidar_opt_ms"] == 22.00
    assert sim["ex19_13_8cam_2lidar_opt_fps"] == 45.45
    assert sim["ex19_13_speedup_x"] == 4.73

    wd = actual["dual_thor_safety_watchdog"]
    assert wd["takeover_latency_ms"] == 12.50
    assert wd["meets_asil_d_ftti"] is True
    assert wd["watchdog_lateral_drift_mm"] == 3.02
    assert wd["restart_250ms_lateral_drift_mm"] == 351.25
    assert wd["drift_reduction_x"] == 116.31
    assert wd["ex19_14_watchdog_drift_mm"] == 2.50
    assert wd["ex19_14_restart_200ms_drift_mm"] == 190.00
    assert wd["ex19_14_reduction_x"] == 76.00
    print("Experiment 19-03 verified successfully.")


if __name__ == "__main__":
    main()
