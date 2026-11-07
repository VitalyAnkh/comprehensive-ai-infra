#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 19-01."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH, RESULTS_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    results = json.loads(Path(RESULTS_PATH).read_text())
    assert actual == expected == results, f"Mismatch:\nactual={actual}\nexpected={expected}\nresults={results}"
    assert actual["realtime_control_and_vlalagbench"]["blind_drift_cloud_120ms_mm"] == 180.0
    assert actual["realtime_control_and_vlalagbench"]["blind_drift_sys1_5ms_mm"] == 7.5
    assert actual["realtime_control_and_vlalagbench"]["blind_drift_05ms_200ms_mm"] == 100.0
    assert actual["realtime_control_and_vlalagbench"]["newest_pred_weight_share"] == 0.1192
    assert actual["realtime_control_and_vlalagbench"]["oldest_pred_weight_share_k6"] == 0.0954
    assert actual["realtime_control_and_vlalagbench"]["newest_pred_weight_share_k6"] == 0.2594
    assert actual["realtime_control_and_vlalagbench"]["dynamic_success_120ms_pct"] == 32.0
    assert actual["dual_system_thor_isolation"]["sys1_wcet_mig_isolated_ms"] == 4.1
    assert actual["gpu_native_sim_4096"]["cpu_gpu_pcie_bw_saved_gb_s"] == 24.22
    assert actual["gpu_native_sim_4096"]["total_sim_hbm_mib"] == 8448.0
    assert actual["gpu_native_sim_4096"]["total_sim_hbm_gib"] == 8.25
    assert actual["gpu_native_sim_4096"]["cpu_gpu_pcie_bw_saved_tier_b_gb_s"] == 43.39
    assert actual["gpu_native_sim_4096"]["total_sim_hbm_tier_b_gib"] == 25.52
    assert actual["shadow_mode_edge_triage"]["raw_tb_per_day"] == 2.592
    assert actual["shadow_mode_edge_triage"]["triage_gb_per_day"] == 12.96
    assert actual["shadow_mode_edge_triage"]["compression_ratio_x"] == 200.0
    print("Experiment 19-01 verified successfully.")


if __name__ == "__main__":
    main()
