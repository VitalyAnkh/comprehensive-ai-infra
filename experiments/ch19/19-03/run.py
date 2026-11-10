#!/usr/bin/env python3
"""Experiment 19-03: Closed-Loop 3DGS Neural Sensor Re-Simulation & Dual Drive Thor ASIL-D Safety Watchdog."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_PATH = ROOT / "expected" / "metrics.json"
RESULTS_PATH = ROOT / "results" / "summary.json"


def compute_metrics() -> dict:
    # 1. Worked Example 19-7(1): 6-camera 1080p + 128-beam LiDAR 3DGS closed-loop sensor simulation
    total_gaussians = 2_500_000
    cull_ratio = 0.72
    active_gaussians = int(round(total_gaussians * (1.0 - cull_ratio)))
    naive_6cam_ms = 6 * 8.50
    naive_1lidar_ms = 24.00
    naive_total_ms = round(naive_6cam_ms + naive_1lidar_ms, 2)
    naive_fps = round(1000.0 / naive_total_ms, 2)

    batched_6cam_ms = 11.40
    rtcore_1lidar_ms = 5.40
    opt_total_ms = round(batched_6cam_ms + rtcore_1lidar_ms, 2)
    opt_fps = round(1000.0 / opt_total_ms, 2)
    render_speedup_x = round(naive_total_ms / opt_total_ms, 2)

    # Exercise 19-13: 8-camera 1080p + dual 128-beam LiDAR
    ex13_naive_ms = round(8 * 8.50 + 2 * 18.00, 2)
    ex13_naive_fps = round(1000.0 / ex13_naive_ms, 2)
    ex13_opt_ms = round(14.20 + 7.80, 2)
    ex13_opt_fps = round(1000.0 / ex13_opt_ms, 2)
    ex13_speedup_x = round(ex13_naive_ms / ex13_opt_ms, 2)

    # 2. Worked Example 19-7(2): Dual Drive Thor ASIL-D Simplex Watchdog takeover at 120 km/h
    t_detect_ms = 5.00
    t_switch_ms = 1.50
    t_mpc_ms = 4.00
    t_act_ms = 2.00
    takeover_latency_ms = round(t_detect_ms + t_switch_ms + t_mpc_ms + t_act_ms, 2)
    asil_d_ftti_ms = 15.0

    dt_wd = takeover_latency_ms / 1000.0
    dt_sw = 250.0 / 1000.0
    v_y0 = 0.18
    a_y_fault = 9.80
    drift_wd_mm = round((v_y0 * dt_wd + 0.5 * a_y_fault * (dt_wd ** 2)) * 1000.0, 2)
    drift_sw_mm = round((v_y0 * dt_sw + 0.5 * a_y_fault * (dt_sw ** 2)) * 1000.0, 2)
    drift_reduction_x = round(drift_sw_mm / drift_wd_mm, 2)

    # Exercise 19-14: 90 km/h (25.0 m/s), v_y0 = 0.15 m/s, a_y_fault = 8.0 m/s^2, restart = 200.0 ms
    dt_sw_ex14 = 200.0 / 1000.0
    ex14_drift_wd_mm = round((0.15 * dt_wd + 0.5 * 8.0 * (dt_wd ** 2)) * 1000.0, 2)
    ex14_drift_sw_mm = round((0.15 * dt_sw_ex14 + 0.5 * 8.0 * (dt_sw_ex14 ** 2)) * 1000.0, 2)
    ex14_reduction_x = round(ex14_drift_sw_mm / ex14_drift_wd_mm, 2)

    return {
        "closed_loop_3dgs_sensor_sim": {
            "total_gaussians": total_gaussians,
            "cull_ratio": cull_ratio,
            "active_gaussians": active_gaussians,
            "naive_6cam_ms": naive_6cam_ms,
            "naive_1lidar_ms": naive_1lidar_ms,
            "naive_total_ms": naive_total_ms,
            "naive_fps": naive_fps,
            "batched_6cam_ms": batched_6cam_ms,
            "rtcore_1lidar_ms": rtcore_1lidar_ms,
            "opt_total_ms": opt_total_ms,
            "opt_fps": opt_fps,
            "render_speedup_x": render_speedup_x,
            "meets_30hz_hil_sla": opt_total_ms <= 33.33,
            "ex19_13_8cam_2lidar_naive_ms": ex13_naive_ms,
            "ex19_13_8cam_2lidar_naive_fps": ex13_naive_fps,
            "ex19_13_8cam_2lidar_opt_ms": ex13_opt_ms,
            "ex19_13_8cam_2lidar_opt_fps": ex13_opt_fps,
            "ex19_13_speedup_x": ex13_speedup_x,
        },
        "dual_thor_safety_watchdog": {
            "t_detect_ms": t_detect_ms,
            "t_switch_ms": t_switch_ms,
            "t_mpc_ms": t_mpc_ms,
            "t_act_ms": t_act_ms,
            "takeover_latency_ms": takeover_latency_ms,
            "asil_d_ftti_ms": asil_d_ftti_ms,
            "meets_asil_d_ftti": takeover_latency_ms <= asil_d_ftti_ms,
            "watchdog_lateral_drift_mm": drift_wd_mm,
            "restart_250ms_lateral_drift_mm": drift_sw_mm,
            "drift_reduction_x": drift_reduction_x,
            "ex19_14_watchdog_drift_mm": ex14_drift_wd_mm,
            "ex19_14_restart_200ms_drift_mm": ex14_drift_sw_mm,
            "ex19_14_reduction_x": ex14_reduction_x,
        },
    }


def main() -> None:
    metrics = compute_metrics()
    EXPECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    for p in (EXPECTED_PATH, RESULTS_PATH):
        p.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
