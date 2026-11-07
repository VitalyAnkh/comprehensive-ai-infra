#!/usr/bin/env python3
"""Experiment 19-01: Embodied AI VLA hard real-time drift, VLALagBench degradation, System 2/1 WCET isolation, and 4,096-env GPU simulation ledger."""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"
RESULTS_PATH = HERE / "results" / "summary.json"


def compute_metrics() -> dict:
    # 1. Blind-flight displacement dx = v * dt and VLALagBench dynamic vs static success rate
    # (a) High-speed conveyor grasp (v = 1.5 m/s, 120 ms vs 5 ms)
    velocity_m_s = 1.5
    dt_cloud_ms = 120.0
    dt_sys1_ms = 5.0
    blind_drift_cloud_mm = round(velocity_m_s * dt_cloud_ms, 2)  # 180.0 mm
    blind_drift_sys1_mm = round(velocity_m_s * dt_sys1_ms, 2)  # 7.5 mm
    drift_reduction_x = round(blind_drift_cloud_mm / blind_drift_sys1_mm, 2)  # 24.0x
    # (b) Precision peg-in-hole (v = 0.5 m/s, 200 ms -> 100.0 mm, Ex 19-1)
    blind_drift_05ms_200ms_mm = round(0.5 * 200.0, 2)  # 100.0 mm

    # Temporal Ensembling weights w_k = exp(-m * k)
    # (a) H=16, m=0.1 -> newest w_0 / sum = 0.1192
    horizon_h = 16
    decay_m = 0.1
    raw_weights = [math.exp(-decay_m * k) for k in range(horizon_h)]
    weight_sum = sum(raw_weights)
    newest_pred_weight_share = round(raw_weights[0] / weight_sum, 4)  # 0.1192
    # (b) K=6, m=0.2 (Ex 19-2) -> oldest w_5 / sum = 0.0954, newest w_0 / sum = 0.2594
    weights_k6 = [math.exp(-0.2 * k) for k in range(6)]
    sum_k6 = sum(weights_k6)
    oldest_pred_weight_share_k6 = round(weights_k6[5] / sum_k6, 4)  # 0.0954
    newest_pred_weight_share_k6 = round(weights_k6[0] / sum_k6, 4)  # 0.2594

    # VLALagBench static vs dynamic grasp success rate (%) at 20 ms vs 120 ms (>100 ms threshold)
    static_success_20ms_pct = 88.0
    static_success_120ms_pct = 82.0
    dynamic_success_20ms_pct = 84.0
    dynamic_success_120ms_pct = 32.0
    dynamic_drop_pct_points = round(dynamic_success_20ms_pct - dynamic_success_120ms_pct, 2)  # 52.0

    # 2. System 2 (2 Hz VLM, 500 ms) + System 1 (200 Hz / 100 Hz Flow-Matching ODE) WCET on Jetson Thor
    sys2_freq_hz = 2.0
    sys1_freq_hz = 200.0
    freq_ratio = int(sys1_freq_hz / sys2_freq_hz)  # 100 System-1 ticks per System-2 update
    sys1_wcet_shared_unisolated_ms = 18.5  # Violates 10 ms / 5 ms deadline due to System 2 Prefill contention
    sys1_wcet_mig_isolated_ms = 4.1  # Meets 5.0 ms hard real-time deadline
    ddpm_50step_latency_ms = 41.0
    flow_matching_5step_latency_ms = 4.1
    flow_matching_speedup_x = round(ddpm_50step_latency_ms / flow_matching_5step_latency_ms, 2)  # 10.0x

    # 3. 4,096-Environment GPU-Native Physics + Ray-Tracing + Policy Simulation Ledger
    num_envs = 4096
    sim_step_hz = 60
    # Tier A (128x128 RGB x2 + 64-dim FP32 state = 98,560 B/env -> 24.22 GB/s, 8.25 GiB HBM)
    obs_bytes_per_env = 2 * (128 * 128 * 3) + 64 * 4  # 98,560 B
    cpu_gpu_pcie_bw_gb_s = round(num_envs * sim_step_hz * obs_bytes_per_env / 1e9, 2)  # 24.22 GB/s saved
    physics_state_mib = round(num_envs * 64 / 1024, 2)  # 256.0 MiB
    bvh_geometry_mib = 512.0  # 512.0 MiB shared mesh & BVH
    tiled_render_fb_mib = round(num_envs * 384 / 1024, 2)  # 1,536.0 MiB
    policy_rollout_mib = 6144.0  # 6,144.0 MiB
    total_sim_hbm_mib = round(
        physics_state_mib + bvh_geometry_mib + tiled_render_fb_mib + policy_rollout_mib, 2
    )  # 8,448.0 MiB (8.25 GiB)
    total_sim_hbm_gib = round(total_sim_hbm_mib / 1024.0, 2)  # 8.25 GiB

    # Tier B (224x224 stereo packed 176,032 B + 128-dim FP32 state 512 B = 176,544 B/env -> 43.39 GB/s, 25.52 GiB HBM, Ex 19-4)
    obs_bytes_tier_b = 176032 + 128 * 4  # 176,544 B
    cpu_gpu_pcie_bw_tier_b_gb_s = round(num_envs * sim_step_hz * obs_bytes_tier_b / 1e9, 2)  # 43.39 GB/s
    total_sim_hbm_tier_b_mib = 2048.0 + 4096.0 + 3600.0 + 16384.0  # 26,128.0 MiB
    total_sim_hbm_tier_b_gib = round(total_sim_hbm_tier_b_mib / 1024.0, 2)  # 25.52 GiB

    # 4. Shadow Mode Edge Triage Bandwidth Reduction (Ex 19-5)
    raw_mb_per_s = 180.0
    hours_per_day = 4.0
    raw_tb_per_day = round(raw_mb_per_s * hours_per_day * 3600.0 / 1e6, 3)  # 2.592 TB/day
    triage_ratio = 0.005
    triage_gb_per_day = round(raw_tb_per_day * 1000.0 * triage_ratio, 2)  # 12.96 GB/day
    compression_ratio_x = round(1.0 / triage_ratio, 1)  # 200.0x

    return {
        "experiment_id": "19-01",
        "realtime_control_and_vlalagbench": {
            "blind_drift_cloud_120ms_mm": blind_drift_cloud_mm,
            "blind_drift_sys1_5ms_mm": blind_drift_sys1_mm,
            "drift_reduction_x": drift_reduction_x,
            "blind_drift_05ms_200ms_mm": blind_drift_05ms_200ms_mm,
            "newest_pred_weight_share": newest_pred_weight_share,
            "oldest_pred_weight_share_k6": oldest_pred_weight_share_k6,
            "newest_pred_weight_share_k6": newest_pred_weight_share_k6,
            "static_success_20ms_pct": static_success_20ms_pct,
            "static_success_120ms_pct": static_success_120ms_pct,
            "dynamic_success_20ms_pct": dynamic_success_20ms_pct,
            "dynamic_success_120ms_pct": dynamic_success_120ms_pct,
            "dynamic_drop_pct_points": dynamic_drop_pct_points,
        },
        "dual_system_thor_isolation": {
            "sys2_freq_hz": sys2_freq_hz,
            "sys1_freq_hz": sys1_freq_hz,
            "freq_ratio": freq_ratio,
            "sys1_wcet_shared_unisolated_ms": sys1_wcet_shared_unisolated_ms,
            "sys1_wcet_mig_isolated_ms": sys1_wcet_mig_isolated_ms,
            "flow_matching_speedup_x": flow_matching_speedup_x,
        },
        "gpu_native_sim_4096": {
            "num_envs": num_envs,
            "obs_bytes_per_env": obs_bytes_per_env,
            "cpu_gpu_pcie_bw_saved_gb_s": cpu_gpu_pcie_bw_gb_s,
            "total_sim_hbm_mib": total_sim_hbm_mib,
            "total_sim_hbm_gib": total_sim_hbm_gib,
            "obs_bytes_tier_b": obs_bytes_tier_b,
            "cpu_gpu_pcie_bw_saved_tier_b_gb_s": cpu_gpu_pcie_bw_tier_b_gb_s,
            "total_sim_hbm_tier_b_mib": total_sim_hbm_tier_b_mib,
            "total_sim_hbm_tier_b_gib": total_sim_hbm_tier_b_gib,
        },
        "shadow_mode_edge_triage": {
            "raw_tb_per_day": raw_tb_per_day,
            "triage_gb_per_day": triage_gb_per_day,
            "compression_ratio_x": compression_ratio_x,
        },
    }


def main() -> None:
    metrics = compute_metrics()
    payload = json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    EXPECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPECTED_PATH.write_text(payload)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(payload)
    print(payload.strip())


if __name__ == "__main__":
    main()
