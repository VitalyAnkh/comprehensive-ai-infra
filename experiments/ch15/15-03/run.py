#!/usr/bin/env python3
"""Experiment 15-03: In-Silicon Microfluidic Cooling & 1.2 GW Microgrid Frequency Control."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"
RESULTS_PATH = HERE / "results" / "summary.json"


def solve_in_silicon_microfluidic_thermal_hydraulics(
    p_chip_w: float = 2500.0,
    die_area_cm2: float = 9.0,
    t_inlet_c: float = 45.0,
    t_j_limit_c: float = 85.0,
    flow_rate_lpm: float = 1.80,
    r_coldplate_total_kw: float = 0.028,
    r_cond_si_kw: float = 0.0018,
    r_conv_micro_kw: float = 0.0042,
    ch_width_um: float = 50.0,
    ch_depth_um: float = 300.0,
    delta_p_kpa: float = 38.4,
    pump_eff: float = 0.70,
) -> dict:
    heat_flux_w_cm2 = p_chip_w / die_area_cm2
    d_h_um = 2.0 * ch_width_um * ch_depth_um / (ch_width_um + ch_depth_um)
    m_dot_kg_s = (flow_rate_lpm / 60.0) * 0.990
    c_p_j_kgk = 4180.0
    r_cal_kw = round(1.0 / (2.0 * m_dot_kg_s * c_p_j_kgk), 4)
    r_micro_total_kw = round(r_cond_si_kw + r_conv_micro_kw + r_cal_kw, 4)

    t_j_coldplate_c = t_inlet_c + p_chip_w * r_coldplate_total_kw
    t_j_micro_c = t_inlet_c + p_chip_w * r_micro_total_kw
    t_in_max_coldplate_c = t_j_limit_c - p_chip_w * r_coldplate_total_kw
    t_in_max_micro_c = t_j_limit_c - p_chip_w * r_micro_total_kw

    q_m3_s = (flow_rate_lpm / 1000.0) / 60.0
    pump_power_w = (delta_p_kpa * 1000.0 * q_m3_s) / pump_eff
    pump_ratio_pct = (pump_power_w / p_chip_w) * 100.0
    return {
        "heat_flux_w_cm2": round(heat_flux_w_cm2, 2),
        "d_h_um": round(d_h_um, 2),
        "r_cal_kw": r_cal_kw,
        "r_micro_total_kw": r_micro_total_kw,
        "r_reduction_ratio": round(r_coldplate_total_kw / r_micro_total_kw, 2),
        "t_j_coldplate_c": round(t_j_coldplate_c, 1),
        "t_j_micro_c": round(t_j_micro_c, 1),
        "t_in_max_coldplate_c": round(t_in_max_coldplate_c, 1),
        "t_in_max_micro_c": round(t_in_max_micro_c, 1),
        "pump_power_w": round(pump_power_w, 2),
        "pump_ratio_pct": round(pump_ratio_pct, 4),
    }


def simulate_gw_campus_microgrid_swing_and_vsg_control(
    p_base_mw: float = 1200.0,
    f0_hz: float = 60.0,
    p_smr_mw: float = 300.0,
    h_smr_s: float = 5.0,
    p_gt_mw: float = 600.0,
    h_gt_s: float = 3.5,
    p_bess_mw: float = 400.0,
    h_bess_syn_s: float = 13.5,
    delta_p_step_mw: float = 450.0,
    rocof_limit_hz_s: float = 1.50,
    d_load_pu: float = 1.2,
    r_gt_droop: float = 0.04,
    r_bess_droop: float = 0.015,
) -> dict:
    h_phys_s = (p_smr_mw * h_smr_s + p_gt_mw * h_gt_s) / p_base_mw
    h_with_vsg_s = h_phys_s + (p_bess_mw * h_bess_syn_s) / p_base_mw

    rocof_no_vsg_hz_s = (delta_p_step_mw / (2.0 * h_phys_s * p_base_mw)) * f0_hz
    rocof_with_vsg_hz_s = (delta_p_step_mw / (2.0 * h_with_vsg_s * p_base_mw)) * f0_hz

    h_sys_req_s = (delta_p_step_mw * f0_hz) / (2.0 * rocof_limit_hz_s * p_base_mw)
    h_bess_min_s = max(0.0, (h_sys_req_s - h_phys_s) * (p_base_mw / p_bess_mw))

    beta_no_vsg_mw_per_hz = (
        (p_gt_mw / (r_gt_droop * f0_hz)) + (d_load_pu * p_base_mw / f0_hz)
    )
    beta_with_vsg_mw_per_hz = beta_no_vsg_mw_per_hz + (
        p_bess_mw / (r_bess_droop * f0_hz)
    )
    df_peak_no_vsg_hz = (delta_p_step_mw / beta_no_vsg_mw_per_hz) * 1.75
    df_peak_with_vsg_hz = (delta_p_step_mw / beta_with_vsg_mw_per_hz) * 1.18
    return {
        "h_phys_s": round(h_phys_s, 2),
        "h_with_vsg_s": round(h_with_vsg_s, 2),
        "rocof_no_vsg_hz_s": round(rocof_no_vsg_hz_s, 3),
        "rocof_with_vsg_hz_s": round(rocof_with_vsg_hz_s, 3),
        "h_bess_min_s": round(h_bess_min_s, 2),
        "df_peak_no_vsg_hz": round(df_peak_no_vsg_hz, 3),
        "df_peak_with_vsg_hz": round(df_peak_with_vsg_hz, 3),
    }


def optimize_electro_thermal_joint_dvfs_and_load_shaping(
    n_subclusters: int = 8,
    raw_step_drop_mw: float = 450.0,
    stagger_efficiency: float = 0.68,
    freq_scale_during_peak: float = 0.92,
    compute_fraction: float = 0.82,
    t_inlet_c: float = 45.0,
    p_static_w: float = 300.0,
    p_dyn_nom_w: float = 2200.0,
    r_micro_kw: float = 0.010,
) -> dict:
    effective_step_mw = raw_step_drop_mw * (1.0 - stagger_efficiency)
    p_dyn_scaled_w = p_dyn_nom_w * (freq_scale_during_peak ** 2.4)
    p_chip_scaled_w = p_static_w + p_dyn_scaled_w
    t_j_controlled_c = t_inlet_c + p_chip_scaled_w * r_micro_kw
    active_duty = 0.20
    step_overhead_pct = (
        active_duty * compute_fraction * ((1.0 / freq_scale_during_peak) - 1.0) * 100.0
    )
    return {
        "n_subclusters": n_subclusters,
        "effective_step_mw": round(effective_step_mw, 1),
        "p_chip_scaled_w": round(p_chip_scaled_w, 1),
        "t_j_controlled_c": round(t_j_controlled_c, 2),
        "step_overhead_pct": round(step_overhead_pct, 2),
    }


def compute_metrics() -> dict:
    return {
        "experiment_id": "15-03",
        "in_silicon_microfluidics": solve_in_silicon_microfluidic_thermal_hydraulics(),
        "gw_microgrid_vsg": simulate_gw_campus_microgrid_swing_and_vsg_control(),
        "electro_thermal_dvfs": optimize_electro_thermal_joint_dvfs_and_load_shaping(),
    }


def main() -> None:
    metrics = compute_metrics()
    EXPECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not EXPECTED_PATH.exists():
        EXPECTED_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    RESULTS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
