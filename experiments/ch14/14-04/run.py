#!/usr/bin/env python3
"""Experiment 14-04: 800V HVDC Busbar Loss, VPD & ASHRAE W45 Liquid Cooling Ledger."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"
RESULTS_PATH = HERE / "results" / "summary.json"


def compute_hvdc_busbar_and_vpd() -> dict:
    p_132_w = 132_000.0
    p_165_w = 165_000.0
    r_05_ohm = 0.0005

    i_48v = p_132_w / 48.0
    i_800v = p_132_w / 800.0
    loss_48v = (i_48v ** 2) * r_05_ohm
    loss_800v = (i_800v ** 2) * r_05_ohm
    loss_800v_165kw = ((p_165_w / 800.0) ** 2) * r_05_ohm

    p_140_w = 140_000.0
    r_08_ohm = 0.0008
    i_140_48v = p_140_w / 48.0
    i_140_800v = p_140_w / 800.0
    loss_140_48v = (i_140_48v ** 2) * r_08_ohm
    loss_140_800v = (i_140_800v ** 2) * r_08_ohm

    i_chip_core = 2400.0
    p_lat_chip = (i_chip_core ** 2) * 35e-6
    p_vpd_chip = (i_chip_core ** 2) * 3.5e-6
    vpd_rack_saving_kw = 72 * (p_lat_chip - p_vpd_chip) / 1000.0

    return {
        "current_48v_amp": round(i_48v, 2),
        "current_800v_amp": round(i_800v, 2),
        "busbar_loss_48v_watt": round(loss_48v, 2),
        "busbar_loss_800v_watt": round(loss_800v, 4),
        "hvdc_i2r_loss_reduction_x": round(loss_48v / loss_800v, 2),
        "busbar_loss_800v_165kw_watt": round(loss_800v_165kw, 2),
        "rack140_loss_48v_watt": round(loss_140_48v, 2),
        "rack140_loss_800v_watt": round(loss_140_800v, 2),
        "vpd_72gpu_saving_kw": round(vpd_rack_saving_kw, 2),
    }


def compute_ashrae_w45_and_microfluidics() -> dict:
    cp_kj_kgk = 4.18
    delta_t_k = 12.0
    rho_50c = 0.990

    m_dot_132 = 132.0 / (cp_kj_kgk * delta_t_k)
    flow_132_rho1 = round(m_dot_132 * 60.0, 1)
    flow_132_w45 = round((m_dot_132 / rho_50c) * 60.0, 1)
    flow_165_w45 = round(flow_132_w45 * (165.0 / 132.0), 1)

    case_temp_1400w = 49.0 + 1400.0 * 0.018
    t_in_max_cp_2kw = 85.0 - 2000.0 * (0.020 + 0.012)
    t_in_max_notim_2kw = 85.0 - 2000.0 * 0.012
    t_in_max_jet_2kw = 85.0 - 2000.0 * 0.008

    return {
        "w45_flow_l_per_min_132kw": flow_132_w45,
        "w45_flow_l_per_min_132kw_rho1": flow_132_rho1,
        "w45_flow_l_per_min_165kw": flow_165_w45,
        "coldplate_1400w_case_temp_c": round(case_temp_1400w, 1),
        "microfluidics_2000w_max_inlet_coldplate_c": round(t_in_max_cp_2kw, 1),
        "microfluidics_2000w_max_inlet_notim_c": round(t_in_max_notim_2kw, 1),
        "microfluidics_2000w_max_inlet_jet_c": round(t_in_max_jet_2kw, 1),
    }


def compute_metrics() -> dict:
    return {
        "hvdc_and_vpd": compute_hvdc_busbar_and_vpd(),
        "ashrae_w45_cooling": compute_ashrae_w45_and_microfluidics(),
    }


def main() -> None:
    metrics = compute_metrics()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
