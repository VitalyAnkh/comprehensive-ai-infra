"""Ch14: 48V vs 800V HVDC busbar loss, ASHRAE W45 liquid cooling, microfluidics, BESS, and campus DiLoCo DCI."""
from __future__ import annotations

from .. import hardware
from ..sources import model_config


def calculate(
    rack_power_kw: float = 132.0,
    nextgen_rack_power_kw: float = 165.0,
    busbar_resistance_mohm: float = 0.50,
    coolant_inlet_c: float = 45.0,
    coolant_delta_t_c: float = 12.0,
    campus_grid_mw: float = 100.0,
    diloco_inner_steps: int = 500,
    dci_bandwidth_gbps: float = 400.0,
) -> dict:
    dsv3 = model_config("deepseek-v3")
    hw_map = {d["id"]: d for d in hardware.catalog()["devices"]}

    # 1. 算例 14-1: 132 kW liquid-cooled rack on 48 VDC vs 800 VDC busbar (R_bus = 0.5 mOhm)
    r_ohm = busbar_resistance_mohm * 1e-3
    i_48v = round(rack_power_kw * 1000.0 / 48.0, 2)               # 2,750.0 A
    i_800v = round(rack_power_kw * 1000.0 / 800.0, 2)             # 165.0 A
    loss_48v_w = round((i_48v**2) * r_ohm, 2)                     # 3,781.25 W (2.86% of rack)
    loss_800v_w = round((i_800v**2) * r_ohm, 4)                   # 13.6125 W
    i2r_reduction_ratio = round((800.0 / 48.0) ** 2, 2)           # 277.78x
    i_800v_165kw = round(nextgen_rack_power_kw * 1000.0 / 800.0, 3)  # 206.25 A
    loss_800v_165kw_w = round((i_800v_165kw**2) * r_ohm, 2)       # 21.27 W

    # 2. 算例 14-4: 132 kW rack under ASHRAE W45 (T_in = 45 C, dT = 12 K, c_p = 4.18 kJ/(kg*K), rho = 0.990 kg/L)
    cp_kj_per_kg_k = 4.18
    rho_kg_per_l = 0.990
    mass_flow_kg_s = rack_power_kw / (cp_kj_per_kg_k * coolant_delta_t_c)  # 2.63158 kg/s
    flow_l_per_min_132kw = round((mass_flow_kg_s / rho_kg_per_l) * 60.0, 1)  # 159.5 L/min (157.9 at rho=1.0)
    flow_l_per_min_rho1 = round(mass_flow_kg_s * 60.0, 1)                    # 157.9 L/min
    flow_l_per_min_165kw = round(flow_l_per_min_132kw * (nextgen_rack_power_kw / rack_power_kw), 1)  # 199.4 L/min

    # Cold-plate case temperature check for 1,400 W GPU chip (local T_fluid = 49.0 C, Psi_ca = 0.018 C/W -> 74.2 C < 85 C)
    chip_power_w = 1400.0
    psi_ca_c_per_w = 0.018
    local_fluid_c = 49.0
    t_case_1400w_c = round(local_fluid_c + chip_power_w * psi_ca_c_per_w, 1)  # 74.2 C

    # 3. 算例 14-8: 2,000 W next-gen chip Direct-to-Silicon Microfluidics eliminating R_TIM = 0.02 C/W (T_j,max = 85.0 C)
    t_j_max_c = 85.0
    max_inlet_coldplate_c = round(t_j_max_c - 2000.0 * (0.020 + 0.012), 1)      # 21.0 C
    max_inlet_microfluidics_base_c = round(t_j_max_c - 2000.0 * 0.012, 1)       # 61.0 C (+40.0 C)
    max_inlet_microfluidics_jet_c = round(t_j_max_c - 2000.0 * 0.008, 1)        # 69.0 C

    # 4. 100 MW campus electricity & BESS peak shaving ledger
    annual_mwh = campus_grid_mw * 8760.0
    annual_elec_saving_musd = round((annual_mwh * (120.0 - 40.0)) / 1e6, 2)     # $70.08M/yr ($0.12 vs $0.04/kWh)
    bess_2hr_20pct_mwh = round(campus_grid_mw * 0.20 * 2.0, 2)                  # 40.0 MWh

    # 5. Controlled multi-campus DCI DiLoCo outer synchronization (Ch14.4.3: H=500 inner steps on 400 Gbps dedicated fiber)
    total_params_b = 671.0
    active_params_b = 37.0
    pseudo_grad_bytes = total_params_b * 1e9 * 2.0                              # 1.342 TB BF16 outer delta
    dci_eff_Bps = dci_bandwidth_gbps * 1e9 / 8.0 * 0.85                         # 42.5 GB/s effective
    ring_allreduce_2campus_s = round(pseudo_grad_bytes / dci_eff_Bps, 2)        # 31.58 s
    inner_step_s = 4.20
    inner_loop_window_s = round(diloco_inner_steps * inner_step_s, 2)           # 2,100.0 s (35.0 min)
    blocking_comm_overhead_pct = round(
        ring_allreduce_2campus_s / (inner_loop_window_s + ring_allreduce_2campus_s) * 100.0, 2
    )  # 1.48% (0.0% exposed under streaming shard overlap)
    per_step_naive_dp_gbps = round((pseudo_grad_bytes * 8.0 / inner_step_s) / 1e9, 1)  # 2,556.2 Gbps -> impossible over DCI
    diloco_avg_dci_gbps = round(per_step_naive_dp_gbps / diloco_inner_steps, 2)        # 5.11 Gbps average (< 400 Gbps)

    return {
        "schema": "h2-topic-v1",
        "calculation": "datacenter-power-cooling",
        "chapter": 14,
        "title": "第 14 章：800V HVDC 母排铜损、ASHRAE W45 温水液冷、硅微流道直冷与跨园区 DCI 分层训练账本",
        "scenario": {
            "rack_power_kw": rack_power_kw,
            "nextgen_rack_power_kw": nextgen_rack_power_kw,
            "busbar_resistance_mohm": busbar_resistance_mohm,
            "coolant_inlet_c": coolant_inlet_c,
            "coolant_delta_t_c": coolant_delta_t_c,
            "campus_grid_mw": campus_grid_mw,
            "diloco_inner_steps": diloco_inner_steps,
            "dci_bandwidth_gbps": dci_bandwidth_gbps,
            "rubin_hbm_bw_tbps": hw_map["rubin-nvl72-product-profile"]["memory"]["bandwidth_bytes_per_second"] / 1e12,
            "deepseek_v3_layers": dsv3["num_hidden_layers"],
        },
        "summary": {
            "current_48v_amp": i_48v,
            "current_800v_amp": i_800v,
            "busbar_loss_48v_watt": loss_48v_w,
            "busbar_loss_800v_watt": loss_800v_w,
            "hvdc_i2r_loss_reduction_x": i2r_reduction_ratio,
            "busbar_loss_800v_165kw_watt": loss_800v_165kw_w,
            "w45_flow_l_per_min_132kw": flow_l_per_min_132kw,
            "w45_flow_l_per_min_132kw_rho1": flow_l_per_min_rho1,
            "w45_flow_l_per_min_165kw": flow_l_per_min_165kw,
            "coldplate_1400w_case_temp_c": t_case_1400w_c,
            "microfluidics_2000w_max_inlet_coldplate_c": max_inlet_coldplate_c,
            "microfluidics_2000w_max_inlet_notim_c": max_inlet_microfluidics_base_c,
            "microfluidics_2000w_max_inlet_jet_c": max_inlet_microfluidics_jet_c,
            "campus_100mw_elec_saving_musd_per_yr": annual_elec_saving_musd,
            "bess_20pct_2hr_capacity_mwh": bess_2hr_20pct_mwh,
            "dci_671b_naive_per_step_gbps": per_step_naive_dp_gbps,
            "dci_671b_diloco_h500_avg_gbps": diloco_avg_dci_gbps,
            "dci_671b_outer_sync_seconds": ring_allreduce_2campus_s,
            "dci_inner_500step_window_seconds": inner_loop_window_s,
            "dci_blocking_comm_overhead_pct": blocking_comm_overhead_pct,
            "active_params_b": active_params_b,
        },
        "assumptions": [
            "算例 14-1：132 kW 机柜在 R_bus = 0.5 mOhm 下，48 VDC 电流 2,750 A、铜损 3,781.25 W；升级 800 VDC 后电流降至 165 A、铜损降至 13.6125 W（缩减 277.78x）。",
            "算例 14-4 与 14-8：ASHRAE W45（45°C 进液、12 K 温升）下 132 kW 需水流量 159.5 L/min，1,400 W 芯片壳温 74.2°C < 85°C；2,000 W 芯片采用硅背面微通道直冷消除 R_TIM=0.02°C/W 后最大允许进液温度从 21.0°C 跃升至 61.0–69.0°C。",
            "跨园区 DCI 分层训练代表同一机构受控双园区 400G 专线（H=500 步外同步开销仅 1.48%，流式分片重叠后暴露为 0），与第 16 章公网不可信算力池严格区分。",
        ],
    }
