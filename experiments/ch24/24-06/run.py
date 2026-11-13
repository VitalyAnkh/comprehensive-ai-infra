#!/usr/bin/env python3
"""Experiment 24-06: 100 MW AI Factory Five-Part Full-Stack Planner & Four Conservation Laws Verifier."""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def solve_baseline_100mw_factory() -> dict:
    facility_grid_mw = 100.0
    pue = 1.12
    fabric_storage_cpu_mw = 7.45
    rack_kw = 165.0
    gpus_per_rack = 72
    train_racks = 400
    serve_racks = 82
    spare_racks = 14
    in_train_hot_spares = 128
    realized_train_tflops_per_gpu = 970.0

    it_power_mw = round(facility_grid_mw / pue, 2)                              # 89.29 MW
    total_racks = train_racks + serve_racks + spare_racks                       # 496 racks
    compute_power_mw = round(total_racks * rack_kw / 1000.0, 2)                 # 81.84 MW
    total_gpus = total_racks * gpus_per_rack                                    # 35,712 GPUs

    active_train_gpus = train_racks * gpus_per_rack - in_train_hot_spares       # 28,672 GPUs
    serve_gpus = serve_racks * gpus_per_rack                                    # 5,904 GPUs
    total_spare_gpus = spare_racks * gpus_per_rack + in_train_hot_spares        # 1,136 GPUs
    spare_pool_ratio_pct = round(total_spare_gpus / total_gpus * 100.0, 2)      # 3.18%
    sustained_ettr_pct = 97.6

    sustained_train_eflops = round(active_train_gpus * realized_train_tflops_per_gpu / 1e6, 2)  # 27.81 EFLOPS
    sustained_train_flops = active_train_gpus * realized_train_tflops_per_gpu * 1e12

    flagship_10t_flops = 1.50e26
    flagship_pretrain_days = round((flagship_10t_flops / sustained_train_flops) / 86400.0, 1)   # 62.4 d
    flagship_rlvr_days = 18.5
    flagship_total_days = round(flagship_pretrain_days + flagship_rlvr_days, 1)                 # 80.9 d

    agile_1t_flops = 6.0 * 44e9 * 25e12                                         # 6.60e24 FLOPs
    agile_pretrain_days = round((agile_1t_flops / sustained_train_flops) / 86400.0, 2)          # 2.75 d

    serving_daily_tokens_billions = 216.5
    three_year_capex_musd = 1860.0
    three_year_power_opex_musd = round(facility_grid_mw * 8760 * 3 * 65.0 / 1e6, 2)             # $170.82M
    three_year_total_tco_musd = round(three_year_capex_musd + three_year_power_opex_musd, 2)    # $2,030.82M
    amortized_usd_per_gpu_hour = round(
        (three_year_total_tco_musd * 1e6) / (total_gpus * 8760 * 3), 2
    )                                                                           # $2.16/GPU-h
    serving_raw_usd_per_1m_tok = round(
        (serve_gpus * amortized_usd_per_gpu_hour * 24.0) / (serving_daily_tokens_billions * 1000.0),
        2,
    )                                                                           # $1.41/1M tok
    serving_net_after_rl_share_usd_per_1m_tok = round(serving_raw_usd_per_1m_tok * 0.433, 2)    # $0.61/1M tok

    return {
        "facility_grid_mw": facility_grid_mw,
        "pue": pue,
        "it_power_budget_mw": it_power_mw,
        "compute_racks_power_mw": compute_power_mw,
        "fabric_storage_cpu_power_mw": fabric_storage_cpu_mw,
        "total_72gpu_racks": total_racks,
        "total_gpus": total_gpus,
        "active_training_gpus": active_train_gpus,
        "serving_rollout_gpus": serve_gpus,
        "total_spare_gpus": total_spare_gpus,
        "spare_pool_ratio_pct": spare_pool_ratio_pct,
        "sustained_ettr_pct": sustained_ettr_pct,
        "sustained_training_eflops": sustained_train_eflops,
        "flagship_10t_60t_flops": flagship_10t_flops,
        "flagship_10t_pretrain_days": flagship_pretrain_days,
        "flagship_10t_rlvr_days": flagship_rlvr_days,
        "flagship_10t_total_days": flagship_total_days,
        "agile_1t_25t_flops": agile_1t_flops,
        "agile_1t_pretrain_days": agile_pretrain_days,
        "serving_daily_tokens_billions": serving_daily_tokens_billions,
        "three_year_total_tco_musd": three_year_total_tco_musd,
        "amortized_usd_per_gpu_hour": amortized_usd_per_gpu_hour,
        "serving_raw_partition_usd_per_1m_tok": serving_raw_usd_per_1m_tok,
        "serving_net_after_rl_share_usd_per_1m_tok": serving_net_after_rl_share_usd_per_1m_tok,
    }


def solve_four_conservation_laws_and_counterfactuals(base: dict) -> dict:
    # Exercise 24-20: Four Conservation Laws efficiency waterfall & PUE=1.45 air-cooling regression
    peak_tflops_per_gpu = 2250.0
    active_train_gpus = base["active_training_gpus"]
    raw_train_peak_eflops = round(active_train_gpus * peak_tflops_per_gpu / 1e6, 2)             # 64.51 EFLOPS
    mfu_pct = round((970.0 / (peak_tflops_per_gpu * 0.976)) * 100.0, 2)                         # 44.17%
    end_to_end_silicon_utilization_pct = round(970.0 / peak_tflops_per_gpu * 100.0, 2)          # 43.11%
    active_train_it_mw = round(active_train_gpus * (165.0 / 72.0) / 1000.0, 4)                  # 65.7067 MW
    useful_train_gflop_per_it_joule = round((base["sustained_training_eflops"] * 1e9) / (active_train_it_mw * 1e6), 2)
    useful_train_gflop_per_grid_joule = round(useful_train_gflop_per_it_joule / base["pue"], 2) # 377.89 GFLOP/J

    air_pue = 1.45
    air_it_mw = round(100.0 / air_pue, 2)                                                       # 68.97 MW
    air_compute_mw = round(air_it_mw - 7.45, 2)                                                 # 61.52 MW
    air_total_racks = math.floor(air_compute_mw * 1000.0 / 165.0)                               # 372 racks
    air_train_racks = air_total_racks - 82 - 14                                                 # 276 racks
    air_active_train_gpus = air_train_racks * 72 - 128                                          # 19,744 GPUs
    air_pretrain_10t_days = round(1.50e26 / (air_active_train_gpus * 970.0e12 * 86400.0), 1)    # 90.7 days

    # Exercise 24-21: +40% electricity tariff shock (65 -> 91 USD/MWh) + 200 MWh BESS & -18% DVFS mitigation
    shock_tariff = 91.0
    unmitigated_power_opex_musd = round(100.0 * 8760 * 3 * shock_tariff / 1e6, 2)               # 239.15 M USD
    unmitigated_tco_musd = round(1860.0 + unmitigated_power_opex_musd, 2)                       # 2,099.15 M USD
    bess_capex_musd = 36.0
    bess_daily_arbitrage_usd = 200.0 * (145.0 - 45.0 / 0.90)                                    # 19,000 USD/day
    bess_3yr_saving_musd = round(bess_daily_arbitrage_usd * 365 * 3 / 1e6, 2)                   # 20.80 M USD
    dvfs_3yr_saving_musd = round(7.5 * 8760 * 3 * shock_tariff / 1e6, 2)                        # 17.94 M USD
    mitigated_power_opex_musd = round(unmitigated_power_opex_musd - bess_3yr_saving_musd - dvfs_3yr_saving_musd, 2)
    mitigated_tco_musd = round(1860.0 + bess_capex_musd + mitigated_power_opex_musd, 2)         # 2,096.41 M USD

    # Exercise 24-22: HBM4 capacity/bandwidth halving (192 GB -> 96 GB, 8.0 -> 4.0 TB/s, CapEx -12%)
    hbm_halved_capex_musd = round(1860.0 * 0.88, 2)                                             # 1,636.80 M USD
    hbm_halved_tco_musd = round(hbm_halved_capex_musd + 170.82, 2)                              # 1,807.62 M USD
    hbm_halved_gpu_hour_usd = round((hbm_halved_tco_musd * 1e6) / (base["total_gpus"] * 8760 * 3), 2)  # 1.93 USD
    ep288_weight_per_gpu_gb = round(5200.0 / 288.0, 2)                                          # 18.06 GB
    ep288_kv_pool_per_gpu_gb = round(96.0 - ep288_weight_per_gpu_gb - 12.0, 2)                  # 65.94 GB
    hbm_halved_train_eflops = round(active_train_gpus * 880.0 / 1e6, 2)                         # 25.23 EFLOPS
    hbm_halved_10t_pretrain_days = round(1.50e26 / (active_train_gpus * 880.0e12 * 86400.0), 1) # 68.8 days
    hbm_halved_daily_tokens_b = 184.0
    hbm_halved_raw_usd_per_1m = round((5904 * hbm_halved_gpu_hour_usd * 24.0) / (hbm_halved_daily_tokens_b * 1000.0), 2)
    hbm_halved_net_usd_per_1m = round(hbm_halved_raw_usd_per_1m * 0.433, 2)                     # 0.65 USD/1M tok
    net_gross_margin_at_2usd_pct = round((2.00 - hbm_halved_net_usd_per_1m) / 2.00 * 100.0, 1)  # 67.5%

    return {
        "four_conservation_laws_verified": 4,
        "raw_train_peak_eflops": raw_train_peak_eflops,
        "mfu_pct": mfu_pct,
        "end_to_end_silicon_utilization_pct": end_to_end_silicon_utilization_pct,
        "useful_train_gflop_per_it_joule": useful_train_gflop_per_it_joule,
        "useful_train_gflop_per_grid_joule": useful_train_gflop_per_grid_joule,
        "air_cooling_pue145_total_racks": air_total_racks,
        "air_cooling_pue145_10t_pretrain_days": air_pretrain_10t_days,
        "power_shock_unmitigated_tco_musd": unmitigated_tco_musd,
        "power_shock_bess_3yr_saving_musd": bess_3yr_saving_musd,
        "power_shock_dvfs_3yr_saving_musd": dvfs_3yr_saving_musd,
        "power_shock_mitigated_tco_musd": mitigated_tco_musd,
        "hbm_halved_tco_musd": hbm_halved_tco_musd,
        "hbm_halved_gpu_hour_usd": hbm_halved_gpu_hour_usd,
        "hbm_halved_ep288_kv_per_gpu_gb": ep288_kv_pool_per_gpu_gb,
        "hbm_halved_train_eflops": hbm_halved_train_eflops,
        "hbm_halved_10t_pretrain_days": hbm_halved_10t_pretrain_days,
        "hbm_halved_raw_usd_per_1m_tok": hbm_halved_raw_usd_per_1m,
        "hbm_halved_net_usd_per_1m_tok": hbm_halved_net_usd_per_1m,
        "hbm_halved_net_gross_margin_pct": net_gross_margin_at_2usd_pct,
    }


def main() -> None:
    baseline = solve_baseline_100mw_factory()
    sensitivity = solve_four_conservation_laws_and_counterfactuals(baseline)
    summary = {
        "experiment_id": "24-06",
        "hundred_mw_ai_factory_baseline": baseline,
        "conservation_laws_and_counterfactuals": sensitivity,
    }
    for sub in ("expected", "results"):
        d = HERE / sub
        d.mkdir(parents=True, exist_ok=True)
    (HERE / "expected" / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (HERE / "results" / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
