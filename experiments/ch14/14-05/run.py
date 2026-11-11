#!/usr/bin/env python3
"""Experiment 14-05: Controlled Dual-Campus 400G Optical DCI (DeepSeek-V3 671B) & Campus Energy Ledger."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"
RESULTS_PATH = HERE / "results" / "summary.json"


def compute_controlled_dci_deepseek_v3() -> dict:
    total_params_b = 671.0
    active_params_b = 37.0
    layers = 61
    step_time_s = 4.20
    inner_steps_h = 500
    dci_bw_gbps = 400.0
    dci_eff = 0.85

    pseudo_grad_bytes = total_params_b * 1e9 * 2.0
    naive_per_step_gbps = round((pseudo_grad_bytes * 8.0 / step_time_s) / 1e9, 1)
    diloco_avg_gbps = round(naive_per_step_gbps / inner_steps_h, 2)

    dci_eff_bps = dci_bw_gbps * 1e9 / 8.0 * dci_eff
    outer_sync_s = round(pseudo_grad_bytes / dci_eff_bps, 2)
    inner_window_s = round(inner_steps_h * step_time_s, 2)
    blocking_overhead_pct = round(outer_sync_s / (inner_window_s + outer_sync_s) * 100.0, 2)

    shard_sync_s = outer_sync_s / layers
    shard_budget_s = inner_window_s / layers
    exposed_streaming_s = max(0.0, shard_sync_s - shard_budget_s)

    return {
        "total_params_b": total_params_b,
        "active_params_b": active_params_b,
        "deepseek_v3_layers": layers,
        "dci_671b_naive_per_step_gbps": round(naive_per_step_gbps, 1),
        "dci_671b_diloco_h500_avg_gbps": round(diloco_avg_gbps, 2),
        "dci_671b_outer_sync_seconds": round(outer_sync_s, 2),
        "dci_inner_500step_window_seconds": round(inner_window_s, 1),
        "dci_blocking_comm_overhead_pct": round(blocking_overhead_pct, 2),
        "streaming_shard_sync_seconds": round(shard_sync_s, 3),
        "streaming_shard_window_seconds": round(shard_budget_s, 2),
        "streaming_exposed_stall_seconds": round(exposed_streaming_s, 2),
    }


def compute_campus_power_and_bess_economics() -> dict:
    campus_mw = 100.0
    delta_price_usd_per_kwh = 0.12 - 0.04
    hours_per_year = 8760.0
    annual_saving_musd = (campus_mw * 1000.0 * hours_per_year * delta_price_usd_per_kwh) / 1e6
    three_year_saving_musd = annual_saving_musd * 3.0
    bess_20pct_2hr_mwh = campus_mw * 0.20 * 2.0

    return {
        "campus_100mw_elec_saving_musd_per_yr": round(annual_saving_musd, 2),
        "campus_100mw_3yr_saving_musd": round(three_year_saving_musd, 2),
        "bess_20pct_2hr_capacity_mwh": round(bess_20pct_2hr_mwh, 1),
    }


def compute_metrics() -> dict:
    return {
        "controlled_optical_dci": compute_controlled_dci_deepseek_v3(),
        "campus_economics": compute_campus_power_and_bess_economics(),
    }


def main() -> None:
    metrics = compute_metrics()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
