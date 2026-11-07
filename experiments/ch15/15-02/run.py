#!/usr/bin/env python3
"""Experiment 15-02: Campus BESS arbitrage, Kareus energy consolidation, and DiLoCo/Di-PS WAN sync."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"


def compute_metrics() -> dict:
    # 1. DiLoCo / Streaming DiLoCo WAN ledger (100B model)
    sync_bits_gb = 3200.0
    step_s = 2.0
    h_steps = 500
    link_gbps = 10.0
    bw_h1_gbps = round(sync_bits_gb / step_s, 2)
    bw_diloco_avg_gbps = round(sync_bits_gb / (h_steps * step_s), 2)
    blocking_stall_s = round(sync_bits_gb / link_gbps, 2)
    num_shards = 25
    shard_interval_s = round((h_steps / num_shards) * step_s, 2)
    shard_tx_s = round((sync_bits_gb / num_shards) / link_gbps, 2)
    streaming_exposed_stall_s = max(0.0, shard_tx_s - shard_interval_s)

    # 2. Kareus dynamic + static energy consolidation (Exercise 15-12)
    p_static_w = 180.0
    p_dyn_w = 520.0
    utilization = 0.40
    p_uniform_per_gpu_w = round(p_static_w + utilization * p_dyn_w, 2)
    p_consolidated_zero_sleep_w = round(utilization * (p_static_w + p_dyn_w), 2)
    p_sleep_w = 15.0
    p_consolidated_15w_sleep_w = round(
        utilization * (p_static_w + p_dyn_w) + (1.0 - utilization) * p_sleep_w, 2
    )
    saving_pct_zero_sleep = round(
        (p_uniform_per_gpu_w - p_consolidated_zero_sleep_w) / p_uniform_per_gpu_w * 100.0, 2
    )
    saving_pct_15w_sleep = round(
        (p_uniform_per_gpu_w - p_consolidated_15w_sleep_w) / p_uniform_per_gpu_w * 100.0, 2
    )

    # 3. 100 MWh Campus BESS peak-valley arbitrage (Exercise 15-13)
    bess_mwh = 100.0
    eta_rt = 0.88
    delta_price_nominal = 0.08
    c_off = 0.04
    c_peak = 0.12
    # Net revenue per cycle under two standard accounting conventions:
    # (a) Effective discharged energy * spread: E * eta_rt * 0.08
    # (b) Exact buy-at-valley / discharge-at-peak: E * (eta_rt * c_peak - c_off)
    daily_rev_spread_usd = round(bess_mwh * 1000.0 * eta_rt * delta_price_nominal, 2)
    annual_rev_spread_usd = round(daily_rev_spread_usd * 365.0, 2)
    daily_rev_exact_usd = round(bess_mwh * 1000.0 * (eta_rt * c_peak - c_off), 2)
    annual_rev_exact_usd = round(daily_rev_exact_usd * 365.0, 2)

    return {
        "experiment_id": "15-02",
        "diloco_wan": {
            "bw_h1_gbps": bw_h1_gbps,
            "bw_diloco_avg_gbps": bw_diloco_avg_gbps,
            "blocking_stall_s": blocking_stall_s,
            "shard_interval_s": shard_interval_s,
            "shard_tx_s": shard_tx_s,
            "streaming_exposed_stall_s": streaming_exposed_stall_s,
        },
        "kareus_power": {
            "p_uniform_per_gpu_w": p_uniform_per_gpu_w,
            "p_consolidated_zero_sleep_w": p_consolidated_zero_sleep_w,
            "p_consolidated_15w_sleep_w": p_consolidated_15w_sleep_w,
            "saving_pct_zero_sleep": saving_pct_zero_sleep,
            "saving_pct_15w_sleep": saving_pct_15w_sleep,
        },
        "bess_100mwh": {
            "daily_rev_spread_usd": daily_rev_spread_usd,
            "annual_rev_spread_usd": annual_rev_spread_usd,
            "daily_rev_exact_usd": daily_rev_exact_usd,
            "annual_rev_exact_usd": annual_rev_exact_usd,
        },
    }


def main() -> None:
    metrics = compute_metrics()
    EXPECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPECTED_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
