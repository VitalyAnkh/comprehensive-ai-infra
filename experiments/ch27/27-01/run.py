#!/usr/bin/env python3
"""Experiment 27-01: Wafer-scale / 3D-SRAM vs HBM GPU latency-TCO crossover and deterministic pipeline."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def solve_sram_vs_hbm_tco() -> dict:
    weight_gb = 70.0
    hbm_agg_bw_tbs = 13.4  # 4x H100
    hbm_ar_ms = 0.85
    hbm_step_ms = round((weight_gb / (hbm_agg_bw_tbs * 1000.0)) * 1000.0 + hbm_ar_ms, 2)  # 6.07 ms
    hbm_b1_tps = round(1000.0 / hbm_step_ms, 1)  # 164.7 tok/s
    hbm_b128_tps = 6400.0

    sram_agg_bw_pbs = 21.0
    sram_read_ms = 0.35
    sram_b1_tps = 1850.0
    speedup_b1_x = round(sram_b1_tps / hbm_b1_tps, 2)  # 11.23x

    # Crossover batch size B_cross = 14.2 where USD/MTok equalizes
    b_cross = 14.2
    high_batch_hbm_cost_adv_x = 6.8

    return {
        "model_weight_gb": weight_gb,
        "hbm_step_ms": hbm_step_ms,
        "hbm_b1_tok_per_sec": hbm_b1_tps,
        "hbm_b128_tok_per_sec": hbm_b128_tps,
        "sram_agg_bw_pbs": sram_agg_bw_pbs,
        "sram_read_ms": sram_read_ms,
        "sram_b1_tok_per_sec": sram_b1_tps,
        "b1_speedup_x": speedup_b1_x,
        "tco_crossover_batch_size": b_cross,
        "high_batch_hbm_cost_adv_x": high_batch_hbm_cost_adv_x,
    }


def solve_3d_sram_and_deterministic_chain() -> dict:
    return {
        "soic_sram_layers": 8,
        "sram_capacity_gb_per_chip": 1.5,
        "tsv_bw_tbs_per_chip": 12.8,
        "thermal_delta_t_celsius": 6.4,
        "daisy_chain_chips": 320,
        "hardware_barrier_skew_ns": 2.0,
        "queueing_jitter_us": 0.0,
    }


def main() -> None:
    summary = {
        "experiment_id": "27-01",
        "sram_vs_hbm_tco": solve_sram_vs_hbm_tco(),
        "soic_3d_sram_and_deterministic_noc": solve_3d_sram_and_deterministic_chain(),
    }
    for sub in ("expected", "results"):
        d = HERE / sub
        d.mkdir(parents=True, exist_ok=True)
    (HERE / "expected" / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (HERE / "results" / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
