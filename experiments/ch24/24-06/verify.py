#!/usr/bin/env python3
"""Verify Experiment 24-06 against expected metrics and calculations/results/ch24-ai-factory-capstone.json."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]


def main() -> None:
    exp = json.loads((HERE / "expected" / "metrics.json").read_text())
    res = json.loads((HERE / "results" / "summary.json").read_text())
    assert exp == res, "Expected and results mismatch"

    b = res["hundred_mw_ai_factory_baseline"]
    c = res["conservation_laws_and_counterfactuals"]

    assert b["facility_grid_mw"] == 100.0
    assert b["it_power_budget_mw"] == 89.29
    assert b["compute_racks_power_mw"] == 81.84
    assert b["total_72gpu_racks"] == 496
    assert b["total_gpus"] == 35712
    assert b["active_training_gpus"] == 28672
    assert b["serving_rollout_gpus"] == 5904
    assert b["total_spare_gpus"] == 1136
    assert b["spare_pool_ratio_pct"] == 3.18
    assert b["sustained_ettr_pct"] == 97.6
    assert b["sustained_training_eflops"] == 27.81
    assert b["flagship_10t_pretrain_days"] == 62.4
    assert b["flagship_10t_total_days"] == 80.9
    assert b["agile_1t_pretrain_days"] == 2.75
    assert b["three_year_total_tco_musd"] == 2030.82
    assert b["amortized_usd_per_gpu_hour"] == 2.16
    assert b["serving_raw_partition_usd_per_1m_tok"] == 1.41
    assert b["serving_net_after_rl_share_usd_per_1m_tok"] == 0.61

    assert c["four_conservation_laws_verified"] == 4
    assert c["air_cooling_pue145_total_racks"] == 372
    assert c["air_cooling_pue145_10t_pretrain_days"] == 90.7
    assert c["power_shock_mitigated_tco_musd"] == 2096.41
    assert c["hbm_halved_10t_pretrain_days"] == 68.8
    assert c["hbm_halved_net_usd_per_1m_tok"] == 0.65

    golden_path = REPO_ROOT / "calculations" / "results" / "ch24-ai-factory-capstone.json"
    if golden_path.exists():
        golden = json.loads(golden_path.read_text())["summary"]
        for k in (
            "facility_grid_mw",
            "it_power_budget_mw",
            "compute_racks_power_mw",
            "total_72gpu_racks",
            "total_gpus",
            "active_training_gpus",
            "serving_rollout_gpus",
            "total_spare_gpus",
            "sustained_training_eflops",
            "flagship_10t_pretrain_days",
            "flagship_10t_total_days",
            "agile_1t_pretrain_days",
            "three_year_total_tco_musd",
            "amortized_usd_per_gpu_hour",
            "serving_raw_partition_usd_per_1m_tok",
            "serving_net_after_rl_share_usd_per_1m_tok",
        ):
            assert b[k] == golden[k], f"Mismatch vs ch24-ai-factory-capstone.json on {k}: {b[k]} != {golden[k]}"

    print("Experiment 24-06 verified successfully.")


if __name__ == "__main__":
    main()
