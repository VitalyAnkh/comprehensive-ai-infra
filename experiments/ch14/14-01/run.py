#!/usr/bin/env python3
"""Experiment 14-01: Liquid cooling thermal resistance, LogicFolding paradox, and TAPAS routing."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"


def compute_metrics() -> dict:
    # 1. LogicFolding power density paradox
    v_old, v_new = 0.85, 0.55
    voltage_sq_ratio = round((v_new / v_old) ** 2, 4)
    power_ratio = 0.75
    area_ratio = 0.60
    power_density_ratio = round(power_ratio / area_ratio, 4)

    # 2. TAPAS row-level and GPU-level admission & break-even
    row_base_kw = [96, 80]
    req_kw, cap_kw = 6, 100
    row_lat_s = [0.8, 1.3]
    deadline_s = 1.0
    feasible = [i for i, b in enumerate(row_base_kw) if b + req_kw <= cap_kw]
    admitted = [i for i in feasible if row_lat_s[i] <= deadline_s]
    breakeven_requests = int(round(3.0 / 0.1))
    thermal_margin_c = round(85.0 - (68.5 + 1.8 * 6), 1)
    n_min_tokens = int(120.0 // (48.0 - 38.0)) + 1

    # 3. Direct-to-silicon microfluidics eliminating R_TIM = 0.02 C/W at 2,000 W
    p_chip_w = 2000.0
    r_tim_cw = 0.02
    r_conv_coldplate_cw = 0.012
    r_conv_microfluidics_cw = 0.008
    t_j_max_c = 85.0
    t_in_max_coldplate_c = round(t_j_max_c - p_chip_w * (r_tim_cw + r_conv_coldplate_cw), 2)
    t_in_max_elim_tim_only_c = round(t_j_max_c - p_chip_w * r_conv_coldplate_cw, 2)
    delta_t_in_from_tim_c = round(p_chip_w * r_tim_cw, 2)
    t_in_max_microfluidics_c = round(t_j_max_c - p_chip_w * r_conv_microfluidics_cw, 2)

    return {
        "experiment_id": "14-01",
        "logicfolding": {
            "voltage_sq_ratio": voltage_sq_ratio,
            "power_ratio": power_ratio,
            "area_ratio": area_ratio,
            "power_density_ratio": power_density_ratio,
        },
        "tapas": {
            "feasible_rows": feasible,
            "admitted_rows": admitted,
            "breakeven_requests": breakeven_requests,
            "thermal_margin_c": thermal_margin_c,
            "n_min_tokens": n_min_tokens,
        },
        "microfluidics_2kw": {
            "p_chip_w": p_chip_w,
            "r_tim_cw": r_tim_cw,
            "t_in_max_coldplate_c": t_in_max_coldplate_c,
            "t_in_max_elim_tim_only_c": t_in_max_elim_tim_only_c,
            "delta_t_in_from_tim_c": delta_t_in_from_tim_c,
            "t_in_max_microfluidics_c": t_in_max_microfluidics_c,
        },
    }


def main() -> None:
    metrics = compute_metrics()
    EXPECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPECTED_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
