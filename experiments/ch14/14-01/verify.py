#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 14-01."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    assert actual == expected, f"Mismatch:\nactual={actual}\nexpected={expected}"
    assert abs(actual["logicfolding"]["power_density_ratio"] - 1.25) < 1e-6
    assert actual["tapas"]["feasible_rows"] == [1]
    assert actual["tapas"]["admitted_rows"] == []
    assert actual["tapas"]["breakeven_requests"] == 30
    assert abs(actual["tapas"]["thermal_margin_c"] - 5.7) < 1e-6
    assert actual["tapas"]["n_min_tokens"] == 13
    assert abs(actual["microfluidics_2kw"]["delta_t_in_from_tim_c"] - 40.0) < 1e-6
    print("Experiment 14-01 verified successfully.")


if __name__ == "__main__":
    main()
