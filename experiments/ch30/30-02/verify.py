#!/usr/bin/env python3
"""Verify Experiment 30-02."""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    exp = json.loads((HERE / "expected" / "metrics.json").read_text())
    res = json.loads((HERE / "results" / "summary.json").read_text())
    assert exp == res, "Expected and results mismatch"

    dvfs = res["per_kernel_dvfs"]
    assert math.isclose(dvfs["freq_downclock_pct"], 18.0, rel_tol=1e-6)
    assert math.isclose(dvfs["dynamic_power_saving_pct"], 26.5, rel_tol=1e-6)

    rca = res["autonomous_aiops_rca"]
    assert rca["cluster_gpus"] == 32768
    assert math.isclose(rca["mttr_reduction_x"], 360.0, rel_tol=1e-6)
    assert math.isclose(rca["auto_ettr_pct"], 97.8, rel_tol=1e-6)
    assert math.isclose(rca["six_month_saved_usd_millions"], 13.5, rel_tol=1e-6)

    dt = res["digital_twin_cosim"]
    assert math.isclose(dt["prevented_moe_slowdown_pct"], 21.4, rel_tol=1e-6)
    print("Experiment 30-02 verified successfully.")


if __name__ == "__main__":
    main()
