#!/usr/bin/env python3
"""Verification script for Experiment 23-01."""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results" / "lifecycle_and_preregistration.json"


def main():
    if not RES.exists():
        print("ERROR: results file missing; run run.py first.", file=sys.stderr)
        sys.exit(1)
    data = json.loads(RES.read_text())

    bc = data["beyond_chinchilla"]
    assert abs(bc["breakeven_t_tokens"] - 3.0) < 1e-6, f"Unexpected breakeven: {bc}"
    assert abs(bc["breakeven_days"] - 150.0) < 1e-6, f"Unexpected days: {bc}"
    assert abs(bc["net_savings_2yr_usd"] - 3_480_000.0) < 1e-2, f"Unexpected savings: {bc}"

    k192 = data["k192_preregistration"]
    assert k192["count"] == 10
    assert abs(k192["median_r"] - 0.693141) < 1e-6, f"Unexpected median_r: {k192}"
    assert k192["in_interval"] is False
    assert k192["verdict"] == "refuted_in_scope"

    ns = data["neusight_stage_calibration"]
    assert ns["aggregate_ape"] == 0.0
    assert abs(ns["stage_apes"][0] - 0.333333) < 1e-5
    assert abs(ns["stage_apes"][1] - 0.5) < 1e-6
    assert ns["halving_stage1_predicted_total_ms"] == 6.0
    assert ns["halving_stage1_actual_total_ms"] == 7.0

    print("PASS: Experiment 23-01 verified successfully.")


if __name__ == "__main__":
    main()
