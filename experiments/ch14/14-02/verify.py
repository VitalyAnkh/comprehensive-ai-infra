#!/usr/bin/env python3
"""Verification script for Experiment 14-02."""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results" / "telescoping_and_farskip.json"


def main():
    if not RES.exists():
        print("ERROR: results file missing; run run.py first.", file=sys.stderr)
        sys.exit(1)
    data = json.loads(RES.read_text())

    tw = data["telescoping_waterfall"]
    assert abs(tw["telescoping_product"] - tw["direct_ratio"]) < 1e-6
    assert abs(tw["direct_ratio"] - 25.074627) < 1e-5
    assert tw["conditional_ratios"][0] == 8.75
    assert tw["conditional_ratios"][1] == 2.0

    fs = data["farskip_collective"]
    assert abs(fs["m=0"]["total_forward_ms"] - 120.0) < 1e-4
    assert abs(fs["m=1"]["total_forward_ms"] - 72.8) < 1e-4
    assert abs(fs["m=1"]["speedup_vs_m0"] - 1.6484) < 1e-3

    gqa = data["gqa_tp_replication"]
    assert gqa["n_kv=8"]["replication_factor"] == 1
    assert gqa["n_kv=2"]["replication_factor"] == 4
    assert gqa["n_kv=8"]["cluster_8gpu_physical_gib"] == gqa["n_kv=2"]["cluster_8gpu_physical_gib"] == 16.0

    print("PASS: Experiment 14-02 verified successfully.")


if __name__ == "__main__":
    main()
