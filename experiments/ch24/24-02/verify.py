#!/usr/bin/env python3
"""Verify Experiment 24-02."""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    exp = json.loads((HERE / "expected" / "metrics.json").read_text())
    res = json.loads((HERE / "results" / "summary.json").read_text())
    assert exp == res, "Expected and results mismatch"

    ep = res["deepep"]
    assert ep["normal_mode_sm_count"] == 24
    assert math.isclose(ep["normal_mode_bw_gbs"], 153.0, rel_tol=1e-6)
    assert math.isclose(ep["low_latency_dispatch_us"], 163.0, rel_tol=1e-6)
    assert math.isclose(ep["low_latency_combine_us"], 194.0, rel_tol=1e-6)
    assert ep["low_latency_sm_occupied"] == 0

    dp = res["dualpipe"]
    assert dp["pp_stages"] == 16
    assert dp["microbatches_per_direction"] == 16
    assert dp["bidirectional_microbatches"] == 32
    assert math.isclose(dp["exposed_alltoall_stall_pct"], 0.0, abs_tol=1e-9)
    assert dp["bubble_slots_1f1b"] == 15
    assert dp["bubble_slots_dualpipe"] == 7
    assert math.isclose(dp["bubble_1f1b_pct"], 31.91, rel_tol=1e-4)
    assert math.isclose(dp["bubble_dualpipe_pct"], 17.95, rel_tol=1e-4)
    print("Experiment 24-02 verified successfully.")


if __name__ == "__main__":
    main()
