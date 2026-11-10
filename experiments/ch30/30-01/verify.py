#!/usr/bin/env python3
"""Verify Experiment 30-01."""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    exp = json.loads((HERE / "expected" / "metrics.json").read_text())
    res = json.loads((HERE / "results" / "summary.json").read_text())
    assert exp == res, "Expected and results mismatch"

    k = res["rlvr_kernel_synthesis"]
    assert k["benchmark_operators"] == 250
    assert math.isclose(k["zero_shot_compile_correct_pct"], 34.0, rel_tol=1e-6)
    assert k["nsight_feedback_rounds"] == 8
    assert math.isclose(k["multi_turn_compile_correct_pct"], 94.8, rel_tol=1e-6)
    assert math.isclose(k["speedup_vs_eager_x"], 3.42, rel_tol=1e-6)
    assert math.isclose(k["speedup_vs_torch_compile_x"], 1.58, rel_tol=1e-6)
    assert math.isclose(k["saved_hbm_traffic_gbs"], 38.4, rel_tol=1e-6)
    print("Experiment 30-01 verified successfully.")


if __name__ == "__main__":
    main()
