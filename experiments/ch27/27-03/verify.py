#!/usr/bin/env python3
"""Verify experiment 27-03 results against expected metrics."""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    exp = json.loads((HERE / "expected" / "metrics.json").read_text())
    res = json.loads((HERE / "results" / "summary.json").read_text())
    assert exp == res
    b = res["bitnet_b158_and_tmac"]
    assert math.isclose(b["shannon_entropy_bits"], 1.585, rel_tol=1e-6)
    assert math.isclose(b["packed_bits_per_weight"], 1.60, rel_tol=1e-6)
    assert math.isclose(b["model_70b_bitnet_gb"], 14.0, rel_tol=1e-6)
    assert math.isclose(b["compression_vs_fp16_x"], 10.0, rel_tol=1e-6)
    assert math.isclose(b["tmac_speedup_x"], 6.4, rel_tol=1e-6)
    assert math.isclose(b["dynamic_mul_power_saving_pct"], 88.5, rel_tol=1e-6)
    print("Experiment 27-03 verified successfully.")


if __name__ == "__main__":
    main()
