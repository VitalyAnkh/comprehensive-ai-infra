#!/usr/bin/env python3
"""Verify experiment 23-08 results against expected metrics."""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    exp = json.loads((HERE / "expected" / "metrics.json").read_text())
    res = json.loads((HERE / "results" / "summary.json").read_text())
    assert exp == res
    hft = res["sub_millisecond_hft_ai"]
    assert math.isclose(hft["baseline_p50_us"], 78.0, rel_tol=1e-6)
    assert math.isclose(hft["baseline_p9999_us"], 420.0, rel_tol=1e-6)
    assert math.isclose(hft["persistent_p50_us"], 16.5, rel_tol=1e-6)
    assert math.isclose(hft["persistent_p9999_us"], 24.8, rel_tol=1e-6)
    assert math.isclose(hft["p9999_tail_reduction_x"], 16.94, rel_tol=1e-6)
    print("Experiment 23-08 verified successfully.")


if __name__ == "__main__":
    main()
