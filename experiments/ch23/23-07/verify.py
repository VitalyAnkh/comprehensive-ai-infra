#!/usr/bin/env python3
"""Verify experiment 23-07 results against expected metrics."""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    exp = json.loads((HERE / "expected" / "metrics.json").read_text())
    res = json.loads((HERE / "results" / "summary.json").read_text())
    assert exp == res
    tco = res["sram_vs_hbm_tco"]
    assert math.isclose(tco["hbm_b1_tok_per_sec"], 164.7, rel_tol=1e-6)
    assert math.isclose(tco["sram_b1_tok_per_sec"], 1850.0, rel_tol=1e-6)
    assert math.isclose(tco["b1_speedup_x"], 11.23, rel_tol=1e-6)
    assert math.isclose(tco["tco_crossover_batch_size"], 14.2, rel_tol=1e-6)
    assert math.isclose(tco["high_batch_hbm_cost_adv_x"], 6.8, rel_tol=1e-6)
    print("Experiment 23-07 verified successfully.")


if __name__ == "__main__":
    main()
