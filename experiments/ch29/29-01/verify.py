#!/usr/bin/env python3
"""Verify Experiment 29-01."""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    exp = json.loads((HERE / "expected" / "metrics.json").read_text())
    res = json.loads((HERE / "results" / "summary.json").read_text())
    assert exp == res, "Expected and results mismatch"

    m = res["flashmla"]
    assert m["mla_kv_elements"] == 576
    assert m["mla_kv_bytes_per_token"] == 1152
    assert math.isclose(m["kv_compression_vs_mha64_x"], 28.44, rel_tol=1e-6)
    assert math.isclose(m["achieved_bw_gbs"], 3000.0, rel_tol=1e-6)
    assert math.isclose(m["bw_utilization_pct"], 89.55, rel_tol=1e-6)
    assert math.isclose(m["achieved_tflops"], 580.0, rel_tol=1e-6)

    g = res["deepgemm"]
    assert g["jit_regs_per_thread"] == 192
    assert math.isclose(g["ffma_cycle_overhead_pct"], 3.12, rel_tol=1e-6)
    assert math.isclose(g["achieved_fp8_tflops"], 1358.0, rel_tol=1e-6)
    print("Experiment 29-01 verified successfully.")


if __name__ == "__main__":
    main()
