#!/usr/bin/env python3
"""Verify Experiment 24-03."""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    exp = json.loads((HERE / "expected" / "metrics.json").read_text())
    res = json.loads((HERE / "results" / "summary.json").read_text())
    assert exp == res, "Expected and results mismatch"

    fs = res["three_fs_and_smallpond"]
    assert fs["storage_nodes"] == 180
    assert fs["total_nvme_ssds"] == 2880
    assert math.isclose(fs["aggregate_read_bw_tbs"], 6.60, rel_tol=1e-6)
    assert fs["graysort_time_sec"] == 1814

    kt = res["mooncake_and_ktransformers"]
    assert math.isclose(kt["prefill_tok_per_sec"], 286.4, rel_tol=1e-6)
    assert math.isclose(kt["decode_tok_per_sec"], 18.6, rel_tol=1e-6)
    assert math.isclose(kt["speedup_vs_llamacpp_x"], 13.8, rel_tol=1e-6)
    print("Experiment 24-03 verified successfully.")


if __name__ == "__main__":
    main()
