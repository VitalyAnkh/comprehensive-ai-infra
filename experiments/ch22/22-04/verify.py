#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 22-04."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH, RESULTS_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    results = json.loads(Path(RESULTS_PATH).read_text())
    assert actual == expected == results, f"Mismatch:\nactual={actual}\nexpected={expected}\nresults={results}"
    assert actual["hardware_profile"]["int8_to_fp64_peak_ratio_x"] == 112.5
    assert actual["hardware_profile"]["signed_int8_max_k_no_overflow"] == 133144
    bench = actual["ozaki_s7_benchmark"]
    assert bench["num_int8_gemms"] == 28
    assert bench["effective_mantissa_bits"] == 51.8
    assert bench["compute_ceiling_x"] == 4.02
    assert bench["vs_native_fp64_speedup_x"] == 3.68
    assert bench["equiv_fp64_tflops"] == 147.2
    assert bench["numerical_test_rel_frobenius_less_than_1e14"] is True
    print("Experiment 22-04 verified successfully.")


if __name__ == "__main__":
    main()
