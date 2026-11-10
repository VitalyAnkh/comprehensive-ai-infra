#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 21-02."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH, RESULTS_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    results = json.loads(Path(RESULTS_PATH).read_text())
    assert actual == expected == results, f"Mismatch:\nactual={actual}\nexpected={expected}\nresults={results}"
    assert actual["hstu_stochastic_length"]["quadratic_flop_reduction_pct"] == 56.2
    assert actual["mfalcon_microbatch_amortization"]["naive_total_tflops"] == 1409.3
    assert actual["mfalcon_microbatch_amortization"]["mfalcon_total_tflops"] == 31.46
    assert actual["mfalcon_microbatch_amortization"]["flop_reduction_x"] == 44.8
    assert actual["mfalcon_microbatch_amortization"]["h100_qps"] == 485
    assert actual["mfalcon_microbatch_amortization"]["items_scored_per_sec"] == 496640
    print("Experiment 21-02 verified successfully.")


if __name__ == "__main__":
    main()
