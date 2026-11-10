#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 13-03."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH, RESULTS_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    results = json.loads(Path(RESULTS_PATH).read_text())
    assert actual == expected == results, f"Mismatch:\nactual={actual}\nexpected={expected}\nresults={results}"
    assert actual["rubric_distillation"]["distillation_compute_leverage_x"] == 55.82
    assert actual["synthetic_data_factory"]["throughput_gain_x"] == 7.2
    assert actual["synthetic_data_factory"]["raw_cost_reduction_x"] == 13.16
    assert actual["synthetic_data_factory"]["effective_verified_yield_pct"] == 58.75
    assert actual["synthetic_data_factory"]["verified_usd_per_1m_tokens"] == 0.075
    print("Experiment 13-03 verified successfully.")


if __name__ == "__main__":
    main()
