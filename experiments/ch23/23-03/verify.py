#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 23-03."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH, SUMMARY_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    summary = json.loads(Path(SUMMARY_PATH).read_text())
    assert actual == expected == summary, f"Mismatch:\nactual={actual}\nexpected={expected}"
    g8 = [p for p in actual["granularity_sweep"] if p["G"] == 8][0]
    assert g8["layer_latency_ms"] == 0.74 and g8["loss_delta_nats"] == -0.072
    assert actual["chip_comparison"]["rubin_ridge_flop_per_b"] == 2272.73
    assert actual["chip_comparison"]["tpuv7_ridge_flop_per_b"] == 626.05
    assert actual["chip_comparison"]["rubin_step_ms"] == 11.86
    assert actual["chip_comparison"]["tpuv7_step_ms"] == 33.64
    assert actual["hwnas_optimal"]["best_G"] == 8
    assert actual["hwnas_optimal"]["best_attn_ratio"] == 0.25
    assert actual["hwnas_optimal"]["best_tpot_ms"] == 12.91
    assert actual["hwnas_optimal"]["best_state_gib"] == 28.25
    print("Experiment 23-03 verified successfully.")


if __name__ == "__main__":
    main()
