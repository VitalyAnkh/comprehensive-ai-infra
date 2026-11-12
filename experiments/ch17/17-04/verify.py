#!/usr/bin/env python3
"""Verifier for Experiment 17-04."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"
SUMMARY_PATH = HERE / "results" / "summary.json"


def main() -> None:
    if not EXPECTED_PATH.exists() or not SUMMARY_PATH.exists():
        print("Missing expected/metrics.json or results/summary.json; run run.py first.", file=sys.stderr)
        sys.exit(1)
    expected = json.loads(EXPECTED_PATH.read_text())
    actual = json.loads(SUMMARY_PATH.read_text())
    assert expected == actual, f"Mismatch:\nexpected={expected}\nactual={actual}"

    sft = actual["sft_varlen_and_zigzag"]
    assert sft["sft_pad_to_64k_mfu_pct"] == 13.6
    assert sft["sft_varlen_unbalanced_mfu_pct"] == 41.2
    assert sft["sft_varlen_zigzag_mfu_pct"] == 56.4
    assert sft["sft_varlen_throughput_speedup_x"] == 4.15
    assert sft["balanced_quad_max_to_mean"] < sft["naive_quad_max_to_mean"]
    assert sft["rank0_position_reset_verified"] is True
    assert sft["zigzag_cp4_equal_causal_blocks"] is True

    pref = actual["preference_alignment_ledger"]
    assert pref["dpo_32b_static_state_gb"] == 576.0
    assert pref["simpo_32b_static_state_gb"] == 512.0
    assert pref["simpo_step_flop_saving_pct"] == 25.0
    print("Experiment 17-04 verified: all locked metrics match.")


if __name__ == "__main__":
    main()
