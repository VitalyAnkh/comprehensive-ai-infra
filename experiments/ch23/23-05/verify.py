#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 23-05."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH, RESULTS_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    results = json.loads(Path(RESULTS_PATH).read_text())
    assert actual == expected == results, f"Mismatch:\nactual={actual}\nexpected={expected}\nresults={results}"
    assert actual["blt_entropy_patching"]["saved_vocab_mem_gb"] == 4.194
    assert actual["blt_entropy_patching"]["seq_len_reduction_x"] == 2.67
    assert actual["blt_entropy_patching"]["e2e_flop_reduction_pct"] == 52.4
    assert actual["titans_ttt_chunk_dual"]["chunk_arith_intensity"] == 48.0
    assert actual["titans_ttt_chunk_dual"]["ttt_speedup_x"] == 24.6
    assert actual["titans_ttt_chunk_dual"]["fast_weight_mb_per_layer"] == 16.0
    print("Experiment 23-05 verified successfully.")


if __name__ == "__main__":
    main()
