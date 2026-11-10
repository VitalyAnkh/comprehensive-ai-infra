#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 23-04."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH, RESULTS_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    results = json.loads(Path(RESULTS_PATH).read_text())
    assert actual == expected == results, f"Mismatch:\nactual={actual}\nexpected={expected}\nresults={results}"
    assert actual["block_diffusion_roofline"]["ar_tok_per_sec"] == 100.0
    assert actual["block_diffusion_roofline"]["dllm_forwards"] == 80
    assert actual["block_diffusion_roofline"]["weight_read_reduction_x"] == 6.4
    assert actual["block_diffusion_roofline"]["dllm_tok_per_sec"] == 592.6
    assert actual["block_diffusion_roofline"]["speedup_vs_ar_x"] == 5.93
    assert actual["confidence_remasking_early_exit"]["early_stop_forwards"] == 52
    assert actual["confidence_remasking_early_exit"]["early_stop_tok_per_sec"] == 911.7
    print("Experiment 23-04 verified successfully.")


if __name__ == "__main__":
    main()
