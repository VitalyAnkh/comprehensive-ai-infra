#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 26-03."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH, RESULTS_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    results = json.loads(Path(RESULTS_PATH).read_text())
    assert actual == expected == results, f"Mismatch:\nactual={actual}\nexpected={expected}\nresults={results}"
    assert actual["coconut_latent_loop"]["latent_thought_steps"] == 100
    assert actual["coconut_latent_loop"]["latency_reduction_pct"] == 69.2
    assert actual["coconut_latent_loop"]["coconut_speedup_x"] == 3.25
    assert actual["mod_stream_compaction"]["uncompacted_warp_eff_pct"] == 58.1
    assert actual["mod_stream_compaction"]["compacted_warp_eff_pct"] == 96.4
    assert actual["mod_stream_compaction"]["mod_layer_speedup_x"] == 3.32
    print("Experiment 26-03 verified successfully.")


if __name__ == "__main__":
    main()
