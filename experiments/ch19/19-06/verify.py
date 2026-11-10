#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 19-06."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH, RESULTS_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    results = json.loads(Path(RESULTS_PATH).read_text())
    assert actual == expected == results, f"Mismatch:\nactual={actual}\nexpected={expected}\nresults={results}"
    assert actual["avatar_lipsync_alignment_ex22_6"]["raw_steady_skew_ms"] == 44.0
    assert actual["avatar_lipsync_alignment_ex22_6"]["aligned_steady_skew_ms"] == 18.0
    assert actual["avatar_lipsync_alignment_ex22_6"]["aligned_p99_skew_ms"] == 33.0
    assert actual["avatar_lipsync_alignment_ex22_6"]["skew_reduction_pct"] == 59.09
    assert actual["avatar_lipsync_alignment_ex22_6"]["avatar_ttfa_ms"] == 190.0
    assert actual["global_edge_sfu_mesh_ex22_7"]["net_edge_ms"] == 98.5
    assert actual["global_edge_sfu_mesh_ex22_7"]["net_reduction_pct"] == 43.71
    assert actual["global_edge_sfu_mesh_ex22_7"]["ttfa_edge_ms"] == 181.5
    assert actual["global_edge_sfu_mesh_ex22_7"]["ttfa_speedup_x"] == 1.67
    assert actual["global_edge_sfu_mesh_ex22_7"]["bw_compression_x"] == 43.64
    assert actual["global_edge_sfu_mesh_ex22_7"]["bw_saving_pct"] == 97.71
    assert actual["exercises_22_11_and_22_12"]["p11_aligned_steady_skew_ms"] == 16.0
    assert actual["exercises_22_11_and_22_12"]["p12_ttfa_edge_ms"] == 185.5
    print("Experiment 19-06 verified successfully.")


if __name__ == "__main__":
    main()
