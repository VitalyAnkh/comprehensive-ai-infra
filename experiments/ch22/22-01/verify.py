#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 22-01."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH, RESULTS_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    results = json.loads(Path(RESULTS_PATH).read_text())
    assert actual == expected == results, f"Mismatch:\nactual={actual}\nexpected={expected}\nresults={results}"
    assert actual["latency_comparison"]["cascaded_total_ttfa_ms"] == 1280.0
    assert actual["latency_comparison"]["duplex_total_ttfa_ms"] == 164.0
    assert actual["latency_comparison"]["ttfa_speedup_x"] == 7.8
    assert actual["rvq_codec_ledger"]["rvq_bitrate_kbps"] == 1.1
    assert actual["rvq_codec_ledger"]["codec_compression_x"] == 349.09
    assert actual["temporal_depth_cudagraph"]["total_frame_eager_ms"] == 38.54
    assert actual["temporal_depth_cudagraph"]["total_frame_cudagraph_ms"] == 11.5
    assert actual["temporal_depth_cudagraph"]["depth_speedup_x"] == 4.38
    assert actual["temporal_depth_cudagraph"]["full_step_speedup_x"] == 3.35
    print("Experiment 22-01 verified successfully.")


if __name__ == "__main__":
    main()
