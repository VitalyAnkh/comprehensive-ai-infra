#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 14-02."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    assert actual == expected, f"Mismatch:\nactual={actual}\nexpected={expected}"
    assert actual["diloco_wan"]["bw_h1_gbps"] == 1600.0
    assert actual["diloco_wan"]["bw_diloco_avg_gbps"] == 3.2
    assert actual["diloco_wan"]["streaming_exposed_stall_s"] == 0.0
    assert actual["kareus_power"]["p_uniform_per_gpu_w"] == 388.0
    assert actual["kareus_power"]["p_consolidated_zero_sleep_w"] == 280.0
    assert actual["kareus_power"]["p_consolidated_15w_sleep_w"] == 289.0
    print("Experiment 14-02 verified successfully.")


if __name__ == "__main__":
    main()
