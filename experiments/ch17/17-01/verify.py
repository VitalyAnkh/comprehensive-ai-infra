#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 17-01."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    assert actual == expected, f"Mismatch:\nactual={actual}\nexpected={expected}"
    assert actual["qwen3_vl_4b_state"]["visual_tokens"] == 400
    assert actual["qwen3_vl_4b_state"]["ec_mib"] == 7.8125
    assert actual["qwen3_vl_4b_state"]["kv_mib"] == 56.25
    assert actual["qwen3_vl_4b_state"]["kv_to_ec_ratio"] == 7.2
    assert actual["video_dit_2d_usp"]["n_vid_tokens"] == 489600
    assert actual["video_dit_2d_usp"]["ulysses_64_per_layer_mib"] == 239.0625
    assert actual["video_dit_2d_usp"]["ring_64_total_mib"] == 7530.46875
    assert actual["video_dit_2d_usp"]["inter_node_reduction_ratio"] == 9.0
    assert actual["vae_and_teacache"]["teacache_speedup"] == 2.12
    print("Experiment 17-01 verified successfully.")


if __name__ == "__main__":
    main()
