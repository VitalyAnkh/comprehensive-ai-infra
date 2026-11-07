#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 16-01."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    assert actual == expected, f"Mismatch:\nactual={actual}\nexpected={expected}"
    assert actual["pipellm"]["t_serial_ms"] == 138.0
    assert actual["pipellm"]["t_pipe_ms"] == 62.0
    assert actual["pipellm"]["triple_buffer_mib"] == 192
    assert actual["pipellm"]["speculative_p_breakeven"] == 0.4
    assert abs(actual["moe_traffic_padding"]["bandwidth_tax_ratio"] - 0.35) < 1e-4
    assert abs(actual["opml_slashing_game"]["p_min_approx"] - 5e-6) < 1e-10
    print("Experiment 16-01 verified successfully.")


if __name__ == "__main__":
    main()
