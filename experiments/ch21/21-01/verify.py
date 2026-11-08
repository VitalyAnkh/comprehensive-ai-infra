#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 21-01."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH, RESULTS_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    results = json.loads(Path(RESULTS_PATH).read_text())
    assert actual == expected == results, f"Mismatch:\nactual={actual}\nexpected={expected}\nresults={results}"
    assert actual["agentic_worm_40turns"]["cumulative_prefill_tokens"] == 1640000
    assert actual["agentic_worm_40turns"]["worm_reuse_ratio_pct"] == 95.12
    assert actual["agentic_worm_40turns"]["prefill_to_decode_ratio"] == 205.0
    assert actual["tool_wait_phase_diagram"]["kv_size_gib"] == 9.0
    assert actual["tool_wait_phase_diagram"]["swap_in_ms"] == 187.5
    assert actual["tool_wait_phase_diagram"]["crossover_wait_sec"] == 2.16
    assert actual["tool_wait_phase_diagram"]["hbm_cost_saving_at_15s_pct"] == 85.6
    assert actual["kv_global_router_and_cachegen"]["router_ttft_speedup_x"] == 3.0
    assert actual["kv_global_router_and_cachegen"]["cachegen_compression_x"] == 4.21
    assert actual["kv_global_router_and_cachegen"]["cachegen_transfer_ms"] == 385.9
    print("Experiment 21-01 verified successfully.")


if __name__ == "__main__":
    main()
