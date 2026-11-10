#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 18-02."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH, RESULTS_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    results = json.loads(Path(RESULTS_PATH).read_text())
    assert actual == expected == results, f"Mismatch:\nactual={actual}\nexpected={expected}\nresults={results}"
    assert actual["firecracker_cow_and_mcts_tree"]["cow_fork_ms"] == 12.0
    assert actual["firecracker_cow_and_mcts_tree"]["fork_speedup_x"] == 125.0
    assert actual["firecracker_cow_and_mcts_tree"]["cow_tree_ram_mib"] == 896.0
    assert actual["firecracker_cow_and_mcts_tree"]["ram_saving_pct"] == 89.06
    assert actual["firecracker_cow_and_mcts_tree"]["total_sync_rollback_ms"] == 8.5
    assert actual["locked_case_i19_telemetry_drain"]["time_to_overflow_sec"] == 81.92
    assert actual["locked_case_i19_telemetry_drain"]["accumulated_mib"] == 46.875
    assert actual["locked_case_i19_telemetry_drain"]["drain_time_sec"] == 76.92
    assert actual["mcp_dynamic_tool_rag"]["schema_reduction_pct"] == 99.2
    assert actual["mcp_dynamic_tool_rag"]["schema_compression_x"] == 125.0
    print("Experiment 18-02 verified successfully.")


if __name__ == "__main__":
    main()
