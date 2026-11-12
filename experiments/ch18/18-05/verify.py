#!/usr/bin/env python3
"""Verifier for Experiment 18-05."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"
SUMMARY_PATH = HERE / "results" / "summary.json"


def main() -> None:
    if not EXPECTED_PATH.exists() or not SUMMARY_PATH.exists():
        print("Missing expected/metrics.json or results/summary.json; run run.py first.", file=sys.stderr)
        sys.exit(1)
    expected = json.loads(EXPECTED_PATH.read_text())
    actual = json.loads(SUMMARY_PATH.read_text())
    assert expected == actual, f"Mismatch:\nexpected={expected}\nactual={actual}"

    dag = actual["multi_agent_dag_tail"]
    assert dag["k16_required_branch_percentile"] == 99.937
    assert dag["k12_barrier_join_p99_seconds"] == 19.8
    assert dag["k12_quorum_slack_p99_seconds"] == 7.4
    assert dag["tail_latency_reduction_pct"] == 62.6

    spec = actual["speculative_tool_and_saga"]
    assert spec["speculative_tool_saved_latency_ms"] == 210.7
    assert spec["speculative_tool_wasted_query_pct"] == 16.3
    assert spec["saga_consistent"] is True
    assert spec["mutating_speculation_blocked"] is True
    print("Experiment 18-05 verified: all locked metrics match.")


if __name__ == "__main__":
    main()
