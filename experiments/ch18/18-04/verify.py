#!/usr/bin/env python3
"""Verifier for Experiment 18-04."""
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

    mem = actual["bitmask_memory_ledger"]
    assert mem["qwen3_bitmask_uint32_words"] == 4748
    assert mem["qwen3_bitmask_bytes_per_req"] == 18992
    assert mem["qwen3_batch64_bitmask_mib"] == 1.16
    assert mem["deepseek_bitmask_uint32_words"] == 4040
    assert mem["deepseek_bitmask_bytes_per_req"] == 16160
    assert mem["adaptive_merge_verified"] is True

    ov = actual["async_pda_gpu_overlap"]
    assert ov["naive_fsm_throughput_drop_pct"] == 81.9
    assert ov["xgrammar_batch64_cpu_ms"] == 2.1
    assert ov["xgrammar_effective_step_ms"] == 8.5
    assert ov["xgrammar_throughput_loss_pct"] == 0.0
    assert ov["max_zero_overhead_batch_size"] == 259
    print("Experiment 18-04 verified: all locked metrics match.")


if __name__ == "__main__":
    main()
