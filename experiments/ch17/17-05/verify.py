#!/usr/bin/env python3
"""Verifier for Experiment 17-05."""
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

    dist = actual["distillation_32bit_ledger"]
    assert dist["qwen_200b_full_logits_bf16_pb"] == 60.7744
    assert dist["deepseek_200b_full_logits_bf16_pb"] == 51.712
    assert dist["top128_unpacked_6b_tb"] == 153.6
    assert dist["top128_bitpacked_32bit_tb"] == 102.4
    assert dist["top128_compression_vs_full_x"] == 593.5
    assert dist["codec_token_id_lossless"] is True
    assert dist["codec_max_abs_err"] <= 0.25
    assert dist["offline_vs_online_teacher_crossover_epochs"] == 1.88
    assert dist["minitron_tensor_core_128_aligned"] is True

    arl = actual["bounded_staleness_agentic_rl"]
    assert arl["max_staleness_version_lag"] == 3
    assert arl["agentic_rl_sync_gpu_idle_pct"] == 68.0
    assert arl["agentic_rl_async_bounded_speedup_x"] == 2.64
    print("Experiment 17-05 verified: all locked metrics match.")


if __name__ == "__main__":
    main()
