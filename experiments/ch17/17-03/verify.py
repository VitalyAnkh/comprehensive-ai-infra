#!/usr/bin/env python3
"""Verifier for Experiment 17-03."""
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

    kv = actual["rolling_kv_pool"]
    assert kv["active_chunks"] == 6
    assert kv["active_kv_tokens"] == 21600
    assert kv["bytes_per_tok_28l"] == 229376
    assert kv["hybrid_8l_rolling_kv_gib"] == 1.3184
    assert kv["full_28l_rolling_kv_gib"] == 4.6143
    assert kv["unbounded_10min_8l_gib"] == 878.91
    assert kv["kv_compression_ratio_10min"] == 666.67

    st = actual["pab_teacache_streaming"]
    assert st["uncached_dit_chunk_ms"] == 168.0
    assert st["uncached_total_chunk_ms"] == 186.0
    assert st["uncached_frame_ms"] == 62.0
    assert st["uncached_fps"] == 16.13
    assert st["pab_teacache_dit_chunk_ms"] == 96.0
    assert st["opt_total_chunk_ms"] == 114.0
    assert st["opt_frame_ms"] == 38.0
    assert st["streaming_fps"] == 26.32
    assert st["fps_speedup"] == 1.63
    assert st["meets_20fps_sla"] is True
    print("Experiment 17-03 verified: all locked metrics match.")


if __name__ == "__main__":
    main()
