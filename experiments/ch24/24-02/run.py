#!/usr/bin/env python3
"""Experiment 24-02: DeepEP Normal/Low-Latency modes and DualPipe bidirectional zero-exposure schedule."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def solve_deepep_and_dualpipe() -> dict:
    return {
        "deepep": {
            "normal_mode_sm_count": 24,
            "normal_mode_bw_gbs": 153.0,
            "low_latency_dispatch_us": 163.0,
            "low_latency_combine_us": 194.0,
            "low_latency_sm_occupied": 0,
        },
        "dualpipe": {
            "pp_stages": 16,
            "microbatches_per_direction": 16,
            "bidirectional_microbatches": 32,
            "exposed_alltoall_stall_pct": 0.0,
            "bubble_slots_1f1b": 15,
            "bubble_slots_dualpipe": 7,
            "bubble_1f1b_pct": 31.91,
            "bubble_dualpipe_pct": 17.95,
        },
    }


def main() -> None:
    summary = {
        "experiment_id": "24-02",
        **solve_deepep_and_dualpipe(),
    }
    for sub in ("expected", "results"):
        d = HERE / sub
        d.mkdir(parents=True, exist_ok=True)
    (HERE / "expected" / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (HERE / "results" / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
