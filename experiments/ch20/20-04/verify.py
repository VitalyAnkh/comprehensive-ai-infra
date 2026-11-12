#!/usr/bin/env python3
"""Verifier for Experiment 20-04."""
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
    expected = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
    actual = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    assert expected == actual, f"Mismatch:\nexpected={expected}\nactual={actual}"

    sp = actual["sensor_pipeline_ex20_8"]
    assert sp["nv12_bytes_per_frame"] == 3_456_000
    assert sp["raw_camera_ingress_gbps"] == 1.244
    assert sp["ros2_serialized_dram_gbps"] == 11.20
    assert sp["dmabuf_iceoryx_dram_gbps"] == 2.49
    assert sp["dram_bandwidth_reduction_x"] == 4.50
    assert sp["ros2_sensor_to_vit_p99_ms"] == 18.60
    assert sp["ros2_sensor_p999_jitter_ms"] == 9.40
    assert sp["dmabuf_sensor_to_vit_p99_ms"] == 0.85
    assert sp["dmabuf_sensor_p999_jitter_ms"] == 0.14
    assert sp["sensor_latency_speedup_x"] == 21.88
    assert sp["meets_sub_150us_jitter_sla"] is True

    ex15 = actual["exercise_20_15_4k_8cam"]
    assert ex15["frame_4k_bytes"] == 12_441_600
    assert ex15["raw_4k_ingress_gbps"] == 2.986
    assert ex15["ros2_4k_dram_gbps"] == 26.87
    assert ex15["dmabuf_4k_dram_gbps"] == 5.97
    assert ex15["saved_4k_dram_gbps"] == 20.90
    assert ex15["saved_blind_drift_mm"] == 51.35

    ex16 = actual["exercise_20_16_iceoryx_pool"]
    assert ex16["min_slots_per_camera"] == 6
    assert ex16["total_pool_slots"] == 36
    assert ex16["total_pool_bytes"] == 124_416_000
    assert ex16["total_pool_mib"] == 118.65
    print("Experiment 20-04 verified: all locked metrics match.")


if __name__ == "__main__":
    main()
