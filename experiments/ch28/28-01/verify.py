#!/usr/bin/env python3
import json
from pathlib import Path


def verify():
    base = Path(__file__).resolve().parent
    exp = json.loads((base / "expected" / "metrics.json").read_text(encoding="utf-8"))
    res = json.loads((base / "results" / "summary.json").read_text(encoding="utf-8"))
    assert exp == res, "Mismatch between expected and results"
    assert res["streaming_diloco"]["std_wan_bw_gbps"] == 700.0
    assert res["streaming_diloco"]["streaming_wan_bw_mbps"] == 350.0
    assert res["streaming_diloco"]["bw_compression_vs_allreduce_x"] == 2000.0
    assert res["streaming_diloco"]["effective_compute_utilization_pct"] == 99.4
    print("Experiment 28-01 verified successfully.")


if __name__ == "__main__":
    verify()
