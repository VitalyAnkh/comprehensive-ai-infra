#!/usr/bin/env python3
import json
from pathlib import Path


def verify():
    base = Path(__file__).resolve().parent
    exp = json.loads((base / "expected" / "metrics.json").read_text(encoding="utf-8"))
    res = json.loads((base / "results" / "summary.json").read_text(encoding="utf-8"))
    assert exp == res, "Mismatch between expected and results"
    assert res["byzantine_and_dht"]["detection_prob_10rounds_pct"] > 99.9
    assert res["byzantine_and_dht"]["dht_ettr_pct"] == 96.8
    assert res["guardrail_and_sae_steering"]["exposed_guardrail_latency_ms"] < 3.5
    assert res["guardrail_and_sae_steering"]["in_situ_overhead_pct"] == 0.08
    print("Experiment 28-02 verified successfully.")


if __name__ == "__main__":
    verify()
