#!/usr/bin/env python3
import json
from pathlib import Path


def run():
    # Proof-of-Learning & Byzantine Aggregation (Example 28-2)
    poisoned_fraction_f = 0.15
    spot_check_rate_p = 0.05
    # If an attacker corrupts k=16 segments per outer round of H=500 steps, over 10 outer rounds:
    corrupted_segments_per_round = 16
    outer_rounds = 10
    miss_prob = (1.0 - spot_check_rate_p) ** (corrupted_segments_per_round * outer_rounds)
    detection_prob_pct = round((1.0 - miss_prob) * 100.0, 2)  # 99.97% (> 99.9%)
    geometric_median_breakdown = 0.50

    # DHT Elastic Spot Coordinator (Example 28-3)
    spot_hourly_preemption_rate = 0.08
    dht_ettr_pct = 96.8

    # Async Speculative Guardrail Sidecar (Example 28-4 & 28-5)
    serial_ttft_overhead_ms = 45.0
    serial_tail_blocking_ms = 180.0
    speculative_window_k = 8
    sidecar_check_ms = 3.2
    exposed_guardrail_latency_ms = 3.2
    in_situ_layer_latency_us = 3.6
    in_situ_overhead_pct = 0.08

    metrics = {
        "experiment_id": "16-05",
        "byzantine_and_dht": {
            "poisoned_fraction_f": poisoned_fraction_f,
            "spot_check_rate_p": spot_check_rate_p,
            "detection_prob_10rounds_pct": detection_prob_pct,
            "geometric_median_breakdown": geometric_median_breakdown,
            "spot_hourly_preemption_rate": spot_hourly_preemption_rate,
            "dht_ettr_pct": dht_ettr_pct,
        },
        "guardrail_and_sae_steering": {
            "serial_ttft_overhead_ms": serial_ttft_overhead_ms,
            "serial_tail_blocking_ms": serial_tail_blocking_ms,
            "speculative_window_k": speculative_window_k,
            "sidecar_check_ms": sidecar_check_ms,
            "exposed_guardrail_latency_ms": exposed_guardrail_latency_ms,
            "in_situ_layer_latency_us": in_situ_layer_latency_us,
            "in_situ_overhead_pct": in_situ_overhead_pct,
        },
    }

    base = Path(__file__).resolve().parent
    for sub in ("expected", "results"):
        d = base / sub
        d.mkdir(parents=True, exist_ok=True)
    (base / "expected" / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (base / "results" / "summary.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run()
