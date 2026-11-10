#!/usr/bin/env python3
"""Experiment 17-02: DistRS 216,000-token sunk-work cutoff, geo-distributed async RL 14,745.6x compression, and PRM CoW tree search."""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"
RESULTS_PATH = HERE / "results" / "baseline.json"


def compute_metrics() -> dict:
    # 1. DistRS heavy-tail conditional remaining time & 216,000-token sunk-work boundary (Example 18-8 & Exercise 18-10)
    tasks_bimodal = [1.0] * 9 + [100.0]
    unconditional_mean_s = round(sum(tasks_bimodal) / len(tasks_bimodal), 2)  # 10.9 s
    elapsed_s = 10.0
    naive_rem_10s = round(max(unconditional_mean_s - elapsed_s, 0.0), 2)  # 0.9 s
    cond_rem_10s = 90.0
    estimation_error_ratio = round(cond_rem_10s / naive_rem_10s, 1)  # 100.0x
    sunk_work_tokens = 6 * 60 * 600  # 216,000 tokens

    # Exercise 18-10 trimodal distribution: 80% 2s, 15% 20s, 5% 200s
    trimodal_mean_s = round(0.80 * 2.0 + 0.15 * 20.0 + 0.05 * 200.0, 2)  # 14.6 s
    trimodal_cond_rem_15s = round((0.15 * (20.0 - 15.0) + 0.05 * (200.0 - 15.0)) / (0.15 + 0.05), 2)  # 50.0 s
    trimodal_cond_rem_25s = round(200.0 - 25.0, 2)  # 175.0 s

    # 2. Geo-distributed heterogeneous async RL bandwidth & IS ratio filtering (Example 18-9 & Exercise 18-13)
    num_trajs = 1024
    avg_len = 16384
    total_tokens = num_trajs * avg_len  # 16,777,216 tokens
    kv_bytes_per_token = 147456  # 36-layer GQA
    compact_bytes_per_token = 10  # int32 token_id (4B) + float32 logprob (4B) + uint16 version (2B)

    kv_payload_gb = round((total_tokens * kv_bytes_per_token) / 1e9, 1)  # 2473.9 GB
    compact_payload_mb = round((total_tokens * compact_bytes_per_token) / 1e6, 5)  # 167.77216 MB
    compact_payload_mib = round((total_tokens * compact_bytes_per_token) / (1024 * 1024), 1)  # 160.0 MiB
    uplink_compression_ratio = round(kv_bytes_per_token / compact_bytes_per_token, 1)  # 14745.6x

    wan_bw_gb_s = 1.25  # 10 Gbps = 1.25 GB/s
    kv_upload_sec = round((total_tokens * kv_bytes_per_token / 1e9) / wan_bw_gb_s, 1)  # 1979.1 s
    compact_upload_ms = round(((total_tokens * compact_bytes_per_token / 1e9) / wan_bw_gb_s) * 1000.0, 1)  # 134.2 ms
    fp8_delta_weight_gb = 8.0
    fp8_delta_sync_sec = round(fp8_delta_weight_gb / wan_bw_gb_s, 2)  # 6.4 s

    # Sequence-level geometric mean IS ratio filter check
    sample_logprob_diffs = [0.08, 0.12, -0.04, 0.16]
    seq_geom_is_ratio = round(math.exp(sum(sample_logprob_diffs) / len(sample_logprob_diffs)), 4)  # exp(0.08) = 1.0833

    # 3. PagedAttention CoW tree search memory saving & Speculative ToT latency (Exercises 18-11 & 18-12)
    root_tokens = 2048
    b1, l1 = 8, 256
    b2, l2 = 4, 256
    num_leaves = b1 * b2  # 32
    full_copy_token_slots = num_leaves * (root_tokens + l1 + l2)  # 81,920
    cow_shared_token_slots = root_tokens + b1 * l1 + num_leaves * l2  # 12,288
    cow_memory_saving_pct = round((1.0 - cow_shared_token_slots / full_copy_token_slots) * 100.0, 1)  # 85.0%
    full_copy_gib = round((full_copy_token_slots * kv_bytes_per_token) / (1024**3), 4)  # 11.25 GiB
    cow_shared_gib = round((cow_shared_token_slots * kv_bytes_per_token) / (1024**3), 4)  # 1.6875 GiB

    depth_d = 20
    t_gen_s, t_prm_s = 1.2, 0.8
    hit_rate = 0.80
    serial_tot_sec = round(depth_d * (t_gen_s + t_prm_s), 2)  # 40.0 s
    spec_tot_sec = round(depth_d * (hit_rate * t_gen_s + (1.0 - hit_rate) * (2.0 * t_gen_s)) + t_prm_s, 2)  # 29.6 s

    return {
        "experiment_id": "17-02",
        "distrs_and_heavy_tail": {
            "unconditional_mean_s": unconditional_mean_s,
            "naive_rem_10s": naive_rem_10s,
            "cond_rem_10s": cond_rem_10s,
            "estimation_error_ratio": estimation_error_ratio,
            "sunk_work_tokens": sunk_work_tokens,
            "trimodal_mean_s": trimodal_mean_s,
            "trimodal_cond_rem_15s": trimodal_cond_rem_15s,
            "trimodal_cond_rem_25s": trimodal_cond_rem_25s,
        },
        "geo_distributed_async_rl": {
            "total_tokens": total_tokens,
            "kv_payload_gb": kv_payload_gb,
            "compact_payload_mb": compact_payload_mb,
            "compact_payload_mib": compact_payload_mib,
            "uplink_compression_ratio": uplink_compression_ratio,
            "kv_upload_sec": kv_upload_sec,
            "compact_upload_ms": compact_upload_ms,
            "fp8_delta_sync_sec": fp8_delta_sync_sec,
            "seq_geom_is_ratio": seq_geom_is_ratio,
        },
        "prm_cow_tree_search": {
            "full_copy_token_slots": full_copy_token_slots,
            "cow_shared_token_slots": cow_shared_token_slots,
            "cow_memory_saving_pct": cow_memory_saving_pct,
            "full_copy_gib": full_copy_gib,
            "cow_shared_gib": cow_shared_gib,
            "serial_tot_sec": serial_tot_sec,
            "spec_tot_sec": spec_tot_sec,
        },
    }


def main() -> None:
    metrics = compute_metrics()
    for p in (EXPECTED_PATH, RESULTS_PATH):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
