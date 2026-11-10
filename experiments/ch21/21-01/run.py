#!/usr/bin/env python3
"""Experiment 21-01: Trillion-parameter DLRM three-tier embedding cache and TorchRec All-to-All overlap."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"
RESULTS_PATH = HERE / "results" / "summary.json"


def compute_metrics() -> dict:
    # 1. 10.0 TB Trillion-Parameter DLRM Three-Tier Embedding Cache (Ex 25-1)
    num_rows = 20_000_000_000  # 2 x 10^10 feature IDs
    embed_dim = 128
    bytes_per_row = 500  # 256B FP16 weight + 230B quantized optimizer state + 14B metadata
    total_table_tb = round((num_rows * bytes_per_row) / 1e12, 1)  # 10.0 TB
    zipf_alpha = 1.12
    temporal_drift = 0.038

    hbm_capacity_gb = 320.0
    cxl_capacity_tb = 2.0
    ssd_capacity_tb = 10.0

    p_hbm = 0.820
    p_cxl = 0.155
    p_ssd = 0.025
    t_hbm_us = 0.35
    t_cxl_us = 1.80
    t_ssd_us = 65.00

    weighted_latency_us = round(p_hbm * t_hbm_us + p_cxl * t_cxl_us + p_ssd * t_ssd_us, 2)  # 2.19 us
    pure_ssd_speedup_x = round(t_ssd_us / weighted_latency_us, 1)  # 29.7x

    # 2. 64-GPU TorchRec Sparse All-to-All Sharding & Dense Overlap (Ex 25-2)
    num_gpus = 64
    batch_per_gpu = 2048
    pooling_factor_l = 320
    fp16_bytes = 2
    per_gpu_send_mb = round(
        (batch_per_gpu * pooling_factor_l * embed_dim * fp16_bytes * (num_gpus - 1) / num_gpus) / 1e6, 2
    )  # 165.15 MB
    naive_skew_factor = 2.85
    balanced_skew_factor = 1.06
    bw_per_gpu_gbs = 114.5
    naive_a2a_ms = round((per_gpu_send_mb * naive_skew_factor) / bw_per_gpu_gbs, 2)  # 4.11 ms
    balanced_a2a_ms = round((per_gpu_send_mb * balanced_skew_factor) / bw_per_gpu_gbs, 2)  # 1.53 ms
    bottom_mlp_compute_ms = 1.85
    exposed_naive_comm_ms = round(max(0.0, naive_a2a_ms - bottom_mlp_compute_ms), 2)  # 2.26 ms
    exposed_balanced_comm_ms = round(max(0.0, balanced_a2a_ms - bottom_mlp_compute_ms), 2)  # 0.00 ms

    return {
        "experiment_id": "21-01",
        "three_tier_embedding_cache": {
            "num_rows": num_rows,
            "embed_dim": embed_dim,
            "bytes_per_row": bytes_per_row,
            "total_table_tb": total_table_tb,
            "zipf_alpha": zipf_alpha,
            "temporal_drift": temporal_drift,
            "hbm_capacity_gb": hbm_capacity_gb,
            "cxl_capacity_tb": cxl_capacity_tb,
            "ssd_capacity_tb": ssd_capacity_tb,
            "p_hbm": p_hbm,
            "p_cxl": p_cxl,
            "p_ssd": p_ssd,
            "weighted_latency_us": weighted_latency_us,
            "pure_ssd_speedup_x": pure_ssd_speedup_x,
        },
        "torchrec_all_to_all_overlap": {
            "num_gpus": num_gpus,
            "per_gpu_send_mb": per_gpu_send_mb,
            "naive_skew_factor": naive_skew_factor,
            "balanced_skew_factor": balanced_skew_factor,
            "naive_a2a_ms": naive_a2a_ms,
            "balanced_a2a_ms": balanced_a2a_ms,
            "bottom_mlp_compute_ms": bottom_mlp_compute_ms,
            "exposed_naive_comm_ms": exposed_naive_comm_ms,
            "exposed_balanced_comm_ms": exposed_balanced_comm_ms,
        },
    }


def main() -> None:
    metrics = compute_metrics()
    payload = json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    EXPECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPECTED_PATH.write_text(payload)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(payload)
    print(payload.strip())


if __name__ == "__main__":
    main()
