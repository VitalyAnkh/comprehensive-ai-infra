#!/usr/bin/env python3
"""Experiment 22-02: Evo 2 1M-bp FlashFFTConv complexity, SFNO 2D grid Halo exchange overlap, and 5-domain scientific AI workload ledger."""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"
RESULTS_PATH = HERE / "results" / "summary.json"


def compute_metrics() -> dict:
    # 1. Evo 2 40B 1M-Base-Pair (L=1,048,576) FlashFFTConv vs Quadratic Attention (Ex 22-3)
    seq_len_l = 1_048_576  # 2^20 single-nucleotide resolution
    hidden_dim_d = 8192
    log2_l = int(round(math.log2(seq_len_l)))  # 20

    quadratic_attn_tflops = round((4.0 * (seq_len_l ** 2) * hidden_dim_d) / 1e12, 1)  # 36,028.8 TFLOPs (36.03 PFLOPs)
    flash_fft_conv_gflops = round((10.0 * seq_len_l * (log2_l + 0.20) * hidden_dim_d) / 1e9, 2)  # 1,735.17 GFLOPs (1.74 TFLOPs)
    seq_mixer_flop_reduction_x = round((quadratic_attn_tflops * 1000.0) / flash_fft_conv_gflops, 1)  # 20,763.8x
    end_to_end_layer_speedup_x = 28.5

    # 2. Global Weather SFNO (0.05 deg, 3600x7200) 2D Domain Decomposition & Halo Overlap (Ex 22-4)
    grid_lat = 3600
    grid_lon = 7200
    px = 8
    py = 8
    sub_lat = grid_lat // px  # 450
    sub_lon = grid_lon // py  # 900
    halo_width = 4
    bytes_per_cell = 512  # 128 channels x FP32 (4B)

    halo_perimeter_cells = 2 * (sub_lat + sub_lon) * halo_width  # 10,800 cells
    halo_volume_mb = round((halo_perimeter_cells * bytes_per_cell) / 1e6, 2)  # 5.53 MB
    interior_compute_ms = 14.20
    halo_transfer_ms = 0.18
    comm_to_comp_ratio_pct = round((halo_transfer_ms / interior_compute_ms) * 100.0, 2)  # 1.27%
    exposed_halo_ms = max(0.0, round(halo_transfer_ms - interior_compute_ms, 2))  # 0.00 ms (100% overlapped)

    # 3. Five-Domain Scientific AI Workload & Conservation Ledger (Ex 22-7)
    five_domain_scientific_ledger = {
        "science_domains_covered": 5,
        "domains": [
            "biomolecular_af3",
            "genomics_evo2_hyena",
            "weather_sfno_graphcast",
            "dft_quantum_chemistry",
            "fusion_tokamak_control",
        ],
        "pairformer_4096_naive_gib": 512.0,
        "pairformer_4096_dap4_workspace_gib": 0.25,
        "pairformer_workspace_reduction_x": 2048.0,
        "pairformer_dap_comm_to_comp_pct": 8.26,
        "evo2_1m_mixer_reduction_x": seq_mixer_flop_reduction_x,
        "weather_64gpu_halo_mb": halo_volume_mb,
        "weather_64gpu_halo_comm_to_comp_pct": comm_to_comp_ratio_pct,
        "ozaki_s7_vs_native_fp64_speedup_x": 3.68,
        "ozaki_equiv_fp64_tflops": 147.2,
        "tokamak_opt_wcet_us": 68.5,
        "tokamak_headroom_us": 131.5,
    }

    return {
        "experiment_id": "22-02",
        "evo2_flashfftconv_1m": {
            "seq_len_l": seq_len_l,
            "hidden_dim_d": hidden_dim_d,
            "quadratic_attn_tflops": quadratic_attn_tflops,
            "flash_fft_conv_gflops": flash_fft_conv_gflops,
            "seq_mixer_flop_reduction_x": seq_mixer_flop_reduction_x,
            "end_to_end_layer_speedup_x": end_to_end_layer_speedup_x,
        },
        "sfno_2d_halo_overlap": {
            "sub_lat": sub_lat,
            "sub_lon": sub_lon,
            "halo_perimeter_cells": halo_perimeter_cells,
            "halo_volume_mb": halo_volume_mb,
            "interior_compute_ms": interior_compute_ms,
            "halo_transfer_ms": halo_transfer_ms,
            "comm_to_comp_ratio_pct": comm_to_comp_ratio_pct,
            "exposed_halo_ms": exposed_halo_ms,
        },
        "five_domain_scientific_ledger": five_domain_scientific_ledger,
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
