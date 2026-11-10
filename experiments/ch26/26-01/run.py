#!/usr/bin/env python3
"""Experiment 26-01: Block Diffusion LLM Roofline phase transition and dynamic confidence remasking."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"
RESULTS_PATH = HERE / "results" / "summary.json"


def compute_metrics() -> dict:
    # 1. 32B FP8 Block Diffusion vs Autoregressive LLM (Ex 26-1)
    model_params_b = 32.0
    weight_bytes_gb = 32.0
    hbm_bw_gbs = 3350.0
    gen_tokens = 512

    ar_steps = 512
    ar_step_ms = 10.00  # 32.0 / 3.350 + 0.45 ms overhead
    ar_total_sec = round((ar_steps * ar_step_ms) / 1000.0, 2)  # 5.12 s
    ar_tok_per_sec = round(gen_tokens / ar_total_sec, 1)  # 100.0 tok/s

    block_size_k = 64
    denoise_steps_t = 10
    num_blocks = gen_tokens // block_size_k  # 8 blocks
    dllm_forwards = num_blocks * denoise_steps_t  # 80 forwards
    weight_read_reduction_x = round(ar_steps / dllm_forwards, 2)  # 6.40x
    dllm_step_ms = 10.80
    dllm_total_sec = round((dllm_forwards * dllm_step_ms) / 1000.0, 3)  # 0.864 s
    dllm_tok_per_sec = round(gen_tokens / dllm_total_sec, 1)  # 592.6 tok/s
    speedup_vs_ar_x = round(dllm_tok_per_sec / ar_tok_per_sec, 2)  # 5.93x
    dllm_8b_tok_per_sec = 1420.0

    # 2. Dynamic Confidence Threshold Remasking (Ex 26-2)
    confidence_gamma = 0.92
    step_saving_pct = 35.0
    avg_steps_per_block = round(denoise_steps_t * (1.0 - step_saving_pct / 100.0), 1)  # 6.5 steps
    early_stop_forwards = int(round(num_blocks * avg_steps_per_block))  # 52 forwards
    early_stop_total_sec = round((early_stop_forwards * dllm_step_ms) / 1000.0, 4)  # 0.5616 s
    early_stop_tok_per_sec = round(gen_tokens / early_stop_total_sec, 1)  # 911.7 tok/s
    early_stop_speedup_x = round(early_stop_tok_per_sec / ar_tok_per_sec, 2)  # 9.12x

    return {
        "experiment_id": "26-01",
        "block_diffusion_roofline": {
            "model_params_b": model_params_b,
            "gen_tokens": gen_tokens,
            "ar_steps": ar_steps,
            "ar_total_sec": ar_total_sec,
            "ar_tok_per_sec": ar_tok_per_sec,
            "block_size_k": block_size_k,
            "denoise_steps_t": denoise_steps_t,
            "dllm_forwards": dllm_forwards,
            "weight_read_reduction_x": weight_read_reduction_x,
            "dllm_total_sec": dllm_total_sec,
            "dllm_tok_per_sec": dllm_tok_per_sec,
            "speedup_vs_ar_x": speedup_vs_ar_x,
            "dllm_8b_tok_per_sec": dllm_8b_tok_per_sec,
        },
        "confidence_remasking_early_exit": {
            "confidence_gamma": confidence_gamma,
            "step_saving_pct": step_saving_pct,
            "avg_steps_per_block": avg_steps_per_block,
            "early_stop_forwards": early_stop_forwards,
            "early_stop_tok_per_sec": early_stop_tok_per_sec,
            "early_stop_speedup_x": early_stop_speedup_x,
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
