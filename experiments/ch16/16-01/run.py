#!/usr/bin/env python3
"""Experiment 16-01: PipeLLM confidential pipeline, MoE traffic padding, and OPML slashing game."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"


def compute_metrics() -> dict:
    # 1. PipeLLM 5-stage triple-buffering pipeline (Exercise 16-2)
    chunk_mib = 64
    num_chunks = 20
    t_h2d, t_dec, t_comp, t_enc, t_d2h = 2.0, 0.2, 2.5, 0.2, 2.0
    t_serial_ms = round(num_chunks * (t_h2d + t_dec + t_comp + t_enc + t_d2h), 2)
    t_gpu_stage_ms = round(t_dec + t_comp + t_enc, 2)
    t_step_ms = max(t_h2d, t_gpu_stage_ms, t_d2h)
    t_pipe_ms = round(num_chunks * t_step_ms + t_h2d + t_d2h, 2)
    speedup_ratio = round(t_serial_ms / t_pipe_ms, 4)
    triple_buffer_mib = 3 * chunk_mib

    # 2. Speculative AES-CTR break-even hit rate (Exercise 16-3)
    dt_hit_ms, dt_miss_ms = 2.4, 1.6
    p_breakeven = round(dt_miss_ms / (dt_hit_ms + dt_miss_ms), 4)

    # 3. MoE traffic side-channel constant-rate dummy token padding (Exercise 16-11)
    ep_ranks = 16
    alpha_hot = 0.35
    mu_tokens_per_rank = 400
    baseline_total_tokens = ep_ranks * mu_tokens_per_rank
    padded_tokens_per_rank = int(round(mu_tokens_per_rank * (1.0 + alpha_hot)))
    padded_total_tokens = ep_ranks * padded_tokens_per_rank
    bandwidth_tax_ratio = round((padded_total_tokens - baseline_total_tokens) / baseline_total_tokens, 4)

    # 4. OPML optimistic verification slashing threshold (Exercise 16-12)
    delta_c_usd = 0.005
    slash_usd = 1000.0
    p_min_exact = delta_c_usd / (slash_usd + delta_c_usd)
    p_min_approx = delta_c_usd / slash_usd

    return {
        "experiment_id": "16-01",
        "pipellm": {
            "t_serial_ms": t_serial_ms,
            "t_pipe_ms": t_pipe_ms,
            "speedup_ratio": speedup_ratio,
            "triple_buffer_mib": triple_buffer_mib,
            "speculative_p_breakeven": p_breakeven,
        },
        "moe_traffic_padding": {
            "ep_ranks": ep_ranks,
            "alpha_hot": alpha_hot,
            "baseline_total_tokens": baseline_total_tokens,
            "padded_total_tokens": padded_total_tokens,
            "bandwidth_tax_ratio": bandwidth_tax_ratio,
        },
        "opml_slashing_game": {
            "delta_c_usd": delta_c_usd,
            "slash_usd": slash_usd,
            "p_min_exact": round(p_min_exact, 10),
            "p_min_approx": round(p_min_approx, 10),
        },
    }


def main() -> None:
    metrics = compute_metrics()
    EXPECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPECTED_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
