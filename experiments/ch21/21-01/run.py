#!/usr/bin/env python3
"""Experiment 21-01: 40-turn Agentic WORM loop, Tool-Wait phase diagram, Four-Tier KV Router, and CacheGen streaming."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"
RESULTS_PATH = HERE / "results" / "summary.json"


def compute_metrics() -> dict:
    # 1. 40-turn Agentic WORM (Write-Once-Read-Many) context loop (Ex 21-1)
    turns_k = 40
    l0_tokens = 2000
    act_tokens = 200
    obs_tokens = 1800
    delta_l = act_tokens + obs_tokens  # 2,000 tokens per turn

    turn_lengths = [l0_tokens + (k - 1) * delta_l for k in range(1, turns_k + 1)]
    final_turn_tokens = turn_lengths[-1]  # 80,000 tokens
    cumulative_prefill_tokens = sum(turn_lengths)  # 1,640,000 tokens
    unique_written_tokens = final_turn_tokens  # 80,000 tokens
    reread_worm_tokens = cumulative_prefill_tokens - unique_written_tokens  # 1,560,000 tokens
    worm_reuse_ratio_pct = round(reread_worm_tokens / cumulative_prefill_tokens * 100.0, 2)  # 95.12%
    total_decode_tokens = turns_k * act_tokens  # 8,000 tokens
    prefill_to_decode_ratio = round(cumulative_prefill_tokens / total_decode_tokens, 1)  # 205.0

    # 2. Tool-Wait Hiatus three-regime crossover at L = 65,536 tokens on Qwen3-8B (144 KiB/tok) (Ex 21-1)
    prefix_tokens = 65536
    kv_kib_per_token = 144.0
    kv_size_gib = round(prefix_tokens * kv_kib_per_token / (1024.0 * 1024.0), 2)  # 9.00 GiB
    prefill_rate_tok_s = 131072.0
    recompute_ms = round(prefix_tokens / prefill_rate_tok_s * 1000.0, 2)  # 500.0 ms
    pcie_bw_gib_s = 48.0
    swap_in_ms = round(kv_size_gib / pcie_bw_gib_s * 1000.0, 2)  # 187.5 ms
    roundtrip_swap_ms = round(2.0 * swap_in_ms, 2)  # 375.0 ms
    crossover_wait_sec = 2.16
    tool_wait_sec = 15.0
    hbm_cost_saving_at_15s_pct = round((1.0 - crossover_wait_sec / tool_wait_sec) * 100.0, 2)  # 85.60%
    ttft_speedup_vs_recompute_x = round(recompute_ms / swap_in_ms, 2)  # 2.67x

    # 3. Four-Tier KV-Aware Global Radix Router (Dynamo / HiCache) (Ex 21-2)
    # Candidate A: queue 120 ms, 90% HBM hit -> 50 ms prefill = 170 ms
    # Candidate B (naive least-loaded): queue 10 ms, 0% hit -> 500 ms prefill = 510 ms
    # Candidate C (remote RDMA hit): queue 20 ms, 80% RDMA hit (200 ms) + 20% miss (100 ms) = 320 ms
    ttft_node_a_ms = 170.0
    ttft_node_b_ms = 510.0
    ttft_node_c_ms = 320.0
    router_ttft_speedup_x = round(ttft_node_b_ms / ttft_node_a_ms, 2)  # 3.00x

    # 4. CacheGen Anchor-Delta Entropy Streaming over 50 Gbps (5.81 GiB/s usable) cross-rack link (Ex 21-2)
    orig_bits = 16.0
    cachegen_bits = 3.80
    cachegen_compression_x = round(orig_bits / cachegen_bits, 2)  # 4.21x
    compressed_kv_gib = round(kv_size_gib * (cachegen_bits / orig_bits), 4)  # 2.1375 GiB
    net_bw_gib_s = 5.81
    decode_overhead_ms = 18.0
    raw_transfer_ms = round(kv_size_gib / net_bw_gib_s * 1000.0, 1)  # 1549.1 ms
    cachegen_transfer_ms = round(compressed_kv_gib / net_bw_gib_s * 1000.0 + decode_overhead_ms, 1)  # 385.9 ms
    cachegen_speedup_vs_raw_x = round(raw_transfer_ms / cachegen_transfer_ms, 2)  # 4.01x
    critical_raw_bw_gib_s = round(kv_size_gib / (recompute_ms / 1000.0), 2)  # 18.00 GiB/s

    return {
        "experiment_id": "21-01",
        "agentic_worm_40turns": {
            "turns_k": turns_k,
            "final_turn_tokens": final_turn_tokens,
            "cumulative_prefill_tokens": cumulative_prefill_tokens,
            "unique_written_tokens": unique_written_tokens,
            "reread_worm_tokens": reread_worm_tokens,
            "worm_reuse_ratio_pct": worm_reuse_ratio_pct,
            "total_decode_tokens": total_decode_tokens,
            "prefill_to_decode_ratio": prefill_to_decode_ratio,
        },
        "tool_wait_phase_diagram": {
            "prefix_tokens": prefix_tokens,
            "kv_size_gib": kv_size_gib,
            "recompute_ms": recompute_ms,
            "swap_in_ms": swap_in_ms,
            "roundtrip_swap_ms": roundtrip_swap_ms,
            "crossover_wait_sec": crossover_wait_sec,
            "hbm_cost_saving_at_15s_pct": hbm_cost_saving_at_15s_pct,
            "ttft_speedup_vs_recompute_x": ttft_speedup_vs_recompute_x,
        },
        "kv_global_router_and_cachegen": {
            "ttft_node_a_ms": ttft_node_a_ms,
            "ttft_node_b_ms": ttft_node_b_ms,
            "ttft_node_c_ms": ttft_node_c_ms,
            "router_ttft_speedup_x": router_ttft_speedup_x,
            "cachegen_compression_x": cachegen_compression_x,
            "compressed_kv_gib": compressed_kv_gib,
            "raw_transfer_ms": raw_transfer_ms,
            "cachegen_transfer_ms": cachegen_transfer_ms,
            "cachegen_speedup_vs_raw_x": cachegen_speedup_vs_raw_x,
            "critical_raw_bw_gib_s": critical_raw_bw_gib_s,
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
