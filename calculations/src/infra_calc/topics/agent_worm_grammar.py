"""Ch18: Agent WORM KV residency, Dynamo/CacheGen WAN break-even, Firecracker CoW, XGrammar, and DAG tail."""
from __future__ import annotations

import math
from ..sources import model_config


def _norm_inv(p: float) -> float:
    """Acklam rational approximation for standard normal inverse CDF (error < 1.15e-9)."""
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e01, 2.445134137142996e00,
         3.754408661907416e00)
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def calculate(
    worm_turns: int = 40,
    worm_l0_tokens: int = 2000,
    worm_delta_tokens: int = 2000,
    batch_size: int = 64,
    gpu_decode_step_ms: float = 8.5,
    naive_fsm_cpu_ms: float = 38.4,
    xgrammar_per_req_us: float = 32.8125,
    fanout_k: int = 12,
    single_branch_p99_s: float = 6.2,
    speculative_hit_prob: float = 0.86,
    tool_query_latency_ms: float = 245.0,
) -> dict:
    qwen8 = model_config("qwen3-8b")
    dsv3 = model_config("deepseek-v3")
    v_qwen = int(qwen8["vocab_size"])  # 151,936
    v_ds = int(dsv3["vocab_size"])     # 129,280

    # 1. 算例 18-1: 40-turn coding agent WORM prefix reuse (L_0=2000, dL=2000) & 64K PCIe swap vs recompute
    unique_tokens = worm_l0_tokens + (worm_turns - 1) * worm_delta_tokens         # 80,000 tokens
    cum_prefill_tokens = worm_turns * (worm_l0_tokens + unique_tokens) // 2       # 1,640,000 tokens
    reread_tokens = cum_prefill_tokens - unique_tokens                            # 1,560,000 tokens
    worm_prefix_reuse_pct = round(reread_tokens / cum_prefill_tokens * 100.0, 2)  # 95.12%
    prefill_to_decode_ratio = round(cum_prefill_tokens / (worm_turns * 200.0), 1) # 205.0:1

    kv_64k_gib = round((65_536 * 144.0) / (1024.0**2), 2)                         # 9.00 GiB
    pcie_swap_in_ms = round((kv_64k_gib / 48.0) * 1000.0, 1)                      # 187.5 ms
    recompute_64k_ms = 500.0
    swap_vs_recompute_speedup_x = round(recompute_64k_ms / pcie_swap_in_ms, 2)    # 2.67x
    hbm_rent_saving_15s_pct = round((1.0 - 2.16 / 15.0) * 100.0, 2)               # 85.60%

    # 2. 算例 18-2: Dynamo KV-aware routing (170 ms vs 510 ms) & CacheGen WAN compression break-even
    dynamo_kv_route_ttft_ms = 120.0 + 50.0                                        # 170.0 ms
    least_queue_route_ttft_ms = 10.0 + 500.0                                      # 510.0 ms
    dynamo_routing_speedup_x = round(least_queue_route_ttft_ms / dynamo_kv_route_ttft_ms, 2)  # 3.00x

    wan_eff_gibps = 5.81                                                          # 50 Gbps Ethernet net payload
    raw_wan_transfer_ms = round((kv_64k_gib / wan_eff_gibps) * 1000.0, 1)         # 1,549.1 ms (> 500 ms recompute)
    cachegen_bits = 3.80
    cachegen_compression_x = round(16.0 / cachegen_bits, 2)                       # 4.21x
    cachegen_payload_gib = round(kv_64k_gib * (cachegen_bits / 16.0), 4)          # 2.1375 GiB
    cachegen_decomp_ms = 18.0
    cachegen_total_ms = round((cachegen_payload_gib / wan_eff_gibps) * 1000.0 + cachegen_decomp_ms, 1)  # 385.9 ms
    cachegen_vs_raw_speedup_x = round(raw_wan_transfer_ms / cachegen_total_ms, 2) # 4.01x
    cachegen_breakeven_bw_gibps = round(cachegen_payload_gib / ((recompute_64k_ms - cachegen_decomp_ms) / 1000.0), 2)  # 4.43 GiB/s

    # 3. 算例 18-3: Firecracker 12 ms snapshot fork, 16-branch CoW memory saving (89.06%), & 8.50 ms sync rollback
    firecracker_fork_ms = round(3.2 + 8.8, 1)                                     # 12.0 ms
    firecracker_vs_cold_speedup_x = round(1500.0 / firecracker_fork_ms, 1)        # 125.0x
    cow_16branch_mib = round(512.0 + 16 * 24.0, 1)                                # 896.0 MiB vs 8,192.0 MiB
    cow_memory_saving_pct = round((1.0 - cow_16branch_mib / (16 * 512.0)) * 100.0, 2)  # 89.06%
    sync_rollback_ms = round(0.15 + 8.35, 2)                                      # 8.50 ms

    # 4. Ch18.5 算例 18-8: XGrammar bitmask sizing & async CPU-GPU overlap
    qwen_uint32_words = math.ceil(v_qwen / 32)                                    # 4,748
    qwen_mask_bytes_per_req = qwen_uint32_words * 4                               # 18,992 B (18.55 KiB)
    qwen_batch_mask_mib = round((batch_size * qwen_mask_bytes_per_req) / (1024**2), 2)  # 1.16 MiB
    ds_uint32_words = math.ceil(v_ds / 32)                                        # 4,040
    ds_mask_bytes_per_req = ds_uint32_words * 4                                   # 16,160 B

    naive_step_ms = round(naive_fsm_cpu_ms + gpu_decode_step_ms, 2)               # 46.9 ms
    throughput_drop_pct = round((1.0 - gpu_decode_step_ms / naive_step_ms) * 100.0, 1)  # 81.9%
    xgrammar_batch_cpu_ms = round(batch_size * xgrammar_per_req_us / 1000.0, 2)   # 2.10 ms
    xgrammar_effective_step_ms = round(max(gpu_decode_step_ms, xgrammar_batch_cpu_ms), 2)  # 8.50 ms
    xgrammar_throughput_loss_pct = round((xgrammar_effective_step_ms - gpu_decode_step_ms) / gpu_decode_step_ms * 100.0, 2)

    # 5. Ch18.6 算例 18-9 & 18-10: Multi-agent DAG fan-out tail amplification & speculative read-only tools
    u_join_12 = 0.99 ** (1.0 / fanout_k)
    u_join_16 = 0.99 ** (1.0 / 16)
    required_branch_percentile_k16 = round(u_join_16 * 100.0, 3)                  # 99.937
    alpha_tail = 0.4673
    t0_scale = single_branch_p99_s * ((1.0 - 0.99) ** alpha_tail)
    join_k12_p99_s = round(t0_scale * ((1.0 - u_join_12) ** (-alpha_tail)), 1)    # 19.8 s
    quorum_10of12_slack_p99_s = 7.4

    expected_saved_ms = round(speculative_hit_prob * tool_query_latency_ms, 1)    # 210.7 ms
    extra_wasted_query_pct = round((1.0 - speculative_hit_prob) / speculative_hit_prob * 100.0, 1)  # 16.3%

    return {
        "schema": "h2-topic-v1",
        "calculation": "agent-worm-grammar",
        "chapter": 18,
        "title": "第 18 章：智能体 WORM 状态复用、CacheGen 跨网回本、XGrammar 位掩码与多智能体 DAG 尾延迟账本",
        "scenario": {
            "worm_turns": worm_turns,
            "batch_size": batch_size,
            "qwen3_vocab_size": v_qwen,
            "deepseek_v3_vocab_size": v_ds,
            "gpu_decode_step_ms": gpu_decode_step_ms,
            "naive_fsm_cpu_ms": naive_fsm_cpu_ms,
            "fanout_k": fanout_k,
            "single_branch_p99_s": single_branch_p99_s,
            "normal_z99_ref": round(_norm_inv(0.99), 4),
        },
        "summary": {
            "worm_40turn_unique_tokens": unique_tokens,
            "worm_40turn_cum_prefill_tokens": cum_prefill_tokens,
            "worm_40turn_prefix_reuse_pct": worm_prefix_reuse_pct,
            "worm_prefill_to_decode_ratio": prefill_to_decode_ratio,
            "kv_64k_pcie_swap_in_ms": pcie_swap_in_ms,
            "kv_64k_swap_vs_recompute_speedup_x": swap_vs_recompute_speedup_x,
            "worm_15s_wait_hbm_rent_saving_pct": hbm_rent_saving_15s_pct,
            "dynamo_kv_router_speedup_x": dynamo_routing_speedup_x,
            "cachegen_compression_x": cachegen_compression_x,
            "cachegen_50g_restore_ms": cachegen_total_ms,
            "cachegen_vs_raw_transfer_speedup_x": cachegen_vs_raw_speedup_x,
            "cachegen_breakeven_bw_gibps": cachegen_breakeven_bw_gibps,
            "firecracker_fork_ms": firecracker_fork_ms,
            "firecracker_16branch_cow_saving_pct": cow_memory_saving_pct,
            "firecracker_sync_rollback_ms": sync_rollback_ms,
            "qwen3_bitmask_uint32_words": qwen_uint32_words,
            "qwen3_bitmask_bytes_per_req": qwen_mask_bytes_per_req,
            "qwen3_batch64_bitmask_mib": qwen_batch_mask_mib,
            "deepseek_bitmask_uint32_words": ds_uint32_words,
            "deepseek_bitmask_bytes_per_req": ds_mask_bytes_per_req,
            "naive_fsm_throughput_drop_pct": throughput_drop_pct,
            "xgrammar_batch64_cpu_ms": xgrammar_batch_cpu_ms,
            "xgrammar_effective_step_ms": xgrammar_effective_step_ms,
            "xgrammar_throughput_loss_pct": xgrammar_throughput_loss_pct,
            "k16_required_branch_percentile": required_branch_percentile_k16,
            "k12_barrier_join_p99_seconds": join_k12_p99_s,
            "k12_quorum_slack_p99_seconds": quorum_10of12_slack_p99_s,
            "speculative_tool_saved_latency_ms": expected_saved_ms,
            "speculative_tool_wasted_query_pct": extra_wasted_query_pct,
        },
        "assumptions": [
            "算例 18-1 至 18-3：40 轮编程智能体（L0=2,000, dL=2,000）前缀复用率达 95.12%（P/D=205:1）；CacheGen 4.21x 压缩将 50G 跨网拉取从 1,549.1 ms 压至 385.9 ms（< 500 ms 重算线，回本带宽仅需 4.43 GiB/s）；Firecracker CoW 16 分支省内存 89.06%。",
            "算例 18-8 至 18-10：Qwen3 (V=151,936) 词表位掩码占 18,992 B/请求（B=64 共 1.16 MiB），CPU 异步生成 2.10 ms 完全隐藏于 8.50 ms GPU 解码步之下；K=12 扇出尾延迟从 6.2 s 放大至 19.8 s，经关键路径调度与 10/12 定额收敛至 7.4 s。",
        ],
    }
