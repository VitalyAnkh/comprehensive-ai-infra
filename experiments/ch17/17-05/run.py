#!/usr/bin/env python3
"""Experiment 17-05: 32-bit Top-128 distillation ledger, Minitron structured pruning, and bounded-staleness Agentic RL."""
from __future__ import annotations

import json
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"
SUMMARY_PATH = HERE / "results" / "summary.json"


def pack_top1_entry_32bit(token_id: int, logit_f32: float) -> int:
    """Pack 18-bit token_id (supports V up to 262,144) + 14-bit truncated BF16/FP14 into one uint32."""
    assert 0 <= token_id < (1 << 18), f"token_id {token_id} exceeds 18-bit range"
    # IEEE-754 float32 -> top 14 bits (1 sign + 8 exp + 5 mantissa bits = FP14 with BF16 dynamic range)
    raw_u32 = struct.unpack("<I", struct.pack("<f", logit_f32))[0]
    fp14 = (raw_u32 >> 18) & 0x3FFF
    return ((token_id & 0x3FFFF) << 14) | fp14


def unpack_top1_entry_32bit(word_u32: int) -> tuple[int, float]:
    token_id = (word_u32 >> 14) & 0x3FFFF
    raw_u32 = (word_u32 & 0x3FFF) << 18
    logit_approx = struct.unpack("<f", struct.pack("<I", raw_u32))[0]
    return token_id, logit_approx


def compute_metrics() -> dict:
    # 1. Distillation storage & offline-vs-online crossover ledger (Example 17-12 & Exercise 17-18)
    distill_tokens = 200_000_000_000
    vocab_qwen = 151_936
    vocab_ds = 129_280
    top_k = 128

    full_qwen_bf16_pb = round(distill_tokens * vocab_qwen * 2 / 1e15, 4)        # 60.7744 PB
    full_ds_bf16_pb = round(distill_tokens * vocab_ds * 2 / 1e15, 4)            # 51.7120 PB
    qwen_100b_bf16_pb = round((distill_tokens // 2) * vocab_qwen * 2 / 1e15, 4) # 30.3872 PB (~30.39 PB)
    unpacked_6b_top128_tb = round(distill_tokens * top_k * 6 / 1e12, 2)         # 153.60 TB
    bitpacked_4b_top128_tb = round(distill_tokens * top_k * 4 / 1e12, 2)        # 102.40 TB
    compression_vs_full_x = round((full_qwen_bf16_pb * 1000.0) / bitpacked_4b_top128_tb, 1)  # 593.5x

    # Verify 32-bit codec on highest Qwen3 token ID (151,935) and typical logits
    test_pairs = [(151_935, 14.75), (129_279, 8.125), (42, -3.5), (0, 0.625)]
    max_abs_err = 0.0
    exact_ids = True
    for tid, val in test_pairs:
        packed = pack_top1_entry_32bit(tid, val)
        rec_id, rec_val = unpack_top1_entry_32bit(packed)
        exact_ids = exact_ids and (rec_id == tid)
        max_abs_err = max(max_abs_err, abs(rec_val - val))

    offline_gen_gpu_h = 14_800.0
    storage_3mo_equiv_gpu_h = 3_200.0
    online_nvfp4_teacher_gpu_h_per_epoch = 9_600.0
    crossover_epochs = round((offline_gen_gpu_h + storage_3mo_equiv_gpu_h) / online_nvfp4_teacher_gpu_h_per_epoch, 2)  # 1.88

    # 2. Minitron structured width/depth pruning from 32B -> 16B (Tensor Core 128-multiple alignment check)
    orig_hidden, pruned_hidden = 5120, 3840       # both divisible by 128
    orig_ffn, pruned_ffn = 25600, 16384           # both divisible by 128
    orig_layers, pruned_layers = 64, 48
    tensor_core_128_aligned = (pruned_hidden % 128 == 0) and (pruned_ffn % 128 == 0)

    # 3. 32-turn Agentic RL synchronous barrier vs bounded-staleness async (Delta_max = 3) (Example 17-13)
    max_staleness_lag = 3
    sync_gpu_idle_pct = 68.0
    sync_updates_per_hour = 4.50
    async_bounded_updates_per_hour = 11.88
    agentic_rl_speedup_x = round(async_bounded_updates_per_hour / sync_updates_per_hour, 2)  # 2.64x

    return {
        "experiment_id": "17-05",
        "distillation_32bit_ledger": {
            "distill_tokens": distill_tokens,
            "qwen3_vocab_size": vocab_qwen,
            "deepseek_v3_vocab_size": vocab_ds,
            "top_k_logits": top_k,
            "qwen_200b_full_logits_bf16_pb": full_qwen_bf16_pb,
            "deepseek_200b_full_logits_bf16_pb": full_ds_bf16_pb,
            "qwen_100b_full_logits_bf16_pb": qwen_100b_bf16_pb,
            "top128_unpacked_6b_tb": unpacked_6b_top128_tb,
            "top128_bitpacked_32bit_tb": bitpacked_4b_top128_tb,
            "top128_compression_vs_full_x": compression_vs_full_x,
            "codec_token_id_lossless": exact_ids,
            "codec_max_abs_err": round(max_abs_err, 6),
            "offline_vs_online_teacher_crossover_epochs": crossover_epochs,
            "minitron_tensor_core_128_aligned": tensor_core_128_aligned,
            "minitron_pruned_layers": pruned_layers,
        },
        "bounded_staleness_agentic_rl": {
            "max_staleness_version_lag": max_staleness_lag,
            "agentic_rl_sync_gpu_idle_pct": sync_gpu_idle_pct,
            "sync_updates_per_hour": sync_updates_per_hour,
            "async_bounded_updates_per_hour": async_bounded_updates_per_hour,
            "agentic_rl_async_bounded_speedup_x": agentic_rl_speedup_x,
        },
    }


def main() -> None:
    metrics = compute_metrics()
    for p in (EXPECTED_PATH, SUMMARY_PATH):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
