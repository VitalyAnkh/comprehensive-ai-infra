"""Ch17: GRPO colocation resharding, RollPacker, Varlen SFT packing, 32-bit Top-128 distillation, and Agentic RL."""
from __future__ import annotations

from ..sources import model_config


def calculate(
    max_seq_len: int = 65_536,
    mean_seq_len: int = 4_200,
    distill_tokens: int = 200_000_000_000,
    top_k_logits: int = 128,
    max_staleness_version_lag: int = 3,
) -> dict:
    qwen32 = model_config("qwen3-32b")
    dsv3 = model_config("deepseek-v3")
    vocab_qwen = int(qwen32["vocab_size"])  # 151,936
    vocab_ds = int(dsv3["vocab_size"])      # 129,280

    # 1. 算例 17-1: PPO (Actor+Critic+Ref+Reward) vs GRPO/DAPO static state ledger on 32B model
    params_32b = 32e9
    ppo_static_gb = round((2 * 16.0 + 2 * 2.0) * params_32b / 1e9, 1)   # 1,152.0 GB
    grpo_static_gb = round((1 * 16.0 + 1 * 2.0) * params_32b / 1e9, 1)  # 576.0 GB (-50.0%)
    dapo_kl_free_static_gb = round(16.0 * params_32b / 1e9, 1)          # 512.0 GB

    # 2. 算例 17-2: 64-GPU colocated ZeRO-3 global resharding vs 3D-HybridEngine intra-node micro-group resharding
    naive_zero3_reshard_s = round(63.0 / 40.0, 3)                       # 1.575 s (1,575 ms over 40 GB/s RoCE)
    hybrid_engine_reshard_s = round(14.0 / 400.0, 3)                    # 0.035 s (35 ms over 400 GB/s NVSwitch)
    hybrid_reshard_speedup_x = round(naive_zero3_reshard_s / hybrid_engine_reshard_s, 1)  # 45.0x

    # 3. 算例 17-4 & 17-5: RollPacker speculative pool (160x10=1,600 vs 1,024), stream overlap, & 2-slot length bias
    rollpacker_target_trajs = 128 * 8                                   # 1,024
    rollpacker_spec_trajs = 160 * 10                                    # 1,600
    rollpacker_overprovision_pct = round((rollpacker_spec_trajs - rollpacker_target_trajs) / rollpacker_target_trajs * 100.0, 2)  # 56.25%
    serial_rl_step_s = 40.0 + 16.0 + 4.0                                # 60.0 s
    stream_overlap_step_s = max(40.0 + 3.0, 10.0 + 3.0 + 16.0) + 8.0 + 4.0      # 55.0 s
    over_migration_counterexample_s = max(40.0 + 10.0, 10.0 + 10.0 + 16.0) + 8.0 + 4.0  # 62.0 s
    two_slot_long_commit_prob_pct = round((0.5**2) * 100.0, 1)          # 25.0% (halved from 50.0% prior)

    # 4. Ch17.5 算例 17-11: 64k SFT heavy-tailed corpus (mean 4.2k) Pad-to-Max vs cu_seqlens + ZigZag Ring
    naive_pad_mfu_pct = 13.6
    varlen_unbalanced_mfu_pct = 41.2
    varlen_zigzag_mfu_pct = 56.44
    varlen_speedup_x = round(varlen_zigzag_mfu_pct / naive_pad_mfu_pct, 2)  # 4.15x

    # 5. Ch17.5 算例 17-12: Distilling Qwen3-32B (V=151,936) -> 16B over 200B tokens
    full_vocab_bf16_pb = round(distill_tokens * vocab_qwen * 2 / 1e15, 4)        # 60.7744 PB (~60.8 PB)
    full_vocab_ds_bf16_pb = round(distill_tokens * vocab_ds * 2 / 1e15, 4)       # 51.7120 PB
    unpacked_top128_tb = round(distill_tokens * top_k_logits * 6 / 1e12, 2)      # 153.60 TB (int32 4B + BF16 2B)
    bitpacked_32b_top128_tb = round(distill_tokens * top_k_logits * 4 / 1e12, 2)  # 102.40 TB (18-bit ID + 14-bit FP14)
    compression_vs_full_x = round((full_vocab_bf16_pb * 1000.0) / bitpacked_32b_top128_tb, 1)  # 593.5x

    offline_gen_gpu_hours = 14_800.0
    storage_3mo_equiv_gpu_hours = 3_200.0
    online_nvfp4_teacher_extra_gpu_hours_per_epoch = 9_600.0
    crossover_epochs = round((offline_gen_gpu_hours + storage_3mo_equiv_gpu_hours) / online_nvfp4_teacher_extra_gpu_hours_per_epoch, 2)

    # 6. Ch17.6 算例 17-13: 32-turn code-repair Agentic RL sync barrier vs bounded-staleness async (Delta_max = 3)
    sync_gpu_idle_pct = 68.0
    sync_updates_per_hour = 4.50
    async_bounded_updates_per_hour = 11.88
    agentic_rl_speedup_x = round(async_bounded_updates_per_hour / sync_updates_per_hour, 2)  # 2.64x

    return {
        "schema": "h2-topic-v1",
        "calculation": "posttrain-rlvr-distill",
        "chapter": 17,
        "title": "第 17 章：GRPO 共置重分片、RollPacker 流式重叠、变长 SFT 装箱、32-bit 蒸馏与异步 Agentic RL 账本",
        "scenario": {
            "max_seq_len": max_seq_len,
            "mean_seq_len": mean_seq_len,
            "distill_tokens": distill_tokens,
            "qwen3_vocab_size": vocab_qwen,
            "deepseek_v3_vocab_size": vocab_ds,
            "top_k_logits": top_k_logits,
            "max_staleness_version_lag": max_staleness_version_lag,
        },
        "summary": {
            "ppo_32b_static_state_gb": ppo_static_gb,
            "grpo_32b_static_state_gb": grpo_static_gb,
            "dapo_32b_kl_free_state_gb": dapo_kl_free_static_gb,
            "zero3_naive_reshard_seconds": naive_zero3_reshard_s,
            "hybrid_engine_reshard_seconds": hybrid_engine_reshard_s,
            "hybrid_engine_reshard_speedup_x": hybrid_reshard_speedup_x,
            "rollpacker_speculative_trajectories": rollpacker_spec_trajs,
            "rollpacker_overprovision_pct": rollpacker_overprovision_pct,
            "rollpacker_serial_step_seconds": serial_rl_step_s,
            "rollpacker_stream_overlap_seconds": stream_overlap_step_s,
            "rollpacker_overmigration_counterexample_seconds": over_migration_counterexample_s,
            "two_slot_naive_truncation_long_prob_pct": two_slot_long_commit_prob_pct,
            "sft_pad_to_64k_mfu_pct": naive_pad_mfu_pct,
            "sft_varlen_unbalanced_mfu_pct": varlen_unbalanced_mfu_pct,
            "sft_varlen_zigzag_mfu_pct": round(varlen_zigzag_mfu_pct, 1),
            "sft_varlen_throughput_speedup_x": varlen_speedup_x,
            "qwen_200b_full_logits_bf16_pb": full_vocab_bf16_pb,
            "deepseek_200b_full_logits_bf16_pb": full_vocab_ds_bf16_pb,
            "top128_unpacked_6b_tb": unpacked_top128_tb,
            "top128_bitpacked_32bit_tb": bitpacked_32b_top128_tb,
            "top128_compression_vs_full_x": compression_vs_full_x,
            "offline_vs_online_teacher_crossover_epochs": crossover_epochs,
            "agentic_rl_sync_gpu_idle_pct": sync_gpu_idle_pct,
            "agentic_rl_async_bounded_speedup_x": agentic_rl_speedup_x,
        },
        "assumptions": [
            "算例 17-1 至 17-5：GRPO 省去 Critic 与 RM 将 32B 静态状态从 1,152 GB 压至 576 GB；3D-HybridEngine 机内微分组重分片将切换耗时从 1.575 s 压至 0.035 s（45.0x）；RollPacker 流式重叠将步长从 60 s 压至 55 s 并避免 1/2 -> 1/4 长度坍缩。",
            "算例 17-11 至 17-13：cu_seqlens 装箱 + ZigZag 负载均衡将 64k SFT MFU 从 13.6% 提至 56.4%（4.15x）；Qwen3（V=151,936）采用 18-bit ID + 14-bit FP14 打包为 32-bit（4 B/entry），200B 词元 Top-128 仅 102.4 TB；异步有界陈旧度 Agentic RL 提速 2.64x。",
        ],
    }
