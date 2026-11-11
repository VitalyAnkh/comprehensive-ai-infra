"""Ch19: 3D Video DiT token/USP scaling, E-PD pool sizing, and 80 ms full-duplex RVQ speech rollback."""
from __future__ import annotations

from ..sources import model_config


def calculate(
    video_seconds: int = 10,
    fps: int = 24,
    height: int = 1080,
    width: int = 1920,
    temporal_stride: int = 4,
    spatial_stride: int = 8,
    patch_size: int = 2,
    dit_hidden_dim: int = 4096,
    sp_gpus: int = 64,
    speech_frame_hz: float = 12.5,
    rvq_codebooks: int = 8,
    barge_in_gen_frame: int = 65,
    barge_in_play_ms: float = 4320.0,
) -> dict:
    vl4b = model_config("qwen3-vl-4b")
    omni30b = model_config("qwen3-omni-30b-a3b-instruct")
    wan5b = model_config("wan2.2-ti2v-5b")

    # 1. 算例 19-2 & 19-3: 10s 1080p@24fps 3D DiT tokens (489,600) & 3D Causal VAE spatial-temporal tiling
    raw_frames = video_seconds * fps                                       # 240
    latent_t = raw_frames // temporal_stride                               # 60
    padded_h = ((height + 15) // 16) * 16                                  # 1088
    padded_w = ((width + 15) // 16) * 16                                   # 1920
    tokens_h = padded_h // (spatial_stride * patch_size)                   # 68
    tokens_w = padded_w // (spatial_stride * patch_size)                   # 120
    video_3d_tokens = latent_t * tokens_h * tokens_w                       # 489,600 tokens

    vae_untiled_1x_gib = round((240 * 1080 * 1920 * 128 * 2) / (1024**3), 4)            # 118.6523 GiB
    vae_tiled_1x_gib = round((16 * (270 * 1.125) * (480 * 1.125) * 128 * 2) / (1024**3), 4)  # 0.6257 GiB
    vae_tiled_2x_peak_gib = round(2.0 * vae_tiled_1x_gib, 4)                            # 1.2514 GiB
    vae_peak_compression_x = round(vae_untiled_1x_gib / vae_tiled_2x_peak_gib, 1)       # 94.8x

    # 2. 算例 19-4: 64-GPU Video DiT (N=489,600, d=4,096, BF16 2B) Ulysses vs Ring vs 8x8 2D-USP (in MiB)
    act_tensor_mib = (video_3d_tokens * dit_hidden_dim * 2) / (1024**2)                 # 3,825.0 MiB
    ulysses_per_gpu_mib = round(4.0 * act_tensor_mib / sp_gpus, 4)                      # 239.0625 MiB
    ring_step_mib = round(2.0 * act_tensor_mib / sp_gpus, 5)                            # 119.53125 MiB
    ring_total_64gpu_mib = round((sp_gpus - 1) * ring_step_mib, 5)                      # 7,530.46875 MiB
    usp_8x8_cross_node_mib = round((8 - 1) * ring_step_mib, 5)                          # 836.71875 MiB
    usp_vs_ring_rdma_reduction_x = round(ring_total_64gpu_mib / usp_8x8_cross_node_mib, 1)  # 9.0x

    # 3. 算例 19-6: 4-GPU disaggregated E + PD pool sizing & 75% Encoder Cache (EC) hit shift
    ec_bytes_per_4img_req = 4 * 8_192_000                                               # 32,768,000 B (31.25 MiB)
    e_to_pd_link_cap_req_s = round(300_000_000.0 / ec_bytes_per_4img_req, 2)            # 9.16 req/s
    best_4gpu_4img_h0_req_s = 6.00                                                      # (2E, 2PD) -> min(6, 8, 9.16) = 6.00
    best_4gpu_4img_h75_req_s = e_to_pd_link_cap_req_s                                   # (1E, 3PD) -> min(12, 12, 9.16) = 9.16

    # 4. 算例 19-9, 19-11 & 19-12: Full-duplex S2S TTFA (164 ms vs 1,280 ms), CUDA Graph (3.35x), & Barge-in rollback
    speech_frame_budget_ms = round(1000.0 / speech_frame_hz, 2)                         # 80.0 ms
    cascade_ttfa_ms = 480.0 + 220.0 + 280.0 + 240.0 + 60.0                              # 1,280.0 ms
    duplex_server_frame_ms = round(6.0 + 11.5 + 6.5, 2)                                 # 24.0 ms
    duplex_ttfa_ms = round(25.0 + 35.0 + duplex_server_frame_ms + speech_frame_budget_ms, 2)  # 164.0 ms
    duplex_vs_cascade_speedup_x = round(cascade_ttfa_ms / duplex_ttfa_ms, 2)            # 7.80x

    depth_micro_kernels = rvq_codebooks * 6 * 8                                         # 384 kernels
    dual_ar_eager_ms = round(3.50 + depth_micro_kernels * (0.035 + 0.05625), 2)         # 38.54 ms
    dual_ar_cudagraph_ms = round(3.50 + 8.00, 2)                                        # 11.50 ms
    cudagraph_speedup_x = round(dual_ar_eager_ms / dual_ar_cudagraph_ms, 2)             # 3.35x

    played_frames = int(barge_in_play_ms // speech_frame_budget_ms)                     # 54 frames
    rolled_back_frames = barge_in_gen_frame - played_frames                             # 11 frames (880.0 ms)
    rolled_back_kv_tokens = rolled_back_frames * (2 * rvq_codebooks + 1)                # 11 * 17 = 187 tokens
    paged_kv_rollback_ms = 0.12

    return {
        "schema": "h2-topic-v1",
        "calculation": "multimodal-dit-duplex",
        "chapter": 19,
        "title": "第 19 章：三维视频 DiT 序列并行通信、E-PD 解耦配比与全双工 RVQ 语音回滚账本",
        "scenario": {
            "video_seconds": video_seconds,
            "fps": fps,
            "resolution": f"{width}x{height}",
            "dit_hidden_dim": dit_hidden_dim,
            "sp_gpus": sp_gpus,
            "speech_frame_hz": speech_frame_hz,
            "rvq_codebooks": rvq_codebooks,
            "wan2_2_5b_config_dim": int(wan5b.get("dim", 3072)),
            "qwen3_vl_model_type": vl4b.get("model_type"),
            "qwen3_omni_model_type": omni30b.get("model_type"),
        },
        "summary": {
            "video_10s_1080p_3d_tokens": video_3d_tokens,
            "vae_untiled_1x_gib": vae_untiled_1x_gib,
            "vae_tiled_2x_peak_gib": vae_tiled_2x_peak_gib,
            "vae_tiling_peak_compression_x": vae_peak_compression_x,
            "dit_single_tensor_mib": act_tensor_mib,
            "ulysses_64gpu_per_layer_comm_mib": ulysses_per_gpu_mib,
            "ring_64gpu_per_layer_comm_mib": ring_total_64gpu_mib,
            "usp_8x8_cross_node_comm_mib": usp_8x8_cross_node_mib,
            "usp_8x8_vs_ring_rdma_reduction_x": usp_vs_ring_rdma_reduction_x,
            "epd_4gpu_4img_h0_throughput_req_s": best_4gpu_4img_h0_req_s,
            "epd_4gpu_4img_h75_throughput_req_s": best_4gpu_4img_h75_req_s,
            "cascade_speech_ttfa_ms": cascade_ttfa_ms,
            "duplex_speech_ttfa_ms": duplex_ttfa_ms,
            "duplex_vs_cascade_speedup_x": duplex_vs_cascade_speedup_x,
            "speech_12_5hz_frame_budget_ms": speech_frame_budget_ms,
            "duplex_server_frame_compute_ms": duplex_server_frame_ms,
            "dual_ar_depth_micro_kernels": depth_micro_kernels,
            "dual_ar_eager_frame_ms": dual_ar_eager_ms,
            "dual_ar_cudagraph_frame_ms": dual_ar_cudagraph_ms,
            "dual_ar_cudagraph_speedup_x": cudagraph_speedup_x,
            "barge_in_played_frame": played_frames,
            "barge_in_rolled_back_frames": rolled_back_frames,
            "barge_in_rolled_back_kv_tokens": rolled_back_kv_tokens,
            "barge_in_pointer_reset_ms": paged_kv_rollback_ms,
        },
        "assumptions": [
            "算例 19-2 至 19-6：1080p 视频补齐至 1088 后生成 60×68×120 = 489,600 个词元；3D 因果 VAE 分块将峰值显存从 118.65 GiB 压至 1.25 GiB（94.8x）；8×8 2D-USP（d=4,096）将跨机 RDMA 流量从 7,530.47 MiB 压至 836.72 MiB（9.0x）。",
            "算例 19-9 至 19-12：原生全双工 S2S 首包延迟 164.0 ms（较串联 1,280.0 ms 快 7.80x）；整步 CUDA Graph 消除 384 个微算子发射风暴（38.54 ms -> 11.50 ms，3.35x）；RTP 游标回滚 11 帧（187 词元）仅需 0.12 ms。",
        ],
    }
