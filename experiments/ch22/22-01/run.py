#!/usr/bin/env python3
"""Experiment 22-01: Cascaded vs Native Full-Duplex S2S latency, RVQ Codec bitrate, and Temporal+Depth CUDA Graph fusion."""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"
RESULTS_PATH = HERE / "results" / "summary.json"


def compute_metrics() -> dict:
    # 1. Cascaded ASR-LLM-TTS vs Native Full-Duplex S2S TTFA latency breakdown (Ex 22-1)
    cascaded_stages_ms = {
        "vad_silence_wait_ms": 480.0,
        "streaming_asr_ms": 220.0,
        "llm_ttft_and_clause_buffer_ms": 280.0,
        "tts_first_chunk_ms": 240.0,
        "network_and_io_ms": 60.0,
    }
    cascaded_total_ttfa_ms = round(sum(cascaded_stages_ms.values()), 2)  # 1280.0 ms

    duplex_stages_ms = {
        "client_mic_aec_opus_ms": 25.0,
        "webrtc_rtt_and_jitter_buffer_ms": 35.0,
        "codec_encode_ms": 6.0,
        "temporal_plus_depth_gpu_ms": 11.5,
        "causal_vocoder_decode_ms": 6.5,
        "frame_tick_alignment_ms": 80.0,
    }
    duplex_total_ttfa_ms = round(sum(duplex_stages_ms.values()), 2)  # 164.0 ms
    human_perception_threshold_ms = 200.0
    ttfa_speedup_x = round(cascaded_total_ttfa_ms / duplex_total_ttfa_ms, 2)  # 7.80x

    # 2. RVQ Neural Audio Codec (Mimi 12.5 Hz, Q=8 codebooks, V=2048) bitrate equation (Ex 22-2)
    pcm_sample_rate_hz = 24000
    pcm_bits_per_sample = 16
    raw_pcm_kbps = round(pcm_sample_rate_hz * pcm_bits_per_sample / 1000.0, 2)  # 384.0 kbps

    frame_rate_hz = 12.5
    frame_period_ms = round(1000.0 / frame_rate_hz, 2)  # 80.0 ms
    num_codebooks_q = 8
    vocab_size_va = 2048
    bits_per_token = int(round(math.log2(vocab_size_va)))  # 11 bits
    tokens_per_sec_per_stream = round(frame_rate_hz * num_codebooks_q, 1)  # 100.0 tok/s
    joint_step_tokens = 2 * num_codebooks_q + 1  # 8 user + 8 assistant + 1 inner monologue text = 17 tokens/step
    rvq_bitrate_kbps = round(tokens_per_sec_per_stream * bits_per_token / 1000.0, 2)  # 1.10 kbps
    codec_compression_x = round(raw_pcm_kbps / rvq_bitrate_kbps, 2)  # 349.09x

    # 3. Temporal (7B, 1 step) + Depth (100M, Q=8 steps) GPU Roofline & Full-Step CUDA Graph (Ex 22-3)
    temporal_step_ms = 3.50
    depth_layers = 6
    kernels_per_layer = 8
    total_depth_kernels = num_codebooks_q * depth_layers * kernels_per_layer  # 384 micro-kernels
    launch_overhead_per_kernel_us = 35.0
    eager_launch_overhead_ms = round(total_depth_kernels * launch_overhead_per_kernel_us / 1000.0, 2)  # 13.44 ms
    eager_unpinned_mem_ms = round(num_codebooks_q * depth_layers * 0.45, 2)  # 21.60 ms
    depth_eager_ms = round(eager_launch_overhead_ms + eager_unpinned_mem_ms, 2)  # 35.04 ms
    total_frame_eager_ms = round(temporal_step_ms + depth_eager_ms, 2)  # 38.54 ms

    depth_cudagraph_l2_ms = 8.00
    total_frame_cudagraph_ms = round(temporal_step_ms + depth_cudagraph_l2_ms, 2)  # 11.50 ms
    depth_speedup_x = round(depth_eager_ms / depth_cudagraph_l2_ms, 2)  # 4.38x
    full_step_speedup_x = round(total_frame_eager_ms / total_frame_cudagraph_ms, 2)  # 3.35x

    return {
        "experiment_id": "22-01",
        "latency_comparison": {
            "cascaded_total_ttfa_ms": cascaded_total_ttfa_ms,
            "duplex_total_ttfa_ms": duplex_total_ttfa_ms,
            "human_perception_threshold_ms": human_perception_threshold_ms,
            "ttfa_speedup_x": ttfa_speedup_x,
        },
        "rvq_codec_ledger": {
            "raw_pcm_kbps": raw_pcm_kbps,
            "frame_rate_hz": frame_rate_hz,
            "frame_period_ms": frame_period_ms,
            "num_codebooks_q": num_codebooks_q,
            "bits_per_token": bits_per_token,
            "tokens_per_sec_per_stream": tokens_per_sec_per_stream,
            "joint_step_tokens": joint_step_tokens,
            "rvq_bitrate_kbps": rvq_bitrate_kbps,
            "codec_compression_x": codec_compression_x,
        },
        "temporal_depth_cudagraph": {
            "temporal_step_ms": temporal_step_ms,
            "total_depth_kernels": total_depth_kernels,
            "depth_eager_ms": depth_eager_ms,
            "total_frame_eager_ms": total_frame_eager_ms,
            "depth_cudagraph_l2_ms": depth_cudagraph_l2_ms,
            "total_frame_cudagraph_ms": total_frame_cudagraph_ms,
            "depth_speedup_x": depth_speedup_x,
            "full_step_speedup_x": full_step_speedup_x,
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
