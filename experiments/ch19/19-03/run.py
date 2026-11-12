#!/usr/bin/env python3
"""Experiment 19-03: Block-causal KV rolling window and PAB/TeaCache reuse for >20 FPS streaming world models."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"
SUMMARY_PATH = HERE / "results" / "summary.json"


def compute_metrics() -> dict:
    # 1. Block-causal rolling KV cache pool (Example 19-8 & Exercise 19-15)
    chunk_rgb_frames = 3
    tokens_per_chunk = 3600  # (720/16) * (1280/16) = 45 * 80
    sink_chunks = 1
    window_chunks = 5
    active_chunks = sink_chunks + window_chunks  # 6
    active_kv_tokens = active_chunks * tokens_per_chunk  # 21,600

    dit_layers = 28
    cross_chunk_layers = 8
    kv_heads = 16
    head_dim = 128
    dtype_bytes = 2
    bytes_per_tok_per_layer = 2 * kv_heads * head_dim * dtype_bytes  # 8,192 B
    bytes_per_tok_28l = dit_layers * bytes_per_tok_per_layer  # 229,376 B

    hybrid_8l_rolling_kv_gib = round(
        (active_kv_tokens * cross_chunk_layers * bytes_per_tok_per_layer) / (1024 ** 3), 4
    )  # 1.3184 GiB
    full_28l_rolling_kv_gib = round(
        (active_kv_tokens * bytes_per_tok_28l) / (1024 ** 3), 4
    )  # 4.6143 GiB

    ten_min_chunks = 4000  # 10 min * 60 s * 20 FPS / 3 frames
    unbounded_10min_8l_gib = round(
        (ten_min_chunks * tokens_per_chunk * cross_chunk_layers * bytes_per_tok_per_layer) / (1024 ** 3), 2
    )  # 878.91 GiB
    unbounded_10min_28l_gib = round(
        (ten_min_chunks * tokens_per_chunk * bytes_per_tok_28l) / (1024 ** 3), 2
    )  # 3076.17 GiB
    kv_compression_ratio_10min = round(ten_min_chunks / active_chunks, 2)  # 666.67x

    # 2. 4-step DMD block-causal latency with PAB + TeaCache (Example 19-8 & Exercise 19-16)
    dmd_steps = 4
    spatial_ffn_ms = 18.0
    temporal_causal_ms = 16.0
    cross_cond_ms = 8.0
    step_full_ms = spatial_ffn_ms + temporal_causal_ms + cross_cond_ms  # 42.0 ms
    vae_chunk_ms = 18.0

    uncached_dit_chunk_ms = round(dmd_steps * step_full_ms, 2)  # 168.0 ms
    uncached_total_chunk_ms = round(uncached_dit_chunk_ms + vae_chunk_ms, 2)  # 186.0 ms
    uncached_frame_ms = round(uncached_total_chunk_ms / chunk_rgb_frames, 2)  # 62.0 ms
    uncached_fps = round(1000.0 / uncached_frame_ms, 2)  # 16.13 FPS

    # Step breakdown: s=4 (42.0 ms), s=3 (18.0 ms), s=2 (2.0 ms TeaCache), s=1 (34.0 ms)
    step_latencies_ms = [42.0, 18.0, 2.0, 34.0]
    pab_teacache_dit_chunk_ms = round(sum(step_latencies_ms), 2)  # 96.0 ms
    opt_total_chunk_ms = round(pab_teacache_dit_chunk_ms + vae_chunk_ms, 2)  # 114.0 ms
    opt_frame_ms = round(opt_total_chunk_ms / chunk_rgb_frames, 2)  # 38.0 ms
    streaming_fps = round(1000.0 / opt_frame_ms, 2)  # 26.32 FPS
    fps_speedup = round(streaming_fps / uncached_fps, 2)  # 1.63x

    return {
        "experiment_id": "19-03",
        "rolling_kv_pool": {
            "chunk_rgb_frames": chunk_rgb_frames,
            "tokens_per_chunk": tokens_per_chunk,
            "sink_chunks": sink_chunks,
            "window_chunks": window_chunks,
            "active_chunks": active_chunks,
            "active_kv_tokens": active_kv_tokens,
            "bytes_per_tok_28l": bytes_per_tok_28l,
            "hybrid_8l_rolling_kv_gib": hybrid_8l_rolling_kv_gib,
            "full_28l_rolling_kv_gib": full_28l_rolling_kv_gib,
            "unbounded_10min_8l_gib": unbounded_10min_8l_gib,
            "unbounded_10min_28l_gib": unbounded_10min_28l_gib,
            "kv_compression_ratio_10min": kv_compression_ratio_10min,
        },
        "pab_teacache_streaming": {
            "dmd_steps": dmd_steps,
            "uncached_dit_chunk_ms": uncached_dit_chunk_ms,
            "uncached_total_chunk_ms": uncached_total_chunk_ms,
            "uncached_frame_ms": uncached_frame_ms,
            "uncached_fps": uncached_fps,
            "step_latencies_ms": step_latencies_ms,
            "pab_teacache_dit_chunk_ms": pab_teacache_dit_chunk_ms,
            "vae_chunk_ms": vae_chunk_ms,
            "opt_total_chunk_ms": opt_total_chunk_ms,
            "opt_frame_ms": opt_frame_ms,
            "streaming_fps": streaming_fps,
            "fps_speedup": fps_speedup,
            "meets_20fps_sla": streaming_fps >= 20.0,
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
