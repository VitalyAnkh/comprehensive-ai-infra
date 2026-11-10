#!/usr/bin/env python3
"""Experiment 19-01: Multimodal EC vs KV amplification, 3D Video DiT 2D-USP communication, and Tiled VAE / TeaCache."""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"


def compute_metrics() -> dict:
    # 1. Qwen3-VL-4B single 640x640 image EC vs KV state size (case-studies/multimodal-stage-placement.md)
    img_h, img_w, patch, merge = 640, 640, 16, 2
    visual_tokens = ((img_h // patch) * (img_w // patch)) // (merge * merge)  # 400
    proj_dim = 2560
    deepstack_blocks = 3
    ec_total_dim = proj_dim * (1 + deepstack_blocks)  # 10,240
    ec_bytes = visual_tokens * ec_total_dim * 2  # 8,192,000 B
    ec_mib = ec_bytes / (1024 * 1024)  # 7.8125 MiB

    llm_layers, kv_heads, head_dim = 36, 8, 128
    kv_bytes_per_token = llm_layers * 2 * kv_heads * head_dim * 2  # 147,456 B
    kv_bytes = visual_tokens * kv_bytes_per_token  # 58,982,400 B
    kv_mib = kv_bytes / (1024 * 1024)  # 56.25 MiB
    kv_to_ec_ratio = round(kv_mib / ec_mib, 4)  # 7.2

    # 2. 3D Video DiT spatiotemporal tokens & 2D-USP (Ulysses x Ring) communication
    duration_s, fps = 10, 24
    raw_frames = duration_s * fps  # 240
    vid_h, vid_w = 1080, 1920
    t_tokens = raw_frames // 4  # 60
    h_tokens = math.ceil(vid_h / 16)  # 68
    w_tokens = math.ceil(vid_w / 16)  # 120
    n_vid_tokens = t_tokens * h_tokens * w_tokens  # 489,600

    d_model = 4096
    elem_bytes = 2  # BF16
    p_total = 64
    ulysses_64_mib = (4 * n_vid_tokens * d_model * elem_bytes / p_total) / (1024 * 1024)  # 239.0625 MiB
    ring_64_step_mib = (2 * n_vid_tokens * d_model * elem_bytes / p_total) / (1024 * 1024)  # 119.53125 MiB
    ring_64_total_mib = (p_total - 1) * ring_64_step_mib  # 7530.46875 MiB

    p_ulysses, p_ring = 8, 8
    usp_inter_node_ring_mib = (p_ring - 1) * ring_64_step_mib  # 836.71875 MiB
    inter_node_reduction_ratio = round(ring_64_total_mib / usp_inter_node_ring_mib, 4)  # 9.0

    # 3. 3D Causal VAE Tiling & TeaCache step reuse speedup
    vae_channels = 128
    untiled_vae_gib = round(
        (raw_frames * vid_h * vid_w * vae_channels * elem_bytes) / (1024**3), 2
    )  # 118.65 GiB
    tiled_frames = 16
    halo_factor = 1.125 * 1.125  # 12.5% spatial halo on H and W
    tiled_vae_gib = round(
        (tiled_frames * (vid_h / 4) * (vid_w / 4) * halo_factor * vae_channels * elem_bytes) / (1024**3), 4
    )

    total_steps, active_steps = 50, 22
    t_step_s, t_fixed_s = 1.20, 3.60
    t_baseline_s = round(total_steps * t_step_s + t_fixed_s, 2)  # 63.60 s
    t_teacache_s = round(active_steps * t_step_s + t_fixed_s, 2)  # 30.00 s
    teacache_speedup = round(t_baseline_s / t_teacache_s, 2)  # 2.12x

    return {
        "experiment_id": "19-01",
        "qwen3_vl_4b_state": {
            "visual_tokens": visual_tokens,
            "ec_bytes": ec_bytes,
            "ec_mib": ec_mib,
            "kv_bytes_per_token": kv_bytes_per_token,
            "kv_bytes": kv_bytes,
            "kv_mib": kv_mib,
            "kv_to_ec_ratio": kv_to_ec_ratio,
        },
        "video_dit_2d_usp": {
            "t_tokens": t_tokens,
            "h_tokens": h_tokens,
            "w_tokens": w_tokens,
            "n_vid_tokens": n_vid_tokens,
            "ulysses_64_per_layer_mib": ulysses_64_mib,
            "ring_64_step_mib": ring_64_step_mib,
            "ring_64_total_mib": ring_64_total_mib,
            "usp_8x8_inter_node_mib": usp_inter_node_ring_mib,
            "inter_node_reduction_ratio": inter_node_reduction_ratio,
        },
        "vae_and_teacache": {
            "untiled_vae_gib": untiled_vae_gib,
            "tiled_vae_gib": tiled_vae_gib,
            "t_baseline_s": t_baseline_s,
            "t_teacache_s": t_teacache_s,
            "teacache_speedup": teacache_speedup,
        },
    }


def main() -> None:
    metrics = compute_metrics()
    EXPECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPECTED_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
