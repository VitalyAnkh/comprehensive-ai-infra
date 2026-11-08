#!/usr/bin/env python3
"""Experiment 17-02: E-P-D-G disaggregation optimization, Transfusion hybrid mask occupancy, and 3DGS-DiT world model."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"
RESULTS_PATH = HERE / "results" / "baseline.json"


def compute_metrics() -> dict:
    # 1. E-P-D-G 4-GPU stage placement & EC cache hit rate shift (Example 17-6 & Exercise 17-11)
    mu_img = 12.0
    mu_pd = 4.0
    single_img_opt_pair = [1, 3]
    single_img_opt_tput = min(mu_img * 1, mu_pd * 3)  # 12.0 req/s
    ec_hit_rate = 0.75
    effective_mu_e_at_75pct = round(mu_img / (1.0 - ec_hit_rate), 2)  # 48.0 req/s

    imgs_per_req = 4
    ec_bytes_per_img = 8192000  # 7.8125 MiB
    link_bw_bytes_s = 300_000_000  # 300 MB/s
    link_cap_req_s = round(link_bw_bytes_s / (imgs_per_req * ec_bytes_per_img), 2)  # 9.16 req/s

    tput_4img_h0 = {
        f"{ne}E_{4-ne}PD": round(min((mu_img * ne) / imgs_per_req, mu_pd * (4 - ne), link_cap_req_s), 2)
        for ne in (1, 2, 3)
    }
    tput_4img_h75 = {
        f"{ne}E_{4-ne}PD": round(
            min((mu_img * ne) / (imgs_per_req * (1.0 - ec_hit_rate)), mu_pd * (4 - ne), link_cap_req_s), 2
        )
        for ne in (1, 2, 3)
    }
    kv_mib, ec_mib = 56.25, 7.8125
    warm_demotion_saving_pct = round((1.0 - ec_mib / kv_mib) * 100.0, 2)  # 86.11%

    # 2. Transfusion / Show-o hybrid causal-bidirectional mask & D->G handoff (Example 17-7 & Exercise 17-13)
    l_t1, l_i1, l_t2 = 1024, 2048, 1024
    l_tot = l_t1 + l_i1 + l_t2  # 4,096
    area_t1 = (l_t1 * l_t1) // 2  # 524,288
    area_i1 = l_i1 * l_t1 + l_i1 * l_i1  # 2,097,152 + 4,194,304 = 6,291,456
    area_t2 = l_t2 * (l_t1 + l_i1) + (l_t2 * l_t2) // 2  # 3,145,728 + 524,288 = 3,670,016
    active_token_pairs = area_t1 + area_i1 + area_t2  # 10,485,760
    full_matrix_pairs = l_tot * l_tot  # 16,777,216
    active_density_pct = round(active_token_pairs / full_matrix_pairs * 100.0, 2)  # 62.5%

    tile_size = 128
    b_t1, b_i1, b_t2 = l_t1 // tile_size, l_i1 // tile_size, l_t2 // tile_size  # 8, 16, 8
    total_tiles = (l_tot // tile_size) ** 2  # 1,024
    tiles_t1 = b_t1 * (b_t1 + 1) // 2  # 36
    tiles_i1 = b_i1 * b_t1 + b_i1 * b_i1  # 128 + 256 = 384
    tiles_t2 = b_t2 * (b_t1 + b_i1) + b_t2 * (b_t2 + 1) // 2  # 192 + 36 = 228
    executed_tiles = tiles_t1 + tiles_i1 + tiles_t2  # 648
    skipped_tiles = total_tiles - executed_tiles  # 376
    skipped_tile_pct = round(skipped_tiles / total_tiles * 100.0, 2)  # 36.72%
    tile_effective_occupancy_pct = round(
        active_token_pairs / (executed_tiles * tile_size * tile_size) * 100.0, 2
    )  # 98.77%

    kv_bytes_per_token = 147456  # 144 KiB/token
    prefix_kv_handoff_mib = round((l_t1 * kv_bytes_per_token) / (1024 * 1024), 2)  # 144.0 MiB
    rdma_bw_mib_s = 50000.0  # 50,000 MiB/s (~50 GB/s)
    kv_handoff_ms = round((prefix_kv_handoff_mib / rdma_bw_mib_s) * 1000.0, 2)  # 2.88 ms

    # 3. 3DGS + Video DiT hybrid world model pipeline latency & FPS
    gs_raster_ms = 2.4
    dit_4step_stream_ms = 24.0
    tiled_vae_ms = 4.5
    total_frame_ms = round(gs_raster_ms + dit_4step_stream_ms + tiled_vae_ms, 2)  # 30.9 ms
    steady_fps = round(1000.0 / total_frame_ms, 2)  # 32.36 FPS

    return {
        "experiment_id": "17-02",
        "epdg_disaggregation": {
            "single_img_opt_pair": single_img_opt_pair,
            "single_img_opt_tput": single_img_opt_tput,
            "effective_mu_e_at_75pct": effective_mu_e_at_75pct,
            "link_cap_req_s": link_cap_req_s,
            "tput_4img_h0": tput_4img_h0,
            "tput_4img_h75": tput_4img_h75,
            "warm_demotion_saving_pct": warm_demotion_saving_pct,
        },
        "transfusion_hybrid_mask": {
            "total_tokens": l_tot,
            "active_token_pairs": active_token_pairs,
            "full_matrix_pairs": full_matrix_pairs,
            "active_density_pct": active_density_pct,
            "total_tiles": total_tiles,
            "executed_tiles": executed_tiles,
            "skipped_tiles": skipped_tiles,
            "skipped_tile_pct": skipped_tile_pct,
            "tile_effective_occupancy_pct": tile_effective_occupancy_pct,
            "prefix_kv_handoff_mib": prefix_kv_handoff_mib,
            "kv_handoff_ms": kv_handoff_ms,
        },
        "hybrid_3dgs_dit_world_model": {
            "gs_raster_ms": gs_raster_ms,
            "dit_4step_stream_ms": dit_4step_stream_ms,
            "tiled_vae_ms": tiled_vae_ms,
            "total_frame_ms": total_frame_ms,
            "steady_fps": steady_fps,
        },
    }


def main() -> None:
    metrics = compute_metrics()
    for p in (EXPECTED_PATH, RESULTS_PATH):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
