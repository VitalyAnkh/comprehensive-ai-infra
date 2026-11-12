#!/usr/bin/env python3
"""Experiment 19-06: Real-time 3D Avatar lip-sync (< 40 ms jitter alignment) & global edge WebRTC SFU mesh."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"
RESULTS_PATH = HERE / "results" / "summary.json"


def compute_metrics() -> dict:
    # 1. Worked Example 19-14: 3D Avatar Lip-Sync & < 40 ms Audio-Video Jitter Alignment
    vocoder_ms = 6.50
    avatar_render_ms = 28.50
    nvenc_encode_ms = 4.00
    vis_path_ms = round(avatar_render_ms + nvenc_encode_ms, 2)  # 32.50
    steady_jitter_skew_ms = 18.00
    p99_extra_jitter_ms = 15.00
    audio_hold_delay_ms = 26.00
    base_audio_ttfa_ms = 164.00

    raw_steady_skew_ms = round((vis_path_ms - vocoder_ms) + steady_jitter_skew_ms, 2)  # 44.00
    raw_p99_skew_ms = round(raw_steady_skew_ms + p99_extra_jitter_ms, 2)  # 59.00
    aligned_steady_skew_ms = round(raw_steady_skew_ms - audio_hold_delay_ms, 2)  # 18.00
    aligned_p99_skew_ms = round(raw_p99_skew_ms - audio_hold_delay_ms, 2)  # 33.00
    skew_reduction_pct = round(100.0 * (1.0 - aligned_steady_skew_ms / raw_steady_skew_ms), 2)  # 59.09
    avatar_ttfa_ms = round(base_audio_ttfa_ms + audio_hold_delay_ms, 2)  # 190.00

    # 2. Worked Example 19-15: Global Anycast Edge WebRTC SFU & Private Backbone RVQ Mesh
    client_aec_ms = 25.00
    pub_rtt_ms = 110.00
    pub_jb_ms = 65.00
    srv_full_ms = 24.00
    unsync_tick_ms = 80.00

    edge_rtt_ms = 12.00
    edge_jb_ms = 10.00
    edge_dsp_ms = 4.50
    backbone_rtt_ms = 68.00
    backbone_jb_ms = 4.00
    srv_core_ms = 18.00
    sync_tick_ms = 40.00

    net_direct_ms = round(pub_rtt_ms + pub_jb_ms, 2)  # 175.00
    net_edge_ms = round(edge_rtt_ms + edge_jb_ms + edge_dsp_ms + backbone_rtt_ms + backbone_jb_ms, 2)  # 98.50
    net_reduction_pct = round(100.0 * (1.0 - net_edge_ms / net_direct_ms), 2)  # 43.71

    ttfa_direct_ms = round(client_aec_ms + net_direct_ms + srv_full_ms + unsync_tick_ms, 2)  # 304.00
    ttfa_edge_ms = round(client_aec_ms + net_edge_ms + srv_core_ms + sync_tick_ms, 2)  # 181.50
    ttfa_speedup_x = round(ttfa_direct_ms / ttfa_edge_ms, 2)  # 1.67

    opus_bi_kbps = 96.00
    rvq_bi_kbps = 2.20
    bw_compression_x = round(opus_bi_kbps / rvq_bi_kbps, 2)  # 43.64
    bw_saving_pct = round(100.0 * (1.0 - rvq_bi_kbps / opus_bi_kbps), 2)  # 97.71

    # 3. Exercises 19-27 & 19-28
    p11_raw_steady_skew_ms = round((27.0 - 5.0) + 16.0, 2)  # 38.00
    p11_raw_p99_skew_ms = round(p11_raw_steady_skew_ms + 12.0, 2)  # 50.00
    p11_aligned_steady_skew_ms = round(p11_raw_steady_skew_ms - 22.0, 2)  # 16.00
    p11_aligned_p99_skew_ms = round(p11_raw_p99_skew_ms - 22.0, 2)  # 28.00
    p11_avatar_ttfa_ms = round(160.0 + 22.0, 2)  # 182.00

    p12_ttfa_direct_ms = round(25.0 + 120.0 + 70.0 + 24.0 + 80.0, 2)  # 319.00
    p12_net_edge_ms = round(10.0 + 8.0 + 4.5 + 76.0 + 4.0, 2)  # 102.50
    p12_ttfa_edge_ms = round(25.0 + p12_net_edge_ms + 18.0 + 40.0, 2)  # 185.50
    p12_speedup_x = round(p12_ttfa_direct_ms / p12_ttfa_edge_ms, 2)  # 1.72

    return {
        "experiment_id": "19-06",
        "avatar_lipsync_alignment_ex19_14": {
            "vis_path_ms": vis_path_ms,
            "raw_steady_skew_ms": raw_steady_skew_ms,
            "raw_p99_skew_ms": raw_p99_skew_ms,
            "aligned_steady_skew_ms": aligned_steady_skew_ms,
            "aligned_p99_skew_ms": aligned_p99_skew_ms,
            "skew_reduction_pct": skew_reduction_pct,
            "avatar_ttfa_ms": avatar_ttfa_ms,
        },
        "global_edge_sfu_mesh_ex19_15": {
            "net_direct_ms": net_direct_ms,
            "net_edge_ms": net_edge_ms,
            "net_reduction_pct": net_reduction_pct,
            "ttfa_direct_ms": ttfa_direct_ms,
            "ttfa_edge_ms": ttfa_edge_ms,
            "ttfa_speedup_x": ttfa_speedup_x,
            "bw_compression_x": bw_compression_x,
            "bw_saving_pct": bw_saving_pct,
        },
        "exercises_19_27_and_19_28": {
            "p11_raw_steady_skew_ms": p11_raw_steady_skew_ms,
            "p11_raw_p99_skew_ms": p11_raw_p99_skew_ms,
            "p11_aligned_steady_skew_ms": p11_aligned_steady_skew_ms,
            "p11_aligned_p99_skew_ms": p11_aligned_p99_skew_ms,
            "p11_avatar_ttfa_ms": p11_avatar_ttfa_ms,
            "p12_ttfa_direct_ms": p12_ttfa_direct_ms,
            "p12_ttfa_edge_ms": p12_ttfa_edge_ms,
            "p12_speedup_x": p12_speedup_x,
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
