#!/usr/bin/env python3
"""Experiment 19-05: Barge-in playback-cursor aligned KV rollback, live video interleaved chunking, and 1-hour acoustic token eviction."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"
RESULTS_PATH = HERE / "results" / "summary.json"


def compute_metrics() -> dict:
    # 1. Barge-in Playback-Cursor Aligned KV Rollback (Example 19-12)
    frame_period_ms = 80.0
    tokens_per_frame = 17  # 8 user audio + 8 assistant audio + 1 inner monologue text
    server_gen_frame = 65
    server_gen_time_ms = server_gen_frame * frame_period_ms  # 5,200.0 ms
    client_playback_rtp_ms = 4320.0
    client_playback_frame = int(client_playback_rtp_ms // frame_period_ms)  # Frame 54

    unplayed_lookahead_frames = server_gen_frame - client_playback_frame  # 11 frames
    unplayed_lookahead_ms = round(unplayed_lookahead_frames * frame_period_ms, 2)  # 880.0 ms
    rolled_back_kv_tokens = unplayed_lookahead_frames * tokens_per_frame  # 187 tokens
    resume_frame_index = client_playback_frame + 1  # Frame 55
    kv_pointer_rollback_ms = 0.12

    # 2. Live Video (1 FPS, 256 tokens) + Audio (12.5 Hz) Interleaved Chunked Prefill (Example 19-13)
    audio_step_ms = 11.5
    visual_tokens_per_sec = 256
    all_at_once_vis_prefill_ms = 72.0
    naive_peak_frame_ms = round(audio_step_ms + all_at_once_vis_prefill_ms, 2)  # 83.5 ms (> 80 ms deadline!)
    deadline_violated_naive = naive_peak_frame_ms > frame_period_ms

    interleave_ticks = 8
    vis_tokens_per_tick = visual_tokens_per_sec // interleave_ticks  # 32 tokens/tick
    interleaved_vis_ms = round(all_at_once_vis_prefill_ms / interleave_ticks, 2)  # 9.0 ms/tick
    interleaved_peak_frame_ms = round(audio_step_ms + interleaved_vis_ms, 2)  # 20.5 ms (<< 80 ms)
    peak_frame_reduction_x = round(naive_peak_frame_ms / interleaved_peak_frame_ms, 2)  # 4.07x

    # 3. 1-Hour Continuous Call Rolling Acoustic & Visual Token Eviction (Example 19-13)
    call_duration_sec = 3600
    total_frames = int(call_duration_sec * 12.5)  # 45,000 frames
    full_audio_text_tokens = total_frames * tokens_per_frame  # 765,000 tokens
    full_visual_tokens = call_duration_sec * visual_tokens_per_sec  # 921,600 tokens
    full_multimodal_tokens = full_audio_text_tokens + full_visual_tokens  # 1,686,600 tokens

    sliding_window_sec = 10
    sliding_window_frames = int(sliding_window_sec * 12.5)  # 125 frames
    anchor_tokens_per_frame = 3  # user q1 + assistant q1 + inner monologue text
    residual_acoustic_per_frame = 14  # user q2..q8 + assistant q2..q8
    retained_audio_text_tokens = (
        total_frames * anchor_tokens_per_frame + sliding_window_frames * residual_acoustic_per_frame
    )  # 136,750 tokens
    retained_visual_tokens = sliding_window_sec * visual_tokens_per_sec  # 2,560 tokens
    retained_multimodal_tokens = retained_audio_text_tokens + retained_visual_tokens  # 139,310 tokens

    audio_kv_saving_pct = round((1.0 - retained_audio_text_tokens / full_audio_text_tokens) * 100.0, 2)  # 82.12%
    multimodal_kv_saving_pct = round((1.0 - retained_multimodal_tokens / full_multimodal_tokens) * 100.0, 2)  # 91.74%

    return {
        "experiment_id": "19-05",
        "barge_in_kv_rollback": {
            "server_gen_frame": server_gen_frame,
            "client_playback_frame": client_playback_frame,
            "unplayed_lookahead_frames": unplayed_lookahead_frames,
            "unplayed_lookahead_ms": unplayed_lookahead_ms,
            "rolled_back_kv_tokens": rolled_back_kv_tokens,
            "resume_frame_index": resume_frame_index,
            "kv_pointer_rollback_ms": kv_pointer_rollback_ms,
        },
        "interleaved_video_audio_pacing": {
            "naive_peak_frame_ms": naive_peak_frame_ms,
            "deadline_violated_naive": deadline_violated_naive,
            "vis_tokens_per_tick": vis_tokens_per_tick,
            "interleaved_peak_frame_ms": interleaved_peak_frame_ms,
            "peak_frame_reduction_x": peak_frame_reduction_x,
        },
        "one_hour_acoustic_eviction": {
            "total_frames": total_frames,
            "full_audio_text_tokens": full_audio_text_tokens,
            "full_multimodal_tokens": full_multimodal_tokens,
            "retained_audio_text_tokens": retained_audio_text_tokens,
            "retained_multimodal_tokens": retained_multimodal_tokens,
            "audio_kv_saving_pct": audio_kv_saving_pct,
            "multimodal_kv_saving_pct": multimodal_kv_saving_pct,
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
