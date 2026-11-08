#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 22-02."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH, RESULTS_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    results = json.loads(Path(RESULTS_PATH).read_text())
    assert actual == expected == results, f"Mismatch:\nactual={actual}\nexpected={expected}\nresults={results}"
    assert actual["barge_in_kv_rollback"]["unplayed_lookahead_frames"] == 11
    assert actual["barge_in_kv_rollback"]["unplayed_lookahead_ms"] == 880.0
    assert actual["barge_in_kv_rollback"]["rolled_back_kv_tokens"] == 187
    assert actual["barge_in_kv_rollback"]["resume_frame_index"] == 55
    assert actual["interleaved_video_audio_pacing"]["naive_peak_frame_ms"] == 83.5
    assert actual["interleaved_video_audio_pacing"]["deadline_violated_naive"] is True
    assert actual["interleaved_video_audio_pacing"]["interleaved_peak_frame_ms"] == 20.5
    assert actual["one_hour_acoustic_eviction"]["audio_kv_saving_pct"] == 82.12
    assert actual["one_hour_acoustic_eviction"]["multimodal_kv_saving_pct"] == 91.74
    print("Experiment 22-02 verified successfully.")


if __name__ == "__main__":
    main()
