#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 17-02."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    assert actual == expected, f"Mismatch:\nactual={actual}\nexpected={expected}"
    assert actual["distrs_and_heavy_tail"]["sunk_work_tokens"] == 216000
    assert actual["distrs_and_heavy_tail"]["estimation_error_ratio"] == 100.0
    assert actual["distrs_and_heavy_tail"]["trimodal_cond_rem_15s"] == 50.0
    assert actual["distrs_and_heavy_tail"]["trimodal_cond_rem_25s"] == 175.0
    assert actual["geo_distributed_async_rl"]["total_tokens"] == 16777216
    assert actual["geo_distributed_async_rl"]["compact_payload_mib"] == 160.0
    assert actual["geo_distributed_async_rl"]["uplink_compression_ratio"] == 14745.6
    assert actual["geo_distributed_async_rl"]["compact_upload_ms"] == 134.2
    assert actual["geo_distributed_async_rl"]["fp8_delta_sync_sec"] == 6.4
    assert actual["prm_cow_tree_search"]["full_copy_token_slots"] == 81920
    assert actual["prm_cow_tree_search"]["cow_shared_token_slots"] == 12288
    assert actual["prm_cow_tree_search"]["cow_memory_saving_pct"] == 85.0
    assert actual["prm_cow_tree_search"]["serial_tot_sec"] == 40.0
    assert actual["prm_cow_tree_search"]["spec_tot_sec"] == 29.6
    print("Experiment 17-02 verified successfully.")


if __name__ == "__main__":
    main()
