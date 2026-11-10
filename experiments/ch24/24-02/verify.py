#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 24-02."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH, RESULTS_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    results = json.loads(Path(RESULTS_PATH).read_text())
    assert actual == expected == results, f"Mismatch:\nactual={actual}\nexpected={expected}\nresults={results}"
    assert actual["evo2_flashfftconv_1m"]["quadratic_attn_tflops"] == 36028.8
    assert actual["evo2_flashfftconv_1m"]["flash_fft_conv_gflops"] == 1735.17
    assert actual["evo2_flashfftconv_1m"]["seq_mixer_flop_reduction_x"] == 20763.8
    assert actual["sfno_2d_halo_overlap"]["halo_volume_mb"] == 5.53
    assert actual["sfno_2d_halo_overlap"]["comm_to_comp_ratio_pct"] == 1.27
    assert actual["sfno_2d_halo_overlap"]["exposed_halo_ms"] == 0.0
    assert actual["grand_encyclopedia_ledger"]["total_chapters"] == 24
    assert actual["grand_encyclopedia_ledger"]["total_chapter_figures"] == 883
    assert actual["grand_encyclopedia_ledger"]["total_production_code_blocks"] == 166
    assert actual["grand_encyclopedia_ledger"]["total_worked_examples"] == 192
    assert actual["grand_encyclopedia_ledger"]["total_exercise_blocks"] == 432
    assert actual["grand_encyclopedia_ledger"]["total_unique_exercise_ids"] == 431
    assert actual["grand_encyclopedia_ledger"]["total_experiment_suites"] == 147
    print("Experiment 24-02 verified successfully.")


if __name__ == "__main__":
    main()
