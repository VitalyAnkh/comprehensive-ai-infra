#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 19-02."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH, RESULTS_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    results = json.loads(Path(RESULTS_PATH).read_text())
    assert actual == expected == results, f"Mismatch:\nactual={actual}\nexpected={expected}\nresults={results}"
    assert actual["three_tier_humanoid_wbc"]["tier3_wcet_us"] == 800.0
    assert actual["three_tier_humanoid_wbc"]["tier3_margin_us"] == 200.0
    assert actual["three_tier_humanoid_wbc"]["slip_disp_100hz_mm"] == 1.80
    assert actual["three_tier_humanoid_wbc"]["slip_disp_1000hz_mm"] == 0.12
    assert actual["three_tier_humanoid_wbc"]["slip_reduction_x"] == 15.0
    assert actual["fast_dct_action_tokenizer"]["naive_tokens"] == 700
    assert actual["fast_dct_action_tokenizer"]["fast_bpe_tokens"] == 80
    assert actual["fast_dct_action_tokenizer"]["compression_ratio_x"] == 8.75
    assert actual["fast_dct_action_tokenizer"]["token_reduction_pct"] == 88.57
    assert actual["fast_dct_action_tokenizer"]["ar_decode_naive_ms"] == 2450.0
    assert actual["fast_dct_action_tokenizer"]["ar_decode_fast_ms"] == 280.0
    assert actual["fast_dct_action_tokenizer"]["energy_retention_pct"] >= 99.4
    assert actual["fast_dct_action_tokenizer"]["recon_rmse_mm"] < 0.35
    assert actual["ptp_alignment_and_cbf_shield"]["software_skew_ms"] == 27.0
    assert actual["ptp_alignment_and_cbf_shield"]["ptp_interp_skew_ms"] == 0.018
    assert actual["ptp_alignment_and_cbf_shield"]["software_pos_err_mm"] == 16.20
    assert actual["ptp_alignment_and_cbf_shield"]["ptp_pos_err_mm"] == 0.011
    assert actual["ptp_alignment_and_cbf_shield"]["v_safe_at_15mm"] == 0.30
    assert actual["ptp_alignment_and_cbf_shield"]["post_cbf_margin"] == 0.0
    assert actual["ptp_alignment_and_cbf_shield"]["safety_violations_500steps"] == 0
    assert actual["ptp_alignment_and_cbf_shield"]["min_dist_mm"] == 1.20
    print("Experiment 19-02 verified successfully.")


if __name__ == "__main__":
    main()
