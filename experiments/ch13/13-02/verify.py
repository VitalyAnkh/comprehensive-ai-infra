#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 13-02."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH, RESULTS_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    results = json.loads(Path(RESULTS_PATH).read_text())
    assert actual == expected == results, f"Mismatch:\nactual={actual}\nexpected={expected}\nresults={results}"
    assert actual["curation_funnel"]["final_curated_tokens_t"] == 15.0
    assert actual["curation_funnel"]["overall_warc_byte_yield_pct"] == 2.4
    assert actual["minhash_lsh_scurve"]["threshold_s_star"] == 0.7071
    assert actual["minhash_lsh_scurve"]["prob_at_s_080_pct"] == 94.7
    assert actual["minhash_lsh_scurve"]["prob_at_s_050_pct"] == 6.07
    assert actual["gpu_fuzzy_dedup_cluster"]["total_signature_gb"] == 2560.0
    assert actual["gpu_fuzzy_dedup_cluster"]["all_to_all_shuffle_gb"] == 2520.0
    assert actual["gpu_fuzzy_dedup_cluster"]["dedup_speedup_x"] == 80.0
    assert actual["gpu_fuzzy_dedup_cluster"]["tco_reduction_x"] == 4.5
    print("Experiment 13-02 verified successfully.")


if __name__ == "__main__":
    main()
