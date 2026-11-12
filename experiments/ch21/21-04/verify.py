#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 21-04."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH, RESULTS_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text(encoding="utf-8"))
    results = json.loads(Path(RESULTS_PATH).read_text(encoding="utf-8"))
    assert actual == expected == results, f"Mismatch:\nactual={actual}\nexpected={expected}\nresults={results}"
    cp = actual["colpali_plaid_compression"]
    hb = actual["hybrid_rerank_budget"]
    assert cp["colpali_bf16_kb_per_page"] == 263.68
    assert cp["colpali_50m_bf16_tb"] == 13.18
    assert cp["plaid_2bit_50m_tb"] == 1.75 and cp["plaid_2bit_compression_x"] == 7.53
    assert cp["plaid_1bit_50m_gb"] == 901.3 and cp["plaid_1bit_compression_x"] == 14.6
    assert cp["plaid_maxsim_p99_ms"] == 14.2
    assert hb["hybrid_bm25_cuvs_ce_p99_ms"] == 27.6
    print("Experiment 21-04 verified successfully.")


if __name__ == "__main__":
    main()
