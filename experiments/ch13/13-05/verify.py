#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 13-05."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH, RESULTS_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    results = json.loads(Path(RESULTS_PATH).read_text())
    assert actual == expected == results, f"Mismatch:\nactual={actual}\nexpected={expected}\nresults={results}"

    pdf = actual["mineru_olmocr_pdf_pipeline"]
    assert pdf["avg_gpu_ms_per_page"] == 17.72
    assert pdf["throughput_pages_per_s_gpu"] == 56.43
    assert pdf["speedup_vs_monolithic_vlm_x"] == 35.27
    assert pdf["speedup_vs_serial_mineru_x"] == 4.51
    assert pdf["total_gpu_hours"] == 2461.11
    assert pdf["total_cost_usd"] == 5414.44
    assert pdf["cost_per_1m_tokens_usd"] == 0.0135

    ast_pack = actual["repo_ast_topological_packing"]
    assert ast_pack["alphabetical_causal_coverage_pct"] == 42.5
    assert ast_pack["topological_causal_coverage_pct"] == 95.0
    assert ast_pack["coverage_gain_pct_points"] == 52.5
    assert ast_pack["ppl_reduction_pct"] == 39.85
    assert ast_pack["packing_efficiency_pct"] == 98.8

    print("Experiment 13-05 verified successfully.")


if __name__ == "__main__":
    main()
