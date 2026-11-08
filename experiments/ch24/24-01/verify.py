#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 24-01."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH, RESULTS_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    results = json.loads(Path(RESULTS_PATH).read_text())
    assert actual == expected == results, f"Mismatch:\nactual={actual}\nexpected={expected}\nresults={results}"
    assert actual["two_stage_msa_disaggregation"]["msa_speedup_x"] == 2250.0
    assert actual["two_stage_msa_disaggregation"]["end_to_end_speedup_x"] == 55.85
    assert actual["two_stage_msa_disaggregation"]["stage2_gpu_utilization_pct"] == 97.56
    assert actual["pairformer_triangle_attention_dap"]["pair_tensor_gib"] == 4.0
    assert actual["pairformer_triangle_attention_dap"]["naive_tri_attn_logits_gib"] == 512.0
    assert actual["pairformer_triangle_attention_dap"]["attn_workspace_reduction_x"] == 2048.0
    print("Experiment 24-01 verified successfully.")


if __name__ == "__main__":
    main()
