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
    assert actual["evo2_flashfftconv_1m"]["quadratic_attn_tflops"] == 36028.8
    assert actual["evo2_flashfftconv_1m"]["flash_fft_conv_gflops"] == 1735.17
    assert actual["evo2_flashfftconv_1m"]["seq_mixer_flop_reduction_x"] == 20763.8
    assert actual["sfno_2d_halo_overlap"]["halo_volume_mb"] == 5.53
    assert actual["sfno_2d_halo_overlap"]["comm_to_comp_ratio_pct"] == 1.27
    assert actual["sfno_2d_halo_overlap"]["exposed_halo_ms"] == 0.0
    ledger = actual["five_domain_scientific_ledger"]
    assert ledger["science_domains_covered"] == 5
    assert ledger["pairformer_workspace_reduction_x"] == 2048.0
    assert ledger["pairformer_dap_comm_to_comp_pct"] == 8.26
    assert ledger["ozaki_s7_vs_native_fp64_speedup_x"] == 3.68
    assert ledger["ozaki_equiv_fp64_tflops"] == 147.2
    assert ledger["tokamak_opt_wcet_us"] == 68.5
    print("Experiment 22-02 verified successfully.")


if __name__ == "__main__":
    main()
