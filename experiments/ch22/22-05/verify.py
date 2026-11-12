#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 22-05."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH, RESULTS_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    results = json.loads(Path(RESULTS_PATH).read_text())
    assert actual == expected == results, f"Mismatch:\nactual={actual}\nexpected={expected}\nresults={results}"
    coup = actual["insitu_dlpack_coupling"]
    assert coup["coupled_field_gb"] == 384.0
    assert coup["per_gpu_field_mb"] == 375.0
    assert coup["filesystem_staged_s"] == 58.2
    assert coup["dlpack_insitu_s"] == 0.41
    assert coup["coupling_insitu_speedup_x"] == 142.0
    cons = actual["conservation_law_projection"]
    assert cons["unconstrained_500step_mass_drift_pct"] == 11.4
    assert cons["projected_500step_mass_drift_rel"] == 4.2e-13
    assert cons["conservation_projection_overhead_pct"] == 3.1
    assert cons["numerical_projection_rel_error_below_1e12"] is True
    print("Experiment 22-05 verified successfully.")


if __name__ == "__main__":
    main()
