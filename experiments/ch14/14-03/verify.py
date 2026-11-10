#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 14-03."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    assert actual == expected, f"Mismatch:\nactual={actual}\nexpected={expected}"
    assert actual["in_silicon_microfluidics"]["heat_flux_w_cm2"] == 277.78
    assert actual["in_silicon_microfluidics"]["r_micro_total_kw"] == 0.01
    assert actual["in_silicon_microfluidics"]["t_j_micro_c"] == 70.0
    assert actual["in_silicon_microfluidics"]["pump_power_w"] == 1.65
    assert actual["gw_microgrid_vsg"]["h_phys_s"] == 3.0
    assert actual["gw_microgrid_vsg"]["h_with_vsg_s"] == 7.5
    assert actual["gw_microgrid_vsg"]["rocof_with_vsg_hz_s"] == 1.5
    assert actual["gw_microgrid_vsg"]["h_bess_min_s"] == 13.5
    assert actual["electro_thermal_dvfs"]["effective_step_mw"] == 144.0
    assert actual["electro_thermal_dvfs"]["step_overhead_pct"] == 1.43
    print("Experiment 14-03 verified successfully.")


if __name__ == "__main__":
    main()
