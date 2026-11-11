#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 14-04."""
from __future__ import annotations

import json
from pathlib import Path
from run import EXPECTED_PATH, compute_metrics


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    assert actual == expected, f"Mismatch:\nactual={actual}\nexpected={expected}"
    assert actual["hvdc_and_vpd"]["busbar_loss_48v_watt"] == 3781.25
    assert actual["hvdc_and_vpd"]["busbar_loss_800v_watt"] == 13.6125
    assert actual["hvdc_and_vpd"]["hvdc_i2r_loss_reduction_x"] == 277.78
    assert actual["hvdc_and_vpd"]["busbar_loss_800v_165kw_watt"] == 21.27
    assert actual["hvdc_and_vpd"]["vpd_72gpu_saving_kw"] == 13.06
    assert actual["ashrae_w45_cooling"]["w45_flow_l_per_min_132kw"] == 159.5
    assert actual["ashrae_w45_cooling"]["coldplate_1400w_case_temp_c"] == 74.2
    print("Experiment 14-04 verified successfully.")


if __name__ == "__main__":
    main()
