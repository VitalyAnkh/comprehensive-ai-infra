#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 19-02."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    assert actual == expected, f"Mismatch:\nactual={actual}\nexpected={expected}"
    assert actual["epdg_disaggregation"]["single_img_opt_tput"] == 12.0
    assert actual["epdg_disaggregation"]["effective_mu_e_at_75pct"] == 48.0
    assert actual["epdg_disaggregation"]["tput_4img_h0"]["2E_2PD"] == 6.0
    assert actual["epdg_disaggregation"]["tput_4img_h75"]["1E_3PD"] == 9.16
    assert actual["epdg_disaggregation"]["warm_demotion_saving_pct"] == 86.11
    assert actual["transfusion_hybrid_mask"]["active_token_pairs"] == 10485760
    assert actual["transfusion_hybrid_mask"]["active_density_pct"] == 62.5
    assert actual["transfusion_hybrid_mask"]["executed_tiles"] == 648
    assert actual["transfusion_hybrid_mask"]["tile_effective_occupancy_pct"] == 98.77
    assert actual["transfusion_hybrid_mask"]["prefix_kv_handoff_mib"] == 144.0
    assert actual["transfusion_hybrid_mask"]["kv_handoff_ms"] == 2.88
    assert actual["hybrid_3dgs_dit_world_model"]["total_frame_ms"] == 30.9
    assert actual["hybrid_3dgs_dit_world_model"]["steady_fps"] == 32.36
    print("Experiment 19-02 verified successfully.")


if __name__ == "__main__":
    main()
