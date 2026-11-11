#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 14-05."""
from __future__ import annotations

import json
from pathlib import Path
from run import EXPECTED_PATH, compute_metrics


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    assert actual == expected, f"Mismatch:\nactual={actual}\nexpected={expected}"
    dci = actual["controlled_optical_dci"]
    assert dci["dci_671b_naive_per_step_gbps"] == 2556.2
    assert dci["dci_671b_diloco_h500_avg_gbps"] == 5.11
    assert dci["dci_671b_outer_sync_seconds"] == 31.58
    assert dci["dci_blocking_comm_overhead_pct"] == 1.48
    assert dci["streaming_exposed_stall_seconds"] == 0.0
    eco = actual["campus_economics"]
    assert eco["campus_100mw_elec_saving_musd_per_yr"] == 70.08
    assert eco["bess_20pct_2hr_capacity_mwh"] == 40.0
    print("Experiment 14-05 verified successfully.")


if __name__ == "__main__":
    main()
