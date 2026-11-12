#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 16-02."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    assert actual == expected, f"Mismatch:\nactual={actual}\nexpected={expected}"
    assert actual["dsec_and_dax"]["avg_sandboxes_per_node"] == 2375
    assert actual["dsec_and_dax"]["guest_struct_page_per_vm_gb"] == 1.0
    assert actual["dsec_and_dax"]["ro_no_dax_total_gb"] == 3264.0
    assert actual["dsec_and_dax"]["ro_with_dax_total_gb"] == 864.0
    assert actual["dsec_and_dax"]["net_host_ram_saved_gb"] == 2400.0
    assert actual["audit_log_drain"]["backlog_mib"] == 46.875
    assert actual["audit_log_drain"]["t_overflow_32m_s"] == 81.92
    assert abs(actual["audit_log_drain"]["t_drain_s"] - 76.9231) < 1e-4
    print("Experiment 16-02 verified successfully.")


if __name__ == "__main__":
    main()
