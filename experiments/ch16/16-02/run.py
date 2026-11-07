#!/usr/bin/env python3
"""Experiment 16-02: DSec sandbox memory ledger, virtio-pmem DAX struct page tax, and I19 audit drain."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"


def compute_metrics() -> dict:
    # 1. DSec scale & virtio-pmem DAX memory ledger (Exercise 16-4 & 16-13)
    cell_nodes = 160
    peak_sandboxes = 380000
    peak_create_per_s = 5000
    avg_sandboxes_per_node = peak_sandboxes // cell_nodes
    struct_page_ratio = 64 / 4096  # 1/64

    # Exercise 16-13: 800 microVMs sharing a 64 GB read-only image (4.0 GB hot subset per VM)
    n_vms = 800
    ro_image_gb = 64.0
    hot_subset_per_vm_gb = 4.0
    naive_full_duplication_gb = (n_vms - 1) * ro_image_gb
    guest_struct_page_per_vm_gb = ro_image_gb * struct_page_ratio
    total_guest_struct_page_gb = n_vms * guest_struct_page_per_vm_gb
    ro_no_dax_total_gb = ro_image_gb + n_vms * hot_subset_per_vm_gb  # 3,264.0 GB
    ro_with_dax_total_gb = ro_image_gb + total_guest_struct_page_gb  # 864.0 GB
    net_host_ram_saved_gb = ro_no_dax_total_gb - ro_with_dax_total_gb  # 2,400.0 GB

    # 2. I19 audit log outage backlog and net drain time (Example 16-6 & Exercise 16-10)
    r_in_kibs = 10 * 20 * 2  # 400 KiB/s
    t_out_s = 120.0
    backlog_kib = r_in_kibs * t_out_s  # 48,000 KiB
    backlog_mib = backlog_kib / 1024.0  # 46.875 MiB
    t_overflow_32m_s = round((32 * 1024) / r_in_kibs, 4)  # 81.92 s
    r_net_kibs = 1024 - r_in_kibs  # 624 KiB/s
    t_drain_s = round(backlog_kib / r_net_kibs, 4)  # 76.9231 s

    return {
        "experiment_id": "16-02",
        "dsec_and_dax": {
            "cell_nodes": cell_nodes,
            "peak_sandboxes": peak_sandboxes,
            "peak_create_per_s": peak_create_per_s,
            "avg_sandboxes_per_node": avg_sandboxes_per_node,
            "struct_page_ratio": struct_page_ratio,
            "n_vms_ex16_13": n_vms,
            "ro_image_gb": ro_image_gb,
            "hot_subset_per_vm_gb": hot_subset_per_vm_gb,
            "naive_full_duplication_saved_gb": naive_full_duplication_gb,
            "guest_struct_page_per_vm_gb": guest_struct_page_per_vm_gb,
            "total_guest_struct_page_gb": total_guest_struct_page_gb,
            "ro_no_dax_total_gb": ro_no_dax_total_gb,
            "ro_with_dax_total_gb": ro_with_dax_total_gb,
            "net_host_ram_saved_gb": net_host_ram_saved_gb,
        },
        "i19_audit_drain": {
            "r_in_kibs": r_in_kibs,
            "backlog_kib": backlog_kib,
            "backlog_mib": backlog_mib,
            "t_overflow_32m_s": t_overflow_32m_s,
            "r_net_kibs": r_net_kibs,
            "t_drain_s": t_drain_s,
        },
    }


def main() -> None:
    metrics = compute_metrics()
    EXPECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPECTED_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
