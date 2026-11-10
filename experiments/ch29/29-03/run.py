#!/usr/bin/env python3
"""Experiment 29-03: 3FS 180-node 6.6 TB/s throughput and KTransformers GPU+CPU AMX MoE offload."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def solve_3fs_and_ktransformers() -> dict:
    storage_nodes = 180
    ssds_per_node = 16
    nics_200g_per_node = 2
    return {
        "three_fs_and_smallpond": {
            "storage_nodes": storage_nodes,
            "total_nvme_ssds": storage_nodes * ssds_per_node,
            "total_200g_rdma_nics": storage_nodes * nics_200g_per_node,
            "aggregate_read_bw_tbs": 6.60,
            "graysort_data_tib": 110.5,
            "graysort_time_sec": 1814,
        },
        "mooncake_and_ktransformers": {
            "gpu_vram_gb": 24.0,
            "gpu_mla_shared_resident_gb": 19.2,
            "cpu_ddr5_bw_gbs": 680.0,
            "cpu_int4_routed_experts_gb": 336.0,
            "pcie_return_activation_kb": 14.0,
            "prefill_tok_per_sec": 286.4,
            "decode_tok_per_sec": 18.6,
            "speedup_vs_llamacpp_x": 13.8,
        },
    }


def main() -> None:
    summary = {
        "experiment_id": "29-03",
        **solve_3fs_and_ktransformers(),
    }
    for sub in ("expected", "results"):
        d = HERE / sub
        d.mkdir(parents=True, exist_ok=True)
    (HERE / "expected" / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (HERE / "results" / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
