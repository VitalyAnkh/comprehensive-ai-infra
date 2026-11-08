#!/usr/bin/env python3
"""Experiment 21-02: Firecracker 12 ms CoW snapshot fork, MCTS synchronized rollback, Locked Case I19 telemetry drain, and MCP Tool RAG."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"
RESULTS_PATH = HERE / "results" / "summary.json"


def compute_metrics() -> dict:
    # 1. Firecracker MicroVM CoW Snapshot Fork & MCTS Tree-of-Actions Synchronized Rollback (Ex 21-3)
    cold_boot_ms = 1500.0
    vmm_clone_ms = 3.2
    uffd_working_set_mib = 5.50
    uffd_bw_mib_s = 625.0
    uffd_hydrate_ms = round(uffd_working_set_mib / uffd_bw_mib_s * 1000.0, 2)  # 8.80 ms
    cow_fork_ms = round(vmm_clone_ms + uffd_hydrate_ms, 2)  # 12.00 ms
    fork_speedup_x = round(cold_boot_ms / cow_fork_ms, 1)  # 125.0x

    mcts_branches = 16
    base_snapshot_mib = 512.0
    dirty_per_branch_mib = 24.0
    full_clone_ram_mib = round(mcts_branches * base_snapshot_mib, 1)  # 8192.0 MiB
    cow_tree_ram_mib = round(base_snapshot_mib + mcts_branches * dirty_per_branch_mib, 1)  # 896.0 MiB
    ram_saving_pct = round((1.0 - cow_tree_ram_mib / full_clone_ram_mib) * 100.0, 2)  # 89.06%

    kv_block_rollback_ms = 0.15
    sandbox_cow_rollback_ms = 8.35
    total_sync_rollback_ms = round(kv_block_rollback_ms + sandbox_cow_rollback_ms, 2)  # 8.50 ms

    # 2. Locked Case I19: Audit Telemetry Outage Backlog Overflow & Recovery Drain (Ex 21-4)
    telemetry_rate_kib_s = 400.0
    buffer_cap_mib = 32.0
    buffer_cap_kib = buffer_cap_mib * 1024.0  # 32,768 KiB
    time_to_overflow_sec = round(buffer_cap_kib / telemetry_rate_kib_s, 2)  # 81.92 s

    outage_duration_sec = 120.0
    accumulated_kib = outage_duration_sec * telemetry_rate_kib_s  # 48,000 KiB
    accumulated_mib = round(accumulated_kib / 1024.0, 3)  # 46.875 MiB
    spilled_to_outbox_mib = round(accumulated_mib - buffer_cap_mib, 3)  # 14.875 MiB

    uplink_bw_mib_s = 1.0
    uplink_bw_kib_s = uplink_bw_mib_s * 1024.0  # 1,024 KiB/s
    net_drain_rate_kib_s = uplink_bw_kib_s - telemetry_rate_kib_s  # 624 KiB/s
    drain_time_sec = round(accumulated_kib / net_drain_rate_kib_s, 2)  # 76.92 s

    # 3. MCP Dynamic Tool RAG & HippoRAG 2 Episodic Memory Consolidation (Ex 21-5)
    total_mcp_tools = 1000
    tokens_per_tool_schema = 180
    full_schema_tokens = total_mcp_tools * tokens_per_tool_schema  # 180,000 tokens
    retrieved_top_k_tools = 8
    rag_schema_tokens = retrieved_top_k_tools * tokens_per_tool_schema  # 1,440 tokens
    schema_reduction_pct = round((1.0 - rag_schema_tokens / full_schema_tokens) * 100.0, 2)  # 99.20%
    schema_compression_x = round(full_schema_tokens / rag_schema_tokens, 1)  # 125.0x

    return {
        "experiment_id": "21-02",
        "firecracker_cow_and_mcts_tree": {
            "cold_boot_ms": cold_boot_ms,
            "cow_fork_ms": cow_fork_ms,
            "fork_speedup_x": fork_speedup_x,
            "mcts_branches": mcts_branches,
            "full_clone_ram_mib": full_clone_ram_mib,
            "cow_tree_ram_mib": cow_tree_ram_mib,
            "ram_saving_pct": ram_saving_pct,
            "total_sync_rollback_ms": total_sync_rollback_ms,
        },
        "locked_case_i19_telemetry_drain": {
            "telemetry_rate_kib_s": telemetry_rate_kib_s,
            "buffer_cap_mib": buffer_cap_mib,
            "time_to_overflow_sec": time_to_overflow_sec,
            "outage_duration_sec": outage_duration_sec,
            "accumulated_mib": accumulated_mib,
            "spilled_to_outbox_mib": spilled_to_outbox_mib,
            "net_drain_rate_kib_s": net_drain_rate_kib_s,
            "drain_time_sec": drain_time_sec,
        },
        "mcp_dynamic_tool_rag": {
            "full_schema_tokens": full_schema_tokens,
            "rag_schema_tokens": rag_schema_tokens,
            "schema_reduction_pct": schema_reduction_pct,
            "schema_compression_x": schema_compression_x,
        },
    }


def main() -> None:
    metrics = compute_metrics()
    payload = json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    EXPECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPECTED_PATH.write_text(payload)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(payload)
    print(payload.strip())


if __name__ == "__main__":
    main()
