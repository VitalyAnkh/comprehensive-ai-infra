#!/usr/bin/env python3
"""Experiment 15-05: Collective Wait-Chain Root-Cause Attribution (Mycroft), Flight Recorder Forensics & Monarch Isolation."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_PATH = ROOT / "expected" / "metrics.json"
RESULTS_PATH = ROOT / "results" / "summary.json"


def analyze_wait_chain_8rank() -> dict:
    # Worked Example 15-9: 8-Rank (TP=2, PP=2, DP=2) collective barrier timeline
    enter_ms = {0: 60.0, 1: 60.0, 2: 60.0, 3: 60.0, 4: 62.0, 5: 200.0, 6: 60.0, 7: 62.0}
    wire_xfer_ms = 4.0
    exit_ms = max(enter_ms.values()) + wire_xfer_ms  # 204.0 ms

    root_rank = max(enter_ms, key=lambda r: enter_ms[r])
    earliest_ms = min(enter_ms.values())
    root_host_stall_ms = round(enter_ms[root_rank] - earliest_ms, 1)  # 140.0 ms
    apparent_nccl_ms = {str(r): round(exit_ms - t, 1) for r, t in enter_ms.items()}
    victim_ranks = [r for r in enter_ms if r != root_rank]

    return {
        "root_cause_rank": root_rank,
        "victim_ranks_stripped": len(victim_ranks),
        "root_enter_ms": enter_ms[root_rank],
        "collective_exit_ms": exit_ms,
        "true_wire_transfer_ms": wire_xfer_ms,
        "root_host_stall_ms": root_host_stall_ms,
        "root_apparent_nccl_ms": apparent_nccl_ms[str(root_rank)],
        "max_victim_apparent_nccl_ms": max(apparent_nccl_ms[str(r)] for r in victim_ranks),
        "apparent_nccl_ms_by_rank": apparent_nccl_ms,
    }


def analyze_monarch_vs_spmd() -> dict:
    # Worked Example 15-10: PyTorch Monarch actor-mesh isolation vs SPMD torchrun global restart
    t_detect_s = 0.12
    t_rebind_s = 0.28
    t_rdma_hydrate_s = 0.45
    monarch_isolation_s = round(t_detect_s + t_rebind_s + t_rdma_hydrate_s, 2)  # 0.85 s

    t_watchdog_s = 28.0
    t_comm_init_s = 76.0
    t_ckpt_load_s = 38.0
    spmd_restart_s = round(t_watchdog_s + t_comm_init_s + t_ckpt_load_s, 1)    # 142.0 s
    speedup_x = round(spmd_restart_s / monarch_isolation_s, 1)                  # 167.1x

    return {
        "monarch_isolation_seconds": monarch_isolation_s,
        "spmd_global_restart_seconds": spmd_restart_s,
        "monarch_vs_spmd_speedup_x": speedup_x,
    }


def analyze_forensics_capstones() -> dict:
    # Exercise 15-17: Multi-hop Kineto flamegraph propagation (Rank 2 H2D PCIe Gen2 downgrade +85 ms)
    # Stage 0 TP={2,3}: Rank 2 enters TP AllReduce at 115.0 ms (30 ms normal + 85 ms H2D stall), Rank 3 at 30.0 ms; TP wire=2.0 ms -> exit 117.0 ms
    # PP Send Rank 2 -> Rank 6 wire=3.0 ms -> Rank 6 receives at 120.0 ms (instead of normal 35.0 ms), forward+backward 40.0 ms -> enters DP AllReduce at 160.0 ms
    # Healthy DP peers (Stage 1 Replica 0: Ranks 0/1 -> Stage 1 Ranks 4/5 in 0-indexed Stage1={4,5,6,7}): enter DP AllReduce at 75.0 ms; DP wire=5.0 ms -> exit 165.0 ms
    rank3_tp_wait_ms = 115.0 - 30.0   # 85.0 ms
    rank6_pp_wait_ms = 120.0 - 35.0   # 85.0 ms
    healthy_dp_wait_ms = 160.0 - 75.0  # 85.0 ms
    step_finish_ms = 165.0

    # Exercise 15-18: NCCL Flight Recorder (K=2048 ring buffer) deadlock & transport hang decoding
    ring_capacity = 2048
    incident_a_ops = {r: ("AllGather" if r == 4 else "AllReduce") for r in range(8)}
    distinct_ops = sorted(set(incident_a_ops.values()))
    incident_a_diagnosis = "COLL_TYPE_MISMATCH_DEADLOCK" if len(distinct_ops) > 1 else "HEALTHY"

    incident_b_seq = 2099
    incident_b_slot = incident_b_seq % ring_capacity  # 51
    coll_rate_per_sec = 320.0
    ring_retention_sec = round(ring_capacity / coll_rate_per_sec, 2)  # 6.40 s

    return {
        "ex15_17_kineto_multihop": {
            "root_cause_rank": 2,
            "rank3_tp_victim_wait_ms": rank3_tp_wait_ms,
            "rank6_pp_victim_wait_ms": rank6_pp_wait_ms,
            "healthy_dp_victim_wait_ms": healthy_dp_wait_ms,
            "corrupted_step_ms": step_finish_ms,
        },
        "ex15_18_flight_recorder": {
            "ring_capacity_entries": ring_capacity,
            "incident_a_divergent_rank": 4,
            "incident_a_diagnosis": incident_a_diagnosis,
            "incident_b_seq": incident_b_seq,
            "incident_b_ring_slot": incident_b_slot,
            "ring_retention_window_sec": ring_retention_sec,
        },
    }


def compute_metrics() -> dict:
    return {
        "experiment_id": "15-05",
        "wait_chain_8rank": analyze_wait_chain_8rank(),
        "monarch_vs_spmd": analyze_monarch_vs_spmd(),
        "forensics_capstones": analyze_forensics_capstones(),
    }


def main() -> None:
    metrics = compute_metrics()
    EXPECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    for p in (EXPECTED_PATH, RESULTS_PATH):
        p.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
