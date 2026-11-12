#!/usr/bin/env python3
"""Experiment 17-03: Multi-Turn Agentic RLVR (SWE-RL) CoW Trajectory Tree Forking and V-trace / TIS Off-Policy Clipping."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"
SUMMARY_PATH = HERE / "results" / "summary.json"


def compute_metrics() -> dict:
    # 1. SWE-RL 4x4 CoW Trajectory Tree Forking (Example 17-10 Part 1 & Exercise 17-15)
    b1, b2 = 4, 4
    num_leaves = b1 * b2  # 16
    t_setup_s = 18.0
    t_turn_s = 12.0
    prefix_turns = 2
    leaf_turns = 2

    linear_sandbox_cpu_sec = round(num_leaves * (t_setup_s + prefix_turns * t_turn_s), 2)  # 672.0 s
    forked_sandbox_cpu_sec = round(t_setup_s + b1 * (prefix_turns * t_turn_s), 2)  # 18.0 + 4 * 24.0 = 114.0 s
    sandbox_cpu_speedup = round(linear_sandbox_cpu_sec / forked_sandbox_cpu_sec, 2)  # 5.89x

    linear_total_cpu_sec = round(linear_sandbox_cpu_sec + num_leaves * (leaf_turns * t_turn_s), 2)  # 1056.0 s
    forked_total_cpu_sec = round(forked_sandbox_cpu_sec + num_leaves * (leaf_turns * t_turn_s), 2)  # 498.0 s
    total_cpu_saving_pct = round((1.0 - forked_total_cpu_sec / linear_total_cpu_sec) * 100.0, 2)  # 52.84%

    l_prompt = 4096
    l_turn_act_obs = 2048
    linear_prefix_kv_slots = num_leaves * (l_prompt + prefix_turns * l_turn_act_obs)  # 131,072
    forked_prefix_kv_slots = l_prompt + b1 * (prefix_turns * l_turn_act_obs)  # 4096 + 4 * 4096 = 20,480
    prefix_kv_saving_pct = round((1.0 - forked_prefix_kv_slots / linear_prefix_kv_slots) * 100.0, 2)  # 84.38%

    # 2. Off-policy IS explosion vs. V-trace / TIS clipping (Example 17-10 Part 2 & Exercise 17-16)
    step_is_ratio = 1.25
    horizon = 10
    unclipped_prod_is = round(step_is_ratio ** horizon, 4)  # 9.3132
    unclipped_var_amp = round(unclipped_prod_is ** 2, 2)  # 86.74x

    rho_bar = 1.2
    c_bar = 1.0
    rho_clipped = min(rho_bar, step_is_ratio)  # 1.2
    c_clipped = min(c_bar, step_is_ratio)  # 1.0
    vtrace_prod_c = round(c_clipped ** (horizon - 1), 4)  # 1.0
    vtrace_last_step_target = round(rho_clipped * 1.0, 4)  # 1.2
    vtrace_first_step_target = round(vtrace_last_step_target * vtrace_prod_c, 4)  # 1.2
    variance_amplification_reduction = round(unclipped_prod_is / vtrace_first_step_target, 2)  # 7.76x

    return {
        "experiment_id": "17-03",
        "cow_trajectory_tree_forking": {
            "branch_factor_l1": b1,
            "branch_factor_l2": b2,
            "num_leaves": num_leaves,
            "linear_sandbox_cpu_sec": linear_sandbox_cpu_sec,
            "forked_sandbox_cpu_sec": forked_sandbox_cpu_sec,
            "sandbox_cpu_speedup": sandbox_cpu_speedup,
            "linear_total_cpu_sec": linear_total_cpu_sec,
            "forked_total_cpu_sec": forked_total_cpu_sec,
            "total_cpu_saving_pct": total_cpu_saving_pct,
            "linear_prefix_kv_slots": linear_prefix_kv_slots,
            "forked_prefix_kv_slots": forked_prefix_kv_slots,
            "prefix_kv_saving_pct": prefix_kv_saving_pct,
        },
        "vtrace_and_tis_off_policy": {
            "step_is_ratio": step_is_ratio,
            "horizon_steps": horizon,
            "unclipped_prod_is": unclipped_prod_is,
            "unclipped_var_amp": unclipped_var_amp,
            "vtrace_rho_bar": rho_bar,
            "vtrace_c_bar": c_bar,
            "vtrace_prod_c": vtrace_prod_c,
            "vtrace_first_step_target": vtrace_first_step_target,
            "vtrace_last_step_target": vtrace_last_step_target,
            "variance_amplification_reduction": variance_amplification_reduction,
        },
    }


def main() -> None:
    metrics = compute_metrics()
    for p in (EXPECTED_PATH, SUMMARY_PATH):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
