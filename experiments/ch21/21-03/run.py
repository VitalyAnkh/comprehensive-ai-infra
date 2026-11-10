#!/usr/bin/env python3
"""Experiment 21-03: GUI incremental ROI vision token cache + Set-of-Marks DOM pruning & multi-agent DAG speculative tool execution."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"
RESULTS_PATH = HERE / "results" / "summary.json"


def compute_metrics() -> dict:
    # 1. Worked Example 21-6: 30-step GUI Incremental ROI Vision Token Cache & SoM DOM Pruning
    steps_ex6 = 30
    full_vis_tokens_ex6 = 2400
    raw_dom_tokens_ex6 = 3600
    dirty_tile_ratio_ex6 = 0.125
    pruned_dom_tokens_ex6 = 450
    full_vit_ms_ex6 = 45.0
    hash_overhead_ms_ex6 = 1.50

    naive_total_tokens_ex6 = steps_ex6 * (full_vis_tokens_ex6 + raw_dom_tokens_ex6)  # 180,000
    inc_vis_tokens_ex6 = int(round(full_vis_tokens_ex6 * dirty_tile_ratio_ex6))  # 300
    opt_total_tokens_ex6 = (full_vis_tokens_ex6 + pruned_dom_tokens_ex6) + (steps_ex6 - 1) * (
        inc_vis_tokens_ex6 + pruned_dom_tokens_ex6
    )  # 24,600
    token_compression_x_ex6 = round(naive_total_tokens_ex6 / opt_total_tokens_ex6, 2)  # 7.32
    token_saving_pct_ex6 = round(100.0 * (1.0 - opt_total_tokens_ex6 / naive_total_tokens_ex6), 2)  # 86.33

    naive_vit_total_ms_ex6 = round(steps_ex6 * full_vit_ms_ex6, 2)  # 1350.00
    inc_vit_step_ms_ex6 = full_vit_ms_ex6 * dirty_tile_ratio_ex6 + hash_overhead_ms_ex6  # 7.125
    opt_vit_total_ms_ex6 = round(full_vit_ms_ex6 + (steps_ex6 - 1) * inc_vit_step_ms_ex6, 2)  # 251.62
    vit_speedup_x_ex6 = round(naive_vit_total_ms_ex6 / (full_vit_ms_ex6 + (steps_ex6 - 1) * inc_vit_step_ms_ex6), 2)  # 5.37

    # 2. Worked Example 21-7: 4-Stage Multi-Agent DAG Speculative Tool Execution
    stages_ex7 = 4
    t_llm_ex7 = 600.0
    t_tool_ex7 = 1400.0
    t_draft_ex7 = 100.0
    t_pipe_overlap_ex7 = 450.0
    t_rollback_ex7 = 12.0
    hit_prob_ex7 = 0.80

    seq_total_ms_ex7 = round(stages_ex7 * (t_llm_ex7 + t_tool_ex7), 2)  # 8000.00
    hit_stage_ms_ex7 = round(t_draft_ex7 + t_tool_ex7 - t_pipe_overlap_ex7, 2)  # 1050.00
    miss_stage_ms_ex7 = round(t_llm_ex7 + t_rollback_ex7 + t_tool_ex7, 2)  # 2012.00
    expected_stage_ms_ex7 = round(hit_prob_ex7 * hit_stage_ms_ex7 + (1.0 - hit_prob_ex7) * miss_stage_ms_ex7, 2)  # 1242.40
    expected_total_ms_ex7 = round(stages_ex7 * expected_stage_ms_ex7, 2)  # 4969.60
    dag_speedup_x_ex7 = round(seq_total_ms_ex7 / expected_total_ms_ex7, 2)  # 1.61
    latency_reduction_pct_ex7 = round(100.0 * (1.0 - expected_total_ms_ex7 / seq_total_ms_ex7), 2)  # 37.88
    breakeven_hit_prob_pct_ex7 = round(100.0 * t_rollback_ex7 / (miss_stage_ms_ex7 - hit_stage_ms_ex7), 2)  # 1.25

    # 3. Exercise 21-11 & Exercise 21-12
    steps_p11 = 40
    naive_tokens_p11 = steps_p11 * (3200 + 4800)  # 320,000
    opt_tokens_p11 = (3200 + 500) + (steps_p11 - 1) * (320 + 500)  # 35,680
    token_comp_p11 = round(naive_tokens_p11 / opt_tokens_p11, 2)  # 8.97
    naive_vit_ms_p11 = round(steps_p11 * 60.0, 2)  # 2400.00
    opt_vit_ms_p11 = round(60.0 + (steps_p11 - 1) * (60.0 * 0.10 + 2.0), 2)  # 372.00
    vit_speedup_p11 = round(naive_vit_ms_p11 / opt_vit_ms_p11, 2)  # 6.45

    seq_total_ms_p12 = round(5 * (500.0 + 1500.0), 2)  # 10000.00
    exp_stage_ms_p12 = round(0.75 * 1100.0 + 0.25 * 2015.0, 2)  # 1328.75
    exp_total_ms_p12 = round(5 * exp_stage_ms_p12, 2)  # 6643.75
    dag_speedup_p12 = round(seq_total_ms_p12 / exp_total_ms_p12, 2)  # 1.51

    return {
        "experiment_id": "21-03",
        "gui_incremental_roi_cache_ex21_6": {
            "steps": steps_ex6,
            "naive_total_tokens": naive_total_tokens_ex6,
            "opt_total_tokens": opt_total_tokens_ex6,
            "token_compression_x": token_compression_x_ex6,
            "token_saving_pct": token_saving_pct_ex6,
            "naive_vit_total_ms": naive_vit_total_ms_ex6,
            "opt_vit_total_ms": opt_vit_total_ms_ex6,
            "vit_speedup_x": vit_speedup_x_ex6,
        },
        "speculative_dag_execution_ex21_7": {
            "num_stages": stages_ex7,
            "seq_total_ms": seq_total_ms_ex7,
            "hit_stage_ms": hit_stage_ms_ex7,
            "miss_stage_ms": miss_stage_ms_ex7,
            "expected_stage_ms": expected_stage_ms_ex7,
            "expected_total_ms": expected_total_ms_ex7,
            "dag_speedup_x": dag_speedup_x_ex7,
            "latency_reduction_pct": latency_reduction_pct_ex7,
            "breakeven_hit_prob_pct": breakeven_hit_prob_pct_ex7,
        },
        "exercises_21_11_and_21_12": {
            "p11_naive_tokens": naive_tokens_p11,
            "p11_opt_tokens": opt_tokens_p11,
            "p11_token_compression_x": token_comp_p11,
            "p11_naive_vit_ms": naive_vit_ms_p11,
            "p11_opt_vit_ms": opt_vit_ms_p11,
            "p11_vit_speedup_x": vit_speedup_p11,
            "p12_seq_total_ms": seq_total_ms_p12,
            "p12_expected_total_ms": exp_total_ms_p12,
            "p12_dag_speedup_x": dag_speedup_p12,
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
