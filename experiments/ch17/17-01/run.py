#!/usr/bin/env python3
"""Experiment 17-01: RLVR memory ledger, Weave/RobustRL recovery, RollPacker bias & normalization, and DistRS."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"


def compute_metrics() -> dict:
    # 1. PPO 4-model vs GRPO 2-model state memory for a 32B model
    psi_b = 32
    train_model_gb = 16 * psi_b  # 512 GB (2B param + 2B grad + 12B FP32 master/Adam)
    infer_model_gb = 2 * psi_b  # 64 GB (BF16 weights)
    ppo_total_state_gb = 2 * train_model_gb + 2 * infer_model_gb  # 1,152 GB (36 * Psi)
    grpo_total_state_gb = train_model_gb + infer_model_gb  # 576 GB (18 * Psi)
    grpo_state_saving_pct = round(
        (ppo_total_state_gb - grpo_total_state_gb) / ppo_total_state_gb * 100.0, 2
    )  # 50.0%

    # 2. Weave complementary RL sharing & RobustRL fault isolation (case-studies/rl-scheduling-and-recovery.md)
    weave_cycle_s = 360 + 120  # 480 s
    weave_peak_h20 = 216
    weave_peak_h800 = 152
    robustrl_full_redo_s = 300 + 30 + 20 + 80 + 300 + 60  # 790 s
    robustrl_preserve_rollout_s = 300 + 30 + 20 + 80 + 60  # 490 s
    robustrl_saved_s = robustrl_full_redo_s - robustrl_preserve_rollout_s  # 300 s

    # 3. RollPacker speculative pool, length bias, and gradient normalization (case-studies/rollout-tail-and-sampling.md)
    target_responses = 128 * 8  # 1,024
    speculative_responses = 160 * 10  # 1,600
    speculative_headroom_pct = round(
        (speculative_responses - target_responses) / target_responses * 100.0, 2
    )  # 56.25%
    long_prob_single_draw = 0.5
    long_prob_two_slot_first = 0.25  # P(both slots draw long response) = (1/2)^2 = 1/4
    serial_step_s = 40 + 16 + 4  # 60 s
    stream_trainer_step_s = max(43, 29) + 8 + 4  # 55 s
    per_microbatch_avg = (1.0 + 3.0) / 2.0  # 2.0 (biased unweighted average of microbatch means)
    true_token_weighted_avg = (1.0 + 9.0) / 4.0  # 2.5 (unbiased global average of {1, 3, 3, 3})

    # 4. DistRS conditional remaining service time & 216,000 token sunk-work boundary
    tasks_s = [1.0] * 9 + [100.0]
    unconditional_mean_s = round(sum(tasks_s) / len(tasks_s), 2)  # 10.9 s
    elapsed_s = 10.0
    naive_remaining_s = round(max(unconditional_mean_s - elapsed_s, 0.0), 2)  # 0.9 s
    survivors = [t - elapsed_s for t in tasks_s if t > elapsed_s]
    conditional_remaining_s = round(sum(survivors) / len(survivors), 2)  # 90.0 s
    sunk_work_tokens = 6 * 60 * 600  # 216,000 tokens

    return {
        "experiment_id": "17-01",
        "ppo_vs_grpo_32b": {
            "ppo_total_state_gb": ppo_total_state_gb,
            "grpo_total_state_gb": grpo_total_state_gb,
            "grpo_state_saving_pct": grpo_state_saving_pct,
        },
        "weave_and_robustrl": {
            "weave_cycle_s": weave_cycle_s,
            "weave_peak_h20": weave_peak_h20,
            "weave_peak_h800": weave_peak_h800,
            "robustrl_full_redo_s": robustrl_full_redo_s,
            "robustrl_preserve_rollout_s": robustrl_preserve_rollout_s,
            "robustrl_saved_s": robustrl_saved_s,
        },
        "rollpacker": {
            "target_responses": target_responses,
            "speculative_responses": speculative_responses,
            "speculative_headroom_pct": speculative_headroom_pct,
            "long_prob_single_draw": long_prob_single_draw,
            "long_prob_two_slot_first": long_prob_two_slot_first,
            "serial_step_s": serial_step_s,
            "stream_trainer_step_s": stream_trainer_step_s,
            "per_microbatch_avg": per_microbatch_avg,
            "true_token_weighted_avg": true_token_weighted_avg,
        },
        "distrs": {
            "unconditional_mean_s": unconditional_mean_s,
            "naive_remaining_at_10s": naive_remaining_s,
            "conditional_remaining_at_10s": conditional_remaining_s,
            "sunk_work_tokens": sunk_work_tokens,
        },
    }


def main() -> None:
    metrics = compute_metrics()
    EXPECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPECTED_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
