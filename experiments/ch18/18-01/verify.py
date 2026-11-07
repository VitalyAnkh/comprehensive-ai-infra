#!/usr/bin/env python3
"""Verify deterministic outputs for Experiment 18-01."""
from __future__ import annotations

import json
from pathlib import Path
from run import compute_metrics, EXPECTED_PATH


def main() -> None:
    actual = compute_metrics()
    expected = json.loads(Path(EXPECTED_PATH).read_text())
    assert actual == expected, f"Mismatch:\nactual={actual}\nexpected={expected}"
    assert actual["ppo_vs_grpo_32b"]["ppo_total_state_gb"] == 1152
    assert actual["ppo_vs_grpo_32b"]["grpo_total_state_gb"] == 576
    assert actual["weave_and_robustrl"]["robustrl_saved_s"] == 300
    assert actual["rollpacker"]["speculative_headroom_pct"] == 56.25
    assert actual["rollpacker"]["long_prob_two_slot_first"] == 0.25
    assert actual["rollpacker"]["per_microbatch_avg"] == 2.0
    assert actual["rollpacker"]["true_token_weighted_avg"] == 2.5
    assert actual["distrs"]["conditional_remaining_at_10s"] == 90.0
    assert actual["distrs"]["sunk_work_tokens"] == 216000
    print("Experiment 18-01 verified successfully.")


if __name__ == "__main__":
    main()
