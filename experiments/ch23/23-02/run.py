#!/usr/bin/env python3
"""Experiment 23-02: LLM rubric distillation to lightweight classifier, offline synthetic data factory Roofline jump, and sandbox rejection sampling."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"
RESULTS_PATH = HERE / "results" / "summary.json"


def compute_metrics() -> dict:
    # 1. LLM Rubric Distillation (70B Teacher -> 110M Classifier) Compute Leverage (Ex 23-4)
    total_docs_b = 50.0  # 50 billion documents
    avg_tokens_per_doc = 500
    total_corpus_tokens = int(total_docs_b * 1e9 * avg_tokens_per_doc)  # 25T tokens

    teacher_70b_tok_s_gpu = 3200.0
    direct_70b_gpu_hours = round(total_corpus_tokens / (teacher_70b_tok_s_gpu * 3600.0), 1)  # 2,170,138.9 GPU-h

    seed_docs_m = 5.0  # 5 million seed docs labeled by 70B teacher
    seed_labeling_gpu_hours = round((seed_docs_m * 1e6 * avg_tokens_per_doc) / (teacher_70b_tok_s_gpu * 3600.0), 1)  # 217.0 GPU-h
    classifier_train_gpu_hours = 80.0
    student_110m_tok_s_gpu = 180000.0
    student_scoring_gpu_hours = round(total_corpus_tokens / (student_110m_tok_s_gpu * 3600.0), 1)  # 38,580.2 GPU-h
    distilled_total_gpu_hours = round(seed_labeling_gpu_hours + classifier_train_gpu_hours + student_scoring_gpu_hours, 1)  # 38,877.2 GPU-h
    distillation_compute_leverage_x = round(direct_70b_gpu_hours / distilled_total_gpu_hours, 2)  # 55.82x

    # 2. Offline Synthetic Data Factory (Batch=2048 Throughput-First) vs Online Serving (Batch=32) & Rejection Sampling (Ex 23-5)
    online_batch_size = 32
    online_tok_s_gpu = 1920.0
    online_mfu_pct = 11.2
    online_gpu_rate_usd_hr = 4.00
    online_usd_per_1m_tokens = round(online_gpu_rate_usd_hr / (online_tok_s_gpu * 3600.0 / 1e6), 3)  # 0.579 USD / 1M

    offline_batch_size = 2048
    offline_tok_s_gpu = 13824.0
    offline_mfu_pct = 58.4
    throughput_gain_x = round(offline_tok_s_gpu / online_tok_s_gpu, 2)  # 7.20x

    slack_spot_gpu_rate_usd_hr = 2.20  # 45% discount via off-peak / slack harvesting
    offline_raw_usd_per_1m_tokens = round(slack_spot_gpu_rate_usd_hr / (offline_tok_s_gpu * 3600.0 / 1e6), 3)  # 0.044 USD / 1M
    raw_cost_reduction_x = round(online_usd_per_1m_tokens / offline_raw_usd_per_1m_tokens, 2)  # 13.16x

    raw_sandbox_pass_yield_pct = 31.25  # 5 / 16 pass compiler + unit tests on first try
    rephrase_salvage_rate = 0.40        # 40% of failed traces salvaged via 1-step self-correction
    effective_verified_yield_pct = round(
        raw_sandbox_pass_yield_pct + rephrase_salvage_rate * (100.0 - raw_sandbox_pass_yield_pct), 2
    )  # 58.75%
    verified_usd_per_1m_tokens = round(offline_raw_usd_per_1m_tokens / (effective_verified_yield_pct / 100.0), 3)  # 0.075 USD / 1M

    return {
        "experiment_id": "23-02",
        "rubric_distillation": {
            "direct_70b_gpu_hours": direct_70b_gpu_hours,
            "seed_labeling_gpu_hours": seed_labeling_gpu_hours,
            "student_scoring_gpu_hours": student_scoring_gpu_hours,
            "distilled_total_gpu_hours": distilled_total_gpu_hours,
            "distillation_compute_leverage_x": distillation_compute_leverage_x,
        },
        "synthetic_data_factory": {
            "online_batch_size": online_batch_size,
            "online_tok_s_gpu": online_tok_s_gpu,
            "online_mfu_pct": online_mfu_pct,
            "online_usd_per_1m_tokens": online_usd_per_1m_tokens,
            "offline_batch_size": offline_batch_size,
            "offline_tok_s_gpu": offline_tok_s_gpu,
            "offline_mfu_pct": offline_mfu_pct,
            "throughput_gain_x": throughput_gain_x,
            "offline_raw_usd_per_1m_tokens": offline_raw_usd_per_1m_tokens,
            "raw_cost_reduction_x": raw_cost_reduction_x,
            "raw_sandbox_pass_yield_pct": raw_sandbox_pass_yield_pct,
            "effective_verified_yield_pct": effective_verified_yield_pct,
            "verified_usd_per_1m_tokens": verified_usd_per_1m_tokens,
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
