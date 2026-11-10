#!/usr/bin/env python3
"""Experiment 23-01: Beyond-Chinchilla Lifecycle Allocation, K192 Preregistration Audit & NeuSight Calibration."""
import json
import pathlib
import statistics

HERE = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"


def compute_beyond_chinchilla():
    # Compare 32B Chinchilla-baseline vs 8B over-trained (20T tokens)
    c_train_32b_usd = 600_000.0
    c_inf_32b_per_m_usd = 0.45
    c_train_8b_usd = 1_500_000.0
    c_inf_8b_per_m_usd = 0.15

    delta_train_usd = c_train_8b_usd - c_train_32b_usd  # 900,000 USD
    delta_inf_per_m_usd = c_inf_32b_per_m_usd - c_inf_8b_per_m_usd  # 0.30 USD / M tokens
    breakeven_m_tokens = delta_train_usd / delta_inf_per_m_usd  # 3,000,000 M = 3.0T tokens
    breakeven_t_tokens = breakeven_m_tokens / 1e6

    daily_tokens = 2.0e10  # 20B tokens/day = 20,000 M tokens/day
    breakeven_days = (breakeven_m_tokens * 1e6) / daily_tokens  # 150.0 days
    two_year_m_tokens = (730 * daily_tokens) / 1e6  # 14,600,000 M tokens
    net_savings_2yr_usd = two_year_m_tokens * delta_inf_per_m_usd - delta_train_usd

    return {
        "delta_train_usd": round(delta_train_usd, 2),
        "delta_inf_per_m_usd": round(delta_inf_per_m_usd, 4),
        "breakeven_t_tokens": round(breakeven_t_tokens, 4),
        "breakeven_days": round(breakeven_days, 2),
        "net_savings_2yr_usd": round(net_savings_2yr_usd, 2),
    }


def audit_k192_preregistration():
    paired_ratios = [
        0.7661106575310103,
        0.6374560811931168,
        0.7039580198331323,
        0.9502700026584732,
        0.9372772983445644,
        0.3734699482077529,
        0.5814393631102742,
        0.6823231258495099,
        0.653319634253581,
        0.7242621041727667,
    ]
    median_r = round(statistics.median(paired_ratios), 6)
    interval_low, interval_high = 0.70, 1.00
    in_interval = interval_low <= median_r <= interval_high
    status = "supported_in_scope" if in_interval else "refuted_in_scope"
    return {
        "count": len(paired_ratios),
        "paired_ratios": paired_ratios,
        "median_r": median_r,
        "preregistered_interval": [interval_low, interval_high],
        "in_interval": in_interval,
        "verdict": status,
    }


def audit_neusight_stage_calibration():
    pred_ms = [8.0, 2.0]
    actual_ms = [6.0, 4.0]
    pred_total = sum(pred_ms)
    actual_total = sum(actual_ms)
    agg_ape = abs(pred_total - actual_total) / actual_total
    stage_apes = [round(abs(p - a) / a, 6) for p, a in zip(pred_ms, actual_ms)]
    halving_stage1_pred_ms = pred_ms[0] / 2.0 + pred_ms[1]
    halving_stage1_actual_ms = actual_ms[0] / 2.0 + actual_ms[1]
    return {
        "predicted_stage_ms": pred_ms,
        "actual_stage_ms": actual_ms,
        "aggregate_ape": round(agg_ape, 6),
        "stage_apes": stage_apes,
        "stage2_predicted_share": round(pred_ms[1] / pred_total, 4),
        "stage2_actual_share": round(actual_ms[1] / actual_total, 4),
        "halving_stage1_predicted_total_ms": halving_stage1_pred_ms,
        "halving_stage1_actual_total_ms": halving_stage1_actual_ms,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "23-01",
        "beyond_chinchilla": compute_beyond_chinchilla(),
        "k192_preregistration": audit_k192_preregistration(),
        "neusight_stage_calibration": audit_neusight_stage_calibration(),
    }
    out_path = RESULTS_DIR / "lifecycle_and_preregistration.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
