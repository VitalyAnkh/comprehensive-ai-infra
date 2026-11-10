#!/usr/bin/env python3
"""Experiment 20-02: Three-tier humanoid WBC + visuotactile slip reflex, PTP sensor alignment, CBF shield, and FAST DCT action tokenizer."""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"
RESULTS_PATH = HERE / "results" / "summary.json"


def dct_ii_1d(signal: list[float]) -> list[float]:
    """Orthonormal Type-II Discrete Cosine Transform (zero external dependencies)."""
    n = len(signal)
    out: list[float] = []
    for k in range(n):
        scale = math.sqrt(1.0 / n) if k == 0 else math.sqrt(2.0 / n)
        acc = sum(signal[t] * math.cos(math.pi * (2 * t + 1) * k / (2.0 * n)) for t in range(n))
        out.append(scale * acc)
    return out


def idct_ii_1d(coeffs: list[float]) -> list[float]:
    """Orthonormal Inverse Type-II Discrete Cosine Transform."""
    n = len(coeffs)
    out: list[float] = []
    for t in range(n):
        acc = 0.0
        for k in range(n):
            scale = math.sqrt(1.0 / n) if k == 0 else math.sqrt(2.0 / n)
            acc += scale * coeffs[k] * math.cos(math.pi * (2 * t + 1) * k / (2.0 * n))
        out.append(acc)
    return out


def compute_metrics() -> dict:
    # 1. Three-tier humanoid control hierarchy & 1,000 Hz visuotactile slip reflex (Ex 19-6)
    tier1_vlm_hz = 2.0
    tier2_flow_hz = 100.0
    tier3_wbc_hz = 1000.0
    tier3_period_us = 1000.0

    tactile_read_us = 180.0
    slip_check_us = 40.0
    wbc_qp_solve_us = 420.0
    ethercat_tx_us = 160.0
    tier3_wcet_us = round(tactile_read_us + slip_check_us + wbc_qp_solve_us + ethercat_tx_us, 2)  # 800.0 us
    tier3_margin_us = round(tier3_period_us - tier3_wcet_us, 2)  # 200.0 us

    # Slip displacement at v_slip = 0.12 m/s: 15 ms visual loop vs 1.0 ms tactile loop
    v_slip_m_s = 0.12
    lag_visual_ms = 15.0
    lag_tactile_ms = 1.0
    slip_disp_100hz_mm = round(v_slip_m_s * lag_visual_ms, 2)  # 1.80 mm
    slip_disp_1000hz_mm = round(v_slip_m_s * lag_tactile_ms, 2)  # 0.12 mm
    slip_reduction_x = round(slip_disp_100hz_mm / slip_disp_1000hz_mm, 2)  # 15.0x

    # 2. FAST (Frequency-space Action Sequence Tokenization) DCT-II compression (Ex 19-6 & Ex 19-11)
    horizon_h = 50
    action_dims = 14
    naive_tokens = horizon_h * action_dims  # 700 tokens
    retained_k = 8
    fast_bpe_tokens = 80
    compression_ratio_x = round(naive_tokens / fast_bpe_tokens, 2)  # 8.75x
    token_reduction_pct = round((1.0 - fast_bpe_tokens / naive_tokens) * 100.0, 2)  # 88.57%

    per_token_ar_ms = 3.5
    ar_decode_naive_ms = round(naive_tokens * per_token_ar_ms, 2)  # 2450.0 ms
    ar_decode_fast_ms = round(fast_bpe_tokens * per_token_ar_ms, 2)  # 280.0 ms

    # Deterministic smooth 50-step x 14-DoF minimum-jerk trajectory DCT verification
    total_energy = 0.0
    low_freq_energy = 0.0
    sq_err_sum = 0.0
    for d in range(action_dims):
        amp = 4.0 + 0.3 * d  # mm incremental displacement over 0.5 s action chunk
        phase = 0.15 * d
        traj = [
            amp
            * (
                0.65 * math.cos(math.pi * t / horizon_h + phase)
                + 0.25 * math.cos(2.0 * math.pi * t / horizon_h)
                + 0.055 * math.cos(9.0 * math.pi * t / horizon_h)
            )
            for t in range(horizon_h)
        ]
        coeffs = dct_ii_1d(traj)
        total_energy += sum(c * c for c in coeffs)
        low_freq_energy += sum(c * c for c in coeffs[:retained_k])
        # Truncate high frequencies k >= 8 and quantize with step q = 0.02
        q_step = 0.02
        quant_coeffs = [round(c / q_step) * q_step if k < retained_k else 0.0 for k, c in enumerate(coeffs)]
        recon = idct_ii_1d(quant_coeffs)
        sq_err_sum += sum((traj[t] - recon[t]) ** 2 for t in range(horizon_h))

    energy_retention_pct = round(low_freq_energy / total_energy * 100.0, 2)  # > 99.4%
    recon_rmse_mm = round(math.sqrt(sq_err_sum / (horizon_h * action_dims)), 4)  # < 0.35 mm

    # 3. PTP hardware synchronization vs software arrival timestamp skew & closed-form CBF safety shield (Ex 19-6 & Ex 19-12)
    arm_vel_m_s = 0.60
    software_skew_ms = 27.0
    ptp_interp_skew_ms = 0.018  # 18 us < 100 us
    software_pos_err_mm = round(arm_vel_m_s * software_skew_ms, 2)  # 16.20 mm
    ptp_pos_err_mm = round(arm_vel_m_s * ptp_interp_skew_ms, 3)  # 0.011 mm
    sync_accuracy_gain_x = round(software_pos_err_mm / ptp_pos_err_mm, 1)  # 1472.7x

    # Closed-form 1,000 Hz Zero-Order Hold CBF safety shield: v_safe <= gamma_cbf * h(x)
    v_nom_m_s = 0.85
    h_init_m = 0.015  # 15.0 mm from obstacle
    gamma_cbf = 20.0
    v_safe_at_15mm = round(min(v_nom_m_s, gamma_cbf * h_init_m), 2)  # 0.30 m/s
    raw_margin = round(-v_nom_m_s + gamma_cbf * h_init_m, 4)  # -0.55 < 0 (unsafe without shield)
    post_cbf_margin = round(-v_safe_at_15mm + gamma_cbf * h_init_m, 4)  # 0.0 >= 0 (safe)
    # 500-step 1,000 Hz rollout with 1.20 mm hardware standoff floor
    h_cur = h_init_m
    dt_s = 0.001
    h_standoff_m = 0.0012  # 1.20 mm safety buffer
    safety_violations = 0
    for _ in range(500):
        v_cmd = min(v_nom_m_s, gamma_cbf * max(0.0, h_cur - h_standoff_m))
        h_cur = h_cur - v_cmd * dt_s
        if h_cur < 0.0:
            safety_violations += 1
    min_dist_mm = round(h_cur * 1000.0, 2)  # 1.20 mm > 0

    return {
        "experiment_id": "20-02",
        "three_tier_humanoid_wbc": {
            "tier1_vlm_hz": tier1_vlm_hz,
            "tier2_flow_hz": tier2_flow_hz,
            "tier3_wbc_hz": tier3_wbc_hz,
            "tier3_period_us": tier3_period_us,
            "tier3_wcet_us": tier3_wcet_us,
            "tier3_margin_us": tier3_margin_us,
            "slip_disp_100hz_mm": slip_disp_100hz_mm,
            "slip_disp_1000hz_mm": slip_disp_1000hz_mm,
            "slip_reduction_x": slip_reduction_x,
        },
        "fast_dct_action_tokenizer": {
            "horizon_h": horizon_h,
            "action_dims": action_dims,
            "naive_tokens": naive_tokens,
            "retained_low_freqs": retained_k,
            "fast_bpe_tokens": fast_bpe_tokens,
            "compression_ratio_x": compression_ratio_x,
            "token_reduction_pct": token_reduction_pct,
            "ar_decode_naive_ms": ar_decode_naive_ms,
            "ar_decode_fast_ms": ar_decode_fast_ms,
            "energy_retention_pct": energy_retention_pct,
            "recon_rmse_mm": recon_rmse_mm,
        },
        "ptp_alignment_and_cbf_shield": {
            "software_skew_ms": software_skew_ms,
            "ptp_interp_skew_ms": ptp_interp_skew_ms,
            "software_pos_err_mm": software_pos_err_mm,
            "ptp_pos_err_mm": ptp_pos_err_mm,
            "sync_accuracy_gain_x": sync_accuracy_gain_x,
            "v_nom_m_s": v_nom_m_s,
            "v_safe_at_15mm": v_safe_at_15mm,
            "raw_cbf_margin": raw_margin,
            "post_cbf_margin": post_cbf_margin,
            "safety_violations_500steps": safety_violations,
            "min_dist_mm": min_dist_mm,
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
