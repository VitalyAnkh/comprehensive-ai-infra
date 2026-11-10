#!/usr/bin/env python3
"""Experiment 16-03: TEE Constant-Rate Traffic Shaping & Verifiable Inference (Freivalds + zkML)."""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"
RESULTS_PATH = HERE / "results" / "summary.json"


def simulate_constant_time_token_padding_and_traffic_shaping(
    true_token_counts: list = [19, 47, 83, 142, 215],
    bucket_granularity: int = 64,
    tick_interval_ms: float = 25.0,
    raw_tpot_ms: float = 22.3,
    packet_bytes_per_tok: int = 4096,
) -> dict:
    raw_total_tokens = sum(true_token_counts)
    padded_counts = [
        int(math.ceil(c / bucket_granularity) * bucket_granularity)
        for c in true_token_counts
    ]
    padded_total_tokens = sum(padded_counts)
    dummy_tokens = padded_total_tokens - raw_total_tokens
    bw_tax_pct = (dummy_tokens / raw_total_tokens) * 100.0
    latency_tax_pct = ((tick_interval_ms - raw_tpot_ms) / raw_tpot_ms) * 100.0
    distinct_buckets = len(set(padded_counts))
    raw_leakage_bits = round(math.log2(256), 2)
    shaped_in_bucket_leakage_bits = 0.0
    return {
        "raw_total_tokens": raw_total_tokens,
        "padded_total_tokens": padded_total_tokens,
        "dummy_tokens": dummy_tokens,
        "bw_tax_pct": round(bw_tax_pct, 2),
        "latency_tax_pct": round(latency_tax_pct, 2),
        "distinct_buckets": distinct_buckets,
        "raw_leakage_bits": raw_leakage_bits,
        "shaped_in_bucket_leakage_bits": shaped_in_bucket_leakage_bits,
        "total_shaped_mib": round(
            (padded_total_tokens * packet_bytes_per_tok) / (1024.0 * 1024.0), 3
        ),
    }


def evaluate_zkml_sumcheck_and_freivalds_auditor(
    m_tokens: int = 4096,
    d_model: int = 4096,
    n_ffn: int = 14336,
    freivalds_k: int = 40,
    sumcheck_prover_mult: float = 185.0,
    naive_zk_prover_mult: float = 12000.0,
) -> dict:
    gemm_flops = 2.0 * m_tokens * d_model * n_ffn
    freivalds_flops = 2.0 * freivalds_k * (
        d_model * n_ffn + m_tokens * d_model + m_tokens * n_ffn
    )
    freivalds_overhead_pct = (freivalds_flops / gemm_flops) * 100.0
    soundness_error = 2.0 ** (-freivalds_k)
    sumcheck_speedup_vs_naive = naive_zk_prover_mult / sumcheck_prover_mult
    return {
        "gemm_gflops": round(gemm_flops / 1e9, 3),
        "freivalds_mflops": round(freivalds_flops / 1e6, 3),
        "freivalds_overhead_pct": round(freivalds_overhead_pct, 4),
        "soundness_error": soundness_error,
        "sumcheck_speedup_vs_naive": round(sumcheck_speedup_vs_naive, 2),
    }


def verify_hybrid_tee_opml_zkml_audit_game(
    n_operators: int = 320,
    interactive_round_rtt_s: float = 5.0,
    single_op_zk_prove_s: float = 1.65,
    single_op_zk_verify_s: float = 0.20,
    interactive_gas_usd: float = 42.0,
    single_zk_verify_gas_usd: float = 4.83,
) -> dict:
    bisect_rounds = math.ceil(math.log2(n_operators))
    interactive_dispute_s = bisect_rounds * interactive_round_rtt_s
    zk_fault_proof_s = single_op_zk_prove_s + single_op_zk_verify_s
    latency_speedup = interactive_dispute_s / zk_fault_proof_s
    gas_saving_pct = (
        (interactive_gas_usd - single_zk_verify_gas_usd) / interactive_gas_usd * 100.0
    )
    return {
        "bisect_rounds": bisect_rounds,
        "interactive_dispute_s": round(interactive_dispute_s, 2),
        "zk_fault_proof_s": round(zk_fault_proof_s, 2),
        "latency_speedup": round(latency_speedup, 2),
        "gas_saving_pct": round(gas_saving_pct, 2),
    }


def compute_metrics() -> dict:
    return {
        "experiment_id": "16-03",
        "traffic_shaping": simulate_constant_time_token_padding_and_traffic_shaping(),
        "freivalds_and_zkml": evaluate_zkml_sumcheck_and_freivalds_auditor(),
        "hybrid_zk_fault_proof": verify_hybrid_tee_opml_zkml_audit_game(),
    }


def main() -> None:
    metrics = compute_metrics()
    EXPECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not EXPECTED_PATH.exists():
        EXPECTED_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    RESULTS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
