#!/usr/bin/env python3
"""Experiment 14-02: Telescoping Cost-Reduction Waterfall, FarSkip-Collective & GQA TP Replication."""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"


def compute_telescoping_waterfall():
    # Locked S0 -> S1 -> S2 -> S3 weight-read lower bound states from Ch14 Section 14.8.2 & calculations.json:
    # S0: 70B, 1 Byte/param (INT8/FP8) on H100 (3.35 TB/s) -> 70 / 3.35 = 20.895522 ms (displayed 20.90 ms)
    # S1: 8B, 1 Byte/param on H100 (3.35 TB/s) -> 8 / 3.35 = 2.388060 ms (displayed 2.39 ms, r1 = 70 / 8 = 8.75x)
    # S2: 8B, 0.5 Byte/param (FP4/INT4) on H100 (3.35 TB/s) -> 4 / 3.35 = 1.194030 ms (displayed 1.19 ms, r2 = 2.00x)
    # S3: 8B, 0.5 Byte/param on H200 (4.80 TB/s) -> 4 / 4.80 = 0.833333 ms (displayed 0.83 ms, r3 = 4.80 / 3.35 = 1.432836x)
    s0_exact = 70.0 / 3.35
    s1_exact = 8.0 / 3.35
    s2_exact = 4.0 / 3.35
    s3_exact = 4.0 / 4.80

    r1 = s0_exact / s1_exact  # 8.75
    r2 = s1_exact / s2_exact  # 2.00
    r3 = s2_exact / s3_exact  # 4.80 / 3.35 = 1.43283582
    product_speedup = r1 * r2 * r3
    direct_ratio = s0_exact / s3_exact
    rounded_display_ratio = 20.90 / 0.83

    host_overhead_sweep = []
    for t_host in [0.0, 0.25, 0.50, 1.00]:
        e2e_speedup = (s0_exact + t_host) / (s3_exact + t_host)
        host_overhead_sweep.append({
            "host_overhead_ms": t_host,
            "e2e_speedup": round(e2e_speedup, 4),
        })

    return {
        "states_exact_ms": {
            "S0": round(s0_exact, 6),
            "S1": round(s1_exact, 6),
            "S2": round(s2_exact, 6),
            "S3": round(s3_exact, 6),
        },
        "states_display_ms": {"S0": 20.90, "S1": 2.39, "S2": 1.19, "S3": 0.83},
        "conditional_ratios": [round(r1, 6), round(r2, 6), round(r3, 6)],
        "telescoping_product": round(product_speedup, 6),
        "direct_ratio": round(direct_ratio, 6),
        "rounded_display_ratio": round(rounded_display_ratio, 4),
        "host_overhead_sweep": host_overhead_sweep,
    }


def simulate_farskip_collective():
    layers = 60
    t_comp_ms = 1.20
    alpha_intra = 0.25
    t_comm_ms = 1.10

    results = {}
    for m in [0, 1, 2]:
        overlap_window_ms = (m + alpha_intra) * t_comp_ms
        exposed_per_covered_layer_ms = max(0.0, t_comm_ms - overlap_window_ms)
        total_exposed_ms = 0.0
        for layer_idx in range(1, layers + 1):
            m_eff = min(m, layers - layer_idx)
            layer_overlap_ms = (m_eff + alpha_intra) * t_comp_ms
            total_exposed_ms += max(0.0, t_comm_ms - layer_overlap_ms)
        total_ms = layers * t_comp_ms + total_exposed_ms
        results[f"m={m}"] = {
            "overlap_window_ms": round(overlap_window_ms, 4),
            "exposed_per_covered_layer_ms": round(exposed_per_covered_layer_ms, 4),
            "total_exposed_comm_ms": round(total_exposed_ms, 4),
            "total_forward_ms": round(total_ms, 4),
        }
    base_ms = results["m=0"]["total_forward_ms"]
    for k, v in results.items():
        v["speedup_vs_m0"] = round(base_ms / v["total_forward_ms"], 4)
    return results


def audit_gqa_tp_replication():
    layers = 64
    d_h = 128
    seq_len = 131_072
    tp = 8
    bytes_per_elem = 1  # FP8 KV

    configs = {}
    for n_kv in [8, 2]:
        rep_factor = max(1, tp // n_kv)
        logical_kv_bytes = layers * 2 * n_kv * d_h * bytes_per_elem * seq_len
        physical_total_bytes = logical_kv_bytes * rep_factor
        per_gpu_gib = (physical_total_bytes / tp) / (1024 ** 3)
        total_gib = physical_total_bytes / (1024 ** 3)
        configs[f"n_kv={n_kv}"] = {
            "replication_factor": rep_factor,
            "logical_kv_gib": round(logical_kv_bytes / (1024 ** 3), 4),
            "per_gpu_physical_gib": round(per_gpu_gib, 4),
            "cluster_8gpu_physical_gib": round(total_gib, 4),
        }
    return configs


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "telescoping_waterfall": compute_telescoping_waterfall(),
        "farskip_collective": simulate_farskip_collective(),
        "gqa_tp_replication": audit_gqa_tp_replication(),
    }
    out_path = RESULTS_DIR / "telescoping_and_farskip.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
