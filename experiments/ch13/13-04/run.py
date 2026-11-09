#!/usr/bin/env python3
"""Experiment 13-04: LMCache Prefix KV Compression, 3-Tier Object Lifecycle, and NVMe WRR QoS."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"
SUMMARY_PATH = HERE / "results" / "summary.json"


@dataclass(frozen=True)
class LMCachePipelineMetrics:
    raw_kv_mib: float
    compressed_kv_mib: float
    compression_ratio: float
    uncompressed_serial_ms: float
    compressed_pipelined_ms: float
    ttft_speedup_x: float


@dataclass(frozen=True)
class ThreeTierObjectLifecycleReport:
    avg_fetch_ms: float
    pure_recompute_ms: float
    speedup_vs_recompute: float
    monthly_storage_cost_usd: float
    all_dram_cost_usd: float
    cost_saving_pct: float


@dataclass(frozen=True)
class NvmeQosIsolationReport:
    fifo_hol_wait_ms: float
    fifo_p99_read_ms: float
    wrr_sliced_wait_ms: float
    wrr_p99_read_ms: float
    tail_latency_reduction_x: float
    kv_guaranteed_bw_gib_s: float


def simulate_lmcache_prefix_compression_and_pipeline(
    seq_tokens: int = 65536,
    layers: int = 64,
    kv_bytes_per_tok_per_layer: int = 2048,
    compression_ratio: float = 3.80,
    gds_bw_gib_s: float = 12.5,
    gpu_decomp_bw_gib_s: float = 80.0,
    layer_compute_ms: float = 1.15,
    chunk_layers: int = 4,
) -> LMCachePipelineMetrics:
    raw_bytes = seq_tokens * layers * kv_bytes_per_tok_per_layer
    raw_mib = raw_bytes / (1024.0 ** 2)
    comp_mib = raw_mib / compression_ratio

    raw_gib = raw_bytes / (1024.0 ** 3)
    uncomp_fetch_ms = (raw_gib / gds_bw_gib_s) * 1000.0
    total_compute_ms = layers * layer_compute_ms
    uncomp_serial_ms = uncomp_fetch_ms + total_compute_ms

    num_chunks = layers // chunk_layers
    comp_gib_per_chunk = (raw_gib / compression_ratio) / num_chunks
    raw_gib_per_chunk = raw_gib / num_chunks
    chunk_io_ms = (
        (comp_gib_per_chunk / gds_bw_gib_s) * 1000.0
        + (raw_gib_per_chunk / gpu_decomp_bw_gib_s) * 1000.0
    )
    chunk_comp_ms = chunk_layers * layer_compute_ms

    pipelined_ms = (
        chunk_io_ms
        + (num_chunks - 1) * max(chunk_io_ms, chunk_comp_ms)
        + chunk_comp_ms
    )
    return LMCachePipelineMetrics(
        raw_kv_mib=round(raw_mib, 2),
        compressed_kv_mib=round(comp_mib, 2),
        compression_ratio=round(compression_ratio, 2),
        uncompressed_serial_ms=round(uncomp_serial_ms, 2),
        compressed_pipelined_ms=round(pipelined_ms, 2),
        ttft_speedup_x=round(uncomp_serial_ms / pipelined_ms, 2),
    )


def evaluate_s3_nvme_dram_three_tier_lifecycle(
    hit_dram: float = 0.45,
    hit_nvme: float = 0.40,
    hit_s3_express: float = 0.15,
    t_dram_ms: float = 42.0,
    t_nvme_ms: float = 273.02,
    t_s3_ms: float = 680.0,
    t_recompute_ms: float = 1850.0,
    pool_total_tib: float = 100.0,
    dram_ratio: float = 0.05,
    nvme_ratio: float = 0.25,
    s3_ratio: float = 0.70,
    cost_dram_tib_mo: float = 320.0,
    cost_nvme_tib_mo: float = 45.0,
    cost_s3_tib_mo: float = 16.0,
) -> ThreeTierObjectLifecycleReport:
    avg_ms = (
        hit_dram * t_dram_ms
        + hit_nvme * t_nvme_ms
        + hit_s3_express * t_s3_ms
    )
    tiered_cost = pool_total_tib * (
        dram_ratio * cost_dram_tib_mo
        + nvme_ratio * cost_nvme_tib_mo
        + s3_ratio * cost_s3_tib_mo
    )
    all_dram_cost = pool_total_tib * cost_dram_tib_mo
    saving_pct = (1.0 - tiered_cost / all_dram_cost) * 100.0
    return ThreeTierObjectLifecycleReport(
        avg_fetch_ms=round(avg_ms, 2),
        pure_recompute_ms=round(t_recompute_ms, 2),
        speedup_vs_recompute=round(t_recompute_ms / avg_ms, 2),
        monthly_storage_cost_usd=round(tiered_cost, 2),
        all_dram_cost_usd=round(all_dram_cost, 2),
        cost_saving_pct=round(saving_pct, 2),
    )


def audit_nvme_wrr_qos_and_hol_blocking(
    ssd_bw_gib_s: float = 10.0,
    fifo_queued_writes: int = 16,
    write_1m_ms: float = 0.80,
    write_slice_kb: int = 128,
    kv_burst_read_1m_ms: float = 1.00,
    wrr_high_weight: int = 8,
    wrr_low_weight: int = 2,
) -> NvmeQosIsolationReport:
    fifo_wait_ms = fifo_queued_writes * write_1m_ms
    fifo_p99_ms = fifo_wait_ms + kv_burst_read_1m_ms

    slice_factor = 1024 // write_slice_kb
    wrr_wait_ms = write_1m_ms / slice_factor
    wrr_share = wrr_high_weight / (wrr_high_weight + wrr_low_weight)
    wrr_p99_ms = wrr_wait_ms + (kv_burst_read_1m_ms / wrr_share)
    kv_bw = ssd_bw_gib_s * wrr_share
    return NvmeQosIsolationReport(
        fifo_hol_wait_ms=round(fifo_wait_ms, 2),
        fifo_p99_read_ms=round(fifo_p99_ms, 2),
        wrr_sliced_wait_ms=round(wrr_wait_ms, 2),
        wrr_p99_read_ms=round(wrr_p99_ms, 2),
        tail_latency_reduction_x=round(fifo_p99_ms / wrr_p99_ms, 2),
        kv_guaranteed_bw_gib_s=round(kv_bw, 2),
    )


def compute_metrics() -> dict:
    return {
        "experiment_id": "13-04",
        "lmcache_pipeline": asdict(simulate_lmcache_prefix_compression_and_pipeline()),
        "three_tier_lifecycle": asdict(evaluate_s3_nvme_dram_three_tier_lifecycle()),
        "nvme_wrr_qos": asdict(audit_nvme_wrr_qos_and_hol_blocking()),
    }


def main() -> None:
    metrics = compute_metrics()
    for p in (EXPECTED_PATH, SUMMARY_PATH):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
