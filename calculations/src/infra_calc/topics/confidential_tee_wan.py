"""Ch16: Bounce-buffer vs Rubin TEE-I/O, AES-CTR pre-masking, DSec microVMs, and untrusted WAN PoL/DiLoCo."""
from __future__ import annotations

from .. import hardware
from ..sources import model_config


def calculate(
    tensor_mib: float = 64.0,
    cpu_bounce_bw_gbps: float = 8.4,
    bounce_dma_dec_ms: float = 0.8,
    rubin_tee_io_bw_gbps: float = 112.0,
    rubin_pipelined_ms: float = 0.40,
    dsec_sandboxes: int = 380_000,
    wan_bandwidth_gbps: float = 5.0,
    byzantine_node_fraction: float = 0.15,
    pol_spot_check_rate: float = 0.05,
    pol_single_round_catch_prob: float = 0.556,
    pol_outer_rounds: int = 10,
) -> dict:
    qwen235 = model_config("qwen3-235b-a22b")
    hw_map = {d["id"]: d for d in hardware.catalog()["devices"]}

    # 1. 算例 16-1: 64 MiB (67.11 MB) tensor software bounce-buffer vs Vera Rubin PCIe Gen6 native TEE-I/O
    tensor_mb = round(tensor_mib * (1024**2) / 1e6, 2)                        # 67.11 MB
    t_cpu_enc_ms = round(tensor_mb / cpu_bounce_bw_gbps, 1)                   # 8.0 ms (7.99 ms)
    t_bounce_ms = round(t_cpu_enc_ms + bounce_dma_dec_ms, 1)                  # 8.8 ms
    t_rubin_single_ms = round(tensor_mb / rubin_tee_io_bw_gbps, 2)            # 0.60 ms
    speedup_single_x = round(t_bounce_ms / t_rubin_single_ms, 1)              # 14.7x (~15x)
    speedup_pipelined_x = round(t_bounce_ms / rubin_pipelined_ms, 1)          # 22.0x

    # 2. 算例 16-3: PipeLLM triple buffer & AES-CTR speculative pre-encryption critical hit rate p*
    triple_buffer_mib = round(3.0 * tensor_mib, 1)                            # 192.0 MiB
    dt_hit_ms = 3.0                                                           # 4.0 ms -> 1.0 ms saves 3.0 ms
    dt_miss_base_ms = 1.5
    dt_miss_tdx_ms = 3.0
    p_star_base_pct = round(dt_miss_base_ms / (dt_hit_ms + dt_miss_base_ms) * 100.0, 1)  # 33.3%
    p_star_tdx_pct = round(dt_miss_tdx_ms / (dt_hit_ms + dt_miss_tdx_ms) * 100.0, 1)     # 50.0%

    # 3. DSec Firecracker microVM page sharing + DAX struct page metadata tax (64 B per 4 KiB page = 1.5625%)
    struct_page_tax_pct = round(64.0 / 4096.0 * 100.0, 4)                     # 1.5625%
    per_vm_unshared_mib = 96.0
    per_vm_base_rootfs_mib = 512.0
    total_naive_tb = round(dsec_sandboxes * (per_vm_base_rootfs_mib + per_vm_unshared_mib) * (1024**2) / 1e12, 2)
    total_shared_dax_tb = round((per_vm_base_rootfs_mib + dsec_sandboxes * per_vm_unshared_mib) * (1024**2) / 1e12, 2)
    memory_saving_pct = round((1.0 - total_shared_dax_tb / total_naive_tb) * 100.0, 2)

    # 4. 算例 16-14 & 16-15: Untrusted WAN Streaming DiLoCo FP4 + Proof-of-Learning sampling game & Weiszfeld median
    params_b = 235.0
    fp4_payload_gb = round(params_b * 0.53125, 2)
    wan_Bps = wan_bandwidth_gbps * 1e9 / 8.0 * 0.88
    num_fragments = 16
    fragment_transfer_s = round((fp4_payload_gb * 1e9 / num_fragments) / wan_Bps, 2)

    escape_prob_10_rounds = (1.0 - pol_single_round_catch_prob) ** pol_outer_rounds  # 0.444^10 = 0.000298
    pol_10round_detect_pct = round((1.0 - escape_prob_10_rounds) * 100.0, 2)         # 99.97%
    geom_median_breakdown_point = 0.50
    geom_median_variance_factor_x = round(1.0 / (1.0 - 2.0 * byzantine_node_fraction), 2)  # 1.43x
    spot_dht_ettr_pct = 96.8

    return {
        "schema": "h2-topic-v1",
        "calculation": "confidential-tee-wan",
        "chapter": 16,
        "title": "第 16 章：Vera Rubin 原生 TEE-I/O、AES-CTR 投机预掩码、微虚机去重与广域网防伪抽检账本",
        "scenario": {
            "tensor_mib": tensor_mib,
            "cpu_bounce_bw_gbps": cpu_bounce_bw_gbps,
            "rubin_tee_io_bw_gbps": rubin_tee_io_bw_gbps,
            "dsec_sandboxes": dsec_sandboxes,
            "wan_bandwidth_gbps": wan_bandwidth_gbps,
            "byzantine_node_fraction": byzantine_node_fraction,
            "pol_spot_check_rate": pol_spot_check_rate,
            "pol_outer_rounds": pol_outer_rounds,
            "qwen235_layers": qwen235["num_hidden_layers"],
            "rubin_hbm_gb": hw_map["rubin-nvl72-product-profile"]["memory"]["nominal_capacity"],
        },
        "summary": {
            "bounce_buffer_64mib_ms": t_bounce_ms,
            "rubin_tee_io_64mib_single_ms": t_rubin_single_ms,
            "rubin_tee_io_64mib_pipelined_ms": rubin_pipelined_ms,
            "tee_io_speedup_single_x": speedup_single_x,
            "tee_io_speedup_pipelined_x": speedup_pipelined_x,
            "pipellm_triple_buffer_mib": triple_buffer_mib,
            "aes_ctr_critical_hit_rate_pct": p_star_base_pct,
            "aes_ctr_critical_hit_rate_tdx_pct": p_star_tdx_pct,
            "dax_struct_page_metadata_tax_pct": struct_page_tax_pct,
            "dsec_380k_naive_ram_tb": total_naive_tb,
            "dsec_380k_shared_dax_ram_tb": total_shared_dax_tb,
            "dsec_memory_saving_pct": memory_saving_pct,
            "wan_streaming_diloco_fp4_payload_gb": fp4_payload_gb,
            "wan_per_fragment_transfer_seconds": fragment_transfer_s,
            "pol_10round_detection_pct": pol_10round_detect_pct,
            "geom_median_breakdown_point": geom_median_breakdown_point,
            "geom_median_variance_inflation_x": geom_median_variance_factor_x,
            "spot_dht_elastic_ettr_pct": spot_dht_ettr_pct,
        },
        "assumptions": [
            "算例 16-1 与 16-3：64 MiB（67.11 MB）张量经软件弹跳缓冲需 8.8 ms，Vera Rubin PCIe Gen6 原生 TEE-I/O 单块仅 0.60 ms（流水 0.40 ms，提速 14.7x–22.0x）；投机预加密临界命中率为 33.3%（开启 CPU TDX 时为 50.0%）。",
            "算例 16-14 与 16-15：开放算力池 K=100 含 f=15% 拜占庭节点时，p_check=5% 随机重放抽检（p_1=0.556）在 10 轮外循环内捕获率达 99.97%，Weiszfeld 几何中位数（崩溃点 0.50）将方差膨胀限制在 1.43x 以内。",
        ],
    }
