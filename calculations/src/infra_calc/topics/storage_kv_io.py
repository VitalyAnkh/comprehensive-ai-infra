"""Ch13: AI storage access patterns, GDS zero-copy, 3FS CRAQ, KV flash layout, and SSD DWPD."""
from __future__ import annotations

import math
from .. import hardware
from ..sources import model_config


def _kv_bytes_per_token_bf16(cfg: dict) -> int:
    tc = cfg.get("text_config", cfg)
    layers = int(tc["num_hidden_layers"])
    if "kv_lora_rank" in tc:
        return layers * (int(tc["kv_lora_rank"]) + int(tc["qk_rope_head_dim"])) * 2
    if tc.get("num_key_value_heads") == 1 and int(tc.get("head_dim", 0)) == 512 and "qk_rope_head_dim" in tc:
        return layers * (int(tc["head_dim"]) + int(tc["qk_rope_head_dim"])) * 2
    return layers * 2 * int(tc["num_key_value_heads"]) * int(tc["head_dim"]) * 2


def calculate(
    checkpoint_gb: float = 140.0,
    nvme_drives: int = 8,
    per_drive_seq_read_gbps: float = 14.0,
    per_drive_gds_eff_gbps: float = 12.5,
    pagecache_rc_limit_gbps: float = 50.0,
    block_tokens: int = 16,
    lba_bytes: int = 4096,
    ssd_drives_kv: int = 4,
    ssd_capacity_tb: float = 3.84,
    ssd_dwpd_rated: float = 3.0,
    ssd_warranty_years: int = 5,
    kv_evict_write_gbps: float = 4.0,
    block_ssd_waf: float = 1.5,
    threefs_nodes: int = 180,
) -> dict:
    qwen8 = model_config("qwen3-8b")
    qwen235 = model_config("qwen3-235b-a22b")
    dsv3 = model_config("deepseek-v3")
    dsv4 = model_config("deepseek-v4-flash")
    hw_map = {d["id"]: d for d in hardware.catalog()["devices"]}

    # 1. 算例 13-1: 8x PCIe Gen5 NVMe SSD loading 140.0 GB checkpoint (GDS P2P DMA vs POSIX PageCache)
    raw_array_gbps = round(nvme_drives * per_drive_seq_read_gbps, 2)        # 112.0 GB/s physical peak
    gds_gbps = round(nvme_drives * per_drive_gds_eff_gbps, 2)              # 100.0 GB/s effective P2P DMA
    pagecache_gbps = round(pagecache_rc_limit_gbps, 2)                     # 50.0 GB/s Root Complex / copy ceiling
    pagecache_rc_traffic_gb = round(2.0 * checkpoint_gb, 2)                # 280.0 GB bidirectional RC traffic
    gds_load_s = round(checkpoint_gb / gds_gbps, 2)                        # 1.40 s
    pagecache_load_s = round(checkpoint_gb / pagecache_gbps, 2)            # 2.80 s
    gds_speedup = round(pagecache_load_s / gds_load_s, 2)                  # 2.0x

    # 2. 算例 13-2: DeepSeek 3FS 180-node storage cluster (16x Gen4 SSDs + 2x200G RDMA NIC per node, CRAQ R=3)
    ssd_raw_per_node_gbps = 16 * 7.0                                       # 112.0 GB/s disk peak
    nic_line_per_node_gbps = (2 * 200.0) / 8.0                             # 50.0 GB/s wire peak
    disk_to_nic_overprovision_x = round(ssd_raw_per_node_gbps / nic_line_per_node_gbps, 2)  # 2.24x
    node_eff_read_gibps = 36.67                                            # 36.67 GiB/s (~39.37 GB/s)
    cluster_read_gibps = round(threefs_nodes * node_eff_read_gibps, 1)     # 6,600.6 GiB/s
    cluster_read_tibps_10e3 = round(cluster_read_gibps / 1000.0, 2)        # 6.60 TiB/s (10^3 GiB convention in Ch13)
    cluster_read_tibps_binary = round(cluster_read_gibps / 1024.0, 2)      # 6.45 TiB/s
    craq_3rep_write_tibps = round(cluster_read_tibps_10e3 / 3.0, 2)        # ~2.20 TiB/s
    kvcache_pool_peak_tib_per_min = 3.66                                   # 3.66 TiB/min (62.46 GiB/s)
    kvcache_pool_peak_gibps = round(kvcache_pool_peak_tib_per_min * 1024.0 / 60.0, 2)

    # 3. 算例 13-3: 32-layer model 16-token KV block recall (2 KiB/layer -> 64 KiB block, 128 blocks = 8 MiB)
    ref_layers = 32
    ref_slice_kib = 2.0
    ref_block_kib = ref_layers * ref_slice_kib                             # 64.0 KiB
    ssd_iops_max = 800_000.0
    ssd_seq_bw_mibps = 12_800.0                                            # 12.5 GiB/s
    critical_io_kib = round((12.5 * 1024.0 * 1024.0) / ssd_iops_max, 3)    # 16.384 KiB
    recall_blocks = 128
    layer_major_ios = recall_blocks * ref_layers                           # 4,096 IOs
    raf_lba_4k = round(4.0 / ref_slice_kib, 2)                             # 2.0x
    raf_nand_page_16k = round(16.0 / ref_slice_kib, 2)                     # 8.0x
    t_layer_major_ms = round(max(layer_major_ios / ssd_iops_max, (layer_major_ios * 4.0 / 1024.0) / ssd_seq_bw_mibps) * 1000.0, 3)  # 5.12 ms
    t_block_major_ms = round(max(recall_blocks / ssd_iops_max, (recall_blocks * ref_block_kib / 1024.0) / ssd_seq_bw_mibps) * 1000.0, 3)  # 0.625 ms
    block_vs_layer_speedup_x = round(t_layer_major_ms / t_block_major_ms, 3)  # 8.192x

    models_kv = {}
    for mid, cfg in (
        ("qwen3-8b", qwen8),
        ("qwen3-235b-a22b", qwen235),
        ("deepseek-v3", dsv3),
        ("deepseek-v4-flash", dsv4),
    ):
        tc = cfg.get("text_config", cfg)
        layers = int(tc["num_hidden_layers"])
        kv_tok = _kv_bytes_per_token_bf16(cfg)
        block_bytes = kv_tok * block_tokens
        per_layer_slice = block_bytes // layers
        is_mla = ("kv_lora_rank" in tc) or (tc.get("num_key_value_heads") == 1 and int(tc.get("head_dim", 0)) == 512)
        sub_tensors_per_layer = 1 if is_mla else 2
        slice_unit = per_layer_slice // sub_tensors_per_layer
        m_layer_ios = layers * sub_tensors_per_layer
        m_layer_phys = m_layer_ios * math.ceil(slice_unit / lba_bytes) * lba_bytes
        m_block_phys = math.ceil(block_bytes / lba_bytes) * lba_bytes
        models_kv[mid] = {
            "layers": layers,
            "kv_bytes_per_token_bf16": kv_tok,
            "block_16tok_logical_bytes": block_bytes,
            "layer_major_iops": m_layer_ios,
            "layer_major_physical_bytes": m_layer_phys,
            "layer_major_read_amplification": round(m_layer_phys / block_bytes, 4),
            "block_major_iops": 1,
            "block_major_physical_bytes": m_block_phys,
            "block_major_read_amplification": round(m_block_phys / block_bytes, 4),
        }

    # 4. 算例 13-9: 8-GPU node KV eviction SSD endurance (4x 3.84 TB = 15.36 TB, 3.0 DWPD rated, 4.0 GB/s host write)
    total_ssd_capacity_tb = round(ssd_drives_kv * ssd_capacity_tb, 2)      # 15.36 TB
    host_write_tb_per_day = round(kv_evict_write_gbps * 86_400.0 / 1000.0, 2)  # 345.6 TB/day
    nand_write_tb_per_day = round(host_write_tb_per_day * block_ssd_waf, 2)    # 518.4 TB/day
    effective_dwpd_unfiltered = round(nand_write_tb_per_day / total_ssd_capacity_tb, 2)  # 33.75 DWPD
    dwpd_overload_x = round(effective_dwpd_unfiltered / ssd_dwpd_rated, 2)               # 11.25x
    burnout_months = round((ssd_warranty_years * 12.0) / dwpd_overload_x, 1)             # 5.3 months
    max_admission_alpha_block_pct = round((ssd_dwpd_rated / effective_dwpd_unfiltered) * 100.0, 2)  # 8.89%
    zns_dwpd_unfiltered = round(host_write_tb_per_day / total_ssd_capacity_tb, 2)        # 22.5 DWPD (WAF = 1.0)
    max_admission_alpha_zns_pct = round((ssd_dwpd_rated / zns_dwpd_unfiltered) * 100.0, 2)  # 13.33%

    return {
        "schema": "h2-topic-v1",
        "calculation": "storage-kv-io",
        "chapter": 13,
        "title": "第 13 章：AI 存储内核旁路（GDS）、3FS 链式复制、KV 闪存块布局与 SSD 寿命准入账本",
        "scenario": {
            "checkpoint_gb": checkpoint_gb,
            "nvme_drives": nvme_drives,
            "per_drive_seq_read_gbps": per_drive_seq_read_gbps,
            "per_drive_gds_eff_gbps": per_drive_gds_eff_gbps,
            "block_tokens": block_tokens,
            "lba_bytes": lba_bytes,
            "ssd_total_capacity_tb": total_ssd_capacity_tb,
            "ssd_dwpd_rated": ssd_dwpd_rated,
            "threefs_nodes": threefs_nodes,
            "reference_hbm_bw_gbps": hw_map["b200-sxm"]["memory"]["bandwidth_bytes_per_second"] / 1e9,
        },
        "summary": {
            "raw_nvme_array_gbps": raw_array_gbps,
            "gds_bandwidth_gbps": gds_gbps,
            "pagecache_bandwidth_gbps": pagecache_gbps,
            "pagecache_rc_traffic_gb": pagecache_rc_traffic_gb,
            "gds_140gb_load_seconds": gds_load_s,
            "pagecache_140gb_load_seconds": pagecache_load_s,
            "gds_vs_pagecache_speedup": gds_speedup,
            "threefs_disk_to_nic_overprovision_x": disk_to_nic_overprovision_x,
            "threefs_180node_read_gibps": cluster_read_gibps,
            "threefs_180node_read_tibps": cluster_read_tibps_10e3,
            "threefs_180node_read_tibps_binary": cluster_read_tibps_binary,
            "threefs_180node_craq_write_tibps": craq_3rep_write_tibps,
            "threefs_kvcache_peak_tib_per_min": kvcache_pool_peak_tib_per_min,
            "threefs_kvcache_peak_gibps": kvcache_pool_peak_gibps,
            "ssd_iops_bw_critical_block_kib": critical_io_kib,
            "ref_32layer_16tok_layer_major_raf_lba": raf_lba_4k,
            "ref_32layer_16tok_layer_major_raf_page16k": raf_nand_page_16k,
            "ref_32layer_128blk_layer_major_ms": t_layer_major_ms,
            "ref_32layer_128blk_block_major_ms": t_block_major_ms,
            "ref_32layer_block_vs_layer_speedup_x": block_vs_layer_speedup_x,
            "qwen3_8b_kv_bytes_per_token": models_kv["qwen3-8b"]["kv_bytes_per_token_bf16"],
            "deepseek_v4_flash_kv_bytes_per_token": models_kv["deepseek-v4-flash"]["kv_bytes_per_token_bf16"],
            "kv_evict_host_write_tb_per_day": host_write_tb_per_day,
            "kv_evict_nand_write_tb_per_day": nand_write_tb_per_day,
            "kv_evict_unfiltered_dwpd": effective_dwpd_unfiltered,
            "kv_evict_burnout_months": burnout_months,
            "kv_write_admission_max_pct_waf1_5": max_admission_alpha_block_pct,
            "kv_write_admission_max_pct_zns_waf1_0": max_admission_alpha_zns_pct,
        },
        "models_kv_layout": models_kv,
        "assumptions": [
            "算例 13-1：8 块 PCIe Gen5 SSD 单盘有效读带宽 12.5 GB/s（聚合 100.0 GB/s，加载 140 GB 耗时 1.40 s），对比 POSIX 页缓存受限于 50.0 GB/s 跨 Root Complex 拷贝（2.80 s，2.0x）。",
            "算例 13-2：180 节点 3FS 存储集群单节点 16×7.0=112.0 GB/s 磁盘相对双口 200G 网卡（50.0 GB/s）超配 2.24x，聚合有效读吞吐达 6,600.6 GiB/s（6.60 TiB/s）。",
            "算例 13-3 与 13-9：Layer-major 2 KiB 单层切片触发 4 KiB LBA 2.0x 读放大（5.12 ms vs Block-major 0.625 ms，8.192x）；4×3.84 TB（3.0 DWPD）SSD 在 4.0 GB/s 换出与 WAF=1.5 下达 33.75 DWPD，需写准入过滤上限 <= 8.89%（ZNS WAF=1.0 下 <= 13.33%）。",
        ],
    }
