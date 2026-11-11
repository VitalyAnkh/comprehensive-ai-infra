# 第 13 章：AI 存储内核旁路（GDS）、3FS 链式复制、KV 闪存块布局与 SSD 寿命准入账本

- 计算标识：`storage-kv-io`（第 13 章）
- 场景输入：`{"block_tokens": 16, "checkpoint_gb": 140.0, "lba_bytes": 4096, "nvme_drives": 8, "per_drive_gds_eff_gbps": 12.5, "per_drive_seq_read_gbps": 14.0, "reference_hbm_bw_gbps": 8000.0, "ssd_dwpd_rated": 3.0, "ssd_total_capacity_tb": 15.36, "threefs_nodes": 180}`

## 核心量化结果

| 指标 | 数值 |
| --- | ---: |
| `raw_nvme_array_gbps` | 112.0 |
| `gds_bandwidth_gbps` | 100.0 |
| `pagecache_bandwidth_gbps` | 50.0 |
| `pagecache_rc_traffic_gb` | 280.0 |
| `gds_140gb_load_seconds` | 1.4 |
| `pagecache_140gb_load_seconds` | 2.8 |
| `gds_vs_pagecache_speedup` | 2.0 |
| `threefs_disk_to_nic_overprovision_x` | 2.24 |
| `threefs_180node_read_gibps` | 6,600.6 |
| `threefs_180node_read_tibps` | 6.6 |
| `threefs_180node_read_tibps_binary` | 6.45 |
| `threefs_180node_craq_write_tibps` | 2.2 |
| `threefs_kvcache_peak_tib_per_min` | 3.66 |
| `threefs_kvcache_peak_gibps` | 62.46 |
| `ssd_iops_bw_critical_block_kib` | 16.384 |
| `ref_32layer_16tok_layer_major_raf_lba` | 2.0 |
| `ref_32layer_16tok_layer_major_raf_page16k` | 8.0 |
| `ref_32layer_128blk_layer_major_ms` | 5.12 |
| `ref_32layer_128blk_block_major_ms` | 0.625 |
| `ref_32layer_block_vs_layer_speedup_x` | 8.192 |
| `qwen3_8b_kv_bytes_per_token` | 147,456 |
| `deepseek_v4_flash_kv_bytes_per_token` | 49,536 |
| `kv_evict_host_write_tb_per_day` | 345.6 |
| `kv_evict_nand_write_tb_per_day` | 518.4 |
| `kv_evict_unfiltered_dwpd` | 33.75 |
| `kv_evict_burnout_months` | 5.3 |
| `kv_write_admission_max_pct_waf1_5` | 8.89 |
| `kv_write_admission_max_pct_zns_waf1_0` | 13.33 |

## 建模口径与物理边界

- 算例 13-1：8 块 PCIe Gen5 SSD 单盘有效读带宽 12.5 GB/s（聚合 100.0 GB/s，加载 140 GB 耗时 1.40 s），对比 POSIX 页缓存受限于 50.0 GB/s 跨 Root Complex 拷贝（2.80 s，2.0x）。
- 算例 13-2：180 节点 3FS 存储集群单节点 16×7.0=112.0 GB/s 磁盘相对双口 200G 网卡（50.0 GB/s）超配 2.24x，聚合有效读吞吐达 6,600.6 GiB/s（6.60 TiB/s）。
- 算例 13-3 与 13-9：Layer-major 2 KiB 单层切片触发 4 KiB LBA 2.0x 读放大（5.12 ms vs Block-major 0.625 ms，8.192x）；4×3.84 TB（3.0 DWPD）SSD 在 4.0 GB/s 换出与 WAF=1.5 下达 33.75 DWPD，需写准入过滤上限 <= 8.89%（ZNS WAF=1.0 下 <= 13.33%）。
