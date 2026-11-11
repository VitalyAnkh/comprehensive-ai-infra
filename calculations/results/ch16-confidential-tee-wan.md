# 第 16 章：Vera Rubin 原生 TEE-I/O、AES-CTR 投机预掩码、微虚机去重与广域网防伪抽检账本

- 计算标识：`confidential-tee-wan`（第 16 章）
- 场景输入：`{"byzantine_node_fraction": 0.15, "cpu_bounce_bw_gbps": 8.4, "dsec_sandboxes": 380000, "pol_outer_rounds": 10, "pol_spot_check_rate": 0.05, "qwen235_layers": 94, "rubin_hbm_gb": 288, "rubin_tee_io_bw_gbps": 112.0, "tensor_mib": 64.0, "wan_bandwidth_gbps": 5.0}`

## 核心量化结果

| 指标 | 数值 |
| --- | ---: |
| `bounce_buffer_64mib_ms` | 8.8 |
| `rubin_tee_io_64mib_single_ms` | 0.6 |
| `rubin_tee_io_64mib_pipelined_ms` | 0.4 |
| `tee_io_speedup_single_x` | 14.7 |
| `tee_io_speedup_pipelined_x` | 22.0 |
| `pipellm_triple_buffer_mib` | 192.0 |
| `aes_ctr_critical_hit_rate_pct` | 33.3 |
| `aes_ctr_critical_hit_rate_tdx_pct` | 50.0 |
| `dax_struct_page_metadata_tax_pct` | 1.5625 |
| `dsec_380k_naive_ram_tb` | 242.26 |
| `dsec_380k_shared_dax_ram_tb` | 38.25 |
| `dsec_memory_saving_pct` | 84.21 |
| `wan_streaming_diloco_fp4_payload_gb` | 124.84 |
| `wan_per_fragment_transfer_seconds` | 14.19 |
| `pol_10round_detection_pct` | 99.97 |
| `geom_median_breakdown_point` | 0.5 |
| `geom_median_variance_inflation_x` | 1.43 |
| `spot_dht_elastic_ettr_pct` | 96.8 |

## 建模口径与物理边界

- 算例 16-1 与 16-3：64 MiB（67.11 MB）张量经软件弹跳缓冲需 8.8 ms，Vera Rubin PCIe Gen6 原生 TEE-I/O 单块仅 0.60 ms（流水 0.40 ms，提速 14.7x–22.0x）；投机预加密临界命中率为 33.3%（开启 CPU TDX 时为 50.0%）。
- 算例 16-14 与 16-15：开放算力池 K=100 含 f=15% 拜占庭节点时，p_check=5% 随机重放抽检（p_1=0.556）在 10 轮外循环内捕获率达 99.97%，Weiszfeld 几何中位数（崩溃点 0.50）将方差膨胀限制在 1.43x 以内。
