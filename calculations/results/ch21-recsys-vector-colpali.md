# 第 21 章：三级嵌入缓存、HSTU/M-FALCON、ColPali/PLAID 残差检索与无锁流式热更新账本

- 计算标识：`recsys-vector-colpali`（第 21 章）
- 场景输入：`{"cuckoo_freq_threshold": 5, "daily_exposure_clicks": 4000000000, "embed_dim": 128, "num_pages": 50000000, "patches_per_page": 1030, "raw_sparse_table_tb": 2.4, "recall_k1": 1000, "rerank_k2": 64, "stream_updates_per_sec": 2500000}`

## 核心量化结果

| 指标 | 数值 |
| --- | ---: |
| `dlrm_3tier_weighted_lookup_us` | 2.19 |
| `dlrm_3tier_vs_ssd_speedup_x` | 29.67 |
| `hstu_stochastic_length_saving_pct` | 56.2 |
| `mfalcon_m1024_flops_reduction_x` | 44.8 |
| `mfalcon_h100_qps` | 485 |
| `colpali_bf16_kb_per_page` | 263.68 |
| `colpali_50m_bf16_tb` | 13.18 |
| `colpali_100m_bf16_tb` | 26.37 |
| `colpali_1b_bf16_tb` | 263.7 |
| `plaid_2bit_bytes_per_vec` | 34.0 |
| `plaid_2bit_50m_tb` | 1.75 |
| `plaid_2bit_compression_x` | 7.53 |
| `plaid_1bit_bytes_per_vec` | 17.5 |
| `plaid_1bit_50m_gb` | 901.3 |
| `plaid_1bit_100m_tb` | 1.8 |
| `plaid_1bit_1b_tb` | 18.0 |
| `plaid_1bit_compression_x` | 14.6 |
| `plaid_maxsim_p99_ms` | 14.2 |
| `hybrid_bm25_cuvs_ce_p99_ms` | 27.6 |
| `cuckoo_tau5_embedding_shrink_pct` | 68.5 |
| `cuckoo_embedding_active_gb` | 756.0 |
| `hot_update_rwlock_p99_ms` | 18.4 |
| `hot_update_epoch_lockfree_p99_ms` | 1.15 |
| `hot_update_tail_reduction_x` | 16.0 |
| `hot_update_epoch_memory_overhead_pct` | 6.2 |

## 建模口径与物理边界

- 算例 21-1、21-3 与 21-4：10.0 TB 稀疏表经 HBM(82%)/CXL(15.5%)/SSD(2.5%) 三级缓存将平均查表压至 2.19 us（29.67x）；HSTU 随机长度压缩省算力 56.2%，M-FALCON 千候选前缀摊销缩减算力 44.80x（32 卡 H100 实例组达 485 QPS）。
- 算例 21-7 与 21-8：ColPali 每页 1,030x128 BF16 占 263.68 KB（50M 页 13.18 TB）；PLAID 2-bit（34 B/vec）压至 1.75 TB（7.53x），1-bit（17.5 B/vec）压至 901.3 GB（14.6x，十亿页 18.0 TB）；BM25+cuVS (K1=1,000) 接 Cross-Encoder 0.6B (K2=64) P99 为 27.6 ms < 30 ms。
- 算例 21-9 与 21-10：日增 40 亿曝光流下频次阈值 tau=5 准入过滤缩减 HBM 活跃表 68.5%；每秒 250 万向量更新下 Epoch 无锁双缓冲将 P99 从 18.4 ms 压至 1.15 ms（额外内存仅 +6.2%）。
