# 第 13 章：百 PB 预训练语料七级提纯漏斗、MinHash-LSH 碰撞积分与 GPU 连通分量去重账本

- 计算标识：`data-funnel-dedup`（第 13 章）
- 场景输入：`{"dedup_gpus": 64, "deepseek_v3_vocab_size": 129280, "jaccard_threshold": 0.8, "num_docs": 5000000000, "num_hashes": 128, "num_snapshots": 25, "qwen3_vocab_size": 151936, "raw_corpus_pb": 100.0, "tokens_per_snapshot_trillion": 0.6}`

## 核心量化结果

| 指标 | 数值 |
| --- | ---: |
| `raw_ingest_pb` | 100.0 |
| `post_html_extract_pb` | 18.0 |
| `post_heuristic_filter_pb` | 9.0 |
| `post_dedup_pb` | 3.0 |
| `dedup_stage_removed_pb` | 6.0 |
| `dedup_stage_removed_pct` | 66.67 |
| `final_curated_corpus_pb` | 2.4 |
| `net_byte_retention_pct` | 2.4 |
| `final_curated_tokens_trillion` | 15.0 |
| `tokenized_utf8_equiv_tb_at_4b` | 60.0 |
| `b16_r8_inflection_s_star` | 0.7071 |
| `b16_r8_p_at_0_50_pct` | 6.07 |
| `b16_r8_p_at_0_70_pct` | 61.33 |
| `b16_r8_p_at_0_80_pct` | 94.7 |
| `b16_r8_p_at_0_90_pct` | 99.99 |
| `b16_r8_fp_integral` | 0.13825 |
| `b16_r8_fn_integral` | 0.00128 |
| `b25_r5_inflection_s_star` | 0.5253 |
| `b32_r8_m256_inflection_s_star` | 0.6484 |
| `minhash_5b_docs_signature_gb` | 2,560.0 |
| `minhash_64gpu_shuffle_gb` | 2,520.0 |
| `cpu_dedup_wall_hours` | 148.0 |
| `gpu_curator_wall_hours` | 1.85 |
| `gpu_vs_cpu_wall_speedup_x` | 80.0 |
| `cpu_dedup_cost_usd` | 42,600.0 |
| `gpu_curator_dedup_cost_usd` | 9,472.0 |
| `gpu_vs_cpu_cost_reduction_x` | 4.5 |

## 建模口径与物理边界

- 算例 13-15：100.0 PB CommonCrawl WARC 归档（25 个合并快照簇）经七级漏斗逐级收敛至 18.00 -> 13.50 -> 9.00 -> 3.00 -> 2.424 -> 2.40 PB（净留存率 2.40%），每快照簇贡献 0.60T 词元，合计产出精确 15.0T 精品训练词元（纯 UTF-8 词元文本按 4 B/tok 折合 60.0 TB）。
- 算例 13-16：M=128 个 32-bit MinHash 切分为 b=16, r=8 时拐点 s* = 0.7071，P(0.50)=6.07%, P(0.70)=61.32%, P(0.80)=94.70%, P(0.90)=99.99%，假阳/假阴面积由 2,000 段复合 Simpson 积分精确计算。
- 算例 13-17：50 亿篇文档（512 B/篇）共 2,560.0 GB 签名，64 卡洗牌流量 2,520.0 GB；64×H100 NeMo Curator 耗时 1.85 h / $9,472，对比 4,096 核 CPU 集群 148.0 h / $42,600 实现 80.0x 提速与 4.50x 成本压降。
