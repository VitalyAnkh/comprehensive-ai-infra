"""Ch13: 100 PB pretraining corpus purification funnel, MinHash-LSH integrals, and GPU dedup."""
from __future__ import annotations

from ..sources import model_config


def _simpson_integral(fn, a: float, b: float, n: int = 2000) -> float:
    h = (b - a) / n
    total = fn(a) + fn(b)
    for i in range(1, n):
        x = a + i * h
        total += (4.0 if i % 2 == 1 else 2.0) * fn(x)
    return total * h / 3.0


def calculate(
    raw_corpus_pb: float = 100.0,
    num_snapshots: int = 25,
    tokens_per_snapshot_trillion: float = 0.60,
    jaccard_threshold: float = 0.80,
    num_hashes: int = 128,
    num_docs: int = 5_000_000_000,
    dedup_gpus: int = 64,
    cpu_wall_hours: float = 148.0,
    cpu_cluster_cost_usd: float = 42_600.0,
    gpu_wall_hours: float = 1.85,
    gpu_cluster_cost_usd: float = 9_472.0,
) -> dict:
    qwen8 = model_config("qwen3-8b")
    dsv3 = model_config("deepseek-v3")

    # 1. 算例 13-15: 100.0 PB CommonCrawl WARC 7-stage purification funnel (25 merged snapshot clusters)
    # Retention ratios: r1=0.18, r2=0.75, r3=2/3, r4=1/3, r5=0.808, r6=0.9901 -> 2.40 PB (2.40% net yield)
    v0 = raw_corpus_pb
    v1 = round(v0 * 0.18, 4)            # 18.00 PB
    v2 = round(v1 * 0.75, 4)            # 13.50 PB
    v3 = round(v2 * (2.0 / 3.0), 4)     # 9.00 PB
    v4 = round(v3 * (1.0 / 3.0), 4)     # 3.00 PB (dedup removes 6.00 PB = 66.67%)
    v5 = round(v4 * 0.808, 4)           # 2.424 PB
    v6 = round(v5 * 0.9901, 2)          # 2.40 PB

    stages = [
        {"stage": "1_raw_warc_ingest", "input_pb": v0, "output_pb": v0, "retention_vs_prev": 1.00},
        {"stage": "2_html_text_extraction", "input_pb": v0, "output_pb": v1, "retention_vs_prev": 0.18},
        {"stage": "3_fasttext_language_id", "input_pb": v1, "output_pb": v2, "retention_vs_prev": 0.75},
        {"stage": "4_gopher_fineweb_heuristics", "input_pb": v2, "output_pb": v3, "retention_vs_prev": round(2.0 / 3.0, 6)},
        {"stage": "5_global_minhash_and_suffix_dedup", "input_pb": v3, "output_pb": v4, "retention_vs_prev": round(1.0 / 3.0, 6)},
        {"stage": "6_classifier_quality_tiering", "input_pb": v4, "output_pb": v5, "retention_vs_prev": 0.808},
        {"stage": "7_benchmark_decontamination", "input_pb": v5, "output_pb": v6, "retention_vs_prev": 0.9901},
    ]
    dedup_removed_pb = round(v3 - v4, 2)                                   # 6.00 PB
    dedup_removed_pct = round((v3 - v4) / v3 * 100.0, 2)                   # 66.67%
    net_byte_retention_pct = round(v6 / v0 * 100.0, 2)                     # 2.40%
    # Each of the 25 merged CommonCrawl snapshot clusters yields 0.60T curated tokens -> 25 * 0.60T = 15.0T tokens total
    # (At ~4.0 B/token uncompressed UTF-8 text, 15.0T tokens = 60.0 TB = 0.060 PB tokenized text; the 2.40 PB represents
    # multi-snapshot sharded/replicated columnar archive & rich metadata across the 100 PB WARC ingest pool.)
    final_tokens_trillion = round(num_snapshots * tokens_per_snapshot_trillion, 2)  # 15.0T tokens
    tokenized_utf8_tb_at_4b = round(final_tokens_trillion * 1e12 * 4.0 / 1e12, 2)   # 60.0 TB

    # 2. 算例 13-16 & 练习 13-21: MinHash-LSH S-curve (b=16, r=8 vs b=25, r=5 for M=128; and b=32, r=8 for M=256)
    lsh_configs = []
    for bands, rows in ((16, 8), (25, 5), (32, 8)):
        s_star = round((1.0 / bands) ** (1.0 / rows), 4)
        fp_area = round(_simpson_integral(lambda s, b=bands, r=rows: 1.0 - (1.0 - s**r) ** b, 0.0, jaccard_threshold), 5)
        fn_area = round(_simpson_integral(lambda s, b=bands, r=rows: (1.0 - s**r) ** b, jaccard_threshold, 1.0), 5)
        lsh_configs.append({
            "bands": bands,
            "rows_per_band": rows,
            "total_hashes_used": bands * rows,
            "inflection_s_star": s_star,
            "collision_prob_at_0_50_pct": round((1.0 - (1.0 - 0.50**rows) ** bands) * 100.0, 2),
            "collision_prob_at_0_70_pct": round((1.0 - (1.0 - 0.70**rows) ** bands) * 100.0, 2),
            "collision_prob_at_0_80_pct": round((1.0 - (1.0 - jaccard_threshold**rows) ** bands) * 100.0, 2),
            "collision_prob_at_0_90_pct": round((1.0 - (1.0 - 0.90**rows) ** bands) * 100.0, 2),
            "false_positive_integral": fp_area,
            "false_negative_integral": fn_area,
        })

    # 3. 算例 13-17: 5e9 docs MinHash signature shuffle & CPU DataTrove vs 64x H100 NeMo Curator (cuDF + cuGraph)
    sig_bytes_per_doc = num_hashes * 4                                     # 512 B/doc
    total_signature_gb = round(num_docs * sig_bytes_per_doc / 1e9, 1)      # 2,560.0 GB
    shuffle_64gpu_gb = round(total_signature_gb * (dedup_gpus - 1) / dedup_gpus, 1)  # 2,520.0 GB
    wall_clock_speedup_x = round(cpu_wall_hours / gpu_wall_hours, 1)       # 80.0x
    cost_reduction_x = round(cpu_cluster_cost_usd / gpu_cluster_cost_usd, 2)  # 4.50x

    return {
        "schema": "h2-topic-v1",
        "calculation": "data-funnel-dedup",
        "chapter": 13,
        "title": "第 13 章：百 PB 预训练语料七级提纯漏斗、MinHash-LSH 碰撞积分与 GPU 连通分量去重账本",
        "scenario": {
            "raw_corpus_pb": raw_corpus_pb,
            "num_snapshots": num_snapshots,
            "tokens_per_snapshot_trillion": tokens_per_snapshot_trillion,
            "jaccard_threshold": jaccard_threshold,
            "num_hashes": num_hashes,
            "num_docs": num_docs,
            "dedup_gpus": dedup_gpus,
            "qwen3_vocab_size": qwen8["vocab_size"],
            "deepseek_v3_vocab_size": dsv3["vocab_size"],
        },
        "summary": {
            "raw_ingest_pb": raw_corpus_pb,
            "post_html_extract_pb": v1,
            "post_heuristic_filter_pb": v3,
            "post_dedup_pb": v4,
            "dedup_stage_removed_pb": dedup_removed_pb,
            "dedup_stage_removed_pct": dedup_removed_pct,
            "final_curated_corpus_pb": v6,
            "net_byte_retention_pct": net_byte_retention_pct,
            "final_curated_tokens_trillion": final_tokens_trillion,
            "tokenized_utf8_equiv_tb_at_4b": tokenized_utf8_tb_at_4b,
            "b16_r8_inflection_s_star": lsh_configs[0]["inflection_s_star"],
            "b16_r8_p_at_0_50_pct": lsh_configs[0]["collision_prob_at_0_50_pct"],
            "b16_r8_p_at_0_70_pct": lsh_configs[0]["collision_prob_at_0_70_pct"],
            "b16_r8_p_at_0_80_pct": lsh_configs[0]["collision_prob_at_0_80_pct"],
            "b16_r8_p_at_0_90_pct": lsh_configs[0]["collision_prob_at_0_90_pct"],
            "b16_r8_fp_integral": lsh_configs[0]["false_positive_integral"],
            "b16_r8_fn_integral": lsh_configs[0]["false_negative_integral"],
            "b25_r5_inflection_s_star": lsh_configs[1]["inflection_s_star"],
            "b32_r8_m256_inflection_s_star": lsh_configs[2]["inflection_s_star"],
            "minhash_5b_docs_signature_gb": total_signature_gb,
            "minhash_64gpu_shuffle_gb": shuffle_64gpu_gb,
            "cpu_dedup_wall_hours": cpu_wall_hours,
            "gpu_curator_wall_hours": gpu_wall_hours,
            "gpu_vs_cpu_wall_speedup_x": wall_clock_speedup_x,
            "cpu_dedup_cost_usd": cpu_cluster_cost_usd,
            "gpu_curator_dedup_cost_usd": gpu_cluster_cost_usd,
            "gpu_vs_cpu_cost_reduction_x": cost_reduction_x,
        },
        "funnel_stages": stages,
        "lsh_band_row_comparison": lsh_configs,
        "assumptions": [
            "算例 13-15：100.0 PB CommonCrawl WARC 归档（25 个合并快照簇）经七级漏斗逐级收敛至 18.00 -> 13.50 -> 9.00 -> 3.00 -> 2.424 -> 2.40 PB（净留存率 2.40%），每快照簇贡献 0.60T 词元，合计产出精确 15.0T 精品训练词元（纯 UTF-8 词元文本按 4 B/tok 折合 60.0 TB）。",
            "算例 13-16：M=128 个 32-bit MinHash 切分为 b=16, r=8 时拐点 s* = 0.7071，P(0.50)=6.07%, P(0.70)=61.32%, P(0.80)=94.70%, P(0.90)=99.99%，假阳/假阴面积由 2,000 段复合 Simpson 积分精确计算。",
            "算例 13-17：50 亿篇文档（512 B/篇）共 2,560.0 GB 签名，64 卡洗牌流量 2,520.0 GB；64×H100 NeMo Curator 耗时 1.85 h / $9,472，对比 4,096 核 CPU 集群 148.0 h / $42,600 实现 80.0x 提速与 4.50x 成本压降。",
        ],
    }
