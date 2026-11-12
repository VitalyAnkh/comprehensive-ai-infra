"""Ch21: Zipf 3-tier cache, HSTU+M-FALCON, ColPali/PLAID 1-bit/2-bit RAG, and Epoch lock-free hot updates."""
from __future__ import annotations

from .. import hardware
from ..sources import model_config


def calculate(
    num_pages: int = 50_000_000,
    patches_per_page: int = 1030,
    embed_dim: int = 128,
    daily_exposure_clicks: int = 4_000_000_000,
    cuckoo_freq_threshold: int = 5,
    raw_sparse_table_tb: float = 2.40,
    stream_updates_per_sec: int = 2_500_000,
) -> dict:
    hw_map = {d["id"]: d for d in hardware.catalog()["devices"]}
    qwen_cfg = model_config("qwen3-8b")
    _ = (hw_map["h100-sxm"], int(qwen_cfg["num_hidden_layers"]))

    # 1. 算例 21-1: 10.0 TB (2e10 x 500 B) 3-tier Zipf (alpha=1.12) embedding cache lookup latency
    p_hbm, p_cxl, p_ssd = 0.820, 0.155, 0.025
    t_hbm_us, t_cxl_us, t_ssd_us = 0.35, 1.80, 65.00
    weighted_lookup_us = round(p_hbm * t_hbm_us + p_cxl * t_cxl_us + p_ssd * t_ssd_us, 2)  # 2.19 us (2.191 us)
    three_tier_speedup_x = round(t_ssd_us / (p_hbm * t_hbm_us + p_cxl * t_cxl_us + p_ssd * t_ssd_us), 2)  # 29.67x

    # 2. 算例 21-3 & 21-4: HSTU stochastic length compression (56.2%) & M-FALCON M=1,024 prefix amortization (44.80x)
    hstu_layer_flops_saving_pct = 56.2
    mfalcon_naive_tflops = round(1024 * 1.37627, 1)                        # 1,409.3 TFLOPs
    mfalcon_amortized_tflops = round(1.376 + 30.084, 2)                    # 31.46 TFLOPs
    mfalcon_reduction_x = round(mfalcon_naive_tflops / mfalcon_amortized_tflops, 2)  # 44.80x
    pod32_sustained_tflops = 32 * 476.815625                               # 15,258.1 TFLOP/s across 32x H100 (~48.2% MFU)
    mfalcon_h100_qps = round(pod32_sustained_tflops / mfalcon_amortized_tflops)  # 485 QPS

    # 3. 算例 21-7 (PLAN.md §5.5): ColPali 50M PDF pages (1,030 patches x 128 dim) BF16 vs PLAID 2-bit / 1-bit
    bf16_bytes_per_vec = embed_dim * 2.0                                   # 256.0 B/vec
    bf16_bytes_per_page = patches_per_page * bf16_bytes_per_vec            # 263,680.0 B = 263.68 KB/page
    bf16_50m_tb = round(num_pages * bf16_bytes_per_page / 1e12, 2)         # 13.18 TB
    bf16_100m_tb = round(100_000_000 * bf16_bytes_per_page / 1e12, 2)      # 26.37 TB
    bf16_1b_tb = round(1_000_000_000 * bf16_bytes_per_page / 1e12, 1)      # 263.7 TB

    # PLAID 2-bit: 32 B (2-bit residual) + 2 B (16-bit centroid ID) = exact 34.0 B/vec -> 35,020 B/page
    plaid_2bit_bytes_per_vec = (embed_dim * 2) / 8.0 + 2.0                 # 34.0 B/vec
    plaid_2bit_bytes_per_page = patches_per_page * plaid_2bit_bytes_per_vec  # 35,020.0 B/page
    plaid_2bit_50m_tb = round(num_pages * plaid_2bit_bytes_per_page / 1e12, 2)  # 1.75 TB (1.751 TB)
    plaid_2bit_compression_x = round(bf16_bytes_per_vec / plaid_2bit_bytes_per_vec, 2)  # 7.53x

    # PLAID 1-bit: 16 B (1-bit binary residual) + 1.5 B (packed 12-bit centroid ID) = exact 17.5 B/vec -> 18,025 B/page
    plaid_1bit_bytes_per_vec = (embed_dim * 1) / 8.0 + 1.5                 # 17.5 B/vec
    plaid_1bit_bytes_per_page = patches_per_page * plaid_1bit_bytes_per_vec  # 18,025.0 B/page
    plaid_1bit_50m_gb = round(num_pages * plaid_1bit_bytes_per_page / 1e9 + 1e-9, 1)  # 901.3 GB (exact 901.25 GB)
    plaid_1bit_100m_tb = round(100_000_000 * plaid_1bit_bytes_per_page / 1e12, 2)     # 1.80 TB
    plaid_1bit_1b_tb = round(1_000_000_000 * plaid_1bit_bytes_per_page / 1e12, 1)     # 18.0 TB
    plaid_1bit_compression_x = round(bf16_bytes_per_vec / plaid_1bit_bytes_per_vec, 1)  # 14.6x (14.63x)
    plaid_maxsim_p99_ms = 14.2

    # 4. 算例 21-8: Under 30 ms SLA, BM25 + cuVS parallel recall (K1=1,000) -> Cross-Encoder (0.6B) reranking (K2=64)
    recall_k1 = 1000
    rerank_k2 = 64
    bm25_wand_p99_ms = 6.4
    cuvs_cagra_p99_ms = 9.8
    parallel_recall_ms = max(bm25_wand_p99_ms, cuvs_cagra_p99_ms)          # 9.8 ms
    rrf_fusion_ms = 0.8
    cross_encoder_0_6b_k64_ms = 14.6
    rpc_budget_ms = 2.4
    hybrid_e2e_p99_ms = round(parallel_recall_ms + rrf_fusion_ms + cross_encoder_0_6b_k64_ms + rpc_budget_ms, 1)  # 27.6 ms < 30 ms

    # 5. 算例 21-9: 4B/day exposure stream, Cuckoo frequency admission (tau = 5) + 30d exponential decay -> 68.5% shrink
    cuckoo_shrink_pct = 68.5
    cuckoo_active_gb = round(raw_sparse_table_tb * 1000.0 * (1.0 - cuckoo_shrink_pct / 100.0), 1)  # 756.0 GB

    # 6. 算例 21-10: 2.5M vector updates/s, segmented RW-lock (18.4 ms) vs Epoch dual-buffer lock-free (1.15 ms, +6.2% mem)
    rwlock_p99_ms = 18.4
    epoch_lockfree_p99_ms = 1.15
    tail_latency_reduction_x = round(rwlock_p99_ms / epoch_lockfree_p99_ms, 1)  # 16.0x
    epoch_extra_memory_overhead_pct = 6.2

    return {
        "schema": "h2-topic-v1",
        "calculation": "recsys-vector-colpali",
        "chapter": 21,
        "title": "第 21 章：三级嵌入缓存、HSTU/M-FALCON、ColPali/PLAID 残差检索与无锁流式热更新账本",
        "scenario": {
            "num_pages": num_pages,
            "patches_per_page": patches_per_page,
            "embed_dim": embed_dim,
            "recall_k1": recall_k1,
            "rerank_k2": rerank_k2,
            "daily_exposure_clicks": daily_exposure_clicks,
            "cuckoo_freq_threshold": cuckoo_freq_threshold,
            "raw_sparse_table_tb": raw_sparse_table_tb,
            "stream_updates_per_sec": stream_updates_per_sec,
        },
        "summary": {
            "dlrm_3tier_weighted_lookup_us": weighted_lookup_us,
            "dlrm_3tier_vs_ssd_speedup_x": three_tier_speedup_x,
            "hstu_stochastic_length_saving_pct": hstu_layer_flops_saving_pct,
            "mfalcon_m1024_flops_reduction_x": mfalcon_reduction_x,
            "mfalcon_h100_qps": mfalcon_h100_qps,
            "colpali_bf16_kb_per_page": round(bf16_bytes_per_page / 1000.0, 2),
            "colpali_50m_bf16_tb": bf16_50m_tb,
            "colpali_100m_bf16_tb": bf16_100m_tb,
            "colpali_1b_bf16_tb": bf16_1b_tb,
            "plaid_2bit_bytes_per_vec": plaid_2bit_bytes_per_vec,
            "plaid_2bit_50m_tb": plaid_2bit_50m_tb,
            "plaid_2bit_compression_x": plaid_2bit_compression_x,
            "plaid_1bit_bytes_per_vec": plaid_1bit_bytes_per_vec,
            "plaid_1bit_50m_gb": plaid_1bit_50m_gb,
            "plaid_1bit_100m_tb": plaid_1bit_100m_tb,
            "plaid_1bit_1b_tb": plaid_1bit_1b_tb,
            "plaid_1bit_compression_x": plaid_1bit_compression_x,
            "plaid_maxsim_p99_ms": plaid_maxsim_p99_ms,
            "hybrid_bm25_cuvs_ce_p99_ms": hybrid_e2e_p99_ms,
            "cuckoo_tau5_embedding_shrink_pct": cuckoo_shrink_pct,
            "cuckoo_embedding_active_gb": cuckoo_active_gb,
            "hot_update_rwlock_p99_ms": rwlock_p99_ms,
            "hot_update_epoch_lockfree_p99_ms": epoch_lockfree_p99_ms,
            "hot_update_tail_reduction_x": tail_latency_reduction_x,
            "hot_update_epoch_memory_overhead_pct": epoch_extra_memory_overhead_pct,
        },
        "assumptions": [
            "算例 21-1、21-3 与 21-4：10.0 TB 稀疏表经 HBM(82%)/CXL(15.5%)/SSD(2.5%) 三级缓存将平均查表压至 2.19 us（29.67x）；HSTU 随机长度压缩省算力 56.2%，M-FALCON 千候选前缀摊销缩减算力 44.80x（32 卡 H100 实例组达 485 QPS）。",
            "算例 21-7 与 21-8：ColPali 每页 1,030x128 BF16 占 263.68 KB（50M 页 13.18 TB）；PLAID 2-bit（34 B/vec）压至 1.75 TB（7.53x），1-bit（17.5 B/vec）压至 901.3 GB（14.6x，十亿页 18.0 TB）；BM25+cuVS (K1=1,000) 接 Cross-Encoder 0.6B (K2=64) P99 为 27.6 ms < 30 ms。",
            "算例 21-9 与 21-10：日增 40 亿曝光流下频次阈值 tau=5 准入过滤缩减 HBM 活跃表 68.5%；每秒 250 万向量更新下 Epoch 无锁双缓冲将 P99 从 18.4 ms 压至 1.15 ms（额外内存仅 +6.2%）。",
        ],
    }
