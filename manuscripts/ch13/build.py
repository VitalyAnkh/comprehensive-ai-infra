#!/usr/bin/env python3
"""Build Chapter 13 figures, locked storage calculations, and offline HTML preview."""
from pathlib import Path
import base64
import hashlib
import json
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figure_style import COL, BORDER_COL, STYLE, Exporter, arrow, box, card, canvas, plot, text
from preview_output import preview_path

plt.rcParams.update(STYLE)


def sha256(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b'version https://git-lfs.github.com/spec/v1'):
        m = re.search(rb'oid sha256:([0-9a-f]{64})', raw)
        if m:
            return m.group(1).decode('ascii')
    return hashlib.sha256(raw).hexdigest()


def build_figure_data() -> dict:
    return {
        "chapter": "ch13",
        "title": "存储系统与数据工程",
        "curation_funnel": {
                "source": "experiments/ch23/23-01/run.py (100 PB CommonCrawl 7-Stage Curation Funnel to 15T Tokens)",
                "raw_warc_pb": 100.0,
                "after_html_extract_pb": 18.0,
                "after_langid_url_pb": 13.5,
                "after_heuristic_pb": 9.0,
                "after_fuzzy_dedup_pb": 3.0,
                "after_quality_tier_pb": 2.424,
                "after_decontam_pb": 2.4,
                "final_curated_tokens_t": 15.0,
                "overall_warc_byte_yield_pct": 2.4,
                "extracted_text_token_yield_pct": 13.33
        },
        "minhash_lsh_scurve": {
                "source": "experiments/ch23/23-01/run.py (MinHash LSH M=128, b=16, r=8 S-Curve Probability)",
                "num_hashes_m": 128,
                "b_bands": 16,
                "r_rows": 8,
                "threshold_s_star": 0.7071,
                "prob_at_s_050_pct": 6.07,
                "prob_at_s_070_pct": 61.33,
                "prob_at_s_080_pct": 94.7,
                "prob_at_s_090_pct": 99.99
        },
        "gpu_fuzzy_dedup_cluster": {
                "source": "experiments/ch23/23-01/run.py (5B Docs 64-GPU NeMo Curator MinHash + cuGraph Union-Find)",
                "num_docs": 5000000000,
                "sig_bytes_per_doc": 512,
                "total_signature_gb": 2560.0,
                "all_to_all_shuffle_gb": 2520.0,
                "cpu_cluster_hours": 148.0,
                "gpu_curator_hours": 1.85,
                "dedup_speedup_x": 80.0,
                "tco_reduction_x": 4.5
        },
        "rubric_distillation": {
                "source": "experiments/ch23/23-02/run.py (70B Teacher to 110M Classifier Rubric Distillation)",
                "direct_70b_gpu_hours": 2170138.9,
                "seed_labeling_gpu_hours": 217.0,
                "student_scoring_gpu_hours": 38580.2,
                "distilled_total_gpu_hours": 38877.2,
                "distillation_compute_leverage_x": 55.82
        },
        "synthetic_data_factory": {
                "source": "experiments/ch23/23-02/run.py (Offline Synthetic Data Factory Batch=2048 vs Online Serving Batch=32)",
                "online_batch_size": 32,
                "online_tok_s_gpu": 1920.0,
                "online_mfu_pct": 11.2,
                "online_usd_per_1m_tokens": 0.579,
                "offline_batch_size": 2048,
                "offline_tok_s_gpu": 13824.0,
                "offline_mfu_pct": 58.4,
                "throughput_gain_x": 7.2,
                "offline_raw_usd_per_1m_tokens": 0.044,
                "raw_cost_reduction_x": 13.16,
                "raw_sandbox_pass_yield_pct": 31.25,
                "effective_verified_yield_pct": 58.75,
                "verified_usd_per_1m_tokens": 0.075
        },
        "mineru_olmocr_pdf_pipeline": {
                "source": "experiments/ch23/23-03/run.py (2M Academic PDF Books 500M Pages MinerU/olmOCR Hybrid GPU Pipeline)",
                "total_books": 2000000,
                "total_pages": 500000000,
                "digital_born_ratio": 0.65,
                "avg_gpu_ms_per_page": 17.72,
                "throughput_pages_per_s_gpu": 56.43,
                "speedup_vs_monolithic_vlm_x": 35.27,
                "speedup_vs_serial_mineru_x": 4.51,
                "total_gpu_hours": 2461.11,
                "total_cost_usd": 5414.44,
                "cost_per_1m_tokens_usd": 0.0135
        },
        "repo_ast_topological_packing": {
                "source": "experiments/ch23/23-03/run.py (64k Repo-Level AST Dependency Graph + Tarjan SCC + Kahn Topological Packing)",
                "num_repo_files": 18,
                "total_cross_file_edges": 40,
                "alphabetical_causal_coverage_pct": 42.5,
                "random_causal_coverage_pct": 50.0,
                "topological_causal_coverage_pct": 95.0,
                "coverage_gain_pct_points": 52.5,
                "ppl_alphabetical": 6.85,
                "ppl_topological": 4.12,
                "ppl_reduction_pct": 39.85,
                "packing_efficiency_pct": 98.8
        },
        "example_13_1_gds": {
            "weight_size_gb": 140.0,
            "ssd_count": 8,
            "per_ssd_raw_gbps": 14.0,
            "per_ssd_eff_gbps": 12.5,
            "posix_rc_traffic_gb": 280.0,
            "posix_eff_bw_gbps": 50.0,
            "posix_load_seconds": 2.80,
            "gds_rc_payload_gb": 0.0,
            "gds_eff_bw_gbps": 100.0,
            "gds_load_seconds": 1.40,
            "speedup": 2.0
        },
        "example_13_2_3fs_craq": {
            "nodes": 180,
            "nics_per_node": 2,
            "nic_gbps": 200,
            "node_net_line_gb_s": 50.0,
            "node_net_eff_gib_s": 36.67,
            "ssds_per_node": 16,
            "ssd_read_gb_s": 7.0,
            "node_disk_raw_gb_s": 112.0,
            "disk_to_net_ratio": 2.24,
            "cluster_read_tib_s": 6.6,
            "kvcache_read_tib_min": 3.66,
            "craq_replicas": 3
        },
        "example_13_3_kv_layout": {
            "layers": 32,
            "tokens_per_block": 16,
            "bytes_per_layer_per_block": 2048,
            "valid_bytes_per_block": 65536,
            "lba_min_bytes": 4096,
            "flash_page_bytes": 16384,
            "layer_major_raf_lba": 2.0,
            "layer_major_raf_page": 8.0,
            "block_major_raf": 1.0,
            "num_blocks_fetched": 128,
            "total_valid_mib": 8.0,
            "ssd_max_iops": 800000,
            "ssd_eff_bw_gib_s": 12.5,
            "layer_major_ios": 4096,
            "layer_major_ms": 5.12,
            "block_major_ios": 128,
            "block_major_ms": 0.625,
            "speedup": 8.192
        },
        "example_13_4_multimodal_dataloader": {
            "clips_per_sec": 4096,
            "frames_per_clip": 16,
            "total_fps": 65536,
            "cpu_fps_per_core": 256,
            "cpu_clips_per_core": 16,
            "required_cpu_cores_soft": 256,
            "host_total_cores": 128,
            "host_reserved_cores": 32,
            "host_avail_decode_cores": 96,
            "cpu_only_supply_ratio": 0.375,
            "gpu_count": 8,
            "nvdec_fps_per_gpu": 9600,
            "nvdec_total_fps": 76800,
            "dali_demux_cpu_cores": 32
        },
        "example_13_5_dsec_erofs": {
            "containers_booted": 8192,
            "nodes": 10,
            "full_pull_minutes_gt": 60,
            "erofs_ondemand_minutes": 35,
            "full_pull_write_gb_per_node_gt": 1600,
            "erofs_ondemand_write_gb_per_node": 700,
            "all_cached_write_gb_per_node": 600,
            "tar_unpack_minutes": 79,
            "erofs_mount_minutes": 45
        },
        "example_13_6_iks_vector_scan": {
            "num_vectors": 33554432,
            "dim": 768,
            "fp32_gib": 96.0,
            "fp16_gib": 48.0,
            "fp16_gb": 51.540,
            "scan_gflops": 51.540,
            "ai_batch1_flop_per_byte": 1.0,
            "ai_batch16_flop_per_byte": 16.0,
            "host_link_gbps": 64.0,
            "host_transfer_ms": 805.306,
            "iks_internal_tbps": 1.0,
            "iks_scan_ms": 51.540,
            "shards": 8,
            "query_broadcast_bytes": 12288,
            "topk_per_shard": 32,
            "response_merge_bytes": 3072
        },
        "experiment_13_01_rag_pipeline": {
            "corpus_docs": 1024,
            "corpus_json_bytes": 466051,
            "corpus_vectors_npy_bytes": 1572992,
            "encoder_dim": 384,
            "encoder_load_s": 1.691928,
            "doc_encode_s": 11.302180,
            "query_encode_s": 0.050709,
            "flat_build_s": 0.000216,
            "flat_bytes": 1572909,
            "hnsw_build_s": 0.023254,
            "hnsw_bytes": 1655730,
            "total_requests": 288,
            "strict_pass": 281,
            "kv_pool_blocks": 910,
            "kv_pool_bytes": 2146959360,
            "kv_block_bytes": 2359296,
            "k4_peak_blocks": 23,
            "k4_peak_bytes": 54263808,
            "k16_peak_blocks": 77,
            "k16_peak_bytes": 181665792,
            "worker_cuda_peak_bytes": 18582745600,
            "tagged_rss_peak_bytes": 2233159680
        },
        "example_13_7_ssd_dwpd": {
            "ssd_count": 4,
            "ssd_capacity_tb": 3.84,
            "total_ssd_capacity_tb": 15.36,
            "kv_evict_gbps": 4.0,
            "host_write_tb_per_day": 345.6,
            "waf": 1.5,
            "nand_write_tb_per_day": 518.4,
            "effective_dwpd": 33.75,
            "target_dwpd_limit": 3.0,
            "max_admission_ratio": 0.0889
        },
        "example_13_11_nixl_pipeline": {
            "total_kv_gb": 8.0,
            "tier_volumes_gb": [3.2, 3.2, 1.6],
            "tier_bw_gbps": [50.0, 40.0, 20.0],
            "tier_init_ms": [0.05, 0.10, 0.15],
            "sync_switch_ms": 0.40,
            "nixl_desc_ms": 0.05,
            "serial_fetch_ms": 225.10,
            "nixl_parallel_ms": 80.20,
            "speedup": 2.81,
            "effective_bw_gbps": 99.75,
            "exposed_after_overlap_ms": 20.05
        },
        "example_13_12_checkpoint_ec": {
            "ckpt_raw_tb": 10.0,
            "total_ckpts": 100,
            "hot_keep_count": 3,
            "cold_count": 97,
            "rep_factor": 3.0,
            "ec_k": 10,
            "ec_m": 4,
            "ec_factor": 1.4,
            "pure_3rep_tb": 3000.0,
            "tiered_ec_tb": 1448.0,
            "saved_tb": 1552.0,
            "saving_pct": 51.73,
            "per_node_parity_mb_s": 66.7
        },
        "experiment_13_04_lmcache_qos": {
            "raw_kv_mib": 8192.0,
            "compressed_kv_mib": 2155.79,
            "compression_ratio": 3.80,
            "uncompressed_serial_ms": 713.60,
            "compressed_pipelined_ms": 273.02,
            "ttft_speedup_x": 2.61,
            "three_tier_avg_ms": 230.11,
            "three_tier_cost_saving_pct": 87.98,
            "fifo_p99_read_ms": 13.80,
            "wrr_p99_read_ms": 1.35,
            "tail_latency_reduction_x": 10.22
        }
    }


def fig_13_1_hierarchy(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.49, 0.96, 0.48, '存储介质金字塔：容量、延迟、IOPS 与带宽账本', color='gray')
    tiers = [
        (0.04, 0.675, 0.29, 0.175, 'L1: GPU HBM3e\n80–192 GiB · 100 ns\n3.35–8.0 TB/s', 'blue'),
        (0.355, 0.675, 0.29, 0.175, 'L2: DRAM / CXL\n0.5–4 TiB · 200 ns\n120–400 GB/s', 'teal'),
        (0.67, 0.675, 0.29, 0.175, 'L3: Gen5 NVMe\n30 TB · 15–80 μs\n100 GB/s (20M IOPS)', 'purple'),
        (0.08, 0.51, 0.40, 0.135, 'L4: 3FS 并行文件池\nPB 级 · 200 μs · 6.6 TiB/s', 'orange'),
        (0.52, 0.51, 0.40, 0.135, 'L5: 冷对象存储 S3\nEB 级 · 5–20 ms · 归档池', 'gray'),
    ]
    for x, y, w, h, label, role in tiers:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.33, 0.762), (0.355, 0.762))
    arrow(ax, (0.645, 0.762), (0.67, 0.762))
    arrow(ax, (0.48, 0.577), (0.52, 0.577))

    card(ax, 0.02, 0.03, 0.96, 0.43, 'AI 四类核心负载的 I/O 访问特征与首要系统瓶颈', color='blue')
    modes = [
        (0.04, 0.20, 0.44, 0.14, '① 预训练多模态流式读\nWebDataset 分片 vs 小文件', 'teal'),
        (0.52, 0.20, 0.44, 0.14, '② Checkpoint 突发聚合写\n万卡同步落盘与 GC 毛刺', 'orange'),
        (0.04, 0.05, 0.44, 0.12, '③ KV 缓存细粒度随机读写\n4–256 KiB 块与读放大', 'purple'),
        (0.52, 0.05, 0.44, 0.12, '④ Serverless 权重流式加载\n140 GB 级 GDS 零拷贝直通', 'blue'),
    ]
    for x, y, w, h, label, role in modes:
        box(ax, x, y, w, h, label, role, size=11)
    ex.save(fig, 'ch13-storage-hierarchy-and-access-patterns')


def fig_13_2_gds(ex: Exporter):
    fig, ax = canvas(3.55)
    card(ax, 0.02, 0.04, 0.46, 0.92, '传统 POSIX 页缓存（2.80 s）', color='orange')
    box(ax, 0.04, 0.68, 0.42, 0.15, '8×Gen5 NVMe SSD\n读出 140 GB 权重分片', 'teal', size=11)
    box(ax, 0.04, 0.46, 0.42, 0.15, 'Root Complex + Host DRAM\nPage Cache 双趟 280 GB', 'orange', size=11)
    box(ax, 0.04, 0.24, 0.42, 0.15, 'CPU 拷贝 + cudaMemcpy\n有效吞吐受限于 ~50 GB/s', 'orange', size=11)
    box(ax, 0.04, 0.07, 0.42, 0.11, 'GPU HBM（2.80 s 完成）', 'blue', size=11)
    arrow(ax, (0.25, 0.68), (0.25, 0.61))
    arrow(ax, (0.25, 0.46), (0.25, 0.39))
    arrow(ax, (0.25, 0.24), (0.25, 0.18))

    card(ax, 0.52, 0.04, 0.46, 0.92, 'GDS P2P DMA 零拷贝（1.40 s）', color='teal')
    box(ax, 0.54, 0.68, 0.42, 0.15, '8×Gen5 NVMe SSD\ncuFileRead + O_DIRECT', 'teal', size=11)
    box(ax, 0.54, 0.46, 0.42, 0.15, '同颗 PCIe Gen5 Switch\nP2P DMA 直通 GPU BAR1', 'purple', size=11)
    box(ax, 0.54, 0.24, 0.42, 0.15, '旁路 Host DRAM 与 CPU\nRoot Complex 载荷 ~0 GB', 'teal', size=11)
    box(ax, 0.54, 0.07, 0.42, 0.11, 'GPU HBM（100 GB/s · 1.40 s）', 'blue', size=11)
    arrow(ax, (0.75, 0.68), (0.75, 0.61))
    arrow(ax, (0.75, 0.46), (0.75, 0.39))
    arrow(ax, (0.75, 0.24), (0.75, 0.18))
    ex.save(fig, 'ch13-posix-pagecache-vs-gds-dma')


def fig_13_3_3fs_craq(ex: Exporter):
    fig, ax = canvas(3.60)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'CRAQ 链式复制协议：链式顺序写与全副本分摊强一致读', color='purple')
    box(ax, 0.04, 0.68, 0.21, 0.15, '3FS 客户端\nRDMA 直通', 'blue', size=11)
    box(ax, 0.29, 0.68, 0.20, 0.15, 'Head 副本 1\n写标记 Dirty', 'purple', size=11)
    box(ax, 0.53, 0.68, 0.20, 0.15, 'Mid 副本 2\n转发 Dirty', 'purple', size=11)
    box(ax, 0.77, 0.68, 0.19, 0.15, 'Tail 副本 3\n提交 Clean', 'teal', size=11)
    arrow(ax, (0.25, 0.755), (0.29, 0.755))
    arrow(ax, (0.49, 0.755), (0.53, 0.755))
    arrow(ax, (0.73, 0.755), (0.77, 0.755))
    box(ax, 0.04, 0.53, 0.92, 0.11, '读取路径：Head / Mid / Tail 任一副本直接响应 Clean 块（3× 读带宽）', 'teal', size=11)

    card(ax, 0.02, 0.03, 0.96, 0.43, '算例 13-2：180 节点 3FS 集群的磁盘—网卡带宽配比（6.6 TiB/s）', color='blue')
    box(ax, 0.04, 0.18, 0.28, 0.16, '16×Gen4 NVMe\n物理读 112 GB/s', 'teal', size=11)
    box(ax, 0.36, 0.18, 0.28, 0.16, '2×200G 网卡\n有效 36.67 GiB/s', 'blue', size=11)
    box(ax, 0.68, 0.18, 0.28, 0.16, '180 节点总读带宽\n聚合 6.6 TiB/s', 'purple', size=11)
    arrow(ax, (0.32, 0.26), (0.36, 0.26))
    arrow(ax, (0.64, 0.26), (0.68, 0.26))
    text(ax, 0.50, 0.09, '磁盘带宽相对网卡超配 2.24×，吸收 3 副本写入、随机 IOPS 与后台 GC', size=11, ha='center', color=COL['ink'])
    ex.save(fig, 'ch13-3fs-craq-architecture')


def fig_13_4_kv_layout(ex: Exporter):
    fig, ax = canvas(3.55)
    card(ax, 0.02, 0.04, 0.46, 0.92, 'Layer-major 布局（5.12 ms）', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '16-token 块跨 32 层分散\n每层 2 KiB（共 64 KiB）', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '触发 4 KiB LBA 最小读粒度\n读放大 RAF = 2.0×（页 8×）', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '128 块需 4,096 次随机读\n800k IOPS 下耗时 5.12 ms', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '有效载荷带宽仅 1.56 GiB/s', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'Block-major 布局（0.625 ms）', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '32 层同块 KV 连续打包\n单块合并为 64 KiB 连续段', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '对齐 4 KiB LBA 与 16 KiB 页\n读放大系数 RAF = 1.0×', 'teal', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '128 块仅 128 次大块顺序读\n12.5 GiB/s 下仅 0.625 ms', 'blue', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '端到端拉取提速 8.192×', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch13-kv-layout-read-amplification')


def fig_13_5_zns_gc(ex: Exporter):
    fig, ax = canvas(3.55)
    card(ax, 0.02, 0.04, 0.46, 0.92, '传统块 SSD：GC 写放大与毛刺', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '不同寿命 KV 与 Checkpoint\n混写于同一 NAND 擦除块', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '随机过期产生碎片页\n触发 FTL 读-改-写搬移', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '写放大 WAF = 2.5–4.0×\nP99 读延迟飙至 8–15 ms', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, 'DWPD 寿命缩水且干扰推理', size=11, ha='center', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'ZNS 分区 SSD：生命周期对齐', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '按会话 KV 或 Step 隔离\n独占 1–2 GiB 顺序追加区', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '同 Zone 数据同步过期失效\n直接 Zone Reset 整区擦除', 'teal', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '零有效页搬移：WAF ≈ 1.0×\n消除 GC 毛刺（P99 < 250 μs）', 'blue', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '闪存擦写寿命延长 2.5–4.0×', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch13-zns-gc-tail-latency')


def fig_13_6_multimodal(ex: Exporter):
    fig, ax = canvas(3.60)
    card(ax, 0.02, 0.52, 0.96, 0.44, '小文件元数据风暴 vs. WebDataset / Tar 连续分片流', color='purple')
    box(ax, 0.04, 0.58, 0.42, 0.22, '千万级离散图文/视频小文件\n每次 open/stat 触发 MDS 锁\n随机读吞吐跌至 <150 MB/s', 'orange', size=11)
    box(ax, 0.54, 0.58, 0.42, 0.22, 'WebDataset 1–2 GB 连续分片\n单次打开顺序流式预取\n元数据降 10,000× · >8 GB/s', 'teal', size=11)
    arrow(ax, (0.46, 0.69), (0.54, 0.69))

    card(ax, 0.02, 0.03, 0.96, 0.45, '算例 13-4：每秒 4,096 段 720p 视频（65,536 fps）解码卸载账本', color='blue')
    box(ax, 0.04, 0.15, 0.42, 0.21, '纯 CPU 软解码（256 fps/核）\n需 256 核 > 96 可用核\n供给率 37.5%，GPU 算力饥饿', 'orange', size=11)
    box(ax, 0.54, 0.15, 0.42, 0.21, 'CPU 32 核解封装 + 8×NVDEC\n硬件解码提供 76,800 fps\n零占 SM，训练 100% 满载', 'blue', size=11)
    arrow(ax, (0.46, 0.255), (0.54, 0.255))
    text(ax, 0.50, 0.08, 'NVDEC 与 Tensor Core 物理独立，解码张量直接驻留显存供 DiT 消费', size=11, ha='center', color=COL['ink'])
    ex.save(fig, 'ch13-multimodal-dataloader-pipeline')


def fig_13_7_parquet_lance(ex: Exporter):
    fig, ax = canvas(3.60)
    card(ax, 0.02, 0.04, 0.46, 0.92, 'WebDataset / Tar 行式流', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '图文/视频/标注按样本捆绑\n顺序流式读取吞吐 >8 GB/s', 'teal', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '缺乏行级偏移索引与列裁剪\n仅读文本列仍须拉取整包图像', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '课程学习按长度重排采样时\n随机跨分片读取退化严重', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '适合固定顺序全模态遍历', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'Parquet / Lance 列式随机寻址', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '按列独立存储 + 页级统计索引\n支持只读文本/元数据列投影', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, 'Lance 定长偏移表 O(1) 寻址\n单次 NVMe I/O 直达目标样本', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '支撑课程学习动态长度分桶\n与多阶段混合权重在线调整', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '兼顾顺序流与亚毫秒点查', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch13-parquet-lance-vs-webdataset-io')


def fig_13_8_dsec_erofs(ex: Exporter):
    fig, ax = canvas(3.60)
    card(ax, 0.02, 0.04, 0.46, 0.92, '全量拉取 + Tar 解包（>60 min）', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '8,192 容器并发拉取完整镜像\n实际仅访问 4.2%–13.3% 数据', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, 'Tar 逐层解压耗时 79 min\n每节点写盘 >1,600 GB（5.5×）', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '传统镜像层重建复杂度\nO(m·N) + O(k·N) 随规模爆炸', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '10 节点启动 >60 min', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'DSec EROFS 按需挂载（35 min）', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '只读 EROFS 块镜像按需读取\n合并大块 I/O 并驻留本地缓存', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '直接挂载耗时 45 min（1.76×）\n每节点写盘 700 GB（省 57%）', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, 'overlayfs 运行时动态组合层\n升级重建降至 O(m) + O(k)', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '8,192 容器 35 min 就绪', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch13-dsec-erofs-ondemand-container-boot')


def fig_13_9_faiss_iks(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.04, 0.46, 0.92, '主机链路搬移（805.3 ms）', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, 'N=2^25, d=768 向量库\nFP32 96 GiB · FP16 48 GiB', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '单查询内积 51.540 GFLOPs\nbatch=1 强度仅 1 FLOP/B', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '经 64 GB/s 主机链路搬移\n48 GiB 供数需 805.306 ms', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, 'B=16 合批强度升至 16 FLOP/B', size=11, ha='center', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, '近存 IKS 8 分片扫描（51.5 ms）', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '存储侧内部聚合带宽 1 TB/s\n本地扫描仅需 51.540 ms', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '主机向 8 分片广播查询向量\n8 × 1,536 B = 12,288 B', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '各分片返回 k=32 (score, id)\n8 × 32 × 12 B = 3,072 B', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '跨链路流量从 48 GiB 降至 15 KiB', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch13-faiss-flat-vs-iks-near-storage')


def fig_13_10_hnsw_diskann(ex: Exporter):
    fig, ax = canvas(3.60)
    card(ax, 0.02, 0.04, 0.46, 0.92, 'HNSW 全内存分层导航图', color='purple', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '原始 FP32 向量常驻 DRAM\n每向量额外存多层邻接表指针', 'purple', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '零层度数 2M（4B/边）\n图拓扑开销达数十至上百 GiB', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '跳表贪心搜索仅几十微秒\n十亿级向量触发 DRAM 容量墙', 'teal', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '高吞吐但内存成本高昂', size=11, ha='center', weight='bold', color=BORDER_COL['purple'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'DiskANN / Vamana 闪存对齐', color='blue', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, 'DRAM 仅驻留轻量 PQ 压缩码\n在内存中快速估算候选距离', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '单节点全精度向量 + 邻居 ID\n紧凑打包进同一 4 KiB 扇区', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, 'Beam Search 并行发起 SSD 读\n6–10 轮 I/O 完成十亿级检索', 'blue', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '消除跨扇区读放大（RAF=1.0×）', size=11, ha='center', weight='bold', color=BORDER_COL['blue'])
    ex.save(fig, 'ch13-hnsw-memory-vs-diskann-ssd-layout')


def fig_13_11_rag_calibration(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.04, 0.46, 0.92, '预注册校准选 k 陷阱（未通过）', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '8 题校准集（Calibration）\nFlat/ef16/ef64 在 k=1 达 8/8', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '按最小合格规则选定 k=1\nef4 仅 7/8（漏 q05）被淘汰', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '16 题留出集（Held-out）\nk=1 在 q14 失守（15/16）', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '严禁事后改 k=4 冒充预选成功', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, '6 个固定配置全通过（24/24）', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '288 请求共 281/288 通过\n7 次失败均输出 UNKNOWN', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, 'Flat/ef16/ef64 × k∈{4,16}\n校准 8/8 与留出 16/16 全过', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '留出集 E2E 中位数（k=4→16）\nef16：117.1 → 149.9 ms', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '检索召回余量决定生成质量下界', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch13-rag-288-requests-calibration-trap')


def fig_13_12_rag_kv_blocks(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.04, 0.46, 0.92, 'vLLM 原生 KV 块分配账本', color='blue', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, 'Worker 固定 KV 池：910 块\n共 2.147 GB（2.36 MB/块）', 'blue', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, 'k=4（输入中位 346.5 tok）\n峰值 23 块 = 54.26 MB', 'teal', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, 'k=16（输入中位 1201.5 tok）\n峰值 77 块 = 181.67 MB', 'purple', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, 'k 增 4× 使单请求 KV 块增 3.35×', size=11, ha='center', weight='bold', color=BORDER_COL['blue'])

    card(ax, 0.52, 0.04, 0.46, 0.92, '多进程资源守卫缺口审计', color='orange', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '原守卫按环境 token 过滤\n仅捕外层进程 RSS 2.23 GB', 'orange', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, 'EngineCore 子进程未继承\n致 raw GPU 采样误报为 0', 'orange', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, 'Worker 内部真实显存峰值\ncuda_peak = 18.583 GB', 'teal', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '须结合会话组与父子 PID 树监控', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])
    ex.save(fig, 'ch13-rag-topk-kv-blocks-and-rlimit-gap')


def fig_13_13_rag_timeline(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.50, 0.96, 0.46, '在线 RAG 五阶段串行时序分解（CPU 检索 → GPU vLLM）', color='purple')
    stages = [
        (0.04, 0.58, 0.17, 0.24, '① 编码\nMiniLM\n~2.1 ms', 'blue'),
        (0.23, 0.58, 0.17, 0.24, '② 检索\nHNSW 24μs\nFlat 278μs', 'teal'),
        (0.42, 0.58, 0.18, 0.24, '③ 拼装\nTokenize\n131–1227词', 'purple'),
        (0.62, 0.58, 0.16, 0.24, '④ Prefill\n23/77 块\n随 k 增长', 'orange'),
        (0.80, 0.58, 0.16, 0.24, '⑤ Decode\n两词口令\nstop 验收', 'teal'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.21, 0.70), (0.23, 0.70))
    arrow(ax, (0.40, 0.70), (0.42, 0.70))
    arrow(ax, (0.60, 0.70), (0.62, 0.70))
    arrow(ax, (0.78, 0.70), (0.80, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.43, '检索耗时节省 vs. 下游 Prefill 膨胀的协同权衡', color='blue')
    box(ax, 0.04, 0.15, 0.44, 0.20, '候选 A（精检 k=8，检索 200 ms）\n1,536 tok · KV 216 MiB\nPrefill 110.2 → 总 310.2 ms', 'teal', size=11)
    box(ax, 0.52, 0.15, 0.44, 0.20, '候选 B（粗检 k=32，检索 20 ms）\n4,608 tok · KV 648 MiB\nPrefill 351.4 → 总 371.4 ms', 'orange', size=11)
    text(ax, 0.50, 0.08, '200 TFLOP/s 下候选 A 快 61.2 ms；算力 >268.0 TFLOP/s 时翻转', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch13-rag-end-to-end-timeline-breakdown')


def fig_13_14_three_tier_kv(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.50, 0.96, 0.46, '跨节点四级 KV 缓存池化阶梯（Mooncake / HiCache）', color='blue')
    tiers = [
        (0.04, 0.58, 0.21, 0.24, 'T0: HBM\n3.35–8 TB/s\n热集驻留', 'blue'),
        (0.28, 0.58, 0.21, 0.24, 'T1: DRAM\n50–64 GB/s\n本机内存', 'teal'),
        (0.52, 0.58, 0.21, 0.24, 'T2: 远端池\n45–50 GB/s\nRDMA 池化', 'purple'),
        (0.76, 0.58, 0.20, 0.24, 'T3: 3FS\n12–62 GiB/s\nNVMe 盘池', 'orange'),
    ]
    for x, y, w, h, label, role in tiers:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.25, 0.70), (0.28, 0.70))
    arrow(ax, (0.49, 0.70), (0.52, 0.70))
    arrow(ax, (0.73, 0.70), (0.76, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.43, '拉取 KV 替代重算 Prefill 的盈亏判据与流水重叠', color='teal')
    box(ax, 0.04, 0.15, 0.44, 0.20, '带宽充裕判据（线性项对比）\nB_tier > B_crit = b_kv·F/(2P)\n拉取固定项快于重算投影', 'teal', size=11)
    box(ax, 0.52, 0.15, 0.44, 0.20, '低带宽临界长度 S_crit\nB_tier < B_crit 时需长上下文\n借二次注意力项反超 I/O', 'purple', size=11)
    text(ax, 0.50, 0.08, '配合第 l 层计算与第 l+1 层异步 DMA 预取重叠，掩盖 80%+ 传输延迟', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch13-three-tier-kv-cache-hierarchy')


def fig_13_15_cacheblend(ex: Exporter):
    fig, ax = canvas(3.60)
    card(ax, 0.02, 0.04, 0.46, 0.92, '传统前缀缓存 vs. 直接拼接', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '严格前缀匹配（Prefix Cache）\n文档顺序变动即致后续失效', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '独立预存文档块直接硬拼接\n虽修正 RoPE 相对位置旋转', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '但缺失跨文档交叉注意力\n多跳问答准确率严重退化', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '无法兼顾任意组合与多跳精度', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'CacheBlend 选择性稀疏重算', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '第 1 层全算定位高偏差词元\n筛出 Top 10%–15% 敏感词', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '85%–90% 词元直接复用 KV\n仅 10%–15% 词元增量重算', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '重算耗时掩盖 SSD 拉取延迟\n恢复全量 Prefill 多跳精度', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, 'TTFT 降 2.2–3.3× 且零精度损失', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch13-cacheblend-non-prefix-kv-fusion')


def fig_13_16_ssd_dwpd(ex: Exporter):
    fig, ax = canvas(3.60)
    card(ax, 0.02, 0.04, 0.46, 0.92, '无节流全量换出（33.75 DWPD）', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '8 卡节点配 4×3.84 TB SSD\n总闪存容量 C = 15.36 TB', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '每秒换出 4 GB 新生成 KV\n主机日写 345.6 TB/day', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '计入 WAF=1.5 写 518.4 TB\n达 33.75 DWPD（超标 11×）', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '不到半年即磨穿 TLC 闪存寿命', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, '准入过滤 + 分层治理（≤3 DWPD）', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, 'Count-Min Sketch 准入过滤\n仅放行命中 ≥2 次的高频块', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, 'DRAM 写缓冲吸收短命会话\n结合 ZNS 将 WAF 压回 1.0×', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '准入率 α ≤ 8.89%（WAF=1.5）\n或 α ≤ 13.33%（ZNS 1.0）', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '守住 3 DWPD 五年服役红线', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch13-ssd-dwpd-endurance-and-admission')


def fig_13_17_diagnosis(ex: Exporter):
    fig, ax = canvas(3.80)
    card(ax, 0.02, 0.03, 0.96, 0.94, 'AI 存储子系统瓶颈五步定量定位工作流', color='blue')
    steps = [
        (0.05, 0.72, 0.42, 0.14, '① 查元数据 IOPS 与打开率\n小文件风暴 → 改 Lance/Tar', 'blue'),
        (0.53, 0.72, 0.42, 0.14, '② 查 PCIe RC 与 CPU 拷贝\n双趟 2V_data → 改 GDS 直通', 'teal'),
        (0.53, 0.50, 0.42, 0.15, '③ 查请求粒度 vs 16KiB 页\nRAF > 1 → 改块连续打包', 'purple'),
        (0.05, 0.50, 0.42, 0.15, '④ 查 P99 毛刺与 DWPD 写放\nGC 阻塞 → ZNS + 准入过滤', 'orange'),
        (0.05, 0.26, 0.90, 0.16, '⑤ 查主机端计算协同（视频 CPU 解码饥饿 / 向量全扫带宽墙）\n卸载至 GPU 片上 NVDEC 硬件解码器或近存检索模块（IKS / DiskANN）', 'teal'),
    ]
    for x, y, w, h, label, role in steps:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.47, 0.79), (0.53, 0.79))
    arrow(ax, (0.74, 0.72), (0.74, 0.65))
    arrow(ax, (0.53, 0.575), (0.47, 0.575))
    arrow(ax, (0.26, 0.50), (0.26, 0.42))
    box(ax, 0.05, 0.08, 0.90, 0.12, '核心判据：让每次传输对齐闪存物理粒度，让数据零拷贝直达算力单元', 'blue', size=11)
    ex.save(fig, 'ch13-storage-diagnosis-workflow')


def fig_13_18_full_stack(ex: Exporter):
    fig, ax = canvas(3.80)
    card(ax, 0.02, 0.50, 0.96, 0.47, 'AI 原生数据底座上层引擎：训练流、检索与多级 KV 池', color='blue')
    top = [
        (0.04, 0.69, 0.44, 0.16, '预训练数据湖（Tar / Lance）\n分片顺序流 + NVDEC 硬解码', 'teal'),
        (0.52, 0.69, 0.44, 0.16, 'RAG 向量库（HNSW / DiskANN）\n近存扫描 IKS + 留出集门控', 'purple'),
        (0.04, 0.52, 0.44, 0.14, '多级 KV 池（Mooncake 池化）\n块连续打包 + CacheBlend', 'blue'),
        (0.52, 0.52, 0.44, 0.14, '万级沙箱镜像（DSec EROFS）\n按需块拉取 + overlayfs 层', 'orange'),
    ]
    for x, y, w, h, label, role in top:
        box(ax, x, y, w, h, label, role, size=11)

    card(ax, 0.02, 0.03, 0.96, 0.43, '底层传输协议栈与持久化介质：内核旁路 + 链式复制 + 闪存对齐', color='teal')
    bot = [
        (0.04, 0.18, 0.28, 0.17, '零拷贝传输通路\nGDS P2P DMA\nio_uring / USRBIO', 'blue'),
        (0.36, 0.18, 0.28, 0.17, 'DeepSeek 3FS\nCRAQ 全副本强一致\n磁盘/网卡 2.24× 超配', 'teal'),
        (0.68, 0.18, 0.28, 0.17, '闪存介质治理\nZNS 顺序追加区\n写准入守住 DWPD', 'purple'),
    ]
    for x, y, w, h, label, role in bot:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.265), (0.36, 0.265))
    arrow(ax, (0.64, 0.265), (0.68, 0.265))
    text(ax, 0.50, 0.08, '全栈协同：消除元数据风暴、Root Complex 绕行、读放大与 GC 尾延迟', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch13-full-stack-ai-data-plane')


def fig_13_19_falconfs(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.04, 0.46, 0.92, '传统 POSIX 逐跳解析（6 次 RPC）', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '客户端逐级 lookup(D=6)\n每打开 1 文件发 6–7 次 RPC', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '客户端 dentry 缓存命中低\n海量随机小文件冲垮目录锁', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '百万文件需 600 万次往返\n千卡 DataLoader 空转等待', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '小文件打开吞吐受限于 RTT', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'FalconFS 服务端复合路径解析', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '路径解析下推至 MDS 服务端\n精简缓存 + 单次复合请求', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '目录混合哈希 + 并发无锁\n消除热点父目录写锁争用', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '网络往返从 O(D) 压至 O(1)\n十亿小文件吞吐提升数倍', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, 'NSDI 26：化解小文件元数据墙', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch13-falconfs-server-side-path-resolution')


def fig_13_20_distvs_helmsman(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.04, 0.46, 0.92, 'DistVS 计算—内存解耦多级筛选', color='blue', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '第一级（计算节点 DRAM）\n低比特粗量化码过滤 90%', 'blue', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '第二级（CXL / RDMA 内存池）\n中等精度单边拉取精筛', 'teal', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '第三级（NVMe 全精度块）\n仅对 Top-M 发起异步点查', 'purple', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '注意：有损召回层级≠无损 KV', size=11, ha='center', weight='bold', color=BORDER_COL['blue'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'Helmsman 大规模聚类闪存 ANNS', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, 'OSDI 26：聚类重返闪存检索\n质心驻 DRAM，簇存 SSD', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '摒弃跨页跳读：同簇连续排布\n随机小读化为对齐顺序块读', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '严守 Recall@k 门槛前提下\n大幅削减单机 DRAM 容量税', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '十亿级 RAG 知识库高性价比底座', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch13-distvs-helmsman-disaggregated-vector-search')


def fig_13_21_copilotio(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.04, 0.46, 0.92, '纯 GPU 直通 I/O 的算力争用陷阱', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '大块裸张量（≥256 KiB）\nGDS P2P DMA 跑满 PCIe', 'teal', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '细粒度随机读 / 解压反序列化\nGPU 轮询队列耗尽寄存器', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, 'I/O 控制流侵占 SM 槽位\n导致主干 GEMM 算子降速', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, 'GPU 擅长矩阵算力而非中断轮询', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'CoPilotIO：CPU 协处理释放 SM', color='blue', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, 'OSDI 26：GPU 仅写轻量门铃\n立即切回执行高密矩阵计算', 'blue', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '空闲 CPU 核接管队列轮询\n完成小块合并与解压校验', 'teal', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '就绪后经锁页缓冲 DMA 入卡\n100% 归还 SM 给核心算子', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '按 I/O 粒度与解压需求动态选路', size=11, ha='center', weight='bold', color=BORDER_COL['blue'])
    ex.save(fig, 'ch13-copilotio-gds-vs-cpu-assisted-io')


def fig_13_22_umap_moonbright(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.04, 0.46, 0.92, 'Umap 分布式 mmap 矩阵访问', color='teal', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '传统 DFS 直接 mmap 大矩阵\n页缓存膨胀易触发容器 OOM', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, 'OSDI 26 Umap：感知矩阵步长\n批量预取 + 定向页缓存回收', 'teal', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '消除用户态双重缓冲拷贝\n严控容器 RSS 峰值免误杀', 'blue', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '主机虚拟内存与远端存储对齐', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'MoonBright 显存页表虚拟化', color='purple', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '传统 VMM 映射慢：CPU 改页表\n叠加全卡 TLB 刷空停顿', 'orange', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, 'OSDI 26 MoonBright：设备端\n物化页表 + 新地址免刷 TLB', 'purple', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '配合 kvcached 虚实地址解耦\n多模型弹性共享物理显存池', 'blue', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '贯通主机—设备统一虚拟页表底座', size=11, ha='center', weight='bold', color=BORDER_COL['purple'])
    ex.save(fig, 'ch13-umap-moonbright-mmap-and-gpu-page-table')


def fig_13_23_tar_vs_lance_seek(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'A. 静态顺序 Shuffle：WebDataset Tar 分片与 FalconFS 小文件对比', color='blue')
    top = [
        (0.04, 0.58, 0.28, 0.24, '离散小文件直读\n深目录 6 次 RPC\nFalconFS 压至 1 次', 'orange'),
        (0.36, 0.58, 0.28, 0.24, 'WebDataset Tar\n1–2 GB 连续打包\n顺序吞吐 >8 GB/s', 'teal'),
        (0.68, 0.58, 0.28, 0.24, '适用边界：\n预训练静态洗牌\n整包全模态遍历', 'blue'),
    ]
    for x, y, w, h, label, role in top:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.36, 0.70))
    arrow(ax, (0.64, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.43, 'B. 课程学习与在线难例重采样：Lance 列式随机点查 + Umap 按需映射', color='teal')
    bot = [
        (0.04, 0.16, 0.44, 0.20, 'Tar 分片按长度/奖励重采样时\n为取 1 条样本被迫读穿整块\n随机读放大高达数十倍', 'orange'),
        (0.52, 0.16, 0.44, 0.20, 'Lance O(1) 偏移表 + Umap 映射\n仅拉取命中样本与目标模态列\n单次 NVMe 点查 <200 μs', 'teal'),
    ]
    for x, y, w, h, label, role in bot:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.48, 0.26), (0.52, 0.26))
    text(ax, 0.50, 0.08, '数据湖演进：顺序吞吐靠连续分片，动态采样靠列式索引与页级映射', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch13-multimodal-tar-webdataset-vs-lance-random')


def fig_13_24_storage_cube(ex: Exporter):
    fig, ax = canvas(3.80)
    card(ax, 0.02, 0.03, 0.96, 0.94, 'AI 原生存储底座三维需求矩阵（带宽 · 延迟 · IOPS）与代表系统映射', color='blue')
    cells = [
        (0.04, 0.68, 0.44, 0.18, '① 训练检查点 & 预训练顺序流\n诉求：聚合带宽（6.6 TiB/s）\n方案：3FS CRAQ + GDS + Tar', 'blue'),
        (0.52, 0.68, 0.44, 0.18, '② 十亿小文件 & 万级容器启动\n诉求：元数据 OPS 与去重\n方案：FalconFS + DSec EROFS', 'teal'),
        (0.04, 0.44, 0.44, 0.18, '③ 多级 KV 缓存卸载与热恢复\n诉求：亚毫秒 P99 与写寿命\n方案：64KiB 块 + ZNS + 写准入', 'purple'),
        (0.52, 0.44, 0.44, 0.18, '④ 十亿级向量 ANNS（RAG 库）\n诉求：高并发点查 + 召回率\n方案：DistVS / Helmsman / IKS', 'orange'),
        (0.04, 0.20, 0.92, 0.18, '⑤ 细粒度压缩 I/O 协同与统一虚拟内存页表（计算—存储边界消融）\n方案：CoPilotIO 协处理 + Umap 页映射 + MoonBright / kvcached', 'teal'),
    ]
    for x, y, w, h, label, role in cells:
        box(ax, x, y, w, h, label, role, size=11)
    box(ax, 0.04, 0.06, 0.92, 0.10, '第 13 章法则：脱离访问粒度、擦写物理与质量门控的存储必遇瓶颈', 'blue', size=11)
    ex.save(fig, 'ch13-ai-storage-bandwidth-latency-iops-cube')


def fig_13_25_nixl(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, 'A. NIXL 统一异步传输控制面', color='blue', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '8.0 GB KV 跨介质注册\nCXL · RDMA · NVMe P2P', 'blue', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '非阻塞复合描述符 (0.05 ms)\n散聚块合并 + 三通道并发', 'teal', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '拓扑感知自动选路\nNVLink > P2P GDS > RoCEv2', 'purple', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '消除跨层 API 割裂与同步等待', size=11, ha='center', weight='bold', color=BORDER_COL['blue'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'B. 算例 13-13：三通道并发预取', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '传统分立 API 串行拉取\n三层累计耗时 225.10 ms', 'orange', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, 'NIXL 并发拉取 80.20 ms\n提速 2.81× (99.75 GB/s)', 'teal', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '与前序块计算重叠 75% 后\n净暴露停顿仅 20.05 ms', 'blue', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '支撑 PD 分离与多级 KV 换入', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch13-nixl-unified-nonblocking-transfer-plane')


def fig_13_26_checkpoint_ec(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, 'A. 热层 3 副本秒级暂存', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '万卡单次快照裸量 10.0 TB\n100 个里程碑共 1.0 PB', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '最近 N_hot=3 个热快照\n3 副本 (3.0×) 占 90.0 TB', 'teal', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '若 100 个全保留 3 副本：\n需消耗 3,000.0 TB 闪存', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '热层：零编码开销、秒级回滚', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'B. 温冷层 RS(10,4) 纠删归档', color='blue', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '100 节点后台编码 66.7 MB/s\n10 数据 + 4 校验块 (1.4×)', 'blue', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '其余 97 个归档 1,358.0 TB\n容忍任意 4 节点同时故障', 'teal', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '算例 13-14：总占 1,448 TB\n净省 1,552 TB (省 51.73%)', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '温冷：容量减半、高可靠归档', size=11, ha='center', weight='bold', color=BORDER_COL['blue'])
    ex.save(fig, 'ch13-checkpoint-replication-to-erasure-coding-lifecycle')


def fig_13_27_lmcache_prefix_kv(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, 'A. 锚点差分压缩与 16 块流水', color='blue', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '64K 前缀原始 KV 8.0 GiB\n未压缩串行耗 713.60 ms', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '锚点差分压缩 3.80×\n降至 2,155.79 MiB 落盘', 'teal', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '16 块 GDS+解压流水重叠\nTTFT 273.02 ms (2.61×)', 'blue', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '逐层组 I/O 掩盖前向计算', size=11, ha='center', weight='bold', color=BORDER_COL['blue'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'B. 100 TiB 三级冷热对象池', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, 'L2 DRAM 热池 (占 5%)\n命中 45% · 42.0 ms', 'blue', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, 'L3 NVMe 温池 (占 25%)\n命中 40% · 273.02 ms', 'teal', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, 'L5 S3 冷池 (70% 命中 15%)\n均 230.11 ms · 省 87.98%', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '比 1,850 ms 重算提速 8.04×', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch13-lmcache-prefix-kv-compression-and-gds-prefetch')


def fig_13_28_nvme_pool_qos(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, 'A. 单 FIFO 队列队头阻塞', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '检查点 1 MiB 突发顺序写\n单次编程占用 0.80 ms', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '前方 16 个写请求排队\n队头阻塞达 12.80 ms', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '叠加 1.00 ms KV 突发读\n读 P99 恶化至 13.80 ms', 'purple', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '大块写阻塞在线推理 TTFT', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'B. 硬件 WRR + 128 KiB 切片', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, 'NVMe WRR 队列权重 8:2\nKV 读锁定 8.0 GiB/s', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '1 MiB 写切为 128 KiB 块\n单块非抢占仅 0.10 ms', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, 'KV 读 P99 降至 1.35 ms\n尾延迟削减达 10.22×', 'teal', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '兼顾高吞吐落盘与毫秒级读', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch13-checkpoint-and-kv-unified-nvme-pool-qos')




def draw_dual_band(
    ex: Exporter,
    stem: str,
    top_title: str,
    top_color: str,
    top_boxes: list[tuple[str, str]],
    bot_title: str,
    bot_color: str,
    bot_boxes: list[tuple[str, str]],
    footer: str,
) -> None:
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, top_title, color=top_color)
    if len(top_boxes) == 3:
        xs = [(0.028, 0.296), (0.352, 0.296), (0.676, 0.296)]
        for (x, w), (lbl, role) in zip(xs, top_boxes):
            box(ax, x, 0.58, w, 0.24, lbl, role, size=11)
        arrow(ax, (0.324, 0.70), (0.352, 0.70))
        arrow(ax, (0.648, 0.70), (0.676, 0.70))
    else:
        xs = [(0.03, 0.45), (0.52, 0.45)]
        for (x, w), (lbl, role) in zip(xs, top_boxes):
            box(ax, x, 0.58, w, 0.24, lbl, role, size=11)
        arrow(ax, (0.48, 0.70), (0.52, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, bot_title, color=bot_color)
    if len(bot_boxes) == 3:
        xs = [(0.028, 0.296), (0.352, 0.296), (0.676, 0.296)]
        for (x, w), (lbl, role) in zip(xs, bot_boxes):
            box(ax, x, 0.13, w, 0.20, lbl, role, size=11)
        arrow(ax, (0.324, 0.23), (0.352, 0.23))
        arrow(ax, (0.648, 0.23), (0.676, 0.23))
    else:
        xs = [(0.03, 0.45), (0.52, 0.45)]
        for (x, w), (lbl, role) in zip(xs, bot_boxes):
            box(ax, x, 0.13, w, 0.20, lbl, role, size=11)
        arrow(ax, (0.48, 0.23), (0.52, 0.23))

    text(ax, 0.50, 0.06, footer, size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, stem)




def main():
    ex = Exporter(HERE)
    fig_13_1_hierarchy(ex)
    fig_13_2_gds(ex)
    fig_13_3_3fs_craq(ex)
    fig_13_4_kv_layout(ex)
    fig_13_5_zns_gc(ex)
    fig_13_6_multimodal(ex)
    fig_13_7_parquet_lance(ex)
    fig_13_8_dsec_erofs(ex)
    fig_13_9_faiss_iks(ex)
    fig_13_10_hnsw_diskann(ex)
    fig_13_11_rag_calibration(ex)
    fig_13_12_rag_kv_blocks(ex)
    fig_13_13_rag_timeline(ex)
    fig_13_14_three_tier_kv(ex)
    fig_13_15_cacheblend(ex)
    fig_13_16_ssd_dwpd(ex)
    fig_13_17_diagnosis(ex)
    fig_13_18_full_stack(ex)
    fig_13_19_falconfs(ex)
    fig_13_20_distvs_helmsman(ex)
    fig_13_21_copilotio(ex)
    fig_13_22_umap_moonbright(ex)
    fig_13_23_tar_vs_lance_seek(ex)
    fig_13_24_storage_cube(ex)
    fig_13_25_nixl(ex)
    fig_13_26_checkpoint_ec(ex)
    fig_13_27_lmcache_prefix_kv(ex)
    fig_13_28_nvme_pool_qos(ex)

    draw_dual_band(
        ex,
        'ch13-100pb-commoncrawl-to-15t-tokens-curation-funnel',
        '100 PB CommonCrawl 原始网页七级提纯漏斗（前四级过滤）',
        'blue',
        [
            ('① 原始归档 100 PB\n正文抽取留 18 PB\n剥离标签与广告脚本', 'blue'),
            ('② 语言与启发式过滤\n18 PB -> 9.0 PB\nGopher/FineWeb 规则', 'purple'),
            ('③ 全局模糊+精确去重\n9.0 PB -> 3.0 PB\n剔除 66.7% 重复冗余', 'orange'),
        ],
        '模型质量分级与去污收束（算例 13-15：产出 15.0T 精品词元）',
        'teal',
        [
            ('④ 质量分级与合成改写：\n3.0 PB -> 2.424 PB\nNemotron-CC 抢救高价值页', 'blue'),
            ('⑤ 评测基准 13-gram 去污：\n最终留存 2.40 PB（2.40%）\n25 个快照合 15.0T 词元', 'teal'),
        ],
        '从 100 PB 原始爬虫到 2.40 PB 精品语料，去重与分级决定上限',
    )

    draw_dual_band(
        ex,
        'ch13-minhash-lsh-band-row-s-curve-probability',
        'MinHash LSH 局部敏感哈希分桶参数（M = b x r = 128）',
        'purple',
        [
            ('5-gram 词片集合\n128 个独立哈希函数\n签名仅 512 B/doc', 'blue'),
            ('划分为 b=16 个分桶\n每桶含 r=8 行签名\n桶内全等即触发候选', 'purple'),
            ('拐点阈值 s*=0.7071\n公式 (1/b)^(1/r)\n陡峭分离相似与无关', 'teal'),
        ],
        'S 型碰撞概率曲线 P(s) = 1 - (1 - s^r)^b 数值特征（算例 13-16）',
        'teal',
        [
            ('无关文档（s = 0.50）：\n误碰率仅 6.07%\n避免平方级无效比对', 'orange'),
            ('近似重复文档（s = 0.80）：\n捕获率高达 94.70%\ns=0.90 时捕获率 99.99%', 'teal'),
        ],
        '调节 (b, r) 行列配比，在计算开销与召回率间精确塑形',
    )

    draw_dual_band(
        ex,
        'ch13-distributed-minhash-shuffle-and-gpu-union-find',
        '50 亿篇文档（2.56 TB 签名）分布式 GPU 模糊去重流水线',
        'blue',
        [
            ('① GPU 哈希签名\n50 亿文档 x 512 B\n总签名体积 2.56 TB', 'blue'),
            ('② 按分桶全互换洗牌\n64 卡洗牌 2.52 TB\n同桶文档发往同卡', 'purple'),
            ('③ cuGraph 连通分量\n大星/小星指针跳跃\n毫秒级收敛聚簇代表', 'teal'),
        ],
        '分布式图连通分量（Union-Find）消除传递性重复簇（算例 13-17）',
        'teal',
        [
            ('传递性闭包聚类（A~B, B~C）：\n大星/小星指针跳跃迭代\n将百万级重复网页缩为单簇', 'blue'),
            ('每簇仅留最高质量代表文档：\n剔除 66.7% 跨站转载与镜像\n避免预训练双下降与背诵退化', 'teal'),
        ],
        '将文本去重转化为分布式图连通分量求解，是百亿文档去重标准解法',
    )

    draw_dual_band(
        ex,
        'ch13-cpu-datatrove-vs-gpu-nemo-curator-throughput',
        'CPU 集群（DataTrove）vs. GPU 流水线（NeMo Curator）对比',
        'orange',
        [
            ('4,096 核 CPU 集群：\n哈希与磁盘洗牌受限\n耗时 148.0 小时', 'orange'),
            ('64 张 H100 GPU 集群：\ncuDF + NVLink 洗牌\n仅需 1.85 小时（80x）', 'teal'),
        ],
        '百 TB 语料去重端到端耗时与云租用总成本（TCO，算例 13-17）',
        'teal',
        [
            ('算力吞吐跃迁（快 80.0 倍）：\n从 6.17 天缩至 111 分钟\n支撑预训练配方日级快速迭代', 'blue'),
            ('云端租用总成本压降 4.50 倍：\nCPU 需 42,600 美元\nGPU 仅需 9,472 美元', 'teal'),
        ],
        'GPU 高带宽显存与高速互连洗牌在数据去重中具备压倒性优势',
    )

    draw_dual_band(
        ex,
        'ch13-document-fuzzy-dedup-vs-suffix-array-substring-dedup',
        '文档级 MinHash 模糊去重与段落级后缀数组精确去重协同',
        'blue',
        [
            ('第一层：MinHash\n粒度：整篇文档级\n剔除改写与转载文章', 'blue'),
            ('第二层：后缀数组\n粒度：50 词元子串\n线性构建全局索引', 'purple'),
            ('原位剥离模板噪声\n切除页眉导航菜单\n与免责声明重复尾', 'teal'),
        ],
        '双层混合去重对模型困惑度（PPL）与记忆化泄露的联合抑制',
        'teal',
        [
            ('防止长文被整篇误杀：\n正文独特但含公用页脚的网页\n经后缀数组剥离页脚得以保留', 'blue'),
            ('消除隐私与版权原样背诵：\n连续 50 词元重复率 < 0.1%\n显著改善上下文学习泛化能力', 'teal'),
        ],
        '文档级模糊去重抓宏观冗余，后缀数组子串去重精准剔除微观模板',
    )

    draw_dual_band(
        ex,
        'ch13-llm-rubric-distillation-to-fasttext-bert-classifier',
        '大模型质量打分蒸馏（70B 教师 -> 110M 轻量分类器）三级杠杆',
        'purple',
        [
            ('① 70B 教师精标种子\n标注 500 万篇文档\n仅耗 217.0 卡时', 'blue'),
            ('② 蒸馏 110M 分类器\n学习教育与推理评分\n训练耗 80.0 卡时', 'purple'),
            ('③ 全量 500 亿篇打分\n吞吐 180k tok/s/卡\n耗 38,580 卡时', 'teal'),
        ],
        '直接 70B 全量打分 vs. 蒸馏轻量分类器算力账本对比（算例 13-18）',
        'teal',
        [
            ('直接用 70B 打分 500 亿篇：\n需 2,170,138.9 GPU 小时\n折合成本高达 868 万美元', 'orange'),
            ('三级蒸馏管线总计：\n仅需 38,877.2 GPU 小时\n算力节省 55.82 倍（省 98.2%）', 'teal'),
        ],
        '用教师模型定义品味、轻量模型扫平百亿语料，是数据分级的黄金法则',
    )

    draw_dual_band(
        ex,
        'ch13-nemotron-cc-classifier-ensembling-and-rephrasing',
        'Nemotron-CC 多分类器集成打分与合成改写抢救（Rephrasing）',
        'blue',
        [
            ('多分类器互补集成\n教育打分+指令打分\n划分 20 个质量桶', 'blue'),
            ('高质桶受控上采样\nTop-5% 精品训 3 轮\n低质桶尽早丢弃', 'purple'),
            ('高价值残缺页抢救\n含稀缺知识但排版碎\n送入大模型合成改写', 'teal'),
        ],
        '突破“激进过滤导致知识枯竭”的合成改写机制',
        'teal',
        [
            ('为什么不能盲目丢弃中低分网页：\n许多排障贴格式杂乱但含稀缺经验\n直接过滤会损失长尾专业知识', 'orange'),
            ('百科风格合成改写（Rephrase）：\n将破碎问答重写为结构化教材段落\n使有效精品词元总量扩充 1.8 倍', 'teal'),
        ],
        '分类器集成实现精细分层，合成改写将粗糙矿石冶炼成精品语料',
    )

    draw_dual_band(
        ex,
        'ch13-benchmark-decontamination-ngram-and-embedding-hybrid',
        '评测基准防污染（Benchmark Decontamination）双层净化管线',
        'orange',
        [
            ('构建全基准指纹库\nMMLU/GSM8K/代码库\n提取全部测试题文本', 'blue'),
            ('第一道防线：13-gram\n多模式串自动机匹配\n拦截字面精确重合', 'orange'),
            ('第二道防线：语义向量\n稠密余弦 > 0.85\n拦截同义改写与翻译', 'purple'),
        ],
        '防止预训练语料刷榜失真（Benchmark Leakage）的双层防线',
        'teal',
        [
            ('为什么单靠 13-gram 远远不够：\n网页常含翻译成中文的英文测试题\n或变量名被替换的算法竞赛原题', 'orange'),
            ('双层混合去污代价与收益：\n仅剔除 0.99%（2.424->2.40 PB）\n确保评测分数反映真实泛化能力', 'teal'),
        ],
        '字面 n-gram 结合稠密向量检索，是守卫基准评测科学性的必由之路',
    )

    draw_dual_band(
        ex,
        'ch13-doremi-regmix-domain-weight-optimization',
        '基于代理模型的领域配比寻优（DoReMi / RegMix）与三阶段退火',
        'blue',
        [
            ('① 小代理模型实验\n用 280M/1B 跑网格\n拟合领域权重曲面', 'blue'),
            ('② 求解最优配比\n提升代码与理科权重\n压降泛娱乐网页比例', 'purple'),
            ('③ 迁移至千亿主干\n零超参试错直接命中\n验证集损失帕累托', 'teal'),
        ],
        '现代预训练三阶段动态数据退火（Curriculum Annealing）',
        'teal',
        [
            ('阶段 1–2（前 85% 训练词元）：\n万亿级多样性网页建立世界常识\n中段逐步拉升代码与长文配比', 'blue'),
            ('阶段 3（末段 15% 极速退火期）：\n注入 70% 合成数学与精校百科\n伴随学习率余弦归零实现跃升', 'teal'),
        ],
        '随训练阶段动态调整数据配比，能以相同算力榨出显著更高模型性能',
    )

    draw_dual_band(
        ex,
        'ch13-online-serving-vs-offline-synthetic-factory-roofline',
        '在线低延迟推理服务 vs. 离线合成数据工厂的 Roofline 跃迁',
        'purple',
        [
            ('在线实时服务（B=32）：\n受制于 TPOT < 25 ms\n卡在 HBM 访存带宽墙\nMFU 仅 11.2%（1,920 tok/s）', 'orange'),
            ('离线合成工厂（B=2,048）：\n无交互延迟 SLO 约束\n推入 Tensor Core 计算饱和区\nMFU 达 58.4%（13,824 tok/s）', 'teal'),
        ],
        '卸下实时延迟枷锁后的单卡吞吐与单位词元成本剧变（算例 13-19）',
        'teal',
        [
            ('单卡生成吞吐暴增 7.20 倍：\n结合前缀共享与分页 KV 卸载\n每次权重读取被数千序列摊薄', 'blue'),
            ('原始词元成本下降 13.16 倍：\n从在线 0.579 美元/百万词元\n降至离线 0.044 美元/百万词元', 'teal'),
        ],
        '离线合成数据工厂用延迟容忍度换取极致算术强度与吞吐/美元',
    )

    draw_dual_band(
        ex,
        'ch13-synthetic-data-generator-verifier-rejection-funnel',
        '合成数据工厂“生成—沙箱验证—重写抢救”校验漏斗（算例 13-19）',
        'blue',
        [
            ('① 多样化高温采样\n每题并发采 N=16 条\n原始 0.044 美元/M', 'blue'),
            ('② 沙箱编译与单测\n首轮通过率 31.25%\n拦截幻觉与语法错', 'purple'),
            ('③ 错误轨迹自纠重写\n挽回 40% 失败样本\n良率升至 58.75%', 'teal'),
        ],
        '可验证奖励过滤（RLVR / PRM）杜绝“模型自噬（Model Collapse）”',
        'teal',
        [
            ('为什么未经验证的合成数据有害：\n同分布无过滤自蒸馏放大高频偏见\n导致尾部知识丢失与方差坍缩', 'orange'),
            ('外部真值检验器注入新信息熵：\n编译器、定理证明器与 PRM 把关\n验证词元成本仅 0.075 美元/M', 'teal'),
        ],
        '生成器提供广度，沙箱验证器提供真值，自纠重写最大化算力良率',
    )

    draw_dual_band(
        ex,
        'ch13-slack-capacity-harvesting-for-synthetic-generation',
        '利用集群昼夜波谷与训练维修空窗收割零边际成本合成算力',
        'teal',
        [
            ('推理集群昼夜潮汐\n凌晨 00–08 点低谷\n闲置 40%–65% 算力', 'blue'),
            ('抢占式合成作业注入\n秒级拉起大批次生成\n填满 100% 功耗基线', 'purple'),
            ('百毫秒级优雅让路\n早高峰流量回升时\n按块落盘并立即腾退', 'teal'),
        ],
        '离线合成数据工厂作为数据中心“算力海绵（Compute Sponge）”',
        'blue',
        [
            ('吸收训练集群重调度碎片：\n万卡训练故障缩容或存检查点时\n空闲节点瞬间切入合成生成队列', 'blue'),
            ('等效 GPU 小时单价再降 45%：\n盘活沉没折旧卡时（2.20 美元/h）\n年产数万亿精品推理与代码词元', 'teal'),
        ],
        '离线合成任务具备无状态可抢占性，是削峰填谷提升机队利用率的利器',
    )

    draw_dual_band(
        ex,
        'ch13-pdf-scientific-ocr-layout-math-extraction-pipeline',
        '科学文献与教材 PDF 多模态版面解析及公式还原流水线',
        'blue',
        [
            ('① 版面检测与阅读序\n精准识别双栏、表格\n与跨页插图拓扑', 'blue'),
            ('② 公式与表格还原\n视觉语言小模型转写\nLaTeX 与 Markdown', 'purple'),
            ('③ 乱码与符号清洗\n修复字体映射缺失\n构建图文交错语料', 'teal'),
        ],
        '攻克高阶 STEM 知识壁垒（Nougat / MinerU / olmOCR 范式）',
        'teal',
        [
            ('传统纯文本提取器的灾难：\n分栏错乱、上下标丢失、积分乱码\n使模型学到支离破碎的错误公式', 'orange'),
            ('端到端视觉文档解析吞吐优化：\nTensorRT 加速轻量 VLM 光栅扫描\n单卡每日解析超 50 万页科技文献', 'teal'),
        ],
        '高精度 PDF 版面与公式还原是解锁人类高阶 STEM 知识宝库的钥匙',
    )

    draw_dual_band(
        ex,
        'ch13-tokenized-dataset-mmap-shuffling-and-sequence-packing',
        '万亿词元二进制索引（mmap）全局伪随机洗牌与最佳适应打包',
        'purple',
        [
            ('零拷贝内存映射读取\n.bin 流 + .idx 偏移\nOS 页缓存按需预取', 'blue'),
            ('O(1) 显存全局洗牌\n块级 Feistel 置换\n免物化百亿索引表', 'purple'),
            ('最佳适应序列打包\n线段树匹配剩余空间\n消除文档截断与填充', 'teal'),
        ],
        '朴素拼接截断（Concat-Chunk）vs. 最佳适应无截断打包（OBFD）',
        'teal',
        [
            ('朴素拼接截断的严重隐患：\n15%–25% 长代码与证明被拦腰截断\n破坏跨函数调用与长链因果完整', 'orange'),
            ('最佳适应无截断打包（OBFD）：\n填充率 < 0.5% 且保持文档完整边界\n配合变长注意力杜绝跨文档串扰', 'teal'),
        ],
        '无截断最佳适应打包兼顾了 99.5% 算力填充效率与长程因果语义完整性',
    )

    draw_dual_band(
        ex,
        'ch13-data-lineage-copyright-optout-and-audit-ledger',
        '全生命周期语料血缘账本（Data Provenance）与合规退订回溯',
        'blue',
        [
            ('抓取与过滤数字签名\n记录来源 URL、协议\n与许可证 SPDX 标签', 'blue'),
            ('块级 Merkle 哈希树\n绑定每个训练批次\n至原始文档 SHA256', 'purple'),
            ('版权退订与遗忘接口\n毫秒级定位目标分片\n支持增量退火擦除', 'teal'),
        ],
        '面向主权 AI 与企业合规审计的可验证数据治理底座',
        'teal',
        [
            ('亚秒级反查任意训练步语料来源：\n当模型出现异常输出或合规质询时\n经倒排哈希索引瞬间追溯源头文档', 'blue'),
            ('零停机动态黑名单原地屏蔽：\n在 .idx 索引层原地标记屏蔽位\n免重写数 PB 底层 .bin 数据文件', 'teal'),
        ],
        '可追溯的密码学数据血缘账本，是万亿词元大模型跨越合规红线的护城河',
    )

    draw_dual_band(
        ex,
        'ch13-exascale-data-and-synthetic-factory-master-architecture',
        '百亿亿次 Token 数据工程与合成数据工厂全栈架构总图',
        'teal',
        [
            ('支柱一：七级提纯漏斗\n100 PB -> 15T 词元\n2.40% 严苛留存率', 'blue'),
            ('支柱二：GPU 模糊去重\nMinHash + cuGraph\n80x 提速与 4.5x 降本', 'purple'),
            ('支柱三：合成数据工厂\nB=2048 达 58.4% MFU\n沙箱验证 0.075 美元/M', 'teal'),
        ],
        '从“清洗人类存量网页”到“算力冶炼合成智慧”的范式交接',
        'blue',
        [
            ('人类自然文本存量逼近枯竭：\nGPU 去重与分类蒸馏（55.8x 杠杆）\n榨干存量互联网每一滴高价值信号', 'blue'),
            ('合成数据开启第二增长曲线：\n将空闲算力转化为经沙箱验证的\n数学、代码与智能体轨迹驱动进化', 'teal'),
        ],
        '数据工程已演进为与预训练集群同等规模的核心 AI 基础设施',
    )

    draw_dual_band(
        ex,
        'ch13-mineru-olmocr-pdf-layout-and-formula-gpu-pipeline',
        '百万册学术 PDF 解析：数字页直抽与公式框动态批处理',
        'blue',
        [
            ('① CMap 字符探针\n65% 原生数字页\n正文零拷贝直抽', 'blue'),
            ('② 跨页子框微批\n公式/表格 B=256\n均摊仅 11.00 ms', 'purple'),
            ('③ 扫描页 olmOCR\n35% 乱码扫描页\n连续批处理 30.2 ms', 'teal'),
        ],
        '200 万册（5 亿页）文献库：加权 17.72 ms/页 vs 单体 VLM 625 ms/页',
        'teal',
        [
            ('单体 7B VLM 全页解码瓶颈：\n1.60 页/s · 86,805.6 GPU-h\n已有数字层重复自回归浪费算力', 'orange'),
            ('MinerU + olmOCR 混合双通路：\n56.43 页/s（提速 35.27 倍）\n2,461.1 GPU-h · 0.0135 USD/M', 'teal'),
        ],
        '区分原生数字页与扫描页并对公式子框跨页批处理，将吞吐推高 35.27 倍',
    )

    draw_dual_band(
        ex,
        'ch13-repo-level-ast-dependency-topological-packing',
        '代码仓库 64k 预训练：AST 依赖提取、Tarjan 破环与拓扑装箱',
        'purple',
        [
            ('① Tree-sitter\n解析 import/符号\n建跨文件有向图', 'blue'),
            ('② Tarjan 强连通\n凝聚循环导入环\n化简为严格 DAG', 'purple'),
            ('③ Kahn 拓扑装箱\n被调方排调用方前\n64k 装箱率 98.8%', 'teal'),
        ],
        '因果掩码下跨文件定义可见率：字母序 42.50% -> 拓扑序 95.00%',
        'teal',
        [
            ('路径字母序拼接（42.50%）：\napp/ 排在 core/ 与 utils/ 之前\n57.5% 跨文件调用沦为无定义盲猜', 'orange'),
            ('AST 拓扑序装箱（95.00%）：\n基础库与工具定义驻留左侧前缀\n跨文件 API 困惑度降 39.85%', 'teal'),
        ],
        '按 AST 依赖拓扑排序让因果自注意力自然贴合软件工程的真实调用流向',
    )

    outputs = ex.finish()

    fig_data = build_figure_data()
    (HERE / 'figure-data.json').write_text(json.dumps(fig_data, ensure_ascii=False, indent=2) + '\n')

    fig_index = [
        {
                "figure_id": "图 13-1",
                "number": 1,
                "file": "ch13-storage-hierarchy-and-access-patterns.svg",
                "asset": "ch13/ch13-storage-hierarchy-and-access-patterns.svg",
                "section": "13.1.1",
                "caption": "AI 存储介质金字塔（L1 HBM 至 L5 对象存储）的延迟、IOPS 与带宽量级，以及四类核心 AI 负载的 I/O 访问瓶颈",
                "source_refs": [
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "建立全章五级存储金字塔与四类 AI I/O 访问特征的定量坐标",
                "name": "ch13-storage-hierarchy-and-access-patterns",
                "figure": "13-1"
        },
        {
                "figure_id": "图 13-2",
                "number": 2,
                "file": "ch13-posix-pagecache-vs-gds-dma.svg",
                "asset": "ch13/ch13-posix-pagecache-vs-gds-dma.svg",
                "section": "13.1.2",
                "caption": "算例 13-1：单机 8 盘 PCIe Gen5 NVMe SSD 加载 140 GB 权重时，传统 POSIX 页缓存双重拷贝（280 GB Root Complex 流量，耗时 2.80 s）与 GPUDirect Storage P2P DMA 零拷贝直通（100 GB/s，耗时 1.40 s）对比",
                "source_refs": [
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "直观揭示 Page Cache 在 PCIe Root Complex 上的流量翻倍效应与 GDS P2P DMA 的旁路机理",
                "name": "ch13-posix-pagecache-vs-gds-dma",
                "figure": "13-2"
        },
        {
                "figure_id": "图 13-3",
                "number": 3,
                "file": "ch13-3fs-craq-architecture.svg",
                "asset": "ch13/ch13-3fs-craq-architecture.svg",
                "section": "13.2.1",
                "caption": "DeepSeek 3FS 的 CRAQ 链式复制协议（链式顺序写与全副本强一致分摊读）及算例 13-2 中 180 节点存储集群实现 6.6 TiB/s 聚合读带宽的网卡—磁盘配比账本",
                "source_refs": [
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "展示 CRAQ 如何将强一致读吞吐线性扩展至全部副本，以及磁盘超配 2.24 倍如何掩盖写放大与 GC",
                "name": "ch13-3fs-craq-architecture",
                "figure": "13-3"
        },
        {
                "figure_id": "图 13-4",
                "number": 4,
                "file": "ch13-kv-layout-read-amplification.svg",
                "asset": "ch13/ch13-kv-layout-read-amplification.svg",
                "section": "13.2.2",
                "caption": "算例 13-3：32 层模型卸载 16-token KV 块时，Layer-major 按层分离布局（触发 4 KiB LBA 读放大 RAF=2.0x 与 800k IOPS 墙，耗时 5.12 ms）与 Block-major 全层打包布局（RAF=1.0x，跑满 12.5 GiB/s，耗时 0.625 ms，提速 8.192x）对比",
                "source_refs": [
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "阐明 NAND Flash 最小读粒度与 IOPS 墙如何决定 KV Cache 的跨层连续打包布局",
                "name": "ch13-kv-layout-read-amplification",
                "figure": "13-4"
        },
        {
                "figure_id": "图 13-5",
                "number": 5,
                "file": "ch13-zns-gc-tail-latency.svg",
                "asset": "ch13/ch13-zns-gc-tail-latency.svg",
                "section": "13.2.2",
                "caption": "传统块接口 NVMe SSD 在混写负载下的 FTL 垃圾回收（GC）写放大（WAF=2.5–4.0x）与读尾延迟毛刺，以及 ZNS 分区命名空间 SSD 按会话/检查点生命周期对齐实现 WAF≈1.0x 的机制",
                "source_refs": [
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "解释 SSD P99 尾延迟毛刺与寿命折损的物理根源及 ZNS 顺序追加区的治理方案",
                "name": "ch13-zns-gc-tail-latency",
                "figure": "13-5"
        },
        {
                "figure_id": "图 13-6",
                "number": 6,
                "file": "ch13-multimodal-dataloader-pipeline.svg",
                "asset": "ch13/ch13-multimodal-dataloader-pipeline.svg",
                "section": "13.3.1",
                "caption": "多模态训练中 WebDataset 分片打包消除元数据风暴，以及算例 13-4 中每秒 4,096 段 720p 视频（65,536 fps）从主机 CPU 软解码赤字（缺口 160 核）向 DALI + 8×GPU NVDEC 硬件解码卸载的流水线",
                "source_refs": [
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "展示 PB 级多模态训练如何贯通存储分片顺序供给与片上 NVDEC 异构解码全流程",
                "name": "ch13-multimodal-dataloader-pipeline",
                "figure": "13-6"
        },
        {
                "figure_id": "图 13-7",
                "number": 7,
                "file": "ch13-parquet-lance-vs-webdataset-io.svg",
                "asset": "ch13/ch13-parquet-lance-vs-webdataset-io.svg",
                "section": "13.3.1",
                "caption": "WebDataset 行式顺序流与 Parquet/Lance 列式随机偏移寻址在多模态按列投影与课程学习动态采样下的 I/O 机制对比",
                "source_refs": [
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "对比顺序打包格式与现代列式随机寻址格式在多阶段训练中的适用边界",
                "name": "ch13-parquet-lance-vs-webdataset-io",
                "figure": "13-7"
        },
        {
                "figure_id": "图 13-8",
                "number": 8,
                "file": "ch13-dsec-erofs-ondemand-container-boot.svg",
                "asset": "ch13/ch13-dsec-erofs-ondemand-container-boot.svg",
                "section": "13.3.1",
                "caption": "DSec 平台 8,192 个容器并发启动对比：全量拉取解包（>60 min，单节点写盘 >1,600 GB）与 EROFS 按需块读取 + overlayfs 组合层（35 min，写盘 700 GB，重建复杂度 O(m)+O(k)）",
                "source_refs": [
                        "case-studies/dsec-sandbox-platform.md"
                ],
                "TeachingRole": "展示大规模评测与训练沙箱如何利用只读块镜像按需读取消除镜像拉取风暴",
                "name": "ch13-dsec-erofs-ondemand-container-boot",
                "figure": "13-8"
        },
        {
                "figure_id": "图 13-9",
                "number": 9,
                "file": "ch13-faiss-flat-vs-iks-near-storage.svg",
                "asset": "ch13/ch13-faiss-flat-vs-iks-near-storage.svg",
                "section": "13.3.2",
                "caption": "2^25×768 向量表（FP16 48 GiB）经 64 GB/s 主机链路搬移（805.306 ms）与 8 分片近存 IKS 内部 1 TB/s 扫描（51.540 ms，查询广播 12,288 B、结果归并 3,072 B）对比",
                "source_refs": [
                        "case-studies/retrieval-and-generation.md"
                ],
                "TeachingRole": "用 1 FLOP/B 的极低算术强度解释为何向量扫描应当下沉至近存侧执行",
                "name": "ch13-faiss-flat-vs-iks-near-storage",
                "figure": "13-9"
        },
        {
                "figure_id": "图 13-10",
                "number": 10,
                "file": "ch13-hnsw-memory-vs-diskann-ssd-layout.svg",
                "asset": "ch13/ch13-hnsw-memory-vs-diskann-ssd-layout.svg",
                "section": "13.3.2",
                "caption": "HNSW 全内存分层导航图的邻接指针内存膨胀，以及 DiskANN 将 PQ 压缩向量驻留 DRAM、全精度向量与邻居表紧凑对齐 4 KiB NVMe 扇区的混合检索架构",
                "source_refs": [
                        "case-studies/retrieval-and-generation.md"
                ],
                "TeachingRole": "阐明十亿级向量索引如何通过扇区耦合布局突破 DRAM 容量墙与 SSD 读放大墙",
                "name": "ch13-hnsw-memory-vs-diskann-ssd-layout",
                "figure": "13-10"
        },
        {
                "figure_id": "图 13-11",
                "number": 11,
                "file": "ch13-rag-288-requests-calibration-trap.svg",
                "asset": "ch13/ch13-rag-288-requests-calibration-trap.svg",
                "section": "13.3.2",
                "caption": "实验 13-01（288 请求 Faiss + Qwen3-8B RAG 管线）：校准集（8 题）预选 k=1 在留出集（16 题）因 q14 跌至 15/16 的失守机制，以及 6 个固定 k=4/16 配置单元 24/24 全通过的实测对比",
                "source_refs": [
                        "experiments/ch13/13-01/retrieval-generation/README.md"
                ],
                "TeachingRole": "通过真实负结果揭示小样本校准选择最小 k 的泛化风险与预注册评测纪律",
                "name": "ch13-rag-288-requests-calibration-trap",
                "figure": "13-11"
        },
        {
                "figure_id": "图 13-12",
                "number": 12,
                "file": "ch13-rag-topk-kv-blocks-and-rlimit-gap.svg",
                "asset": "ch13/ch13-rag-topk-kv-blocks-and-rlimit-gap.svg",
                "section": "13.3.2",
                "caption": "实验 13-01 中检索 k=4 与 k=16 对 vLLM 910 块 KV 池的真实占用（23 块 vs. 77 块）及环境变量守卫漏检 EngineCore 子进程（CUDA 分配器峰值 18.58 GB）的边界审计",
                "source_refs": [
                        "experiments/ch13/13-01/retrieval-generation/README.md"
                ],
                "TeachingRole": "连接上游检索 top-k 参数与下游推理引擎物理 KV 块分配及多进程资源监控边界",
                "name": "ch13-rag-topk-kv-blocks-and-rlimit-gap",
                "figure": "13-12"
        },
        {
                "figure_id": "图 13-13",
                "number": 13,
                "file": "ch13-rag-end-to-end-timeline-breakdown.svg",
                "asset": "ch13/ch13-rag-end-to-end-timeline-breakdown.svg",
                "section": "13.3.2",
                "caption": "RAG 请求端到端五阶段关键路径分解，以及算例 13-7 中精检少文档（k=8，检索 200 ms + Prefill 110.2 ms）与粗检多文档（k=32，检索 20 ms + Prefill 351.4 ms）的翻转边界",
                "source_refs": [
                        "case-studies/retrieval-and-generation.md"
                ],
                "TeachingRole": "证明不能孤立优化检索延迟，而必须在同质量下最小化检索加 Prefill 的联合关键路径",
                "name": "ch13-rag-end-to-end-timeline-breakdown",
                "figure": "13-13"
        },
        {
                "figure_id": "图 13-14",
                "number": 14,
                "file": "ch13-three-tier-kv-cache-hierarchy.svg",
                "asset": "ch13/ch13-three-tier-kv-cache-hierarchy.svg",
                "section": "13.4.1",
                "caption": "HBM—主机 DRAM—RDMA 远端内存—NVMe SSD 四级 KV 缓存池（Mooncake / LMCache / HiCache）的带宽阶梯与拉取替代重算的临界条件",
                "source_refs": [
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "建立跨层级 KV 缓存拉取耗时与重算 Prefill 耗时的定量比较框架",
                "name": "ch13-three-tier-kv-cache-hierarchy",
                "figure": "13-14"
        },
        {
                "figure_id": "图 13-15",
                "number": 15,
                "file": "ch13-cacheblend-non-prefix-kv-fusion.svg",
                "asset": "ch13/ch13-cacheblend-non-prefix-kv-fusion.svg",
                "section": "13.4.1",
                "caption": "RAG 多文档任意顺序拼接下的非前缀 KV 复用：直接拼接缺失交叉注意力 vs. CacheBlend 仅对 10%–15% 高偏差词元执行稀疏重算并与 SSD 拉取流水重叠",
                "source_refs": [
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "展示如何突破严格前缀匹配限制，将存储池 KV 复用推广至模块化 RAG 文档块",
                "name": "ch13-cacheblend-non-prefix-kv-fusion",
                "figure": "13-15"
        },
        {
                "figure_id": "图 13-16",
                "number": 16,
                "file": "ch13-ssd-dwpd-endurance-and-admission.svg",
                "asset": "ch13/ch13-ssd-dwpd-endurance-and-admission.svg",
                "section": "13.4.1",
                "caption": "算例 13-9：8 卡节点持续以 4 GB/s 换出 KV 状态时高达 33.75 DWPD 的闪存寿命超限风险，以及通过写准入过滤器（α≤8.89%）与 ZNS 分区将寿命拉回 ≤3 DWPD 安全线的机制",
                "source_refs": [
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "揭示高频 KV 换出对企业级 SSD 擦写寿命的毁灭性冲击及准入节流设计准则",
                "name": "ch13-ssd-dwpd-endurance-and-admission",
                "figure": "13-16"
        },
        {
                "figure_id": "图 13-17",
                "number": 17,
                "file": "ch13-storage-diagnosis-workflow.svg",
                "asset": "ch13/ch13-storage-diagnosis-workflow.svg",
                "section": "13.4.1",
                "caption": "AI 存储子系统性能与寿命瓶颈的五步定量诊断工作流",
                "source_refs": [
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "为系统工程师提供从元数据、PCIe 拓扑、闪存粒度、GC 毛刺到异构解码的排障流程",
                "name": "ch13-storage-diagnosis-workflow",
                "figure": "13-17"
        },
        {
                "figure_id": "图 13-18",
                "number": 18,
                "file": "ch13-full-stack-ai-data-plane.svg",
                "asset": "ch13/ch13-full-stack-ai-data-plane.svg",
                "section": "13.4.1",
                "caption": "贯穿预训练数据湖、RAG 向量知识库、多级 KV 缓存池与万级容器镜像的 AI 原生数据底座全景架构图",
                "source_refs": [
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "串联第 13 章上半部分存储与缓存体系，呈现计算—网络—存储深度融合的现代 AI 数据平面",
                "name": "ch13-full-stack-ai-data-plane",
                "figure": "13-18"
        },
        {
                "figure_id": "图 13-19",
                "number": 19,
                "file": "ch13-falconfs-server-side-path-resolution.svg",
                "asset": "ch13/ch13-falconfs-server-side-path-resolution.svg",
                "section": "13.4.2",
                "caption": "传统 POSIX 逐跳目录解析（D=6 时每次打开触发 6–7 次 RPC）与 FalconFS（NSDI '26）服务端复合路径解析及混合哈希无锁目录布局对比",
                "source_refs": [
                        "research/2026-infra-survey/reading-nsdi-2026.md"
                ],
                "TeachingRole": "展示如何将路径解析下推至服务端以消除十亿级多模态小文件训练中的元数据网络往返放大",
                "name": "ch13-falconfs-server-side-path-resolution",
                "figure": "13-19"
        },
        {
                "figure_id": "图 13-20",
                "number": 20,
                "file": "ch13-multimodal-tar-webdataset-vs-lance-random.svg",
                "asset": "ch13/ch13-multimodal-tar-webdataset-vs-lance-random.svg",
                "section": "13.4.2",
                "caption": "静态顺序 Shuffle（WebDataset Tar 连续分片 / FalconFS）与课程学习及在线难例重采样（Lance O(1) 偏移点查 + Umap 页映射）的磁盘寻道与读放大对比",
                "source_refs": [
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "阐明训练数据访问模式从静态顺序遍历转向动态按长度/奖励抽样时的存储格式选型边界",
                "name": "ch13-multimodal-tar-webdataset-vs-lance-random",
                "figure": "13-20"
        },
        {
                "figure_id": "图 13-21",
                "number": 21,
                "file": "ch13-distvs-helmsman-disaggregated-vector-search.svg",
                "asset": "ch13/ch13-distvs-helmsman-disaggregated-vector-search.svg",
                "section": "13.4.2",
                "caption": "DistVS（NSDI '26）计算—内存解耦三级精度筛选（本地 DRAM 粗筛 → 远端内存精筛 → SSD 全精度重排）与 Helmsman（OSDI '26）大规模倒排聚类闪存 ANNS 架构对比",
                "source_refs": [
                        "research/2026-infra-survey/reading-nsdi-2026.md",
                        "research/2026-infra-survey/reading-osdi-2026.md"
                ],
                "TeachingRole": "对比超大规模向量检索跨越 DRAM、CXL/远端内存与 NVMe SSD 时的多级筛选与连续块布局，并明确区分有损向量召回与无损 KV 分层",
                "name": "ch13-distvs-helmsman-disaggregated-vector-search",
                "figure": "13-21"
        },
        {
                "figure_id": "图 13-22",
                "number": 22,
                "file": "ch13-copilotio-gds-vs-cpu-assisted-io.svg",
                "asset": "ch13/ch13-copilotio-gds-vs-cpu-assisted-io.svg",
                "section": "13.4.2",
                "caption": "大块连续裸张量 GPUDirect Storage（GDS）直通与细粒度/需解压场景下 CoPilotIO（OSDI '26）利用主机 CPU 协处理轮询释放 GPU SM 算力的架构对比",
                "source_refs": [
                        "research/2026-infra-survey/reading-osdi-2026.md"
                ],
                "TeachingRole": "揭示纯 GPU 发起细粒度 I/O 时的寄存器与 SM 争用退化，建立按 I/O 粒度与计算耦合度划分 GDS 与 CPU 协处理的边界",
                "name": "ch13-copilotio-gds-vs-cpu-assisted-io",
                "figure": "13-22"
        },
        {
                "figure_id": "图 13-23",
                "number": 23,
                "file": "ch13-umap-moonbright-mmap-and-gpu-page-table.svg",
                "asset": "ch13/ch13-umap-moonbright-mmap-and-gpu-page-table.svg",
                "section": "13.4.2",
                "caption": "Umap（OSDI '26）分布式文件系统页粒度矩阵 mmap 按需预取免除容器 OOM 与 MoonBright（OSDI '26）设备端 GPU 页表物化及延迟 TLB 同步协同架构",
                "source_refs": [
                        "research/2026-infra-survey/reading-osdi-2026.md"
                ],
                "TeachingRole": "剖析主机侧 DFS 内存映射与设备侧 GPU 页表虚拟化如何消除双重缓冲拷贝与页表同步停顿",
                "name": "ch13-umap-moonbright-mmap-and-gpu-page-table",
                "figure": "13-23"
        },
        {
                "figure_id": "图 13-24",
                "number": 24,
                "file": "ch13-ai-storage-bandwidth-latency-iops-cube.svg",
                "asset": "ch13/ch13-ai-storage-bandwidth-latency-iops-cube.svg",
                "section": "13.4.2",
                "caption": "AI 原生存储底座全场景（检查点、十亿小文件、多级 KV、十亿向量 ANNS、细粒度协同 I/O）在带宽—延迟—IOPS 三维空间的映射总览",
                "source_refs": [
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "总结第 13 章上半部分存储负载特征与对应系统原语的工程选型总表",
                "name": "ch13-ai-storage-bandwidth-latency-iops-cube",
                "figure": "13-24"
        },
        {
                "figure_id": "图 13-25",
                "number": 25,
                "file": "ch13-nixl-unified-nonblocking-transfer-plane.svg",
                "asset": "ch13/ch13-nixl-unified-nonblocking-transfer-plane.svg",
                "section": "13.4.2",
                "caption": "NIXL 跨 HBM、Host DRAM、本地 NVMe 与远端 RDMA 存储池的统一异步传输抽象与路径自动协商",
                "source_refs": [
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "展示 NVIDIA NIXL 如何统一封装多介质零拷贝异步描述符并消减多级 KV 换入换出尾延迟",
                "name": "ch13-nixl-unified-nonblocking-transfer-plane",
                "figure": "13-25"
        },
        {
                "figure_id": "图 13-26",
                "number": 26,
                "file": "ch13-checkpoint-replication-to-erasure-coding-lifecycle.svg",
                "asset": "ch13/ch13-checkpoint-replication-to-erasure-coding-lifecycle.svg",
                "section": "13.4.2",
                "caption": "PB 级检查点从热层 3 副本秒级闪存暂存到温冷层 RS(10,4) 纠删码对象归档的容量与恢复带宽瀑布图",
                "source_refs": [
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "建立十万卡训练集群检查点热层 3 副本与温冷层 RS(10,4) 纠删码的分层生命周期归档账本",
                "name": "ch13-checkpoint-replication-to-erasure-coding-lifecycle",
                "figure": "13-26"
        },
        {
                "figure_id": "图 13-27",
                "number": 27,
                "file": "ch13-lmcache-prefix-kv-compression-and-gds-prefetch.svg",
                "asset": "ch13/ch13-lmcache-prefix-kv-compression-and-gds-prefetch.svg",
                "section": "13.4.2",
                "caption": "LMCache / Mooncake Store 跨请求前缀 KV 锚点差分压缩（3.80×）、16 块逐层组 GDS 异步预取与 GPU 解压流水线（713.60 ms 降至 273.02 ms，提速 2.61×）及 DRAM—NVMe—S3 三级冷热分层",
                "source_refs": [
                        "manuscripts/ch13/figure-data.json",
                        "experiments/ch13/13-04/README.md"
                ],
                "TeachingRole": "展示跨请求前缀 KV 锚点差分压缩、逐层组 GDS 异步流水预取与三级冷热对象存储池的延迟及 TCO 收益",
                "name": "ch13-lmcache-prefix-kv-compression-and-gds-prefetch",
                "figure": "13-27"
        },
        {
                "figure_id": "图 13-28",
                "number": 28,
                "file": "ch13-checkpoint-and-kv-unified-nvme-pool-qos.svg",
                "asset": "ch13/ch13-checkpoint-and-kv-unified-nvme-pool-qos.svg",
                "section": "13.4.2",
                "caption": "训练检查点 1 MiB 突发写与在线 KV 缓存读取共享 NVMe 池时，单一 FIFO 队列队头阻塞（P99 达 13.80 ms）与 NVMe 硬件 WRR（8:2 权重）+ 128 KiB 写微块切片 QoS 隔离（P99 降至 1.35 ms，改善 10.22×）对比",
                "source_refs": [
                        "manuscripts/ch13/figure-data.json",
                        "experiments/ch13/13-04/README.md"
                ],
                "TeachingRole": "剖析共享 NVMe 存储池中大块检查点写对在线 KV 读的队头阻塞机理及硬件 WRR 加写微块切片的 QoS 治理",
                "name": "ch13-checkpoint-and-kv-unified-nvme-pool-qos",
                "figure": "13-28"
        },
        {
                "figure_id": "图 13-29",
                "number": 29,
                "file": "ch13-100pb-commoncrawl-to-15t-tokens-curation-funnel.svg",
                "asset": "ch13/ch13-100pb-commoncrawl-to-15t-tokens-curation-funnel.svg",
                "section": "13.5.1",
                "caption": "从 100 PB CommonCrawl 原始 WARC 归档经七级提纯漏斗产出 15.0T 精品词元的体积与算力瀑布（算例 13-15）",
                "source_refs": [
                        "experiments/ch13/13-02/run.py",
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "展示百 PB 级原始网页经正文抽取、启发式过滤、去重、分级与去污仅留存 2.40% 的全链路账本",
                "name": "ch13-100pb-commoncrawl-to-15t-tokens-curation-funnel",
                "figure": "13-29"
        },
        {
                "figure_id": "图 13-30",
                "number": 30,
                "file": "ch13-minhash-lsh-band-row-s-curve-probability.svg",
                "asset": "ch13/ch13-minhash-lsh-band-row-s-curve-probability.svg",
                "section": "13.5.2",
                "caption": "MinHash LSH 局部敏感哈希在 (b=16, r=8) 分桶配置下的 S 型碰撞概率曲线 P(s) = 1 - (1 - s^r)^b（算例 13-16）",
                "source_refs": [
                        "experiments/ch13/13-02/run.py",
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "推导阈值拐点 s* = 0.7071 以及 s=0.80 捕获率 94.70%、s=0.50 误碰率 6.07% 的概率边界",
                "name": "ch13-minhash-lsh-band-row-s-curve-probability",
                "figure": "13-30"
        },
        {
                "figure_id": "图 13-31",
                "number": 31,
                "file": "ch13-distributed-minhash-shuffle-and-gpu-union-find.svg",
                "asset": "ch13/ch13-distributed-minhash-shuffle-and-gpu-union-find.svg",
                "section": "13.5.2",
                "caption": "50 亿篇文档模糊去重的分布式流水线：Shingle 哈希、按 Band 洗牌与 GPU 连通分量（Large-Star/Small-Star）聚簇（算例 13-17）",
                "source_refs": [
                        "experiments/ch13/13-02/run.py",
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "揭示 2.56 TB 签名跨卡洗牌与分布式图连通分量求解传递性重复簇的系统机理",
                "name": "ch13-distributed-minhash-shuffle-and-gpu-union-find",
                "figure": "13-31"
        },
        {
                "figure_id": "图 13-32",
                "number": 32,
                "file": "ch13-cpu-datatrove-vs-gpu-nemo-curator-throughput.svg",
                "asset": "ch13/ch13-cpu-datatrove-vs-gpu-nemo-curator-throughput.svg",
                "section": "13.5.2",
                "caption": "CPU 集群（DataTrove）与 GPU 加速流水线（NeMo Curator + cuDF + cuGraph）在百 TB 语料去重中的耗时（80.0x 加速）与 TCO（降 4.50x）对比（算例 13-17）",
                "source_refs": [
                        "experiments/ch13/13-02/run.py",
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "对比 4,096 核 CPU 跑 148 小时与 64 卡 H100 跑 1.85 小时的吞吐与成本账本",
                "name": "ch13-cpu-datatrove-vs-gpu-nemo-curator-throughput",
                "figure": "13-32"
        },
        {
                "figure_id": "图 13-33",
                "number": 33,
                "file": "ch13-document-fuzzy-dedup-vs-suffix-array-substring-dedup.svg",
                "asset": "ch13/ch13-document-fuzzy-dedup-vs-suffix-array-substring-dedup.svg",
                "section": "13.5.2",
                "caption": "文档级 MinHash 模糊去重与基于后缀数组（Suffix Array）的 50 词元级精确重复模板/导航栏剥离协同",
                "source_refs": [
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "阐明宏观整篇模糊去重与微观后缀数组子串切除的互补分工",
                "name": "ch13-document-fuzzy-dedup-vs-suffix-array-substring-dedup",
                "figure": "13-33"
        },
        {
                "figure_id": "图 13-34",
                "number": 34,
                "file": "ch13-llm-rubric-distillation-to-fasttext-bert-classifier.svg",
                "asset": "ch13/ch13-llm-rubric-distillation-to-fasttext-bert-classifier.svg",
                "section": "13.6.1",
                "caption": "用 70B 教师模型标注 500 万样本蒸馏 110M 轻量级质量分类器（FineWeb-Edu / DCLM）实现 55.82 倍算力节省的三级漏斗（算例 13-18）",
                "source_refs": [
                        "experiments/ch13/13-03/run.py",
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "证明教师精标种子与轻量分类器全量推理相比直接用大模型打分节省 98.21% 算力",
                "name": "ch13-llm-rubric-distillation-to-fasttext-bert-classifier",
                "figure": "13-34"
        },
        {
                "figure_id": "图 13-35",
                "number": 35,
                "file": "ch13-nemotron-cc-classifier-ensembling-and-rephrasing.svg",
                "asset": "ch13/ch13-nemotron-cc-classifier-ensembling-and-rephrasing.svg",
                "section": "13.6.1",
                "caption": "Nemotron-CC 范式：多分类器集成打分分桶与对高知识价值但表述残缺网页的合成改写抢救（Synthetic Rephrasing）",
                "source_refs": [
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "通过多分类器集成与百科风格合成改写突破激进过滤带来的独特词元枯竭墙",
                "name": "ch13-nemotron-cc-classifier-ensembling-and-rephrasing",
                "figure": "13-35"
        },
        {
                "figure_id": "图 13-36",
                "number": 36,
                "file": "ch13-benchmark-decontamination-ngram-and-embedding-hybrid.svg",
                "asset": "ch13/ch13-benchmark-decontamination-ngram-and-embedding-hybrid.svg",
                "section": "13.6.1",
                "caption": "防止评测集刷榜失真的双层去污管线（13-gram 子串匹配 + 稠密语义向量近邻剔除防止同义改写与跨语言污染）",
                "source_refs": [
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "构建字面 n-gram 自动机与稠密向量检索结合的评测基准防泄漏防线",
                "name": "ch13-benchmark-decontamination-ngram-and-embedding-hybrid",
                "figure": "13-36"
        },
        {
                "figure_id": "图 13-37",
                "number": 37,
                "file": "ch13-doremi-regmix-domain-weight-optimization.svg",
                "asset": "ch13/ch13-doremi-regmix-domain-weight-optimization.svg",
                "section": "13.6.1",
                "caption": "基于小型代理模型（Proxy Model）的领域配比寻优（DoReMi / RegMix）与三阶段预训练动态退火曲线",
                "source_refs": [
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "展示如何用小模型网格搜索最优领域配比并在预训练末段执行高纯度数据极速退火",
                "name": "ch13-doremi-regmix-domain-weight-optimization",
                "figure": "13-37"
        },
        {
                "figure_id": "图 13-38",
                "number": 38,
                "file": "ch13-online-serving-vs-offline-synthetic-factory-roofline.svg",
                "asset": "ch13/ch13-online-serving-vs-offline-synthetic-factory-roofline.svg",
                "section": "13.6.2",
                "caption": "在线低延迟服务（B=32，MFU 11.2%）与离线合成数据工厂（B=2,048，MFU 58.4%，吞吐升 7.20 倍、成本降 13.16 倍）的 Roofline 跃迁（算例 13-19）",
                "source_refs": [
                        "experiments/ch13/13-03/run.py",
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "阐明卸下实时延迟枷锁后通过超大批量将自回归解码推入 Tensor Core 计算饱和区的经济学",
                "name": "ch13-online-serving-vs-offline-synthetic-factory-roofline",
                "figure": "13-38"
        },
        {
                "figure_id": "图 13-39",
                "number": 39,
                "file": "ch13-synthetic-data-generator-verifier-rejection-funnel.svg",
                "asset": "ch13/ch13-synthetic-data-generator-verifier-rejection-funnel.svg",
                "section": "13.6.2",
                "caption": "推理与代码合成数据工厂的“多样化种子 -> 高温批量采样 -> 沙箱测试验证（31.25%）-> 自纠重写（综合良率 58.75%）”校验账本（算例 13-19）",
                "source_refs": [
                        "experiments/ch13/13-03/run.py",
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "利用外部确定性沙箱验证器与错误轨迹重写防止模型自噬并将验证词元成本锁定在 0.075 美元/M",
                "name": "ch13-synthetic-data-generator-verifier-rejection-funnel",
                "figure": "13-39"
        },
        {
                "figure_id": "图 13-40",
                "number": 40,
                "file": "ch13-slack-capacity-harvesting-for-synthetic-generation.svg",
                "asset": "ch13/ch13-slack-capacity-harvesting-for-synthetic-generation.svg",
                "section": "13.6.2",
                "caption": "利用在线推理集群昼夜波谷与训练维护空窗自动拉起离线合成数据作业的零边际硬件成本收割架构",
                "source_refs": [
                        "experiments/ch13/13-03/run.py",
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "通过毫秒级抢占式波谷算力收割将离线合成 GPU 小时单价再压降 45%",
                "name": "ch13-slack-capacity-harvesting-for-synthetic-generation",
                "figure": "13-40"
        },
        {
                "figure_id": "图 13-41",
                "number": 41,
                "file": "ch13-pdf-scientific-ocr-layout-math-extraction-pipeline.svg",
                "asset": "ch13/ch13-pdf-scientific-ocr-layout-math-extraction-pipeline.svg",
                "section": "13.6.2",
                "caption": "高质量学术论文与教材 PDF 语料的版面分析、公式 LaTeX 还原与多模态图文交错提取流水线",
                "source_refs": [
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "解决传统 PDF 文本提取破坏分栏与数学公式的难题，解锁高阶 STEM 专业文献",
                "name": "ch13-pdf-scientific-ocr-layout-math-extraction-pipeline",
                "figure": "13-41"
        },
        {
                "figure_id": "图 13-42",
                "number": 42,
                "file": "ch13-tokenized-dataset-mmap-shuffling-and-sequence-packing.svg",
                "asset": "ch13/ch13-tokenized-dataset-mmap-shuffling-and-sequence-packing.svg",
                "section": "13.6.2",
                "caption": "万亿词元二进制索引（.bin/.idx mmap）的确定性全局伪随机洗牌与最佳适应序列打包（Best-Fit Packing）零填充损耗",
                "source_refs": [
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "通过 Feistel 块级置换与最佳适应无截断打包消除长文档腰斩与跨文档注意力串扰",
                "name": "ch13-tokenized-dataset-mmap-shuffling-and-sequence-packing",
                "figure": "13-42"
        },
        {
                "figure_id": "图 13-43",
                "number": 43,
                "file": "ch13-data-lineage-copyright-optout-and-audit-ledger.svg",
                "asset": "ch13/ch13-data-lineage-copyright-optout-and-audit-ledger.svg",
                "section": "13.6.2",
                "caption": "面向合规与可追溯性的语料血缘账本（Data Provenance）、版权退订剔除与训练批次精确回溯索引",
                "source_refs": [
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "建立从每个训练 Batch 反向追溯至原始网页 SHA-256 与许可证标签的密码学审计链路",
                "name": "ch13-data-lineage-copyright-optout-and-audit-ledger",
                "figure": "13-43"
        },
        {
                "figure_id": "图 13-44",
                "number": 44,
                "file": "ch13-exascale-data-and-synthetic-factory-master-architecture.svg",
                "asset": "ch13/ch13-exascale-data-and-synthetic-factory-master-architecture.svg",
                "section": "13.6.2",
                "caption": "百亿亿次 Token 数据工程与合成数据工厂从原始爬虫、GPU 去重到离线合成退火的全栈总图",
                "source_refs": [
                        "experiments/ch13/13-02/run.py",
                        "experiments/ch13/13-03/run.py",
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "串联第 13 章下半部分七级提纯漏斗、GPU 连通分量去重、分类器蒸馏与合成数据工厂的完整体系",
                "name": "ch13-exascale-data-and-synthetic-factory-master-architecture",
                "figure": "13-44"
        },
        {
                "figure_id": "图 13-45",
                "number": 45,
                "file": "ch13-mineru-olmocr-pdf-layout-and-formula-gpu-pipeline.svg",
                "asset": "ch13/ch13-mineru-olmocr-pdf-layout-and-formula-gpu-pipeline.svg",
                "section": "13.6.2",
                "caption": "百万册级复杂学术 PDF 版面解析与公式还原（MinerU / olmOCR 混合快慢分流与跨页子区域 TensorRT 动态批处理，吞吐提速 35.27 倍）（算例 13-20）",
                "source_refs": [
                        "experiments/ch13/13-05/run.py",
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "通过 65% 原生数字页正文直抽与跨页公式框 TensorRT 动态批处理将单卡吞吐从 1.60 页/s 推高至 56.43 页/s",
                "name": "ch13-mineru-olmocr-pdf-layout-and-formula-gpu-pipeline",
                "figure": "13-45"
        },
        {
                "figure_id": "图 13-46",
                "number": 46,
                "file": "ch13-repo-level-ast-dependency-topological-packing.svg",
                "asset": "ch13/ch13-repo-level-ast-dependency-topological-packing.svg",
                "section": "13.6.2",
                "caption": "代码仓库级跨文件 AST 依赖图提取、Tarjan 强连通破环与 Kahn 拓扑排序序列打包（因果依赖覆盖率从 42.50% 跃升至 95.00%）（算例 13-21）",
                "source_refs": [
                        "experiments/ch13/13-05/run.py",
                        "manuscripts/ch13/figure-data.json"
                ],
                "TeachingRole": "消除路径字母序拼接导致的跨文件定义盲区，使 64k 代码预训练跨文件 API 困惑度降低 39.85%",
                "name": "ch13-repo-level-ast-dependency-topological-packing",
                "figure": "13-46"
        }
]
    (HERE / 'figure-index.json').write_text(json.dumps(fig_index, ensure_ascii=False, indent=2) + '\n')

    md = HERE.parent / '13-存储系统与数据工程.md'
    if md.exists():
        text_md = md.read_text(encoding='utf-8')
        display = re.findall(r'\$\$(.+?)\$\$', text_md, re.S)
        inline = re.findall(r'(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)', text_md)
        formulas = [f.strip() for f in display + inline]
        katex_js = ROOT / 'manuscripts/ch06/vendor/katex/katex.js'
        js = (
            'const katex=require(process.argv[1]);const fs=require("fs");'
            'const a=JSON.parse(fs.readFileSync(0,"utf8"));const e=[];'
            'a.forEach((s,i)=>{try{katex.renderToString(s,{throwOnError:true,displayMode:i<'
            + str(len(display))
            + '})}catch(err){e.push({index:i,formula:s,error:String(err)})}});'
            'console.log(JSON.stringify({checked:a.length,display:'
            + str(len(display))
            + ',inline:'
            + str(len(inline))
            + ',errors:e},null,2));'
        )
        res = subprocess.run(
            ['node', '-e', js, str(katex_js)],
            input=json.dumps(formulas),
            text=True,
            capture_output=True,
            cwd=ROOT,
            check=True,
        )
        math_val = json.loads(res.stdout)
        assert not math_val['errors'], math_val['errors']
        (HERE / 'math-validation.json').write_text(json.dumps(math_val, ensure_ascii=False, indent=2) + '\n')

        import markdown

        def repl_img(m):
            alt, rel = m.group(1), m.group(2)
            p = (md.parent / rel).with_suffix('.png')
            b64 = base64.b64encode(p.read_bytes()).decode()
            return f'<figure><img alt="{alt}" src="data:image/png;base64,{b64}"><figcaption>{alt}</figcaption></figure>'

        work = re.sub(r'!\[([^\]]*)\]\((ch13/[^)]+)\)', repl_img, text_md)
        body = markdown.markdown(work, extensions=['tables', 'fenced_code', 'footnotes', 'toc'])
        html = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>第 13 章：存储系统与数据工程</title></head><body>{body}</body></html>\n'
        out_html = preview_path(md)
        out_html.write_text(html, encoding='utf-8')

    sources = [
        HERE / 'build.py',
        HERE / 'figure-data.json',
        HERE / 'figure-index.json',
        ROOT / 'calculations/results/ch13-storage-kv-io.json',
        ROOT / 'calculations/results/ch13-data-funnel-dedup.json',
        ROOT / 'experiments/ch13/13-02/run.py',
        ROOT / 'experiments/ch13/13-03/run.py',
        ROOT / 'experiments/ch13/13-04/run.py',
        ROOT / 'experiments/ch13/13-05/run.py',
    ]
    if md.exists():
        sources.insert(3, md)
    (HERE / 'sources.json').write_text(
        json.dumps(
            {"chapter": "ch13", "sources": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in sources]},
            ensure_ascii=False,
            indent=2,
        )
        + '\n'
    )
    all_outputs = sorted(outputs) + [
        HERE / 'figure-data.json',
        HERE / 'figure-index.json',
        HERE / 'teaching-layout-validation.json',
    ]
    if (HERE / 'math-validation.json').exists():
        all_outputs.append(HERE / 'math-validation.json')
    (HERE / 'manifest.json').write_text(
        json.dumps(
            {"chapter": "ch13", "outputs": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in all_outputs]},
            ensure_ascii=False,
            indent=2,
        )
        + '\n'
    )
    print(f'Built {len(outputs)} figure files for Chapter 13; 0 layout warnings.')


if __name__ == '__main__':
    main()
