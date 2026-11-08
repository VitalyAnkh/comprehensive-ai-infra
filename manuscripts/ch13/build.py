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
    box(ax, 0.04, 0.20, 0.42, 0.16, '消除用户态双重缓冲拷贝\n严控容器 RSS 水位免误杀', 'blue', size=11)
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

    card(ax, 0.52, 0.04, 0.46, 0.92, 'B. 算例 13-11：三通道并发预取', color='teal', title_size=11)
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
    box(ax, 0.54, 0.20, 0.42, 0.16, '算例 13-12：总占 1,448 TB\n净省 1,552 TB (省 51.73%)', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '温冷：容量减半、高可靠归档', size=11, ha='center', weight='bold', color=BORDER_COL['blue'])
    ex.save(fig, 'ch13-checkpoint-replication-to-erasure-coding-lifecycle')


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
    outputs = ex.finish()

    fig_data = build_figure_data()
    (HERE / 'figure-data.json').write_text(json.dumps(fig_data, ensure_ascii=False, indent=2) + '\n')

    fig_index = [
        {
            "figure_id": "图 13-1",
            "number": 1,
            "file": "ch13-storage-hierarchy-and-access-patterns.svg",
            "asset": "ch13/ch13-storage-hierarchy-and-access-patterns.svg",
            "section": "13.1.2",
            "caption": "AI 存储介质金字塔（L1 HBM 至 L5 对象存储）的延迟、IOPS 与带宽量级，以及四类核心 AI 负载的 I/O 访问瓶颈",
            "source_refs": ["manuscripts/ch13/figure-data.json"],
            "TeachingRole": "建立全章五级存储金字塔与四类 AI I/O 访问特征的定量坐标"
        },
        {
            "figure_id": "图 13-2",
            "number": 2,
            "file": "ch13-posix-pagecache-vs-gds-dma.svg",
            "asset": "ch13/ch13-posix-pagecache-vs-gds-dma.svg",
            "section": "13.2.2",
            "caption": "算例 13-1：单机 8 盘 PCIe Gen5 NVMe SSD 加载 140 GB 权重时，传统 POSIX 页缓存双重拷贝（280 GB Root Complex 流量，耗时 2.80 s）与 GPUDirect Storage P2P DMA 零拷贝直通（100 GB/s，耗时 1.40 s）对比",
            "source_refs": ["manuscripts/ch13/figure-data.json"],
            "TeachingRole": "直观揭示 Page Cache 在 PCIe Root Complex 上的流量翻倍效应与 GDS P2P DMA 的旁路机理"
        },
        {
            "figure_id": "图 13-3",
            "number": 3,
            "file": "ch13-3fs-craq-architecture.svg",
            "asset": "ch13/ch13-3fs-craq-architecture.svg",
            "section": "13.3.3",
            "caption": "DeepSeek 3FS 的 CRAQ 链式复制协议（链式顺序写与全副本强一致分摊读）及算例 13-2 中 180 节点存储集群实现 6.6 TiB/s 聚合读带宽的网卡—磁盘配比账本",
            "source_refs": ["manuscripts/ch13/figure-data.json"],
            "TeachingRole": "展示 CRAQ 如何将强一致读吞吐线性扩展至全部副本，以及磁盘超配 2.24 倍如何掩盖写放大与 GC"
        },
        {
            "figure_id": "图 13-4",
            "number": 4,
            "file": "ch13-kv-layout-read-amplification.svg",
            "asset": "ch13/ch13-kv-layout-read-amplification.svg",
            "section": "13.4.2",
            "caption": "算例 13-3：32 层模型卸载 16-token KV 块时，Layer-major 按层分离布局（触发 4 KiB LBA 读放大 RAF=2.0x 与 800k IOPS 墙，耗时 5.12 ms）与 Block-major 全层打包布局（RAF=1.0x，跑满 12.5 GiB/s，耗时 0.625 ms，提速 8.192x）对比",
            "source_refs": ["manuscripts/ch13/figure-data.json"],
            "TeachingRole": "阐明 NAND Flash 最小读粒度与 IOPS 墙如何决定 KV Cache 的跨层连续打包布局"
        },
        {
            "figure_id": "图 13-5",
            "number": 5,
            "file": "ch13-zns-gc-tail-latency.svg",
            "asset": "ch13/ch13-zns-gc-tail-latency.svg",
            "section": "13.4.3",
            "caption": "传统块接口 NVMe SSD 在混写负载下的 FTL 垃圾回收（GC）写放大（WAF=2.5–4.0x）与读尾延迟毛刺，以及 ZNS 分区命名空间 SSD 按会话/检查点生命周期对齐实现 WAF≈1.0x 的机制",
            "source_refs": ["manuscripts/ch13/figure-data.json"],
            "TeachingRole": "解释 SSD P99 尾延迟毛刺与寿命折损的物理根源及 ZNS 顺序追加区的治理方案"
        },
        {
            "figure_id": "图 13-6",
            "number": 6,
            "file": "ch13-multimodal-dataloader-pipeline.svg",
            "asset": "ch13/ch13-multimodal-dataloader-pipeline.svg",
            "section": "13.5.2",
            "caption": "多模态训练中 WebDataset 分片打包消除元数据风暴，以及算例 13-4 中每秒 4,096 段 720p 视频（65,536 fps）从主机 CPU 软解码赤字（缺口 160 核）向 DALI + 8×GPU NVDEC 硬件解码卸载的流水线",
            "source_refs": ["manuscripts/ch13/figure-data.json"],
            "TeachingRole": "展示 PB 级多模态训练如何贯通存储分片顺序供给与片上 NVDEC 异构解码全流程"
        },
        {
            "figure_id": "图 13-7",
            "number": 7,
            "file": "ch13-parquet-lance-vs-webdataset-io.svg",
            "asset": "ch13/ch13-parquet-lance-vs-webdataset-io.svg",
            "section": "13.5.3",
            "caption": "WebDataset 行式顺序流与 Parquet/Lance 列式随机偏移寻址在多模态按列投影与课程学习动态采样下的 I/O 机制对比",
            "source_refs": ["manuscripts/ch13/figure-data.json"],
            "TeachingRole": "对比顺序打包格式与现代列式随机寻址格式在多阶段训练中的适用边界"
        },
        {
            "figure_id": "图 13-8",
            "number": 8,
            "file": "ch13-dsec-erofs-ondemand-container-boot.svg",
            "asset": "ch13/ch13-dsec-erofs-ondemand-container-boot.svg",
            "section": "13.5.3",
            "caption": "DSec 平台 8,192 个容器并发启动对比：全量拉取解包（>60 min，单节点写盘 >1,600 GB）与 EROFS 按需块读取 + overlayfs 组合层（35 min，写盘 700 GB，重建复杂度 O(m)+O(k)）",
            "source_refs": ["case-studies/dsec-sandbox-platform.md"],
            "TeachingRole": "展示大规模评测与训练沙箱如何利用只读块镜像按需读取消除镜像拉取风暴"
        },
        {
            "figure_id": "图 13-9",
            "number": 9,
            "file": "ch13-faiss-flat-vs-iks-near-storage.svg",
            "asset": "ch13/ch13-faiss-flat-vs-iks-near-storage.svg",
            "section": "13.6.1",
            "caption": "2^25×768 向量表（FP16 48 GiB）经 64 GB/s 主机链路搬移（805.306 ms）与 8 分片近存 IKS 内部 1 TB/s 扫描（51.540 ms，查询广播 12,288 B、结果归并 3,072 B）对比",
            "source_refs": ["case-studies/retrieval-and-generation.md"],
            "TeachingRole": "用 1 FLOP/B 的极低算术强度解释为何向量扫描应当下沉至近存侧执行"
        },
        {
            "figure_id": "图 13-10",
            "number": 10,
            "file": "ch13-hnsw-memory-vs-diskann-ssd-layout.svg",
            "asset": "ch13/ch13-hnsw-memory-vs-diskann-ssd-layout.svg",
            "section": "13.6.1",
            "caption": "HNSW 全内存分层导航图的邻接指针内存膨胀，以及 DiskANN 将 PQ 压缩向量驻留 DRAM、全精度向量与邻居表紧凑对齐 4 KiB NVMe 扇区的混合检索架构",
            "source_refs": ["case-studies/retrieval-and-generation.md"],
            "TeachingRole": "阐明十亿级向量索引如何通过扇区耦合布局突破 DRAM 容量墙与 SSD 读放大墙"
        },
        {
            "figure_id": "图 13-11",
            "number": 11,
            "file": "ch13-rag-288-requests-calibration-trap.svg",
            "asset": "ch13/ch13-rag-288-requests-calibration-trap.svg",
            "section": "13.6.2",
            "caption": "实验 13-01（288 请求 Faiss + Qwen3-8B RAG 管线）：校准集（8 题）预选 k=1 在留出集（16 题）因 q14 跌至 15/16 的失守机制，以及 6 个固定 k=4/16 配置单元 24/24 全通过的实测对比",
            "source_refs": ["experiments/ch13/13-01/retrieval-generation/README.md"],
            "TeachingRole": "通过真实负结果揭示小样本校准选择最小 k 的泛化风险与预注册评测纪律"
        },
        {
            "figure_id": "图 13-12",
            "number": 12,
            "file": "ch13-rag-topk-kv-blocks-and-rlimit-gap.svg",
            "asset": "ch13/ch13-rag-topk-kv-blocks-and-rlimit-gap.svg",
            "section": "13.6.2",
            "caption": "实验 13-01 中检索 k=4 与 k=16 对 vLLM 910 块 KV 池的真实占用（23 块 vs. 77 块）及环境变量守卫漏检 EngineCore 子进程（CUDA 分配器峰值 18.58 GB）的边界审计",
            "source_refs": ["experiments/ch13/13-01/retrieval-generation/README.md"],
            "TeachingRole": "连接上游检索 top-k 参数与下游推理引擎物理 KV 块分配及多进程资源监控边界"
        },
        {
            "figure_id": "图 13-13",
            "number": 13,
            "file": "ch13-rag-end-to-end-timeline-breakdown.svg",
            "asset": "ch13/ch13-rag-end-to-end-timeline-breakdown.svg",
            "section": "13.6.2",
            "caption": "RAG 请求端到端五阶段关键路径分解，以及算例 13-5 中精检少文档（k=8，检索 200 ms + Prefill 110.2 ms）与粗检多文档（k=32，检索 20 ms + Prefill 351.4 ms）的翻转边界",
            "source_refs": ["case-studies/retrieval-and-generation.md"],
            "TeachingRole": "证明不能孤立优化检索延迟，而必须在同质量下最小化检索加 Prefill 的联合关键路径"
        },
        {
            "figure_id": "图 13-14",
            "number": 14,
            "file": "ch13-three-tier-kv-cache-hierarchy.svg",
            "asset": "ch13/ch13-three-tier-kv-cache-hierarchy.svg",
            "section": "13.7.1",
            "caption": "HBM—主机 DRAM—RDMA 远端内存—NVMe SSD 四级 KV 缓存池（Mooncake / LMCache / HiCache）的带宽阶梯与拉取替代重算的临界条件",
            "source_refs": ["manuscripts/ch13/figure-data.json"],
            "TeachingRole": "建立跨层级 KV 缓存拉取耗时与重算 Prefill 耗时的定量比较框架"
        },
        {
            "figure_id": "图 13-15",
            "number": 15,
            "file": "ch13-cacheblend-non-prefix-kv-fusion.svg",
            "asset": "ch13/ch13-cacheblend-non-prefix-kv-fusion.svg",
            "section": "13.7.1",
            "caption": "RAG 多文档任意顺序拼接下的非前缀 KV 复用：直接拼接缺失交叉注意力 vs. CacheBlend 仅对 10%–15% 高偏差词元执行稀疏重算并与 SSD 拉取流水重叠",
            "source_refs": ["manuscripts/ch13/figure-data.json"],
            "TeachingRole": "展示如何突破严格前缀匹配限制，将存储池 KV 复用推广至模块化 RAG 文档块"
        },
        {
            "figure_id": "图 13-16",
            "number": 16,
            "file": "ch13-ssd-dwpd-endurance-and-admission.svg",
            "asset": "ch13/ch13-ssd-dwpd-endurance-and-admission.svg",
            "section": "13.7.2",
            "caption": "算例 13-7：8 卡节点持续以 4 GB/s 换出 KV 状态时高达 33.75 DWPD 的闪存寿命超限风险，以及通过写准入过滤器（α≤8.89%）与 ZNS 分区将寿命拉回 ≤3 DWPD 安全线的机制",
            "source_refs": ["manuscripts/ch13/figure-data.json"],
            "TeachingRole": "揭示高频 KV 换出对企业级 SSD 擦写寿命的毁灭性冲击及准入节流设计准则"
        },
        {
            "figure_id": "图 13-17",
            "number": 17,
            "file": "ch13-storage-diagnosis-workflow.svg",
            "asset": "ch13/ch13-storage-diagnosis-workflow.svg",
            "section": "13.7.2",
            "caption": "AI 存储子系统性能与寿命瓶颈的五步定量诊断工作流",
            "source_refs": ["manuscripts/ch13/figure-data.json"],
            "TeachingRole": "为系统工程师提供从元数据、PCIe 拓扑、闪存粒度、GC 毛刺到异构解码的排障流程"
        },
        {
            "figure_id": "图 13-18",
            "number": 18,
            "file": "ch13-full-stack-ai-data-plane.svg",
            "asset": "ch13/ch13-full-stack-ai-data-plane.svg",
            "section": "13.7.2",
            "caption": "贯穿预训练数据湖、RAG 向量知识库、多级 KV 缓存池与万级容器镜像的 AI 原生数据底座全景架构图",
            "source_refs": ["manuscripts/ch13/figure-data.json"],
            "TeachingRole": "收束第 13 章全章知识体系，呈现计算—网络—存储深度融合的现代 AI 数据平面"
        },
        {
            "figure_id": "图 13-19",
            "number": 19,
            "file": "ch13-falconfs-server-side-path-resolution.svg",
            "asset": "ch13/ch13-falconfs-server-side-path-resolution.svg",
            "section": "13.8.1",
            "caption": "传统 POSIX 逐跳目录解析（D=6 时每次打开触发 6–7 次 RPC）与 FalconFS（NSDI '26）服务端复合路径解析及混合哈希无锁目录布局对比",
            "source_refs": ["research/2026-infra-survey/reading-nsdi-2026.md"],
            "TeachingRole": "展示如何将路径解析下推至服务端以消除十亿级多模态小文件训练中的元数据网络往返放大"
        },
        {
            "figure_id": "图 13-20",
            "number": 20,
            "file": "ch13-multimodal-tar-webdataset-vs-lance-random.svg",
            "asset": "ch13/ch13-multimodal-tar-webdataset-vs-lance-random.svg",
            "section": "13.8.1",
            "caption": "静态顺序 Shuffle（WebDataset Tar 连续分片 / FalconFS）与课程学习及在线难例重采样（Lance O(1) 偏移点查 + Umap 页映射）的磁盘寻道与读放大对比",
            "source_refs": ["manuscripts/ch13/figure-data.json"],
            "TeachingRole": "阐明训练数据访问模式从静态顺序遍历转向动态按长度/奖励抽样时的存储格式选型边界"
        },
        {
            "figure_id": "图 13-21",
            "number": 21,
            "file": "ch13-distvs-helmsman-disaggregated-vector-search.svg",
            "asset": "ch13/ch13-distvs-helmsman-disaggregated-vector-search.svg",
            "section": "13.8.2",
            "caption": "DistVS（NSDI '26）计算—内存解耦三级精度筛选（本地 DRAM 粗筛 → 远端内存精筛 → SSD 全精度重排）与 Helmsman（OSDI '26）大规模倒排聚类闪存 ANNS 架构对比",
            "source_refs": ["research/2026-infra-survey/reading-nsdi-2026.md", "research/2026-infra-survey/reading-osdi-2026.md"],
            "TeachingRole": "对比超大规模向量检索跨越 DRAM、CXL/远端内存与 NVMe SSD 时的多级筛选与连续块布局，并明确区分有损向量召回与无损 KV 分层"
        },
        {
            "figure_id": "图 13-22",
            "number": 22,
            "file": "ch13-copilotio-gds-vs-cpu-assisted-io.svg",
            "asset": "ch13/ch13-copilotio-gds-vs-cpu-assisted-io.svg",
            "section": "13.8.3",
            "caption": "大块连续裸张量 GPUDirect Storage（GDS）直通与细粒度/需解压场景下 CoPilotIO（OSDI '26）利用主机 CPU 协处理轮询释放 GPU SM 算力的架构对比",
            "source_refs": ["research/2026-infra-survey/reading-osdi-2026.md"],
            "TeachingRole": "揭示纯 GPU 发起细粒度 I/O 时的寄存器与 SM 争用退化，建立按 I/O 粒度与计算耦合度划分 GDS 与 CPU 协处理的边界"
        },
        {
            "figure_id": "图 13-23",
            "number": 23,
            "file": "ch13-umap-moonbright-mmap-and-gpu-page-table.svg",
            "asset": "ch13/ch13-umap-moonbright-mmap-and-gpu-page-table.svg",
            "section": "13.8.3",
            "caption": "Umap（OSDI '26）分布式文件系统页粒度矩阵 mmap 按需预取免除容器 OOM 与 MoonBright（OSDI '26）设备端 GPU 页表物化及延迟 TLB 同步协同架构",
            "source_refs": ["research/2026-infra-survey/reading-osdi-2026.md"],
            "TeachingRole": "剖析主机侧 DFS 内存映射与设备侧 GPU 页表虚拟化如何消除双重缓冲拷贝与页表同步停顿"
        },
        {
            "figure_id": "图 13-24",
            "number": 24,
            "file": "ch13-ai-storage-bandwidth-latency-iops-cube.svg",
            "asset": "ch13/ch13-ai-storage-bandwidth-latency-iops-cube.svg",
            "section": "13.8.3",
            "caption": "AI 原生存储底座全场景（检查点、十亿小文件、多级 KV、十亿向量 ANNS、细粒度协同 I/O）在带宽—延迟—IOPS 三维空间的映射总览",
            "source_refs": ["manuscripts/ch13/figure-data.json"],
            "TeachingRole": "总结第 13 章全部存储负载特征与对应系统原语的工程选型总表"
        },
        {
            "figure_id": "图 13-25",
            "number": 25,
            "file": "ch13-nixl-unified-nonblocking-transfer-plane.svg",
            "asset": "ch13/ch13-nixl-unified-nonblocking-transfer-plane.svg",
            "section": "13.8.4",
            "caption": "NIXL 跨 HBM、Host DRAM、本地 NVMe 与远端 RDMA 存储池的统一异步传输抽象与路径自动协商",
            "source_refs": ["manuscripts/ch13/figure-data.json"],
            "TeachingRole": "展示 NVIDIA NIXL 如何统一封装多介质零拷贝异步描述符并消减多级 KV 换入换出尾延迟"
        },
        {
            "figure_id": "图 13-26",
            "number": 26,
            "file": "ch13-checkpoint-replication-to-erasure-coding-lifecycle.svg",
            "asset": "ch13/ch13-checkpoint-replication-to-erasure-coding-lifecycle.svg",
            "section": "13.8.4",
            "caption": "PB 级检查点从热层 3 副本秒级闪存暂存到温冷层 RS(10,4) 纠删码对象归档的容量与恢复带宽瀑布图",
            "source_refs": ["manuscripts/ch13/figure-data.json"],
            "TeachingRole": "建立十万卡训练集群检查点热层 3 副本与温冷层 RS(10,4) 纠删码的分层生命周期归档账本"
        }
    ]
    (HERE / 'figure-index.json').write_text(json.dumps(fig_index, ensure_ascii=False, indent=2) + '\n')

    md = HERE.parent / '13-AI 原生存储系统与数据底座.md'
    if md.exists():
        text_md = md.read_text()
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
            + ',errors:e}));if(e.length)process.exit(1);'
        )
        res = subprocess.run(
            ['node', '-e', js, str(katex_js)],
            input=json.dumps(formulas),
            text=True,
            capture_output=True,
            cwd=ROOT,
            check=True,
        )
        (HERE / 'math-validation.json').write_text(res.stdout)

        import markdown

        def repl_img(m):
            alt, rel = m.group(1), m.group(2)
            p = (md.parent / rel).with_suffix('.png')
            b64 = base64.b64encode(p.read_bytes()).decode()
            return f'<figure><img alt="{alt}" src="data:image/png;base64,{b64}"><figcaption>{alt}</figcaption></figure>'

        work = re.sub(r'!\[([^\]]*)\]\((ch13/[^)]+)\)', repl_img, text_md)
        body = markdown.markdown(work, extensions=['tables', 'fenced_code', 'footnotes', 'toc'])
        html = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>第 13 章：AI 原生存储系统与数据底座</title></head><body>{body}</body></html>\n'
        out_html = preview_path(md)
        out_html.write_text(html)

    sources = [HERE / 'build.py', HERE / 'figure-data.json', HERE / 'figure-index.json']
    if md.exists():
        sources.append(md)
    (HERE / 'sources.json').write_text(
        json.dumps(
            {"sources": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in sources]},
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
            {"outputs": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in all_outputs]},
            ensure_ascii=False,
            indent=2,
        )
        + '\n'
    )
    print(f'Built {len(outputs)} figure files for Chapter 13; 0 layout warnings.')


if __name__ == '__main__':
    main()
