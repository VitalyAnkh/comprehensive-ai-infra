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


def main():
    ex = Exporter(HERE)
    fig_13_1_hierarchy(ex)
    fig_13_2_gds(ex)
    fig_13_3_3fs_craq(ex)
    fig_13_4_kv_layout(ex)
    fig_13_5_zns_gc(ex)
    fig_13_6_multimodal(ex)
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
            "TeachingRole": "展示 PB 级多模态训练如何打通存储分片顺序供给与片上 NVDEC 异构解码闭环"
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
