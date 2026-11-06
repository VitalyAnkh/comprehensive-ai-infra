#!/usr/bin/env python3
"""Build Chapter 14 figures, locked co-design calculations, and offline HTML preview."""
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
        "example_14_1_v41_dual_track": {
            "mla_40layer_b_per_tok": 23040,
            "v4_global_b_per_tok": 3598.72,
            "v4_persistent_b_per_tok": 3514.25,
            "v41_global_b_per_tok": 890.0,
            "32k_mla_mib": 720.0,
            "32k_v4_global_mib": 112.46,
            "32k_v4_total_mib": 115.52,
            "32k_v41_global_mib": 27.8125,
            "32k_v41_total_mib": 30.875,
            "1m_v4_resident_gib": 3.351,
            "1m_v41_resident_gib": 0.849,
            "resident_ratio": 3.95,
            "8k_v4_read_mib": 25.06,
            "8k_v41_read_mib": 22.84,
            "8k_read_ratio": 1.10,
            "1m_v4_read_gib": 3.679,
            "1m_v41_read_gib": 1.466,
            "1m_read_ratio": 2.51,
            "persistent_cache_reduction": 8
        },
        "example_14_2_digital_worker": {
            "v41_state_200k_gib": 1.30,
            "v41_state_1m_gib": 6.42,
            "k3_state_200k_gib": 5.57,
            "k3_state_1m_gib": 26.17,
            "v41_200k_b200_usd_per_m": 5.97,
            "v41_200k_api_usd_per_m": 4.69,
            "v41_1m_b200_usd_per_m": 13.34,
            "v41_1m_api_usd_per_m": 6.41,
            "k3_200k_b300_usd_per_m": 145.13,
            "k3_200k_api_usd_per_m": 65.91,
            "k3_1m_b200_usd_per_m": 496.08,
            "k3_1m_api_usd_per_m": 125.19,
            "k3_fixed_layer_overhead_ms": 13.95,
            "k3_rebuild_200k_ms": 9.18,
            "k3_rebuild_1m_ms": 9.26
        },
        "example_14_3_opentallas": {
            "target_tokens_per_s": 8050,
            "step_budget_us": 124.221,
            "rom_weight_read_us": 7.828,
            "hbm_kv_read_us": 41.979,
            "tensor_compute_us": 25.932,
            "serial_comm_us": 69.821,
            "unpriced_aux_budget_us": 14.568,
            "wafer_hops": 15,
            "hop_ns": 100,
            "layers": 120,
            "wafer_comm_floor_us": 180.0,
            "incremental_nre_usd": 20000000,
            "breakeven_tokens": 40000000000000,
            "machines_1yr": 254,
            "machines_6mo": 508
        },
        "example_14_4_counterfactual": {
            "swa_layers": 40,
            "swa_window_tokens": 128,
            "swa_state_mib": 2.578,
            "candidate_pool_tokens": 16384,
            "reindex_k": 2048
        }
    }


def fig_14_1_simultaneous(ex: Exporter):
    fig, ax = canvas(3.80)
    card(ax, 0.02, 0.50, 0.96, 0.47, '硬件物理边界内化为模型架构正则项的跨层耦合', color='gray')
    hw = [
        (0.04, 0.69, 0.44, 0.16, '微缩块量化（MXFP4 32元素）\n约束激活离群值与谱范数', 'blue'),
        (0.52, 0.69, 0.44, 0.16, 'SFU 与 Tensor Core 配比\n约束 Softmax 频次与能耗', 'teal'),
        (0.04, 0.52, 0.44, 0.14, 'HBM 容量与 SSD 块粒度\n约束持久 KV 与跨层共享', 'purple'),
        (0.52, 0.52, 0.44, 0.14, '超节点互连半径与跳数\n约束专家度与状态重放', 'orange'),
    ]
    for x, y, w, h, label, role in hw:
        box(ax, x, y, w, h, label, role, size=11)

    card(ax, 0.02, 0.03, 0.96, 0.43, '协同设计的四大定量杠杆（Four Co-Design Levers）', color='blue')
    levers = [
        (0.04, 0.20, 0.44, 0.14, '① 数值分布整形\nRoPE-post 与正交旋转', 'blue'),
        (0.52, 0.20, 0.44, 0.14, '② 跨层状态锚点化\nCSA2 4-Owner 共享 KV', 'teal'),
        (0.04, 0.05, 0.44, 0.12, '③ 有界重算代传输\n128-token SWA 局部重放', 'purple'),
        (0.52, 0.05, 0.44, 0.12, '④ 前向计算重分布\nCED 早期丢弃与深编码', 'orange'),
    ]
    for x, y, w, h, label, role in levers:
        box(ax, x, y, w, h, label, role, size=11)
    ex.save(fig, 'ch14-codesign-simultaneous-equations')


def fig_14_2_dual_track(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.04, 0.46, 0.92, '轨道一：驻留容量（恒定 3.95×）', color='teal', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, 'V4 Flash（40 层独立 KV）\n3514.25 B/tok · 32K 112.46M', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, 'V4.1 Flash（4-Owner 共享）\n890 B/tok · 32K 27.8125M', 'teal', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '1M 上下文驻留容量对比\n3.351 GiB → 0.849 GiB', 'blue', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '压缩比 3.95× 与长度 H 无关', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])

    card(ax, 0.52, 0.04, 0.46, 0.92, '轨道二：单步读取（1.10→2.51×）', color='blue', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '8K 短上下文：Top-k 主导\n25.06 → 22.84 MiB (1.10×)', 'purple', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '128K–200K：索引占比抬升\n36 追随层免扫索引显效', 'teal', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '1M 超长上下文：索引主导\n3.679 → 1.466 GiB (2.51×)', 'blue', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '随 H 拉长逼近渐近上限 ~2.60×', size=11, ha='center', weight='bold', color=BORDER_COL['blue'])
    ex.save(fig, 'ch14-v41-dual-track-context-scaling')


def fig_14_3_four_stage(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'V4.1 Flash 四阶段全局注意力流水线（4 Owner + 36 Follower）', color='purple')
    stages = [
        (0.04, 0.58, 0.20, 0.24, '① Full 建库\n4 Owner 层\n写 KV+索引', 'blue'),
        (0.28, 0.58, 0.21, 0.24, '② 16K 候选池\nOwner 粗筛\n共享候选集', 'purple'),
        (0.53, 0.58, 0.20, 0.24, '③ Reindex\n16K 池内精排\n选 2,048 块', 'teal'),
        (0.77, 0.58, 0.19, 0.24, '④ Reuse\n追随层直取\n2,048 索引', 'orange'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.24, 0.70), (0.28, 0.70))
    arrow(ax, (0.49, 0.70), (0.53, 0.70))
    arrow(ax, (0.73, 0.70), (0.77, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.43, '持久化前缀缓存压缩至 1/8 的正交拆解（1/2 × 1/4 = 1/8）', color='teal')
    box(ax, 0.04, 0.16, 0.43, 0.19, '因子一 (1/2)：免存 40 层 SWA\n128-token 重算代 2.578 MiB 块', 'blue', size=11)
    box(ax, 0.53, 0.16, 0.43, 0.19, '因子二 (1/4)：CSA2 跨层共享\n全局 KV 3514.25→890 B/tok', 'teal', size=11)
    arrow(ax, (0.47, 0.255), (0.53, 0.255))
    text(ax, 0.50, 0.08, '两项正交机制叠加，使持久化前缀缓存体积降至原架构的约 1/8', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch14-v41-four-stage-attention-pipeline')


def fig_14_4_digital_worker(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.50, 0.96, 0.46, '算例 14-2：24×7 数字员工（≥30 tok/s）未优化基线成本对比', color='blue')
    box(ax, 0.04, 0.56, 0.44, 0.25, 'V4.1 Flash（1.30→6.42 GiB）\n200K：B200 5.97 vs API 4.69\n1M：B200 13.34 vs API 6.41 ($/M)', 'teal', size=11)
    box(ax, 0.52, 0.56, 0.44, 0.25, 'Kimi K3 基线（5.57→26.17 GiB）\n200K：B300 145.13 vs 65.91\n1M：B200 496.08 vs 125.19 ($/M)', 'orange', size=11)

    card(ax, 0.02, 0.03, 0.96, 0.43, '性能归因审计（OPTIMIZATION-AUDIT）：区分架构代价与运行时伪影', color='orange')
    box(ax, 0.04, 0.16, 0.44, 0.19, '伪影一：逐层固定调度 150 μs\n93 层累加 13.95 ms/tok', 'orange', size=11)
    box(ax, 0.52, 0.16, 0.44, 0.19, '伪影二：10% 重建 + 漏算 DSpark\n摊入 9.18–9.26 ms/tok（占七成）', 'purple', size=11)
    text(ax, 0.50, 0.08, '审计结论：未启用 MTP/DSpark 与算子融合的基线不能外推为架构劣势', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch14-digital-worker-v41-vs-k3-cost')


def fig_14_5_opentallas(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.50, 0.96, 0.46, '算例 14-3（Case B）：OpenTallas 单步 124.221 μs 四资源账本', color='teal')
    res_boxes = [
        (0.04, 0.58, 0.21, 0.24, 'ROM 权重读\n7.828 μs\n并行隐藏', 'teal'),
        (0.28, 0.58, 0.21, 0.24, 'HBM KV 读\n41.979 μs\n记忆主导', 'blue'),
        (0.52, 0.58, 0.21, 0.24, 'Tensor 计算\n25.932 μs\n重叠隐藏', 'purple'),
        (0.76, 0.58, 0.20, 0.24, '通信+余量\n69.82+14.57\n=84.389 μs', 'orange'),
    ]
    for x, y, w, h, label, role in res_boxes:
        box(ax, x, y, w, h, label, role, size=11)

    card(ax, 0.02, 0.03, 0.96, 0.43, 'Case D 晶圆级跳数延迟越界与 Case E/F 2000 万美元 NRE 回本边界', color='orange')
    box(ax, 0.04, 0.15, 0.44, 0.20, '晶圆级扩展直径反超（Case D）\n15 跳 × 100 ns × 120 层 = 180 μs\n单凭通信即击穿 124.221 μs 预算', 'orange', size=11)
    box(ax, 0.52, 0.15, 0.44, 0.20, '2000 万美元 NRE 回本（Case F）\n需产 40 万亿 tok（省 $0.50/M）\n1 年需 254 台 · 半年需 508 台', 'blue', size=11)
    text(ax, 0.50, 0.08, '专用 ASIC 必须在物理跳数延迟与模型半衰期之间同时闭合方程', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch14-opentallas-four-resource-and-wafer-hop')


def fig_14_6_decision_tree(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.03, 0.96, 0.94, '全书总结：AI Infra 架构师的五步定量诊断与反事实协同决策树', color='blue')
    steps = [
        (0.05, 0.72, 0.42, 0.14, '第一步：锁定量纲与负载边界\n区分十进制/二进制 · 明确 SLO', 'blue'),
        (0.53, 0.72, 0.42, 0.14, '第二步：建立四资源下界方程\n计算 / 访存 / 通信 / 存储 I/O', 'teal'),
        (0.53, 0.50, 0.42, 0.15, '第三步：定位瓶颈与翻转点\nRoofline · 利特尔 · Amdahl', 'purple'),
        (0.05, 0.50, 0.42, 0.15, '第四步：剥离架构与软件伪影\n审计逐层固定开销与双重拷贝', 'orange'),
        (0.05, 0.26, 0.90, 0.16, '第五步：跨层反事实推演与 TCO 均衡（Counterfactual Co-Design）\n当 HBM 免费 / 互连翻倍 / 半衰期减半时，重验算法—系统—硬件最优解', 'teal'),
    ]
    for x, y, w, h, label, role in steps:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.47, 0.79), (0.53, 0.79))
    arrow(ax, (0.74, 0.72), (0.74, 0.65))
    arrow(ax, (0.53, 0.575), (0.47, 0.575))
    arrow(ax, (0.26, 0.50), (0.26, 0.42))
    box(ax, 0.05, 0.08, 0.90, 0.12, '终极准则：以物理下界指导架构取舍，让每焦耳能量转化为有效智能', 'blue', size=11)
    ex.save(fig, 'ch14-five-step-diagnostic-decision-tree')


def main():
    ex = Exporter(HERE)
    fig_14_1_simultaneous(ex)
    fig_14_2_dual_track(ex)
    fig_14_3_four_stage(ex)
    fig_14_4_digital_worker(ex)
    fig_14_5_opentallas(ex)
    fig_14_6_decision_tree(ex)
    outputs = ex.finish()

    fig_data = build_figure_data()
    (HERE / 'figure-data.json').write_text(json.dumps(fig_data, ensure_ascii=False, indent=2) + '\n')

    fig_index = [
        {
            "figure_id": "图 14-1",
            "number": 1,
            "file": "ch14-codesign-simultaneous-equations.svg",
            "asset": "ch14/ch14-codesign-simultaneous-equations.svg",
            "section": "14.1.2",
            "caption": "模型—系统—硬件协同设计的跨层耦合约束（MXFP4 块粒度、SFU 配比、HBM 容量与超节点半径）及四大定量设计杠杆",
            "source_refs": ["manuscripts/ch14/figure-data.json"],
            "TeachingRole": "建立第 14 章跨层联立方程组与四大协同设计杠杆的总览框架"
        },
        {
            "figure_id": "图 14-2",
            "number": 2,
            "file": "ch14-v41-dual-track-context-scaling.svg",
            "asset": "ch14/ch14-v41-dual-track-context-scaling.svg",
            "section": "14.2.1",
            "caption": "算例 14-1：在 8K 至 1M 上下文扩展过程中，DeepSeek V4 Flash 与 V4.1 Flash 的驻留 KV 容量（恒定缩小 3.95x）与单步 Decode 理论读取量（从 1.10x 演进至 2.51x）的双轨分离规律",
            "source_refs": ["manuscripts/ch14/figure-data.json"],
            "TeachingRole": "直观区分驻留显存容量压缩比与单步注意力读取带宽压缩比的不同演进轨迹"
        },
        {
            "figure_id": "图 14-3",
            "number": 3,
            "file": "ch14-v41-four-stage-attention-pipeline.svg",
            "asset": "ch14/ch14-v41-four-stage-attention-pipeline.svg",
            "section": "14.2.3",
            "caption": "V4.1 Flash 的四阶段全局注意力执行流水线（Full -> 16,384 候选池 -> Reindex -> Reuse）以及持久化前缀缓存缩减至 1/8（1/2 SWA 免存 × 1/4 CSA2 跨层共享）的正交拆解",
            "source_refs": ["manuscripts/ch14/figure-data.json"],
            "TeachingRole": "剖析 4 个 Owner 层与 36 个 Follower 层如何协同消除重复索引扫描并实现 1/8 持久化缓存"
        },
        {
            "figure_id": "图 14-4",
            "number": 4,
            "file": "ch14-digital-worker-v41-vs-k3-cost.svg",
            "asset": "ch14/ch14-digital-worker-v41-vs-k3-cost.svg",
            "section": "14.3.2",
            "caption": "算例 14-2：24×7 数字员工在 200K/1M 上下文下 V4.1 Flash 与 Kimi K3 的跨代 GPU 选卡及 API 成本对比，以及揭示约七成耗时来自逐层固定开销与频繁重建假设的性能归因审计",
            "source_refs": ["manuscripts/ch14/figure-data.json"],
            "TeachingRole": "演示如何在真实工程评估中剥离未优化运行时伪影与模型架构内生差异"
        },
        {
            "figure_id": "图 14-5",
            "number": 5,
            "file": "ch14-opentallas-four-resource-and-wafer-hop.svg",
            "asset": "ch14/ch14-opentallas-four-resource-and-wafer-hop.svg",
            "section": "14.4.2",
            "caption": "算例 14-3：OpenTallas 专用 ASIC 单步 124.221 μs（8,050 tok/s）的四资源时间账本（Case B）、晶圆级 15 跳 180 μs 通信直径越界（Case D）以及 2,000 万美元 NRE 的 40 万亿 token 回本边界（Case F）",
            "source_refs": ["manuscripts/ch14/figure-data.json"],
            "TeachingRole": "用严密的四资源时间线与跨芯片跳数下界审计专用推理芯片的物理与商业可行性"
        },
        {
            "figure_id": "图 14-6",
            "number": 6,
            "file": "ch14-five-step-diagnostic-decision-tree.svg",
            "asset": "ch14/ch14-five-step-diagnostic-decision-tree.svg",
            "section": "14.5.2",
            "caption": "全书收束：AI 基础设施架构师的五步定量诊断法与跨层反事实协同设计决策树",
            "source_refs": ["manuscripts/ch14/figure-data.json"],
            "TeachingRole": "收束全书十四章知识体系，提供可复用的一线 AI Infra 性能诊断与架构设计闭环"
        }
    ]
    (HERE / 'figure-index.json').write_text(json.dumps(fig_index, ensure_ascii=False, indent=2) + '\n')

    md = HERE.parent / '14-模型与系统硬件协同设计.md'
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

        work = re.sub(r'!\[([^\]]*)\]\((ch14/[^)]+)\)', repl_img, text_md)
        body = markdown.markdown(work, extensions=['tables', 'fenced_code', 'footnotes', 'toc'])
        html = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>第 14 章：模型与系统硬件协同设计</title></head><body>{body}</body></html>\n'
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
    print(f'Built {len(outputs)} figure files for Chapter 14; 0 layout warnings.')


if __name__ == '__main__':
    main()
