#!/usr/bin/env python3
"""Verify Chapter 13 structure, figure layout, mathematical formulas, and numerical examples."""
from pathlib import Path
from urllib.parse import urlsplit, unquote
import hashlib
import json
import math
import re
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
from preview_output import preview_path

checks = []


def check(name: str, value: bool):
    checks.append({"check": name, "passed": bool(value)})


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def sha256(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b'version https://git-lfs.github.com/spec/v1'):
        m = re.search(rb'oid sha256:([0-9a-f]{64})', raw)
        if m:
            return m.group(1).decode('ascii')
    return hashlib.sha256(raw).hexdigest()


md = HERE.parent / '13-AI 原生存储系统与数据底座.md'
s = md.read_text()
page = preview_path(md).read_text()

expected_headings = [
    '13.1', '13.1.1', '13.1.2',
    '13.2', '13.2.1', '13.2.2', '13.2.3',
    '13.3', '13.3.1', '13.3.2', '13.3.3',
    '13.4', '13.4.1', '13.4.2', '13.4.3',
    '13.5', '13.5.1', '13.5.2', '13.5.3',
    '13.6', '13.6.1', '13.6.2',
    '13.7', '13.7.1', '13.7.2',
    '13.8', '13.8.1', '13.8.2', '13.8.3', '13.8.4',
]
found_headings = re.findall(r'^#{2,3} (13\.\d+(?:\.\d+)?) ', s, re.M)
check('all eight sections and twenty-two subsections match outline', found_headings == expected_headings)

h2s = re.findall(r'^## (.+)$', s, re.M)
check('last H2 is 本章小结', len(h2s) >= 7 and h2s[-1].strip() == '本章小结')

exercises = re.findall(r'^>\s*\*\*(?:练习|实验) 13-(\d+)[：:]', s, re.M)
check('seventeen exercises 13-1 to 13-17 in order', exercises == [str(i) for i in range(1, 18)])

code_blocks = re.findall(r'```python\n.*?```', s, re.S)
check('three production Python code blocks', len(code_blocks) == 3)

worked_examples = re.findall(r'^>\s*\*\*算例\s*13-', s, re.M)
check('fourteen standardized blockquote worked examples', len(worked_examples) == 14)

table_captions = re.findall(r'^\*表 13-(\d+)　', s, re.M)
check('ten standardized italic table captions 13-1 to 13-10', table_captions == [str(i) for i in range(1, 11)])

fig_index = json.loads((HERE / 'figure-index.json').read_text())
fig_count = len(fig_index)
check('twenty-six figures in figure-index.json', fig_count == 26)
captions = re.findall(r'^\*图 13-(\d+)(?:：|　)', s, re.M)
check('sequential external captions 13-1 to 13-26', captions == [str(i) for i in range(1, fig_count + 1)])

imgs = re.findall(r'!\[[^\]]*\]\((ch13/[^)]+)\)', s)
check('active illustrations match figure-index.json', imgs == [row.get('asset') or f"ch13/{row['file']}" for row in fig_index])

for rel in imgs:
    p = md.parent / rel
    text_svg = ''.join(ET.parse(p).getroot().itertext())
    check('no internal figure numbering: ' + p.stem, re.search(r'图\s*\d+\s*[-－–]\s*\d+', text_svg) is None)
    check('SVG PNG PDF available: ' + p.stem, all(p.with_suffix(e).exists() for e in ['.svg', '.png', '.pdf']))
    check('PDF is binary not LFS pointer: ' + p.stem, p.with_suffix('.pdf').read_bytes().startswith(b'%PDF'))

missing = []
for p in [md, HERE / 'README.md', HERE / 'reading-notes.md']:
    for link in re.findall(r'\]\(([^)]+)\)', p.read_text()):
        u = urlsplit(link)
        if not u.scheme and u.path and not (p.parent / unquote(u.path)).exists():
            missing.append((str(p), link))
check('local links exist', not missing)

defs = re.findall(r'^\[\^([^\]]+)\]:', s, re.M)
refs = re.findall(r'\[\^([^\]]+)\](?!:)', s)
check('all footnotes used and defined once', set(defs) == set(refs) and len(defs) == len(set(defs)) and len(defs) >= 8)

for name, key in [('sources.json', 'sources'), ('manifest.json', 'outputs')]:
    data = json.loads((HERE / name).read_text())[key]
    mismatches = [row['path'] for row in data if sha256(ROOT / row['path']) != row['sha256']]
    check(name + ' integrity', not mismatches)

math_val = json.loads((HERE / 'math-validation.json').read_text())
check('no formula errors', not math_val['errors'] and 'katex-error' not in page)

layout_val = json.loads((HERE / 'teaching-layout-validation.json').read_text())
check(
    'book-size figure layout',
    len(layout_val) == 26 and all(r['width_pt'] == 420 and r['min_label_pt'] >= 11 and not r['text_extent_warnings'] for r in layout_val),
)

d = json.loads((HERE / 'figure-data.json').read_text())
e1 = d['example_13_1_gds']
check(
    'Example 13-1 GDS arithmetic',
    close(e1['weight_size_gb'] / e1['posix_eff_bw_gbps'], e1['posix_load_seconds'])
    and close(e1['weight_size_gb'] / e1['gds_eff_bw_gbps'], e1['gds_load_seconds'])
    and close(e1['posix_load_seconds'] / e1['gds_load_seconds'], e1['speedup']),
)
e2 = d['example_13_2_3fs_craq']
check(
    'Example 13-2 3FS arithmetic',
    close(e2['ssds_per_node'] * e2['ssd_read_gb_s'], e2['node_disk_raw_gb_s'])
    and close(e2['node_disk_raw_gb_s'] / e2['node_net_line_gb_s'], e2['disk_to_net_ratio'])
    and close(round(e2['nodes'] * e2['node_net_eff_gib_s'] / 1000, 1), e2['cluster_read_tib_s']),
)
e3 = d['example_13_3_kv_layout']
check(
    'Example 13-3 KV layout arithmetic',
    e3['layers'] * e3['bytes_per_layer_per_block'] == e3['valid_bytes_per_block']
    and close(e3['lba_min_bytes'] / e3['bytes_per_layer_per_block'], e3['layer_major_raf_lba'])
    and close(e3['layer_major_ios'] / e3['ssd_max_iops'] * 1000, e3['layer_major_ms'])
    and close(e3['total_valid_mib'] / (e3['ssd_eff_bw_gib_s'] * 1024) * 1000, e3['block_major_ms'])
    and close(e3['layer_major_ms'] / e3['block_major_ms'], e3['speedup']),
)
e4 = d['example_13_4_multimodal_dataloader']
check(
    'Example 13-4 multimodal decoding arithmetic',
    e4['clips_per_sec'] * e4['frames_per_clip'] == e4['total_fps']
    and e4['total_fps'] // e4['cpu_fps_per_core'] == e4['required_cpu_cores_soft']
    and e4['host_total_cores'] - e4['host_reserved_cores'] == e4['host_avail_decode_cores']
    and close(e4['host_avail_decode_cores'] / e4['required_cpu_cores_soft'], e4['cpu_only_supply_ratio'])
    and e4['gpu_count'] * e4['nvdec_fps_per_gpu'] == e4['nvdec_total_fps'] > e4['total_fps'],
)

e11 = d['example_13_11_nixl_pipeline']
check(
    'example 13-11 NIXL multi-tier parallel transfer arithmetic',
    close(sum(e11['tier_volumes_gb']), e11['total_kv_gb'])
    and close(e11['serial_fetch_ms'], 225.10)
    and close(e11['nixl_parallel_ms'], 80.20)
    and close(round(e11['serial_fetch_ms'] / e11['nixl_parallel_ms'], 2), e11['speedup']),
)

e12 = d['example_13_12_checkpoint_ec']
check(
    'example 13-12 checkpoint RS(10,4) erasure coding arithmetic',
    close(e12['total_ckpts'] * e12['ckpt_raw_tb'] * e12['rep_factor'], e12['pure_3rep_tb'])
    and close(
        e12['hot_keep_count'] * e12['ckpt_raw_tb'] * e12['rep_factor']
        + e12['cold_count'] * e12['ckpt_raw_tb'] * e12['ec_factor'],
        e12['tiered_ec_tb'],
    )
    and close(e12['pure_3rep_tb'] - e12['tiered_ec_tb'], e12['saved_tb'])
    and close(round(e12['saved_tb'] / e12['pure_3rep_tb'] * 100.0, 2), e12['saving_pct']),
)

check('zero exclamation marks', '！' not in s)

result = {
    "passed": all(x['passed'] for x in checks),
    "checks": checks,
    "missing_links": missing,
}
(HERE / 'validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(
    json.dumps(
        {
            "passed": result['passed'],
            "checks": len(checks),
            "failures": [x for x in checks if not x['passed']],
        },
        ensure_ascii=False,
        indent=2,
    )
)
raise SystemExit(0 if result['passed'] else 1)
