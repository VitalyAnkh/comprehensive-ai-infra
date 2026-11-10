#!/usr/bin/env python3
"""Verify Chapter 17 structure, figure layout, mathematical formulas, and numerical examples."""
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


md = HERE.parent / '17-多模态、视频生成（DiT）与世界模型基础设施.md'
s = md.read_text()
page = preview_path(md).read_text()

expected_headings = [
    '17.1', '17.1.1', '17.1.2', '17.1.3',
    '17.2', '17.2.1', '17.2.2', '17.2.3',
    '17.3', '17.3.1', '17.3.2', '17.3.3',
    '17.4', '17.4.1', '17.4.2', '17.4.3',
]
found_headings = re.findall(r'^#{2,3} (17\.\d+(?:\.\d+)?) ', s, re.M)
check('all four sections and twelve subsections match outline', found_headings == expected_headings)

h2s = re.findall(r'^## (.+)$', s, re.M)
check('last H2 is 本章小结', len(h2s) >= 6 and h2s[-1].strip() == '本章小结')

exercises = re.findall(r'^>\s*\*\*(?:练习|实验) 17-(\d+)[：:（]', s, re.M)
check('sixteen exercises 17-1 to 17-16 in order', exercises == [str(i) for i in range(1, 17)])

code_blocks = re.findall(r'```python\n.*?```', s, re.S)
check('nine production Python code blocks', len(code_blocks) == 9 and len(re.findall(r'```', s)) == 18)

worked_examples = re.findall(r'^>\s*\*\*算例\s*17-', s, re.M)
check('eight standardized blockquote worked examples', len(worked_examples) == 8)

table_captions = re.findall(r'^\*表 17-(\d+)　', s, re.M)
check('seven standardized italic table captions 17-1 to 17-7', table_captions == [str(i) for i in range(1, 8)])

fig_index = json.loads((HERE / 'figure-index.json').read_text())
fig_count = len(fig_index)
check('twenty-two figures in figure-index.json', fig_count == 22)
captions = re.findall(r'^\*图 17-(\d+)(?:：|　)', s, re.M)
check('sequential captions 17-1 to 17-22', sorted(captions, key=int) == [str(i) for i in range(1, fig_count + 1)])

imgs = re.findall(r'!\[.+?\]\((ch17/[^)]+)\)', s)
check('twenty-two active illustrations match figure-index.json', set(imgs) == {row.get('asset') or f"ch17/{row['file']}" for row in fig_index} and len(imgs) == 22)

for rel in imgs:
    p = md.parent / rel
    text_svg = ''.join(ET.parse(p).getroot().itertext())
    check('no internal figure numbering: ' + p.stem, re.search(r'图\s*\d+\s*[-－–]\s*\d+', text_svg) is None)
    check('SVG PNG PDF available: ' + p.stem, all(p.with_suffix(e).exists() for e in ['.svg', '.png', '.pdf']))
    check('PDF is binary not LFS pointer: ' + p.stem, p.with_suffix('.pdf').read_bytes().startswith(b'%PDF'))

missing = []
for link in re.findall(r'\]\(([^)]+)\)', s):
    u = urlsplit(link)
    if not u.scheme and u.path and not (md.parent / unquote(u.path)).exists():
        missing.append((str(md), link))
check('local links exist', not missing)

defs = re.findall(r'^\[\^([^\]]+)\]:', s, re.M)
refs = re.findall(r'\[\^([^\]]+)\](?!:)', s)
check('all footnotes used and defined once', set(defs) == set(refs) and len(defs) == len(set(defs)) and len(defs) >= 15)

for name, key in [('sources.json', 'sources'), ('manifest.json', 'outputs')]:
    data = json.loads((HERE / name).read_text())[key]
    mismatches = [row['path'] for row in data if sha256(ROOT / row['path']) != row['sha256']]
    check(name + ' integrity', not mismatches)

math_val = json.loads((HERE / 'math-validation.json').read_text())
check('no formula errors', not math_val['errors'] and 'katex-error' not in page)

layout_val = json.loads((HERE / 'teaching-layout-validation.json').read_text())
check(
    'book-size figure layout',
    len(layout_val) == 22 and all(r['width_pt'] == 420 and r['min_label_pt'] >= 11 and not r['text_extent_warnings'] for r in layout_val),
)

d = json.loads((HERE / 'figure-data.json').read_text())
tm = d['transfusion_hybrid_mask']
check(
    'example 17-7 Transfusion hybrid mask and D->G handoff arithmetic',
    tm['text_prefix_tokens'] + tm['image_block_tokens'] + tm['text_suffix_tokens'] == tm['total_tokens']
    and close(round(tm['active_token_pairs'] / tm['full_matrix_pairs'] * 100.0, 2), tm['active_density_pct'])
    and tm['executed_tiles'] == 648
    and close(round(tm['active_token_pairs'] / (tm['executed_tiles'] * 128 * 128) * 100.0, 2), tm['tile_occupancy_pct'])
    and close(tm['kv_handoff_mib'] / 50000.0 * 1000.0, tm['kv_handoff_ms_at_50gbs']),
)

wm = d['hybrid_3dgs_dit_world_model']
check(
    '3DGS + Video DiT world model latency and FPS arithmetic',
    close(round(wm['gs_raster_ms'] + wm['dit_4step_stream_ms'] + wm['tiled_vae_ms'], 2), wm['total_frame_ms'])
    and close(round(1000.0 / wm['total_frame_ms'], 2), wm['steady_fps']),
)

cwm = d['causal_streaming_world_model_20fps']
check(
    'example 17-8 causal streaming world model KV rolling and PAB/TeaCache FPS arithmetic',
    cwm['total_active_chunks'] == cwm['sink_chunks'] + cwm['rolling_window_chunks'] == 6
    and cwm['active_kv_tokens'] == 21600
    and close(cwm['pab_teacache_dit_chunk_ms'] + cwm['causal_vae_chunk_ms'], cwm['total_chunk_ms'])
    and close(cwm['total_chunk_ms'] / cwm['chunk_rgb_frames'], cwm['per_frame_latency_ms'])
    and close(round(1000.0 / cwm['per_frame_latency_ms'], 2), cwm['streaming_fps']),
)

passed = sum(1 for c in checks if c['passed'])
print(json.dumps({"chapter": "ch17", "passed": passed, "total": len(checks), "checks": checks}, ensure_ascii=False, indent=2))
if passed != len(checks):
    sys.exit(1)
