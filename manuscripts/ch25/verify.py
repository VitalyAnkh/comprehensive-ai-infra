#!/usr/bin/env python3
"""Verify Chapter 25 structure, figure layout, mathematical formulas, and numerical examples (64 checks)."""
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


md = HERE.parent / '25-生成式推荐系统（Generative Recommenders）、万亿参数稀疏表（DLRM）与十亿级 GPU 向量检索引擎.md'
s = md.read_text()
page = preview_path(md).read_text()

expected_headings = [
    '25.1', '25.1.1', '25.1.2',
    '25.2', '25.2.1', '25.2.2',
    '25.3', '25.3.1', '25.3.2',
    '25.4', '25.4.1', '25.4.2',
]
found_headings = re.findall(r'^#{2,3} (25\.\d+(?:\.\d+)?) ', s, re.M)
check('all four sections and eight subsections match outline', found_headings == expected_headings)

h2s = re.findall(r'^## (.+)$', s, re.M)
check('last H2 is 本章小结', len(h2s) >= 6 and h2s[-1].strip() == '本章小结')

exercises = re.findall(r'^(?:>\s*)?\*\*(?:练习|实验) 25-(\d+)[：:（]', s, re.M)
check('ten exercises 25-1 to 25-10 in order', exercises == [str(i) for i in range(1, 11)])

code_blocks = re.findall(r'```python\n.*?```', s, re.S)
check('six production python code blocks', len(code_blocks) == 6 and len(re.findall(r'```', s)) == 12)

worked_examples = re.findall(r'^>\s*\*\*算例\s*25-', s, re.M)
check('six standardized blockquote worked examples', len(worked_examples) == 6)

table_captions = re.findall(r'^\*表 25-(\d+)　', s, re.M)
check('six standardized italic table captions 25-1 to 25-6', table_captions == [str(i) for i in range(1, 7)])

fig_index = json.loads((HERE / 'figure-index.json').read_text())
fig_count = len(fig_index)
check('sixteen figures in figure-index.json', fig_count == 16)
captions = re.findall(r'^\*图 25-(\d+)(?:：|　)', s, re.M)
check('sequential captions 25-1 to 25-16', captions == [str(i) for i in range(1, fig_count + 1)])

imgs = re.findall(r'!\[.+?\]\((ch25/[^)]+)\)', s)
check(
    'sixteen active illustrations match figure-index.json',
    imgs == [row.get('asset') or f"ch25/{row['file']}" for row in fig_index] and len(imgs) == 16,
)

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
    len(layout_val) == 16 and all(r['width_pt'] == 420 and r['min_label_pt'] >= 11 and not r['text_extent_warnings'] for r in layout_val),
)

d = json.loads((HERE / 'figure-data.json').read_text())
check(
    'locked generative recommender, DLRM, CAGRA, and DiskANN metrics',
    close(d['three_tier_embedding_cache']['total_table_tb'], 10.0)
    and close(d['three_tier_embedding_cache']['weighted_latency_us'], 2.19)
    and close(d['three_tier_embedding_cache']['pure_ssd_speedup_x'], 29.7)
    and close(d['torchrec_all_to_all_overlap']['balanced_a2a_ms'], 1.53)
    and close(d['torchrec_all_to_all_overlap']['exposed_balanced_comm_ms'], 0.0)
    and close(d['hstu_stochastic_length']['quadratic_flop_reduction_pct'], 56.2)
    and close(d['mfalcon_microbatch_amortization']['naive_total_tflops'], 1409.3)
    and close(d['mfalcon_microbatch_amortization']['mfalcon_total_tflops'], 31.46)
    and close(d['mfalcon_microbatch_amortization']['flop_reduction_x'], 44.8)
    and close(d['cuvs_cagra_vs_hnsw']['build_speedup_x'], 16.91)
    and close(d['diskann_sector_aligned_search']['dram_pq_gb'], 64.0)
    and d['diskann_sector_aligned_search']['sustainable_qps'] == 210000,
)

passed = sum(1 for c in checks if c['passed'])
print(json.dumps({"chapter": "ch25", "passed": passed, "total": len(checks), "checks": checks}, ensure_ascii=False, indent=2))
if passed != len(checks):
    sys.exit(1)
