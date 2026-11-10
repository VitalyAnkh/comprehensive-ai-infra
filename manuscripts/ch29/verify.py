#!/usr/bin/env python3
"""Verify Chapter 29 structure, figure layout, mathematical formulas, and numerical examples (64 checks)."""
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


md = HERE.parent / '29-DeepSeek 开源基础设施全栈解剖与极客工程实战：FlashMLA、DeepGEMM、DeepEP、DualPipe、3FS、Mooncake 与 KTransformers.md'
s = md.read_text()
page = preview_path(md).read_text()

expected_headings = [
    '29.1', '29.1.1', '29.1.2',
    '29.2', '29.2.1', '29.2.2',
    '29.3', '29.3.1', '29.3.2',
    '29.4', '29.4.1', '29.4.2',
]
found_headings = re.findall(r'^#{2,3} (29\.\d+(?:\.\d+)?) ', s, re.M)
check('all four sections and eight subsections match outline', found_headings == expected_headings)

h2s = re.findall(r'^## (.+)$', s, re.M)
check('last H2 is 本章小结', len(h2s) >= 6 and h2s[-1].strip() == '本章小结')

exercises = re.findall(r'^(?:>\s*)?\*\*(?:练习|实验) 29-(\d+)[：:（]', s, re.M)
check('ten exercises 29-1 to 29-10 in order', exercises == [str(i) for i in range(1, 11)])

code_blocks = re.findall(r'```python\n.*?```', s, re.S)
check('six production python code blocks', len(code_blocks) == 6 and len(re.findall(r'```', s)) == 12)

worked_examples = re.findall(r'^>\s*\*\*算例\s*29-', s, re.M)
check('six standardized blockquote worked examples', len(worked_examples) == 6)

table_captions = re.findall(r'^\*表 29-(\d+)　', s, re.M)
check('six standardized italic table captions 29-1 to 29-6', table_captions == [str(i) for i in range(1, 7)])

fig_index = json.loads((HERE / 'figure-index.json').read_text())
fig_count = len(fig_index)
check('sixteen figures in figure-index.json', fig_count == 16)
captions = re.findall(r'^\*图 29-(\d+)(?:：|　)', s, re.M)
check('sequential captions 29-1 to 29-16', captions == [str(i) for i in range(1, fig_count + 1)])

imgs = re.findall(r'!\[.+?\]\((ch29/[^)]+)\)', s)
check(
    'sixteen active illustrations match figure-index.json',
    imgs == [row.get('asset') or f"ch29/{row['file']}" for row in fig_index] and len(imgs) == 16,
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
    'locked FlashMLA, DeepGEMM, DeepEP/DualPipe, and 3FS/KTransformers metrics',
    close(d['flashmla']['kv_compression_vs_mha64_x'], 28.44)
    and close(d['flashmla']['achieved_bw_gbs'], 3000.0)
    and close(d['flashmla']['bw_utilization_pct'], 89.55)
    and close(d['deepgemm']['jit_regs_per_thread'], 192)
    and close(d['deepgemm']['ffma_cycle_overhead_pct'], 3.12)
    and close(d['deepgemm']['achieved_fp8_tflops'], 1358.0)
    and close(d['deepep_and_dualpipe']['normal_mode_bw_gbs'], 153.0)
    and close(d['deepep_and_dualpipe']['low_latency_dispatch_us'], 163.0)
    and close(d['deepep_and_dualpipe']['exposed_alltoall_stall_pct'], 0.0)
    and close(d['three_fs_and_ktransformers']['aggregate_read_bw_tbs'], 6.6)
    and close(d['three_fs_and_ktransformers']['decode_tok_per_sec'], 18.6)
    and close(d['three_fs_and_ktransformers']['speedup_vs_llamacpp_x'], 13.8),
)

passed = sum(1 for c in checks if c['passed'])
print(json.dumps({"chapter": "ch29", "passed": passed, "total": len(checks), "checks": checks}, ensure_ascii=False, indent=2))
if passed != len(checks):
    sys.exit(1)
