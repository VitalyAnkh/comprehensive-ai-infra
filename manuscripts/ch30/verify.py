#!/usr/bin/env python3
"""Verify Chapter 30 structure, figure layout, mathematical formulas, and numerical examples (64 checks)."""
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


md = HERE.parent / '30-自演进 AI 基础设施（AI for AI Infra）：大模型驱动的算子自动合成、强化学习编译寻优与三十章全栈终极收束.md'
s = md.read_text()
page = preview_path(md).read_text()

expected_headings = [
    '30.1', '30.1.1', '30.1.2',
    '30.2', '30.2.1', '30.2.2',
    '30.3', '30.3.1', '30.3.2',
    '30.4', '30.4.1', '30.4.2',
]
found_headings = re.findall(r'^#{2,3} (30\.\d+(?:\.\d+)?) ', s, re.M)
check('all four sections and eight subsections match outline', found_headings == expected_headings)

h2s = re.findall(r'^## (.+)$', s, re.M)
check('last H2 is 本章小结', len(h2s) >= 6 and h2s[-1].strip() == '本章小结')

exercises = re.findall(r'^(?:>\s*)?\*\*(?:练习|实验) 30-(\d+)[：:（]', s, re.M)
check('ten exercises 30-1 to 30-10 in order', exercises == [str(i) for i in range(1, 11)])

code_blocks = re.findall(r'```python\n.*?```', s, re.S)
check('six production python code blocks', len(code_blocks) == 6 and len(re.findall(r'```', s)) == 12)

worked_examples = re.findall(r'^>\s*\*\*算例\s*30-', s, re.M)
check('six standardized blockquote worked examples', len(worked_examples) == 6)

table_captions = re.findall(r'^\*表 30-(\d+)　', s, re.M)
check('six standardized italic table captions 30-1 to 30-6', table_captions == [str(i) for i in range(1, 7)])

fig_index = json.loads((HERE / 'figure-index.json').read_text())
fig_count = len(fig_index)
check('sixteen figures in figure-index.json', fig_count == 16)
captions = re.findall(r'^\*图 30-(\d+)(?:：|　)', s, re.M)
check('sequential captions 30-1 to 30-16', captions == [str(i) for i in range(1, fig_count + 1)])

imgs = re.findall(r'!\[.+?\]\((ch30/[^)]+)\)', s)
check(
    'sixteen active illustrations match figure-index.json',
    imgs == [row.get('asset') or f"ch30/{row['file']}" for row in fig_index] and len(imgs) == 16,
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
    'locked KernelBench RLVR, Per-Kernel DVFS, Autonomous AIOps, and Grand 30-Chapter Ledger metrics',
    close(d['kernelbench_rlvr']['rlvr_8turn_correctness_pct'], 94.8)
    and close(d['kernelbench_rlvr']['rlvr_8turn_speedup_vs_eager_x'], 3.42)
    and close(d['kernelbench_rlvr']['rlvr_8turn_speedup_vs_compile_x'], 1.58)
    and close(d['kernelbench_rlvr']['register_fusion_hbm_saved_gbps'], 38.4)
    and close(d['auto_parallel_dvfs']['dynamic_power_saving_pct'], 26.5)
    and close(d['autonomous_aiops_twin']['mttr_compression_x'], 360.0)
    and close(d['autonomous_aiops_twin']['ettr_autonomous_pct'], 97.8)
    and close(d['autonomous_aiops_twin']['six_month_saved_compute_usd_m'], 13.5)
    and d['grand_30chapter_ledger']['total_chapters'] == 30
    and d['grand_30chapter_ledger']['total_figures'] == 980
    and d['grand_30chapter_ledger']['runnable_experiments'] == 165,
)

passed = sum(1 for c in checks if c['passed'])
print(json.dumps({"chapter": "ch30", "passed": passed, "total": len(checks), "checks": checks}, ensure_ascii=False, indent=2))
if passed != len(checks):
    sys.exit(1)
