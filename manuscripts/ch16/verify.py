#!/usr/bin/env python3
"""Verify Chapter 16 structure, figure layout, mathematical formulas, and numerical examples."""
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


md = HERE.parent / '16-机密 AI 计算、可信执行环境与安全沙箱底座.md'
s = md.read_text()
page = preview_path(md).read_text()

expected_headings = [
    '16.1', '16.1.1', '16.1.2',
    '16.2', '16.2.1', '16.2.2',
    '16.3', '16.3.1', '16.3.2',
    '16.4', '16.4.1', '16.4.2', '16.4.3',
    '16.5', '16.5.1', '16.5.2',
    '16.6', '16.6.1', '16.6.2', '16.6.3', '16.6.4',
]
found_headings = re.findall(r'^#{2,3} (16\.\d+(?:\.\d+)?) ', s, re.M)
check('all six sections and fifteen subsections match outline', found_headings == expected_headings)

h2s = re.findall(r'^## (.+)$', s, re.M)
check('last H2 is 本章小结', len(h2s) >= 6 and h2s[-1].strip() == '本章小结')

exercises = re.findall(r'^>\s*\*\*(?:练习|实验) 16-(\d+)[：:]', s, re.M)
check('eighteen exercises 16-1 to 16-18 in order', exercises == [str(i) for i in range(1, 19)])

code_blocks = re.findall(r'```python\n.*?```', s, re.S)
check('six production Python code blocks', len(code_blocks) == 6 and len(re.findall(r'```', s)) == 12)

worked_examples = re.findall(r'^>\s*\*\*算例\s*16-', s, re.M)
check('twelve standardized blockquote worked examples', len(worked_examples) == 12)

table_captions = re.findall(r'^\*表 16-(\d+)　', s, re.M)
check('twelve standardized italic table captions 16-1 to 16-12', table_captions == [str(i) for i in range(1, 13)])

fig_index = json.loads((HERE / 'figure-index.json').read_text())
fig_count = len(fig_index)
check('twenty-six figures in figure-index.json', fig_count == 26)
captions = re.findall(r'^\*图 16-(\d+)(?:：|　)', s, re.M)
check('sequential external captions 16-1 to 16-26', captions == [str(i) for i in range(1, fig_count + 1)])

imgs = re.findall(r'!\[.+?\]\((ch16/[^)]+)\)', s)
check('active illustrations match figure-index.json', imgs == [row.get('asset') or f"ch16/{row['file']}" for row in fig_index])

for rel in imgs:
    p = md.parent / rel
    text_svg = ''.join(ET.parse(p).getroot().itertext())
    check('no internal figure numbering: ' + p.stem, re.search(r'图\s*\d+\s*[-－–]\s*\d+', text_svg) is None)
    check('SVG PNG PDF available: ' + p.stem, all(p.with_suffix(e).exists() for e in ['.svg', '.png', '.pdf']))
    check('PDF is binary not LFS pointer: ' + p.stem, p.with_suffix('.pdf').read_bytes().startswith(b'%PDF'))

missing = []
for link in re.findall(r'\]\(([^)]+)\)', s):
    u = urlsplit(link)
    raw_path = unquote(u.path)
    if not u.scheme and raw_path and not (md.parent / raw_path).exists():
        missing.append((str(md), link))
check('local links exist', not missing)

defs = re.findall(r'^\[\^([^\]]+)\]:', s, re.M)
refs = re.findall(r'\[\^([^\]]+)\](?!:)', s)
check('all footnotes used and defined once', set(defs) == set(refs) and len(defs) == len(set(defs)) and len(defs) >= 7)

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

d = json.loads((HERE / 'figure-data.json').read_text())['locked_truths']
e11 = d['lease_bound_weights_example_16_11']
check(
    'example 16-11 lease-bound weight protection and HBM zeroization arithmetic',
    close(e11['heartbeat_interval_s'] * 10.0, e11['grace_period_s'])
    and close(e11['weight_gb'] / e11['hbm_write_bw_gbps'] * 1000.0, e11['zeroization_ms'])
    and close(e11['dek_rotation_ms'] + e11['zeroization_ms'], 20.50),
)

e12 = d['tournament_watermark_example_16_12']
check(
    'example 16-12 fused tournament watermark Z-score arithmetic',
    close(round(2.0 * math.sqrt(e12['tournament_layers_m'] * e12['tokens_L']) * (e12['effective_win_prob'] - 0.50), 2), e12['z_score_at_200'])
    and math.ceil(16.0 / (4.0 * e12['tournament_layers_m'] * ((e12['effective_win_prob'] - 0.50) ** 2))) == e12['min_tokens_for_z4'],
)

passed = sum(1 for c in checks if c['passed'])
print(json.dumps({"chapter": "ch16", "passed": passed, "total": len(checks), "checks": checks}, ensure_ascii=False, indent=2))
if passed != len(checks):
    sys.exit(1)
