#!/usr/bin/env python3
"""Verify Chapter 19 structure, figure layout, mathematical formulas, and numerical examples."""
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


md = HERE.parent / '19-具身智能（VLA）、实时控制与大规模物理仿真底座.md'
s = md.read_text()
page = preview_path(md).read_text()

expected_headings = [
    '19.1', '19.1.1', '19.1.2', '19.1.3',
    '19.2', '19.2.1', '19.2.2',
    '19.3', '19.3.1', '19.3.2', '19.3.3',
    '19.4', '19.4.1', '19.4.2', '19.4.3', '19.4.4',
]
found_headings = re.findall(r'^#{2,3} (19\.\d+(?:\.\d+)?) ', s, re.M)
check('all four sections and twelve subsections match outline', found_headings == expected_headings)

h2s = re.findall(r'^## (.+)$', s, re.M)
check('last H2 is 本章小结', len(h2s) >= 6 and h2s[-1].strip() == '本章小结')

exercises = re.findall(r'^>\s*\*\*(?:练习|实验) 19-(\d+)[：:（]', s, re.M)
check('twelve exercises 19-1 to 19-12 in order', exercises == [str(i) for i in range(1, 13)])

code_blocks = re.findall(r'```(?:python|cpp)\n.*?```', s, re.S)
check('eight production code blocks (7 python + 1 cpp)', len(code_blocks) == 8 and len(re.findall(r'```', s)) == 16)

worked_examples = re.findall(r'^>\s*\*\*算例\s*19-', s, re.M)
check('six standardized blockquote worked examples', len(worked_examples) == 6)

table_captions = re.findall(r'^\*表 19-(\d+)　', s, re.M)
check('eight standardized italic table captions 19-1 to 19-8', table_captions == [str(i) for i in range(1, 9)])

fig_index = json.loads((HERE / 'figure-index.json').read_text())
fig_count = len(fig_index)
check('eighteen figures in figure-index.json', fig_count == 18)
captions = re.findall(r'^\*图 19-(\d+)(?:：|　)', s, re.M)
check('sequential captions 19-1 to 19-18', sorted(captions, key=int) == [str(i) for i in range(1, fig_count + 1)])

imgs = re.findall(r'!\[.+?\]\((ch19/[^)]+)\)', s)
check('eighteen active illustrations match figure-index.json', set(imgs) == {row.get('asset') or f"ch19/{row['file']}" for row in fig_index} and len(imgs) == 18)

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
check('all footnotes used and defined once', set(defs) == set(refs) and len(defs) == len(set(defs)) and len(defs) >= 9)

for name, key in [('sources.json', 'sources'), ('manifest.json', 'outputs')]:
    data = json.loads((HERE / name).read_text())[key]
    mismatches = [row['path'] for row in data if sha256(ROOT / row['path']) != row['sha256']]
    check(name + ' integrity', not mismatches)

math_val = json.loads((HERE / 'math-validation.json').read_text())
check('no formula errors', not math_val['errors'] and 'katex-error' not in page)

layout_val = json.loads((HERE / 'teaching-layout-validation.json').read_text())
check(
    'book-size figure layout',
    len(layout_val) == 18 and all(r['width_pt'] == 420 and r['min_label_pt'] >= 11 and not r['text_extent_warnings'] for r in layout_val),
)

d = json.loads((HERE / 'figure-data.json').read_text())
fast = d['three_tier_wbc_and_fast_tokenizer']['fast_dct_action_tokenizer']
check(
    'example 19-6 FAST DCT action tokenizer compression arithmetic',
    fast['chunk_horizon_steps'] * fast['action_dims'] == fast['naive_time_domain_tokens']
    and close(round(fast['naive_time_domain_tokens'] / fast['fast_bpe_tokens'], 2), fast['compression_ratio_x'])
    and close(fast['ar_decode_naive_ms'] / fast['ar_decode_fast_ms'], fast['compression_ratio_x']),
)

passed = sum(1 for c in checks if c['passed'])
print(json.dumps({"chapter": "ch19", "passed": passed, "total": len(checks), "checks": checks}, ensure_ascii=False, indent=2))
if passed != len(checks):
    sys.exit(1)
