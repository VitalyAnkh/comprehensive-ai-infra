#!/usr/bin/env python3
"""Verify Chapter 18 structure, figure layout, mathematical formulas, and numerical examples."""
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


md = HERE.parent / '18-大规模后训练（RLVR）、异步强化学习与推理时扩展系统.md'
s = md.read_text()
page = preview_path(md).read_text()

expected_headings = [
    '18.1', '18.1.1', '18.1.2',
    '18.2', '18.2.1', '18.2.2',
    '18.3', '18.3.1', '18.3.2', '18.3.3',
    '18.4', '18.4.1', '18.4.2', '18.4.3',
]
found_headings = re.findall(r'^#{2,3} (18\.\d+(?:\.\d+)?) ', s, re.M)
check('all four sections and ten subsections match outline', found_headings == expected_headings)

h2s = re.findall(r'^## (.+)$', s, re.M)
check('last H2 is 本章小结', len(h2s) >= 6 and h2s[-1].strip() == '本章小结')

exercises = re.findall(r'^>\s*\*\*(?:练习|实验) 18-(\d+)[：:（]', s, re.M)
check('sixteen exercises 18-1 to 18-16 in order', exercises == [str(i) for i in range(1, 17)])

code_blocks = re.findall(r'```python\n.*?```', s, re.S)
check('ten production Python code blocks', len(code_blocks) == 10 and len(re.findall(r'```', s)) == 20)

worked_examples = re.findall(r'^>\s*\*\*算例\s*18-', s, re.M)
check('ten standardized blockquote worked examples', len(worked_examples) == 10)

table_captions = re.findall(r'^\*表 18-(\d+)　', s, re.M)
check('nine standardized italic table captions 18-1 to 18-9', table_captions == [str(i) for i in range(1, 10)])

fig_index = json.loads((HERE / 'figure-index.json').read_text())
fig_count = len(fig_index)
check('twenty-two figures in figure-index.json', fig_count == 22)
captions = re.findall(r'^\*图 18-(\d+)(?:：|　)', s, re.M)
check('sequential captions 18-1 to 18-22', sorted(captions, key=int) == [str(i) for i in range(1, fig_count + 1)])

imgs = re.findall(r'!\[.+?\]\((ch18/[^)]+)\)', s)
check('twenty-two active illustrations match figure-index.json', set(imgs) == {row.get('asset') or f"ch18/{row['file']}" for row in fig_index} and len(imgs) == 22)

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
check('all footnotes used and defined once', set(defs) == set(refs) and len(defs) == len(set(defs)) and len(defs) >= 19)

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
geo = d['geo_distributed_async_rl']
check(
    'example 18-9 geo-distributed async RL WAN compression arithmetic',
    geo['batch_trajectories'] * geo['avg_seq_len'] == geo['total_tokens']
    and close(round(geo['total_tokens'] * geo['kv_bytes_per_token'] / 1e9, 1), geo['kv_payload_gb'])
    and close(round(geo['total_tokens'] * geo['compact_bytes_per_token'] / (1024 * 1024), 1), geo['compact_payload_mib'])
    and close(round(geo['kv_bytes_per_token'] / geo['compact_bytes_per_token'], 1), geo['compression_ratio']),
)

mt = d['multi_turn_agentic_rlvr_and_vtrace']
check(
    'example 18-10 multi-turn agentic RLVR CoW tree forking and V-trace arithmetic',
    mt['branch_factor_l1'] * mt['branch_factor_l2'] == 16
    and close(16 * (mt['root_setup_sec'] + mt['prefix_turns'] * mt['turn_exec_sec']), mt['linear_sandbox_cpu_sec'])
    and close(mt['root_setup_sec'] + 4 * (mt['prefix_turns'] * mt['turn_exec_sec']), mt['forked_sandbox_cpu_sec'])
    and close(round(mt['linear_sandbox_cpu_sec'] / mt['forked_sandbox_cpu_sec'], 2), mt['sandbox_cpu_speedup'])
    and close(round((1.0 - mt['forked_prefix_kv_slots'] / mt['linear_prefix_kv_slots']) * 100.0, 2), mt['prefix_kv_saving_pct'])
    and close(round(mt['step_is_ratio'] ** mt['horizon_steps'], 4), mt['unclipped_prod_is']),
)

passed = sum(1 for c in checks if c['passed'])
print(json.dumps({"chapter": "ch18", "passed": passed, "total": len(checks), "checks": checks}, ensure_ascii=False, indent=2))
if passed != len(checks):
    sys.exit(1)
