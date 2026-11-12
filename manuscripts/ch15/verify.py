#!/usr/bin/env python3
"""Verify Chapter 15 structure, figure layout, mathematical formulas, and numerical examples."""
from pathlib import Path
from urllib.parse import urlsplit, unquote
import hashlib
import json
import math
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
from preview_output import preview_path

checks = []


def check(name: str, value: bool):
    checks.append({"check": name, "passed": bool(value)})


def sha256(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b'version https://git-lfs.github.com/spec/v1'):
        m = re.search(rb'oid sha256:([0-9a-f]{64})', raw)
        if m:
            return m.group(1).decode('ascii')
    return hashlib.sha256(raw).hexdigest()


md = HERE.parent / '15-集群可靠性与基准评测.md'
s = md.read_text(encoding='utf-8')
page = preview_path(md).read_text(encoding='utf-8')

expected_headings = [
    "15.1", "15.1.1", "15.1.2",
    "15.2", "15.2.1", "15.2.2",
    "15.3", "15.3.1", "15.3.2",
    "15.4", "15.4.1", "15.4.2",
    "15.5", "15.5.1", "15.5.2", "15.5.3",
    "15.6", "15.6.1", "15.6.2",
]
found_headings = re.findall(r'^#{2,3} (15\.\d+(?:\.\d+)?) ', s, re.M)
check('all sections and subsections match outline', found_headings == expected_headings)
check('chapter line count meets Path H2 target (>= 1744)', len(s.splitlines()) >= 1744)
check('zero H5 headings and zero digit-prefixed H4 headings', not re.findall(r'^#####\s', s, re.M) and not re.findall(r'^####\s+\d', s, re.M))
check('zero legacy ch20 footnotes or stale Poisson rounding', not re.search(r'\[\^ch20-|0\.99839|0\.99907|99\.907', s))

h2s = re.findall(r'^## (.+)$', s, re.M)
check('last H2 is 本章小结', len(h2s) >= 6 and h2s[-1].strip() == '本章小结')

exercises = re.findall(r'^(?:>\s*)?\*\*(?:(?:练习|实验|习题|拓展练习|思考题)\s+)?15-(\d+)\b(?!\s*解答)', s, re.M)
check('sequential exercises 15-1 to 15-18', exercises == [str(i) for i in range(1, 18 + 1)])
answers = re.findall(r'^-\s*\*\*15-(\d+)\s*解答\*\*', s, re.M)
check('all 18 exercises have step-by-step derivations', answers == [str(i) for i in range(1, 18 + 1)])

py_blocks = re.findall(r'^```python\b', s, re.M)
cpp_blocks = re.findall(r'^```cpp\b', s, re.M)
check('expected production code blocks', len(py_blocks) == 10 and len(cpp_blocks) == 0)

worked_examples = re.findall(r'^>\s*\*\*算例\s*15-(\d+)\b', s, re.M)
check('sequential blockquote worked examples 15-1 to 15-10', worked_examples == [str(i) for i in range(1, 10 + 1)])

table_captions = re.findall(r'^\*表\s*15-(\d+)(?:：|　|\s)', s, re.M)
check('sequential italic table captions 15-1 to 15-12', table_captions == [str(i) for i in range(1, 12 + 1)])

fig_index = json.loads((HERE / 'figure-index.json').read_text(encoding='utf-8'))
fig_count = len(fig_index)
check('expected figures in figure-index.json', fig_count == 24)
captions = re.findall(r'^\*图\s*15-(\d+)(?:：|　|\s)', s, re.M)
check('sequential captions 15-1 to 15-24', captions == [str(i) for i in range(1, fig_count + 1)])

imgs = re.findall(r'!\[[^\n]*\]\((ch15/[^)]+)\)', s)
check(
    'active illustrations match figure-index.json',
    imgs == [row.get('asset') or f"ch15/{row['file']}" for row in fig_index] and len(imgs) == 24,
)

for rel in imgs:
    p = md.parent / rel
    root_el = ET.parse(p).getroot()
    text_svg = ''.join(root_el.itertext())
    has_raw_text_el = any(el.tag.endswith('text') for el in root_el.iter())
    check('no internal figure numbering and paths-only SVG: ' + p.stem, re.search(r'图\s*\d+\s*[-－–]\s*\d+', text_svg) is None and not has_raw_text_el)
    check('SVG PNG PDF available: ' + p.stem, all(p.with_suffix(e).exists() for e in ['.svg', '.png', '.pdf']))
    check('PDF is binary not LFS pointer: ' + p.stem, p.with_suffix('.pdf').read_bytes().startswith(b'%PDF'))

missing = []
for link in re.findall(r'\]\(([^)]+)\)', s):
    u = urlsplit(link.strip('<>'))
    if not u.scheme and u.path and not (md.parent / unquote(u.path)).exists():
        missing.append((str(md), link))
check('local links exist', not missing)

defs = re.findall(r'^\[\^([^\]]+)\]:', s, re.M)
refs = re.findall(r'\[\^([^\]]+)\](?!:)', s)
check('all footnotes used and defined once', set(refs) == set(defs) and len(defs) == len(set(defs)) and len(defs) == 28)

golden = json.loads((ROOT / 'calculations/results/ch15-fleet-reliability-profiler.json').read_text(encoding='utf-8'))
g_sum = golden['summary']
check(
    'golden calc.py ch15-fleet-reliability-profiler metrics locked in manuscript',
    math.isclose(g_sum['fleet_100k_mtbf_minutes'], 30.41)
    and math.isclose(g_sum['ebpf_dynolog_total_overhead_pct'], 0.42)
    and math.isclose(g_sum['cupti_16k_burned_gpus_exact'], 1310.72)
    and math.isclose(g_sum['trace_storage_reduction_x'], 788.9)
    and g_sum['wait_chain_root_cause_rank'] == 5
    and math.isclose(g_sum['monarch_isolation_seconds'], 0.85)
    and math.isclose(g_sum['monarch_vs_spmd_speedup_x'], 167.1)
    and ('0.42%' in s or r'0.42\%' in s)
    and all(tok in s for tok in ['30.41', '1,310.72', '788.9', '0.85', '142.0', '167.1']),
)

for name, key in [('sources.json', 'sources'), ('manifest.json', 'outputs')]:
    data = json.loads((HERE / name).read_text(encoding='utf-8'))[key]
    mismatches = [row['path'] for row in data if 'path' in row and sha256(ROOT / row['path']) != row['sha256']]
    check(name + ' integrity', not mismatches)

math_val = json.loads((HERE / 'math-validation.json').read_text(encoding='utf-8'))
check('KaTeX math validation zero errors', not math_val.get('errors'))

layout_val = json.loads((HERE / 'teaching-layout-validation.json').read_text(encoding='utf-8'))
check(
    'layout validation zero warnings',
    (
        isinstance(layout_val, list)
        and len(layout_val) == fig_count
        and all(not row.get('warnings') and row.get('passed', True) for row in layout_val)
    )
    or (
        isinstance(layout_val, dict)
        and layout_val.get('total_warnings', 0) == 0
        and layout_val.get('all_passed', False)
    ),
)

check('HTML preview generated and non-empty', '<html' in page and len(page) > 10000)

exp_dir = ROOT / 'experiments' / 'ch15'
for suite in sorted(exp_dir.glob('*')):
    if suite.is_dir() and (suite / 'verify.py').exists():
        res = subprocess.run([sys.executable, 'verify.py'], cwd=suite, capture_output=True, text=True)
        check(f'experiment suite {suite.name} verify.py', res.returncode == 0)

failed = [c for c in checks if not c['passed']]
print(json.dumps({"total": len(checks), "passed": len(checks) - len(failed), "failed": failed}, ensure_ascii=False, indent=2))
if failed:
    sys.exit(1)
