#!/usr/bin/env python3
"""Verify Chapter 24 structure, figure layout, mathematical formulas, and numerical examples."""
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


md = HERE.parent / '24-全栈跨层优化与自演进系统.md'
s = md.read_text(encoding='utf-8')
page = preview_path(md).read_text(encoding='utf-8')

expected_headings = ["24.1", "24.1.1", "24.1.2", "24.2", "24.2.1", "24.2.2", "24.3", "24.3.1", "24.3.2", "24.4", "24.4.1", "24.4.2", "24.5", "24.5.1", "24.5.2", "24.6", "24.6.1", "24.6.2", "24.7", "24.7.1", "24.7.2", "24.8", "24.8.1", "24.8.2"]
found_headings = re.findall(r'^#{2,3} (24\.\d+(?:\.\d+)?) ', s, re.M)
check('all sections and subsections match outline', found_headings == expected_headings)

h2s = re.findall(r'^## (.+)$', s, re.M)
check('last H2 is 本章小结', len(h2s) >= 5 and h2s[-1].strip() == '本章小结')

exercises = re.findall(r'^(?:>\s*)?\*\*(?:(?:练习|实验|习题|拓展练习|思考题)\s+)?24-(\d+)\b(?!\s*解答)', s, re.M)
check('sequential exercises 24-1 to 24-22', exercises == [str(i) for i in range(1, 22 + 1)])
check('no ASCII-digit-prefixed H4 headings', not re.findall(r'^####\s+\d', s, re.M))
check('links to ch24-ai-factory-capstone.json and all 8 case studies', all(
    ref in s for ref in [
        '../calculations/results/ch24-ai-factory-capstone.json',
        '../case-studies/hundred-megawatt-ai-factory-capstone.md',
        '../case-studies/kernel-and-fleet-efficiency.md',
        '../case-studies/profile-and-plan-ranking.md',
        '../case-studies/expert-dispatch-and-resizing.md',
        '../case-studies/pd-af-heterogeneous.md',
        '../case-studies/weight-offload-execution.md',
        '../case-studies/fusion-legality-and-precision.md',
        '../case-studies/optimization-evaluation-and-deployment.md',
    ]
))

py_blocks = re.findall(r'^```python\b', s, re.M)
cpp_blocks = re.findall(r'^```cpp\b', s, re.M)
check('expected production code blocks', len(py_blocks) == 12 and len(cpp_blocks) == 0)

worked_examples = re.findall(r'^>\s*\*\*算例\s*24-(\d+)\b', s, re.M)
check('sequential blockquote worked examples 24-1 to 24-12', worked_examples == [str(i) for i in range(1, 12 + 1)])

table_captions = re.findall(r'^\*表\s*24-(\d+)(?:：|　|\s)', s, re.M)
check('sequential italic table captions 24-1 to 24-12', table_captions == [str(i) for i in range(1, 12 + 1)])

fig_index = json.loads((HERE / 'figure-index.json').read_text(encoding='utf-8'))
fig_count = len(fig_index)
check('expected figures in figure-index.json', fig_count == 32)
captions = re.findall(r'^\*图\s*24-(\d+)(?:：|　|\s)', s, re.M)
check('sequential captions 24-1 to 24-32', captions == [str(i) for i in range(1, fig_count + 1)])

imgs = re.findall(r'!\[[^\n]*\]\((ch24/[^)]+)\)', s)
check(
    'active illustrations match figure-index.json',
    imgs == [row.get('asset') or f"ch24/{row['file']}" for row in fig_index] and len(imgs) == 32,
)

for rel in imgs:
    p = md.parent / rel
    text_svg = ''.join(ET.parse(p).getroot().itertext())
    check('no internal figure numbering: ' + p.stem, re.search(r'图\s*\d+\s*[-－–]\s*\d+', text_svg) is None)
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
check('all footnotes used and defined once', set(refs) <= set(defs) and len(defs) == len(set(defs)) and len(defs) == 32)

for name, key in [('sources.json', 'sources'), ('manifest.json', 'outputs')]:
    data = json.loads((HERE / name).read_text(encoding='utf-8'))[key]
    mismatches = [row['path'] for row in data if 'path' in row and sha256(ROOT / row['path']) != row['sha256']]
    check(name + ' integrity', not mismatches)

math_val = json.loads((HERE / 'math-validation.json').read_text(encoding='utf-8'))
check('KaTeX math validation zero errors', not math_val.get('errors'))

layout_val = json.loads((HERE / 'teaching-layout-validation.json').read_text(encoding='utf-8'))
check('layout validation zero warnings', layout_val.get('total_warnings', 0) == 0 and layout_val.get('all_passed', False))

check('HTML preview generated and non-empty', '<html' in page and len(page) > 10000)

exp_dir = ROOT / 'experiments' / 'ch24'
for suite in sorted(exp_dir.glob('*')):
    if suite.is_dir() and (suite / 'verify.py').exists():
        res = subprocess.run([sys.executable, 'verify.py'], cwd=suite, capture_output=True, text=True)
        check(f'experiment suite {suite.name} verify.py', res.returncode == 0)

failed = [c for c in checks if not c['passed']]
print(json.dumps({"total": len(checks), "passed": len(checks) - len(failed), "failed": failed}, ensure_ascii=False, indent=2))
if failed:
    sys.exit(1)
