#!/usr/bin/env python3
"""Verify Chapter 23 structure, figure layout, mathematical formulas, and numerical examples."""
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


md = HERE.parent / '23-软硬件协同与新型架构.md'
s = md.read_text(encoding='utf-8')
page = preview_path(md).read_text(encoding='utf-8')

expected_headings = ["23.1", "23.1.1", "23.1.2", "23.1.3", "23.2", "23.2.1", "23.2.2", "23.2.3", "23.3", "23.3.1", "23.3.2", "23.4", "23.4.1", "23.4.2", "23.5", "23.5.1", "23.5.2", "23.5.3", "23.6", "23.6.1", "23.6.2", "23.7", "23.7.1", "23.7.2", "23.8", "23.8.1", "23.8.2", "23.9", "23.9.1", "23.9.2", "23.9.3", "23.10", "23.10.1", "23.10.2", "23.11", "23.11.1", "23.11.2", "23.12", "23.12.1", "23.12.2", "23.13", "23.13.1", "23.13.2", "23.14", "23.14.1", "23.14.2", "23.15", "23.15.1", "23.15.2", "23.16", "23.16.1", "23.16.2", "23.17", "23.17.1", "23.17.2"]
found_headings = re.findall(r'^#{2,3} (23\.\d+(?:\.\d+)?) ', s, re.M)
check('all sections and subsections match outline', found_headings == expected_headings)

h2s = re.findall(r'^## (.+)$', s, re.M)
check('last H2 is 本章小结', len(h2s) >= 5 and h2s[-1].strip() == '本章小结')

exercises = re.findall(r'^(?:>\s*)?\*\*(?:(?:练习|实验|习题|拓展练习|思考题)\s+)?23-(\d+)\b(?!\s*解答)', s, re.M)
check('sequential exercises 23-1 to 23-39', exercises == [str(i) for i in range(1, 39 + 1)])

py_blocks = re.findall(r'^```python\b', s, re.M)
cpp_blocks = re.findall(r'^```cpp\b', s, re.M)
check('expected production code blocks', len(py_blocks) == 18 and len(cpp_blocks) == 0)

worked_examples = re.findall(r'^>\s*\*\*算例\s*23-(\d+)\b', s, re.M)
check('sequential blockquote worked examples 23-1 to 23-25', worked_examples == [str(i) for i in range(1, 25 + 1)])

table_captions = re.findall(r'^\*表\s*23-(\d+)(?:：|　|\s)', s, re.M)
check('sequential italic table captions 23-1 to 23-24', table_captions == [str(i) for i in range(1, 24 + 1)])

fig_index = json.loads((HERE / 'figure-index.json').read_text(encoding='utf-8'))
fig_count = len(fig_index)
check('expected figures in figure-index.json', fig_count == 60)
captions = re.findall(r'^\*图\s*23-(\d+)(?:：|　|\s)', s, re.M)
check('sequential captions 23-1 to 23-60', captions == [str(i) for i in range(1, fig_count + 1)])

imgs = re.findall(r'!\[[^\n]*\]\((ch23/[^)]+)\)', s)
check(
    'active illustrations match figure-index.json',
    imgs == [row.get('asset') or f"ch23/{row['file']}" for row in fig_index] and len(imgs) == 60,
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
check('all footnotes used and defined once', set(refs) <= set(defs) and len(defs) == len(set(defs)) and len(defs) == 49)

for name, key in [('sources.json', 'sources'), ('manifest.json', 'outputs')]:
    data = json.loads((HERE / name).read_text(encoding='utf-8'))[key]
    mismatches = [row['path'] for row in data if 'path' in row and sha256(ROOT / row['path']) != row['sha256']]
    check(name + ' integrity', not mismatches)

math_val = json.loads((HERE / 'math-validation.json').read_text(encoding='utf-8'))
check('KaTeX math validation zero errors', not math_val.get('errors'))

layout_val = json.loads((HERE / 'teaching-layout-validation.json').read_text(encoding='utf-8'))
check('layout validation zero warnings', layout_val.get('total_warnings', 0) == 0 and layout_val.get('all_passed', False))

check('HTML preview generated and non-empty', '<html' in page and len(page) > 10000)

exp_dir = ROOT / 'experiments' / 'ch23'
for suite in sorted(exp_dir.glob('*')):
    if suite.is_dir() and (suite / 'verify.py').exists():
        res = subprocess.run([sys.executable, 'verify.py'], cwd=suite, capture_output=True, text=True)
        check(f'experiment suite {suite.name} verify.py', res.returncode == 0)

failed = [c for c in checks if not c['passed']]
print(json.dumps({"total": len(checks), "passed": len(checks) - len(failed), "failed": failed}, ensure_ascii=False, indent=2))
if failed:
    sys.exit(1)
