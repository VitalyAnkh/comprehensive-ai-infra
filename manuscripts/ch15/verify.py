#!/usr/bin/env python3
"""Verify Chapter 15 structure, figure layout, mathematical formulas, and numerical examples."""
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


md = HERE.parent / '15-吉瓦级 AI 数据中心、电力热学与跨地域协同训练.md'
s = md.read_text()
page = preview_path(md).read_text()

expected_headings = [
    '15.1', '15.1.1', '15.1.2', '15.1.3',
    '15.2', '15.2.1', '15.2.2', '15.2.3',
    '15.3', '15.3.1', '15.3.2', '15.3.3',
    '15.4', '15.4.1', '15.4.2', '15.4.3',
    '15.5', '15.5.1', '15.5.2',
    '15.6', '15.6.1', '15.6.2', '15.6.3', '15.6.4',
]
found_headings = re.findall(r'^#{2,3} (15\.\d+(?:\.\d+)?) ', s, re.M)
check('all six sections and eighteen subsections match outline', found_headings == expected_headings)

h2s = re.findall(r'^## (.+)$', s, re.M)
check('last H2 is 本章小结', len(h2s) >= 6 and h2s[-1].strip() == '本章小结')

exercises = re.findall(r'^>\s*\*\*(?:练习|实验) 15-(\d+)[：:]', s, re.M)
check('sixteen exercises 15-1 to 15-16 in order', exercises == [str(i) for i in range(1, 17)])

code_blocks = re.findall(r'```python\n.*?```', s, re.S)
check('three production Python code blocks', len(code_blocks) == 3 and len(re.findall(r'```', s)) == 6)

worked_examples = re.findall(r'^>\s*\*\*算例\s*15-', s, re.M)
check('twelve standardized blockquote worked examples', len(worked_examples) == 12)

table_captions = re.findall(r'^\*表 15-(\d+)　', s, re.M)
check('eleven standardized italic table captions 15-1 to 15-11', table_captions == [str(i) for i in range(1, 12)])

fig_index = json.loads((HERE / 'figure-index.json').read_text())
fig_count = len(fig_index)
check('twenty-four figures in figure-index.json', fig_count == 24)
captions = re.findall(r'^\*图 15-(\d+)(?:：|　)', s, re.M)
check('sequential external captions 15-1 to 15-24', captions == [str(i) for i in range(1, fig_count + 1)])

imgs = re.findall(r'!\[.+?\]\((ch15/[^)]+)\)', s)
check('active illustrations match figure-index.json', imgs == [row.get('asset') or f"ch15/{row['file']}" for row in fig_index])

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
check('all footnotes used and defined once', set(defs) == set(refs) and len(defs) == len(set(defs)) and len(defs) >= 8)

for name, key in [('sources.json', 'sources'), ('manifest.json', 'outputs')]:
    data = json.loads((HERE / name).read_text())[key]
    mismatches = [row['path'] for row in data if sha256(ROOT / row['path']) != row['sha256']]
    check(name + ' integrity', not mismatches)

math_val = json.loads((HERE / 'math-validation.json').read_text())
check('no formula errors', not math_val['errors'] and 'katex-error' not in page)

layout_val = json.loads((HERE / 'teaching-layout-validation.json').read_text())
check(
    'book-size figure layout',
    len(layout_val) == 24 and all(r['width_pt'] == 420 and r['min_label_pt'] >= 11 and not r['text_extent_warnings'] for r in layout_val),
)

d = json.loads((HERE / 'figure-data.json').read_text())
e11 = d['hvdc_vpd_example_15_11']
check(
    'example 15-11 800V HVDC busbar and VPD arithmetic',
    close(round(e11['rack_power_kw'] * 1000.0 / 48.0, 2), e11['current_48v_a'])
    and close(round(e11['rack_power_kw'] * 1000.0 / 800.0, 2), e11['current_800v_a'])
    and close(round((e11['rack_power_kw'] * 1000.0 / 48.0) ** 2 * 0.0008, 2), e11['i2r_loss_48v_w'])
    and close(round((e11['rack_power_kw'] * 1000.0 / 800.0) ** 2 * 0.0008, 2), e11['i2r_loss_800v_w'])
    and close(round((800.0 / 48.0) ** 2, 2), e11['loss_reduction_ratio'])
    and close(round((e11['rack_power_kw'] * 1000.0 / 48.0) / 2.0, 2), e11['copper_area_48v_mm2'])
    and close(round((e11['rack_power_kw'] * 1000.0 / 800.0) / 2.0, 2), e11['copper_area_800v_mm2']),
)

e12 = d['sofc_ere_example_15_12']
check(
    'example 15-12 SOFC microgrid and ERE heat recovery arithmetic',
    close(e12['it_power_mw'] * e12['pue'], e12['total_power_mw'])
    and close(e12['it_power_mw'] * e12['heat_recovery_ratio'], e12['reused_heat_mw'])
    and close((e12['total_power_mw'] - e12['reused_heat_mw']) / e12['it_power_mw'], e12['ere'])
    and close(round(e12['reused_heat_mw'] / e12['total_power_mw'], 4), e12['erf'])
    and close(e12['reused_heat_mw'] * 8760.0 / 1000.0, e12['annual_reused_gwh'])
    and close(e12['annual_reused_gwh'] * 0.20, e12['annual_co2_saved_kt']),
)

passed = sum(1 for c in checks if c['passed'])
print(json.dumps({"chapter": "ch15", "passed": passed, "total": len(checks), "checks": checks}, ensure_ascii=False, indent=2))
if passed != len(checks):
    sys.exit(1)
