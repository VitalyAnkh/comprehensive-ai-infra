#!/usr/bin/env python3
"""Verify Chapter 14 structure, figure layout, mathematical formulas, and numerical examples."""
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


md = HERE.parent / '14-智算中心与电力热学系统.md'
s = md.read_text(encoding='utf-8')
page = preview_path(md).read_text(encoding='utf-8')

expected_headings = ["14.1", "14.1.1", "14.1.2", "14.1.3", "14.2", "14.2.1", "14.2.2", "14.2.3", "14.3", "14.3.1", "14.3.2", "14.3.3", "14.4", "14.4.1", "14.4.2", "14.4.3", "14.5", "14.5.1", "14.5.2", "14.6", "14.6.1", "14.6.2", "14.6.3", "14.6.4"]
found_headings = re.findall(r'^#{2,3} (14\.\d+(?:\.\d+)?) ', s, re.M)
check('all sections and subsections match outline', found_headings == expected_headings)
check('zero H5+ headings', not re.findall(r'^#{5,}\s+', s, re.M))
check('zero ASCII-digit-prefixed H4 headings', not re.findall(r'^####\s+[0-9]+', s, re.M))

h2s = re.findall(r'^## (.+)$', s, re.M)
check('last H2 is 本章小结', len(h2s) >= 5 and h2s[-1].strip() == '本章小结')

exercises = re.findall(r'^(?:>\s*)?\*\*(?:(?:练习|实验|习题|拓展练习|思考题)\s+)?14-(\d+)\b(?!\s*解答)', s, re.M)
check('sequential exercises 14-1 to 14-18', exercises == [str(i) for i in range(1, 18 + 1)])
derivs = re.findall(r'^>\s*-\s*\*\*参考(?:推导与答案|验证断言与(?:期望输出|推导))要点\*\*：', s, re.M)
check('all 18 exercises include step-by-step reference derivations', len(derivs) == 18)

py_blocks = re.findall(r'^```python\b', s, re.M)
cpp_blocks = re.findall(r'^```cpp\b', s, re.M)
check('expected production code blocks', len(py_blocks) == 6 and len(cpp_blocks) == 0)

worked_examples = re.findall(r'^>\s*\*\*算例\s*14-(\d+)\b', s, re.M)
check('sequential blockquote worked examples 14-1 to 14-12', worked_examples == [str(i) for i in range(1, 12 + 1)])

table_captions = re.findall(r'^\*表\s*14-(\d+)(?:：|　|\s)', s, re.M)
check('sequential italic table captions 14-1 to 14-12', table_captions == [str(i) for i in range(1, 12 + 1)])

fig_index = json.loads((HERE / 'figure-index.json').read_text(encoding='utf-8'))
fig_count = len(fig_index)
check('expected figures in figure-index.json', fig_count == 26)
captions = re.findall(r'^\*图\s*14-(\d+)(?:：|　|\s)', s, re.M)
check('sequential captions 14-1 to 14-26', captions == [str(i) for i in range(1, fig_count + 1)])

imgs = re.findall(r'!\[[^\n]*\]\((ch14/[^)]+)\)', s)
check(
    'active illustrations match figure-index.json',
    imgs == [row.get('asset') or f"ch14/{row['file']}" for row in fig_index] and len(imgs) == 26,
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
check('all footnotes used and defined once', set(refs) <= set(defs) and len(defs) == len(set(defs)) and len(defs) == 22)

for name, key in [('sources.json', 'sources'), ('manifest.json', 'outputs')]:
    data = json.loads((HERE / name).read_text(encoding='utf-8'))[key]
    mismatches = [row['path'] for row in data if 'path' in row and sha256(ROOT / row['path']) != row['sha256']]
    check(name + ' integrity', not mismatches)

math_val = json.loads((HERE / 'math-validation.json').read_text(encoding='utf-8'))
check('KaTeX math validation zero errors', not math_val.get('errors'))

layout_val = json.loads((HERE / 'teaching-layout-validation.json').read_text(encoding='utf-8'))
if isinstance(layout_val, list):
    check('layout validation zero warnings', len(layout_val) == 26 and all(not r.get('warnings') for r in layout_val))
else:
    check('layout validation zero warnings', layout_val.get('total_warnings', 0) == 0 and layout_val.get('all_passed', False))

check('HTML preview generated and non-empty', '<html' in page and len(page) > 10000)
calc_dc = json.loads((ROOT / 'calculations/results/ch14-datacenter-power-cooling.json').read_text(encoding='utf-8'))['summary']
exp_14_04 = json.loads((ROOT / 'experiments/ch14/14-04/expected/metrics.json').read_text(encoding='utf-8'))
exp_14_05 = json.loads((ROOT / 'experiments/ch14/14-05/expected/metrics.json').read_text(encoding='utf-8'))
check(
    'golden calc datacenter-power-cooling matches manuscript and 14-04/14-05',
    calc_dc['current_48v_amp'] == 2750.0
    and calc_dc['current_800v_amp'] == 165.0
    and calc_dc['hvdc_i2r_loss_reduction_x'] == 277.78
    and calc_dc['w45_flow_l_per_min_132kw'] == 159.5
    and calc_dc['w45_flow_l_per_min_132kw_rho1'] == 157.9
    and calc_dc['w45_flow_l_per_min_165kw'] == 199.4
    and calc_dc['coldplate_1400w_case_temp_c'] == 74.2
    and calc_dc['dci_671b_outer_sync_seconds'] == 31.58
    and calc_dc['dci_blocking_comm_overhead_pct'] == 1.48
    and calc_dc['campus_100mw_elec_saving_musd_per_yr'] == 70.08
    and exp_14_04['ashrae_w45_cooling']['w45_flow_l_per_min_132kw'] == 159.5
    and exp_14_04['ashrae_w45_cooling']['w45_flow_l_per_min_132kw_rho1'] == 157.9
    and exp_14_05['controlled_optical_dci']['dci_671b_outer_sync_seconds'] == 31.58
    and '159.5' in s
    and '157.9' in s
    and '157.8' not in s,
)

exp_dir = ROOT / 'experiments' / 'ch14'
for suite in sorted(exp_dir.glob('*')):
    if suite.is_dir() and (suite / 'verify.py').exists():
        res = subprocess.run([sys.executable, 'verify.py'], cwd=suite, capture_output=True, text=True)
        check(f'experiment suite {suite.name} verify.py', res.returncode == 0)

failed = [c for c in checks if not c['passed']]
print(json.dumps({"total": len(checks), "passed": len(checks) - len(failed), "failed": failed}, ensure_ascii=False, indent=2))
if failed:
    sys.exit(1)
