#!/usr/bin/env python3
"""Verify Chapter 16 structure, figure layout, mathematical formulas, and numerical examples."""
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


md = HERE.parent / '16-机密计算与可信安全系统.md'
s = md.read_text(encoding='utf-8')
page = preview_path(md).read_text(encoding='utf-8')

expected_headings = [
    "16.1", "16.1.1", "16.1.2",
    "16.2", "16.2.1", "16.2.2",
    "16.3", "16.3.1", "16.3.2",
    "16.4", "16.4.1", "16.4.2",
    "16.5", "16.5.1", "16.5.2",
]
found_headings = re.findall(r'^#{2,3} (16\.\d+(?:\.\d+)?) ', s, re.M)
check('all sections and subsections match outline', found_headings == expected_headings)
check('chapter line count meets Path H2 target (>= 2413)', len(s.splitlines()) >= 2413)
check('zero H5 headings and zero digit-prefixed H4 headings', not re.findall(r'^#####\s', s, re.M) and not re.findall(r'^####\s+\d', s, re.M))
check('zero internal scaffolding or wrong part taxonomy tokens', not re.search(r'六大部分|六部二十四章|第六部分|I19|i19_|TC-\d+|Paper\s*#\d+|\\tag\{', s))

h2s = re.findall(r'^## (.+)$', s, re.M)
check('last H2 is 本章小结', len(h2s) == 8 and h2s[-1].strip() == '本章小结')

exercises = re.findall(r'^(?:>\s*)?\*\*(?:(?:练习|实验|习题|拓展练习|思考题)\s+)?16-(\d+)\b(?!\s*解答)', s, re.M)
check('sequential exercises 16-1 to 16-28', exercises == [str(i) for i in range(1, 28 + 1)])
answers = re.findall(r'^-\s*\*\*16-(\d+)\s*解答', s, re.M)
check('all 28 exercises have step-by-step derivations', answers == [str(i) for i in range(1, 28 + 1)])

py_blocks = re.findall(r'^```python\b', s, re.M)
cpp_blocks = re.findall(r'^```cpp\b', s, re.M)
check('expected production code blocks', len(py_blocks) == 12 and len(cpp_blocks) == 0)

worked_examples = re.findall(r'^>\s*\*\*算例\s*16-(\d+)\b', s, re.M)
check('sequential blockquote worked examples 16-1 to 16-18', worked_examples == [str(i) for i in range(1, 18 + 1)])

table_captions = re.findall(r'^\*表\s*16-(\d+)(?:：|　|\s)', s, re.M)
check('sequential italic table captions 16-1 to 16-18', table_captions == [str(i) for i in range(1, 18 + 1)])

fig_index = json.loads((HERE / 'figure-index.json').read_text(encoding='utf-8'))
fig_count = len(fig_index)
check('expected figures in figure-index.json', fig_count == 42)
captions = re.findall(r'^\*图\s*16-(\d+)(?:：|　|\s)', s, re.M)
check('sequential captions 16-1 to 16-42', captions == [str(i) for i in range(1, fig_count + 1)])

imgs = re.findall(r'!\[[^\n]*\]\((ch16/[^)]+)\)', s)
check(
    'active illustrations match figure-index.json',
    imgs == [row.get('asset') or f"ch16/{row['file']}" for row in fig_index] and len(imgs) == 42,
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
check('all footnotes used and defined once', set(refs) == set(defs) and len(defs) == len(set(defs)) and len(defs) == 34)

golden = json.loads((ROOT / 'calculations/results/ch16-confidential-tee-wan.json').read_text(encoding='utf-8'))
g_sum = golden['summary']
check(
    'golden calc.py ch16-confidential-tee-wan metrics locked in manuscript',
    math.isclose(g_sum['bounce_buffer_64mib_ms'], 8.8)
    and math.isclose(g_sum['rubin_tee_io_64mib_single_ms'], 0.6)
    and math.isclose(g_sum['rubin_tee_io_64mib_pipelined_ms'], 0.4)
    and math.isclose(g_sum['pipellm_triple_buffer_mib'], 192.0)
    and math.isclose(g_sum['dax_struct_page_metadata_tax_pct'], 1.5625)
    and math.isclose(g_sum['dsec_380k_naive_ram_tb'], 242.26)
    and math.isclose(g_sum['dsec_380k_shared_dax_ram_tb'], 38.25)
    and math.isclose(g_sum['dsec_memory_saving_pct'], 84.21)
    and math.isclose(g_sum['wan_streaming_diloco_fp4_payload_gb'], 124.84)
    and math.isclose(g_sum['wan_per_fragment_transfer_seconds'], 14.19)
    and math.isclose(g_sum['pol_10round_detection_pct'], 99.97)
    and math.isclose(g_sum['geom_median_variance_inflation_x'], 1.43)
    and math.isclose(g_sum['spot_dht_elastic_ettr_pct'], 96.8)
    and all(tok in s for tok in ['8.8', '0.60', '0.40', '192', '1.5625', '242.26', '38.25', '84.21', '124.84', '14.19', '99.97', '1.43', '96.8']),
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

exp_dir = ROOT / 'experiments' / 'ch16'
exp_scaffolding_hits = []
for suite in sorted(exp_dir.glob('*')):
    if suite.is_dir():
        for fp in sorted(suite.rglob('*')):
            if fp.suffix in {'.py', '.json', '.md'} and '__pycache__' not in fp.parts:
                if re.search(r'I19|i19_|Example 28-', fp.read_text(encoding='utf-8')):
                    exp_scaffolding_hits.append(str(fp))
    if suite.is_dir() and (suite / 'verify.py').exists():
        res = subprocess.run([sys.executable, 'verify.py'], cwd=suite, capture_output=True, text=True)
        check(f'experiment suite {suite.name} verify.py', res.returncode == 0)
check('experiment ch16 files free of legacy I19 or Ch28 scaffolding', not exp_scaffolding_hits)

failed = [c for c in checks if not c['passed']]
print(json.dumps({"total": len(checks), "passed": len(checks) - len(failed), "failed": failed}, ensure_ascii=False, indent=2))
if failed:
    sys.exit(1)
