#!/usr/bin/env python3
"""Verify Chapter 13 structure, figure layout, mathematical formulas, and numerical examples."""
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


md = HERE.parent / '13-存储系统与数据工程.md'
s = md.read_text(encoding='utf-8')
page = preview_path(md).read_text(encoding='utf-8')

expected_headings = [
    "13.1", "13.1.1", "13.1.2",
    "13.2", "13.2.1", "13.2.2",
    "13.3", "13.3.1", "13.3.2",
    "13.4", "13.4.1", "13.4.2",
    "13.5", "13.5.1", "13.5.2",
    "13.6", "13.6.1", "13.6.2",
]
found_headings = re.findall(r'^#{2,3} (13\.\d+(?:\.\d+)?) ', s, re.M)
check('all sections and subsections match outline', found_headings == expected_headings)
check('zero H5+ headings', not re.findall(r'^#{5,}\s+', s, re.M))
check('zero ASCII-digit-prefixed H4 headings', not re.findall(r'^####\s+[0-9]+', s, re.M))

h2s = re.findall(r'^## (.+)$', s, re.M)
check('last H2 is 本章小结', len(h2s) >= 5 and h2s[-1].strip() == '本章小结')

exercises = re.findall(r'^(?:>\s*)?\*\*(?:(?:练习|实验|习题|拓展练习|思考题)\s+)?13-(\d+)\b(?!\s*解答)', s, re.M)
check('sequential exercises 13-1 to 13-31', exercises == [str(i) for i in range(1, 31 + 1)])

solutions = re.findall(r'^####\s+(?:练习|实验)\s+13-(\d+)\s+解答$', s, re.M)
check('complete sequential solutions 13-1 to 13-31', solutions == [str(i) for i in range(1, 31 + 1)])

py_blocks = re.findall(r'^```python\b', s, re.M)
cpp_blocks = re.findall(r'^```cpp\b', s, re.M)
check('expected production code blocks', len(py_blocks) == 14 and len(cpp_blocks) == 0)

worked_examples = re.findall(r'^>\s*\*\*算例\s*13-(\d+)\b', s, re.M)
check('sequential blockquote worked examples 13-1 to 13-21', worked_examples == [str(i) for i in range(1, 21 + 1)])

table_captions = re.findall(r'^\*表\s*13-(\d+)(?:：|　|\s)', s, re.M)
check('sequential italic table captions 13-1 to 13-18', table_captions == [str(i) for i in range(1, 18 + 1)])

fig_index = json.loads((HERE / 'figure-index.json').read_text(encoding='utf-8'))
fig_count = len(fig_index)
check('expected figures in figure-index.json', fig_count == 46)
captions = re.findall(r'^\*图\s*13-(\d+)(?:：|　|\s)', s, re.M)
check('sequential captions 13-1 to 13-46', captions == [str(i) for i in range(1, fig_count + 1)])

imgs = re.findall(r'!\[[^\n]*\]\((ch13/[^)]+)\)', s)
check(
    'active illustrations match figure-index.json',
    imgs == [row.get('asset') or f"ch13/{row['file']}" for row in fig_index] and len(imgs) == 46,
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
check('all footnotes used and defined once', set(refs) <= set(defs) and len(defs) == len(set(defs)) and len(defs) == 42)

for name, key in [('sources.json', 'sources'), ('manifest.json', 'outputs')]:
    data = json.loads((HERE / name).read_text(encoding='utf-8'))[key]
    mismatches = [row['path'] for row in data if 'path' in row and sha256(ROOT / row['path']) != row['sha256']]
    check(name + ' integrity', not mismatches)

math_val = json.loads((HERE / 'math-validation.json').read_text(encoding='utf-8'))
check('KaTeX math validation zero errors', not math_val.get('errors'))

layout_val = json.loads((HERE / 'teaching-layout-validation.json').read_text(encoding='utf-8'))
if isinstance(layout_val, list):
    check('layout validation zero warnings', len(layout_val) == 46 and all(not r.get('warnings') for r in layout_val))
else:
    check('layout validation zero warnings', layout_val.get('total_warnings', 0) == 0 and layout_val.get('all_passed', False))

check('HTML preview generated and non-empty', '<html' in page and len(page) > 10000)

calc_kv = json.loads((ROOT / 'calculations/results/ch13-storage-kv-io.json').read_text(encoding='utf-8'))['summary']
calc_dedup = json.loads((ROOT / 'calculations/results/ch13-data-funnel-dedup.json').read_text(encoding='utf-8'))['summary']
exp_13_02 = json.loads((ROOT / 'experiments/ch13/13-02/expected/metrics.json').read_text(encoding='utf-8'))
check(
    'golden calc storage-kv-io invariants match manuscript',
    calc_kv['gds_bandwidth_gbps'] == 100.0
    and calc_kv['gds_140gb_load_seconds'] == 1.4
    and calc_kv['threefs_180node_read_tibps'] == 6.6
    and calc_kv['ref_32layer_block_vs_layer_speedup_x'] == 8.192
    and calc_kv['qwen3_8b_kv_bytes_per_token'] == 147456
    and calc_kv['kv_write_admission_max_pct_waf1_5'] == 8.89
    and calc_kv['kv_write_admission_max_pct_zns_waf1_0'] == 13.33
    and '147{,}456' in s
    and '8.192' in s
    and '8.89%' in s,
)
check(
    'golden calc data-funnel-dedup invariants match manuscript and 13-02',
    calc_dedup['final_curated_corpus_pb'] == 2.4
    and calc_dedup['net_byte_retention_pct'] == 2.4
    and calc_dedup['final_curated_tokens_trillion'] == 15.0
    and calc_dedup['b16_r8_inflection_s_star'] == 0.7071
    and calc_dedup['b16_r8_p_at_0_80_pct'] == 94.7
    and calc_dedup['minhash_64gpu_shuffle_gb'] == 2520.0
    and calc_dedup['gpu_vs_cpu_wall_speedup_x'] == 80.0
    and calc_dedup['gpu_vs_cpu_cost_reduction_x'] == 4.5
    and exp_13_02['curation_funnel']['after_decontam_pb'] == 2.4
    and '0.7071' in s
    and '2{,}520.0' in s,
)

exp_dir = ROOT / 'experiments' / 'ch13'
for suite in sorted(exp_dir.glob('*')):
    if suite.is_dir() and (suite / 'verify.py').exists():
        res = subprocess.run([sys.executable, 'verify.py'], cwd=suite, capture_output=True, text=True)
        check(f'experiment suite {suite.name} verify.py', res.returncode == 0)

failed = [c for c in checks if not c['passed']]
print(json.dumps({"total": len(checks), "passed": len(checks) - len(failed), "failed": failed}, ensure_ascii=False, indent=2))
if failed:
    sys.exit(1)
