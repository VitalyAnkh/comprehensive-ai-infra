#!/usr/bin/env python3
"""Verify Chapter 14 structure, figure layout, mathematical formulas, and numerical examples."""
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


md = HERE.parent / '14-模型与系统硬件协同设计.md'
s = md.read_text()
page = preview_path(md).read_text()

expected_headings = [
    '14.1', '14.1.1', '14.1.2', '14.1.3',
    '14.2', '14.2.1', '14.2.2', '14.2.3',
    '14.3', '14.3.1', '14.3.2',
    '14.4', '14.4.1', '14.4.2',
    '14.5', '14.5.1', '14.5.2', '14.5.3',
    '14.6', '14.6.1', '14.6.2',
    '14.7', '14.7.1', '14.7.2',
    '14.8', '14.8.1', '14.8.2',
    '14.9', '14.9.1', '14.9.2', '14.9.3',
]
found_headings = re.findall(r'^#{2,3} (14\.\d+(?:\.\d+)?) ', s, re.M)
check('all nine sections and twenty-two subsections match outline', found_headings == expected_headings)

h2s = re.findall(r'^## (.+)$', s, re.M)
check('last H2 is 本章小结', len(h2s) >= 7 and h2s[-1].strip() == '本章小结')

exercises = re.findall(r'^>\s*\*\*(?:练习|实验) 14-(\d+)[：:]', s, re.M)
check('seventeen exercises 14-1 to 14-17 in order', exercises == [str(i) for i in range(1, 18)])

code_blocks = re.findall(r'```python\n.*?```', s, re.S)
check('three production Python code blocks', len(code_blocks) == 3)

worked_examples = re.findall(r'^>\s*\*\*算例\s*14-', s, re.M)
check('thirteen standardized blockquote worked examples', len(worked_examples) == 13)

table_captions = re.findall(r'^\*表 14-(\d+)　', s, re.M)
check('eleven standardized italic table captions 14-1 to 14-11', table_captions == [str(i) for i in range(1, 12)])

fig_index = json.loads((HERE / 'figure-index.json').read_text())
fig_count = len(fig_index)
check('twenty-six figures in figure-index.json', fig_count == 26)
captions = re.findall(r'^\*图 14-(\d+)(?:：|　)', s, re.M)
check('sequential external captions 14-1 to 14-26', captions == [str(i) for i in range(1, fig_count + 1)])

imgs = re.findall(r'!\[[^\]]*\]\((ch14/[^)]+)\)', s)
check('active illustrations match figure-index.json', imgs == [row.get('asset') or f"ch14/{row['file']}" for row in fig_index])

for rel in imgs:
    p = md.parent / rel
    text_svg = ''.join(ET.parse(p).getroot().itertext())
    check('no internal figure numbering: ' + p.stem, re.search(r'图\s*\d+\s*[-－–]\s*\d+', text_svg) is None)
    check('SVG PNG PDF available: ' + p.stem, all(p.with_suffix(e).exists() for e in ['.svg', '.png', '.pdf']))
    check('PDF is binary not LFS pointer: ' + p.stem, p.with_suffix('.pdf').read_bytes().startswith(b'%PDF'))

missing = []
for p in [md, HERE / 'README.md', HERE / 'reading-notes.md']:
    for link in re.findall(r'\]\(([^)]+)\)', p.read_text()):
        u = urlsplit(link)
        if not u.scheme and u.path and not (p.parent / unquote(u.path)).exists():
            missing.append((str(p), link))
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
    len(layout_val) == 26 and all(r['width_pt'] == 420 and r['min_label_pt'] >= 11 and not r['text_extent_warnings'] for r in layout_val),
)

d = json.loads((HERE / 'figure-data.json').read_text())
e1 = d['example_14_1_v41_dual_track']
check(
    'example 14-1 V4.1 dual-track arithmetic',
    close(32768 * e1['v41_global_b_per_tok'] / (1024 ** 2), e1['32k_v41_global_mib'])
    and close(round(e1['v4_persistent_b_per_tok'] / e1['v41_global_b_per_tok'], 2), e1['resident_ratio'])
    and close(round(e1['1m_v4_read_gib'] / e1['1m_v41_read_gib'], 2), e1['1m_read_ratio']),
)

e2 = d['example_14_2_digital_worker']
check(
    'example 14-2 digital worker audit arithmetic',
    close(93 * 0.15, e2['k3_fixed_layer_overhead_ms'])
    and e2['v41_200k_b200_usd_per_m'] == 5.97
    and e2['k3_1m_b200_usd_per_m'] == 496.08,
)

e3 = d['example_14_3_opentallas']
check(
    'example 14-3 OpenTallas step and hop arithmetic',
    close(e3['wafer_hops'] * e3['hop_ns'] * e3['layers'] / 1000.0, e3['wafer_comm_floor_us'])
    and e3['wafer_comm_floor_us'] > e3['step_budget_us']
    and e3['machines_6mo'] == 2 * e3['machines_1yr'],
)

e4 = d['example_14_4_counterfactual']
check(
    'example 14-4 counterfactual SWA and candidate pool',
    e4['candidate_pool_tokens'] == 8 * e4['reindex_k']
    and close(e4['swa_state_mib'], 2.578),
)

e11 = d['example_14_11_hwnas_hybrid_layers']
check(
    'example 14-11 HW-NAS hybrid layer allocation arithmetic',
    e11['optimal_full_layers'] + e11['optimal_linear_layers'] == e11['total_layers']
    and close(
        (e11['optimal_full_layers'] * e11['full_layer_mib'] + e11['optimal_linear_layers'] * e11['linear_layer_mib']) / 1024.0,
        e11['resident_state_gib'],
    )
    and e11['resident_state_gib'] <= e11['kv_budget_gib']
    and e11['step_tpot_ms'] <= e11['tpot_budget_ms'],
)

e12 = d['example_14_12_sparse_upcycling']
check(
    'example 14-12 Sparse Upcycling compute saving arithmetic',
    close(6.0 * e12['active_params_b'] * 1e9 * e12['scratch_tokens_t'] * 1e12 / 1e24, e12['scratch_flops_e24'])
    and close(round((e12['scratch_flops_e24'] - e12['migrate_flops_e24']) / e12['scratch_flops_e24'] * 100.0, 2), e12['compute_saving_pct']),
)

check('zero exclamation marks', '！' not in s)

result = {
    "passed": all(x['passed'] for x in checks),
    "checks": checks,
    "missing_links": missing,
}
(HERE / 'validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')

failed = [c for c in checks if not c['passed']]
print(json.dumps({"passed": not failed, "checks": len(checks), "failures": failed}, ensure_ascii=False, indent=2))
sys.exit(1 if failed else 0)
