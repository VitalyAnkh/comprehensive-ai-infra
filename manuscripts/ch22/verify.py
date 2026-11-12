#!/usr/bin/env python3
"""Verify Chapter 22 structure, figure layout, mathematical formulas, and numerical examples."""
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


md = HERE.parent / '22-科学计算智能系统.md'
s = md.read_text(encoding='utf-8')
page = preview_path(md).read_text(encoding='utf-8')

expected_headings = [
    "22.1", "22.1.1", "22.1.2",
    "22.2", "22.2.1", "22.2.2",
    "22.3", "22.3.1", "22.3.2",
    "22.4", "22.4.1", "22.4.2",
    "22.5", "22.5.1", "22.5.2",
    "22.6", "22.6.1", "22.6.2",
]
found_headings = re.findall(r'^#{2,3} (22\.\d+(?:\.\d+)?) ', s, re.M)
check('all sections and subsections match outline', found_headings == expected_headings)
check('chapter line count meets Path H2 target (>= 1538)', len(s.splitlines()) >= 1538)
check('zero H5 headings and zero digit-prefixed H4 headings', not re.findall(r'^#####\s', s, re.M) and not re.findall(r'^####\s+\d', s, re.M))
numbered_titles = re.findall(r'^#{2,3}\s+22\.\d+(?:\.\d+)?\s+(.+)$', s, re.M)
check(
    'numbered H2/H3 headings within 8-26 CJK characters',
    bool(numbered_titles) and all(8 <= len(re.findall(r'[\u4e00-\u9fff]', t)) <= 26 for t in numbered_titles),
)
s_no_fences = re.sub(r'```[\s\S]*?```', '', s)
inline_dollar_spans = [
    m.group(0)
    for line in s_no_fences.splitlines()
    for m in re.finditer(r'(`+)([^`\n]+)\1', line)
    if '$' in m.group(2)
]
check('zero bare $ inside inline backticks', not inline_dollar_spans)
check(
    'zero internal scaffolding, meta-repo counts, or banned buzzwords',
    not re.search(
        r'六大部分|六部二十四章|第六部分|^#{2,3}\s+24\.|算例\s*24-|ch24-|EncyclopediaGrandLedger|883\s*幅|884\s*幅|147\s*套|\bI1[0-9]\b|\bI2[01]\b|双向奔赴|打满|拉满|推满|水位|死死|暴降|骤降|惊人|锁死|打爆|抓手|卡死|暴跌|暴涨|暴增|闭环|！|file://',
        s,
        re.M,
    ),
)

h2s = re.findall(r'^## (.+)$', s, re.M)
check('last H2 is 本章小结 and total H2 count is 9 (6 numbered + 3 backmatter)', len(h2s) == 9 and h2s[-1].strip() == '本章小结')

exercises = re.findall(r'^(?:>\s*)?\*\*(?:(?:练习|实验|习题|拓展练习|思考题)\s+)?22-(\d+)\b(?!\s*解答)', s, re.M)
check('sequential exercises 22-1 to 22-16', exercises == [str(i) for i in range(1, 16 + 1)])
answers = [
    m.group(1) or m.group(2)
    for m in re.finditer(r'^(?:-\s*\*\*22-(\d+)\s*解答\*\*|####\s+(?:练习|实验)\s+22-(\d+)\s+解答)', s, re.M)
]
check('all 16 exercises have step-by-step derivations', answers == [str(i) for i in range(1, 16 + 1)])

py_blocks = re.findall(r'^```python\b', s, re.M)
cpp_blocks = re.findall(r'^```cpp\b', s, re.M)
check('expected production code blocks (10 Python)', len(py_blocks) == 10 and len(cpp_blocks) == 0)

worked_examples = re.findall(r'^>\s*\*\*算例\s*22-(\d+)\b', s, re.M)
check('sequential blockquote worked examples 22-1 to 22-10', worked_examples == [str(i) for i in range(1, 10 + 1)])

table_captions = re.findall(r'^\*表\s*22-(\d+)(?:：|　|\s)', s, re.M)
check('sequential italic table captions 22-1 to 22-9', table_captions == [str(i) for i in range(1, 9 + 1)])

fig_index = json.loads((HERE / 'figure-index.json').read_text(encoding='utf-8'))
fig_count = len(fig_index)
check('expected figures in figure-index.json (22)', fig_count == 22)
captions = re.findall(r'^\*图\s*22-(\d+)(?:：|　|\s)', s, re.M)
check('sequential captions 22-1 to 22-22', captions == [str(i) for i in range(1, fig_count + 1)])

imgs = re.findall(r'!\[[^\n]*\]\((ch22/[^)]+)\)', s)
check(
    'active illustrations match figure-index.json',
    imgs == [row.get('asset') or f"ch22/{row['file']}" for row in fig_index] and len(imgs) == 22,
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
for doc in [md, ROOT / 'case-studies/scientific-ai-and-ozaki-fp64.md']:
    doc_s = doc.read_text(encoding='utf-8')
    for link in re.findall(r'\]\(([^)]+)\)', doc_s):
        u = urlsplit(link.strip('<>'))
        if not u.scheme and u.path and not (doc.parent / unquote(u.path)).exists():
            missing.append((str(doc), link))
check('local links exist (manuscript and new case study)', not missing)
check(
    'required Chapter 22 case studies linked',
    all(cs in s for cs in ['scientific-ai-and-ozaki-fp64.md', 'attention-tiles-and-io.md', 'mesh-shape-and-slicing.md']),
)

defs = re.findall(r'^\[\^([^\]]+)\]:', s, re.M)
refs = re.findall(r'\[\^([^\]]+)\](?!:)', s)
check('all footnotes used and defined once (26)', set(refs) == set(defs) and len(defs) == len(set(defs)) and len(defs) == 26)

golden = json.loads((ROOT / 'calculations/results/ch22-science-ozaki-coupling.json').read_text(encoding='utf-8'))
g_sum = golden['summary']
check(
    'golden calc.py ch22-science-ozaki-coupling metrics locked in manuscript',
    g_sum['science_domains_covered'] == 5
    and math.isclose(g_sum['pairformer_4096_naive_gib'], 512.0)
    and math.isclose(g_sum['pairformer_4096_dap4_workspace_gib'], 0.25)
    and math.isclose(g_sum['pairformer_workspace_reduction_x'], 2048.0)
    and math.isclose(g_sum['pairformer_dap_comm_to_comp_pct'], 8.26)
    and math.isclose(g_sum['weather_64gpu_halo_mb'], 5.53)
    and math.isclose(g_sum['weather_64gpu_halo_comm_to_comp_pct'], 1.27)
    and g_sum['ozaki_signed_int8_max_k_no_overflow'] == 133144
    and g_sum['ozaki_s7_int8_gemms'] == 28
    and math.isclose(g_sum['ozaki_s7_effective_mantissa_bits'], 51.8)
    and math.isclose(g_sum['ozaki_int8_to_fp64_peak_ratio_x'], 112.5)
    and math.isclose(g_sum['ozaki_s7_compute_ceiling_x'], 4.02)
    and math.isclose(g_sum['ozaki_vs_native_fp64_speedup_x'], 3.68)
    and math.isclose(g_sum['ozaki_equiv_fp64_tflops'], 147.2)
    and math.isclose(g_sum['coupling_384gb_filesystem_s'], 58.2)
    and math.isclose(g_sum['coupling_384gb_dlpack_insitu_s'], 0.41)
    and math.isclose(g_sum['coupling_insitu_speedup_x'], 142.0)
    and math.isclose(g_sum['unconstrained_500step_mass_drift_pct'], 11.4)
    and math.isclose(g_sum['projected_500step_mass_drift_rel'], 4.2e-13, rel_tol=1e-6)
    and math.isclose(g_sum['conservation_projection_overhead_pct'], 3.1)
    and all(
        tok in s
        for tok in [
            '512.0', '0.25', '2,048', '8.26', '5.53', '1.27',
            '133,144', '28', '51.8', '112.5', '4.02', '3.68', '147.2',
            '58.2', '0.41', '142.0', '11.4', '4.2', '3.1',
        ]
    ),
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

exp_dir = ROOT / 'experiments' / 'ch22'
exp_scaffolding_hits = []
for suite in sorted(exp_dir.glob('*')):
    if suite.is_dir():
        for fp in sorted(suite.rglob('*')):
            if fp.suffix in {'.py', '.json', '.md'} and '__pycache__' not in fp.parts:
                txt = fp.read_text(encoding='utf-8')
                if re.search(r'Chapter 24|Ex\s*24-\d+|Section\s*24\.|24\.[1-6]\s*节|24\.\d+\.\d+', txt):
                    exp_scaffolding_hits.append(str(fp))
    if suite.is_dir() and (suite / 'verify.py').exists():
        res = subprocess.run([sys.executable, 'verify.py'], cwd=suite, capture_output=True, text=True)
        check(f'experiment suite {suite.name} verify.py', res.returncode == 0)
check('experiment ch22 files free of legacy Ch24 scaffolding', not exp_scaffolding_hits)

failed = [c for c in checks if not c['passed']]
print(json.dumps({"total": len(checks), "passed": len(checks) - len(failed), "failed": failed}, ensure_ascii=False, indent=2))
if failed:
    sys.exit(1)
