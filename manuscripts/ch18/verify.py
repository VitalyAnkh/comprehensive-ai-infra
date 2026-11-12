#!/usr/bin/env python3
"""Verify Chapter 18 structure, figure layout, mathematical formulas, and numerical examples."""
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


md = HERE.parent / '18-自主智能体系统.md'
s = md.read_text(encoding='utf-8')
page = preview_path(md).read_text(encoding='utf-8')

expected_headings = [
    "18.1", "18.1.1", "18.1.2",
    "18.2", "18.2.1", "18.2.2",
    "18.3", "18.3.1", "18.3.2",
    "18.4", "18.4.1", "18.4.2",
    "18.5", "18.5.1", "18.5.2",
    "18.6", "18.6.1", "18.6.2",
]
found_headings = re.findall(r'^#{2,3} (18\.\d+(?:\.\d+)?) ', s, re.M)
check('all sections and subsections match outline', found_headings == expected_headings)
check('chapter line count meets Path H2 target (>= 1671)', len(s.splitlines()) >= 1671)
check('zero H5 headings and zero digit-prefixed H4 headings', not re.findall(r'^#####\s', s, re.M) and not re.findall(r'^####\s+\d', s, re.M))
numbered_titles = re.findall(r'^#{2,3}\s+18\.\d+(?:\.\d+)?\s+(.+)$', s, re.M)
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
check('zero internal scaffolding or wrong part taxonomy tokens', not re.search(r'六大部分|六部二十四章|第六部分|I19|i19|TC-\d+|Paper\s*#\d+|\\tag\{', s))

h2s = re.findall(r'^## (.+)$', s, re.M)
check('last H2 is 本章小结 and total H2 count is 9 (6 numbered + 3 backmatter)', len(h2s) == 9 and h2s[-1].strip() == '本章小结')

exercises = re.findall(r'^(?:>\s*)?\*\*(?:(?:练习|实验|习题|拓展练习|思考题)\s+)?18-(\d+)\b(?!\s*解答)', s, re.M)
check('sequential exercises 18-1 to 18-16', exercises == [str(i) for i in range(1, 16 + 1)])
answers = re.findall(r'^####\s+(?:练习|实验)\s+18-(\d+)\s+解答', s, re.M)
check('all 16 exercises have step-by-step derivations', answers == [str(i) for i in range(1, 16 + 1)])

py_blocks = re.findall(r'^```python\b', s, re.M)
cpp_blocks = re.findall(r'^```cpp\b', s, re.M)
check('expected production code blocks (10 Python)', len(py_blocks) == 10 and len(cpp_blocks) == 0)

worked_examples = re.findall(r'^>\s*\*\*算例\s*18-(\d+)\b', s, re.M)
check('sequential blockquote worked examples 18-1 to 18-10', worked_examples == [str(i) for i in range(1, 10 + 1)])

table_captions = re.findall(r'^\*表\s*18-(\d+)(?:：|　|\s)', s, re.M)
check('sequential italic table captions 18-1 to 18-9', table_captions == [str(i) for i in range(1, 9 + 1)])

fig_index = json.loads((HERE / 'figure-index.json').read_text(encoding='utf-8'))
fig_count = len(fig_index)
check('expected figures in figure-index.json (22)', fig_count == 22)
captions = re.findall(r'^\*图\s*18-(\d+)(?:：|　|\s)', s, re.M)
check('sequential captions 18-1 to 18-22', captions == [str(i) for i in range(1, fig_count + 1)])

imgs = re.findall(r'!\[[^\n]*\]\((ch18/[^)]+)\)', s)
check(
    'active illustrations match figure-index.json',
    imgs == [row.get('asset') or f"ch18/{row['file']}" for row in fig_index] and len(imgs) == 22,
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
check('all footnotes used and defined once (26)', set(refs) == set(defs) and len(defs) == len(set(defs)) and len(defs) == 26)

golden = json.loads((ROOT / 'calculations/results/ch18-agent-worm-grammar.json').read_text(encoding='utf-8'))
g_sum = golden['summary']
check(
    'golden calc.py ch18-agent-worm-grammar metrics locked in manuscript',
    math.isclose(g_sum['worm_40turn_prefix_reuse_pct'], 95.12)
    and math.isclose(g_sum['worm_prefill_to_decode_ratio'], 205.0)
    and math.isclose(g_sum['cachegen_vs_raw_transfer_speedup_x'], 4.01)
    and math.isclose(g_sum['firecracker_fork_ms'], 12.0)
    and g_sum['qwen3_bitmask_uint32_words'] == 4748
    and g_sum['qwen3_bitmask_bytes_per_req'] == 18992
    and g_sum['deepseek_bitmask_bytes_per_req'] == 16160
    and math.isclose(g_sum['qwen3_batch64_bitmask_mib'], 1.16)
    and math.isclose(g_sum['naive_fsm_throughput_drop_pct'], 81.9)
    and math.isclose(g_sum['xgrammar_batch64_cpu_ms'], 2.1)
    and math.isclose(g_sum['k16_required_branch_percentile'], 99.937)
    and math.isclose(g_sum['k12_barrier_join_p99_seconds'], 19.8)
    and math.isclose(g_sum['k12_quorum_slack_p99_seconds'], 7.4)
    and math.isclose(g_sum['speculative_tool_saved_latency_ms'], 210.7)
    and math.isclose(g_sum['speculative_tool_wasted_query_pct'], 16.3)
    and all(tok in s for tok in ['95.12', '2.16', '18.0', '4.01', '12.0', '4{,}748', '18{,}992', '16{,}160', '1.16', '81.9', '2.10', '99.937', '19.8', '7.4', '210.7', '16.3']),
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

exp_dir = ROOT / 'experiments' / 'ch18'
exp_scaffolding_hits = []
for suite in sorted(exp_dir.glob('*')):
    if suite.is_dir():
        for fp in sorted(suite.rglob('*')):
            if fp.suffix in {'.py', '.json', '.md'} and '__pycache__' not in fp.parts:
                if re.search(r'I19|i19_|Example 21-|Chapter 21', fp.read_text(encoding='utf-8')):
                    exp_scaffolding_hits.append(str(fp))
    if suite.is_dir() and (suite / 'verify.py').exists():
        res = subprocess.run([sys.executable, 'verify.py'], cwd=suite, capture_output=True, text=True)
        check(f'experiment suite {suite.name} verify.py', res.returncode == 0)
check('experiment ch18 files free of legacy I19 or Ch21 scaffolding', not exp_scaffolding_hits)

failed = [c for c in checks if not c['passed']]
print(json.dumps({"total": len(checks), "passed": len(checks) - len(failed), "failed": failed}, ensure_ascii=False, indent=2))
if failed:
    sys.exit(1)
