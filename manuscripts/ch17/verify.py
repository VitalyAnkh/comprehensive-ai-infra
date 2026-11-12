#!/usr/bin/env python3
"""Verify Chapter 17 structure, figure layout, mathematical formulas, and numerical examples."""
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


md = HERE.parent / '17-后训练与强化学习系统.md'
s = md.read_text(encoding='utf-8')
page = preview_path(md).read_text(encoding='utf-8')

expected_headings = [
    "17.1", "17.1.1", "17.1.2",
    "17.2", "17.2.1", "17.2.2",
    "17.3", "17.3.1", "17.3.2", "17.3.3",
    "17.4", "17.4.1", "17.4.2", "17.4.3",
    "17.5", "17.5.1", "17.5.2",
    "17.6", "17.6.1", "17.6.2",
]
found_headings = re.findall(r'^#{2,3} (17\.\d+(?:\.\d+)?) ', s, re.M)
check('all sections and subsections match outline', found_headings == expected_headings)
check('chapter line count meets Path H2 target (>= 1904)', len(s.splitlines()) >= 1904)
check('zero H5 headings and zero digit-prefixed H4 headings', not re.findall(r'^#####\s', s, re.M) and not re.findall(r'^####\s+\d', s, re.M))
numbered_titles = re.findall(r'^#{2,3}\s+17\.\d+(?:\.\d+)?\s+(.+)$', s, re.M)
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
check('zero internal scaffolding or wrong part taxonomy tokens', not re.search(r'六大部分|六部二十四章|第六部分|I19|i19_|TC-\d+|Paper\s*#\d+|\\tag\{', s))

h2s = re.findall(r'^## (.+)$', s, re.M)
check('last H2 is 本章小结 and total H2 count is 9 (6 numbered + 3 backmatter)', len(h2s) == 9 and h2s[-1].strip() == '本章小结')

exercises = re.findall(r'^(?:>\s*)?\*\*(?:(?:练习|实验|习题|拓展练习|思考题)\s+)?17-(\d+)\b(?!\s*解答)', s, re.M)
check('sequential exercises 17-1 to 17-20', exercises == [str(i) for i in range(1, 20 + 1)])
answers = re.findall(r'^-\s*\*\*17-(\d+)\s*解答', s, re.M)
check('all 20 exercises have step-by-step derivations', answers == [str(i) for i in range(1, 20 + 1)])

py_blocks = re.findall(r'^```python\b', s, re.M)
cpp_blocks = re.findall(r'^```cpp\b', s, re.M)
check('expected production code blocks (12 Python)', len(py_blocks) == 12 and len(cpp_blocks) == 0)

worked_examples = re.findall(r'^>\s*\*\*算例\s*17-(\d+)\b', s, re.M)
check('sequential blockquote worked examples 17-1 to 17-13', worked_examples == [str(i) for i in range(1, 13 + 1)])

table_captions = re.findall(r'^\*表\s*17-(\d+)(?:：|　|\s)', s, re.M)
check('sequential italic table captions 17-1 to 17-11', table_captions == [str(i) for i in range(1, 11 + 1)])

fig_index = json.loads((HERE / 'figure-index.json').read_text(encoding='utf-8'))
fig_count = len(fig_index)
check('expected figures in figure-index.json (26)', fig_count == 26)
captions = re.findall(r'^\*图\s*17-(\d+)(?:：|　|\s)', s, re.M)
check('sequential captions 17-1 to 17-26', captions == [str(i) for i in range(1, fig_count + 1)])

imgs = re.findall(r'!\[[^\n]*\]\((ch17/[^)]+)\)', s)
check(
    'active illustrations match figure-index.json',
    imgs == [row.get('asset') or f"ch17/{row['file']}" for row in fig_index] and len(imgs) == 26,
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
check('all footnotes used and defined once (25)', set(refs) == set(defs) and len(defs) == len(set(defs)) and len(defs) == 25)

golden = json.loads((ROOT / 'calculations/results/ch17-posttrain-rlvr-distill.json').read_text(encoding='utf-8'))
g_sum = golden['summary']
check(
    'golden calc.py ch17-posttrain-rlvr-distill metrics locked in manuscript',
    math.isclose(g_sum['ppo_32b_static_state_gb'], 1152.0)
    and math.isclose(g_sum['grpo_32b_static_state_gb'], 576.0)
    and math.isclose(g_sum['hybrid_engine_reshard_speedup_x'], 45.0)
    and math.isclose(g_sum['rollpacker_stream_overlap_seconds'], 55.0)
    and math.isclose(g_sum['sft_pad_to_64k_mfu_pct'], 13.6)
    and math.isclose(g_sum['sft_varlen_zigzag_mfu_pct'], 56.4)
    and math.isclose(g_sum['sft_varlen_throughput_speedup_x'], 4.15)
    and math.isclose(g_sum['qwen_200b_full_logits_bf16_pb'], 60.7744)
    and math.isclose(g_sum['top128_unpacked_6b_tb'], 153.6)
    and math.isclose(g_sum['top128_bitpacked_32bit_tb'], 102.4)
    and math.isclose(g_sum['top128_compression_vs_full_x'], 593.5)
    and math.isclose(g_sum['offline_vs_online_teacher_crossover_epochs'], 1.88)
    and math.isclose(g_sum['agentic_rl_sync_gpu_idle_pct'], 68.0)
    and math.isclose(g_sum['agentic_rl_async_bounded_speedup_x'], 2.64)
    and all(tok in s for tok in ['1{,}152', '576', '45.0', '55', '13.6', '56.4', '4.15', '153.6', '102.4', '593.5', '1.88', '68', '2.64']),
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

exp_dir = ROOT / 'experiments' / 'ch17'
exp_scaffolding_hits = []
for suite in sorted(exp_dir.glob('*')):
    if suite.is_dir():
        for fp in sorted(suite.rglob('*')):
            if fp.suffix in {'.py', '.json', '.md'} and '__pycache__' not in fp.parts:
                if re.search(r'I19|i19_|Example 18-|Chapter 18', fp.read_text(encoding='utf-8')):
                    exp_scaffolding_hits.append(str(fp))
    if suite.is_dir() and (suite / 'verify.py').exists():
        res = subprocess.run([sys.executable, 'verify.py'], cwd=suite, capture_output=True, text=True)
        check(f'experiment suite {suite.name} verify.py', res.returncode == 0)
check('experiment ch17 files free of legacy Ch18 scaffolding', not exp_scaffolding_hits)

failed = [c for c in checks if not c['passed']]
print(json.dumps({"total": len(checks), "passed": len(checks) - len(failed), "failed": failed}, ensure_ascii=False, indent=2))
if failed:
    sys.exit(1)
