#!/usr/bin/env python3
"""Verify Chapter 21 structure, figure layout, mathematical formulas, and numerical examples."""
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


md = HERE.parent / '21-推荐检索与向量数据库.md'
s = md.read_text(encoding='utf-8')
page = preview_path(md).read_text(encoding='utf-8')

expected_headings = [
    "21.1", "21.1.1", "21.1.2",
    "21.2", "21.2.1", "21.2.2",
    "21.3", "21.3.1", "21.3.2",
    "21.4", "21.4.1", "21.4.2",
    "21.5", "21.5.1", "21.5.2",
    "21.6", "21.6.1", "21.6.2",
]
found_headings = re.findall(r'^#{2,3} (21\.\d+(?:\.\d+)?) ', s, re.M)
check('all sections and subsections match outline', found_headings == expected_headings)
check('chapter line count meets Path H2 target (>= 1550)', len(s.splitlines()) >= 1550)
check('zero H5 headings and zero digit-prefixed H4 headings', not re.findall(r'^#####\s', s, re.M) and not re.findall(r'^####\s+\d', s, re.M))
numbered_titles = re.findall(r'^#{2,3}\s+21\.\d+(?:\.\d+)?\s+(.+)$', s, re.M)
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
    'zero internal scaffolding or stale chapter/model tokens',
    not re.search(
        r'六大部分|六部二十四章|第六部分|25\.\d+\s*节|^#{2,3}\s+25\.|算例\s*25-|ch25-|第\s*26\s*章|双向奔赴|打满|拉满|推满|水位|死死|暴降|骤降|惊人|锁死|打爆|抓手|卡死|暴跌|暴涨|暴增|闭环|！|file://|固定开销',
        s,
        re.M,
    ),
)

h2s = re.findall(r'^## (.+)$', s, re.M)
check('last H2 is 本章小结 and total H2 count is 9 (6 numbered + 3 backmatter)', len(h2s) == 9 and h2s[-1].strip() == '本章小结')

exercises = re.findall(r'^(?:>\s*)?\*\*(?:(?:练习|实验|习题|拓展练习|思考题)\s+)?21-(\d+)\b(?!\s*解答)', s, re.M)
check('sequential exercises 21-1 to 21-16', exercises == [str(i) for i in range(1, 16 + 1)])
answers = [
    m.group(1) or m.group(2)
    for m in re.finditer(r'^(?:-\s*\*\*21-(\d+)\s*解答\*\*|####\s+(?:练习|实验)\s+21-(\d+)\s+解答)', s, re.M)
]
check('all 16 exercises have step-by-step derivations', answers == [str(i) for i in range(1, 16 + 1)])

py_blocks = re.findall(r'^```python\b', s, re.M)
cpp_blocks = re.findall(r'^```cpp\b', s, re.M)
check('expected production code blocks (10 Python)', len(py_blocks) == 10 and len(cpp_blocks) == 0)

worked_examples = re.findall(r'^>\s*\*\*算例\s*21-(\d+)\b', s, re.M)
check('sequential blockquote worked examples 21-1 to 21-10', worked_examples == [str(i) for i in range(1, 10 + 1)])

table_captions = re.findall(r'^\*表\s*21-(\d+)(?:：|　|\s)', s, re.M)
check('sequential italic table captions 21-1 to 21-9', table_captions == [str(i) for i in range(1, 9 + 1)])

fig_index = json.loads((HERE / 'figure-index.json').read_text(encoding='utf-8'))
fig_count = len(fig_index)
check('expected figures in figure-index.json (22)', fig_count == 22)
captions = re.findall(r'^\*图\s*21-(\d+)(?:：|　|\s)', s, re.M)
check('sequential captions 21-1 to 21-22', captions == [str(i) for i in range(1, fig_count + 1)])

imgs = re.findall(r'!\[[^\n]*\]\((ch21/[^)]+)\)', s)
check(
    'active illustrations match figure-index.json',
    imgs == [row.get('asset') or f"ch21/{row['file']}" for row in fig_index] and len(imgs) == 22,
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
for doc in [md, ROOT / 'case-studies/generative-recsys-and-colpali-rag.md']:
    doc_s = doc.read_text(encoding='utf-8')
    for link in re.findall(r'\]\(([^)]+)\)', doc_s):
        u = urlsplit(link.strip('<>'))
        if not u.scheme and u.path and not (doc.parent / unquote(u.path)).exists():
            missing.append((str(doc), link))
check('local links exist (manuscript and new case study)', not missing)
check(
    'required Chapter 21 case studies linked',
    all(cs in s for cs in ['generative-recsys-and-colpali-rag.md', 'retrieval-and-generation.md', 'memory-criticality-and-tiering.md']),
)

defs = re.findall(r'^\[\^([^\]]+)\]:', s, re.M)
refs = re.findall(r'\[\^([^\]]+)\](?!:)', s)
check('all footnotes used and defined once (23)', set(refs) == set(defs) and len(defs) == len(set(defs)) and len(defs) == 23)

golden = json.loads((ROOT / 'calculations/results/ch21-recsys-vector-colpali.json').read_text(encoding='utf-8'))
g_sum = golden['summary']
check(
    'golden calc.py ch21-recsys-vector-colpali metrics locked in manuscript',
    math.isclose(g_sum['dlrm_3tier_weighted_lookup_us'], 2.19)
    and math.isclose(g_sum['dlrm_3tier_vs_ssd_speedup_x'], 29.67)
    and math.isclose(g_sum['hstu_stochastic_length_saving_pct'], 56.2)
    and math.isclose(g_sum['mfalcon_m1024_flops_reduction_x'], 44.8)
    and g_sum['mfalcon_h100_qps'] == 485
    and math.isclose(g_sum['colpali_bf16_kb_per_page'], 263.68)
    and math.isclose(g_sum['colpali_50m_bf16_tb'], 13.18)
    and math.isclose(g_sum['colpali_100m_bf16_tb'], 26.37)
    and math.isclose(g_sum['colpali_1b_bf16_tb'], 263.7)
    and math.isclose(g_sum['plaid_2bit_bytes_per_vec'], 34.0)
    and math.isclose(g_sum['plaid_2bit_50m_tb'], 1.75)
    and math.isclose(g_sum['plaid_2bit_compression_x'], 7.53)
    and math.isclose(g_sum['plaid_1bit_bytes_per_vec'], 17.5)
    and math.isclose(g_sum['plaid_1bit_50m_gb'], 901.3)
    and math.isclose(g_sum['plaid_1bit_100m_tb'], 1.8)
    and math.isclose(g_sum['plaid_1bit_1b_tb'], 18.0)
    and math.isclose(g_sum['plaid_1bit_compression_x'], 14.6)
    and math.isclose(g_sum['plaid_maxsim_p99_ms'], 14.2)
    and math.isclose(g_sum['hybrid_bm25_cuvs_ce_p99_ms'], 27.6)
    and math.isclose(g_sum['cuckoo_tau5_embedding_shrink_pct'], 68.5)
    and math.isclose(g_sum['cuckoo_embedding_active_gb'], 756.0)
    and math.isclose(g_sum['hot_update_rwlock_p99_ms'], 18.4)
    and math.isclose(g_sum['hot_update_epoch_lockfree_p99_ms'], 1.15)
    and math.isclose(g_sum['hot_update_tail_reduction_x'], 16.0)
    and math.isclose(g_sum['hot_update_epoch_memory_overhead_pct'], 6.2)
    and all(
        tok in s
        for tok in [
            '2.19', '29.67', '56.2', '44.80', '485',
            '263.68', '13.18', '26.37', '263.7',
            '34.0', '1.75', '7.53', '17.5', '901.3', '1.80', '18.0', '14.6',
            '14.2', '27.6', '68.5', '756.0', '18.4', '1.15', '16.0', '6.2',
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

exp_dir = ROOT / 'experiments' / 'ch21'
exp_scaffolding_hits = []
for suite in sorted(exp_dir.glob('*')):
    if suite.is_dir():
        for fp in sorted(suite.rglob('*')):
            if fp.suffix in {'.py', '.json', '.md'} and '__pycache__' not in fp.parts:
                txt = fp.read_text(encoding='utf-8')
                if re.search(r'Chapter 25|Ex\s*25-\d+|Section\s*25\.|25\.[1-6]\s*节|25\.\d+\.\d+', txt):
                    exp_scaffolding_hits.append(str(fp))
    if suite.is_dir() and (suite / 'verify.py').exists():
        res = subprocess.run([sys.executable, 'verify.py'], cwd=suite, capture_output=True, text=True)
        check(f'experiment suite {suite.name} verify.py', res.returncode == 0)
check('experiment ch21 files free of legacy Ch25 scaffolding', not exp_scaffolding_hits)

failed = [c for c in checks if not c['passed']]
print(json.dumps({"total": len(checks), "passed": len(checks) - len(failed), "failed": failed}, ensure_ascii=False, indent=2))
if failed:
    sys.exit(1)
