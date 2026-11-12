#!/usr/bin/env python3
"""Verify Chapter 19 structure, figure layout, mathematical formulas, and numerical examples."""
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


md = HERE.parent / '19-多模态生成与实时语音.md'
s = md.read_text(encoding='utf-8')
page = preview_path(md).read_text(encoding='utf-8')

expected_headings = [
    "19.1", "19.1.1", "19.1.2", "19.1.3",
    "19.2", "19.2.1", "19.2.2", "19.2.3",
    "19.3", "19.3.1", "19.3.2", "19.3.3",
    "19.4", "19.4.1", "19.4.2", "19.4.3",
    "19.5", "19.5.1", "19.5.2",
    "19.6", "19.6.1", "19.6.2",
    "19.7", "19.7.1", "19.7.2",
    "19.8", "19.8.1", "19.8.2",
]
found_headings = re.findall(r'^#{2,3} (19\.\d+(?:\.\d+)?) ', s, re.M)
check('all sections and subsections match outline', found_headings == expected_headings)
check('chapter line count meets Path H2 target (>= 2825)', len(s.splitlines()) >= 2825)
check('zero H5 headings and zero digit-prefixed H4 headings', not re.findall(r'^#####\s', s, re.M) and not re.findall(r'^####\s+\d', s, re.M))
numbered_titles = re.findall(r'^#{2,3}\s+19\.\d+(?:\.\d+)?\s+(.+)$', s, re.M)
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
    'zero internal scaffolding or legacy 22.x / #127 tokens',
    not re.search(r'六大部分|六部二十四章|第六部分|TC-\d+|Paper\s*#\d+|#127\b|\\tag\{22\.|(?<!\d)22\.[1-8](?:\.\d+)?\s*节|抓手|\[\^ch17-', s),
)

h2s = re.findall(r'^## (.+)$', s, re.M)
check('last H2 is 本章小结 and total H2 count is 11 (8 numbered + 3 backmatter)', len(h2s) == 11 and h2s[-1].strip() == '本章小结')

exercises = re.findall(r'^(?:>\s*)?\*\*(?:(?:练习|实验|习题|拓展练习|思考题)\s+)?19-(\d+)\b(?!\s*解答)', s, re.M)
check('sequential exercises 19-1 to 19-28', exercises == [str(i) for i in range(1, 28 + 1)])
answers = [
    m.group(1) or m.group(2)
    for m in re.finditer(r'^(?:-\s*\*\*19-(\d+)\s*解答\*\*|####\s+(?:练习|实验)\s+19-(\d+)\s+解答)', s, re.M)
]
check('all 28 exercises have step-by-step derivations', answers == [str(i) for i in range(1, 28 + 1)])

py_blocks = re.findall(r'^```python\b', s, re.M)
cpp_blocks = re.findall(r'^```cpp\b', s, re.M)
check('expected production code blocks (16 Python)', len(py_blocks) == 16 and len(cpp_blocks) == 0)

worked_examples = re.findall(r'^>\s*\*\*算例\s*19-(\d+)\b', s, re.M)
check('sequential blockquote worked examples 19-1 to 19-15', worked_examples == [str(i) for i in range(1, 15 + 1)])

table_captions = re.findall(r'^\*表\s*19-(\d+)(?:：|　|\s)', s, re.M)
check('sequential italic table captions 19-1 to 19-14', table_captions == [str(i) for i in range(1, 14 + 1)])

fig_index = json.loads((HERE / 'figure-index.json').read_text(encoding='utf-8'))
fig_count = len(fig_index)
check('expected figures in figure-index.json (40)', fig_count == 40)
captions = re.findall(r'^\*图\s*19-(\d+)(?:：|　|\s)', s, re.M)
check('sequential captions 19-1 to 19-40', captions == [str(i) for i in range(1, fig_count + 1)])

imgs = re.findall(r'!\[[^\n]*\]\((ch19/[^)]+)\)', s)
check(
    'active illustrations match figure-index.json',
    imgs == [row.get('asset') or f"ch19/{row['file']}" for row in fig_index] and len(imgs) == 40,
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
check(
    'required Chapter 19 case studies linked',
    all(cs in s for cs in ['multimodal-stage-placement.md', 'queqiao.md', 'generative-multimodal-models.md']),
)

defs = re.findall(r'^\[\^([^\]]+)\]:', s, re.M)
refs = re.findall(r'\[\^([^\]]+)\](?!:)', s)
check('all footnotes used and defined once (35)', set(refs) == set(defs) and len(defs) == len(set(defs)) and len(defs) == 35)

golden = json.loads((ROOT / 'calculations/results/ch19-multimodal-dit-duplex.json').read_text(encoding='utf-8'))
g_sum = golden['summary']
check(
    'golden calc.py ch19-multimodal-dit-duplex metrics locked in manuscript',
    g_sum['video_10s_1080p_3d_tokens'] == 489600
    and math.isclose(g_sum['vae_tiling_peak_compression_x'], 94.8)
    and math.isclose(g_sum['usp_8x8_cross_node_comm_mib'], 836.71875)
    and math.isclose(g_sum['usp_8x8_vs_ring_rdma_reduction_x'], 9.0)
    and math.isclose(g_sum['epd_4gpu_4img_h0_throughput_req_s'], 6.0)
    and math.isclose(g_sum['epd_4gpu_4img_h75_throughput_req_s'], 9.16)
    and math.isclose(g_sum['duplex_speech_ttfa_ms'], 164.0)
    and math.isclose(g_sum['duplex_vs_cascade_speedup_x'], 7.8)
    and g_sum['dual_ar_depth_micro_kernels'] == 384
    and math.isclose(g_sum['dual_ar_cudagraph_frame_ms'], 11.5)
    and math.isclose(g_sum['dual_ar_cudagraph_speedup_x'], 3.35)
    and g_sum['barge_in_played_frame'] == 54
    and g_sum['barge_in_rolled_back_frames'] == 11
    and g_sum['barge_in_rolled_back_kv_tokens'] == 187
    and math.isclose(g_sum['barge_in_pointer_reset_ms'], 0.12)
    and all(tok in s for tok in ['489,600', '94.8', '836.72', '9.0', '9.16', '164.0', '7.8', '384', '11.5', '3.35', '54', '11', '187', '0.12']),
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

exp_dir = ROOT / 'experiments' / 'ch19'
exp_scaffolding_hits = []
for suite in sorted(exp_dir.glob('*')):
    if suite.is_dir():
        for fp in sorted(suite.rglob('*')):
            if fp.suffix in {'.py', '.json', '.md'} and '__pycache__' not in fp.parts:
                txt = fp.read_text(encoding='utf-8')
                if re.search(r'Chapter 17|Chapter 22|Ex\s*(?:17|22)-\d+|(?<!\d)17\.[1-8](?:\.\d+)?|(?<!\d)22\.[1-8](?:\.\d+)?', txt):
                    exp_scaffolding_hits.append(str(fp))
    if suite.is_dir() and (suite / 'verify.py').exists():
        res = subprocess.run([sys.executable, 'verify.py'], cwd=suite, capture_output=True, text=True)
        check(f'experiment suite {suite.name} verify.py', res.returncode == 0)
check('experiment ch19 files free of legacy Ch17 or Ch22 scaffolding', not exp_scaffolding_hits)

failed = [c for c in checks if not c['passed']]
print(json.dumps({"total": len(checks), "passed": len(checks) - len(failed), "failed": failed}, ensure_ascii=False, indent=2))
if failed:
    sys.exit(1)
