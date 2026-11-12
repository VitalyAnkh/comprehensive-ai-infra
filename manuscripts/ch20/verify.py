#!/usr/bin/env python3
"""Verify Chapter 20 structure, figure layout, mathematical formulas, and numerical examples."""
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


md = HERE.parent / '20-具身智能与物理仿真.md'
s = md.read_text(encoding='utf-8')
page = preview_path(md).read_text(encoding='utf-8')

expected_headings = [
    "20.1", "20.1.1", "20.1.2", "20.1.3",
    "20.2", "20.2.1", "20.2.2",
    "20.3", "20.3.1", "20.3.2", "20.3.3",
    "20.4", "20.4.1", "20.4.2", "20.4.3", "20.4.4",
    "20.5", "20.5.1", "20.5.2",
]
found_headings = re.findall(r'^#{2,3} (20\.\d+(?:\.\d+)?) ', s, re.M)
check('all sections and subsections match outline', found_headings == expected_headings)
check('chapter line count meets Path H2 target (>= 1720)', len(s.splitlines()) >= 1720)
check('zero H5 headings and zero digit-prefixed H4 headings', not re.findall(r'^#####\s', s, re.M) and not re.findall(r'^####\s+\d', s, re.M))
numbered_titles = re.findall(r'^#{2,3}\s+20\.\d+(?:\.\d+)?\s+(.+)$', s, re.M)
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
    not re.search(r'六大部分|六部二十四章|第六部分|前十八章|DeepSeek\s*V4\.1|TC-\d+|Paper\s*#\d+|\\tag\{|双向奔赴|\[\^ch19-', s),
)

h2s = re.findall(r'^## (.+)$', s, re.M)
check('last H2 is 本章小结 and total H2 count is 8 (5 numbered + 3 backmatter)', len(h2s) == 8 and h2s[-1].strip() == '本章小结')

exercises = re.findall(r'^(?:>\s*)?\*\*(?:(?:练习|实验|习题|拓展练习|思考题)\s+)?20-(\d+)\b(?!\s*解答)', s, re.M)
check('sequential exercises 20-1 to 20-18', exercises == [str(i) for i in range(1, 18 + 1)])
answers = [
    m.group(1) or m.group(2)
    for m in re.finditer(r'^(?:-\s*\*\*20-(\d+)\s*解答\*\*|####\s+(?:练习|实验)\s+20-(\d+)\s+解答)', s, re.M)
]
check('all 18 exercises have step-by-step derivations', answers == [str(i) for i in range(1, 18 + 1)])

py_blocks = re.findall(r'^```python\b', s, re.M)
cpp_blocks = re.findall(r'^```cpp\b', s, re.M)
check('expected production code blocks (10 Python + 1 C++)', len(py_blocks) == 10 and len(cpp_blocks) == 1)

worked_examples = re.findall(r'^>\s*\*\*算例\s*20-(\d+)\b', s, re.M)
check('sequential blockquote worked examples 20-1 to 20-10', worked_examples == [str(i) for i in range(1, 10 + 1)])

table_captions = re.findall(r'^\*表\s*20-(\d+)(?:：|　|\s)', s, re.M)
check('sequential italic table captions 20-1 to 20-11', table_captions == [str(i) for i in range(1, 11 + 1)])

fig_index = json.loads((HERE / 'figure-index.json').read_text(encoding='utf-8'))
fig_count = len(fig_index)
check('expected figures in figure-index.json (23)', fig_count == 23)
captions = re.findall(r'^\*图\s*20-(\d+)(?:：|　|\s)', s, re.M)
check('sequential captions 20-1 to 20-23', captions == [str(i) for i in range(1, fig_count + 1)])

imgs = re.findall(r'!\[[^\n]*\]\((ch20/[^)]+)\)', s)
check(
    'active illustrations match figure-index.json',
    imgs == [row.get('asset') or f"ch20/{row['file']}" for row in fig_index] and len(imgs) == 23,
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
    'required Chapter 20 case studies linked',
    all(cs in s for cs in ['embodied-soc-and-world-model.md', 'multimodal-stage-placement.md', 'macbook-rtx-pro6000.md']),
)

defs = re.findall(r'^\[\^([^\]]+)\]:', s, re.M)
refs = re.findall(r'\[\^([^\]]+)\](?!:)', s)
check('all footnotes used and defined once (26)', set(refs) == set(defs) and len(defs) == len(set(defs)) and len(defs) == 26)

golden = json.loads((ROOT / 'calculations/results/ch20-embodied-wcet-sim.json').read_text(encoding='utf-8'))
g_sum = golden['summary']
check(
    'golden calc.py ch20-embodied-wcet-sim metrics locked in manuscript',
    g_sum['sys1_to_sys2_freq_ratio'] == 100
    and math.isclose(g_sum['jetson_shared_wcet_ms'], 18.5)
    and math.isclose(g_sum['jetson_shared_deadline_miss_pct'], 28.4)
    and math.isclose(g_sum['jetson_mig_isolated_wcet_ms'], 4.15)
    and math.isclose(g_sum['sim_4096env_pcie_tier_a_gbps'], 24.22)
    and math.isclose(g_sum['sim_4096env_pcie_tier_b_gbps'], 43.39)
    and math.isclose(g_sum['sim_4096env_gpu_hbm_tier_a_gib'], 8.25)
    and math.isclose(g_sum['sim_4096env_gpu_hbm_tier_b_gib'], 25.52)
    and g_sum['nv12_bytes_per_frame'] == 3456000
    and math.isclose(g_sum['raw_camera_ingress_gbps'], 1.244)
    and math.isclose(g_sum['ros2_serialized_dram_gbps'], 11.2)
    and math.isclose(g_sum['dmabuf_iceoryx_dram_gbps'], 2.49)
    and math.isclose(g_sum['dram_bandwidth_reduction_x'], 4.5)
    and math.isclose(g_sum['ros2_sensor_to_vit_p99_ms'], 18.6)
    and math.isclose(g_sum['dmabuf_sensor_to_vit_p99_ms'], 0.85)
    and math.isclose(g_sum['sensor_latency_speedup_x'], 21.88)
    and math.isclose(g_sum['reflex_wcet_unisolated_ms'], 1.82)
    and math.isclose(g_sum['reflex_wcet_slc_qos_isolated_ms'], 0.41)
    and math.isclose(g_sum['world_model_64gpu_eval_hours'], 0.4)
    and math.isclose(g_sum['world_model_64gpu_cost_usd'], 768.0)
    and math.isclose(g_sum['physical_robot_eval_cost_usd'], 111360.0)
    and math.isclose(g_sum['world_model_cost_compression_x'], 145.0)
    and all(
        tok in s
        for tok in [
            '18.5', '28.4', '24.22', '43.39', '8.25', '25.52',
            '1.244', '11.20', '2.49', '4.5', '18.6', '0.85', '21.88',
            '1.82', '0.41', '0.40', '768', '111{,}360', '145',
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

exp_dir = ROOT / 'experiments' / 'ch20'
exp_scaffolding_hits = []
for suite in sorted(exp_dir.glob('*')):
    if suite.is_dir():
        for fp in sorted(suite.rglob('*')):
            if fp.suffix in {'.py', '.json', '.md'} and '__pycache__' not in fp.parts:
                txt = fp.read_text(encoding='utf-8')
                if re.search(r'Chapter 19|Ex\s*19-\d+|(?<!\d)19\.[1-5](?:\.\d+)?', txt):
                    exp_scaffolding_hits.append(str(fp))
    if suite.is_dir() and (suite / 'verify.py').exists():
        res = subprocess.run([sys.executable, 'verify.py'], cwd=suite, capture_output=True, text=True)
        check(f'experiment suite {suite.name} verify.py', res.returncode == 0)
check('experiment ch20 files free of legacy Ch19 scaffolding', not exp_scaffolding_hits)

failed = [c for c in checks if not c['passed']]
print(json.dumps({"total": len(checks), "passed": len(checks) - len(failed), "failed": failed}, ensure_ascii=False, indent=2))
if failed:
    sys.exit(1)
