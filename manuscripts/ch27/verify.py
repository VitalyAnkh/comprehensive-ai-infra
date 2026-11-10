#!/usr/bin/env python3
"""Verify Chapter 27 structure, figure layout, mathematical formulas, and numerical examples (64 checks)."""
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


md = HERE.parent / '27-晶圆级与 3D-SRAM 专用推理芯片（Cerebras-Groq）、微秒级极低延迟 AI 与三值量化（BitNet b1.58）系统.md'
s = md.read_text()
page = preview_path(md).read_text()

expected_headings = [
    '27.1', '27.1.1', '27.1.2',
    '27.2', '27.2.1', '27.2.2',
    '27.3', '27.3.1', '27.3.2',
    '27.4', '27.4.1', '27.4.2',
]
found_headings = re.findall(r'^#{2,3} (27\.\d+(?:\.\d+)?) ', s, re.M)
check('all four sections and eight subsections match outline', found_headings == expected_headings)

h2s = re.findall(r'^## (.+)$', s, re.M)
check('last H2 is 本章小结', len(h2s) >= 6 and h2s[-1].strip() == '本章小结')

exercises = re.findall(r'^(?:>\s*)?\*\*(?:练习|实验) 27-(\d+)[：:（]', s, re.M)
check('ten exercises 27-1 to 27-10 in order', exercises == [str(i) for i in range(1, 11)])

code_blocks = re.findall(r'```python\n.*?```', s, re.S)
check('six production python code blocks', len(code_blocks) == 6 and len(re.findall(r'```', s)) == 12)

worked_examples = re.findall(r'^>\s*\*\*算例\s*27-', s, re.M)
check('six standardized blockquote worked examples', len(worked_examples) == 6)

table_captions = re.findall(r'^\*表 27-(\d+)　', s, re.M)
check('six standardized italic table captions 27-1 to 27-6', table_captions == [str(i) for i in range(1, 7)])

fig_index = json.loads((HERE / 'figure-index.json').read_text())
fig_count = len(fig_index)
check('sixteen figures in figure-index.json', fig_count == 16)
captions = re.findall(r'^\*图 27-(\d+)(?:：|　)', s, re.M)
check('sequential captions 27-1 to 27-16', captions == [str(i) for i in range(1, fig_count + 1)])

imgs = re.findall(r'!\[.+?\]\((ch27/[^)]+)\)', s)
check(
    'sixteen active illustrations match figure-index.json',
    imgs == [row.get('asset') or f"ch27/{row['file']}" for row in fig_index] and len(imgs) == 16,
)

for rel in imgs:
    p = md.parent / rel
    text_svg = ''.join(ET.parse(p).getroot().itertext())
    check('no internal figure numbering: ' + p.stem, re.search(r'图\s*\d+\s*[-－–]\s*\d+', text_svg) is None)
    check('SVG PNG PDF available: ' + p.stem, all(p.with_suffix(e).exists() for e in ['.svg', '.png', '.pdf']))
    check('PDF is binary not LFS pointer: ' + p.stem, p.with_suffix('.pdf').read_bytes().startswith(b'%PDF'))

missing = []
for link in re.findall(r'\]\(([^)]+)\)', s):
    u = urlsplit(link)
    if not u.scheme and u.path and not (md.parent / unquote(u.path)).exists():
        missing.append((str(md), link))
check('local links exist', not missing)

defs = re.findall(r'^\[\^([^\]]+)\]:', s, re.M)
refs = re.findall(r'\[\^([^\]]+)\](?!:)', s)
check('all footnotes used and defined once', set(defs) == set(refs) and len(defs) == len(set(defs)) and len(defs) >= 15)

for name, key in [('sources.json', 'sources'), ('manifest.json', 'outputs')]:
    data = json.loads((HERE / name).read_text())[key]
    mismatches = [row['path'] for row in data if sha256(ROOT / row['path']) != row['sha256']]
    check(name + ' integrity', not mismatches)

math_val = json.loads((HERE / 'math-validation.json').read_text())
check('no formula errors', not math_val['errors'] and 'katex-error' not in page)

layout_val = json.loads((HERE / 'teaching-layout-validation.json').read_text())
check(
    'book-size figure layout',
    len(layout_val) == 16 and all(r['width_pt'] == 420 and r['min_label_pt'] >= 11 and not r['text_extent_warnings'] for r in layout_val),
)

d = json.loads((HERE / 'figure-data.json').read_text())
check(
    'locked SRAM/WSE-3/Groq, SoIC, HFT Megakernel, and BitNet/T-MAC metrics',
    close(d['sram_vs_hbm_tco']['sram_b1_tok_per_sec'], 1850.0)
    and close(d['sram_vs_hbm_tco']['b1_speedup_x'], 11.23)
    and close(d['sram_vs_hbm_tco']['tco_crossover_batch_size'], 14.2)
    and close(d['sram_vs_hbm_tco']['high_batch_hbm_cost_adv_x'], 6.8)
    and close(d['soic_3d_sram_and_deterministic_noc']['sram_capacity_gb_per_chip'], 1.5)
    and close(d['soic_3d_sram_and_deterministic_noc']['hardware_barrier_skew_ns'], 2.0)
    and close(d['sub_millisecond_hft_ai']['persistent_p50_us'], 16.5)
    and close(d['sub_millisecond_hft_ai']['persistent_p9999_us'], 24.8)
    and close(d['sub_millisecond_hft_ai']['p9999_tail_reduction_x'], 16.94)
    and close(d['bitnet_b158_and_tmac']['packed_bits_per_weight'], 1.6)
    and close(d['bitnet_b158_and_tmac']['model_70b_bitnet_gb'], 14.0)
    and close(d['bitnet_b158_and_tmac']['tmac_speedup_x'], 6.4)
    and close(d['bitnet_b158_and_tmac']['dynamic_mul_power_saving_pct'], 88.5),
)

passed = sum(1 for c in checks if c['passed'])
print(json.dumps({"chapter": "ch27", "passed": passed, "total": len(checks), "checks": checks}, ensure_ascii=False, indent=2))
if passed != len(checks):
    sys.exit(1)
