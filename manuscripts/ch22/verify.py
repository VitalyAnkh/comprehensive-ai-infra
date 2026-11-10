#!/usr/bin/env python3
"""Verify Chapter 22 structure, figure layout, mathematical formulas, and numerical examples."""
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


md = HERE.parent / '22-全双工实时语音（Full-Duplex Speech）、神经音频编解码与原生全模态流式底座.md'
s = md.read_text()
page = preview_path(md).read_text()

expected_headings = [
    '22.1', '22.1.1', '22.1.2',
    '22.2', '22.2.1', '22.2.2',
    '22.3', '22.3.1', '22.3.2',
    '22.4', '22.4.1', '22.4.2',
]
found_headings = re.findall(r'^#{2,3} (22\.\d+(?:\.\d+)?) ', s, re.M)
check('all four sections and eight subsections match outline', found_headings == expected_headings)

h2s = re.findall(r'^## (.+)$', s, re.M)
check('last H2 is 本章小结', len(h2s) >= 6 and h2s[-1].strip() == '本章小结')

exercises = re.findall(r'^>\s*\*\*(?:练习|实验) 22-(\d+)[：:（]', s, re.M)
check('twelve exercises 22-1 to 22-12 in order', exercises == [str(i) for i in range(1, 13)])

code_blocks = re.findall(r'```python\n.*?```', s, re.S)
check('seven production python code blocks', len(code_blocks) == 7 and len(re.findall(r'```', s)) == 14)

worked_examples = re.findall(r'^>\s*\*\*算例\s*22-', s, re.M)
check('seven standardized blockquote worked examples', len(worked_examples) == 7)

table_captions = re.findall(r'^\*表 22-(\d+)　', s, re.M)
check('seven standardized italic table captions 22-1 to 22-7', table_captions == [str(i) for i in range(1, 8)])

fig_index = json.loads((HERE / 'figure-index.json').read_text())
fig_count = len(fig_index)
check('eighteen figures in figure-index.json', fig_count == 18)
captions = re.findall(r'^\*图 22-(\d+)(?:：|　)', s, re.M)
check('sequential captions 22-1 to 22-18', sorted(captions, key=int) == [str(i) for i in range(1, fig_count + 1)])

imgs = re.findall(r'!\[.+?\]\((ch22/[^)]+)\)', s)
check('eighteen active illustrations match figure-index.json', set(imgs) == {row.get('asset') or f"ch22/{row['file']}" for row in fig_index} and len(imgs) == 18)

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
check('all footnotes used and defined once', set(defs) == set(refs) and len(defs) == len(set(defs)) and len(defs) >= 20)

for name, key in [('sources.json', 'sources'), ('manifest.json', 'outputs')]:
    data = json.loads((HERE / name).read_text())[key]
    mismatches = [row['path'] for row in data if sha256(ROOT / row['path']) != row['sha256']]
    check(name + ' integrity', not mismatches)

math_val = json.loads((HERE / 'math-validation.json').read_text())
check('no formula errors', not math_val['errors'] and 'katex-error' not in page)

layout_val = json.loads((HERE / 'teaching-layout-validation.json').read_text())
check(
    'book-size figure layout',
    len(layout_val) == 18 and all(r['width_pt'] == 420 and r['min_label_pt'] >= 11 and not r['text_extent_warnings'] for r in layout_val),
)

d = json.loads((HERE / 'figure-data.json').read_text())
barge = d['barge_in_kv_rollback']
check(
    'example 22-4 barge-in playback cursor KV rollback at frame 54 (187 tokens)',
    barge['client_playback_frame'] == 54
    and barge['unplayed_lookahead_frames'] == 11
    and barge['rolled_back_kv_tokens'] == 187,
)
av22 = d['avatar_lipsync_and_edge_sfu_mesh']
check(
    'examples 22-6 and 22-7 avatar lip-sync alignment and global edge SFU mesh',
    close(av22['aligned_steady_skew_ms'], 18.0)
    and close(av22['aligned_p99_skew_ms'], 33.0)
    and close(av22['avatar_ttfa_ms'], 190.0)
    and close(av22['ttfa_edge_ms'], 181.5)
    and close(av22['bw_saving_pct'], 97.71),
)

passed = sum(1 for c in checks if c['passed'])
print(json.dumps({"chapter": "ch22", "passed": passed, "total": len(checks), "checks": checks}, ensure_ascii=False, indent=2))
if passed != len(checks):
    sys.exit(1)
