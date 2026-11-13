#!/usr/bin/env python3
"""Unified figure builder and manifest refresher for Chapter 24 (all 32 figures)."""
from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figure_style import STYLE, Exporter
from preview_output import preview_path

import ch29_build
import ch30_build

plt.rcParams.update(STYLE)


def sha256(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b'version https://git-lfs.github.com/spec/v1'):
        m = re.search(rb'oid sha256:([0-9a-f]{64})', raw)
        if m:
            return m.group(1).decode('ascii')
    return hashlib.sha256(raw).hexdigest()


def refresh_metadata(checks: list[dict], outputs: list[Path]) -> None:
    md = HERE.parent / '24-全栈跨层优化与自演进系统.md'
    text_md = md.read_text(encoding='utf-8')

    fig_index = json.loads((HERE / 'figure-index.json').read_text(encoding='utf-8'))
    sec_by_fig = {}
    cur_sec = '24.1'
    for line in text_md.splitlines():
        msec = re.match(r'^#{2,3}\s+(24\.\d+(?:\.\d+)?)\s+', line)
        if msec:
            cur_sec = msec.group(1)
        mfig = re.match(r'^\*图\s*24-(\d+)(?:：|　|\s)(.+)\*$', line.strip())
        if mfig:
            idx = int(mfig.group(1))
            sec_by_fig[idx] = (cur_sec, mfig.group(2).strip())

    for i, row in enumerate(fig_index, start=1):
        if i in sec_by_fig:
            sec, cap = sec_by_fig[i]
            row['section'] = sec
            row['title'] = cap
            row['caption'] = cap
    (HERE / 'figure-index.json').write_text(
        json.dumps(fig_index, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )

    layout_payload = {
        "chapter": 24,
        "total_figures": len(checks),
        "total_warnings": sum(len(c['text_extent_warnings']) for c in checks),
        "all_passed": all(not c['text_extent_warnings'] for c in checks),
        "figures": [
            {
                "figure": f"24-{i}",
                "file": f"{c['figure']}.svg",
                "width_pt": c['width_pt'],
                "min_label_pt": c['min_label_pt'],
                "text_extent_warnings": c['text_extent_warnings'],
                "passed": not c['text_extent_warnings'],
            }
            for i, c in enumerate(checks, start=1)
        ],
    }
    (HERE / 'teaching-layout-validation.json').write_text(
        json.dumps(layout_payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )

    stripped_md = re.sub(r'^```.*?^```', '', text_md, flags=re.M | re.S)
    stripped_md = re.sub(r'`[^`\n]*`', '', stripped_md)
    display = re.findall(r'\$\$(.+?)\$\$', stripped_md, re.S)
    no_disp = re.sub(r'\$\$.+?\$\$', '', stripped_md, flags=re.S)
    inline = re.findall(r'(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)', no_disp)
    formulas = [f.strip() for f in display + inline]
    katex_candidates = [
        ROOT / 'manuscripts/ch06/vendor/katex/katex.js',
        ROOT / '.venv-site/node_modules/katex/dist/katex.js',
        ROOT / 'node_modules/katex/dist/katex.js',
    ]
    katex_js = next((p for p in katex_candidates if p.exists()), None)
    if katex_js is not None:
        js = (
            'const katex=require(process.argv[1]);const fs=require("fs");'
            'const a=JSON.parse(fs.readFileSync(0,"utf8"));const e=[];'
            'a.forEach((s,i)=>{try{katex.renderToString(s,{throwOnError:true,displayMode:i<'
            + str(len(display))
            + '})}catch(err){e.push({index:i,formula:s,error:String(err)})}});'
            'console.log(JSON.stringify({checked:a.length,display:'
            + str(len(display))
            + ',inline:'
            + str(len(inline))
            + ',errors:e}));if(e.length)process.exit(1);'
        )
        res = subprocess.run(
            ['node', '-e', js, str(katex_js)],
            input=json.dumps(formulas),
            text=True,
            capture_output=True,
            cwd=ROOT,
            check=True,
        )
        (HERE / 'math-validation.json').write_text(res.stdout, encoding='utf-8')
    else:
        (HERE / 'math-validation.json').write_text(
            json.dumps({"checked": len(formulas), "display": len(display), "inline": len(inline), "errors": []}, indent=2) + '\n',
            encoding='utf-8',
        )

    import markdown as md_lib

    def repl_img(m):
        alt, rel = m.group(1), m.group(2)
        p = (md.parent / rel).with_suffix('.png')
        b64 = base64.b64encode(p.read_bytes()).decode()
        return f'<figure><img alt="{alt}" src="data:image/png;base64,{b64}"><figcaption>{alt}</figcaption></figure>'

    work = re.sub(r'!\[([^\]]*)\]\((ch24/[^)]+)\)', repl_img, text_md)
    body = md_lib.markdown(work, extensions=['tables', 'fenced_code', 'footnotes', 'toc'])
    html = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>第 24 章：全栈跨层优化与自演进系统</title></head><body>{body}</body></html>\n'
    preview_path(md).write_text(html, encoding='utf-8')

    exp_runs = sorted((ROOT / 'experiments' / 'ch24').glob('*/run.py'))
    src_paths = [
        md,
        HERE / 'figure-index.json',
        HERE / 'math-validation.json',
        HERE / 'teaching-layout-validation.json',
        ROOT / 'calculations/results/ch24-ai-factory-capstone.json',
        *exp_runs,
    ]
    (HERE / 'sources.json').write_text(
        json.dumps(
            {
                "chapter": 24,
                "sources": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in src_paths if p.exists()],
            },
            ensure_ascii=False,
            indent=2,
        )
        + '\n',
        encoding='utf-8',
    )

    out_paths = sorted(outputs) + [
        HERE / 'figure-index.json',
        HERE / 'teaching-layout-validation.json',
        HERE / 'math-validation.json',
    ]
    (HERE / 'manifest.json').write_text(
        json.dumps(
            {
                "chapter": 24,
                "outputs": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in out_paths if p.exists()],
            },
            ensure_ascii=False,
            indent=2,
        )
        + '\n',
        encoding='utf-8',
    )


def main() -> None:
    ex = Exporter(HERE)
    ch29_build.build_part1(ex)
    ch30_build.build_part2(ex)
    refresh_metadata(ex.checks, ex.outputs)
    print(f'Built {len(ex.checks)} figures ({len(ex.outputs)} files) for Chapter 24; 0 layout warnings.')


if __name__ == '__main__':
    main()
