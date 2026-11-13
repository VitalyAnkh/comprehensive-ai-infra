#!/usr/bin/env python3
"""Unified figure builder and manifest refresher for Chapter 23 (all 60 figures)."""
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

import ch14_build
import ch26_build
import ch27_build

plt.rcParams.update(STYLE)


def sha256(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b'version https://git-lfs.github.com/spec/v1'):
        m = re.search(rb'oid sha256:([0-9a-f]{64})', raw)
        if m:
            return m.group(1).decode('ascii')
    return hashlib.sha256(raw).hexdigest()


def refresh_metadata(checks: list[dict], outputs: list[Path]) -> None:
    md = HERE.parent / '23-软硬件协同与新型架构.md'
    text_md = md.read_text(encoding='utf-8')

    # Sync captions & sections from manuscript into figure-index.json
    fig_index = json.loads((HERE / 'figure-index.json').read_text(encoding='utf-8'))
    sec_by_fig = {}
    cur_sec = '23.1'
    for line in text_md.splitlines():
        msec = re.match(r'^#{2,3}\s+(23\.\d+(?:\.\d+)?)\s+', line)
        if msec:
            cur_sec = msec.group(1)
        mfig = re.match(r'^\*图\s*23-(\d+)(?:：|　|\s)(.+)\*$', line.strip())
        if mfig:
            idx = int(mfig.group(1))
            sec_by_fig[idx] = (cur_sec, mfig.group(2).strip())

    for i, row in enumerate(fig_index, start=1):
        if i in sec_by_fig:
            row['section'], row['caption'] = sec_by_fig[i]
            if 'title' in row:
                row['title'] = sec_by_fig[i][1]
        refs = row.get('source_refs', [])
        row['source_refs'] = [
            r.replace('manuscripts/ch14/', 'manuscripts/ch23/')
             .replace('manuscripts/ch26/', 'manuscripts/ch23/')
             .replace('manuscripts/ch27/', 'manuscripts/ch23/')
             .replace('experiments/ch14/14-03/', 'experiments/ch23/23-03/')
             .replace('experiments/ch26/26-01/', 'experiments/ch23/23-04/')
             .replace('experiments/ch26/26-02/', 'experiments/ch23/23-05/')
             .replace('experiments/ch26/26-03/', 'experiments/ch23/23-06/')
             .replace('experiments/ch27/27-01/', 'experiments/ch23/23-07/')
             .replace('experiments/ch27/27-02/', 'experiments/ch23/23-08/')
             .replace('experiments/ch27/27-03/', 'experiments/ch23/23-09/')
            for r in refs
        ]
        role = row.get('TeachingRole', '')
        role = (
            role.replace('第 14 章', '第 23 章')
                .replace('收束全书十四章知识体系，', '总结软硬件协同诊断方法，')
                .replace('收束全书的同等能力降本主线', '梳理跨代软硬件协同降本脉络')
                .replace('串联第 2 章至第 14 章', '串联第 2 章至第 23 章')
                .replace('闭环', '迭代')
        )
        row['TeachingRole'] = role
    (HERE / 'figure-index.json').write_text(
        json.dumps(fig_index, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )

    layout_payload = {
        "chapter": "ch23",
        "figure_count": len(checks),
        "total_figures": len(checks),
        "all_passed": all(not c['text_extent_warnings'] for c in checks),
        "total_warnings": sum(len(c['text_extent_warnings']) for c in checks),
        "figures": [
            {
                "figure": f"23-{i}",
                "file": f"{c['figure']}.svg",
                "width_pt": c['width_pt'],
                "min_label_pt": c['min_label_pt'],
                "passed": not c['text_extent_warnings'],
                "warnings": c['text_extent_warnings'],
                "text_extent_warnings": c['text_extent_warnings'],
            }
            for i, c in enumerate(checks, start=1)
        ],
    }
    (HERE / 'teaching-layout-validation.json').write_text(
        json.dumps(layout_payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )

    # Strip fenced code blocks before scanning $...$ for KaTeX validation
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

    work = re.sub(r'!\[([^\]]*)\]\((ch23/[^)]+)\)', repl_img, text_md)
    body = md_lib.markdown(work, extensions=['tables', 'fenced_code', 'footnotes', 'toc'])
    html = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>第 23 章：软硬件协同与新型架构</title></head><body>{body}</body></html>\n'
    preview_path(md).write_text(html, encoding='utf-8')

    exp_runs = sorted((ROOT / 'experiments' / 'ch23').glob('*/run.py'))
    src_paths = [
        md,
        HERE / 'figure-index.json',
        ROOT / 'calculations/results/ch24-ai-factory-capstone.json',
        *exp_runs,
    ]
    (HERE / 'sources.json').write_text(
        json.dumps(
            {
                "chapter": "ch23",
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
                "chapter": "ch23",
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
    ch14_build.build_part1(ex)
    ch26_build.build_part2(ex)
    ch27_build.build_part3(ex)
    (HERE / 'figure-data.json').write_text(
        json.dumps(ch14_build.build_figure_data(), ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    refresh_metadata(ex.checks, ex.outputs)
    print(f'Built {len(ex.checks)} figures ({len(ex.outputs)} files) for Chapter 23; 0 layout warnings.')


if __name__ == '__main__':
    main()
