#!/usr/bin/env python3
"""Build ElegantBook PDFs from the canonical manuscripts (Pandoc + XeLaTeX)."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MANUSCRIPTS = ROOT / 'manuscripts'


def run(command, *, cwd=HERE, log=None):
    if log:
        with log.open('w') as stream:
            result = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f'Build failed; see {log}\n' + log.read_text(errors='replace')[-6000:])
    else:
        subprocess.run(command, cwd=cwd, check=True)


def prepare(source, dest, assets, source_ref=None):
    text = source.read_text()
    number = int(source.name[:2])
    if number == 0:
        text = re.sub(r'^# 前言$', '# 前言 {.unnumbered}', text, flags=re.M)
        text = re.sub(r'^(## .+)$', r'\1 {.unnumbered}', text, flags=re.M)
    text = re.sub(r'^# 第\s*\d+\s*章[：:\s]*(.*)$', rf'# \1 {{#chapter-{number}}}', text, flags=re.M)
    text = re.sub(r'<a id="([^"]+)"></a>\s*\n+(#{1,6} [^\n]+)',
                  lambda m: m[2]+' {#'+m[1]+'}', text)
    # Manuscripts repeat image alt text as the following italic caption. Merge them.
    text = re.sub(r'!\[[^\n]*\]\(([^)]+)\)\s*\n\s*\*([^\n]+)\*',
                  lambda m: f'![{m[2]}]({m[1]})', text)

    def image(match):
        original = (source.parent / match[2]).resolve()
        selected = original.with_suffix('.pdf') if original.with_suffix('.pdf').exists() else original.with_suffix('.png')
        if not selected.exists():
            raise FileNotFoundError(original)
        if selected.read_bytes().startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError(f'Figure is an LFS pointer; run git lfs pull: {selected}')
        assets.append(selected)
        return f'![{match[1]}]({selected.as_posix()})'
    text = re.sub(r'!\[([^\n]*)\]\(([^)]+)\)', image, text)
    # Prefix ordinary links with their real relative location from the delivered PDF.
    def link(match):
        target = match[2]
        if re.match(r'[a-z]+:|#', target):
            return match[0]
        from urllib.parse import unquote
        resolved = (source.parent / unquote(target.split('#')[0])).resolve()
        if resolved.is_relative_to(ROOT):
            fragment = '#' + target.split('#',1)[1] if '#' in target else ''
            relative = resolved.relative_to(ROOT).as_posix()
            if source_ref:
                from urllib.parse import quote
                return f'[{match[1]}](https://github.com/VitalyAnkh/comprehensive-ai-infra/blob/{quote(source_ref, safe="")}/{quote(relative, safe="/")}{fragment})'
            return f'[{match[1]}](../{relative}{fragment})'
        return match[0]
    text = re.sub(r'!\[[^\n]*\]\([^\n)]+\)|(?<!!)\[([^\n\[\]]+)\]\(([^\n)]+)\)',
                  lambda m: m[0] if m[1] is None else link(m), text)
    dest.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chapter', type=int, choices=range(1,25), help='Build one chapter with its original chapter number')
    parser.add_argument('--output-dir', type=Path, default=HERE, help='Output directory relative to book/ (default: book/)')
    parser.add_argument('--source-ref', help='Git commit for portable GitHub links in released PDFs')
    args = parser.parse_args()
    output_dir = (HERE / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    name = f'AI-Infra-Book-Chapter-{args.chapter:02}' if args.chapter else 'AI-Infra-Book'
    work = HERE / 'build' / name
    work.mkdir(parents=True, exist_ok=True)
    numbers = [args.chapter] if args.chapter else range(1,25)
    sources = [next(MANUSCRIPTS.glob(f'{n:02}-*.md')) for n in numbers]
    inputs, assets = [], []
    for source in sources:
        target = work / source.name
        prepare(source,target,assets,args.source_ref)
        inputs.append(target)
    before = work / 'frontmatter.tex'
    edition = f'第 {args.chapter} 章排版样张' if args.chapter else 'v1.0'
    before.write_text('\\renewcommand{\\BookEdition}{'+edition+'}\n\\input{cover.tex}\n'
                     '\\pagenumbering{Roman}\n')
    front_sources = []
    if not args.chapter:
        preface = MANUSCRIPTS / '00-前言.md'
        prepared_preface = work / preface.name
        prepare(preface, prepared_preface, assets, args.source_ref)
        preface_tex = work / 'preface.tex'
        run(['pandoc', str(prepared_preface), '--to=latex',
             '--lua-filter='+str(HERE/'layout.lua'),
             '--top-level-division=chapter', '-o', str(preface_tex)],
            log=work / 'preface-pandoc.log')
        before.write_text(before.read_text() + '\\input{' + str(preface_tex) + '}\n\\clearpage\n')
        front_sources.append(preface)
    # mainmatter starts after the TOC, immediately before the first source chapter.
    first = inputs[0]
    first.write_text('```{=latex}\n\\clearpage\n\\pagenumbering{arabic}\n'
                     f'\\setcounter{{chapter}}{{{(args.chapter or 1)-1}}}\n```\n\n'+first.read_text())
    tex = work / 'book.tex'
    command = ['pandoc', *map(str,inputs), '--file-scope', '--standalone',
               '--from=markdown+lists_without_preceding_blankline', '--to=latex',
               '--top-level-division=chapter', '--toc', '--toc-depth=2', '--number-sections',
               '--lua-filter='+str(HERE/'layout.lua'),
               '-V','documentclass=elegantbook', '-V','classoption=lang=cn',
               '-V','classoption=nofont', '-V','classoption=cyan', '-V','classoption=device=normal',
               '-V','author=李博杰', '--metadata','title-meta=深入理解 AI Infra：量化分析与系统设计',
               '--metadata','author-meta=李博杰', '-H', str(HERE/'preamble.tex'),
               '--include-before-body='+str(before), '--highlight-style=kate',
               '--columns=100', '-o',str(tex)]
    run(command,log=work/'pandoc.log')
    # XeLaTeX runs from book/ so imported series template and cover resolve locally.
    for iteration in range(1,4):
        run(['xelatex','-interaction=nonstopmode','-halt-on-error', '-file-line-error',
             '-output-directory='+str(work),str(tex)],log=work/f'xelatex-{iteration}.log')
    output = output_dir / f'{name}.pdf'
    staged_output = output.with_suffix('.pdf.tmp')
    shutil.copy2(work/'book.pdf',staged_output)
    staged_output.replace(output)
    log = (work/'book.log').read_text(errors='replace')
    warnings = [line for line in log.splitlines() if any(x in line for x in ['Overfull','Missing character:','undefined references','LaTeX Warning:'])]
    report = dict(output=str(output), source_ref=args.source_ref, chapters=[int(s.name[:2]) for s in sources],
                  engine='Pandoc + XeLaTeX / ElegantBook (AI Agent Book series template)',
                  sources=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in front_sources + sources],
                  figure_count=len(assets),warnings=warnings)
    (output_dir/f'{name}-build.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if not args.chapter and shutil.which('pdfseparate') and shutil.which('pdftoppm'):
        if shutil.which('gs'):
            run(['gs','-q','-dBATCH','-dNOPAUSE','-sDEVICE=pdfwrite','-dFirstPage=1','-dLastPage=1',
                 '-sOutputFile='+str(output_dir/'AI-Infra-Book-Cover.pdf'),str(output)])
        else:
            run(['pdfseparate','-f','1','-l','1',str(output),str(output_dir/'AI-Infra-Book-Cover.pdf')])
        run(['pdftoppm','-f','1','-l','1','-scale-to','1600','-png','-singlefile',
             str(output),str(output_dir/'AI-Infra-Book-Cover')])
    print(output)
    print(f'{len(sources)} chapters, {len(assets)} figures; {len(warnings)} layout/font warnings; details: {work}/book.log')


if __name__ == '__main__':
    main()
