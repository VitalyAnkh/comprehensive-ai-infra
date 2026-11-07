#!/usr/bin/env python3
"""Verify all generated internal page, fragment and asset links."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1] / 'build/site'

class Links(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.path, self.ids, self.links = path, set(), []
        self.feed(path.read_text(encoding='utf-8'))

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        for key in ('href', 'src'):
            if key in attrs:
                self.links.append(attrs[key])

pages = {p.resolve(): Links(p) for p in ROOT.rglob('*.html')}
errors = []
for path, page in pages.items():
    for link in page.links:
        u = urlsplit(link)
        if u.scheme or u.netloc or u.path.startswith('/'):
            continue
        target = (path.parent / unquote(u.path)).resolve() if u.path else path
        if target.is_dir():
            target /= 'index.html'
        if not target.exists():
            errors.append(f'{path.relative_to(ROOT)}: missing {link}')
        elif u.fragment and target in pages and unquote(u.fragment) not in pages[target].ids:
            errors.append(f'{path.relative_to(ROOT)}: missing anchor {link}')
# Each edition: preface, twelve chapters and its own search index (see scripts/build_site.py).
EDITIONS = {
    '简体中文': (ROOT, 'manuscripts/00-前言.html', lambda n: f'manuscripts/{n:02}-*.html'),
    'English': (ROOT / 'en', 'introduction.html', lambda n: f'chapter{n:02}.html'),
    '繁體中文': (ROOT / 'zh-tw', 'introduction.html', lambda n: f'chapter{n:02}.html'),
}
for name, (site, preface, chapter) in EDITIONS.items():
    if not (site / preface).exists():
        errors.append(f'{name}: missing preface')
    max_ch = 21 if name == '简体中文' else 13
    missing = [n for n in range(1, max_ch) if not any(site.glob(chapter(n)))]
    if missing:
        errors.append(f'{name}: missing chapters {missing}')
    if not (site / 'search/search_index.json').exists():
        errors.append(f'{name}: missing search index')
if errors:
    raise SystemExit('\n'.join(errors))
print(f'PASS: {len(pages)} pages; {len(EDITIONS)} editions with preface, chapters (20 zh / 12 translations) and search index; '
      'internal links and images')
