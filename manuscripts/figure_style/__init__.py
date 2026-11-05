"""Book-size vector drawings; geometry is designed at its final reading size."""
from pathlib import Path
import json
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from figure_style.typography import FAMILY, configure_font

configure_font()

COL = dict(
    ink='#0F172A',
    muted='#475569',
    line='#475569',
    blue='#DCEBFA',
    green='#D8F1E5',
    teal='#D5F5F0',
    orange='#FDE8CC',
    purple='#EAE0F6',
    gray='#F1F5F9',
    white='#FFFFFF',
)

BORDER_COL = dict(
    blue='#2B6CB0',
    green='#2F855A',
    teal='#0F766E',
    orange='#C05621',
    purple='#6B46C1',
    gray='#64748B',
    white='#94A3B8',
    ink='#0F172A',
    line='#475569',
)

HEADER_COL = dict(
    blue='#C3DDF6',
    green='#BCE5D2',
    teal='#B2ECE4',
    orange='#FBD3A7',
    purple='#DAC8F0',
    gray='#E2E8F0',
    white='#F1F5F9',
)

TEXT_COL = dict(
    blue='#1E3A8A',
    green='#14532D',
    teal='#134E4A',
    orange='#7C2D12',
    purple='#3B0764',
    gray='#1E293B',
    white='#0F172A',
)

STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': [FAMILY, 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'font.size': 12,
    'font.weight': 'normal',
    'axes.titleweight': 'bold',
    'text.color': COL['ink'],
    'axes.labelcolor': COL['ink'],
    'axes.edgecolor': '#64748B',
    'axes.linewidth': 0.95,
    'xtick.color': COL['ink'],
    'ytick.color': COL['ink'],
    'axes.labelsize': 12,
    'axes.titlesize': 13.5,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'legend.frameon': True,
    'legend.facecolor': '#FFFFFF',
    'legend.edgecolor': '#CBD5E1',
    'legend.framealpha': 0.94,
    'grid.color': '#CBD5E1',
    'grid.alpha': 0.45,
    'svg.fonttype': 'path',
    'pdf.fonttype': 42,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
}


def canvas(height=3.6):
    fig, ax = plt.subplots(figsize=(420 / 72, height))
    fig.subplots_adjust(left=0.035, right=0.965, bottom=0.035, top=0.965)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    return fig, ax


def plot(height=3.6, left=0.17, bottom=0.20, right=0.95, top=0.93):
    fig, ax = plt.subplots(figsize=(420 / 72, height))
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    ax.spines[['top', 'right']].set_visible(False)
    return fig, ax


def text(ax, x, y, label, size=12, ha='left', **kw):
    if 'weight' not in kw and size >= 13.5:
        kw['weight'] = 'bold'
    if 'color' not in kw:
        kw['color'] = COL['ink']
    return ax.text(x, y, label, fontsize=size, ha=ha, va='center', linespacing=1.36, **kw)


def box(ax, x, y, w, h, label='', color='blue', size=12, **kw):
    face = COL.get(color, color)
    edge = BORDER_COL.get(color, COL['line'])
    is_container = (label == '' and (w > 0.60 or h > 0.32))
    if not is_container and w >= 0.055 and h >= 0.055 and color != 'white':
        shadow = FancyBboxPatch(
            (x + 0.003, max(0.002, y - 0.005)),
            w,
            h,
            boxstyle='round,pad=0.002,rounding_size=0.010',
            facecolor='#0F172A',
            edgecolor='none',
            alpha=0.065,
        )
        ax.add_patch(shadow)
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle='round,pad=0.002,rounding_size=0.010',
        facecolor=face,
        edgecolor=edge,
        linewidth=1.15 if not is_container else 1.0,
        linestyle='--' if (is_container and color == 'gray') else '-',
    )
    ax.add_patch(patch)
    if label:
        tcol = TEXT_COL.get(color, COL['ink'])
        weight = kw.pop('weight', 'medium' if size >= 11.5 else 'normal')
        t = text(ax, x + w / 2, y + h / 2, label, size, ha='center', color=tcol, weight=weight, **kw)
        t._enclosing_patch = patch
    return patch


def card(ax, x, y, w, h, title, body='', color='blue', title_size=11.5, body_size=11.0):
    """Two-tier publication card with header band, subtle shadow, and semantic border."""
    if isinstance(body, (list, tuple)):
        body = '\n'.join(str(line) for line in body)
    face = COL.get(color, color)
    head_face = HEADER_COL.get(color, COL['gray'])
    edge = BORDER_COL.get(color, COL['line'])
    tcol = TEXT_COL.get(color, COL['ink'])
    shadow = FancyBboxPatch(
        (x + 0.003, max(0.002, y - 0.006)),
        w,
        h,
        boxstyle='round,pad=0.002,rounding_size=0.012',
        facecolor='#0F172A',
        edgecolor='none',
        alpha=0.07,
    )
    ax.add_patch(shadow)
    base = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle='round,pad=0.002,rounding_size=0.012',
        facecolor=face,
        edgecolor=edge,
        linewidth=1.25,
    )
    ax.add_patch(base)
    if body:
        hh = min(max(h * 0.38, 0.048), 0.095)
        head = FancyBboxPatch(
            (x, y + h - hh),
            w,
            hh,
            boxstyle='round,pad=0.002,rounding_size=0.010',
            facecolor=head_face,
            edgecolor=edge,
            linewidth=1.0,
        )
        ax.add_patch(head)
        t1 = text(ax, x + w / 2, y + h - hh / 2, title, title_size, ha='center', color=tcol, weight='bold')
        t2 = text(ax, x + w / 2, y + (h - hh) / 2, body, body_size, ha='center', color=COL['ink'], weight='normal')
        t1._enclosing_patch = base
        t2._enclosing_patch = base
    elif h >= 0.24:
        hh = min(max(h * 0.12, 0.052), 0.076)
        head = FancyBboxPatch(
            (x, y + h - hh),
            w,
            hh,
            boxstyle='round,pad=0.002,rounding_size=0.010',
            facecolor=head_face,
            edgecolor=edge,
            linewidth=1.0,
        )
        ax.add_patch(head)
        t1 = text(ax, x + w / 2, y + h - hh / 2, title, title_size, ha='center', color=tcol, weight='bold')
        t1._enclosing_patch = base
    else:
        t1 = text(ax, x + w / 2, y + h / 2, title, title_size, ha='center', color=tcol, weight='bold')
        t1._enclosing_patch = base
    return base


def badge(ax, x, y, label, color='white', size=11.0, ha='center'):
    """Pill callout badge with opaque background so edge labels never collide with lines."""
    edge = BORDER_COL.get(color, '#CBD5E1') if color != 'white' else '#CBD5E1'
    face = COL.get(color, '#FFFFFF') if color != 'white' else '#FFFFFF'
    tcol = TEXT_COL.get(color, COL['ink'])
    return ax.text(
        x,
        y,
        label,
        fontsize=size,
        ha=ha,
        va='center',
        color=tcol,
        weight='medium',
        bbox=dict(boxstyle='round,pad=0.22,rounding_size=0.35', facecolor=face, edgecolor=edge, linewidth=0.85, alpha=0.96),
    )


def arrow(ax, start, end, kind='data', label='', rad=0.0, color=None):
    # Dashed edges denote control; solid edges carry data. Explain locally in captions.
    col = BORDER_COL.get(color, color) if color else '#334155'
    conn = f'arc3,rad={rad}' if rad else 'arc3,rad=0'
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle='-|>',
            mutation_scale=11.5,
            linewidth=1.3,
            color=col,
            linestyle='--' if kind == 'control' else '-',
            connectionstyle=conn,
            shrinkA=3,
            shrinkB=3,
        )
    )
    if label:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        badge(ax, mx, my, label, size=11.0)


class Exporter:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.outputs = []
        self.checks = []

    def save(self, fig, name):
        for ax in fig.axes:
            legend = ax.get_legend()
            if legend is not None and ax.axison:
                # If legend is placed inside the plot area where gridlines/curves exist, give it an opaque pill frame.
                fig.canvas.draw()
                r0 = fig.canvas.get_renderer()
                ax_bb = ax.get_window_extent(r0)
                leg_bb = legend.get_window_extent(r0)
                if leg_bb.y1 <= ax_bb.y1 + 4 and leg_bb.y0 >= ax_bb.y0 - 4:
                    legend.set_frame_on(True)
                    frame = legend.get_frame()
                    frame.set_facecolor('#FFFFFF')
                    frame.set_edgecolor('#CBD5E1')
                    frame.set_alpha(0.93)
                    frame.set_linewidth(0.8)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        warnings = []
        sizes = []
        labels = list(fig.texts)
        for ax in fig.axes:
            labels.extend(ax.texts)
            legend = ax.get_legend()
            if legend is not None:
                labels.extend(legend.get_texts())
            if ax.axison:
                labels.extend([ax.title, ax.xaxis.label, ax.yaxis.label])
                for axis in [ax.xaxis, ax.yaxis]:
                    lo, hi = sorted(axis.get_view_interval())
                    for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                        if lo <= tick.get_loc() <= hi:
                            labels.extend([tick.label1, tick.label2])
                    labels.append(axis.get_offset_text())
        for t in labels:
            if not t.get_visible() or not t.get_text():
                continue
            bb = t.get_window_extent(renderer)
            sizes.append(t.get_fontsize())
            if bb.x0 < 0 or bb.y0 < 0 or bb.x1 > fig.bbox.width or bb.y1 > fig.bbox.height:
                warnings.append(t.get_text())
            patch = getattr(t, '_enclosing_patch', None)
            if patch is not None:
                pbb = patch.get_window_extent(renderer)
                if bb.x0 < pbb.x0 + 3.5 or bb.x1 > pbb.x1 - 3.5:
                    warnings.append(f'BOX_OVERFLOW:{t.get_text()} (text_w={bb.width:.1f} > box_w={pbb.width:.1f})')
        record = dict(
            figure=name,
            width_pt=round(fig.get_figwidth() * 72, 4),
            min_label_pt=min(sizes),
            text_extent_warnings=warnings,
        )
        self.checks.append(record)
        if warnings:
            raise ValueError(f'{name}: labels outside canvas: {warnings}')
        if min(sizes) < 11:
            raise ValueError(f'{name}: label smaller than 11 pt')
        for ext in ['svg', 'png', 'pdf']:
            path = self.directory / f'{name}.{ext}'
            fig.savefig(path, dpi=240)
            self.outputs.append(path)
        plt.close(fig)

    def finish(self):
        (self.directory / 'teaching-layout-validation.json').write_text(
            json.dumps(self.checks, ensure_ascii=False, indent=2) + '\n'
        )
        return self.outputs
