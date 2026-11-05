"""L2 organisation, cross-SM handoff and synchronisation scopes, from the RTX PRO 6000 measurements."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
from figure_style import COL, BORDER_COL, TEXT_COL, STYLE, canvas, plot, text, box, card, badge, arrow

SOURCE = 'calculations/sources/opentallas/blackwell-sync-latency.json'


def ladder_rows(m, english=False):
    L = lambda zh, en: en if english else zh
    b = m['boundaries_ns']
    return [
        (L('SM 内栅栏', 'Barrier within an SM'), b['syncthreads_one_block'], 'green'),
        (L('线程块簇内同步', 'Within a thread\nblock cluster'), b['cluster_sync_8_blocks'], 'purple'),
        (L('跨 SM 一对一交接', 'One-to-one\ncross-SM handoff'), sum(b['handoff_release_acquire']) / 2, 'blue'),
        (L('全部 SM 汇合', 'All-SM barrier'), b['all_sm_counter_barrier'][0], 'teal'),
        (L('全部 SM 汇合\n并送达激活向量', 'All-SM barrier\n+ vector delivery'), b['all_sm_gather_with_vector_bf16_8kib'], 'orange'),
    ]


def draw(out, root, english=False):
    L = lambda zh, en: en if english else zh
    m = json.loads((Path(root) / SOURCE).read_text())

    def save(f, n):
        out.save(f, 'figure-4-' + n)

    with plt.rc_context(STYLE):
        # 1. Where L2 sits: the only storage every SM reaches.
        f, a = canvas(5.0)
        for g, x0 in enumerate([0.03, 0.52]):
            box(a, x0, 0.58, 0.45, 0.39, '', 'gray')
            text(a, x0 + 0.02, 0.93, f'GPC {g}', 11.5, weight='bold')
            for i in range(3):
                box(a, x0 + 0.02 + i * 0.145, 0.68, 0.125, 0.19, L('SM\n共享内存', 'SM\nshared\nmemory'), 'blue', 11, weight='bold')
        a.add_patch(plt.Rectangle((0.04, 0.655), 0.285, 0.235, fill=False, ec=BORDER_COL['purple'], lw=1.4, ls='--'))
        text(a, 0.1825, 0.62, L('线程块簇', 'Thread block cluster'), 11, ha='center', weight='bold', color=TEXT_COL['purple'])
        box(a, 0.03, 0.38, 0.94, 0.12, L('片上交叉互连（Crossbar）', 'On-chip crossbar'), 'orange', 11.5, weight='bold')
        for i in range(6):
            box(a, 0.03 + i * 0.16, 0.17, 0.14, 0.11, L('L2 分片', 'L2 slice'), 'green', 11, weight='bold')
        text(a, 0.50, 0.12, L(f"L2 共 {m['device']['l2_mib']} MiB，每个分片负责一部分地址", f"{m['device']['l2_mib']} MiB of L2; each slice owns part of the addresses"), 11, ha='center', color=COL['muted'], weight='bold')
        box(a, 0.03, 0.02, 0.94, 0.07, L('显存控制器与显存', 'Memory controllers and GPU memory'), 'gray', 11, weight='bold')
        arrow(a, (0.4125, 0.68), (0.4125, 0.50), 'blue')
        arrow(a, (0.36, 0.38), (0.40, 0.28), 'orange')
        arrow(a, (0.44, 0.28), (0.74, 0.38), 'green')
        arrow(a, (0.7475, 0.50), (0.7475, 0.68), 'orange')
        text(a, 0.425, 0.54, L('① 写入', '① Write'), 11, weight='bold', color=TEXT_COL['blue'])
        text(a, 0.76, 0.54, L('② 读取', '② Read'), 11, weight='bold', color=TEXT_COL['orange'])
        save(f, 'l2-structure')

        # 2. One cross-SM handoff, time downward.
        f, a = canvas(5.2)
        for x, label, c in [(0.24, L('生产者 SM', 'Producer SM'), 'blue'), (0.56, L('L2 分片', 'L2 slice'), 'green'), (0.85, L('消费者 SM', 'Consumer SM'), 'orange')]:
            box(a, x - 0.11, 0.88, 0.22, 0.09, label, c, 11, weight='bold')
            a.plot([x, x], [0.05, 0.88], color='#94A3B8', lw=1.0, ls=':')

        def msg(x0, y0, x1, y1, label, above=True, col='blue'):
            arrow(a, (x0, y0), (x1, y1), col)
            text(a, (x0 + x1) / 2, (y0 + y1) / 2 + (0.032 if above else -0.032), label, 11, ha='center', weight='bold')

        P, S, C = 0.24, 0.56, 0.85
        msg(P, 0.82, S, 0.75, L('写数据', 'Write data'), True, 'blue')
        msg(S, 0.73, P, 0.66, L('确认', 'Ack'), False, 'green')
        a.plot([0.19, 0.19], [0.81, 0.67], color=BORDER_COL['purple'], lw=2.4)
        text(a, 0.02, 0.74, L('fence：\n等待确认', 'Fence:\nwait\nfor ack'), 11, weight='bold', color=TEXT_COL['purple'])
        msg(P, 0.52, S, 0.45, L('写标志', 'Write flag'), True, 'blue')
        msg(C, 0.80, S, 0.73, L('读标志', 'Read flag'), True, 'orange')
        msg(S, 0.71, C, 0.64, L('旧值', 'Old value'), False, 'gray')
        msg(C, 0.46, S, 0.39, L('读标志', 'Read flag'), True, 'orange')
        msg(S, 0.37, C, 0.30, L('新值', 'New value'), False, 'green')
        msg(C, 0.25, S, 0.18, L('读数据', 'Read data'), True, 'orange')
        msg(S, 0.16, C, 0.09, L('数据', 'Data'), False, 'green')
        a.plot([0.89, 0.89], [0.80, 0.30], color=BORDER_COL['orange'], lw=2.4)
        text(a, 0.91, 0.55, L('轮询', 'Poll'), 11, weight='bold', color=TEXT_COL['orange'])
        save(f, 'cross-sm-handoff')

        # 3. Cost of each synchronisation scope, measured.
        rows = ladder_rows(m, english)
        f, a = plot(3.8, left=0.31, bottom=0.19, right=0.95, top=0.92)
        for i, (label, ns, c) in enumerate(rows):
            a.barh(i, ns, color=COL[c], edgecolor=BORDER_COL[c], linewidth=1.2, height=0.55)
            a.text(ns + 22, i, f'{ns:,.0f} ns', va='center', fontsize=11, weight='bold', color=COL['ink'])
        a.set(
            yticks=range(len(rows)),
            yticklabels=[r[0] for r in rows],
            xlim=(0, 1280),
            xticks=range(0, 1201, 200),
            ylim=(len(rows) - 0.4, -0.6),
            xlabel=L('一次同步的实测时间（ns）', 'Time per synchronization (ns)'),
        )
        a.grid(axis='x', linestyle=':', alpha=0.45, color='#CBD5E1')
        save(f, 'sync-ladder')
    return {'source': SOURCE, 'ladder': [{'scope': r[0], 'ns': r[1]} for r in rows]}
