"""V4/V4.1 conversation diagrams in the shared Hands-On book style."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
from figure_style import COL, BORDER_COL, STYLE, canvas, plot, text, box, card, badge, arrow

ROOT = Path(__file__).resolve().parents[1]


def draw(ch, out):
    data = json.loads((ROOT / 'calculations/results/v41-throughline.json').read_text())
    with plt.rc_context(STYLE):
        if ch == 2:
            f, a = canvas(5.5)
            card(a, 0.05, 0.81, 0.90, 0.13, '输入提示词序列（Prompt Tokens）', '先完整通过前 20 层编码器（Encoder）', 'blue')
            card(a, 0.05, 0.51, 0.41, 0.18, '编码器末层表示投影', '生成 4 个锚点层的全局 MLA KV', 'green')
            card(a, 0.54, 0.51, 0.41, 0.18, '早期丢弃（CED）截断', '仅保留末尾 ≤128 个位置输出', 'orange')
            arrow(a, (0.255, 0.81), (0.255, 0.69), color='blue')
            arrow(a, (0.745, 0.81), (0.745, 0.69), color='orange')
            card(a, 0.54, 0.24, 0.41, 0.18, '后 20 层解码器局部重放', '仅对末尾窗口构建局部 SWA', 'orange')
            arrow(a, (0.745, 0.51), (0.745, 0.42), color='orange')
            a.plot([0.255, 0.255], [0.51, 0.33], color=BORDER_COL['green'], lw=1.3)
            arrow(a, (0.255, 0.33), (0.54, 0.33), color='green')
            badge(a, 0.255, 0.42, '全局 KV 供解码器跨层读取', 'green', size=11)
            card(a, 0.05, 0.04, 0.90, 0.13, '自回归生成新 token（Decode）', '穿行完整 40 层：20 层编码器 + 20 层解码器', 'purple')
            arrow(a, (0.745, 0.24), (0.745, 0.17), color='purple')
            out.save(f, 'figure-2-v41-ced-path')

            f, a = canvas(5.2)
            text(a, 0.24, 0.95, 'V4-Flash：每层各自保存状态', 12.0, ha='center')
            text(a, 0.75, 0.95, 'V4.1 Flash：4 锚点层跨层共享', 12.0, ha='center')
            for i in range(3):
                y = 0.70 - i * 0.21
                card(a, 0.02, y, 0.21, 0.15, f'层 {i + 1}', '独立注意力层', 'orange')
                card(a, 0.29, y, 0.18, 0.15, '本层 KV', '独立驻留', 'blue')
                arrow(a, (0.29, y + 0.075), (0.23, y + 0.075), color='blue')
            for i, (label, ratio, users) in enumerate([
                ('锚点层 2', '压缩比 2:1', '共享给层 3–7'),
                ('锚点层 8', '压缩比 2:1', '共享给层 9–13'),
                ('锚点层 14', '压缩比 2:1', '共享给层 15–19'),
                ('锚点层 20', '全分辨率 1:1', '共享给层 21–39'),
            ]):
                y = 0.72 - i * 0.18
                card(a, 0.52, y, 0.19, 0.135, label, ratio, 'blue', title_size=11.0, body_size=11.0)
                box(a, 0.75, y, 0.22, 0.135, users, 'green', 11)
                arrow(a, (0.71, y + 0.067), (0.75, y + 0.067), color='green')
            badge(a, 0.50, 0.105, '各层仍保留独立查询 Q 与 128-token 局部 SWA', 'gray', size=11)
            text(a, 0.5, 0.025, '层编号从 0 开始；全局状态合计压缩至 890 B/token', 11, ha='center', color=COL['muted'])
            out.save(f, 'figure-2-v41-sharing')

            f, a = canvas(5.1)
            card(a, 0.08, 0.78, 0.84, 0.15, '阶段 1 · Full 全量索引扫描（锚点层）', '扫描全量历史压缩索引（64 B/条目），每块含 8 个缓存条目', 'blue')
            card(a, 0.08, 0.54, 0.84, 0.15, '阶段 2 · 截断生成有界候选池（Candidate Pool）', '保留得分最高的最多 16,384 个候选条目，隔离超长历史长尾', 'green')
            card(a, 0.08, 0.30, 0.84, 0.15, '阶段 3 · Reindex 候选池内重选', '仅在 16,384 个候选条目内重排选出 Top-512 精确条目', 'orange')
            card(a, 0.08, 0.06, 0.84, 0.15, '阶段 4 · Reuse 跨层直接复用索引', '后续跟随层直接读取已选出的 512 个条目，免去重复索引扫描', 'purple')
            for y, c in [(0.78, 'blue'), (0.54, 'green'), (0.30, 'orange')]:
                arrow(a, (0.50, y), (0.50, y - 0.09), color=c)
            out.save(f, 'figure-2-v41-selection')
        elif ch == 3:
            f, a = canvas(4.8)
            text(a, 0.04, 0.94, '从空状态处理 8,192 token 输入的前向激活体积对比', 13.5)
            card(a, 0.04, 0.71, 0.92, 0.14, '传统全层 Prefill', '8,192 token × 40 层 = 327,680 token·层', 'blue')
            card(a, 0.04, 0.44, 0.58, 0.17, 'CED 编码器阶段', '8,192 token × 前 20 层 = 163,840 token·层', 'green')
            card(a, 0.66, 0.44, 0.30, 0.17, '解码器局部重放', '128 × 后 20 层 = 2,560', 'orange')
            badge(a, 0.50, 0.34, '主干激活规模：327,680 → 166,400 token·层（缩减近一半）', 'green', size=11.5)
            card(a, 0.04, 0.12, 0.92, 0.14, '需单独计入的固定与辅助项', '全局 KV 投影、注意力核心运算及 MoE 路由开销', 'gray')
            text(a, 0.5, 0.035, '进入逐 token Decode 阶段后仍穿行完整 40 层主干', 11, ha='center', color=COL['muted'])
            out.save(f, 'figure-3-v41-ced')
        elif ch == 5:
            f, axes = plt.subplots(2, 1, figsize=(420 / 72, 5.3))
            f.subplots_adjust(left=0.25, right=0.94, top=0.92, bottom=0.13, hspace=0.62)
            rows = data['cache_cases'][0]['models']
            for ax, key, title in [
                (axes[0], 'global_history_bytes', '8K 上下文全局 KV 驻留量'),
                (axes[1], 'decode_selected_history_read_bytes', '8K 单步注意力 KV 与索引读取量'),
            ]:
                vals = [r[key] / 2**20 for r in rows]
                ax.spines[['top', 'right']].set_visible(False)
                ax.barh([1, 0], vals, color=[COL['blue'], COL['green']], height=0.48, edgecolor=[BORDER_COL['blue'], BORDER_COL['green']], linewidth=1.2)
                for y, v in zip([1, 0], vals):
                    ax.text(v + 0.5, y, f'{v:.3f} MiB', va='center', fontsize=11, weight='medium')
                ax.set(
                    yticks=[1, 0],
                    yticklabels=['V4-Flash', 'V4.1 Flash'],
                    xlim=(0, 36),
                    ylim=(-0.65, 1.7),
                    xticks=[0, 10, 20, 30],
                    xlabel='数据量（MiB）',
                )
                ax.grid(axis='x', alpha=0.35)
                ax.set_title(title, fontsize=13, weight='bold')
            out.save(f, 'figure-5-v41-traffic')
        elif ch == 8:
            f, a = canvas(5.9)
            text(a, 0.04, 0.95, '多轮对话前缀命中判断与 SWA 局部状态按需恢复路径', 13.5)
            branches = [
                (0.73, '路径 A · 热缓存全命中', '全局 KV 与编码器 SWA 均在显存', '直接送入编码器', '仅处理本轮新增输入 token', 'green'),
                (0.50, '路径 B · 仅全局锚点 KV 命中', '已淘汰占一半容量的 40 层 SWA', '末尾 128-token 局部重放', '重建编码器 SWA 后处理新输入', 'blue'),
                (0.27, '路径 C · 全局 KV 未命中', '前缀超出缓存或首次到达', '编码器重算缺失前缀', '补齐全局 KV 与 SWA 后处理新输入', 'orange'),
            ]
            for y, ltitle, lbody, rtitle, rbody, col in branches:
                card(a, 0.03, y, 0.44, 0.16, ltitle, lbody, col, title_size=11.0, body_size=11.0)
                card(a, 0.57, y, 0.40, 0.16, rtitle, rbody, 'orange' if col == 'orange' else 'green', title_size=11.0, body_size=11.0)
                arrow(a, (0.47, y + 0.08), (0.57, y + 0.08), color=col)
            badge(a, 0.50, 0.205, '三条路径汇合后统一进入解码器阶段', 'purple', size=11)
            card(a, 0.03, 0.04, 0.94, 0.125, '解码器 20 层末尾窗口重放 → 构建解码器局部 SWA → 自回归生成输出', '', 'purple', title_size=11.5)
            out.save(f, 'figure-8-v41-recovery')
        elif ch == 9:
            f, a = plot(4.1, left=0.25, bottom=0.20)
            route = data['routing']
            transfer = route['remote_transfer_ms']
            replay = route['remote_replay_ms']
            a.barh(2, 10, height=0.52, color=COL['gray'], edgecolor=BORDER_COL['gray'], linewidth=1.15)
            a.barh(1, transfer, height=0.52, color=COL['blue'], edgecolor=BORDER_COL['blue'], linewidth=1.15)
            a.barh(1, replay, left=transfer, height=0.52, color=COL['orange'], edgecolor=BORDER_COL['orange'], linewidth=1.15)
            a.barh(0, 20, height=0.52, color=COL['gray'], edgecolor=BORDER_COL['gray'], linewidth=1.15)
            a.text(10.5, 2, '10 ms', va='center', fontsize=11, weight='medium')
            a.text(route['remote_ready_ms'] + 0.5, 1, '12.666 ms', va='center', fontsize=11, weight='bold', color='#1E3A8A')
            a.text(20.5, 0, '20 ms', va='center', fontsize=11, weight='medium')
            a.text(2.3, 1, '跨网传输', ha='center', va='center', fontsize=11, weight='medium', color='#1E3A8A')
            a.text(8.7, 1, '编码器恢复', ha='center', va='center', fontsize=11, weight='medium', color='#7C2D12')
            a.set(
                yticks=[2, 1, 0],
                yticklabels=['实例 A：短排队', '实例 B：拉取与恢复', '实例 A：长排队'],
                xlim=(0, 26.5),
                ylim=(-0.7, 2.9),
                xlabel='首包开始前的准备时间（ms）',
            )
            a.grid(axis='x', alpha=0.35)
            out.save(f, 'figure-9-v41-routing')
