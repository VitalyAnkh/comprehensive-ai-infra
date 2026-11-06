"""Chapter 2: publication-grade book-size figures (420pt). Follows objects from one position to a full request."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from figure_style import (
    COL, BORDER_COL, HEADER_COL, TEXT_COL, STYLE,
    canvas, plot, text, box, card, badge, arrow, Exporter
)


def draw(here, data):
    out = Exporter(here)

    def save(f, name):
        out.save(f, 'figure-2-' + name)

    def cells(a, x, y, count, w=0.65, h=0.10, colors=None, labels=None):
        for j in range(count):
            box(
                a,
                x + j * w / count,
                y,
                w / count - 0.006,
                h,
                labels[j] if labels else str(j + 1),
                colors[j] if colors else 'blue',
                11,
            )

    def bars(name, labels, values, xlabel, theme='blue'):
        f, a = plot(4.1 if len(values) > 4 else 3.4, left=0.31)
        a.barh(
            range(len(values)),
            values,
            color=COL[theme],
            edgecolor=BORDER_COL[theme],
            linewidth=1.25,
            height=0.55,
        )
        a.set(yticks=range(len(values)), yticklabels=labels, xlim=(0, max(values) * 1.32), xlabel=xlabel)
        a.invert_yaxis()
        a.grid(axis='x', linestyle=':', alpha=0.45, color='#CBD5E1')
        for i, v in enumerate(values):
            a.text(v + max(values) * 0.025, i, f'{v:,.2f}', va='center', fontsize=11, weight='bold', color=COL['ink'])
        save(f, name)

    with plt.rc_context(STYLE):
        # 1 & 2. figure-2-1-dependencies & figure-2-causal-dependencies
        for mode, name, title in [
            ('rnn', '1-dependencies', '同层时间递推：必须先完成前一个 token'),
            ('causal', 'causal-dependencies', '因果注意力：各 token 读取上一层的因果前缀'),
        ]:
            f, a = canvas(3.8)
            text(a, 0.04, 0.93, title, 13, weight='bold')
            xs = [0.24, 0.45, 0.66, 0.87]
            ys = [0.20, 0.48, 0.76]
            for l, y in enumerate(ys):
                text(a, 0.04, y, ['输入层', '第 1 层', '第 2 层'][l], 11, weight='bold')
                for j, x in enumerate(xs):
                    if l:
                        for k in ([j] if mode == 'rnn' else range(j + 1)):
                            arrow(a, (xs[k], ys[l - 1] + 0.04), (x, y - 0.04), color=BORDER_COL['blue'])
                        if mode == 'rnn' and j:
                            arrow(a, (xs[j - 1] + 0.035, y), (x - 0.035, y), color=BORDER_COL['orange'])
                    a.scatter(x, y, s=420, facecolor=COL['blue'], edgecolor=BORDER_COL['blue'], linewidth=1.4, zorder=5)
            for j, x in enumerate(xs):
                text(a, x, 0.06, f'token {j+1}', 11, ha='center', weight='bold')
            save(f, name)

        # 3. figure-2-qkv-objects
        f, a = canvas(4.3)
        text(a, 0.04, 0.94, '从同一输入隐藏向量线性投影产生 Q、K、V 三种表示', 13, weight='bold')
        card(a, 0.25, 0.73, 0.50, 0.14, '当前 token 的隐藏向量 X', '输入形状 m × d', 'gray', title_size=11, body_size=11)
        for x, label, desc, c in [
            (0.04, '查询 Q', '主动检索特征', 'orange'),
            (0.36, '键 K', '位置特征标签', 'blue'),
            (0.68, '值 V', '待汇总内容表示', 'green'),
        ]:
            arrow(a, (0.50, 0.73), (x + 0.14, 0.62), color=BORDER_COL[c])
            card(a, x, 0.43, 0.28, 0.19, label, desc, c, title_size=11, body_size=11)
        badge(a, 0.50, 0.31, '查询 Q 与历史键 K 点积打分 → 归一化得到各位置权重 A', 'white', size=11)
        card(a, 0.06, 0.06, 0.88, 0.16, '按权重 A 加权汇总各位置的值 V，形成注意力输出', r'输出后再经线性投影 $W_o$ 返回主干维度 d', 'purple', title_size=11, body_size=11)
        arrow(a, (0.50, 0.26), (0.50, 0.22), color=BORDER_COL['purple'])
        save(f, 'qkv-objects')

        # 4. figure-2-causal-pairs
        f, a = canvas(3.8)
        text(a, 0.04, 0.94, '已有 2 个历史 token，本次再输入 3 个 token 的有效配对', 13, weight='bold')
        for i in range(3):
            y = 0.65 - i * 0.19
            text(a, 0.04, y + 0.065, f'新 token {i+1}', 11, weight='bold')
            for j in range(5):
                face = COL['blue'] if j < 2 else COL['green'] if j < 3 + i else '#F8FAFC'
                edge = BORDER_COL['blue'] if j < 2 else BORDER_COL['green'] if j < 3 + i else '#CBD5E1'
                a.add_patch(Rectangle((0.29 + j * 0.125, y), 0.118, 0.13, facecolor=face, edgecolor=edge, lw=1.1))
                if j < 3 + i:
                    text(a, 0.349 + j * 0.125, y + 0.065, '✓', 12, ha='center', weight='bold', color=COL['ink'])
        text(a, 0.415, 0.83, '已有历史 (S=2)', 11, ha='center', color=TEXT_COL['blue'], weight='bold')
        text(a, 0.73, 0.83, '本次输入 (P=3)', 11, ha='center', color=TEXT_COL['green'], weight='bold')
        box(a, 0.03, 0.05, 0.94, 0.11, '有效查询—键配对总数：历史配对 3×2 = 6 ＋ 新输入下三角配对 6 ＝ 12', 'orange', size=11)
        save(f, 'causal-pairs')

        # 5. figure-2-2-layer
        f, a = canvas(4.8)
        text(a, 0.04, 0.95, '单层 Transformer：注意力子层交换位置信息，前馈子层变换特征', 13, weight='bold')
        stages = [
            ('输入张量：每 token 4096 维隐藏向量', 'gray'),
            (r'RMSNorm → 注意力打分与汇总 → 输出投影 $W_o$', 'blue'),
            ('第一处残差连接：与注意力子层输入逐元素相加', 'green'),
            ('RMSNorm → SwiGLU 前馈网络（gate / up / down）', 'orange'),
            ('第二处残差连接：与前馈子层输入逐元素相加', 'green'),
        ]
        for i, (s, c) in enumerate(stages):
            y = 0.77 - i * 0.16
            box(a, 0.19, y, 0.76, 0.115, s, c, size=11)
            if i < 4:
                arrow(a, (0.57, y), (0.57, y - 0.045), color=BORDER_COL[c])
        for y1, y2 in [(0.77, 0.507), (0.45, 0.187)]:
            a.plot([0.19, 0.065, 0.065, 0.19], [y1 + 0.035, y1 + 0.035, y2, y2], color=BORDER_COL['green'], lw=1.3)
            arrow(a, (0.065, y2), (0.19, y2), color=BORDER_COL['green'])
        text(a, 0.05, 0.055, '左侧旁路原样保留子层输入，算完子层后逐元素相加', 11, color=TEXT_COL['green'], weight='bold')
        save(f, '2-layer')

        # 6. figure-2-ffn-gates
        f, a = canvas(4.0)
        box(a, 0.28, 0.81, 0.44, 0.12, '输入 X：m × 4096 维', 'gray', size=11, weight='bold')
        card(a, 0.04, 0.52, 0.41, 0.18, 'gate 投影 → SiLU 激活', 'm × 4096 → m × 12288', 'orange', title_size=11, body_size=11)
        card(a, 0.55, 0.52, 0.41, 0.18, 'up 线性升维投影', 'm × 4096 → m × 12288', 'blue', title_size=11, body_size=11)
        arrow(a, (0.50, 0.81), (0.245, 0.70), color=BORDER_COL['orange'])
        arrow(a, (0.50, 0.81), (0.755, 0.70), color=BORDER_COL['blue'])
        box(a, 0.14, 0.27, 0.72, 0.13, '逐元素相乘：SiLU(gate) ⊙ up（12288 维）', 'green', size=11, weight='bold')
        arrow(a, (0.245, 0.52), (0.40, 0.40), color=BORDER_COL['orange'])
        arrow(a, (0.755, 0.52), (0.60, 0.40), color=BORDER_COL['blue'])
        box(a, 0.08, 0.05, 0.84, 0.13, 'down 线性降维投影：m × 12288 → 返回 m × 4096 维', 'purple', size=11, weight='bold')
        arrow(a, (0.50, 0.27), (0.50, 0.18), color=BORDER_COL['purple'])
        save(f, 'ffn-gates')

        # 7 & 8. figure-2-history & figure-2-history-batch
        f, a = canvas(4.2)
        text(a, 0.04, 0.94, '自回归解码：每一步追加 1 个新 token，并重读全部已有上下文', 13, weight='bold')
        for i, n in enumerate([4, 5, 6, 7]):
            y = 0.71 - i * 0.18
            text(a, 0.03, y + 0.055, f'第 {i+1} 步', 11, weight='bold')
            cells(a, 0.20, y, n + 1, w=0.74, h=0.12, colors=['blue'] * n + ['orange'])
        box(a, 0.03, 0.04, 0.94, 0.10, '蓝色格：从缓存重读的历史位置 · 橙色格：本步新生成并追加的位置', 'gray', size=11)
        save(f, 'history')

        f, a = plot(3.5)
        B = np.arange(1, 65)
        W = data['teaching_diagrams']['history']['shared_weight_bytes'] / 2**30
        a.axhline(W, color=BORDER_COL['blue'], lw=2.2, label='每批共享读取一遍模型权重 (14.1 GiB)')
        a.plot(B, B * 8192 * 147456 / 2**30, color=BORDER_COL['orange'], lw=2.2, label='8K 上下文 KV 缓存 × 请求数 B')
        a.axvline(13, color='#64748B', ls=':', lw=1.4)
        a.scatter([13], [13 * 8192 * 147456 / 2**30], color=BORDER_COL['orange'], s=45, zorder=5)
        a.text(15, 16.5, '交点 B = 13', fontsize=11, weight='bold', color=TEXT_COL['orange'])
        a.set(xlim=(0, 64), ylim=(0, 75), xlabel='批内并发请求数 B', ylabel='单步理论读取量（GiB）')
        a.grid(True, linestyle=':', alpha=0.45, color='#CBD5E1')
        a.legend(loc='upper left', frameon=True)
        save(f, 'history-batch')

        # 9 & 10. figure-2-3-sharing & figure-2-4-cache
        f, a = canvas(5.6)
        for row, (title, groups) in enumerate([
            ('MHA：每个查询头各自拥有一组独立 KV', 4),
            ('GQA：每两个查询头共享一组 KV', 2),
            ('MQA：全部四个查询头共用唯一一组 KV', 1),
        ]):
            top = 0.95 - row * 0.32
            text(a, 0.04, top, title, 12.5, weight='bold')
            for j in range(4):
                x = 0.08 + j * 0.225
                box(a, x, top - 0.13, 0.15, 0.075, f'Q{j+1}', 'orange', 11, weight='bold')
                g = j if groups == 4 else j // 2 if groups == 2 else 0
                dest = 0.08 + (g + 0.5) * 0.825 / groups
                arrow(a, (x + 0.075, top - 0.13), (dest, top - 0.20), color=BORDER_COL['blue'])
            for g in range(groups):
                x = 0.08 + g * 0.825 / groups
                box(a, x, top - 0.28, 0.825 / groups - 0.02, 0.08, f'KV 组 {g+1}', 'blue', 11, weight='bold')
        save(f, '3-sharing')

        bars('4-cache', ['MHA：32 组', 'GQA：8 组', 'MQA：1 组'], data['figure_2_4']['qwen_variants_mib'], '8K 上下文状态容量（MiB）', theme='blue')

        # 11 & 12. figure-2-mla-paths & figure-2-mla-capacity
        f, a = canvas(4.8)
        text(a, 0.04, 0.94, '路径一（展开路径）：先将每个历史潜变量展开为各头 K', 12.5, weight='bold')
        card(a, 0.04, 0.67, 0.26, 0.16, '缓存潜变量 c', '512 维紧凑向量', 'blue', title_size=11, body_size=11)
        card(a, 0.37, 0.67, 0.26, 0.16, '上投影展开 K', r'c · $U_k$ 展开各头', 'green', title_size=11, body_size=11)
        card(a, 0.70, 0.67, 0.26, 0.16, '注意力点积打分', r'q · $K^\top$ 计算得分', 'purple', title_size=11, body_size=11)
        arrow(a, (0.30, 0.75), (0.37, 0.75), color=BORDER_COL['blue'])
        arrow(a, (0.63, 0.75), (0.70, 0.75), color=BORDER_COL['green'])
        text(a, 0.50, 0.58, r'展开路径对全部历史位置执行 K = c $U_k$，显存占用激增', 11, ha='center', color=TEXT_COL['green'], weight='bold')

        text(a, 0.04, 0.45, '路径二（紧凑权重吸收路径）：先将上投影吸收到当前查询 q', 12.5, weight='bold')
        card(a, 0.04, 0.19, 0.26, 0.16, '当前查询头 q', '仅单步查询位置', 'orange', title_size=11, body_size=11)
        card(a, 0.37, 0.19, 0.26, 0.16, '权重吸收变换', r'计算 q · $U_k^\top$', 'orange', title_size=11, body_size=11)
        card(a, 0.70, 0.19, 0.26, 0.16, '潜空间直接打分', '直接点积历史 c', 'purple', title_size=11, body_size=11)
        arrow(a, (0.30, 0.27), (0.37, 0.27), color=BORDER_COL['orange'])
        arrow(a, (0.63, 0.27), (0.70, 0.27), color=BORDER_COL['orange'])
        box(a, 0.66, 0.03, 0.30, 0.095, '直接读取缓存潜变量 c', 'blue', size=11, weight='bold')
        arrow(a, (0.81, 0.125), (0.81, 0.19), color=BORDER_COL['blue'])
        save(f, 'mla-paths')

        bars('mla-capacity', ['紧凑潜变量', '展开各头 KV'], data['figure_2_4']['k3_mla_paths_mib'], 'Kimi K3 的 24 层 MLA 状态（MiB）', theme='purple')

        # 13, 14, 15. figure-2-5-sparse, figure-2-sparse-capacity, figure-2-compression-steps
        f, a = canvas(4.4)
        text(a, 0.04, 0.94, '压缩稀疏注意力（CSA）：先形成块压缩条目，再由索引器筛选 top-k', 12.5, weight='bold')
        cells(a, 0.05, 0.73, 8, w=0.90, h=0.11)
        for i in range(2):
            arrow(a, (0.27 + i * 0.45, 0.72), (0.27 + i * 0.45, 0.60), color=BORDER_COL['green'])
            card(a, 0.09 + i * 0.45, 0.43, 0.36, 0.16, f'压缩条目 {i+1}（4:1 合成）', '512 维主记录 + 128 维索引', 'green', title_size=11, body_size=11)
        card(a, 0.04, 0.07, 0.26, 0.18, '1  当前查询', '生成索引查询向量', 'orange', title_size=11, body_size=11)
        card(a, 0.37, 0.07, 0.26, 0.18, '2  扫描轻量索引', '128 维索引打分排序', 'purple', title_size=11, body_size=11)
        card(a, 0.70, 0.07, 0.26, 0.18, '3  读取选中条目', '仅读 top-512 条目', 'green', title_size=11, body_size=11)
        arrow(a, (0.30, 0.16), (0.37, 0.16), color=BORDER_COL['orange'])
        arrow(a, (0.63, 0.16), (0.70, 0.16), color=BORDER_COL['purple'])
        save(f, '5-sparse')

        c = data['figure_2_5']['components']
        bars(
            'sparse-capacity',
            ['窗口状态', '压缩表示', '索引条目', '压缩缓冲'],
            [
                c['window_history_bytes'] / 2**20,
                c['compressed_history_bytes'] / 2**20,
                c['index_history_bytes'] / 2**20,
                data['figure_2_5']['compressor_buffer_bytes'] / 2**20,
            ],
            'DeepSeek V4-Flash 的 8K 上下文状态（MiB）',
            theme='green',
        )

        f, a = canvas(3.5)
        text(a, 0.04, 0.92, '块压缩缓冲的周期性更新与条目发布（以 4:1 为例）', 12.5, weight='bold')
        for i in range(4):
            x = 0.04 + i * 0.24
            box(a, x, 0.52, 0.20, 0.22, f'到达步 {i+1}\n块内进度 {i+1}/4', 'orange' if i == 3 else 'blue', 11, weight='bold')
            if i < 3:
                arrow(a, (x + 0.20, 0.63), (x + 0.24, 0.63), color=BORDER_COL['blue'])
        card(a, 0.49, 0.08, 0.47, 0.24, '凑满 4 个 token：发布压缩条目', '写入全局缓存与索引并清空缓冲', 'green', title_size=11, body_size=11)
        arrow(a, (0.86, 0.52), (0.76, 0.32), color=BORDER_COL['orange'])
        box(a, 0.04, 0.11, 0.41, 0.17, '前 3 步暂存于未完成块缓冲', 'gray', size=11)
        save(f, 'compression-steps')

        # 16 & 17. figure-2-recurrence & figure-2-6-hybrid
        f, a = canvas(3.9)
        text(a, 0.04, 0.93, '线性递推注意力：上下文增长，固定大小状态矩阵原地更新', 12.5, weight='bold')
        for x, title, c in [(0.05, r'旧状态 $\mathcal{S}_{t-1}$', 'blue'), (0.38, r'新外积 $k_t v_t^{\top}$', 'orange'), (0.71, r'新状态 $\mathcal{S}_t$', 'green')]:
            for i in range(3):
                for j in range(3):
                    a.add_patch(Rectangle((x + j * 0.08, 0.46 + i * 0.08), 0.078, 0.078, facecolor=COL[c], edgecolor=BORDER_COL[c], lw=1.0))
            text(a, x + 0.12, 0.35, title, 11.5, ha='center', weight='bold', color=TEXT_COL[c])
        text(a, 0.33, 0.58, '+', 16, ha='center', weight='bold')
        text(a, 0.665, 0.58, '=', 16, ha='center', weight='bold')
        box(a, 0.05, 0.08, 0.90, 0.14, r'当前查询读取固定状态矩阵：$o_t = q_t^{\top}\mathcal{S}_t$（状态体积与长度无关）', 'purple', size=11, weight='bold')
        save(f, 'recurrence')

        f, a = canvas(4.2)
        text(a, 0.04, 0.94, 'Qwen3.6 混合架构：按层交错注意力机制，统一接入 MoE 专家', 12.5, weight='bold')
        card(a, 0.04, 0.61, 0.42, 0.21, '30 层线性递推注意力', '固定大小 FP32 递推矩阵 + 短卷积', 'blue', title_size=11, body_size=11)
        card(a, 0.54, 0.61, 0.42, 0.21, '10 层完整注意力（GQA）', '随上下文增长的逐位置精确 KV', 'green', title_size=11, body_size=11)
        box(a, 0.18, 0.35, 0.64, 0.13, '层内 RMSNorm 归一化与残差相加连接', 'gray', size=11, weight='bold')
        arrow(a, (0.25, 0.61), (0.40, 0.48), color=BORDER_COL['blue'])
        arrow(a, (0.75, 0.61), (0.60, 0.48), color=BORDER_COL['green'])
        card(a, 0.04, 0.06, 0.43, 0.19, '路由专家：256 选 8', r'动态分派，每专家处理 $t_e$ 行', 'orange', title_size=11, body_size=11)
        card(a, 0.53, 0.06, 0.43, 0.19, '共享专家（1 个）', '无条件处理本次输入的全部 m 行', 'purple', title_size=11, body_size=11)
        arrow(a, (0.40, 0.35), (0.255, 0.25), color=BORDER_COL['orange'])
        arrow(a, (0.60, 0.35), (0.745, 0.25), color=BORDER_COL['purple'])
        save(f, '6-hybrid')

        # 18 & 19. figure-2-7-state-growth & figure-2-state-access
        comparison = json.loads((here / 'model-comparison.json').read_text())
        d = comparison['state_curves']
        names = dict(zip([x['model_id'] for x in comparison['models']], ['V4.1 Flash', 'Qwen3-8B', 'Qwen3.6', 'V4-Flash', 'Kimi K3']))
        palette_5 = ['#DC2626', BORDER_COL['blue'], BORDER_COL['purple'], BORDER_COL['green'], BORDER_COL['orange']]
        for field, name, ylabel in [
            ('resident_bytes', '7-state-growth', '单请求状态容量（GiB）'),
            ('accounted_access_bytes', 'state-access', '每步状态访问量（GiB）'),
        ]:
            f, a = plot(4.3, left=0.19, bottom=0.18)
            for key, col in zip(names, palette_5):
                a.plot(np.array(d['lengths']) / 1024, np.array(d[field][key]) / 2**30, 'o-', label=names[key], color=col, lw=2.0, markersize=5)
            a.set(xscale='log', yscale='log', xlabel='上下文长度（千 token，对数轴）', ylabel=ylabel + '，对数轴')
            a.grid(True, linestyle=':', alpha=0.45, color='#CBD5E1')
            a.legend(frameon=True, fontsize=11, loc='upper left')
            save(f, name)

        # 20. figure-2-expert-reuse
        f, a = canvas(4.1)
        text(a, 0.04, 0.94, 'MoE 批处理权重复用：总分派次数相同，命中的专家数相差 32 倍', 12.5, weight='bold')
        for y, title, n, c, bw in [
            (0.54, '情形一（均匀分散）：读取 1.5 GiB 专家权重', 256, 'blue', '每专家仅 2 行输入'),
            (0.11, '情形二（高度集中）：仅读 48 MiB 专家权重', 8, 'orange', '每专家复用 64 行输入'),
        ]:
            card(a, 0.04, y, 0.38, 0.22, '64 个并发 token', '每词选 8 → 共 512 次分派', 'gray', title_size=11, body_size=11)
            arrow(a, (0.42, y + 0.11), (0.52, y + 0.11), color=BORDER_COL[c])
            card(a, 0.52, y, 0.44, 0.22, f'命中 {n} 个不同路由专家', bw, c, title_size=11, body_size=11)
            text(a, 0.04, y + 0.26, title, 11.5, weight='bold', color=TEXT_COL[c])
        save(f, 'expert-reuse')

        # 21. figure-2-8-residual
        f, a = canvas(4.3)
        for i, (title, streams, desc) in enumerate([
            ('普通残差连接：仅保留单路 4096 维旁路', 1, '单路原输入与子层输出直接逐元素相加'),
            ('流形约束超连接（mHC）：并行维护 4 路状态', 4, '4 路状态先加权汇合为 4096 维供子层计算，算完再分发写回 4 路'),
        ]):
            y = 0.55 - i * 0.44
            text(a, 0.04, y + 0.33, title, 12.5, weight='bold')
            for j in range(streams):
                box(a, 0.04 + j * 0.10, y + 0.11, 0.08, 0.11, f'路 {j+1}', 'blue', 11, weight='bold')
            card(a, 0.54, y + 0.08, 0.42, 0.17, '子层计算（注意力 / MoE）', '输入宽度仍保持 4096 维', 'orange', title_size=11, body_size=11)
            arrow(a, (0.44 if streams == 4 else 0.13, y + 0.165), (0.54, y + 0.165), color=BORDER_COL['blue'])
            text(a, 0.04, y - 0.02, desc, 11, color=COL['muted'])
        save(f, '8-residual')

        # 22-27. Architecture & Resource comparison bars
        arch = data['teaching_diagrams']['architecture']
        model_labels = ['DeepSeek\nV4.1 Flash', 'Qwen3-8B', 'Qwen3.6', 'DeepSeek\nV4-Flash', 'Kimi K3']
        for key, label, name, th in [
            ('layers', '主干层数', 'architecture', 'blue'),
            ('hidden', '主干隐藏维度', 'architecture-width', 'purple'),
            ('experts', '每个 MoE 层的路由专家数', 'architecture-experts', 'orange'),
        ]:
            bars(name, model_labels, arch[key], label, theme=th)

        models = data['teaching_diagrams']['resource_comparison']['models']
        model_labels_c = ['DeepSeek\nV4.1 Flash', 'Qwen3-8B', 'Qwen3.6', 'DeepSeek\nV4-Flash', 'Kimi K3\n紧凑']
        for key, div, label, name, th in [
            ('uniform_bf16_bytes', 1e9, '完整 BF16 权重（GB）', 'resources', 'blue'),
            ('decode_matrix_flops', 1e9, '8K 上下文单步矩阵运算（GFLOPs）', 'resources-compute', 'orange'),
            ('state_8192_bytes', 2**20, '8K 上下文状态（MiB）', 'resources-state', 'green'),
        ]:
            bars(name, model_labels_c, [m[key] / div for m in models], label, theme=th)

        # 28. figure-2-v41-prefill-breakdown
        f, a = plot(4.2, left=0.33, bottom=0.18)
        f.subplots_adjust(top=0.84)
        stages_v41 = ['前 20 层编码器\n(8192 位置)', '后 20 层解码器\n及全局投影', '全模型 8K Prefill\n总矩阵运算']
        ref_vals = [142.139, 140.246, 282.386]
        ced_vals = [141.727, 2.219, 143.947]
        yy = np.arange(3)
        a.barh(yy - 0.18, ref_vals, height=0.32, color=COL['orange'], edgecolor=BORDER_COL['orange'], linewidth=1.2, label='40 层全层前向基线')
        a.barh(yy + 0.18, ced_vals, height=0.32, color=COL['blue'], edgecolor=BORDER_COL['blue'], linewidth=1.2, label='CED + 末尾 128 窗口重放')
        for y_i, v in zip(yy - 0.18, ref_vals):
            a.text(v + 5.0, y_i, f'{v:.2f}', va='center', fontsize=11, color=TEXT_COL['orange'], weight='bold')
        for y_i, v in zip(yy + 0.18, ced_vals):
            a.text(v + 5.0, y_i, f'{v:.2f}', va='center', fontsize=11, color=TEXT_COL['blue'], weight='bold')
        a.set(yticks=yy, yticklabels=stages_v41, xlim=(0, 355), xlabel='矩阵运算量（TFLOPs）')
        a.invert_yaxis()
        a.grid(axis='x', linestyle=':', alpha=0.45, color='#CBD5E1')
        a.legend(loc='upper left', bbox_to_anchor=(-0.15, 1.18), ncol=2, frameon=False)
        save(f, 'v41-prefill-breakdown')

        # 29 & 30. figure-2-long-context-compute & figure-2-long-context-state
        long_data = json.loads((here / 'long-context-comparison.json').read_text())
        f, a = plot(5.4, left=0.31, bottom=0.16)
        f.subplots_adjust(top=0.86)
        for start, offset, th, label in [(0, -0.18, 'blue', '8K 上下文'), (10, 0.18, 'orange', '1M 可见 token')]:
            vals = [r['matrix_flops'] / 1e9 for r in long_data['models'][start:start + 5]]
            yy = np.arange(5) + offset
            a.barh(yy, np.array(vals) - 1, left=1, height=0.30, color=COL[th], edgecolor=BORDER_COL[th], linewidth=1.2, label=label)
            for y_i, value in zip(yy, vals):
                a.text(value * 1.12, y_i, f'{value:,.2f}', va='center', fontsize=11, weight='bold', color=TEXT_COL[th])
        a.set(
            yticks=range(5),
            yticklabels=model_labels_c,
            xscale='log',
            xlim=(1, 22000),
            xlabel='单步矩阵运算（GFLOPs，对数轴）',
            ylim=(4.65, -0.65),
        )
        a.set_xticks([1, 10, 100, 1000, 10000], ['1', '10', '100', '1000', '10000'])
        a.minorticks_off()
        a.legend(loc='upper left', bbox_to_anchor=(0, 1.15), ncol=2, frameon=False)
        a.grid(axis='x', linestyle=':', alpha=0.45, color='#CBD5E1')
        save(f, 'long-context-compute')

        f, a = plot(5.4, left=0.31, bottom=0.16)
        f.subplots_adjust(top=0.86)
        for start, offset, th, label in [(0, -0.18, 'green', '8K 上下文'), (10, 0.18, 'purple', '1M 可见 token')]:
            vals = [r['state_bytes'] / 2**30 for r in long_data['models'][start:start + 5]]
            yy = np.arange(5) + offset
            a.barh(yy, vals, left=0.01, height=0.30, color=COL[th], edgecolor=BORDER_COL[th], linewidth=1.2, label=label)
            for y_i, value in zip(yy, vals):
                a.text((value + 0.01) * 1.15, y_i, f'{value:,.3f}', va='center', fontsize=11, weight='bold', color=TEXT_COL[th])
        a.set(
            yticks=range(5),
            yticklabels=model_labels_c,
            xscale='log',
            xlim=(0.01, 600),
            xlabel='单请求状态容量（GiB，对数轴）',
            ylim=(4.65, -0.65),
        )
        a.set_xticks([0.01, 0.1, 1, 10, 100], ['0.01', '0.1', '1', '10', '100'])
        a.minorticks_off()
        a.legend(loc='upper left', bbox_to_anchor=(0, 1.15), ncol=2, frameon=False)
        a.grid(axis='x', linestyle=':', alpha=0.45, color='#CBD5E1')
        save(f, 'long-context-state')

        # 31 & 32. figure-2-9-capacity & figure-2-history-capacity
        rows = data['figure_2_9']['capacity_rows']
        f, a = plot(3.9, left=0.27, bottom=0.24)
        left = np.zeros(3)
        for key, label, col in [('weight_bytes', '模型权重', 'blue'), ('workspace_bytes', '工作区预留', 'orange'), ('kv_bytes_per_request', '单请求 KV', 'green')]:
            vals = np.array([r[key] / 1e9 for r in rows])
            a.barh(range(3), vals, left=left, height=0.50, label=label, color=COL[col], edgecolor=BORDER_COL[col], linewidth=1.2)
            left += vals
        for i, r in enumerate(rows):
            a.plot([r['capacity_bytes'] / 1e9] * 2, [i - 0.35, i + 0.35], color='#B91C1C', lw=1.8, ls='--')
        a.set(
            yticks=range(3),
            yticklabels=['Qwen BF16\nRTX 4090', '70B 8-bit\nH100 SXM', '70B 4-bit\nH100 SXM'],
            xlim=(0, 88),
            xlabel='显存容量占用（GB）',
        )
        a.invert_yaxis()
        a.grid(axis='x', linestyle=':', alpha=0.45, color='#CBD5E1')
        a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
        save(f, '9-capacity')

        d_hc = data['figure_2_9']['history_capacity']
        bars('history-capacity', ['8K 上下文', '32K 上下文'], d_hc['maximum_requests'], '容量允许的并发请求数', theme='green')

        # 33. ch02-dit-vae-memory-and-dllm-attention (NEW 420pt figure for Batch 1)
        f, a = canvas(6.3)
        card(a, 0.03, 0.53, 0.94, 0.44, 'A  多模态扩散算力与 3D 因果 VAE 特征缓存克隆尖峰', '', 'blue', title_size=11.5)
        card(
            a, 0.055, 0.72, 0.43, 0.16,
            'Qwen-Image-2512 (20B)',
            '100 次调用 · 7830.4 TFLOPs\n占全链路算力 99.85%',
            'orange', title_size=11, body_size=11
        )
        card(
            a, 0.515, 0.72, 0.43, 0.16,
            'FLUX.2-klein-4B (蒸馏)',
            '4 步单分支 · 139.3 TFLOPs\n算力降 56.2× · VAE 248 MiB',
            'green', title_size=11, body_size=11
        )
        card(
            a, 0.055, 0.55, 0.89, 0.145,
            '3D 因果 VAE 解码：算力仅占 0.15% (11.7 TFLOPs)，主导显存尖峰',
            '30 槽位缓存 1.125 GiB 经 .clone() 翻倍至 2.250 GiB + 1.0 GiB 激活',
            'purple', title_size=11, body_size=11
        )

        card(a, 0.03, 0.03, 0.94, 0.47, 'B  注意力掩码拓扑与 KV 缓存跨步不变性边界（AR vs dLLM）', '', 'green', title_size=11.5)
        card(
            a, 0.055, 0.305, 0.89, 0.115,
            '1  自回归因果掩码（AR）：下三角隔离未来，历史 KV 严格不变',
            r'单步输入 $m = 1$（矩阵乘退化为向量乘，受限于 HBM 访存带宽）',
            'blue', title_size=11, body_size=11
        )
        card(
            a, 0.055, 0.175, 0.89, 0.115,
            '2  全序列双向去噪（朴素 dLLM）：双向互看，揭开词元改变全局状态',
            '历史 KV 无法跨去噪步冻结复用，每步须重算全序列激活',
            'orange', title_size=11, body_size=11
        )
        card(
            a, 0.055, 0.045, 0.89, 0.115,
            '3  分块半自回归扩散（Block dLLM）：块间因果掩码 + 块内双向去噪',
            r'冻结提示词与历史块 KV；块内 $m = B_{\mathrm{blk}}$ 并行重算（转向算力受限）',
            'green', title_size=11, body_size=11
        )
        out.save(f, 'ch02-dit-vae-memory-and-dllm-attention')

        # 34 & 35. figure-2-10-request & figure-2-request-compute
        f, a = canvas(4.8)
        text(a, 0.04, 0.94, '输入 128 个 token、生成返回 4 个输出 token 的调用链', 13, weight='bold')
        for i in range(4):
            y = 0.72 - i * 0.21
            box(a, 0.03, y, 0.24, 0.135, '输入 128 个' if i == 0 else rf'输入 $y_{i}$', 'blue', 11, weight='bold')
            box(a, 0.36, y, 0.27, 0.135, 'Prefill 调用' if i == 0 else f'Decode 调用 {i}', 'orange', 11, weight='bold')
            box(a, 0.72, y, 0.25, 0.135, rf'产出 $y_{i+1}$', 'green', 11, weight='bold')
            arrow(a, (0.27, y + 0.067), (0.36, y + 0.067), color=BORDER_COL['blue'])
            arrow(a, (0.63, y + 0.067), (0.72, y + 0.067), color=BORDER_COL['orange'])
            text(a, 0.495, y - 0.042, f'调用后缓存保留 {128+i} 个位置状态', 11, ha='center', color=COL['muted'])
        save(f, '10-request')

        bars(
            'request-compute',
            model_labels_c,
            [r['matrix_flops'] / 1e12 for r in comparison['requests']],
            '完整请求矩阵运算（TFLOPs）',
            theme='blue',
        )

        # 36. ch02-capability-density-and-kv-pareto (Path B Fig 2-40)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(420 / 72, 4.3))
        f.subplots_adjust(left=0.14, right=0.96, bottom=0.22, top=0.85, wspace=0.50)
        for ax in (ax1, ax2):
            ax.spines[['top', 'right']].set_visible(False)

        models_cd = ['Llama2\n70B', 'Llama3\n8B', 'Gemma2\n9B']
        weights_gb = [140.0, 16.0, 18.5]
        mmlu_scores = [69.7, 66.6, 71.3]
        x_cd = np.array([0.0, 1.35, 2.75])
        bars_cd = ax1.bar(x_cd, weights_gb, width=0.56, color=[COL['orange'], COL['blue'], COL['green']], edgecolor=[BORDER_COL['orange'], BORDER_COL['blue'], BORDER_COL['green']], linewidth=1.2)
        for xi, w_val, m_val in zip(x_cd, weights_gb, mmlu_scores):
            w_str = f'{w_val:.0f} GB' if w_val in (140.0, 16.0) else f'{w_val:.1f} GB'
            ax1.text(xi, w_val + 4.5, f'{w_str}\n({m_val})', ha='center', va='bottom', fontsize=11, weight='bold', color=COL['ink'])
        ax1.set_xticks(x_cd, models_cd)
        ax1.set_xlim(-0.68, 3.45)
        ax1.set_ylim(0, 185)
        ax1.set_ylabel('BF16 权重（GB，括号为 MMLU）', fontsize=11)
        ax1.set_title('A  能力密度与权重压缩', loc='left', fontsize=12, weight='bold', pad=8)
        ax1.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E1')

        # Panel B: 1M context State Capacity (GiB) vs Attention+Index Compute (GFLOPs)
        pts_1m = [
            ('V4.1', 3.138, 25.23, 'green', (7, -8)),
            ('V4', 6.735, 113.80, 'teal', (-24, 10)),
            ('Qwen3.6', 20.060, 171.80, 'purple', (6, -15)),
            ('K3', 27.423, 5257.04, 'orange', (-12, -16)),
            ('Qwen3-8B', 144.000, 618.48, 'blue', (-48, 11)),
        ]
        pareto_x = [3.138, 6.735, 20.060, 144.000]
        pareto_y = [25.23, 113.80, 171.80, 618.48]
        ax2.plot(pareto_x, pareto_y, ls='--', lw=1.3, color='#64748B', zorder=2)
        for name_m, sx, sy, col_m, (dx, dy) in pts_1m:
            ax2.scatter([sx], [sy], s=60, color=BORDER_COL[col_m], facecolor=COL[col_m], linewidth=1.6, zorder=4)
            ax2.annotate(name_m, (sx, sy), xytext=(dx, dy), textcoords='offset points', fontsize=11, weight='bold', color=COL['ink'])
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlim(1.8, 380)
        ax2.set_ylim(12, 12000)
        ax2.set_xlabel('1M 状态容量（GiB）', fontsize=11)
        ax2.set_ylabel('1M 注意力运算（GFLOPs）', fontsize=11)
        ax2.set_title('B  1M 状态—算力前沿', loc='left', fontsize=12, weight='bold', pad=8)
        ax2.grid(True, which='major', linestyle=':', alpha=0.5, color='#CBD5E1')
        out.save(f, 'ch02-capability-density-and-kv-pareto')

    from core_principles_figures import draw as draw_principles
    draw_principles(2, out)
    from v41_case_figures import draw as draw_v41
    draw_v41(2, out)
    return out.finish()
