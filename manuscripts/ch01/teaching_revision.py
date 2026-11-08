"""Chapter 1: publication-grade book-size figures (420pt). Inputs come from build.py's locked evidence."""
import numpy as np
import matplotlib.pyplot as plt
from figure_style import (
    COL, BORDER_COL, HEADER_COL, TEXT_COL, STYLE,
    canvas, plot, text, box, card, badge, arrow, Exporter
)


def draw(here, data):
    out = Exporter(here)
    with plt.rc_context(STYLE):
        # 1. figure-1-1-panorama: Six-layer AI Infra vertical stack + cross-layer governance
        f, a = canvas(6.0)
        layers = [
            ('L6', '应用与任务', '对话、编程助手、多模态 · 延迟与吞吐 SLO', 'orange'),
            ('L5', '模型与负载', '稠密/MoE 架构、长上下文 · 算力与显存需求', 'blue'),
            ('L4', '训练与推理系统', '分布式并行、请求路由、连续批处理与 KV 管理', 'green'),
            ('L3', '算子与编译运行时', '算子融合、计算图编译、显存分配与异步流调度', 'purple'),
            ('L2', '处理器与存储', 'GPU / NPU / CPU · SRAM、HBM 与持久化存储', 'blue'),
            ('L1', '互联与数据中心', '片间互联（Scale-Up）、跨机网络、供电与散热', 'gray'),
        ]
        for i, (lv, title, body, c) in enumerate(layers):
            y = 0.855 - i * 0.135
            card(a, 0.03, y, 0.68, 0.112, f'{lv}  {title}', body, c, title_size=12, body_size=11)
            if i < 5:
                arrow(a, (0.37, y), (0.37, y - 0.023), color=BORDER_COL[c])
        card(
            a, 0.735, 0.18, 0.235, 0.787,
            '跨层平台治理',
            '资源调度\n算力分配\n齐步排程\n\n运行环境\n容器镜像\n故障隔离\n\n可观测性\n全栈追踪\n能效核算',
            'green', title_size=12, body_size=11
        )
        box(a, 0.03, 0.035, 0.94, 0.095, '共同物理边界：先进封装 · 机架供电与液冷散热 · 集群高可用', 'gray', size=11)
        out.save(f, 'figure-1-1-panorama')

        # 2. figure-1-2-request: Online inference request lifecycle
        f, a = canvas(4.7)
        for idx, (x, title, body, c) in enumerate([
            (0.02, '1  应用与客户端', '组装提示词与上下文', 'orange'),
            (0.36, '2  服务网关入口', '鉴权、限流与配额', 'gray'),
            (0.70, '3  全局请求路由', '感知负载选定实例', 'purple'),
        ], start=1):
            card(a, x, 0.74, 0.28, 0.20, title, body, c, title_size=12, body_size=11)
        arrow(a, (0.30, 0.84), (0.36, 0.84))
        arrow(a, (0.64, 0.84), (0.70, 0.84))

        card(a, 0.02, 0.05, 0.96, 0.56, '被选中的推理服务实例（单卡或多卡协同）', '', 'gray', title_size=13)
        card(a, 0.05, 0.14, 0.27, 0.34, '实例调度器', '分词编码与排队\n动态组成连续 Batch', 'green', title_size=12, body_size=11)
        card(a, 0.365, 0.14, 0.27, 0.34, 'CPU 运行时', '分配 KV 显存块\n异步下发算子', 'blue', title_size=12, body_size=11)
        card(a, 0.68, 0.14, 0.27, 0.34, 'GPU 与 HBM', '读写权重与 KV\n执行 Prefill/Decode', 'orange', title_size=12, body_size=11)
        arrow(a, (0.32, 0.31), (0.365, 0.31))
        arrow(a, (0.635, 0.31), (0.68, 0.31))
        arrow(a, (0.84, 0.74), (0.84, 0.61), color=BORDER_COL['purple'])
        text(a, 0.52, 0.655, '请求路由转发进入目标实例', 11, ha='center', color=TEXT_COL['purple'], weight='bold')
        out.save(f, 'figure-1-2-request')

        # 3. figure-1-weight-lifetime: Persistent weight vs per-step execution
        f, a = canvas(4.1)
        card(a, 0.03, 0.76, 0.28, 0.19, '持久化模型文件', '磁盘 / 对象存储', 'gray', title_size=12, body_size=11)
        card(a, 0.37, 0.76, 0.60, 0.19, 'GPU 显存（HBM 常驻权重）', '启动时一次性加载，后续各生成步反复读取', 'blue', title_size=12, body_size=11)
        arrow(a, (0.31, 0.855), (0.37, 0.855), color=BORDER_COL['blue'])
        for i, (x, s) in enumerate([(0.04, '第 1 步生成'), (0.37, '第 2 步生成'), (0.70, '第 3 步生成')], start=1):
            card(a, x, 0.21, 0.26, 0.31, s, 'GPU 计算单元\n读取权重并运算', 'green', title_size=12, body_size=11)
            arrow(a, (0.44 + (i - 1) * 0.23, 0.76), (x + 0.13, 0.52), color=BORDER_COL['blue'])
        badge(a, 0.48, 0.64, '每生成一个输出 token，均从 HBM 顺序读取权重', 'white', size=11)
        arrow(a, (0.30, 0.365), (0.37, 0.365), color=BORDER_COL['green'])
        arrow(a, (0.63, 0.365), (0.70, 0.365), color=BORDER_COL['green'])
        box(a, 0.04, 0.04, 0.92, 0.11, '自回归依赖：前一步输出的 token 与追加的 KV 状态成为下一步输入', 'orange', size=11)
        out.save(f, 'figure-1-weight-lifetime')

        # 4. figure-1-3-datacenter: Datacenter to Supernode hierarchy
        f, a = canvas(5.2)
        card(a, 0.03, 0.79, 0.43, 0.17, '在线入口与共享存储', '网关路由 · 数据集与检查点', 'gray', title_size=12, body_size=11)
        card(a, 0.54, 0.79, 0.43, 0.17, '其他超节点（Scale-Out）', '跨机架并行训练与推理副本', 'purple', title_size=12, body_size=11)
        box(a, 0.08, 0.60, 0.84, 0.105, '数据中心高速网络（RoCEv2 / InfiniBand 跨机互联）', 'green', size=12, weight='bold')
        arrow(a, (0.245, 0.79), (0.36, 0.705))
        arrow(a, (0.755, 0.79), (0.64, 0.705))

        card(a, 0.02, 0.03, 0.96, 0.49, '放大一个超节点（高带宽低延迟域 Scale-Up）', '', 'gray', title_size=12)
        card(a, 0.06, 0.27, 0.40, 0.15, '主机 CPU 与系统主存', '控制面调度 · 算子下发', 'blue', title_size=11, body_size=11)
        card(a, 0.56, 0.27, 0.38, 0.15, '可编程网卡（SmartNIC）', 'RDMA 零拷贝 · 协议卸载', 'green', title_size=11, body_size=11)
        arrow(a, (0.75, 0.60), (0.75, 0.52), color=BORDER_COL['green'])
        arrow(a, (0.46, 0.345), (0.56, 0.345))
        card(a, 0.06, 0.06, 0.38, 0.15, 'GPU 组 A + HBM 显存', '张量计算与本地高带宽显存', 'orange', title_size=11, body_size=11)
        card(a, 0.56, 0.06, 0.38, 0.15, 'GPU 组 B + HBM 显存', '张量计算与本地高带宽显存', 'orange', title_size=11, body_size=11)
        arrow(a, (0.25, 0.27), (0.25, 0.21))
        arrow(a, (0.75, 0.27), (0.75, 0.21))
        arrow(a, (0.44, 0.135), (0.56, 0.135), color=BORDER_COL['orange'])
        arrow(a, (0.56, 0.135), (0.44, 0.135), color=BORDER_COL['orange'])
        text(a, 0.50, 0.175, '内部互联', 11, ha='center', color=TEXT_COL['orange'], weight='bold')
        out.save(f, 'figure-1-3-datacenter')

        # 5. figure-1-4-numbers: Key latency numbers across orders of magnitude
        f, a = plot(3.6, left=0.27)
        names = ['L1/L2 缓存', '系统主存访问', '机房网络往返', '机械磁盘寻道']
        vals = [3.0, 100.0, 500000.0, 10000000.0]
        colors = [COL['purple'], COL['blue'], COL['green'], COL['orange']]
        edges = [BORDER_COL['purple'], BORDER_COL['blue'], BORDER_COL['green'], BORDER_COL['orange']]
        a.barh(names, vals, color=colors, edgecolor=edges, linewidth=1.3, height=0.52)
        a.set_xscale('log')
        a.set_xlim(1, 3e8)
        a.set_xlabel('典型访问延迟（ns，对数刻度）')
        a.invert_yaxis()
        a.grid(axis='x', linestyle=':', alpha=0.5, color='#CBD5E1')
        for i, (v, label) in enumerate(zip(vals, ['0.5–7 ns', '100 ns (0.1 μs)', '0.5 ms', '10 ms'])):
            a.text(v * 1.45, i, label, va='center', fontsize=11, color=COL['ink'], weight='bold')
        out.save(f, 'figure-1-4-numbers')

        # 6. figure-1-amdahl-coding: Amdahl's law speedup curves vs acceleratable fraction f
        f, a = plot(3.9, left=0.16, bottom=0.18)
        frac = np.linspace(0.0, 0.88, 200)
        sp_5 = 1.0 / ((1.0 - frac) + frac / 5.0)
        sp_10 = 1.0 / ((1.0 - frac) + frac / 10.0)
        sp_inf = 1.0 / (1.0 - frac)
        a.axvspan(20, 30, color=COL['orange'], alpha=0.45)
        a.axvspan(70, 80, color=COL['green'], alpha=0.45)
        a.plot(frac * 100, sp_inf, color='#64748B', ls=':', lw=1.8, label='理论上限 $1/(1-f)$')
        a.plot(frac * 100, sp_10, color=BORDER_COL['green'], ls='--', lw=2.0, label='局部加速 $s = 10$')
        a.plot(frac * 100, sp_5, color=BORDER_COL['blue'], ls='-', lw=2.3, label='局部加速 $s = 5$')
        a.scatter([20, 30, 70, 80], [1 / (0.8 + 0.2 / 5), 1 / (0.7 + 0.3 / 5), 1 / (0.3 + 0.7 / 5), 1 / (0.2 + 0.8 / 5)], color=BORDER_COL['blue'], s=36, zorder=5)
        a.text(
            25, 3.55, '大型企业团队\n(20%–30%)\n1.19×–1.32×',
            ha='center', va='center', fontsize=11, weight='bold', color=TEXT_COL['orange'],
            bbox=dict(boxstyle='round,pad=0.28', facecolor='#FFFFFF', edgecolor=BORDER_COL['orange'], lw=1.0, alpha=0.96), zorder=6
        )
        a.text(
            62, 4.85, '精简团队\n(70%–80%)\n2.27×–2.78×',
            ha='center', va='center', fontsize=11, weight='bold', color=TEXT_COL['green'],
            bbox=dict(boxstyle='round,pad=0.28', facecolor='#FFFFFF', edgecolor=BORDER_COL['green'], lw=1.0, alpha=0.96), zorder=6
        )
        a.set(
            xlim=(0, 90),
            ylim=(1.0, 6.4),
            xticks=[0, 20, 40, 60, 80],
            xticklabels=['0%', '20%', '40%', '60%', '80%'],
            xlabel='可加速环节在原任务中的耗时占比 f',
            ylabel='端到端总加速比（倍）',
        )
        a.grid(True, linestyle=':', alpha=0.45, color='#CBD5E1')
        a.legend(loc='upper left', frameon=True)
        out.save(f, 'figure-1-amdahl-coding')

        # 7. figure-1-capacity-path: 70B capacity budget vs 80 GB H100 limit
        capacity = data['capacity_example']
        bf16 = capacity['bf16_weight_bytes'] / 1e9
        int8 = capacity['int8_weight_and_metadata_bytes'] / 1e9
        f, a = plot(3.6, left=0.25)
        rows_cap = [
            (3, bf16, 'orange', f'{bf16:.2f} GB'),
            (2, bf16 / 2, 'blue', f'{bf16/2:.2f} GB'),
            (1, bf16 / 2, 'blue', f'{bf16/2:.2f} GB'),
            (0, int8, 'green', f'{int8:.2f} GB'),
        ]
        for y, v, c, desc in rows_cap:
            a.barh(y, v, height=0.55, color=COL[c], edgecolor=BORDER_COL[c], linewidth=1.3)
            a.text(v + 3.0, y, desc, va='center', fontsize=11, color=COL['ink'], weight='bold')
        a.axvline(80, color='#B45309', ls='--', lw=1.6)
        a.text(82, 3.62, '单卡容量 80 GB', fontsize=11, color='#B45309', weight='bold')
        a.axhline(0.5, color='#CBD5E1', lw=0.9)
        a.set(
            yticks=[3, 2, 1, 0],
            yticklabels=['BF16 单卡', 'BF16 卡 0', 'BF16 卡 1', '8 比特单卡'],
            xlim=(0, 195),
            ylim=(-0.55, 4.05),
            xlabel='权重及量化附加数据（GB）',
            xticks=[0, 40, 80, 120, 160],
        )
        out.save(f, 'figure-1-capacity-path')

        # 8. figure-1-read-path: Single-step weight read bandwidth lower bound
        f, a = canvas(3.2)
        card(a, 0.03, 0.46, 0.35, 0.44, 'HBM 显存', '常驻 70 GB 权重\n（每参数 1 字节）', 'blue', title_size=12, body_size=11)
        card(a, 0.62, 0.46, 0.35, 0.44, 'GPU 计算单元', '接收权重与激活\n完成乘加运算', 'green', title_size=12, body_size=11)
        arrow(a, (0.38, 0.68), (0.62, 0.68), color=BORDER_COL['blue'])
        text(a, 0.50, 0.77, '峰值带宽', 11, ha='center', color=TEXT_COL['blue'], weight='bold')
        text(a, 0.50, 0.56, '3350 GB/s', 12, ha='center', color=TEXT_COL['blue'], weight='bold')
        box(a, 0.03, 0.08, 0.94, 0.25, '单步权重读取下界：70 GB ÷ 3350 GB/s ≈ 20.90 ms（每步完整读取）', 'orange', size=11, weight='bold')
        out.save(f, 'figure-1-read-path')

        # 9. figure-1-5-budget: Compute vs memory lower bound under hardware upgrades
        f, a = plot(3.8, left=0.24)
        comp = data['teaching']['compute_ms']
        mem = data['teaching']['weight_read_ms']
        labels = ['原加速器', '算力翻倍', '带宽翻倍']
        y = np.arange(3)
        mem_vals = [mem, mem, mem / 2]
        comp_vals = [comp, comp / 2, comp]
        a.barh(y + 0.17, mem_vals, height=0.30, color=COL['blue'], edgecolor=BORDER_COL['blue'], linewidth=1.2, label='读取权重下界')
        a.barh(y - 0.17, comp_vals, height=0.30, color=COL['orange'], edgecolor=BORDER_COL['orange'], linewidth=1.2, label='矩阵计算下界')
        for i, v in enumerate(mem_vals):
            a.text(v + 0.45, i + 0.17, f'{v:.2f} ms', fontsize=11, va='center', color=TEXT_COL['blue'], weight='bold')
        for i, v in enumerate(comp_vals):
            a.text(v + 0.45, i - 0.17, f'{v:.2f} ms', fontsize=11, va='center', color=TEXT_COL['orange'])
        a.set(yticks=y, yticklabels=labels, xlim=(0, 26.5), xlabel='单步资源服务时间下界（ms）')
        a.invert_yaxis()
        a.legend(loc='lower right', frameon=True)
        out.save(f, 'figure-1-5-budget')

        # 10. figure-1-batch-reuse: Weight read amortized across 8 batched requests
        f, a = canvas(3.7)
        card(a, 0.04, 0.66, 0.92, 0.25, 'HBM 一次读取：70 GB 模型权重（耗时仍约 20.90 ms）', '同一份权重读入片上缓存后，复用于批内 8 条并发请求', 'blue', title_size=12, body_size=11)
        for i in range(8):
            x = 0.04 + i * 0.117
            box(a, x, 0.28, 0.102, 0.20, f'请求 {i+1}', 'green', size=11, weight='bold')
            arrow(a, (x + 0.051, 0.66), (x + 0.051, 0.48), color=BORDER_COL['blue'])
        box(a, 0.04, 0.05, 0.92, 0.15, '8 请求并发：整批读取仍约 20.90 ms → 每输出 token 分摊降至 2.61 ms', 'orange', size=11, weight='bold')
        out.save(f, 'figure-1-batch-reuse')

        # 11. figure-1-batch-transition: Crossing batch size B* ≈ 148
        d = data['teaching_diagrams']['batch_transition']
        b = np.array(d['batch'])
        f, a = plot(3.8)
        a.plot(b, comp * b, color=BORDER_COL['orange'], lw=2.0, label='矩阵计算 (0.1415 × B)')
        a.axhline(mem, color=BORDER_COL['blue'], lw=2.0, label='权重读取 (20.90 ms)')
        a.plot(b, np.maximum(comp * b, mem), color=COL['ink'], ls='--', lw=2.2, label='两者最大值')
        a.axvline(d['crossing_batch'], color='#64748B', ls=':', lw=1.5)
        a.scatter([d['crossing_batch']], [mem], color=BORDER_COL['orange'], s=55, zorder=5)
        a.text(d['crossing_batch'] + 12, mem + 4.5, '转折点 $B^* \\approx 148$', fontsize=11, color=COL['ink'], weight='bold')
        a.set(xlim=(0, 512), ylim=(0, 82), xlabel='批内并发请求数 B', ylabel='单步时间下界（ms）')
        a.legend(loc='upper left', frameon=True)
        out.save(f, 'figure-1-batch-transition')

        # 12. figure-1-batch-throughput: Ideal throughput upper bound vs batch size
        f, a = plot(3.5)
        a.plot(b, d['throughput'], color=BORDER_COL['blue'], lw=2.2, label='理想输出吞吐上界')
        a.fill_between(b, 0, d['throughput'], color=COL['blue'], alpha=0.35)
        a.axvline(d['crossing_batch'], ls=':', lw=1.5, color='#64748B')
        a.text(18, 4800, '带宽受限区\n随 B 线性增长', fontsize=11, color=TEXT_COL['blue'], weight='bold')
        a.text(165, 5600, '计算受限区（$B \\geq 148$）\n趋近 7067 token/s', fontsize=11, color=TEXT_COL['orange'], weight='bold')
        a.set(xlim=(0, 512), ylim=(0, 8200), xlabel='批内并发请求数 B', ylabel='理想输出吞吐（token/s）')
        out.save(f, 'figure-1-batch-throughput')

        # 13 & 14. figure-1-measured-throughput and figure-1-measured-tpot
        rows = data['measured_short_group']
        for key, ylabel, name, theme in [
            ('throughput', '整批输出吞吐（token/s）', 'figure-1-measured-throughput', 'blue'),
            ('tpot_ms', '每请求输出间隔 TPOT（ms）', 'figure-1-measured-tpot', 'orange'),
        ]:
            f, a = plot(3.4)
            vals = [r[key] for r in rows]
            xs = list(range(len(rows)))
            a.plot(xs, vals, 'o-', color=BORDER_COL[theme], markerfacecolor=COL[theme], markeredgewidth=1.8, markersize=7, lw=2.2)
            a.fill_between(xs, 0, vals, color=COL[theme], alpha=0.28)
            for i, v in enumerate(vals):
                a.annotate(f'{v:.2f}', (i, v), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=11, weight='bold', color=TEXT_COL[theme])
            a.set(
                xlim=(-0.45, 3.45),
                ylim=(0, max(vals) * 1.26),
                xticks=xs,
                xticklabels=[f"B={r['batch']}" for r in rows],
                xlabel='并发请求数档位（Qwen3-8B BF16 实测）',
                ylabel=ylabel,
            )
            out.save(f, name)

        # 15. figure-1-step-breakdown: Measured TPOT breakdown vs physical bandwidth lower bound
        f, a = plot(3.6, left=0.31)
        stages_bd = ['初始实测\n(含逐步日志)', '关闭日志后\n客户端实测', 'GPU 事件计时\n(仅模型前向)', '理想访存下界\n(权重+历史KV)']
        ms_bd = [26.5, 12.1, 10.7, 8.62]
        mbu_bd = ['33%', '71%', '81%', '100%']
        cols_bd = [COL['gray'], COL['orange'], COL['blue'], COL['green']]
        edges_bd = [BORDER_COL['gray'], BORDER_COL['orange'], BORDER_COL['blue'], BORDER_COL['green']]
        y_bd = np.arange(4)
        a.barh(y_bd, ms_bd, height=0.54, color=cols_bd, edgecolor=edges_bd, linewidth=1.3)
        for i, (v, mbu) in enumerate(zip(ms_bd, mbu_bd)):
            a.text(v + 0.6, i, f'{v:.2f} ms (MBU {mbu})', va='center', fontsize=11, weight='bold', color=COL['ink'])
        a.set(yticks=y_bd, yticklabels=stages_bd, xlim=(0, 46), xlabel='单步解码耗时（ms）与显存带宽利用率（MBU）')
        a.invert_yaxis()
        a.grid(axis='x', linestyle=':', alpha=0.5, color='#CBD5E1')
        out.save(f, 'figure-1-step-breakdown')

        # 16. figure-1-design-tpu: Domain-specific systolic matrix architecture
        f, a = canvas(3.7)
        card(a, 0.02, 0.05, 0.96, 0.90, '定制 AI 加速器（以 TPU 为例）：围绕矩阵运算重组芯片面积', '', 'gray', title_size=12)
        card(a, 0.05, 0.24, 0.24, 0.48, 'HBM 与输入缓冲', '批量预取权重\n与激活张量', 'blue', title_size=11, body_size=11)
        card(a, 0.34, 0.24, 0.32, 0.48, '二维脉动矩阵阵列', '权重固定，激活与部分和\n在乘加单元间直接传递', 'orange', title_size=11, body_size=11)
        card(a, 0.71, 0.24, 0.24, 0.48, '向量与输出缓冲', '融合激活函数\n写回片上缓存', 'green', title_size=11, body_size=11)
        arrow(a, (0.29, 0.48), (0.34, 0.48), color=BORDER_COL['blue'])
        arrow(a, (0.66, 0.48), (0.71, 0.48), color=BORDER_COL['orange'])
        box(a, 0.05, 0.08, 0.90, 0.10, '设计收益：省去通用指令译码与寄存器往返，提升每瓦矩阵算力', 'purple', size=11)
        out.save(f, 'figure-1-design-tpu')

        # 17. figure-1-design-smartnic: Programmable SmartNIC / DPU data-path & policy offload
        f, a = canvas(3.7)
        card(a, 0.02, 0.05, 0.96, 0.90, '可编程网卡（SmartNIC）：在数据必经之路上卸载协议与策略', '', 'gray', title_size=12)
        card(a, 0.05, 0.24, 0.24, 0.48, '数据中心网络', '以太网 / RoCE\n微秒级报文流', 'blue', title_size=11, body_size=11)
        card(a, 0.34, 0.24, 0.32, 0.48, '可编程网卡', '硬件流水线 + Arm 核\n完成分包、加密与卸载', 'orange', title_size=11, body_size=11)
        card(a, 0.71, 0.24, 0.24, 0.48, '主机 CPU 与 GPU', 'RDMA 直达显存\n释放主机核心', 'green', title_size=11, body_size=11)
        arrow(a, (0.29, 0.48), (0.34, 0.48), color=BORDER_COL['blue'])
        arrow(a, (0.66, 0.48), (0.71, 0.48), color=BORDER_COL['orange'])
        box(a, 0.05, 0.08, 0.90, 0.10, '设计边界：数据面卸载收益显著；控制策略卸载受跨 PCIe 延迟约束', 'purple', size=11)
        out.save(f, 'figure-1-design-smartnic')

        # 18. ch01-fleet-goodput-and-smartnic-offload: Dual-panel new figure (MPG waterfall + SmartNIC offload boundary)
        f, _ = canvas(4.0)
        f.clf()
        ax1 = f.add_axes([0.16, 0.18, 0.29, 0.70])
        ax2 = f.add_axes([0.59, 0.18, 0.36, 0.70])

        stages = ['总容量 C', '单卡分配', '齐步就绪 A', '有效进展 G', '参考时间 I']
        pcts = [100.0, 97.2, 83.3, 66.7, 33.3]
        bar_cols = [COL['gray'], COL['purple'], COL['blue'], COL['green'], COL['orange']]
        edge_cols = [BORDER_COL['gray'], BORDER_COL['purple'], BORDER_COL['blue'], BORDER_COL['green'], BORDER_COL['orange']]
        y_pos = np.arange(len(stages))
        ax1.barh(y_pos, pcts, height=0.55, color=bar_cols, edgecolor=edge_cols, linewidth=1.2)
        for i, p in enumerate(pcts):
            ax1.text(p + 3.0, i, f'{p:.1f}%', va='center', fontsize=11, weight='bold', color=COL['ink'])
        ax1.set_yticks(y_pos, stages)
        ax1.invert_yaxis()
        ax1.set_xlim(0, 155)
        ax1.set_xlabel('占总容量比例（%）', fontsize=11)
        ax1.set_title('A  机队效率漏斗', loc='left', fontsize=12, weight='bold', pad=8)
        ax1.grid(axis='x', linestyle=':', alpha=0.5, color='#CBD5E1')

        d_pct = np.linspace(0, 10, 101)
        ratio = 32.0 * (1.0 - d_pct / 100.0) / 30.0
        ax2.plot(d_pct, ratio, color=BORDER_COL['blue'], lw=2.2)
        ax2.axhline(1.0, color='#64748B', ls='--', lw=1.4)
        ax2.axvline(6.25, color=BORDER_COL['orange'], ls=':', lw=1.5)
        ax2.fill_between(d_pct, 1.0, ratio, where=(ratio >= 1.0), color=COL['green'], alpha=0.45)
        ax2.fill_between(d_pct, ratio, 1.0, where=(ratio < 1.0), color=COL['orange'], alpha=0.35)
        ax2.scatter([2.0, 6.25, 8.0], [32 * 0.98 / 30, 1.0, 32 * 0.92 / 30], color=[BORDER_COL['green'], BORDER_COL['orange'], '#DC2626'], s=45, zorder=5)
        ax2.text(1.0, 1.055, '+4.53%', fontsize=11, weight='bold', color=TEXT_COL['green'])
        ax2.text(6.55, 1.012, '6.25%', fontsize=11, weight='bold', color=TEXT_COL['orange'])
        ax2.text(6.65, 0.955, '-1.87%', fontsize=11, weight='bold', color='#B91C1C')
        ax2.set_xlim(0, 10.5)
        ax2.set_ylim(0.93, 1.085)
        ax2.set_xlabel('单核吞吐降幅 d（%）', fontsize=11)
        ax2.set_ylabel('相对总吞吐倍数', fontsize=11)
        ax2.set_title('B  策略卸载翻转边界', loc='left', fontsize=12, weight='bold', pad=8)
        ax2.grid(True, linestyle=':', alpha=0.5, color='#CBD5E1')
        out.save(f, 'ch01-fleet-goodput-and-smartnic-offload')

        # 19. figure-1-design-ub: Supernode Unified Bus (UB / NVLink Switch) architecture
        f, a = canvas(3.8)
        card(a, 0.02, 0.05, 0.96, 0.90, '超节点统一互联（UB / NVLink Switch）：对等高带宽资源池', '', 'gray', title_size=12)
        for i, (x, label, c) in enumerate([
            (0.05, '节点 0：加速器', 'blue'),
            (0.36, '节点 1：加速器', 'blue'),
            (0.67, '共享 CPU 与内存池', 'purple'),
        ]):
            card(a, x, 0.55, 0.28, 0.24, label, '计算与显存统一编址', c, title_size=11, body_size=11)
            arrow(a, (x + 0.14, 0.55), (x + 0.14, 0.43), color=BORDER_COL['green'])
            arrow(a, (x + 0.14, 0.43), (x + 0.14, 0.55), color=BORDER_COL['green'])
        card(
            a, 0.05, 0.21, 0.90, 0.22,
            '超节点统一互联交换矩阵（高带宽 · 低延迟 · 统一内存语义）',
            '支持跨设备直接 Load/Store 与集合通信，消除主机协议栈拷贝',
            'green', title_size=11, body_size=11
        )
        box(a, 0.05, 0.08, 0.90, 0.095, '适用场景：大模型张量并行（TP）与专家并行（EP）高速通信', 'orange', size=11)
        out.save(f, 'figure-1-design-ub')

        # 20. ch01-cloud-vs-ai-datacenter-and-patterns (Path B Fig 1-21)
        f, a = canvas(6.6)
        card(a, 0.02, 0.52, 0.96, 0.45, 'A  传统通用云 vs. AI 数据中心：六维架构重心迁移（D11-01）', '', 'blue', title_size=11.5)
        rows_a = [
            (0.755, '传统云：OS/进程接口\n一机切分多租户实例', 'AI 工厂：上下文/模型接口\n多卡聚合单一实例', 'blue', 'green'),
            (0.645, '传统云：异步 RPC 流量\n局部无状态副本容错', 'AI 工厂：高频集合通信\n全局同步屏障扩散', 'gray', 'orange'),
            (0.535, '传统云：资源超分复用\n看重 CPU/内存分配率', 'AI 工厂：齐步协同调度\n看重 SLO 产出与 MPG', 'purple', 'teal'),
        ]
        for y_row, left_lbl, right_lbl, c_l, c_r in rows_a:
            box(a, 0.045, y_row, 0.43, 0.092, left_lbl, c_l, size=11)
            box(a, 0.525, y_row, 0.43, 0.092, right_lbl, c_r, size=11)
            arrow(a, (0.475, y_row + 0.046), (0.525, y_row + 0.046), color=BORDER_COL['blue'])

        card(a, 0.02, 0.03, 0.96, 0.46, 'B  经典计算机系统六大设计模式在 LLM 系统中的映射（D11-03）', '', 'green', title_size=11.5)
        patterns = [
            (0.045, 0.255, '1 分层与缓存', '页缓存 → 前缀 KV\n与三级状态池', 'blue'),
            (0.355, 0.255, '2 重算换搬运', '重物化 → 检查点\n与 FA 在线重算', 'orange'),
            (0.665, 0.255, '3 旁路零拷贝', '内核旁路 → RDMA\n与 GDS 直通', 'teal'),
            (0.045, 0.055, '4 重叠与流水', '双缓冲 → 算子流水\n与通信重叠', 'green'),
            (0.355, 0.055, '5 批量与分摊', '批量 I/O → 连续批\n分摊权重读取', 'purple'),
            (0.665, 0.055, '6 解耦与池化', '存算分离 → PD/AF\n与集中 KV 池', 'gray'),
        ]
        for px, py, ptitle, pbody, pcol in patterns:
            card(a, px, py, 0.29, 0.17, ptitle, pbody, pcol, title_size=11, body_size=11)
        out.save(f, 'ch01-cloud-vs-ai-datacenter-and-patterns')

        # 21. ch01-token-cost-telescoping-chain (Path B Fig 1-22)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(420 / 72, 4.0))
        f.subplots_adjust(left=0.25, right=0.96, bottom=0.20, top=0.86, wspace=0.68)
        for ax in (ax1, ax2):
            ax.spines[['top', 'right']].set_visible(False)

        benchmarks = ['MMLU ≥ 64.8', 'MMLU ≥ 68.0', 'MMLU ≥ 86.0', 'GPQA ≥ 33.0', 'HumanEval ≥ 67']
        ratios_ep = [286, 29, 208, 313, 375]
        cols_ep = [COL['blue'], COL['gray'], COL['green'], COL['teal'], COL['orange']]
        edges_ep = [BORDER_COL['blue'], BORDER_COL['gray'], BORDER_COL['green'], BORDER_COL['teal'], BORDER_COL['orange']]
        y_ep = np.arange(len(benchmarks))
        ax1.barh(y_ep, ratios_ep, height=0.56, color=cols_ep, edgecolor=edges_ep, linewidth=1.2)
        for i, r_val in enumerate(ratios_ep):
            ax1.text(r_val + 12, i, f'{r_val}×', va='center', fontsize=11, weight='bold', color=COL['ink'])
        ax1.set_yticks(y_ep, benchmarks)
        ax1.invert_yaxis()
        ax1.set_xlim(0, 560)
        ax1.set_xlabel('价格下降倍数（×）', fontsize=11)
        ax1.set_title('A  固定能力价格降幅', loc='left', fontsize=12, weight='bold', pad=8)
        ax1.grid(axis='x', linestyle=':', alpha=0.5, color='#CBD5E1')

        steps_s = ['S0: 70B 1B', 'S1: 8B 1B', 'S2: 8B 0.5B', 'S3: H200']
        ms_vals = [20.90, 2.39, 1.19, 0.83]
        cols_s = [COL['orange'], COL['blue'], COL['green'], COL['teal']]
        edges_s = [BORDER_COL['orange'], BORDER_COL['blue'], BORDER_COL['green'], BORDER_COL['teal']]
        y_s = np.arange(len(steps_s))
        ax2.barh(y_s, ms_vals, height=0.54, color=cols_s, edgecolor=edges_s, linewidth=1.2)
        for i, v_ms in enumerate(ms_vals):
            ax2.text(v_ms + 0.8, i, f'{v_ms:.2f} ms', va='center', fontsize=11, weight='bold', color=COL['ink'])
        ax2.axvline(1.0, color='#B91C1C', ls='--', lw=1.3)
        ax2.set_yticks(y_s, steps_s)
        ax2.invert_yaxis()
        ax2.set_xlim(0, 33.0)
        ax2.set_xlabel('权重读取下界（ms）', fontsize=11)
        ax2.set_title('B  下界折叠 (25.1×)', loc='left', fontsize=12, weight='bold', pad=8)
        ax2.grid(axis='x', linestyle=':', alpha=0.5, color='#CBD5E1')
        out.save(f, 'ch01-token-cost-telescoping-chain')

        # 22. ch01-host-dispatch-ghost-wave-zoc (Path D Fig 1-23)
        f, a = canvas(6.9)
        card(a, 0.02, 0.51, 0.96, 0.46, 'A  主机调度旁路（ghOSt）：切换税与本地派发', '', 'blue', title_size=11.5)
        card(
            a, 0.045, 0.675, 0.43, 0.20,
            '传统跨代理往返切换',
            '切换耗时 3.32~4.04 μs\n4 μs 任务加两次 1 μs 等待\n延迟 +50% · 吞吐 -33.3%',
            'orange', title_size=11, body_size=11
        )
        card(
            a, 0.525, 0.675, 0.43, 0.20,
            'ghOSt 本地调度 (9ca0a1f)',
            '目标核本地派发仅 426 ns\nOpen → commit → succeeded\n状态字屏障消除抢占竞态',
            'green', title_size=11, body_size=11
        )
        arrow(a, (0.475, 0.775), (0.525, 0.775), color=BORDER_COL['green'])
        box(
            a, 0.045, 0.535, 0.91, 0.115,
            '微秒级控制面定律：1~10 μs 算子下发受限于内核切换税\n本地直接派发与三阶段原子屏障可将调度开销压缩近十倍',
            'gray', size=11
        )

        card(a, 0.02, 0.03, 0.96, 0.46, 'B  策略卸载（Wave）与虚拟智能网卡（ZOC）', '', 'green', title_size=11.5)
        card(
            a, 0.045, 0.21, 0.43, 0.20,
            'Wave 策略卸载 (ASPLOS \'26)',
            '16 核 Arm 卸载内存页策略\nAgent 内存 600 → 720 GiB\n32 核满载，瓶颈转入 CPU',
            'purple', title_size=11, body_size=11
        )
        card(
            a, 0.525, 0.21, 0.43, 0.20,
            'ZOC 虚拟网卡 (NSDI \'26)',
            '复用主机空闲核 + 服务 VM\n接管虚拟交换慢路径与策略\n随负载弹性扩缩，零定制硬件',
            'teal', title_size=11, body_size=11
        )
        box(
            a, 0.045, 0.055, 0.91, 0.12,
            '卸载判据：释放主存/核心的净增益须大于跨总线单核降幅 d',
            'blue', size=11
        )
        out.save(f, 'ch01-host-dispatch-ghost-wave-zoc')

        # 23. ch01-xprof-profinfer-critical-path (Path D Fig 1-24)
        f, a = canvas(6.4)
        card(a, 0.02, 0.49, 0.96, 0.48, 'A  全栈时间线与跨层关键路径有向无环图（DAG）', '', 'blue', title_size=11.5)
        lanes = [
            (0.79, 'L1 主机调度流', 'Python API → 连续批编排 → 算子下发', 'blue'),
            (0.65, 'L2 GPU 计算流', 'CUPTI 追踪 · Prefill/Decode 算子执行', 'orange'),
            (0.51, 'L3 网卡通信流', 'RDMA 队列 · AllReduce/All-to-All 传输', 'green'),
        ]
        for y_l, l_title, l_body, l_col in lanes:
            box(a, 0.045, y_l, 0.26, 0.105, l_title, l_col, size=11, weight='bold')
            box(a, 0.335, y_l, 0.62, 0.105, l_body, 'gray', size=11)
            arrow(a, (0.305, y_l + 0.052), (0.335, y_l + 0.052), color=BORDER_COL[l_col])

        card(a, 0.02, 0.03, 0.96, 0.43, 'B  三类剖析器协同定位关键路径暴露量', '', 'purple', title_size=11.5)
        profilers = [
            (0.045, 'XProf', '统一全局时间轴\nRoofline 瓶颈归因', 'blue'),
            (0.355, 'ProfInfer', 'eBPF 无侵入追踪\n捕获缺页与锁争用', 'teal'),
            (0.665, 'StriaTrace', '锚定跨卡同步屏障\n逆向定位慢节点流', 'orange'),
        ]
        for px, ptitle, pbody, pcol in profilers:
            card(a, px, 0.175, 0.29, 0.205, ptitle, pbody, pcol, title_size=11, body_size=11)
        box(
            a, 0.045, 0.05, 0.91, 0.095,
            '关键路径判据：仅优化最长路径上的非重叠暴露段才能降低端到端延迟',
            'green', size=11
        )
        out.save(f, 'ch01-xprof-profinfer-critical-path')

        # 24. ch01-five-workload-paradigms-arithmetic-intensity (Path E3 Fig 1-25)
        f, a = canvas(6.8)
        card(a, 0.02, 0.49, 0.96, 0.48, 'A  五大 AI 负载范式的算术强度与主瓶颈迁移', '', 'blue', title_size=11.5)
        paradigms_top = [
            (0.045, '1 单轮对话', '状态单次读取\n受限 HBM 带宽', 'blue'),
            (0.355, '2 自主智能体', '90% 前缀复用\n受限状态池容量', 'purple'),
            (0.665, '3 全双工语音', '12.5 Hz 双流\n80 ms 硬实时墙', 'orange'),
        ]
        for px, ptitle, pbody, pcol in paradigms_top:
            card(a, px, 0.68, 0.29, 0.20, ptitle, pbody, pcol, title_size=11, body_size=11)
        card(
            a, 0.045, 0.51, 0.43, 0.145,
            '4 合成数据工厂 (Ch23)',
            '无首字约束 · 追求吞吐/美元',
            'green', title_size=11, body_size=11
        )
        card(
            a, 0.525, 0.51, 0.43, 0.145,
            '5 科学智能 AI4S (Ch24)',
            '三角注意力与长序列通信',
            'teal', title_size=11, body_size=11
        )

        card(a, 0.02, 0.03, 0.96, 0.43, 'B  三维特征空间：算术强度 · 状态驻留 · 尾延迟', '', 'purple', title_size=11.5)
        dims_3d = [
            (0.045, '算术强度 I', '单步解码 ~1\n合成批 >200', 'blue'),
            (0.355, '状态驻留时长', '单轮毫秒即释\n智能体跨小时', 'purple'),
            (0.665, '尾延迟敏感度', '语音 <80 ms\n合成容忍分钟', 'orange'),
        ]
        for dx, dtitle, dbody, dcol in dims_3d:
            card(a, dx, 0.175, 0.29, 0.205, dtitle, dbody, dcol, title_size=11, body_size=11)
        box(
            a, 0.045, 0.05, 0.91, 0.095,
            '范式转移：主瓶颈从显存带宽转向状态池容量、微秒抖动与跨卡通信',
            'green', size=11
        )
        out.save(f, 'ch01-five-workload-paradigms-arithmetic-intensity')

        # 25. ch01-north-star-efficiency-waterfall-chain (Path E3 Fig 1-26)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(420 / 72, 4.1))
        f.subplots_adjust(left=0.22, right=0.96, bottom=0.20, top=0.85, wspace=0.64)
        for ax in (ax1, ax2):
            ax.spines[['top', 'right']].set_visible(False)

        wf_stages = ['铭牌峰值算力', '调度与容错后', '机队算子 MPG', '服务 SLO 达标', '验证任务净产率']
        wf_vals = [100.0, 66.7, 33.3, 26.0, 16.9]
        cols_wf = [COL['gray'], COL['blue'], COL['teal'], COL['orange'], COL['green']]
        edges_wf = [BORDER_COL['gray'], BORDER_COL['blue'], BORDER_COL['teal'], BORDER_COL['orange'], BORDER_COL['green']]
        y_wf = np.arange(len(wf_stages))
        ax1.barh(y_wf, wf_vals, height=0.55, color=cols_wf, edgecolor=edges_wf, linewidth=1.2)
        for i, v_w in enumerate(wf_vals):
            ax1.text(v_w + 3.0, i, f'{v_w:.1f}%', va='center', fontsize=11, weight='bold', color=COL['ink'])
        ax1.set_yticks(y_wf, wf_stages)
        ax1.invert_yaxis()
        ax1.set_xlim(0, 138.0)
        ax1.set_xlabel('剩余有效算力比例（%）', fontsize=11)
        ax1.set_title('A  北极星效率瀑布', loc='left', fontsize=12, weight='bold', pad=8)
        ax1.grid(axis='x', linestyle=':', alpha=0.5, color='#CBD5E1')

        opt_modes = ['当前基线', '单提算子 PG', '全栈五级协同']
        opt_vals = [16.9, 20.3, 33.3]
        cols_op = [COL['orange'], COL['blue'], COL['green']]
        edges_op = [BORDER_COL['orange'], BORDER_COL['blue'], BORDER_COL['green']]
        y_op = np.arange(len(opt_modes))
        ax2.barh(y_op, opt_vals, height=0.50, color=cols_op, edgecolor=edges_op, linewidth=1.2)
        for i, v_o in enumerate(opt_vals):
            ax2.text(v_o + 1.5, i, f'{v_o:.1f}%', va='center', fontsize=11, weight='bold', color=COL['ink'])
        ax2.set_yticks(y_op, opt_modes)
        ax2.invert_yaxis()
        ax2.set_xlim(0, 50.0)
        ax2.set_xlabel('端到端任务效率（%）', fontsize=11)
        ax2.set_title('B  全栈协同倍增效应', loc='left', fontsize=12, weight='bold', pad=8)
        ax2.grid(axis='x', linestyle=':', alpha=0.5, color='#CBD5E1')
        out.save(f, 'ch01-north-star-efficiency-waterfall-chain')

    from core_principles_figures import draw as draw_principles
    draw_principles(1, out)
    return out.finish()
