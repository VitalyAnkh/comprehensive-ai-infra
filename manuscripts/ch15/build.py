#!/usr/bin/env python3
"""Build Chapter 15 figures, locked calculations, and offline HTML preview."""
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figure_style import COL, BORDER_COL, STYLE, Exporter, arrow, box, canvas, card, text
from preview_output import preview_path

plt.rcParams.update(STYLE)


def sha256(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b'version https://git-lfs.github.com/spec/v1'):
        m = re.search(rb'oid sha256:([0-9a-f]{64})', raw)
        if m:
            return m.group(1).decode('ascii')
    return hashlib.sha256(raw).hexdigest()


def fig_15_1_mtbf_scaling(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '集群 MTBF_cluster = MTBF_node / N 的反比缩减规律（算例 15-1）', color='orange')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '1,024 卡集群\nMTBF ≈ 49.4 小时\n约两天 1 次中断', 'teal'),
        (0.355, 0.58, 0.29, 0.24, 'Meta 16,384 卡\n54 天 419 次中断\nMTBF = 3.09 小时', 'blue'),
        (0.68, 0.58, 0.29, 0.24, '100,000 卡集群\nMTBF 降至 30.4 分\n每半小时必发异常', 'orange'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, 'Meta Llama 3 405B（16K 卡 54 天）真实中断账本：硬件占 78%', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, 'GPU 与 HBM3 硬件故障（58.7%）\nHBM ECC 错、SRAM 奇偶错\nNVLink 与光模块闪断（19.3%）', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '软件与基础设施故障（22.0%）\n内核崩溃、存储超时与供电抖动\n促使容错从分钟重启走向秒级自愈', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '十万卡下故障不再是偶发异常，而是每 30 分钟必至的稳态过程', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch15-cluster-mtbf-scaling-and-llama3-failures')


def fig_15_2_hot_spare_pool(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '热备节点池（Hot-Spare Pool）泊松维修队列模型（算例 15-2）', color='teal')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '故障率 λ = 2.0 /h\n每小时坏 2 台节点\n立刻切入热备顶替', 'orange'),
        (0.355, 0.58, 0.29, 0.24, '维修 MTTR = 12 h\n稳态在修节点均值\nL = λ·MTTR = 24 台', 'blue'),
        (0.68, 0.58, 0.29, 0.24, '99.9% 不降级门限\nP(K ≤ S) ≥ 0.999\n最小热备 S = 40 台', 'teal'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '为什么只按均值配 24 台备机不可行？以及超节点局部性约束', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, '若仅配 S = L = 24 台热备节点\n泊松分布下 44.6% 时间备机耗尽\n备机枯竭将被迫降级或停机等待', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '仅多配 16 台（占集群 < 0.4%）\n即可换取 99.91% 零降级可用率\n空闲热备平时可运行可抢占评测', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '用 < 0.4% 热备容量吸收 12 小时维修延迟是维持完整拓扑的优选方案', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch15-hot-spare-pool-queuing-dimensioning')


def fig_15_3_sdc_contamination(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '静默数据损坏（SDC）逃逸 HBM ECC 并污染万卡权重的路径', color='orange')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '① HBM3 读取正常\n权重离开显存时\nECC 校验完全通过', 'teal'),
        (0.355, 0.58, 0.29, 0.24, '② Tensor 核内翻转\n电压跌落/时序违例\n无 ECC 累加器出错', 'orange'),
        (0.68, 0.58, 0.29, 0.24, '③ 写回并 AllReduce\n赋予错误值合法 ECC\n1 次归约传染万卡', 'orange'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '两类 SDC 破坏形态：显式指数发散 vs. 隐式尾数漂移', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, 'BF16 高位指数翻转（0 → 1）\n数值瞬间放大 2^64 倍\n数步后引发 Loss 尖峰与动量失效', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '尾数位或符号位静默偏移（+64）\n不触发 NaN，训练曲线表面平稳\n持续拉偏收敛且难以事后定位坏卡', 'purple', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, 'HBM ECC 只能保护存储与传输，无法保护无纠错编码的算术单元', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch15-sdc-bitflip-allreduce-contamination')


def fig_15_4_abft_checksum(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '算子级代数不变量校验（ABFT 行列校验和，算例 15-3）', color='teal')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '① 构造行列校验和\nc_A^T = 1^T A\nr_B = B 1', 'blue'),
        (0.355, 0.58, 0.29, 0.24, '② Tensor 核矩阵乘\n计算 C = A B\n附带算外围内积', 'teal'),
        (0.68, 0.58, 0.29, 0.24, '③ 残差十字定位\nΔr_i≠0 且 Δc_j≠0\n锁定坏元素 C[i,j]', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '为什么 ABFT 比双重执行冗余（DMR 100% 算力税）低 1,365 倍开销？', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, '对 M = N = K = 4,096 的 GEMM\n主矩阵乘需 137.44 GFLOPs\n外围校验仅需 100.66 MFLOPs', 'teal', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '附加开销仅 3/4096 = 0.0732%\n注入 (2,3) 处 +64.0 静默翻转\n凭残差立刻定位并原地减去 64', 'blue', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '利用线性结合律将 O(N^3) 校验降维成 O(N^2) 向量内积拦截 SDC', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch15-abft-checksum-matrix-multiplication')


def fig_15_5_safeguarding_opguard(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '3,500 万卡时在线 SDC 拦截与 OpGuard 位级定位（OSDI 2026）', color='blue')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '第一道：归约前拦截\n跨 DP 秩比对梯度范数\n防异常污染全局', 'teal'),
        (0.355, 0.58, 0.29, 0.24, '第二道：确定性对齐\n固定原子加与归约序\n健康卡位级一致', 'blue'),
        (0.68, 0.58, 0.29, 0.24, '第三道：算子级重放\n跨卡比对位级哈希\n毫秒级锁定故障卡', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '化解“浮点非确定性掩盖硬件真错误”的诊断难题', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '传统痛点：原子加与乱序归约\n使两张正常卡输出本就不同\n无法靠哈希区分良性差异与 SDC', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, 'OpGuard 强制确定性归约顺序\n使健康 GPU 输出 100% 位级相同\n任何 1 比特差异即确诊硬件 SDC', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '先在 AllReduce 前守住全局权重，再用位级确定性重放指认故障卡', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch15-safeguarding-sdc-and-opguard-bitwise-debug')


def fig_15_6_microbatch_rerun(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, '全局检查点回滚（数小时）', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, 'AllReduce 后才发现 Loss 尖峰\n万卡权重与动量已被污染', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '回滚至 2 小时前落盘检查点\n且不知哪张卡发生静默翻转', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '若永久故障卡仍留在集群内\n恢复后再次污染全部万卡', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '事后回滚既丢算力又难定位故障源', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, '微批次重跑与热摘除（50 ms）', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '在优化器 step() 前拦截异常\nHBM 主权重与动量完好无损', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '耗时 50 ms 原地重算当前微批\n恢复一致则判为宇宙射线软错误', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '若重算仍错则锁定特定故障卡\n热摘除该卡切入热备，零回滚', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '50 ms 微批重算替代 2 小时回滚', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch15-microbatch-rerun-vs-checkpoint-rollback')


def fig_15_7_trainmover(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, 'torchrun 全局销毁重启（300 s）', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '单卡掉线触发全网 SIGTERM\n九万九千张健康卡连带销毁', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '重新初始化 CUDA 与 NCCL 环\n万卡 TCP 握手耗 120–180 s', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '从远端存储重载数 TB 检查点\nETTR 降至 63.22%（算例 15-1）', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '一卡故障引发万卡连带重启', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'TrainMover 秒级热接管', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '热备常驻预初始化守护进程\n预建 CUDA 上下文与备用 QP', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '虚拟通信组局部替换故障 Rank\n仅重连邻居 QP，其余卡零感知', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '同 DP 邻居经 RDMA 推送状态\n5 s 内恢复，ETTR 升至 92.34%', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '保留健康进程上下文实现秒级接管', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch15-trainmover-subsecond-hot-migration')


def fig_15_8_straggler_blame(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'NCCL Flight Recorder 与到达偏斜矩阵（算例 15-4）', color='blue')
    stages = [
        (0.03, 0.58, 0.29, 0.24, 'Rank 0,1,2 准时到\nt = 0.0 ms 入集合\n空转等待 2.0 ms', 'teal'),
        (0.355, 0.58, 0.29, 0.24, 'Rank 3 慢节点根因\n降频致 t = 2.0 ms\n才启动集合算子', 'orange'),
        (0.68, 0.58, 0.29, 0.24, '网络交换仅 0.4 ms\nt = 2.4 ms 全体完\n火焰图误显 2.4 ms', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '为什么盲目升级网络带宽只省 0.2 ms，而定位慢节点能省 2.0 ms？', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '误把等待当传输（算例 15-4）\n带宽翻倍仅把 0.4 压至 0.2 ms\n总墙钟 2.4 → 2.2 ms（省 8.3%）', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '提取算子入队时间戳偏斜矩阵\n锁定最晚到达的根因 Rank 3\n剔除慢卡后降至 0.4 ms（省 83%）', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '同步通信耗时长往往是健康卡在等慢卡，而非网络拥塞', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch15-nccl-flight-recorder-straggler-blame')


def fig_15_9_optical_gray_failure(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '800G 光模块劣化引发 FEC 重传（灰故障）的演化特征', color='orange')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '① 光功率轻微衰减\nPre-FEC 升至 10^-5\nFEC 纠错掩盖异常', 'teal'),
        (0.355, 0.58, 0.29, 0.24, '② 纠错码字近饱和\n不可纠帧触发重传\nAllReduce 现毛刺', 'orange'),
        (0.68, 0.58, 0.29, 0.24, '③ 间歇性端口闪断\n未 Down 但重传频发\n拖慢轨道平面吞吐', 'orange'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '基于光模块遥测与 FEC 纠错桶直方图的预测性隔离', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, '传统监控只看 Link Up/Down\n对“在线但慢 20%”的灰故障失察\n致万卡长期处于亚健康降速态', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '采集 FEC 高阶纠错桶（Bin 7–15）\n在未丢包前提前标记劣化链路\n趁检查点间隙切换路由或备机', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '监控 Pre-FEC 误码斜率是在灰故障拖慢集群前将其拦截的关键', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch15-optical-ber-fec-gray-failure-detection')


def fig_15_10_benchmark_traps(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'AI 系统跑分五大失真陷阱与加速比悖论（算例 15-5）', color='orange')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '陷阱 ①–②：随机词元\n与无 SLO 极限吞吐\n掩盖投机与排队', 'orange'),
        (0.355, 0.58, 0.29, 0.24, '陷阱 ③：加速比悖论\nA 平均 5.25x 却耗时\n210 μs 慢于基线', 'orange'),
        (0.68, 0.58, 0.29, 0.24, '陷阱 ④–⑤：顺序时漂\n与静默精度退化\n误把节流当加速', 'orange'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '锁定反例复核：加速比翻转阈值与 ABBA 时漂抵消', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '加速比翻转点 p* = 15/19 ≈ 0.7895\n仅短形状 > 78.95% 时候选 A 才赢\n等频下候选 B（100 μs）快 2.1x', 'teal', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '消除 [0, 10, 20, 30] μs 热漂移\nAABB 顺序误判 A 快 10 μs\nABBA 镜像顺序还原真实差 +10 μs', 'blue', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '切勿对不同形状加速比取算术平均，必须汇报真实分布加权耗时', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch15-five-benchmark-gaming-traps')


def fig_15_11_four_pillar_eval(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '工业级 AI 基础设施四维评测与无偏性能剖析体系', color='blue')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '① 算子与通信基准\nFlashInfer + CCEval\n锁形状与影子流量', 'blue'),
        (0.355, 0.58, 0.29, 0.24, '② 漂移与 SLO 基准\nDriftBench 合成流\n测突发下 Goodput', 'teal'),
        (0.68, 0.58, 0.29, 0.24, '③ 可信剖析与门禁\nBlink 消采样失真\n+ KL 散度质量门禁', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '为什么传统 PMU 采样剖析在扁平运行时上失真？（Blink OSDI 2026）', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '短函数与关中断致采样失真\nPMU 采样点滑移与系统性漏采\n夸大长算子占比、漏报主机开销', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, 'Blink 校准剖析 + 两阶段金丝雀\n先在影子流量验 KL 散度与 P99\n再按 ABBA 交错放量验端到端收益', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '可信剖析工具（Blink）与质量门禁是防止优化流于局部指标的双重保障', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch15-flashinfer-bench-cceval-and-blink-profiling')


def fig_15_12_ettr_surface(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '十万卡有效训练时间比（ETTR）与 Young-Daly 最优周期（算例 15-1）', color='teal')
    stages = [
        (0.03, 0.58, 0.29, 0.24, 'Young-Daly 公式\nτ = sqrt(2·C·M)\n平衡写盘与回滚', 'blue'),
        (0.355, 0.58, 0.29, 0.24, '落盘 + 全量重启\nC=180s, T_rec=300s\nETTR = 63.22%', 'orange'),
        (0.68, 0.58, 0.29, 0.24, '内存快照 + 热接管\nC=5s, T_rec=5s\nETTR = 92.34%', 'teal'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '十万卡集群损失的 36.78% 算力去哪了？如何追回 29.12 个百分点？', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, 'M = 1,800 s（30 分钟一次故障）\n传统方案花 300 s 重启 + 402 s 重算\n近四成算力在空转与重算中损耗', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, 'Checkmate 内存快照（5 s）叠加\nTrainMover 无重启接管（5 s）\n挽回 +29,120 张等效 GPU 算力', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '将检查点与恢复压至秒级等价于在十万卡集群释放近三万张 GPU', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch15-effective-training-time-ratio-ettr-model')


def fig_15_13_burn_in_pipeline(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '新节点入网与维修归队后的四阶自动化烤机（Burn-in）准入流水线', color='blue')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '① 满载热压测\n持续 TDP 30 分钟\n筛查冷板与供电', 'blue'),
        (0.355, 0.58, 0.29, 0.24, '② 确定性 SDC 压测\n多精度 GEMM+ABFT\n拦截静默翻转卡', 'teal'),
        (0.68, 0.58, 0.29, 0.24, '③ 光链路裕量压测\n验 BER < 10^-8\n通过后入热备池', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '自动化机队健康自治状态机（Fleet Health State Machine）', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '禁止未烤机节点直入训练主池\n维修节点常存松动光模块或老化\n直入会导致修好一台却拖停全网', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '在役节点 → 热摘除 → 隔离诊断\n→ 维修 → 四阶烤机 → 回热备池\n全程零人工介入自动流转', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '把准入烤机前置在热备池门外是防带病节点二次污染集群的准则', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch15-automated-burn-in-fleet-health-pipeline')


def fig_15_14_dual_track_ledger(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '第 1–15 章集群底座双轨主线（Dense vs. MoE）总账本', color='blue')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '第 1–7 章：算力互联\nQwen3-8B 单卡端侧\nvs 671B 双平面网', 'blue'),
        (0.355, 0.58, 0.29, 0.24, '第 8–12 章：训推协同\nPD 分离、DeepEP\n弹性调度与端边云', 'teal'),
        (0.68, 0.58, 0.29, 0.24, '第 13–15 章：数据底座\nPB 存储、吉瓦液冷\n防 SDC 与基准评测', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '双轨模型在集群底座层的核心物理尺度与容错对照', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '轨道 A：Qwen3-8B（36 层 GQA）\n权重 16.4 GB / INT4 4.6 GB\nKV 144 KB/tok，无状态快速扩容', 'blue', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '轨道 B：DeepSeek 671B MoE\nMLA 压 KV 至 70.2 KB/tok\n万卡训练需秒级快照与热备顶替', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '一切集群工程抉择最终都还原为算力、显存、带宽、功耗与可靠性的账本', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch15-twenty-chapter-dual-track-master-ledger')


def fig_15_15_knowledge_graph(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '集群基础设施知识坐标系：从硬件网络到可靠性与后训练', color='teal')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '第 1–7 章：基础硬件\n模型负载与加速器\n超节点与数据中心网', 'blue'),
        (0.355, 0.58, 0.29, 0.24, '第 8–12 章：训推运行\n单机与分布式推理\n预训练、调度与端云', 'teal'),
        (0.68, 0.58, 0.29, 0.24, '第 13–17 章：集群底座\n数据存储、供电液冷\n容错可观测与后训练', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '跨层协同设计（Cross-Layer Co-Design）的双向耦合脉络', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, '自顶向下：算法演进重塑系统负载\nGQA/MLA 省显存 → MoE 增通信\n→ 后训练与智能体推高环境状态', 'blue', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '自底向上：物理定律划定算法边界\nSRAM/HBM 容量、超节点域宽\n与 SDC 失效率反过来约束模型', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '集群可靠性与无偏评测是承接底层硬件并支撑上层应用的枢纽层', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch15-twenty-chapter-architecture-knowledge-graph')


def fig_15_16_ten_principles(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '规模化生产集群的十条第一性原理工程准则（第 1–5 条）', color='blue')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '① 算术强度分界\n② 显存容量带宽双墙\n先算字节再谈算力', 'blue'),
        (0.355, 0.58, 0.29, 0.24, '③ 通信域物理分层\nTP 留柜内、EP 走轨\nDP/PP 跨层重叠', 'teal'),
        (0.68, 0.58, 0.29, 0.24, '④ 负载解耦与池化\nPD 分离、存储分层\n与异构沙箱分池', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '规模化生产集群的十条工程准则（第 6–10 条：物理、容错与评测）', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '⑤–⑦ 尾部气泡治理、全周期经济\n与机房供电/液冷/存储物理约束\n优化每焦耳与每美元有效产出', 'teal', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '⑧–⑩ 确定性边界、十万卡容错\n（防 SDC + 秒级接管）与评测科学\n以真实分布与质量门禁核验收益', 'blue', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '回归物理边界、坚持定量核算、以可观测性与混沌演练护航生产集群', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch15-ten-first-principles-of-ai-infra')


def fig_15_17_chaos_fault_injection(ex: Exporter) -> None:
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '四维主动混沌故障注入演练（200 次注入，总召回率 99.5%）', color='orange')
    faults = [
        (0.03, 0.58, 0.29, 0.24, '① 算子 SDC 注入\n翻转 +32..+64\nABFT 100% 捕获\nMTTD = 0.042 s', 'orange'),
        (0.355, 0.58, 0.29, 0.24, '② 慢卡与光模块\n+2.0 ms 偏斜/FEC\n偏斜阵 99% 定位\nMTTD = 3.2–6.8 s', 'purple'),
        (0.68, 0.58, 0.29, 0.24, '③ 掉卡与金丝雀\nTrainMover 0.68 s\n>3% P99 熔断回滚\nMTTR = 3.85 s', 'teal'),
    ]
    for x, y, w, h, label, role in faults:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '混沌演练 + 两阶段金丝雀门禁 vs. 传统被动排障收益对比', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '传统被动运维（无 GameDay）\n告警链路失效、慢卡长期潜伏\n人工重启耗时 1,500 s（25 min）', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '主动混沌检验 + 自治 SRE 协同\n199/200 故障秒级自动拦截隔离\n恢复提速 389.6×，ETTR 92.34%', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '未经主动故障注入检验的容错状态机，在十万卡故障并发时易引发级联退化', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch15-chaos-fault-injection-and-canary-rollout-gate')


def fig_15_18_frontier_bridge(ex: Exporter) -> None:
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '第 1–15 章可靠性底座 → 第 16–24 章安全与前沿系统', color='blue')
    box(ax, 0.04, 0.58, 0.44, 0.24, '第 16–17 章：机密与后训练\nGPU TEE 加密 + 广域网防投毒\nSFT 变长装箱 + 异步 RLVR', 'blue', size=11)
    box(ax, 0.52, 0.58, 0.44, 0.24, '第 18–24 章：前沿与全栈系统\n智能体、多模态语音与具身控制\n向量检索、科学计算与智算工厂', 'purple', size=11)
    arrow(ax, (0.48, 0.70), (0.52, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '从集群可靠性工程迈向全栈可观测性（15.6）与上层负载', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '本章 15.6 深化：零侵入可观测\neBPF 0.42% 剖析 + 等待链归因\n为自动容错状态机提供毫秒输入', 'blue', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '跨章延伸：定量工程准则复用\n局部性、异步解耦与尾延迟治理\n贯穿后续机密、RL 与全栈架构', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '字节守恒、拓扑同构、无偏评测与因果取证贯穿全书二十四章体系', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch15-core-cluster-to-four-new-frontiers-bridge')


def fig_15_19_oobleck_failover(ex: Exporter) -> None:
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'Oobleck 预实例化异构流水线模板无重启容错（1.45 s 恢复）', color='blue')
    steps = [
        (0.03, 0.58, 0.29, 0.24, '① 预实例化双模板\n标准 8 阶 T_8 与\n紧凑 7 阶 T_7 域', 'blue'),
        (0.355, 0.58, 0.29, 0.24, '② 局部层分片补齐\n检测 0.25+补齐 0.85s\nRDMA 拉取冗余分片', 'teal'),
        (0.68, 0.58, 0.29, 0.24, '③ 异构微批重配额\n同步 0.35s 零重启\n30×T_8 + 2×T_7', 'purple'),
    ]
    for x, y, w, h, label, role in steps:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '32 条 8 阶流水线（256 节点）掉线 2 台无备机时恢复对比', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '传统全局销毁重启（300.00 s）\n销毁 2,048 卡进程并重载磁盘\n无备机时全集群阻塞停机', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, 'Oobleck 模板切换（1.45 s）\n恢复提速 206.90×（零全局重启）\n保留 98.75% 稳态训练吞吐', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '编译期预建立多阶流水线通信域可在热备池耗尽时实现秒级降级续训', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch15-oobleck-pre-instantiated-pipeline-template-failover')


def fig_15_20_bursty_goodput(ex: Exporter) -> None:
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'Kingman 排队近似公式与推理压测流量变异系数（CV）失真', color='orange')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '失真均匀定速压测\nC_a = 0, C_s = 0\n排队 W_q 恒为 0 ms', 'orange'),
        (0.355, 0.58, 0.29, 0.24, '真实泊松长尾流量\nC_a = 1.0, C_s = 1.2\n变异因子达 1.22', 'purple'),
        (0.68, 0.58, 0.29, 0.24, '高负载排队非线性放大\nρ/(1-ρ) 放大 5.67×\nW_q 升至 829.60 ms', 'blue'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '利用率 ρ = 0.85 下 TTFT 与 SLO Goodput（≤500 ms）失真对比', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '均匀发包虚报（TTFT=80.0ms）\n宣称 100% 达标（7.08 req/s）\n掩盖线上突发排队尾延迟', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '泊松长尾实测（TTFT=909.6ms）\n膨胀 11.37×，仅 48.77% 达标\n真实 Goodput 仅 3.45 req/s', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '满足平均 TTFT ≤ 300 ms 要求系统利用率 ρ ≤ 60.04%（预留约 40% 裕量）', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch15-goodput-sla-under-bursty-poisson-traffic')


def fig_15_21_three_tier_observability(ex: Exporter) -> None:
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '三级分层遥测金字塔：化解万卡“观测者悖论”（算例 15-8）', color='blue')
    tiers = [
        (0.03, 0.58, 0.29, 0.24, 'L0 常驻零侵入\neBPF+PMU 采样\n损耗 0.35%', 'teal'),
        (0.355, 0.58, 0.29, 0.24, 'L1 异常触发层\nDynolog >3σ\nKineto 抓 5 步', 'blue'),
        (0.68, 0.58, 0.29, 0.24, 'L2 存证黑匣子\nFlight Record\n2,048 环槽', 'purple'),
    ]
    for x, y, w, h, label, role in tiers:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '16,384 卡常驻 CUPTI vs. 分层按需追踪账本对比', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '常驻 CUPTI（8.0% 损耗）\n占 1,310.72 张等效 GPU\n日写 1.42 PB 追踪数据', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, 'eBPF+Kineto（损 0.42%）\n仅占 68.81 张等效 GPU\n日写 1.80 TB（省 788.9×）', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '用内核采样守住常态低开销，仅在偏离时拉起微秒火焰图', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch15-ebpf-strobelight-dynolog-kineto-three-tier-observability')


def fig_15_22_wait_chain_dag(ex: Exporter) -> None:
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '8-Rank 等待链有向图归因：受害者 vs. 肇事者（算例 15-9）', color='orange')
    nodes = [
        (0.03, 0.58, 0.29, 0.24, 'Rank 5 根因慢卡\n读数抖 +140 ms\nt_in=200.0 ms', 'orange'),
        (0.355, 0.58, 0.29, 0.24, 'PP 下游 6,7\n点对点受阻\nt_in=201.0 ms', 'purple'),
        (0.68, 0.58, 0.29, 0.24, '健康 Rank 0–4\n58–59 ms 入队\n等 142–144 ms', 'teal'),
    ]
    for x, y, w, h, label, role in nodes:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, 'Mycroft / HTA 跨流事件依赖 DAG 逆向溯源机理', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, '朴素火焰图耗时排行误判\n7 个受害者耗 142–144 ms\n根因 Rank 5 仅显 4.0 ms', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '沿屏障与 P2P 边逆向溯源\n锁定最晚入队的 Rank 5\n剥离 7 个假阳性受害节点', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '同步屏障中通信核最短且入队最晚的节点才是慢节点源头', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch15-collective-wait-chain-victim-vs-root-cause-attribution-dag')


def fig_15_23_flight_recorder_forensics(ex: Exporter) -> None:
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'Flight Recorder 环形缓冲（K=2,048）死锁取证（练习 15-18）', color='purple')
    steps = [
        (0.03, 0.58, 0.29, 0.24, '① 环形槽常驻\nidx=seq%2048\n存 op/字节/QP', 'blue'),
        (0.355, 0.58, 0.29, 0.24, '② 序列 #18,451\n槽 idx=51 挂起\nRank 0–6 归约', 'orange'),
        (0.68, 0.58, 0.29, 0.24, '③ 分支错位证据\nRank 7 入广播\n确诊类型错配', 'purple'),
    ]
    for x, y, w, h, label, role in steps:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '区分“控制流不匹配死锁”与“RDMA 物理断链挂起”', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '控制流死锁（代码缺陷）\n同 seq 下算子或字节不同\n免换硬件直接修复分支', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '物理断链（网卡/光模块）\n同 seq 入队但某 QP 超时\n6.40 s 窗内锁定坏端口', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '看门狗超时前转储 2,048 槽黑匣子，免 SIGKILL 毁现场', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch15-nccl-flight-recorder-ring-buffer-deadlock-forensics')


def fig_15_24_monarch_fault_isolation(ex: Exporter) -> None:
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '传统 SPMD 命运绑定 vs. Monarch 单控制器网格（算例 15-10）', color='teal')
    card(ax, 0.02, 0.03, 0.46, 0.43, 'SPMD 全局重启（142.0 s）', color='orange', title_size=11)
    box(ax, 0.04, 0.14, 0.42, 0.21, '控制与数据紧耦于每张卡\n单卡异常致全员进程退出\nTCP 重建耗 142.0 s', 'orange', size=11)

    card(ax, 0.52, 0.03, 0.46, 0.43, 'Monarch 局部隔离（0.85 s）', color='teal', title_size=11)
    box(ax, 0.54, 0.14, 0.42, 0.21, '主控捕获子网格异常引用\n围栏+重绑+RDMA 仅 0.85s\n提速 167.1×，健康零中断', 'teal', size=11)

    stages = [
        (0.03, 0.58, 0.29, 0.24, '单控制器编排\n主控持全局图\n发多维网格令', 'blue'),
        (0.355, 0.58, 0.29, 0.24, '子网格隔离\n捕获局部异常\n仅摘受损切片', 'teal'),
        (0.68, 0.58, 0.29, 0.24, 'RDMA 重绑定\n热备原地接管\n0.85 s 续训', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))
    ex.save(fig, 'ch15-monarch-single-controller-actor-mesh-fault-isolation')


def build_chapter() -> None:
    ex = Exporter(HERE)
    fig_15_1_mtbf_scaling(ex)
    fig_15_2_hot_spare_pool(ex)
    fig_15_3_sdc_contamination(ex)
    fig_15_4_abft_checksum(ex)
    fig_15_5_safeguarding_opguard(ex)
    fig_15_6_microbatch_rerun(ex)
    fig_15_7_trainmover(ex)
    fig_15_8_straggler_blame(ex)
    fig_15_9_optical_gray_failure(ex)
    fig_15_10_benchmark_traps(ex)
    fig_15_11_four_pillar_eval(ex)
    fig_15_12_ettr_surface(ex)
    fig_15_13_burn_in_pipeline(ex)
    fig_15_14_dual_track_ledger(ex)
    fig_15_15_knowledge_graph(ex)
    fig_15_16_ten_principles(ex)
    fig_15_17_chaos_fault_injection(ex)
    fig_15_18_frontier_bridge(ex)
    fig_15_19_oobleck_failover(ex)
    fig_15_20_bursty_goodput(ex)
    fig_15_21_three_tier_observability(ex)
    fig_15_22_wait_chain_dag(ex)
    fig_15_23_flight_recorder_forensics(ex)
    fig_15_24_monarch_fault_isolation(ex)
    outputs = ex.finish()

    fig_index = [
        {
            "figure_id": "图 15-1",
            "number": 1,
            "file": "ch15-cluster-mtbf-scaling-and-llama3-failures.svg",
            "asset": "ch15/ch15-cluster-mtbf-scaling-and-llama3-failures.svg",
            "section": "15.1.1",
            "caption": "集群规模从 1K 扩至 100K 卡时 MTBF_cluster 降至 30.41 分钟的反比规律及 Meta 16,384 卡 419 次中断分类账本（算例 15-1）",
            "source_refs": ["experiments/ch15/15-01/run.py", "manuscripts/ch15/figure-data.json"],
            "TeachingRole": "展示十万卡规模下硬件故障从偶发异常演变为每半小时必至的稳态过程"
        },
        {
            "figure_id": "图 15-2",
            "number": 2,
            "file": "ch15-hot-spare-pool-queuing-dimensioning.svg",
            "asset": "ch15/ch15-hot-spare-pool-queuing-dimensioning.svg",
            "section": "15.1.2",
            "caption": "基于泊松修复队列模型（λ=2/h, MTTR=12h）的热备节点池（Hot-Spare Pool S_min=40）配置与可用性曲线（算例 15-2）",
            "source_refs": ["experiments/ch15/15-01/run.py", "research/2026-infra-survey/reading-mlsys-2026.md"],
            "TeachingRole": "推导以 99.9% 概率保证万卡训练不降级所需的最小热备节点数"
        },
        {
            "figure_id": "图 15-3",
            "number": 3,
            "file": "ch15-sdc-bitflip-allreduce-contamination.svg",
            "asset": "ch15/ch15-sdc-bitflip-allreduce-contamination.svg",
            "section": "15.2.1",
            "caption": "单颗 GPU Tensor Core 静默比特翻转（SDC）如何逃过 HBM ECC 并通过一次 AllReduce 瞬间污染万卡权重",
            "source_refs": ["research/2026-infra-survey/reading-osdi-2026.md", "manuscripts/ch15/figure-data.json"],
            "TeachingRole": "剖析无 ECC 保护的数据通路翻转如何被硬件赋予合法校验码并传染全集群"
        },
        {
            "figure_id": "图 15-4",
            "number": 4,
            "file": "ch15-abft-checksum-matrix-multiplication.svg",
            "asset": "ch15/ch15-abft-checksum-matrix-multiplication.svg",
            "section": "15.2.2",
            "caption": "基于代数不变量（ABFT 行列校验和）以 0.0732%（<0.5%）算力开销在线拦截并定位矩阵乘硬件错误（算例 15-3）",
            "source_refs": ["experiments/ch15/15-01/run.py", "manuscripts/ch15/figure-data.json"],
            "TeachingRole": "证明利用线性结合律可将 O(N^3) 矩阵乘校验降维至 O(N^2) 外围向量内积"
        },
        {
            "figure_id": "图 15-5",
            "number": 5,
            "file": "ch15-safeguarding-sdc-and-opguard-bitwise-debug.svg",
            "asset": "ch15/ch15-safeguarding-sdc-and-opguard-bitwise-debug.svg",
            "section": "15.2.2",
            "caption": "3,500 万 GPU 小时在线 SDC 感知隔离（Safeguarding LLM Training, OSDI 2026）与控制良性非确定性的位级对齐定位（OpGuard, OSDI 2026）",
            "source_refs": ["research/2026-infra-survey/reading-osdi-2026.md"],
            "TeachingRole": "结合 AllReduce 前统计拦截与确定性重放解决浮点非确定性掩盖 SDC 的难题"
        },
        {
            "figure_id": "图 15-6",
            "number": 6,
            "file": "ch15-microbatch-rerun-vs-checkpoint-rollback.svg",
            "asset": "ch15/ch15-microbatch-rerun-vs-checkpoint-rollback.svg",
            "section": "15.2.2",
            "caption": "检测到异常后原地重跑当前微批次（耗时 50 ms）区分瞬态软错误与永久硬件故障 vs. 盲目回滚数小时全局检查点",
            "source_refs": ["research/2026-infra-survey/reading-osdi-2026.md", "manuscripts/ch15/figure-data.json"],
            "TeachingRole": "展示在优化器更新前拦截异常并重算微批次如何避免回滚数小时进度"
        },
        {
            "figure_id": "图 15-7",
            "number": 7,
            "file": "ch15-trainmover-subsecond-hot-migration.svg",
            "asset": "ch15/ch15-trainmover-subsecond-hot-migration.svg",
            "section": "15.3.1",
            "caption": "传统全局销毁重启（torchrun 耗时 300 s）vs. TrainMover（OSDI 2026）利用预热备用进程与 RDMA 重映射实现秒级无重启接管",
            "source_refs": ["research/2026-infra-survey/reading-osdi-2026.md", "experiments/ch15/15-01/run.py"],
            "TeachingRole": "解析虚拟通信域局部重连如何消除九万九千张健康卡的连带重启开销"
        },
        {
            "figure_id": "图 15-8",
            "number": 8,
            "file": "ch15-nccl-flight-recorder-straggler-blame.svg",
            "asset": "ch15/ch15-nccl-flight-recorder-straggler-blame.svg",
            "section": "15.3.2",
            "caption": "利用 NCCL Flight Recorder 与集合通信到达时间偏斜矩阵（Arrival Skew，算例 15-4）从全网阻塞假象中定位唯一根因慢节点",
            "source_refs": ["case-studies/collective-paths-and-diagnosis.md", "experiments/ch15/15-01/run.py"],
            "TeachingRole": "证明同步集合通信中升级带宽仅省 0.2 ms、而消除 2.0 ms 入队偏斜才是关键"
        },
        {
            "figure_id": "图 15-9",
            "number": 9,
            "file": "ch15-optical-ber-fec-gray-failure-detection.svg",
            "asset": "ch15/ch15-optical-ber-fec-gray-failure-detection.svg",
            "section": "15.3.2",
            "caption": "光模块劣化引发 FEC 纠错重传（灰故障 Gray Failure）导致通信有效带宽周期性下降的特征识别",
            "source_refs": ["manuscripts/ch15/figure-data.json"],
            "TeachingRole": "通过监控 Pre-FEC 误码率与高阶纠错符号桶在链路彻底断连前主动隔离灰故障"
        },
        {
            "figure_id": "图 15-10",
            "number": 10,
            "file": "ch15-five-benchmark-gaming-traps.svg",
            "asset": "ch15/ch15-five-benchmark-gaming-traps.svg",
            "section": "15.4.1",
            "caption": "AI 系统跑分中的五大失真陷阱与算术平均加速比悖论（候选 A 5.25x 反慢于候选 B 2.0x，算例 15-5）",
            "source_refs": ["case-studies/optimization-evaluation-and-deployment.md", "experiments/ch15/15-01/run.py"],
            "TeachingRole": "复算加速比平均悖论（p*=15/19）与 AABB/ABBA 线性时漂抵消实验"
        },
        {
            "figure_id": "图 15-11",
            "number": 11,
            "file": "ch15-flashinfer-bench-cceval-and-blink-profiling.svg",
            "asset": "ch15/ch15-flashinfer-bench-cceval-and-blink-profiling.svg",
            "section": "15.4.2",
            "caption": "FlashInfer-Bench（算子基准）、CCEval（集合通信基准）、DriftBench（负载漂移）与 Blink（OSDI 2026 消除采样失真）的四维评测体系",
            "source_refs": ["research/2026-infra-survey/reading-osdi-2026.md", "research/2026-infra-survey/reading-mlsys-2026.md"],
            "TeachingRole": "建立从无偏硬件剖析（Blink）到影子流量质量门禁的工业级评测标准"
        },
        {
            "figure_id": "图 15-12",
            "number": 12,
            "file": "ch15-effective-training-time-ratio-ettr-model.svg",
            "asset": "ch15/ch15-effective-training-time-ratio-ettr-model.svg",
            "section": "15.1.1",
            "caption": "十万卡集群有效训练时间比（ETTR）对检查点周期 τ_ckpt、恢复时间 T_rec 与故障率 λ 的响应对比（63.22% vs. 92.34%）",
            "source_refs": ["experiments/ch15/15-01/run.py", "manuscripts/ch15/figure-data.json"],
            "TeachingRole": "量化展示内存检查点与秒级热恢复如何为十万卡集群挽回 29.12% 的有效算力"
        },
        {
            "figure_id": "图 15-13",
            "number": 13,
            "file": "ch15-automated-burn-in-fleet-health-pipeline.svg",
            "asset": "ch15/ch15-automated-burn-in-fleet-health-pipeline.svg",
            "section": "15.3.1",
            "caption": "新节点入网与故障修复后的自动化烤机（Burn-in：满载热压测 + 确定性 SDC 校验 + 光链路裕量扫描）准入流水线",
            "source_refs": ["manuscripts/ch15/figure-data.json"],
            "TeachingRole": "设计自动化机队健康状态机防止维修后带病节点重新污染在役训练集群"
        },
        {
            "figure_id": "图 15-14",
            "number": 14,
            "file": "ch15-twenty-chapter-dual-track-master-ledger.svg",
            "asset": "ch15/ch15-twenty-chapter-dual-track-master-ledger.svg",
            "section": "15.5.1",
            "caption": "第 1–15 章通用集群底座双轨贯穿案例（Qwen3-8B Dense 轨与 DeepSeek 671B MoE 轨）核心工程与可靠性参数对照总表",
            "source_refs": ["manuscripts/ch15/figure-data.json"],
            "TeachingRole": "横向串联前十五章集群底座双轨模型在算力、显存、通信、能耗与容错维度的定量参数"
        },
        {
            "figure_id": "图 15-15",
            "number": 15,
            "file": "ch15-twenty-chapter-architecture-knowledge-graph.svg",
            "asset": "ch15/ch15-twenty-chapter-architecture-knowledge-graph.svg",
            "section": "15.5.1",
            "caption": "集群基础设施知识坐标系：从硬件网络、训推调度到存储供电、可靠性工程与后训练衔接图",
            "source_refs": ["manuscripts/ch15/figure-data.json"],
            "TeachingRole": "建立自顶向下算法负载与自底向上物理定律双向耦合的集群知识坐标系"
        },
        {
            "figure_id": "图 15-16",
            "number": 16,
            "file": "ch15-ten-first-principles-of-ai-infra.svg",
            "asset": "ch15/ch15-ten-first-principles-of-ai-infra.svg",
            "section": "15.5.2",
            "caption": "规模化生产集群架构师的十条第一性原理工程准则总结图",
            "source_refs": ["manuscripts/ch15/figure-data.json"],
            "TeachingRole": "提炼贯穿通用集群底座的十条第一性原理与定量工程准则"
        },
        {
            "figure_id": "图 15-17",
            "number": 17,
            "file": "ch15-chaos-fault-injection-and-canary-rollout-gate.svg",
            "asset": "ch15/ch15-chaos-fault-injection-and-canary-rollout-gate.svg",
            "section": "15.5.3",
            "caption": "面向十万卡集群的混沌故障注入（SDC 比特翻转 / RoCE 灰丢包 / 慢节点）与金丝雀灰度上线门禁",
            "source_refs": ["experiments/ch15/15-02/run.py", "manuscripts/ch15/figure-data.json"],
            "TeachingRole": "通过 200 次主动混沌演练（99.5% 召回率，MTTR = 3.85 s）验证十万卡自愈流程"
        },
        {
            "figure_id": "图 15-18",
            "number": 18,
            "file": "ch15-core-cluster-to-four-new-frontiers-bridge.svg",
            "asset": "ch15/ch15-core-cluster-to-four-new-frontiers-bridge.svg",
            "section": "15.5.3",
            "caption": "从第 1–15 章通用集群可靠性底座迈向全栈可观测性（15.6）、机密后训练（第 16–17 章）与前沿系统（第 18–24 章）的架构衔接图",
            "source_refs": ["manuscripts/ch15/figure-data.json"],
            "TeachingRole": "衔接本章可观测性故障取证专节并引出后续机密安全、后训练与前沿应用负载"
        },
        {
            "figure_id": "图 15-19",
            "number": 19,
            "file": "ch15-oobleck-pre-instantiated-pipeline-template-failover.svg",
            "asset": "ch15/ch15-oobleck-pre-instantiated-pipeline-template-failover.svg",
            "section": "15.5.3",
            "caption": "Oobleck / Parachute 预实例化异构流水线模板无重启容错重配置（1.45 s 恢复 vs. 300 s 全局重启，算例 15-7）",
            "source_refs": ["experiments/ch15/15-03/run.py", "manuscripts/ch15/figure-data.json"],
            "TeachingRole": "展示编译期预建立多阶流水线模板如何在无热备顶替时以 1.45 s 切换实现 98.75% 算力保持"
        },
        {
            "figure_id": "图 15-20",
            "number": 20,
            "file": "ch15-goodput-sla-under-bursty-poisson-traffic.svg",
            "asset": "ch15/ch15-goodput-sla-under-bursty-poisson-traffic.svg",
            "section": "15.5.3",
            "caption": "推理系统在失真均匀到达（CV=0）与真实泊松突发长尾流量（CV≥1）下的排队放大与 SLO Goodput 退化曲线",
            "source_refs": ["experiments/ch15/15-03/run.py", "manuscripts/ch15/figure-data.json"],
            "TeachingRole": "用排队论 Kingman 公式揭示均匀定速压测掩盖 11.37× TTFT 膨胀与 51.27% Goodput 缩水的机理"
        },
        {
            "figure_id": "图 15-21",
            "number": 21,
            "file": "ch15-ebpf-strobelight-dynolog-kineto-three-tier-observability.svg",
            "asset": "ch15/ch15-ebpf-strobelight-dynolog-kineto-three-tier-observability.svg",
            "section": "15.6.1",
            "caption": "三级分层遥测金字塔（L0 eBPF Strobelight + L1 Dynolog/Kineto + L2 Flight Recorder）与 16,384 卡追踪开销对比（算例 15-8）",
            "source_refs": ["calculations/results/ch15-fleet-reliability-profiler.json", "experiments/ch15/15-04/run.py"],
            "TeachingRole": "证明分层按需剖析可将万卡追踪损耗从 8.0%（1.42 PB/天）压降至 0.42%（1.80 TB/天）"
        },
        {
            "figure_id": "图 15-22",
            "number": 22,
            "file": "ch15-collective-wait-chain-victim-vs-root-cause-attribution-dag.svg",
            "asset": "ch15/ch15-collective-wait-chain-victim-vs-root-cause-attribution-dag.svg",
            "section": "15.6.2",
            "caption": "8-Rank TP×PP×DP 集合通信等待链有向无环图（Mycroft / HTA）剥离 7 个受害者并锁定根因慢节点 Rank 5（算例 15-9）",
            "source_refs": ["calculations/results/ch15-fleet-reliability-profiler.json", "experiments/ch15/15-05/run.py"],
            "TeachingRole": "揭示真凶节点在集合通信核内耗时最短（4.0 ms）而健康受害节点空转 142–144 ms 的反转规律"
        },
        {
            "figure_id": "图 15-23",
            "number": 23,
            "file": "ch15-nccl-flight-recorder-ring-buffer-deadlock-forensics.svg",
            "asset": "ch15/ch15-nccl-flight-recorder-ring-buffer-deadlock-forensics.svg",
            "section": "15.6.2",
            "caption": "NCCL Flight Recorder 环形缓冲区（K=2,048）槽位映射与集合通信类型错位死锁取证流程（练习 15-18）",
            "source_refs": ["experiments/ch15/15-05/run.py", "manuscripts/ch15/figure-data.json"],
            "TeachingRole": "展示如何利用 2,048 槽内存黑匣子在进程被终止前区分代码分支错位死锁与物理 RDMA 断链"
        },
        {
            "figure_id": "图 15-24",
            "number": 24,
            "file": "ch15-monarch-single-controller-actor-mesh-fault-isolation.svg",
            "asset": "ch15/ch15-monarch-single-controller-actor-mesh-fault-isolation.svg",
            "section": "15.6.2",
            "caption": "传统 SPMD 全员销毁重建（142.0 s）vs. PyTorch Monarch 单控制器 Actor 网格局部隔离恢复（0.85 s，算例 15-10）",
            "source_refs": ["calculations/results/ch15-fleet-reliability-profiler.json", "experiments/ch15/15-05/run.py"],
            "TeachingRole": "解析控制面与数据面解耦如何将分布式子网格容错恢复提速 167.1 倍"
        },
    ]
    for row in fig_index:
        row['figure'] = f"15-{row['number']}"
        row['name'] = Path(row['file']).stem
    (HERE / 'figure-index.json').write_text(json.dumps(fig_index, ensure_ascii=False, indent=2) + '\n')

    md = HERE.parent / '15-集群可靠性与基准评测.md'
    if md.exists():
        text_md = md.read_text()
        display = re.findall(r'\$\$(.+?)\$\$', text_md, re.S)
        inline = re.findall(r'(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)', text_md)
        formulas = [f.strip() for f in display + inline]
        katex_js = ROOT / 'manuscripts/ch06/vendor/katex/katex.js'
        js = (
            'const katex=require(process.argv[1]);const fs=require("fs");'
            'const a=JSON.parse(fs.readFileSync(0,"utf8"));const e=[];'
            'a.forEach((s,i)=>{try{katex.renderToString(s,{throwOnError:true,displayMode:i<'
            + str(len(display))
            + '})}catch(err){e.push({i,s,err:err.message})}});'
            'console.log(JSON.stringify({total:a.length,display:'
            + str(len(display))
            + ',inline:'
            + str(len(inline))
            + ',errors:e}));'
        )
        res = subprocess.run(
            ['node', '-e', js, str(katex_js)],
            input=json.dumps(formulas),
            text=True,
            capture_output=True,
            check=True,
        )
        math_val = json.loads(res.stdout)
        assert not math_val['errors'], math_val['errors']
        (HERE / 'math-validation.json').write_text(json.dumps(math_val, ensure_ascii=False, indent=2) + '\n')

        import markdown

        def repl_img(m):
            alt, rel = m.group(1), m.group(2)
            p = (md.parent / rel).with_suffix('.png')
            b64 = base64.b64encode(p.read_bytes()).decode()
            return f'<figure><img alt="{alt}" src="data:image/png;base64,{b64}"><figcaption>{alt}</figcaption></figure>'

        work = re.sub(r'!\[([^\]]*)\]\((ch15/[^)]+)\)', repl_img, text_md)
        body = markdown.markdown(work, extensions=['tables', 'fenced_code', 'footnotes', 'toc'])
        html = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>第 15 章：集群可靠性与基准评测</title></head><body>{body}</body></html>\n'
        out_html = preview_path(md)
        out_html.write_text(html)

    sources = [
        HERE / 'build.py',
        HERE / 'figure-data.json',
        HERE / 'figure-index.json',
        ROOT / 'calculations/results/ch15-fleet-reliability-profiler.json',
        ROOT / 'case-studies/collective-paths-and-diagnosis.md',
        ROOT / 'case-studies/optimization-evaluation-and-deployment.md',
        ROOT / 'experiments/ch15/15-01/run.py',
        ROOT / 'experiments/ch15/15-02/run.py',
        ROOT / 'experiments/ch15/15-03/run.py',
        ROOT / 'experiments/ch15/15-04/run.py',
        ROOT / 'experiments/ch15/15-05/run.py',
    ]
    if md.exists():
        sources.append(md)
    (HERE / 'sources.json').write_text(
        json.dumps(
            {"sources": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in sources]},
            ensure_ascii=False,
            indent=2,
        )
        + '\n'
    )
    all_outputs = sorted(outputs) + [
        HERE / 'figure-data.json',
        HERE / 'figure-index.json',
        HERE / 'teaching-layout-validation.json',
    ]
    if (HERE / 'math-validation.json').exists():
        all_outputs.append(HERE / 'math-validation.json')
    (HERE / 'manifest.json').write_text(
        json.dumps(
            {"outputs": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in all_outputs]},
            ensure_ascii=False,
            indent=2,
        )
        + '\n'
    )
    print(f'Built {len(outputs)} figure files for Chapter 15; 0 layout warnings.')


if __name__ == '__main__':
    build_chapter()
