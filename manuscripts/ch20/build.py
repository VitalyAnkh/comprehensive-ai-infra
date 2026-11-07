#!/usr/bin/env python3
"""Build Chapter 20 figures, locked calculations, and offline HTML preview."""
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


def fig_20_1_mtbf_scaling(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '集群 MTBF_cluster = MTBF_node / N 的反比坍缩（算例 20-1）', color='orange')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '1,024 卡集群\nMTBF ≈ 49.4 小时\n约两天 1 次中断', 'teal'),
        (0.355, 0.58, 0.29, 0.24, 'Meta 16,384 卡\n54 天 419 次中断\nMTBF = 3.09 小时', 'blue'),
        (0.68, 0.58, 0.29, 0.24, '100,000 卡集群\nMTBF 跌至 30.4 分\n每半小时必发异常', 'orange'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, 'Meta Llama 3 405B（16K 卡 54 天）真实中断账本：硬件占 78%', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, 'GPU 与 HBM3 硬件故障（58.7%）\nHBM ECC 错、SRAM 奇偶错\nNVLink 与光模块闪断（19.3%）', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '软件与基础设施故障（22.0%）\n内核崩溃、存储超时与供电抖动\n倒逼容错从分钟重启走向秒级自愈', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '十万卡下故障不再是偶发异常，而是每 30 分钟必至的稳态过程', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch20-cluster-mtbf-scaling-and-llama3-failures')


def fig_20_2_hot_spare_pool(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '热备节点池（Hot-Spare Pool）泊松维修队列模型（算例 20-2）', color='teal')
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
    box(ax, 0.52, 0.13, 0.44, 0.20, '仅多配 16 台（占集群 < 0.4%）\n即可换取 99.91% 零降级可用率\n空闲热备平时可跑可抢占评测', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '用 < 0.4% 热备容量吸收 12 小时维修延迟是维持满血拓扑的最优解', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch20-hot-spare-pool-queuing-dimensioning')


def fig_20_3_sdc_contamination(ex: Exporter):
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

    card(ax, 0.02, 0.03, 0.96, 0.42, '两类 SDC 破坏形态：显式指数爆炸 vs. 隐式尾数漂移', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, 'BF16 高位指数翻转（0 → 1）\n数值瞬间放大 2^64 倍\n数步后引发 Loss 尖峰与动量报废', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '尾数位或符号位静默偏移（+64）\n不触发 NaN，训练曲线表面平稳\n持续拉偏收敛且无法事后抓坏卡', 'purple', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, 'HBM ECC 只能保护存储与传输，无法保护无纠错编码的算术单元', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch20-sdc-bitflip-allreduce-contamination')


def fig_20_4_abft_checksum(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '算子级代数不变量校验（ABFT 行列校验和，算例 20-3）', color='teal')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '① 构造行列校验和\nc_A^T = 1^T A\nr_B = B 1', 'blue'),
        (0.355, 0.58, 0.29, 0.24, '② Tensor 核矩阵乘\n计算 C = A B\n附带算外围内积', 'teal'),
        (0.68, 0.58, 0.29, 0.24, '③ 残差十字定位\nΔr_i≠0 且 Δc_j≠0\n锁定坏元素 C[i,j]', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '为什么 ABFT 比双重执行冗余（DMR 100% 算力税）便宜 1,365 倍？', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, '对 M = N = K = 4,096 的 GEMM\n主矩阵乘需 137.44 GFLOPs\n外围校验仅需 100.66 MFLOPs', 'teal', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '附加开销仅 3/4096 = 0.0732%\n注入 (2,3) 处 +64.0 静默翻转\n凭残差立刻定位并原地减去 64', 'blue', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '利用线性结合律将 O(N^3) 校验降维成 O(N^2) 向量内积拦截 SDC', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch20-abft-checksum-matrix-multiplication')


def fig_20_5_safeguarding_opguard(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '3,500 万卡时在线 SDC 拦截与 OpGuard 位级定位（OSDI 26）', color='blue')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '第一道：归约前拦截\n跨 DP 秩比对梯度范数\n防异常污染全局', 'teal'),
        (0.355, 0.58, 0.29, 0.24, '第二道：确定性对齐\n固定原子加与归约序\n健康卡位级一致', 'blue'),
        (0.68, 0.58, 0.29, 0.24, '第三道：算子级重放\n跨卡比对位级哈希\n毫秒级揪出坏卡', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '破解“浮点非确定性掩盖硬件真错误”的诊断死结', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '传统痛点：原子加与乱序归约\n使两张正常卡输出本就不同\n无法靠哈希区分良性差异与 SDC', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, 'OpGuard 强制确定性归约顺序\n使健康 GPU 输出 100% 位级相同\n任何 1 比特差异即确诊硬件 SDC', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '先在 AllReduce 前守住全局权重，再用位级确定性重放指认坏卡', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch20-safeguarding-sdc-and-opguard-bitwise-debug')


def fig_20_6_microbatch_rerun(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, '全局检查点回滚（数小时）', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, 'AllReduce 后才发现 Loss 突刺\n万卡权重与动量已被污染', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '回滚至 2 小时前落盘检查点\n且不知哪张卡发生静默翻转', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '若永久坏卡仍留在集群内\n恢复后再次污染全部万卡', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '事后回滚既丢算力又抓不出真凶', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, '微批次重跑与热踢出（50 ms）', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '在优化器 step() 前拦截异常\nHBM 主权重与动量完好无损', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '耗时 50 ms 原地重算当前微批\n恢复一致则判为宇宙射线软错误', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '若重算仍错则锁定特定坏卡\n热踢出该卡切入热备，零回滚', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '50 ms 微批重算替代 2 小时回滚', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch20-microbatch-rerun-vs-checkpoint-rollback')


def fig_20_7_trainmover(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, 'torchrun 全局销毁重启（300 s）', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '单卡掉线触发全网 SIGTERM\n九万九千张健康卡陪葬销毁', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '重新初始化 CUDA 与 NCCL 环\n万卡 TCP 握手耗 120–180 s', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '从远端存储重载数 TB 检查点\nETTR 跌至 63.22%（算例 20-1）', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '一卡故障、万卡陪葬重启', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'TrainMover 秒级热接管', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '热备常驻预初始化守护进程\n预建 CUDA 上下文与备用 QP', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '虚拟通信组局部替换故障 Rank\n仅重连邻居 QP，其余卡零感知', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '同 DP 邻居经 RDMA 推状态\n5 s 内恢复，ETTR 升至 92.34%', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '保留健康进程上下文实现秒级接管', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch20-trainmover-subsecond-hot-migration')


def fig_20_8_straggler_blame(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'NCCL Flight Recorder 与到达偏斜矩阵（算例 20-4）', color='blue')
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
    box(ax, 0.04, 0.13, 0.44, 0.20, '误把等待当传输（算例 20-4）\n带宽翻倍仅把 0.4 压至 0.2 ms\n总墙钟 2.4 → 2.2 ms（省 8.3%）', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '提取算子入队时间戳偏斜矩阵\n锁定最晚到达的根因 Rank 3\n剔除慢卡后降至 0.4 ms（省 83%）', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '同步通信耗时长往往是健康卡在等慢卡，而非网络拥塞', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch20-nccl-flight-recorder-straggler-blame')


def fig_20_9_optical_gray_failure(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '800G 光模块劣化引发 FEC 重传（灰故障）的演化特征', color='orange')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '① 光功率轻微衰减\nPre-FEC 升至 10^-5\nFEC 纠错掩盖异常', 'teal'),
        (0.355, 0.58, 0.29, 0.24, '② 纠错码字近饱和\n不可纠帧触发重传\nAllReduce 现毛刺', 'orange'),
        (0.68, 0.58, 0.29, 0.24, '③ 间歇性端口闪断\n未 Down 但重传风暴\n拖垮轨道平面吞吐', 'orange'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '基于光模块遥测与 FEC 纠错桶直方图的预测性隔离', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, '传统监控只看 Link Up/Down\n对“活着但慢 20%”的灰故障失明\n致万卡长期处于亚健康降速态', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '采集 FEC 高阶纠错桶（Bin 7–15）\n在未丢包前提前标记劣化链路\n趁检查点间隙切换路由或备机', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '监控 Pre-FEC 误码斜率是在灰故障拖垮集群前将其拦截的关键', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch20-optical-ber-fec-gray-failure-detection')


def fig_20_10_benchmark_traps(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'AI 系统跑分五大失真陷阱与加速比悖论（算例 20-5）', color='orange')
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
    text(ax, 0.50, 0.06, '严禁对不同形状加速比取算术平均，必须汇报真实分布加权耗时', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch20-five-benchmark-gaming-traps')


def fig_20_11_four_pillar_eval(ex: Exporter):
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

    card(ax, 0.02, 0.03, 0.96, 0.42, '为什么传统 PMU 采样剖析在扁平运行时上失真？（Blink OSDI 26）', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '短函数与关中断致采样失真\nPMU 采样点滑移与系统性漏采\n夸大长算子占比、漏报主机开销', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, 'Blink 校准剖析 + 两阶段金丝雀\n先在影子流量验 KL 散度与 P99\n再按 ABBA 交错放量验端到端收益', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '可信剖析工具（Blink）与质量门禁是防止优化沦为数字游戏的双保险', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch20-flashinfer-bench-cceval-and-blink-profiling')


def fig_20_12_ettr_surface(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '十万卡有效训练时间比（ETTR）与 Young-Daly 最优周期（算例 20-1）', color='teal')
    stages = [
        (0.03, 0.58, 0.29, 0.24, 'Young-Daly 公式\nτ = sqrt(2·C·M)\n平衡写盘与回滚', 'blue'),
        (0.355, 0.58, 0.29, 0.24, '落盘 + 全量重启\nC=180s, T_rec=300s\nETTR = 63.22%', 'orange'),
        (0.68, 0.58, 0.29, 0.24, '内存快照 + 热接管\nC=5s, T_rec=5s\nETTR = 92.34%', 'teal'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '十万卡集群损失的 36.78% 算力去哪了？如何夺回 29.12 个百分点？', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, 'M = 1,800 s（30 分钟一次故障）\n传统方案花 300 s 重启 + 402 s 重算\n近四成算力在空转与重算中蒸发', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, 'Checkmate 内存快照（5 s）叠加\nTrainMover 无重启接管（5 s）\n挽回 +29,120 张等效 GPU 算力', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '将检查点与恢复压至秒级等价于在十万卡集群白捡近三万张 GPU', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch20-effective-training-time-ratio-ettr-model')


def fig_20_13_burn_in_pipeline(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '新节点入网与维修归队后的四阶自动化烤机（Burn-in）准入流水线', color='blue')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '① 满载热压测\n跑满 TDP 30 分钟\n筛查冷板与供电', 'blue'),
        (0.355, 0.58, 0.29, 0.24, '② 确定性 SDC 压测\n多精度 GEMM+ABFT\n拦截静默翻转卡', 'teal'),
        (0.68, 0.58, 0.29, 0.24, '③ 光链路裕量压测\n验 BER < 10^-8\n通过后入热备池', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '闭环机队健康自治状态机（Fleet Health State Machine）', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '严禁未烤机节点直入训练主池\n维修节点常存松动光模块或老化\n直入会导致修好一台却炸停全网', 'orange', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '在役节点 → 热踢出 → 隔离诊断\n→ 维修 → 四阶烤机 → 回热备池\n全程零人工介入闭环运转', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '把准入烤机挡在热备池门外是防带病节点二次感染集群的铁律', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch20-automated-burn-in-fleet-health-pipeline')


def fig_20_14_dual_track_ledger(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '全书二十章双轨贯穿主线（Dense 轨 vs. MoE 轨）收官总账本', color='blue')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '第 1–7 章：算力互联\nQwen3-8B 单卡端侧\nvs 671B 双平面网', 'blue'),
        (0.355, 0.58, 0.29, 0.24, '第 8–14 章：训推存储\nPD 分离、DeepEP\n过度训练百倍降本', 'teal'),
        (0.68, 0.58, 0.29, 0.24, '第 15–20 章：六大前沿\n吉瓦供电、机密沙箱\nDiT/RLVR/VLA/容错', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '双轨模型贯穿二十章的核心物理尺度对比', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '轨道 A：Qwen3-8B（36 层 GQA）\n权重 16.4 GB / INT4 4.6 GB\nKV 144 KB/tok，代表单卡与端侧', 'blue', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '轨道 B：DeepSeek 671B MoE\nMLA 压 KV 至 70.2 KB/tok\n代表超节点、EP 与 RLVR 旗舰', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '一切系统架构抉择最终都还原为算力、显存、带宽、功耗与可靠性的账本', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch20-twenty-chapter-dual-track-master-ledger')


def fig_20_15_knowledge_graph(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '全书二十章知识体系全景图：三大部分与六大前沿底座', color='teal')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '第一部分（1–3 章）\n模型与负载硬件视角\nMoE/MLA 与 SLO', 'blue'),
        (0.355, 0.58, 0.29, 0.24, '第二部分（4–7 章）\n芯片算子与集群网络\n超节点与无损网', 'teal'),
        (0.68, 0.58, 0.29, 0.24, '第三部分（8–20 章）\n训推、云边、供电安全\nDiT/RLVR/VLA/容错', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '跨层协同设计（Cross-Layer Co-Design）的闭环脉络', color='blue')
    box(ax, 0.04, 0.13, 0.44, 0.20, '自顶向下：算法演进重塑系统负载\nGQA/MLA 救显存 → MoE 推高通信\n→ Long-CoT 与 3D DiT 改训推比', 'blue', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '自底向上：物理定律划定算法边界\nSRAM/HBM 容量、超节点域宽\n与 SDC 失效率反过来决定模型', 'teal', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '优秀架构师既懂顶层模型数学，又能一路向下推演到硅片与变压器', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch20-twenty-chapter-architecture-knowledge-graph')


def fig_20_16_ten_principles(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, 'AI 基础设施架构师的十条第一性原理铁律（第 1–5 条）', color='blue')
    stages = [
        (0.03, 0.58, 0.29, 0.24, '① 算术强度分界\n② 显存容量带宽双墙\n先算字节再谈算力', 'blue'),
        (0.355, 0.58, 0.29, 0.24, '③ 通信域物理分层\nTP 留柜内、EP 走轨\nDP/PP 跨层重叠', 'teal'),
        (0.68, 0.58, 0.29, 0.24, '④ 负载解耦与池化\nPD 分离、EPD 分离\n与 RLVR 异步分池', 'purple'),
    ]
    for x, y, w, h, label, role in stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.355, 0.70))
    arrow(ax, (0.645, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '十条第一性原理铁律（第 6–10 条：尾部、物理、容错与评测）', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '⑤–⑦ 尾部气泡治理、全周期经济\n与机房供电/液冷/安全物理约束\n最大化每焦耳与每美元有效产出', 'teal', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '⑧–⑩ 硬实时 WCET、十万卡容错\n（防 SDC + 秒级接管）与评测科学\n只认真实分布与质量门禁下吞吐', 'blue', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '回归物理常识、坚持定量核算、敬畏生产现实——全书二十章收官', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch20-ten-first-principles-of-ai-infra')


def build_chapter() -> None:
    ex = Exporter(HERE)
    fig_20_1_mtbf_scaling(ex)
    fig_20_2_hot_spare_pool(ex)
    fig_20_3_sdc_contamination(ex)
    fig_20_4_abft_checksum(ex)
    fig_20_5_safeguarding_opguard(ex)
    fig_20_6_microbatch_rerun(ex)
    fig_20_7_trainmover(ex)
    fig_20_8_straggler_blame(ex)
    fig_20_9_optical_gray_failure(ex)
    fig_20_10_benchmark_traps(ex)
    fig_20_11_four_pillar_eval(ex)
    fig_20_12_ettr_surface(ex)
    fig_20_13_burn_in_pipeline(ex)
    fig_20_14_dual_track_ledger(ex)
    fig_20_15_knowledge_graph(ex)
    fig_20_16_ten_principles(ex)
    outputs = ex.finish()

    fig_index = [
        {
            "figure_id": "图 20-1",
            "number": 1,
            "file": "ch20-cluster-mtbf-scaling-and-llama3-failures.svg",
            "asset": "ch20/ch20-cluster-mtbf-scaling-and-llama3-failures.svg",
            "section": "20.1.1",
            "caption": "集群规模从 1K 扩至 100K 卡时 MTBF_cluster 跌至 30.41 分钟的反比规律及 Meta 16,384 卡 419 次中断分类账本（算例 20-1）",
            "source_refs": ["experiments/ch20/20-01/run.py", "manuscripts/ch20/figure-data.json"],
            "TeachingRole": "展示十万卡规模下硬件故障从偶发异常演变为每半小时必至的稳态过程"
        },
        {
            "figure_id": "图 20-2",
            "number": 2,
            "file": "ch20-hot-spare-pool-queuing-dimensioning.svg",
            "asset": "ch20/ch20-hot-spare-pool-queuing-dimensioning.svg",
            "section": "20.1.2",
            "caption": "基于泊松修复队列模型（λ=2/h, MTTR=12h）的热备节点池（Hot-Spare Pool S_min=40）配置与可用性曲线（算例 20-2）",
            "source_refs": ["experiments/ch20/20-01/run.py", "research/2026-infra-survey/reading-mlsys-2026.md"],
            "TeachingRole": "推导以 99.9% 概率保证万卡训练不降级所需的最小热备节点数"
        },
        {
            "figure_id": "图 20-3",
            "number": 3,
            "file": "ch20-sdc-bitflip-allreduce-contamination.svg",
            "asset": "ch20/ch20-sdc-bitflip-allreduce-contamination.svg",
            "section": "20.2.1",
            "caption": "单颗 GPU Tensor Core 静默比特翻转（SDC）如何逃过 HBM ECC 并通过一次 AllReduce 瞬间污染万卡权重",
            "source_refs": ["research/2026-infra-survey/reading-osdi-2026.md", "manuscripts/ch20/figure-data.json"],
            "TeachingRole": "剖析无 ECC 保护的数据通路翻转如何被硬件赋予合法校验码并传染全集群"
        },
        {
            "figure_id": "图 20-4",
            "number": 4,
            "file": "ch20-abft-checksum-matrix-multiplication.svg",
            "asset": "ch20/ch20-abft-checksum-matrix-multiplication.svg",
            "section": "20.2.2",
            "caption": "基于代数不变量（ABFT 行列校验和）以 0.0732%（<0.5%）算力开销在线拦截并定位矩阵乘硬件错误（算例 20-3）",
            "source_refs": ["experiments/ch20/20-01/run.py", "manuscripts/ch20/figure-data.json"],
            "TeachingRole": "证明利用线性结合律可将 O(N^3) 矩阵乘校验降维至 O(N^2) 外围向量内积"
        },
        {
            "figure_id": "图 20-5",
            "number": 5,
            "file": "ch20-safeguarding-sdc-and-opguard-bitwise-debug.svg",
            "asset": "ch20/ch20-safeguarding-sdc-and-opguard-bitwise-debug.svg",
            "section": "20.2.2",
            "caption": "3,500 万 GPU 小时在线 SDC 感知隔离（Safeguarding LLM Training OSDI '26）与控制良性非确定性的位级对齐定位（OpGuard OSDI '26）",
            "source_refs": ["research/2026-infra-survey/reading-osdi-2026.md"],
            "TeachingRole": "结合 AllReduce 前统计拦截与确定性重放解决浮点非确定性掩盖 SDC 的难题"
        },
        {
            "figure_id": "图 20-6",
            "number": 6,
            "file": "ch20-microbatch-rerun-vs-checkpoint-rollback.svg",
            "asset": "ch20/ch20-microbatch-rerun-vs-checkpoint-rollback.svg",
            "section": "20.2.2",
            "caption": "检测到异常后原地重跑当前微批次（耗时 50 ms）区分瞬态软错误与永久硬件故障 vs. 盲目回滚数小时全局检查点",
            "source_refs": ["research/2026-infra-survey/reading-osdi-2026.md", "manuscripts/ch20/figure-data.json"],
            "TeachingRole": "展示在优化器更新前拦截异常并重算微批次如何避免回滚数小时进度"
        },
        {
            "figure_id": "图 20-7",
            "number": 7,
            "file": "ch20-trainmover-subsecond-hot-migration.svg",
            "asset": "ch20/ch20-trainmover-subsecond-hot-migration.svg",
            "section": "20.3.1",
            "caption": "传统全局销毁重启（torchrun 耗时 300 s）vs. TrainMover（OSDI '26）利用预热备用进程与 RDMA 重映射实现秒级无重启接管",
            "source_refs": ["research/2026-infra-survey/reading-osdi-2026.md", "experiments/ch20/20-01/run.py"],
            "TeachingRole": "解析虚拟通信域局部重连如何消除九万九千张健康卡的陪葬重启开销"
        },
        {
            "figure_id": "图 20-8",
            "number": 8,
            "file": "ch20-nccl-flight-recorder-straggler-blame.svg",
            "asset": "ch20/ch20-nccl-flight-recorder-straggler-blame.svg",
            "section": "20.3.2",
            "caption": "利用 NCCL Flight Recorder 与集合通信到达时间偏斜矩阵（Arrival Skew，算例 20-4）从全网阻塞假象中揪出唯一根因慢节点",
            "source_refs": ["case-studies/collective-paths-and-diagnosis.md", "experiments/ch20/20-01/run.py"],
            "TeachingRole": "证明同步集合通信中升级带宽仅省 0.2 ms、而消除 2.0 ms 入队偏斜才是正解"
        },
        {
            "figure_id": "图 20-9",
            "number": 9,
            "file": "ch20-optical-ber-fec-gray-failure-detection.svg",
            "asset": "ch20/ch20-optical-ber-fec-gray-failure-detection.svg",
            "section": "20.3.2",
            "caption": "光模块劣化引发 FEC 纠错重传（灰故障 Gray Failure）导致通信有效带宽周期性塌陷的特征识别",
            "source_refs": ["manuscripts/ch20/figure-data.json"],
            "TeachingRole": "通过监控 Pre-FEC 误码率与高阶纠错符号桶在链路彻底断连前主动隔离灰故障"
        },
        {
            "figure_id": "图 20-10",
            "number": 10,
            "file": "ch20-five-benchmark-gaming-traps.svg",
            "asset": "ch20/ch20-five-benchmark-gaming-traps.svg",
            "section": "20.4.1",
            "caption": "AI 系统跑分中的五大失真陷阱与算术平均加速比悖论（候选 A 5.25x 反慢于候选 B 2.0x，算例 20-5）",
            "source_refs": ["case-studies/optimization-evaluation-and-deployment.md", "experiments/ch20/20-01/run.py"],
            "TeachingRole": "复算加速比平均悖论（p*=15/19）与 AABB/ABBA 线性时漂抵消实验"
        },
        {
            "figure_id": "图 20-11",
            "number": 11,
            "file": "ch20-flashinfer-bench-cceval-and-blink-profiling.svg",
            "asset": "ch20/ch20-flashinfer-bench-cceval-and-blink-profiling.svg",
            "section": "20.4.2",
            "caption": "FlashInfer-Bench（算子基准）、CCEval（集合通信基准）、DriftBench（负载漂移）与 Blink（OSDI '26 消除采样失真）的四维评测体系",
            "source_refs": ["research/2026-infra-survey/reading-osdi-2026.md", "research/2026-infra-survey/reading-mlsys-2026.md"],
            "TeachingRole": "建立从无偏硬件剖析（Blink）到影子流量质量门禁的工业级评测标准"
        },
        {
            "figure_id": "图 20-12",
            "number": 12,
            "file": "ch20-effective-training-time-ratio-ettr-model.svg",
            "asset": "ch20/ch20-effective-training-time-ratio-ettr-model.svg",
            "section": "20.1.1",
            "caption": "十万卡集群有效训练时间比（ETTR）对检查点周期 τ_ckpt、恢复时间 T_rec 与故障率 λ 的响应对比（63.22% vs. 92.34%）",
            "source_refs": ["experiments/ch20/20-01/run.py", "manuscripts/ch20/figure-data.json"],
            "TeachingRole": "量化展示内存检查点与秒级热恢复如何为十万卡集群挽回 29.12% 的有效算力"
        },
        {
            "figure_id": "图 20-13",
            "number": 13,
            "file": "ch20-automated-burn-in-fleet-health-pipeline.svg",
            "asset": "ch20/ch20-automated-burn-in-fleet-health-pipeline.svg",
            "section": "20.3.1",
            "caption": "新节点入网与故障修复后的自动化烤机（Burn-in：满载热压测 + 确定性 SDC 校验 + 光链路裕量扫描）准入流水线",
            "source_refs": ["manuscripts/ch20/figure-data.json"],
            "TeachingRole": "设计闭环机队健康状态机防止维修后带病节点重新污染在役训练集群"
        },
        {
            "figure_id": "图 20-14",
            "number": 14,
            "file": "ch20-twenty-chapter-dual-track-master-ledger.svg",
            "asset": "ch20/ch20-twenty-chapter-dual-track-master-ledger.svg",
            "section": "20.5.1",
            "caption": "全书 20 章双轨贯穿案例（Qwen3-8B Dense 轨与 DeepSeek V4.1 / MoE 轨）从第 1 章到第 20 章的完整工程参数演进总表",
            "source_refs": ["manuscripts/ch20/figure-data.json"],
            "TeachingRole": "横向串联全书二十章双轨模型在算力、显存、通信、能耗、安全与容错维度的定量参数"
        },
        {
            "figure_id": "图 20-15",
            "number": 15,
            "file": "ch20-twenty-chapter-architecture-knowledge-graph.svg",
            "asset": "ch20/ch20-twenty-chapter-architecture-knowledge-graph.svg",
            "section": "20.5.1",
            "caption": "全书 20 章“模型与负载 → 芯片与系统 → 推理、训练与六大前沿底座”的知识体系全景图",
            "source_refs": ["manuscripts/ch20/figure-data.json"],
            "TeachingRole": "建立自顶向下算法负载与自底向上物理定律双向耦合的全书知识坐标系"
        },
        {
            "figure_id": "图 20-16",
            "number": 16,
            "file": "ch20-ten-first-principles-of-ai-infra.svg",
            "asset": "ch20/ch20-ten-first-principles-of-ai-infra.svg",
            "section": "20.5.2",
            "caption": "AI 基础设施架构师的十条第一性原理终极法则总结图",
            "source_refs": ["manuscripts/ch20/figure-data.json"],
            "TeachingRole": "提炼贯穿全书二十章的十条第一性原理与定量工程铁律"
        }
    ]
    for row in fig_index:
        row['figure'] = f"20-{row['number']}"
        row['name'] = Path(row['file']).stem
    (HERE / 'figure-index.json').write_text(json.dumps(fig_index, ensure_ascii=False, indent=2) + '\n')

    md = HERE.parent / '20-十万卡集群可靠性工程、静默数据损坏（SDC）防御与基准评测科学.md'
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

        work = re.sub(r'!\[([^\]]*)\]\((ch20/[^)]+)\)', repl_img, text_md)
        body = markdown.markdown(work, extensions=['tables', 'fenced_code', 'footnotes', 'toc'])
        html = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>第 20 章：十万卡集群可靠性工程、静默数据损坏（SDC）防御与基准评测科学</title></head><body>{body}</body></html>\n'
        out_html = preview_path(md)
        out_html.write_text(html)

    sources = [
        HERE / 'build.py',
        HERE / 'figure-data.json',
        HERE / 'figure-index.json',
        ROOT / 'case-studies/collective-paths-and-diagnosis.md',
        ROOT / 'case-studies/optimization-evaluation-and-deployment.md',
        ROOT / 'experiments/ch20/20-01/run.py',
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
    print(f'Built {len(outputs)} figure files for Chapter 20; 0 layout warnings.')


if __name__ == '__main__':
    build_chapter()
