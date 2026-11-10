#!/usr/bin/env python3
"""Build Chapter 16 figures, locked calculations, and offline HTML preview."""
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


def fig_16_1_threat_model(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, '机密 AI 计算：防宿主窥探', color='blue', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '资产：闭源模型权重\n隐私提示词与 KV 缓存', 'blue', size=11)
    box(ax, 0.04, 0.44, 0.42, 0.16, '威胁：不可信云宿主机\nPCIe 嗅探与内存转储', 'orange', size=11)
    box(ax, 0.04, 0.22, 0.42, 0.16, '防线：CPU+GPU 双 TEE\n内存加密与 TEE-I/O', 'teal', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.60))
    arrow(ax, (0.25, 0.44), (0.25, 0.38))
    text(ax, 0.25, 0.11, '由外向内防护：宿主机不可信', size=11, ha='center', weight='bold', color=BORDER_COL['blue'])

    card(ax, 0.52, 0.04, 0.46, 0.92, '安全沙箱：防 Agent 逃逸', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '资产：宿主 Linux 内核\n训练内网与评测评分器', 'teal', size=11)
    box(ax, 0.54, 0.44, 0.42, 0.16, '威胁：百万 RL Agent\n恶意调用·内网扫描', 'orange', size=11)
    box(ax, 0.54, 0.22, 0.42, 0.16, '防线：Firecracker 虚机\nAppArmor + eBPF 白名单', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.60))
    arrow(ax, (0.75, 0.44), (0.75, 0.38))
    text(ax, 0.75, 0.11, '由内向外隔离：载荷不可信', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch16-confidential-ai-vs-sandbox-threat-model')


def fig_16_2_remote_attestation(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.52, 0.96, 0.44, 'CPU TEE（TDX/SEV-SNP）与 GPU TEE 联合远程证明四步握手', color='blue')
    steps = [
        (0.04, 0.60, 0.21, 0.22, '① 硬件度量\n固件哈希\n签名报告', 'blue'),
        (0.28, 0.60, 0.21, 0.22, '② 证书校验\nKMS 核验\n根证书链', 'teal'),
        (0.52, 0.60, 0.21, 0.22, '③ 信道绑定\nSPDM 建立\n机密会话', 'purple'),
        (0.76, 0.60, 0.20, 0.22, '④ 密钥释放\n仅向 TEE\n注入密钥', 'orange'),
    ]
    for x, y, w, h, label, role in steps:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.25, 0.71), (0.28, 0.71))
    arrow(ax, (0.49, 0.71), (0.52, 0.71))
    arrow(ax, (0.73, 0.71), (0.76, 0.71))

    card(ax, 0.02, 0.04, 0.96, 0.42, '加密显存边界：宿主机 Hypervisor 与驱动仅见密文', color='teal')
    box(ax, 0.04, 0.14, 0.44, 0.20, '不可信宿主机 OS 与驱动\n仅管页表映射与 DMA 调度\n读取 HBM 仅获密文乱码', 'orange', size=11)
    box(ax, 0.52, 0.14, 0.44, 0.20, 'GPU TEE 安全区（HBM 保护）\n拷贝引擎内联加解密与验签\n明文权重与激活永不外泄', 'teal', size=11)
    arrow(ax, (0.48, 0.24), (0.52, 0.24))
    text(ax, 0.50, 0.08, '未授权内存转储或重放均触发认证标签（MAC）校验熔断', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch16-cpu-gpu-tee-remote-attestation')


def fig_16_3_tee_io(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.52, 0.96, 0.44, '传统软件弹跳缓冲（Bounce Buffer）：双重拷贝与 CPU AES 瓶颈', color='orange')
    b_stages = [
        (0.04, 0.60, 0.21, 0.22, '① 私有内存\n64 MiB 明文\nCPU 待传', 'blue'),
        (0.28, 0.60, 0.21, 0.22, '② CPU 加密\nAES-GCM\n8 GB/s', 'orange'),
        (0.52, 0.60, 0.21, 0.22, '③ 弹跳缓冲\n共享内存\nPCIe DMA', 'orange'),
        (0.76, 0.60, 0.20, 0.22, '④ GPU 解密\nCE 验签\n共 8.8 ms', 'purple'),
    ]
    for x, y, w, h, label, role in b_stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.25, 0.71), (0.28, 0.71))
    arrow(ax, (0.49, 0.71), (0.52, 0.71))
    arrow(ax, (0.73, 0.71), (0.76, 0.71))

    card(ax, 0.02, 0.04, 0.96, 0.42, 'Vera Rubin 原生 TEE-I/O：PCIe Gen6 与 NVLink 6 线速机密互连', color='teal')
    box(ax, 0.04, 0.14, 0.44, 0.20, 'PCIe Gen6 TDISP / IDE 加密\n256 GB/s 链路硬件内联\n零弹跳缓冲，仅需 0.4 ms', 'teal', size=11)
    box(ax, 0.52, 0.14, 0.44, 0.20, 'NVLink 6 机密超节点互连\n单卡 3.6 TB/s·整柜 260 TB/s\n72 卡集合通信原生线速加密', 'blue', size=11)
    arrow(ax, (0.48, 0.24), (0.52, 0.24))
    text(ax, 0.50, 0.08, '算例 16-1：64 MiB 机密传输从 8.8 ms 降至 0.4 ms（提速 22 倍）', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch16-bounce-buffer-vs-rubin-tee-io')


def fig_16_4_pipellm(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.52, 0.96, 0.44, 'PipeLLM（ASPLOS \'25）五阶段机密张量卸载流水线（64 MiB）', color='blue')
    p_stages = [
        (0.04, 0.60, 0.16, 0.22, '① H2D\n密文拷入\n显存', 'blue'),
        (0.23, 0.60, 0.16, 0.22, '② 解密\n掩码异或\n明文', 'teal'),
        (0.42, 0.60, 0.16, 0.22, '③ 计算\n矩阵算子\n执行', 'purple'),
        (0.61, 0.60, 0.16, 0.22, '④ 加密\n输出激活\n异或', 'orange'),
        (0.80, 0.60, 0.16, 0.22, '⑤ D2H\n密文回传\n主机', 'gray'),
    ]
    for x, y, w, h, label, role in p_stages:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.20, 0.71), (0.23, 0.71))
    arrow(ax, (0.39, 0.71), (0.42, 0.71))
    arrow(ax, (0.58, 0.71), (0.61, 0.71))
    arrow(ax, (0.77, 0.71), (0.80, 0.71))

    card(ax, 0.02, 0.04, 0.96, 0.42, '算例 16-2：三缓冲显存驻留账本（3 × 64 = 192 MiB）', color='teal')
    box(ax, 0.04, 0.14, 0.28, 0.20, 'Buffer 0 (64 MiB)\n第 k+1 层\nH2D 预取与解密', 'blue', size=11)
    box(ax, 0.36, 0.14, 0.28, 0.20, 'Buffer 1 (64 MiB)\n第 k 层\nGPU 算子计算', 'teal', size=11)
    box(ax, 0.68, 0.14, 0.28, 0.20, 'Buffer 2 (64 MiB)\n第 k-1 层\n加密与 D2H 回写', 'orange', size=11)
    text(ax, 0.50, 0.08, '五阶段跨三套独立硬件队列并发重叠，周期取决于最慢单阶段', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch16-pipellm-five-stage-confidential-pipeline')


def fig_16_5_aes_ctr_otp(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.52, 0.96, 0.44, 'AES-CTR 一次性掩码（OTP）预生成与在线异或解耦', color='teal')
    box(ax, 0.04, 0.60, 0.44, 0.22, '离线/空闲期预生成（非关键路径）\n用密钥与递增计数器预生成掩码\nPad = AES_K(Nonce || Ctr)', 'teal', size=11)
    box(ax, 0.52, 0.60, 0.44, 0.22, '在线关键路径（HBM 带宽级异或）\nCipher = Plain ⊕ Pad\n严禁同一 Nonce 掩码复用两次', 'blue', size=11)
    arrow(ax, (0.48, 0.71), (0.52, 0.71))

    card(ax, 0.02, 0.04, 0.96, 0.42, '算例 16-3：投机分支预加密的收益与未命中惩罚边界', color='orange')
    box(ax, 0.04, 0.14, 0.44, 0.20, '投机命中（概率 p）\n提前完成候选分支加密与准备\n节省关键路径等待 Δt_hit', 'teal', size=11)
    box(ax, 0.52, 0.14, 0.44, 0.20, '投机未命中（概率 1 - p）\n废弃密文并非免费：耗主机拷贝\n与 PCIe 带宽惩罚 Δt_miss', 'orange', size=11)
    text(ax, 0.50, 0.08, '正收益条件：p · Δt_hit > (1 - p) · Δt_miss（须计入主机拷贝代价）', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch16-aes-ctr-otp-speculative-encryption')


def fig_16_6_escape_failures(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.51, 0.96, 0.45, 'DeepSeek DSec 实录：共享宿主内核下 RL Agent 的三类破坏', color='orange')
    modes = [
        (0.04, 0.57, 0.28, 0.25, '① 文件系统破坏\nXFS_IOC_SWAPEXT\n损毁 XFS 元数据\n覆盖 /bin/bash', 'orange'),
        (0.36, 0.57, 0.28, 0.25, '② 内核与日志洪水\n读 kpagecgroup\n触发宿主内核缺陷\nyes 刷爆数十GB', 'purple'),
        (0.68, 0.57, 0.28, 0.25, '③ 内网与奖励作弊\n扫描端口伪造\nchronus 套接字\nGo 代理偷答案', 'blue'),
    ]
    for x, y, w, h, label, role in modes:
        box(ax, x, y, w, h, label, role, size=11)

    card(ax, 0.02, 0.04, 0.96, 0.42, '架构结论：自主代码 Agent 必须从裸容器升级为硬件虚拟化', color='teal')
    box(ax, 0.04, 0.14, 0.44, 0.20, '裸容器（共享宿主内核）\n系统调用直通宿主 Linux\n单点内核漏洞即全机沦陷', 'orange', size=11)
    box(ax, 0.52, 0.14, 0.44, 0.20, '独立 Guest 内核（微虚机）\nKVM 虚拟化阻断宿主攻击面\n叠加 AppArmor 与 eBPF', 'teal', size=11)
    arrow(ax, (0.48, 0.24), (0.52, 0.24))
    text(ax, 0.50, 0.08, 'RL 探索天然奖励“走捷径得分”，必须假设沙箱内代码怀有恶意', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch16-agent-container-escape-failure-modes')


def fig_16_7_dsec_arch(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.52, 0.96, 0.44, 'DeepSeek DSec 四类隔离后端（160 节点·38 万并发沙箱）', color='blue')
    backends = [
        (0.04, 0.60, 0.21, 0.22, '① 预热 FnCall\n无状态函数\n毫秒级复用', 'blue'),
        (0.28, 0.60, 0.21, 0.22, '② VM 容器\nQEMU 承载\n≥3,200/机', 'teal'),
        (0.52, 0.60, 0.21, 0.22, '③ Firecracker\n微虚拟机\n≥800/机', 'purple'),
        (0.76, 0.60, 0.20, 0.22, '④ 完整 QEMU\nAndroid 与\nGPU 虚拟化', 'orange'),
    ]
    for x, y, w, h, label, role in backends:
        box(ax, x, y, w, h, label, role, size=11)

    card(ax, 0.02, 0.04, 0.96, 0.42, 'DSec 核心资源指标与四大底层支撑技术', color='teal')
    box(ax, 0.04, 0.14, 0.44, 0.20, '生产吞吐与超卖密度\n日创建 300 万·峰值 >5,000/s\n90% 沙箱平均 CPU ≤ 5%', 'blue', size=11)
    box(ax, 0.52, 0.14, 0.44, 0.20, '四大底层支撑技术\n按需 EROFS·pmem+DAX\nDAMON 气球·核调度+eBPF', 'teal', size=11)
    arrow(ax, (0.48, 0.24), (0.52, 0.24))
    text(ax, 0.50, 0.08, '160 节点（30,000 核 / 250 TB DRAM）支撑日均 300 万次沙箱评测', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch16-dsec-firecracker-microvm-architecture')


def fig_16_8_pmem_dax(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.52, 0.96, 0.44, '只读层去重：virtio-blk 双重缓存 vs. virtio-pmem + DAX', color='teal')
    box(ax, 0.04, 0.60, 0.44, 0.22, '传统块设备（virtio-blk）\n宿主页缓存 + 800 份客户机缓存\n相同镜像被重复缓存数百份', 'orange', size=11)
    box(ax, 0.52, 0.60, 0.44, 0.22, 'virtio-pmem + DAX 直映\n绕过 Guest 缓存共享宿主物理页\n实测削减 40.2% 节点峰值内存', 'teal', size=11)
    arrow(ax, (0.48, 0.71), (0.52, 0.71))

    card(ax, 0.02, 0.04, 0.96, 0.42, '算例 16-4：可写大磁盘为何不盲目开 DAX（struct page 税）', color='orange')
    box(ax, 0.04, 0.14, 0.44, 0.20, 'Guest struct page 固定税\n每 4 KiB 页耗 64 B（占 1/64）\n128 GB 盘耗 2 GB 客户机内存', 'orange', size=11)
    box(ax, 0.52, 0.14, 0.44, 0.20, 'DSec 分层挂载策略\n只读 EROFS 层用 pmem+DAX\n可写大盘用 DAMON+气球回收', 'blue', size=11)
    arrow(ax, (0.48, 0.24), (0.52, 0.24))
    text(ax, 0.50, 0.08, '开启 DAX 还会使瞬时 CPU 峰值由 26.5% 升至 41.4%，须按盘大小取舍', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch16-virtio-pmem-dax-vs-struct-page-tax')


def fig_16_9_damon_balloon(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.52, 0.96, 0.44, '可写磁盘与堆内存回收：DAMON + virtio-balloon 联动', color='blue')
    d_steps = [
        (0.04, 0.60, 0.21, 0.22, '① 采样监控\nDAMON 追踪\n页访问热度', 'blue'),
        (0.28, 0.60, 0.21, 0.22, '② 识别冷页\n超时未访页\n标为可回收', 'teal'),
        (0.52, 0.60, 0.21, 0.22, '③ 气球上报\n2 MiB 粒度\n空闲页报告', 'purple'),
        (0.76, 0.60, 0.20, 0.22, '④ 宿主释放\nmadvise 归还\n物理内存', 'orange'),
    ]
    for x, y, w, h, label, role in d_steps:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.25, 0.71), (0.28, 0.71))
    arrow(ax, (0.49, 0.71), (0.52, 0.71))
    arrow(ax, (0.73, 0.71), (0.76, 0.71))

    card(ax, 0.02, 0.04, 0.96, 0.42, '生产实测收益：order-9（2 MiB）粒度与时间积分内存下降', color='teal')
    box(ax, 0.04, 0.14, 0.44, 0.20, '默认空闲页报告：order-9\n按 512 × 4 KiB = 2 MiB 块上报\n兼顾回收效率与虚机退出开销', 'blue', size=11)
    box(ax, 0.52, 0.14, 0.44, 0.20, 'DSec 生产集群实测收益\n时间积分内存（GB·s）降 21.2%\n保障单机 800+ 微虚机稳定', 'teal', size=11)
    arrow(ax, (0.48, 0.24), (0.52, 0.24))
    text(ax, 0.50, 0.08, '与只读层 pmem+DAX（峰值 -40.2%）正交配合，消除虚机双重内存开销', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch16-damon-ept-balloon-reclaim')


def fig_16_10_core_sched(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.52, 0.96, 0.44, '算例 16-5：延迟敏感（LS）评测与尽力而为（BE）同机混部', color='orange')
    cases = [
        (0.04, 0.60, 0.28, 0.22, '① 默认 CFS 混部\n50% BE 同机\nLS 耗时 +45.2%', 'orange'),
        (0.36, 0.60, 0.28, 0.22, '② 仅 SCHED_IDLE\nSMT 仍争流水线\n最多改善 3.4%', 'purple'),
        (0.68, 0.60, 0.28, 0.22, '③ IDLE+核调度\nPR_SCHED_CORE\n膨胀压至 +17.3%', 'teal'),
    ]
    for x, y, w, h, label, role in cases:
        box(ax, x, y, w, h, label, role, size=11)

    card(ax, 0.02, 0.04, 0.96, 0.42, '为什么需要 SMT 核调度及剩余 17.3% 干扰来源', color='blue')
    box(ax, 0.04, 0.14, 0.44, 0.20, 'prctl(PR_SCHED_CORE)\n禁止 BE 与 LS 线程同时共处\n同一物理核的两个 SMT 逻辑核', 'teal', size=11)
    box(ax, 0.52, 0.14, 0.44, 0.20, '剩余 +17.3% 膨胀的物理根源\n多核高负载触发睿频降频\n及末级缓存与内存带宽争用', 'blue', size=11)
    arrow(ax, (0.48, 0.24), (0.52, 0.24))
    text(ax, 0.50, 0.08, '耗时敏感型 RL 评测若不开启核调度，混部噪声将直接污染奖励信号', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch16-sched-idle-and-core-scheduling')


def fig_16_11_ebpf_allowlist(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.52, 0.96, 0.44, 'DSec 纵深防御：AppArmor + 动态 eBPF 网络白名单', color='blue')
    net_steps = [
        (0.04, 0.60, 0.21, 0.22, '① 沙箱进程\n发起 socket\n连接请求', 'blue'),
        (0.28, 0.60, 0.21, 0.22, '② AppArmor\n拦截违规文件\n与本地套接字', 'teal'),
        (0.52, 0.60, 0.21, 0.22, '③ 内核 eBPF\n按任务查核\nIP 白名单', 'purple'),
        (0.76, 0.60, 0.20, 0.22, '④ 裁决执行\n阻断内网探测\n与未授权外联', 'orange'),
    ]
    for x, y, w, h, label, role in net_steps:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.25, 0.71), (0.28, 0.71))
    arrow(ax, (0.49, 0.71), (0.52, 0.71))
    arrow(ax, (0.73, 0.71), (0.76, 0.71))

    card(ax, 0.02, 0.04, 0.96, 0.42, '精准封堵两类生产越权：内网服务保护与评测防作弊', color='teal')
    box(ax, 0.04, 0.14, 0.44, 0.20, '阻断内网横向移动\n默认拒绝 RFC1918 私网网段\n保护 chronus 与元数据服务', 'orange', size=11)
    box(ax, 0.52, 0.14, 0.44, 0.20, '任务级细粒度放行\n闭卷代码评测彻底断网防作弊\n工具调用仅放行指定 API', 'teal', size=11)
    arrow(ax, (0.48, 0.24), (0.52, 0.24))
    text(ax, 0.50, 0.08, '在宿主内核态 eBPF 钩子处直接阻断，沙箱内获 root 也无法绕过', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch16-ebpf-cgroup-network-allowlist')


def fig_16_12_audit_drain(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.52, 0.96, 0.44, '算例 16-6（I19）：日志后端中断 120 s 积压与溢出点', color='orange')
    out_boxes = [
        (0.04, 0.60, 0.28, 0.22, '稳态事件到达率\n10 任务/s × 20 条\n400 KiB/s', 'blue'),
        (0.36, 0.60, 0.28, 0.22, '中断 120 s 积压\n24,000 条事件\n共 46.875 MiB', 'orange'),
        (0.68, 0.60, 0.28, 0.22, '32 MiB 缓冲溢出\n32,768 / 400\n第 81.92 s 丢账', 'purple'),
    ]
    for x, y, w, h, label, role in out_boxes:
        box(ax, x, y, w, h, label, role, size=11)

    card(ax, 0.02, 0.04, 0.96, 0.42, '恢复期排空方程：必须扣除持续涌入的新事件流（净排空率）', color='teal')
    box(ax, 0.04, 0.14, 0.44, 0.20, '常见错算：忽略新到达流量\n误算 46.875 / 1.0 = 46.88 s\n低估恢复期高水位持续时长', 'orange', size=11)
    box(ax, 0.52, 0.14, 0.44, 0.20, '正确净排空（64 MiB 缓冲）\nR_net = 1,024 - 400 = 624 KiB/s\nT_drain = 48,000 / 624 ≈ 76.92 s', 'teal', size=11)
    arrow(ax, (0.48, 0.24), (0.52, 0.24))
    text(ax, 0.50, 0.08, '本地持久缓冲须 ≥ 64 MiB，且按净排空速率 624 KiB/s 核算恢复窗口', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch16-agent-audit-log-outage-and-drain')


def fig_16_13_idempotent_commit(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.52, 0.96, 0.44, '工具调用外部副作用防重：幂等键（Idempotency Key）两阶段提交', color='blue')
    i_steps = [
        (0.04, 0.60, 0.21, 0.22, '① 生成幂等键\n任务+步号\n参数哈希', 'blue'),
        (0.28, 0.60, 0.21, 0.22, '② 查状态账本\n若已提交则\n返回旧结果', 'teal'),
        (0.52, 0.60, 0.21, 0.22, '③ 外部执行\n写执行中态\n调外部 API', 'purple'),
        (0.76, 0.60, 0.20, 0.22, '④ 落盘确认\n存响应快照\n置已提交态', 'orange'),
    ]
    for x, y, w, h, label, role in i_steps:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.25, 0.71), (0.28, 0.71))
    arrow(ax, (0.49, 0.71), (0.52, 0.71))
    arrow(ax, (0.73, 0.71), (0.76, 0.71))

    card(ax, 0.02, 0.04, 0.96, 0.42, '算例 16-7：ACK 丢失或沙箱抢占重启下的精准重放', color='teal')
    box(ax, 0.04, 0.14, 0.44, 0.20, '无幂等网关：直接重试调用\n导致重复扣款或外部状态污染\n离线回放时外部环境已漂移', 'orange', size=11)
    box(ax, 0.52, 0.14, 0.44, 0.20, '幂等网关：拦截重复副作用\n重传已缓存的确定性响应快照\n至少一次传输 + 恰好一次生效', 'teal', size=11)
    arrow(ax, (0.48, 0.24), (0.52, 0.24))
    text(ax, 0.50, 0.08, '把“消息重传”与“外部动作重做”解耦，同时支撑确定性离线回放', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch16-idempotent-tool-commit-and-replay')


def fig_16_14_telemetry_vs_ledger(ex: Exporter):
    fig, ax = canvas(3.65)
    card(ax, 0.02, 0.04, 0.46, 0.92, '可采样通道：性能诊断 Trace', color='blue', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '数据：算子耗时与调度跨度\nOpenTelemetry Span 明细', 'blue', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '尾部采样：正常请求抽 1%\n慢请求与报错 100% 保留', 'teal', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '拥塞降级：缓冲满允许丢弃\n不影响计费与训练正确性', 'gray', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '目标：低开销定位系统尾延迟', size=11, ha='center', weight='bold', color=BORDER_COL['blue'])

    card(ax, 0.52, 0.04, 0.46, 0.92, '不可采样：计费与准入账本', color='orange', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '数据：Token 计费·幂等键\nRL 轨迹策略版本准入记录', 'orange', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '严禁采样：100% 写持久 WAL\n加密哈希链防篡改与漏记', 'purple', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '拥塞反压：逼近上限主动限流\n宁可暂停接单也绝不丢账', 'teal', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '目标：零丢账·零重扣·零脏轨迹', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])
    ex.save(fig, 'ch16-telemetry-sampling-vs-billing-ledger')


def fig_16_15_full_stack_map(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.52, 0.96, 0.44, '全书十六章知识坐标系：从矩阵乘法到吉瓦级可信数据中心', color='blue')
    layers = [
        (0.04, 0.59, 0.21, 0.24, '第一部分\nCh1–3\n负载与模型', 'blue'),
        (0.28, 0.59, 0.21, 0.24, '第二部分\nCh4–7\n芯片与网络', 'teal'),
        (0.52, 0.59, 0.21, 0.24, '第三部分\nCh8–12\n推理与训练', 'purple'),
        (0.76, 0.59, 0.20, 0.24, '第四部分\nCh13–16\n电热与安全', 'orange'),
    ]
    for x, y, w, h, label, role in layers:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.25, 0.71), (0.28, 0.71))
    arrow(ax, (0.49, 0.71), (0.52, 0.71))
    arrow(ax, (0.73, 0.71), (0.76, 0.71))

    card(ax, 0.02, 0.04, 0.96, 0.42, '贯穿全书的五个第一性追问（Five Fundamental Questions）', color='teal')
    box(ax, 0.04, 0.15, 0.44, 0.19, '① 搬什么·② 搬多少·③ 搬几次\n决定数值表示（FP8/FP4/MLA）\n与各级存储层次复用率', 'blue', size=11)
    box(ax, 0.52, 0.15, 0.44, 0.19, '④ 经过哪里·⑤ 谁必须等它\n决定超节点拓扑、同步屏障、\n功率平滑与 TEE-I/O 边界', 'teal', size=11)
    arrow(ax, (0.48, 0.245), (0.52, 0.245))
    text(ax, 0.50, 0.08, '一切架构演进皆围绕缩短关键路径等待、逼近带宽与能效极限展开', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch16-sixteen-chapter-full-stack-map')


def fig_16_16_master_checklist(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.04, 0.96, 0.92, 'AI 基础设施架构师的六维终极核查表（Master Checklist）', color='blue')
    dims = [
        (0.04, 0.56, 0.28, 0.25, '① 算力与精度\n算术强度拐点\nFP8/FP4 尾项', 'blue'),
        (0.36, 0.56, 0.28, 0.25, '② 存力与容量\n权重+KV+激活\nHBM4·SSD 寿命', 'teal'),
        (0.68, 0.56, 0.28, 0.25, '③ 运力与同步\nTPOT 通信预算\nNVLink6·轨隔离', 'purple'),
        (0.04, 0.18, 0.28, 0.25, '④ 电力与热学\n800 VDC·削峰20%\nW45 温水冷却', 'orange'),
        (0.36, 0.18, 0.28, 0.25, '⑤ 可信与沙箱\nTEE-I/O 线速加密\nDAX 去重·核调度', 'teal'),
        (0.68, 0.18, 0.28, 0.25, '⑥ 生命周期成本\n训练+推理总成本\n条件比率防重计', 'blue'),
    ]
    for x, y, w, h, label, role in dims:
        box(ax, x, y, w, h, label, role, size=11)
    text(ax, 0.50, 0.09, '先做数量级信封背面估算，再做受控实验与预注册证伪', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch16-ai-infra-architect-master-checklist')


def fig_16_17_moe_padding(ex: Exporter):
    fig, ax = canvas(3.80)
    card(ax, 0.02, 0.04, 0.46, 0.92, '密文下 MoE 流量侧信道', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '即便启用 TEE-I/O 载荷加密\n攻击者仍观测跨节点包计数', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '代码/医疗/金融提示词触发\n特定 EP Rank 流量偏斜', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '结合投机解码步长波动指纹\n反推用户敏感领域与提示词', 'purple', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '加密隐藏内容未藏流量形状', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, '恒定速率哑元填充防御', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '为各 EP Rank 设统一桶深\nC_pad = μ · (1 + α) 槽位', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '不足槽位注入加密哑元词元\n跨节点 All-to-All 字节恒定', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '练习 16-11：当 α = 0.35 时\n付 35% 带宽税换零偏斜泄露', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '以受控带宽税抹平侧信道特征', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch16-moe-routing-traffic-side-channel-and-padding')


def fig_16_18_verifiable_triangle(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '可验证推理（Verifiable Inference）三路线对比', color='blue')
    routes = [
        (0.04, 0.58, 0.28, 0.24, '① 硬件 TEE 证明\n开销 < 5% 极低\n依芯片硬件信任根', 'teal'),
        (0.36, 0.58, 0.28, 0.24, '② 密码学 zkML\n数学无条件可验证\n算力放大千至十万倍', 'purple'),
        (0.68, 0.58, 0.28, 0.24, '③ 乐观抽检 OPML\n仅抽检 p 比例复算\n质押罚没博弈保障', 'blue'),
    ]
    for x, y, w, h, label, role in routes:
        box(ax, x, y, w, h, label, role, size=11)

    card(ax, 0.02, 0.03, 0.96, 0.42, '防止云端 API 偷换降级模型（如 70B 换 8B）的分层选型', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '千亿大模型高吞吐在线服务\n采用 GPU TEE 签名激活摘要\n叠加 OPML 低比例随机抽检', 'teal', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '高敏金融风控与局部审计裁决\n对争议层或轻量判别器用 zkML\n生成零知识简洁非交互证明', 'blue', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '日常流量走 TEE + OPML 抽检，争议仲裁触发确定性复算证明', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch16-verifiable-inference-tee-zkml-opml-triangle')


def fig_16_19_opml_slashing(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, '乐观可验证推理（OPML）的博弈树与纳什均衡条件', color='blue')
    nodes = [
        (0.04, 0.58, 0.28, 0.24, '① 偷换模型收益\n每次省算力 Δc\n未抽中率 1 - p', 'orange'),
        (0.36, 0.58, 0.28, 0.24, '② 随机抽检复算\n抽检率 p 触发\n确定性重放比对', 'blue'),
        (0.68, 0.58, 0.28, 0.24, '③ 欺诈证明罚没\n一旦检出不一致\n罚没押金 S_slash', 'purple'),
    ]
    for x, y, w, h, label, role in nodes:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.32, 0.70), (0.36, 0.70))
    arrow(ax, (0.64, 0.70), (0.68, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, '练习 16-12：Δc=0.005 美元、S_slash=1,000 美元诚实边界', color='teal')
    box(ax, 0.04, 0.13, 0.44, 0.20, '激励相容不等式（诚实占优）\np · S_slash > (1 - p) · Δc\n导出 p_min = Δc / (S + Δc)', 'teal', size=11)
    box(ax, 0.52, 0.13, 0.44, 0.20, '代入 Δc=0.005, S=1,000 美元\np_min ≈ 0.0005%（5×10^-6）\n万分之几抽检即可杜绝作弊', 'blue', size=11)
    arrow(ax, (0.48, 0.23), (0.52, 0.23))
    text(ax, 0.50, 0.06, '高额质押金将验证开销压缩五个数量级，前提是算子确定性可复现', size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, 'ch16-optimistic-verification-slashing-game')


def fig_16_20_janus_nested(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, '传统嵌套虚拟化（页表税）', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '公有云租户 L1 VM 内部\n再起 L2 沙箱微虚拟机', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '三维嵌套扩展页表（EPT 嵌套）\nTLB 缺失最坏触发 124 次访存', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, 'VM-Exit 陷 L0 再转 L1\n系统调用与 I/O 延迟翻倍', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '双重页表吞噬沙箱吞吐', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'JANUS（OSDI 26）协同虚拟化', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '跨世界协同（Cross-World）\n将 L2 沙箱提为 L0 同级域', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '扁平化单级 EPT 硬件页表\n回落至 24 次访存消除放大', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '安全委托通道连 L1 管理面\n兼得租户封装与裸机性能', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '公有云 VM 内微虚机逼近原生', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch16-janus-nested-virtualization-fast-path')


def fig_16_21_blowfish_prism(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.04, 0.46, 0.92, 'Blowfish（OSDI 26）内存池', color='blue', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '面向主机 + CXL 解耦内存池\n跨数百个高密沙箱微虚拟机', 'blue', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '区分真空闲页与冷可回收页\n优先调度本地 DRAM 热页', 'teal', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '突发唤醒亚毫秒级按需回填\n消除传统气球回收长尾卡顿', 'purple', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, 'CPU 沙箱：跨虚机弹性内存池', size=11, ha='center', weight='bold', color=BORDER_COL['blue'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'Prism（OSDI 26）显存气球', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '面向同卡共置的多模型实例\n打破静态显存预留分区墙', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, 'GPU 虚拟显存气球动态伸缩\n低谷模型归还物理 HBM 页', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '突发模型零拷贝借空闲显存\n支撑多模型 KV 时间复用', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, 'GPU 推理：跨模型显存气球', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch16-blowfish-elastic-sandbox-memory-pool')


def fig_16_22_defense_in_depth(ex: Exporter):
    fig, ax = canvas(4.15)
    card(ax, 0.02, 0.03, 0.96, 0.94, '可信 AI 基础设施纵深防御全景（Defense-in-Depth）', color='teal')
    layers = [
        (0.05, 0.66, 0.42, 0.20, '① 硬件机密底座（16.2–3）\nCPU+GPU 双 TEE 远程证明\nRubin TEE-I/O 与 PipeLLM', 'blue'),
        (0.53, 0.66, 0.42, 0.20, '② 侧信道与验证（16.6.1–2）\nMoE 恒定桶深哑元填充防御\nTEE 签名 + OPML 抽检罚没', 'teal'),
        (0.53, 0.37, 0.42, 0.20, '③ 高密微虚机（16.4 & 6.3）\nFirecracker + JANUS 旁路\npmem+DAX · Blowfish/Prism', 'purple'),
        (0.05, 0.37, 0.42, 0.20, '④ 零信任与审计（16.4–5）\nSCHED_IDLE + 核调度去噪\neBPF 白名单 · 幂等网关排空', 'orange'),
    ]
    for x, y, w, h, label, role in layers:
        box(ax, x, y, w, h, label, role, size=11)
    arrow(ax, (0.47, 0.76), (0.53, 0.76))
    arrow(ax, (0.74, 0.66), (0.74, 0.57))
    arrow(ax, (0.53, 0.47), (0.47, 0.47))
    arrow(ax, (0.26, 0.57), (0.26, 0.66))
    box(ax, 0.05, 0.08, 0.90, 0.21, '实验验证：ch16/16-01（PipeLLM·MoE 填充·OPML 博弈）\n与 ch16/16-02（DSec 38 万沙箱·DAX 元数据税·I19 排空）', 'blue', size=11)
    ex.save(fig, 'ch16-trusted-ai-infrastructure-defense-in-depth')


def fig_16_23_lease_bound_weights(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, 'A. 远程证明租约绑定权重分发', color='blue', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '磁盘/主机常驻密文模型权重\n140 GB AES-GCM-256 加密块', 'blue', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, 'CPU+GPU 双 TEE 联合远程证明\nKMS 签发租约密钥 (心跳 60 s)', 'teal', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '仅在 GPU 加密 HBM 内解封 DEK\n宿主机 root 零明文访问权限', 'purple', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '私有化部署严防闭源权重被盗', size=11, ha='center', weight='bold', color=BORDER_COL['blue'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'B. 算例 16-11：宽限期与硬件擦除', color='orange', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '断网宽限期 T_grace = 600 s\n容忍最多 10 次心跳闪断', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '超时或拆机篡改触发硬件熔断\n0.50 ms 销毁片上寄存器 DEK', 'orange', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '控制器按 7.0 TB/s 并行覆写\n20.0 ms 完成 140 GB 显存清零', 'blue', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '密码学擦除 + 物理显存覆写双保险', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])
    ex.save(fig, 'ch16-lease-bound-encrypted-weight-distribution')


def fig_16_24_fused_watermark(ex: Exporter):
    fig, ax = canvas(3.75)
    card(ax, 0.02, 0.04, 0.46, 0.92, 'A. TEE 算子内联锦标赛水印', color='purple', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '传统用户态 Python 后处理水印\n易被恶意宿主机截获并剥离', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, 'TEE 采样算子融合 (SynthID)\nm=4 层伪随机锦标赛非失真采样', 'purple', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, '保持单步分布无偏 (零 PPL 退化)\n辅以指令/数据 KV 污点标签隔离', 'teal', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '算子级硬件绑定阻断无损剥离', size=11, ha='center', weight='bold', color=BORDER_COL['purple'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'B. 算例 16-12：Z 分数检验边界', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, 'm=4 层 · 熵 H=1.8 bits/tok\n单层有效胜率 p_eff = 0.62', 'blue', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, 'L=200 词元 (N=800 位检验)\nZ = 6.79 · FPR ≈ 9.86e-11', 'teal', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '锁定 Z ≥ 4.0 合规取证门限\n最小检测长度 L_min = 70 词元', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '短文本即可实现高置信度内容溯源', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch16-fused-cryptographic-logit-watermark-kernel')


def fig_16_25_timing_shaping(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.04, 0.46, 0.92, 'A. 投机解码与变长侧信道泄露', color='orange', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, '投机解码每步接受 1–5 词元\n包间隔 Δt_i 暴露领域词频', 'orange', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '微批真实长度 {19,47,83,142,215}\n总词元 506 · 泄露 8.0 bits', 'orange', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, 'PMIC 功耗阶跃与 TTFT 跳变\n暴露跨租户前缀缓存命中', 'orange', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '仅加密载荷无法掩盖时序包长指纹', size=11, ha='center', weight='bold', color=BORDER_COL['orange'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'B. 恒定节拍与 64-Token 分桶整形', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '节拍锁相 Δτ_tick = 25.0 ms\n相对平均 TPOT 延迟税 12.11%', 'teal', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '向上对齐 64 倍数 {64..256}\n填充后 704 词元 (带宽税 39.13%)', 'blue', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '尾部步注入等长 4 KB 哑元包\n桶内互信息泄露 = 0.0 bits', 'purple', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '时空双轴整形彻底阻断被动侧信道', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch16-tee-timing-side-channel-traffic-shaping')


def fig_16_26_verifiable_zkml(ex: Exporter):
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.04, 0.46, 0.92, 'A. 密码学算子校验与专用 zkML', color='blue', title_size=11)
    box(ax, 0.04, 0.66, 0.42, 0.16, 'Freivalds 随机投影 (k=40)\n检验 A(Br)=Cr · 漏检 2^-40', 'blue', size=11)
    box(ax, 0.04, 0.43, 0.42, 0.16, '4096×4096×14336 FFN 层\n校验仅占 GEMM 算力 2.23%', 'teal', size=11)
    box(ax, 0.04, 0.20, 0.42, 0.16, 'zkLLM Sumcheck + tlookup\n比通用电路提速 64.86× (185×)', 'purple', size=11)
    arrow(ax, (0.25, 0.66), (0.25, 0.59))
    arrow(ax, (0.25, 0.43), (0.25, 0.36))
    text(ax, 0.25, 0.10, '线性投影抽检 + 多线性扩展证明', size=11, ha='center', weight='bold', color=BORDER_COL['blue'])

    card(ax, 0.52, 0.04, 0.46, 0.92, 'B. 触发式单算子 zk-Fault-Proof', color='teal', title_size=11)
    box(ax, 0.54, 0.66, 0.42, 0.16, '传统 OPML 320 算子二分交互\n需 9 轮 · 耗时 45.0 s (Gas $42)', 'orange', size=11)
    box(ax, 0.54, 0.43, 0.42, 0.16, '定位唯一争议算子后非交互证明\n生成 1.65 s + 链上验证 0.20 s', 'teal', size=11)
    box(ax, 0.54, 0.20, 0.42, 0.16, '裁决总延迟 1.85 s (快 24.32×)\n链上 Gas 降至 $4.83 (省 88.5%)', 'blue', size=11)
    arrow(ax, (0.75, 0.66), (0.75, 0.59))
    arrow(ax, (0.75, 0.43), (0.75, 0.36))
    text(ax, 0.75, 0.10, '兼具 TEE 实时性与零知识数学终局', size=11, ha='center', weight='bold', color=BORDER_COL['teal'])
    ex.save(fig, 'ch16-verifiable-inference-optimistic-and-zkml-proof')


def main():
    ex = Exporter(HERE)
    fig_16_1_threat_model(ex)
    fig_16_2_remote_attestation(ex)
    fig_16_3_tee_io(ex)
    fig_16_4_pipellm(ex)
    fig_16_5_aes_ctr_otp(ex)
    fig_16_6_escape_failures(ex)
    fig_16_7_dsec_arch(ex)
    fig_16_8_pmem_dax(ex)
    fig_16_9_damon_balloon(ex)
    fig_16_10_core_sched(ex)
    fig_16_11_ebpf_allowlist(ex)
    fig_16_12_audit_drain(ex)
    fig_16_13_idempotent_commit(ex)
    fig_16_14_telemetry_vs_ledger(ex)
    fig_16_15_full_stack_map(ex)
    fig_16_16_master_checklist(ex)
    fig_16_17_moe_padding(ex)
    fig_16_18_verifiable_triangle(ex)
    fig_16_19_opml_slashing(ex)
    fig_16_20_janus_nested(ex)
    fig_16_21_blowfish_prism(ex)
    fig_16_22_defense_in_depth(ex)
    fig_16_23_lease_bound_weights(ex)
    fig_16_24_fused_watermark(ex)
    fig_16_25_timing_shaping(ex)
    fig_16_26_verifiable_zkml(ex)
    outputs = ex.finish()

    fig_index = [
        {
            "figure_id": "图 16-1",
            "number": 1,
            "file": "ch16-confidential-ai-vs-sandbox-threat-model.svg",
            "asset": "ch16/ch16-confidential-ai-vs-sandbox-threat-model.svg",
            "section": "16.1.1",
            "caption": "AI 基础设施的双向信任边界：防宿主机窥探的机密 AI 计算（TEE）vs. 防自主 Agent 逃逸的安全沙箱（microVM）",
            "source_refs": ["case-studies/dsec-sandbox-platform.md", "manuscripts/ch16/figure-data.json"],
            "TeachingRole": "建立机密计算（宿主机不可信）与安全沙箱（工作负载不可信）的正交双向威胁模型"
        },
        {
            "figure_id": "图 16-2",
            "number": 2,
            "file": "ch16-cpu-gpu-tee-remote-attestation.svg",
            "asset": "ch16/ch16-cpu-gpu-tee-remote-attestation.svg",
            "section": "16.2.1",
            "caption": "CPU TEE（Intel TDX / AMD SEV-SNP）与 GPU TEE 的联合远程证明四步握手及 HBM 加密显存边界",
            "source_refs": ["references/text/nvidia-rubin-arch.txt", "manuscripts/ch16/figure-data.json"],
            "TeachingRole": "展示硬件信任根如何确保模型解密密钥仅在通过度量核验的可信执行环境内部释放"
        },
        {
            "figure_id": "图 16-3",
            "number": 3,
            "file": "ch16-bounce-buffer-vs-rubin-tee-io.svg",
            "asset": "ch16/ch16-bounce-buffer-vs-rubin-tee-io.svg",
            "section": "16.2.2",
            "caption": "传统软件弹跳缓冲双重拷贝加密 vs. Vera Rubin 原生 PCIe Gen6（256 GB/s）与 NVLink 6（3.6 TB/s）硬件内联线速 TEE-I/O（算例 16-1）",
            "source_refs": ["references/text/nvidia-rubin-arch.txt", "manuscripts/ch16/figure-data.json"],
            "TeachingRole": "量化消除主机弹跳缓冲与 CPU AES-GCM 瓶颈后 64 MiB 机密传输 22 倍的延迟压缩"
        },
        {
            "figure_id": "图 16-4",
            "number": 4,
            "file": "ch16-pipellm-five-stage-confidential-pipeline.svg",
            "asset": "ch16/ch16-pipellm-five-stage-confidential-pipeline.svg",
            "section": "16.3.1",
            "caption": "PipeLLM（ASPLOS '25）五阶段机密张量卸载流水线（H2D → Decrypt → Compute → Encrypt → D2H）与 192 MiB 三缓冲显存账本（算例 16-2）",
            "source_refs": ["research/2026-infra-survey/reading-asplos-2025.md", "references/proceedings/ASPLOS/2025/pipellm-reading.json"],
            "TeachingRole": "拆解机密卸载中用 3 × 64 MiB = 192 MiB 显存三缓冲换取加解密与传输重叠的流水机制"
        },
        {
            "figure_id": "图 16-5",
            "number": 5,
            "file": "ch16-aes-ctr-otp-speculative-encryption.svg",
            "asset": "ch16/ch16-aes-ctr-otp-speculative-encryption.svg",
            "section": "16.3.2",
            "caption": "AES-CTR 一次性掩码（OTP）离线预生成与投机分支预加密的命中收益及未命中带宽惩罚边界（算例 16-3）",
            "source_refs": ["research/2026-infra-survey/reading-asplos-2025.md", "references/proceedings/ASPLOS/2025/pipellm-reading.json"],
            "TeachingRole": "阐明在线异或加速的原理，并指出投机未命中时的额外主机拷贝与总线带宽并非免费"
        },
        {
            "figure_id": "图 16-6",
            "number": 6,
            "file": "ch16-agent-container-escape-failure-modes.svg",
            "asset": "ch16/ch16-agent-container-escape-failure-modes.svg",
            "section": "16.4.1",
            "caption": "DeepSeek DSec 生产环境中自主 RL Agent 在共享宿主机内核容器下的三类真实越界与故障案例",
            "source_refs": ["case-studies/dsec-sandbox-platform.md", "manuscripts/ch16/figure-data.json"],
            "TeachingRole": "用 XFS_IOC_SWAPEXT、/proc/kpagecgroup 与 chronus 伪造实证为何裸容器无法隔离强化学习 Agent"
        },
        {
            "figure_id": "图 16-7",
            "number": 7,
            "file": "ch16-dsec-firecracker-microvm-architecture.svg",
            "asset": "ch16/ch16-dsec-firecracker-microvm-architecture.svg",
            "section": "16.4.1",
            "caption": "DeepSeek DSec 安全沙箱平台架构：单 Cell 160 节点支撑 38 万并发沙箱与四类隔离后端（FnCall / VM 容器 / Firecracker / QEMU VM）",
            "source_refs": ["case-studies/dsec-sandbox-platform.md", "manuscripts/ch16/figure-data.json"],
            "TeachingRole": "展示工业级百万并发 Agent 沙箱如何按任务形态匹配四类虚拟化后端与高密度超卖"
        },
        {
            "figure_id": "图 16-8",
            "number": 8,
            "file": "ch16-virtio-pmem-dax-vs-struct-page-tax.svg",
            "asset": "ch16/ch16-virtio-pmem-dax-vs-struct-page-tax.svg",
            "section": "16.4.2",
            "caption": "virtio-pmem + DAX 消除宿主—客户机双重页缓存（峰值内存 -40.2%）与每 128 GB 设备消耗 2 GB struct page 元数据的权衡（算例 16-4）",
            "source_refs": ["case-studies/dsec-sandbox-platform.md", "manuscripts/ch16/figure-data.json"],
            "TeachingRole": "精确核算 64 B / 4 KiB = 1/64 的 Guest 内核描述符税，解释为何仅对只读镜像层启用 DAX"
        },
        {
            "figure_id": "图 16-9",
            "number": 9,
            "file": "ch16-damon-ept-balloon-reclaim.svg",
            "asset": "ch16/ch16-damon-ept-balloon-reclaim.svg",
            "section": "16.4.2",
            "caption": "DAMON 冷页监控与 virtio-balloon 空闲页报告（order-9 = 2 MiB 块）将时间积分内存（GB·s）降低 21.2% 的联动机制",
            "source_refs": ["case-studies/dsec-sandbox-platform.md", "manuscripts/ch16/figure-data.json"],
            "TeachingRole": "揭示针对大容量可写虚拟磁盘与客户机堆内存的动态冷页回收流水线"
        },
        {
            "figure_id": "图 16-10",
            "number": 10,
            "file": "ch16-sched-idle-and-core-scheduling.svg",
            "asset": "ch16/ch16-sched-idle-and-core-scheduling.svg",
            "section": "16.4.3",
            "caption": "延迟敏感（LS）与尽力而为（BE）任务同机混部下，CFS 默认调度（+45.2%）vs. SCHED_IDLE + SMT 核调度（+17.3%）对比（算例 16-5）",
            "source_refs": ["case-studies/dsec-sandbox-platform.md", "manuscripts/ch16/figure-data.json"],
            "TeachingRole": "证明仅调低优先级不足以消除同物理核 SMT 超线程争用，必须开启 prctl(PR_SCHED_CORE)"
        },
        {
            "figure_id": "图 16-11",
            "number": 11,
            "file": "ch16-ebpf-cgroup-network-allowlist.svg",
            "asset": "ch16/ch16-ebpf-cgroup-network-allowlist.svg",
            "section": "16.4.3",
            "caption": "AppArmor 本地访问控制与按任务动态更新的 cgroup eBPF 网络白名单流水线",
            "source_refs": ["case-studies/dsec-sandbox-platform.md", "manuscripts/ch16/figure-data.json"],
            "TeachingRole": "展示如何在宿主机内核态阻断沙箱对集群内网的横向扫描与评测作弊连接"
        },
        {
            "figure_id": "图 16-12",
            "number": 12,
            "file": "ch16-agent-audit-log-outage-and-drain.svg",
            "asset": "ch16/ch16-agent-audit-log-outage-and-drain.svg",
            "section": "16.5.1",
            "caption": "算例 16-6（I19）：日志后端中断 120 s 积压 46.875 MiB（32 MiB 缓冲于 81.92 s 溢出）及恢复后按 624 KiB/s 净速率在 76.92 s 排空",
            "source_refs": ["case-studies/evaluation-and-agent-records.md", "manuscripts/ch16/figure-data.json"],
            "TeachingRole": "通过排队论净排空方程纠正忽略持续新到达流量的工程误算"
        },
        {
            "figure_id": "图 16-13",
            "number": 13,
            "file": "ch16-idempotent-tool-commit-and-replay.svg",
            "asset": "ch16/ch16-idempotent-tool-commit-and-replay.svg",
            "section": "16.5.1",
            "caption": "基于幂等键（Idempotency Key）与两阶段状态日志的工具调用网关：分离结果重传与外部副作用重做（算例 16-7）",
            "source_refs": ["case-studies/evaluation-and-agent-records.md", "manuscripts/ch16/figure-data.json"],
            "TeachingRole": "确保网络抖动或沙箱抢占恢复时绝不触发重复扣款或脏外部写入，并支持离线确定性回放"
        },
        {
            "figure_id": "图 16-14",
            "number": 14,
            "file": "ch16-telemetry-sampling-vs-billing-ledger.svg",
            "asset": "ch16/ch16-telemetry-sampling-vs-billing-ledger.svg",
            "section": "16.5.1",
            "caption": "可采样性能诊断 Trace（尾部采样）与不可采样计费/幂等键/RL 轨迹版本准入账本的双通道分流架构",
            "source_refs": ["case-studies/evaluation-and-agent-records.md", "manuscripts/ch16/figure-data.json"],
            "TeachingRole": "划清诊断可观测性与严谨财务/训练证据链的系统边界"
        },
        {
            "figure_id": "图 16-15",
            "number": 15,
            "file": "ch16-sixteen-chapter-full-stack-map.svg",
            "asset": "ch16/ch16-sixteen-chapter-full-stack-map.svg",
            "section": "16.5.2",
            "caption": "全书十六章全栈知识坐标系：以“五个第一性追问”贯穿模型、芯片、网络、推理/训练、存储、电热与可信安全",
            "source_refs": ["manuscripts/ch16/figure-data.json"],
            "TeachingRole": "收束全书十六章四大部分的演进脉络与统一分析方法论"
        },
        {
            "figure_id": "图 16-16",
            "number": 16,
            "file": "ch16-ai-infra-architect-master-checklist.svg",
            "asset": "ch16/ch16-ai-infra-architect-master-checklist.svg",
            "section": "16.5.2",
            "caption": "AI 基础设施架构师的六维终极核查表（算力、存力、运力、电力热学、可信安全、全生命周期成本）",
            "source_refs": ["manuscripts/ch16/figure-data.json"],
            "TeachingRole": "为读者提供跨层系统评审与容量规划的随身工程指南"
        },
        {
            "figure_id": "图 16-17",
            "number": 17,
            "file": "ch16-moe-routing-traffic-side-channel-and-padding.svg",
            "asset": "ch16/ch16-moe-routing-traffic-side-channel-and-padding.svg",
            "section": "16.6.1",
            "caption": "密文状态下 MoE All-to-All 专家负载偏斜泄露提示词特征 vs. 恒定桶大小哑元填充（Dummy Padding）防御",
            "source_refs": ["experiments/ch16/16-01/run.py"],
            "TeachingRole": "揭示即使开启 TEE-I/O 载荷加密，专家路由包计数仍构成侧信道，并量化 (1+α) 哑元填充带宽税"
        },
        {
            "figure_id": "图 16-18",
            "number": 18,
            "file": "ch16-verifiable-inference-tee-zkml-opml-triangle.svg",
            "asset": "ch16/ch16-verifiable-inference-tee-zkml-opml-triangle.svg",
            "section": "16.6.2",
            "caption": "可验证推理三路线（硬件 TEE 证明、密码学 zkML、乐观抽检欺诈证明 OPML）在证明开销、信任假设与模型规模上的对比",
            "source_refs": ["experiments/ch16/16-01/run.py"],
            "TeachingRole": "对比 TEE 签名（<5% 开销）、zkML（10³–10⁵x 膨胀）与 OPML 乐观抽检的适用边界"
        },
        {
            "figure_id": "图 16-19",
            "number": 19,
            "file": "ch16-optimistic-verification-slashing-game.svg",
            "asset": "ch16/ch16-optimistic-verification-slashing-game.svg",
            "section": "16.6.2",
            "caption": "乐观抽检可验证推理（OPML）的博弈树：抽检概率 p、造假收益 Δc 与质押罚没金 S_slash 的诚实边界曲线",
            "source_refs": ["experiments/ch16/16-01/run.py"],
            "TeachingRole": "导出激励相容条件 p·S_slash > (1-p)·Δc，证明高额质押金可将验证抽检率压至 10⁻⁵ 量级"
        },
        {
            "figure_id": "图 16-20",
            "number": 20,
            "file": "ch16-janus-nested-virtualization-fast-path.svg",
            "asset": "ch16/ch16-janus-nested-virtualization-fast-path.svg",
            "section": "16.6.3",
            "caption": "传统二级页表嵌套虚拟化（N_guest × N_host 查表放大）vs. JANUS（OSDI '26）扁平化安全沙箱隔离",
            "source_refs": ["research/2026-infra-survey/reading-osdi-2026.md"],
            "TeachingRole": "解析公有云虚拟机内部运行轻量微虚机时的二维 EPT 页表税与 JANUS 跨世界扁平化机制"
        },
        {
            "figure_id": "图 16-21",
            "number": 21,
            "file": "ch16-blowfish-elastic-sandbox-memory-pool.svg",
            "asset": "ch16/ch16-blowfish-elastic-sandbox-memory-pool.svg",
            "section": "16.6.3",
            "caption": "Blowfish（OSDI '26 #50）在解耦内存池上的高密微虚拟机弹性气球回收与 Prism（OSDI '26 #5）跨模型显存协调机制",
            "source_refs": ["research/2026-infra-survey/reading-osdi-2026.md", "experiments/ch16/16-02/run.py"],
            "TeachingRole": "将 CPU 侧跨虚机解耦内存弹性气球（Blowfish）与 GPU 侧跨模型显存气球（Prism）纳入统一池化视图"
        },
        {
            "figure_id": "图 16-22",
            "number": 22,
            "file": "ch16-trusted-ai-infrastructure-defense-in-depth.svg",
            "asset": "ch16/ch16-trusted-ai-infrastructure-defense-in-depth.svg",
            "section": "16.6.3",
            "caption": "可信 AI 基础设施纵深防御全景（硬件 TEE-I/O + 流量填充 + 远程证明/抽检 + Firecracker 微虚机 + 幂等审计账本）",
            "source_refs": ["experiments/ch16/16-01/run.py", "experiments/ch16/16-02/run.py"],
            "TeachingRole": "收束第 16 章硬件机密底座、侧信道与可验证层、高密微虚机沙箱与可恢复审计账本的四层纵深防御"
        },
        {
            "figure_id": "图 16-23",
            "number": 23,
            "file": "ch16-lease-bound-encrypted-weight-distribution.svg",
            "asset": "ch16/ch16-lease-bound-encrypted-weight-distribution.svg",
            "section": "16.6.4",
            "caption": "私有化部署中基于远程证明租约（Lease-Bound KMS）与 GPU TEE 的闭源模型权重防导出硬件绑定流程",
            "source_refs": ["manuscripts/ch16/figure-data.json"],
            "TeachingRole": "展示如何通过周期性远程证明心跳与硬件熔断显存清零保护私有化部署的百亿美金模型权重"
        },
        {
            "figure_id": "图 16-24",
            "number": 24,
            "file": "ch16-fused-cryptographic-logit-watermark-kernel.svg",
            "asset": "ch16/ch16-fused-cryptographic-logit-watermark-kernel.svg",
            "section": "16.6.4",
            "caption": "在 TEE 采样算子内原位融合伪随机锦标赛水印（SynthID/Gumbel Watermark）及统计验证检测边界",
            "source_refs": ["manuscripts/ch16/figure-data.json"],
            "TeachingRole": "将密码学生成水印下沉至 GPU TEE 采样算子内部以阻断宿主机剥离，并推导 Z 分数检测最小词元数"
        },
        {
            "figure_id": "图 16-25",
            "number": 25,
            "file": "ch16-tee-timing-side-channel-traffic-shaping.svg",
            "asset": "ch16/ch16-tee-timing-side-channel-traffic-shaping.svg",
            "section": "16.6.4",
            "caption": "GPU TEE 恒定节拍词元释放（Tick-Locked Pacing）与 64-Token 分桶流量整形防御 PCIe/功耗侧信道",
            "source_refs": ["experiments/ch16/16-03/run.py", "manuscripts/ch16/figure-data.json"],
            "TeachingRole": "量化 25.0 ms 恒定节拍（延迟税 12.11%）与 64-Token 分桶填充（带宽税 39.13%）实现桶内零信息泄露"
        },
        {
            "figure_id": "图 16-26",
            "number": 26,
            "file": "ch16-verifiable-inference-optimistic-and-zkml-proof.svg",
            "asset": "ch16/ch16-verifiable-inference-optimistic-and-zkml-proof.svg",
            "section": "16.6.4",
            "caption": "三级可验证推理架构（TEE 实时签名 + Freivalds 投影抽检 + 触发式单算子 zkML 零知识故障证明）",
            "source_refs": ["experiments/ch16/16-03/run.py", "manuscripts/ch16/figure-data.json"],
            "TeachingRole": "建立 Freivalds 随机投影（2.23% 算力开销）与触发式单算子 zk-Fault-Proof（1.85 s 裁决，提速 24.32×）协同架构"
        }
    ]
    (HERE / 'figure-index.json').write_text(json.dumps(fig_index, ensure_ascii=False, indent=2) + '\n')

    md = HERE.parent / '16-机密 AI 计算、可信执行环境与安全沙箱底座.md'
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
        (HERE / 'math-validation.json').write_text(res.stdout)

        import markdown

        def repl_img(m):
            alt, rel = m.group(1), m.group(2)
            p = (md.parent / rel).with_suffix('.png')
            b64 = base64.b64encode(p.read_bytes()).decode()
            return f'<figure><img alt="{alt}" src="data:image/png;base64,{b64}"><figcaption>{alt}</figcaption></figure>'

        work = re.sub(r'!\[([^\]]*)\]\((ch16/[^)]+)\)', repl_img, text_md)
        body = markdown.markdown(work, extensions=['tables', 'fenced_code', 'footnotes', 'toc'])
        html = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>第 16 章：机密 AI 计算、可信执行环境与安全沙箱底座</title></head><body>{body}</body></html>\n'
        out_html = preview_path(md)
        out_html.write_text(html)

    sources = [HERE / 'build.py', HERE / 'figure-data.json', HERE / 'figure-index.json']
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
    print(f'Built {len(outputs)} figure files for Chapter 16; 0 layout warnings.')


if __name__ == '__main__':
    main()
