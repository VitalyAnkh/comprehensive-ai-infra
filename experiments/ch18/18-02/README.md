# 实验 18-02：Firecracker 微虚机 12 ms CoW 快照分叉、MCTS 环境回溯显存账本与审计遥测断网排空（81.92 s 触顶 / 76.92 s 排空）复算器

本实验复现第 18 章 18.3–18.4 节、算例 18-3、算例 18-4、算例 18-5 与练习 18-5 至 18-10 的核心定量模型与算法：

1. **Firecracker 微虚机快照克隆（`MAP_PRIVATE` CoW + `userfaultfd`）与 MCTS 树搜索同步回溯（算例 18-3）**：
   - 传统容器/虚机冷启动耗时 $T_{\mathrm{cold}} = \mathbf{1,500.0\ \mathrm{ms}}$（基础镜像常驻内存 $512.0\ \mathrm{MiB}$）；
   - 基于预热快照的写时复制（CoW）克隆耗时仅为 $3.2\ \mathrm{ms}$（VMM 元数据克隆）$+ 8.8\ \mathrm{ms}$（$5.50\ \mathrm{MiB}$ 工作集缺页预取）$= \mathbf{12.0\ \mathrm{ms}}$（启动提速 **$125.0\times$**）；
   - 在 $B_{\mathrm{tree}} = 16$ 分支的 MCTS 环境搜索中，每分支私有脏页仅占 $24.0\ \mathrm{MiB}$，总内存占用为 $512 + 16 \times 24 = \mathbf{896.0\ \mathrm{MiB}}$（相比全量复制 $8,192.0\ \mathrm{MiB}$ 节省 **$89.06\%$**），且破坏性分支的“KV 块表 $+$ 沙箱快照”同步回滚耗时仅 **$8.50\ \mathrm{ms} < 10.0\ \mathrm{ms}$**。
2. **审计遥测日志断网积压溢出与恢复排空账本（算例 18-4，锚定 [`case-studies/evaluation-and-agent-records.md`](../../../case-studies/evaluation-and-agent-records.md)）**：
   - 审计遥测产生速率 $r_{\mathrm{gen}} = 400\ \mathrm{KiB/s}$，本地内存缓冲上限 $B_{\mathrm{buf}} = 32\ \mathrm{MiB} = 32,768\ \mathrm{KiB}$，断网后缓冲触顶时间为 $t_{\mathrm{full}} = 32,768 / 400 = \mathbf{81.92\ \mathrm{s}}$；
   - 在 $T_{\mathrm{out}} = 120.0\ \mathrm{s}$ 断网期间累计产生 $120 \times 400 / 1,024 = \mathbf{46.875\ \mathrm{MiB}}$ 积压（其中 $14.875\ \mathrm{MiB}$ 溢写至本地 NVMe 事务外发箱 Outbox）；
   - 网络恢复后以 $r_{\mathrm{up}} = 1.0\ \mathrm{MiB/s} = 1,024\ \mathrm{KiB/s}$ 并发排空，扣除持续产生的 $400\ \mathrm{KiB/s}$ 新日志后，净排空速率为 $624\ \mathrm{KiB/s}$，完全排空积压耗时为 $48,000 / 624 = \mathbf{76.92\ \mathrm{s}}$。
3. **MCP 网关动态工具检索（Dynamic Tool RAG）与 `HippoRAG 2` 长期记忆压缩（算例 18-5）**：
   - 从 $1,000$ 个挂载工具（全量 Schema $180,000$ 词元）中按需检索 Top-$8$ 工具（仅 $1,440$ 词元），消除 **$99.20\%$** 的工具 Schema 词元税（压缩 **$125.0\times$**）。

## 运行方式

```bash
python3 experiments/ch18/18-02/run.py
python3 experiments/ch18/18-02/verify.py
```
