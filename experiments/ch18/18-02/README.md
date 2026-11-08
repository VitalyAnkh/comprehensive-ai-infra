# 实验 18-02：DistRS 216,000 词元沉没成本裁剪、跨地域异步 Rollout 14,745.6x 带宽压缩与 PRM CoW 树搜索显存复算器

本实验复现第 18 章 18.4.1–18.4.3 节、算例 18-8、算例 18-9 与练习 18-11、18-12、18-13、实验 18-14 的核心定量模型：

1. **`DistRS`（NSDI '26）重尾验证条件剩余时间与 `216,000` 词元沉没工作量裁剪阈值（算例 18-8 & 练习 18-10）**：
   - 在双峰重尾验证分布（9 个 $1\ \mathrm{s}$ 任务 + 1 个 $100\ \mathrm{s}$ 任务，无条件均值 $\mathbb{E}[S] = 10.9\ \mathrm{s}$）中，运行至 $e = 10\ \mathrm{s}$ 时朴素估计仅剩 **$0.9\ \mathrm{s}$**，而真实条件剩余时间 $\mathbb{E}[S - 10 \mid S > 10]$ 高达 **$90.0\ \mathrm{s}$**（相差 **$100.0\times$**）；
   - 6 个超期并发任务按 $60\ \mathrm{s} \times 600\ \mathrm{tok/s}$ 累积的沉没词元工作量边界精确为 **$216,000$ 词元**。
2. **跨地域异构算力异步 RL（`Prime-RL` / `OpenDiLoCo-RL`）极窄带宽回传与双级过滤（算例 18-9 & 练习 18-13）**：
   - 单步收集 $1{,}024$ 条平均长度 $16{,}384$ 词元的轨迹（合计 **$16,777,216$ 词元**）：若回传 36 层 KV 缓存（$147{,}456\ \mathrm{B/tok}$），数据量高达 **$2,473.9\ \mathrm{GB}$**，在 $10\ \mathrm{Gbps}$ 跨机房专线上阻塞 **$1,979.1\ \mathrm{s}$**；
   - 改为仅回传 `10 B/tok` 紧凑三元组 `(token_id: int32, behavior_logprob: float32, version: uint16)` 后，全批上行体积骤降至 **$167.77\ \mathrm{MB} = 160.0\ \mathrm{MiB}$**，带宽压缩比达 **$14,745.6\times$**，$10\ \mathrm{Gbps}$ 专线仅需 **$134.2\ \mathrm{ms}$** 即可传完；
   - 配合每 $2$ 步异步广播 FP8 差分权重（$8.0\ \mathrm{GB}$，耗时 **$6.4\ \mathrm{s}$** 完全隐藏在 $300\ \mathrm{s}$ Rollout 期）及陈旧度 $\tau \le 2$、序列几何均值重要性比率 $\bar{r}_{\mathrm{seq}} \in [0.5, 2.0]$ 双级过滤，保证跨洲异构采样稳定收敛。
3. **推理时扩展 PagedAttention 写时复制（CoW）树搜索显存节省与投机越障（练习 18-11 & 18-12）**：
   - 根前缀 $2{,}048$ 词元派生 $8 \times 4 = 32$ 条叶分支（每层步长 $256$ 词元）：独立深拷贝需 **$81,920$ 词元槽位**（$11.25\ \mathrm{GiB}$），而 PagedAttention CoW 树共享仅需 **$12,288$ 词元槽位**（$1.6875\ \mathrm{GiB}$），节省 **$85.0\%$** 显存；
   - `Breaking the Reward Barrier`（OSDI '26）投机思维树探索（$D=20, t_{\mathrm{gen}}=1.2\ \mathrm{s}, t_{\mathrm{prm}}=0.8\ \mathrm{s}$, 命中率 $80\%$）将串行 $40.0\ \mathrm{s}$ 延迟压降至 **$29.6\ \mathrm{s}$**。

## 运行方式

```bash
python3 experiments/ch18/18-02/run.py
python3 experiments/ch18/18-02/verify.py
```
