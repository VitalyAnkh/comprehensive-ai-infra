# 实验 15-01：十万卡集群故障率（Young-Daly 最优检查点周期）、热备池泊松排队模型、ABFT 矩阵乘 SDC 拦截与评测失真反例复算器

本实验复现第 15 章 20.1–20.4 节与练习 20-1、20-2、20-4、20-7、20-8、20-10 的核心定量模型：

1. **Meta Llama 3 405B `16,384` 卡 `54` 天 `419` 次故障账本外推至 `100,000` 卡及 Young-Daly 最优检查点周期（算例 20-1）**：
   - `16,384` 卡 `54` 天（$1,296\ \mathrm{h}$）共发生 `419` 次非计划中断，集群 $\mathrm{MTBF}_{16\mathrm{K}} = 1,296 / 419 \approx \mathbf{3.09\ \mathrm{h}}$；线性外推至 $100,000$ 卡时，$\mathrm{MTBF}_{100\mathrm{K}} \approx \mathbf{0.5068\ \mathrm{h}} = \mathbf{30.41\ \mathrm{min}}$；
   - 按 Young-Daly 公式 $\tau_{\mathrm{opt}} = \sqrt{2 C_{\mathrm{ckpt}} M}$（取 $M = 1,800\ \mathrm{s}$）：传统落盘检查点（$C_{\mathrm{ckpt}} = 180\ \mathrm{s}$，重启恢复 $T_{\mathrm{rec}} = 300\ \mathrm{s}$）的最优间隔为 **$805.0\ \mathrm{s}$**，有效训练时间比（ETTR）跌至 **$63.22\%$**；采用内存异步检查点（$C_{\mathrm{ckpt}} = 5\ \mathrm{s}$）配合 `TrainMover` 亚秒/秒级无重启接管（$T_{\mathrm{rec}} = 5\ \mathrm{s}$），最优间隔为 **$134.16\ \mathrm{s}$**，十万卡 ETTR 提升至 **$92.34\%$**。
2. **热备节点池（Hot-Spare Pool）泊松维修队列容量规划（MLSys '26 #33，算例 20-2）**：
   - 在节点故障到达率 $\lambda = 2.0\ \mathrm{h}^{-1}$、平均更换修复时间 $\mathrm{MTTR}_{\mathrm{repair}} = 12.0\ \mathrm{h}$ 下，在修节点稳态均值为 $L = \lambda \cdot \mathrm{MTTR}_{\mathrm{repair}} = \mathbf{24.0}$；以 $\ge 99.9\%$ 概率保证训练零降级所需的最小热备节点数为 **$S_{\min} = 40\ \text{个节点}$**。
3. **ABFT 矩阵乘代数不变量校验和（$\mathbf{1}^T(AB) = (\mathbf{1}^T A)B$）开销与静默数据损坏（SDC）单比特翻转定位（算例 20-3）**：
   - 对 $4,096 \times 4,096$ 矩阵乘，ABFT 外围行列校验和仅增加 **$0.0732\% < 0.5\%$** 的 FLOPs 开销；注入 $(2, 3)$ 元素 $+64.0$ 静默翻转后，凭行残差与列残差非零坐标 $100\%$ 精确定位 $(2, 3)$ 并原地修复。
4. **慢节点到达偏斜与基准评测失真反例（锁定 [`case-studies/collective-paths-and-diagnosis.md`](../../../case-studies/collective-paths-and-diagnosis.md) 与 [`case-studies/optimization-evaluation-and-deployment.md`](../../../case-studies/optimization-evaluation-and-deployment.md)，算例 20-5 & 20-6）**：
   - 集合通信到达偏斜：4 个 Rank 在 `[0, 0, 0, 2.0] ms` 就绪 + `0.4 ms` 交换，带宽翻倍仅省 **$0.2\ \mathrm{ms}$**，消除慢节点就绪偏斜省 **$2.0\ \mathrm{ms}$**；
   - `FlashInfer-Bench` 加速比平均悖论：候选 A `[10, 200] μs` 算术平均加速比 **$5.25\times$** 却在等频下耗时 **$210\ \mu\mathrm{s} > 200\ \mu\mathrm{s}$**，候选 B `[50, 50] μs` 平均加速比 **$2.0\times$** 耗时 **$100\ \mu\mathrm{s}$**（翻转阈值 $p^* = 15/19 \approx 0.7895$）；`AABB` vs. `ABBA` 消除 `[0, 10, 20, 30] μs` 线性时漂。

## 运行方式

```bash
python3 experiments/ch15/15-01/run.py
python3 experiments/ch15/15-01/verify.py
```
