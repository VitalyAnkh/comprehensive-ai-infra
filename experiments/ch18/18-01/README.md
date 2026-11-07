# 实验 18-01：RLVR（PPO vs. GRPO）显存账本、Weave/RobustRL 容错、RollPacker 长度偏差修正与 DistRS 条件剩余时间模拟器

本实验复现第 18 章 18.1–18.4 节与练习 18-1、18-4、18-6、18-9、18-12 的核心定量模型：

1. **PPO 四模型 vs. GRPO 无 Critic 架构的训练态显存账本（算例 18-1）**：
   - 以 $\Psi = 32\mathrm{B}$ 参数模型为例，单模型完整训练态（BF16 权重 $2\Psi$ + 梯度 $2\Psi$ + FP32 主权重与 AdamW 一二阶动量 $12\Psi$）占 $16\Psi = 512\ \mathrm{GB}$，纯推理态占 $2\Psi = 64\ \mathrm{GB}$；
   - PPO 需驻留 Actor（$16\Psi$）+ Critic（$16\Psi$）+ Reference（$2\Psi$）+ Reward（$2\Psi$）共 **$36\Psi = 1,152\ \mathrm{GB}$**；GRPO 消除 Critic 并以规则验证器替代神经网络 RM，仅需 Actor（$16\Psi$）+ Reference（$2\Psi$）共 **$18\Psi = 576\ \mathrm{GB}$**（模型状态显存直降 **$50.0\%$**）。
2. **`Weave` 互补作业池化与 `RobustRL` 分角色容错（算例 18-3 & 18-7，锁定 [`case-studies/rl-scheduling-and-recovery.md`](../../../case-studies/rl-scheduling-and-recovery.md)）**：
   - 作业 A（`360 s` Rollout + `120 s` 训练）与作业 B（`120 s` Rollout + `360 s` 训练）在 `480 s` 周期内互补错峰，共享 `216 H20` 与 `152 H800`；
   - `RobustRL`：单轮 `300 s` Rollout + `60 s` 更新，在更新 `30 s` 处发生故障（检测 `20 s`、恢复 `80 s`），若整步重做需 `300 + 30 + 20 + 80 + 300 + 60 = 790 s`，而保留已完成 Rollout 轨迹仅重算更新只需 `300 + 30 + 20 + 80 + 60 = 490 s`（净省 **`300 s`**）。
3. **`RollPacker` 投机采样、双槽位长度偏差与全局总词元归一化（算例 18-5 & 18-6，锁定 [`case-studies/rollout-tail-and-sampling.md`](../../../case-studies/rollout-tail-and-sampling.md)）**：
   - 投机扩容：`160 × 10 = 1,600` 候选轨迹 vs. 目标 `128 × 8 = 1,024`（**$+56.25\%$** 裕度），双槽位竞争下短回复占比从 $1/2$ 漂移至 **$3/4$**（长回复压缩至 **$1/4$**）；
   - 流式训练重叠（8 卡：10s 完成一半，3s 迁移使剩余 Rollout 在 43s 结束，早到一半在 4 卡用 16s 于 29s 算完，后一半在 8 卡用 8s + 4s 更新）将周期从 `40 + 16 + 4 = 60 s` 压至 `max(43, 29) + 8 + 4 = 55 s`；不均等微批（Microbatch 1 含 1 项 `{1.0}`，Microbatch 2 含 3 项 `{3.0, 3.0, 3.0}`）下，按微批均值直接平均得到 `(1 + 3)/2 = 2.0`（发生偏置，放大微批 1 权重），而按全局有效总样本/总词元数加权归一化得到无偏真值 `(1 + 9)/4 = 2.5`。
4. **`DistRS` 条件剩余服务时间 $\mathbb{E}[S - e \mid S > e]$ 与 `216,000` 词元沉没工作量边界（算例 18-8，锁定 [`case-studies/reward-deadlines-and-feedback.md`](../../../case-studies/reward-deadlines-and-feedback.md)）**：
   - 9 个 $1\ \mathrm{s}$ 任务 + 1 个 $100\ \mathrm{s}$ 任务的无条件均值为 $10.9\ \mathrm{s}$；当任务已运行 $e = 10\ \mathrm{s}$ 时，朴素估计误判剩余 $0.9\ \mathrm{s}$，而真实条件期望剩余时间为 **$90.0\ \mathrm{s}$**；
   - 6 个存活长尾验证任务在 $60\ \mathrm{s}$ 额外预算、$600\ \mathrm{tok/s}$ 下对应的沉没生成预算为 **`216,000` tokens**。

## 运行方式

```bash
python3 experiments/ch18/18-01/run.py
python3 experiments/ch18/18-01/verify.py
```
