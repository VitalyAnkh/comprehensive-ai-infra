# 实验 15-04：eBPF + Dynolog/Kineto 三级按需剖析开销对比与 Blink 采样滑移校准仿真器

本实验复现第 15 章 15.4.2 节与 15.6.1 节、算例 15-8、练习 15-15 的全部锁定定量账本，并与 `calculations/results/ch15-fleet-reliability-profiler.json` 严格同源：

1. **16,384 卡集群常驻全量 CUPTI 追踪 vs. 三级分层遥测金字塔（`eBPF Strobelight` L0 + `Dynolog/Kineto` L1，算例 15-8）**：
   - **全量常驻 CUPTI 追踪**：吞吐退化 **$8.0\%$**，在 $16{,}384$ 卡集群上等效损耗 **$1{,}310.72$ 张 GPU**（约 **$1{,}311$ 张**），按单卡 $1.0038\ \mathrm{MB/s}$ 产率每日产生 **$1.42\ \mathrm{PB}$**（$1{,}420\ \mathrm{TB}$）追踪日志；
   - **三级按需遥测架构**：内核态 `eBPF`（`uprobes` + `Strobelight` PMU）L0 全机队常驻损耗仅 **$0.35\%$**（日产 $1.70\ \mathrm{TB}$ 聚合栈摘要），配合守护进程 `Dynolog` 在步耗时偏离中位数 $> 3\sigma$ 时按需唤醒 `PyTorch Kineto` L1 抓取 $5$ 步微秒级轨迹（机队分摊占空比 $0.875\%$，分摊损耗 **$0.07\%$**，日增 $0.10\ \mathrm{TB}$ 轨迹）；
   - **端到端收益**：总算力损耗压缩至 **$0.42\%$**（仅占 **$68.81$ 张等效 GPU**，净挽回 **$1{,}241.91$ 张 GPU**），日追踪存储降至 **$1.80\ \mathrm{TB}$**，存储体积缩减 **$788.9\times$**。
2. **扁平微秒级负载上的硬件 PMU 采样滑移（Sampling Skid）逆矩阵校准（`Blink` OSDI 2026，练习 15-15）**：
   - 对三段式微秒级函数链 `[tokenize_dispatch, launch_stub, post_hook]`，在滑移转移矩阵 $M$ 与观测计数 $y = [300, 500, 200]^T$ 下，逆投影精确还原真实耗时占比 **$[50.0\%, 40.0\%, 10.0\%]$**，消除传统采样器把 $20\%$ 开销转嫁给下游函数的归因失真。

## 运行与验证

```bash
python3 experiments/ch15/15-04/run.py
python3 experiments/ch15/15-04/verify.py
```
