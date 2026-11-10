# 第 14 章阅读与复现笔记：模型与系统硬件协同设计

1. **DeepSeek V4 $\to$ V4.1 Flash 跨层协同设计专题**：
   - 参阅仓库研究报告 [`research/deepseek-v41-codesign-2026-09-10/README.md`](../../research/deepseek-v41-codesign-2026-09-10/README.md)，复核 `8K -> 1M` 上下文下驻留容量（恒定缩小 `3.95x`）与单步 Decode 读取量（从 `1.10x` 演进至 `2.51x`）的双轨演进模型，以及 `Full -> 16,384 候选池 -> Reindex -> Reuse` 四阶段注意力执行流水线。
2. **$24\times 7$ 数字员工跨代 GPU 成本与性能归因审计**：
   - 参阅案例分析 [`case-studies/single-agent-serving-cost.md`](../../case-studies/single-agent-serving-cost.md) 及可复现审计报告 [`experiments/ch13/13-06/single-agent-serving/OPTIMIZATION-AUDIT.md`](../../experiments/ch13/13-06/single-agent-serving/OPTIMIZATION-AUDIT.md)，重点理解如何剥离 `13.95 ms/tok` 的逐层固定调度开销与 `9.18–9.26 ms/tok` 的频繁重建假设。
3. **OpenTallas 专用推理 ASIC 四资源账本与晶圆级跳数下界**：
   - 参阅案例分析 [`case-studies/opentallas.md`](../../case-studies/opentallas.md)，复核 Case B 单步 `124.221 μs`（`8,050 tok/s`）的四资源时间线与 `14.568 μs` 辅助算子余量，以及 Case D 晶圆级 `15 跳 × 100 ns × 120 层 = 180 μs` 的通信直径反超推导。
