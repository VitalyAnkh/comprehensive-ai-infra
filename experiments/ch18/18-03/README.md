# 实验 18-03：多轮工具交互 RLVR（SWE-RL）的 CoW 环境快照树分叉与离策略 V-trace 截断仿真

本实验对应第 18 章 18.4.3 节新增小节、算例 18-10 以及练习 18-15、实验 18-16，零依赖复现：
1. **SWE-RL 4×4 环境状态快照分叉（CRIU/OverlayFS + PagedAttention CoW）**：核算 $G=16$ 条叶轨迹下线性独立重跑（$672.0\ \mathrm{s}$ 容器耗时、$131,072$ 前缀词元槽位）与两级 CoW 快照树分叉（$114.0\ \mathrm{s}$ 容器耗时、$5.89\times$ 加速、$20,480$ 前缀词元槽位、净省 $84.38\%$）的精确账本；
2. **异步离策略（Staleness $\Delta k=2$）重要性采样爆炸与 V-trace / TIS 双阈值截断**：对比 10 步连乘 IS 权重（$1.25^{10} \approx 9.3132$，方差放大 $86.74\times$）与 V-trace 轨迹截断（$c_m = \min(1.0, \tau_m) = 1.0$、$\rho_m = \min(1.2, \tau_m) = 1.2$）的递归价值修正收敛特性，以及双侧观测掩码下策略梯度对环境观察词元的零梯度隔离。

## 运行与验证

```bash
python3 experiments/ch18/18-03/run.py
python3 experiments/ch18/18-03/verify.py
```
