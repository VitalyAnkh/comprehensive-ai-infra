# 实验 16-01：PipeLLM 三缓冲加密流水线、MoE 流量侧信道填充与 OPML 可验证推理博弈仿真

本实验复现第 16 章 16.2–16.3 节与 16.6.1–16.6.2 节的核心机密计算与可验证推理账本：

1. **Vera Rubin 原生 TEE-I/O vs. 传统软件弹跳缓冲（算例 16-1 & 练习 16-1）**：
   - $64\ \mathrm{MiB}$ 激活块从 $8.8\ \mathrm{ms}$ 压缩至 $0.4\ \mathrm{ms}$（$22.0\times$ 加速）；
   - $128\ \mathrm{MiB}$ 激活块从 $27.36\ \mathrm{ms}$ 压缩至 $1.40\ \mathrm{ms}$（$19.54\times$ 加速）。
2. **PipeLLM 五阶段三缓冲机密卸载流水线与 AES-CTR 投机预加密（算例 16-2, 16-3 & 练习 16-2, 16-3）**：
   - 单块 $64\ \mathrm{MiB}$、五阶段耗时 $[2.0, 0.2, 2.5, 0.2, 2.0]\ \mathrm{ms}$：连续处理 $M=20$ 块时，串行总耗时 $138.0\ \mathrm{ms}$，三缓冲（$192\ \mathrm{MiB}$）流水线耗时 $62.0\ \mathrm{ms}$（加速比 $2.2258\times$）；
   - 投机预加密命中节省 $\Delta t_{\mathrm{hit}} = 2.4\ \mathrm{ms}$、未命中惩罚 $\Delta t_{\mathrm{miss}} = 1.6\ \mathrm{ms}$ 时，盈亏平衡命中率为 $p^* = 40.0\%$。
3. **MoE 通信流量侧信道恒定速率填充与 OPML 乐观抽检博弈（练习 16-11 & 练习 16-12）**：
   - 在 $P_{\mathrm{EP}} = 16$ 卡专家并行中，为防御专家流量侧信道将每卡接收槽统一填充至热专家上界 $\mu(1 + \alpha)$，额外 All-to-All 带宽税恰为 $\alpha$（当 $\alpha = 0.35$ 时带宽放大 $1.35\times$）；
   - 在乐观抽样验证（OPML）中，单次偷换小模型节省成本 $\Delta c = 0.005\ \mathrm{USD}$、质押罚没金 $S_{\mathrm{slash}} = 1,000\ \mathrm{USD}$ 时，由诚实边界 $p \cdot S_{\mathrm{slash}} > (1-p)\Delta c$ 得最小随机抽检率 $p_{\min} = \frac{\Delta c}{S_{\mathrm{slash}} + \Delta c} \approx 4.999975 \times 10^{-6}$（约 $0.0005\%$）。

## 运行方式

```bash
python3 experiments/ch16/16-01/run.py
python3 experiments/ch16/16-01/verify.py
```
