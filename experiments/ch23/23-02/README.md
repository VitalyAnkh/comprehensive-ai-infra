# 实验 23-02：2023–2026 逐级条件比率瀑布、FarSkip-Collective 跳层通信掩盖与 GQA 复制税模拟器

本实验对应第 23 章 `练习 14-11`、`练习 14-13`、`练习 14-14` 与 `实验 14-15`，仅依赖 Python 3 标准库即可独立运行：

1. **逐级条件比率连乘（Telescoping Waterfall）**：严格对齐第 14.8.2 节与 `research/token-cost-2023-2026/calculations.json` 的四个锁定状态 $S_0 \to S_1 \to S_2 \to S_3$（$70/3.35 \approx 20.90\ \mathrm{ms} \to 8/3.35 \approx 2.39\ \mathrm{ms} \to 4/3.35 \approx 1.19\ \mathrm{ms} \to 4/4.80 \approx 0.83\ \mathrm{ms}$；逐级条件比率 $r_1 = 8.75\times$、$r_2 = 2.00\times$、$r_3 = 4.80/3.35 \approx 1.4328\times$；精确连乘加速比 $25.0746\times \approx 25.1\times$）及加入主机固定调度开销 $t_{\mathrm{host}}$ 后的端到端加速比稀释；
2. **FarSkip-Collective 跨层跳步掩盖模拟**：复算 60 层 MoE 模型在 $m=0$（逐层同步阻塞，总耗时 $120.0\ \mathrm{ms}$）与 $m=1$（跨层延后消费，前 59 层通信完全掩盖、末层暴露 $0.80\ \mathrm{ms}$，总耗时 $72.80\ \mathrm{ms}$，加速比 $1.6484\times$）下的逐层暴露通信时间；
3. **GQA 键值头复制因子核查**：对比 $n_{\mathrm{kv}}=8$ 与 $n_{\mathrm{kv}}=2$ 在 $\mathrm{TP}=8$ 部署下的复制因子（$1\times$ vs. $4\times$）与 8 卡物理 KV 显存占用（同为 $16.0\ \mathrm{GiB}$）。

## 运行与校验

```bash
python3 experiments/ch23/23-02/run.py
python3 experiments/ch23/23-02/verify.py
```
