# 实验 24-01：AlphaFold 3 / Boltz-1 两阶段解耦、MMseqs2-GPU 加速比与 Pairformer 分块三角注意力/动态轴向并行核算器

本实验对应第 24 章 24.2 节与算例 24-1、24-2，复算：
1. 传统 CPU MSA 搜索（`2,700.0 s`）导致 GPU 空转 `98.25%` vs. `MMseqs2-GPU`（`1.20 s`，`2,250.0x` 加速）+ 两阶段解耦后端到端延迟降至 `49.20 s`（提速 `55.85x`，Stage 2 GPU 利用率升至 `97.56%`）；
2. $N_{\mathrm{res}} = 4,096$ 超大复合物的对表示张量 $Z$（`4.00 GiB`）、朴素物化 $\mathcal{O}(N_{\mathrm{res}}^3)$ 三角注意力中间张量（`512.00 GiB` OOM）以及分块三角注意力核融合 + 4 卡动态轴向并行（DAP）将峰值注意力工作区压降至 `0.25 GiB/GPU`（降低 `2,048.0x`）。

## 运行与校验
```bash
python3 experiments/ch24/24-01/run.py
python3 experiments/ch24/24-01/verify.py
```
