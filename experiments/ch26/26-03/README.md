# 实验 26-03：Coconut 连续潜空间思维链回环与 Mixture-of-Depths (MoD) 流紧缩

本实验对应第 26 章 26.4 节算例 26-5 与算例 26-6，验证：
1. **Coconut（Chain of Continuous Thought）连续潜空间推理零拷贝回环**：
   - 每个连续潜向量压缩替代 $c = 3$ 个显式离散思维链词元，且在潜空间思考阶段完全跳过 LM Head 词表投影与离散采样同步（每步耗时从 `10.00 ms` 降至 `9.24 ms`），将 300 词元离散推理链压缩为 100 步潜向量演化，端到端推理延迟降低 `69.2%`（提速 `3.25x`）。
2. **Mixture-of-Depths (MoD) 词元级动态跳层与流紧缩（Stream Compaction）**：
   - 在层保留率 $\rho = 0.25$ 下，若直接在原序列位置靠掩码跳过计算，因 Warp 分支发散与非合并访存，有效硬件利用率退化至 `58.1%`；
   - 执行片上前缀和流紧缩（Gather 活跃词元为紧凑连续张量 -> 满载 GEMM -> Scatter 回原位置）后，有效 Warp 满载率恢复至 `96.4%`，实测单层净加速 `3.32x`。

## 运行与验证

```bash
python3 experiments/ch26/26-03/run.py
python3 experiments/ch26/26-03/verify.py
```
