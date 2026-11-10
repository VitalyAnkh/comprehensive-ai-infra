# 实验 23-05：Byte Latent Transformer (BLT) 字节熵动态分块与 Titans/TTT 分块对偶扫描

本实验对应第 23 章 26.2 与 26.3 节算例 26-3 与算例 26-4，验证：
1. **Byte Latent Transformer (BLT) 告别 256k 词表与字节条件熵动态分块（$\tau_H = 1.35\ \mathrm{bits}$）**：
   - 移除 $V=256,000, d=4,096$ 的输入 Embedding 与输出 LM Head 矩阵，节省 `4.194 GB` 常驻显存及每词元 `2.10 GFLOPs` 词表投影计算；
   - 在高可预测性代码/模板段将平均 Patch 跨度从 `4.5 B` 拉长至 `12.0 B`，使骨干潜空间 Transformer 序列长度缩短 `2.67x`，端到端推理 FLOPs 降低 `52.4%`。
2. **测试时记忆网络（Google Titans / TTT-Linear）分块对偶形式（`ttt_chunk_scan`，$b=64$）**：
   - 逐词元串行更新 $128 \times 128$ 快速权重矩阵：算术强度仅 `1.5 FLOP/B`（吞吐 `42.0k tok/s`）；
   - 分块大小 $b=64$ 矩阵对偶形式：算术强度跃升至 `48.0 FLOP/B`，Tensor Core 吞吐达 `1,033.2k tok/s`（提速 `24.60x`），且在 `200 万`（$2 \times 10^6$）上下文下快速权重显存恒定为 `16.0 MB/layer`（$\mathcal{O}(1)$ 零膨胀）。

## 运行与验证

```bash
python3 experiments/ch23/23-05/run.py
python3 experiments/ch23/23-05/verify.py
```
