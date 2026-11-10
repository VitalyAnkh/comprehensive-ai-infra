# 实验 19-01：多模态视觉特征膨胀（Qwen3-VL-4B）、3D 视频 DiT 序列并行（2D USP）与 3D VAE 分块显存复算器

本实验复现第 19 章 17.1–17.4 节与练习 17-1、17-2、17-4、17-9、17-12 的核心定量模型：

1. **`Qwen3-VL-4B` 视觉编码器特征（EC）与主干 KV 缓存的 $7.2\times$ 状态体积膨胀（算例 17-1，锁定 [`case-studies/multimodal-stage-placement.md`](../../../case-studies/multimodal-stage-placement.md)）**：
   - 单张 $640 \times 640$ 图像经 Patch $16$ 与 $2\times 2$ 空间合并生成 $400$ 个视觉词元；
   - 包含 3 个 DeepStack 层（`#5, #11, #17`）的视觉编码器输出张量 `[400, 10240]`（BF16）占 **$8,192,000\ \mathrm{B} = 7.8125\ \mathrm{MiB}$**；
   - 进入 36 层 LLM 主干（8 KV 头，头维 128，BF16）预填充后，KV 缓存占 $400 \times 147,456\ \mathrm{B} = 58,982,400\ \mathrm{B} =$ **$56.25\ \mathrm{MiB}$**，状态体积精确放大 **$7.2\times$**。
2. **3D 视频 DiT（$489,600$ Tokens）的 2D 统一序列并行（Ulysses $\times$ Ring USP）通信量账本（算例 17-2 & 17-3）**：
   - $10\ \mathrm{s} \times 24\ \mathrm{fps}$、$1080\mathrm{p}$（$1920 \times 1080$）视频经 $4\times 8\times 8$ 3D VAE 与 $1\times 2\times 2$ Patchify 后产生 $60 \times 68 \times 120 =$ **$489,600$ 个时空词元**；
   - 在 $P = 64$ 卡、隐藏维 $d = 4,096$（BF16）下，对比纯 Ulysses（单层每卡 All-to-All $\frac{4Nd}{P} = 239.0625\ \mathrm{MiB}$）、纯 Ring Attention（全环 63 步每卡 $\frac{2(P-1)Nd}{P} = 7,530.46875\ \mathrm{MiB}$）与 **2D USP（节点内 $P_U=8$ Ulysses + 跨节点 $P_R=8$ Ring，跨机 RDMA 流量降至 $836.71875\ \mathrm{MiB}$，压缩 $9.0\times$）**。
3. **3D 因果 VAE 时空分块解码（Tiled VAE）与 TeaCache 时间步残差复用加速比（算例 17-4 & 17-5）**：
   - 未分块全分辨率 3D VAE 中间特征图峰值达 $118.65\ \mathrm{GiB}$，启用 $4\times 4$ 空间分块 + 16 帧滑窗（含 $12.5\%$ Halo 重叠）后单缓冲压至 $0.6257\ \mathrm{GiB}$（双缓冲峰值约 $1.25\ \mathrm{GiB}$）；
   - 50 步 DiT 去噪中通过 TeaCache 跳过 28 步冗余重算，端到端耗时从 $63.60\ \mathrm{s}$ 降至 $30.00\ \mathrm{s}$（**$2.12\times$ 加速**）。

## 运行方式

```bash
python3 experiments/ch19/19-01/run.py
python3 experiments/ch19/19-01/verify.py
```
