# 实验 19-04：级联语音管线 vs. 原生全双工 S2S（Moshi / Qwen-Omni）延迟拆解、RVQ 比特率方程与 Temporal+Depth 双层 Transformer 80 ms 帧预算复算器

本实验复现第 19 章 19.5–19.6 节、算例 19-9、算例 19-10、算例 19-11 与练习 19-17 至 19-21 的核心定量模型与算法：

1. **传统四级串联管线（`VAD -> ASR -> LLM -> TTS`）vs. 原生全双工 S2S 端到端语音延迟（TTFA）账本（算例 19-9）**：
   - 级联流水线：VAD 尾音静默等待 $480.0\ \mathrm{ms}$ + 流式 ASR 识别 $220.0\ \mathrm{ms}$ + LLM 首字与凑句缓冲 $280.0\ \mathrm{ms}$ + TTS 首包合成 $240.0\ \mathrm{ms}$ + 网络与音频 I/O $60.0\ \mathrm{ms} = \mathbf{1,280.0\ \mathrm{ms}}$（超出人类自然对话 $200\ \mathrm{ms}$ 感知阈值 $6.40\times$）；
   - 原生全双工 S2S（$12.5\ \mathrm{Hz}$，$80.0\ \mathrm{ms}$ 帧节拍）：端侧采集与 AEC $25.0\ \mathrm{ms}$ + WebRTC 传输与抖动缓冲 $35.0\ \mathrm{ms}$ + 神经编解码编码 $6.0\ \mathrm{ms}$ + Temporal+Depth 双层整步 CUDA Graph 推理 $11.5\ \mathrm{ms}$ + 流式因果声码器还原 $6.5\ \mathrm{ms}$ + 单帧节拍对齐 $80.0\ \mathrm{ms} = \mathbf{164.0\ \mathrm{ms} < 200.0\ \mathrm{ms}}$（延迟压缩 **$7.80\times$**，彻底消除 VAD 等待）。
2. **残差向量量化（RVQ：`Mimi`）帧率—码本—比特率方程（算例 19-10）**：
   - 原始 $24\ \mathrm{kHz}$ 16-bit 单声道 PCM 码率为 $384.0\ \mathrm{kbps}$；
   - `Mimi` 神经编解码器工作在 $f_s = 12.5\ \mathrm{Hz}$，$Q = 8$ 层码本（$1$ 层语义码本 $+$ $7$ 层声学残差码本），词表 $V_a = 2,048$（$11\ \mathrm{bits/token}$），单流词元速率为 **$100.0\ \mathrm{tokens/s}$**，比特率仅为 **$1.10\ \mathrm{kbps}$**（压缩比高达 **$349.09\times$**）。
3. **Temporal（7B）+ Depth（100M，$Q=8$ 步串行）双层 Transformer 的 CUDA Graph 整步融合（算例 19-11）**：
   - 朴素逐算子动态发射下，Depth 阶段 $48$ 层块（$384$ 个微算子）产生 $13.44\ \mathrm{ms}$ 启动阻塞与 $21.60\ \mathrm{ms}$ 访存耗时（Depth 耗时 $35.04\ \mathrm{ms}$，整帧耗时 $38.54\ \mathrm{ms}$）；
   - 通过整步 CUDA Graph 捕获 + L2 缓存常驻固定，Depth 耗时压至 **$8.00\ \mathrm{ms}$**（提速 $4.38\times$），整帧 GPU 耗时锁定为 $3.50 + 8.00 = \mathbf{11.50\ \mathrm{ms} \ll 80.0\ \mathrm{ms}}$（提速 **$3.35\times$**）。

## 运行方式

```bash
python3 experiments/ch19/19-04/run.py
python3 experiments/ch19/19-04/verify.py
```
