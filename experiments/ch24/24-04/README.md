# 实验 24-04：基于 Nsight 硬件剖析反馈的多轮 RLVR 内核自动合成（KernelBench / Kevin）进化收敛模拟器

本实验复现第 24 章 24.5 节的核心量化结论：
1. 在 `KernelBench` 250 个代表性算子基准上，单轮零样本生成（Zero-Shot LLM）编译且数值正确率仅为 `34.0%`，超越 `torch.compile` 比例仅 `18.0%`；
2. 引入 $T_{\mathrm{iter}} = 8$ 轮 Nsight Compute 硬件计数器反馈后，正确率跃升至 **`94.8%`**，在复杂长尾融合算子上相对 PyTorch Eager 取得平均 **`3.42x`** 加速、相对 `torch.compile` 取得 **`1.58x`** 加速，并通过寄存器级持久化融合省去 **`38.4 GB/s`** 中间激活落盘流量。

## 运行与校验
```bash
python3 experiments/ch24/24-04/run.py
python3 experiments/ch24/24-04/verify.py
```
