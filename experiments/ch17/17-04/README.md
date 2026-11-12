# 实验 17-04：长序列变长装箱（`cu_seqlens`）、二次计算量均衡与偏好对齐显存账本

本实验对应第 17 章 17.5.1 节、算例 17-11 与练习 17-17，复现：
1. 重尾长上下文指令微调语料（最大序列长度 $L_{\max}=65{,}536$、均值 $\bar{L}=4{,}200$）下，`Pad-to-Max`、未均衡 `cu_seqlens` 变长打包与 `cu_seqlens` + 二次计算量（$\sum_i L_i^2$）ZigZag Ring 负载均衡三者的 MFU 演进（`13.6%` $\to$ `41.2%` $\to$ `56.4%`）及 **`4.15x`** 有效吞吐加速；
2. 跨文档交叉注意力污染阻断：验证分段 `cu_seqlens` 前缀和索引与重置式 RoPE `position_ids` 严格消除跨样本泄漏；
3. 偏好对齐系统（DPO 需共驻参考策略 vs SimPO / ORPO 无参考策略）的显存与前向计算通道对比账本。

## 运行与校验

```bash
python3 experiments/ch17/17-04/run.py
python3 experiments/ch17/17-04/verify.py
```
