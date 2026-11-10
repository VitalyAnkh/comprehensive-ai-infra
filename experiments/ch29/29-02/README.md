# 实验 29-02：DeepEP 双模式（NVLink+RDMA / 纯 IBGDA 零 SM 占用）与 DualPipe 双向流水线零气泡验证器

本实验复现第 29 章 29.2 节的核心量化结论：
1. **DeepEP**：Normal 模式（NVLink-RDMA 混合双跳）仅用 `24` 个通信 SM 即达 **`153.0 GB/s`** 转发吞吐；Low-Latency 纯 `IBGDA` 模式在 128 专家下实现 **`163.0 us` Dispatch** 与 **`194.0 us` Combine**，且通过后台 Hook 轮询实现 **`0` 个 SM 占用**；
2. **DualPipe**：在 $PP = 16$、双向微批 $2M = 32$ 下，通过将反向传播拆分为细粒度 `B_I`（对输入求导）与 `B_W`（对权重求导），实现 **`0.0%` 暴露跨节点 All-to-All 通信停顿**。

## 运行与校验
```bash
python3 experiments/ch29/29-02/run.py
python3 experiments/ch29/29-02/verify.py
```
