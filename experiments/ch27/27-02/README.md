# 实验 27-02：高频量化微秒级 AI（内核旁路 + 持久化轮询 Megakernel + isolcpus）16.5 us 端到端尾延迟模拟器

本实验对应第 27 章 27.3 节（算例 27-4），对比传统 Linux Socket + PyTorch 栈（中位数 `78.0 us`，P99.99 `420.0 us`）与内核旁路网络（`ef_vi` / GPUDirect RDMA）+ 单持久化轮询 CUDA 内核（Persistent Megakernel）+ `isolcpus` 核隔离改造后的微秒级推理系统（中位数 `16.5 us`，P99.99 `24.8 us`，尾延迟削减 `16.94x`）。
