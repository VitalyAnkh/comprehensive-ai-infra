# 实验 15-05：分布式集合通信等待链有向图（Mycroft）受害者剥离、NCCL Flight Recorder 取证与 Monarch 局部自愈器

本实验复现第 15 章 15.6.2 节、算例 15-9、算例 15-10 及取证综合题 15-16、15-17、15-18 的全部锁定账本，并与 `calculations/results/ch15-fleet-reliability-profiler.json` 严格同源：

1. **8-Rank `TP×PP×DP` 集合通信等待链有向图（`Mycroft / HTA`）受害者剥离与根因归因（算例 15-9）**：
   - 在 8-Rank（`TP=2, PP=2, DP=2`）步骤屏障中，Rank `0, 1, 2, 3, 6` 于 $60.0\ \mathrm{ms}$ 入队、Rank `4, 7` 于 $62.0\ \mathrm{ms}$ 入队，唯独 **Rank `5`** 因 $140.0\ \mathrm{ms}$ 主机 DataLoader / PCIe 阻塞迟至 **$200.0\ \mathrm{ms}$** 入队，全组于 $204.0\ \mathrm{ms}$ 完成 $4.0\ \mathrm{ms}$ 真实网络归约；
   - 朴素火焰图上 $7$ 个健康受害节点显示 `ncclKernel_AllReduce` 耗时高达 $142.0\text{–}144.0\ \mathrm{ms}$，而肇事者 Rank `5` 仅显示 $4.0\ \mathrm{ms}$；有向等待图逆向求解 $\arg\max_r t_{\mathrm{enter}}(r) = \mathbf{5}$ 精准剥离 **$7$ 个假阳性受害节点**，锁定唯一根因 `Rank 5`。
2. **`PyTorch Monarch` 单控制器 Actor Mesh 局部故障隔离 vs. 传统 SPMD `torchrun` 全局重启（算例 15-10 & 练习 15-16）**：
   - 传统 SPMD 全局销毁重建耗时 **$142.0\ \mathrm{s}$**（看门狗与清理 $28.0\ \mathrm{s}$ + `TCPStore/QP` 重建 $76.0\ \mathrm{s}$ + 检查点重载 $38.0\ \mathrm{s}$）；
   - `Monarch` 控制面网格通过心跳感知（$0.12\ \mathrm{s}$）、故障子网格摘除重绑（$0.28\ \mathrm{s}$）与对端 RDMA 状态注水（$0.45\ \mathrm{s}$），仅需 **$0.85\ \mathrm{s}$** 完成无中断隔离恢复，提速 **$167.1\times$**。
3. **跨层 Kineto 火焰图级联传播取证（练习 15-17）与 `NCCL Flight Recorder` 环路死锁 / 传输超时分类解码（练习 15-18）**：
   - 验证 `Rank 2` H2D 降速 $85.0\ \mathrm{ms}$ 沿 `TP -> PP -> DP` 级联传染的多跳等待链还原，以及 `Flight Recorder`（环形缓冲区容量 $K=2{,}048$）区分条件分支算子错位死锁（`COLL_TYPE_MISMATCH_DEADLOCK`）与底层 RDMA QP 超时挂死（`RDMA_TRANSPORT_TIMEOUT`）的确定性解码规则。

## 运行与验证

```bash
python3 experiments/ch15/15-05/run.py
python3 experiments/ch15/15-05/verify.py
```
