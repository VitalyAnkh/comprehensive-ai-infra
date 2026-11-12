# 实验 20-04：端侧多相机 ISP -> DMA-BUF 零拷贝传感链路 vs. ROS 2 序列化拷贝的 WCET 与抖动验证器

本实验复现第 20 章 20.5.1 节算例 20-8 以及练习 20-15、练习 20-16 的全部锁定工程指标（并与 `calculations/results/ch20-embodied-wcet-sim.json` 同源对齐）：

1. **6 路 `1920×1200@60fps` `NV12` 相机传感链路的 DRAM 带宽与端到端延迟拆解（算例 20-8）**：
   - 单帧 `NV12`（`1.5 B/pixel`）体积为 $1{,}920 \times 1{,}200 \times 1.5 = \mathbf{3{,}456{,}000\ \mathrm{B}}$，6 路 60 FPS 原始传感器流入速率为 **$1.244\ \mathrm{GB/s}$**（精确 $1.24416\ \mathrm{GB/s}$）；
   - **传统 `ROS 2` 套接字序列化拷贝路径**：经历 `1` 次 ISP 写出与 `4` 次全帧拷贝（每次读写双向），累计 **9 次 DRAM 遍历**，消耗 **$11.20\ \mathrm{GB/s}$** 内存带宽；端到端传感至 ViT 输入 P99 延迟高达 **$18.60\ \mathrm{ms}$**，`P99.9` 调度抖动为 **$9.40\ \mathrm{ms}$**；
   - **硬件级 `CSI/GMSL -> ISP -> DMA-BUF / NvSciBuf + Iceoryx` 零拷贝同步路径**：仅需 ISP 统一物理页写出与 GPU ViT 融合采样直读 **2 次 DRAM 遍历**，内存流量压降至 **$2.49\ \mathrm{GB/s}$**（缩减 **$4.50\times$**），传感传输 P99 延迟压降至 **$0.85\ \mathrm{ms}$**（提速 **$21.88\times$**），`P99.9` 抖动收敛至 **$0.14\ \mathrm{ms} < 0.15\ \mathrm{ms}$**。
2. **8 路 4K@30fps 环视扩展与 `Iceoryx` 无锁共享内存池无饥饿槽位下界（练习 20-15 & 练习 20-16）**：
   - 8 路 `3840×2160@30fps` `NV12` 原始流入速率达 **$2.986\ \mathrm{GB/s}$**，零拷贝路径将 DRAM 带宽消耗从 **$26.87\ \mathrm{GB/s}$** 降至 **$5.97\ \mathrm{GB/s}$**（净省 **$20.90\ \mathrm{GB/s}$**），并在 $v = 1.20\ \mathrm{m/s}$ 下消除 **$51.35\ \mathrm{mm}$** 传感搬移盲飞位移；
   - 依据 `Iceoryx` 零拷贝借用（Loan/Borrow）槽位守恒公式 $S_{\min} = 1_{\mathrm{write}} + 1_{\mathrm{queue}} + \sum_j H_j$，单路相机至少需配置 **$6$ 个槽位**（6 路合计 $36$ 个槽位、驻留池容量 **$118.65\ \mathrm{MiB}$**）方可保证慢速 System 2 持锁期间高频发布者零阻塞。

## 运行与校验

```bash
python3 experiments/ch20/20-04/run.py
python3 experiments/ch20/20-04/verify.py
```
