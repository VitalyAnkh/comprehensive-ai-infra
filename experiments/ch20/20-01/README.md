# 实验 20-01：具身智能 VLA 双频控制回路（System 2 + System 1）盲飞位移、VLALagBench 成功率退化与单卡 4,096 并行物理仿真显存/SM 预算器

本实验复现第 20 章 20.1–20.4 节与练习 20-1、20-2、20-4、20-6、实验 20-10 的核心定量模型：

1. **硬实时盲飞位移 $\Delta x = v \cdot \Delta t_{\mathrm{inf}}$ 与动作块时间集成（Temporal Ensembling，算例 20-1 & 20-2）**：
   - 在机械臂末端或自动驾驶车辆运动速度 $v = 1.5\ \mathrm{m/s}$ 下，云端软实时推理延迟 $\Delta t_{\mathrm{cloud}} = 120\ \mathrm{ms}$ 造成盲飞位移 **$\Delta x_{\mathrm{cloud}} = 180.0\ \mathrm{mm}$**，而在机载 System 1 硬实时控制周期 $\Delta t_{\mathrm{sys1}} = 5\ \mathrm{ms}$（$200\ \mathrm{Hz}$）下盲飞位移压降至 **$\Delta x_{\mathrm{sys1}} = 7.5\ \mathrm{mm}$**（下降 $24.0\times$）；
   - 在 VLALagBench 动态目标抓取基准中，当感知—动作延迟从 $20\ \mathrm{ms}$ 升至 $120\ \mathrm{ms}$（超过 $100\ \mathrm{ms}$ 观测过时阈值）时，静态抓取成功率仅从 $88.0\%$ 微降至 $82.0\%$，而动态抓取成功率从 $84.0\%$ 跌至 **$32.0\% < 40.0\%$**（净跌 $52.0$ 个百分点）。
2. **System 2（$2\ \mathrm{Hz}$ VLM）+ System 1（$100\text{–}200\ \mathrm{Hz}$ Flow-Matching 动作头）在 Jetson Thor 上的 SM/MIG 隔离与 WCET 保障（算例 20-3）**：
   - 未隔离竞争模式下，System 2 的视觉 Prefill 抢占 SM 与内存带宽，使 System 1（5 步 Euler ODE 积分）的 P99 最坏执行时间（WCET）从 $3.2\ \mathrm{ms}$ 恶化至 **$18.5\ \mathrm{ms}$**（超出 $10\ \mathrm{ms}$ 截止期）；
   - 启用硬件 SM/MIG 分区与锁页无锁环形缓冲后，System 1 的 P99 WCET 严格收敛在 **$4.1\ \mathrm{ms} \le 5.0\ \mathrm{ms}$**。
3. **单卡 $N_{\mathrm{env}} = 4,096$ 并行 GPU 物理仿真（Isaac Lab / MuJoCo Warp / Genesis）的零拷贝带宽节省与显存账本（算例 20-4）**：
   - 消除 CPU 仿真 + PCIe 往返后，在 $4,096$ 环境、$60\ \mathrm{Hz}$ 下避免每秒 **$24.22\ \mathrm{GB/s}$** 的跨总线搬移；
   - 单卡 $4,096$ 环境总显存占用为：接触与刚体状态 $256.0\ \mathrm{MiB}$ + BVH 几何共享 $512.0\ \mathrm{MiB}$ + 双目渲染帧缓冲 $1,536.0\ \mathrm{MiB}$ + 策略网络与 Rollout 缓冲 $6,144.0\ \mathrm{MiB} =$ **$8,448.0\ \mathrm{MiB}$（$8.25\ \mathrm{GiB}$）**。

## 运行方式

```bash
python3 experiments/ch20/20-01/run.py
python3 experiments/ch20/20-01/verify.py
```
