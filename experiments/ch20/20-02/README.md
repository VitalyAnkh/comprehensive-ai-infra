# 实验 20-02：十万卡集群混沌故障注入（Chaos GameDay）、Arrival Skew 慢节点定位、TrainMover 热迁移与跑分失真修正器

本实验复现第 20 章 20.3–20.5 节、算例 20-6 与练习 20-11、实验 20-12 的核心定量模型与算法：

1. **十万卡集群主动混沌故障注入演练（Chaos GameDay，算例 20-6）**：
   - 在预生产/影子集群中向四类故障域确定性注入 $N_{\mathrm{chaos}} = 200$ 次故障（每类 $50$ 次）：
     1. 算子位级 SDC 翻转（$+32.0\text{–}+64.0$）：`ABFT` + `Safeguarding` + `OpGuard` 捕获 $50/50$（召回率 **$100.0\%$**，$\mathrm{MTTD} = 0.042\ \mathrm{s}$）；
     2. 计算/PCIe 慢节点（Straggler）：`NCCL Flight Recorder` 到达偏斜矩阵捕获 $50/50$（召回率 **$100.0\%$**，$\mathrm{MTTD} = 3.20\ \mathrm{s}$）；
     3. 800G 光模块 RS-FEC 高阶桶劣化灰故障：DOM/FEC 斜率预警捕获 $49/50$（召回率 **$98.0\%$**，$\mathrm{MTTD} = 6.80\ \mathrm{s}$）；
     4. HBM 双比特 ECC / 掉卡故障：`TrainMover` 预热备机切换捕获 $50/50$（召回率 **$100.0\%$**，$\mathrm{MTTD} = 0.84\ \mathrm{s}$）；
   - 综合故障召回率达 **$199/200 = 99.50\%$**，机队平均检测时间 $\mathrm{MTTD} = \mathbf{2.70\ \mathrm{s}}$，自动自愈恢复时间 $\mathrm{MTTR}_{\mathrm{auto}} = \mathbf{3.85\ \mathrm{s}}$（相比传统人工值班重启 $1,500.0\ \mathrm{s}$ 提速 **$389.61\times$**）。
2. **8-Rank 集合通信 Arrival Skew Matrix 慢节点反演与 `TrainMover` 亚秒级热迁移（算例 20-6 & 实验 20-12）**：
   - 7 个健康 Rank 在 $t = 10.0\ \mathrm{ms}$ 入队，第 5 号慢节点因 PCIe 降级在 $t = 12.0\ \mathrm{ms}$ 入队（入队偏斜 $\Delta t_{\mathrm{skew}} = \mathbf{2.00\ \mathrm{ms}}$），实际 Ring 传输耗时 $T_{\mathrm{xfer}} = 0.40\ \mathrm{ms}$；
   - 健康 Rank 观测到 $2.40\ \mathrm{ms}$ 通信耗时（其中 **$83.33\%$** 为等待第 5 号慢卡，仅 **$16.67\%$** 为真实传输）；通过入队偏斜矩阵唯一锁定 `Rank 5`，并由 `TrainMover` 在 **$0.68\ \mathrm{s}$** 内完成局部 RDMA QP 重映射与状态热接管（相比传统 $300.0\ \mathrm{s}$ 全局重启提速 **$441.18\times$**）。
3. **四大跑分失真修正与两阶段金丝雀门禁验证（算例 20-5、练习 20-7/20-8 & 实验 20-12）**：
   - 验证 $5.25\times$ 算术平均加速比实则 $0.9524\times$ 倒退、`ABBA` 消除 $-16.0\ \mu\mathrm{s}$ / $-24.0\ \mu\mathrm{s}$ 线性热漂移偏置，以及金丝雀门禁捕获 $+4.8\%$（$> +3.0\%$）P99 退化自动回滚。

## 运行方式

```bash
python3 experiments/ch20/20-02/run.py
python3 experiments/ch20/20-02/verify.py
```
