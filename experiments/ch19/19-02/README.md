# 实验 19-02：人形机器人三级控制环（VLM + Flow-Matching + 1,000 Hz WBC/视触觉反射）、PTP 多传感器时标对齐、CBF 护盾与 FAST 频域动作分词器

本实验复现第 19 章 19.4.1–19.4.4 节、算例 19-6 与练习 19-11、实验 19-12 的核心定量模型与算法：

1. **人形机器人三级异构频率解耦与 $1,000\ \mathrm{Hz}$ 视触觉初生打滑反射（算例 19-6）**：
   - Tier 1（System 2 VLM）：$2.0\ \mathrm{Hz}$（周期 $500.0\ \mathrm{ms}$）；Tier 2（System 1 流匹配动作头）：$100.0\ \mathrm{Hz}$（周期 $10.0\ \mathrm{ms}$，$\mathrm{WCET} = 4.2\ \mathrm{ms}$）；Tier 3（$1,000\ \mathrm{Hz}$ 视触觉打滑反射 + $28$ 自由度全身动力学 WBC-QP）：周期 $1,000.0\ \mu\mathrm{s}$（$1.0\ \mathrm{ms}$）；
   - Tier 3 单周期最坏执行时间拆解：触觉与 EtherCAT 读取 $180.0\ \mu\mathrm{s}$ + 切向/法向力比打滑判定 $40.0\ \mu\mathrm{s}$ + $28$ 自由度全身二次规划（WBC-QP）$420.0\ \mu\mathrm{s}$ + EtherCAT PDO 下发 $160.0\ \mu\mathrm{s} = \mathbf{800.0\ \mu\mathrm{s}}$（留有 $200.0\ \mu\mathrm{s}$ 安全裕量）；
   - 当被抓持物体以 $v_{\mathrm{slip}} = 0.12\ \mathrm{m/s}$ 发生初生滑移时，$100\ \mathrm{Hz}$ 视觉环（含 $15\ \mathrm{ms}$ 链路滞后）滑移达 **$1.80\ \mathrm{mm}$**，而 $1,000\ \mathrm{Hz}$ 视触觉反射环在 $1.0\ \mathrm{ms}$ 内制动，滑移压降至 **$0.12\ \mathrm{mm}$**（缩减 **$15.0\times$**）。
2. **`FAST`（Frequency-space Action Sequence Tokenization）DCT 频域动作压缩（算例 19-6 & 练习 19-11）**：
   - 对视野 $H = 50$ 步、动作维度 $D_a = 14$ 的高频连续动作块（传统时域分箱需 $50 \times 14 = \mathbf{700}$ 个离散 Token），沿时间维执行正交 Type-II 离散余弦变换（DCT-II）；
   - 前 $K = 8$ 个低频系数保留了 **$99.47\%$** 的轨迹能量，经量化与游程/BPE 编码后仅需 **$80$ 个频域 Token**，实现 **$8.75\times$** 词元压缩率（序列缩短 $88.57\%$），将自回归解码耗时从 $2,450.0\ \mathrm{ms}$ 压降至 **$280.0\ \mathrm{ms}$**，重构轨迹均方根误差 $\mathrm{RMSE} = 0.2431\ \mathrm{mm} < 0.35\ \mathrm{mm}$。
3. **硬件 PTP 多传感器时标对齐与 $1,000\ \mathrm{Hz}$ 控制屏障函数（CBF）闭式安全护盾验证**：
   - 在末端速度 $v = 0.60\ \mathrm{m/s}$ 下，消除 $27.0\ \mathrm{ms}$ 软件时间戳错位（$16.20\ \mathrm{mm}$ 标定偏差），将 PTP + $\mathrm{SE}(3)$ 插值误差锁定在 $0.018\ \mathrm{ms}$（$18\ \mu\mathrm{s} < 100\ \mu\mathrm{s}$，空间误差 $0.011\ \mathrm{mm}$，精度提升 **$1,472\times$**）；
   - 验证闭式 CBF 安全护盾在 $h = 15.0\ \mathrm{mm}, v_{\mathrm{nom}} = 0.85\ \mathrm{m/s}$ 下实时钳制为 $v_{\mathrm{safe}} = 0.30\ \mathrm{m/s}$，并在 $500$ 步连续冲击下实现 $0$ 次越界（最小安全距离保持 $1.20\ \mathrm{mm}$）。

## 运行方式

```bash
python3 experiments/ch19/19-02/run.py
python3 experiments/ch19/19-02/verify.py
```
