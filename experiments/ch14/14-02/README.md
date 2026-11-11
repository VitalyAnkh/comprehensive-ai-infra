# 实验 14-02：园区百兆瓦储能套利、Kareus 动静态能耗合并与 Streaming DiLoCo / Di-PS 广域网账本

本实验复现第 14 章 14.4 节与 14.6.2–14.6.3 节的能量调度与跨园区协同训练账本：

1. **Streaming DiLoCo 与 Di-PS 跨园区训练带宽账本（算例 14-7）**：
   - $100\mathrm{B}$ 模型（$V_{\mathrm{sync}} = 3,200\ \mathrm{Gb}$，$T_{\mathrm{step}} = 2.0\ \mathrm{s}$）：每步同步需 $1,600\ \mathrm{Gbps}$，DiLoCo（$H=500$）平均带宽降至 $3.2\ \mathrm{Gbps}$，但在 $10\ \mathrm{Gbps}$ 专线上阻塞 $320\ \mathrm{s}$；
   - Streaming DiLoCo 切分为 $M=25$ 个分片（每 $40\ \mathrm{s}$ 传一个 $128\ \mathrm{Gb}$ 分片，耗时 $12.8\ \mathrm{s} < 40\ \mathrm{s}$），暴露停顿降为 $0\ \mathrm{s}$。
2. **Kareus（OSDI '26）动静态能耗联合优化（练习 14-12）**：
   - 给定单卡静态漏电功耗 $P_{\mathrm{static}} = 180\ \mathrm{W}$、满载动态功耗 $P_{\mathrm{dyn}} = 520\ \mathrm{W}$，在集群负载率 $u = 40\%$ 下，均匀分摊单卡功耗为 $180 + 0.40 \times 520 = 388\ \mathrm{W}$；
   - 采用 Kareus 将负载紧缩合并至 $40\%$ 活跃卡满载运行（$700\ \mathrm{W}$）、其余 $60\%$ 空闲卡进入深度休眠（$P_{\mathrm{sleep}} = 15\ \mathrm{W}$），等效单卡平均功耗降至 $0.40 \times 700 + 0.60 \times 15 = 289\ \mathrm{W}$（若休眠态彻底关断至 $0\ \mathrm{W}$ 则为 $280\ \mathrm{W}$），分别节电 $25.52\%$ 与 $27.84\%$。
3. **百兆瓦时园区储能系统（BESS）峰谷套利与电网调频账本（练习 14-13）**：
   - $E_{\mathrm{BESS}} = 100\ \mathrm{MWh}$，谷电充电价 $c_{\mathrm{off}} = 0.04\ \mathrm{USD/kWh}$、峰电放电顶替价 $c_{\mathrm{peak}} = 0.12\ \mathrm{USD/kWh}$（名义价差 $0.08\ \mathrm{USD/kWh}$），往返循环效率 $\eta_{\mathrm{rt}} = 88\%$。

## 运行方式

```bash
python3 experiments/ch14/14-02/run.py
python3 experiments/ch14/14-02/verify.py
```
