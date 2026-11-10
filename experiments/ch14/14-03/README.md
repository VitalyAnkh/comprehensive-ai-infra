# 实验 14-03：硅内三维分水微通道直冷（Direct-to-Silicon Microfluidics）与 1.2 GW 园区孤岛微网（SMR + 燃机 + 构网型 BESS）动态调频仿真

本实验复现第 14 章 15.6.4 节新增专题（练习 15-17、练习 15-18）的热工水力方程与园区孤岛微网转子摇摆方程：

1. **硅内三维分水微通道直冷（2,500 W 芯片，$q'' = 277.78\ \mathrm{W/cm^2}$）**：
   - 对比传统外贴铜冷板（$R_{\mathrm{th, cp}} = 0.028\ \mathrm{K/W}$，$T_{\mathrm{in}} = 45.0^\circ\mathrm{C}$ 下结温达 $115.0^\circ\mathrm{C}$）与硅内三维分水微通道直冷（$R_{\mathrm{cond, Si}} = 0.0018\ \mathrm{K/W}$、$R_{\mathrm{conv, micro}} = 0.0042\ \mathrm{K/W}$、$R_{\mathrm{cal}} = 0.0040\ \mathrm{K/W}$，总热阻 $0.0100\ \mathrm{K/W}$，降低 $2.80\times$）；
   - 验证在 ASHRAE W45（$45.0^\circ\mathrm{C}$）下结温保持 $70.0^\circ\mathrm{C} \le 85.0^\circ\mathrm{C}$（最高允许进水温度 $60.0^\circ\mathrm{C}$），且在 $\Delta P = 38.4\ \mathrm{kPa}$、$\dot{V} = 1.80\ \mathrm{L/min}$ 下水力泵功仅为 $1.65\ \mathrm{W}$（占芯片热功耗 $0.0658\%$）。
2. **1.2 GW 园区孤岛微网（300 MW SMR + 600 MW 燃机 + 400 MW 构网型 BESS）调频整定**：
   - 在 $\Delta P_{\mathrm{step}} = 450.0\ \mathrm{MW}$ 同步训练阶跃卸载下，无虚拟惯量时物理惯量 $H_{\mathrm{phys}} = 3.00\ \mathrm{s}$、初始 $\mathrm{RoCoF}_0 = 3.750\ \mathrm{Hz/s}$；
   - 整定 $400\ \mathrm{MW}$ 构网型 BESS 虚拟同步惯量 $H_{\mathrm{BESS}} = 13.50\ \mathrm{s}$ 后，系统等效惯量升至 $7.50\ \mathrm{s}$，初始 $\mathrm{RoCoF}_0$ 压至 $1.500\ \mathrm{Hz/s}$，暂态极值频偏从 $2.874\ \mathrm{Hz}$ 压至 $0.739\ \mathrm{Hz}$。
3. **8 子集群微批次错峰与电热联合 DVFS 闭环控制**：
   - 将瞬态阶跃负载削减至 $144.0\ \mathrm{MW}$，芯片结温控制在 $66.01^\circ\mathrm{C}$，端到端训练步时间开销仅为 $1.43\%$。

## 运行方式

```bash
python3 experiments/ch14/14-03/run.py
python3 experiments/ch14/14-03/verify.py
```
