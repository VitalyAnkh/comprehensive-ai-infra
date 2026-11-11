# 实验 14-04：800V HVDC 机柜母排铜损、板级垂直供电（VPD）与 ASHRAE W45 温水液冷热阻复算器

本实验复现第 14 章 14.1.1 节、14.3.1 节与 14.6.4 节的高压直流供电母排损耗、芯片背面垂直供电（VPD）与 ASHRAE W45 温水直冷热工账本（对齐 `calculations/calc.py datacenter-power-cooling`）：

1. **48 VDC vs. 800 VDC 机柜铜母排焦耳损耗账本（算例 14-1、算例 14-11、练习 14-1、练习 14-15）**：
   - $132\ \mathrm{kW}$ 机柜在 $R_{\mathrm{bus}} = 0.5\ \mathrm{m\Omega}$ 下：$48\ \mathrm{VDC}$ 电流达 $2,750.0\ \mathrm{A}$、母排铜损 $3,781.25\ \mathrm{W}$；升级 $800\ \mathrm{VDC}$ 后电流降至 $165.0\ \mathrm{A}$、铜损降至 $13.6125\ \mathrm{W}$（缩减 $277.78\times$）；下一代 $165\ \mathrm{kW}$ 机柜在 $800\ \mathrm{VDC}$ 下母排铜损仅为 $21.27\ \mathrm{W}$。
   - $140\ \mathrm{kW}$ 机柜在 $R_{\mathrm{bus}} = 0.8\ \mathrm{m\Omega}$ 下：铜损由 $6,805.56\ \mathrm{W}$ 压降至 $24.50\ \mathrm{W}$；$72\times\mathrm{GPU}$ 板级背面垂直供电（VPD，$35\ \mathrm{\mu\Omega} \to 3.5\ \mathrm{\mu\Omega}$）整柜额外节省 $13.06\ \mathrm{kW}$。
2. **ASHRAE W45 温水直冷流量与结温边界（算例 14-4、算例 14-8、练习 14-5）**：
   - 在 $T_{\mathrm{in}} = 45.0^\circ\mathrm{C}$、$\Delta T = 12.0\ \mathrm{K}$ 下，$132\ \mathrm{kW}$ 机柜冷却水流量为 $159.5\ \mathrm{L/min}$（按 $50^\circ\mathrm{C}$ 水密度 $0.990\ \mathrm{kg/L}$；若按 $1.0\ \mathrm{kg/L}$ 近似则为 $157.9\ \mathrm{L/min}$），$165\ \mathrm{kW}$ 机柜为 $199.4\ \mathrm{L/min}$；$1,400\ \mathrm{W}$ 芯片壳温为 $74.2^\circ\mathrm{C} < 85.0^\circ\mathrm{C}$。

## 运行方式

```bash
python3 experiments/ch14/14-04/run.py
python3 experiments/ch14/14-04/verify.py
```
