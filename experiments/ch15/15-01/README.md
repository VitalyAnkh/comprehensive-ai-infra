# 实验 15-01：吉瓦级机柜液冷热阻网络、LogicFolding 功率密度悖论与 TAPAS 热感知路由回本仿真

本实验复现第 15 章 15.3 节与 15.6.1 节的核心热力学与热感知调度账本：

1. **LogicFolding 三维折叠芯片功率密度悖论（算例 15-5）**：
   - 供电电压从 $0.85\ \mathrm{V}$ 降至 $0.55\ \mathrm{V}$ 时，动态功耗电压平方因子降至 $(0.55/0.85)^2 \approx 0.4187$（约 $0.42\times$）；
   - 初代折叠芯片总功耗降至 $0.75\times$（$-25\%$），但芯片投影面积缩减至 $0.60\times$（$-40\%$），单位面积热流密度反升至 $0.75 / 0.60 = 1.25\times$（$+25\%$）。
2. **TAPAS 热感知与功率封顶重配置回本阈值（算例 15-6）**：
   - 机柜排功率 $[96, 80]\ \mathrm{kW}$ 在 $+6\ \mathrm{kW}$ 请求下可行集为 `[1]`，延迟 $[0.8, 1.3]\ \mathrm{s}$ 在 $1.0\ \mathrm{s}$ 截止期下准入集为 `[]`；
   - 重配置停顿 $T_{\mathrm{reconfig}} = 3.0\ \mathrm{s}$、每请求节省 $\Delta t = 0.1\ \mathrm{s}$ 时，最小回本请求数为 $N^* = 30$；
   - 单卡级热裕度 $85.0 - (68.5 + 1.8 \times 6) = 5.7^\circ\mathrm{C}$，热迁移 $120\ \mathrm{ms}$ 停顿在解码步耗时从 $48\ \mathrm{ms}$ 降至 $38\ \mathrm{ms}$ 时需至少 $13$ 个生成词元回本。
3. **硅背面微通道直冷（Direct-to-Silicon Microfluidics）进液温度提升（练习 15-11）**：
   - 在 $P_{\mathrm{chip}} = 2,000\ \mathrm{W}$ 下，消除导热界面材料热阻 $R_{\mathrm{TIM}} = 0.02\ \mathrm{^\circ C/W}$ 可直接降低结壳温差 $40.0^\circ\mathrm{C}$（或在相同结温红线下将最大允许进液温度提升 $+40.0^\circ\mathrm{C}$）。

## 运行方式

```bash
python3 experiments/ch15/15-01/run.py
python3 experiments/ch15/15-01/verify.py
```
