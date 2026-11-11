# 实验 14-05：受控双园区 400G 光纤 DCI 分层训练重叠（DeepSeek-V3 671B）与园区电价/储能账本复算器

本实验复现第 14 章 14.1.3 节、14.4.3 节与 14.6.2 节的受控双园区专用 DWDM 光互连（DCI）分层同步账本与 $100\ \mathrm{MW}$ 园区供电/储能经济账本（对齐 `calculations/calc.py datacenter-power-cooling`）：

1. **DeepSeek-V3（总参 671B / 活跃 37B，61 层）受控双园区 400G 光纤 DCI 训练账本**：
   - 朴素每步同步（双园区 $K=2$ 全双工环规约系数 $2(K-1)/K = 1.0$，单向发出 BF16 梯度/伪梯度量 $1.342\ \mathrm{TB}$，单步计算窗口 $T_{\mathrm{step}} = 4.20\ \mathrm{s}$）要求跨城专线持续提供 $2,556.2\ \mathrm{Gbps}$；
   - 采用双循环分层同步（$H = 500$ 步内循环窗口 $2,100.0\ \mathrm{s}$）后，平均跨城带宽降至 $5.11\ \mathrm{Gbps}$；在单条 $400\ \mathrm{Gbps}$ 受控园区 DCI 光纤（按 $85\%$ 有效净荷计为 $42.5\ \mathrm{GB/s}$）上外同步耗时 $31.58\ \mathrm{s}$，阻塞占比仅 $1.48\%$；
   - 进一步启用 61 层流式分片重叠（Streaming DiLoCo）后，每层分片传输耗时约 $0.518\ \mathrm{s}$，完全隐藏于约 $34.4\ \mathrm{s}$ 的层分片内循环间隔之下，暴露跨城通信停顿降为 $0.0\ \mathrm{s}$。
2. **100 MW 园区电价敏感性与储能配平账本（算例 14-3、算例 14-9）**：
   - 工业电价由 $0.12\ \mathrm{USD/kWh}$ 降至 $0.04\ \mathrm{USD/kWh}$ 时，$100\ \mathrm{MW}$ 园区每年节省电费 $7,008$ 万美元（三年节省 $2.1024$ 亿美元）；
   - 按峰值负载 $20\%$、持续 $2\ \mathrm{h}$ 配置削峰稳频储能容量为 $40.0\ \mathrm{MWh}$，扩展至 $100\ \mathrm{MWh}$ 时支持峰谷套利与电网一次调频复用。

## 运行方式

```bash
python3 experiments/ch14/14-05/run.py
python3 experiments/ch14/14-05/verify.py
```
