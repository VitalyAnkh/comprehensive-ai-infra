# 实验 15-03：Oobleck 预实例化流水线无重启容错重配置与推理服务泊松突发流量 Goodput 评测器

本实验复算并校验第 15 章 15.5.3 节算例 15-7、练习 15-13 与实验 15-14 的全部锁定工程账本：

1. **Oobleck / Parachute 预实例化异构流水线模板无重启容错重配置（算例 15-7 & 练习 15-13）**：
   - $P = 32$ 条 8 阶段数据并行流水线（$256$ 台节点）突发 $F = 2$ 台节点掉线且无热备顶替时：传统全局重启耗时 $300.00\ \mathrm{s}$，而 `Oobleck` 预实例化模板重配置仅需 $0.25 + 0.85 + 0.35 = \mathbf{1.45\ \mathrm{s}}$（恢复提速 **$206.90\times$**），30 条 $\mathcal{T}_8$ + 2 条 $\mathcal{T}_7$（效率 $80\%$）保留 **$98.75\%$** 稳态吞吐（`Parachute` 摘除 2 条管线保留 **$93.75\%$**）；
   - 练习 15-13（$64$ 条管线突发 $F = 4$ 台节点掉线，$\mathcal{T}_7$ 效率 $82\%$）：`Parachute` 保留 **$93.75\%$** 吞吐，`Oobleck` 保留 **$98.88\%$** 吞吐，每天净多挽回 **$629.76$** 个等效健康节点小时（$5,038.08$ GPU 小时/天）。
2. **推理压测均匀到达（$C_a=C_s=0$）vs. 泊松突发长尾流量（$C_a=1.0, C_s=1.2$）的 Goodput 崩塌（算例 15-7 & 实验 15-14）**：
   - 在 $\rho = 0.85, T_s = 120.0\ \mathrm{ms}, T_{\mathrm{prefill}} = 80.0\ \mathrm{ms}, \mathrm{SLO}_{\mathrm{TTFT}} = 500.0\ \mathrm{ms}$ 下：均匀定速压测排队 $0.00\ \mathrm{ms}$、$\mathrm{TTFT} = 80.00\ \mathrm{ms}$、虚报 $100\%$ 达标（Goodput $= 7.08\ \mathrm{req/s}$）；真实泊松长尾流量下排队飙升至 **$829.60\ \mathrm{ms}$**、平均 $\mathrm{TTFT} = \mathbf{909.60\ \mathrm{ms}}$（膨胀 **$11.37\times$**），SLO 达标率跌至 **$48.77\%$**，真实 Goodput 仅为 **$3.45\ \mathrm{req/s}$**（缩水 **$51.27\%$**），满足平均 $\mathrm{TTFT} \le 300.0\ \mathrm{ms}$ 的最大安全利用率为 **$60.04\%$**；
   - 实验 15-14（$\rho = 0.80, T_s = 100.0\ \mathrm{ms}, T_{\mathrm{prefill}} = 60.0\ \mathrm{ms}, C_a = C_s = 1.5$）：排队等待时间从 $0.00\ \mathrm{ms}$ 飙升至 **$900.00\ \mathrm{ms}$**，平均 $\mathrm{TTFT}$ 从 $60.00\ \mathrm{ms}$ 膨胀至 **$960.00\ \mathrm{ms}$**（膨胀 **$16.00\times$**）。

## 运行与验证

```bash
python3 experiments/ch15/15-03/run.py
python3 experiments/ch15/15-03/verify.py
```
