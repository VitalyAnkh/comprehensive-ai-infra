# 实验 14-03：面向 Vera Rubin 与 TPU v7 的下一代 MoE 超参数与芯片带宽比联合 Pareto 寻优器

本实验复现第 14 章 14.9.3 节“面向 Vera Rubin 与 TPU v7 的下一代 MoE 超参数协同搜索（专家粒度、激活比、注意力混合比与芯片内存带宽比联合 Pareto 寻优）”中的三组定量账本：

1. **MoE 细粒度专家切分倍数 $G$ 的表达收益与硬件瓦片/通信开销帕累托拐点**：
   - 保持每层激活 FFN 维度预算 $k(G) \cdot d_{\mathrm{exp}}(G) = 8,192$ 不变，扫描切分倍数 $G \in \{1, 2, 4, 8, 16\}$（总路由专家数 $E(G) = 64 G$，激活专家数 $k(G) = 2 G$，单专家隐藏维 $d_{\mathrm{exp}}(G) = 4,096 / G$）；
   - 当 $G = 8$（即 $E = 512, k = 16, d_{\mathrm{exp}} = 512$，恰好对齐矩阵核心最小高效列宽 $N_{\mathrm{tile}} = 512$）时，单层步延迟控制在 **$0.74\ \mathrm{ms}$**（瓦片有效利用率 $100\%$），质量损失相对 $G=1$ 下降 **$0.0720\ \mathrm{nats}$**，取得质量与延迟效用函数的全局最优（效用分 **$3.870$**）；若继续切分至 $G = 16$（$d_{\mathrm{exp}} = 256 < 512$，瓦片利用率跌至 $50\%$），瓦片填充折损使延迟跳升至 **$2.02\ \mathrm{ms}$**。
2. **Vera Rubin NVL72 与 TPU v7（Ironwood）在 671B/37B MoE 上的算存比与通信拐点对比**：
   - **Vera Rubin**（$P_{\mathrm{FP4}} = 50,000\ \mathrm{TFLOP/s}$，$B_{\mathrm{HBM4}} = 22.0\ \mathrm{TB/s}$，NVLink 6 单卡双向 $3.6\ \mathrm{TB/s}$）：Roofline 临界算术强度高达 $\mathcal{I}_{\mathrm{Rubin}}^* = \mathbf{2,272.73\ \mathrm{FLOP/B}}$；在 `EP=64`、`batch=128` Decode 下，单步权重与状态读取耗时 **$8.64\ \mathrm{ms}$**、All-to-All 通信耗时 **$1.42\ \mathrm{ms}$**，单步总周期 **$11.86\ \mathrm{ms}$**（吞吐 **$10,792.6\ \mathrm{tok/s}$**）；
   - **TPU v7 Ironwood**（$P_{\mathrm{FP8}} = 4,614\ \mathrm{TFLOP/s}$，$B_{\mathrm{HBM3e}} = 7.37\ \mathrm{TB/s}$，3D Torus ICI $1.2\ \mathrm{TB/s}$）：Roofline 临界算术强度为 $\mathcal{I}_{\mathrm{TPUv7}}^* = \mathbf{626.05\ \mathrm{FLOP/B}}$；在 `EP=64`、`batch=128` FP8 Decode 下，单步读取耗时 **$25.78\ \mathrm{ms}$**、ICI 通信耗时 **$4.26\ \mathrm{ms}$**，单步总周期 **$33.64\ \mathrm{ms}$**（吞吐 **$3,805.0\ \mathrm{tok/s}$**）。
3. **三维超参数 $(G, \rho_{\mathrm{act}}, r_{\mathrm{attn}})$ 硬件感知闭环搜索（HW-NAS Loop）**：
   - 在单步延迟上限 $\mathrm{TPOT} \le 15.0\ \mathrm{ms}$、单卡状态显存预算 $\le 40.0\ \mathrm{GiB}$ 约束下，联合搜索 $(G, \rho_{\mathrm{act}}, r_{\mathrm{attn}})$，锁定最优配置为 **$(G^*=8, \rho_{\mathrm{act}}^*=0.03125, r_{\mathrm{attn}}^*=0.25)$**（即 $1:3$ 全注意力与线性注意力混合 + $16/512$ 细粒度 MoE），单步耗时 **$12.91\ \mathrm{ms}$**、单卡状态显存 **$28.25\ \mathrm{GiB}$**、综合质量得分 **$96.80$**。

## 运行方式

```bash
python3 experiments/ch14/14-03/run.py
python3 experiments/ch14/14-03/verify.py
```
