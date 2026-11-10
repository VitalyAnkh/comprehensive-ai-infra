# 实验 20-03：端到端自动驾驶 3DGS 多传感器闭环重仿真与车载双 Drive Thor ASIL-D 看门狗接管

本实验复算并校验第 20 章 19.4.4 节算例 19-7、练习 19-13 与实验 19-14 的全部锁定工程账本：

1. **云端 3DGS 动静态解耦多传感器闭环重渲染（算例 19-7 & 练习 19-13）**：
   - 6 路 1080p 环视相机 + 128 线激光雷达：朴素串行渲染耗时 $75.00\ \mathrm{ms/step}$（$13.33\ \mathrm{FPS}$）；启用联合视锥与距离 LOD 剔除（剔除 $72\%$，保留 $700,000$ 活跃高斯点）+ 6 路批量瓦片光栅化（$11.40\ \mathrm{ms}$）+ RT Core 光线—高斯 BVH 求交（$5.40\ \mathrm{ms}$）后，单步总耗时降至 **$16.80\ \mathrm{ms/step}$**（稳态帧率 **$59.52\ \mathrm{FPS}$**，加速 **$4.46\times$**）；
   - 8 路 1080p 环视 + 双 128 线激光雷达（练习 19-13）：串行 $104.00\ \mathrm{ms/step}$（$9.62\ \mathrm{FPS}$）降至批量 **$22.00\ \mathrm{ms/step}$**（**$45.45\ \mathrm{FPS}$**，加速 **$4.73\times$**）。
2. **车载双 Drive Thor 异构 SoC 的 ASIL-D Simplex 安全降级接管（算例 19-7 & 实验 19-14）**：
   - 四阶段硬实时接管总延迟：$5.00 + 1.50 + 4.00 + 2.00 = \mathbf{12.50\ \mathrm{ms}} \le 15.0\ \mathrm{ms}$（满足 ISO 26262 ASIL-D FTTI）；
   - $120\ \mathrm{km/h}$ 故障瞬态横向偏离：硬件看门狗热备接管锁死在 **$3.02\ \mathrm{mm}$** vs. 单芯片软件崩溃重启（$250.0\ \mathrm{ms}$）失控漂移 **$351.25\ \mathrm{mm}$**（偏离缩减 **$116.31\times$**）；
   - $90\ \mathrm{km/h}$ 场景（实验 19-14）：硬件看门狗偏离 **$2.50\ \mathrm{mm}$** vs. 软件重启（$200.0\ \mathrm{ms}$）偏离 **$190.00\ \mathrm{mm}$**（缩减 **$76.00\times$**）。

## 运行与验证

```bash
python3 experiments/ch20/20-03/run.py
python3 experiments/ch20/20-03/verify.py
```
