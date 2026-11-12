# 实验 16-05：去中心化算力池拜占庭梯度抽检（Proof-of-Learning）与异步伴飞护栏（Speculative Guardrail）模拟器

本实验对标第 16 章 16.4.2–16.5.1 节与算例 16-14、16-15、16-16、16-17，验证开放算力池 Proof-of-Learning 随机抽检捕获率（> 99.9%）、DHT 弹性 Spot 实例有效训练时间比（ETTR = 96.8%）、双流异步伴飞护栏（K=8 滑动窗口，暴露延迟 3.2 ms < 3.5 ms）以及残差流原位 SAE 正交投影钳位（每层耗时 3.6 us < 4 us，总开销 0.08%）。

## 运行方式

```bash
python3 experiments/ch16/16-05/run.py
python3 experiments/ch16/16-05/verify.py
```
