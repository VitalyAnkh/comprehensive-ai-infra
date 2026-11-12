# 实验 16-04：跨洲际广域网 Streaming DiLoCo（H=500 + FP4 伪梯度流水）2,000x 带宽压缩与零暴露开销模拟器

本实验对标第 16 章 16.4.1 节与算例 16-13，核算跨 4 个地理区域（RTT = 85 ms）联合训练 70B 模型（BF16 参数 140.0 GB，单步耗时 2.40 s）时，标准跨地域 AllReduce（700.0 Gbps）、基础 DiLoCo（H=500）与 Streaming DiLoCo + FP4 伪梯度流水（350.0 Mbps，带宽需求降低 2,000x，暴露等待时间 0.0 s，算力有效利用率 99.4%）的通信账本。

## 运行方式

```bash
python3 experiments/ch16/16-04/run.py
python3 experiments/ch16/16-04/verify.py
```
