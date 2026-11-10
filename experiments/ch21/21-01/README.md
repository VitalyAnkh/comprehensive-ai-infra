# 实验 21-01：万亿参数 DLRM 三级嵌入缓存（HBM-CXL-SSD）与 TorchRec 稀疏 All-to-All 重叠

本实验对应第 21 章 25.1 节算例 25-1 与算例 25-2，验证：
1. **10.0 TB 万亿参数稀疏嵌入表三级缓存（HBM 320 GB / CXL 2.0 TB / NVMe SSD 10.0 TB）**：
   - 在幂律分布 $\alpha = 1.12$ 与 15 分钟滑动窗口时间漂移因子 $\eta_{\mathrm{drift}} = 0.038$ 下，HBM 热表命中率 `82.0%`（延迟 `0.35 us`）、CXL 温表命中率 `15.5%`（延迟 `1.80 us`）、NVMe GDS 冷表穿透率 `2.5%`（延迟 `65.0 us`）；
   - 加权平均查表延迟从纯 SSD 的 `65.00 us` 压降至 `2.19 us`（等效吞吐提升 `29.68x` / `29.7x`）。
2. **64 卡 TorchRec 稀疏 All-to-All 查表与底端稠密 MLP 双流并发重叠**：
   - 热度感知分桶切分（Load-Balanced Sharding）将跨卡倾斜因子从随机哈希的 `2.85` 压降至 `1.06`，All-to-All 通信耗时从 `4.12 ms` 降至 `1.53 ms`，被底端 MLP 计算（`1.85 ms`）100% 掩盖（暴露通信延迟 `0.00 ms`）。

## 运行与验证

```bash
python3 experiments/ch21/21-01/run.py
python3 experiments/ch21/21-01/verify.py
```
