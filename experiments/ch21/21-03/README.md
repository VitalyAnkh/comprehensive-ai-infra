# 实验 21-03：十亿级向量索引 GPU cuVS CAGRA 建图加速与 DiskANN 4 KB 扇区对齐检索

本实验对应第 21 章 25.3 与 25.4 节算例 25-5 与算例 25-6，验证：
1. **10 亿（$10^9$）条 768 维 FP16 向量（`1.536 TB`）索引构建**：
   - 双路 128 核 CPU 构建 `HNSW`（$M=32, \mathrm{efConstruction}=200$）耗时 `11.50 hours`；
   - 8 卡 H100 运行 `cuVS CAGRA`（Warp Splitting 距离核 + 两阶段路径绕行剪枝 + 反向边注入）耗时仅 `0.68 hours`（`40.8 min`），实现 `16.91x` 建图提速。
2. **DiskANN 4 KB 磁盘扇区物理对齐与 io_uring 异步波束搜索**：
   - 主机 DRAM 常驻 `64 B/vec` PQ 压缩码（`10 亿`条仅占 `64.0 GB`）；
   - NVMe SSD 将每条原始向量（`1,536 B`）与 $R=64$ 个 `uint32` 邻居 ID（`256 B`）打包为 `1,792 B`（每 `4,096 B` 物理扇区容纳 `2` 个完整节点）；
   - 在波束宽度 $W=4$、跳数 $S=6$ 下单次查询仅触发 `24` 次 4 KB 随机读，在 4 块 PCIe 5.0 NVMe SSD（聚合 `6.0M IOPS`）上实现 `210,000 QPS`、Recall@10 = `95.4%`、P99 延迟 `2.85 ms`。

## 运行与验证

```bash
python3 experiments/ch21/21-03/run.py
python3 experiments/ch21/21-03/verify.py
```
