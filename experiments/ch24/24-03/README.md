# 实验 24-03：3FS 180 节点 6.6 TB/s 链式复制吞吐与 KTransformers 单卡 GPU + 双路 CPU AMX 异构 MoE 规划器

本实验复现第 24 章 24.3 与 24.4 节的核心量化结论：
1. **3FS & Smallpond**：`180` 台存储节点（`2,880` 块 NVMe SSD + `360` 张 200G RDMA 网卡）实现 **`6.60 TB/s`** 持续聚合随机读吞吐，GraySort `110.5 TiB` 仅耗时 **`30 min 14 s`（`1,814 s`）**；
2. **KTransformers**：单张 `24 GB` RTX 4090D（驻留 MLA 与共享专家 `19.2 GB`）配合双路 Xeon 8488C（`680.0 GB/s` DDR5 + AMX 驻留 `336.0 GB` INT4 路由专家），使 `DeepSeek-V3 671B` 预填充达 **`286.4 tok/s`**、解码达 **`18.6 tok/s`（较 llama.cpp 提速 `13.8x`）**。

## 运行与校验
```bash
python3 experiments/ch24/24-03/run.py
python3 experiments/ch24/24-03/verify.py
```
