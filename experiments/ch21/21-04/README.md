# 实验 21-04：ColPali 多向量视觉检索 PLAID 残差压缩与三路混合重排预算核算

本实验对应第 21 章 21.5 节算例 21-7 与算例 21-8，验证：
1. **ColPali 视觉文档晚交互（Late Interaction）存储膨胀与 PLAID 1-bit / 2-bit 残差压缩**：
   - 每页 PDF 编码为 $N_p = 1{,}030$ 个 $d = 128$ 维视觉 Patch 词元，朴素 `BF16` 占 `263.68 KB/page`（`5,000 万`页占 `13.18 TB`、`1 亿`页占 `26.37 TB`、`10 亿`页占 `263.7 TB`）；
   - `PLAID 2-bit`（`2 B` 质心 ID + `32 B` 残差 = `34.0 B/vec`）将 `5,000 万`页压缩 `7.53x` 至 `1.75 TB`；`PLAID 1-bit`（`1.5 B` 紧凑质心 ID + `16 B` 二值残差 = `17.5 B/vec`）压缩 `14.6x` 至 `901.3 GB`（`1 亿`页 `1.80 TB`、`10 亿`页 `18.0 TB`），配合质心初筛与融合 `MaxSim` 内核实现单查询 `P99 = 14.2 ms`。
2. **三路混合检索与交叉编码器（Cross-Encoder 0.6B）级联预算分配**：
   - `BM25 Block-Max WAND`（`6.4 ms`）与 `cuVS CAGRA`（`9.8 ms`）并发召回 $K_1 = 1{,}000$，经 `RRF` 融合去重（`0.8 ms`）后选取 $K_2 = 64$ 送入 `0.6B` 交叉编码器紧凑装箱重排（`14.6 ms`），核心检索重排耗时 `25.2 ms`，计入 `2.4 ms` 网关开销后端到端 `P99 = 27.6 ms < 30.0 ms`。

## 运行与验证

```bash
python3 experiments/ch21/21-04/run.py
python3 experiments/ch21/21-04/verify.py
```
