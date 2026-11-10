# 实验 13-05：百万册级复杂学术 PDF `MinerU`/`olmOCR` 混合 GPU 管线与 64k 代码仓库级 AST 拓扑排序装箱复算器

本实验对应第 13 章 **23.4.2 节（第五专题）、算例 23-6、算例 23-7、练习 23-11 与实验 23-12**：
1. 复算 $2,000,000$ 册（$5 \times 10^8$ 页，产出 $400\ \mathrm{B}$ 词元）理工科学术 PDF 在 `MinerU` 原生数字页正文直抽 + 跨页公式框 TensorRT 动态批处理（$p_{\mathrm{digital}} = 65.0\%, T_{\mathrm{dig}} = 11.00\ \mathrm{ms/page}$）与 `olmOCR` 扫描页连续批处理（$35.0\%, T_{\mathrm{scan}} = 30.20\ \mathrm{ms/page}$）混合双通路下的加权单页 GPU 耗时（`17.72 ms/page`）、单卡吞吐（`56.43 pages/s/GPU`）、相对单体 7B VLM 的 `35.27x` 加速比、总 GPU 小时数（`2,461.11 GPU-h`）与每百万词元成本（`$0.0135 / 1M tokens`）；
2. 复算 64k（$L_{\mathrm{ctx}} = 65,536$）代码预训练中，基于 `Tree-sitter` AST 跨文件依赖图提取、Tarjan 强连通分量（SCC）破环凝聚与 Kahn 拓扑排序装箱相对路径字母序（`42.50%`）和随机序（`50.00%`）将因果依赖覆盖率提升至 `95.00%`（`+52.50` 个百分点）、跨文件 API 调用困惑度降低 `39.85%`（`6.85 -> 4.12`）及窗口装箱利用率 `98.80%` 的完整账本。

## 运行与校验
```bash
python3 experiments/ch13/13-05/run.py
python3 experiments/ch13/13-05/verify.py
```
