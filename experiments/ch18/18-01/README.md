# 实验 18-01：40 轮自主智能体 WORM 前缀复用率、工具等待三态相图、四级 KV 全局路由（Dynamo/HiCache）与 CacheGen 流式压缩回本计算器

本实验复现第 18 章 21.1–21.2 节、算例 21-1、算例 21-2 与练习 21-1 至 21-5 的核心定量模型与算法：

1. **40 轮自主智能体（ReAct / SWE-Agent）的“一写多读（WORM）”上下文雪崩账本（算例 21-1）**：
   - 初始任务提示词 $L_0 = 2,000$ 词元，每轮生成动作 $L_{\mathrm{act}} = 200$ 词元、环境返回观测反馈 $\Delta L_{\mathrm{obs}} = 1,800$ 词元（每轮上下文净增 $\Delta L = 2,000$ 词元）；
   - 在 $K = 40$ 轮循环中，第 $40$ 轮输入长度达 $80,000$ 词元，累计预填充总词元达 **$1,640,000$ 词元**，其中重复读取历史前缀的词元高达 **$1,560,000$ 词元**，**WORM 前缀复用率达 $39/41 \approx \mathbf{95.12\%}$**，预填充与解码词元比高达 **$205.0 : 1$**。
2. **工具等待期（Tool-Wait Hiatus）三段式驻留/换出/重算相图（算例 21-1）**：
   - 对 $L_{\mathrm{prefix}} = 65,536$（$64\mathrm{K}$）词元前缀（Qwen3-8B GQA KV 体积 **$9.00\ \mathrm{GiB}$**），重算耗时 $500.0\ \mathrm{ms}$，PCIe 5.0（单向有效 $48.0\ \mathrm{GiB/s}$）单向换入耗时 **$187.5\ \mathrm{ms}$**（双向 $375.0\ \mathrm{ms}$）；
   - HBM 原地驻留与 Host DRAM 换出的成本盈亏平衡等待时间为 **$t_{\mathrm{offload}}^* = 2.16\ \mathrm{s}$**；在 $t_{\mathrm{wait}} = 15.0\ \mathrm{s}$ 工具等待下，分层换出节省 **$85.60\%$** 的 HBM 显存租金成本，且恢复首字延迟比重算快 **$2.67\times$**。
3. **四级 KV 缓存全局路由器（NVIDIA Dynamo / SGLang HiCache）与 `CacheGen` 流式残差熵编码（算例 21-2）**：
   - 全局 KV 感知路由器综合权衡节点排队延迟与前缀命中率，为 $64\mathrm{K}$ 请求选择节点 A（排队 $120\ \mathrm{ms}$ + $90\%$ 命中预填充 $50\ \mathrm{ms} = \mathbf{170.0\ \mathrm{ms}}$），相比朴素最少负载路由（节点 B，$510.0\ \mathrm{ms}$）将 TTFT 降低 **$3.00\times$**；
   - `CacheGen` 将 BF16 KV 缓存压缩至 $3.80\ \mathrm{bits/param}$（压缩比 **$4.21\times$**，体积从 $9.00\ \mathrm{GiB}$ 压至 **$2.1375\ \mathrm{GiB}$**），在 $50\ \mathrm{Gbps}$（有效 $5.81\ \mathrm{GiB/s}$）跨机架网络上耗时（含 $18.0\ \mathrm{ms}$ 解压）从 $1,549.1\ \mathrm{ms}$ 降至 **$385.9\ \mathrm{ms} < 500.0\ \mathrm{ms}$**（快 **$4.01\times$**），成功击败本地重算。

## 运行方式

```bash
python3 experiments/ch18/18-01/run.py
python3 experiments/ch18/18-01/verify.py
```
