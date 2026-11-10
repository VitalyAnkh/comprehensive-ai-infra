# 实验 16-03：机密推理恒定节拍分桶流量整形（Traffic Shaping）与 Freivalds / zkML 可验证推理承诺

本实验复现第 16 章 16.6.4 节新增专题（练习 16-17、练习 16-18）的侧信道流量整形与密码学可验证推理账本：

1. **恒定节拍词元释放（Tick-Locked Pacing）与 64-Token 分桶流量整形（练习 16-17）**：
   - 给定微批次真实输出词元长度 $\{19, 47, 83, 142, 215\}$（合计 $506$ 个词元），按 $B_{\mathrm{pad}} = 64$ 向上对齐后填充总词元数为 $704$（注入 $198$ 个哑元包，带宽税 $39.13\%$，整形总体积 $2.75\ \mathrm{MiB}$）；
   - 在原始平均 TPOT $\bar{t}_{\mathrm{tpot}} = 22.3\ \mathrm{ms}$、恒定节拍 $\Delta \tau_{\mathrm{tick}} = 25.0\ \mathrm{ms}$ 下，相对延迟税为 $12.11\%$，桶内时序与长度互信息泄露降为 $0.0\ \mathrm{bits}$。
2. **Freivalds 随机线性投影校验与专用 Sumcheck zkML 算力分析（练习 16-18）**：
   - 对 $4{,}096 \times 4{,}096 \times 14{,}336$ 的 FFN 矩阵乘（原生算力 $481.036\ \mathrm{GFLOPs}$），执行 $k = 40$ 轮 Freivalds 随机二元向量校验仅需 $10{,}737.418\ \mathrm{MFLOPs}$（占原生矩阵乘的 $2.2321\%$），伪造漏检率上界为 $2^{-40} \approx 9.09 \times 10^{-13}$；
   - 专用 `zkLLM` Sumcheck + `tlookup` 协议（$185\times$ 膨胀）较通用电路（$12{,}000\times$ 膨胀）提速 $64.86\times$。
3. **触发式单算子零知识故障证明（Single-Operator `zk-Fault-Proof`）**：
   - 将 $N_{\mathrm{op}} = 320$ 步模型的争议裁决耗时从传统 OPML 的 $9$ 轮交互（$45.0\ \mathrm{s}$，Gas 费 $42.0\ \text{美元}$）压缩至 $1.85\ \mathrm{s}$（提速 $24.32\times$，Gas 费降至 $4.83\ \text{美元}$，节省 $88.50\%$）。

## 运行方式

```bash
python3 experiments/ch16/16-03/run.py
python3 experiments/ch16/16-03/verify.py
```
