# 实验 13-04：分布式 KV Cache 前缀压缩预取（LMCache / Mooncake Store）与共享 NVMe 存储池 QoS 隔离仿真

本实验复现第 13 章 13.8.4 节“分布式 KV Cache 存储池与对象存储冷热分层（LMCache / Mooncake Store 跨请求 Prefix 压缩与 GDS 异步预取）”中的三组核心系统账本：

1. **LMCache / CacheGen 跨请求前缀 KV 压缩与逐层 GDS 异步预取流水线**：
   - 针对 $L = 64$ 层、上下文长度 $S = 65,536$（$64\mathrm{K}$）、每层每词元 KV 状态 $2,048\ \mathrm{B}$ 的长上下文请求，原始前缀 KV 体积为 $8,192.0\ \mathrm{MiB}$（$8.0\ \mathrm{GiB}$）；
   - 采用锚点差分量化与算术编码（压缩比 $\rho_{\mathrm{comp}} = 3.80\times$）后，落盘与传输体积降至 $2,155.79\ \mathrm{MiB}$（$2.105\ \mathrm{GiB}$）；
   - 按每 $G_{\mathrm{layer}} = 4$ 层切分为 $16$ 个流水块，在单盘 GDS 带宽 $12.5\ \mathrm{GiB/s}$、GPU 解压吞吐 $80.0\ \mathrm{GiB/s}$ 与每层计算 $1.15\ \mathrm{ms}$ 下，端到端恢复与首词元耗时从无压缩串行的 $713.60\ \mathrm{ms}$ 降至 **$273.02\ \mathrm{ms}$**（提速 **$2.61\times$**）。
2. **DRAM—NVMe—对象存储（S3 Express）三级冷热前缀生命周期与 TCO 账本**：
   - 在 $100\ \mathrm{TiB}$ 全局前缀池（DRAM 占 $5\%$、NVMe 占 $25\%$、低延迟对象存储占 $70\%$，命中率分别为 $45\%$、$40\%$、$15\%$）下，加权平均恢复耗时为 **$230.11\ \mathrm{ms}$**（相比 $1,850.0\ \mathrm{ms}$ 全量重算提速 **$8.04\times$**），月度存储硬件成本从全 DRAM 驻留的 $32,000\ \mathrm{USD}$ 降至 **$3,845.0\ \mathrm{USD}$**（节省 **$87.98\%$**）。
3. **共享 NVMe 存储池中训练检查点突发写与在线 KV 读取的队头阻塞（HoL）及硬件 WRR + 切片 QoS 治理**：
   - 单一 FIFO 队列下，$16$ 个未切片的 $1\ \mathrm{MiB}$ 检查点写请求（单次写耗时 $0.80\ \mathrm{ms}$）造成 $12.80\ \mathrm{ms}$ 队头阻塞，使 $1\ \mathrm{MiB}$（$16 \times 64\ \mathrm{KiB}$）KV 突发读的 P99 延迟飙升至 **$13.80\ \mathrm{ms}$**；
   - 启用 NVMe 硬件加权轮询（WRR，$w_{\mathrm{H}}:w_{\mathrm{L}} = 8:2$，保障 KV 读 $80\%$ 带宽份额即 $8.0\ \mathrm{GiB/s}$）并将检查点写切分为 $128\ \mathrm{KiB}$ 微块（单块非抢占等待仅 $0.10\ \mathrm{ms}$）后，KV 突发读 P99 延迟压降至 **$1.35\ \mathrm{ms}$**（尾延迟削减 **$10.22\times$**）。

## 运行方式

```bash
python3 experiments/ch13/13-04/run.py
python3 experiments/ch13/13-04/verify.py
```
