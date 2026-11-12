# 实验 16-02：DSec 38 万并发沙箱内存账本、virtio-pmem DAX 去重与审计日志中断排空仿真

本实验复现第 16 章 16.2.2–16.3.1 节与练习 16-10、16-13、16-14 的安全沙箱与可恢复审计账本：

1. **DeepSeek DSec 生产沙箱内存与核调度账本（算例 16-4, 16-5 & 练习 16-13）**：
   - 单 Cell `160` 节点支撑 `380,000` 个并发沙箱（单机平均 `2,375` 个）、峰值创建 `5,000` 个/秒；
   - `virtio-pmem` + DAX 消除宿主—客户机双重页缓存使峰值内存下降 $40.2\%$，但每 $4\ \mathrm{KiB}$ 页消耗 $64\ \mathrm{B}$ `struct page`（占比 $1/64$，即每 $64\ \mathrm{GB}$ 消耗 $1.0\ \mathrm{GB}$ 元数据，$128\ \mathrm{GB}$ 消耗 $2.0\ \mathrm{GB}$）；
   - 单节点运行 $N=800$ 个 microVM 共享一块 $64\ \mathrm{GB}$ 只读基础镜像时，传统独立页缓存（每个虚机平均在客户机页缓存中驻留 $4.0\ \mathrm{GB}$ 工具链热点页，外加宿主机 $64\ \mathrm{GB}$ 页缓存）共消耗 $64 + 800 \times 4.0 = 3,264\ \mathrm{GB}$ 内存（若全量缓存则高达 $800 \times 64 = 51,200\ \mathrm{GB}$）；启用 `virtio-pmem` + DAX 后，宿主机侧仅需 $64\ \mathrm{GB}$ 单份共享页缓存，每个 Guest 承担 $64 / 64 = 1.0\ \mathrm{GB}$ `struct page`（800 个虚机合计 $800\ \mathrm{GB}$，只读层总内存 $864\ \mathrm{GB}$），整机净节省 $3,264 - 864 = 2,400\ \mathrm{GB}$ 主机内存（若将只读镜像裁剪为 $16\ \mathrm{GB}$，则单 Guest `struct page` 仅 $256\ \mathrm{MiB}$，$800$ 个虚机总计 $200\ \mathrm{GB}$）。
2. **算例 16-6 审计日志后端中断积压与恢复排空**：
   - 稳态事件流 $R_{\mathrm{in}} = 10 \times 20 \times 2 = 400\ \mathrm{KiB/s}$；
   - 后端中断 $120\ \mathrm{s}$ 积压 $48,000\ \mathrm{KiB} = 46.875\ \mathrm{MiB}$（$32\ \mathrm{MiB}$ 缓冲在 $81.92\ \mathrm{s}$ 溢出）；
   - 恢复后最大写出速率 $1,024\ \mathrm{KiB/s}$ 扣除持续新到达流量后的净排空速率为 $624\ \mathrm{KiB/s}$，完全排空耗时 $48,000 / 624 \approx 76.9231\ \mathrm{s}$。

## 运行方式

```bash
python3 experiments/ch16/16-02/run.py
python3 experiments/ch16/16-02/verify.py
```
