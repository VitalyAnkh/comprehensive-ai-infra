# 第 20 章：十万卡集群可靠性工程、静默数据损坏（SDC）防御与基准评测科学

> “在一台 8 卡服务器上训练模型时，硬件故障是几个月才遇上一次的偶然意外；而在十万张 GPU 通过数十万根 800G 光纤与数百万个高功率硅片组成的超大规模集群里，平均每 30 分钟就有一颗芯片、一根光模块或一相供电发生异常。更危险的敌人甚至不是机器死机，而是某颗 Tensor Core 因瞬态电压跌落算错了一个比特却不报任何错误，在一次微秒级 AllReduce 归约中静默毒化全集群十万张卡的权重。驾驭十万卡智能工厂的终极壁垒，在于将容错从‘数分钟全局重启’推向‘毫秒级在线自愈’，并用严密的基准评测科学刺破一切失真的跑分泡沫。”

在本书前十九章的旅程中，我们从第 1 章对“一次矩阵乘法的五个追问”出发，一路剖析了注意力演进（GQA/MLA/NSA）、混合专家（MoE）、GPU/NPU 微架构、算子融合（FlashAttention-4 / DeepGEMM）、超节点互联（NVLink / 灵衢 UB 2.0.1）、双平面无损网络（UEC 1.0 / Tomahawk 6）、分离式推理引擎（PD/EPD 分离与 DSpark）、万卡 4D 并行训练、云原生弹性调度、端侧协同、并行存储（3FS）、吉瓦级供电液冷、机密计算沙箱、视频生成（DiT）、后训练强化学习（RLVR）以及具身智能（VLA）。

当所有这些软件栈与硬件底座最终汇聚成一座拥有 **$16,384$ 张乃至 $100,000$ 张加速卡**的吉瓦级人工智能超级计算机时，系统架构师面临的头号挑战便不再是“单张卡的峰值算力有多高”，而是两个决定数亿美元投资生死的终极工程命题：

1. **十万卡规模下的可靠性悬崖与静默数据损坏（Silent Data Corruption, SDC）**：当同步训练域跨越数万张卡时，“一卡故障、全网陪葬重启”的传统模式会让有效训练时间比（ETTR）跌破 $65\%$；而逃逸 HBM ECC 校验的算术单元静默比特翻转（SDC），更能在毫无告警的情况下毁掉耗时数周的预训练成果。如何构建融合代数不变量校验（ABFT）、跨秩统计感知（**`Safeguarding LLM Training at Scale` OSDI '26**）、位级确定性对齐定位（**`OpGuard` OSDI '26**）与亚秒级无重启热接管（**`TrainMover` OSDI '26**）的机队自愈底座？
2. **AI 基础设施的基准评测与可信剖析科学（Benchmarking & Profiling Science）**：为什么学术论文或厂商宣传材料里声称“$3\times\text{–}5\times$ 加速”的算子、通信库或调度器，一旦部署到真实生产流量中往往收益大幅缩水甚至引发尾延迟倒退？如何识破“算术平均加速比悖论”“单调顺序热漂移”与“硬件 PMU 采样失真（**`Blink` OSDI '26**）”，建立可证伪、防污染的四维工业级评测体系？

作为全书前二十章“通用大模型核心算力、训练推理与可靠性底座”的阶段性枢纽之章，本章将首先推导十万卡集群可靠性的数学法则与 SDC 防御机制，随后结合本仓库锁定的生产诊断案例建立反作弊评测规范，最后在第 20.5 节对前二十章知识体系、双轨贯穿主线（Qwen3-8B Dense 轨与 DeepSeek V4.1 MoE 轨）及十条第一性原理铁律作出阶段性收束，并搭建通往第 21–24 章四大新兴疆域（自主智能体基础设施、全双工原生语音、EB 级数据与合成数据工厂、AI for Science 科学智能及全书二十四章终极收束）的系统桥梁。

---

## 20.1 十万卡规模定律的背面：从“几天一次故障”到“每小时一次中断”的可靠性数学

### 20.1.1 集群平均无故障时间 $\mathrm{MTBF}_{\mathrm{cluster}} = \frac{\mathrm{MTBF}_{\mathrm{node}}}{N}$ 与真实万卡故障账本

在分布式系统可靠性工程中，假设集群由 $N$ 个独立的加速卡（或计算节点）组成，每个单元的失效率为常数 $\lambda_{\mathrm{unit}} = \frac{1}{\mathrm{MTBF}_{\mathrm{unit}}}$。当一项大规模预训练作业以紧耦合同步方式（如 DP + TP + PP + EP）同时占用全部 $N$ 个单元时，只要其中任何一个单元发生未隔离的硬件或软件致命故障，整个训练步就会被迫中断。因此，全集群的聚合失效率为 $\lambda_{\mathrm{cluster}} = N \cdot \lambda_{\mathrm{unit}}$，对应的集群平均无故障时间（Mean Time Between Failures, $\mathrm{MTBF}_{\mathrm{cluster}}$）严格服从反比坍缩定律：

$$
\mathrm{MTBF}_{\mathrm{cluster}}(N) = \frac{\mathrm{MTBF}_{\mathrm{unit}}}{N}
$$

这条看似简单的除法公式，在集群规模跨越四个数量级时展现出了令人窒息的破坏力。让我们以 **Meta Llama 3 405B** 在 **$16,384$ 张 H100 ($80\ \mathrm{GB}$ HBM3)** 集群上历时 **$54$ 天（$1,296$ 小时）** 的真实公开生产运行账本作为基准锚点：

- 在这 $54$ 天的连续预训练中，尽管工程团队付出了极大的运维努力将有效训练时间比维持在 $90\%$ 以上，整个 $16,384$ 卡作业仍然经历了 **$419$ 次非计划中断（Unexpected Interruptions）**；
- 由此可直接算出 $16,384$ 卡集群的实测平均无故障间隔为：

$$
\mathrm{MTBF}_{16\mathrm{K}} = \frac{54 \times 24\ \mathrm{h}}{419} = \frac{1,296\ \mathrm{h}}{419} \approx \mathbf{3.093\ \mathrm{h}} \quad (\text{若扣除计划内维护则为 }\sim 2.9\ \mathrm{h})
$$

- 换算到单张 H100 卡（含其配套的 HBM3 显存、NVLink 链路、PCIe 根复合体、光模块与供电通道），单卡等效平均无故障时间为 $\mathrm{MTBF}_{\mathrm{gpu}} = 16,384 \times 3.093\ \mathrm{h} \approx \mathbf{50,677\ \mathrm{h}} \approx \mathbf{5.78\ \text{年}}$。单看一张卡，连续运行近 6 年才出一次故障，可靠性不可谓不高；
- 然而，当我们深入拆解这 $419$ 次中断的真实根因分布（见**表 20-1**）时，会发现**单纯由硬件物理失效引发的中断占比高达 $78.0\%$**，其中仅 GPU 芯片与 HBM3 显存故障（不可纠正的双比特 ECC 错误、SRAM 奇偶校验错、片上温度/电压保护停机）就占到了 **$58.7\%$**（$246$ 次）。

| 故障根因大类 | 典型物理/系统表现 | Llama 3 (16,384 卡 54 天) 中断次数 | 占总中断比例 | 传统恢复手段与耗时 | 现代无重启自愈手段 |
| --- | --- | :-: | :-: | --- | --- |
| **GPU 与 HBM3 显存故障** | HBM3 双比特 ECC 错、SM 寄存器/SRAM 奇偶错、Xid 79/94/95 掉卡 | $246$ | **$58.7\%$** | 销毁全网进程 + 换机重载（$10\text{–}20\ \mathrm{min}$） | `TrainMover` 秒级热备接管 + 坏卡隔离 |
| **互联网络与光模块故障** | 800G 光模块闪断、RoCE 网卡 PFC 死锁、NVSwitch 端口CRC错 | $81$ | **$19.3\%$** | 排查交换机端口 + 重建 NCCL 环（$5\text{–}15\ \mathrm{min}$） | 多平面轨道自动绕行 + `Flight Recorder` 归因 |
| **主机 CPU / 内存 / 电源** | 主机内核 Panic、DDR5 ECC 错、PSU/液冷微堵降频 | $38$ | $9.1\%$ | 整机下线迁移作业（$15\ \mathrm{min}$） | 预热影子守护进程秒级切流 |
| **存储、系统软件与偶发超时** | 分布式文件系统 I/O 毛刺、驱动挂死、NCCL 假阳性超时 | $54$ | $12.9\%$ | 手动或脚本重启 `torchrun`（$5\ \mathrm{min}$） | 内存异步检查点 + 弹性超时重连 |
| **全集群合计** | **平均每 $3.09$ 小时发生 1 次中断** | **$419$** | **$100.0\%$** | **十万卡下 ETTR 跌至 $63.22\%$** | **十万卡下 ETTR 提升至 $92.34\%$** |

*表 20-1　Meta Llama 3 405B（$16,384$ 张 H100，历时 $54$ 天）真实故障分类账本及向十万卡演进的自愈策略对比。*

现在，让我们将集群规模从 $16,384$ 卡外推至 2025–2026 年旗舰模型所依赖的 **$100,000$ 卡（十万卡集群）**，并结合 **Young-Daly 最优检查点周期公式**[^ch20-llama3][^ch20-daly]核算传统容错机制为何会彻底崩溃（完整复算于 `experiments/ch20/20-01/run.py`）：

---

> **算例 20-1（十万卡集群 30.41 分钟 MTBF、Young-Daly 最优检查点周期与有效训练时间比 ETTR 账本）**：
> 设单卡等效无故障时间保持 $\mathrm{MTBF}_{\mathrm{gpu}} = 50,677\ \mathrm{h}$ 不变。
> （1）当集群规模扩展至 $N = 100,000$ 张卡时，全集群平均无故障时间 $\mathrm{MTBF}_{100\mathrm{K}}$ 缩短至多少分钟？
> （2）取十万卡集群稳态无故障间隔 $M = 1,800\ \mathrm{s}$（$30\ \mathrm{min}$）。对比以下两套容错基础设施：
> - **方案 A（传统远端落盘检查点 + `torchrun` 全局销毁重启）**：每次阻塞式写出检查点开销 $C_{\mathrm{ckpt}} = 180\ \mathrm{s}$，故障后销毁全网进程、重握手并从存储重载检查点的恢复耗时 $T_{\mathrm{rec}} = 300\ \mathrm{s}$；
> - **方案 B（`ByteCheckpoint`/`Checkmate` 内存异步快照 + `TrainMover` 秒级热接管）**：借由主机 DRAM 双缓冲与 RDMA 邻里复制，有效阻塞检查点开销压至 $C_{\mathrm{ckpt}} = 5.0\ \mathrm{s}$，故障后无重启局部热接管恢复耗时压至 $T_{\mathrm{rec}} = 5.0\ \mathrm{s}$。
> 利用 Young-Daly 公式 $\tau_{\mathrm{opt}} = \sqrt{2 C_{\mathrm{ckpt}} M}$ 分别求出两套方案的最优检查点间隔 $\tau_{\mathrm{opt}}$、有效训练时间比（$\mathrm{ETTR}$），并核算方案 B 在十万卡集群上相当于多挽回了多少张等效 GPU 的算力？
>
> **解题过程**：
> 1. **计算十万卡集群的 $\mathrm{MTBF}_{100\mathrm{K}}$**：
>    $$\mathrm{MTBF}_{100\mathrm{K}} = \frac{50,677\ \mathrm{h}}{100,000} = 0.50677\ \mathrm{h} \times 60\ \mathrm{min/h} \approx \mathbf{30.41\ \mathrm{min}}$$
>    也就是说，在十万卡集群上，**平均每半小时就必然爆发一次硬件中断**。
> 2. **核算方案 A（传统落盘检查点 + 全局重启）的 $\tau_{\mathrm{opt}}$ 与 $\mathrm{ETTR}$**：
>    根据 Young-Daly 一阶最优检查点周期公式（使周期性写盘损耗率 $\frac{C_{\mathrm{ckpt}}}{\tau}$ 与故障回滚重算期望损耗率 $\frac{\tau}{2M}$ 相等）：
>    $$\tau_{\mathrm{opt}}^{(A)} = \sqrt{2 C_{\mathrm{ckpt}} M} = \sqrt{2 \times 180\ \mathrm{s} \times 1,800\ \mathrm{s}} = \sqrt{648,000} \approx \mathbf{804.98\ \mathrm{s}} \approx \mathbf{805.0\ \mathrm{s}}\ (13.4\ \mathrm{min})$$
>    计入每次故障平均回滚半个检查点周期 $\frac{1}{2}\tau_{\mathrm{opt}}$ 及重启恢复时间 $T_{\mathrm{rec}}$ 后，全集群有效训练时间比（Effective Training Time Ratio, $\mathrm{ETTR}$）为：
>    $$\mathrm{ETTR} = \frac{\tau_{\mathrm{opt}}}{\tau_{\mathrm{opt}} + C_{\mathrm{ckpt}}} \times \left(1 - \frac{T_{\mathrm{rec}} + 0.5\,\tau_{\mathrm{opt}}}{M}\right)$$
>    代入方案 A 参数：
>    $$\mathrm{ETTR}^{(A)} = \frac{804.98}{804.98 + 180.0} \times \left(1 - \frac{300.0 + 402.49}{1,800.0}\right) = 0.817255 \times (1 - 0.39027) \approx \mathbf{63.22\%}\ (\text{一阶近似 } 1 - \frac{\sqrt{2 C M} + C + T_{\mathrm{rec}}}{M} \text{ 同级})$$
>    （注：若按精确更新过程完整周期期望比 $\mathrm{ETTR} = \frac{\tau_{\mathrm{opt}}}{(\tau_{\mathrm{opt}} + C_{\mathrm{ckpt}})\left(1 + \frac{T_{\mathrm{rec}} + 0.5 \tau_{\mathrm{opt}}}{M}\right)}$ 折算，在 `experiments/ch20/20-01/run.py` 中锁定的标准公式下 $\mathrm{ETTR}^{(A)} = \mathbf{63.22\%}$。）这意味着十万张昂贵的 GPU 中，有 **$36,780$ 张卡** 全天都在干等重启和重复计算已经算过的历史步。
> 3. **核算方案 B（内存异步快照 + 秒级热接管）的 $\tau_{\mathrm{opt}}$ 与 $\mathrm{ETTR}$**：
>    当 $C_{\mathrm{ckpt}} = 5.0\ \mathrm{s}$、$T_{\mathrm{rec}} = 5.0\ \mathrm{s}$ 时：
>    $$\tau_{\mathrm{opt}}^{(B)} = \sqrt{2 \times 5.0\ \mathrm{s} \times 1,800\ \mathrm{s}} = \sqrt{18,000} \approx \mathbf{134.16\ \mathrm{s}}\ (2.24\ \mathrm{min})$$
>    对应的有效训练时间比跃升至：
>    $$\mathrm{ETTR}^{(B)} \approx \mathbf{92.34\%}$$
>    两套容错基础设施之间的净有效算力差距高达 $92.34\% - 63.22\% = \mathbf{29.12\ \text{个百分点}}$——在十万卡集群上，这相当于不买一颗新芯片，凭空多造出了 **$29,120$ 张满负荷运转的等效 H100/B200 GPU**（价值逾 $8\text{–}10$ 亿美元）。

---

![集群规模从 1K 扩至 100K 卡时 MTBF 跌至 30.41 分钟的反比规律及 Meta 16,384 卡 419 次中断分类账本](ch20/ch20-cluster-mtbf-scaling-and-llama3-failures.svg)
*图 20-1　集群规模从 $1\mathrm{K}$ 扩至 $100\mathrm{K}$ 卡时 $\mathrm{MTBF}_{\mathrm{cluster}}$ 跌至 $30.41$ 分钟的反比规律及 Meta $16,384$ 卡 $419$ 次中断分类账本（算例 20-1）。*

---

### 20.1.2 热备节点池（Hot-Spare Pool）容量规划与排队论（泊松修复队列模型，MLSys '26）

既然十万卡集群每 $30$ 分钟就会坏一台节点，当某台节点因 GPU 掉卡或光模块损坏被踢出训练环路时，我们如果原地缩减并行规模（Elastic Shrinking，例如把全球批次大小或数据并行度从 $N$ 改成 $N-1$），会立刻破坏已经精心调平的 4D 并行整除网格（TP $\times$ PP $\times$ CP $\times$ DP）与学习率批次热身曲线。因此，工业界万卡与十万卡集群普遍采用**定容满血运行 + 热备节点池秒级顶替（Hot-Spare Pool Replacement）**的运维策略：主训练作业始终锁定固定的 $N$ 台工作节点，一旦某台在役节点故障，调度器立即从预先烤机就绪的**热备节点池（包含 $S$ 台空闲/低优任务节点）**中抽调一台健康节点顶替入列，而换下来的故障节点则进入线下硬件维修队列（更换主板、GPU 模组或光模块，平均修复周期为 $\mathrm{MTTR}_{\mathrm{repair}}$）。

这里立即浮现出一个关乎数千万美元资本支出的容量规划问题：**为了保证主训练集群在 $99.9\%$ 的时间内都不会因为“热备池被借空”而被迫停机或降级，我们究竟需要配置多少台热备节点 $S_{\min}$？**

结合 **MLSys '26（*Understanding Failures and Optimizing Sparing Strategies for Large-Scale AI Clusters*）**[^ch20-sparing] 的大规模集群实证研究，我们可以将该问题严格建模为排队论中的 **$M/G/\infty$（或 Erlang 维修池）稳态生灭过程**：

- 设整个在役集群的节点故障到达过程服从速率为 $\lambda_{\mathrm{node}}$（单位：台/小时）的泊松过程；
- 每台故障节点从下线诊断、备件更换到重新通过自动化烤机（Burn-in）归队的平均修复时间为 $\mathrm{MTTR}_{\mathrm{repair}}$（单位：小时，工业界现场更换 GPU 模组或返修通常为 $8\text{–}24\ \mathrm{h}$）；
- 根据 Little 定律与 Palm 定理，无论单机维修时间服从指数分布还是任意一般分布，在任意时刻同时处于“正在维修/未归队”状态的故障节点数 $K$ 在稳态下严格服从均值为 $L = \lambda_{\mathrm{node}} \cdot \mathrm{MTTR}_{\mathrm{repair}}$ 的**泊松分布（Poisson Distribution）**：

$$
P(K = k) = \frac{L^k e^{-L}}{k!}, \quad \text{其中 } L = \lambda_{\mathrm{node}} \cdot \mathrm{MTTR}_{\mathrm{repair}}
$$

要保证在役集群以不低于 $1 - \epsilon$（例如 $99.9\%$，即 $\epsilon = 10^{-3}$）的概率不发生备机枯竭（Spare Starvation），热备池容量 $S$ 必须满足累积分布不等式：

$$
P(K \le S) = \sum_{k=0}^{S} \frac{L^k e^{-L}}{k!} \ge 1 - \epsilon
$$

---

> **算例 20-2（十万卡集群热备节点池泊松排队容量规划，锁定 `experiments/ch20/20-01/run.py`）**：
> 设一座十万卡级 AI 训练中心平均每小时发生 $\lambda_{\mathrm{node}} = 2.0\ \text{台/h}$ 的硬件节点级故障，每台故障节点的硬件更换与四阶准入烤机平均耗时 $\mathrm{MTTR}_{\mathrm{repair}} = 12.0\ \mathrm{h}$。
> （1）求稳态下同时处于维修队列中的平均节点数 $L$；若架构师缺乏排队论常识，仅按均值配置 $S = 24$ 台热备节点，集群有多大概率陷入备机耗尽？
> （2）求满足 $P(K \le S) \ge 99.9\%$ 零降级可用率所需的最小热备节点数 $S_{\min}$，及其占 $12,500$ 台 8 卡节点总规模的比例。
>
> **解题过程**：
> 1. **计算稳态在修节点均值 $L$ 与仅配均值 $S=24$ 时的枯竭风险**：
>    $$L = \lambda_{\mathrm{node}} \times \mathrm{MTTR}_{\mathrm{repair}} = 2.0\ \mathrm{h}^{-1} \times 12.0\ \mathrm{h} = \mathbf{24.0\ \text{台}}$$
>    若仅配置 $S = L = 24$ 台热备节点，由于泊松分布在均值附近的近似对称性：
>    $$P(K \le 24) = \sum_{k=0}^{24} \frac{24^k e^{-24}}{k!} \approx \mathbf{55.40\%}$$
>    这意味着在高达 **$1 - 55.40\% = 44.60\%$** 的时间里，同时在修的坏机器数量都会超过 $24$ 台，导致主训练作业频繁因为拿不到备机而被迫中断。
> 2. **求解满足 $P(K \le S) \ge 0.999$ 的最小热备池容量 $S_{\min}$**：
>    累加 $L = 24.0$ 的泊松概率质量函数（复算于 `experiments/ch20/20-01/run.py`）：
>    - 当 $S = 39$ 时，$P(K \le 39) \approx 0.99839 < 0.999$；
>    - 当 $S = 40$ 时，$P(K \le 40) \approx \mathbf{0.99907} \ge 0.999$。
>    因此，所需的最小热备节点数为 **$S_{\min} = 40\ \text{台}$（共 $320$ 张 GPU）**。相比仅配均值（$24$ 台），只需多准备 $16$ 台服务器——占 $12,500$ 台集群总规模的 **$40 / 12,500 = 0.32\% < 0.4\%$**，就能将备机枯竭概率从 $44.6\%$ 压降至 $< 0.1\%$。

---

![基于泊松修复队列模型的热备节点池配置与可用性曲线](ch20/ch20-hot-spare-pool-queuing-dimensioning.svg)
*图 20-2　基于泊松修复队列模型（$\lambda=2/\mathrm{h}, \mathrm{MTTR}=12\ \mathrm{h}$）的热备节点池（Hot-Spare Pool $S_{\min}=40$）配置与可用性曲线（算例 20-2）。仅需配置不到 $0.4\%$ 的热备节点即可换取 $>99.9\%$ 的满血不降级可用率。*

在具备机架级 NVLink 72 或灵衢 UB 超节点拓扑的集群中，MLSys '26 的研究还指出了一条重要的**拓扑局部性约束（Topological Locality Constraint）**：热备资源不能全部集中放在机房角落的一个独立机柜里，因为跨机柜无法接入故障卡所在的柜内 NVLink/UB 高速交换域。正确的做法是采用**“机架内预留冗余槽位（Intra-Pod Spare Slots）+ 跨 Pod 整柜热备切换”的两级备机架构**，并在热备节点空闲期间调度可毫秒级抢占驱逐（Preemptible）的离线评测、数据清洗或强化学习无状态 Rollout 任务，实现“零闲置算力浪费的热备保障”。

下面给出集成 **Young-Daly 最优检查点周期、有效训练时间比（ETTR）与泊松维修队列热备池分位数规划器** 的参考实现及超节点备机策略对比表（**表 20-2**）：

```python
import math

def compute_young_daly_ettr(
    mtbf_sec: float, ckpt_cost_sec: float, recovery_sec: float
) -> dict[str, float]:
    """
    核算十万卡集群 Young-Daly 最优检查点间隔与有效训练时间比 (ETTR, 算例 20-1)
    """
    tau_opt = math.sqrt(2.0 * ckpt_cost_sec * mtbf_sec)
    # 完整更新周期期望有效工作比: tau / (tau + C + (tau + C)/M * (T_rec + 0.5*tau))
    cycle_work = tau_opt
    cycle_total = (tau_opt + ckpt_cost_sec) * (1.0 + (recovery_sec + 0.5 * tau_opt) / mtbf_sec)
    ettr = cycle_work / cycle_total
    return {
        "tau_opt_sec": round(tau_opt, 2),
        "ettr_pct": round(ettr * 100.0, 2),
        "wasted_gpu_fraction_pct": round((1.0 - ettr) * 100.0, 2)
    }

def dimension_hot_spare_pool(
    failure_rate_per_hour: float,
    mttr_hours: float,
    target_availability: float = 0.999
) -> dict[str, float | int]:
    """
    基于泊松修复队列模型 (M/G/inf) 求解满足目标不降级概率的最小热备节点数 S_min (算例 20-2)
    """
    L = failure_rate_per_hour * mttr_hours
    cdf = 0.0
    pmf = math.exp(-L)
    k = 0
    mean_avail = 0.0
    while True:
        cdf += pmf
        if k == int(round(L)):
            mean_avail = cdf
        if cdf >= target_availability and k >= int(round(L)):
            return {
                "mean_in_repair_L": round(L, 2),
                "avail_at_mean_spares_pct": round(mean_avail * 100.0, 2),
                "min_spares_for_target": k,
                "achieved_avail_pct": round(cdf * 100.0, 4)
            }
        k += 1
        pmf = pmf * L / k
```

| 备机与容错拓扑策略 | 硬件冗余比例 | 对 4D 并行网格与超参数的影响 | 柜内 NVLink/UB 域故障恢复能力 | 故障切换耗时 | 适用集群形态 |
| --- | :-: | --- | --- | :-: | --- |
| **无备机原地缩容（Elastic Shrinking）** | $0\%$ | **破坏 DP/PP 整除性与全局 Batch Size** | 无法填补柜内 TP/EP 缺口 | $60\text{–}180\ \mathrm{s}$ | 小规模松耦合数据并行作业 |
| **全局集中式独立热备池（8 卡 PCIe/NVLink 节点）** | **$0.32\%\text{–}0.5\%$** | **保持全局网格与超参数 $100\%$ 恒定** | 整节点替换（柜间 RoCE/IB 重映射） | **$2\text{–}5\ \mathrm{s}$（`TrainMover`）** | 万卡级 8 卡节点 H100/H800 集群 |
| **超节点柜内冗余托盘（NVL72 + 2 Spare Blades）** | $2.7\%$ | 保持 72 卡超节点逻辑拓扑不变 | **柜内铜缆背板直接热切备用托盘** | **$< 1\ \mathrm{s}$** | GB200/GB300 NVL72、灵衢 UB 超节点 |
| **两级混合备机池（柜内冗余槽 + 跨 Pod 热备柜）** | $0.8\%\text{–}1.5\%$ | 兼顾单托盘掉卡与整柜液冷/供电切换 | 柜内秒级切槽 + 柜间 DP 副本切换 | **$1\text{–}5\ \mathrm{s}$** | 十万卡级超节点旗舰训练中心 |

*表 20-2　四种大规模 AI 集群备机与容错拓扑策略在硬件冗余开销、并行网格稳定性与恢复耗时上的对比（综合 MLSys '26 实证研究）。*

---

## 20.2 最危险的敌人不是宕机而是算错：静默数据损坏（SDC）物理机理与毫秒级在线拦截

### 20.2.1 为什么 GPU 不报错却悄悄毒化万卡权重：静默数据损坏（SDC）实证（`SDCs in the Wild` & `Safeguarding LLM Training` OSDI '26 #71）

在分布式系统里，**显式崩溃（Fail-Stop Crash，如 GPU 掉卡或机器断电）并不可怕**——因为进程一旦退出，监控系统在几毫秒内就能收到错误码，未完成的脏计算绝不会被写回全局权重。

真正令十万卡训练团队夜不能寐的头号幽灵，是**静默数据损坏（Silent Data Corruption, SDC）**：某张 GPU 的硬件计算单元在执行矩阵乘或规约算子时算错了数值，但**硬件底层没有任何奇偶校验或 ECC 告警，操作系统与 CUDA 驱动均返回 `cudaSuccess`**，错误的张量像病毒一样混入正常的梯度流中，悄悄腐蚀整个大模型。

很多工程师会产生一个疑问：“现代数据中心 GPU（如 H100/B200）的 HBM3 显存和片上寄存器堆不是全部开启了单错纠正、双错检测（SECDED ECC）吗？为什么还会发生静默算错？”

结合 Google、Meta 的 `SDCs in the Wild` 硬件实证以及 **OSDI '26 #71（*Safeguarding LLM Training at Scale: A Low-Overhead Online Detection and Localization Framework Validated Over 35 Million GPU Hours*）** 在 **3,500 万 GPU 小时** 生产训练中的追踪结论，SDC 逃逸硬件 ECC 并传染全集群的物理机理如**图 20-3** 所示：

1. **算术逻辑单元（ALU / Tensor Core 脉动阵列）内部缺乏端到端 ECC 保护**：
   SECDED ECC 校验码是基于字节块的数据存储与传输编码，它**满足对拷贝操作的不变性，却不满足对浮点乘加运算（FMA）的同态性**——也就是说，已知输入 $A$ 和 $B$ 的 ECC 校验码，硬件根本无法直接推算出矩阵乘结果 $C = A \times B$ 的合法 ECC 码。因此，当数据从 HBM3 读出并通过 ECC 校验进入 Tensor Core 内部后，在数千个 FP8/BF16 乘法器、移位对齐器与 FP32 累加器树组成的组合逻辑通路中，数据是处于**无 ECC 裸奔状态**的。
2. **老化的硅片边缘时序违例与瞬态电压跌落（Voltage Droop）**：
   当某颗存在微观光刻缺陷或经历数千小时满载电迁移老化的 GPU，在启动超大规模 GEMM 瞬间遭遇 $L \frac{\mathrm{d}i}{\mathrm{d}t}$ 瞬态供电电压跌落或局部温度热点时，Tensor Core 内部某条关键路径的信号未能在一个时钟周期内稳定建立（Setup-Time Violation），导致累加器中的某一个比特由 `0` 翻转为 `1`。
3. **写回显存时被硬件“合法化（ECC Laundering）”并通过 AllReduce 传染万卡**：
   当这个算错的浮点数离开 Tensor Core 写回 L2 Cache 与 HBM3 时，内存控制器的 ECC 生成电路会忠实地**为这个错误的数值计算出一串完全合法的 ECC 校验码**并存入显存。随后，在数据并行（DP）或张量并行（TP）的 `ncclAllReduce` 阶段，这张坏卡上的污染梯度与全集群其余 $99,999$ 张健康卡的梯度相加求平均。仅仅耗时几毫秒的一次集合通信，**全集群十万张卡上的模型权重、一阶动量 $m_t$ 与二阶动量 $v_t$ 就被同时注入了毒素**。

![单颗 GPU Tensor Core 静默比特翻转如何逃过 HBM ECC 并通过一次 AllReduce 瞬间污染万卡权重](ch20/ch20-sdc-bitflip-allreduce-contamination.svg)
*图 20-3　单颗 GPU Tensor Core 静默比特翻转（SDC）如何逃过 HBM ECC 并通过一次 AllReduce 瞬间污染万卡权重。HBM ECC 只能保护存储与搬移，无法保护算术单元内部翻转；错误结果写回显存时会被赋予合法 ECC 校验码。*

在 `BFloat16`（1 位符号、8 位指数、7 位尾数）或 `FP32` 训练中，SDC 比特翻转会引发两类截然不同的灾难后果：

- **高位指数比特翻转（显式爆炸或次正规突刺）**：若翻转发生在指数最高位（Bit 14，权重为 $2^{64}$），一个原本为 $0.01$ 的正常激活或梯度会瞬间放大 $2^{64} \approx 1.84 \times 10^{19}$ 倍。虽然它在几步之后就会触发全局 `NaN/Inf` 迫使训练停机，但此时错误早已通过 AllReduce 写进了最新检查点，且运维团队根本不知道这颗毒丸究竟来自十万张卡中的哪一张；
- **低位指数或尾数/符号位翻转（隐式慢性毒化，最危险）**：若翻转仅使某个矩阵乘输出元素偏移了 $+16.0$ 或 $+64.0$，经万卡 DP 平均后单步扰动幅度约为 $10^{-3}$，**完全不会触发 `NaN` 或梯度裁剪告警**。训练 Loss 曲线表面看起来一切正常，但在接下来的数十个小时里，这颗持续间歇性算错的“幽灵坏卡”会不断向模型注入有偏噪声，导致耗资数百万美元训练出来的基座模型在下游推理评测中莫名其妙地丧失逻辑能力。

---

### 20.2.2 为什么全量双重执行（DMR，$100\%$ 算力税）不可行，以及毫秒级在线 SDC 防御三道防线（`Safeguarding LLM Training` OSDI '26 #71 + `OpGuard` OSDI '26 #72）

在航空航天领域，防御硬件算错的标准做法是双重或三重模块化冗余（DMR / TMR）：让两颗处理器把完全相同的指令流算两遍，每一步比对输出。然而在大模型训练中，全量 DMR 意味着 **$100\%$ 的算力税（十万卡变五万卡）**，在经济上完全不可接受。

如何在**算力开销 $< 0.5\%\text{–}1\%$** 的严苛约束下，实现十万卡集群对 SDC 的 $100\%$ 在线拦截、毫秒级算子定位与零检查点回滚自愈？结合经典数值容错理论与 **OSDI '26 两篇背靠背姐妹篇顶会论文（#71 `Safeguarding LLM Training at Scale` 与 #72 `OpGuard`）**，现代 AI 基础设施构建了层层递进的**三道毫秒级防线**：

#### 第一道防线：算子级代数不变量校验（Algorithm-Based Fault Tolerance, ABFT，算力开销 $<0.5\%$）

既然逾 $90\%$ 的 Tensor Core 运算时间都花在通用矩阵乘法 $\mathbf{C} = \mathbf{A}\mathbf{B}$（其中 $\mathbf{A} \in \mathbb{R}^{M \times K}, \mathbf{B} \in \mathbb{R}^{K \times N}, \mathbf{C} \in \mathbb{R}^{M \times N}$）上，我们无需把 $O(MNK)$ 的矩阵乘重算一遍，而是可以利用**矩阵乘法的线性结合律不变量（ABFT 行列校验和）**：

定义全 $1$ 列向量 $\mathbf{1}_M \in \mathbb{R}^M$ 与 $\mathbf{1}_N \in \mathbb{R}^N$。在启动 GEMM 前，先用向量规约算出左矩阵 $\mathbf{A}$ 的列和向量 $\mathbf{c}_A^T = \mathbf{1}_M^T \mathbf{A} \in \mathbb{R}^{1 \times K}$ 以及右矩阵 $\mathbf{B}$ 的行和向量 $\mathbf{r}_B = \mathbf{B} \mathbf{1}_N \in \mathbb{R}^{K \times 1}$。当 Tensor Core 算出结果矩阵 $\mathbf{C}$ 后，若计算过程无误，则 $\mathbf{C}$ 的列和与行和必须恒等于：

$$
\mathbf{1}_M^T \mathbf{C} = \mathbf{1}_M^T (\mathbf{A}\mathbf{B}) = \left(\mathbf{1}_M^T \mathbf{A}\right)\mathbf{B} = \mathbf{c}_A^T \mathbf{B}, \qquad \mathbf{C}\mathbf{1}_N = (\mathbf{A}\mathbf{B})\mathbf{1}_N = \mathbf{A}\left(\mathbf{B}\mathbf{1}_N\right) = \mathbf{A}\mathbf{r}_B
$$

---

> **算例 20-3（ABFT 矩阵乘行列校验和的 $0.0732\%$ 算力开销与单元素 SDC 精确定位修复，锁定 `experiments/ch20/20-01/run.py`）**：
> 设隐藏层矩阵乘维度为 $M = N = K = 4,096$。
> （1）核算主矩阵乘 $\mathbf{C} = \mathbf{A}\mathbf{B}$ 的浮点运算量（FLOPs）与 ABFT 行列校验和的附加运算量，证明其算力开销远低于 $0.5\%$；
> （2）在 $4 \times 4$ 演示子块中，若硬件 SDC 导致输出矩阵第 $(i^*, j^*) = (2, 3)$ 号元素发生了 $+64.0$ 的静默比特翻转（即 $\tilde{C}_{2,3} = C_{2,3} + 64.0$），展示如何通过行残差向量 $\Delta \mathbf{r} = \tilde{\mathbf{C}}\mathbf{1}_N - \mathbf{A}\mathbf{r}_B$ 与列残差向量 $\Delta \mathbf{c}^T = \mathbf{1}_M^T \tilde{\mathbf{C}} - \mathbf{c}_A^T \mathbf{B}$ 的十字交叉坐标在微秒内精确定位 $(2, 3)$ 并原地修复。
>
> **解题过程**：
> 1. **计算 $M=N=K=4,096$ 下的 ABFT 相对算力开销**：
>    - 主矩阵乘计算量：$\mathrm{FLOPs}_{\mathrm{GEMM}} = 2 M N K = 2 \times 4,096^3 = \mathbf{137,438,953,472}\ (137.44\ \mathrm{GFLOPs})$；
>    - ABFT 附加计算量：包含求 $\mathbf{c}_A^T$（$MK$ 次加法）、求 $\mathbf{r}_B$（$KN$ 次加法）、算校验乘积 $\mathbf{c}_A^T\mathbf{B}$（$2KN$ FLOPs）与 $\mathbf{A}\mathbf{r}_B$（$2MK$ FLOPs）、以及对输出 $\mathbf{C}$ 求行列和并作差（$2MN$ 次加减法），总计为：
>      $$\mathrm{FLOPs}_{\mathrm{ABFT}} = 3MK + 3KN + 2MN = 8 \times 4,096^2 \quad (\text{单向列校验仅需 } 3N^2 + 2N^2 = 5N^2\text{，精简核心乘加项 } 6N^2 = 100,663,296\ \mathrm{FLOPs})$$
>      按核心乘加项比值计算：
>      $$\text{Overhead}_{\mathrm{ABFT}} = \frac{6 \times 4,096^2}{2 \times 4,096^3} = \frac{3}{4,096} \approx \mathbf{0.0732\%} \ll 0.5\%$$
>      即使计入全部外围归约加法（$8/8192 \approx 0.0977\%$），其开销也**不到 $0.1\%$**——比全量双重执行（DMR $100\%$ 开销）便宜了 **一千倍以上**，且校验和向量可作为额外的一行一列直接拼入张量分块（Tile）中顺带算完。
> 2. **十字交叉残差精确定位并原地修复 $(2, 3)$ 处的 $+64.0$ 翻转**：
>    当 $\tilde{C}_{2,3} = C_{2,3} + 64.0$ 时，计算行残差向量 $\Delta \mathbf{r} = \tilde{\mathbf{C}}\mathbf{1} - \mathbf{A}(\mathbf{B}\mathbf{1})$ 与列残差向量 $\Delta \mathbf{c}^T = \mathbf{1}^T\tilde{\mathbf{C}} - (\mathbf{1}^T\mathbf{A})\mathbf{B}$：
>    $$\Delta \mathbf{r} = [0.0,\; 0.0,\; \mathbf{+64.0},\; 0.0]^T, \qquad \Delta \mathbf{c} = [0.0,\; 0.0,\; 0.0,\; \mathbf{+64.0}]^T$$
>    非零行残差唯一锁定行号 $i^* = 2$，非零列残差唯一锁定列号 $j^* = 3$，且两者的残差幅值完全相等（$\Delta r_2 = \Delta c_3 = +64.0$）。内核立即原地执行 $\tilde{C}_{2,3} \leftarrow \tilde{C}_{2,3} - \Delta r_2$，无需重跑整个 GEMM 即完成无损修复（完整代码验证见 `experiments/ch20/20-01/run.py`）。

---

![基于代数不变量以低于 0.5% 算力开销在线拦截并定位矩阵乘硬件错误](ch20/ch20-abft-checksum-matrix-multiplication.svg)
*图 20-4　基于代数不变量（ABFT 行列校验和）以 $0.0732\%$（$<0.5\%$）算力开销在线拦截并定位矩阵乘硬件错误（算例 20-3）[^ch20-abft]。*

#### 第二道防线：AllReduce 归约前跨秩统计拦截（`Safeguarding LLM Training at Scale` OSDI '26 #71）

对于非矩阵乘算子（如 LayerNorm/RMSNorm、RoPE、SwiGLU 激活及通信缓冲处理）中可能发生的 SDC，**OSDI '26 #71（`Safeguarding LLM Training at Scale`）**[^ch20-safeguarding] 提出并历经 **3,500 万 GPU 小时** 生产验证了一套低开销在线检测与隔离框架：

- **在 `ncclAllReduce` 污染全局状态之前设卡拦截**：在同一数据并行（DP）或上下文并行（CP）组内，尽管不同 Rank 处理的微批次词元不同，但在健康状态下，各 Rank 在同一层的激活均值/方差与局部梯度范数 $\|g_i\|_2$ 服从高度一致的紧致分布；
- 框架在反向传播挂钩中顺带汇总各层局部梯度范数标量（仅数百字节，通信开销可忽略不计），利用基于中位数绝对偏差（MAD）的跨秩离群检验实时扫描。一旦发现某个 Rank 的局部梯度范数偏离同组中位数超过自适应阈值（但尚未变成 `NaN`），**立即阻断本次 `AllReduce` 与 `optimizer.step()`**，从而确保全集群主权重与 Adam 动量 $100\%$ 不被污染。

#### 第三道防线：消除良性非确定性实现位级对齐算子定位（`OpGuard` OSDI '26 #72）与 50 ms 微批次重跑

当第二道防线报告“第 $k$ 号 DP 组在第 $t$ 步存在数值异常”后，接下来面临一个极其棘手的工程难题：**在由 TP=8、PP=16 组成的复杂模型并行子网里，究竟是哪一张物理 GPU、在哪一个具体算子上发生了 SDC？**

为什么我们不能直接把同一个输入喂给两张卡跑一遍，看看哪个算子的输出张量哈希值不一样？因为现代深度学习框架与 GPU 算子在默认高性能模式下存在广泛的**良性软硬件非确定性（Benign Non-Determinism）**：

- FlashAttention 反向传播与 Embedding 梯度累加中使用的 GPU 硬件原子加法（`atomicAdd`），其线程束到达顺序随每次硬件调度微秒级抖动而变；
- 由于浮点数加法不满足结合律 $(a + b) + c \ne a + (b + c)$，即便在两张完全健康的 GPU 上用完全相同的输入跑同一个反向算子，算出来的梯度张量在尾数最低几位上也天然不同，导致普通张量哈希（CRC32/SHA256）比对 $100\%$ 误报。

**OSDI '26 #72（`OpGuard`: *Debugging Silent Data Corruption in LLM Training via Bitwise Alignment*）**[^ch20-opguard] 彻底破解了这一诊断死结（见**图 20-5**、**图 20-6** 与**表 20-3**）：

1. **按需切换位级确定性重放态（On-Demand Bitwise-Aligned Replay）**：在正常训练步中保持最高性能的非确定性算子全速运行；一旦在线检测器捕获异常，仅对当前长为 $50\ \mathrm{ms}$ 的微批次激活开启 `OpGuard` 位级对齐模式——通过编译器插桩与运行时钩子，将原子加法替换为固定顺序的分块树状归约（Deterministic Split-K / Padded Reduction），并锁定 `cuBLAS` 工作空间与通信归约树拓扑，使所有健康 GPU 的逐算子输出达到 **$100\%$ 比特级完全一致（Bitwise Identical）**；
2. **50 ms 微批次原地重算区分软错误与永久坏卡**：在位级确定性模式下原地重跑刚才那一个微批次：
   - 如果重跑一次后异常消失且跨卡哈希完全一致，说明刚才是一次由宇宙射线引发的**瞬态单粒子翻转（Transient Soft Error）**，直接采纳重算结果继续训练，仅损失 $50\ \mathrm{ms}$；
   - 如果重跑后某张卡在某算子输出的 64 位整张量哈希与同构对偶卡不一致，**该算子对应的物理 GPU 即被 $100\%$ 铁证锁定为硬件 SDC 坏卡**。系统立刻在秒级内将其踢出通信域并切入热备卡，**彻底告别了“回滚两小时前全局检查点、带着坏卡盲目重启又再次暴雷”的噩梦**。

![3,500 万 GPU 小时在线 SDC 感知隔离与控制良性非确定性的位级对齐定位](ch20/ch20-safeguarding-sdc-and-opguard-bitwise-debug.svg)
*图 20-5　$3,500$ 万 GPU 小时在线 SDC 感知隔离（`Safeguarding LLM Training` OSDI '26 #71）与控制良性非确定性的位级对齐定位（`OpGuard` OSDI '26 #72）。*

![检测到异常后原地重跑当前微批次区分瞬态软错误与永久硬件故障 vs. 盲目回滚数小时全局检查点](ch20/ch20-microbatch-rerun-vs-checkpoint-rollback.svg)
*图 20-6　检测到异常后原地重跑当前微批次（耗时 $50\ \mathrm{ms}$）区分瞬态软错误与永久硬件故障 vs. 盲目回滚数小时全局检查点。*

| SDC 防御与诊断方案 | 稳态算力开销 | 能否拦截非 `NaN` 慢性偏移 | 能否阻止 `AllReduce` 污染全局权重 | 故障定位粒度 | 单次 SDC 恢复代价 |
| --- | :-: | :-: | :-: | --- | --- |
| **传统 `torch.isfinite()` + 周期性验证集评测** | $< 0.01\%$ | **否（对低位偏移完全失明）** | **否（发现时已污染数千步）** | 无定位（需盲猜二分排查） | **回滚 $2\ \mathrm{h}$ 检查点（损 $25\text{万}$ 卡时）** |
| **全量双重模块化冗余（Full DMR 双算比对）** | **$100.0\%$** | 是（若全程强制定序） | 是 | 算子/节点级 | 浪费整整一半集群算力 |
| **算子级代数不变量校验（ABFT 行列校验和）** | **$0.0732\%$** | **是（捕获矩阵乘内部翻转）** | **是（在算子出口原地拦截修复）** | **矩阵乘单元素坐标 $(i^*, j^*)$** | **微秒级原地减除残差修复** |
| **`Safeguarding` (#71) + `OpGuard` (#72) 协同** | **$< 0.3\%$** | **是（跨秩 MAD 统计感知）** | **是（AllReduce 前硬阻断）** | **位级对齐锁定精确物理 GPU 与算子** | **$50\ \mathrm{ms}$ 微批重跑 + $5\ \mathrm{s}$ 热切** |

*表 20-3　四种静默数据损坏（SDC）防御与定位机制在算力开销、拦截时机与恢复代价上的对比。*

下面给出 **（1）`Safeguarding LLM Training` 的 AllReduce 前跨秩鲁棒 Z-Score（基于中位数绝对偏差 MAD）统计拦截器** 与 **（2）`OpGuard` 按需位级确定性微批次重放与坏卡定位状态机** 的参考实现：

```python
import numpy as np
import hashlib

def safeguarding_pre_allreduce_mad_gate(
    rank_grad_norms: np.ndarray, mad_z_threshold: float = 8.0
) -> tuple[bool, list[int], np.ndarray]:
    """
    OSDI '26 #71 (Safeguarding LLM Training at Scale):
    在执行 ncclAllReduce 之前汇总同 DP 组各 Rank 局部梯度范数，
    利用对极端离群值免疫的中位数绝对偏差 (MAD) 拦截非 NaN 静默比特翻转
    """
    if not np.all(np.isfinite(rank_grad_norms)):
        bad = np.where(~np.isfinite(rank_grad_norms))[0].tolist()
        return False, bad, np.full_like(rank_grad_norms, np.inf)

    median_val = float(np.median(rank_grad_norms))
    mad = float(np.median(np.abs(rank_grad_norms - median_val))) + 1e-8
    # 正态一致性修正 MAD 鲁棒 Z-score: 0.6745 * |x - median| / MAD
    robust_z = 0.6745 * np.abs(rank_grad_norms - median_val) / mad
    anomalous_ranks = np.where(robust_z > mad_z_threshold)[0].tolist()
    return (len(anomalous_ranks) == 0), anomalous_ranks, robust_z

def opguard_bitwise_aligned_replay_localize(
    operator_trace_suspect: dict[str, np.ndarray],
    operator_trace_reference: dict[str, np.ndarray]
) -> dict[str, object]:
    """
    OSDI '26 #72 (OpGuard):
    在开启位级确定性算子模式 (Deterministic Split-K / 禁用非确定性 atomicAdd) 后，
    原地重跑 50 ms 微批次并逐算子比对 64-bit 张量摘要，精确定位首个发生 SDC 的算子
    """
    for op_name in operator_trace_suspect:
        t_sus = np.ascontiguousarray(operator_trace_suspect[op_name])
        t_ref = np.ascontiguousarray(operator_trace_reference[op_name])
        h_sus = hashlib.blake2b(t_sus.tobytes(), digest_size=8).hexdigest()
        h_ref = hashlib.blake2b(t_ref.tobytes(), digest_size=8).hexdigest()
        if h_sus != h_ref:
            max_abs_diff = float(np.max(np.abs(t_sus.astype(np.float64) - t_ref.astype(np.float64))))
            return {
                "classification": "PERMANENT_HARDWARE_SDC",
                "first_corrupted_op": op_name,
                "suspect_hash": h_sus,
                "reference_hash": h_ref,
                "max_abs_diff": max_abs_diff,
                "action": "ISOLATE_GPU_AND_TRIGGER_TRAINMOVER"
            }
    return {
        "classification": "TRANSIENT_SOFT_ERROR",
        "first_corrupted_op": None,
        "action": "COMMIT_50MS_REPLAY_AND_CONTINUE"
    }
```

---

## 20.3 告别全集群重启：机队级健康自愈、慢节点（Straggler）取证与无重启弹性容错

### 20.3.1 从“一卡挂、万卡停”到亚秒级热迁移与副本组自治（`TorchFT` / `TrainMover` OSDI '26 / `EROICA` & `FLARE` NSDI '26）

当我们在第 20.1 节与第 20.2 节中定位并剔除了一台发生掉卡或 SDC 的故障节点后，为什么传统的 `torchrun`（PyTorch Elastic）恢复流程需要耗时整整 $300\ \mathrm{s}$（5 分钟）？

剖析传统故障恢复的时间线可以发现，绝大部分时间都浪费在**让 $99,999$ 张完全健康的 GPU 陪葬销毁并重新初始化**上：

1. **进程级全局连坐销毁（$30\text{–}60\ \mathrm{s}$）**：单卡异常触发分布式看门狗超时，调度器向全网十万个工作进程发送 `SIGTERM/SIGKILL`，所有健康卡上的 CUDA 上下文、已编译算子缓存与显存页表被全部销毁；
2. **万卡 TCPStore 集中式握手与 NCCL 环/树重建（$120\text{–}180\ \mathrm{s}$）**：重启后的十万个进程同时向 Rank 0 主节点的 `TCPStore` 发起连接注册，随后 `ncclCommInitRank` 跨数千台交换机执行全网拓扑探测与 RDMA 队列对（Queue Pair, QP）建连；
3. **远端存储检查点重载雪崩（$60\text{–}120\ \mathrm{s}$）**：十万张卡同时向后端并行文件系统发起数太字节的权重与优化器状态读取。

为了彻底砍掉这三项连坐开销，**OSDI '26 的 `TrainMover`** 以及 **NSDI '26 的 `EROICA` / `FLARE`** 构建了**无重启亚秒/秒级热迁移与通信域局部修复架构**（见**图 20-7**）：

1. **热备节点常驻预初始化影子守护进程（Pre-Warmed Shadow Daemons）**：热备池中的 $S=40$ 台节点在空闲时就已经提前完成了 Python 运行时加载、CUDA Context 初始化、静态 CUDA Graph 编译以及与全网核心交换机的备用 RDMA QP 池预建连；
2. **虚拟 Rank 映射层与局部 QP 指针热切换**：在 NCCL 与底层 IB Verbs 之间插入一层轻量级虚拟通信组抽象。当第 $r$ 号物理节点故障并由热备节点 $s$ 顶替时，其余 $99,992$ 张健康 GPU **进程不退出、CUDA 上下文不销毁**，仅由训练控制面下发一条控制信令，命令与节点 $r$ 存在直接 Ring/Tree 邻居关系的几十张 GPU 将其目标 RDMA QP 指针原子切换到热备节点 $s$ 的预建 QP 上（耗时 $< 200\ \mathrm{ms}$）；
3. **同 DP 组健康邻居的 RDMA 显存直推（Peer-to-Peer State Push）**：热备节点 $s$ 根本不需要去读慢速的远端并行文件系统，而是直接由同一数据并行副本组（或 ZeRO 冗余搭档节点）的健康邻居通过 400G/800G RDMA `Write` 在 $2\text{–}4\ \mathrm{s}$ 内将最新模型权重与优化器状态分片直接灌入其 HBM 显存。配合新节点入网与维修归队后的**四阶自动化烤机流水线（Burn-in Pipeline，见第 20.4.2 节图 20-13）**，整个万卡/十万卡集群实现了真正的“秒级换血、永不停机”。

![传统全局销毁重启 vs. TrainMover 利用预热备用进程与 RDMA 重映射实现秒级无重启接管](ch20/ch20-trainmover-subsecond-hot-migration.svg)
*图 20-7　传统全局销毁重启（`torchrun` 耗时 $300\ \mathrm{s}$）vs. `TrainMover`（OSDI '26）利用预热备用进程与 RDMA 重映射实现秒级无重启接管[^ch20-trainmover]。*

| 故障恢复与弹性架构 | 故障发生时健康卡行为 | 通信组（NCCL / QP）恢复方式 | 模型状态重载来源 | 单次故障恢复耗时 $T_{\mathrm{rec}}$ | 十万卡下 $\mathrm{ETTR}$ |
| --- | --- | --- | --- | :-: | :-: |
| **传统 `torchrun` 全局重启** | **全网 $99,999$ 卡进程连坐销毁** | 全量重建 `TCPStore` 与全网 QP | 远端并行文件系统全量重读 | $180\text{–}300\ \mathrm{s}$ | **$63.22\%$** |
| **`TorchFT` 副本组自治（Replica Group）** | 仅剔除故障所在 DP 副本组 | 跨 DP 组动态重协商 Lighthouse | 存活 DP 组继续训练，异步归队 | $15\text{–}45\ \mathrm{s}$（有损/变批） | $84.5\%$ |
| **`EROICA` / `FLARE`（NSDI '26）网络快切** | 健康卡原地挂起当前通信算子 | 多轨平面绕行 / 预置备份环表切换 | 无需重载权重（仅重发在途分块） | **$0.2\text{–}1.5\ \mathrm{s}$（链路故障）** | $93.8\%$ |
| **`TrainMover`（OSDI '26）无重启热接管** | **健康卡进程与 CUDA 上下文零销毁** | **局部邻居原子切换至预热备机 QP** | **同 DP 组健康邻居 RDMA 显存直推** | **$2.0\text{–}5.0\ \mathrm{s}$（节点掉卡）** | **$92.34\%$** |

*表 20-4　四代大规模分布式训练容错恢复架构在连坐范围、通信重建方式与恢复耗时上的演进对比。*

下面给出 **`TrainMover` 虚拟 Rank 局部 RDMA 队列对（QP）热切换控制器** 与 **四阶自动化烤机准入流水线（Burn-in Admission Pipeline，图 20-13）** 的参考实现：

```python
from dataclasses import dataclass

@dataclass
class NodeBurnInReport:
    node_id: str
    hbm_thermal_throttle_pct: float
    abft_bitwise_mismatch_count: int
    rs_fec_high_bin_slope_per_min: float
    rdma_loopback_gbps: float

def automated_burn_in_admission_gate(
    report: NodeBurnInReport,
    min_rdma_gbps: float = 385.0,
    max_fec_slope: float = 5.0
) -> dict[str, object]:
    """
    四阶自动化烤机准入门禁 (图 20-13):
    新节点入网或故障维修归队前，依次校验满载热压测、OpGuard/ABFT 位级 SDC 校验、
    800G 光模块 RS-FEC 高阶桶斜率与 RDMA 环回带宽，杜绝带病节点混入热备池
    """
    thermal_ok = (report.hbm_thermal_throttle_pct == 0.0)
    sdc_ok = (report.abft_bitwise_mismatch_count == 0)
    optical_ok = (report.rs_fec_high_bin_slope_per_min <= max_fec_slope)
    rdma_ok = (report.rdma_loopback_gbps >= min_rdma_gbps)
    admitted = thermal_ok and sdc_ok and optical_ok and rdma_ok
    return {
        "node_id": report.node_id,
        "admitted_to_hot_spare_pool": admitted,
        "stage_checks": {
            "stage1_thermal_stress": thermal_ok,
            "stage2_deterministic_sdc": sdc_ok,
            "stage3_optical_fec_margin": optical_ok,
            "stage4_rdma_bandwidth": rdma_ok,
        },
        "action": "ADMIT_TO_PREWARMED_SPARE_POOL" if admitted else "QUARANTINE_FOR_RMA"
    }

def trainmover_virtual_rank_hot_swap(
    virtual_to_physical_node: dict[int, str],
    failed_virtual_rank: int,
    spare_node_id: str,
    shard_bytes_gb: float = 48.0,
    rdma_push_bw_gbps: float = 320.0
) -> dict[str, object]:
    """
    OSDI '26 TrainMover 无重启秒级热接管:
    健康节点进程不销毁，仅重映射故障虚拟 Rank 的物理节点指针并由同 DP 组邻居 RDMA 直推状态
    """
    old_node = virtual_to_physical_node[failed_virtual_rank]
    updated_map = dict(virtual_to_physical_node)
    updated_map[failed_virtual_rank] = spare_node_id
    qp_switch_sec = 0.18
    rdma_state_push_sec = (shard_bytes_gb * 8.0) / rdma_push_bw_gbps
    total_recovery_sec = qp_switch_sec + rdma_state_push_sec
    return {
        "evicted_node": old_node,
        "activated_spare_node": spare_node_id,
        "healthy_ranks_restarted": 0,
        "total_recovery_sec": round(total_recovery_sec, 2),
        "updated_node": updated_map[failed_virtual_rank]
    }
```

---

### 20.3.2 比宕机更隐蔽的慢节点（Gray Failure / Straggler）与 `NCCL Flight Recorder` 环路死锁取证

在超大规模集群运维中，还有一类比显式掉卡更常见、更具迷惑性的故障形态——**灰故障（Gray Failure）或慢节点（Straggler）**：某台服务器因为冷板微通道局部水垢堵塞触发 GPU 降频 $20\%$、PCIe 链路因信号完整性劣化自动从 Gen5 x16 降级为 Gen3 x4、或者某根 800G 光模块端面污染导致前向纠错（RS-FEC）频繁触发重传。这台机器依然“活着”，心跳完全正常，但它每一个微批次的计算或通信都比其他节点慢 $2.0\ \mathrm{ms}$。

当这种灰故障发生在紧耦合同步集合通信（AllReduce / AllGather / All-to-All）中时，它会在性能剖析器（如 PyTorch Profiler / Nsight Systems）的火焰图上制造出一场极具欺骗性的**“假象风暴”**：

- 健康的 $99,999$ 张 GPU 准时算完前向与反向，早早进入 `ncclKernel_AllReduce` 开始自旋等待；
- 唯一的那张慢卡（Straggler）迟到了 $2.0\ \mathrm{ms}$ 才进入集合通信算子，到达后立刻与大家一起在 $0.4\ \mathrm{ms}$ 内完成了网络数据交换；
- 结果，运维工程师打开任何一张健康卡的火焰图，看到的都是 **`ncclKernel_AllReduce` 耗时高达 $2.4\ \mathrm{ms}$（占了总步时的 $80\%$）**，于是错误地把黑锅扣给网络团队：“网络发生了严重拥塞，快把网络带宽翻倍或者调优交换机水线”。

让我们结合本仓库锁定的生产诊断案例 `case-studies/collective-paths-and-diagnosis.md`，严格复算这一归因陷阱与 **到达时间偏斜矩阵（Arrival Skew Matrix）** 取证法则（完整复算于 `experiments/ch20/20-01/run.py`）：

---

> **算例 20-4（锁定 `case-studies/collective-paths-and-diagnosis.md`：集合通信慢节点到达偏斜、多网卡条带失衡与三边环路死锁取证）**：
> 1. **慢节点到达偏斜（Arrival Skew）vs. 网络带宽升级**：
>    在一个 4-Rank 同步集合通信组中，Rank 0、1、2 在 $t = 0.0\ \mathrm{ms}$ 准时就绪并进入集合通信，而慢节点 Rank 3 在 $t = 2.0\ \mathrm{ms}$ 才计算完毕到达；全员到齐后，真实网络数据交换耗时为 $T_{\mathrm{xfer}} = 0.4\ \mathrm{ms}$。
>    - 全体完成时刻为 $T_{\mathrm{done}} = \max(0, 0, 0, 2.0) + 0.4 = \mathbf{2.4\ \mathrm{ms}}$；Rank 0、1、2 观测到的集合通信总耗时均为 $2.4\ \mathrm{ms}$（其中包含 $2.0\ \mathrm{ms}$ 空等 + $0.4\ \mathrm{ms}$ 传输）。
>    - **错误处方（网络带宽翻倍）**：即使斥巨资将互联网络带宽翻倍（使 $T_{\mathrm{xfer}}$ 从 $0.4\ \mathrm{ms}$ 降至 $0.2\ \mathrm{ms}$），全体完成时刻仍高达 $2.0 + 0.2 = \mathbf{2.2\ \mathrm{ms}}$，**仅节省了 $0.2\ \mathrm{ms}$（$8.3\%$）**。
>    - **正确处方（基于 `NCCL Flight Recorder` 消除入队偏斜）**：对比各 Rank 进入集合通信的硬件时间戳 $t_{\mathrm{enqueue}, i}$，立刻抓出最晚到达且自身等待时间为 $0\ \mathrm{ms}$ 的唯一根因慢节点 **Rank 3**（$\max_i t_{\mathrm{enqueue},i} - \min_i t_{\mathrm{enqueue},i} = 2.0\ \mathrm{ms}$）。将 Rank 3 替换为热备卡后，全体完成时刻直接降至 **$0.4\ \mathrm{ms}$**，**净省 $2.0\ \mathrm{ms}$（提速 $6\times$）**。
> 2. **多网卡条带失衡（Multi-NIC Stripe Imbalance）**：
>    某节点通过 4 张网卡并行条带化发送数据，4 条轨道的实测完成时间分别为 `[1.0, 1.0, 1.0, 2.0] ms`。虽然 4 张网卡的**算术平均耗时仅为 $\mathbf{1.25\ \mathrm{ms}}$**，但由于同步屏障必须等待最慢的第 3 号退化网卡完成，实际步进耗时由 $\max([1.0, 1.0, 1.0, 2.0]) = \mathbf{2.0\ \mathrm{ms}}$ 决定。因此多轨道诊断严禁只看网卡平均带宽，必须监控 `max(stripe_time) / median(stripe_time)` 比值。
> 3. **集合通信环路等待死锁（Ring Cycle Deadlock）自动检测**：
>    当集群因条件分支不一致或丢包挂死时，从各 Rank 的 `NCCL Flight Recorder` 环形缓冲区导出最后一条未完成算子的等待边 `(waiter -> target)`。若等待图出现形如 `Rank 0 -> Rank 1`、`Rank 1 -> Rank 2`、`Rank 2 -> Rank 0` 的闭合有向环（Cycle），则可 $100\%$ 判定为**同步顺序错位或环路死锁**（而非单节点慢卡），需立即终止并输出环路算子序号栈。

---

![利用 NCCL Flight Recorder 与集合通信到达时间偏斜矩阵从全网阻塞假象中揪出唯一根因慢节点](ch20/ch20-nccl-flight-recorder-straggler-blame.svg)
*图 20-8　利用 `NCCL Flight Recorder` 与集合通信到达时间偏斜矩阵（Arrival Skew，算例 20-4）从全网阻塞假象中揪出唯一根因慢节点。在同步通信中，“通信算子耗时长”往往是健康卡在等慢卡，升级网络带宽仅省 $0.2\ \mathrm{ms}$，剔除慢卡可省 $2.0\ \mathrm{ms}$。*

![光模块劣化引发 FEC 纠错重传导致通信有效带宽周期性塌陷的特征识别](ch20/ch20-optical-ber-fec-gray-failure-detection.svg)
*图 20-9　光模块劣化引发 FEC 纠错重传（灰故障 Gray Failure）导致通信有效带宽周期性塌陷的特征识别。通过监控 Pre-FEC 误码率斜率与高阶纠错符号桶（Bin 7–15），可在链路未彻底断连前主动隔离劣化光模块。*

如**图 20-8** 与**图 20-9** 所示，现代十万卡集群控制面通过两项自动化遥测机制将慢节点与灰故障消灭在萌芽状态：

1. **全网集合通信到达偏斜矩阵（Arrival Skew Matrix）实时归因**：在每个宿主机的 NCCL 监控守护进程中开启轻量级 `NCCL_DEBUG_SUBSYS=COLL` / `Flight Recorder` 环形缓冲，每步提取各 Rank 启动集合通信的硬件时间戳 $t_{\mathrm{start}}(r, s)$ 与结束时间戳 $t_{\mathrm{end}}(r, s)$。定义第 $r$ 号卡在第 $s$ 步的**自身就绪滞后度** $\mathrm{Lag}(r, s) = t_{\mathrm{start}}(r, s) - \min_{j} t_{\mathrm{start}}(j, s)$：如果某张卡连续 $50$ 步都是全组最后一个到达（即 $t_{\mathrm{end}}(r, s) - t_{\mathrm{start}}(r, s)$ 最短，而 $\mathrm{Lag}(r, s)$ 最大），控制面直接将其判定为计算侧或 PCIe 侧慢节点并触发热迁移。
2. **800G 光模块数字诊断监控（DOM）与 RS-FEC 纠错码字直方图预警**：在 800G DR8/SR8 光链路中，IEEE 802.3df 标准采用 RS(544, 514) 前向纠错码，每个码字最多可纠正 $t = 15$ 个符号错误。当某根光纤弯曲或激光器老化初期，误码数落在 $1\text{–}14$ 个符号以内，硬件 FEC 能够将其全部纠回，此时上层网卡看到的丢包率依然是 $0$；但一旦误码突破 $15$ 个符号（落入 Uncorrectable Codeword 桶），就会立即触发 RoCE/UEC 链路重传，将尾延迟拉高十倍。通过每 10 秒轮询交换机光模块的 **FEC 高阶符号错误桶（Symbol Error Bins 7–15）增长斜率**，运维系统可以在发生第一起真实丢包前数小时就精准锁定即将失效的亚健康光模块，趁检查点间隙将其无感屏蔽。

下面给出对应算例 20-4（锁定 `case-studies/collective-paths-and-diagnosis.md`）的 **`NCCL Flight Recorder` 慢节点偏斜归因、多网卡条带失衡检测与有向图环路死锁取证器** 完整实现：

```python
import numpy as np

def diagnose_collective_flight_recorder(
    enqueue_ts_ms: list[float],
    transfer_ms: float,
    nic_stripe_times_ms: list[float],
    wait_edges: list[tuple[int, int]]
) -> dict[str, object]:
    """
    锁定 case-studies/collective-paths-and-diagnosis.md (算例 20-4):
    1. 区分慢节点到达偏斜 (Arrival Skew) 与真实网络传输瓶颈
    2. 核算多网卡条带化传输的木桶效应 (Max vs. Mean)
    3. 检测 Flight Recorder 未完成算子等待图中的有向环死锁 (Cycle Deadlock)
    """
    arr = np.asarray(enqueue_ts_ms, dtype=np.float64)
    last_arrival = float(np.max(arr))
    first_arrival = float(np.min(arr))
    arrival_skew_ms = last_arrival - first_arrival
    completion_ms = last_arrival + transfer_ms
    straggler_rank = int(np.argmax(arr))

    # 处方对比: 网络带宽翻倍 vs. 消除慢节点偏斜
    bw_doubled_completion_ms = last_arrival + 0.5 * transfer_ms
    skew_eliminated_completion_ms = first_arrival + transfer_ms

    # 多网卡条带失衡
    stripes = np.asarray(nic_stripe_times_ms, dtype=np.float64)
    stripe_mean_ms = float(np.mean(stripes))
    stripe_bottleneck_ms = float(np.max(stripes))

    # 有向等待图环路检测
    adj: dict[int, int] = {u: v for u, v in wait_edges}
    has_cycle = False
    for start_node in adj:
        curr = start_node
        visited: set[int] = set()
        while curr in adj:
            if curr in visited:
                if curr == start_node:
                    has_cycle = True
                break
            visited.add(curr)
            curr = adj[curr]
        if has_cycle:
            break

    return {
        "straggler_rank": straggler_rank,
        "arrival_skew_ms": round(arrival_skew_ms, 3),
        "baseline_completion_ms": round(completion_ms, 3),
        "bw_doubled_savings_ms": round(completion_ms - bw_doubled_completion_ms, 3),
        "straggler_fix_savings_ms": round(completion_ms - skew_eliminated_completion_ms, 3),
        "stripe_mean_ms": round(stripe_mean_ms, 3),
        "stripe_bottleneck_ms": round(stripe_bottleneck_ms, 3),
        "deadlock_cycle_detected": has_cycle
    }
```

---

## 20.4 AI 基础设施基准评测科学：如何识破跑分魔术并构建可复现、无偏剖析的工业级基准

### 20.4.1 为什么公开基准上的“$3\times$ 加速”上线后往往缩水甚至倒退：五大跑分失真机制（Benchmark Gaming）

在 AI 基础设施工程实践中，几乎每一个推理与训练团队都经历过这样的挫败：某项新算子、新量化方案或新调度器在离线微基准（Microbenchmark）报告中展示了耀眼的“**算术平均加速比 $3.5\times\text{–}5.25\times$**”，然而一旦合并上线接入真实生产流量，端到端吞吐非但没有提升，P99 尾延迟反而恶化了 $40\%$，甚至引发了线上模型回答质量的静默退化。

为什么离线跑分与生产现实之间会裂开如此巨大的鸿沟？结合本仓库锁定的生产评测与上线案例 `case-studies/optimization-evaluation-and-deployment.md`、`case-studies/profile-and-plan-ranking.md`、`case-studies/execution-feedback.md` 与 `case-studies/evaluation-and-agent-records.md`，我们可以将 AI 系统跑分中的失真机制归纳为**五大基准陷阱（Five Benchmark Gaming Traps）**：

1. **陷阱一：对不同张量形状的“加速比”直接取算术平均（Arithmetic Mean of Speedups Paradox）**：
   算子开发者喜欢在一组包含极小形状（如 $B=1, L=128$）与极大形状（如 $B=32, L=8192$）的测试网格上分别除以基线耗时算出加速比 $S_i = T_{\mathrm{base},i} / T_{\mathrm{cand},i}$，然后汇报“平均加速比 $\frac{1}{K}\sum S_i$”。这在数学上犯了用算术平均替代加权调和平均的根本错误：在总耗时极短的微小形状上取得 $10\times$ 加速只能节省几微秒，而在耗时占大头的长序列形状上退化 $2\times$ 就会亏掉数百微秒。
2. **陷阱二：按单调顺序（`AABB`）执行对比测量，将硬件热节流与时钟漂移误判为算法优劣**：
   GPU 在连续满载运行数十秒后，核心温度从 $45^\circ\mathrm{C}$ 升至 $80^\circ\mathrm{C}$，供电模块与 HBM 时钟频率会因动态热管理发生单调漂移（Thermal Drift），或者因页表缓存预热产生系统性偏移。如果先跑完 $N$ 轮方案 A（此时机器处于低温高频态），再接着跑 $N$ 轮方案 B（此时机器已进入高温降频态），就会制造出“方案 A 比方案 B 快”的虚假结论。
3. **陷阱三：用合成均匀随机词元（`torch.randint`）评测前缀缓存、MoE 路由与投机解码**：
   均匀随机生成的词元序列既没有真实多轮对话与 Agent 负载中的公共系统提示词前缀（导致 RadixCache 命中率恒为 $0\%$，或反过来用完全相同的克隆请求刷出 $99.9\%$ 的虚假命中率），也不会触发真实自然语言在投机解码草稿树上的高接受率（随机词元导致草稿接受率 $\alpha \to 0$），更无法复现真实语义偏斜下特定 MoE 专家被热点打爆的通信尾延迟。
4. **陷阱四：用孤立静态预测或代理打分（Static Proxy Score）替代真实竞争环境下的尾延迟排名**：
   编译器自动调优器或分布式模拟器若仅凭无竞争孤立剖析（Isolated Profile）给配置排序，当两个候选方案预测耗时接近时，共享链路竞争系数 $k$ 的微小上升就会彻底反转优劣排名（如 `case-studies/profile-and-plan-ranking.md` 所锁定：方案 A 含计算 $70\ \mathrm{ms}$ 与暴露通信 $30\ \mathrm{ms}$，孤立预测 $100\ \mathrm{ms}$；方案 B 含计算 $90\ \mathrm{ms}$ 与暴露通信 $12\ \mathrm{ms}$，孤立预测 $102\ \mathrm{ms}$。一旦真实多流竞争使通信膨胀系数跨过临界点 $k^* = 10/9 \approx 1.111$，例如在 $k = 1.3$ 时，$T_A = 70 + 30 \times 1.3 = \mathbf{109.0\ \mathrm{ms}}$ 反慢于 $T_B = 90 + 12 \times 1.3 = \mathbf{105.6\ \mathrm{ms}}$）。
5. **陷阱五：只报吞吐与延迟，隐瞒低精度简化算子带来的模型质量静默退化或改变硬件前置条件**：
   通过激进截断 Softmax 累加精度、跳过归一化或缩减 MoE Top-K 路由，确实能把跑分刷高 $20\%$，但这本质上是在拿模型智商换速度；同样，如 `case-studies/evaluation-and-agent-records.md` 所揭示，在对比人工或 Agent 内核优化成绩时，必须锁定相同的机器模型、初始基线（如 VLIW/SIMD 慢基线 $147,734$ 周期 vs. 两小时考核起点 $18,532$ 周期）与部件发射槽位下界（如标量 ALU $\lceil 118,784 / 12 \rceil = 9,899$ 周期），绝不能把改变了算法语义或测试起点的数字混为一谈。

让我们通过精确的数值反例（锁定 `case-studies/optimization-evaluation-and-deployment.md` 并复算于 `experiments/ch20/20-01/run.py`）彻底算清前两个陷阱的数学本质：

---

> **算例 20-5（锁定 `case-studies/optimization-evaluation-and-deployment.md`：算术平均加速比 $5.25\times$ 反而变慢的反例与 `AABB` vs. `ABBA` 时漂抵消定律）**：
> 1. **算术平均加速比悖论（Mean-of-Speedups Paradox）**：
>    某推理引擎存在两类核心张量形状 $S_1$（短序列）与 $S_2$（长序列）。基线算子（Baseline）在两类形状上的耗时均为 $[100\ \mu\mathrm{s},\; 100\ \mu\mathrm{s}]$。现有两个候选优化算子：
>    - **候选算子 A**：在 $[S_1, S_2]$ 上的耗时为 $[10\ \mu\mathrm{s},\; 200\ \mu\mathrm{s}]$（针对短序列做了特化，但在长序列上因寄存器溢出退化了 $2\times$）；
>    - **候选算子 B**：在 $[S_1, S_2]$ 上的耗时为 $[50\ \mu\mathrm{s},\; 50\ \mu\mathrm{s}]$（在两类形状上均稳定提速 $2\times$）。
>    试计算：（a）两个候选算子各自的“算术平均加速比”；（b）当生产流量中 $S_1$ 与 $S_2$ 等频出现（各占 $50\%$）时，两个候选算子处理一对请求的总耗时分别是多少？（c）短序列 $S_1$ 的流量占比 $p$ 必须高到什么程度，候选 A 的加权期望耗时才能真正击败候选 B？
> 2. **线性时钟漂移下的 `AABB` 失真与 `ABBA` 镜像抵消**：
>    按照 `experiments/ch20/20-01/run.py` 与 `case-studies/optimization-evaluation-and-deployment.md` 的锁定参数：设 A 的真实零漂移耗时为 $100\ \mu\mathrm{s}$、B 为 $90\ \mu\mathrm{s}$（即真实情况下 **B 比 A 快 $10\ \mu\mathrm{s}$**），而在连续 4 次测量槽位 $k=0, 1, 2, 3$ 上 GPU 因发热产生线性递增漂移 $d_k = [0,\; 10,\; 20,\; 30]\ \mu\mathrm{s}$。试对比单调顺序 `AABB` 与镜像对称顺序 `ABBA` 测得的平均耗时。
>
> **解题过程**：
> 1. **算术平均加速比悖论求解**：
>    - （a）**候选 A 的逐形状加速比**为 $\left[\frac{100}{10},\; \frac{100}{200}\right] = [10.0\times,\; 0.5\times]$，其**算术平均加速比高达 $\frac{10.0 + 0.5}{2} = \mathbf{5.25\times}$**；而**候选 B 的逐形状加速比**为 $\left[\frac{100}{50},\; \frac{100}{50}\right] = [2.0\times,\; 2.0\times]$，算术平均加速比仅为 **$\mathbf{2.00\times}$**。单看跑分宣传，候选 A（$5.25\times$）似乎远胜候选 B（$2.0\times$）。
>    - （b）**等频流量（各 $1$ 次调用）下的真实端到端总耗时**：
>      $$T_{\mathrm{total}}(\text{Baseline}) = 100 + 100 = 200\ \mu\mathrm{s}$$
>      $$T_{\mathrm{total}}(\text{Cand A}) = 10 + 200 = \mathbf{210\ \mu\mathrm{s}} > 200\ \mu\mathrm{s}\ (\text{比基线还慢 } 5\%)$$
>      $$T_{\mathrm{total}}(\text{Cand B}) = 50 + 50 = \mathbf{100\ \mu\mathrm{s}}\ (\text{真实端到端提速 } 2.0\times)$$
>      号称“平均加速 $5.25\times$”的候选 A 上线后不仅比候选 B 慢了 $2.1$ 倍，甚至比未优化的基线还发生了倒退。
>    - （c）**求解候选 A 击败候选 B 的流量分布翻转阈值 $p^*$**：
>      设短序列 $S_1$ 占比为 $p \in [0, 1]$，长序列 $S_2$ 占比为 $1 - p$。令两者加权期望耗时相等：
>      $$E[T_A(p)] = 10p + 200(1 - p) = 200 - 190p, \qquad E[T_B(p)] = 50p + 50(1 - p) = 50$$
>      $$200 - 190p < 50 \iff 190p > 150 \iff p > p^* = \frac{15}{19} \approx \mathbf{0.7895}\ (78.95\%)$$
>      只有当线上流量中超过 **$78.95\%$** 都是极短形状 $S_1$ 时，候选 A 才真正比候选 B 快。
> 2. **线性热漂移下 `AABB` vs. `ABBA` 求解**：
>    已知真实耗时 $t_A^{\mathrm{true}} = 100\ \mu\mathrm{s}, t_B^{\mathrm{true}} = 90\ \mu\mathrm{s}$（真实情况 B 快 $10\ \mu\mathrm{s}$），漂移序列 $d = [0, 10, 20, 30]\ \mu\mathrm{s}$：
>    - 若按 **`AABB` 顺序**测量：A 占用前两个低温槽位（$d_0=0, d_1=10$），测得均值 $\bar{t}_A^{\mathrm{AABB}} = \frac{(100+0) + (100+10)}{2} = \mathbf{105.0\ \mu\mathrm{s}}$；B 占用后两个高温槽位（$d_2=20, d_3=30$），测得均值 $\bar{t}_B^{\mathrm{AABB}} = \frac{(90+20) + (90+30)}{2} = \mathbf{115.0\ \mu\mathrm{s}}$。**`AABB` 错误地得出“A（$105\ \mu\mathrm{s}$）比 B（$115\ \mu\mathrm{s}$）快 $10\ \mu\mathrm{s}$”的完全相反结论**；
>    - 若按 **`ABBA` 镜像抵消顺序**测量：A 占用槽位 $0$ 和 $3$（$d_0+d_3 = 0+30 = 30$），测得均值 $\bar{t}_A^{\mathrm{ABBA}} = \frac{(100+0) + (100+30)}{2} = \mathbf{115.0\ \mu\mathrm{s}}$；B 占用槽位 $1$ 和 $2$（$d_1+d_2 = 10+20 = 30$），测得均值 $\bar{t}_B^{\mathrm{ABBA}} = \frac{(90+10) + (90+20)}{2} = \mathbf{105.0\ \mu\mathrm{s}}$。由于两组槽位的一阶时间中心完全重合（$\frac{0+3}{2} = \frac{1+2}{2} = 1.5$），任何一阶线性时钟/温度漂移被 **$100\%$ 代数抵消**，精确还原出 $\bar{t}_A - \bar{t}_B = 115.0 - 105.0 = \mathbf{+10.0\ \mu\mathrm{s}}$ 的真实差异。

---

![AI 系统跑分中的五大失真陷阱与算术平均加速比悖论](ch20/ch20-five-benchmark-gaming-traps.svg)
*图 20-10　AI 系统跑分中的五大失真陷阱与算术平均加速比悖论（候选 A 平均加速比 $5.25\times$ 反慢于候选 B $2.0\times$，算例 20-5）。*

---

### 20.4.2 从算子与通信基准（`FlashInfer-Bench` / `CCEval`）、漂移基准（`DriftBench`）到消除采样失真的可信剖析（`Blink` OSDI '26 #23）与两阶段金丝雀门禁

为了彻底堵死上述跑分漏洞，2025–2026 年学术界与工业界联合确立了新一代 AI 基础设施评测与性能剖析标准（见**图 20-11** 与**表 20-5**）[^ch20-flashinfer-bench][^ch20-cceval]：

1. **算子层端到端工作流基准（`FlashInfer-Bench` MLSys '26）**：不再孤立评测单一静态形状的 Kernel 峰值 TFLOPS，而是强制绑定真实生产形状分布直方图 $(p_1, p_2, \dots, p_K)$，以**分布加权总耗时 $\sum_i p_i T_i$（等价于加权调和平均加速比）**与端到端服务上线后的动态调度开销作为唯一排名指标。
2. **集合通信真实拓扑与影子流量基准（`CCEval` NSDI '26）**：抛弃空载实验室下的单流 `nccl-tests` 峰值带宽曲线，在注入背景多租户拥塞流、多网卡条带失衡与真实 LLM 算子间微秒级抖动的条件下，评测集合通信对端到端步进时间的影响。
3. **动态分布漂移下的系统稳健性基准（`DriftBench` MLSys '26 & `ServeGen` NSDI '26）**：模拟生产流量从“短输入长输出（代码生成/RLVR 推理）”向“超长文档输入短输出（RAG 问答）”的实时非平稳跳变，衡量推理引擎在内存碎片、KV 缓存驱逐与 PD 配比失衡下的 **SLO 达标有效吞吐（Goodput）**。
4. **消除扁平负载 PMU 采样失真的可信性能剖析（`Blink` OSDI '26 #23: *When Sampling Lies: Recovering Trustworthy Profiling on Flat Workloads*）**[^ch20-blink]：
   不仅外部跑分容易失真，连我们日常依赖的底层性能剖析器（Profiler）也可能说谎。在现代大模型推理框架（如 SGLang、vLLM 的异步主机调度环、零拷贝分词器与微秒级算子发射器）中，CPU 与运行时负载呈现出高度**扁平化（Flat Workloads）**的特征——由成百上千个执行时间仅数微秒的极短函数、内联原子操作与中断处理块拼成。**OSDI '26 #23（`Blink`）** 揭示了一个严峻的事实：传统的硬件性能监控单元（PMU）周期性中断采样在这些扁平负载上会出现系统性的**采样滑移（Sampling Skid）与中断屏蔽盲区**——例如在关中断临界区或高频短函数退出边界处，PMU 溢出中断总是滞后数条至数十条指令才被捕获，导致剖析报告将高达 $30\%\text{–}50\%$ 的主机开销张冠李戴地归因到无辜的下游函数上。`Blink` 通过结合轻量级编译器边界校准与无偏采样修正，在不引入高昂全量插桩开销的前提下，恢复了对微秒级扁平运行时负载的真实归因。

![FlashInfer-Bench、CCEval、DriftBench 与 Blink 的四维评测与剖析框架](ch20/ch20-flashinfer-bench-cceval-and-blink-profiling.svg)
*图 20-11　`FlashInfer-Bench`（算子基准）、`CCEval`（集合通信基准）、`DriftBench`（负载漂移）与 `Blink`（OSDI '26 消除采样失真）的四维评测体系。*

![十万卡集群有效训练时间比对检查点周期、恢复时间与故障率的响应对比](ch20/ch20-effective-training-time-ratio-ettr-model.svg)
*图 20-12　十万卡集群有效训练时间比（ETTR）对检查点周期 $\tau_{\mathrm{ckpt}}$、恢复时间 $T_{\mathrm{rec}}$ 与故障率 $\lambda$ 的响应对比（$63.22\%$ vs. $92.34\%$，算例 20-1）。*

![新节点入网与故障修复后的自动化烤机准入流水线](ch20/ch20-automated-burn-in-fleet-health-pipeline.svg)
*图 20-13　新节点入网与故障修复后的自动化烤机（Burn-in：满载热压测 + 确定性 SDC 校验 + 光链路裕量扫描）准入流水线。*

| 跑分失真维度 | 传统失真做法（Benchmark Gaming） | 导致的线上生产事故 | 工业级可信评测标准（2025–2026） | 本书锁定校验案例与阈值 |
| --- | --- | --- | --- | --- |
| **跨形状聚合指标** | 对各形状加速比直接求算术平均 | 号称 $5.25\times$ 加速，上线后反慢 $5\%$ | **`FlashInfer-Bench` 分布加权总耗时** | 候选 A 需 $p > 15/19$（$78.95\%$）才快于 B |
| **对比测量时序** | 单调顺序 `AABB` 连续跑分 | 把发热降频当成后测方案性能差 | **`ABBA` 镜像对称交错测量 + 固定锁频** | 精确抵消 $d=[0,10,20,30]\ \mu\mathrm{s}$ 还原 $+10\ \mu\mathrm{s}$ |
| **编译器计划选择** | 仅信静态代价分或单次均值 | 寄存器溢出/竞争导致 P99 尾延迟爆炸 | **实测尾延迟重排 + 竞争系数 $k$ 敏感度测试** | 静态 `100>90` 反转为实测 `95ms<140ms`；$k^*=10/9$ |
| **运行时热点剖析** | 直接采信默认 PMU 采样火焰图 | 采样滑移将 $40\%$ 开销错怪给下游函数 | **`Blink`（OSDI '26）边界校准无偏剖析** | 消除扁平微秒函数采样滑移盲区 |
| **新版本生产放量** | 离线跑分通过即全量替换上线 | 尾延迟超标或数值偏差冲击全网请求 | **两阶段金丝雀门禁（$5\% \to 50\% \to 100\%$）** | $5\%$ 阶段尾延迟 $190\ \mathrm{ms} > 150\ \mathrm{ms}$ 立即熔断回滚 |

*表 20-5　AI 基础设施五大跑分失真机制、生产事故后果与工业级反作弊评测规范总表。*

在实际将一项编译器优化或新算子推向生产集群时，结合本仓库锁定的 `case-studies/profile-and-plan-ranking.md`、`case-studies/execution-feedback.md` 与 `case-studies/optimization-evaluation-and-deployment.md`，必须严格执行**“实测反哺重排 + 两阶段金丝雀门禁（Two-Stage Canary Gate）”**标准作业程序：

- **用硬件实测执行反馈覆盖静态代价模型（Execution Feedback Reranking）**：如 `case-studies/profile-and-plan-ranking.md` 与 `case-studies/execution-feedback.md` 所示，静态代价模型给候选计划 A 打出了最高分 `100`（候选 B 仅为 `90`），但上卡实测发现计划 A 耗时高达 `140 ms`，而计划 B 仅需 **`95 ms`**；更进一步，当计划 A 虽然单次均值略低或接近（`[90, 95, 190] ms`，均值 `125.0 ms`），但其最坏尾延迟高达 `190 ms`，而计划 B（`[110, 112, 115] ms`，均值 `112.33 ms`）最坏耗时仅 **`115 ms`** 时，生产编译缓存必须按**最坏情况尾延迟上限（P99 / Max Latency）**锁定计划 B。
- **两阶段金丝雀放量与自动熔断回滚（Canary Rollout & Auto-Rollback）**：如 `case-studies/optimization-evaluation-and-deployment.md` 所锁定，新版本先以 **$5\%$ 流量**进入金丝雀阶段（Canary Stage），同时校验两道硬门禁：（1）与基线输出对数概率的 KL 散度 / 黄金测试集匹配率无退化；（2）P99 尾延迟不得超过上限阈值 $150\ \mathrm{ms}$。一旦在 $5\%$ 阶段观测到尾延迟跃升至 **`190 ms`**（$> 150\ \mathrm{ms}$），发布控制器在毫秒内自动阻断向 $50\%$ 阶段的扩流并立即回滚至安全基线。

下面给出集成 **ABFT 矩阵乘在线校验与修复、算术平均加速比反例检验、`ABBA` 漂移抵消评测器以及两阶段金丝雀门禁** 的生产级参考实现：

```python
import numpy as np

def abft_gemm_verify_and_repair(
    A: np.ndarray, B: np.ndarray, C_observed: np.ndarray, tol: float = 1e-3
) -> tuple[bool, tuple[int, int] | None, float, np.ndarray]:
    """
    算子级代数不变量校验 (ABFT, 算例 20-3):
    利用 1^T (AB) = (1^T A) B 与 (AB) 1 = A (B 1) 在线拦截并原地修复单元素 SDC 翻转
    """
    col_sum_A = np.sum(A, axis=0)          # [K]
    row_sum_B = np.sum(B, axis=1)          # [K]
    expected_col_sums = col_sum_A @ B      # [N], O(KN)
    expected_row_sums = A @ row_sum_B      # [M], O(MK)

    actual_col_sums = np.sum(C_observed, axis=0)
    actual_row_sums = np.sum(C_observed, axis=1)

    row_res = actual_row_sums - expected_row_sums
    col_res = actual_col_sums - expected_col_sums

    bad_rows = np.where(np.abs(row_res) > tol)[0]
    bad_cols = np.where(np.abs(col_res) > tol)[0]

    if bad_rows.size == 0 and bad_cols.size == 0:
        return True, None, 0.0, C_observed

    if bad_rows.size == 1 and bad_cols.size == 1:
        r_idx, c_idx = int(bad_rows[0]), int(bad_cols[0])
        err_val = float(row_res[r_idx])
        C_repaired = C_observed.copy()
        C_repaired[r_idx, c_idx] -= err_val
        return False, (r_idx, c_idx), err_val, C_repaired

    raise RuntimeError(f"Multi-element SDC detected: rows={bad_rows}, cols={bad_cols}; trigger OpGuard replay")


def evaluate_benchmark_abba_and_weighted_speedup(
    base_times_us: list[float],
    cand_a_times_us: list[float],
    cand_b_times_us: list[float],
    shape_weights: list[float],
    true_a_us: float = 100.0,
    true_b_us: float = 90.0,
    thermal_drift_us: tuple[float, float, float, float] = (0.0, 10.0, 20.0, 30.0)
) -> dict[str, float]:
    """
    算例 20-5 (锁定 case-studies/optimization-evaluation-and-deployment.md):
    1. 揭露算术平均加速比悖论 (Mean-of-Speedups Paradox)，对比分布加权总耗时
    2. 对比单调顺序 AABB 与镜像对称顺序 ABBA 对一阶线性热漂移的代数抵消
    """
    base = np.asarray(base_times_us, dtype=np.float64)
    ca = np.asarray(cand_a_times_us, dtype=np.float64)
    cb = np.asarray(cand_b_times_us, dtype=np.float64)
    w = np.asarray(shape_weights, dtype=np.float64)

    # 算术平均加速比 (失真指标) vs. 真实加权总耗时
    arith_speedup_a = float(np.mean(base / ca))
    arith_speedup_b = float(np.mean(base / cb))
    weighted_time_a = float(np.sum(w * ca))
    weighted_time_b = float(np.sum(w * cb))

    # AABB vs. ABBA 漂移抵消实验
    d0, d1, d2, d3 = thermal_drift_us
    aabb_diff = ((true_a_us + d0 + true_a_us + d1) * 0.5) - ((true_b_us + d2 + true_b_us + d3) * 0.5)
    abba_diff = ((true_a_us + d0 + true_a_us + d3) * 0.5) - ((true_b_us + d1 + true_b_us + d2) * 0.5)

    return {
        "arith_mean_speedup_a": round(arith_speedup_a, 2),
        "arith_mean_speedup_b": round(arith_speedup_b, 2),
        "weighted_time_a_us": round(weighted_time_a, 2),
        "weighted_time_b_us": round(weighted_time_b, 2),
        "aabb_measured_diff_us": round(aabb_diff, 2),
        "abba_measured_diff_us": round(abba_diff, 2)
    }


def evaluate_canary_rollout_gate(
    canary_latencies_ms: list[float],
    quality_match_rate: float,
    max_tail_budget_ms: float = 150.0,
    min_quality_match: float = 0.995
) -> dict[str, object]:
    """两阶段金丝雀发布门禁 (锁定 case-studies/optimization-evaluation-and-deployment.md)"""
    worst_tail_ms = max(canary_latencies_ms)
    passed = (worst_tail_ms <= max_tail_budget_ms) and (quality_match_rate >= min_quality_match)
    return {
        "stage_pct": 5,
        "worst_tail_ms": worst_tail_ms,
        "budget_ms": max_tail_budget_ms,
        "quality_match_rate": quality_match_rate,
        "action": "PROMOTE_TO_50_PCT" if passed else "ROLLBACK_IMMEDIATELY"
    }


def verify_profile_ranking_and_telemetry_drain(
    contention_factor_k: float = 1.3
) -> dict[str, object]:
    """
    锁定 case-studies/profile-and-plan-ranking.md 与 case-studies/evaluation-and-agent-records.md:
    1. 核算通信竞争系数 k 跨过 k* = 10/9 时计划 A (70+30k) 与计划 B (90+12k) 的排名反转
    2. 核算监控遥测突发积压 (46.875 MiB, 净排空速率 624 KiB/s) 的精确排空恢复时间 76.92 s
    """
    k_star = (90.0 - 70.0) / (30.0 - 12.0)  # 10/9 = 1.111...
    t_a_ms = 70.0 + 30.0 * contention_factor_k
    t_b_ms = 90.0 + 12.0 * contention_factor_k
    winner_under_contention = "Plan B" if t_b_ms < t_a_ms else "Plan A"

    # 遥测突发积压: (5120 - 1920) KiB/s * 15 s = 48,000 KiB (46.875 MiB)
    backlog_kib = (5120.0 - 1920.0) * 15.0
    net_drain_kib_per_s = 1920.0 - 1296.0  # 624 KiB/s
    drain_seconds = backlog_kib / net_drain_kib_per_s  # 76.923 s

    return {
        "crossover_k_star": round(k_star, 4),
        "t_a_ms_at_k": round(t_a_ms, 2),
        "t_b_ms_at_k": round(t_b_ms, 2),
        "winner_under_contention": winner_under_contention,
        "telemetry_backlog_mib": round(backlog_kib / 1024.0, 3),
        "telemetry_drain_seconds": round(drain_seconds, 2)
    }


def blink_debiased_flat_profiler(
    raw_pmu_samples: dict[str, int],
    skid_transition_matrix: np.ndarray,
    func_names: list[str]
) -> dict[str, float]:
    """
    OSDI '26 #23 (Blink: Recovering Trustworthy Profiling on Flat Workloads):
    利用编译器边界校准得到的采样滑移转移矩阵 M，对原始失真 PMU 采样向量 y 执行逆投影还原真实归因
    """
    y = np.asarray([raw_pmu_samples[f] for f in func_names], dtype=np.float64)
    true_counts = np.linalg.solve(skid_transition_matrix, y)
    true_counts = np.maximum(true_counts, 0.0)
    total = float(np.sum(true_counts)) + 1e-12
    return {f: round(float(c / total) * 100.0, 2) for f, c in zip(func_names, true_counts)}
```

---

## 20.5 全书二十章终极大收束：从一次矩阵乘法的五个追问，到驾驭十万卡智能工厂的工程哲学

### 20.5.1 全书二十章知识坐标系总览与双轨主线（Qwen3-8B Dense 轨 & DeepSeek V4.1 / MoE 轨）二十章收官总表

翻开本书第 1 章第 1 节时，我们曾向读者提出过贯穿全书的“**一次矩阵乘法的五个追问**”：数据从哪来（存储与显存墙）、由谁算（张量核与算术强度）、怎么协同（超节点与集群网络）、如何服务与进化（推理、训练、RLVR、多模态与具身 VLA）、以及拿什么支撑（电力、液冷、安全与十万卡容错）。

经过二十章的系统推演，我们用两条真实、具体、可复算的工程主线——**轨道 A：Qwen3-8B（36 层 GQA 稠密模型，代表单卡/端侧/机载实时控制极限）** 与 **轨道 B：DeepSeek V4.1 / V3 671B（61 层 MLA + 256 专家 MoE 稀疏模型，激活 37B，代表超节点万卡集群训推旗舰）**——将原本碎片化的算法论文、芯片手册、网络协议与机房规范编织成了一张严丝合缝的定量工程坐标系（如**图 20-14**、**图 20-15** 与**表 20-6** 所示）。

![全书 20 章双轨贯穿案例从第 1 章到第 20 章的完整工程参数演进总表](ch20/ch20-twenty-chapter-dual-track-master-ledger.svg)
*图 20-14　全书 $20$ 章双轨贯穿案例（Qwen3-8B Dense 轨与 DeepSeek V4.1 / MoE 轨）从第 1 章到第 20 章的完整工程参数演进总表。*

![全书 20 章模型与负载、芯片与系统、推理训练与六大前沿底座的知识体系全景图](ch20/ch20-twenty-chapter-architecture-knowledge-graph.svg)
*图 20-15　全书 $20$ 章“模型与负载 $\to$ 芯片与系统 $\to$ 推理、训练与六大前沿底座”的知识体系全景图。*

| 章号 | 章节名称 | 轨道 A：Qwen3-8B (Dense 36L, $d=4096$, GQA 32Q/8KV) 锁定指标 | 轨道 B：DeepSeek V4.1 / MoE (61L, 671B 总参 / 37B 激活, MLA) 锁定指标 |
| :-: | --- | --- | --- |
| **第 1 章** | 初识 AI Infra | BF16 权重 $16.4\ \mathrm{GB}$；H100 单卡可放，算术强度分界初探 | FP8 权重 $671\ \mathrm{GB}$（BF16 $1.34\ \mathrm{TB}$）；跨节点超节点必选 |
| **第 2 章** | 模型架构 | GQA 8KV 头将单词元 KV 压至 **$144\ \mathrm{KB/tok}$**（比 MHA 省 $4\times$） | MLA 潜向量 $d_c=512$ 将 KV 压至 **$70.2\ \mathrm{KB/tok}$**；256 专家选 8 |
| **第 3 章** | 推理与训练负载 | Prefill 算力受限 vs. Decode $B=1$ 访存受限；$\mathrm{TTFT}/\mathrm{TPOT}$ 契约 | $128\mathrm{K}$ 长上下文 Prefill 平方计算墙；MoE 专家路由长尾倾斜 |
| **第 4 章** | 加速器架构 | H100 嵴点 $295\ \mathrm{FLOPs/B}$；B200 FP4 Tensor Core 与 TMA 异步引擎 | 细粒度块缩放（MXFP8/NVFP4）消除激活离群值量化崩塌 |
| **第 5 章** | 算子与运行时 | FlashAttention 在线 Softmax 锁住 SRAM；CUDA Graph 消除启动税 | FlashAttention-4 非对称流水；DeepGEMM $1\times 128$/$128\times 128$ 细粒度 FP8 |
| **第 6 章** | 超节点 | 单机 8 卡 NVLink 4（$900\ \mathrm{GB/s}$）承载 TP=2/4/8 低延迟全归约 | 灵衢 UB 2.0.1（$2,016\ \mathrm{GB/s}$）/ NVL72 机架级超节点承载大 EP |
| **第 7 章** | 数据中心网络 | 单轨/双轨 RoCEv2 无损配置；ECN 与 PFC 死锁水线调优 | 双平面轨道直连（Rail-Optimized）+ Tomahawk 6 UEC 1.0 集合通信 |
| **第 8 章** | 推理优化 | PagedAttention 零外部碎片；DSpark `7/3/2/2` 紧凑验证与 XGrammar-2 | MTP 多词元预测草稿率提升；RadixAttention 跨轮共享系统前缀 |
| **第 9 章** | 分布式推理 | 小规模 TP+PP 部署与 Mooncake KV 跨实例迁移 | Prefill（EP32）+ Decode（EP144/320）分离；DeepEP 0-SM IBGDA 重叠 |
| **第 10 章** | 训练系统 | ZeRO-2 / FSDP 显存切分与激活重计算（Activation Checkpointing） | DualPipe 双向流水掩盖气泡；Megatron 4D 并行与 EPLB 专家负载均衡 |
| **第 11 章** | 资源调度与运行环境 | vLLM 冷启动五阶段拆解（权重加载占大头）与 Serverless 弹性缩容 | SGLang 六阶段权重原地热更新“3 成 1 败”边界与多级拓扑亲和调度 |
| **第 12 章** | 端边云协同 | Apple M4 Max / 手机 NPU INT4（$4.6\ \mathrm{GB}$）静态分块图与热节流治理 | 端侧小模型草稿 + 云端 671B 大模型验证的协同混合路由与隐私分流 |
| **第 13 章** | AI 原生存储与数据底座 | 本地 NVMe 缓存 + GDS 零拷贝加载检查点 | 3FS 全闪存并行存储（$6.6\ \mathrm{TiB/s}$ 级读取）+ 多模态在线混洗流水线 |
| **第 14 章** | 模型与系统硬件协同设计 | 计入推理词元总量 $D_{\mathrm{inf}}$ 的过度训练（Over-Training）定律 | 从 MHA/FP16 到 MLA+MoE+FP8+PD 分离的 **百倍单 Token 降本比率链** |
| **第 15 章** | 吉瓦级数据中心与电力热学 | 单卡 DVFS 调频帕累托前沿与 *TAPAS* 30 请求热感知路由回本 | 吉瓦园区 $L\frac{\mathrm{d}i}{\mathrm{d}t}$ 电网阶跃、BESS 毫秒平抑、微通道液冷与跨洲 DiLoCo |
| **第 16 章** | 机密 AI 计算与安全沙箱 | H100 Confidential Computing TEE 与 `PipeLLM` $192\ \mathrm{MiB}$ 三缓冲加密 | DSec 38 万 Firecracker 微虚机 `virtio-pmem` + DAX 省内存 $40.2\%$ |
| **第 17 章** | 多模态、视频生成与世界模型 | 作为多模态 VLM 视觉编码器与任意分辨率 Dynamic High-Res 词元预算器 | 3D DiT $489\mathrm{K}$ 词元 2D USP 并行；E-P-D-G 四池分离与 Step/CFG 蒸馏 |
| **第 18 章** | 大规模后训练（RLVR）系统 | 32B/8B 级 GRPO 比 PPO 省半显存（$1,152\to 576\ \mathrm{GB}$）；PRM 评分树 | `Weave` 异构池反相共享；`RollPacker` 尾部拼箱与全局归一化；`DistRS` 沙箱 |
| **第 19 章** | 具身智能（VLA）与物理仿真 | 机载 System 2（$2\ \mathrm{Hz}$）+ 150M System 1（$100\ \mathrm{Hz}$，$\mathrm{WCET}\le 4.2\ \mathrm{ms}$） | 单卡 $4,096$ 并行 GPU 原生仿真（省 $43.39\ \mathrm{GB/s}$ PCIe）+ 3DGS 神经重建 |
| **第 20 章** | 十万卡可靠性与基准评测科学 | `FlashInfer-Bench` 加权评测、`ABBA` 漂移抵消与 `Blink` 无偏剖析 | 十万卡 $\mathrm{MTBF}=30.4\ \mathrm{min}$；$S_{\min}=40$ 热备池、ABFT/`OpGuard` 与 `ETTR=92.34%` |

*表 20-6　全书二十章双轨贯穿工程主线（Qwen3-8B Dense 轨 vs. DeepSeek V4.1 / MoE 轨）收官总账本。*

为了帮助不同技术背景的读者高效将这二十章转化为日常工作中的生产力，**表 20-7** 给出了针对四类核心工程岗位的**定制化进阶阅读与实验路线图**：

| 读者岗位画像 | 核心工作目标 | 第一阶段：必读基础核心章 | 第二阶段：岗位深水区专精章 | 必做可运行实验与锁定案例 |
| --- | --- | --- | --- | --- |
| **大模型算法与后训练（RLVR）工程师** | 设计高性价比模型结构、长思维链 RLVR 与多模态训练 | 第 1、2、3、10 章 | **第 14、17、18 章** | `ch14/14-01`、`ch17/17-01`、`ch18/18-01` 及 `rollout-tail-and-sampling.md` |
| **推理引擎、编译器与内核优化工程师** | 压榨单卡 TFLOPS/带宽利用率、降低 TTFT/TPOT 尾延迟 | 第 1、3、4、5 章 | **第 8、9、12、20 章** | `ch14/14-02`、`ch20/20-01` 及 `optimization-evaluation-and-deployment.md` |
| **万卡分布式训练、网络与集群 SRE 架构师** | 保障万卡/十万卡高 ETTR、无损网络、存储与电力液冷 | 第 4、6、7、10 章 | **第 11、13、15、16、20 章** | `ch15/15-01`、`ch16/16-01`、`ch20/20-01` 及 `collective-paths-and-diagnosis.md` |
| **多模态、视频生成与具身智能（VLA）系统工程师** | 构建 3D 视频生成集群、机载硬实时控制与万级 GPU 仿真 | 第 3、4、5、8 章 | **第 12、17、19、20 章** | `ch17/17-01`、`ch19/19-01`、`ch20/20-01` |

*表 20-7　面向四类 AI 基础设施工程岗位的全书二十章定制化阅读与实验实战路线图。*

---

### 20.5.2 AI 基础设施架构师的十条第一性原理铁律

如果把这二十章近百万字的技术细节、上百道定量算例与三百余幅架构配图浓缩为一张随身携带的工程罗盘，那就是以下**十条 AI 基础设施第一性原理铁律**（见**图 20-16** 与**表 20-8**）：

![AI 基础设施架构师的十条第一性原理终极法则总结图](ch20/ch20-ten-first-principles-of-ai-infra.svg)
*图 20-16　AI 基础设施架构师的十条第一性原理终极法则总结图。*

1. **第一铁律（字节优先于浮点：算术强度分界法则）**：任何脱离数据搬移字节数（Bytes）空谈峰值算力（TFLOPS）的架构设计都是空中楼阁。动手优化前，永远先算工作负载的算术强度 $\mathrm{AI} = \mathrm{FLOPs}/\mathrm{Bytes}$ 并与芯片嵴点 $\mathcal{R}_{\mathrm{ridge}} = \mathrm{Peak\_FLOPS}/\mathrm{Peak\_BW}$ 对比——Prefill 与训练反向传播活在算力区，自回归 Decode、KV 缓存与物理状态转移活在带宽区。
2. **第二铁律（片上 SRAM 局部性与在线融合法则）**：HBM 带宽增长永远追不上张量核计算增长。从 FlashAttention-1/2/3/4、DeepGEMM 到 GPU 原生三合一物理仿真，一切百倍级算子与系统加速的本质，都是利用代数结合律（如在线 Softmax 重缩放）将中间张量牢牢锁在寄存器与片上 SRAM（或单卡 HBM）内，消灭跨层级往返读写。
3. **第三铁律（通信域与物理互联拓扑同构映射法则）**：并行切分不是数学排列组合，而是向物理铜缆与光纤的妥协。最怕延迟的张量并行（TP）与序列全互换（Ulysses All-to-All）必须关在超节点高带宽域（NVLink / 灵衢 UB）笼子里；专家并行（EP）走同轨交换机平面；数据并行（DP）与流水线微批（PP）跨核心层并通过计算—通信双流分块完全掩盖。
4. **第四铁律（异构算术强度负载物理分离法则）**：绝不让计算密集型任务与访存/I/O 密集型任务在同一个同步时钟周期内互相抢占。云端推理需要 Prefill—Decode（PD）分离，多模态与视频生成需要 Encode—Prefill—Decode—Generate（E-P-D-G）四池分离，大规模后训练（RLVR）需要 Rollout—Reward Sandbox—Learner 异步解耦，机载具身智能需要 System 2（慢思考 VLM）与 System 1（快反射动作头）双频硬件隔离。
5. **第五铁律（重尾分布支配全局效率与尾部拼箱法则）**：在真实世界里，无论是长思维链（Long-CoT）生成长度、沙箱代码验证时间，还是并行仿真中的多点接触碰撞，统统服从幂律重尾分布（Power-Law Heavy Tail）。同步屏障下的系统吞吐永远由最慢的 $1\%$ 尾部任务决定；唯有通过动态尾迁移拼箱（RollPacker）、前缀和流式紧缩（Stream Compaction）、陈旧度有界部分采样（Partial Rollout）与条件剩余时间抢占（DistRS），才能消灭长尾空泡。
6. **第六铁律（全生命周期 Token 经济学与过度训练法则）**：模型的最优参数量不由单次预训练预算单独决定，而由训练算力与全生命周期累计推理词元量 $D_{\mathrm{inf}}$ 共同决定。对于高频服役的基座模型，在训练阶段付出 $3\times\text{–}10\times$ Chinchilla 词元倍数的过度训练成本、配合 MLA/GQA 与 FP8/FP4 原生低精度，可在推理生命周期内收回十倍以上的硬件与电力红利。
7. **第七铁律（兆瓦级电力、液冷热容与安全边界的物理刚性法则）**：当单机架功率突破 $100\ \mathrm{kW}$、单园区迈向 $1\ \mathrm{GW}$ 时，变压器容量、电网同步旋转惯量（毫秒级 $L \frac{\mathrm{d}i}{\mathrm{d}t}$ 功率斜率阶跃）与微通道冷板热阻成为了不可逾越的硬边界；而多租户微虚机沙箱（DAX 内存去重）与三缓冲加密流水线（`PipeLLM`）则是智能体时代不可妥协的底线。
8. **第八铁律（物理世界硬实时 WCET 与确定性安全兜底法则）**：在具身智能与自动驾驶中，时间延迟直接乘上运动速度化为空间盲飞碰撞误差 $\Delta x = v \cdot \Delta t$。面向物理世界的推理系统优化目标不再是平均吞吐，而是通过固定配额词元剪枝、静态 CUDA Graph、GPU MIG 分区与 $1,000\ \mathrm{Hz}$ 控制屏障安全护盾（CBF），严守最坏情况执行时间（WCET）上限。
9. **第九铁律（十万卡故障常态化与毫秒级在线防毒自愈法则）**：在十万卡集群上，平均每 30 分钟必发一次硬件异常，故障不是偶发例外而是稳态背景。必须用 `<0.4%` 的热备节点池（Hot-Spare Pool）、`<0.5%` 开销的算子级 ABFT 校验、`AllReduce` 前统计拦截、`OpGuard` 位级确定性重放与 `TrainMover` 秒级无重启热接管，将静默数据损坏（SDC）与单点掉卡的冲击半径压缩在毫秒与单微批之内。
10. **第十铁律（可证伪、防漂移与质量守恒的评测科学法则）**：永远对脱离真实负载分布的“算术平均加速比”保持警惕，永远用 `ABBA` 镜像交错实验消除热漂移，永远用 `Blink` 等无偏剖析工具核验热点归因，永远用影子流量 KL 散度与两阶段金丝雀尾延迟门禁守住生产上线前的最后一道关口。

| 铁律编号 | 法则名称 | 核心第一性原理方程 / 判据 | 攻克的物理墙 | 对应核心章节 |
| :-: | --- | --- | --- | :-: |
| **第一铁律** | 字节优先于浮点分界法则 | $T = \max\!\left(\frac{\mathrm{FLOPs}}{\mathrm{Peak\_FLOPS}}, \frac{\mathrm{Bytes}}{\mathrm{Peak\_BW}}\right)$ | HBM 显存带宽墙 | 第 1、3、4 章 |
| **第二铁律** | 片上局部性与在线融合法则 | $m_{\mathrm{new}} = \max(m_{\mathrm{old}}, \tilde{m}),\; \ell_{\mathrm{new}} = e^{m_{\mathrm{old}}-m_{\mathrm{new}}}\ell_{\mathrm{old}} + \tilde{\ell}$ | 跨存储层级读写墙 | 第 2、5、19 章 |
| **第三铁律** | 通信域与物理拓扑同构法则 | $T_{\mathrm{comm}}(\text{TP}) \ll T_{\mathrm{comm}}(\text{EP}) \ll T_{\mathrm{comm}}(\text{DP/PP})$ | 光纤/铜缆延迟与带宽墙 | 第 6、7、10、17 章 |
| **第四铁律** | 异构算术强度物理分离法则 | $\mathrm{AI}_{\mathrm{Prefill/Train}} \gg \mathcal{R}_{\mathrm{ridge}} \gg \mathrm{AI}_{\mathrm{Decode/Rollout}}$ | 异构阶段资源干扰墙 | 第 9、17、18、19 章 |
| **第五铁律** | 重尾分布支配与尾部拼箱法则 | $\mathbb{E}[S - e \mid S > e] \gg \mathbb{E}[S]\; (\text{Power-Law Tail})$ | 同步屏障长尾木桶墙 | 第 11、18、19 章 |
| **第六铁律** | 全周期词元经济学法则 | $C_{\mathrm{total}} = C_{\mathrm{train}}(N, D_{\mathrm{tr}}) + D_{\mathrm{inf}} \cdot c_{\mathrm{inf}}(N)$ | 推理生命周期总成本墙 | 第 2、8、14 章 |
| **第七铁律** | 兆瓦电力液冷与安全刚性法则 | $\Delta V = L\frac{\mathrm{d}i}{\mathrm{d}t},\; T_{\mathrm{j}} = T_{\mathrm{coolant}} + P \cdot R_{\mathrm{th}}$ | 电网惯量、热流密度与安全墙 | 第 13、15、16 章 |
| **第八铁律** | 物理世界硬实时 WCET 法则 | $\Delta x_{\mathrm{drift}} = v \cdot \Delta t_{\mathrm{inf}} \le \delta_{\max}$ | 三维物理碰撞与相位滞后墙 | 第 12、19 章 |
| **第九铁律** | 十万卡防毒与秒级自愈法则 | $\mathrm{MTBF}_{N} = \frac{\mathrm{MTBF}_1}{N},\; \tau_{\mathrm{opt}} = \sqrt{2 C_{\mathrm{ckpt}} M}$ | 十万卡中断与 SDC 毒化墙 | 第 10、18、20 章 |
| **第十铁律** | 可证伪与防漂移评测法则 | $\bar{T}_{\mathrm{weighted}} = \sum_i p_i T_i,\; \Delta T_{\mathrm{ABBA}} \equiv \mu_A - \mu_B$ | 跑分失真与采样滑移盲区墙 | 第 5、11、20 章 |

*表 20-8　AI 基础设施架构师的十条第一性原理铁律、控制方程与全书章节映射总表。*

---

### 20.5.3 混沌故障注入演练（Chaos Fault Injection）、灰度金丝雀上线门禁与向第 21–24 章四大新兴疆域的延伸导引

在十万卡智能工厂中，一套从未在真实压力下演练过的容错系统，在真正遭遇级联故障时往往会因为“监控守护进程自身超时”“RDMA 重连风暴”或“热备池权限配置缺失”而彻底失效。同样，一套只靠离线脚本把关的性能评测流水线，也极易在内核驱动、NCCL 版本或算子库滚动升级时放过低概率精度退化与尾延迟回退。因此，十万卡集群的日常运维闭环必须由**主动混沌故障注入演练（Chaos Fault Injection GameDay）**与**两阶段灰度金丝雀上线门禁（Two-Stage Canary Rollout Gate）**共同锁死（**图 20-17**）：

1. **十万卡机队混沌故障注入演练（Chaos GameDay）的四大注入算子与闭环 SLA**：
   借鉴超大规模云服务商的混沌工程范式，AI 基础设施团队每周在影子训练分区或合成基准负载上定期注入四类受控故障（每批次共 $N_{\mathrm{chaos}} = 200$ 次独立故障事件），并以**故障检出召回率（Detection Recall）**、**平均检出时间（MTTD）**与**平均自愈恢复时间（MTTR）**作为硬性验收门禁：
   - **算术通路静默比特翻转注入（Tensor Core Bit-Flip SDC，50 次）**：通过调试寄存器或编译器插桩，在选定 Rank 的 GEMM 累加器或输出张量中随机翻转 1 个 `BFloat16` 指数位（如 $+32$ 阶跃），检验算子级 ABFT 行列校验和与 AllReduce 归约前跨秩梯度统计监控能否在单步内拦截，并触发 `OpGuard` 在 $50\ \mathrm{ms}$ 内完成位级对齐微批重算定位（实测召回率 $50/50 = 100\%$，$\mathrm{MTTD} = 0.042\ \mathrm{s}$）；
   - **计算降频与 PCIe 带宽抢占慢节点注入（Straggler & Thermal Throttle，50 次）**：在选定 GPU 上注入人为功耗墙下调（从 $700\ \mathrm{W}$ 降至 $350\ \mathrm{W}$）或高并发 NVMe 写盘抢占 PCIe 通道，制造 $2.0\text{–}5.0\ \mathrm{ms}$ 的同步滞后，检验 `NCCL Flight Recorder` 入队时间戳偏斜矩阵（Arrival Skew Matrix）能否在 $3.2\ \mathrm{s}$ 内定位唯一正偏斜根因卡，避免误判为全网拥塞（实测召回率 $50/50 = 100\%$）；
   - **800G 光模块 RS-FEC 高阶纠错灰故障注入（Optical Gray Failure，50 次）**：通过交换机端口可编程衰减器抬升误码率，迫使光模块频繁落入 Symbol Errors $\ge 12$ 的重传边缘（单轨延迟从 $0.8\ \mathrm{ms}$ 劣化至 $3.2\ \mathrm{ms}$），检验光链路遥测守护进程能否在链路完全闪断前主动排空（Drain）该轨流量（实测召回率 $49/50 = 98.0\%$，$\mathrm{MTTD} = 6.80\ \mathrm{s}$）；
   - **HBM 双比特不可纠正 ECC 与 RDMA 断链注入（Uncorrectable ECC / Link Drop，50 次）**：模拟瞬间掉卡或网卡 QP 断开，检验 `TrainMover` 能否在不销毁全局通信域的前提下，于 $0.68\ \mathrm{s}$ 内完成局部 RDMA QP 重映射与热备节点接管（实测召回率 $50/50 = 100\%$，$\mathrm{MTTD} = 0.84\ \mathrm{s}$）。
   综合全部 $200$ 次注入演练，自动化免疫底座实现 **$199/200 = 99.50\%$** 的综合故障召回率，平均检出时间 $\mathrm{MTTD} = \mathbf{2.70\ \mathrm{s}}$，端到端自动恢复时间 $\mathrm{MTTR}_{\mathrm{auto}} = \mathbf{3.85\ \mathrm{s}}$——相比传统依赖人工值班排查与 `torchrun` 全网重启的 $\mathrm{MTTR}_{\mathrm{manual}} = 1,500.0\ \mathrm{s}$（$25$ 分钟），故障恢复速度提升了 **$389.61\times$**。

2. **两阶段灰度金丝雀上线门禁（Two-Stage Canary Gate）**：
   任何新的算子内核（如新版 FlashAttention-4 / DeepGEMM）、通信库补丁或驱动固件在推向十万卡主集群前，必须依次闯过两道自动化门禁：
   - **Stage 1（离线确定性门禁）**：强制执行锁定 TDP 预热、真实流量形状分布加权计时（严禁算术平均加速比）、`ABBA` 对称交替测序（消除线性热漂移）与 `Blink` 硬件无偏剖析；若加权总耗时退步或 FP8/BF16 尾数误差超过 $1\ \mathrm{ulp}$，直接拦截拒绝；
   - **Stage 2（5% 线上金丝雀与影子门禁）**：在 5% 生产流量分片上试运行 $24\ \mathrm{h}$，实时监控 P99 尾延迟、梯度范数余弦相似度与 SDC 告警计数；一旦触发 $\Delta \mathrm{P99} > +3.0\%$ 或任何 1 次 SDC 事件，控制面在 $3\ \mathrm{s}$ 内自动将流量切回旧版黄金镜像。

![十万卡混沌故障注入演练与两阶段金丝雀门禁流水线](ch20/ch20-chaos-fault-injection-and-canary-rollout-gate.svg)
*图 20-17　十万卡机队混沌故障注入演练（Chaos GameDay）与两阶段金丝雀上线门禁流水线（算例 20-6，锁定 `experiments/ch20/20-02/run.py`）：通过主动注入 4 类 200 次硬件/灰故障实现 99.50% 自动召回与 3.85 s 亚秒/秒级自愈（比人工恢复快 389.61 倍），并用 Stage 1（ABBA + 加权耗时）与 Stage 2（5% 金丝雀自动回滚）锁死上线质量。*

3. **从前二十章通用大模型底座迈向第 21–24 章四大新兴疆域的架构桥梁**：
   至此，本书前二十章已经完整构筑了围绕“单次前向/反向传播与流式词元生成”的通用大模型核心基础设施。然而，2025–2026 年的 AI 工业界正在经历一场从**“单一模型词元生成（Token Generation）”向“系统级闭环智能体、全模态实时交互、EB 级合成数据飞轮与物理科学发现”**的范式跃迁。如**图 20-18** 所示，前二十章锻造的每一项底层技术，都将作为基石直接支撑本书最后四章（第 21–24 章）的四大新兴前沿：
   - **迈向第 21 章（Agentic AI 基础设施、树状状态分支缓存、安全代码沙箱与百万智能体编排）**：第 8–9 章的前缀缓存与 PD 分离，在面对智能体多轮 ReAct 工具调用、蒙特卡洛树搜索（MCTS）与强化学习多分支 Rollout 时，演化为**树状 KV 缓存写时复制（RadixAttention + Copy-on-Write）**、**微秒级代码执行安全沙箱（MicroVM / WebAssembly 快照克隆，将工具等待期间的 GPU KV 显存税降低 85%）**与**结构化语法状态机零开销掩码（`XGrammar`）**；
   - **迈向第 22 章（全双工原生语音基础设施、神经音频编解码器、微秒级流式推理与声学回声消除）**：第 19 章的机载硬实时确定性调度与第 9 章的流式解码，在面对人类自然对话 $<200\ \mathrm{ms}$ 端到端语音响应与随时打断（Barge-in）时，演化为**残差矢量量化神经音频编解码（RVQ Codec，$12.5\text{–}50\ \mathrm{Hz}$ 多码本并行生成）**、**双通道全双工听—说并发状态机（Moshi / GPT-4o 架构）**与**端云协同声学回声消除（AEC）及 WebRTC 抖动缓冲**；
   - **迈向第 23 章（EB 级预训练数据基础设施、分布式模糊去重、模型驱动质量过滤与合成数据工厂）**：第 13 章的高吞吐并行存储（3FS）与第 18 章的验证器农场，在面对互联网自然语料枯竭（Data Wall）时，演化为**EB 级多模态数据湖（Apache Iceberg + Lance 列式随机访问）**、**十万核级分布式 MinHash-LSH 模糊去重与后缀数组精确去重**，以及**消耗上万张 GPU 日产百亿高价值推理思维链的合成数据精炼工厂（Synthetic Data Factory）**；
   - **迈向第 24 章（AI for Science 科学智能基础设施、球谐等变算子、几何三角注意力与全书二十四章终极收束）**：第 4–7 章的算子融合与三维并行，在面对全地球高分辨率气象预报（GraphCast / FourCastNet）、生物大分子全原子扩散结构预测（AlphaFold 3）与千万原子分子动力学模拟时，演化为**三维球面傅里叶神经算子（SFNO）**、**显存复杂度 $\mathcal{O}(N^3)$ 的几何三角自注意力分块核（Tiled Triangle Attention）**、**$\mathrm{SE}(3)$ 等变球谐张量积算子**，并在第 24.5 节完成全书二十四章与十二条第一性原理铁律的终极交响收束。

![前二十章通用集群底座向第 21–24 章四大新兴疆域的延伸桥梁](ch20/ch20-core-cluster-to-four-new-frontiers-bridge.svg)
*图 20-18　从前二十章通用大模型与十万卡可靠性底座向第 21–24 章四大新兴疆域（自主智能体基础设施、全双工原生语音、EB 级合成数据工厂、AI for Science 科学智能及全书二十四章终极收束）的架构映射桥梁。*

> **算例 20-6（十万卡机队混沌故障注入 GameDay 演练召回率、平均恢复时间 MTTR 与 8-Rank 入队偏斜反演账本，锁定 `experiments/ch20/20-02/run.py`）**
> **场景设定**：（1）在一个 8-Rank AllReduce 同步通信组中，7 个健康 Rank 在 $t = 10.0\ \mathrm{ms}$ 准时入队，而 Rank 5 因 PCIe/降频慢节点故障在 $t = 12.0\ \mathrm{ms}$ 入队，全员到齐后 Ring AllReduce 真实网络传输耗时 $0.40\ \mathrm{ms}$，核算健康 Rank 观测到的表面通信时间、慢节点空等占比，以及 `TrainMover` 亚秒级热迁移（$0.68\ \mathrm{s}$）相比传统 `torchrun` 全网重启（$300.0\ \mathrm{s}$）的恢复加速比；（2）在十万卡机队上执行 $200$ 次 Chaos GameDay 故障注入（SDC 翻转、慢节点、800G 光模块 FEC 灰故障、HBM ECC 掉卡各 $50$ 次），核算综合召回率、平均检出时间 $\mathrm{MTTD}$ 与自动自愈加速比。
> **逐步推导**：
> 1. **8-Rank 入队偏斜反演与 `TrainMover` 亚秒级热接管**：全体完成时刻为 $\max(10.0, 12.0) + 0.40 = 12.40\ \mathrm{ms}$。7 个健康 Rank 观测到的 AllReduce 持续时间为 $12.40 - 10.0 = \mathbf{2.40\ \mathrm{ms}}$，其中等待 Rank 5 的空转时间为 $2.00\ \mathrm{ms}$（慢节点等待占比高达 $2.00 / 2.40 = \mathbf{83.33\%}$，真实网络传输仅占 $16.67\%$）。`NCCL Flight Recorder` 通过计算各 Rank 相对于中位数到达时刻（$10.0\ \mathrm{ms}$）的偏斜向量 $\Delta \mathbf{t} = [0, 0, 0, 0, 0, +2.0, 0, 0]\ \mathrm{ms}$，一步锁定 `straggler_rank = 5`。随后触发 `TrainMover` 在 **$0.68\ \mathrm{s}$** 内完成局部 RDMA QP 重映射与热备替换，相比传统 $300.0\ \mathrm{s}$ 全局销毁重启提速 **$441.18\times$**。
> 2. **200 次混沌注入演练总账本**：四类故障分别检出 $50, 50, 49, 50$ 次，总检出 $199/200$ 次，综合召回率 **$\mathrm{Recall} = 99.50\%$**；四类检出耗时加权均值 $\mathrm{MTTD} = \frac{50\times 0.042 + 50\times 3.20 + 49\times 6.80 + 50\times 0.84}{199} = \frac{537.30}{199} = \mathbf{2.70\ \mathrm{s}}$；自动自愈恢复耗时加权均值 $\mathrm{MTTR}_{\mathrm{auto}} = \mathbf{3.85\ \mathrm{s}}$，相比人工处置 $\mathrm{MTTR}_{\mathrm{manual}} = 1,500.0\ \mathrm{s}$ 提速 **$389.61\times$**。

下面给出集合通信入队偏斜矩阵（Arrival Skew Matrix）慢节点定位器与十万卡混沌演练 SLA 审计器的生产级参考实现：

```python
from typing import Dict, List


class ChaosGameDayAndArrivalSkewAuditor:
    """集合通信入队偏斜慢节点定位器与十万卡 Chaos GameDay 自动自愈审计器。"""

    @staticmethod
    def diagnose_arrival_skew(enqueue_ms: List[float], net_tx_ms: float) -> Dict[str, float]:
        sorted_t = sorted(enqueue_ms)
        median_t = sorted_t[len(sorted_t) // 2]
        skews = [round(t - median_t, 4) for t in enqueue_ms]
        max_skew = max(skews)
        straggler_rank = skews.index(max_skew)
        completion_ms = max(enqueue_ms) + net_tx_ms
        healthy_duration_ms = completion_ms - min(enqueue_ms)
        wait_share_pct = round(100.0 * max_skew / healthy_duration_ms, 2)
        return {
            "straggler_rank": float(straggler_rank),
            "max_skew_ms": round(max_skew, 4),
            "observed_allreduce_ms": round(healthy_duration_ms, 4),
            "straggler_wait_share_pct": wait_share_pct,
        }

    @staticmethod
    def audit_chaos_gameday(campaigns: List[Dict[str, float]], manual_mttr_s: float = 1500.0) -> Dict[str, float]:
        total_injected = sum(int(c["injected"]) for c in campaigns)
        total_detected = sum(int(c["detected"]) for c in campaigns)
        mttd_s = sum(c["detected"] * c["mttd_s"] for c in campaigns) / max(1, total_detected)
        mttr_auto_s = sum(c["detected"] * c["mttr_s"] for c in campaigns) / max(1, total_detected)
        return {
            "total_injected": float(total_injected),
            "detection_recall_pct": round(100.0 * total_detected / max(1, total_injected), 2),
            "mttd_sec": round(mttd_s, 2),
            "mttr_auto_sec": round(mttr_auto_s, 2),
            "mttr_speedup_vs_manual": round(manual_mttr_s / mttr_auto_s, 2),
        }
```

#### 无中断弹性重配置（Parachute / Oobleck 预实例化流水线模板秒级容错降级）与面向推理系统 Goodput 的动态负载压测生成科学（Poisson 突发请求与长尾长度分布建模）

在超大规模集群工程与基准评测科学中，还有两项直接决定生产 SLA 真实底线的核心技术：**（1）当热备节点池（Hot-Spare Pool）因突发机架级断电或交换机故障暂时耗尽时，如何在不销毁重创全局 `NCCL` 通信域、不重载远端检查点的前提下，通过异构流水线模板实现秒级降级续训（Elastic Reconfiguration without Restart）**[^ch20-oobleck][^ch20-parachute][^ch20-unicron]；**（2）在评测大模型在线推理服务时，如何戳穿“均匀定速发包（Uniform Pacing）+ 固定长度 Prompt”制造的虚假吞吐繁荣，建立基于泊松突发到达（Poisson Bursty Arrival）与长尾序列分布的 SLO Goodput（达标有效吞吐）评测规范**[^ch20-mlperf][^ch20-splitwise][^ch20-distserve]。

![图 20-19：Oobleck / Parachute 预实例化异构流水线模板无重启容错重配置（1.45 s 恢复 vs. 300 s 全局重启，算例 20-7）](ch20/ch20-oobleck-pre-instantiated-pipeline-template-failover.svg)

*图 20-19　Oobleck / Parachute 预实例化异构流水线模板无重启容错重配置（1.45 s 恢复 vs. 300 s 全局重启，算例 20-7）。*

如图 20-19 所示，传统弹性训练框架（如原生 `torchrun` Elastic 或 `Varuna`）在遭遇节点损失且无热备顶替时，必须销毁所有幸存节点上的进程、重新执行 TCPStore 握手与 `ncclCommInitRank` 全网建链，并从磁盘检查点重新切分加载模型权重（单次重构耗时高达 $240\text{–}300\ \mathrm{s}$）[^ch20-varuna][^ch20-bamboo][^ch20-megascale]。而以 **Oobleck（SOSP '23）与 Parachute** 为代表的无重启弹性训练架构通过**编译期预实例化多套异构流水线模板（Pre-Instantiated Heterogeneous Pipeline Templates）**彻底消除了重初始化开销[^ch20-oobleck][^ch20-parachute]：
1. **多阶流水线模板与通信域预建立**：在作业启动阶段，运行时同时在显存/主机内存中注册标准 $S_{\mathrm{std}} = 8$ 阶段流水线模板 $\mathcal{T}_8$（每阶段承载 $8$ 层 Transformer）与紧凑 $S_{\mathrm{deg}} = 7$ 阶段容错模板 $\mathcal{T}_7$（通过开启细粒度激活重计算将 $64$ 层模型重映射至 $7$ 台节点，每节点承载 $8\text{–}10$ 层），并预先完成两组模板对应的点对点与集合通信域句柄初始化；
2. **局部层分片热拼装与异构微批配额调度**：当 $32$ 条数据并行流水线中有 $F = 2$ 条各损失 1 台节点时，其余 $30$ 条健康流水线完全不受影响（继续运行 8 阶段模板 $\mathcal{T}_8$）；仅受损的 2 条流水线在 **$1.45\ \mathrm{s}$** 内激活 7 阶段模板 $\mathcal{T}_7$，利用训练跨流水线副本已持有的冗余模型分片通过 RDMA/NVLink 补齐缺失层状态，并由全局调度器动态调整各流水线的微批配额（或如 `Parachute` 直接将受损流水线的微批平滑分摊至健康流水线），实现零全局重启、高达 **$93.75\%\text{–}98.75\%$** 的稳态算力保持率。

![图 20-20：推理系统在作弊式均匀到达（CV=0）与真实泊松突发长尾流量（CV≥1）下的排队膨胀与 SLO Goodput 崩塌曲线](ch20/ch20-goodput-sla-under-bursty-poisson-traffic.svg)

*图 20-20　推理系统在作弊式均匀到达（CV=0）与真实泊松突发长尾流量（CV≥1）下的排队膨胀与 SLO Goodput 崩塌曲线。*

在推理系统基准评测侧，20.4.1 节剖析了算子级跑分失真，而在端到端服务级（Serving Benchmark）评测中，最具迷惑性的反模式是**“用恒定步调发生器（Constant-Rate Closed-Loop / Uniform Pacing）评测在线推理吞吐”**（图 20-20）[^ch20-sarathi][^ch20-kingman][^ch20-sharegpt]：
- **排队论 Kingman 公式揭示的突发性黑洞**：设推理引擎的平均单请求服务耗时为 $T_s$，系统平均利用率为 $\rho = \lambda T_s \in (0, 1)$，请求到达间隔的变异系数（Coefficient of Variation，即标准差除以均值）为 $C_a$，服务时间（由输入/输出序列长度分布决定）的变异系数为 $C_s$。根据排队论经典的 **Kingman G/G/1 近似公式**[^ch20-kingman]，请求在调度队列中的平均排队等待时间 $W_q$ 为：
  $$W_q \approx \frac{\rho}{1 - \rho} \cdot \frac{C_a^2 + C_s^2}{2} \cdot T_s$$
- **为什么均匀发包（$C_a = 0$）+ 固定长度（$C_s = 0$）是彻头彻尾的跑分作弊**：当评测脚本每隔精确的 $\Delta t = 1/\lambda$ 毫秒均匀发送一个长度完全相同的请求时，$C_a^2 + C_s^2 = 0$，即便把 GPU 利用率推高到 $\rho = 0.85\text{–}0.95$，队列等待时间依然恒为 **$W_q = 0$**，测得的首词元延迟（TTFT）永远等于纯算子执行时间 $T_{\mathrm{prefill}}$。然而一旦上线面对真实人类用户或智能体并发（到达过程服从泊松过程 $C_a = 1.0$，且 `ShareGPT` 长尾长度分布使 $C_s \ge 1.2$），因子 $\frac{C_a^2 + C_s^2}{2} = 1.22$ 瞬间激活 $\frac{\rho}{1-\rho}$ 的非线性爆炸，导致 $\rho = 0.85$ 下的排队延迟飙升十余倍，过半请求违背 TTFT SLO 红线，使真实**达标有效吞吐（Goodput，即同时满足 $\mathrm{TTFT} \le \mathrm{SLO}_1$ 且 $\mathrm{TPOT} \le \mathrm{SLO}_2$ 的每秒完成请求数）**发生断崖式崩塌。

> **算例 20-7（Oobleck 预实例化异构流水线模板 1.45 s 无重启重配置与泊松突发流量下的推理 Goodput 崩塌核算，锁定 `experiments/ch20/20-03/run.py`）：**
>
> **问题设定**：
> 1. **Oobleck 预实例化异构流水线无重启容错账本**：某大模型训练任务运行在 $P = 32$ 条数据并行流水线上，每条标准流水线含 $S = 8$ 台节点（共 $256$ 台节点，$2,048$ 张 GPU）。当热备池暂时耗尽且突发 $F = 2$ 台节点掉线（分别位于流水线 #3 与 #19）时，对比以下两种恢复路径：（a）传统 `torchrun` 全局销毁重启并从磁盘重载检查点（耗时 $T_{\mathrm{restart}} = 300.00\ \mathrm{s}$）；（b）`Oobleck` 预实例化模板重配置——包括故障确认与路由切换 $t_{\mathrm{detect}} = 0.25\ \mathrm{s}$、跨副本缺失层分片 RDMA 补齐 $t_{\mathrm{shard}} = 0.85\ \mathrm{s}$、异构微批配额同步对齐 $t_{\mathrm{sync}} = 0.35\ \mathrm{s}$（其中 2 条降级为 7 阶段模板的流水线因开启局部重计算单步吞吐为原管线的 $80.0\%$，其余 30 条 8 阶段健康流水线吞吐为 $100.0\%$）。求 `Oobleck` 端到端重配置延迟 $T_{\mathrm{oobleck}}$、相对传统重启的恢复加速比，以及在无备机顶替期间集群保留的有效吞吐百分比（同时给出 `Parachute` 直接摘除 2 条受损流水线时的保留吞吐百分比）。
> 2. **推理压测均匀到达（$C_a=0, C_s=0$）vs. 真实泊松长尾流量（$C_a=1.0, C_s=1.2$）的 Goodput 核算**：设某推理实例的平均单请求服务时间为 $T_s = 120.0\ \mathrm{ms}$（饱和吞吐率 $\mu = \frac{1000}{120.0} \approx 8.3333\ \mathrm{req/s}$），纯 Prefill 算子执行耗时 $T_{\mathrm{prefill}} = 80.0\ \mathrm{ms}$，首词元延迟 SLO 上限为 $\mathrm{SLO}_{\mathrm{TTFT}} = 500.0\ \mathrm{ms}$（最大容许排队时间 $W_{\max} = 420.0\ \mathrm{ms}$）。当系统在利用率 $\rho = 0.85$（到达率 $\lambda = 0.85 \mu \approx 7.0833\ \mathrm{req/s}$）下运行时，分别计算均匀定速固定长度压测与真实泊松长尾压测下的平均排队时间 $W_q$、平均 $\mathrm{TTFT}$、按指数尾部模型 $P(W_q \le W_{\max}) = 1 - \rho \exp(-W_{\max}/W_q)$ 估算的 SLO 达标率与有效吞吐 Goodput（$\mathrm{req/s}$），并求出在真实突发流量下若要求平均 $\mathrm{TTFT} \le 300.0\ \mathrm{ms}$（即 $W_q \le 220.0\ \mathrm{ms}$）所允许的最大安全系统利用率 $\rho_{\max}$。
>
> **逐项推导**：
>
> 1. **Oobleck 预实例化流水线模板重配置延迟与有效算力保持率**：
>    - **端到端无重启重配置耗时与加速比**：
>      $$T_{\mathrm{oobleck}} = t_{\mathrm{detect}} + t_{\mathrm{shard}} + t_{\mathrm{sync}} = 0.25 + 0.85 + 0.35 = \mathbf{1.45\ \mathrm{s}}$$
>      相比传统全局销毁重启（$T_{\mathrm{restart}} = 300.00\ \mathrm{s}$），恢复速度提升 **$\frac{300.00}{1.45} \approx \mathbf{206.90\times}$**。
>    - **无备机顶替期间的稳态吞吐保持率**：
>      - 在 `Oobleck` 异构模板混合运行（30 条 $\mathcal{T}_8$ + 2 条 $\mathcal{T}_7$）下，集群保留的有效训练吞吐比例为：
>        $$\eta_{\mathrm{oobleck}} = \frac{30 \times 1.00 + 2 \times 0.80}{32} = \frac{31.60}{32} = \mathbf{98.75\%}$$
>      - 若采用 `Parachute` 弹性降级数据并行度（摘除 2 条受损流水线，30 条健康流水线无中断继续运行），保留吞吐比例为：
>        $$\eta_{\mathrm{parachute}} = \frac{30}{32} = \mathbf{93.75\%}$$
>
> 2. **均匀定速压测 vs. 真实泊松长尾压测的排队膨胀与 Goodput 崩塌**：
>    - **作弊式均匀定速 + 固定长度压测（$C_a = 0, C_s = 0$）**：
>      $$W_{q,\mathrm{uniform}} = \mathbf{0.00\ \mathrm{ms}},\qquad \mathrm{TTFT}_{\mathrm{uniform}} = 80.00 + 0.00 = \mathbf{80.00\ \mathrm{ms}} \le 500.0\ \mathrm{ms}$$
>      表面 SLO 达标率为 **$100.00\%$**，宣称 Goodput $= \lambda = \mathbf{7.08\ \mathrm{req/s}}$。
>    - **真实泊松突发 + 长尾长度压测（$C_a = 1.0, C_s = 1.2$）**：变异因子 $\frac{C_a^2 + C_s^2}{2} = \frac{1.00 + 1.44}{2} = 1.22$，平均排队等待时间飙升至：
>      $$W_{q,\mathrm{bursty}} = \frac{0.85}{1 - 0.85} \times 1.22 \times 120.0\ \mathrm{ms} = 5.6667 \times 146.4\ \mathrm{ms} = \mathbf{829.60\ \mathrm{ms}}$$
>      平均首词元延迟暴涨至 $\mathrm{TTFT}_{\mathrm{bursty}} = 80.00 + 829.60 = \mathbf{909.60\ \mathrm{ms}}$（膨胀 **$\frac{909.60}{80.00} \approx \mathbf{11.37\times}$**）。在 $W_{\max} = 420.0\ \mathrm{ms}$ 预算下，SLO 达标率跌至：
>      $$P_{\mathrm{slo}} = 1 - 0.85 \exp\!\left(-\frac{420.0}{829.60}\right) = 1 - 0.85 \times 0.60274 \approx \mathbf{48.77\%}$$
>      真实有效吞吐骤降至 $\mathrm{Goodput}_{\mathrm{bursty}} = 7.0833 \times 0.48767 \approx \mathbf{3.45\ \mathrm{req/s}}$（比均匀压测宣称的 $7.08\ \mathrm{req/s}$ 缩水 **$51.27\%$**）。
>    - **满足平均 $\mathrm{TTFT} \le 300.0\ \mathrm{ms}$（即 $W_q \le 220.0\ \mathrm{ms}$）的最大安全利用率 $\rho_{\max}$**：
>      $$\frac{\rho_{\max}}{1 - \rho_{\max}} \le \frac{220.0}{1.22 \times 120.0} = \frac{220.0}{146.4} \approx 1.50273 \implies \rho_{\max} = \frac{1.50273}{2.50273} \approx \mathbf{60.04\%}$$
>      这从数学上解释了为何在线生产推理集群必须预留约 $35\%\text{–}40\%$ 的突发排队缓冲水位（Headroom），严禁按均匀压测的 $85\%$ 利用率满载排布。

表 20-9 总结了无重启弹性流水线重配置与面向 SLO Goodput 的真实动态流量评测规范：

| 工程与评测维度 | 传统脆弱基线 / 失真压测手法 | 现代高可靠弹性与反作弊评测规范（Oobleck + Bursty Goodput） | 定量改善或失真揭露幅度 |
| :--- | :--- | :--- | :---: |
| **无备机时节点掉线恢复** | `torchrun` 全局销毁重建通信域并读盘（$300.00\ \mathrm{s}$） | **Oobleck 预实例化异构流水线模板切换（$1.45\ \mathrm{s}$）** | 恢复提速 **$206.90\times$** |
| **掉线 2/256 节点期间吞吐** | 全部 $2,048$ 卡停机阻塞等待（有效吞吐 $0\%$） | **30 条 $\mathcal{T}_8$ + 2 条 $\mathcal{T}_7$ 异构续训（保留 $98.75\%$ 吞吐）** | 挽回 **$98.75\%$** 算力 |
| **推理压测请求到达分布** | 作弊式均匀定速发包（$C_a=0, C_s=0$） | **泊松突发到达（$C_a=1.0$）+ 长尾长度分布（$C_s=1.2$）** | 还原真实线上排队尾延迟 |
| **$\rho=0.85$ 下平均 TTFT** | 虚假测量值 **$80.00\ \mathrm{ms}$**（排队 $0\ \mathrm{ms}$） | 真实平均耗时 **$909.60\ \mathrm{ms}$**（排队高达 $829.60\ \mathrm{ms}$） | 揭露 **$11.37\times$** 延迟低估 |
| **$\rho=0.85$ 下 SLO Goodput** | 虚报 $100\%$ 达标（**$7.08\ \mathrm{req/s}$**） | 仅 **$48.77\%$** 达标（真实 Goodput **$3.45\ \mathrm{req/s}$**） | 揭露 **$51.27\%$** 有效吞吐虚标 |

*表 20-9　无重启预实例化流水线容错重配置（Oobleck / Parachute）与推理服务突发流量 Goodput 评测对比。*

下面的参考实现代码将预实例化异构流水线模板容错调度器与基于 Kingman 排队论的突发流量 Goodput 审计器封装为生产级校验引擎（对应配套实验 `experiments/ch20/20-03`）：

```python
import math
from dataclasses import dataclass

@dataclass
class OobleckElasticPipelineAndGoodputBenchEngine:
    """
    复现算例 20-7 (对应 experiments/ch20/20-03)：
    1. Oobleck / Parachute 预实例化异构流水线模板无重启容错重配置 (1.45 s vs. 300.0 s)
    2. 均匀定速压测 (CV=0) vs. 泊松突发长尾流量 (Ca=1.0, Cs=1.2) 的 Kingman 排队膨胀与 Goodput 审计
     """
    num_pipelines: int = 32
    stages_per_pipeline: int = 8
    failed_nodes: int = 2
    degraded_template_eff: float = 0.80

    def audit_oobleck_failover(
        self,
        t_detect_s: float = 0.25,
        t_shard_s: float = 0.85,
        t_sync_s: float = 0.35,
        t_full_restart_s: float = 300.0,
    ) -> dict[str, float]:
        t_oobleck_s = round(t_detect_s + t_shard_s + t_sync_s, 2)
        healthy_pipes = self.num_pipelines - self.failed_nodes
        oobleck_retain_pct = round(
            100.0 * (healthy_pipes * 1.0 + self.failed_nodes * self.degraded_template_eff) / self.num_pipelines, 2
        )
        parachute_retain_pct = round(100.0 * healthy_pipes / self.num_pipelines, 2)
        return {
            "oobleck_reconfig_sec": t_oobleck_s,
            "full_restart_sec": round(t_full_restart_s, 2),
            "reconfig_speedup_x": round(t_full_restart_s / t_oobleck_s, 2),
            "oobleck_retained_throughput_pct": oobleck_retain_pct,
            "parachute_retained_throughput_pct": parachute_retain_pct,
        }

    @staticmethod
    def audit_bursty_goodput_vs_uniform(
        service_ms: float = 120.0,
        prefill_ms: float = 80.0,
        slo_ttft_ms: float = 500.0,
        rho: float = 0.85,
        ca_bursty: float = 1.0,
        cs_bursty: float = 1.2,
        target_mean_ttft_ms: float = 300.0,
    ) -> dict[str, float]:
        mu_req_s = 1000.0 / service_ms
        arrival_req_s = rho * mu_req_s
        w_max_ms = slo_ttft_ms - prefill_ms

        ttft_uniform_ms = prefill_ms
        goodput_uniform = round(arrival_req_s, 2)

        var_factor = 0.5 * (ca_bursty ** 2 + cs_bursty ** 2)
        wq_bursty_ms = (rho / (1.0 - rho)) * var_factor * service_ms
        ttft_bursty_ms = prefill_ms + wq_bursty_ms
        slo_attain_ratio = 1.0 - rho * math.exp(-w_max_ms / wq_bursty_ms)
        goodput_bursty = round(arrival_req_s * slo_attain_ratio, 2)

        target_wq_ms = target_mean_ttft_ms - prefill_ms
        odds = target_wq_ms / (var_factor * service_ms)
        rho_max_pct = round(100.0 * (odds / (1.0 + odds)), 2)

        return {
            "ttft_uniform_ms": round(ttft_uniform_ms, 2),
            "goodput_uniform_req_s": goodput_uniform,
            "wq_bursty_ms": round(wq_bursty_ms, 2),
            "ttft_bursty_ms": round(ttft_bursty_ms, 2),
            "ttft_inflation_x": round(ttft_bursty_ms / ttft_uniform_ms, 2),
            "slo_attainment_bursty_pct": round(100.0 * slo_attain_ratio, 2),
            "goodput_bursty_req_s": goodput_bursty,
            "goodput_drop_pct": round(100.0 * (goodput_uniform - goodput_bursty) / goodput_uniform, 2),
            "max_safe_utilization_pct": rho_max_pct,
        }


if __name__ == "__main__":
    eng = OobleckElasticPipelineAndGoodputBenchEngine()
    ob = eng.audit_oobleck_failover()
    gp = eng.audit_bursty_goodput_vs_uniform()
    assert ob["oobleck_reconfig_sec"] == 1.45 and ob["reconfig_speedup_x"] == 206.90
    assert ob["oobleck_retained_throughput_pct"] == 98.75 and ob["parachute_retained_throughput_pct"] == 93.75
    assert gp["wq_bursty_ms"] == 829.60 and gp["ttft_bursty_ms"] == 909.60
    assert gp["slo_attainment_bursty_pct"] == 48.77 and gp["goodput_bursty_req_s"] == 3.45
    assert gp["max_safe_utilization_pct"] == 60.04
```

---

## 常见误区

1. **误区一：“每张 H100 GPU 的平均无故障时间长达 5 年以上，而且显存开启了硬件 ECC，所以跑几万张卡的训练只要每隔几小时存一次远端磁盘检查点就高枕无忧了。”**
   **纠正**：如算例 20-1 所示，单卡 5.78 年的 MTBF 除以 $100,000$ 张卡后，集群级 MTBF 会坍缩至 **$30.41$ 分钟**；若仍采用耗时 $180\ \mathrm{s}$ 的远端落盘检查点与耗时 $300\ \mathrm{s}$ 的 `torchrun` 全局销毁重启，全集群有效训练时间比（ETTR）将跌至 **$63.22\%$**（白白蒸发 $36,780$ 张 GPU 的算力）。更致命的是，HBM ECC 无法保护 Tensor Core 算术通路内部的瞬态电压/老化比特翻转（SDC），一旦算错的值写回显存就会获得合法 ECC 码并在一次 AllReduce 中毒化全网权重。必须结合内存异步快照、秒级热迁移（`TrainMover`）与在线 SDC 拦截（ABFT + `Safeguarding` + `OpGuard`）。
2. **误区二：“为了应付十万卡集群每小时坏 2 台机器、每台修 12 小时的故障率，备机池只要按平均值 $2 \times 12 = 24$ 台准备就够了。”**
   **纠正**：如算例 20-2 所示，在修节点数服从均值 $L = 24.0$ 的泊松分布；若只配 $S = 24$ 台热备节点，有高达 **$44.60\%$** 的时间在修坏机器数量会超过 24 台，导致主训练集群频繁因备机枯竭而停机或降级。按泊松累积分布 $P(K \le S) \ge 99.9\%$ 规划，需配置 $S_{\min} = \mathbf{40}$ 台热备节点（仅占万节点集群不到 $0.4\%$）。
3. **误区三：“训练变慢时，打开 Profiler 看到所有卡上 `ncclKernel_AllReduce` 都占了 80% 的时间，这铁定是数据中心网络拥塞或带宽不够。”**
   **纠正**：如算例 20-4（锁定 `case-studies/collective-paths-and-diagnosis.md`）所示，在紧耦合同步集合通信中，只要有 1 张慢卡（Straggler）因降频或 PCIe 降速迟到了 $2.0\ \mathrm{ms}$，其余 $99,999$ 张准时到达的健康卡都会在 `ncclKernel_AllReduce` 内自旋空等这 $2.0\ \mathrm{ms}$。此时盲目将网络带宽翻倍只能把 $0.4\ \mathrm{ms}$ 的真实传输压到 $0.2\ \mathrm{ms}$（仅省 $8.3\%$），而通过 `NCCL Flight Recorder` 入队时间戳偏斜矩阵抓出最晚到达的慢卡，才能立竿见影省掉全部 $2.0\ \mathrm{ms}$（提速 $6\times$）。
4. **误区四：“评测新算子时，在 10 个不同形状上算出各自的加速比再求算术平均，得到平均加速 5.25 倍，上线后肯定比平均加速 2.0 倍的算子快。”**
   **纠正**：如算例 20-5（锁定 `case-studies/optimization-evaluation-and-deployment.md`）所示，候选算子 A 在短形状上快 $10\times$（$100 \to 10\ \mu\mathrm{s}$）、在长形状上慢 $0.5\times$（$100 \to 200\ \mu\mathrm{s}$），算术平均加速比高达 **$5.25\times$**，但在等频流量下总耗时为 **$210\ \mu\mathrm{s}$**，不仅比候选 B（$50+50=100\ \mu\mathrm{s}$）慢一倍以上，甚至比基线（$200\ \mu\mathrm{s}$）还慢。算子评测严禁对加速比取算术平均，必须按真实生产形状分布汇报加权总耗时。

---

## 练习与实验

> **练习 20-1（十万卡集群 MTBF 外推与 Young-Daly 最优检查点周期推演）：**
> 某新一代 $65,536$ 卡（64K 卡）超大规模训练集群，实测单卡等效平均无故障时间为 $\mathrm{MTBF}_{\mathrm{gpu}} = 45,000\ \mathrm{h}$。
> （1）求该 $65,536$ 卡集群的平均无故障间隔 $\mathrm{MTBF}_{64\mathrm{K}}$（以分钟和秒为单位）；
> （2）若采用传统并行文件系统落盘检查点（单次阻塞写盘耗时 $C_{\mathrm{ckpt}} = 150\ \mathrm{s}$，故障后全局重启与重载恢复耗时 $T_{\mathrm{rec}} = 240\ \mathrm{s}$），求 Young-Daly 最优检查点间隔 $\tau_{\mathrm{opt}} = \sqrt{2 C_{\mathrm{ckpt}} M}$ 及对应的有效训练时间比（ETTR）；
> （3）若升级为 `ByteCheckpoint`/`Checkmate` 内存异步检查点（$C_{\mathrm{ckpt}} = 4.0\ \mathrm{s}$）配合 `TrainMover` 无重启热接管（$T_{\mathrm{rec}} = 6.0\ \mathrm{s}$），求新的 $\tau_{\mathrm{opt}}$ 与 ETTR，并计算相当于为该集群净挽回了多少张 GPU 的有效算力。

> **练习 20-2（热备节点池泊松维修队列与分位数容量规划）：**
> 某万卡训练中心共有 $4,096$ 台 8 卡节点，平均节点故障到达率为 $\lambda = 1.5\ \text{台/h}$。
> （1）当线下硬件维修团队的平均修复归队周期为 $\mathrm{MTTR}_{\mathrm{repair}} = 10.0\ \mathrm{h}$ 时，求稳态在修节点均值 $L$，并利用正态近似 $S_{\min} \approx \lceil L + z_{0.999}\sqrt{L} \rceil$（其中 $z_{0.999} = 3.09$）或精确泊松分布估算满足 $99.9\%$ 不降级概率所需的最小热备节点数；
> （2）若通过备件前置与自动化四阶烤机流水线将 $\mathrm{MTTR}_{\mathrm{repair}}$ 从 $10.0\ \mathrm{h}$ 缩短至 $4.0\ \mathrm{h}$，所需的最小热备节点数可减少多少台？

> **练习 20-3（BFloat16 静默数据损坏 SDC 比特翻转的数值放大效应）：**
> IEEE 754 / `BFloat16` 格式由 1 位符号位（Bit 15）、8 位指数位（Bits 14..7，偏移量 Bias $= 127$）和 7 位尾数位（Bits 6..0）组成。设某正常梯度元素的值为 $g = 0.125$（二进制表示为符号 `0`，指数 $124 = \text{0b01111100}$，尾数 `0000000`）。
> （1）若宇宙射线或电压跌落导致指数第 5 位（Bit 12，对应指数权重 $+32$）由 `0` 静默翻转为 `1`，翻转后的浮点数 $\tilde{g}$ 变成了多少？在 $16,384$ 卡数据并行 AllReduce 求平均后，单卡这 1 个比特的翻转会给全集群平均梯度带来多大的绝对偏差？
> （2）为什么这种幅度（既非 `NaN` 亦非 `Inf`）的翻转能够轻易穿透传统的 `torch.isfinite()` 检查？

> **练习 20-4（ABFT 矩阵乘代数不变量校验开销与双错误检测边界）：**
> 在形状为 $M = 8,192,\; N = 8,192,\; K = 8,192$ 的 FP8/FP32 矩阵乘法 $\mathbf{C} = \mathbf{A}\mathbf{B}$ 中：
> （1）核算主矩阵乘的 FLOPs 与 ABFT 行列双向校验和（$6N^2$ 核心乘加项）的相对算力开销百分比；
> （2）若同一次 GEMM 中恰好有同一行的两个元素 $(i, j_1)$ 与 $(i, j_2)$ 分别发生了 $+32.0$ 与 $-32.0$ 的反号错误，仅靠行和校验 $\mathbf{C}\mathbf{1}_N = \mathbf{A}(\mathbf{B}\mathbf{1}_N)$ 能否发现？为什么同时检查列和校验 $\mathbf{1}_M^T\mathbf{C} = (\mathbf{1}_M^T\mathbf{A})\mathbf{B}$ 或采用加权校验向量 $\mathbf{w} = [1, 2, \dots, N]^T$ 可以 $100\%$ 捕获此类抵消错误？

> **练习 20-5（`OpGuard` 良性非确定性消除与 50 ms 微批次重跑收益核算）：**
> 某十万卡集群每 $100$ 小时发生 $1$ 次硬件 SDC 事件。
> （1）若无在线拦截，每次 SDC 平均在污染全局权重 $1.5$ 小时后才通过验证集困惑度崩盘被发现，随后需回滚至 $2.0$ 小时前的干净检查点并耗费 $0.5$ 小时排查坏卡，求每百小时因 SDC 损失的集群墙钟时间与 GPU 小时数；
> （2）若部署 `Safeguarding LLM Training`（AllReduce 前拦截）+ `OpGuard`（$50\ \mathrm{ms}$ 位级对齐微批重算定位）+ `TrainMover`（$5.0\ \mathrm{s}$ 热备切换），求每次 SDC 事件的端到端处置耗时与每百小时节省的 GPU 小时数。

> **练习 20-6（锁定 `case-studies/collective-paths-and-diagnosis.md`：集合通信慢节点偏斜与多轨失衡复算）：**
> 在一个 8-Rank AllReduce 通信组中，前 7 个 Rank 的就绪到达时刻均为 $t = 10.0\ \mathrm{ms}$，而第 7 号 Rank 因 SSD 日志刷盘抢占 PCIe 在 $t = 14.5\ \mathrm{ms}$ 才到达；全员到齐后的网络 Ring AllReduce 传输耗时为 $0.5\ \mathrm{ms}$。
> （1）求前 7 个 Rank 在 Profiler 中看到的 AllReduce 算子持续时间，以及其中真实的“网络传输时间占比”与“慢节点等待时间占比”；
> （2）若某节点使用 4 张 400G 网卡条带化传输同一集合通信负载，其中 3 张网卡耗时 `0.8 ms`，第 4 张网卡因光模块 FEC 重传耗时 `3.2 ms`，求 4 张网卡的平均耗时与实际决定训练步进的木桶短板耗时。

> **练习 20-7（锁定 `case-studies/optimization-evaluation-and-deployment.md`：加速比平均悖论与翻转阈值推导）：**
> 基线算子在形状 $S_1$ 与 $S_2$ 上的耗时分别为 $T_{\mathrm{base}} = [80\ \mu\mathrm{s},\; 120\ \mu\mathrm{s}]$。候选算子 X 的耗时为 $T_X = [10\ \mu\mathrm{s},\; 240\ \mu\mathrm{s}]$，候选算子 Y 的耗时为 $T_Y = [40\ \mu\mathrm{s},\; 60\ \mu\mathrm{s}]$。
> （1）分别计算候选 X 与候选 Y 的算术平均加速比；
> （2）求在等频流量（$p = 0.5$）下两者的总耗时与真实端到端加速比，并求出候选 X 的期望耗时小于候选 Y 所需的短形状占比阈值 $p^*$。

> **练习 20-8（锁定 `case-studies/optimization-evaluation-and-deployment.md`：线性热漂移下的 `AABB` 偏差与 `ABBA` 抵消证明）：**
> 设待对比的两个算子 A 与 B 的真实稳态耗时分别为 $\mu_A$ 与 $\mu_B$。在连续 4 个测量窗口 $k = 0, 1, 2, 3$ 中，系统背景温度引发的附加线性延迟漂移为 $d_k = k \cdot \delta$（其中 $\delta = 12.0\ \mu\mathrm{s}$）。
> （1）证明按 `AABB` 顺序测量时，样本均值差 $\bar{T}_A^{\mathrm{AABB}} - \bar{T}_B^{\mathrm{AABB}}$ 会产生恒等于 $-2\delta = -24.0\ \mu\mathrm{s}$ 的系统性偏差；
> （2）证明按 `ABBA` 顺序测量时，样本均值差 $\bar{T}_A^{\mathrm{ABBA}} - \bar{T}_B^{\mathrm{ABBA}} \equiv \mu_A - \mu_B$，线性漂移项 $\delta$ 被完全消除。

> **练习 20-9（全书双轨贯穿综合题：从单卡算术强度到十万卡容错的端到端账本）：**
> 结合表 20-6 中的双轨模型参数：
> （1）在轨道 A（Qwen3-8B，BF16 权重 $16.4\ \mathrm{GB}$，KV 缓存 $144\ \mathrm{KB/tok}$）中，若机载端将其 INT4 量化（权重 $4.6\ \mathrm{GB}$）部署于单颗 Jetson Thor 的 75% MIG 分区（有效带宽 $600\ \mathrm{GB/s}$），求 $B=1$ 自回归生成一个词元的理论访存下限延迟，并说明为何必须配合 $150\ \mathrm{M}$ System 1 流匹配动作头才能满足 $100\ \mathrm{Hz}$（$10\ \mathrm{ms}$）全身控制；
> （2）在轨道 B（DeepSeek V4.1 671B，激活 $37\ \mathrm{B}$）的万卡预训练与 RLVR 后训练中，总结 MLA、DeepEP、DualPipe、`RollPacker` 与 `OpGuard`/`TrainMover` 五大系统技术分别攻克了哪一堵物理墙。

> **实验 20-10（可独立运行实验：十万卡 Young-Daly 周期、热备池排队模型、ABFT SDC 注入修复与评测反例复算器）：**
> 运行并阅读本章配套实验 `experiments/ch20/20-01/run.py` 与 `experiments/ch20/20-01/verify.py`：
> （1）复算算例 20-1（Meta $16\mathrm{K}$ 卡 $419$ 次故障外推至十万卡 $\mathrm{MTBF} = 30.41\ \mathrm{min}$，以及传统落盘检查点 $\mathrm{ETTR} = 63.22\%$ vs. 内存快照+热迁移 $\mathrm{ETTR} = 92.34\%$）与算例 20-2（泊松热备池 $S_{\min} = 40$）；
> （2）复算算例 20-3（$4,096$ 阶矩阵乘 ABFT $0.0732\%$ 开销与 $(2, 3)$ 坐标 $+64.0$ 翻转原地修复）以及算例 20-4 / 20-5（慢节点 $2.0\ \mathrm{ms}$ 偏斜 vs. $0.2\ \mathrm{ms}$ 带宽收益、$5.25\times$ 加速比悖论 $p^* = 15/19$ 与 `ABBA` 漂移抵消）。

> **练习 20-11（集合通信 Arrival Skew Matrix 慢节点根因反演与光模块 FEC 重传灰故障定位）：**
> 在某 $16$-Rank 跨机架 AllReduce 通信组中，`NCCL Flight Recorder` 记录到单步集合通信的入队时刻：除 Rank 11 在 $t_{11} = 28.5\ \mathrm{ms}$ 入队外，其余 $15$ 个健康 Rank 均在 $t = 22.5\ \mathrm{ms}$ 入队；全员到齐后 Ring AllReduce 的网络传输耗时为 $1.5\ \mathrm{ms}$。
> （1）求其余 $15$ 个健康 Rank 在 Profiler 中观测到的 AllReduce 算子持续时间，以及其中等待 Rank 11 的慢节点空转占比；
> （2）若将集群网络带宽直接翻倍（使真实网络传输耗时从 $1.5\ \mathrm{ms}$ 降至 $0.75\ \mathrm{ms}$），而未隔离慢节点 Rank 11，求健康 Rank 观测到的 AllReduce 持续时间仅缩短了百分之几？若改用 `TrainMover` 在 $0.68\ \mathrm{s}$ 内将 Rank 11 热迁移至健康热备卡（消除 $6.0\ \mathrm{ms}$ 入队偏斜），单步 AllReduce 耗时可获得多少倍的真实加速？

> **实验 20-12（运行 `experiments/ch20/20-02` 复算 Arrival Skew 慢节点定位、TrainMover 亚秒级热迁移与四大跑分失真修正）：**
> 运行并阅读本章第二个配套实验 `experiments/ch20/20-02/run.py` 与 `experiments/ch20/20-02/verify.py`：
> （1）复算算例 20-6 中的 8-Rank 入队偏斜矩阵诊断（精确锁定 `straggler_rank = 5`、偏斜 $2.0\ \mathrm{ms}$、慢节点等待占比 $83.33\%$）以及 `TrainMover` 亚秒级热迁移（$0.68\ \mathrm{s}$ vs. 传统重启 $300.0\ \mathrm{s}$，提速 $441.18\times$）；
> （2）复算四大跑分失真修正与两阶段金丝雀门禁（$5.25\times$ 算术平均加速比实则 $0.9524\times$ 倒退、`ABBA` 消除 $-16.0\ \mu\mathrm{s}$ 漂移偏置、金丝雀捕获 $+4.8\%$ P99 退化自动回滚），以及十万卡 $200$ 次 Chaos GameDay 故障注入演练（$99.50\%$ 召回率、$\mathrm{MTTD} = 2.70\ \mathrm{s}$、$\mathrm{MTTR}_{\mathrm{auto}} = 3.85\ \mathrm{s}$，比人工恢复快 $389.61\times$）。

> **练习 20-13（Oobleck 预实例化异构流水线模板在多节点同时掉线下的吞吐保持率推演）：**
> 承接算例 20-7，某 $64$ 条数据并行流水线（每条 $8$ 阶段，共 $512$ 台节点）的训练任务在热备节点耗尽期间突发 $F = 4$ 台节点掉线（分别位于 4 条不同的数据并行流水线中）。
> （1）若采用 `Parachute` 策略直接摘除这 4 条受损流水线、保持其余 $60$ 条流水线无中断运行，集群保留的有效训练吞吐百分比是多少？
> （2）若采用 `Oobleck` 预实例化模板重配置，将这 4 条受损流水线在 $1.45\ \mathrm{s}$ 内切换为 7 阶段紧凑模板 $\mathcal{T}_7$（单条降级管线因开启局部激活重计算，有效吞吐为原 8 阶段管线的 $82.0\%$），求集群保留的有效训练吞吐百分比，以及相比 `Parachute` 策略每天（$24\ \mathrm{h}$）净多挽回多少个等效健康节点小时（Node-Hours）？

> **实验 20-14（推理服务泊松突发长尾流量下的 Kingman 排队膨胀与 Goodput 校验实验 `experiments/ch20/20-03`）：**
> 承接算例 20-7，设某推理引擎的平均单请求服务耗时为 $T_s = 100.0\ \mathrm{ms}$（饱和吞吐率 $\mu = 10.0\ \mathrm{req/s}$），纯 Prefill 耗时 $T_{\mathrm{prefill}} = 60.0\ \mathrm{ms}$，当系统工作在利用率 $\rho = 0.80$（到达率 $\lambda = 8.0\ \mathrm{req/s}$）时：
> （1）对比均匀定速固定长度压测（$C_a = 0, C_s = 0$）与极端智能体并发突发流量（$C_a = 1.5, C_s = 1.5$）下的平均排队等待时间 $W_q$ 与平均首词元延迟 $\mathrm{TTFT}$；
> （2）运行 `python3 experiments/ch20/20-03/run.py && python3 experiments/ch20/20-03/verify.py` 校验算例 20-7、练习 20-13 与本题的全部锁定指标。

### 可运行实验（20-01、20-02 与 20-03）

- **实验目录**：`experiments/ch20/20-01`（十万卡 Young-Daly 周期、热备池排队模型与 ABFT SDC 修复器）、`experiments/ch20/20-02`（Arrival Skew 慢节点定位、`TrainMover` 热迁移、四大跑分失真修正与 Chaos GameDay 演练审计器）、`experiments/ch20/20-03`（Oobleck 预实例化流水线无重启容错重配置与泊松突发流量 Goodput 评测器）
- **实验目标**：一键复现并校验本章全部锁定算例与生产案例账本，包括十万卡 $\mathrm{MTBF} = 30.41\ \mathrm{min}$、Young-Daly 最优检查点周期与有效训练时间比（$\mathrm{ETTR}\ 63.22\% \to 92.34\%$）、泊松热备节点池（$S_{\min}=40$）、ABFT 矩阵乘行列校验和（$0.0732\%$ 开销与原地修复）（`20-01`），8-Rank 入队偏斜慢节点定位（$83.33\%$ 等待占比）、`TrainMover` $0.68\ \mathrm{s}$ 热接管（快 $441.18\times$）、四大跑分失真修正与 $200$ 次 Chaos GameDay 故障注入演练（$99.50\%$ 召回率、$\mathrm{MTTR}_{\mathrm{auto}} = 3.85\ \mathrm{s}$）（`20-02`），以及 `Oobleck` $1.45\ \mathrm{s}$ 预实例化模板重配置（快 $206.90\times$，保留 $98.75\%$ 吞吐）与 $\rho=0.85$ 泊松突发流量下 $\mathrm{TTFT}$ 从 $80.00\ \mathrm{ms}$ 膨胀至 $909.60\ \mathrm{ms}$（Goodput 缩水 $51.27\%$）（`20-03`）。
- **执行与校验命令**：在终端依次执行 `python3 experiments/ch20/20-01/run.py && python3 experiments/ch20/20-01/verify.py`、`python3 experiments/ch20/20-02/run.py && python3 experiments/ch20/20-02/verify.py` 以及 `python3 experiments/ch20/20-03/run.py && python3 experiments/ch20/20-03/verify.py`。

---

### 参考解答与详尽推导

- **20-1 解答**：
  1. $65,536$ 卡集群的 $\mathrm{MTBF}_{64\mathrm{K}} = \frac{45,000\ \mathrm{h}}{65,536} \approx 0.68665\ \mathrm{h} = \mathbf{41.20\ \mathrm{min}} = \mathbf{2,471.9\ \mathrm{s}}$。
  2. 传统方案（$C = 150\ \mathrm{s}, T_{\mathrm{rec}} = 240\ \mathrm{s}, M = 2,471.9\ \mathrm{s}$）：$\tau_{\mathrm{opt}} = \sqrt{2 \times 150 \times 2,471.9} \approx \mathbf{861.1\ \mathrm{s}}$；按算例 20-1 公式计算，$\mathrm{ETTR} = \frac{861.1}{861.1 + 150} \times \left(1 - \frac{240 + 430.55}{2,471.9}\right) \approx 0.8516 \times 0.7287 \approx \mathbf{62.06\%}$（或按标准分步期望公式约 $67.8\%$）。
  3. 现代方案（$C = 4.0\ \mathrm{s}, T_{\mathrm{rec}} = 6.0\ \mathrm{s}$）：$\tau_{\mathrm{opt}} = \sqrt{2 \times 4.0 \times 2,471.9} \approx \mathbf{140.6\ \mathrm{s}}$；对应的 $\mathrm{ETTR} = \frac{140.6}{144.6} \times \left(1 - \frac{6.0 + 70.3}{2,471.9}\right) \approx 0.9723 \times 0.9691 \approx \mathbf{94.23\%}$。净提升超过 $26\text{–}32$ 个百分点，相当于为该 $65,536$ 卡集群净挽回了 **$17,300\text{–}21,000$ 张等效 GPU**。

- **20-2 解答**：
  1. 当 $\lambda = 1.5\ \mathrm{h}^{-1}, \mathrm{MTTR} = 10.0\ \mathrm{h}$ 时，稳态在修均值 $L = 1.5 \times 10.0 = \mathbf{15.0\ \text{台}}$。查泊松累积分布表，满足 $P(K \le S) \ge 0.999$ 的最小整数为 **$S_{\min} = 28\ \text{台}$**（正态近似 $15 + 3.09\sqrt{15} = 26.97 \to 27\text{–}28$ 台）。
  2. 当 $\mathrm{MTTR}$ 缩短至 $4.0\ \mathrm{h}$ 时，稳态在修均值降至 $L' = 1.5 \times 4.0 = \mathbf{6.0\ \text{台}}$，满足 $99.9\%$ 可用率的最小热备数降至 **$S'_{\min} = 15\ \text{台}$**，直接节省了 $28 - 15 = \mathbf{13\ \text{台}}$ 8 卡服务器（$104$ 张 GPU）。

- **20-3 解答**：
  1. 正常值 $g = 0.125 = 2^{-3}$（指数位码值 $124 = 127 - 3$）。当 Bit 12（指数权重 $+32$）由 `0` 翻为 `1` 后，新指数码值变为 $124 + 32 = 156$，对应的真实指数为 $156 - 127 = 29$，故翻转后的数值变为 $\tilde{g} = 2^{29} = \mathbf{536,870,912.0}$（放大了 $2^{32} \approx 4.29 \times 10^9$ 倍）。经 $16,384$ 卡（$2^{14}$）AllReduce 平均后，全集群平均梯度偏移了 $\frac{2^{29}}{2^{14}} = 2^{15} = \mathbf{32,768.0}$，足以在一步更新内将整层权重彻底冲毁。
  2. 因为指数码值 $156 < 255$（`BFloat16` 中指数全 `1` 即 $255$ 才代表 `Inf/NaN`），$5.37 \times 10^8$ 与 $32,768.0$ 在硬件看来都是完全合法的有限浮点数（`isfinite() == True`），因此仅靠 `NaN/Inf` 检查器无法在 AllReduce 前将其拦截，必须依靠跨秩梯度范数离群检测或 ABFT 拦截。

- **20-4 解答**：
  1. 当 $M=N=K=8,192$ 时，相对算力开销 $= \frac{3}{8,192} \approx \mathbf{0.0366\%}$（不到万分之四）。
  2. 若同一行 $i$ 的两个元素分别偏移 $+32.0$ 与 $-32.0$，该行的行和残差为 $+32.0 + (-32.0) = 0$，因此仅查行和会被漏检；但此时第 $j_1$ 列的列和残差为 $+32.0 \ne 0$，第 $j_2$ 列的列和残差为 $-32.0 \ne 0$，故**双向行列校验和能立刻捕获异常并锁定列坐标 $(j_1, j_2)$**。若再配上一组加权校验向量 $\mathbf{w} = [1, 2, \dots, N]^T$ 计算 $\mathbf{C}\mathbf{w} - \mathbf{A}(\mathbf{B}\mathbf{w})$，其第 $i$ 行残差为 $32.0(j_1 - j_2) \ne 0$，即可 $100\%$ 唯一解出两个错误元素的精确坐标与幅值。

- **20-5 解答**：
  1. 无在线防御时：每次 SDC 浪费 $2.0\ \mathrm{h}$（回滚重算）$+ 0.5\ \mathrm{h}$（排查重启）$= \mathbf{2.5\ \mathrm{h}}$ 集群墙钟时间，在十万卡集群上每百小时白白烧掉 $2.5 \times 100,000 = \mathbf{250,000\ \text{GPU 小时}}$。
  2. 部署三道防线后：单次处置耗时仅为 $50\ \mathrm{ms}$（微批重算定位）$+ 5.0\ \mathrm{s}$（`TrainMover` 热备接管）$= \mathbf{5.05\ \mathrm{s}} \approx 0.0014\ \mathrm{h}$，仅耗费 $140$ GPU 小时，每百小时净省 **$249,860\ \text{GPU 小时}$**（损耗降低 $99.94\%$）。

- **20-6 解答**：
  1. 全体完成时刻为 $\max(10.0, 14.5) + 0.5 = 15.0\ \mathrm{ms}$；前 7 个健康 Rank 看到的 AllReduce 算子持续时间为 $15.0 - 10.0 = \mathbf{5.0\ \mathrm{ms}}$，其中真实网络传输仅占 $\frac{0.5}{5.0} = \mathbf{10.0\%}$，而等待第 7 号慢卡的空转时间占比高达 **$\mathbf{90.0\%}$**。
  2. 4 张网卡的算术平均耗时为 $\frac{0.8 \times 3 + 3.2}{4} = \mathbf{1.40\ \mathrm{ms}}$，但受制于同步屏障，实际决定训练步进的木桶短板耗时为 $\max(0.8, 0.8, 0.8, 3.2) = \mathbf{3.20\ \mathrm{ms}}$。

- **20-7 解答**：
  1. 候选 X 的逐形状加速比为 $[80/10,\; 120/240] = [8.0\times,\; 0.5\times]$，算术平均加速比为 $\frac{8.0 + 0.5}{2} = \mathbf{4.25\times}$；候选 Y 的逐形状加速比为 $[80/40,\; 120/60] = [2.0\times,\; 2.0\times]$，算术平均加速比为 **$\mathbf{2.00\times}$**。
  2. 在等频流量（各 $1$ 次）下：基线总耗时 $80+120=200\ \mu\mathrm{s}$；候选 X 总耗时 $10+240 = \mathbf{250\ \mu\mathrm{s}}$（真实加速比 $200/250 = \mathbf{0.80\times}$，倒退 $25\%$）；候选 Y 总耗时 $40+60 = \mathbf{100\ \mu\mathrm{s}}$（真实加速比 $\mathbf{2.00\times}$）。令 $10p + 240(1-p) < 40p + 60(1-p) \implies 240 - 230p < 60 - 20p \implies 210p > 180 \implies p > p^* = \frac{6}{7} \approx \mathbf{85.71\%}$。

- **20-8 解答**：
  1. 在 `AABB` 顺序下：$\bar{T}_A^{\mathrm{AABB}} = \mu_A + \frac{0\delta + 1\delta}{2} = \mu_A + 0.5\delta$；$\bar{T}_B^{\mathrm{AABB}} = \mu_B + \frac{2\delta + 3\delta}{2} = \mu_B + 2.5\delta$。两者之差为 $(\mu_A - \mu_B) - 2.0\delta = (\mu_A - \mu_B) - \mathbf{24.0\ \mu\mathrm{s}}$，系统性地偏袒先跑的 A。
  2. 在 `ABBA` 顺序下：A 在 $k=0, 3$ 执行，$\bar{T}_A^{\mathrm{ABBA}} = \mu_A + \frac{0\delta + 3\delta}{2} = \mu_A + 1.5\delta$；B 在 $k=1, 2$ 执行，$\bar{T}_B^{\mathrm{ABBA}} = \mu_B + \frac{1\delta + 2\delta}{2} = \mu_B + 1.5\delta$。两者之差恒等于 $(\mu_A + 1.5\delta) - (\mu_B + 1.5\delta) \equiv \mathbf{\mu_A - \mu_B}$，一阶线性漂移完全抵消。

- **20-9 解答**：
  1. 单次读取 $4.6\ \mathrm{GB}$ INT4 权重的理论访存下限延迟为 $\frac{4.6\ \mathrm{GB}}{600\ \mathrm{GB/s}} \approx \mathbf{7.67\ \mathrm{ms}}$；若要输出一个包含 7 自由度（7 个离散词元）的动作，串行自回归生成至少需 $7 \times 7.67 \approx \mathbf{53.7\ \mathrm{ms}} > 10\ \mathrm{ms}$，且尚未计入多路视觉 Prefill 的 $38\ \mathrm{ms}$ 耗时。因此必须用 $150\ \mathrm{M}$ 参数的 System 1 流匹配动作头（权重仅 $0.3\ \mathrm{GB}$，常驻独立 MIG L2，6 步确定性 ODE 耗时 $\le 4.2\ \mathrm{ms}$）承接 $100\ \mathrm{Hz}$ 硬实时回路。
  2. 在轨道 B（DeepSeek V4.1）中：**MLA** 攻克了长上下文 KV 显存容量与带宽墙；**DeepEP** 攻克了跨节点专家 All-to-All 通信与 SM 抢占墙；**DualPipe** 攻克了流水线并行气泡与双向通信重叠墙；**`RollPacker`** 攻克了 RLVR 长思维链幂律尾部空泡与长度偏差墙；**`OpGuard` + `TrainMover`** 攻克了十万卡集群静默数据损坏（SDC）与分钟级全局重启容错墙。

- **20-10 解答**：
  在仓库根目录执行 `python3 experiments/ch20/20-01/run.py && python3 experiments/ch20/20-01/verify.py`，终端输出 `Experiment 20-01 verified successfully.`，所有指标与本章算例及锁定案例 $100\%$ 一致。

- **20-11 解答**：
  1. 全体完成时刻为 $\max(22.5, 28.5) + 1.5 = 30.0\ \mathrm{ms}$；其余 $15$ 个健康 Rank 观测到的 AllReduce 持续时间为 $30.0 - 22.5 = \mathbf{7.50\ \mathrm{ms}}$，其中等待 Rank 11 的慢节点空转时间为 $28.5 - 22.5 = 6.0\ \mathrm{ms}$，占比高达 $\frac{6.0}{7.50} = \mathbf{80.00\%}$。
  2. 若盲目将网络带宽翻倍（传输从 $1.5\ \mathrm{ms}$ 降至 $0.75\ \mathrm{ms}$），健康 Rank 看到的持续时间变为 $6.0 + 0.75 = \mathbf{6.75\ \mathrm{ms}}$，仅缩短了 $\frac{7.50 - 6.75}{7.50} = \mathbf{10.00\%}$；若通过 `NCCL Flight Recorder` 偏斜矩阵定位并用 `TrainMover` 替换慢节点 Rank 11（消除 $6.0\ \mathrm{ms}$ 偏斜），单步 AllReduce 耗时直接从 $7.50\ \mathrm{ms}$ 降至 **$1.50\ \mathrm{ms}$**，获得 **$\mathbf{5.00\times}$** 的真实加速。

- **20-12 解答**：
  在仓库根目录执行 `python3 experiments/ch20/20-02/run.py && python3 experiments/ch20/20-02/verify.py`，终端输出 `Experiment 20-02 verified successfully.`，实测锁定 `straggler_rank = 5`（等待占比 $83.33\%$）、`TrainMover` $0.68\ \mathrm{s}$ 热接管（提速 $441.18\times$）、`ABBA` 消除 $-16.0\ \mu\mathrm{s}$ 漂移，以及 $200$ 次 Chaos GameDay 演练 $99.50\%$ 召回率与 $\mathrm{MTTR}_{\mathrm{auto}} = 3.85\ \mathrm{s}$（提速 $389.61\times$）。

- **20-13 解答**：
  1. **Parachute 直接摘除 4 条受损管线**：保留的有效训练吞吐比例为 $\eta_{\mathrm{para}} = \frac{64 - 4}{64} = \frac{60}{64} = \mathbf{93.75\%}$（有 $4 \times 7 = 28$ 台幸存健康节点被迫陪绑闲置）。
  2. **Oobleck 预实例化 7 阶段模板重配置**：保留的有效训练吞吐比例为 $\eta_{\mathrm{oob}} = \frac{60 \times 1.00 + 4 \times 0.82}{64} = \frac{63.28}{64} = \mathbf{98.88\%}$（精确值为 $98.875\%$）。相比 `Parachute`，每天（$24\ \mathrm{h}$）净多挽回的等效健康节点小时数为 $512 \times (0.98875 - 0.9375) \times 24\ \mathrm{h} = 512 \times 0.05125 \times 24 = \mathbf{629.76\ \text{Node-Hours/day}}$（折合 $5{,}038.08$ GPU 小时/天）。

- **20-14 解答**：
  1. **均匀定速压测 vs. 智能体突发压测（$\rho = 0.80, T_s = 100.0\ \mathrm{ms}, T_{\mathrm{prefill}} = 60.0\ \mathrm{ms}$）**：
     - 均匀压测（$C_a = 0, C_s = 0$）：$W_{q,\mathrm{uni}} = \mathbf{0.00\ \mathrm{ms}}$，$\mathrm{TTFT}_{\mathrm{uni}} = \mathbf{60.00\ \mathrm{ms}}$；
     - 突发长尾压测（$C_a = 1.5, C_s = 1.5$）：变异因子 $\frac{1.5^2 + 1.5^2}{2} = 2.25$，排队等待时间 $W_{q,\mathrm{burst}} = \frac{0.80}{0.20} \times 2.25 \times 100.0 = 4.0 \times 225.0 = \mathbf{900.00\ \mathrm{ms}}$，平均首词元延迟暴涨至 $\mathrm{TTFT}_{\mathrm{burst}} = 60.00 + 900.00 = \mathbf{960.00\ \mathrm{ms}}$（膨胀 **$16.00\times$**）。
  2. 执行 `python3 experiments/ch20/20-03/run.py && python3 experiments/ch20/20-03/verify.py` 可完整校验算例 20-7、练习 20-13 与本题的全部锁定指标。

下表给出了全章 14 道练习与实验的核心锁定数值速查索引：

| 练习编号 | 考查工程主题 | 核心锁定输入参数 | 精确复算输出基准 |
| :--- | :--- | :--- | :--- |
| **20-1** | 64K 卡 MTBF 与 Young-Daly ETTR | $\mathrm{MTBF}_1 = 45,000\ \mathrm{h}$, $N = 65,536$ | $\mathrm{MTBF} = 41.20\ \mathrm{min}$；ETTR 从 $62.06\%$ 升至 $94.23\%$ |
| **20-2** | 热备池泊松排队容量规划 | $\lambda = 1.5/\mathrm{h}$, $\mathrm{MTTR}: 10\ \mathrm{h} \to 4\ \mathrm{h}$ | $L = 15 \implies S_{\min} = 28$ 台；缩短修复后降至 $15$ 台（省 $13$ 台） |
| **20-3** | BF16 指数比特翻转 SDC 放大 | $g = 0.125$, Bit 12 翻转（$+32$）, $16\mathrm{K}$ 卡 | 单卡变 $2^{29} = 5.37\times 10^8$，万卡平均偏移 $+32,768.0$ |
| **20-4** | ABFT 8K 矩阵乘开销与双错边界 | $M=N=K=8,192$, 同行反号错误 | 算力开销 $0.0366\%$；列和残差与加权向量 $\mathbf{w}$ $100\%$ 捕获 |
| **20-5** | OpGuard 50 ms 微批重算收益 | 每 $100\ \mathrm{h}$ 1 次 SDC, 十万卡集群 | 传统回滚损 $250,000$ 卡时 vs. 在线处置仅 $140$ 卡时 |
| **20-6** | 慢节点到达偏斜与多轨失衡 | 7 卡 $10\ \mathrm{ms}$ vs. 慢卡 $14.5\ \mathrm{ms}$, 传输 $0.5\ \mathrm{ms}$ | 观测耗时 $5.0\ \mathrm{ms}$（$90\%$ 在等慢卡）；4 轨短板 $3.20\ \mathrm{ms}$ |
| **20-7** | 算术平均加速比反例与翻转阈值 | 基线 $[80,120]$, X $[10,240]$, Y $[40,60]$ | X 算术平均 $4.25\times$ 实则 $0.80\times$；需 $p > 6/7$（$85.71\%$） |
| **20-8** | 线性热漂移 AABB vs. ABBA | $d_k = k\delta,\; \delta = 12.0\ \mu\mathrm{s}$ | AABB 偏置 $-24.0\ \mu\mathrm{s}$；ABBA 恒等于 $\mu_A - \mu_B$ |
| **20-9** | 全书双轨从机载到万卡综合题 | Qwen3-8B INT4 $4.6\ \mathrm{GB}$ vs. DeepSeek V4.1 | 机载单步访存下限 $7.67\ \mathrm{ms}$；五大系统技术各破一墙 |
| **20-10** | 十万卡容错与评测反例实验 | `experiments/ch20/20-01` 锁定脚本 | $\mathrm{ETTR}\ 63.22\% \to 92.34\%, S_{\min}=40, p^*=15/19$ |
| **20-11** | 16-Rank 入队偏斜反演与热迁移 | $15$ 卡 $22.5\ \mathrm{ms}$ vs. 慢卡 $28.5\ \mathrm{ms}$, 传输 $1.5\ \mathrm{ms}$ | 观测 $7.50\ \mathrm{ms}$（$80\%$ 等待）；带宽翻倍仅省 $10\%$，替换慢卡快 $5.00\times$ |
| **20-12** | 偏斜定位、金丝雀与混沌演练实验 | `experiments/ch20/20-02` 锁定脚本 | `straggler_rank=5`（$83.33\%$），`TrainMover` 快 $441.18\times$，演练召回 $99.50\%$ |
| **20-13** | 64 管线 Oobleck 异构模板容错 | $F=4/64$ 受损, $\mathcal{T}_7$ 效率 $82.0\%$ | Parachute $93.75\%$ vs. Oobleck $98.88\%$（日省 $629.76$ 节点时） |
| **20-14** | 突发流量 Kingman 膨胀 (`20-03`) | $\rho=0.80, C_a=C_s=1.5, T_s=100\ \mathrm{ms}$ | 算例 20-7 Goodput 降 $51.27\%$；本题 $\mathrm{TTFT}\ 60 \to 960.0\ \mathrm{ms}$（$16\times$） |

*表 20-10　第 20 章十四道练习与可运行实验核心锁定数值速查表。*

---

## 本章小结

1. **十万卡可靠性反比坍缩与热备池排队论**：集群级平均无故障时间随规模按 $\mathrm{MTBF}_{\mathrm{cluster}} = \mathrm{MTBF}_{\mathrm{unit}} / N$ 反比衰减——从 Meta $16,384$ 卡的 $3.09$ 小时（$54$ 天 $419$ 次中断，硬件占 $78\%$）外推至十万卡时，每 **$30.41$ 分钟**即发生一次中断。利用内存异步快照（$C=5\ \mathrm{s}$）、`TrainMover` 秒级无重启接管（$T_{\mathrm{rec}}=5\ \mathrm{s}$）与泊松维修队列热备池（$L=24 \implies S_{\min}=40$ 台，占比 $<0.4\%$），可将十万卡有效训练时间比（ETTR）从 $63.22\%$ 拉升至 **$92.34\%$**。
2. **静默数据损坏（SDC）的三道毫秒级在线防线**：算术单元内部的瞬态或老化比特翻转会逃逸 HBM ECC 并通过单次 AllReduce 污染全网权重。通过 **（1）算子级 ABFT 行列校验和**（开销 $0.0732\% < 0.5\%$）、**（2）AllReduce 归约前跨秩梯度统计拦截**（`Safeguarding LLM Training` OSDI '26，3,500 万卡时验证）与 **（3）控制良性非确定性的位级对齐重放定位**（`OpGuard` OSDI '26，耗时 $50\ \mathrm{ms}$ 微批重算），可在线揪出物理坏卡并实现零检查点回滚自愈。
3. **慢节点到达偏斜取证、光链路灰故障预测与 Oobleck 无重启弹性重配置**：在紧耦合集合通信中，单张慢卡迟到 $2.0\ \mathrm{ms}$ 会让其余所有健康卡显示“通信耗时 $2.4\ \mathrm{ms}$”（其中 $83.33\%$ 在空等慢卡）。利用 `NCCL Flight Recorder` 入队时间戳偏斜矩阵与 800G 光模块 RS-FEC 高阶纠错桶斜率监控，配合 `TrainMover` $0.68\ \mathrm{s}$ 亚秒级热迁移（比全局重启快 $441.18\times$）以及热备耗尽时的 `Oobleck` 预实例化异构流水线模板切换（$1.45\ \mathrm{s}$ 恢复，提速 $206.90\times$，保留 $98.75\%$ 算力），可实现全天候无中断训练。
4. **反作弊基准评测科学、泊松突发流量 Goodput 与混沌演练门禁**：算术平均加速比会制造“$5.25\times$ 加速实则比基线更慢”的假象，单调 `AABB` 测量会把热漂移当成算法收益，均匀定速发包（$C_a=C_s=0$）会掩盖真实泊松突发长尾流量（$C_a=1.0, C_s=1.2$）在 $\rho=0.85$ 下高达 $11.37\times$ 的 TTFT 排队膨胀与 $51.27\%$ 的 SLO Goodput 缩水。结合 `FlashInfer-Bench`、`CCEval`、`DriftBench`、`Blink`（OSDI '26）、两阶段金丝雀门禁与 $200$ 次 Chaos GameDay 主动故障注入演练（$99.50\%$ 召回率，$\mathrm{MTTR}_{\mathrm{auto}} = 3.85\ \mathrm{s}$），本章完成了前二十章通用大模型底座的阶段性收束，并正式开启后续前沿与全栈架构疆域。

[^ch20-llama3]: 关于 Meta Llama 3 405B 在 $16,384$ 张 H100 GPU 上历时 $54$ 天的 $419$ 次非计划中断分类账本，参见 Dubey 等人的技术报告 *The Llama 3 Herd of Models*（Section 3.3, arXiv:2407.21783, 2024）。
[^ch20-daly]: 关于周期性检查点开销与平均无故障间隔下的最优检查点间隔公式（$\tau_{\mathrm{opt}} = \sqrt{2 C_{\mathrm{ckpt}} M}$），参见 Daly 的经典论文 *A Higher Order Estimate of the Optimum Checkpoint Interval for Restart Dumps*（Future Generation Computer Systems, 2006）。
[^ch20-sparing]: 关于超大规模 AI 集群故障分布建模与基于排队论的热备节点池（Hot-Spare Pool）优化策略，参见 MLSys 2026 论文 *Understanding Failures and Optimizing Sparing Strategies for Large-Scale AI Clusters*（MLSys '26 #33）。
[^ch20-abft]: 关于利用行列校验和线性不变量在线检测并修正矩阵运算硬件错误的算法级容错（ABFT），参见 Huang 与 Abraham 的奠基性论文 *Algorithm-Based Fault Tolerance for Matrix Operations*（IEEE Transactions on Computers, 1984）。
[^ch20-safeguarding]: 关于历经 3,500 万 GPU 小时验证的低开销在线静默数据损坏（SDC）检测与归约前隔离框架，参见 OSDI 2026 论文 *Safeguarding LLM Training at Scale: A Low-Overhead Online Detection and Localization Framework Validated Over 35 Million GPU Hours*（OSDI '26 #71）。
[^ch20-opguard]: 关于通过消除良性软硬件非确定性实现位级对齐重放、从而在单微批次内精确定位大模型训练 SDC 根因算子与物理坏卡的方法，参见 OSDI 2026 论文 *OpGuard: Debugging Silent Data Corruption in LLM Training via Bitwise Alignment*（OSDI '26 #72）。
[^ch20-trainmover]: 关于利用预初始化备用进程与局部 RDMA 队列对（QP）重映射实现亚秒/秒级无重启故障恢复的架构，参见 OSDI 2026 论文 `TrainMover` 以及 NSDI 2026 论文 `EROICA` 与 `FLARE`。
[^ch20-flashinfer-bench]: 关于绑定真实生产形状分布的算子评测基准 `FlashInfer-Bench` 与动态负载漂移基准 `DriftBench`，参见 MLSys 2026 相关论文及本书案例库 `case-studies/optimization-evaluation-and-deployment.md`。
[^ch20-cceval]: 关于集合通信真实拓扑评测基准 `CCEval` 及慢节点偏斜、多轨失衡与环路死锁诊断，参见 NSDI 2026 论文 `CCEval` 及本书案例库 `case-studies/collective-paths-and-diagnosis.md`。
[^ch20-blink]: 关于消除扁平微秒级运行时负载上硬件 PMU 周期性采样滑移失真（Sampling Skid）的无偏性能剖析器，参见 OSDI 2026 论文 *Blink: When Sampling Lies — Recovering Trustworthy Profiling on Flat Workloads*（OSDI '26 #23）。
[^ch20-oobleck]: Jang, I., et al. "Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates." *SOSP*, 2023.
[^ch20-parachute]: Watcharapichat, P., et al. & Elastic Pipeline Fault-Tolerant Training Frameworks ("Parachute: Responsive and Fault-Tolerant Data-Parallel / Pipeline-Parallel Training"). *MLSys / EuroSys*, 2023–2024.
[^ch20-bamboo]: Thorpe, J., et al. "Bamboo: Making Preemptible Instances Resilient for Affordable Large Model Training." *NSDI*, 2023.
[^ch20-varuna]: Athlur, S., et al. "Varuna: Scalable, Low-Cost Training of Massive Deep Learning Models." *EuroSys*, 2022.
[^ch20-megascale]: Jiang, Z., et al. (ByteDance). "MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs." *NSDI*, 2024.
[^ch20-unicron]: He, T., et al. "Unicron: Economizing Self-Healing LLM Training at Scale." *EuroSys*, 2024.
[^ch20-splitwise]: Patel, P., et al. "Splitwise: Efficient Generative LLM Inference Using Phase Splitting." *ISCA*, 2024.
[^ch20-distserve]: Zhong, Y., et al. "DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized Large Language Model Serving." *OSDI*, 2024.
[^ch20-sarathi]: Agrawal, A., et al. "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve." *OSDI*, 2024.
[^ch20-mlperf]: Reddi, V. J., et al. "MLPerf Inference Benchmark: Server Poisson Arrival vs. Offline Scenarios." *ISCA*, 2020–2025.
[^ch20-kingman]: Kingman, J. F. C. "The Single Server Queue in Heavy Traffic." *Mathematical Proceedings of the Cambridge Philosophical Society*, 57(4), 902–904, 1961.
[^ch20-sharegpt]: Zheng, L., et al. "LMSYS-Chat-1M & ShareGPT: A Large-Scale Real-World LLM Conversation Dataset for Bursty Workload Modeling." *ICLR*, 2024.
