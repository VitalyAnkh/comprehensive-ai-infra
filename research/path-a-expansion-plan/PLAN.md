# 《深入理解 AI Infra》路径 A 全景升级与扩写执行计划（Master Plan）

> **版本**：v2.0（2026-11-06，根据第 1 轮审阅全面校准文件名、小节落点、练习编号、算例锁定真值与构建脚本边界）  
> **目标定位**：采用**路径 A（既有 12 章全量深度扩充 + 新增第 13 章《AI 原生存储系统与数据底座》+ 新增第 14 章《模型与系统硬件协同设计》）**，将仓库内已储备但尚未被正文直接引用的 **16 篇深度案例（`case-studies/`）**、专项研究报告（`research/`）以及 **2025–2026 年顶会与工业界前沿机制**系统化织入全书；全程按章节分批推进，严格执行“**调用 `codex exec` 使用 `image_gen` 工具与 `gpt-image-2.6-sunburst` 模型（最高质量 `quality="high"`）结合 `figure_style` 绘制 420pt 出版级配图三件套（`.svg`、`.png`、`.pdf`）$\to$ `refine-prose` 深度润色去 AI 腔 $\to$ 本地自动化脚本全套校验 $\to$ Gemini 4 Argon High 同配置 Subagent 多轮审阅直至收敛 $\to$ 每完成一个批次推送部署**”的闭环工作流。

---

## 一、 总体架构设计与工程不变量

### 1.1 为什么选择“12 章原位扩充 + 末尾新增第 13、14 章”的拓扑结构
本书仓库拥有一套严密的自动化构建与验证体系（[`scripts/verify_core_principles.py`](../../scripts/verify_core_principles.py)、[`scripts/build_site.py`](../../scripts/build_site.py)、[`scripts/check_site.py`](../../scripts/check_site.py)、[`book/build_pdf.py`](../../book/build_pdf.py)），且第 1～12 章的章号（`ch01`～`ch12`）、跨章图号引用（`图 X-Y`）、基线练习/实验编号（`练习 X-Y` / `实验 X-Y` / `习题 X-Y`）以及多语种对照目录（`book-en/`、`book-zh-tw/`、`book-ru/`）均与 `01`～`12` 的编号强绑定。

若在中间插入新章导致后继章节顺延改号，会破坏历史实验目录（`experiments/ch01`～`ch12`）与跨语种锚点映射。因此，**路径 A 采取“原位强化 1～12 章 + 尾部追加第 13、14 章”的无损扩展架构**：

1. **第一阶段：全栈需求、模型架构与负载形态（第 1～3 章）**
   - [`manuscripts/01-初识 AI Infra.md`](../../manuscripts/01-%E5%88%9D%E8%AF%86%20AI%20Infra.md)、[`manuscripts/02-模型架构.md`](../../manuscripts/02-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84.md)、[`manuscripts/03-推理与训练负载.md`](../../manuscripts/03-%E6%8E%A8%E7%90%86%E4%B8%8E%E8%AE%AD%E7%BB%83%E8%B4%9F%E8%BD%BD.md)
   - 强化芯片内数据移动物理代价（LogicFolding）、网络包处理核数预算与主机调度策略卸载（可编程网卡 / Wave）、机队级全栈效率分解（Google MLSys 2026 $\mathrm{MPG}=\mathrm{SG}\times\mathrm{RG}\times\mathrm{PG}$）、跨层 KV 共享（CSA2）深度账本、扩散语言模型（dLLM）与多模态图像/视频生成（`Qwen-Image-2512` vs. `FLUX.2-klein-4B` DiT + 3D Causal VAE）、检索—生成协同预算（RAG + 近存向量检索）以及具身智能（VLA）双环路负载。
2. **第二阶段：加速器、算子运行时、超节点与数据中心网络（第 4～7 章）**
   - [`manuscripts/04-加速器架构.md`](../../manuscripts/04-%E5%8A%A0%E9%80%9F%E5%99%A8%E6%9E%B6%E6%9E%84.md)、[`manuscripts/05-算子与运行时.md`](../../manuscripts/05-%E7%AE%97%E5%AD%90%E4%B8%8E%E8%BF%90%E8%A1%8C%E6%97%B6.md)、[`manuscripts/06-超节点.md`](../../manuscripts/06-%E8%B6%85%E8%8A%82%E7%82%B9.md)、[`manuscripts/07-数据中心网络.md`](../../manuscripts/07-%E6%95%B0%E6%8D%AE%E4%B8%AD%E5%BF%83%E7%BD%91%E7%BB%9C.md)
   - 强化统一内存 vs. 独立显存单步带宽下界（M2 Max vs. RTX PRO 6000）、GPU 机密计算（TEE Bounce Buffer）、FlashInfer `plan/run` 二阶段解耦与 `ggml` Metal 算子分派、800 VDC 高压供电与共封装光学（CPO）、分层内存关键度迁移（PACT/Camp）、Trace 驱动并行方案排序以及多轨分层集合通信与光模块闪断容错。
3. **第三阶段：推理优化、分布式推理、训练系统、资源调度与端边云（第 8～12 章）**
   - [`manuscripts/08-推理优化.md`](../../manuscripts/08-%E6%8E%A8%E7%90%86%E4%BC%98%E5%8C%96.md)、[`manuscripts/09-分布式推理.md`](../../manuscripts/09-%E5%88%86%E5%B8%83%E5%BC%8F%E6%8E%A8%E7%90%86.md)、[`manuscripts/10-训练系统.md`](../../manuscripts/10-%E8%AE%AD%E7%BB%83%E7%B3%BB%E7%BB%9F.md)、[`manuscripts/11-资源调度与运行环境.md`](../../manuscripts/11-%E8%B5%84%E6%BA%90%E8%B0%83%E5%BA%A6%E4%B8%8E%E8%BF%90%E8%A1%8C%E7%8E%AF%E5%A2%83.md)、[`manuscripts/12-端边云协同.md`](../../manuscripts/12-%E7%AB%AF%E8%BE%B9%E4%BA%91%E5%8D%8F%E5%90%8C.md)
   - 强化结构化生成与语法掩码重叠（XGrammar）、KV 压缩长度膨胀反转与混合模型检查点（Marconi）、A100/H20 与 KTransformers 异构 PD/AF 切分、万卡静默数据损坏（SDC）排查、RL 后训练长尾采样与流式梯度（RollPacker）、多作业 RL 交错与角色解耦恢复（Weave/RobustRL）、权重交接显存峰值控制（Weight Handoff）、沙箱快照与 CXL 共享只读页（E2B/CXLfork）、24×7 数字员工成本核算及跨广域网语音传输（Queqiao）与移动端闪存流式推理。
4. **第四阶段（新增）：AI 原生存储底座与跨层协同设计终章（第 13～14 章）**
   - **新增第 13 章《AI 原生存储系统与数据底座》**（[`manuscripts/13-AI 原生存储系统与数据底座.md`](../../manuscripts/13-AI%20%E5%8E%9F%E7%94%9F%E5%AD%98%E5%82%A8%E7%B3%BB%E7%BB%9F%E4%B8%8E%E6%95%B0%E6%8D%AE%E5%BA%95%E5%BA%A7.md)，配图目录 `manuscripts/ch13/`）：承接第 9.5.2 节与第 10.4.3 节，下沉剖析存储协议栈与硬件介质——POSIX Page Cache vs. GPUDirect Storage（GDS）P2P DMA 零拷贝、`io_uring`/SPDK/USRBIO、AI 原生分布式文件系统（DeepSeek 3FS / CRAQ 链式复制）、NVMe SSD 擦写块/页粒度导致的读写放大（WAF/RAF）与 ZNS 分区命名空间、Layer-major vs. Block-major 布局在 $4\ \mathrm{KiB}$ LBA 上的 IOPS 放大，以及 PB 级多模态训练数据预处理流水线。
   - **新增第 14 章《模型与系统硬件协同设计》**（[`manuscripts/14-模型与系统硬件协同设计.md`](../../manuscripts/14-%E6%A8%A1%E5%9E%8B%E4%B8%8E%E7%B3%BB%E7%BB%9F%E7%A1%AC%E4%BB%B6%E5%8D%8F%E5%90%8C%E8%AE%BE%E8%AE%A1.md)，配图目录 `manuscripts/ch14/`）：定位为**“跨层联立方程组与反事实推演（Counterfactual Co-Design）”终章**，避免与前 12 章已铺垫的基础概念简单重复，重点落地 [`research/deepseek-v41-codesign-2026-09-10/README.md`](../../research/deepseek-v41-codesign-2026-09-10/README.md) 第 3.2、4、5 节与 [`case-studies/opentallas.md`](../../case-studies/opentallas.md) Case B/D/E/F 的深水区内容（长上下文双轨演进表、V4.1 Flash vs. Kimi K3 在 $200\mathrm{K}/1\mathrm{M}$ 四代 GPU 上的单流 Agent 吞吐与固定开销审计、OpenTallas 四资源账本与晶圆级 15 跳通信直径越界、四大反事实思想实验）。

---

## 二、 仓库 16 篇“未直接引用”案例的精确分类与 100% 吸收映射表

目前 [`case-studies/README.md`](../../case-studies/README.md) 第 123–138 行列有 **16 篇“延伸阅读（正文未直接引用）”案例**。经逐行核对正文，这 16 篇分为两类：
- **A 类（6 篇，正文已有部分主体或算例，但漏挂了 `../case-studies/*.md` 脚注链接或仅缺局部子项）**：严禁在正文重复编写已有算例或重复画图，只补齐 `../case-studies/*.md` 脚注链接及尚未入书的增量推导/子项。
- **B 类（10 篇，正文中完全未出现，需完整撰写机制、算例与脚注链接）**。

下表给出全部 16 篇案例的精准落点与锁定真值：

| 序号 | 类别 | 案例文件（`case-studies/`） | 精确目标章节与真实小节号 | 已有内容边界与本次待补齐的锁定真值 / 核心机制 |
| :---: | :---: | :--- | :--- | :--- |
| 1 | **A 类** | [`logicfolding-energy.md`](../../case-studies/logicfolding-energy.md) | 第 1 章 `1.2.2`、第 4 章 `4.1.3` | `4.1.3`（第 122–150 行及图 4-4）已写出 `1.54 pJ -> 0.21 pJ`、`0.16 pJ` 寄存器能量与 `0.40 mm²` 面积。**本次动作**：在 `4.1.3` 补挂 `../case-studies/logicfolding-energy.md` 脚注链接并补入 $P_{\mathrm{dynamic}}\approx \alpha C V^2 f$ 降压算例（$0.85\ \mathrm{V}\to 0.55\ \mathrm{V}$ 动态功耗比 $(0.55/0.85)^2\approx 0.42$）；在 `1.2.2` 补充简短引述与链接。 |
| 2 | **A 类** | [`programmable-nic.md`](../../case-studies/programmable-nic.md) | 第 1 章 `1.4.2`、第 7 章 `7.3.4` | `1.4.2`（第 413 行 $3\sim 6$ 核）与 `7.3.4`（第 583–588 行及图 7-28 KV-Direct 64 tags / `60 MOPS`）已在正文。**本次动作**：在 `1.4.2` 与 `7.3.4` 补挂 `../case-studies/programmable-nic.md` 脚注链接，并在 `7.3.4` 补入 ClickNP FPGA 空间流水线与 CPU 核转发的对比。 |
| 3 | **A 类** | [`kernel-and-fleet-efficiency.md`](../../case-studies/kernel-and-fleet-efficiency.md) | 第 1 章 `1.3.4`、第 4 章 `4.2.3`、第 10 章 `10.6.1` | `4.2.3`（第 316–339 行 `例 4-1` 及图 4-11）**已完整写出 FlashAttention-4 的 $128\times 128$ 分块与 `1024/768/1024` 周期账本，切勿在第 4 章重复建算例或重复画图**！**本次动作**：① 在 `4.2.3` 补挂 `../case-studies/kernel-and-fleet-efficiency.md` 脚注链接及 vLLM/SGLang 后端分派表；② 将正文中尚未出现的 **Google MLSys 2026 机队效率框架（$\mathrm{MPG}=\mathrm{SG}\times\mathrm{RG}\times\mathrm{PG}$）及本书构造的同型 100-GPU 池教学算例（$\mathrm{SG}=83.33\%\times\mathrm{RG}=80\%\times\mathrm{PG}=50\%=33.33\%$，明确标注为教学算例而非论文实测值）** 写入第 1 章 `1.3.4` 与第 10 章 `10.6.1`。 |
| 4 | **A 类** | [`macbook-rtx-pro6000.md`](../../case-studies/macbook-rtx-pro6000.md) | 第 4 章 `4.3.1`/`4.8.3`、第 5 章 `5.4.1`、第 12 章 `12.1.5` | `4.8.3`（图 4-43、4-44、`实验 4-6`）已有 Q 投影实测。**本次动作**：① 在 `4.3.1`/`4.8.3` 补充 `4 GiB` 单步读取带宽下界表（M2 Pro `200 GB/s` 需 `21.47 ms`、M2 Max `400 GB/s` 需 `10.74 ms`、RTX PRO 6000 `1,792 GB/s` 需 `2.40 ms`，比值 `4.48` / `8.96`）并挂案例链接；② 在 `5.4.1` 补充 Ollama v0.20.7 `ggml` Metal 缓冲区与算子分派；③ 在 `12.1.5` 补充 Ollama Generate API 计时字段拆解。 |
| 5 | **A 类** | [`cache-and-reconfiguration.md`](../../case-studies/cache-and-reconfiguration.md) | 第 5 章 `5.5.2`、第 8 章 `8.3.4`/`8.4.4`、第 11 章 `11.3.1` | `5.5.2` 已有 FlashInfer 实测，`8.3.3` 已提 Marconi。**本次动作**：① 在 `5.5.2` 补入 `Qwen3-8B` 36 层共用一次 `plan`（$20\ \mu\mathrm{s}$）vs. 逐层重做（$720\ \mu\mathrm{s}$）手算推导、V4/K3 异构层分组 `plan` 约束及案例链接；② 在 `8.3.4` 补入 Marconi 二次出现识别保存/三次出现受益及 `FLOPs/byte` 淘汰权重；③ 在 **`8.4.4`** 完整补入 **Rethinking KV Cache Compression（压缩后单步从 $50$ 升至 $60\ \mathrm{tok/s}$，但输出从 $200$ 涨至 $300\ \mathrm{tok}$，总耗时从 $0.5+200/50=4.5\ \mathrm{s}$ 反增至 $0.5+300/60=5.5\ \mathrm{s}$）**；④ 在 **`11.3.1`** 完整补入 **Rubick 弹性重配（$1.2\ \mathrm{s}\to 1.0\ \mathrm{s/step}$，切换耗时 $80\ \mathrm{s}$ 需剩余 $>400$ 步才回本）**。 |
| 6 | **A 类** | [`queqiao.md`](../../case-studies/queqiao.md) | 第 3 章 `3.3.3`、第 12 章 `12.3.5` | 第 12 章已有专门小节 `12.3.5 案例：Queqiao 如何加速跨地域语音服务`（图 12-21、12-22）。**本次动作**：在 `3.3.3` 与 `12.3.5` 补挂 `../case-studies/queqiao.md` 脚注链接，并在 `12.3.5` 显式补入三行十进制时间预算表（理想串行发送 $355{,}000\times 8\div 333{,}000{,}000\approx 8.5\ \mathrm{ms}$、$\mathrm{BDP}=333{,}000{,}000\times 0.2\div 8\approx 8.3\ \mathrm{MB}$、简化时间预算 $200+8.5+30\approx 239\ \mathrm{ms}$）。 |
| 7 | **B 类** | [`host-policy-and-dispatch.md`](../../case-studies/host-policy-and-dispatch.md) | 第 1 章 `1.4.2`、第 11 章 `11.1.2` | ASPLOS 2026 Wave 策略卸载至 SmartNIC Arm 核：32 核主机释放 2 个管理核后吞吐比 $32(1-d)/30$ 的翻转阈值 $d < 6.25\%$（$d=2\%$ 时净增 $4.53\%$，$d=8\%$ 时净降 $1.87\%$）；预取读取 $0.75\ \mu\mathrm{s}$ 与可重叠工作 $1\ \mu\mathrm{s}/0.2\ \mu\mathrm{s}$；以及 `11.1.2` Agent 平台（驻留 $30\ \mathrm{s}$、占 $2\ \mathrm{GiB}$、执行 $3\ \text{核秒}$）在 $600\ \mathrm{GiB}$ 内存封顶下吞吐锁死在 $10\ \text{项/s}$，唯有升至 $720\ \mathrm{GiB}$ 后 CPU 卸载才释放真实吞吐。 |
| 8 | **B 类** | [`memory-criticality-and-tiering.md`](../../case-studies/memory-criticality-and-tiering.md) | 第 6 章 `6.6`、第 9 章 `9.5.2` | ASPLOS 2026 PACT/Camp 分层内存：`Qwen3-8B` 单层 FFN（$288\ \mathrm{MiB}$）在源端 $100/40/8\ \mathrm{GiB/s}$ 与 H2D $24\ \mathrm{GiB/s}$ 下的传输下界 $\max(288/B_{\mathrm{source}}, 288/24)$（分别为 $11.72\ \mathrm{ms}$、$11.72\ \mathrm{ms}$、$35.16\ \mathrm{ms}$），结合 $16\ \mathrm{ms}/40\ \mathrm{ms}$ 预取提前量推导暴露等待（$0\ \mathrm{ms}$ vs. $19.16\ \mathrm{ms}$）与页迁移回本不等式（单次迁移 $35.16\ \mathrm{ms}$，在 $16\ \mathrm{ms}$ 提前量下需未来 2 次复用省 $38.31\ \mathrm{ms}$ 才回本）。 |
| 9 | **B 类** | [`profile-and-plan-ranking.md`](../../case-studies/profile-and-plan-ranking.md) | 第 6 章 `6.6.3`、第 10 章 `10.6.1` | MICRO 2024 vTrain 配置排序反转：候选 A（计算 $70\ \mathrm{ms}$ + 暴露通信 $30\ \mathrm{ms}$）与候选 B（计算 $90\ \mathrm{ms}$ + 暴露通信 $12\ \mathrm{ms}$）孤立预测为 $100\ \mathrm{ms}$ vs. $102\ \mathrm{ms}$，当真实训练通信膨胀系数 $k > 10/9 \approx 1.11$ 时排序反转（如 $k=1.3$ 时 $T_A=109\ \mathrm{ms} > T_B=105.6\ \mathrm{ms}$）。 |
| 10 | **B 类** | [`structured-generation.md`](../../case-studies/structured-generation.md) | 第 8 章 `8.2.4`/`8.5.1`、第 11 章 `11.2.1` | MLSys 2025 XGrammar（严格采用案例锁定真值）：`Qwen3-8B`（词表 $151{,}936$，$B=64$）32-bit 位图掩码每行 $\lceil 151936/32\rceil\times 4 = 18{,}992\ \mathrm{B}$，$B=64$ 共 $1{,}215{,}488\ \mathrm{B}$（**$1.159\ \mathrm{MiB}$**），按 $25\ \mathrm{GB/s}$ H2D 传输需 **$48.62\ \mu\mathrm{s}$**；GPU 侧读写整张 FP32 logits `[64, 151936]` 一读一写共 $77{,}791{,}232\ \mathrm{B}$（**$74.188\ \mathrm{MiB}$**），按 $1\ \mathrm{TB/s}$ 显存带宽需 **$77.79\ \mu\mathrm{s}$**；CPU 掩码单请求 $50\ \mu\mathrm{s}$、64 路串行 **$3.2\ \mathrm{ms}$**，GPU 前向 **$2\ \mathrm{ms}$**（串行步时 **$5.2\ \mathrm{ms}$** $\to$ CPU/GPU 流水重叠后 $\max(3.2, 2.0)=\mathbf{3.2\ \mathrm{ms}}$ $\to$ CPU 并行压至 $<2\ \mathrm{ms}$ 后步时回到 **$2\ \mathrm{ms}$**）；投机解码 4 草稿 + 1 bonus（共 320 行）位图增至 $6{,}077{,}440\ \mathrm{B}$（**$5.796\ \mathrm{MiB}$**）及拒绝时的 PDA 栈回滚。 |
| 11 | **B 类** | [`pd-af-heterogeneous.md`](../../case-studies/pd-af-heterogeneous.md) | 第 9 章 `9.2.3`、`9.3` | 区分阶段切分（PD：A100 80GB SXM 跑 prefill + H20 96GB 跑 decode，$L=32, S=8{,}192, H_{kv}=8, d=128, b=2$ 产生 **$1\ \mathrm{GiB}$** KV，在 $25\ \mathrm{GB/s}$ 跨池链路上仅有效载荷即需 **$42.95\ \mathrm{ms}$**）与层内算子切分（AF：RTX 4090 跑 Attention + 双路 Xeon Gold 6454S 跑 MoE 专家的 KTransformers，每步跨 36 层往返 72 次交接）。 |
| 12 | **B 类** | [`rollout-tail-and-sampling.md`](../../case-studies/rollout-tail-and-sampling.md) | 第 10 章 `10.5.2`/`10.5.3`、第 11 章 `11.3.4` | NSDI 2026 RollPacker：① 超额提交（$160\times 10=1600$ vs. $128\times 8=1024$，多提交 $56.25\%$）抢先完成导致的采样分布偏移反例（长回答保留概率从 $1/2$ 降为 $1/4$，单次采样期望 $5.5\ \mathrm{s}$ vs. 双槽抢先墙钟 $3.25\ \mathrm{s}$ 但耗 $6.5\ \text{槽·秒}$）；② Stream Trainer 半数卡转训练的 KV 容量检查（`Qwen3-8B` 32 条 $8\mathrm{K}$ 轨迹占 $36\ \mathrm{GiB}$、4 卡各 $9\ \mathrm{GiB}\le 12\ \mathrm{GiB}$ 可行；$16\mathrm{K}$ 占 $72\ \mathrm{GiB}$、单卡 $18\ \mathrm{GiB}$ 越界）与时间线（串行 $60\ \mathrm{s}\to$ 正常迁移 $\max(43,29)+8+4=55\ \mathrm{s}\to$ 若拖慢 rollout 则 $\max(50,36)+8+4=62\ \mathrm{s}$）；③ 非均匀微批梯度归一化反例（$(1\times 1+3\times 3)/4=2.5 \neq (1+3)/2=2$；逐样本裁剪 $1.75$ vs. 整批裁剪 $2.0$）。 |
| 13 | **B 类** | [`rl-scheduling-and-recovery.md`](../../case-studies/rl-scheduling-and-recovery.md) | 第 10 章 `10.5.2`/`10.5.3`、第 11 章 `11.3.3` | OSDI 2026 Weave 互补 RL 作业（A: $360\mathrm{s}/120\mathrm{s}$，B: $120\mathrm{s}/360\mathrm{s}$）共享一对池实现 $480\ \mathrm{s}$ 无气泡周期（加 $5\ \mathrm{s}$ 热切换为 $490\ \mathrm{s}$，冷启动 $45\ \mathrm{s}$ 则达 $570\ \mathrm{s} > 528\ \mathrm{s}$ SLO），以及主机内存驻留陷阱（$400+500+150=1050\ \mathrm{GiB} > 1\ \mathrm{TiB}$）；OSDI 2026 RobustRL 角色解耦故障恢复（更新至 $30\ \mathrm{s}$ 故障，整轮重做 $300+30+20+80+300+60=790\ \mathrm{s}$ vs. 保留轨迹仅重做更新 $300+30+20+80+60=490\ \mathrm{s}$，省下 $300\ \mathrm{s}$ rollout）。 |
| 14 | **B 类** | [`weight-handoff.md`](../../case-studies/weight-handoff.md) | 第 5 章 `5.5.2`、第 10 章 `10.5.2` | `Qwen3-8B`（$8{,}190{,}735{,}360$ 参数，BF16 权重 **$15.2564\ \mathrm{GiB}$**）在 $80\ \mathrm{GB}$（$74.5\ \mathrm{GiB}$）H100 上训推切换：训练独占 $40+4=44\ \mathrm{GiB}$，rollout 独占 $15.2564+24+4=43.2564\ \mathrm{GiB}$；无序恢复峰值 **$83.2564\ \mathrm{GiB}$**（OOM）vs. 分阶段有序交接峰值 $\max(40+15.2564+4, 15.2564+24+4)=\mathbf{59.2564\ \mathrm{GiB}}$；`Qwen3-235B` 路由专家（94 层 $\times$ 128 专家 $\times 36\ \mathrm{MiB}=\mathbf{423\ \mathrm{GiB}}$）在 EP16 下按需接收（**$26.4375\ \mathrm{GiB}$**，接收端下界从 $18.17\ \mathrm{s}$ 降至 $1.14\ \mathrm{s}$）受限于 8 生产端出口（**$2.27\ \mathrm{s}$**），以及最大原子切片（Embedding $151936\times 4096$ BF16 占 $1.1592\ \mathrm{GiB}$，双缓冲槽至少需 **$2.3184\ \mathrm{GiB}$**）。 |
| 15 | **B 类** | [`snapshot-residency-and-first-use.md`](../../case-studies/snapshot-residency-and-first-use.md) | 第 11 章 `11.2.2`、`11.2.3` | E2B Firecracker `UFFDIO_COPY` 按需缺页与 ASPLOS 2025 CXLfork 共享只读页对比：100 个派生自 $2\ \mathrm{GiB}$ 模板的沙箱（触及 $512\ \mathrm{MiB}$，其中写 $128\ \mathrm{MiB}$、只读 $384\ \mathrm{MiB}$，管理开销 $4\ \mathrm{MiB}$）在全量复制（$200.390625\ \mathrm{GiB}$）、按需复制（$50.390625\ \mathrm{GiB}$）、只读共享（$12.890625\ \mathrm{GiB}$）及热点本地化（$37.890625\ \mathrm{GiB}$）下对应 $64\ \mathrm{GiB}$ 预算的容量上界（**31 / 127 / 496 / 168 个环境**）；以及 64 个离散 $4\ \mathrm{KiB}$ 页分布在 64 个 $2\ \mathrm{MiB}$ 块时触发大页读放大（$256\ \mathrm{KiB}/0.244\ \mathrm{ms}\to 128\ \mathrm{MiB}/125\ \mathrm{ms}$）。 |
| 16 | **B 类** | [`single-agent-serving-cost.md`](../../case-studies/single-agent-serving-cost.md) | 第 11 章 `11.4.4`、第 14 章 `14.3` | 一名 24×7 数字员工在 $200\mathrm{K}/1\mathrm{M}$ 上下文下的自建 GPU 池（H100/H200/B200/B300）vs. 官方 API 成本对决：对比 DeepSeek V4 Flash（逻辑状态 $1.30\to 6.42\ \mathrm{GiB}$）与 Kimi K3（逻辑状态 $5.57\to 26.17\ \mathrm{GiB}$）计入周期性前缀重建后的真实吞吐，以及 `OPTIMIZATION-AUDIT.md` 揭示的 K3 未优化基线中约七成耗时来自逐层固定调度开销与频繁重建假设。 |

---

## 三、 逐章扩充与新章建设详单（第 1～14 章）

> **编号对齐承诺**：以下各章的小节落点与新增练习/实验编号均已逐一核对 `manuscripts/01-*.md`～`12-*.md` 当前真实结构，确保零断号、零重号、前缀 100% 匹配。

### 第 1 章：初识 AI Infra（[`manuscripts/01-初识 AI Infra.md`](../../manuscripts/01-%E5%88%9D%E8%AF%86%20AI%20Infra.md)）
1. **扩充位置与内容**：
   - **`1.2.2 分析 AI 系统需要哪些关键数字`**：合入 [`logicfolding-energy.md`](../../case-studies/logicfolding-energy.md) 的背景引述与脚注链接，从芯片内部金属连线充放电解释“数据移动不仅发生在卡间和片外，片上路径长度同样消耗能量”。
   - **`1.3.4 实测为何达不到极限：模型漏项还是系统开销`**：合入 [`kernel-and-fleet-efficiency.md`](../../case-studies/kernel-and-fleet-efficiency.md) 的 **Google MLSys 2026 机队级有效产出分解框架（$\mathrm{MPG}=\mathrm{SG}\times\mathrm{RG}\times\mathrm{PG}$）**，以及本书基于该框架构造的 100-GPU 同型资源池教学算例（明确声明具体数字为本书教学算例而非论文实测值）。
   - **`1.4.1 CPU：控制流、系统调用与驱动调度` 与 `1.4.2 SmartNIC：网卡承担协议处理与数据搬移`**：在 `1.4.2` 已有的 $3\sim 6$ 核包转发算例处补挂 [`programmable-nic.md`](../../case-studies/programmable-nic.md) 脚注链接，并新增 [`host-policy-and-dispatch.md`](../../case-studies/host-policy-and-dispatch.md)（ASPLOS 2026 Wave）的 32 核主机策略卸载翻转分析。
2. **新增定量算例**：
   - **算例 1-A（100-GPU 资源池的机队效率连乘折损教学算例）**：总供给 $360{,}000\ \mathrm{GPU\cdot s}$，扣除碎片后 $\mathrm{SG}=83.33\%$，扣除抢占/故障重做后 $\mathrm{RG}=80\%$，结合单卡内核执行效率 $\mathrm{PG}=50\%$，得出最终机队有效产出 $\mathrm{MPG}=33.33\%$。
   - **算例 1-B（32 核主机将调度策略卸载至 SmartNIC 的吞吐翻转阈值）**：由 $32(1-d)/30 > 1$ 导出单核跨 PCIe 损耗率阈值 $d < 6.25\%$（$d=2\%$ 时净增 $4.53\%$，$d=8\%$ 时反降 $1.87\%$），以及 $0.75\ \mu\mathrm{s}$ 读取在 $1\ \mu\mathrm{s}$ vs. $0.2\ \mu\mathrm{s}$ 重叠窗口下的暴露等待。
3. **新增配图（`codex exec` + `gpt-image-2.6-sunburst` `quality="high"` + `figure_style` 420pt）**：
   - `ch01-fleet-goodput-and-smartnic-offload`：机队级全栈效率三层漏斗（$\mathrm{SG}\to\mathrm{RG}\to\mathrm{PG}$）与主机/SmartNIC 策略卸载翻转曲线图。
4. **新增练习（接原书最大编号 `练习 1-6`）**：
   - **`> **练习 1-7：机队全栈效率（MPG）分解与智算中心电费账单核算。**`**
   - **`> **练习 1-8：主机管理策略卸载至 SmartNIC 的吞吐翻转边界与预取隐藏条件。**`**

---

### 第 2 章：模型架构（[`manuscripts/02-模型架构.md`](../../manuscripts/02-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84.md)）
1. **扩充位置与内容**：
   - **`2.3.6 跨层共享与稀疏选择：从全层独立缓存到少量锚点层`**：在现有 CSA2 `890 B/token` 基础上，深化 [`research/deepseek-v41-codesign-2026-09-10/README.md`](../../research/deepseek-v41-codesign-2026-09-10/README.md) 中 V4 Flash（`3514.25 B/token` 全局 + SWA）到 V4.1 Flash（4 个 Owner 层保留全局 MLA KV + 36 层 SWA）在 $32\mathrm{K}$ 上下文下的完整容量对比，并补充 **非自回归扩散大语言模型（dLLM，如 LLaDA、Mercury）双向去噪注意力** 对前缀缓存不变性的挑战。
   - **`2.6.2 扩散 Transformer（DiT）：全序列双向注意力与多步去噪`**：合入 [`generative-multimodal-models.md`](../../case-studies/generative-multimodal-models.md)、[`research/generative-image-analysis/PLAN.md`](../../research/generative-image-analysis/PLAN.md) 与 [`calculations/results/image-generation-book.md`](../../calculations/results/image-generation-book.md) 的锁定真值，对比 `Qwen-Image-2512`（20B MMDiT，50 步双分支 CFG）与 `FLUX.2-klein-4B`（4B 蒸馏 DiT，4 步单分支）的骨干算力与 3D Causal VAE 解码激活峰值。
2. **新增定量算例**：
   - **算例 2-A（`Qwen-Image-2512` vs. `FLUX.2-klein-4B` 在 $1024\times 1024$ 下的算力与 VAE 激活账本）**：`Qwen-Image-2512` 骨干单分支单步 `78.304 TFLOPs`（50 步双分支 CFG 共 `7,830.39 TFLOPs` = `7.83 PFLOPs`），VAE 解码 `11.68 TFLOPs`，但 VAE 解码因 30 次因果卷积特征缓存克隆（`_feat_map`）驻留 `2,416,443,392 B`（`2.250 GiB` / `2.42 GB`，算子边界峰值 `2.263 GiB`），逼近 60 层骨干单层边界峰值（`3.162 GiB`）；而 `FLUX.2-klein-4B` 4 步蒸馏单分支仅 `139.28 TFLOPs`（单步 `34.820 TFLOPs`），VAE 解码占 `10.47 TFLOPs`（激活峰值 `248.38 MiB`）。
3. **新增配图（420pt）**：
   - `ch02-dit-vae-memory-and-dllm-attention`：自回归因果掩码 vs. dLLM 双向去噪掩码，以及 `Qwen-Image-2512` / `FLUX.2-klein-4B` 骨干与 VAE 算力/显存对比图。
4. **新增练习（接原书最大编号 `练习 2-11`）**：
   - **`> **练习 2-12：图像生成模型中 DiT 步数蒸馏与 3D Causal VAE 特征缓存显存峰值核算。**`**
   - **`> **练习 2-13：非自回归扩散语言模型（dLLM）双向去噪步数与自回归访存带宽墙的翻转条件。**`**

---

### 第 3 章：推理与训练负载（[`manuscripts/03-推理与训练负载.md`](../../manuscripts/03-%E6%8E%A8%E7%90%86%E4%B8%8E%E8%AE%AD%E7%BB%83%E8%B4%9F%E8%BD%BD.md)）
1. **扩充位置与内容**：
   - **`3.1.3 请求为什么在表面上相似、成本上却差十倍` 与 `3.2 长上下文、检索增强与条件计算`**：深化 [`retrieval-and-generation.md`](../../case-studies/retrieval-and-generation.md)（ASPLOS 2025 IKS）的锁定真值算例，揭示粗检索省下检索时间却因召回文档过多推高 GPU Prefill 耗时的反直觉定律。
   - **`3.3.2 音频与图像生成的计算过程` 与 `3.3.3 连续感知、首段响应与打断`**：补挂 [`queqiao.md`](../../case-studies/queqiao.md) 与 [`generative-multimodal-models.md`](../../case-studies/generative-multimodal-models.md) 脚注链接，深化 `Qwen3-Omni-30B-A3B`（Thinker/Talker/Code2Wav）与 `Fish Audio S2 Pro`（Dual-AR 慢/快双自回归）首包链路，并新增 **具身智能（Physical AI / VLA，如 OpenVLA、$\pi_0$）** 的“云端 $5\text{–}10\ \mathrm{Hz}$ 慢规划 + 端侧 $500\text{–}1000\ \mathrm{Hz}$ 实时控制”双环路负载特征。
2. **新增定量算例**：
   - **算例 3-A（RAG 精检索 8 篇 vs. 粗检索 32 篇的端到端延迟反转账本）**：基于 `Qwen3-8B`（`512` query + `8×128` vs. `32×128` 文档 token，总输入 `1,536` vs. `4,608` token，KV 增量 `216 MiB` vs. `648 MiB`，矩阵计算量 `22.033 TFLOPs` vs. `70.275 TFLOPs`），在 `200 TFLOP/s` 有效 Prefill 吞吐下，候选 A（检索 `200 ms` + Prefill `110.168 ms` = `310.168 ms`）快于候选 B（检索 `20 ms` + Prefill `351.378 ms` = `371.378 ms`，Prefill 暴涨 `241.210 ms` 反超检索省下的 `180 ms`，净慢 `61.210 ms`；翻转算力阈值为 `268.011 TFLOP/s`）。
   - **算例 3-B（具身智能 VLA 双环路频率配比与动作块 Action Chunking 时间预算）**。
3. **新增配图（420pt）**：
   - `ch03-rag-prefill-tradeoff-and-vla-loop`：RAG 检索耗时与 GPU Prefill 耗时叠加翻转图及具身智能 VLA 快慢双环路时序图。
4. **新增练习（接原书最大编号 `练习 3-12`）**：
   - **`> **练习 3-13：RAG 系统中向量检索召回量与 LLM Prefill 算力吞吐的翻转点推导。**`**
   - **`> **练习 3-14：具身智能（VLA）与双自回归（Dual-AR）流式语音的首包延迟与抖动预算。**`**

---

### 第 4 章：加速器架构（[`manuscripts/04-加速器架构.md`](../../manuscripts/04-%E5%8A%A0%E9%80%9F%E5%99%A8%E6%9E%B6%E6%9E%84.md)）
1. **扩充位置与内容（严格去重，不重复编写已有的 `4.1.3` 图 4-4 和 `4.2.3` `例 4-1` 图 4-11）**：
   - **`4.1.3 芯片面积、功耗与封装的限制`**：在现有 LogicFolding 段落补挂 [`logicfolding-energy.md`](../../case-studies/logicfolding-energy.md) 脚注链接，并补充 NPU 同性能降压算例（电压从 $0.85\ \mathrm{V}$ 降至 $0.55\ \mathrm{V}$ 时动态功耗比 $(0.55/0.85)^2\approx 0.42$）。
   - **`4.2.3 从矩阵乘到注意力核函数`**：在现有 FlashAttention-4 `例 4-1`（第 316–339 行）后补挂 [`kernel-and-fleet-efficiency.md`](../../case-studies/kernel-and-fleet-efficiency.md) 脚注链接，并补充 vLLM/SGLang 在 Hopper（FA3）与 Blackwell（FA4 / FlashInfer `trtllm-gen`）上的后端分派对照表。
   - **`4.3.1 容量、带宽与访问延迟` 与 `4.8.3`**：补挂 [`macbook-rtx-pro6000.md`](../../case-studies/macbook-rtx-pro6000.md) 脚注链接，补充 `4 GiB` 单步读取带宽下界表（M2 Pro `200 GB/s` 需 `21.47 ms`、M2 Max `400 GB/s` 需 `10.74 ms`、RTX PRO 6000 `1,792 GB/s` 需 `2.40 ms`，带宽比值 `4.48` 与 `8.96`）。
   - **`4.5.2 CPU 与加速器之间的数据访问`**：新增 **GPU 机密计算（Hopper H100 / Blackwell B200 Confidential Computing TEE）** 的硬件安全边界、CPU Bounce Buffer AES-GCM 加解密流水线，以及对短输入 TTFT 与长序列大 Batch TPOT 的非对称性能影响。
2. **新增定量算例**：
   - **算例 4-A（M2 Pro / M2 Max 统一内存 vs. RTX PRO 6000 独立显存的 `4 GiB` 单步读取下界与 PCIe 交接账本）**。
   - **算例 4-B（GPU TEE 机密计算模式下 CPU Bounce Buffer AES-GCM 加解密对 H2D/D2H 有效带宽与 TTFT/TPOT 的开销核算）**。
3. **新增配图（420pt）**：
   - `ch04-unified-memory-vs-gddr7-and-gpu-tee`：Apple 统一内存与独立 GPU GDDR7 数据路径对比及 GPU TEE 机密计算 Bounce Buffer 加密流水线图。
4. **新增实验（接原书最大编号 `实验 4-10`，严格保持 `> **实验 4-Y：` 格式）**：
   - **`> **实验 4-11：统一内存（M2 Max）与独立显存（RTX PRO 6000）在单步解码与跨设备交接下的带宽下界核验。**`**
   - **`> **实验 4-12：加速器机密计算（GPU TEE）开启前后主机—设备 DMA 吞吐与短请求 TTFT 附加开销建模。**`**

---

### 第 5 章：算子与运行时（[`manuscripts/05-算子与运行时.md`](../../manuscripts/05-%E7%AE%97%E5%AD%90%E4%B8%8E%E8%BF%90%E8%A1%8C%E6%97%B6.md)）
1. **扩充位置与内容**：
   - **`5.4.1 编译流程与中间表示` 与 `5.4.5 TileLang-Ascend`**：合入 [`macbook-rtx-pro6000.md`](../../case-studies/macbook-rtx-pro6000.md) 中 Ollama v0.20.7 `ggml` Metal 后端的统一内存缓冲区管理与算子融合边界，并横向对比 **TileLang / ThunderKittens / CuTe DSL** 的块级张量编程抽象。
   - **`5.5.2 CUDA Graph：提交收益与输入复制开销` 与 `5.5.3 动态形状与编译开销`**：在现有 FlashInfer 实测旁完整合入 [`cache-and-reconfiguration.md`](../../case-studies/cache-and-reconfiguration.md) 的 **`plan/run` 二阶段分离手算推导**（`Qwen3-8B` 36 层同构注意力共享一次 `plan` 耗时 $20\ \mu\mathrm{s}$ vs. 逐层重做 $720\ \mu\mathrm{s}$，以及 DeepSeek V4 / Kimi K3 混合不同类型层时需按层族维护多份计划的约束），并合入 [`weight-handoff.md`](../../case-studies/weight-handoff.md) 中 RL 引擎休眠（Sleep Mode）保留虚拟地址以维持 CUDA Graph 有效性的机制。
2. **新增定量算例**：
   - **算例 5-A（FlashInfer `plan/run` 跨层复用与混合模型多计划表开销核算）**：对比 36 层同构 GQA 共用 1 份计划（$20\ \mu\mathrm{s}$）与含全注意力 + 滑动窗口/线性注意力的异构模型分组生成计划的主机调度账本。
3. **新增配图（420pt）**：
   - `ch05-flashinfer-plan-run-and-sleep-vaddr`：FlashInfer `plan`（CPU 元数据写预留缓冲）与 `run`（CUDA Graph 固定地址执行）分离及 Sleep Mode 虚拟地址保留机制图。
4. **新增实验（接原书最大编号 `实验 5-11`）**：
   - **`> **实验 5-12：动态请求下注意力算子 `plan/run` 分离、多层计划复用与虚拟地址保留的耗时拆解。**`**

---

### 第 6 章：超节点（[`manuscripts/06-超节点.md`](../../manuscripts/06-%E8%B6%85%E8%8A%82%E7%82%B9.md)）
1. **扩充位置与内容**：
   - **`6.5.2 互联介质、供电与物理规模` 与 `6.5.3 NVIDIA：从八卡服务器到机柜`**：深入剖析 **铜缆背板极限（224G SerDes Copper Flyover $<2\ \mathrm{m}$）与共封装光学（CPO / Silicon Photonics，面向 NVL144/NVL576 高密机架）** 的功耗（$\mathrm{pJ/bit}$）与失效率权衡，以及单机柜功率从 $120\ \mathrm{kW}$ 跃升至 $500\text{–}600\ \mathrm{kW}$ 时为何必须从 $48\ \mathrm{V}/54\ \mathrm{V}$ 升级至 **$800\ \mathrm{VDC}$ 高压直流供电**。
   - **`6.6 超节点内的异构协同与内存池化`**：完整合入 [`memory-criticality-and-tiering.md`](../../case-studies/memory-criticality-and-tiering.md)（ASPLOS 2026 PACT 与 Camp）的 `Qwen3-8B` $288\ \mathrm{MiB}$ FFN 权重借用算例，并在 `6.6.3` 引入 [`profile-and-plan-ranking.md`](../../case-studies/profile-and-plan-ranking.md) 的候选并行方案初筛原则。
2. **新增定量算例**：
   - **算例 6-A（分层内存源端带宽、H2D 瓶颈、预取提前量与页迁移回本阈值）**：核算 $288\ \mathrm{MiB}$ 权重在源端 $100/40/8\ \mathrm{GiB/s}$ 与 H2D $24\ \mathrm{GiB/s}$ 下的下界（$11.72\ \mathrm{ms}$ vs. $35.16\ \mathrm{ms}$），在 $16\ \mathrm{ms}$ 预取提前量下 $8\ \mathrm{GiB/s}$ 源端仍暴露 $19.16\ \mathrm{ms}$ 等待，单次页迁移（$35.16\ \mathrm{ms}$）需未来至少 2 次关键路径使用（省 $38.31\ \mathrm{ms}$）才能回本；若提前量达 $40\ \mathrm{ms}$ 则迁移无可见时间收益。
   - **算例 6-B（$500\ \mathrm{kW}$ 超节点机柜在 $50\ \mathrm{V}$ vs. $800\ \mathrm{VDC}$ 供电下的母排电流 $10{,}000\ \mathrm{A}\to 625\ \mathrm{A}$ 与 $I^2R$ 损耗缩减 $256$ 倍账本）**。
3. **新增配图（420pt）**：
   - `ch06-memory-tiering-prefetch-and-800vdc-cpo`：分层内存预取提前量与暴露等待时间线，以及 $800\ \mathrm{VDC}$ + CPO 超节点机柜物理架构图。
4. **新增实验（接原书最大编号 `实验 6-13`）**：
   - **`> **实验 6-14：分层内存中预取提前量、暴露停顿与冷热页迁移（Page Promotion）的净收益边界。**`**
   - **`> **实验 6-15：高功率超节点机柜的 800 VDC 高压供电母排损耗与光电互连能耗核算。**`**

---

### 第 7 章：数据中心网络（[`manuscripts/07-数据中心网络.md`](../../manuscripts/07-%E6%95%B0%E6%8D%AE%E4%B8%AD%E5%BF%83%E7%BD%91%E7%BB%9C.md)）
1. **扩充位置与内容**：
   - **`7.3.4 案例：KV-Direct 如何绕过主机内存转发`**：在现有 64 tags / `60 MOPS` 算例（第 583–588 行及图 7-28）处补挂 [`programmable-nic.md`](../../case-studies/programmable-nic.md) 脚注链接，并补充 ClickNP FPGA 元素级空间流水线与 CPU 核处理的延迟对比。
   - **`7.5.3 多路径与重传`**：补充万卡集群光模块温度漂移导致的间歇性微闪断（Link Flapping）检测，以及 UEC 数据包喷洒（Packet Spraying）与选择性重传（SACK）相比传统 RoCE Go-Back-N 的抗丢包收益。
   - **`7.6.2 跨节点分层集合通信` 与 `7.6.4 集合通信的耗时建模与实测校准`**：深化机架内 NVLS（NVLink SHARP）网内规约与跨机架 Rail-Optimized 多轨并发规约的两级时间建模。
2. **新增定量算例**：
   - **算例 7-A（1,024 卡集群上扁平 Ring vs. 机架内 NVLS + 跨机架多轨 AllReduce 同步 $16\ \mathrm{GiB}$ 梯度的分阶段耗时账本）**。
3. **新增配图（420pt）**：
   - `ch07-nvls-multirail-allreduce-and-sack`：机架内 NVLS 网内规约 + 跨机架多轨并行 AllReduce 数据流与微闪断下 SACK vs. Go-Back-N 吞吐恢复对比图。
4. **新增实验（接原书最大编号 `实验 7-17`）**：
   - **`> **实验 7-18：万卡链路微闪断（$10^{-4}$ 丢包率）下 Go-Back-N 与选择性重传（SACK）的有效带宽校准。**`**

---

### 第 8 章：推理优化（[`manuscripts/08-推理优化.md`](../../manuscripts/08-%E6%8E%A8%E7%90%86%E4%BC%98%E5%8C%96.md)）
1. **扩充位置与内容**：
   - **`8.2.4 采样、提前停止与结构化解码开销` 与 `8.5.1 投机解码`**：完整合入 [`structured-generation.md`](../../case-studies/structured-generation.md)（MLSys 2025 **XGrammar**）的全部锁定真值算例（32-bit 位图掩码大小、H2D 传输耗时、GPU logits 读写耗时、CPU 64 路掩码生成与 GPU 前向的流水重叠，以及投机解码 320 行多草稿掩码与拒绝时的下推自动机栈回滚），并在 `8.5` 补充 **扩散语言模型（`Fast-dLLM`）分块近似 KV 缓存与基于置信度阈值的并行解码**。
   - **`8.3.3 对话与智能体的前缀复用` 与 `8.3.4 缓存准入、淘汰、换出与重计算`**：补入 [`cache-and-reconfiguration.md`](../../case-studies/cache-and-reconfiguration.md) 中 **Marconi** 混合 SSM/Attention 模型检查点准入（纯输入公共前缀二次出现识别保存、三次出现受益）与 `FLOPs/byte` 淘汰权重。
   - **`8.4.4 稀疏与量化结合时的误差累积`**：完整合入 [`cache-and-reconfiguration.md`](../../case-studies/cache-and-reconfiguration.md) 中的 **Rethinking KV Cache Compression（有损压缩使 decode 速率提高 $20\%$ 却因输出长度膨胀 $50\%$ 导致整次请求多花 $22\%$ 时间）**。
2. **新增定量算例（严格采用 `case-studies/` 锁定真值）**：
   - **算例 8-A（XGrammar 词表位图掩码传输、Logits 访存与 CPU-GPU 流水重叠账本）**：`Qwen3-8B`（词表 $151{,}936$，$B=64$）32-bit 位图每行 $\lceil 151936/32\rceil\times 4 = 18{,}992\ \mathrm{B}$，$B=64$ 共 $1{,}215{,}488\ \mathrm{B}$（$1.159\ \mathrm{MiB}$），在 $25\ \mathrm{GB/s}$ H2D 链路上仅需 $48.62\ \mu\mathrm{s}$；GPU 侧一读一写 FP32 logits `[64, 151936]` 共 $77{,}791{,}232\ \mathrm{B}$（$74.188\ \mathrm{MiB}$），在 $1\ \mathrm{TB/s}$ 显存带宽下需 $77.79\ \mu\mathrm{s}$；CPU 单请求掩码生成 $50\ \mu\mathrm{s}$、64 路串行 $3.2\ \mathrm{ms}$，GPU 前向 $2\ \mathrm{ms}$，串行总步时 $5.2\ \mathrm{ms}$，CPU/GPU 跨步流水重叠后降至 $\max(3.2, 2.0)=3.2\ \mathrm{ms}$，将 CPU 掩码并行化压至 $<2\ \mathrm{ms}$ 后步时回到 $2.0\ \mathrm{ms}$；投机解码（4 草稿 + 1 bonus，320 行）位图增至 $6{,}077{,}440\ \mathrm{B}$（$5.796\ \mathrm{MiB}$）。
   - **算例 8-B（KV 缓存压缩后单步提速但输出变长导致端到端耗时反转）**：设 $\mathrm{TTFT}=0.5\ \mathrm{s}$，原配置生成 $200\ \mathrm{tok}$、速率 $50\ \mathrm{tok/s}$，总耗时 $0.5 + 200/50 = 4.5\ \mathrm{s}$；压缩后速率升至 $60\ \mathrm{tok/s}$（快 $20\%$），但生成长度增至 $300\ \mathrm{tok}$（长 $50\%$），总耗时变为 $0.5 + 300/60 = 5.5\ \mathrm{s}$（反慢约 $22\%$）。
3. **新增配图（420pt）**：
   - `ch08-xgrammar-overlap-and-kv-compression-reversal`：XGrammar 位图掩码 CPU/GPU 重叠与投机回滚栈（左）及 KV 压缩下长度膨胀导致完成时间反转曲线（右）。
4. **新增练习（接原书最大编号 `练习 8-11`）**：
   - **`> **练习 8-12：结构化生成（XGrammar）位图掩码传输、CPU/GPU 重叠与投机解码多草稿掩码核算。**`**
   - **`> **练习 8-13：有损 KV 压缩下的输出长度膨胀反转判据与混合模型（Marconi）检查点准入分析。**`**

---

### 第 9 章：分布式推理（[`manuscripts/09-分布式推理.md`](../../manuscripts/09-%E5%88%86%E5%B8%83%E5%BC%8F%E6%8E%A8%E7%90%86.md)）
1. **扩充位置与内容**：
   - **`9.2.2 KV 传输与两端内存占用` 与 `9.5.3 持久化、检查点与部分重算`**：深化 [`research/deepseek-v41-codesign-2026-09-10/README.md`](../../research/deepseek-v41-codesign-2026-09-10/README.md) 中 **SWA 128-token 局部重算替代跨网 KV 传输** 的定量边界（在 PD 分离下，36 个非 Owner SWA 层无需跨网传输 `2.578 MiB` KV 状态，只需传末尾 128 个 Token ID 并在目标端有界重算）。
   - **`9.2.3 预算匹配与木桶效应` 与 `9.3 Attention-FFN 分离`**：完整合入 [`pd-af-heterogeneous.md`](../../case-studies/pd-af-heterogeneous.md)，对比 **A100 80GB SXM Prefill + H20 96GB Decode** 阶段切分与 **RTX 4090 + 双路 Xeon Gold 6454S（KTransformers）** 层内 Attention-FFN 异构切分的交接频率与带宽账本。
2. **新增定量算例**：
   - **算例 9-A（A100 Prefill + H20 Decode 跨池 $1\ \mathrm{GiB}$ KV 传输下界 vs. KTransformers 逐层 AF 交接）**：32 层 GQA 模型（$S=8{,}192, H_{kv}=8, d=128, b=2$）单请求产生 $1\ \mathrm{GiB}$ KV，在 $25\ \mathrm{GB/s}$ 跨池链路上仅有效载荷传输即需 $42.95\ \mathrm{ms}$；对比 AF 分离在每生成步跨 36 层发生 72 次细粒度激活与专家结果交接的延迟敏感性。
3. **新增配图（420pt）**：
   - `ch09-pd-a100-h20-vs-af-ktransformers`：A100/H20 异构 PD 阶段切分与 KTransformers GPU/CPU 层内 AF 切分的数据交接路径与频率对比图。
4. **新增练习（接原书最大编号 `练习 9-13`）**：
   - **`> **练习 9-14：A100 Prefill 与 H20 Decode 异构池的 KV 传输下界、按层流水重叠与木桶配比计算。**`**
   - **`> **练习 9-15：PD 分离下滑动窗口层（SWA）“跨网传输 KV”与“仅传末尾 Token 本地重算”的盈亏临界点。**`**

---

### 第 10 章：训练系统（[`manuscripts/10-训练系统.md`](../../manuscripts/10-%E8%AE%AD%E7%BB%83%E7%B3%BB%E7%BB%9F.md)）
1. **扩充位置与内容**：
   - **`10.4 容错、检查点与长稳运行`**：新增 **万卡集群静默数据损坏（Silent Data Corruption, SDC / 亚健康“幽灵坏卡”）** 的物理成因、AllReduce 梯度污染扩散机制与算子级校验和/确定性微批次重放隔离方案。
   - **`10.5.2 阶段配比、资源复用与权重同步` 与 `10.5.3 长尾、中断与可复现性`**：完整合入 3 篇 RL 训练系统深度案例：
     1. [`weight-handoff.md`](../../case-studies/weight-handoff.md)（`Qwen3-8B` 训推显存有序交接峰值 $59.2564\ \mathrm{GiB}$ 避免 $83.2564\ \mathrm{GiB}$ OOM；`Qwen3-235B` 路由专家 $423\ \mathrm{GiB}$ 在 EP16 下按需接收 $26.4375\ \mathrm{GiB}$、8 生产端出口下界 $2.27\ \mathrm{s}$，以及 Embedding $1.1592\ \mathrm{GiB}$ 最大原子切片所需的 $2.3184\ \mathrm{GiB}$ 双缓冲槽）；
     2. [`rollout-tail-and-sampling.md`](../../case-studies/rollout-tail-and-sampling.md)（NSDI 2026 **RollPacker**：超额采样 $160\times 10=1600$ 抢先完成导致长回答保留概率从 $1/2$ 降为 $1/4$ 的分布偏移反例；Stream Trainer 腾出半数卡转训练的 KV 容量检查 $36\ \mathrm{GiB}$ vs. $72\ \mathrm{GiB}$ 与时间线 $60\ \mathrm{s}\to 55\ \mathrm{s}$ vs. $62\ \mathrm{s}$；非均匀微批梯度归一化 $(1\times 1+3\times 3)/4=2.5 \neq 2.0$ 及梯度裁剪次序反例）；
     3. [`rl-scheduling-and-recovery.md`](../../case-studies/rl-scheduling-and-recovery.md)（OSDI 2026 **RobustRL**：trainer 在更新至 $30\ \mathrm{s}$ 时故障，整轮重做需 $790\ \mathrm{s}$，保留已生成 rollout 轨迹仅重做更新步降至 $490\ \mathrm{s}$，省下 $300\ \mathrm{s}$）。
   - **`10.6.1 性能模型与配置选择`**：合入 [`profile-and-plan-ranking.md`](../../case-studies/profile-and-plan-ranking.md)（MICRO 2024 **vTrain** 通信争用膨胀系数 $k > 10/9 \approx 1.11$ 导致的候选方案 A/B 排序反转）与 [`kernel-and-fleet-efficiency.md`](../../case-studies/kernel-and-fleet-efficiency.md) 的机队级 $\mathrm{MPG}$ 视角。
2. **新增定量算例**：
   - **算例 10-A（`Qwen3-8B` 训推显存有序交接与 `Qwen3-235B` EP16 权重分发及双缓冲下界账本）**。
   - **算例 10-B（RollPacker 流式训练半数卡切换时间线、采样分布偏移反例与非均匀微批梯度归一化核算）**。
   - **算例 10-C（RobustRL 角色解耦故障恢复 $790\ \mathrm{s}\to 490\ \mathrm{s}$ 与 vTrain $k=10/9$ 配置排序反转推算）**。
3. **新增配图（420pt）**：
   - `ch10-rl-weight-handoff-and-stream-trainer`：`Qwen3-8B` 有序显存交接峰值控制（上）、`RollPacker` Stream Trainer 动态卡池切换与 `RobustRL` 角色解耦恢复时间线（下）。
4. **新增习题（接原书最大编号 `习题 10-12`，严格保持 `> **习题 10-Y · ...**` 格式）**：
   - **`> **习题 10-13 · 综合分析：RL 训练中超额采样分布偏移、流式卡池切换 KV 预算与非均匀微批梯度归约**`**
   - **`> **习题 10-14 · 分析：235B MoE 权重分片分发（Sharded RDT）的生产端/接收端带宽下界与原子切片双缓冲设计**`**

---

### 第 11 章：资源调度与运行环境（[`manuscripts/11-资源调度与运行环境.md`](../../manuscripts/11-%E8%B5%84%E6%BA%90%E8%B0%83%E5%BA%A6%E4%B8%8E%E8%BF%90%E8%A1%8C%E7%8E%AF%E5%A2%83.md)）
1. **扩充位置与内容**：
   - **`11.1.2 并发环境、驻留容量与外部瓶颈`**：合入 [`host-policy-and-dispatch.md`](../../case-studies/host-policy-and-dispatch.md) 的完整平台预算（每任务驻留 $30\ \mathrm{s}$、占 $2\ \mathrm{GiB}$、耗 $3\ \text{核秒}$：在 $600\ \mathrm{GiB}$ 环境内存下吞吐被锁死在 $10\ \text{项/s}$，将环境内存扩至 $720\ \mathrm{GiB}$ 后 SmartNIC 释放的 CPU 核才将吞吐推至 $10.453\ \text{项/s}$）。
   - **`11.2.1 工具调用与执行边界`、`11.2.2 沙箱生命周期` 与 `11.2.3 快照、缓存与按需恢复`**：补挂 [`structured-generation.md`](../../case-studies/structured-generation.md) 工具参数语法约束调用链，并完整合入 [`snapshot-residency-and-first-use.md`](../../case-studies/snapshot-residency-and-first-use.md)（E2B Firecracker `UFFDIO_COPY` 按需缺页 vs. ASPLOS 2025 **CXLfork** 共享只读页的四种内存放置预算，以及 64 个离散 $4\ \mathrm{KiB}$ 页在 $2\ \mathrm{MiB}$ 大页下的 $512\times$ 读放大）。
   - **`11.3.1 弹性资源分配`、`11.3.3 多作业交错` 与 `11.3.4 验证期限与奖励过滤`**：合入 [`cache-and-reconfiguration.md`](../../case-studies/cache-and-reconfiguration.md)（MLSys 2025 **Rubick** 弹性训练重配 $>400$ 步回本阈值）、[`rl-scheduling-and-recovery.md`](../../case-studies/rl-scheduling-and-recovery.md)（OSDI 2026 **Weave** 双 RL 作业互补交错与主机 $1050\ \mathrm{GiB} > 1\ \mathrm{TiB}$ 内存驻留陷阱）以及 [`rollout-tail-and-sampling.md`](../../case-studies/rollout-tail-and-sampling.md)（RollPacker 代码验证自适应超时 $\min(\max(2\mathrm{s}, 1.5\times t_{\mathrm{anchor}}), 30\mathrm{s})$ 的误杀率权衡）。
   - **`11.4.4 容量采购与按量调用`**：合入 [`single-agent-serving-cost.md`](../../case-studies/single-agent-serving-cost.md)（24×7 数字员工在 $200\mathrm{K}/1\mathrm{M}$ 上下文下的自建 GPU 池 vs. 官方 API 成本核算）。
2. **新增定量算例**：
   - **算例 11-A（100 个派生自 $2\ \mathrm{GiB}$ 模板的沙箱在四种内存放置下的容量上界与大页读放大账本）**：全量复制（$2{,}052\ \mathrm{MiB}/\text{环境}$，100 个占 $200.39\ \mathrm{GiB}$）、按需复制（$516\ \mathrm{MiB}$，占 $50.39\ \mathrm{GiB}$）、CXL 只读共享（$132\ \mathrm{MiB}$，占 $12.89\ \mathrm{GiB}$）与热点本地化（$388\ \mathrm{MiB}$，占 $37.89\ \mathrm{GiB}$）在 $64\ \mathrm{GiB}$ 本地预算下的并发上限（31 / 127 / 496 / 168 个）；以及 64 个离散 $4\ \mathrm{KiB}$ 页按小页取（$256\ \mathrm{KiB}$，$0.244\ \mathrm{ms}$）vs. 按 $2\ \mathrm{MiB}$ 大页取（$128\ \mathrm{MiB}$，$125\ \mathrm{ms}$）。
   - **算例 11-B（Weave 互补 RL 作业共享周期 $480\ \mathrm{s}/490\ \mathrm{s}/570\ \mathrm{s}$ 与 Rubick $>400$ 步重配回本阈值核算）**。
3. **新增配图（420pt）**：
   - `ch11-sandbox-cxlfork-and-weave-gantt`：沙箱快照四种内存放置容量对比（上）与 Weave 跨作业互补交错及主机驻留上限图（下）。
4. **新增实验（接原书最大编号 `实验 11-12`）**：
   - **`> **实验 11-13：沙箱环境快照按需缺页（Demand Paging）、CXL 只读共享与大页读放大的容量及首次访问延迟核算。**`**
   - **`> **实验 11-14：多作业 RL 互补交错（Weave）的主机内存驻留验证与弹性重配（Rubick）切换步数阈值推算。**`**

---

### 第 12 章：端边云协同（[`manuscripts/12-端边云协同.md`](../../manuscripts/12-%E7%AB%AF%E8%BE%B9%E4%BA%91%E5%8D%8F%E5%90%8C.md)）
1. **扩充位置与内容**：
   - **`12.1.5 端侧执行资源与本地部署`**：合入 [`macbook-rtx-pro6000.md`](../../case-studies/macbook-rtx-pro6000.md) 的 Ollama Generate API 计时口径拆解（`load_duration`、`prompt_eval_duration`、`eval_duration` 与端到端墙钟时间差额），并新增 **端侧闪存流式按需加载（LLM in a Flash / UFS 4.0 共激活神经元捆绑读取）**。
   - **`12.3.5 案例：Queqiao 如何加速跨地域语音服务`**：在现有 Queqiao 专节补挂 [`queqiao.md`](../../case-studies/queqiao.md) 脚注链接，并显式补入三行十进制预算算式表（理想串行发送 $8.5\ \mathrm{ms}$、$\mathrm{BDP}\approx 8.3\ \mathrm{MB}$、简化预算 $239\ \mathrm{ms}$）。
   - **`12.4 端边云协同架构设计`**：新增 **具身智能（Embodied VLA）车载/机器人端边云双环路协同**——端侧 Orin/Thor 芯片运行 $1000\ \mathrm{Hz}$ 实时控制与动作块缓存（Action Chunking），云端运行 $5\ \mathrm{Hz}$ 视觉语言重规划，抵抗无线弱网 P99 抖动。
2. **新增定量算例**：
   - **算例 12-A（移动端 UFS 4.0 闪存随机读 IOPS 墙与共激活神经元捆绑读取账本）**：对比按单神经元行（$2\ \mathrm{KiB}$ 随机读，有效带宽 $400\ \mathrm{MB/s}$）与按共激活分块捆绑（$64\ \mathrm{KiB}$ 连续块读，有效带宽 $3.2\ \mathrm{GB/s}$）流式加载稀疏 FFN 权重时的 $8\times$ 延迟缩减。
   - **算例 12-B（具身机器人端边云动作块缓存 Action Chunking 的最大断网容忍时间窗口计算）**。
3. **新增配图（420pt）**：
   - `ch12-ufs-flash-streaming-and-vla-chunking`：移动端 UFS 4.0 共激活神经元捆绑读取布局（左）与具身机器人端云双环路动作块滑窗容错图（右）。
4. **新增练习（接原书最大编号 `**拓展练习 12-12`，严格保持第 12 章 `**拓展练习 12-Y　...**` 格式）**：
   - **`**拓展练习 12-13　端侧 UFS 4.0 闪存流式加载稀疏大模型时的块粒度、预取窗口与 DRAM 缓冲池设计〔延伸，★★〕。**`**
   - **`**拓展练习 12-14　弱网高抖动下具身智能（VLA）端边云双环路控制的动作块长度与安全降级阈值推导〔延伸，★★〕。**`**

---

### 新增第 13 章：AI 原生存储系统与数据底座（[`manuscripts/13-AI 原生存储系统与数据底座.md`](../../manuscripts/13-AI%20%E5%8E%9F%E7%94%9F%E5%AD%98%E5%82%A8%E7%B3%BB%E7%BB%9F%E4%B8%8E%E6%95%B0%E6%8D%AE%E5%BA%95%E5%BA%A7.md)）
> **章节定位与去重边界**：通过显式交叉引用承接第 9.5.2 节（KV 多级缓存容量与 SSD DWPD 寿命估算）和第 10.4.3 节（异步 Checkpoint 与 Young–Daly 公式），避免重复推导容量与最佳保存间隔公式，将重心完全下沉到前 12 章未展开的 **存储协议栈、内核旁路 DMA 通路、AI 原生分布式文件系统（3FS/CRAQ）与 NAND Flash 物理介质读写放大治理**。

1. **完整章节大纲（H2 / H3）**：
   - **13.1 从算力墙到 I/O 墙：AI 负载的四种存储访问模式**
     - 13.1.1 预训练小文件随机读、Checkpoint 突发聚合写、KV 块随机读写与 Serverless 权重流式加载
     - 13.1.2 存储介质金字塔（HBM $\to$ DRAM/CXL $\to$ PCIe Gen5 NVMe SSD $\to$ 对象存储）的延迟、IOPS 与带宽账本
   - **13.2 GPUDirect Storage (GDS) 与内核旁路 I/O 协议栈**
     - 13.2.1 传统 POSIX 页缓存（Page Cache）路径的双重拷贝、CPU 中转与 PCIe Root Complex 带宽翻倍瓶颈
     - 13.2.2 GDS P2P DMA 零拷贝路径：`NVMe SSD -> PCIe Switch -> GPU BAR1` 的直接传输与队列深度要求
     - 13.2.3 用户态异步 I/O：`io_uring`、SPDK 轮询驱动与 RDMA USRBIO 零拷贝网卡—磁盘直通
   - **13.3 AI 原生并行文件系统：以 DeepSeek 3FS 为例**
     - 13.3.1 为什么传统 Lustre/POSIX 在万卡随机采样与 KVCache 卸载下遭遇元数据锁与长尾延迟瓶颈
     - 13.3.2 **CRAQ（Chain Replication with Apportioned Queries）** 链式复制：如何在保证强一致性的同时将读吞吐线性扩展至全部副本节点
     - 13.3.3 180 节点存储集群实现 `6.6 TiB/s` 聚合读带宽与 `3.66 TiB/min` KVCache 吞吐的网卡—磁盘配比算例
   - **13.4 NVMe SSD 物理特性、读写放大与 KV Cache 存储布局（Mooncake / LMCache）**
     - 13.4.1 NAND Flash 页（$4\text{–}16\ \mathrm{KiB}$ LBA）与擦除块（Block）物理特性：细粒度随机拉取 KV 导致的 IOPS 墙与读放大（RAF）
     - 13.4.2 Layer-major（按层连续）vs. Block-major（按 Token 块连续）布局在 NVMe SSD 上的传输效率翻转条件
     - 13.4.3 垃圾回收（GC）引发的写放大（WAF）与 P99 尾延迟毛刺，以及 ZNS（分区命名空间）SSD 顺序追加优化
   - **13.5 多模态 PB 级训练数据预处理与流式供给流水线**
     - 13.5.1 Tar/WebDataset/Parquet 分片打包消除千万级小文件 `open()/stat()` 元数据风暴
     - 13.5.2 视频与高分辨率图文训练中主机 CPU 解码瓶颈、GPU NVDEC/DALI 硬件解码卸载与预取缓冲池设计
   - **常见误区**（5 条高频 AI 存储系统认知误区辨析）
   - **本章小结**（严格作为全章最后一个二级标题）
2. **核心定量算例（4 个）**：
   - **算例 13-1**：传统 POSIX Page Cache 路径 vs. GDS P2P DMA 路径在单机 8 张 PCIe Gen5 NVMe SSD（单盘 $14\ \mathrm{GB/s}$）加载 $140\ \mathrm{GB}$ 权重时的 PCIe Root Complex 流量减半与耗时对比。
   - **算例 13-2**：DeepSeek 3FS 180 节点存储集群（每节点 $2\times 200\ \mathrm{Gbps}$ 网卡 + 16 块 NVMe SSD）的网卡—磁盘带宽匹配与 CRAQ 3 副本读写吞吐推导。
   - **算例 13-3**：32 层模型在卸载 `16 tokens/block` 的 KV 块时，Layer-major 布局（32 次离散小 I/O，单次 $<4\ \mathrm{KiB}$ 触发读放大）与聚合 Block-major 布局（$256\ \mathrm{KiB}$ 连续块）在 NVMe SSD 上的实际耗时对比（IOPS 受限 vs. 带宽受限）。
   - **算例 13-4**：视频 DiT 训练中每秒消费 $4{,}096$ 段 $720\mathrm{p}$ 视频片段时的主机 CPU 软解码核数赤字与 GPU 硬件解码卸载配比。
3. **配套配图（6 幅，`codex exec` + `gpt-image-2.6-sunburst` `quality="high"` + `figure_style` 420pt，存入 `manuscripts/ch13/`）**：
   - `ch13-storage-hierarchy-and-access-patterns`（图 13-1：AI 四类负载的存储介质金字塔与访问特征）
   - `ch13-posix-pagecache-vs-gds-dma`（图 13-2：传统 Page Cache 双重拷贝与 GPUDirect Storage DMA 直通对比）
   - `ch13-3fs-craq-architecture`（图 13-3：DeepSeek 3FS 架构与 CRAQ 链式复制读写路径）
   - `ch13-kv-layout-read-amplification`（图 13-4：KV Cache 在 NVMe SSD 上的 Layer-major vs. Block-major 读放大对比）
   - `ch13-zns-gc-tail-latency`（图 13-5：SSD 垃圾回收写放大毛刺与 ZNS 顺序块生命周期对齐）
   - `ch13-multimodal-dataloader-pipeline`（图 13-6：PB 级多模态训练数据分片打包与 CPU/GPU 协同解码流水线）
4. **配套练习与实验（6 道，严格使用 `> **练习 13-Y：...**` / `> **实验 13-6：...**` 格式以兼容 `verify_core_principles.py`）**：
   - `> **练习 13-1：GPUDirect Storage (GDS) 与 PCIe 拓扑带宽核算。**`
   - `> **练习 13-2：CRAQ 链式复制在节点故障与写密集突发期间的读写吞吐推导。**`
   - `> **练习 13-3：KV 缓存 NVMe SSD 卸载的块大小、读放大系数（RAF）与 IOPS/带宽翻转点。**`
   - `> **练习 13-4：ZNS SSD 与传统块 SSD 在高频 Checkpoint 覆盖写下的写放大（WAF）与寿命对比。**`
   - `> **练习 13-5：海量小文件合并分片（WebDataset/Tar）对元数据服务器 IOPS 与寻道时间的缩减倍率。**`
   - `> **实验 13-6：测量不同 I/O 块大小与队列深度下直接 I/O（`O_DIRECT`）与页缓存的吞吐拐点及尾延迟。**`

---

### 新增第 14 章：模型与系统硬件协同设计（[`manuscripts/14-模型与系统硬件协同设计.md`](../../manuscripts/14-%E6%A8%A1%E5%9E%8B%E4%B8%8E%E7%B3%BB%E7%BB%9F%E7%A1%AC%E4%BB%B6%E5%8D%8F%E5%90%8C%E8%AE%BE%E8%AE%A1.md)）
> **章节定位与去重边界**：由于第 2、4、6、8、9 章已分步介绍了 CSA2 `890 B/token`（`2.3.6`）、CED 早期丢弃（`2.6`/`3.2`）、SWA 128-token 重放（`8.3.3`）与 OpenTallas `$20M` ROI（`4.8.4`），本章**绝不将上述概念当成初次登场的知识点重复介绍**，而是定位为**“跨层联立方程组、长上下文双轨演进与反事实推演（Counterfactual Co-Design）”终章**，专门落地 [`research/deepseek-v41-codesign-2026-09-10/README.md`](../../research/deepseek-v41-codesign-2026-09-10/README.md) 第 3.2、4、5 节、[`case-studies/single-agent-serving-cost.md`](../../case-studies/single-agent-serving-cost.md)（含 `OPTIMIZATION-AUDIT.md`）与 [`case-studies/opentallas.md`](../../case-studies/opentallas.md) Case B/D/E 中尚未入书的深水区推导。

1. **完整章节大纲（H2 / H3）**：
   - **14.1 协同设计的联立方程：当硬件边界成为模型结构的正则项**
     - 14.1.1 算存比、微缩块粒度（`MXFP4` 32 元素块）、片上特殊函数单元（SFU）配比与超节点互连半径的跨层耦合
     - 14.1.2 协同设计的四大定量杠杆：`RoPE-post` 数值分布整形、跨层状态锚点化、有界重算替代跨网传输、前向计算重分布
   - **14.2 案例深剖一：DeepSeek V4 $\to$ V4.1 Flash 的长上下文双轨演进（$8\mathrm{K}\to 1\mathrm{M}$）**
     - 14.2.1 **驻留容量 vs. 单步 Decode 逻辑读取量的分离演进**：完整展开 [`deepseek-v41-codesign`](../../research/deepseek-v41-codesign-2026-09-10/README.md) 表 3.2，剖析为何 $8\mathrm{K}\to 128\mathrm{K}\to 200\mathrm{K}\to 1\mathrm{M}$ 下驻留 KV 容量恒定缩小 $3.95\times$（如 $32\mathrm{K}$ 下 V4 全局 KV `112.46 MiB` vs. V4.1 Flash 4-Owner `27.8125 MiB`），而单步 Decode 逻辑读取字节却随上下文拉长从缩小 $1.10\times$ 演进至缩小 $2.51\times$
     - 14.2.2 **四阶段注意力执行流水线（`Full -> 16,384 候选池 -> Reindex -> Reuse`）**：核算 `64 B/token` 索引扫描、`16,384` 候选池截断与 36 个 Follower 层直接复用 Top-$k$ 索引如何消去重复检索
     - 14.2.3 **持久化缓存 $1/8$ 压缩拆解（$\frac{1}{2}\times\frac{1}{4}$）**：$\frac{1}{2}$ 来自用 128-token 有界重算免除原本占约一半持久化缓存的 40 层 SWA 状态（`2.578 MiB`），$\frac{1}{4}$ 来自 CSA2 将全局 KV 从 `3514.25 B/token` 压至 `890 B/token`
   - **14.3 案例深剖二：V4.1 Flash 与 Kimi K3 在 $200\mathrm{K}/1\mathrm{M}$ 数字员工场景下的跨代硬件对决**
     - 14.3.1 完整落地 [`single-agent-serving-cost.md`](../../case-studies/single-agent-serving-cost.md)：对比 V4.1 Flash（`TP2/4/8`，状态 $1.30\to 6.42\ \mathrm{GiB}$）与 Kimi K3（`69 KDA + 24 MLA`，`PP2×TP8 + DCP8`，状态 $5.57\to 26.17\ \mathrm{GiB}$）在 H100、H200、B200、B300 四代硬件上的单流吞吐、周期性前缀重建成本与官方 API 价格交点
     - 14.3.2 **性能归因审计（`OPTIMIZATION-AUDIT.md`）**：揭示为何 K3 在未做分桶 CUDA Graph 与 MTP/DSpark 优化的基线中，约七成估算耗时来自逐层固定调度开销与频繁重建假设，演示如何区分“模型架构劣势”与“运行时未优化伪影”
   - **14.4 案例深剖三：专用推理 ASIC（OpenTallas）的四资源账本与晶圆级通信直径审计**
     - 14.4.1 **单步 `124.221 μs`（`8,050 tok/s`）的四资源时间账本（[`opentallas.md`](../../case-studies/opentallas.md) Case B）**：拆解 ROM 权重读取（`7.828 μs`）、HBM KV 读取（`41.979 μs`）、Tensor 计算（`25.932 μs`）、串行集合通信（`69.821 μs`）以及未被计价的辅助算子剩余预算（仅 `14.568 μs`）
     - 14.4.2 **晶圆级扩展的通信直径反超（Case D）与等面积公平性审计（Case E/F）**：核算 `15 跳 × 100 ns/跳 × 120 层 = 180 μs` 跨芯片跳数延迟如何单凭通信就击穿 `124.221 μs` 整步预算，并衔接 `$20M` 增量 NRE 在 `40 万亿 token`（254 台跑 1 年 vs. 半年模型寿命需 508 台）下的商业边界
   - **14.5 四个反事实思想实验与全书五步定量诊断总纲**
     - 14.5.1 **四个反事实推演（[`deepseek-v41-codesign`](../../research/deepseek-v41-codesign-2026-09-10/README.md) §5）**：① 若 HBM 容量与带宽免费，哪些架构设计会消失、哪些仍会保留？② 当 `16,384` 候选池在多跳大海捞针中漏检时会发生什么？③ 精确回放 vs. 128-token 有界重放的语义偏移边界在哪里？④ 等算力预算 vs. 等参数规模的公平对比准则
     - 14.5.2 全书收束：AI Infra 架构师的五步定量诊断决策树
   - **常见误区**（5 条跨层协同设计深水区误区辨析）
   - **本章小结**（严格作为全章最后一个二级标题）
2. **核心定量算例（4 个，全部锚定仓库锁定真值）**：
   - **算例 14-1（V4 vs. V4.1 Flash 在 $8\mathrm{K}/32\mathrm{K}/128\mathrm{K}/200\mathrm{K}/1\mathrm{M}$ 下的驻留容量与单步 Decode 逻辑读取双轨核算）**：在 $32\mathrm{K}$ 下，40 层标准 MLA（`576 B/token/层`）需 `720 MiB`（$1\mathrm{K}$ 为 `22.5 MiB`），V4 Flash 全局 KV 为 `112.46 MiB`（含 SWA 共 `115.52 MiB`），V4.1 Flash 4-Owner 全局 KV 仅为 $32{,}768\times 890\ \mathrm{B}=\mathbf{27.8125\ \mathrm{MiB}}$；在 $1\mathrm{M}$ 下驻留容量从 `3.351 GiB` 压至 `0.849 GiB`（$3.95\times$），单步读取从 `3.679 GiB` 压至 `1.466 GiB`（$2.51\times$）。
   - **算例 14-2（24×7 数字员工在 $200\mathrm{K}/1\mathrm{M}$ 下 V4.1 Flash vs. Kimi K3 的 B200/B300 选卡与逐层固定开销审计账本）**。
   - **算例 14-3（OpenTallas Case B 单步 `124.221 μs` 四资源分解与 Case D 晶圆级 `180 μs` 跳数延迟越界推算）**。
   - **算例 14-4（反事实推演：当 HBM 带宽/容量趋于免费或模型半衰期从 12 个月缩至 6 个月时，CSA2、CED 与 `$20M` ASIC 流片的存废翻转点）**。
3. **配套配图（6 幅，`codex exec` + `gpt-image-2.6-sunburst` `quality="high"` + `figure_style` 420pt，存入 `manuscripts/ch14/`）**：
   - `ch14-codesign-simultaneous-equations`（图 14-1：模型—系统—硬件协同设计的跨层耦合与四大杠杆）
   - `ch14-v41-dual-track-context-scaling`（图 14-2：$8\mathrm{K}\to 1\mathrm{M}$ 上下文下 V4 与 V4.1 Flash 驻留容量及单步 Decode 读取量的双轨演进曲线）
   - `ch14-v41-four-stage-attention-pipeline`（图 14-3：`Full -> 16,384 候选池 -> Reindex -> Reuse` 四阶段流水与 $1/8$ 持久化缓存拆解）
   - `ch14-digital-worker-v41-vs-k3-cost`（图 14-4：24×7 数字员工在 H100/H200/B200/B300 与官方 API 下的成本交点及 K3 固定开销审计）
   - `ch14-opentallas-four-resource-and-wafer-hop`（图 14-5：OpenTallas 单步 `124.221 μs` 四资源时间线与晶圆级 `180 μs` 通信直径越界对照）
   - `ch14-five-step-diagnostic-decision-tree`（图 14-6：全书总结——AI Infra 五步定量诊断法与反事实设计决策树）
4. **配套练习与实验（6 道，严格使用 `> **练习 14-Y：...**` / `> **实验 14-6：...**` 格式）**：
   - `> **练习 14-1：推导 V4.1 Flash 在任意上下文长度 $H$ 下的驻留容量压缩比与单步 Decode 逻辑读取压缩比渐近极限。**`
   - `> **练习 14-2：核算 PD 分离下 40 层 SWA（`2.578 MiB`）跨网传输与 128-token 局部重算在不同 RDMA 带宽下的翻转阈值。**`
   - `> **练习 14-3：复算 Kimi K3 在 $1\mathrm{M}$ 上下文下扣除逐层固定调度开销并开启 MTP/DSpark 后的单流吞吐上限。**`
   - `> **练习 14-4：OpenTallas Case B 中若辅助算子（RMSNorm/RoPE/SiLU）无法完全隐藏于 `14.568 μs` 窗口，推导实际吞吐退化比例。**`
   - `> **练习 14-5：反事实分析——若下一代 HBM4 带宽翻倍但片上 SRAM 容量不变，FlashAttention-4 与 MLA 的算存平衡点将如何移动？**`
   - `> **实验 14-6：全栈综合推演：给定 24×7 数字员工 SLO（$\ge 30\ \mathrm{tok/s}$，$1\mathrm{M}$ 上下文）与预算，从零完成模型、并行切分、存储卸载与硬件选型设计。**`

---

## 四、 构建与校验基础设施完备性适配方案（支撑 14 章体系）

为确保 Batch 1～6 推进期间随时可构建、且 Batch 7 新增第 13、14 章后全套校验脚本、在线站点与 PDF 编译 100% 绿灯通过，明确以下 **7 项工程适配规则**：

1. **[`scripts/verify_core_principles.py`](../../scripts/verify_core_principles.py)（Batch 7 同步更新，含 `if n <= 12:` 历史快照越界保护）**：
   - 将第 10 行的 `for n in range(1,13):` 扩展为 `for n in range(1,15):`。
   - **关键防崩溃保护**：脚本第 23–27 行读取 `archive/reviews/core-principles-revision-2026-09-10/before/manuscripts/` 校验历史练习编号不被删除，而该历史归档目录仅有 `01`～`12` 章！因此**必须将第 23–27 行的 `before` 读取与对比逻辑包裹在 `if n <= 12:` 分支内**；对于 `n in (13, 14)`，改为校验其练习/实验编号列表 `exercise(s)` 从 `f'{n}-1'` 到 `f'{n}-{len(exercise(s))}'` 严格非空且连续递增。
2. **[`scripts/build_site.py`](../../scripts/build_site.py)（Batch 7 同步更新两处章节上限）**：
   - 在 `chapters(edition)` 函数第 57 行：将 `found = [(n, p) for n, p in found if n <= 12]` 改为按语种区分上限：`max_ch = 14 if edition == 'zh' else 12; found = [(n, p) for n, p in found if n <= max_ch]`。
   - 在第 64 行：将 `if [n for n, _ in found] != list(range(13)):` 改为：`expected = list(range(15)) if edition == 'zh' else list(range(13))`（保证英文 `en`、繁体中文 `zh-tw`、俄文 `ru` 维持既有 12 章构建不报错，而简体中文 `zh` 完整构建前言 + 14 章）。
3. **[`scripts/check_site.py`](../../scripts/check_site.py)（Batch 7 同步更新）**：
   - 在第 46 行：将 `missing = [n for n in range(1, 13) if not any(site.glob(chapter(n)))]` 改为：`max_ch = 15 if name == '简体中文' else 13; missing = [n for n in range(1, max_ch) if not any(site.glob(chapter(n)))]`，并同步更新末尾统计提示信息。
4. **[`book/build_pdf.py`](../../book/build_pdf.py)（Batch 7 同步更新两处 `range`）**：
   - 将第 69 行 `--chapter` 的 `choices=range(1,13)` 与第 79 行的 `numbers = [args.chapter] if args.chapter else range(1,13)` 统一扩展为 `range(1,15)`。
5. **全站导航与导读索引文件同步更新（Batch 7 必改清单，含 [`website/index.md`](../../website/index.md)）**：
   - [`website/index.md`](../../website/index.md)（简体中文网站首页）：将第 21–39 行的 `## 十二章导读` 表格和第 54 行的 `前言与十二章` 更新为 `## 十四章导读` 与 `前言与十四章`，补齐第 13、14 章导读行。
   - 同步更新 [`README.md`](../../README.md)、[`manuscripts/README.md`](../../manuscripts/README.md)、[`manuscripts/00-前言.md`](../../manuscripts/00-%E5%89%8D%E8%A8%80.md) 的章节结构导读。
   - 同步更新 [`case-studies/README.md`](../../case-studies/README.md)：将末尾“延伸阅读（正文未直接引用）”下的全部 16 篇案例移入各自首次引用的章节表格中，标注完整的“引用章节”列表，实现 **未直接引用案例清零（0 篇遗留）**。
6. **[`.gitattributes`](../../.gitattributes) 与三件套配图（`.svg` / `.png` / `.pdf`）非 LFS 指针约束**：
   - `.gitattributes` 末尾已配置 `/manuscripts/ch*/*.pdf !filter !diff !merge binary`。生成每幅新图的三件套（`.svg`、`.png`、`.pdf`）时，必须验证 `.svg` 根节点包含 `width="420pt"`，且 `.png` 与 `.pdf` 均为真实二进制文件（非 `version https://git-lfs.github.com/spec/v1` 指针），防止 `build_site.py` 与 `book/build_pdf.py` 触发 LFS 指针异常。
7. **历史封存目录 [`experiments/ch13/`](../../experiments/README.md) 保护规则**：
   - 仓库中已存在 `experiments/ch13/`（含 `13-01/retrieval-generation`、`13-02`、`13-06/single-agent-serving`），被多篇 `case-studies/` 作为历史封存证据直接引用。**执行期间严禁删除、移动或覆盖 `experiments/ch13/` 下任何已有目录与文件**；若第 13 章（AI 原生存储）需新增配套脚本，统一放入 `calculations/` 或不冲突的 `experiments/ch13/storage-*` 子目录。

---

## 五、 分批执行计划与标准化质量闭环工作流（7 个 Batch）

全书共 **14 章**，按 **每 2 章一个 Batch（共 7 个 Batch）** 稳步推进：

### 5.1 7 个执行批次划分

| 批次 | 涵盖章节（真实文件名） | 核心交付物概览 |
| :---: | :--- | :--- |
| **Batch 1** | **第 1 章（`01-初识 AI Infra.md`）+ 第 2 章（`02-模型架构.md`）** | 消化 `logicfolding`（1.2.2）、`kernel-and-fleet-efficiency`（1.3.4 机队 MPG 教学算例）、`programmable-nic` & `host-policy-and-dispatch`（1.4.2）、CSA2 深度账本（2.3.6）、dLLM 与 `Qwen-Image-2512`/`FLUX.2-klein-4B` DiT+VAE 真值算例（2.6.2）；新增 2 幅 `420pt` 配图、3 个算例、4 道练习（`1-7`、`1-8`、`2-12`、`2-13`）。 |
| **Batch 2** | **第 3 章（`03-推理与训练负载.md`）+ 第 4 章（`04-加速器架构.md`）** | 消化 `retrieval-and-generation`（3.1.3/3.2 真值 `310.168 ms` vs `371.378 ms`）、`queqiao` & `generative-multimodal-models` & 具身 VLA 双环路（3.3.2/3.3.3）、`logicfolding` 降压算例与链接（4.1.3）、FA4 链接与后端分派表（4.2.3，不重复建算例）、`macbook-rtx-pro6000` `4 GiB` 下界表（4.3.1/4.8.3）、GPU TEE Bounce Buffer（4.5.2）；新增 2 幅 `420pt` 配图、4 个算例、4 道练习/实验（`3-13`、`3-14`、`4-11`、`4-12`）。 |
| **Batch 3** | **第 5 章（`05-算子与运行时.md`）+ 第 6 章（`06-超节点.md`）** | 消化 `macbook-rtx-pro6000` Metal 算子分派与 TileLang/CuTe DSL（5.4.1/5.4.5）、FlashInfer `plan/run` $20\ \mu\mathrm{s}$ vs $720\ \mu\mathrm{s}$ 推导与 Sleep 虚拟地址保留（5.5.2/5.5.3）、CPO 共封装光学与 $800\ \mathrm{VDC}$ 高压供电（6.5.2/6.5.3）、`memory-criticality-and-tiering` PACT/Camp（6.6）与 `profile-and-plan-ranking`（6.6.3）；新增 2 幅 `420pt` 配图、3 个算例、3 道实验（`5-12`、`6-14`、`6-15`）。 |
| **Batch 4** | **第 7 章（`07-数据中心网络.md`）+ 第 8 章（`08-推理优化.md`）** | 消化 `programmable-nic` 链接与 ClickNP 对比（7.3.4）、光模块微闪断与 UEC SACK（7.5.3）、1,024 卡 NVLS+多轨 AllReduce（7.6.2/7.6.4）、`structured-generation` XGrammar 锁定真值（8.2.4/8.5.1）、`Fast-dLLM`、Marconi 检查点准入（8.3.4）与 `Rethinking KV Cache Compression` $4.5\ \mathrm{s}\to 5.5\ \mathrm{s}$ 反转（8.4.4）；新增 2 幅 `420pt` 配图、3 个算例、3 道实验/练习（`7-18`、`8-12`、`8-13`）。 |
| **Batch 5** | **第 9 章（`09-分布式推理.md`）+ 第 10 章（`10-训练系统.md`）** | 消化 SWA 128-token 本地重算（9.2.2/9.5.3）、`pd-af-heterogeneous` A100/H20 vs KTransformers（9.2.3/9.3）、万卡 SDC 静默损坏排查（10.4）、`weight-handoff` + `rollout-tail-and-sampling`（RollPacker）+ `rl-scheduling-and-recovery`（RobustRL）（10.5.2/10.5.3）、`profile-and-plan-ranking` vTrain $k=10/9$ 反转（10.6.1）；新增 2 幅 `420pt` 配图、4 个算例、4 道练习/习题（`9-14`、`9-15`、`10-13`、`10-14`）。 |
| **Batch 6** | **第 11 章（`11-资源调度与运行环境.md`）+ 第 12 章（`12-端边云协同.md`）** | 消化 `host-policy-and-dispatch` $600\to 720\ \mathrm{GiB}$ 预算（11.1.2）、`snapshot-residency-and-first-use` E2B/CXLfork（11.2.2/11.2.3）、Rubick $>400$ 步重配（11.3.1）、Weave 互补交错（11.3.3）、`single-agent-serving-cost`（11.4.4）、Ollama 计时口径与 UFS 4.0 闪存流式推理（12.1.5）、`queqiao` 三行十进制预算表与链接（12.3.5）、具身 VLA 动作块缓存（12.4）；新增 2 幅 `420pt` 配图、4 个算例、4 道实验/拓展练习（`11-13`、`11-14`、`12-13`、`12-14`）。 |
| **Batch 7** | **新增第 13 章（`13-AI 原生存储系统与数据底座.md`）+ 新增第 14 章（`14-模型与系统硬件协同设计.md`）+ 全书收官** | 撰写第 13 章与第 14 章全文，生成 `ch13/`（6 幅）与 `ch14/`（6 幅）共 **12 幅 `420pt` 配图**、8 个核心算例、12 道练习/实验；同步升级 `verify_core_principles.py` / `build_site.py` / `check_site.py` / `book/build_pdf.py` 至 `range(1, 15)`，并更新 `website/index.md`、`README.md`、`manuscripts/README.md`、`00-前言.md` 与 `case-studies/README.md`（16 篇未引用案例清零）。 |

---

### 5.2 每个 Batch 的 6 步标准化执行规程（SOP）与 3 条防呆红线

对每个 Batch 内的每一章，严格按以下 6 步执行：

1. **步骤 1：调用 `codex exec`（强制指定 `gpt-image-2.6-sunburst` 最高质量）与 `manuscripts/figure_style` 绘制 `420pt` 出版级配图**
   - 调用 `codex exec` 启用 `image_gen` 工具，**强制指定使用 `gpt-image-2.6-sunburst` 模型并开启最高质量（`quality="high"`）** 生成高保真技术架构图或结合 `manuscripts/figure_style/` 矢量排版封装，落盘至 `manuscripts/chXX/` 的三件套（`.svg`、`.png`、`.pdf`）必须满足：`.svg` 根节点具有 `width="420pt"`、最小字号 $\ge 11\ \mathrm{pt}$、无文字重叠，且 `.pdf` 为非 LFS 指针的真实二进制文件；正文中统一以 `![alt](chXX/...svg)` 引用。
   - 将新图注册至 `manuscripts/chXX/figure-index.json`。
   - **【防呆红线 1：图号重排与跨章引用同步】** 若在章节中部插入新图，该章后续所有配图的图号 `Y` 均会顺延 `+1`。不仅要顺延更新本章后续所有 `*图 X-Y　...*` 题注及本章内文中的 `图 X-Y` 引用，**还必须用 `rg "图 X-"` 扫描全书所有 `manuscripts/*.md`，若其他章节存在指向本章后续图号的跨章引用，必须一并同步更新**。
2. **步骤 2：内容扩充与 [`refine-prose`](../../.agents/skills/refine-prose/SKILL.md) 深度润色**
   - 严格遵循“**实际工程场景/痛点引入 $\to$ 物理直觉与数学公式推导 $\to$ 锁定真值的手算算例与翻转条件 $\to$ 启发式量化练习/实验**”的教学法。
   - **【防呆红线 2：16 篇案例脚注链接闭环】** 本批次负责消化的每一篇 `case-studies/*.md`，都必须作为 Markdown 链接 `[...](../case-studies/<name>.md)` 规范写入该章文末的脚注定义 `[^id]: ...` 中，并在正文对应算例处引用 `[^id]`。
   - 执行 `refine-prose` 全套文风清洗：彻底清除 AI 生造词、翻译腔与口语俚语，确保**感叹号 `！` 零容忍**、**繁体字零残留**、**180 词严格禁用词表（`strict_ai`）零命中**。
3. **步骤 3：本地自动化校验门禁（Pre-Commit Gate）**
   - **【防呆红线 3：构建脚本切换时序】** 在 **Batch 1～Batch 6** 期间，`verify_core_principles.py`、`build_site.py`、`check_site.py`、`book/build_pdf.py` 保持 `range(1, 13)` 不变，使每批均能立即通过全套校验；进入 **Batch 7** 创建第 13、14 章时，在同一批次原子更新上述脚本至 `range(1, 15)`（含 `if n <= 12:` 历史快照保护）。
   - 运行 `python3 scripts/verify_core_principles.py`（检查并还原 `archive/reviews/core-principles-revision-2026-09-10/core-validation.json`），确认 `passed: true, errors: []`。
   - 运行感叹号、繁体字与 180 词 `strict_ai` 禁用词扫描脚本，确认计数全部为 `0`。
   - 运行 `git diff --check` 确认无空白字符格式错误。
4. **步骤 4：提交初版 Git Commit**
   - 将当前章节的正文、新增配图三件套与 `figure-index.json` 提交为原子 Commit。
5. **步骤 5：启动 Gemini 4 Argon High 同配置 Subagent 多轮审校闭环（直至收敛）**
   - 调用 `invoke_subagent`（`TypeName: 'self'`, `Model: 'inherit'`），由具备高推理强度的 Gemini 4 Argon High 同配置 Subagent 对照 `git diff` 与章节全文，从**教学法清晰度、现代汉语书面语文雅度、技术/数值/公式/图号/练习编号/脚注零差错**三个维度进行严苛审阅。
   - 若 Subagent 提出任何修改意见，立即使用 `edit_file` 逐项修复 $\to$ 重新通过步骤 3 校验 $\to$ 提交修复 Commit $\to$ 使用 `send_message` 发回同一 Subagent 复核，循环迭代（最多 5 轮）直至获得明确的 **`Approved / Converged`（终审通过）** 结论。
6. **步骤 6：批次网站构建验证与推送部署（Push & Deploy）**
   - 每完成一个 Batch（两章均获得 Subagent `Approved / Converged`），运行 `python3 scripts/build_site.py && python3 scripts/check_site.py` 验证整站构建与所有内部锚点/图片链接 100% 完好，随后还原临时生成的 `.mkdocs-build-*.yml`（若未纳入版本变更），执行 `git push origin main` 推送部署。
