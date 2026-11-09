# 路径 F3（30 章皇冠全栈终极百科版 / 30-Chapter Crown-Jewel Full-Stack Encyclopedia Edition）工程级总体扩充计划

> **文档状态**：Engineering-Grade Master Plan（面向 Gemini 4 Argon High 独立评审与 15 批次自动化执行）
> **基线版本**：Path E3 收敛提交 `HEAD = 0790cc9a`（24 章，正文 `32,678` 行，编号章节配图 `835` 幅 + 未编号全书导读总图 `1` 幅 = `836` 幅，Python 代码块 `113` 个 + C++ 代码块 `1` 个 = `114` 个类型化代码块，标准化编号算例 `156` 个，标准斜体表题 `281` 个，课后思考题 `384` 块 / `383` 个唯一题号，学术脚注 `602` 条，可运行实验套件 `135` 个）
> **目标版本**：Path F3 终极皇冠全栈版（**30 章**，正文 **`43,548+` 行**，编号章节配图 **`979` 幅** + 重绘未编号 30 章全书导读总图 `1` 幅 = **`980` 幅** / `2,940` 个矢量与位图文件，Python 代码块 **`201` 个** + C++ 代码块 `1` 个 = **`202` 个类型化代码块**，标准化编号算例 **`228` 个**，标准斜体表题 **`341` 个**，课后思考题 **`492` 块 / `491` 个唯一题号**，学术脚注 **`812` 条**，零依赖可运行实验套件 **`165` 个**）

---

## 1. 全书现状精确基线审计（`HEAD = 0790cc9a`）与 Path F3 演进目标矩阵

### 1.1 为什么在 24 章（Path E3）之上启动 Path F3（30 章皇冠全栈版）？

在完成 Path E3 后，《AI 基础设施系统工程》已建立起覆盖基础硬件、分布式训练/推理、光互连、机密计算、多模态、RLVR、具身智能、智能体、全双工语音、百 PB 级数据工程与科学智能（AI4Science）的 24 章体系。但通过脚本级全量审计与对 **2025–2026 全球工业界算力版图及系统顶会（OSDI / SOSP / NSDI / ISCA / ASPLOS / MLSys / SC）** 的对标，当前版本仍存在两大维度的提升空间：

1. **存量 24 章（`Ch01–Ch24`）内部的结构性温差与最新代际拼图补齐**：
   - **生产级代码块温差**：第 1、3、6、7、9、11、12、13、14、15、16 章（共 11 个章节）每章仅有 **`3` 个**带语言标识的代码块，第 2、4、8、10 章仅有 **`4` 个**代码块，而第 17–24 章每章拥有 `6–9` 个生产级代码块；
   - **学术文献脚注温差**：第 1–14 章每章拥有 `19–60` 条详尽的一手学术脚注（共 `523` 条），而第 15–24 章每章仅有 **`3–10` 条**脚注（其中第 17 章仅 `3` 条、第 16/18 章仅 `7` 条，10 章合计仅 `79` 条）；
   - **可运行实验套件温差**：第 1–12 章每章拥有 `7–10` 个实验目录（共 `110` 个），而第 14–24 章每章仅有 **`2` 个**实验目录（第 13 章为 `3` 个）；
   - **2025–2026 旗舰芯片与关键算法拼图待合入**：存量章节需系统合入 **NVIDIA Vera Rubin (`NVL144` / `Rubin Ultra NVL576`, 2048-bit `HBM4` `19.2 TB/s`)**、**Google TPU v7 (`Ironwood` `192 GiB` HBM3e, `9,216` 卡推理 Pod)**、**AMD Instinct `MI355X` / `MI400` (`Helios` 开放机柜)**、**DeepSeek 原生稀疏注意力（`NSA`）与块稀疏注意力（`MoBA`）**、**新一代推理引擎零开销异步重叠调度器（`vLLM V1` / `SGLang` Overlap Scheduler / `NVIDIA Dynamo`）** 以及 **`KTransformers` 单卡 GPU + 双路 CPU AMX 异构运行 671B MoE** 等一线生产技术。
2. **全球 AI 数据中心 6 大核心前沿与工业支柱领域的专章空白（`Ch25–Ch30`）**：
   - **空白领域 1（第 25 章）——生成式推荐（Generative Recommenders / `Meta HSTU` / `OneRec`）、万亿参数稀疏表（`DLRM` / `TorchRec`）与十亿级 GPU 向量检索引擎（`cuVS CAGRA` / `DiskANN`）**：搜广推与企业级向量检索/GraphRAG 占据全球互联网巨头 50% 以上的日常 AI 算力与现金流，亟需专章剖析从级联 DLRM 到生成式序列转导器 `HSTU` 及 GPU/SSD 向量图索引的系统底座；
   - **空白领域 2（第 26 章）——后自回归生成范式基础设施：扩散大语言模型（`dLLM`: `Mercury` / `LLaDA` / `Block Diffusion`）、字节级无分词架构（`Byte Latent Transformer, BLT`）与测试时记忆网络（`Google Titans` / `TTT` / `Coconut`）**：打破“BPE 分词 + 逐词元自回归解码 + 静态权重”三大传统假设的下一代基础模型系统栈；
   - **空白领域 3（第 27 章）——晶圆级与 3D-SRAM 专用推理芯片（`Cerebras WSE-3` / `Groq LPU` / `3D V-Cache`）、微秒级高频极低延迟 AI（Sub-Millisecond AI）与三值量化（`BitNet b1.58` / `T-MAC`）**：面向 `< 50 us` 金融量化/高频控制及 `> 1,500 tok/s` 极致交互推理的非 HBM 范式与全片上静态确定性调度系统；
   - **空白领域 4（第 28 章）——主权 AI、广域网去中心化低通信训练（`DiLoCo` / `Streaming DiLoCo` / `Prime Intellect`）、实时安全护栏（Guardrails）、激活干预与密码学水印（`SynthID-Text`）安全底座**：解决跨地域/跨云低带宽（`1–10 Gbps`）聚合训练千亿模型，以及生产推理毫秒级安全伴飞、防越狱特征钳位与权重供应链防投毒；
   - **空白领域 5（第 29 章）——DeepSeek 开源基础设施全栈解剖与极客工程实战（`FlashMLA` / `DeepGEMM` / `DeepEP` / `DualPipe` / `3FS` / `Mooncake` / `KTransformers`）**：将现代大模型开源系统工程皇冠上的七大组件进行源码级、寄存器/Warp 级与集群数据流级的端到端贯通解剖；
   - **空白领域 6（第 30 章）——自演进 AI 基础设施（AI for AI Infra）：大模型驱动的 GPU 内核自动合成（`KernelBench` / `Kevin`）、强化学习编译寻优与三十章全栈终极收束**：剖析如何用 RLVR + 硬件性能计数器反馈闭环自动合成超越人类专家的 CUDA/Triton 内核、自动求解 5D 并行策略，并对全书 30 章五大架构部类进行大一统收束。

---

### 1.2 逐章精确基线与 Path F3 目标增量总表（`Ch01–Ch30`）

下表给出了当前仓库 `HEAD = 0790cc9a` 的精确脚本实测基线，以及 Path F3 完成后每一章的精确目标值（所有列算术和均经严格校验）：

| 章号 | 章节名称（与 `manuscripts/` 磁盘文件名一致） | 基线行数 | 目标行数 (`+Δ`) | 基线配图 | 目标配图 (`+Δ`) | 基线 Python 代码 | 目标 Python 代码 (`+Δ`) | 基线算例 | 目标算例 (`+Δ`) | 基线表题 | 目标表题 (`+Δ`) | 基线习题 | 目标习题 (`+Δ`) | 基线脚注 | 目标脚注 (`+Δ`) | 基线实验 | 目标实验 (`+Δ`) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **01** | `01-初识 AI Infra.md`（含重绘未编号全书导读总图 `figure-1-book-roadmap`） | `1,095` | `1,305+` (`+210`) | `26` | `28` (`+2`) | `3` | `6` (`+3`) | `4` | `6` (`+2`) | `10` | `11` (`+1`) | `14` | `16` (`+2`) | `25` | `25` (`+0`) | `8` | `8` (`+0`) |
| **02** | `02-模型架构.md`（增补 MLA 矩阵吸收证明与混合注意力状态敏感度） | `1,853` | `2,053+` (`+200`) | `44` | `46` (`+2`) | `4` | `6` (`+2`) | `5` | `7` (`+2`) | `41` | `42` (`+1`) | `18` | `20` (`+2`) | `35` | `35` (`+0`) | `9` | `9` (`+0`) |
| **03** | `03-推理与训练负载.md`（增补多轮 Agent 缓存相变与扩散去噪负载） | `1,293` | `1,503+` (`+210`) | `39` | `41` (`+2`) | `3` | `6` (`+3`) | `4` | `6` (`+2`) | `14` | `15` (`+1`) | `19` | `21` (`+2`) | `49` | `49` (`+0`) | `10` | `10` (`+0`) |
| **04** | `04-加速器架构.md`（增补 NVFP4 微缩放数据通路与 CXL 3.1 内存池排队） | `1,826` | `2,026+` (`+200`) | `51` | `53` (`+2`) | `4` | `6` (`+2`) | `5` | `7` (`+2`) | `19` | `20` (`+1`) | `18` | `20` (`+2`) | `60` | `60` (`+0`) | `7` | `7` (`+0`) |
| **05** | `05-算子与运行时.md`（增补变长序列 Split-KV 持久调度与 JIT 特化） | `1,641` | `1,821+` (`+180`) | `48` | `50` (`+2`) | `9` | `10` (`+1`) | `5` | `7` (`+2`) | `17` | `18` (`+1`) | `18` | `20` (`+2`) | `44` | `44` (`+0`) | `9` | `9` (`+0`) |
| **06** | `06-超节点.md`（增补 NVLink 6 576 卡正交背板与 3D Torus 对分带宽） | `1,824` | `2,034+` (`+210`) | `65` | `67` (`+2`) | `3` | `6` (`+3`) | `5` | `7` (`+2`) | `21` | `22` (`+1`) | `20` | `22` (`+2`) | `44` | `44` (`+0`) | `10` | `10` (`+0`) |
| **07** | `07-数据中心网络.md`（增补 1.6T 网卡、Google Falcon 与 MoE 自适应路由） | `1,854` | `2,064+` (`+210`) | `69` | `71` (`+2`) | `3` | `6` (`+3`) | `5` | `7` (`+2`) | `11` | `12` (`+1`) | `23` | `25` (`+2`) | `42` | `42` (`+0`) | `10` | `10` (`+0`) |
| **08** | `08-推理优化.md`（增补 `vLLM V1` / `SGLang` 零开销异步重叠调度） | `1,445` | `1,645+` (`+200`) | `44` | `46` (`+2`) | `4` | `6` (`+2`) | `5` | `7` (`+2`) | `8` | `9` (`+1`) | `18` | `20` (`+2`) | `36` | `36` (`+0`) | `9` | `9` (`+0`) |
| **09** | `09-分布式推理.md`（增补 `NVIDIA Dynamo` 路由与 `KTransformers`） | `1,513` | `1,723+` (`+210`) | `54` | `56` (`+2`) | `3` | `6` (`+3`) | `5` | `7` (`+2`) | `11` | `12` (`+1`) | `20` | `22` (`+2`) | `41` | `41` (`+0`) | `10` | `10` (`+0`) |
| **10** | `10-训练系统.md`（增补 `Megatron-Core` / `TorchTitan` 细粒度选择性重计算） | `1,714` | `1,914+` (`+200`) | `56` | `58` (`+2`) | `4` | `6` (`+2`) | `5` | `7` (`+2`) | `13` | `14` (`+1`) | `20` | `22` (`+2`) | `37` | `37` (`+0`) | `10` | `10` (`+0`) |
| **11** | `11-资源调度与运行环境.md`（增补 `CuMemMap` 毫秒级显存休眠与 RDMA 多播冷启动） | `1,494` | `1,704+` (`+210`) | `57` | `59` (`+2`) | `3` | `6` (`+3`) | `5` | `7` (`+2`) | `9` | `10` (`+1`) | `19` | `21` (`+2`) | `34` | `34` (`+0`) | `10` | `10` (`+0`) |
| **12** | `12-端边云协同.md`（增补 `ExecuTorch` 编译器与 `M5 Ultra` TB5 RDMA 环网） | `1,407` | `1,617+` (`+210`) | `42` | `44` (`+2`) | `3` | `6` (`+3`) | `4` | `6` (`+2`) | `9` | `10` (`+1`) | `19` | `21` (`+2`) | `37` | `37` (`+0`) | `8` | `8` (`+0`) |
| **13** | `13-AI 原生存储系统与数据底座.md`（增补 `LMCache` 与 `13-04`） | `1,271` | `1,481+` (`+210`) | `26` | `28` (`+2`) | `3` | `6` (`+3`) | `14` | `14` (`+0`) | `10` | `11` (`+1`) | `17` | `19` (`+2`) | `22` | `22` (`+0`) | `3` | `4` (`+1`) |
| **14** | `14-模型与系统硬件协同设计.md`（增补联合 Pareto 与 `14-03`） | `1,260` | `1,470+` (`+210`) | `26` | `28` (`+2`) | `3` | `6` (`+3`) | `13` | `13` (`+0`) | `11` | `12` (`+1`) | `17` | `19` (`+2`) | `19` | `19` (`+0`) | `2` | `3` (`+1`) |
| **15** | `15-吉瓦级 AI 数据中心、电力热学与跨地域协同训练.md`（增补脚注与 `15-03`） | `1,163` | `1,383+` (`+220`) | `24` | `26` (`+2`) | `3` | `6` (`+3`) | `12` | `12` (`+0`) | `11` | `12` (`+1`) | `16` | `18` (`+2`) | `9` | `21` (`+12`) | `2` | `3` (`+1`) |
| **16** | `16-机密 AI 计算、可信执行环境与安全沙箱底座.md`（增补脚注与 `16-03`） | `1,072` | `1,292+` (`+220`) | `24` | `26` (`+2`) | `3` | `6` (`+3`) | `12` | `12` (`+0`) | `11` | `12` (`+1`) | `16` | `18` (`+2`) | `7` | `19` (`+12`) | `2` | `3` (`+1`) |
| **17** | `17-多模态、视频生成（DiT）与世界模型基础设施.md`（增补脚注与 `17-03`） | `1,290` | `1,470+` (`+180`) | `20` | `22` (`+2`) | `8` | `9` (`+1`) | `7` | `8` (`+1`) | `6` | `7` (`+1`) | `14` | `16` (`+2`) | `3` | `15` (`+12`) | `2` | `3` (`+1`) |
| **18** | `18-大规模后训练（RLVR）、异步强化学习与推理时扩展系统.md`（增补脚注与 `18-03`） | `1,266` | `1,446+` (`+180`) | `20` | `22` (`+2`) | `9` | `10` (`+1`) | `9` | `10` (`+1`) | `8` | `9` (`+1`) | `14` | `16` (`+2`) | `7` | `19` (`+12`) | `2` | `3` (`+1`) |
| **19** | `19-具身智能（VLA）、实时控制与大规模物理仿真底座.md`（含 1 个 C++ 块；增补脚注与 `19-03`） | `1,213` | `1,393+` (`+180`) | `18` | `20` (`+2`) | `7` | `8` (`+1`) | `6` | `7` (`+1`) | `8` | `9` (`+1`) | `12` | `14` (`+2`) | `9` | `21` (`+12`) | `2` | `3` (`+1`) |
| **20** | `20-十万卡集群可靠性工程、静默数据损坏（SDC）防御与基准评测科学.md`（增补脚注与 `20-03`） | `1,130` | `1,310+` (`+180`) | `18` | `20` (`+2`) | `6` | `7` (`+1`) | `6` | `7` (`+1`) | `9` | `10` (`+1`) | `12` | `14` (`+2`) | `10` | `22` (`+12`) | `2` | `3` (`+1`) |
| **21** | 自主智能体基础设施（Agentic Infra）、Computer Use 沙箱与分层 KV 记忆底座（增补 `21-03`） | `1,050` | `1,250+` (`+200`) | `16` | `18` (`+2`) | `6` | `7` (`+1`) | `5` | `7` (`+2`) | `6` | `7` (`+1`) | `10` | `12` (`+2`) | `8` | `20` (`+12`) | `2` | `3` (`+1`) |
| **22** | 全双工实时语音（Full-Duplex Speech）、神经音频编解码与原生全模态流式底座（增补 `22-03`） | `1,025` | `1,225+` (`+200`) | `16` | `18` (`+2`) | `6` | `7` (`+1`) | `5` | `7` (`+2`) | `6` | `7` (`+1`) | `10` | `12` (`+2`) | `8` | `20` (`+12`) | `2` | `3` (`+1`) |
| **23** | 百亿亿次 Token 数据工程、GPU 加速模糊去重与合成数据工厂底座（增补 `23-03`） | `1,005` | `1,225+` (`+220`) | `16` | `18` (`+2`) | `6` | `8` (`+2`) | `5` | `7` (`+2`) | `6` | `7` (`+1`) | `10` | `12` (`+2`) | `8` | `20` (`+12`) | `2` | `3` (`+1`) |
| **24** | 科学智能（AI for Science）基础设施：生物大分子、百万碱基基因组与物理场网格系统（增补 `24-03`） | `974` | `1,194+` (`+220`) | `16` | `18` (`+2`) | `6` | `8` (`+2`) | `5` | `7` (`+2`) | `6` | `7` (`+1`) | `10` | `12` (`+2`) | `8` | `20` (`+12`) | `2` | `3` (`+1`) |
| **25** | **【新章】生成式推荐（HSTU）、万亿稀疏表（DLRM）与 GPU 向量检索** | `0` | `1,000+` (`+1,000`) | `0` | `16` (`+16`) | `0` | `6` (`+6`) | `0` | `6` (`+6`) | `0` | `6` (`+6`) | `0` | `10` (`+10`) | `0` | `15` (`+15`) | `0` | `3` (`+3`) |
| **26** | **【新章】扩散语言模型（dLLM）、无分词（BLT）与测试时记忆（Titans）** | `0` | `1,000+` (`+1,000`) | `0` | `16` (`+16`) | `0` | `6` (`+6`) | `0` | `6` (`+6`) | `0` | `6` (`+6`) | `0` | `10` (`+10`) | `0` | `15` (`+15`) | `0` | `3` (`+3`) |
| **27** | **【新章】晶圆级/3D-SRAM 推理芯片、微秒级 AI 与三值量化（BitNet）** | `0` | `1,000+` (`+1,000`) | `0` | `16` (`+16`) | `0` | `6` (`+6`) | `0` | `6` (`+6`) | `0` | `6` (`+6`) | `0` | `10` (`+10`) | `0` | `15` (`+15`) | `0` | `3` (`+3`) |
| **28** | **【新章】主权 AI、广域网去中心化训练（DiLoCo）、护栏与密码学水印** | `0` | `1,000+` (`+1,000`) | `0` | `16` (`+16`) | `0` | `6` (`+6`) | `0` | `6` (`+6`) | `0` | `6` (`+6`) | `0` | `10` (`+10`) | `0` | `15` (`+15`) | `0` | `3` (`+3`) |
| **29** | **【新章】DeepSeek 开源基础设施全栈源码级解剖与极客工程实战** | `0` | `1,000+` (`+1,000`) | `0` | `16` (`+16`) | `0` | `6` (`+6`) | `0` | `6` (`+6`) | `0` | `6` (`+6`) | `0` | `10` (`+10`) | `0` | `15` (`+15`) | `0` | `3` (`+3`) |
| **30** | **【新章】自演进 AI 基础设施（AI for AI Infra）与三十章全栈大一统收束** | `0` | `1,000+` (`+1,000`) | `0` | `16` (`+16`) | `0` | `6` (`+6`) | `0` | `6` (`+6`) | `0` | `6` (`+6`) | `0` | `10` (`+10`) | `0` | `15` (`+15`) | `0` | `3` (`+3`) |
| **合计** | **全书 30 章总计（不含 `00-前言.md`；另含 Ch19 C++ 代码块 1 个）** | **`32,678`** | **`43,548+` (`+10,870+`)** | **`835`** | **`979` (`+144`)** | **`113`** | **`201` (`+88`)** | **`156`** | **`228` (`+72`)** | **`281`** | **`341` (`+60`)** | **`384`** | **`492` (`+108`)** | **`602`** | **`812` (`+210`)** | **`135`** | **`165` (`+30`)** |

> [!IMPORTANT]
> **全书均衡化硬指标承诺（Zero-Asymmetry Guarantee）**：
> 完成 Path F3 后，全书 **30 个章节（`Ch01–Ch30`）中的每一章** 都将同时满足：
> 1. 章节篇幅 $\ge 1,000$ 行（全书共 `43,548+` 行，平均每章 `1,451+` 行）；
> 2. 出版级 `420pt` 编号章节配图 $\ge 16$ 幅（`Ch01–Ch24` 均 $\ge 18$ 幅，`Ch25–Ch30` 均为 `16` 幅，全书共 `979` 幅编号章节配图 + `1` 幅未编号全书 30 章导读总图 `ch01/figure-1-book-roadmap` = `980` 幅配图）；
> 3. 带语言标识的生产级系统代码块 $\ge 6$ 个（全书共 `201` 个 Python 代码块 + `Ch19` 既有 `1` 个 C++ 代码块 = `202` 个类型化代码块，彻底消灭早期章节只有 3–4 个代码块的不均衡现象）；
> 4. 标准化引用块算例（`> **算例 X-Y（...）**`）$\ge 6$ 个；
> 5. 标准化斜体表题（`*表 X-Y　...*`）$\ge 6$ 个（且每章均配备章末课后习题核心锁定参数与标准答案速查表）；
> 6. 课后思考题 $\ge 10$ 道（`Ch01–Ch24` 均 $\ge 12$ 道，`Ch25–Ch30` 均为 `10` 道，全书 `492` 个习题块 / `491` 个唯一题号，100% 附带详尽分步数学推导）；
> 7. 一手学术文献与工业规范脚注 $\ge 15$ 条（彻底消除 `Ch15–Ch24` 脚注偏少的短板）；
> 8. 零依赖确定性可运行实验套件（`experiments/chXX/XX-YY`）$\ge 3$ 个（全书 `165` 个实验目录，100% 包含 `README.md`、`run.py`、`verify.py`、`expected/metrics.json`、`results/summary.json`）。

---

## 2. 非破坏性验证与构建门禁不变量（Non-Destructive Invariants Specification）

为确保在扩充 `Ch01–Ch24` 及新建 `Ch25–Ch30` 的全过程中，仓库现有的所有有效自动化校验脚本始终 100% 绿灯通过且不污染历史归档 JSON，执行时必须严格遵守以下脚本级不变量：

### 2.1 `Ch01–Ch12` 大纲顺序锁、配图单调落位锁、习题正则锁与历史脚本边界
1. **历史归档脚本（禁止运行）vs. 现行活跃门禁（必须更新并运行）的严格区分**：
   - `manuscripts/ch01/verify.py` 在仓库中不存在；而 [`manuscripts/verify_teaching_book.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/verify_teaching_book.py) 与 `manuscripts/ch02..08/verify.py` 是 `2026-09-10` 轮次的历史归档脚本（硬编码了 Path A 之前的历史图数和旧标题，一旦执行会覆写受 Git 跟踪的 `manuscripts/ch02..08/validation.json` 与 `archive/reviews/book-teaching-rewrite-2026-09-10/final-validation.json` 并写入 `"passed": false`）。因此，**严禁执行 `manuscripts/verify_teaching_book.py` 与 `manuscripts/ch02..08/verify.py`**，保持 `manuscripts/ch02..08/validation.json` 只读不动。
   - 相反，**[`manuscripts/ch09/verify.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/ch09/verify.py) 至 [`manuscripts/ch24/verify.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/ch24/verify.py)（以及新建的 `manuscripts/ch25..30/verify.py`）全部是现行活跃门禁（`exit 0`）**，在扩充对应章节时必须同步更新其中的计数/哈希断言并执行通过。此外，在更新 `Ch01–Ch12` 时，必须同步刷新各章 `manuscripts/chXX/manifest.json`（文件 SHA-256 哈希）、`figure-index.json`、`teaching-layout-validation.json`、`math-validation.json` 与 `preview_output/` HTML。
2. **H2 / H3 编号大纲顺序锁与新图单调递增落位规则（Monotonic Figure Placement & Heading Lock）**：
   - `Ch09–Ch12` 的 `verify.py` 会比对正文中的 `^#{2,3} X.Y` 标题序列与 `outline.md` 完全一致；同时全书 [`scripts/verify_core_principles.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/scripts/verify_core_principles.py) 强制校验每章最后一个二级标题为 `## 本章小结`，且正文中所有章节配图引用编号必须严格单调递增（`图 X-1, 图 X-2, ..., 图 X-(N+2)`）并与 `manuscripts/chXX/figure-index.json` 的数组顺序 100% 一致。
   - **黄金落位准则**：在 `Ch01–Ch24` 中新增的深度专题（使用四级标题 `#### 专题名称`，严禁新增 `## X.Y` 或 `### X.Y.Z` 编号标题）、新增的 `+2` 幅配图 `图 X-(N+1)` 与 `图 X-(N+2)`、新增的代码块与算例，**一律放置在所在章节最后一个编号小节的末尾（即紧接既有最后一幅图 `图 X-N` 之后、`## 常见误区` / `## 练习与实验` / `## 本章小结` 之前）**，确保全章配图编号与算例编号严格单调递增，零乱序风险。
3. **课后习题精确正则匹配锁（Exact Exercise Regex Lock）**：
   - [`scripts/verify_core_principles.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/scripts/verify_core_principles.py) 第 23 行统计全书习题使用的**真实正则表达式**为：
     ```python
     all_ex = re.findall(r'^(?:>\s*)?\*\*(?:(?:练习|实验|习题|拓展练习)\s+)?(\d+-\d+)\b(?!\s*解答)', s, re.M)
     ```
     注意该正则**只匹配行首粗体 `**...` 或引用块粗体 `> **...`，绝不匹配 `### ` 标题**。因此，各章新增的 `+2` 道思考题必须严格采用如下与该章既有习题及单章 `verify.py` 完全兼容的粗体/引用块格式：
     - **`Ch01–Ch08`**：统一采用 `> **练习 X-N〔延伸〕：...**`（或 `> **实验 X-N · 延伸：...**`），紧接该章既有最后一题之后，解答小节统一采用 `#### 练习 X-N 解答`；
     - **`Ch09` 与 `Ch10`**：严格沿用章内既有无前缀粗体格式 `**9-21 ...**`、`**9-22 ...**` 与 `**10-21 ...**`、`**10-22 ...**`（`ch09/verify.py` 与 `ch10/verify.py` 切片取前 10 题校验 `1..10` 并断言总题数 `>= 15`，刷新 `manifest.json` SHA-256 后即原生通过）；
     - **`Ch11`**：严格沿用 `11-18..19` 的格式 `> **实验 11-20 · ...〔延伸〕**` 与 `> **实验 11-21 · ...〔延伸〕**`（`ch11/verify.py` 切片前 17 题检查 `11-1..17` 并断言 `len >= 17`，刷新 `manifest.json` 后原生通过）；
     - **`Ch12`**：严格沿用 `12-11..19` 的格式 `**拓展练习 12-20　...**` 与 `**拓展练习 12-21　...**`（题号后接全角空格 `　`），并同步将 [`manuscripts/ch12/verify.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/ch12/verify.py) 第 166–168 行的 `range(11, 20)` / `'exercises': 19` 更新为 `range(11, 22)` / `'exercises': 21`；
     - **`Ch13–Ch24` 及新建 `Ch25–Ch30`**：统一采用 `**练习 XX-N（...）**`（或与所在章一致的 `**练习 XX-N：...**`）格式，并同步更新 `manuscripts/chXX/verify.py` 中的 `exercises` 计数。
4. **外部图注标点锁（External Caption Punctuation Lock）**：
   - `Ch05` 与 `Ch12` 要求外部图注使用中文冒号 `*图 5-X：...*` 与 `*图 12-X：...*`；其余章节使用全角空格 `*图 X-Y　...*`（或兼容冒号/全角空格）。新增配图时严格沿用所在章节既有图注分隔符。

### 2.2 `Ch13–Ch24` 及新建 `Ch25–Ch30` 强契约校验不变量（`manuscripts/ch13..30/verify.py`）
- `Ch13–Ch24` 的 `verify.py` 均采用精确计数断言（检查 `fig_count`、`exercises`、`len(code_blocks)`、`len(worked_examples)`、`table_captions`、`sources.json` 与 `manifest.json` 哈希）。其中 [`manuscripts/ch19/verify.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/ch19/verify.py) 第 53 行统计全部类型化代码块（含 `Ch19` 既有的 1 个 `cpp` 代码块），在 `Ch19` 新增 1 个 Python 代码块后（Python 代码块从 `7` 增至 `8`，总类型化代码块从 `8` 增至 `9`），同步将其第 53 行更新为 `len(code_blocks) == 9 and len(re.findall(r'```', s)) == 18`。
- 在扩充 `Ch13–Ch24` 时，每章同步更新对应的 `manuscripts/chXX/build.py`、`manuscripts/chXX/figure-data.json` 与 `manuscripts/chXX/verify.py` 中的目标计数值，并重新运行 `build.py && verify.py` 刷新 `figure-index.json`、`sources.json`、`teaching-layout-validation.json`、`math-validation.json` 与 `manifest.json`。
- 新建 `Ch25–Ch30` 时，每章均配备包含 **64 项检查（64/64 checks）** 的 `manuscripts/chXX/verify.py`（检查章节四节八小节大纲、末节 `## 本章小结`、10 道习题 `XX-1..10`、6 个生产级代码块、6 个引用块算例 `算例 XX-1..6`、6 个斜体表题 `表 XX-1..6`、16 幅 `420pt` 配图的 `.svg/.png/.pdf` 完整性与零内部图号、15 条脚注定义与引用匹配、公式语法与锁定数值锚点）。

### 2.3 全书排版与零 HTML 铁律（Typography & Zero-HTML Gate）
- [`scripts/check_no_html.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/scripts/check_no_html.py) 强制要求所有 Markdown 源文件**严禁包含任何原生 HTML 标签**（如 `<details>`、`<summary>`、`<br>`、`<sub>`、`<sup>`）。所有新增习题解析统一采用纯 Markdown 三级/四级标题（`### 参考解答与详尽推导` + `#### 练习 X-Y 解答`）。
- 全书正文与图注严禁出现感叹号（`！` 或英文散文 `!`）、严禁出现未转义货币符号 `$`（统一写为 `美元` 或 `USD`）、严禁出现繁体中文字形（如将“瓶颈、体系、网络”误写为繁体字形），所有仓库内路径在正文中一律使用行内代码反引号包裹（如 `` `experiments/ch25/25-01` ``）。

---

## 3. 存量 24 章（`Ch01–Ch24`）逐章原位深化与均衡化工程蓝图

本节详细规定第 1–24 章（Batches 1–12）每一章新增的 **2 幅 420pt 配图（共 `+48` 幅，统一置于各章最后一个编号小节末尾以确保图号严格单调递增）**、**1–3 个生产级 Python 代码块（共 `+52` 个）**、**编号算例（共 `+36` 个）**、**标准表题（共 `+24` 个，并同步更新各章末习题速查表）**、**2 道课后思考题（共 `+48` 道）**、**学术脚注（`Ch15–Ch24` 共 `+120` 条）** 与 **可运行实验套件（`Ch13–Ch24` 共 `+12` 个）**：

### 3.1 Batch 1：第 1 章 & 第 2 章原位深化
- **第 1 章（[`manuscripts/01-初识 AI Infra.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/01-%E5%88%9D%E8%AF%86%20AI%20Infra.md)）**：
  - **重绘未编号全书导读总图**：更新 [`manuscripts/ch01/build_roadmap.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/ch01/build_roadmap.py)，将未编号全书导读总图 `ch01/figure-1-book-roadmap.{svg,png,pdf}`（独立于编号 `图 1-1` `ch01/figure-1-programmability`）升级为覆盖 **第 1–30 章五大架构部类** 的全景路线图。
  - **新增深度专题（置于 `1.4.5` 节末尾、`图 1-26` 之后）**：增补 `#### 推理摊销全生命周期扩展律（Inference-Amortized Scaling Law）与 30 章五大负载谱系`，剖析从自回归大模型到生成式推荐（HSTU）、扩散语言模型（dLLM）与去中心化算力网的全栈负载演进及 30 章知识体系映射。
  - **新增 2 幅编号配图（`26 -> 28`）**：
    - `图 1-27` (`ch01-inference-amortized-overtraining-scaling-curve`)：全生命周期推理摊销下最优参数量 $N^*$ 与最优预训练词元数 $D_{\mathrm{train}}^*$ 的偏移曲线（解释为何 `8B` 模型需训 `15T–20T` 词元）；
    - `图 1-28` (`ch01-thirty-chapter-five-pillar-workload-spectrum`)：全书 30 章五大算力象限（稠密/MoE 预训练、长思维链 RLVR、搜广推 HSTU/向量检索、极低延迟 SRAM/扩散推理、广域网/科学智能）的算术强度与通信敏感度谱系。
  - **新增 3 个生产级代码块（`3 -> 6`）**：（1）推理摊销全生命周期最优模型规模与过训练比率求解器 `optimize_inference_amortized_scaling()`；（2）多模态 + 智能体 + 生成式推荐混合数据中心机队容量分配规划器；（3）跨代际加速器（H100 / B200 / Vera Rubin R100 / TPU v7）TCO 与能效前沿分析器。
  - **新增 2 个编号算例（`4 -> 6`）**：`算例 1-5`（累计服务 `500T` 推理词元下的最优过训练倍数 $D/N$ 推导）、`算例 1-6`（30 章五大典型负载的机柜级功耗与网络带宽配比核算）。
  - **新增 1 张表格（`10 -> 11`）** 与 **2 道思考题（`14 -> 16`：`> **练习 1-15〔延伸〕：...**`、`> **练习 1-16〔延伸〕：...**`）**。
- **第 2 章（[`manuscripts/02-模型架构.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/02-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84.md)）**：
  - **新增深度专题（置于 `2.6.5` 节末尾、`图 2-44` 之后，与既有 `NSA` 基础介绍互补且零重复）**：增补 `#### 多头潜在注意力（MLA）矩阵吸收恒等变换证明与混合线性—全注意力跨层状态容量分配`，严格推导解码期将 $W^{UK}$ 吸收进 $W^Q$、将 $W^{UV}$ 吸收进 $W^O$ 的张量收缩结合律与条件数稳定性，以及混合架构（如 `3:1` 线性状态层 + 全注意力层）下的 RoPE 频率解耦敏感度与跨向后章节（第 26、29 章）的数学桥接。
  - **新增 2 幅编号配图（`44 -> 46`）**：
    - `图 2-45` (`ch02-mla-matrix-absorption-tensor-contraction-datapath`)：`MLA` 训练期显式解压与推理期 $W^{UK}/W^{UV}$ 矩阵吸收前后的张量维数、FLOPs 与 HBM 字节流对比；
    - `图 2-46` (`ch02-hybrid-linear-full-attention-state-capacity-pareto`)：混合线性递归状态与全注意力 KV 缓存的层间配比、针海检索（NIAH）召回率与显存容量 Pareto 前沿。
  - **新增 2 个生产级代码块（`4 -> 6`）**：（1）`MLA` 矩阵吸收前后数值等价性验证与算术强度拐点计算器；（2）混合线性—全注意力层间状态分配与块稀疏路由掩码（`MoBA`）算术强度模拟器。
  - **新增 2 个编号算例（`5 -> 7`）**：`算例 2-6`（`MLA` 矩阵吸收在批大小 $B=1..128$ 下的计算冗余与 HBM 带宽净收益精确盈亏平衡点推导）、`算例 2-7`（64 层混合架构 `3:1` 线性状态 + 全注意力的 `128k` 上下文常驻状态账本核算）。
  - **新增 1 张表格（`41 -> 42`）** 与 **2 道思考题（`18 -> 20`：`> **练习 2-19〔延伸〕：...**`、`> **练习 2-20〔延伸〕：...**`）**。

### 3.2 Batch 2：第 3 章 & 第 4 章原位深化
- **第 3 章（[`manuscripts/03-推理与训练负载.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/03-%E6%8E%A8%E7%90%86%E4%B8%8E%E8%AE%AD%E7%BB%83%E8%B4%9F%E8%BD%BD.md)）**：
  - **新增深度专题（置于最后编号小节末尾、`图 3-39` 之后）**：增补 `#### 长上下文预填充—解码动态比例漂移、多轮 Agent 前缀缓存命中率相变与批量扩散去噪负载特征`。
  - **新增 2 幅编号配图（`39 -> 41`）**：`图 3-40` (`ch03-multi-turn-prefix-cache-hit-rate-roofline-shift`)、`图 3-41` (`ch03-autoregressive-vs-block-diffusion-workload-profile`)。
  - **新增 3 个生产级代码块（`3 -> 6`）**、**2 个编号算例（`4 -> 6`：`算例 3-5`、`算例 3-6`）**、**1 张表格（`14 -> 15`）**、**2 道思考题（`19 -> 21`：`> **练习 3-20〔延伸〕：...**`、`> **练习 3-21〔延伸〕：...**`）**。
- **第 4 章（[`manuscripts/04-加速器架构.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/04-%E5%8A%A0%E9%80%9F%E5%99%A8%E6%9E%B6%E6%9E%84.md)）**：
  - **新增深度专题（置于 `4.8.6` 节末尾、`图 4-51` 之后，与既有 Rubin/HBM4 概览互补且零重复）**：增补 `#### 第五代 Tensor Core 微缩放浮点（NVFP4 / MXFP4）硬件数据通路信噪比与 CXL 3.1 内存池化排队延迟模型`，深入推导 `NVFP4`（每 16 个 `E2M1` 元素共享 `E4M3` 二级微块缩放因子）的硬件点积树量化信噪比（SQNR）及 `CXL 3.1` Type-3 交换机池化内存域在突发访存下的排队延迟边界（并与第 27 章全片上 SRAM 架构形成对比）。
  - **新增 2 幅编号配图（`51 -> 53`）**：`图 4-52` (`ch04-nvfp4-mxfp4-two-level-microscaling-tensor-core-datapath`)、`图 4-53` (`ch04-cxl31-memory-pooling-fabric-queueing-latency-model`)。
  - **新增 2 个生产级代码块（`4 -> 6`）**、**2 个编号算例（`5 -> 7`：`> **算例 4-6（...）**`、`> **算例 4-7（...）**`）**、**1 张表格（`19 -> 20`）**、**2 道思考题（`18 -> 20`：`> **实验 4-19 · 延伸：...**`、`> **实验 4-20 · 延伸：...**`）**。

### 3.3 Batch 3：第 5 章 & 第 6 章原位深化
- **第 5 章（[`manuscripts/05-算子与运行时.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/05-%E7%AE%97%E5%AD%90%E4%B8%8E%E8%BF%90%E8%A1%8C%E6%97%B6.md)）**：
  - **新增深度专题（置于最后编号小节末尾、`图 5-48` 之后）**：增补 `#### 变长序列 Split-KV 持久内核负载均衡（FlashMLA / FlashDecoding++ Tile Scheduler）与运行期 JIT 模板特化消除寄存器溢出`。
  - **新增 2 幅编号配图（`48 -> 50`）**：`图 5-49` (`ch05-variable-length-split-kv-persistent-tile-scheduler`)、`图 5-50` (`ch05-jit-compile-time-shape-specialization-register-pressure`)。
  - **新增 1 个生产级代码块（`9 -> 10`）**、**2 个编号算例（`5 -> 7`：`算例 5-6`、`算例 5-7`）**、**1 张表格（`17 -> 18`）**、**2 道思考题（`18 -> 20`：`> **实验 5-19 · 延伸：...**`、`> **实验 5-20 · 延伸：...**`）**。
- **第 6 章（[`manuscripts/06-超节点.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/06-%E8%B6%85%E8%8A%82%E7%82%B9.md)）**：
  - **新增深度专题（置于 `6.7.6` 节末尾、`图 6-65` 之后，与既有超节点概览互补且零重复）**：增补 `#### NVLink 6（3.6 TB/s）576 卡正交无缆背板信号完整性损耗预算与全连接交换 vs. 3D Torus 对分带宽对比`，定量建模 `Kyber` 576 卡正交 PCB 中板（Orthogonal Midplane）在 `224G PAM4` SerDes 下的通道插入损耗（$\le 28\ \mathrm{dB}$）、反射串扰预算，以及全互连 NVSwitch 域与 `TPU v7` 3D Torus 光互连拓扑在 AllReduce / All-to-All 下的对分带宽与跳数延迟交叉点。
  - **新增 2 幅编号配图（`65 -> 67`）**：`图 6-66` (`ch06-rubin-ultra-nvl576-orthogonal-midplane-insertion-loss-budget`)、`图 6-67` (`ch06-nvswitch-full-bisection-vs-3d-torus-collective-scaling`)。
  - **新增 3 个生产级代码块（`3 -> 6`）**、**2 个编号算例（`5 -> 7`：`算例 6-6`、`算例 6-7`）**、**1 张表格（`21 -> 22`）**、**2 道思考题（`20 -> 22`：`> **实验 6-21 · 延伸：...**`、`> **实验 6-22 · 延伸：...**`）**。

### 3.4 Batch 4：第 7 章 & 第 8 章原位深化
- **第 7 章（[`manuscripts/07-数据中心网络.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/07-%E6%95%B0%E6%8D%AE%E4%B8%AD%E5%BF%83%E7%BD%91%E7%BB%9C.md)）**：
  - **新增深度专题（置于最后编号小节末尾、`图 7-69` 之后）**：增补 `#### 1.6Tbps（8x200G PAM4）以太网、Google Falcon 硬件可靠传输协议与 ConnectX-8 / Spectrum-X800 面对 MoE All-to-All 突发微流的自适应无损路由`。
  - **新增 2 幅编号配图（`69 -> 71`）**：`图 7-70` (`ch07-1600g-connectx8-and-google-falcon-hardware-transport`)、`图 7-71` (`ch07-moe-incast-adaptive-routing-and-credit-loop`)。
  - **新增 3 个生产级代码块（`3 -> 6`）**、**2 个编号算例（`5 -> 7`：`算例 7-6`、`算例 7-7`）**、**1 张表格（`11 -> 12`）**、**2 道思考题（`23 -> 25`：`> **实验 7-24 · 延伸：...**`、`> **实验 7-25 · 延伸：...**`）**。
- **第 8 章（[`manuscripts/08-推理优化.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/08-%E6%8E%A8%E7%90%86%E4%BC%98%E5%8C%96.md)）**：
  - **新增深度专题（置于最后编号小节末尾、`图 8-44` 之后）**：增补 `#### 推理引擎零开销异步重叠调度器（vLLM V1 / SGLang Zero-Overhead Overlap Scheduler）：消除控制面、前缀树匹配与采样同步空泡`。
  - **新增 2 幅编号配图（`44 -> 46`）**：`图 8-45` (`ch08-synchronous-engine-bubble-vs-vllm-v1-sglang-overlap-scheduler`)、`图 8-46` (`ch08-async-future-token-id-pipeline-on-device-sampling`)。
  - **新增 2 个生产级代码块（`4 -> 6`）**、**2 个编号算例（`5 -> 7`：`算例 8-6`、`算例 8-7`）**、**1 张表格（`8 -> 9`）**、**2 道思考题（`18 -> 20`：`> **练习 8-19〔延伸〕：...**`、`> **练习 8-20〔延伸〕：...**`）**。

### 3.5 Batch 5：第 9 章 & 第 10 章原位深化
- **第 9 章（[`manuscripts/09-分布式推理.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/09-%E5%88%86%E5%B8%83%E5%BC%8F%E6%8E%A8%E7%90%86.md)）**：
  - **新增深度专题（置于最后编号小节末尾、`图 9-54` 之后）**：增补 `#### NVIDIA Dynamo 与 Mooncake 全局 KV 缓存智能路由器及 KTransformers CPU-GPU 异构混合专家卸载架构`。
  - **新增 2 幅编号配图（`54 -> 56`）**：`图 9-55` (`ch09-nvidia-dynamo-mooncake-kv-aware-global-router`)、`图 9-56` (`ch09-ktransformers-gpu-mla-and-cpu-amx-routed-experts`)。
  - **新增 3 个生产级代码块（`3 -> 6`）**、**2 个编号算例（`5 -> 7`：`算例 9-6`、`算例 9-7`）**、**1 张表格（`11 -> 12`）**、**2 道思考题（`20 -> 22`：`**9-21 ...**`、`**9-22 ...**`，刷新 `ch09/manifest.json` 并通过 `ch09/verify.py`）**。
- **第 10 章（[`manuscripts/10-训练系统.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/10-%E8%AE%AD%E7%BB%83%E7%B3%BB%E7%BB%9F.md)）**：
  - **新增深度专题（置于最后编号小节末尾、`图 10-56` 之后）**：增补 `#### Megatron-Core (MCore) / TorchTitan 下 5D 混合并行与细粒度选择性重计算（Core Attention + MLA Up-Projection + SwiGLU）显存—吞吐联合优化`。
  - **新增 2 幅编号配图（`56 -> 58`）**：`图 10-57` (`ch10-selective-activation-recomputation-mla-and-swiglu`)、`图 10-58` (`ch10-megatron-core-torchtitan-5d-parallel-memory-breakdown`)。
  - **新增 2 个生产级代码块（`4 -> 6`）**、**2 个编号算例（`5 -> 7`：`算例 10-6`、`算例 10-7`）**、**1 张表格（`13 -> 14`）**、**2 道思考题（`20 -> 22`：`**10-21 ...**`、`**10-22 ...**`，刷新 `ch10/manifest.json` 并通过 `ch10/verify.py`）**。

### 3.6 Batch 6：第 11 章 & 第 12 章原位深化
- **第 11 章（[`manuscripts/11-资源调度与运行环境.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/11-%E8%B5%84%E6%BA%90%E8%B0%83%E5%BA%A6%E4%B8%8E%E8%BF%90%E8%A1%8C%E7%8E%AF%E5%A2%83.md)）**：
  - **新增深度专题（置于最后编号小节末尾、`图 11-57` 之后）**：增补 `#### 训练与在线推理解耦混部下的毫秒级显存休眠唤醒（vLLM Sleep Level 1/2 + CUDA CuMemMap）与 Serverless GPU 权重 RDMA 多播冷启动`。
  - **新增 2 幅编号配图（`57 -> 59`）**：`图 11-58` (`ch11-cumemmap-virtual-memory-sleep-wake-colocation`)、`图 11-59` (`ch11-serverless-gpu-rdma-multicast-cold-start`)。
  - **新增 3 个生产级代码块（`3 -> 6`）**、**2 个编号算例（`5 -> 7`：`算例 11-6`、`算例 11-7`）**、**1 张表格（`9 -> 10`）**、**2 道思考题（`19 -> 21`：`> **实验 11-20 · ...〔延伸〕**`、`> **实验 11-21 · ...〔延伸〕**`，刷新 `ch11/manifest.json` 并通过 `ch11/verify.py`）**。
- **第 12 章（[`manuscripts/12-端边云协同.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/12-%E7%AB%AF%E8%BE%B9%E4%BA%91%E5%8D%8F%E5%90%8C.md)）**：
  - **新增深度专题（置于最后编号小节末尾、`图 12-42` 之后）**：增补 `#### 端侧 PyTorch 原生运行时（ExecuTorch 编译器与 NPU 算子委派）与桌面级高带宽统一内存微型集群（Apple M4/M5 Ultra Thunderbolt 5 RDMA 环网与 DGX Spark GB10）`。
  - **新增 2 幅编号配图（`42 -> 44`）**：`图 12-43` (`ch12-executorch-aot-export-and-npu-delegate-pipeline`)、`图 12-44` (`ch12-unified-memory-desktop-cluster-tb5-rdma-ring`)。
  - **新增 3 个生产级代码块（`3 -> 6`）**、**2 个编号算例（`4 -> 6`：`算例 12-5`、`算例 12-6`）**、**1 张表格（`9 -> 10`）**、**2 道思考题（`19 -> 21`：`**拓展练习 12-20　...**`、`**拓展练习 12-21　...**`，同步更新 `ch12/verify.py` 第 166–168 行 `range(11, 22)` / `'exercises': 21`）**。

### 3.7 Batch 7：第 13 章 & 第 14 章原位深化（含新增实验 `13-04`、`14-03`）
- **第 13 章（[`manuscripts/13-AI 原生存储系统与数据底座.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/13-AI%20%E5%8E%9F%E7%94%9F%E5%AD%98%E5%82%A8%E7%B3%BB%E7%BB%9F%E4%B8%8E%E6%95%B0%E6%8D%AE%E5%BA%95%E5%BA%A7.md)）**：
  - **新增深度专题（置于 `13.4.2` 节末尾、`图 13-26` 之后）**：增补 `#### 分布式 KV Cache 存储池与对象存储冷热分层（LMCache / Mooncake Store 跨请求 Prefix 压缩与 GDS 异步预取）`。
  - **新增 2 幅编号配图（`26 -> 28`）**：`图 13-27` (`ch13-lmcache-prefix-kv-compression-and-gds-prefetch`)、`图 13-28` (`ch13-checkpoint-and-kv-unified-nvme-pool-qos`)。
  - **新增 3 个生产级代码块（`3 -> 6`）**、**1 张表格（`10 -> 11`）**、**2 道思考题（`17 -> 19`：`**练习 13-18**`、`**练习 13-19**`）**、**新增实验套件 [`experiments/ch13/13-04/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch13/13-04)**（分布式 KV Cache 压缩存储与 GDS 预取带宽仿真）。
- **第 14 章（[`manuscripts/14-模型与系统硬件协同设计.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/14-%E6%A8%A1%E5%9E%8B%E4%B8%8E%E7%B3%BB%E7%BB%9F%E7%A1%AC%E4%BB%B6%E5%8D%8F%E5%90%8C%E8%AE%BE%E8%AE%A1.md)）**：
  - **新增深度专题（置于 `14.4.2` 节末尾、`图 14-26` 之后）**：增补 `#### 面向 Vera Rubin 与 TPU v7 的下一代 MoE 超参数协同搜索（专家粒度、激活比、注意力混合比与芯片内存带宽比联合 Pareto 寻优）`。
  - **新增 2 幅编号配图（`26 -> 28`）**：`图 14-27` (`ch14-moe-granularity-vs-memory-bandwidth-pareto`)、`图 14-28` (`ch14-hardware-aware-neural-architecture-search-loop`)。
  - **新增 3 个生产级代码块（`3 -> 6`）**、**1 张表格（`11 -> 12`）**、**2 道思考题（`17 -> 19`：`**练习 14-18**`、`**练习 14-19**`）**、**新增实验套件 [`experiments/ch14/14-03/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch14/14-03)**（MoE 专家粒度与硬件字节/浮点比联合 Pareto 寻优器）。

### 3.8 Batch 8：第 15 章 & 第 16 章原位深化（含 `+24` 条脚注与新增实验 `15-03`、`16-03`）
- **第 15 章（[`manuscripts/15-吉瓦级 AI 数据中心、电力热学与跨地域协同训练.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/15-%E5%90%89%E7%93%A6%E7%BA%A7%20AI%20%E6%95%B0%E6%8D%AE%E4%B8%AD%E5%BF%83%E3%80%81%E7%94%B5%E5%8A%9B%E7%83%AD%E5%AD%A6%E4%B8%8E%E8%B7%A8%E5%9C%B0%E5%9F%9F%E5%8D%8F%E5%90%8C%E8%AE%AD%E7%BB%83.md)）**：
  - **新增深度专题（置于 `15.4.2` 节末尾、`图 15-24` 之后）**：增补 `#### 微通道芯片内液冷（Microfluidics Direct-to-Silicon Cooling，热通量 > 250 W/cm²）与吉瓦级园区孤岛微网（SMR + 燃气轮机 + 构网型储能 BESS）动态调频控制`。
  - **新增 2 幅编号配图（`24 -> 26`）**：`图 15-25` (`ch15-in-silicon-microfluidic-cooling-thermal-resistance`)、`图 15-26` (`ch15-gw-campus-smr-gas-bess-microgrid-frequency-control`)。
  - **新增 3 个生产级代码块（`3 -> 6`）**、**1 张表格（`11 -> 12`）**、**2 道思考题（`16 -> 18`：`**练习 15-17**`、`**练习 15-18**`）**、**新增 12 条权威学术/工程规范脚注（`9 -> 21`）**、**新增实验套件 [`experiments/ch15/15-03/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch15/15-03)**。
- **第 16 章（[`manuscripts/16-机密 AI 计算、可信执行环境与安全沙箱底座.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/16-%E6%9C%BA%E5%AF%86%20AI%20%E8%AE%A1%E7%AE%97%E3%80%81%E5%8F%AF%E4%BF%A1%E6%89%A7%E8%A1%8C%E7%8E%AF%E5%A2%83%E4%B8%8E%E5%AE%89%E5%85%A8%E6%B2%99%E7%AE%B1%E5%BA%95%E5%BA%A7.md)）**：
  - **新增深度专题（置于 `16.4.2` 节末尾、`图 16-24` 之后）**：增补 `#### 机密推理下的电磁/功耗/PCIe 时序侧信道防御（Constant-Time Token Padding & Traffic Shaping）与零知识机器学习证明（zkML / 可验证推理承诺）`。
  - **新增 2 幅编号配图（`24 -> 26`）**：`图 16-25` (`ch16-tee-timing-side-channel-traffic-shaping`)、`图 16-26` (`ch16-verifiable-inference-optimistic-and-zkml-proof`)。
  - **新增 3 个生产级代码块（`3 -> 6`）**、**1 张表格（`11 -> 12`）**、**2 道思考题（`16 -> 18`：`**练习 16-17**`、`**练习 16-18**`）**、**新增 12 条权威学术脚注（`7 -> 19`）**、**新增实验套件 [`experiments/ch16/16-03/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch16/16-03)**。

### 3.9 Batch 9：第 17 章 & 第 18 章原位深化（含 `+24` 条脚注与新增实验 `17-03`、`18-03`）
- **第 17 章（[`manuscripts/17-多模态、视频生成（DiT）与世界模型基础设施.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/17-%E5%A4%9A%E6%A8%A1%E6%80%81%E3%80%81%E8%A7%86%E9%A2%91%E7%94%9F%E6%88%90%EF%BC%88DiT%EF%BC%89%E4%B8%8E%E4%B8%96%E7%95%8C%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD.md)）**：
  - **新增深度专题（置于 `17.4.2` 节末尾、`图 17-20` 之后）**：增补 `#### 自回归流式因果视频世界模型（CausVid / Oasis / Matrix-Game 20 FPS 实时交互生成）的块因果 KV 缓存滚动窗口与跨帧特征复用（TeaCache / PAB）`。
  - **新增 2 幅编号配图（`20 -> 22`）**：`图 17-21` (`ch17-causal-streaming-world-model-20fps-kv-rolling`)、`图 17-22` (`ch17-teacache-timestep-embedding-aware-activation-reuse`)。
  - **新增 1 个生产级代码块（`8 -> 9`）**、**1 个编号算例（`7 -> 8`：`算例 17-8`）**、**1 张表格（`6 -> 7`）**、**2 道思考题（`14 -> 16`：`**练习 17-15**`、`**练习 17-16**`）**、**新增 12 条权威学术脚注（`3 -> 15`）**、**新增实验套件 [`experiments/ch17/17-03/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch17/17-03)**。
- **第 18 章（[`manuscripts/18-大规模后训练（RLVR）、异步强化学习与推理时扩展系统.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/18-%E5%A4%A7%E8%A7%84%E6%A8%A1%E5%90%8E%E8%AE%AD%E7%BB%83%EF%BC%88RLVR%EF%BC%89%E3%80%81%E5%BC%82%E6%AD%A5%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E4%B8%8E%E6%8E%A8%E7%90%86%E6%97%B6%E6%89%A9%E5%B1%95%E7%B3%BB%E7%BB%9F.md)）**：
  - **新增深度专题（置于 `18.4.2` 节末尾、`图 18-20` 之后）**：增补 `#### 多轮工具交互强化学习（Multi-Turn Agentic RLVR，如 SWE-RL / DeepResearch-RL）的环境状态检查点分叉、长轨迹分段优势估计与离策略重要性采样截断（V-trace / TIS）`。
  - **新增 2 幅编号配图（`20 -> 22`）**：`图 18-21` (`ch18-multi-turn-agentic-rlvr-trajectory-tree-forking`)、`图 18-22` (`ch18-asynchronous-off-policy-importance-sampling-clipping`)。
  - **新增 1 个生产级代码块（`9 -> 10`）**、**1 个编号算例（`9 -> 10`：`算例 18-10`）**、**1 张表格（`8 -> 9`）**、**2 道思考题（`14 -> 16`：`**练习 18-15**`、`**练习 18-16**`）**、**新增 12 条权威学术脚注（`7 -> 19`）**、**新增实验套件 [`experiments/ch18/18-03/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch18/18-03)**。

### 3.10 Batch 10：第 19 章 & 第 20 章原位深化（含 `+24` 条脚注与新增实验 `19-03`、`20-03`）
- **第 19 章（[`manuscripts/19-具身智能（VLA）、实时控制与大规模物理仿真底座.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/19-%E5%85%B7%E8%BA%AB%E6%99%BA%E8%83%BD%EF%BC%88VLA%EF%BC%89%E3%80%81%E5%AE%9E%E6%97%B6%E6%8E%A7%E5%88%B6%E4%B8%8E%E5%A4%A7%E8%A7%84%E6%A8%A1%E7%89%A9%E7%90%86%E4%BB%BF%E7%9C%9F%E5%BA%95%E5%BA%A7.md)）**：
  - **新增深度专题（置于 `19.4.2` 节末尾、`图 19-18` 之后）**：增补 `#### 端到端自动驾驶世界模型闭环仿真器（3D Gaussian Splatting 传感器高保真闭环重渲染）与车载双 Orin/Thor 异构 SoC 安全降级接管（Safety Fallback Watchdog）实时调度`。
  - **新增 2 幅编号配图（`18 -> 20`）**：`图 19-19` (`ch19-3dgs-neural-sensor-closed-loop-re-simulation`)、`图 19-20` (`ch19-dual-thor-soc-safety-fallback-watchdog-pipeline`)。
  - **新增 1 个生产级 Python 代码块（Python 代码块 `7 -> 8`，加上章内既有 1 个 C++ 代码块，总类型化代码块 `8 -> 9`，同步更新 `manuscripts/ch19/verify.py` 第 53 行为 `len(code_blocks) == 9 and len(re.findall(r'```', s)) == 18`）**、**1 个编号算例（`6 -> 7`：`算例 19-7`）**、**1 张表格（`8 -> 9`）**、**2 道思考题（`12 -> 14`：`**练习 19-13**`、`**练习 19-14**`）**、**新增 12 条权威学术脚注（`9 -> 21`）**、**新增实验套件 [`experiments/ch19/19-03/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch19/19-03)**。
- **第 20 章（[`manuscripts/20-十万卡集群可靠性工程、静默数据损坏（SDC）防御与基准评测科学.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/20-%E5%8D%81%E4%B8%87%E5%8D%A1%E9%9B%86%E7%BE%A4%E5%8F%AF%E9%9D%A0%E6%80%A7%E5%B7%A5%E7%A8%8B%E3%80%81%E9%9D%99%E9%BB%98%E6%95%B0%E6%8D%AE%E6%8D%9F%E5%9D%8F%EF%BC%88SDC%EF%BC%89%E9%98%B2%E5%BE%A1%E4%B8%8E%E5%9F%BA%E5%87%86%E8%AF%84%E6%B5%8B%E7%A7%91%E5%AD%A6.md)）**：
  - **新增深度专题（置于 `20.4.2` 节末尾、`图 20-18` 之后）**：增补 `#### 无中断弹性重配置（Parachute / Oobleck 预实例化流水线模板秒级容错降级）与面向推理系统 Goodput 的动态负载压测生成科学（Poisson 突发请求与长尾长度分布建模）`。
  - **新增 2 幅编号配图（`18 -> 20`）**：`图 20-19` (`ch20-oobleck-pre-instantiated-pipeline-template-failover`)、`图 20-20` (`ch20-goodput-sla-under-bursty-poisson-traffic`)。
  - **新增 1 个生产级代码块（`6 -> 7`）**、**1 个编号算例（`6 -> 7`：`算例 20-7`）**、**1 张表格（`9 -> 10`）**、**2 道思考题（`12 -> 14`：`**练习 20-13**`、`**练习 20-14**`）**、**新增 12 条权威学术脚注（`10 -> 22`）**、**新增实验套件 [`experiments/ch20/20-03/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch20/20-03)**。

### 3.11 Batch 11：第 21 章 & 第 22 章原位深化（含 `+24` 条脚注与新增实验 `21-03`、`22-03`）
- **第 21 章（[`manuscripts/21-自主智能体基础设施（Agentic Infra）、Computer Use 沙箱与分层 KV 记忆底座.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/21-%E8%87%AA%E4%B8%BB%E6%99%BA%E8%83%BD%E4%BD%93%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD%EF%BC%88Agentic%20Infra%EF%BC%89%E3%80%81Computer%20Use%20%E6%B2%99%E7%AE%B1%E4%B8%8E%E5%88%86%E5%B1%82%20KV%20%E8%AE%B0%E5%BF%86%E5%BA%95%E5%BA%A7.md)）**：
  - **新增深度专题（置于 `21.4.2` 节末尾、`图 21-16` 之后）**：增补 `#### 智能体长程浏览器/GUI 视觉动作流（WebArena / OSWorld）的增量 DOM+截图双通道编码缓存与多智能体工作流有向无环图（DAG）关键路径投机预执行（Speculative Tool Execution）`。
  - **新增 2 幅编号配图（`16 -> 18`）**：`图 21-17` (`ch21-speculative-tool-execution-and-rollback-dag`)、`图 21-18` (`ch21-gui-incremental-region-of-interest-vision-token-cache`)。
  - **新增 1 个生产级代码块（`6 -> 7`）**、**2 个编号算例（`5 -> 7`：`算例 21-6`、`算例 21-7`）**、**1 张表格（`6 -> 7`）**、**2 道思考题（`10 -> 12`：`**练习 21-11**`、`**练习 21-12**`）**、**新增 12 条权威学术脚注（`8 -> 20`）**、**新增实验套件 [`experiments/ch21/21-03/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch21/21-03)**。
- **第 22 章（[`manuscripts/22-全双工实时语音（Full-Duplex Speech）、神经音频编解码与原生全模态流式底座.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/22-%E5%85%A8%E5%8F%8C%E5%B7%A5%E5%AE%9E%E6%97%B6%E8%AF%AD%E9%9F%B3%EF%BC%88Full-Duplex%20Speech%EF%BC%89%E3%80%81%E7%A5%9E%E7%BB%8F%E9%9F%B3%E9%A2%91%E7%BC%96%E8%A7%A3%E7%A0%81%E4%B8%8E%E5%8E%9F%E7%94%9F%E5%85%A8%E6%A8%A1%E6%80%81%E6%B5%81%E5%BC%8F%E5%BA%95%E5%BA%A7.md)）**：
  - **新增深度专题（置于 `22.4.2` 节末尾、`图 22-16` 之后）**：增补 `#### 多人会议空间音频分离、实时音视频数字人唇形同步（3D Avatar Lip-Sync < 40 ms 音画对齐抖动缓冲）与跨地域全球边缘 WebRTC SFU 媒体网格就近卸载`。
  - **新增 2 幅编号配图（`16 -> 18`）**：`图 22-17` (`ch22-realtime-avatar-lip-sync-audio-video-jitter-alignment`)、`图 22-18` (`ch22-global-edge-webrtc-sfu-to-gpu-duplex-mesh`)。
  - **新增 1 个生产级代码块（`6 -> 7`）**、**2 个编号算例（`5 -> 7`：`算例 22-6`、`算例 22-7`）**、**1 张表格（`6 -> 7`）**、**2 道思考题（`10 -> 12`：`**练习 22-11**`、`**练习 22-12**`）**、**新增 12 条权威学术脚注（`8 -> 20`）**、**新增实验套件 [`experiments/ch22/22-03/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch22/22-03)**。

### 3.12 Batch 12：第 23 章 & 第 24 章原位深化（含 `+24` 条脚注与新增实验 `23-03`、`24-03`）
- **第 23 章（[`manuscripts/23-百亿亿次 Token 数据工程、GPU 加速模糊去重与合成数据工厂底座.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/23-%E7%99%BE%E4%BA%BF%E4%BA%BF%E6%AC%A1%20Token%20%E6%95%B0%E6%8D%AE%E5%B7%A5%E7%A8%8B%E3%80%81GPU%20%E5%8A%A0%E9%80%9F%E6%A8%A1%E7%B3%8A%E5%8E%BB%E9%87%8D%E4%B8%8E%E5%90%88%E6%88%90%E6%95%B0%E6%8D%AE%E5%B7%A5%E5%8E%82%E5%BA%95%E5%BA%A7.md)）**：
  - **新增深度专题（置于 `23.4.2` 节末尾、`图 23-16` 之后）**：增补 `#### 百万册级复杂学术 PDF / 扫描件视觉版面解析与公式还原管线（MinerU / olmOCR GPU 吞吐量优化）与代码仓库级跨文件依赖拓扑排序（Repo-Level AST Dependency Packing，支撑 64k–128k 代码预训练上下文）`。
  - **新增 2 幅编号配图（`16 -> 18`）**：`图 23-17` (`ch23-mineru-olmocr-pdf-layout-and-formula-gpu-pipeline`)、`图 23-18` (`ch23-repo-level-ast-dependency-topological-packing`)。
  - **新增 2 个生产级代码块（`6 -> 8`）**、**2 个编号算例（`5 -> 7`：`算例 23-6`、`算例 23-7`）**、**1 张表格（`6 -> 7`）**、**2 道思考题（`10 -> 12`：`**练习 23-11**`、`**练习 23-12**`）**、**新增 12 条权威学术脚注（`8 -> 20`）**、**新增实验套件 [`experiments/ch23/23-03/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch23/23-03)**。
- **第 24 章（[`manuscripts/24-科学智能（AI for Science）基础设施：生物大分子、百万碱基基因组与物理场网格系统.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/24-%E7%A7%91%E5%AD%A6%E6%99%BA%E8%83%BD%EF%BC%88AI%20for%20Science%EF%BC%89%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD%EF%BC%9A%E7%94%9F%E7%89%A9%E5%A4%A7%E5%88%86%E5%AD%90%E3%80%81%E7%99%BE%E4%B8%87%E7%A2%B1%E5%9F%BA%E5%9F%BA%E5%9B%A0%E7%BB%84%E4%B8%8E%E7%89%A9%E7%90%86%E5%9C%BA%E7%BD%91%E6%A0%BC%E7%B3%BB%E7%BB%9F.md)）**：
  - **新增深度专题（置于 `24.4.2` 节末尾、`图 24-16` 之后）**：增补 `#### 托卡马克核聚变等离子体磁流体（MHD）撕裂模毫秒级实时强化学习控制推理底座与单细胞亿级转录组图谱（scGPT / Geneformer）稀疏基因表达矩阵加速`。
  - **第 24 章“第四部分收束”与第 30 章“全书大一统收束”的平滑过渡规范**：
    - `24.4` 节将定位调整为 **“前四部分（第 1–24 章：硬件、网络、分布式训推与科学负载篇）阶段性综合与向第五部分（第 25–30 章）的桥梁”**；
    - 同步更新 [`manuscripts/ch24/figure-data.json`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/ch24/figure-data.json)、[`manuscripts/ch24/verify.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/ch24/verify.py) 与 [`experiments/ch24/24-02/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch24/24-02) 中的配图/实验账本锚点（在 Batch 12 完成时 反映 `Ch01–Ch24` 扩充后的前 24 章阶段总账本：`883` 幅编号章节配图 + `1` 幅导读图 = `884` 幅图、`147` 个实验套件，并在第 30 章 `30.4` 节与 [`experiments/ch30/30-03/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch30/30-03) 中完成全书 30 章 `979 + 1 = 980` 幅图与 `165` 个实验套件的终极总收束）。
  - **新增 2 幅编号配图（`16 -> 18`）**：`图 24-17` (`ch24-tokamak-plasma-tearing-mode-sub-ms-rl-control`)、`图 24-18` (`ch24-single-cell-atlas-sparse-gene-expression-dataloader`)。
  - **新增 2 个生产级代码块（`6 -> 8`）**、**2 个编号算例（`5 -> 7`：`算例 24-6`、`算例 24-7`）**、**1 张表格（`6 -> 7`）**、**2 道思考题（`10 -> 12`：`**练习 24-11**`、`**练习 24-12**`）**、**新增 12 条权威学术脚注（`8 -> 20`）**、**新增实验套件 [`experiments/ch24/24-03/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch24/24-03)**。

---

## 4. 新建第 25–30 章（第五部分：全场景前沿范式、极客源码解剖与自演进系统篇）完整架构与数学规范

新建的第 25–30 章每章均严格遵循统一的高标准工程契约：
- **章节篇幅**：$\ge 1,000$ 行中文技术散文；
- **大纲结构**：4 个二级标题（`## XX.1`–`## XX.4`）+ 8 个三级标题（每节 2 个 `### XX.Y.1`、`### XX.Y.2`）+ 末节 `## 本章小结`（无数字前缀）；
- **出版级 420pt 配图**：每章 **16 幅**（`图 XX-1` 至 `图 XX-16`，`.svg` / `.png` / `.pdf` 三格式齐全，`svg.fonttype == 'path'`，零内部图号，零文字重叠，共 `96` 幅新图）；
- **生产级系统代码块**：每章 **6 个**带类型标识的生产级 Python 代码块（共 `36` 个代码块）；
- **标准化引用块算例**：每章 **6 个** `> **算例 XX-1..6（...）**`（共 `36` 个算例）；
- **标准化斜体表题**：每章 **6 个** `*表 XX-1..6　...*`（含章末 `*表 XX-6　第 XX 章课后思考题核心锁定参数与标准答案速查表*`，共 `36` 张表）；
- **课后思考题与详尽解析**：每章 **10 道** `**练习 XX-1..10**` + `### 参考解答与详尽推导`（共 `60` 道新习题）；
- **学术文献脚注**：每章 **15 条** 一手顶会/工业白皮书脚注 `[^chXX-1]`–`[^chXX-15]`（共 `90` 条新脚注）；
- **零依赖可运行实验套件**：每章 **3 个** 独立实验目录 `experiments/chXX/XX-01..03`（共 `18` 个新实验目录）。

---

### 4.1 第 25 章：`manuscripts/25-生成式推荐系统（Generative Recommenders）、万亿参数稀疏表（DLRM）与十亿级 GPU 向量检索引擎.md`

#### 4.1.1 四节八小节大纲
- `## 25.1 万亿参数搜广推稀疏嵌入表（DLRM / TorchRec）的三级分层存储与通信底座`
  - `### 25.1.1 幂律特征访问（Zipf 分布）下的 HBM—DRAM（CXL）—NVMe 三级动态嵌入缓存与无锁哈希表`
  - `### 25.1.2 稀疏查表全互换（All-to-All Embedding Lookup）与稠密特征交互的 Roofline 拐点及分桶负载均衡`
- `## 25.2 从级联漏斗到生成式推荐系统（Generative Recommenders：Meta HSTU 与 OneRec）`
  - `### 25.2.1 层次化序列转导单元（HSTU）的点积非归一化注意力内核与随机序列长度压缩（Stochastic Length）`
  - `### 25.2.2 候选集微批前缀摊销算法（M-FALCON）：单次用户状态编码并行打分千级候选物品的算力跃迁`
- `## 25.3 十亿至百亿级向量数据库与 GPU 图索引硬件加速（NVIDIA cuVS CAGRA / Milvus / FAISS）`
  - `### 25.3.1 为什么十亿级 CPU HNSW 建图需要十小时？GPU CAGRA Warp 级并行局部图构建与两阶段度剪枝`
  - `### 25.3.2 乘积量化（IVF-PQ / RaBitQ）非对称距离查找表（LUT）在 GPU 片上共享内存的寄存器级融合`
- `## 25.4 SSD 驻留百亿向量图检索（DiskANN / Vamana）、实时流式入库与 GraphRAG 混合检索系统`
  - `### 25.4.1 DiskANN 4 KB 磁盘扇区物理对齐（向量与近邻表同页打包）与 io_uring / GDS 异步波束搜索（Beam Search）`
  - `### 25.4.2 在线流式增量索引合并（FreshDiskANN）、多跳社区图谱检索（GraphRAG）与尾延迟 SLA 保障`
- `## 本章小结`

#### 4.1.2 核心数学方程与锁定数值锚点（Locked Numerical Anchors）
1. **万亿参数 DLRM 三级嵌入缓存命中率与有效带宽放大**：
   - 总嵌入表规模 $V_{\mathrm{total}} = 10.0\ \mathrm{TB}$（$2 \times 10^{10}$ 个特征 ID，嵌入维数 $D = 128$；每行包含 $128 \times 2\ \mathrm{B} = 256\ \mathrm{B}$ FP16 权重、$128 \times 1.8\ \mathrm{B} = 230\ \mathrm{B}$ 量化二阶动量状态与 $14\ \mathrm{B}$ LFUCache 时间戳/版本元数据，合计精确 $500\ \mathrm{B/row}$）。特征静态频率服从幂律分布 $p(k) \propto k^{-\alpha}$（$\alpha = 1.12$），叠加在线滑动窗口（`15 min`）时间局部性漂移因子 $\eta_{\mathrm{drift}} = 0.038$ 后：
     - 第一层 GPU HBM 热表缓存（容量 $320\ \mathrm{GB}$，占总表 $3.2\%$）：捕获 **$82.0\%$** 的特征访问（延迟 $0.35\ \mu\mathrm{s}$，聚合带宽 $24.0\ \mathrm{TB/s}$）；
     - 第二层 Host CXL/DDR5 温表缓存（容量 $2.0\ \mathrm{TB}$，占总表 $20.0\%$）：捕获 **$15.5\%$** 的特征访问（延迟 $1.8\ \mu\mathrm{s}$，聚合带宽 $1.2\ \mathrm{TB/s}$）；
     - 第三层 NVMe SSD 冷表存储（容量 $10.0\ \mathrm{TB}$）：仅穿透剩余 **$2.5\%$** 的长尾请求（通过 `GDS` 批量合并读取，延迟 $65.0\ \mu\mathrm{s}$）；
   - 加权平均查表延迟从纯 SSD 的 $65.0\ \mu\mathrm{s}$ 降至 **`2.19 us`**，等效吞吐提升 **`29.7x`**。
2. **生成式推荐 `Meta HSTU` 与 `M-FALCON` 微批候选摊销模型**：
   - 用户历史行为序列长度 $L_{\mathrm{hist}} = 4,096$，隐藏维 $d = 1,024$，层数 $N_L = 16$（每层含 $U, V, Q, K$ 四个线性投影与点积聚合注意力，模型有效参数量 $N_{\mathrm{param}} = 168.0\ \mathrm{M}$，包含二次注意力项在内的单次历史序列前向计算量为 $1.376\ \mathrm{TFLOPs}$），单次请求需对 $M = 1,024$ 个召回候选物品打分：
     - **传统逐物品独立交叉打分**：将每个候选物品分别拼接在历史序列末尾独立前向 $1,024$ 次，总计算量 $\mathrm{FLOPs}_{\mathrm{naive}} = 1,024 \times 1.376\ \mathrm{TFLOPs} = \mathbf{1,409.3\ \mathrm{TFLOPs}}$，难以在 `15 ms` SLA 内完成；
     - **`M-FALCON` 因果掩码前缀共享 + 候选微批并行**：历史序列 $L_{\mathrm{hist}} = 4,096$ 仅编码 **$1$ 次**（耗时 `1.376 TFLOPs`），随后将 $M = 1,024$ 个候选物品打包在序列尾部共享同一份历史 KV 状态并行前向（交叉注意力掩码禁止候选物品互相看见，候选打分阶段仅需 `30.084 TFLOPs`），单请求总计算量从 `1,409.3 TFLOPs` 骤降至 **`31.46 TFLOPs`**，实现 **`44.80x` 算力缩减**，单卡 H100 吞吐达 **`485 QPS`（即每秒打分 `496,640` 个候选物品，P99 延迟 `11.8 ms`）**。
3. **十亿向量索引 `GPU cuVS CAGRA` vs. `CPU HNSW` 与 `DiskANN` 4 KB 扇区对齐**：
   - `10 亿`（$10^9$）条 $D = 768$ 维 FP16 向量（原始向量体积 `1.536 TB`）：
     - 双路 128 核 CPU 构建 `HNSW`（$M=32, \mathrm{efConstruction}=200$）耗时 **`11.50 hours`**；8 卡 H100 运行 `cuVS CAGRA`（先构建 $k$-NN 候选图再执行两阶段路径绕行剪枝与反向边注入）耗时仅 **`0.68 hours`（`40.8 min`）**，建图提速 **`16.91x`**；
     - `DiskANN` 将每条向量压缩为 `64 B` PQ 码常驻主机内存（`10 亿` 条仅占 **`64.0 GB`**），而在 NVMe SSD 上将每条原始 `768` 维 FP16 向量（`1,536 B`）与 $R = 64$ 个 `uint32` 邻居 ID（`256 B`）紧凑打包为 `1,792 B`（每 `4,096 B` 物理扇区容纳 2 个完整图节点），在 Beam Width $W = 4$、迭代跳数 $S = 6$ 跳下，单次查询仅触发 $W \times S = \mathbf{24}$ 次 4 KB 随机读，在 4 块 PCIe 5.0 NVMe SSD（聚合 `6.0M IOPS`）上实现 **`210,000 QPS`、Recall@10 = `95.4%`、P99 延迟 `2.85 ms`**。

#### 4.1.3 第 25 章 16 幅配图清单（`manuscripts/ch25/`）
1. `ch25-trillion-param-dlrm-three-tier-hbm-cxl-ssd-embedding`
2. `ch25-zipf-power-law-hot-embedding-cache-hit-curve`
3. `ch25-torchrec-all-to-all-sparse-lookup-and-bucket-sharding`
4. `ch25-cascade-dlrm-vs-generative-recommender-hstu-paradigm`
5. `ch25-hstu-pointwise-aggregated-attention-and-stochastic-length`
6. `ch25-m-falcon-candidate-microbatch-prefix-amortization`
7. `ch25-hstu-online-event-streaming-and-delta-kv-update`
8. `ch25-cpu-hnsw-vs-gpu-cuvs-cagra-graph-build-pipeline`
9. `ch25-cagra-warp-splitting-distance-and-two-stage-pruning`
10. `ch25-ivf-pq-shared-memory-lut-asymmetric-distance-kernel`
11. `ch25-diskann-4kb-sector-aligned-vector-and-neighbor-layout`
12. `ch25-diskann-io-uring-gds-beam-search-pipeline`
13. `ch25-freshdiskann-streaming-insert-delete-consolidation`
14. `ch25-graphrag-hierarchical-community-and-vector-hybrid-retrieval`
15. `ch25-search-ads-rec-gpu-fleet-tco-and-sla-roofline`
16. `ch25-generative-recommendation-and-vector-engine-synthesis`

#### 4.1.4 第 25 章代码块（6 个）、算例（6 个）、表格（6 张）与思考题（10 道）精确规格清单
- **6 个生产级 Python 代码块**：
  1. `ThreeTierEmbeddingCacheSimulator`：万亿参数 DLRM 幂律 + 时间漂移三级（HBM / CXL / NVMe GDS）嵌入缓存命中率与延迟求解器；
  2. `TorchRecSparseAllToAllPlanner`：稀疏嵌入表按访问热度与行维度的分桶负载均衡与 `All-to-All` 通信张量打包器；
  3. `HSTUPointwiseAttentionAndStochasticLength`：`Meta HSTU` 非归一化点积注意力内核与随机序列长度压缩（Stochastic Length）算子实现；
  4. `MFalconCandidateMicrobatchEngine`：`M-FALCON` 用户历史前缀 KV 共享与千级候选物品因果掩码并行打分引擎（验证 `44.80x` 算力缩减）；
  5. `CagraTwoStageGraphPruner`：GPU `cuVS CAGRA` 局部 $k$-NN 图两阶段路径绕行剪枝、反向边注入与 Shared Memory PQ-LUT 距离核模拟器；
  6. `DiskAnnSectorAlignedBeamSearch`：`DiskANN` `4,096 B` 物理扇区双节点对齐打包、`io_uring` 异步波束搜索与 `FreshDiskANN` 流式合并引擎。
- **6 个标准化编号算例**：
  - `算例 25-1`：`10.0 TB` 稀疏表在 $\alpha = 1.12, \eta_{\mathrm{drift}} = 0.038$ 下三级缓存（`82.0% / 15.5% / 2.5%`）加权查表延迟 `2.19 us` 推导；
  - `算例 25-2`：64 卡 TorchRec 稀疏 `All-to-All` 查表与稠密交叉网络并发重叠的 Roofline 带宽盈亏平衡点核算；
  - `算例 25-3`：`HSTU` 随机序列长度压缩（$\beta = 0.75$）将长尾用户 `8,192` 行为序列复杂度降低 `56.2%` 的精度—算力账本；
  - `算例 25-4`：`M-FALCON` 单请求 `4,096` 历史词元 + `1,024` 候选物品从 `1,409.3 TFLOPs` 降至 `31.46 TFLOPs`（`44.80x`）及单卡 `485 QPS` 核算；
  - `算例 25-5`：`10 亿` 条 `768` 维向量在 8 卡 H100 上运行 `cuVS CAGRA`（`40.8 min`）对比双路 128 核 CPU `HNSW`（`11.50 h`，`16.91x` 提速）拆解；
  - `算例 25-6`：`DiskANN` `64.0 GB` 主机 PQ 码 + `4,096 B` 扇区双节点对齐在 $W=4, S=6$（`24` 次随机读）下实现 `210,000 QPS` 吞吐核算。
- **6 张标准化斜体表题**：`表 25-1`（DLRM 三级存储介质参数与命中率对比）、`表 25-2`（传统级联推荐漏斗 vs. 生成式推荐 `HSTU/OneRec` 架构对比）、`表 25-3`（`M-FALCON` 不同候选批大小下的算术强度与延迟对比）、`表 25-4`（十亿级向量索引 `HNSW` / `IVF-PQ` / `CAGRA` / `DiskANN` 性能矩阵）、`表 25-5`（`GraphRAG` 多跳社区摘要与向量混合检索延迟分解）、`表 25-6`（第 25 章课后思考题核心锁定参数与标准答案速查表）。
- **10 道课后思考题（`**练习 25-1**` 至 `**练习 25-10**`）**：覆盖幂律缓存未命中率积分推导、无锁 Cuckoo 哈希冲突退化、All-to-All 热度倾斜分桶、HSTU 取代 Softmax 的数值原因、M-FALCON 掩码 FLOPs 精算、在线实时 Delta KV 增量更新、CAGRA Warp Splitting 距离计算、IVF-PQ 共享内存 Bank 冲突消除、DiskANN 扇区跨页未对齐惩罚以及 FreshDiskANN 墓碑合并带宽分配。

---

### 4.2 第 26 章：`manuscripts/26-后自回归生成范式基础设施：扩散大语言模型（dLLM）、字节级无分词架构（BLT）与测试时记忆网络（Titans-TTT）.md`

#### 4.2.1 四节八小节大纲
- `## 26.1 扩散大语言模型（Diffusion LLM：Mercury / LLaDA / Block Diffusion）并行去噪推理底座`
  - `### 26.1.1 从逐词元自回归（I = 1 FLOP/B）到块掩码并行去噪（I = 64 FLOP/B）的 Roofline 相变与千词元每秒吞吐`
  - `### 26.1.2 半自回归块扩散（Block Diffusion）的块间 KV 缓存固化、块内双向注意力与动态置信度重掩码调度`
- `## 26.2 字节级无分词动态分块架构（Byte Latent Transformer, BLT）的变长张量系统`
  - `### 26.2.1 告别 256k 词表瓶颈：轻量级字节条件熵模型在线动态分块（Entropy Patching）与算力按需分配`
  - `### 26.2.2 局部字节编码器—全局潜空间 Transformer 的非规则交叉注意力（Ragged Cross-Attention）打包内核`
- `## 26.3 测试时训练与神经长期记忆网络（Google Titans / TTT-Linear / TTT-MLP）的片上快速权重更新`
  - `### 26.3.1 将上下文记忆建模为在线梯度下降：惊奇度驱动动量更新（Surprise Momentum）与权重衰减遗忘门`
  - `### 26.3.2 分块对偶形式（Chunk-wise Dual Form `ttt_chunk_scan`）：把串行梯度步转化为 Tensor Core 矩阵乘`
- `## 26.4 连续潜空间推理（Coconut）与动态计算深度路由（Mixture-of-Depths / Early-Exit）运行时`
  - `### 26.4.1 连续思维链（Coconut）的隐藏状态零拷贝回环反馈与多路径潜空间隐式波束搜索`
  - `### 26.4.2 词元级动态跳层（Mixture-of-Depths）引发的 GPU Warp 分支发散与流紧缩（Stream Compaction）重聚`
- `## 本章小结`

#### 4.2.2 核心数学方程与锁定数值锚点（Locked Numerical Anchors）
1. **块扩散语言模型（`Block Diffusion LLM`）vs. 自回归语言模型（`AR LLM`）的解码 Roofline 相变**：
   - 给定参数量 $N = 32\ \mathrm{B}$（FP8 权重占 `32.0 GB`）的模型，在单张 H100（HBM 带宽 $B_{\mathrm{mem}} = 3,350\ \mathrm{GB/s}$，FP8 Tensor Core 有效算力 $F_{\mathrm{peak}} = 1,200\ \mathrm{TFLOPS}$，拐点算术强度 $I^* \approx 358\ \mathrm{FLOP/Byte}$）上生成 $L_{\mathrm{gen}} = 512$ 个词元（并发批大小 $B = 1$）：
     - **标准自回归（AR）解码**：共需串行执行 $512$ 步，每步仅生成 $1$ 个词元，单步权重读取 $32.0\ \mathrm{GB}$，耗时 $t_{\mathrm{step}}^{\mathrm{AR}} \approx \frac{32.0\ \mathrm{GB}}{3,350\ \mathrm{GB/s}} + 0.45\ \mathrm{ms} = 10.00\ \mathrm{ms}$，生成速度仅为 **`100.0 tok/s`**（总耗时 `5.12 s`）；
     - **半自回归块扩散（Block Diffusion，块大小 $K = 64$，每块自适应去噪步数 $T_{\mathrm{denoise}} = 10$ 步）**：生成 $512$ 个词元仅需 $512 / 64 = 8$ 个块，共计 $8 \times 10 = \mathbf{80}$ 次模型前向（权重读取次数减少 **$6.40x$**），每次前向同时并行处理 $K = 64$ 个词元（单次前向耗时仅 $10.80\ \mathrm{ms}$），总耗时降至 **`0.864 s`**，单请求有效生成速度跃升至 **`592.6 tok/s`**（提速 **`5.93x`**；在 `8B` 模型上更可达 **`1,420 tok/s`**）。
2. **`Byte Latent Transformer (BLT)` 字节熵动态分块与 LM Head 显存/算力节省**：
   - 当字节流条件熵 $H(b_t \mid b_{<t}) > \tau_H = 1.35\ \mathrm{bits}$ 时切分新 Patch，平均每 Patch 包含 $p_{\mathrm{avg}} = 4.5\ \text{bytes}$：
     - 彻底移除词表大小 $V = 256,000, d = 4,096$ 的输入 Embedding 与输出 LM Head 矩阵（节省 $2 \times 256,000 \times 4,096 \times 2\ \mathrm{B} = \mathbf{4.194\ \mathrm{GB}}$ 常驻显存及每词元 `2.10 GFLOPs` 词表投影计算）；
     - 在可预测性高的代码缩进、HTML 标签或重复模板段，动态 Patch 跨度自动拉长至 $12\text{–}16\ \text{bytes}$，使骨干潜空间 Transformer 的序列长度缩短 **`2.67x`**，端到端推理 FLOPs 降低 **`52.4%`**。
3. **测试时记忆网络（`Titans` / `TTT-Linear`）分块对偶化（Chunk Size $b = 64$）加速**：
   - 快速权重矩阵 $W_t \in \mathbb{R}^{d_{\mathrm{head}} \times d_{\mathrm{head}}}$（$d_{\mathrm{head}} = 128$），在线更新律 $W_t = \alpha_t W_{t-1} - \eta_t \nabla_W \|W_{t-1} k_t - v_t\|_2^2$：
     - 若逐词元串行更新，每词元都需在寄存器与 SRAM 间读写 $128 \times 128$ 矩阵，算术强度仅 $I = 1.5\ \mathrm{FLOP/B}$（CUDA Core 受限，吞吐仅 `42k tok/s`）；
     - 采用块大小 $b = 64$ 的分块矩阵对偶形式 $W_{t+b} = \alpha^b W_t - 2\eta (W_t K_{[t:t+b]} - V_{[t:t+b]}) K_{[t:t+b]}^\top$，将 64 步逐词元外积融合为 `[128, 64] x [64, 128]` 的 Tensor Core `MMA` 指令，算术强度提升至 **`48.0 FLOP/B`**，内核吞吐飙升至 **`1,033k tok/s`（提速 `24.60x`）**，且在 `200 万`（$2 \times 10^6$）词元上下文下快速权重显存恒定为 **`16.0 MB / layer`（$\mathcal{O}(1)$ 零膨胀）**。

#### 4.2.3 第 26 章 16 幅配图清单（`manuscripts/ch26/`）
1. `ch26-autoregressive-vs-diffusion-llm-roofline-phase-transition`
2. `ch26-block-diffusion-semi-autoregressive-kv-cache-and-mask`
3. `ch26-dynamic-confidence-threshold-remasking-scheduler`
4. `ch26-dllm-speculative-parallel-verification-hybrid-engine`
5. `ch26-bpe-tokenizer-bottleneck-vs-blt-byte-entropy-patching`
6. `ch26-blt-local-byte-encoder-global-latent-cross-attention`
7. `ch26-blt-ragged-patch-packing-and-load-balancing`
8. `ch26-titans-short-term-attention-and-long-term-neural-memory`
9. `ch26-ttt-token-by-token-vs-chunkwise-dual-form-tensor-core`
10. `ch26-titans-2m-context-constant-memory-vs-kv-cache-growth`
11. `ch26-coconut-continuous-latent-thought-zero-copy-loop`
12. `ch26-coconut-latent-tree-breadth-first-parallel-expansion`
13. `ch26-mixture-of-depths-token-routing-and-warp-divergence`
14. `ch26-stream-compaction-gather-scatter-for-early-exit-layers`
15. `ch26-post-autoregressive-architectures-throughput-latency-pareto`
16. `ch26-next-gen-foundation-model-infrastructure-synthesis`

#### 4.2.4 第 26 章代码块（6 个）、算例（6 个）、表格（6 张）与思考题（10 道）精确规格清单
- **6 个生产级 Python 代码块**：
  1. `BlockDiffusionDenoisingScheduler`：半自回归块扩散（`Block Diffusion`）块间 KV 固化与动态置信度重掩码推理调度器；
  2. `DLLMRooflineThroughputAnalyzer`：自回归解码（`AR LLM`）与并行扩散去噪（`dLLM`）在不同块大小 $K$ 与去噪步数 $T$ 下的 Roofline 吞吐求解器；
  3. `BLTEntropyDynamicPatcher`：`Byte Latent Transformer` 字节流条件熵在线分块器与非规则交叉注意力（Ragged Cross-Attention）偏移量索引生成器；
  4. `TitansChunkwiseDualFormScan`：`Google Titans` / `TTT-Linear` 惊奇度动量快速权重更新的分块对偶矩阵乘（`ttt_chunk_scan`）内核模拟器；
  5. `CoconutLatentThoughtLoopEngine`：`Coconut` 连续潜空间思维链零拷贝隐藏状态回环与多分支潜空间波束扩展运行时；
  6. `MixtureOfDepthsStreamCompactor`：词元级动态跳层（`Mixture-of-Depths`）流紧缩（Stream Compaction）Gather/Scatter 与 Warp 满载率优化器。
- **6 个标准化编号算例**：
  - `算例 26-1`：`32B` FP8 模型在 $K=64, T_{\mathrm{denoise}}=10$ 下从自回归 `100.0 tok/s` 跃升至 `592.6 tok/s`（`5.93x` 提速）完整时延拆解；
  - `算例 26-2`：动态置信度重掩码（Confidence Threshold $\gamma = 0.92$）在代码生成任务中提前终止 35% 冗余去噪步的算力节省核算；
  - `算例 26-3`：`BLT` 移除 `256k` 词表节省 `4.194 GB` 显存及平均 Patch 跨度从 `4.5 B` 拉长至 `12.0 B` 时端到端 FLOPs 降低 `52.4%` 推导；
  - `算例 26-4`：`Titans` / `TTT-Linear` 采用分块大小 $b=64$ 将算术强度从 `1.5 FLOP/B` 提升至 `48.0 FLOP/B`（吞吐提速 `24.60x`）与 `2M` 上下文 `16.0 MB/layer` 恒定显存核算；
  - `算例 26-5`：`Coconut` 每步压缩 $c=3$ 个离散推理词元为 1 个连续潜向量并省去 LM Head + 采样同步的端到端延迟收益；
  - `算例 26-6`：`Mixture-of-Depths`（保留率 $\rho = 0.25$）未执行流紧缩时的 Warp 有效利用率退化（`58.1%`）与流紧缩后恢复至 `96.4%` 的开销平衡推导。
- **6 张标准化斜体表题**：`表 26-1`（AR LLM vs. 全序列扩散 LLaDA vs. 半自回归 Block Diffusion 系统特征对比）、`表 26-2`（BPE 分词架构与 BLT 字节级无分词架构显存及算力账本）、`表 26-3`（Softmax KV Cache、线性 SSM 与 Titans/TTT 测试时神经记忆对比）、`表 26-4`（显式离散 CoT 与 Coconut 连续潜空间推理性能对比）、`表 26-5`（动态深度路由 MoD 与 Early-Exit 硬件调度对比）、`表 26-6`（第 26 章课后思考题核心锁定参数与标准答案速查表）。
- **10 道课后思考题（`**练习 26-1**` 至 `**练习 26-10**`）**：覆盖块扩散最优块大小 $K^*$ 导数求解、块间固化 KV 缓存误差传播、置信度重掩码退火曲线、BLT 字节熵阈值与序列压缩率、Ragged Cross-Attention 分块对齐填充率、TTT 分块对偶形式精确代数等价条件、惊奇度动量衰减率稳定性、Coconut 多路径潜空间搜索树分叉显存、MoD Top-K 因果预测器设计及后自回归混合推理引擎架构综合。

---

### 4.3 第 27 章：`manuscripts/27-晶圆级与 3D-SRAM 专用推理芯片（Cerebras-Groq）、微秒级极低延迟 AI 与三值量化（BitNet b1.58）系统.md`

#### 4.3.1 四节八小节大纲
- `## 27.1 全片上 SRAM 驻留架构（Cerebras WSE-3 / Groq LPU / 3D-SRAM）与 HBM 架构的第一性原理对决`
  - `### 27.1.1 为什么 HBM 在 Batch=1 下注定撞墙？晶圆级芯片（44 GB SRAM / 21 PB/s）与多芯片 SRAM 串联菊花链`
  - `### 27.1.2 3D 垂直堆叠 SRAM（TSMC SoIC 3D V-Cache / TeRAM）的硅穿孔（TSV）带宽密度、热阻耦合与 CapEx 交叉点`
- `## 27.2 零硬件抖动的软件静态确定性编译器（Cycle-Accurate Deterministic Execution）`
  - `### 27.2.1 消灭缓存未命中、动态分支预测与网络包仲裁：编译期单时钟周期级数据流编排（Software-Scheduled NoC）`
  - `### 27.2.2 跨数百颗 SRAM 芯片的确定性张量—流水混合切分与微秒级同步屏障（Zero-Jitter Hardware Barrier）`
- `## 27.3 亚毫秒级（< 50 us）金融量化、高频风控与实时物理控制 AI 推理系统工程`
  - `### 27.3.1 内核旁路网络（DPDK / Solarflare ef_vi）+ FPGA/NIC PCIe P2P 直写 GPU 显存零拷贝数据通路`
  - `### 27.3.2 持久化轮询 CUDA 内核（Persistent Polling Kernel）、CPU 核心物理隔离（isolcpus）与 P99.99 尾延迟猎杀`
- `## 27.4 极致三值量化（BitNet b1.58）与无乘法器查找表计算引擎（T-MAC / BitBLAS）`
  - `### 27.4.1 1.58-Bit 三值权重（{-1, 0, +1}）的信息论边界、三值打包编码（5 权重压入 1 Byte）与加法树硬件映射`
  - `### 27.4.2 预计算激活查找表（Activation LUT）：利用 SIMD 寄存器洗牌指令（pshufb / tbl）与片上 SRAM 实现十倍能效跃迁`
- `## 本章小结`

#### 4.3.2 核心数学方程与锁定数值锚点（Locked Numerical Anchors）
1. **全片上 SRAM 串联（`Groq LPU` / `Cerebras WSE-3`）vs. `8x B200` HBM 超节点的延迟—成本（TCO）交叉方程**：
   - 运行 `Llama-3.3 70B`（FP8 权重 `70.0 GB`）：
     - **方案 A（HBM 架构：`4x H100` 张量并行）**：聚合 HBM 带宽 $4 \times 3.35 = 13.4\ \mathrm{TB/s}$，$B=1$ 时单步权重读取耗时 $5.22\ \mathrm{ms}$ + AllReduce $0.85\ \mathrm{ms} = 6.07\ \mathrm{ms}$（单用户速度 **`164.7 tok/s`**）；但在 $B=128$ 时总吞吐达 **`6,400 tok/s`**（单卡成本摊销低）；
     - **方案 B（全片上 SRAM 菊花链：如 320 颗 `230 MB` SRAM 芯片或 2 台 `44 GB` WSE-3，聚合片上 SRAM 带宽 `21.0–25.6 PB/s`）**：权重 100% 驻留片上 SRAM，单步权重读取仅需 **`0.35 ms`**，单用户生成速度高达 **`1,850.0 tok/s`**（较 H100 提速 **`11.23x`**）；
     - 定量推导出 SRAM 架构与 HBM 架构的**每百万词元成本（$\mathrm{USD/MTok}$）盈亏平衡并发度 $B_{\mathrm{cross}} = 14.2$**：当场景要求超低并发、极致单流速度（如多轮串行 Agent 工具调用或实时代码补全，$B \le 14$）时 SRAM 芯片具备延迟统治力；当离线批处理或高并发共享（$B \ge 32$）时 HBM GPU 具有 **`6.8x` 成本优势**。
2. **亚毫秒级高频量化 AI 推理延迟拆解账本（`< 40 us` P99.99）**：
   - 传统 Linux Socket + PyTorch 栈：内核协议栈拷贝 `18.0 us` + Python/C++ 调度 `12.0 us` + `cudaMemcpyAsync` `6.5 us` + 6 层时序 Transformer 每次 `cudaLaunchKernel` 开销（$6 \times 4.5\ \mu\mathrm{s} = 27.0\ \mu\mathrm{s}$）+ 算子计算 `14.5 us` = **`78.0 us`**（且受 OS 调度中断影响，P99.99 飙升至 **`420.0 us`**）；
   - **极低延迟全链路改造**：网卡 `ef_vi` / `GPUDirect RDMA` 直写锁定显存环形缓冲区（`2.8 us`）+ **单持久化轮询 CUDA 内核（Persistent Megakernel 在 GPU 端死循环轮询显存标志位，`0.0 us` Launch 开销，全模型驻留 L2 Cache 与 Shared Memory，计算耗时 `11.4 us`）** + GDR 网卡直发指令（`2.3 us`）= 端到端中位数延迟 **`16.5 us`**，配合 `isolcpus` 屏蔽核间中断后 **P99.99 尾延迟严格锁定在 `24.8 us` 以内**（尾延迟削减 **`16.94x`**）。
3. **`BitNet b1.58` 三值编码与 `T-MAC` 查找表（LUT）加速**：
   - 每个三值权重 $w \in \{-1, 0, +1\}$ 理论信息熵为 $\log_2 3 \approx \mathbf{1.585\ \mathrm{bits}}$。工程实现中将每 **$5$ 个三值权重**（$3^5 = 243 \le 256$）无损编码进 **`1 Byte`（`uint8`）**，实测每权重仅占 **`1.60 bits`**（相比 FP16 压缩 **`10.0x`**，`70B` 模型仅需 **`14.0 GB`** 内存）；
   - `T-MAC` 将每 $g = 4$ 个连续激活值预计算其全部 $3^4 = 81$（或对称折半为 $40$）种加减组合存入 128-bit SIMD 寄存器表，内循环将乘法累加替换为 `pshufb` / `vtbl` 字节查表指令，在端侧芯片上实现 **`6.4x` 推理提速与 `88.5%` 乘法器动态功耗消除**。

#### 4.3.3 第 27 章 16 幅配图清单（`manuscripts/ch27/`）
1. `ch27-sram-wafer-scale-vs-hbm-gpu-memory-wall-comparison`
2. `ch27-cerebras-wse3-wafer-scale-noc-and-weight-streaming`
3. `ch27-groq-lpu-320-chip-deterministic-daisy-chain-pipeline`
4. `ch27-3d-stacked-sram-soic-tsv-thermal-and-bandwidth-model`
5. `ch27-sram-asic-vs-hbm-gpu-tco-crossover-by-batch-size`
6. `ch27-hardware-dynamic-arbitration-vs-cycle-accurate-compiler`
7. `ch27-software-scheduled-tensor-stream-instruction-alignment`
8. `ch27-sub-millisecond-hft-ai-kernel-bypass-nic-to-gpu-p2p`
9. `ch27-cuda-launch-overhead-vs-persistent-polling-megakernel`
10. `ch27-linux-isolcpus-rt-throttling-and-p9999-tail-latency`
11. `ch27-bitnet-b158-ternary-weight-packing-5trits-per-byte`
12. `ch27-fp16-mac-unit-vs-ternary-adder-tree-silicon-area-energy`
13. `ch27-tmac-precomputed-activation-lut-simd-pshufb-kernel`
14. `ch27-bitblas-mixed-precision-w158a8-tensor-core-dequant`
15. `ch27-ultra-low-latency-and-1bit-inference-pareto-frontier`
16. `ch27-extreme-latency-and-ternary-computing-synthesis`

#### 4.3.4 第 27 章代码块（6 个）、算例（6 个）、表格（6 张）与思考题（10 道）精确规格清单
- **6 个生产级 Python 代码块**：
  1. `SramWaferVsHbmTcoCrossoverSolver`：全片上 SRAM 菊花链（`Groq LPU` / `WSE-3`）与 HBM 超节点在 `Llama-3.3 70B` 下的并发度—TCO 交叉点求解器；
  2. `CycleAccurateDeterministicCompiler`：零硬件仲裁软件静态确定性数据流编译器（计算周期级张量流发射时刻表与跨芯片同步偏移）；
  3. `SubMillisecondPersistentMegakernelSim`：内核旁路网络（`ef_vi`）+ PCIe P2P + 持久化轮询 CUDA Megakernel 微秒级推理时序模拟器；
  4. `TailLatencyJitterAnalyzer`：OS 调度中断、`cudaLaunchKernel` 开销与 `isolcpus` 核隔离对 P50 / P99 / P99.99 尾延迟的统计仿真器；
  5. `BitNetTernaryCodec`：`BitNet b1.58` 三值权重 `5-trits-per-byte`（$3^5 = 243 \le 256$）无损打包编解码器与容量核算器；
  6. `TMacSimdLutEngine`：`T-MAC` 预计算激活查找表（Activation LUT）与 128-bit SIMD 寄存器查表内循环模拟器（验证 `6.4x` 提速与能耗节省）。
- **6 个标准化编号算例**：
  - `算例 27-1`：`70B` FP8 模型在 `4x H100`（`164.7 tok/s`）与全片上 SRAM 阵列（`1,850.0 tok/s`，提速 `11.23x`）下的延迟与盈亏平衡并发度 $B_{\mathrm{cross}} = 14.2$ 推导；
  - `算例 27-2`：3D 堆叠 SRAM（TSMC SoIC 8 层 3D V-Cache，`1.5 GB/chip`）的硅穿孔（TSV）带宽密度与逻辑裸片温升 $\Delta T$ 耦合核算；
  - `算例 27-3`：320 颗 SRAM 芯片流水线菊花链在静态时钟对齐（$\pm 2\ \mathrm{ns}$ 硬件屏障）下的零排队抖动证明；
  - `算例 27-4`：高频量化 AI 从传统 Socket+PyTorch（中位数 `78.0 us` / P99.99 `420.0 us`）降至持久化轮询 Megakernel（中位数 `16.5 us` / P99.99 `24.8 us`）全账本拆解；
  - `算例 27-5`：`BitNet b1.58` 每 5 个三值权重压入 `1 Byte`（`1.60 bits/weight`）使 `70B` 模型缩至 `14.0 GB`（放入单颗大容量 3D-SRAM 或消费级端侧内存）容量核算；
  - `算例 27-6`：`T-MAC` $g=4$ 分组查找表预计算开销与矩阵乘主循环 `pshufb` 查表指令比（动态乘法功耗降低 `88.5%`）推导。
- **6 张标准化斜体表题**：`表 27-1`（HBM3e/HBM4 GPU、Groq LPU 与 Cerebras WSE-3 架构参数对比）、`表 27-2`（动态硬件缓存/路由仲裁 vs. 静态编译期确定性数据流对比）、`表 27-3`（亚毫秒级高频 AI 推理软件栈各阶段延迟对比）、`表 27-4`（FP16、INT8、INT4 与 BitNet b1.58 硬件乘加器面积及能耗对比）、`表 27-5`（`T-MAC` 与 `BitBLAS` 在不同批大小下的解码吞吐对比）、`表 27-6`（第 27 章课后思考题核心锁定参数与标准答案速查表）。
- **10 道课后思考题（`**练习 27-1**` 至 `**练习 27-10**`）**：覆盖 SRAM 容量成本对长上下文 KV 缓存的制约、3D-SRAM 热阻网络、静态编译器面对动态 MoE 路由的确定性填充策略、PCIe 5.0 P2P TLP 包开销、持久化轮询内核与 GPU 功耗发热管理、Linux RT 内核中断屏蔽、三值编码信息效率（`1.585 / 1.60 = 99.06%`）、T-MAC 最优分组大小 $g$ 与寄存器容量边界、W1.58A8 混合精度反量化流水线以及微秒级 AI 系统全链路容错设计。

---

### 4.4 第 28 章：`manuscripts/28-主权 AI、广域网去中心化低通信训练（DiLoCo）、实时护栏（Guardrails）与密码学水印安全底座.md`

#### 4.4.1 四节八小节大纲
- `## 28.1 跨洲际广域网与零散算力池的低通信训练（DiLoCo / Streaming DiLoCo / OpenDiLoCo）`
  - `### 28.1.1 双层优化器解耦：本地内循环 AdamW（H=500 步）与广域网外循环 Nesterov 伪梯度动量更新`
  - `### 28.1.2 Streaming DiLoCo 分片流水重叠与 FP4 伪梯度压缩：将跨地域带宽需求降低两千倍且零暴露停顿`
- `## 28.2 去中心化异构算力网络的拜占庭容错（Proof-of-Learning）、弹性扩缩容与掉线自愈`
  - `### 28.2.1 开放不可信算力池的投毒梯度防御：随机重放抽检（Spot-Check Verification）与鲁棒几何中位数聚合`
  - `### 28.2.2 基于分布式哈希表（DHT / Hivemind）的优化器状态分片与抢占式实例（Spot GPUs）秒级无损进出`
- `## 28.3 生产级大模型在线实时护栏（Guardrail Sidecars）与表示工程激活干预（Activation Steering）`
  - `### 28.3.1 异步流式伴飞护栏架构：主模型解码与轻量级安全分类器（ShieldGemma / Llama-Guard）的投机窗口对齐`
  - `### 28.3.2 零延迟原位安全防御：推理残差流中的稀疏自编码器（SAE）特征监控与有害方向正交投影钳位`
- `## 28.4 密码学无损生成水印（SynthID-Text）与模型权重供应链防篡改体系`
  - `### 28.4.1 分布式张量并行采样下的伪随机种子同步与锦标赛水印采样（Tournament Watermarking）数学无偏性`
  - `### 28.4.2 告别 Pickle 反序列化恶意代码漏洞：Safetensors 零拷贝 mmap 页对齐加载与权重哈希树签名校验`
- `## 本章小结`

#### 4.4.2 核心数学方程与锁定数值锚点（Locked Numerical Anchors）
1. **`Streaming DiLoCo` 广域网带宽压缩比与暴露通信开销**：
   - 在跨 4 个地理区域（RTT $= 85\ \mathrm{ms}$）的算力池上联合训练 `70B` 模型（BF16 参数 `140.0 GB`），单副本本地单步训练耗时 $t_{\mathrm{step}} = 2.40\ \mathrm{s}$：
     - **标准跨地域 AllReduce（每步同步）**：需在 $2.40\ \mathrm{s}$ 内完成 Ring AllReduce 传输 $2 \times \frac{3}{4} \times 140.0\ \mathrm{GB} = 210.0\ \mathrm{GB}$，要求跨地域专线带宽 $\ge \mathbf{700.0\ \mathrm{Gbps}}$（否则通信占比 $> 90\%$）；
     - **基础 `DiLoCo`（内循环步数 $H = 500$，外循环同步间隔 $T_{\mathrm{sync}} = 500 \times 2.40 = 1,200\ \mathrm{s}$）**：每 `20 分钟` 才同步一次伪梯度 $\Delta \theta^{(k)} = \theta^{(k,0)} - \theta^{(k,H)}$，通信频率降低 **$500x$**，但第 500 步末尾仍会出现突发同步尖峰；
     - **`Streaming DiLoCo` + FP4 E2M1 伪梯度块量化（`35.0 GB`）**：将模型参数切分为 $M = 25$ 个大小 $1.40\ \mathrm{GB}$ 的分片，每完成 $\Delta H = 20$ 步本地更新即在后台流式传输 1 个分片（耗时窗口 $48.0\ \mathrm{s}$），所需广域网持续带宽骤降至：
       $$B_{\mathrm{WAN}}^{\mathrm{stream}} = \frac{2 \times \frac{3}{4} \times 35.0\ \mathrm{GB} \times 8\ \mathrm{bits/B}}{1,200\ \mathrm{s}} = \mathbf{350.0\ \mathrm{Mbps}}\ (\approx 0.35\ \mathrm{Gbps})$$
       相比标准跨地域数据并行，**广域网带宽需求降低整整 `2,000x`**，且后台流式传输与前向/反向计算 100% 重叠（**暴露广域网等待时间为 `0.0 s`，算力有效利用率 `99.4%`**）。
2. **异步流式伴飞护栏（Speculative Guardrail Sidecar）与原位 SAE 激活钳位**：
   - 若将 `Llama-Guard 8B` 串行挡在主模型（`70B`）生成流之前或之后，会额外增加 `45.0 ms` 首词元延迟（TTFT）与 `180.0 ms` 尾包审查阻塞；
   - 采用**双流异步伴飞 + 块粒度（每 $K=8$ 个生成词元）滑动窗口投机放行**：主模型生成的词元立即流式下发客户端显示缓冲，同时旁路拷贝至同节点 MIG 切片上的 `1B/2B` 蒸馏护栏模型并行打分（单次审查仅 `3.2 ms`），端到端正常请求 **暴露护栏延迟 `< 3.5 ms`**；结合残差流原位投影 $h' \leftarrow h - \max(0, \langle h, v_{\mathrm{harm}}\rangle) v_{\mathrm{harm}}$，每层仅增加 2 次向量点积（耗时 `< 4 us/layer`，开销 **`0.08%`**）。
3. **`SynthID-Text` 锦标赛采样水印的检出率与零困惑度退化**：
   - 在深度 $m = 30$ 层的锦标赛采样中，仅需生成 **$200$ 个词元**，在假阳性率 $\mathrm{FPR} = 10^{-6}$ 下即可实现 **`99.2%` 的真阳性检出率（TPR）**，而模型输出分布的期望交叉熵偏差 $\Delta \mathrm{PPL} < \mathbf{0.01}$。

#### 4.4.3 第 28 章 16 幅配图清单（`manuscripts/ch28/`）
1. `ch28-wan-allreduce-bottleneck-vs-diloco-two-level-optimizer`
2. `ch28-streaming-diloco-fragment-pipeline-and-fp4-outer-delta`
3. `ch28-diloco-inner-step-h-vs-validation-loss-and-bandwidth`
4. `ch28-proof-of-learning-spot-check-and-byzantine-median`
5. `ch28-dht-hivemind-elastic-optimizer-sharding-spot-recovery`
6. `ch28-serial-guardrail-blocking-vs-async-speculative-sidecar`
7. `ch28-sliding-window-token-stream-guardrail-rollback`
8. `ch28-in-situ-residual-stream-sae-feature-clamping`
9. `ch28-representation-engineering-harmful-subspace-projection`
10. `ch28-synthid-text-tournament-sampling-tree-mechanism`
11. `ch28-distributed-tp-prng-seed-synchronization-for-watermark`
12. `ch28-watermark-token-length-vs-detection-tpr-at-low-fpr`
13. `ch28-pickle-arbitrary-code-exploit-vs-safetensors-zero-copy-mmap`
14. `ch28-merkle-tree-weight-signature-and-tee-attestation-boot`
15. `ch28-sovereign-ai-multi-region-compliance-and-audit-architecture`
16. `ch28-decentralized-training-and-trustworthy-ai-infra-synthesis`

#### 4.4.4 第 28 章代码块（6 个）、算例（6 个）、表格（6 张）与思考题（10 道）精确规格清单
- **6 个生产级 Python 代码块**：
  1. `StreamingDiLoCoOptimizerSimulator`：跨洲际广域网双层优化器（内循环 AdamW $H=500$ + 外循环 Nesterov 动量）与 `Streaming DiLoCo` FP4 伪梯度流水调度器；
  2. `ByzantineRobustGradientAggregator`：开放算力池随机重放抽检（Proof-of-Learning Spot-Check）与 Weiszfeld 几何中位数拜占庭鲁棒聚合器；
  3. `DhtElasticSpotWorkerCoordinator`：基于分布式哈希表（DHT / Hivemind）的优化器状态分片与抢占式 Spot GPU 秒级进出恢复协调器；
  4. `AsyncSpeculativeGuardrailSidecar`：在线推理双流异步伴飞护栏（`K=8` 滑动窗口投机放行与违规回滚中断）控制器；
  5. `ResidualStreamSaeClamper`：推理残差流原位稀疏自编码器（SAE）特征监控与有害方向正交投影钳位算子；
  6. `SynthIdTournamentWatermarkAndSafetensorsVerifier`：`SynthID-Text` $m=30$ 层锦标赛无偏水印采样器与 `Safetensors` 页对齐 Merkle 签名校验器。
- **6 个标准化编号算例**：
  - `算例 28-1`：跨 4 洲际区域训练 `70B` 模型从标准 AllReduce（需 `700.0 Gbps`）到 `Streaming DiLoCo` + FP4（仅需 `350.0 Mbps`，压缩 `2,000x` 且暴露等待 `0.0 s`）完整推导；
  - `算例 28-2`：开放算力池存在 $f = 15\%$ 投毒节点时，抽检率 $p_{\mathrm{check}} = 5\%$ 在 10 轮外循环内捕获欺诈节点的概率（`> 99.9%`）与几何中位数崩溃点核算；
  - `算例 28-3`：抢占式 Spot 实例每小时中断率 $\lambda = 8\%$ 下，DHT 分片冗余修复与毫秒级 Peer 状态拉取的有效训练时间比（`ETTR = 96.8%`）推导；
  - `算例 28-4`：串行护栏（增加 `45 ms` TTFT + `180 ms` 尾延迟）vs. 异步伴飞护栏（`K=8` 投机窗口，暴露延迟 `< 3.5 ms`）吞吐与 SLA 对比；
  - `算例 28-5`：残差流原位有害子空间正交投影（每层 2 次 $d=8,192$ 点积，耗时 `< 4 us/layer`，总开销 `0.08%`）的零延迟安全边界计算；
  - `算例 28-6`：`SynthID-Text` 30 层锦标赛在 200 词元下实现 `FPR = 10^-6`、`TPR = 99.2%` 及分布式 TP 伪随机数种子同步带宽核算。
- **6 张标准化斜体表题**：`表 28-1`（标准数据并行、基础 DiLoCo 与 Streaming DiLoCo 广域网通信账本对比）、`表 28-2`（去中心化训练聚合算法均值、裁剪均值、Krum 与几何中位数抗拜占庭能力对比）、`表 28-3`（串行护栏、异步伴飞护栏与原位 SAE 激活钳位系统指标对比）、`表 28-4`（KGW 红绿词表水印 vs. SynthID-Text 锦标赛无损水印数学性质对比）、`表 28-5`（Pickle 格式安全风险与 Safetensors 零拷贝页对齐加载性能对比）、`表 28-6`（第 28 章课后思考题核心锁定参数与标准答案速查表）。
- **10 道课后思考题（`**练习 28-1**` 至 `**练习 28-10**`）**：覆盖内循环步数 $H$ 与外循环动量 $\beta_{\mathrm{outer}}$ 收敛半径、Streaming DiLoCo 滞后伪梯度陈旧度（Staleness）补偿、FP4 E2M1 伪梯度误差反馈（Error Feedback）累加器、Proof-of-Learning 浮点非确定性容差、DHT 虚拟节点一致性哈希、异步伴飞护栏投机窗口大小 $K$ 与客户端撤回率权衡、SAE 激活钳位对正常语义流形的正交性保持、锦标赛采样期望概率分布严格不变性证明、张量并行下 Gumbel 噪声种子广播以及 Safetensors `4,096 B` 页对齐 `mmap` 零拷贝加载吞吐。

---

### 4.5 第 29 章：`manuscripts/29-DeepSeek 开源基础设施全栈解剖与极客工程实战：FlashMLA、DeepGEMM、DeepEP、DualPipe、3FS、Mooncake 与 KTransformers.md`

#### 4.5.1 四节八小节大纲
- `## 29.1 FlashMLA 与 DeepGEMM：榨干 Hopper/Blackwell 每一周期的算子内核艺术`
  - `### 29.1.1 FlashMLA 深度解剖：矩阵吸收后 [512 + 64] 非对称头维的乒乓 Warpgroup 流水与 Split-KV 持久调度（3,000 GB/s）`
  - `### 29.1.2 DeepGEMM 深度解剖：300 行模板 + 运行期 JIT 形状常量化 + FP8（1x128 与 128x128）二级累加（1,350+ TFLOPS）`
- `## 29.2 DeepEP 与 DualPipe：零 SM 占用异构集合通信与双向无气泡流水线协同`
  - `### 29.2.1 DeepEP Normal 模式（NVLink-RDMA 混合转发 150+ GB/s）与 Low-Latency 纯 IBGDA 模式（< 160 us）的 PTX 内存序`
  - `### 29.2.2 DualPipe 双向流水线微批交错矩阵、细粒度前反向拆分（计算/通信解耦）与零通信暴露证明`
- `## 29.3 3FS（Fire-Flyer File System）与 Smallpond：面向百 TB/s 吞吐的去缓存随机读与分布式数据底座`
  - `### 29.3.1 3FS CRAQ 链式复制强一致状态机、O_DIRECT + USRBIO 零拷贝异步环形队列与 180 节点 6.6 TB/s 聚合读实测`
  - `### 29.3.2 Smallpond 基于 DuckDB + 3FS 的无常驻服务（Serverless）PB 级 Parquet 分区洗牌与外存排序`
- `## 29.4 Mooncake 与 KTransformers：从数据中心级 P-D 分离集群到单机混合专家平民化部署`
  - `### 29.4.1 Mooncake Transfer Engine 多网卡 RDMA 拓扑感知路由、基于负载预测的提前拒绝（Early Rejection）与弹性扩缩`
  - `### 29.4.2 KTransformers 极致异构卸载：GPU 驻留 MLA 与共享专家 + 双路 CPU AMX 原位计算 256 个路由专家的 NUMA 绑核工程`
- `## 本章小结`

#### 4.5.2 核心数学方程与锁定数值锚点（Locked Numerical Anchors）
1. **`FlashMLA` 非对称头维（$d_{c} = 512$ 吸收潜向量 $+ d_{R} = 64$ 解耦 RoPE）的片上寄存器与带宽账本**：
   - 在解码阶段，通过将 $W^{UK}$ 吸收进查询投影、将 $W^{UV}$ 吸收进输出投影，每个 KV 词元仅需从 HBM 读取 $512 + 64 = \mathbf{576}$ 个 BF16/FP8 元素（BF16 下 `1,152 B/token`；对比头数 $n_h = 128$、每头常规维数 $d_h = 64$ 的标准未压缩 MHA 每词元 K+V BF16 状态 $2 \times 128 \times 64 \times 2\ \mathrm{B} = 32,768\ \mathrm{B/token}$，实现精确 **`28.44x`** 访存压缩；若对比含 $d_h = 192$ 完整头维的解压 MHA `98,304 B/token`，则压缩比更高达 `85.33x`）；
   - `FlashMLA` 采用 2 个消费者 Warpgroup（共 8 个 Warp）乒乓交替执行 `WGMMA` 与 Softmax 缩放，在 H800（物理峰值带宽 `3,350 GB/s`）上实测跑出 **`3,000 GB/s`（带宽利用率 `89.55%`）** 与 **`580 TFLOPS`**（算术强度拐点处达到理论屋脊上限的 **`94.2%`**）。
2. **`DeepGEMM` 运行期 JIT 编译节省寄存器与二级累加指令开销**：
   - 通过在运行期将矩阵维数 $M, N, K$、分块大小 $B_M, B_N, B_K$ 及流水级数编译为 C++ 立即数常量，编译器得以完全展开循环并省去动态地址寄存器，单线程寄存器占用从静态通用库的 `248` 个降至 **`192` 个**（消除寄存器溢出至 Local Memory）；
   - 每执行 $K_{\mathrm{block}} = 128$ 维的 4 条 `wgmma.mma_async` 指令（Tensor Core 流水执行周期为 `128` 个时钟周期），仅需在 CUDA Core 上插入 `4` 个时钟周期的 `FFMA` 指令将块累加器乘以缩放因子 $s_A \cdot s_B$ 累加至 FP32 主寄存器，二级累加周期占比仅为 $4 / 128 = 3.125\% \approx \mathbf{3.12\%}$，在 H800 上实现 **`1,358 TFLOPS`** 的分组 MoE GEMM 吞吐。
3. **`DeepEP` + `DualPipe` + `3FS` + `KTransformers` 全链路性能锚点**：
   - `DeepEP`：预填充/训练 Normal 模式跨节点转发吞吐 **`153 GB/s`**（仅占 20–24 个通信 SM）；解码 Low-Latency 纯 `IBGDA` 模式在 `128` 专家下 Dispatch 延迟 **`163 us`**、Combine 延迟 **`194 us`**，且通过后台 Hook 轮询实现 **`0` 个 SM 占用**；
   - `3FS`：`180` 台存储节点（每台 $16 \times 14\ \mathrm{TB}$ NVMe SSD + $2 \times 200\ \mathrm{Gbps}$ 网卡）为 `500+` 计算节点提供 **`6.60 TB/s`** 持续聚合随机读吞吐，GraySort `110.5 TiB` 排序仅耗时 **`30 min 14 s`**；
   - `KTransformers`：单张 `24 GB` RTX 4090D（驻留 MLA + 共享专家 `19.2 GB`）配合双路 Xeon 8488C（16 通道 DDR5 带宽 `680 GB/s` 驻留 INT4 路由专家 `336 GB`），通过 CPU `AMX` 就地算完 8 个激活专家后仅传回 `14 KB` 激活值，使 `DeepSeek-V3 671B` 预填充达 **`286.4 tok/s`**、解码达 **`18.6 tok/s`**（较 llama.cpp 提速 **`13.8x`**）。

#### 4.5.3 第 29 章 16 幅配图清单（`manuscripts/ch29/`）
1. `ch29-deepseek-open-infra-seven-pillar-fullstack-architecture`
2. `ch29-flashmla-512-plus-64-pingpong-warpgroup-pipeline`
3. `ch29-flashmla-variable-length-split-kv-load-balancer`
4. `ch29-deepgemm-runtime-jit-shape-constant-specialization`
5. `ch29-deepgemm-fp8-1x128-128x128-two-level-ffma-accumulation`
6. `ch29-deepep-normal-mode-nvlink-rdma-hybrid-forwarding`
7. `ch29-deepep-low-latency-ibgda-zero-sm-hook-overlap`
8. `ch29-dualpipe-bidirectional-pipeline-schedule-and-20-stage-grid`
9. `ch29-dualpipe-fine-grained-backward-input-weight-split`
10. `ch29-3fs-craq-chain-replication-and-usrbio-zero-copy-ring`
11. `ch29-3fs-180-node-6600gbs-throughput-and-graysort-pipeline`
12. `ch29-smallpond-duckdb-on-3FS-serverless-parquet-shuffle`
13. `ch29-mooncake-transfer-engine-multi-nic-rdma-kv-pool`
14. `ch29-mooncake-early-rejection-and-sla-goodput-controller`
15. `ch29-ktransformers-gpu-mla-and-cpu-amx-numa-expert-offload`
16. `ch29-deepseek-fullstack-cost-and-efficiency-synthesis`

#### 4.5.4 第 29 章代码块（6 个）、算例（6 个）、表格（6 张）与思考题（10 道）精确规格清单
- **6 个生产级 Python 代码块**：
  1. `FlashMlaPingpongTileScheduler`：`FlashMLA` `[512 + 64]` 非对称头维双 Warpgroup 乒乓流水与变长序列 Split-KV 持久块调度模拟器；
  2. `DeepGemmJitAndTwoLevelAccumulator`：`DeepGEMM` 运行期 JIT 形状常量化寄存器节省与 FP8 `1x128 / 128x128` 二级累加器仿真器；
  3. `DeepEpDualModeRouter`：`DeepEP` Normal 模式（NVLink-RDMA 混合双跳 `153 GB/s`）与 Low-Latency 纯 `IBGDA` 零 SM 占用钩子模拟器；
  4. `DualPipeBidirectionalGridVerifier`：`DualPipe` 双向流水线微批交错时间表、细粒度 `Backward-for-Input / Backward-for-Weight` 拆分与零通信暴露验证器；
  5. `ThreeFsCraqAndUsrbioSimulator`：`3FS` 链式复制（CRAQ）强一致读写状态机与 `180` 节点 `6.60 TB/s` 聚合零拷贝 I/O 吞吐核算器；
  6. `MooncakeAndKTransformersPlanner`：`Mooncake` P-D 分离提前拒绝控制器与 `KTransformers` 单 GPU + 双路 CPU AMX NUMA 混合专家卸载规划器。
- **6 个标准化编号算例**：
  - `算例 29-1`：`FlashMLA` 矩阵吸收后 `1,152 B/token`（较 `32,768 B/token` 压缩 `28.44x`）及 H800 上 `3,000 GB/s`（`89.55%` 利用率）与 `580 TFLOPS` 屋脊推导；
  - `算例 29-2`：`DeepGEMM` JIT 常量化将寄存器从 `248` 降至 `192` 个及每 `128` 个 Tensor Core 周期插入 `4` 周期 `FFMA`（开销 `3.12%`）跑出 `1,358 TFLOPS` 核算；
  - `算例 29-3`：`DeepEP` 在预填充（`24` 个 SM 达 `153 GB/s`）与解码（`0` 个 SM 纯 IBGDA 达 `163 us` Dispatch）双模式下的带宽与延迟账本；
  - `算例 29-4`：`DualPipe` 在 $PP = 16$ 下通过双向注入 $2M = 32$ 个微批将流水气泡比压低并完全掩盖跨节点 All-to-All 的时间表推导；
  - `算例 29-5`：`3FS` `180` 台存储节点（`2,880` 块 NVMe SSD + `360` 张 `200G` 网卡）实现 `6.60 TB/s` 聚合读与 `Smallpond` `110.5 TiB` GraySort 吞吐拆解；
  - `算例 29-6`：`KTransformers` 在单张 `24 GB` 4090D + 双路 DDR5 CPU AMX（`680 GB/s`）上实现 `671B` 模型预填充 `286.4 tok/s`、解码 `18.6 tok/s`（提速 `13.8x`）全链路核算。
- **6 张标准化斜体表题**：`表 29-1`（DeepSeek 开源七大基础设施组件功能定位与性能指标总表）、`表 29-2`（FlashAttention-3 vs. FlashMLA 非对称头维寄存器与共享内存布局对比）、`表 29-3`（CUTLASS 静态模板库 vs. DeepGEMM 运行期 JIT 编译对比）、`表 29-4`（NCCL All-to-All vs. DeepEP Normal / Low-Latency 双模式对比）、`表 29-5`（Ceph / Lustre 传统缓存文件系统 vs. 3FS 去缓存 RDMA 文件系统对比）、`表 29-6`（第 29 章课后思考题核心锁定参数与标准答案速查表）。
- **10 道课后思考题（`**练习 29-1**` 至 `**练习 29-10**`）**：覆盖 FlashMLA 共享内存块尺寸边界、Split-KV 精确负载均衡分配算法、DeepGEMM 未对齐矩阵尾块（Tail M）填充、FP8 二级累加量化方差边界、DeepEP PTX `release/acquire` 内存序防死锁、IBGDA 队列深度与网卡门铃合并、DualPipe 参数副本显存代价与激活显存节省对冲、3FS CRAQ 尾节点脏读避免、Mooncake 全局 KV 池早拒阈值及 KTransformers 双路 CPU 跨 NUMA QPI/UPI 流量消除。

---

### 4.6 第 30 章：`manuscripts/30-自演进 AI 基础设施（AI for AI Infra）：大模型驱动的算子自动合成、强化学习编译寻优与三十章全栈终极收束.md`

#### 4.6.1 四节八小节大纲
- `## 30.1 大模型与强化学习驱动的 GPU 内核自动合成（KernelBench / Kevin / AI CUDA Engineer）`
  - `### 30.1.1 基于编译校验、数值等价测试与 Nsight Compute 硬件剖析反馈（Profile-in-the-Loop）的多轮 RLVR 算子进化`
  - `### 30.1.2 自动突破人类手写瓶颈：代数重结合、跨算子寄存器级持久化融合与非对称块尺寸（Tile Shape）自主发现`
- `## 30.2 面向并行拓扑、通信调度与芯片布局的自动搜索系统（Auto-Parallelism & AlphaChip）`
  - `### 30.2.1 基于代价模型与整数线性规划/蒙特卡洛树搜索（ILP + MCTS）的 5D 混合并行与自定义集合通信树合成`
  - `### 30.2.2 芯片物理布局强化学习（AlphaChip）与算子级动态电压频率调节（Per-Kernel DVFS）的闭环能效寻优`
- `## 30.3 十万卡智算中心自治运维（Autonomous AIOps）、多智能体根因诊断（RCA）与数字孪生仿真器`
  - `### 30.3.1 融合硬件遥测、NCCL 慢节点轨迹与光模块误码率的时空图诊断智能体（秒级定位慢卡与静默坏块）`
  - `### 30.3.2 数据中心级高保真数字孪生仿真器（Workload-Network-Thermal Co-Simulator）：上线前零成本推演十万卡架构`
- `## 30.4 全书三十章五大架构部类大一统收束：AI 基础设施第一性原理、演进铁律与未来十年展望`
  - `### 30.4.1 从第 1 章到第 30 章：五大部类、979 幅编号章节配图（含导读总图共 980 幅）、201 个 Python 代码块、228 个算例、492 道习题与 165 个实验的全景矩阵`
  - `### 30.4.2 统治 AI 基础设施的四大普适物理守恒律（算术强度、状态局部性、关键路径掩盖、信息—能量转换效率）终极综合`
- `## 本章小结`

#### 4.6.2 核心数学方程与锁定数值锚点（Locked Numerical Anchors）
1. **基于硬件剖析反馈的多轮 RLVR 内核自动合成（`Kevin` / `KernelBench`）收益账本**：
   - 算子合成智能体的奖励函数定义为正确性门控与硬件实测加速比的复合函数：
     $$R(\mathcal{K}) = \mathbb{I}_{\mathrm{compile}}(\mathcal{K}) \cdot \mathbb{I}_{\|\mathcal{K}(x) - f_{\mathrm{ref}}(x)\|_\infty \le \epsilon} \cdot \left(1 + \lambda \cdot \frac{T_{\mathrm{torch.compile}}}{T_{\mathrm{hw}}(\mathcal{K})}\right)$$
   - 在 `KernelBench` 250 个代表性算子基准上：单轮零样本生成（Zero-Shot LLM）编译且数值正确率仅为 `34.0%`，超越 `torch.compile` 的比例仅 `18.0%`；引入 **$T_{\mathrm{iter}} = 8$ 轮 Nsight Compute 硬件计数器反馈（注入 SM 占用率、Shared Memory Bank 冲突率、DRAM 读写字节数与 Warp Stall 原因）** 后，正确率跃升至 **`94.8%`**，在复杂长尾融合算子（如 LayerNorm + RoPE + GQA + FP8 Quant）上相对 PyTorch Eager 取得平均 **`3.42x`** 加速、相对 `torch.compile` 取得 **`1.58x`** 加速。
2. **自治集群 AIOps 多智能体根因诊断（Autonomous RCA）对有效训练时间比（ETTR）的提升**：
   - 在 `32,768` 卡集群上，传统人工排查隐性降速（Straggler / 交换机端口 PFC 死锁 / 光模块 FEC 降级）平均耗时 $\mathrm{MTTR}_{\mathrm{manual}} = 4.20\ \mathrm{hours}$，导致集群 $\mathrm{ETTR} = 88.4\%$；
   - 部署自治 AIOps 诊断与热替换闭环后，通过分析 NCCL Flight Recorder 环路延迟偏度与光模块 SNR 斜率，在 **`42.0 seconds`（`0.0117 hours`）** 内自动隔离亚健康节点并触发预实例化热备顶替，将 $\mathrm{MTTR}$ 压缩 **`360x`**，集群 $\mathrm{ETTR}$ 提升至 **`97.8%`**（在半年训练周期中直接挽回 **`13.5M USD`** 等效算力损耗）。
3. **全书 30 章大一统收束总账本（Grand 30-Chapter Encyclopedia Ledger）**：
   - 全书定格为 **`30` 章（五大部类）**、**`979` 幅编号章节配图（含未编号全书 30 章导读总图 `ch01/figure-1-book-roadmap` 共 `980` 幅配图）**、**`201` 个 Python 代码块（另含第 19 章 1 个 C++ 代码块，共 `202` 个类型化代码块）**、**`228` 个标准化引用块算例**、**`341` 张带题注表格**、**`492` 个思考题块（`491` 个唯一题号）**、**`812` 条学术脚注**、**`165` 个零依赖确定性可运行实验套件**。

#### 4.6.3 第 30 章 16 幅配图清单（`manuscripts/ch30/`）
1. `ch30-llm-rlvr-gpu-kernel-synthesis-closed-loop-architecture`
2. `ch30-nsight-compute-roofline-and-warp-stall-feedback-prompt`
3. `ch30-kernelbench-multi-turn-evolution-accuracy-and-speedup`
4. `ch30-auto-discovered-register-fusion-vs-human-handwritten-kernel`
5. `ch30-ilp-mcts-5d-auto-parallelism-and-collective-tree-search`
6. `ch30-alphachip-rl-macro-placement-and-wirelength-optimization`
7. `ch30-per-kernel-dvfs-frequency-scaling-energy-delay-product`
8. `ch30-autonomous-aiops-straggler-and-pfc-root-cause-graph`
9. `ch30-sub-minute-fault-isolation-mttr-and-ettr-recovery`
10. `ch30-datacenter-digital-twin-workload-network-thermal-cosim`
11. `ch30-thirty-chapter-five-part-encyclopedia-master-atlas`
12. `ch30-pillar-1-to-3-hardware-networking-training-inference-map`
13. `ch30-pillar-4-to-5-multimodal-agent-science-frontier-map`
14. `ch30-four-fundamental-conservation-laws-across-30-chapters`
15. `ch30-ten-year-ai-infrastructure-roadmap-2026-to-2035`
16. `ch30-grand-finale-ai-infrastructure-system-engineering-crown`

#### 4.6.4 第 30 章代码块（6 个）、算例（6 个）、表格（6 张）与思考题（10 道）精确规格清单
- **6 个生产级 Python 代码块**：
  1. `ProfileGuidedKernelSynthesisAgent`：基于编译校验、数值等价门控与 Nsight Compute 计数器反馈的多轮 RLVR 内核进化合成模拟器；
  2. `AutoParallelismIlpMctsSolver`：面向异构超节点拓扑的 5D 混合并行与自定义集合通信树 ILP + MCTS 自动规划器；
  3. `PerKernelDvfsEnergyOptimizer`：算子级动态电压频率调节（Per-Kernel DVFS）与芯片宏单元强化学习布局（`AlphaChip`）能效积（EDP）寻优器；
  4. `AutonomousAiopsStragglerDiagnoser`：融合 NCCL Flight Recorder 环路延迟偏度与光模块 SNR 斜率的亚分钟级故障根因诊断（RCA）器；
  5. `DatacenterDigitalTwinCoSimulator`：十万卡智算中心“工作负载—集合通信网—液冷热力学”三域耦合数字孪生仿真器；
  6. `ThirtyChapterEncyclopediaMasterVerifier`：《AI 基础设施系统工程》全书 30 章五大部类全景账本审计与四大普适物理守恒律综合核算器。
- **6 个标准化编号算例**：
  - `算例 30-1`：`KernelBench` 250 算子在 8 轮 Nsight 反馈下正确率从 `34.0%` 升至 `94.8%` 且复杂融合算子实现 `3.42x`（相对 Eager）/ `1.58x`（相对 `torch.compile`）加速账本；
  - `算例 30-2`：自动合成内核通过寄存器级跨算子融合省去 `38.4 GB/s` 中间激活落盘并自动发现非对称分块 `BM=128, BN=192` 的占用率推导；
  - `算例 30-3`：算子级 DVFS 在访存受限算子（如 AllReduce / Decode Attention）自动降频 `18%` 实现计算性能无损（`< 0.8%`）而动态功耗降低 `26.5%` 的三次方能效方程；
  - `算例 30-4`：`32,768` 卡集群部署自治 AIOps 将隐性故障 $\mathrm{MTTR}$ 从 `4.20 h` 降至 `42.0 s`（压缩 `360x`）、$\mathrm{ETTR}$ 从 `88.4%` 提升至 `97.8%`（半年挽回 `13.5M USD`）推导；
  - `算例 30-5`：数字孪生仿真器在集群投建前识别胖树拓扑局部超额认购（Oversubscription `2:1`）导致 MoE 训练降速 `21.4%` 的预演收益核算；
  - `算例 30-6`：全书 30 章五大部类终极指标总账本（`43,548+` 行、`979+1=980` 幅图、`201+1=202` 个代码块、`228` 个算例、`341` 张表、`492` 块习题、`812` 条脚注、`165` 个实验）与四大守恒律定量闭环验证。
- **6 张标准化斜体表题**：`表 30-1`（手工算子调优、TVM/Triton 自动调优与大模型 RLVR 内核自动合成对比）、`表 30-2`（自动并行系统 Alpa、 Galvatron 与 RL+ILP 混合搜索器能力对比）、`表 30-3`（十万卡集群常见隐性故障特征、遥测信号与自治 AIOps 处置策略矩阵）、`表 30-4`（《AI 基础设施系统工程》全书 30 章五大架构部类核心技术与量化锚点总表）、`表 30-5`（2026–2035 未来十年 AI 基础设施演进路线预测矩阵）、`表 30-6`（第 30 章课后思考题核心锁定参数与标准答案速查表）。
- **10 道课后思考题（`**练习 30-1**` 至 `**练习 30-10**`）**：覆盖 RLVR 内核合成奖励函数防作弊（Reward Hacking）设计、数值等价容差 $\epsilon$ 与 FP8 累加顺序、Nsight Compute 屋脊瓶颈自动分类规则、5D 并行整数线性规划约束条件线性化、算子级 PLL 锁相环调频切换延迟边界、NCCL 环路延迟偏度定位慢节点的图差分算法、数字孪生离散事件仿真加速比、全书四大物理守恒律跨章映射、从“人工设计系统支撑 AI”向“AI 自主演化系统底座”的闭环条件以及全书 30 章综合架构设计大题。

---

## 5. 新增 30 个零依赖确定性可运行实验套件规范（`experiments/`：`135 -> 165`）

每个新增实验目录必须包含 `README.md`、`run.py`（纯 Python 3 标准库实现，零外部依赖，执行耗时 `< 2s`）、`verify.py`、`expected/metrics.json` 与 `results/summary.json`：

### 5.1 存量第 13–24 章新增的 12 个实验套件（使每章实验数 $\ge 3$）
1. `experiments/ch13/13-04`：分布式 KV Cache 块压缩（`LMCache`）与 GDS NVMe 异步预取带宽核算器
2. `experiments/ch14/14-03`：面向 Vera Rubin 与 TPU v7 的 MoE 专家粒度—激活比—带宽比联合 Pareto 寻优器
3. `experiments/ch15/15-03`：芯片内微通道液冷（Microfluidics）热阻网络与吉瓦园区 SMR+储能微网一次调频仿真器
4. `experiments/ch16/16-03`：TEE 机密推理定长填充防时序侧信道开销与乐观抽检可验证推理（Verifiable Inference）博弈模型
5. `experiments/ch17/17-03`：流式因果视频世界模型（20 FPS）滚动 KV 窗口与 `TeaCache` 时间步激活复用模拟器
6. `experiments/ch18/18-03`：多轮 Agentic RLVR 轨迹树环境检查点分叉与异步离策略重要性采样截断（`V-trace`）模拟器
7. `experiments/ch19/19-03`：3DGS 神经传感器闭环重渲染延迟与双 Thor SoC 安全看门狗（Watchdog）降级接管时序验证器
8. `experiments/ch20/20-03`：预实例化流水线模板秒级容错（`Oobleck`）与突发 Poisson 流量下的推理 Goodput SLA 压测器
9. `experiments/ch21/21-03`：智能体 DAG 工具调用投机预执行（Speculative Tool Execution）与 GUI ROI 视觉增量缓存核算器
10. `experiments/ch22/22-03`：全双工数字人 3D 唇形同步（`< 40 ms` 音画抖动对齐）与全球边缘 WebRTC SFU 路由仿真器
11. `experiments/ch23/23-03`：百万册 PDF 视觉解析管线（`MinerU`/`olmOCR`）吞吐与代码仓库级 AST 拓扑打包（Repo-Packing）模拟器
12. `experiments/ch24/24-03`：托卡马克核聚变等离子体磁流体（MHD）亚毫秒级 RL 控制回路与亿级单细胞稀疏基因矩阵加载仿真器

### 5.2 新建第 25–30 章配套的 18 个实验套件（每章 3 个实验）
13. `experiments/ch25/25-01`：万亿参数 DLRM 幂律特征（Zipf $\alpha=1.12$）HBM-CXL-SSD 三级嵌入缓存命中率与延迟仿真器
14. `experiments/ch25/25-02`：生成式推荐 `Meta HSTU` 与 `M-FALCON` 千级候选微批前缀摊销（`44.8x` 算力缩减）性能核算器
15. `experiments/ch25/25-03`：十亿向量 `GPU cuVS CAGRA` 建图加速（`16.91x`）与 `DiskANN` 4 KB 扇区对齐波束搜索模拟器
16. `experiments/ch26/26-01`：块扩散语言模型（`Block Diffusion LLM`）vs. 自回归解码的 Roofline 相变与 `5.93x` 吞吐模拟器
17. `experiments/ch26/26-02`：`Byte Latent Transformer (BLT)` 字节条件熵动态分块（Entropy Patching）与无词表显存节省核算器
18. `experiments/ch26/26-03`：测试时记忆网络（`Titans` / `TTT`）分块对偶矩阵乘（`24.6x` 提速）与 `Coconut` 连续潜空间推理仿真器
19. `experiments/ch27/27-01`：晶圆级/全片上 SRAM 芯片（`WSE-3` / `Groq LPU`）vs. HBM GPU 的延迟—并发度 TCO 交叉点求解器
20. `experiments/ch27/27-02`：高频量化微秒级 AI（内核旁路 + 持久化轮询 Megakernel + `isolcpus`）`16.5 us` 端到端尾延迟模拟器
21. `experiments/ch27/27-03`：`BitNet b1.58` 三值权重紧凑编码（5 权重/Byte）与 `T-MAC` SIMD 查找表（LUT）能效核算器
22. `experiments/ch28/28-01`：跨洲际广域网 `Streaming DiLoCo`（$H=500$ + FP4 伪梯度流水）`2,000x` 带宽压缩与零暴露开销模拟器
23. `experiments/ch28/28-02`：去中心化算力池拜占庭梯度抽检（Proof-of-Learning）与异步伴飞护栏（Speculative Guardrail）模拟器
24. `experiments/ch28/28-03`：`SynthID-Text` 锦标赛采样水印检出率（TPR/FPR）与 `Safetensors` 零拷贝页对齐校验器
25. `experiments/ch29/29-01`：`FlashMLA` 非对称头维（`512+64`）乒乓流水带宽利用率（`3,000 GB/s`）与 `DeepGEMM` 二级累加核算器
26. `experiments/ch29/29-02`：`DeepEP` 双模式（NVLink+RDMA / 纯 IBGDA 零 SM 占用）与 `DualPipe` 双向流水线零气泡验证器
27. `experiments/ch29/29-03`：`3FS` 180 节点 `6.6 TB/s` 链式复制吞吐与 `KTransformers` 单卡 GPU + 双路 CPU AMX 异构 MoE 规划器
28. `experiments/ch30/30-01`：基于 Nsight 硬件剖析反馈的多轮 RLVR 内核自动合成（`KernelBench` / `Kevin`）进化收敛模拟器
29. `experiments/ch30/30-02`：5D 自动并行 ILP+MCTS 求解器与十万卡自治 AIOps 亚分钟级故障隔离（`MTTR` 压缩 `360x`）仿真器
30. `experiments/ch30/30-03`：《AI 基础设施系统工程》全书 30 章五大部类全景指标审计与四大物理守恒律综合核算器

---

## 6. 全书 30 章构建、核心原则校验、MkDocs 站点、PDF/EPUB 与导航升级规范

在扩充进程与最后一个批次（Batch 15）合入第 29–30 章时，同步升级以下全书构建与校验管线：
1. **[`manuscripts/ch01/build_roadmap.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/ch01/build_roadmap.py) 与单章活跃门禁（`manuscripts/ch09..30/verify.py`）**：
   - 在 Batch 1 更新 `manuscripts/ch01/build_roadmap.py`，将未编号全书导读总图 `ch01/figure-1-book-roadmap.{svg,png,pdf}` 升级为 30 章五大架构部类全景图；
   - 对 `Ch01–Ch30` 每章同步更新 `manuscripts/chXX/figure-index.json`、`manuscripts/chXX/manifest.json`（SHA-256）、`teaching-layout-validation.json`、`math-validation.json` 与 `preview_output/` HTML，并运行全部现行活跃单章门禁 `manuscripts/ch09..30/verify.py`（注：严禁执行历史归档脚本 `manuscripts/verify_teaching_book.py` 与 `manuscripts/ch02..08/verify.py`，以免污染受跟踪的历史 `validation.json`）。
2. **[`scripts/verify_core_principles.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/scripts/verify_core_principles.py)**：
   - 将第 10 行 `for n in range(1,25):` 升级为 `for n in range(1,31):`，验证全部 30 章的末节标题 `本章小结`、`979` 幅编号章节配图顺序与文件完整性（连同 `ch01/figure-1-book-roadmap` 共 `980` 幅配图）、`492` 个思考题块（`491` 个唯一题号）、`812` 条脚注定义与引用一致性，并输出更新后的 `archive/reviews/core-principles-revision-2026-09-10/core-validation.json`。
3. **[`scripts/build_site.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/scripts/build_site.py) & [`scripts/check_site.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/scripts/check_site.py)**：
   - 在 `scripts/build_site.py` 第 57–58 行将中文版章节上限从 `n <= 24` / `range(25)` 升级为 `n <= 30` / `range(31)`（共 staging `32` 个中文 Markdown 页面：首页 + 前言 + 30 章）；
   - 在 `scripts/check_site.py` 第 46、54 行将 `max_ch = 25` 升级为 `max_ch = 31`，校验通过输出：`PASS: 78 pages; 3 editions with preface, chapters (30 zh / 12 translations) and search index; internal links and images`。
4. **[`book/build_pdf.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/book/build_pdf.py)、[`book/build_epub.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/book/build_epub.py)、[`book/check_ci_pdf.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/book/check_ci_pdf.py)、[`book/check_typography.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/book/check_typography.py) & [`book/verify_pdf.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/book/verify_pdf.py)**：
   - 将 `book/build_pdf.py` 与 `book/build_epub.py` 中文版章节范围从 `range(1,25)` / `range(25)` 升级为 `range(1,31)` / `range(31)`；
   - 在 `book/check_ci_pdf.py` 的 `expected_text`、[`book/check_typography.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/book/check_typography.py) 与 [`book/verify_pdf.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/book/verify_pdf.py) 中同步升级至 30 章覆盖范围并加入第 25–30 章特征词（如 `'生成式推荐'`、`'自演进'`）。
5. **全书导航与总览文档同步更新**：
   - 同步更新 [`README.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/README.md)、[`website/index.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/website/index.md)、[`manuscripts/README.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/README.md)、[`manuscripts/00-前言.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/00-%E5%89%8D%E8%A8%80.md)、[`case-studies/README.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/README.md) 与 [`experiments/README.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/README.md)，完整呈现全书 30 章五大部类目录与 `165` 个实验套件索引。

---

## 7. 15 批次执行路线图（15-Batch Execution Roadmap）与强制 6 步 SOP

全书 30 章按每批次 2 章划分为 **15 个标准批次（Batches 1–15）**：

| 批次 | 覆盖章节 | 批次核心交付物摘要 |
| :---: | :---: | :--- |
| **Batch 1** | **Ch01 + Ch02** | 重绘未编号 30 章全书导读总图 `figure-1-book-roadmap` + 新增 `图 1-27..28`、`图 2-45..46` + `+5` 代码块 + `+4` 算例 + `+4` 习题 |
| **Batch 2** | **Ch03 + Ch04** | 新增 `图 3-40..41`、`图 4-52..53`（NVFP4 微缩放 / CXL 3.1 内存池）+ `+5` 代码块 + `+4` 算例 + `+4` 习题 |
| **Batch 3** | **Ch05 + Ch06** | 新增 `图 5-49..50`、`图 6-66..67`（NVL576 正交背板损耗 / 3D Torus 对分带宽）+ `+4` 代码块 + `+4` 算例 + `+4` 习题 |
| **Batch 4** | **Ch07 + Ch08** | 新增 `图 7-70..71`、`图 8-45..46`（vLLM V1 / SGLang 零开销调度）+ `+5` 代码块 + `+4` 算例 + `+4` 习题 |
| **Batch 5** | **Ch09 + Ch10** | 新增 `图 9-55..56`（Dynamo / KTransformers）、`图 10-57..58` + `+5` 代码块 + `+4` 算例 + `+4` 习题 |
| **Batch 6** | **Ch11 + Ch12** | 新增 `图 11-58..59`（CuMemMap 休眠）、`图 12-43..44`（ExecuTorch / TB5 RDMA）+ `+6` 代码块 + `+4` 算例 + `+4` 习题 |
| **Batch 7** | **Ch13 + Ch14** | 新增 `图 13-27..28`、`图 14-27..28` + `+6` 代码块 + `+4` 习题 + 实验 `13-04`、`14-03` |
| **Batch 8** | **Ch15 + Ch16** | 新增 `图 15-25..26`、`图 16-25..26` + `+6` 代码块 + `+4` 习题 + `+24` 脚注 + 实验 `15-03`、`16-03` |
| **Batch 9** | **Ch17 + Ch18** | 新增 `图 17-21..22`、`图 18-21..22` + `+2` 代码块 + `+2` 算例 + `+4` 习题 + `+24` 脚注 + 实验 `17-03`、`18-03` |
| **Batch 10** | **Ch19 + Ch20** | 新增 `图 19-19..20`、`图 20-19..20` + `+2` 代码块 + `+2` 算例 + `+4` 习题 + `+24` 脚注 + 实验 `19-03`、`20-03` |
| **Batch 11** | **Ch21 + Ch22** | 新增 `图 21-17..18`、`图 22-17..18` + `+2` 代码块 + `+4` 算例 + `+4` 习题 + `+24` 脚注 + 实验 `21-03`、`22-03` |
| **Batch 12** | **Ch23 + Ch24** | 新增 `图 23-17..18`、`图 24-17..18` + `+4` 代码块 + `+4` 算例 + `+4` 习题 + `+24` 脚注 + 实验 `23-03`、`24-03` |
| **Batch 13** | **New Ch25 + New Ch26** | 新建第 25 章（生成式推荐 HSTU & GPU 向量库）与第 26 章（dLLM / BLT / Titans）：`+32` 图、`+12` 代码、`+12` 算例、`+20` 习题、`+30` 脚注、实验 `25-01..03` & `26-01..03` |
| **Batch 14** | **New Ch27 + New Ch28** | 新建第 27 章（晶圆级/3D-SRAM & 微秒级 AI & BitNet）与第 28 章（DiLoCo & 护栏 & 水印）：`+32` 图、`+12` 代码、`+12` 算例、`+20` 习题、`+30` 脚注、实验 `27-01..03` & `28-01..03` |
| **Batch 15** | **New Ch29 + New Ch30 + 30 章全量集成** | 新建第 29 章（DeepSeek 全栈解剖）与第 30 章（自演进 AI Infra & 30 章收束）：`+32` 图、`+12` 代码、`+12` 算例、`+20` 习题、`+30` 脚注、实验 `29-01..03` & `30-01..03` + 全书 30 章构建/站点/EPUB 升级 |

### 每个批次必须严格执行的 6 步标准作业程序（6-Step Batch SOP）
1. **Step 1（确定性实验先行）**：编写或更新本批次对应的 `experiments/chXX/XX-YY/run.py` 与 `verify.py`，生成锁定的 `expected/metrics.json` 与 `results/summary.json`，确保所有算例、图表与习题锚点拥有统一的机器校验真值源。
2. **Step 2（出版级 420pt 配图生成与逐图视觉审查）**：更新 `manuscripts/chXX/build.py`（及 `figure-data.json`），使用 `/Users/quan.zheng/.cache/uv/archive-v0/McStPUda9BXiDOrZ/bin/python3` 生成 `.svg`/`.png`/`.pdf`（`0 layout warnings`），并调用 `read_file` 逐一视觉检查每一张新生成的 `.png`，确保零文字截断、零重叠、零内部图号。
3. **Step 3（手稿正文、代码块、算例、表题、习题与脚注扩写）**：原位扩写或新建章节 Markdown，严格核对零感叹号、零未转义 `$`、零繁体字、零原生 HTML 标签。
4. **Step 4（自动化门禁全量验证）**：刷新本批次章节的 `figure-index.json`、`manifest.json`（SHA-256）、`teaching-layout-validation.json`、`math-validation.json` 与 `preview_output/` HTML，运行现行活跃单章 `manuscripts/ch09..30/verify.py`（严禁运行历史归档脚本 `manuscripts/verify_teaching_book.py` 与 `manuscripts/ch02..08/verify.py`）、`scripts/verify_core_principles.py`、`scripts/check_no_html.py` 及 `.venv-site/bin/python3 scripts/build_site.py && .venv-site/bin/python3 scripts/check_site.py`，确保 100% 零错误通过。
5. **Step 5（独立 Gemini 4 Argon High 子代理评审与迭代收敛）**：调用 `invoke_subagent`（`TypeName="self"`, `Model="inherit"`）启动独立评审子代理对本批次手稿、代码、算例、配图与实验进行严苛审计，解决全部评审意见直至子代理明确输出 `Status: Approved / Converged`。
6. **Step 6（批次 Git 提交与远端推送）**：更新 `task.md` 进度，执行 `git add`、`git commit` 与 `git push origin main`，确保持久化收敛后再进入下一批次。
