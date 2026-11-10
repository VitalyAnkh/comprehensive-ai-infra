# 深入理解 AI Infra：量化分析与系统设计（清晰易读重述版 / Plain Edition）

[![Build](https://github.com/VitalyAnkh/plain-ai-infra-book/actions/workflows/book-site.yml/badge.svg)](https://github.com/VitalyAnkh/plain-ai-infra-book/actions/workflows/book-site.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Upstream](https://img.shields.io/badge/upstream-bojieli%2Fai--infra--book-brightgreen)](https://github.com/bojieli/ai-infra-book)

**简体中文** · [English](book-en/) · [繁體中文](book-zh-tw/) · [Русский — community edition](book-ru/)

> [!IMPORTANT]
> **关于本仓库（清晰易读重述版）与版权说明**
>
> - **来源与致谢**：本仓库（[`VitalyAnkh/plain-ai-infra-book`](https://github.com/VitalyAnkh/plain-ai-infra-book)）修改自上游作者[李博杰](https://01.me/)（[@bojieli](https://github.com/bojieli)）所著的开源技术书籍[《深入理解 AI Infra：量化分析与系统设计》](https://github.com/bojieli/ai-infra-book)。
> - **改写目标**：原书初稿有 AI 参与写作，部分段落存在拗口难懂的用词、生造概念与欧化长句。本版本在**严格保留原书全部技术事实、数学公式、量化算例、图表与配套实验**的基础上，去掉各类晦涩的 AI 行话（AI Jargon），将正文改写为人类更易读懂、文雅自然、符合现代汉语习惯的清晰书面语。
> - **在线阅读（本仓库重述版）**：**<https://vitalyankh.github.io/plain-ai-infra-book/>**
> - **版权与许可**：本仓库严格遵循上游作者的版权与开源许可协议，采用 **[Apache License 2.0](LICENSE)** 许可发布。原书著作权归上游原作者 **Bojie Li（李博杰）** 所有（Copyright © 2026 Bojie Li）。

> [!TIP]
> **下载与阅读** · [格式与版本说明](#版本与格式说明)
>
> - **简体中文（本仓库清晰易读重述版）**：[在线阅读（GitHub Pages）](https://vitalyankh.github.io/plain-ai-infra-book/) · [GitHub 仓库](https://github.com/VitalyAnkh/plain-ai-infra-book)
> - **简体中文（上游原版，作者 [@bojieli](https://github.com/bojieli)）**：[PDF](https://github.com/bojieli/ai-infra-book/releases/latest/download/AI-Infra-Book.pdf) · [EPUB](https://github.com/bojieli/ai-infra-book/releases/latest/download/AI-Infra-Book.epub) · [原版在线阅读](https://bojieli.github.io/ai-infra-book/)
> - **English（社区翻译，[@tg1482](https://github.com/tg1482)）**: [PDF](https://github.com/bojieli/ai-infra-book/releases/latest/download/AI-Infra-Book-EN.pdf) · [EPUB](https://github.com/bojieli/ai-infra-book/releases/latest/download/AI-Infra-Book-EN.epub) · [Read online](https://vitalyankh.github.io/plain-ai-infra-book/en/)
> - **繁體中文（社区翻译，[@edward821220](https://github.com/edward821220)）**：[PDF](https://github.com/bojieli/ai-infra-book/releases/latest/download/AI-Infra-Book-ZH-TW.pdf) · [EPUB](https://github.com/bojieli/ai-infra-book/releases/latest/download/AI-Infra-Book-ZH-TW.epub) · [線上閱讀](https://vitalyankh.github.io/plain-ai-infra-book/zh-tw/)
> - **Русский — community edition, [@ilkruglov](https://github.com/ilkruglov)**: [PDF](https://github.com/bojieli/ai-infra-book/releases/latest/download/AI-Infra-in-Depth-RU.pdf) · [EPUB](https://github.com/bojieli/ai-infra-book/releases/latest/download/AI-Infra-in-Depth-RU.epub) · [Онлайн-чтение](https://vitalyankh.github.io/plain-ai-infra-book/ru/) · [О переводе](book-ru/)

《深入理解 AI Infra》是 GitHub 上获得 **50k+ Star** 的[《深入理解 AI Agent：设计原理与工程实践》](https://github.com/bojieli/ai-agent-book)的姊妹篇。

写完[《深入理解 AI Agent》](https://github.com/bojieli/ai-agent-book)后，在与读者交流的过程中，我越来越感到：要开发好基于模型的应用，还需要理解它赖以运行的基础设施。大多数软件工程师不必亲自开发操作系统、编译器和芯片，却仍要学习操作系统、编译原理和计算机体系结构，因为申请内存、读取文件、调用函数，背后都有资源与时间代价。基于模型开发应用也是如此。延迟相差几倍，产品体验就可能完全不同；成本相差一个数量级，能够支撑的商业模式也随之改变。

更深层的变化是**编程抽象的上移：从操作系统到模型上下文**。传统的操作系统、编译器和硬件要为事先未知的各种程序提供通用能力，系统优化总要在可编程性与性能之间取舍。如今 LLM 成了最重要的应用，从算子执行到分布式调度，都可以针对特定的模型和加速器架构优化；模型设计也开始反过来适应硬件，DeepSeek V4/V4.1 重新设计长上下文的表示方式，就是一例。从某种意义上说，**模型成了 LLM 时代的操作系统，AI Infra 成了 LLM 时代的计算机体系结构**。《计算机体系结构：量化研究方法》是我在体系结构领域的入门书，而 AI Infra 领域还缺少一本从硬件约束和模型架构出发、量化推导系统设计的书，这是我写作本书的动机。

贯穿全书的方法是**从约束推导设计**：先明确任务与质量要求，列出计算、存储、通信和依赖关系，对照硬件的容量、带宽和算力做数量级估算。这类估算人容易出错，AI 也一样：只算权重读取而忘了 KV 缓存，按峰值算力推算速度而不查带宽能否供给，把工作平分给多张卡却遗漏卡间通信，漏掉任何一项，结论都可能偏离几倍甚至几个数量级。估算还有另一层用意：读博时导师张霖涛博士反复叮嘱，优化一定要做到物理所允许的极限。本书贯彻这一习惯，先按第一性原理算出硬件允许的上限，再看实测离上限还有多远；差距不是模型漏了项，就是系统有可以去掉的开销。从 FPGA 加速 Bing 搜索排序、昇腾 AKG 算子生成到 UB 万卡互联，我反复遇到的是同一条线索：**数据搬移**。本书因此反复追问五个问题：**搬什么、搬多少、搬几次、经过哪里、谁必须等它。** 更多写作背景见[前言](manuscripts/00-前言.md)。

目前书稿仍是初稿，正在持续修订。

## 内容目录

| 章 | 主题 | 主要问题 |
| :--: | --- | --- |
| 1 | [初识 AI Infra](<manuscripts/01-初识 AI Infra.md>) | 一次生成需要多少显存、计算和数据读写？ |
| 2 | [模型架构](manuscripts/02-模型架构.md) | 注意力、历史状态与专家结构如何改变系统需求？ |
| 3 | [推理与训练负载](manuscripts/03-推理与训练负载.md) | 任务阶段、到达模式和状态寿命如何影响资源需求？ |
| 4 | [加速器架构](manuscripts/04-加速器架构.md) | 如何在计算、存储、带宽、功耗与成本之间取舍？ |
| 5 | [算子与运行时](manuscripts/05-算子与运行时.md) | 融合、复用、并发和调度如何减少执行开销？ |
| 6 | [超节点](manuscripts/06-超节点.md) | 多设备协作如何平衡容量、吞吐和同步代价？ |
| 7 | [数据中心网络](manuscripts/07-数据中心网络.md) | 网络带宽、通信方式和拥塞如何影响计算效率？ |
| 8 | [推理优化](manuscripts/08-推理优化.md) | 批处理、KV 管理、卸载与推测解码何时有效？ |
| 9 | [分布式推理](manuscripts/09-分布式推理.md) | 如何放置计算和状态，并处理扩缩容与恢复？ |
| 10 | [训练系统](manuscripts/10-训练系统.md) | 如何安排显存、通信和重算，让训练更高效？ |
| 11 | [资源调度与运行环境](manuscripts/11-资源调度与运行环境.md) | 模型服务和工具环境如何共享资源，减少等待？ |
| 12 | [端边云协同](manuscripts/12-端边云协同.md) | 任务放在本地、边缘还是云端，如何兼顾效果、延迟和成本？ |
| 13 | [AI 原生存储系统与数据底座](<manuscripts/13-AI 原生存储系统与数据底座.md>) | 如何消除页缓存拷贝、闪存读写放大与多模态数据供给瓶颈？ |
| 14 | [模型与系统硬件协同设计](manuscripts/14-模型与系统硬件协同设计.md) | 硬件边界如何反向塑造模型结构，如何用五步定量法贯通跨层设计？ |
| 15 | [吉瓦级 AI 数据中心、电力热学与跨地域协同训练](<manuscripts/15-吉瓦级 AI 数据中心、电力热学与跨地域协同训练.md>) | 如何跨越供电、液冷与功率阶跃边界，支撑吉瓦级园区与跨地域协同训练？ |
| 16 | [机密 AI 计算、可信执行环境与安全沙箱底座](<manuscripts/16-机密 AI 计算、可信执行环境与安全沙箱底座.md>) | 如何构建 CPU—GPU 硬件可信执行环境、远程证明链与微虚拟机安全沙箱？ |
| 17 | [多模态、视频生成（DiT）与世界模型基础设施](<manuscripts/17-多模态、视频生成（DiT）与世界模型基础设施.md>) | 如何破解三维时空词元平方爆炸，构建 E-P-D-G 四池分离与流式生成系统？ |
| 18 | [大规模后训练（RLVR）、异步强化学习与推理时扩展系统](<manuscripts/18-大规模后训练（RLVR）、异步强化学习与推理时扩展系统.md>) | 如何编排多模型异构显存、驯服长思维链幂律长尾并实现秒级沙箱与权重热同步？ |
| 19 | [具身智能（VLA）、实时控制与大规模物理仿真底座](<manuscripts/19-具身智能（VLA）、实时控制与大规模物理仿真底座.md>) | 如何满足毫秒级硬实时控制（WCET）上限并构建万级环境 GPU 原生仿真底座？ |
| 20 | [十万卡集群可靠性工程、静默数据损坏（SDC）防御与基准评测科学](<manuscripts/20-十万卡集群可靠性工程、静默数据损坏（SDC）防御与基准评测科学.md>) | 如何在线拦截静默比特翻转（SDC）、实现秒级无重启自愈并构建可证伪评测体系？ |
| 21 | [自主智能体基础设施（Agentic Infra）、Computer Use 沙箱与分层 KV 记忆底座](<manuscripts/21-自主智能体基础设施（Agentic Infra）、Computer Use 沙箱与分层 KV 记忆底座.md>) | 如何支撑 >95% WORM 前缀复用、四级 KV 缓存全局路由、毫秒级 CoW 沙箱分叉与幂等外发箱？ |
| 22 | [全双工实时语音（Full-Duplex Speech）、神经音频编解码与原生全模态流式底座](<manuscripts/22-全双工实时语音（Full-Duplex Speech）、神经音频编解码与原生全模态流式底座.md>) | 如何突破 200 ms 对话感知阈值，协同 RVQ 编解码、双层 Transformer 与按播放游标对齐的打断回滚？ |
| 23 | [百亿亿次 Token 数据工程、GPU 加速模糊去重与合成数据工厂底座](<manuscripts/23-百亿亿次 Token 数据工程、GPU 加速模糊去重与合成数据工厂底座.md>) | 如何从 100 PB 原始网页经 GPU 模糊去重与离线合成数据工厂锻造万亿级精品词元？ |
| 24 | [科学智能（AI for Science）基础设施：生物大分子、百万碱基基因组与物理场网格系统](<manuscripts/24-科学智能（AI for Science）基础设施：生物大分子、百万碱基基因组与物理场网格系统.md>) | 如何攻克生物大分子三角注意力、百万碱基基因组 FlashFFTConv 与全球气象网格 Halo 交换？ |
| 25 | [生成式推荐系统（Generative Recommenders）、万亿参数稀疏表（DLRM）与十亿级 GPU 向量检索引擎](<manuscripts/25-生成式推荐系统（Generative Recommenders）、万亿参数稀疏表（DLRM）与十亿级 GPU 向量检索引擎.md>) | 如何实现从 DLRM 内存墙向 HSTU 生成式序列转导的跃迁，并协同万亿参数嵌入表与十亿级 GPU 向量检索？ |
| 26 | [后自回归生成范式基础设施：扩散大语言模型（dLLM）、字节级无分词架构（BLT）与测试时记忆网络（Titans-TTT）](<manuscripts/26-后自回归生成范式基础设施：扩散大语言模型（dLLM）、字节级无分词架构（BLT）与测试时记忆网络（Titans-TTT）.md>) | 如何用并行去噪迭代、字节级信息熵动态分块与测试时内循环梯度记忆突破自回归与 KV 缓存瓶颈？ |
| 27 | [晶圆级与 3D-SRAM 专用推理芯片（Cerebras-Groq）、微秒级极低延迟 AI 与三值量化（BitNet b1.58）系统](<manuscripts/27-晶圆级与 3D-SRAM 专用推理芯片（Cerebras-Groq）、微秒级极低延迟 AI 与三值量化（BitNet b1.58）系统.md>) | 如何凭全片上 SRAM 与确定性编译跨越 HBM 带宽墙，并用 1.58-bit 三值加法查找表重塑端侧能效？ |
| 28 | [主权 AI、广域网去中心化低通信训练（DiLoCo）、实时护栏（Guardrails）与密码学水印安全底座](<manuscripts/28-主权 AI、广域网去中心化低通信训练（DiLoCo）、实时护栏（Guardrails）与密码学水印安全底座.md>) | 如何在跨洲低带宽广域网上实现高算力利用率协同训练，并在微秒级预算内挂载实时安全护栏与零失真水印？ |
| 29 | [DeepSeek 开源基础设施全栈解剖与极客工程实战：FlashMLA、DeepGEMM、DeepEP、DualPipe、3FS、Mooncake 与 KTransformers](<manuscripts/29-DeepSeek 开源基础设施全栈解剖与极客工程实战：FlashMLA、DeepGEMM、DeepEP、DualPipe、3FS、Mooncake 与 KTransformers.md>) | 如何将算子层、通信层、流水线层、存储池化层与异构卸载层贯通为生产级全栈开源基础设施？ |
| 30 | [自演进 AI 基础设施（AI for AI Infra）：大模型驱动的算子自动合成、强化学习编译寻优与三十章全栈终极收束](<manuscripts/30-自演进 AI 基础设施（AI for AI Infra）：大模型驱动的算子自动合成、强化学习编译寻优与三十章全栈终极收束.md>) | 如何用强化学习与进化搜索自主合成超越人类专家的底层内核，并收束全书三十章五维守恒定律总账本？ |

## 适合谁读

如果你已经调用过模型 API，或在自己的机器上运行过模型，想进一步弄清模型为什么慢、如何降低成本，这本书可以作为起点。从事系统、网络和芯片工作的工程师，以及相关方向的研究人员和学生，也能从书中看到各层设计与实际模型任务的联系。

阅读时需要一些 Python、线性代数和计算机系统基础，具体要求见[前言](manuscripts/00-前言.md)。建议先读第 1—3 章，了解模型与负载，再根据自己的兴趣选择五条重点路线：

- **模型、智能体与全模态应用路线**：可以重点读第 8、9、17、18、19、21、22、25、26 章，再看第 11、12 章的任务环境与部署，以及第 14、16、20、28 章的架构协同、安全沙箱、评测科学与实时护栏；遇到容量、算子或通信问题时，回到第 4—7 章追踪原因。
- **推理、训练与万卡集群系统路线**：可以重点读第 5—7 章，再看第 8—11、13、15、16、18、20、21、28、29、30 章，检查这些机制如何影响分布式推理、训练、AI 原生存储、吉瓦级园区、机密计算、智能体沙箱、十万卡容错自愈、DeepSeek 全栈与自演进基础设施。
- **硬件、芯片与互连体系结构路线**：可以重点读第 4—7 章、第 14—20 章与第 27、29、30 章，并结合第 21—26 章的前沿算例，检查硬件指标如何转化为实际的服务能力。
- **大规模数据工程、推荐检索与科学智能（AI4Science）路线**：可以重点读第 10、13、18、23、24、25 章，系统掌握百 PB 网页 GPU 模糊去重、万亿词元离线合成数据工厂、生成式推荐与十亿级 GPU 向量检索、生物大分子预测、百万碱基基因组与高分辨率物理场网格并行系统。
- **全书百科全书系统学习路线**：可以按第 1—30 章目录顺序阅读，配合 165 个计算与实验套件，逐步核对自己的理解。

遇到书中的算例，不妨先自己估一下，再看推导和实验结果。也可以换成你正在使用的模型和硬件，看看结论是否改变。

## 配套计算与实验

[配套计算工具](calculations/README.md)可以用来复算书中的数字，也可以换一组模型和输入，估算资源需求。工具附有模型配置和[结果索引](calculations/results/README.md)，静态计算只需 Python 3.10+ 标准库，无需 GPU 或模型权重。

```bash
git clone https://github.com/bojieli/ai-infra-book.git
cd ai-infra-book

# 查看模型支持情况
python3 calculations/calc.py models

# 估算 Qwen3-8B 在 8192-token prefill 下的逐算子资源需求
python3 calculations/calc.py forward --model qwen3-8b --tokens 8192 --format md
```

仓库使用 [Git LFS](https://git-lfs.com/) 保存论文和较大的输入与测量记录，合计约 20 GB。为避免克隆时全部下载，[.lfsconfig](.lfsconfig) 默认跳过所有 LFS 文件，工作区中只留下指针；正文、配图和静态计算都不需要它们。复现某个实验时，只下载对应目录：

```bash
git lfs install
git lfs pull --include="experiments/ch05/05-01/**" --exclude=""
```

[配套实验](experiments/README.md)按章节存放在 `experiments/chXX/XX-YY/` 中，每个实验都附有运行方法、输入条件和结果说明。需要 GPU 的实验会注明硬件与依赖要求；没有相应设备，也可以先阅读已有记录。复现或引用结果时，请留意所用的书稿版本、模型、硬件和输入参数。

## 本地构建

**阅读网站**（Python 3.10+）：

```bash
python3 -m venv .venv-site
source .venv-site/bin/activate
python -m pip install -r website/requirements.txt
python scripts/build_site.py
python scripts/check_site.py
python scripts/build_site.py --serve
```

预览地址为 <http://127.0.0.1:8000>，简体中文位于根目录，英文、繁體中文和俄语分别位于 `en/`、`zh-tw/` 和 `ru/`。生成文件位于 `build/`，详细说明见[网站构建与发布](website/README.md)。

**全书 PDF 与 EPUB**（另需 Pandoc、XeLaTeX 和字体）：

```bash
bash book/build_pdf.sh                        # 简体中文 → book/AI-Infra-Book.pdf
bash book-en/build_pdf.sh                     # English → book-en/AI-Infra-Book-EN.pdf
bash book-zh-tw/build_pdf.sh                  # 繁體中文 → book-zh-tw/AI-Infra-Book-ZH-TW.pdf
python3 book/build_epub.py --edition zh       # EPUB 只需 Pandoc 与 Poppler；--edition 可选 zh、en、zh-tw、ru
```

译本配图存于 Git LFS，构建前先下载：`git lfs pull --include="book-en/images/**,book-zh-tw/images/**" --exclude=""`。依赖、字体及单章编译方法见 [PDF 编译说明](book/README.md)。GitHub Actions 会检查 Pull Request 的网站与各语言的 PDF、EPUB 构建；推送到 `main` 后自动发布 Release（四种语言各含 PDF 与 EPUB）并部署四种语言的在线阅读网站。

## 仓库结构

| 目录 | 内容 |
| --- | --- |
| [manuscripts/](manuscripts/README.md) | 前言、三十章正文、配图与绘图脚本 |
| [experiments/](experiments/README.md) | 按章节组织的 165 个实验套件与运行记录 |
| [calculations/](calculations/README.md) | 资源计算工具、固定输入与复算结果 |
| [case-studies/](case-studies/README.md) | 模型、硬件和系统案例分析，按章节索引 |
| [references/](references/README.md) | 引用资料、来源清单与版本快照 |
| [research/](research/README.md) | 支撑正文的专题调研，以及各章修订记录 |
| [book/](book/README.md) | PDF 模板、构建与校验工具 |
| [book-en/](book-en/README.md) | 英文版（社区翻译）：前言与十二章正文、英文配图、翻译工具与 PDF 构建脚本 |
| [book-zh-tw/](book-zh-tw/README.md) | 繁體中文版（社区贡献）：前言与十二章正文、繁體配图、翻译工具与 PDF 构建脚本 |
| [website/](website/README.md)、[scripts/](scripts/README.md) | 网站资源、构建与检查脚本 |
| [archive/](archive/README.md) | 历史大纲、审阅和写作协调记录 |

## 参与贡献

书稿仍是初稿，欢迎读者参与改进。以下几个方向都有价值，欢迎认领：

- **指出并修正错误**：数字算错、单位或量级不对、引用与原文不符、公式或图表有误。发现一处就值得提一处，[勘误 Issue](https://github.com/bojieli/ai-infra-book/issues/new?template=erratum.yml) 或直接提 PR 都可以。
- **改写讲得不清楚的地方**：推导跳步、概念没有在首次出现时交代、例子不好懂。读不顺的地方多半是书写得不好，欢迎[提出来](https://github.com/bojieli/ai-infra-book/issues/new?template=question.yml)，也欢迎直接给出更好的写法。
- **补充遗漏的重要内容**：某个该讲的机制、架构或权衡没有写进来。
- **修复配套代码的 bug**：`experiments/` 的实验代码与 `calculations/` 的计算 CLI，欢迎修正错误、补充测试或改进可用性。
- **改进网页版**：[在线阅读版](https://bojieli.github.io/ai-infra-book/)的排版、导航、搜索和移动端体验都还有提升空间。
- **翻译**：英文版与繁體中文版已由社区贡献，欢迎翻译为更多语言，可参考 `book-en/` 和 `book-zh-tw/` 的组织方式；也欢迎跟进英文版，使其与中文原版同步。翻译前请先开 Issue 说明计划，便于协调进度、避免重复劳动。

正文的唯一来源是 `manuscripts/` 下的 Markdown，网页版和 PDF 都由它构建生成，改正文请直接改这里。

## 作者与致谢

作者：[李博杰](https://01.me/)（[@bojieli](https://github.com/bojieli)）。

感谢这些年一起做研究和工程的合作者，也感谢相关论文、开源项目与技术文档的作者，以及参与勘误和实验复现的读者。详细致谢见[前言](manuscripts/00-前言.md)，引用来源见正文脚注和[参考资料库](references/README.md)。本书 PDF 沿用《深入理解 AI Agent》的 ElegantBook / XeLaTeX 模板。

英文版由 [@tg1482](https://github.com/tg1482) 翻译并贡献（[#4](https://github.com/bojieli/ai-infra-book/pull/4)）。

繁體中文版由 [@edward821220](https://github.com/edward821220) 翻译并贡献（[#10](https://github.com/bojieli/ai-infra-book/pull/10)）。

俄语 community edition 由 [@ilkruglov](https://github.com/ilkruglov) 组织并完成翻译与校验（[#20](https://github.com/bojieli/ai-infra-book/pull/20)）。

## 贡献者

感谢所有为本书贡献内容、代码、翻译和修正的贡献者！点击头像即可访问他们的 GitHub 主页，也欢迎你加入。

<!-- CONTRIBUTORS: GitHub contributors for bojieli/ai-infra-book, plus translators credited above. -->
<p>
  <a href="https://github.com/bojieli"><img src="https://avatars.githubusercontent.com/u/1421793?v=4&amp;s=128" width="64" height="64" alt="@bojieli" title="@bojieli" /></a>
  <a href="https://github.com/tg1482"><img src="https://avatars.githubusercontent.com/u/31661606?v=4&amp;s=128" width="64" height="64" alt="@tg1482" title="@tg1482" /></a>
  <a href="https://github.com/edward821220"><img src="https://github.com/edward821220.png?s=128" width="64" height="64" alt="@edward821220" title="@edward821220" /></a>
  <a href="https://github.com/determine123"><img src="https://avatars.githubusercontent.com/u/182319045?v=4&amp;s=128" width="64" height="64" alt="@determine123" title="@determine123" /></a>
  <a href="https://github.com/sleeep-early"><img src="https://avatars.githubusercontent.com/u/186607517?v=4&amp;s=128" width="64" height="64" alt="@sleeep-early" title="@sleeep-early" /></a>
</p>
<!-- /CONTRIBUTORS -->

[查看 GitHub 上的全部贡献记录 →](https://github.com/bojieli/ai-infra-book/graphs/contributors)

## 许可与版权说明

本书系在[李博杰（Bojie Li）](https://github.com/bojieli)所著上游开源书籍[《深入理解 AI Infra：量化分析与系统设计》](https://github.com/bojieli/ai-infra-book)基础上改写而成的清晰易读重述版（Plain Edition）。

本书原创正文、配图、配套代码及本仓库的文字润色修改均遵循上游作者的版权与开源协议，采用 [Apache License 2.0](LICENSE) 许可发布。Copyright © 2026 Bojie Li（李博杰）。

仓库中的第三方代码、字体、模板与参考资料保留各自的版权和许可声明，不因收录于本仓库而改用 Apache-2.0；具体来源和使用条件见相应目录。

## 版本与格式说明

- **在线阅读（本仓库重述版）**：[阅读网站（Plain Edition）](https://vitalyankh.github.io/plain-ai-infra-book/)提供章节导航、全文搜索和深色模式，右上角可切换语言；上游原版在线阅读见 [bojieli.github.io/ai-infra-book](https://bojieli.github.io/ai-infra-book/)。
- **PDF**：由 XeLaTeX 排版，公式、表格、脚注和交叉引用完整，推荐通读时使用。GitHub 直接显示 Markdown 时，LaTeX 公式和部分排版常常渲染不全或错位；上方目录链接到各章 Markdown 源文件，便于查找原文和提交勘误。
- **EPUB**：字号可调，版面随屏幕自动重排，适合手机和电子书阅读器。
- **自动发布**：每次更新 `main` 后自动构建并部署到 GitHub Pages（[vitalyankh.github.io/plain-ai-infra-book](https://vitalyankh.github.io/plain-ai-infra-book/)）；上游原版 PDF/EPUB 自动发布见[上游 Releases](https://github.com/bojieli/ai-infra-book/releases)。
- **翻译版本与译者**：简体中文为原版，作者为 [Bojie Li（李博杰）](https://github.com/bojieli)，正文位于 [`manuscripts/`](manuscripts/README.md)。英文版由社区译者 [@tg1482](https://github.com/tg1482) 翻译，位于 [`book-en/`](book-en/)，包含前言与十二章正文、重绘为英文标注的配图和独立的构建脚本。繁體中文版由社区贡献者 [@edward821220](https://github.com/edward821220) 翻译并整理，位于 [`book-zh-tw/`](book-zh-tw/)，包含前言与十二章正文、繁體配图和独立的构建脚本；两项已知配图例外见[繁體版说明](book-zh-tw/README.md)。俄语 community edition 由 [@ilkruglov](https://github.com/ilkruglov) 组织并完成翻译与校验，位于 [`book-ru/`](book-ru/)，依据固定版本的中文原稿制作，原作者仍为 [Bojie Li](https://github.com/bojieli)。各译本的译者、来源版本和许可证见对应目录说明；译本数字、公式和引用以原版为准。
