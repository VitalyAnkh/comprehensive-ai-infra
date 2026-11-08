# 正文书稿

[前言](00-前言.md)介绍写作缘起、作者的 UB 与数据中心网络经历，以及本书的分析方法。前言同时提供三个阶段的结构图、逐章导读和阅读前置条件。

二十四章正文以模型训练与推理的完整执行过程连接跨层优化：应用通过模型与上下文表达行为，各层利用计算关系、状态寿命与任务依赖重新选择实现。算子融合、执行复用、设备交接、上下文共享、AI 原生存储底座、模型—硬件跨层协同设计、吉瓦级 AI 数据中心电力热学、机密 AI 计算、原生多模态与视频生成（3D DiT）、大规模后训练与强化学习（RLVR）、具身智能（VLA）与大规模物理仿真、十万卡集群可靠性工程与基准评测科学、自主智能体（Agentic AI）长程上下文与沙箱底座、全双工实时语音与原生全模态流式底座、百亿亿次 Token 数据工程与合成数据工厂、以及科学智能（AI for Science）基础设施将这一主线融入具体推导。各章插图沿用全书统一的 Hands-On Large Language Models 风格，以 SVG、PNG、PDF 提供（全书共 835 幅章节配图）。

各章均以“本章小结”收尾，资料说明移出正文，来源保留为脚注。

[下载全书 PDF（推荐）](https://github.com/bojieli/ai-infra-book/releases/latest/download/AI-Infra-Book.pdf) · [本轮修改与验证](../archive/reviews/core-principles-revision-2026-09-10/README.md) · [思想实验复算](../calculations/results/core-principles.json)

| 章 | 正文 |
| --- | --- |
| 1 | [初识 AI Infra](<01-初识 AI Infra.md>) |
| 2 | [模型架构](<02-模型架构.md>) |
| 3 | [推理与训练负载](<03-推理与训练负载.md>) |
| 4 | [加速器架构](<04-加速器架构.md>) |
| 5 | [算子与运行时](<05-算子与运行时.md>) |
| 6 | [超节点](<06-超节点.md>) |
| 7 | [数据中心网络](<07-数据中心网络.md>) |
| 8 | [推理优化](<08-推理优化.md>) |
| 9 | [分布式推理](<09-分布式推理.md>) |
| 10 | [训练系统](<10-训练系统.md>) |
| 11 | [资源调度与运行环境](<11-资源调度与运行环境.md>) |
| 12 | [端边云协同](<12-端边云协同.md>) |
| 13 | [AI 原生存储系统与数据底座](<13-AI 原生存储系统与数据底座.md>) |
| 14 | [模型与系统硬件协同设计](<14-模型与系统硬件协同设计.md>) |
| 15 | [吉瓦级 AI 数据中心、电力热学与跨地域协同训练](<15-吉瓦级 AI 数据中心、电力热学与跨地域协同训练.md>) |
| 16 | [机密 AI 计算、可信执行环境与安全沙箱底座](<16-机密 AI 计算、可信执行环境与安全沙箱底座.md>) |
| 17 | [多模态、视频生成（DiT）与世界模型基础设施](<17-多模态、视频生成（DiT）与世界模型基础设施.md>) |
| 18 | [大规模后训练（RLVR）、异步强化学习与推理时扩展系统](<18-大规模后训练（RLVR）、异步强化学习与推理时扩展系统.md>) |
| 19 | [具身智能（VLA）、实时控制与大规模物理仿真底座](<19-具身智能（VLA）、实时控制与大规模物理仿真底座.md>) |
| 20 | [十万卡集群可靠性工程、静默数据损坏（SDC）防御与基准评测科学](<20-十万卡集群可靠性工程、静默数据损坏（SDC）防御与基准评测科学.md>) |
| 21 | [自主智能体基础设施（Agentic Infra）、Computer Use 沙箱与分层 KV 记忆底座](<21-自主智能体基础设施（Agentic Infra）、Computer Use 沙箱与分层 KV 记忆底座.md>) |
| 22 | [全双工实时语音（Full-Duplex Speech）、神经音频编解码与原生全模态流式底座](<22-全双工实时语音（Full-Duplex Speech）、神经音频编解码与原生全模态流式底座.md>) |
| 23 | [百亿亿次 Token 数据工程、GPU 加速模糊去重与合成数据工厂底座](<23-百亿亿次 Token 数据工程、GPU 加速模糊去重与合成数据工厂底座.md>) |
| 24 | [科学智能（AI for Science）基础设施：生物大分子、百万碱基基因组与物理场网格系统](<24-科学智能（AI for Science）基础设施：生物大分子、百万碱基基因组与物理场网格系统.md>) |

配图由各章 `build.py` 生成，统一样式位于 `figure_style/`，本轮新增图位于 `core_principles_figures.py`。图号与阅读顺序由正文派生，记录在各章 `figure-index.json`。

在线阅读由 Markdown 自动构建，详见[网站构建与发布](../website/README.md)。章节构建脚本的临时 HTML 预览写入 `build/legacy/`，不提交到仓库。

在安装 matplotlib、numpy、Markdown、PyMuPDF 和 Playwright 的 Python 环境中，运行 `python manuscripts/ch01/build.py`（其他章节替换章号）；全书 PDF 运行 `python book/build_pdf.py`，另需 Pandoc 与 XeLaTeX。

这里的条件比较属于教学推导，未新增 GPU 性能实测；原始实验、模型配置和版本来源仍保留在对应目录。


V4／V4.1 贯穿案例已沿十二章展开，新增六幅图，见[修改与验证记录](../archive/reviews/v41-throughline-2026-09-10/README.md)。模型状态、CED 输入处理、缓存恢复和路由选择采用同一组[固定条件与复算](../calculations/results/v41-throughline.json)。

V4／V4.1 再次核对后，补充 CED 数据流图，并区分编码器与解码器的 SWA 恢复；全书共 400 多幅插图。见[数据与概念复核记录](../archive/reviews/v4-v41-audit-2026-09-10/README.md)。


2026-09-11 的[统一切分与调度修订](../research/ub-ep-integration-2026-09-10/README.md)衔接第 1、4、5、6、7、9、10 章，补充并行维度总览、SP／CP、UB、专家分离与 1024 卡训练算例。新增七幅图使用 `ub_ep_figures.py` 和统一样式生成，已接入第 6、7、9 章构建脚本。

2026-09-11 的 [UB 量化修订](../research/ub-openurma-quantification-2026-09-11/README.md)把 Unified Bus 的讨论分到两处：第 6.5.5 节从超节点规模推导互联能容纳的主机数、关系建立时间与硅面积代价，第 7.3–7.4 节沿一次远程访问推导阶段延迟、请求速率与状态增长。七幅新图由 `ub_ep_figures.py` 从 `calculations/results/ub-fabric-book.json` 生成，计算主题 `ub-fabric` 已接入 `calc.py`。
