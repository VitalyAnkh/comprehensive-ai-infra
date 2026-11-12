# 案例分析

这里是正文算例背后的展开材料：具体模型、硬件与系统案例的完整推算、来源核对和取舍说明。正文只保留结论与关键推导，脚注链接到对应案例。

下表按引用的章节分组；“引用章节”列出所有链接到该案例的章（全库 74 篇案例已 100% 接入正文第 1～24 章）。

## 第 1 章 · 初识 AI Infra

| 案例 | 引用章节 |
| --- | :--: |
| [主机调度的放置与任务吞吐](host-policy-and-dispatch.md) | 1、11 |
| [从内核瓶颈到训练产出](kernel-and-fleet-efficiency.md) | 1、4、5、10 |
| [LogicFolding：芯片内部的数据移动与能耗](logicfolding-energy.md) | 1、4、14 |
| [可编程网卡：从网络虚拟化到数据访问](programmable-nic.md) | 1、7 |

## 第 2 章 · 模型架构

| 案例 | 引用章节 |
| --- | :--: |
| [从模型结构计算容量、工作量与访问量](model-resource-accounting.md) | 2、7 |
| [理解之外：Omni、语音、图像与视频生成](generative-multimodal-models.md) | 2、3、19 |

## 第 3 章 · 推理与训练负载

| 案例 | 引用章节 |
| --- | :--: |
| [作者材料：上下文与技术判断](author-context-and-design.md) | 3、8、12 |
| [多模态输入的字节、状态与阶段放置](multimodal-stage-placement.md) | 3、12、19、20 |
| [Queqiao 语音传输案例](queqiao.md) | 3、12、19 |
| [检索与生成的协同预算](retrieval-and-generation.md) | 3、13 |
| [训练投入与 Scaling Law 历史比较](scaling-history.md) | 3 |
| [训练计算量的核算](training-compute.md) | 3、10 |
| [请求分布与推理资源配置](workload-and-provisioning.md) | 3、14 |

## 第 4 章 · 加速器架构

| 案例 | 引用章节 |
| --- | :--: |
| [加速器架构与执行比较](accelerator-architecture.md) | 4 |
| [从负载变化理解芯片演进](architecture-evolution.md) | 4 |
| [从算子利用率到执行中的等待](component-utilization-and-overlap.md) | 4 |
| [算子拆分、融合与量化边界](kernel-orchestration-and-quantization.md) | 4、5 |
| [MacBook 与 RTX PRO 6000：从存储组织到本地推理](macbook-rtx-pro6000.md) | 4、5、12、20 |
| [访存并发与带宽估算](memory-bandwidth-and-concurrency.md) | 4 |
| [OpenTallas：从不可变权重到系统可行性的案例与习题](opentallas.md) | 4、23 |

## 第 5 章 · 算子与运行时

| 案例 | 引用章节 |
| --- | :--: |
| [注意力分块：容量、重读与计算安排](attention-tiles-and-io.md) | 5 |
| [片上容量与数据复用](buffer-capacity-and-data-movement.md) | 5 |
| [缓存、图执行与资源重配的取舍](cache-and-reconfiguration.md) | 5、8、11 |
| [从性能反馈到推理、训练优化](execution-feedback.md) | 5 |
| [推理框架近两年关键特性与章节对应](framework-evolution.md) | 5、9 |
| [自动融合：保留什么状态，接受什么数值语义](fusion-legality-and-precision.md) | 5 |
| [图执行的范围与成本](graph-execution-tradeoffs.md) | 5、8 |
| [主机搬运、锁页内存与缓冲复用](host-transfer-and-buffer-lifetime.md) | 5、10 |
| [模型与算子核对笔记](model-operator-examples.md) | 5 |
| [自动优化的成绩与部署收益](optimization-evaluation-and-deployment.md) | 5 |
| [流式交接的顺序与缓冲](stream-order-and-buffer.md) | 5 |
| [RL 阶段切换与权重分发](weight-handoff.md) | 5、10、17 |

## 第 6 章 · 超节点

| 案例 | 引用章节 |
| --- | :--: |
| [集合通信的实际路径与等待来源](collective-paths-and-diagnosis.md) | 6、7、10 |
| [通信调优、计算争用与卸载](communication-tuning.md) | 6、7 |
| [分层内存：容量、等待与迁移收益](memory-criticality-and-tiering.md) | 6、9 |
| [矩阵形状、通信分块与执行条件](mesh-shape-and-slicing.md) | 6 |
| [具体模型的并行推算](model-parallelism.md) | 6、7 |
| [专家执行、容量与服务启动的取舍](moe-and-startup.md) | 6、7、9 |
| [集合通信的物理路径与作业错峰](network-planning-and-collectives.md) | 6、7 |
| [并行方式切换：状态位置与额外驻留](parallel-switching-and-state.md) | 6、9 |
| [PCIe 通信中的中转与 NUMA 放置](pcie-staging-and-numa.md) | 6 |
| [配置预测接近时，应该补测什么](profile-and-plan-ranking.md) | 6、10 |

## 第 7 章 · 数据中心网络

| 案例 | 引用章节 |
| --- | :--: |
| [远端访问：并发窗口之后还要检查什么](remote-ordering-and-completion.md) | 7 |
| [资源共享中的带宽、容量与等待](resource-sharing-and-placement.md) | 7、8、11 |

## 第 8 章 · 推理优化

| 案例 | 引用章节 |
| --- | :--: |
| [分块执行、阶段分离与请求迁移](chunking-and-state-transfer.md) | 8、9 |
| [历史草稿、验证产出与 RL 供给](history-drafts-and-rollout.md) | 8、13、17 |
| [混合模型的前缀状态与恢复成本](hybrid-prefix-state.md) | 2、8 |
| [从业务要求推算推理与训练系统](inference-training-scenarios.md) | 8 |
| [KV 位宽、容量与执行成本](kv-quantization-and-execution.md) | 8 |
| [多 LoRA 服务：共享权重之后还要算什么](multi-lora-serving.md) | 8、9 |
| [推测解码的执行成本与动态预算](speculative-execution.md) | 8 |
| [工具参数生成中的 CPU 工作与状态](structured-generation.md) | 8、11、16、18 |
| [张量压缩、解压与传输](tensor-codec-and-transfer.md) | 8、18 |
| [权重卸载的容量、流量与执行位置](weight-offload-execution.md) | 8、9 |

## 第 9 章 · 分布式推理

| 案例 | 引用章节 |
| --- | :--: |
| [缓存事件与路由判断](cache-events-and-routing.md) | 9 |
| [KV 缓存的取回、共享与路由](cache-tiers-and-routing.md) | 9、18 |
| [专家分派、重排与扩缩容](expert-dispatch-and-resizing.md) | 9、10 |
| [多 LoRA 服务：共享权重之后还要算什么](multi-lora-serving.md) | 8、9 |
| [异构推理的两种切分：A100／H20 与 KTransformers](pd-af-heterogeneous.md) | 9 |

## 第 10 章 · 训练系统

| 案例 | 引用章节 |
| --- | :--: |
| [检查点布局、后台保存与推理加载](checkpoint-layout-and-loading.md) | 10 |
| [MiMo-V2.6 两次 RL 运行的中断与恢复](mimo-v26-rl-interruptions.md) | 10、17 |
| [RL 作业的交错执行与故障恢复](rl-scheduling-and-recovery.md) | 10、11、17 |
| [RL 中的状态、版本与可复现性](rl-state-and-reproducibility.md) | 10、17 |
| [RL 长尾、样本选择与流式训练](rollout-tail-and-sampling.md) | 10、11、17 |
| [训练卸载中的转换、传输与更新位置](training-offload-and-casting.md) | 10 |

## 第 11 章 · 资源调度与运行环境

| 案例 | 引用章节 |
| --- | :--: |
| [DSec：Agent RL 沙箱平台的资源与状态](dsec-sandbox-platform.md) | 11、13、16、17、18 |
| [优化结果与 Agent 执行记录](evaluation-and-agent-records.md) | 11、15、16、18 |
| [调度、模型路由与云端环境：扩写依据](platform-routing.md) | 11 |
| [可抢占 rollout 的权重准备与有效产出](preemptible-rollout-and-weight-readiness.md) | 11、17 |
| [RL 验证的剩余时间与资源配置](reward-deadlines-and-feedback.md) | 11、17 |
| [模型路由：从 token 单价算到任务成本](routing-cost-and-completion.md) | 11 |
| [一名 24×7 数字员工的 serving 成本](single-agent-serving-cost.md) | 11、18、23 |
| [快照恢复、首次访问与环境容量](snapshot-residency-and-first-use.md) | 11、18 |

## 第 12 章 · 端边云协同

| 案例 | 引用章节 |
| --- | :--: |
| [RAW 图片精修的传输与计算](raw-retouching.md) | 12 |

## 第 13 章 · 存储系统与数据工程

| 案例 | 引用章节 |
| --- | :--: |
| [检索与生成的协同预算](retrieval-and-generation.md) | 3、13 |
| [历史草稿、验证产出与 RL 供给](history-drafts-and-rollout.md) | 8、13、17 |
| [DSec：Agent RL 沙箱平台的资源与状态](dsec-sandbox-platform.md) | 11、13、16、17、18 |

## 第 14 章 · 智算中心与电力热学系统

| 案例 | 引用章节 |
| --- | :--: |
| [LogicFolding：芯片内部的数据移动与能耗](logicfolding-energy.md) | 1、4、14 |
| [请求分布与推理资源配置](workload-and-provisioning.md) | 3、14 |

## 第 15 章 · 集群可靠性与基准评测

| 案例 | 引用章节 |
| --- | :--: |
| [从性能反馈到推理、训练优化](execution-feedback.md) | 5、15 |
| [自动优化的成绩与部署收益](optimization-evaluation-and-deployment.md) | 5、15 |
| [集合通信的实际路径与等待来源](collective-paths-and-diagnosis.md) | 6、7、10、15 |
| [配置预测接近时，应该补测什么](profile-and-plan-ranking.md) | 6、10、15 |
| [优化结果与 Agent 执行记录](evaluation-and-agent-records.md) | 11、15、16、18 |

## 第 16 章 · 机密计算与可信安全系统

| 案例 | 引用章节 |
| --- | :--: |
| [DSec：Agent RL 沙箱平台的资源与状态](dsec-sandbox-platform.md) | 11、13、16、17、18 |
| [优化结果与 Agent 执行记录](evaluation-and-agent-records.md) | 11、15、16、18 |
| [工具参数生成中的 CPU 工作与状态](structured-generation.md) | 8、11、16、18 |

## 第 17 章 · 后训练与强化学习系统

| 案例 | 引用章节 |
| --- | :--: |
| [RL 阶段切换与权重分发](weight-handoff.md) | 5、10、17 |
| [历史草稿、验证产出与 RL 供给](history-drafts-and-rollout.md) | 8、13、17 |
| [MiMo-V2.6 两次 RL 运行的中断与恢复](mimo-v26-rl-interruptions.md) | 10、17 |
| [RL 作业的交错执行与故障恢复](rl-scheduling-and-recovery.md) | 10、11、17 |
| [RL 中的状态、版本与可复现性](rl-state-and-reproducibility.md) | 10、17 |
| [RL 长尾、样本选择与流式训练](rollout-tail-and-sampling.md) | 10、11、17 |
| [可抢占 rollout 的权重准备与有效产出](preemptible-rollout-and-weight-readiness.md) | 11、17 |
| [RL 验证的剩余时间与资源配置](reward-deadlines-and-feedback.md) | 11、17 |
| [DSec：Agent RL 沙箱平台的资源与状态](dsec-sandbox-platform.md) | 11、13、16、17、18 |

## 第 18 章 · 自主智能体系统

| 案例 | 引用章节 |
| --- | :--: |
| [KV 缓存的取回、共享与路由](cache-tiers-and-routing.md) | 9、18 |
| [张量压缩、解压与传输](tensor-codec-and-transfer.md) | 8、18 |
| [DSec：Agent RL 沙箱平台的资源与状态](dsec-sandbox-platform.md) | 11、13、16、17、18 |
| [快照恢复、首次访问与环境容量](snapshot-residency-and-first-use.md) | 11、18 |
| [优化结果与 Agent 执行记录](evaluation-and-agent-records.md) | 11、15、16、18 |
| [工具参数生成中的 CPU 工作与状态](structured-generation.md) | 8、11、16、18 |
| [一名 24×7 数字员工的 serving 成本](single-agent-serving-cost.md) | 11、18、23 |

## 第 19 章 · 多模态生成与实时语音

| 案例 | 引用章节 |
| --- | :--: |
| [理解之外：Omni、语音、图像与视频生成](generative-multimodal-models.md) | 2、3、19 |
| [多模态输入的字节、状态与阶段放置](multimodal-stage-placement.md) | 3、12、19、20 |
| [Queqiao 语音传输案例](queqiao.md) | 3、12、19 |

## 第 20 章 · 具身智能与物理仿真

| 案例 | 引用章节 |
| --- | :--: |
| [端侧异构 SoC 零拷贝传感与闭环神经世界模型回归评测](embodied-soc-and-world-model.md) | 20 |
| [多模态输入的字节、状态与阶段放置](multimodal-stage-placement.md) | 3、12、19、20 |
| [MacBook 与 RTX PRO 6000：从存储组织到本地推理](macbook-rtx-pro6000.md) | 4、5、12、20 |

## 第 21 章 · 推荐检索与向量数据库

| 案例 | 引用章节 |
| --- | :--: |
| [检索与生成的协同预算](retrieval-and-generation.md) | 3、13、21 |
| [分层内存：容量、等待与迁移收益](memory-criticality-and-tiering.md) | 6、9、21 |

## 第 22 章 · 科学计算智能系统

| 案例 | 引用章节 |
| --- | :--: |
| [注意力分块：容量、重读与计算安排](attention-tiles-and-io.md) | 5、22 |
| [矩阵形状、通信分块与执行条件](mesh-shape-and-slicing.md) | 6、22 |

## 第 23 章 · 软硬件协同与新型架构

| 案例 | 引用章节 |
| --- | :--: |
| [OpenTallas：从不可变权重到系统可行性的案例与习题](opentallas.md) | 4、23 |
| [一名 24×7 数字员工的 serving 成本](single-agent-serving-cost.md) | 11、18、23 |
| [混合模型的前缀状态与恢复成本](hybrid-prefix-state.md) | 2、8、23 |
| [推测解码的执行成本与动态预算](speculative-execution.md) | 8、23 |
| [加速器架构与执行比较](accelerator-architecture.md) | 4、23 |

## 第 24 章 · 全栈跨层优化与自演进系统

| 案例 | 引用章节 |
| --- | :--: |
| [专家分派、重排与扩缩容](expert-dispatch-and-resizing.md) | 9、10、24 |
| [异构推理的两种切分：A100／H20 与 KTransformers](pd-af-heterogeneous.md) | 9、24 |
| [权重卸载的容量、流量与执行位置](weight-offload-execution.md) | 8、9、24 |
| [自动融合：保留什么状态，接受什么数值语义](fusion-legality-and-precision.md) | 5、24 |
| [自动优化的成绩与部署收益](optimization-evaluation-and-deployment.md) | 5、15、24 |
