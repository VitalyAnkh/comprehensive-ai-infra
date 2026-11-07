# 路径 D 全书百科全书级旗舰主计划（20 章终极版 · 第 1–16 章原位深化 + 第 17–20 章新建 + `experiments/ch14..ch20` 全栈闭环）

> **版本与状态**：v1.1（`Status: Approved / Converged` —— 已通过 Gemini 4 Argon High Subagent 两轮严苛独立审计并零阻塞项收敛）  
> **当前仓库基线（路径 B 完成后，提交 `26cf0999`）**：全书 **16 章正文共 17,842 行**（含 `00-前言.md` 146 行共 17,988 行）、**615 幅 420pt 出版级配图**、**222 个题目块（221 个唯一题号）**  
> **路径 D 目标规模**：升级为 **20 章百科全书级终极体系（预计 ~25,780 行正文 / 731 幅 420pt 出版级配图 / 304 道可复算习题与实验 + 补齐 `experiments/ch14`–`experiments/ch20` 独立可运行实验目录）**  
> **核心工程准则**：
> 1. **100% 锚定一手真值与公开顶会/源码证据**：全面消化仓库内尚未合入正文的底层源码审计、硬件规范与锁定案例库（[`root-dspark-execution/NOTES.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/2026-infra-survey/root-dspark-execution/NOTES.md)、[`root-weight-controller/NOTES.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/2026-infra-survey/root-weight-controller/NOTES.md)、[`UB-ASCEND-NOTES.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/references/UB-ASCEND-NOTES.md)、[`ascend-950-official.txt`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/references/text/ascend-950-official.txt)、[`HARDWARE-COVERAGE.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/references/HARDWARE-COVERAGE.md)、[`GAPS.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/references/GAPS.md)、[`case-studies/*.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/README.md)、[`reading-mlsys-2026.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/2026-infra-survey/reading-mlsys-2026.md)、[`reading-osdi-2026.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/2026-infra-survey/reading-osdi-2026.md)、[`reading-nsdi-2026.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/2026-infra-survey/reading-nsdi-2026.md)）及 2025–2026 工业界旗舰系统（DeepSeek NSA / DeepEP / DeepGEMM、SGLang / vLLM、veRL / HybridFlow、UEC 1.0 / Tomahawk 6、Isaac Lab / Jetson Thor、Meta 16K–100K 集群可靠性与 SDC 防御）。
> 2. **非破坏性增量扩充（零断链、零重排既有编号）**：第 1–16 章所有既有章号、H2/H3 节号、既有图号（`1-1`..`16-16`）与既有习题号 100% 保持不变；新增插图严格按 `sync_figure_index` 单调递增规则追加于各章末尾既有图之后。
> 3. **最高规格配图与去 AI 腔润色门禁**：分 **10 个 Batch（每 Batch 2 章）** 推进，每一批次强制执行 **`codex exec` + `gpt-image-2.6-sunburst` (`quality="high"`) 配图生成 + `420pt` 字体/对齐/排版/配色精细打磨 + [`refine-prose`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/.agents/skills/refine-prose/SKILL.md) 去 AI 腔深度润色 + Gemini 4 Argon High Subagent 审阅收敛（含 `.png` 视觉巡检）**。

---

## 一、全书 20 章体系架构与非破坏性基线总表

### 1.1 为什么构建“第 1–16 章深度加固 + 第 17–20 章四大新范式”的 20 章百科全书体系
1. **彻底消除前 16 章与主干章节之间的体量落差并清空遗留研究笔记**：
   - 当前第 13–16 章（`823`、`819`、`725`、`583` 行）虽然框架完整，但与第 4–12 章（平均 `1,300+` 行）相比仍有较大深化空间；
   - 仓库中仍有若干已完成静态源码审计或规格归档但未写进正文的硬核材料，包括 SGLang `c99d906e` 的 **DSpark 紧凑验证窗口与 DP 共同档位**（[`root-dspark-execution/NOTES.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/2026-infra-survey/root-dspark-execution/NOTES.md)）、**SGLang 磁盘权重更新多 Worker 汇总与恢复边界**（[`root-weight-controller/NOTES.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/2026-infra-survey/root-weight-controller/NOTES.md)）、**华为昇腾 910A/910C/950PR/950DT（第三代达芬奇架构）与灵衢 UB 2.0.1 正式规范（545 页）**（[`UB-ASCEND-NOTES.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/references/UB-ASCEND-NOTES.md)、[`ascend-950-official.txt`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/references/text/ascend-950-official.txt)）、**AMD CDNA 4 (MI350X/MI355X) 与 AWS Trainium2 (NeuronCore-v3, `28 MiB` SBUF/core) / Trainium3 (NeuronCore-v4, `32 MiB` SBUF/core)**（[`HARDWARE-COVERAGE.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/references/HARDWARE-COVERAGE.md)），以及 [`case-studies/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/README.md) 与 MLSys/OSDI/NSDI 2026 的最新锁定算例。
2. **补齐 2025–2026 AI 基础设施最关键的四个独立工程支柱（第 17–20 章）**：
   - **第 17 章（多模态、视频生成 DiT 与世界模型基础设施）**：打破全书前 16 章以 1D 自回归文本为主的局限，系统建立 3D 时空扩散 Transformer（$T\times H\times W$）二次方注意力、3D 因果 VAE 显存洪峰、2D 统一序列并行（Ulysses + Ring USP）、异步块流水（DistriFusion / PipeFusion）、特征复用缓存（TeaCache）与 **EPD 三池解耦（TriInfer MLSys '26）扩展至 E-P-D-G 四阶段池化分离（vLLM-Omni）**的完整定量体系；
   - **第 18 章（大规模后训练 RLVR、异步强化学习与推理时扩展系统）**：将第 10 章 10.6.3 的单节 RL 简介扩充为独立旗舰章，拆解 PPO 四模型 vs. GRPO/DAPO 显存账本、**veRL (`3D-HybridEngine`) 零冗余重分片** vs. **RollArt/Weave 物理训推分离**、长思维链（Long-CoT）幂律尾气泡打包（RollPacker / Seer）、陈旧度有界异步部分采样（Staleness-Bounded Partial Rollout）、分布式验证沙箱调度（DistRS）与测试时树搜索（PRM / Speculative ToT）系统；
   - **第 19 章（具身智能 VLA、实时控制与大规模物理仿真底座）**：覆盖物理世界不可回滚的硬实时截止期（VLALagBench 延迟退化模型）、**System 2（1–5 Hz VLM）+ System 1（50–200 Hz 动作扩散头）双频异步架构**、车载/机器人计算平台（NVIDIA Jetson Thor / Drive Thor）MIG 与 RTOS 确定性调度、万级并行 GPU 物理仿真器（Isaac Lab / Genesis / MuJoCo Warp）的刚体求解—光线追踪—策略推理三合一显存/SM 编排，以及车队影子模式数据回流闭环；
   - **第 20 章（十万卡集群可靠性工程、静默数据损坏 SDC 防御与基准评测科学）**：作为全书 20 章的终极大收束，定量剖析十万卡级集群故障分布（Meta Llama 3 16,384 卡 419 次中断、78% 硬件故障分类与热备件排队论）、**静默数据损坏（SDC）物理诱因、3500 万 GPU 小时在线检测（`Safeguarding LLM Training at Scale` OSDI '26）、算子代数不变量（ABFT）与位级对齐异常算子定位（`OpGuard` OSDI '26）**、无重启弹性容错（TorchFT / TrainMover / EROICA / NCCL Flight Recorder 死锁取证）、AI 基准评测防作弊与无偏性能剖析科学（`FlashInfer-Bench` / `CCEval` / `DriftBench` / `Blink` OSDI '26 / `LiveBench`），以及全书二十章总合龙。

### 1.2 全书 20 章当前精确基线与路径 D 扩充目标总表

| 章号 | 章节文件 | 当前行数 | 目标行数 | 当前图数 (末图号与所在小节) | 新增图数 (新图号区间) | 当前习题数 (末题号与格式) | 新增习题数 (新题号区间) | 所属 Batch |
| :-: | --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 01 | [`01-初识 AI Infra.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/01-%E5%88%9D%E8%AF%86%20AI%20Infra.md) | 737 | ~900 | 22 (`图 1-22` @ `1.4.4`) | +2 (`图 1-23`~`1-24`) | 10 (`> **练习 1-10〔延伸〕...**`) | +2 (`练习 1-11`~`1-12`) | Batch 1 |
| 02 | [`02-模型架构.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/02-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84.md) | 1,370 | ~1,540 | 40 (`图 2-40` @ `2.6.3`) | +2 (`图 2-41`~`2-42`) | 14 (`> **练习 2-14〔延伸〕...**`) | +2 (`练习 2-15`~`2-16`) | Batch 1 |
| 03 | [`03-推理与训练负载.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/03-%E6%8E%A8%E7%90%86%E4%B8%8E%E8%AE%AD%E7%BB%83%E8%B4%9F%E8%BD%BD.md) | 974 | ~1,130 | 35 (`图 3-35` @ `3.6.3`) | +2 (`图 3-36`~`3-37`) | 15 (`> **练习 3-15〔核心〕...**`) | +2 (`练习 3-16`~`3-17`) | Batch 2 |
| 04 | [`04-加速器架构.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/04-%E5%8A%A0%E9%80%9F%E5%99%A8%E6%9E%B6%E6%9E%84.md) | 1,524 | ~1,720 | 47 (`图 4-47` @ `4.8.4`) | +2 (`图 4-48`~`4-49`) | 14 (`> **实验 4-14 · 延伸...**`) | +2 (`实验 4-15`~`4-16`) | Batch 2 |
| 05 | [`05-算子与运行时.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/05-%E7%AE%97%E5%AD%90%E4%B8%8E%E8%BF%90%E8%A1%8C%E6%97%B6.md) | 1,423 | ~1,600 | 44 (`图 5-44` @ `5.6.3`) | +2 (`图 5-45`~`5-46`) | 14 块/13 题号 (`> **实验 5-13 · 核心...**`) | +2 (`实验 5-14`~`5-15`) | Batch 3 |
| 06 | [`06-超节点.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/06-%E8%B6%85%E8%8A%82%E7%82%B9.md) | 1,484 | ~1,660 | 61 (`图 6-61` @ `6.7.4`) | +2 (`图 6-62`~`6-63`) | 16 (`> **实验 6-16 · 核心...**`) | +2 (`实验 6-17`~`6-18`) | Batch 3 |
| 07 | [`07-数据中心网络.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/07-%E6%95%B0%E6%8D%AE%E4%B8%AD%E5%BF%83%E7%BD%91%E7%BB%9C.md) | 1,522 | ~1,690 | 65 (`图 7-65` @ `7.6.4`) | +2 (`图 7-66`~`7-67`) | 19 (`> **实验 7-19 · 核心...**`) | +2 (`实验 7-20`~`7-21`) | Batch 4 |
| 08 | [`08-推理优化.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/08-%E6%8E%A8%E7%90%86%E4%BC%98%E5%8C%96.md) | 1,091 | ~1,270 | 40 (`图 8-40` @ `8.6.3`) | +2 (`图 8-41`~`8-42`) | 14 (`> **练习 8-14 · 核心...**`) | +2 (`练习 8-15`~`8-16`) | Batch 4 |
| 09 | [`09-分布式推理.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/09-%E5%88%86%E5%B8%83%E5%BC%8F%E6%8E%A8%E7%90%86.md) | 1,199 | ~1,360 | 50 (`图 9-50` @ `9.7.3`) | +2 (`图 9-51`~`9-52`) | 16 (`> **练习 9-16 · ...**`) | +2 (`练习 9-17`~`9-18`) | Batch 5 |
| 10 | [`10-训练系统.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/10-%E8%AE%AD%E7%BB%83%E7%B3%BB%E7%BB%9F.md) | 1,271 | ~1,450 | 52 (`图 10-52` @ `10.6.4`) | +2 (`图 10-53`~`10-54`) | 16 (`> **习题 10-16 · ...**`) | +2 (`习题 10-17`~`10-18`) | Batch 5 |
| 11 | [`11-资源调度与运行环境.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/11-%E8%B5%84%E6%BA%90%E8%B0%83%E5%BA%A6%E4%B8%8E%E8%BF%90%E8%A1%8C%E7%8E%AF%E5%A2%83.md) | 1,190 | ~1,350 | 53 (`图 11-53` @ `11.5.3`) | +2 (`图 11-54`~`11-55`) | 15 (`> **实验 11-15 · ...**`) | +2 (`实验 11-16`~`11-17`) | Batch 6 |
| 12 | [`12-端边云协同.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/12-%E7%AB%AF%E8%BE%B9%E4%BA%91%E5%8D%8F%E5%90%8C.md) | 1,107 | ~1,260 | 38 (`图 12-38` @ `12.5.4`) | +2 (`图 12-39`~`12-40`) | 15 (`**拓展练习 12-15　...**`) | +2 (`拓展练习 12-16`~`12-17`) | Batch 6 |
| 13 | [`13-AI 原生存储系统与数据底座.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/13-AI%20%E5%8E%9F%E7%94%9F%E5%AD%98%E5%82%A8%E7%B3%BB%E7%BB%9F%E4%B8%8E%E6%95%B0%E6%8D%AE%E5%BA%95%E5%BA%A7.md) | 823 | ~1,150 | 18 (`图 13-18` @ `13.7.2`) | +6 (`图 13-19`~`13-24`) | 12 (`> **实验 13-12：...**`) | +3 (`练习/实验 13-13`~`13-15`) | Batch 7 |
| 14 | [`14-模型与系统硬件协同设计.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/14-%E6%A8%A1%E5%9E%8B%E4%B8%8E%E7%B3%BB%E7%BB%9F%E7%A1%AC%E4%BB%B6%E5%8D%8F%E5%90%8C%E8%AE%BE%E8%AE%A1.md) | 819 | ~1,150 | 18 (`图 14-18` @ `14.8.2`) | +6 (`图 14-19`~`14-24`) | 12 (`> **实验 14-12：...**`) | +3 (`练习/实验 14-13`~`14-15`) | Batch 7 |
| 15 | [`15-吉瓦级 AI 数据中心、电力热学与跨地域协同训练.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/15-%E5%90%89%E7%93%A6%E7%BA%A7%20AI%20%E6%95%B0%E6%8D%AE%E4%B8%AD%E5%BF%83%E3%80%81%E7%94%B5%E5%8A%9B%E7%83%AD%E5%AD%A6%E4%B8%8E%E8%B7%A8%E5%9C%B0%E5%9F%9F%E5%8D%8F%E5%90%8C%E8%AE%AD%E7%BB%83.md) | 725 | ~1,120 | 16 (`图 15-16` @ `15.5.2`) | +6 (`图 15-17`~`15-22`) | 10 (`> **实验 15-10：...**`) | +4 (`练习/实验 15-11`~`15-14`) | Batch 8 |
| 16 | [`16-机密 AI 计算、可信执行环境与安全沙箱底座.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/16-%E6%9C%BA%E5%AF%86%20AI%20%E8%AE%A1%E7%AE%97%E3%80%81%E5%8F%AF%E4%BF%A1%E6%89%A7%E8%A1%8C%E7%8E%AF%E5%A2%83%E4%B8%8E%E5%AE%89%E5%85%A8%E6%B2%99%E7%AE%B1%E5%BA%95%E5%BA%A7.md) | 583 | ~1,080 | 16 (`图 16-16` @ `16.5.2`) | +6 (`图 16-17`~`16-22`) | 10 (`> **实验 16-10：...**`) | +4 (`练习/实验 16-11`~`16-14`) | Batch 8 |
| 17 | [`17-多模态、视频生成（DiT）与世界模型基础设施.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/17-%E5%A4%9A%E6%A8%A1%E6%80%81%E3%80%81%E8%A7%86%E9%A2%91%E7%94%9F%E6%88%90%EF%BC%88DiT%EF%BC%89%E4%B8%8E%E4%B8%96%E7%95%8C%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD.md) | 新建 | ~1,100 | 0 | +18 (`图 17-1`~`17-18`) | 0 | +12 (`练习/实验 17-1`~`17-12`) | Batch 9 |
| 18 | [`18-大规模后训练（RLVR）、异步强化学习与推理时扩展系统.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/18-%E5%A4%A7%E8%A7%84%E6%A8%A1%E5%90%8E%E8%AE%AD%E7%BB%83%EF%BC%88RLVR%EF%BC%89%E3%80%81%E5%BC%82%E6%AD%A5%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E4%B8%8E%E6%8E%A8%E7%90%86%E6%97%B6%E6%89%A9%E5%B1%95%E7%B3%BB%E7%BB%9F.md) | 新建 | ~1,100 | 0 | +18 (`图 18-1`~`18-18`) | 0 | +12 (`练习/实验 18-1`~`18-12`) | Batch 9 |
| 19 | [`19-具身智能（VLA）、实时控制与大规模物理仿真底座.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/19-%E5%85%B7%E8%BA%AB%E6%99%BA%E8%83%BD%EF%BC%88VLA%EF%BC%89%E3%80%81%E5%AE%9E%E6%97%B6%E6%8E%A7%E5%88%B6%E4%B8%8E%E5%A4%A7%E8%A7%84%E6%A8%A1%E7%89%A9%E7%90%86%E4%BB%BF%E7%9C%9F%E5%BA%95%E5%BA%A7.md) | 新建 | ~1,050 | 0 | +16 (`图 19-1`~`19-16`) | 0 | +10 (`练习/实验 19-1`~`19-10`) | Batch 10 |
| 20 | [`20-十万卡集群可靠性工程、静默数据损坏（SDC）防御与基准评测科学.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/20-%E5%8D%81%E4%B8%87%E5%8D%A1%E9%9B%86%E7%BE%A4%E5%8F%AF%E9%9D%A0%E6%80%A7%E5%B7%A5%E7%A8%8B%E3%80%81%E9%9D%99%E9%BB%98%E6%95%B0%E6%8D%AE%E6%8D%9F%E5%9D%8F%EF%BC%88SDC%EF%BC%89%E9%98%B2%E5%BE%A1%E4%B8%8E%E5%9F%BA%E5%87%86%E8%AF%84%E6%B5%8B%E7%A7%91%E5%AD%A6.md) | 新建 | ~1,100 | 0 | +16 (`图 20-1`~`20-16`) | 0 | +10 (`练习/实验 20-1`~`20-10`) | Batch 10 |
| **合计** | **20 章正文（不含前言 146 行）** | **17,842** | **~25,780** | **615** | **+116（总计 731 幅）** | **222 块 (221 题号)** | **+82（总计 304 块 / 303 题号）** | **10 Batches** |

> [!IMPORTANT]
> **非破坏性编号与检验脚本兼容红线（Invariants）**：
> 1. **章内插图严格单调递增追加规则（兼容 [`manuscripts/book_assets.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/book_assets.py) 与 [`scripts/verify_core_principles.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/scripts/verify_core_principles.py)）**：`sync_figure_index` 强制要求正文中所有图片按阅读顺序满足 `number == f'{chapter}-{i}'`（`1..N` 严格连续递增），且 `verify_core_principles.py` 第 19–22 行强制要求 `got == assets`。因此，对于第 1–16 章，**所有新增插图 `图 X-(last+1)..` 必须物理嵌入在该章最后一个既有插图 `图 X-last` 所在的节（如上表“末图号与所在小节”所示）或紧随其后的新增小节中**；前面小节在正文中直接给出公式/表格并通过“如图 X-Y 所示”前向引用章末综合图，绝不在 `图 X-last` 之前插入新图破坏既有图号。
> 2. **习题题号与格式兼容规则**：
>    - 第 5 章因 `5.1` 节正文有一处加粗列表项 `**5-1 ...**`，导致 `verify_core_principles.py` 正则匹配出 14 个块（含两个 `'5-1'`），但其最后一个习题编号实为 `实验 5-13`；因此第 5 章新增习题严格从 **`实验 5-14`** 与 **`实验 5-15`** 开始编号。
>    - 第 1–11、13–20 章的新增习题统一采用 `> **练习/实验/习题 X-Y...**` 引用块格式，第 10 章保持 `> **习题 10-Y...**` 格式，第 12 章保持该章特有的 `**拓展练习 12-Y　...**` 顶格加粗段落格式；且第 1–12 章 100% 保留 [`archive/reviews/core-principles-revision-2026-09-10/before/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/archive/reviews/core-principles-revision-2026-09-10/before) 中的全部历史题号。
> 3. **严禁改动 `manuscripts/ch01..ch12/sources.json` 与 `manuscripts/ch05/teaching_revision.py`**：[`manuscripts/verify_teaching_book.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/verify_teaching_book.py)（用于历史 12 章教学快照审计）会校验 `sources.json` 所列输入文件的 SHA-256 哈希及 `manuscripts/ch05/teaching_revision.py` 原型脚本字面内容，所有新增脚本或数据文件均使用独立的新文件名，绝不触碰上述历史锁定文件。

---

## 二、仓库剩余未合入一手资产、锁定案例库（`case-studies/*.md`）与 2026 顶会真值 100% 吸收清单

| 资产类别 | 仓库一手文件路径 / 权威文献锚点 | 注入章节 | 核心锁定事实、公式与严格适用边界 |
| --- | --- | --- | --- |
| **SGLang DSpark 紧凑验证窗口与 DP 共同档位** | [`research/2026-infra-survey/root-dspark-execution/NOTES.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/2026-infra-survey/root-dspark-execution/NOTES.md)（固定 SGLang `c99d906effa8bd05573995127f0d4a0984c5a96a`，核验 `planner`、`verify`、`ragged_verify.py` L34–218、`dspark_verify_window.py` L25–226 & L305–460）与 [`case-studies/graph-execution-tradeoffs.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/graph-execution-tradeoffs.md) | 第 8 章 `8.3.3` / `8.6.3` | 1. **预算、真实窗口与执行图档位三本账**：`local_verify_tier_num_tokens` 将每请求至少 1 个 anchor 位置计入 floor，再加草稿预算并以完整窗口上限截断；`schedule_layout` 对逻辑总长向上取整至捕获档位 `graph_num_tokens`（超出捕获范围返回 `None`）；`dp_global_verify_tier_num_tokens` 对跨 DP rank 有效需求取最大值（任一 rank 为负 sentinel 则无共同预算限制）。<br>2. **四请求锁定算例**：4 个请求窗口分别为 `7/3/2/2`（含每请求 1 个 anchor，合计 `14` 行，前缀和边界 `0/7/10/12/14`）；设最大窗口 `7`、每请求最少 `1`、CUDA Graph 档位 `8/16/32`，则草稿预算为 $14 - 4 = 10$，本地执行图档位向上取整为 **`16` 行**（最后 `2` 行填占位值）；若另一 DP rank 需求为 `25`，共同档位升至 **`32` 行**；启用默认关闭的 `_budget_aligned_to_graph_tier` 余量填充时，本地 `16` 行档可将草稿预算上限从 `10` 补到 `12`，共同 `32` 行档（受四请求完整窗口上限 `28` 行截断）可将草稿预算上限补到 `24`；混合窗口 `8/1/3`（合计 `12`）在 floor `24` 与 `8/16/32/64` 档位下得 `32` 行。严格声明：紧凑输入不代表所有输出与中间缓冲同步缩短，也不把占位 cache 值 `0` 写成对任意后端的安全证明。 |
| **SGLang 磁盘权重更新多 Worker 汇总与恢复边界** | [`research/2026-infra-survey/root-weight-controller/NOTES.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/2026-infra-survey/root-weight-controller/NOTES.md)（固定 SGLang `c99d906e` `tokenizer_manager.py`）与 [`parallel-framework-gap-triage.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/2026-infra-survey/parallel-framework-gap-triage.md)（vLLM RDT `sharded_rdt_engine.py` / SkyRL） | 第 10 章 `10.6.3` / `10.6.4` | 1. **六阶段控制流边界**：请求进入（等待暂停条件并持 reader lock）$\to$ 暂停（设 `is_pause`，发出暂停消息 $\ne$ 收到全部 scheduler 静止确认）$\to$ 提交更新（未暂停取 writer lock，已暂停用空上下文）$\to$ 汇总结果（预期数量取自 `elastic_worker_count`，初始为 DP size，收齐后取 `all(r.success)`，不可直接等同于全部 TP/EP rank 数）$\to$ 发布版本（仅成功才更新路径与版本，但失败不发布 $\ne$ 已改权重自动回滚）$\to$ 恢复采样（`continue_generation` 不自动检查上次更新结果）。<br>2. **四响应部分失败教学算例**：4 个预期响应为“成功、成功、失败、成功”时，`all(r.success)` 为 `False`，但前两个接收方可能已覆写新权重；汇总等待受最晚响应约束 $\max_i t_i$，而完整恢复还需计入重载/重建（如 SkyRL 剔除异常实例重分片）、KV 缓存清理与首 token 生成开销。 |
| **华为昇腾 910A/910C/950PR/950DT（第三代达芬奇架构）与灵衢 UB 2.0.1（545 页正式规范）** | [`references/UB-ASCEND-NOTES.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/references/UB-ASCEND-NOTES.md)、[`references/text/ascend-950-official.txt`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/references/text/ascend-950-official.txt)（Table 3-1 & Table 4-2）、[`case-studies/accelerator-architecture.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/accelerator-architecture.md) | 第 4 章 `4.7.1` & 第 6 章 `6.4.1`/`6.7.4` | 1. **昇腾芯片代际与第三代达芬奇（DaVinci）架构官方真值**：910A $\to$ 910B（HCCS 单环 $30\ \mathrm{GB/s}\times 7 = 210\ \mathrm{GB/s}$）$\to$ **910C**（双 die 合封，`128 GB` HBM2e @ `3.2 TB/s`）$\to$ **950PR / 950DT（第三代达芬奇架构 / 第三代 DaVinciCore）**：950PR 含 `128 GB` 自研 HiBL 1.0 @ `1.6 TB/s`（`32/28` Cube Core + `64/56` Vector Core，满血/裁切版），FP8/HiF8 `1 PFLOPS`、FP4/MXFP4 `2 PFLOPS`；950DT 含 **`144 GB` HiZQ 2.0 @ `4 TB/s`**（`36/32/28` Cube Core + `72/64/56` Vector Core）；依据官方白皮书 **Table 4-2**，单 AI Core 片上存储严格锁定为 **`L1 Buffer 512 KB`、`L0A Buffer 64 KB`、`L0B Buffer 64 KB`、`L0C Buffer 256 KB`、`Unified Buffer (UB) 512 KB`**，全芯片共享 **`L2 Cache 128 MB`**（`512B` Cache Line 含 `4 x 128B` Sector，低 4 位地址跨 4 Die / 16 Slice 交错）；微架构引入 **Cube-Vector 核内直连通道**、**RegFile 寄存器双发射 SIMD + SIMD/SIMT 融合编程**、**`BufferID`（`get_buf`/`rel_buf`）硬件同步**以及支持分形格式转换与量化预处理的 **NDDMA**。<br>2. **灵衢 UB 2.0.1 物理与事务层真值（Table 3-1）**：单颗 950 提供 **`2,016 GB/s` 双向（`1,008 GB/s` 单向）** UB 带宽，由 `18 x 4` 组 `112 Gbps` PAM4 SerDes 组成；这 18 个 UB 端口与 **`PCIe 5.0 x16`（`128 GB/s` 双向，静态复用 `4` 个 x4 端口）** 及 **`2 x 400G UBoE / RoCE`（`200 GB/s` 双向，静态复用 `2` 个 x4 端口）** 共享 SerDes 通道；支持 Load/Store 内存语义（Home Agent / Caching Agent，粒度 `64B`）与 **CCU（Collective Communication Unit）** 硬件卸载集合通信；UB 2.0.1 正式公开稿共 `545` 页，明确区分协议支持范围与芯片实装子集。 |
| **AMD CDNA 4 (MI350X/MI355X) 与 AWS Trainium2 (NeuronCore-v3) / Trainium3 (NeuronCore-v4)** | [`references/HARDWARE-COVERAGE.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/references/HARDWARE-COVERAGE.md)、[`aws-neuroncore-v3.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/references/files/documents/aws-neuroncore-v3.md)、[`aws-trainium2.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/references/files/specs/aws-trainium2.md)、[`aws-trainium3.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/references/files/specs/aws-trainium3.md)、[`aws-trainium3-nki.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/references/files/documents/aws-trainium3-nki.md) | 第 4 章 `4.6.2` & `4.7.1` | 1. **AMD MI300X $\to$ MI325X $\to$ MI350X/MI355X (CDNA 4)**：`288 GB` HBM3e，峰值带宽 `8 TB/s`，原生支持 **FP4 / FP6 / Block FP16** 微缩放格式，8 XCD + 2 IOD 芯粒拓扑，Infinity Fabric 互连。<br>2. **AWS Trainium1 $\to$ Trainium2 $\to$ Trainium3 代际真值**：Trainium1（NeuronCore-v2）为 `24 MiB` SBUF/core；**Trainium2（8 个 NeuronCore-v3）** 每核含 TensorEngine（$128\times 128$ 脉动阵列）、VectorEngine、ScalarEngine、GPSIMD（8 核 `@ 2.4 GHz`）以及 **`28 MiB` 软件管理片上缓存 SBUF/core（全芯片 `224 MiB`）+ `2 MiB` PSUM/core**，配备 `96 GiB` HBM3 @ `2.9 TB/s` 与 **NeuronLink-v3**（`1,280 GB/s`，Trn2 UltraServer 64 卡 2D Torus）；**Trainium3（8 个 NeuronCore-v4）** 升级为 **`32 MiB` SBUF/core（全芯片 `256 MiB`）+ `2 MiB` PSUM/core**，原生支持 **Quad-MXFP8/MXFP4**（单核 `315 TFLOPS` 稠密 MXFP8，全芯片 `2.52 PFLOPS`），配 `144 GiB` HBM3e 与 **NeuronLink-v4**（`2,560 GB/s`）+ **NeuronSwitch-v1**（Trn3 UltraServer 最高 144 卡 / `20.7 TB` HBM3e）；同时在正文中如实标注 **Trainium3** 官方文档中的两处公开口径差异（`aws-trainium3.md` Overview 页写 `4.9 TB/s` HBM3e 带宽 / `16` CC-Core vs. `aws-trainium3-nki.md` 架构详页写 `4.7 TB/s` / `20` CC-Core）。 |
| **仓库锁定定量案例库（`case-studies/*.md`）与 `arithmetic.json` 验算锚点** | [`hybrid-prefix-state.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/hybrid-prefix-state.md)、[`accelerator-architecture.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/accelerator-architecture.md)、[`kernel-and-fleet-efficiency.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/kernel-and-fleet-efficiency.md)、[`moe-and-startup.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/moe-and-startup.md)、[`graph-execution-tradeoffs.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/graph-execution-tradeoffs.md)、[`workload-and-provisioning.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/workload-and-provisioning.md)、[`structured-generation.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/structured-generation.md)、[`multimodal-stage-placement.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/multimodal-stage-placement.md)、[`preemptible-rollout-and-weight-readiness.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/preemptible-rollout-and-weight-readiness.md)、[`rl-scheduling-and-recovery.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/rl-scheduling-and-recovery.md)、[`rollout-tail-and-sampling.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/rollout-tail-and-sampling.md)、[`reward-deadlines-and-feedback.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/reward-deadlines-and-feedback.md) | 第 1–18 章 | 1. **Kimi K3 混合前缀状态（`hybrid-prefix-state.md`）**：`69` 层 KDA + `24` 层 MLA，单个 KDA 检查点 `428.5546875 MiB`（`TP=8` 时每卡 `53.5693359375 MiB`）vs. MLA KV `1.0986328125 MiB/token`，以及 `10,752 -> 8,192` token 前缀回退边界（**Marconi** MLSys '25 + **SGLang Unified Radix Cache `5aab054e`** + **vLLM Hybrid KV Cache Coordinator `#37898/#45845/#49502`**）。<br>2. **算子与机队效率（`kernel-and-fleet-efficiency.md`）**：**FlashAttention-4** 在 B200 上 $128\times 128$ 瓦片的周期账本（Tensor Core `1,024` 周期 vs. SMEM `768` 周期 vs. SFU `exp2` `1,024` 周期，多项式模拟消减 `75%` SFU 压力）与 Google TPU 生产机队有效吞吐乘积 **MPG**（$SG=83.33\% \times RG=80.00\% \times PG=50.00\% = 33.33\%$）。<br>3. **MoE 偏斜与冷启动（`moe-and-startup.md`）**：*Breaking the Ice*（`64` 专家、`8` 卡、$B=2,048, k=2$：热专家 `640` vs. 均衡 `256` tokens 致 `2.5x` 尾减速与零拷贝逻辑切分）与 *FaaScale*（MLSys '26：`671B` FP8 `671 GB` 在 `100 Gbps` vs. `400 Gbps` RDMA 下的分层流水冷启动）。<br>4. **生产负载与结构化生成（`workload-and-provisioning.md` & `structured-generation.md`）**：**ServeGen（MLSys/NSDI '26）** 重尾突发（$CV^2=4.0$ vs. 泊松 $CV^2=1.0 \implies$ Kingman 公式下排队长度放大 `2.5x`）、**TokenScale（MLSys '26）** $O(W)$ 滑窗词元速度扩缩、**Grammareng（OSDI '26）** 上下文相关词表（`128,256` 词表中 `37%` 多字节词元）与结构化合规率基准（`Outlines 18.6%` vs. `XGrammar 58.3%` vs. `LLGuidance 68.3%`）。<br>5. **多模态与大规模 RL 锁定算例**：`Qwen3-VL-4B`（$640\times 640 \to 400$ 视觉 Token，DeepStack `[400, 10240]` = `7.8125 MiB` EC vs. `56.25 MiB` KV $\implies 7.2\times$ 带宽放大）与 **TriInfer（MLSys '26）** EPD 解耦；**RLBoost（OSDI '26）**（`8x H200` `83.79 美元/h` + `6x 8xH100` `5.32 美元/h` Spot $\implies 1.381\times$ 吞吐 / `1.224\times` 成本效率及 `3.93 s` 权重传输上界）、**RollPacker（OSDI/NSDI '26）**（`1/2 -> 1/4` 长度偏差反例与总词元归一化）、**DistRS（MLSys/NSDI '26）**（$\mathbb{E}[S-e \mid S>e]$ 条件剩余时间与 `216,000` token 抢占阈值）。 |
| **2026 顶会（MLSys / OSDI / NSDI '26）论文精确映射表** | [`reading-mlsys-2026.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/2026-infra-survey/reading-mlsys-2026.md)、[`reading-osdi-2026.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/2026-infra-survey/reading-osdi-2026.md)、[`reading-nsdi-2026.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/2026-infra-survey/reading-nsdi-2026.md) | 贯穿第 1–20 章 | 严格按原论文真实贡献对号入座：**ECHO（OSDI '26 #2）** 用于 NSA 稀疏注意力 KV 卸载与无损预取；**GraCE（MLSys/OSDI '26）** 用于编译器辅助图捕获（`+66%` 编译时间换 `+20%` 内存优化）；**UCCL-Tran（OSDI '26 #61）** / **AutoCCL（NSDI '25）** 用于集合通信传输调优；**Umap（OSDI '26 #76）** 用于分布式文件系统页粒度 `mmap` 矩阵访问免容器 OOM，配合 **MoonBright（OSDI '26 #109）** 设备端 GPU 页表物化与 **kvcached（MLSys '26）** 虚拟内存弹性 KV 共享；**Blowfish（OSDI '26 #50）** 用于解耦内存下高密虚拟机弹性气球回收，配合 **Prism（OSDI '26 #5）** 跨模型 GPU 显存协调；**Safeguarding LLM Training at Scale（OSDI '26 #71）**（3500 万 GPU 小时在线 SDC 感知与验证分离）配合 **OpGuard（OSDI '26 #72）**（消除良性非确定性实现位级对齐算子级异常定位）；**Blink（OSDI '26 #23）**（消除扁平负载 PMU 采样失真的可信性能剖析）配合 **CCEval（NSDI '26 #117）**、**DriftBench（MLSys '26 #121）**、**FlashInfer-Bench（MLSys '26 #25）** 与 **LiveBench** 构建防作弊评测体系。 |

---

## 三、第 1–16 章原位深化蓝图（Batches 1–8）

### 3.1 Batch 1：第 1 章 + 第 2 章原位深化
- **第 1 章：初识 AI Infra（`737` 行 $\to$ `~900` 行，图 `22 -> 24`，习题 `10 -> 12`）**
  - **`1.3.4` 深化（主机策略调度与虚拟智能网卡卸载）**：结合 [`case-studies/host-policy-and-dispatch.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/host-policy-and-dispatch.md) 与 NSDI '26 **ZOC**，剖析当 GPU 算子步长缩至百微秒级时，Linux CFS 调度延迟如何拖慢发射流水线；对比用户态抢占调度（`ghOSt`）与网络协议栈/虚机管理向 SmartNIC 或本机虚拟 SmartNIC（`Wave` / `ZOC`）卸载的 PCIe 同步代价与核数回收账本。
  - **`1.4.4` 深化（AI 基础设施全栈可观测性与性能剖析体系 + 20 章路线图升级）**：引入 **XProf（MLSys '26）** 与 **ProfInfer（MLSys '26）** 的全栈追踪方法论，建立从主机 Python API、运行时调度队列、CUDA Stream / 硬件性能计数器（CUPTI / RoCP）到 RDMA 网卡队列深度的跨层关键路径归因（Critical Path Analysis）方程；同步将全书路线图（`图 1-1` 与 `1.4.4` 双轨贯穿表）升级为 **20 章体系**。
  - **新增配图（追加于 `1.4.4` 末尾，`图 1-22` 之后）**：
    - `图 1-23`（`ch01/ch01-host-dispatch-ghost-wave-zoc.svg`）：主机内核调度旁路与卸载演进——从 Linux CFS 抖动、`ghOSt` 用户态微秒级抢占到 `Wave`/`ZOC` 智能网卡数据面卸载的延迟与 CPU 核回收对比；
    - `图 1-24`（`ch01/ch01-xprof-profinfer-critical-path.svg`）：跨主机调度、GPU 流执行与 RDMA 集合通信的全栈关键路径剖析（XProf / ProfInfer）与空泡归因有向无环图。
  - **新增习题（追加于 `练习 1-10` 之后）**：
    - `> **练习 1-11〔核心〕：主机数据面卸载（Wave / ZOC）的 PCIe 往返延迟税与宿主机 CPU 核回收盈亏平衡计算**`
    - `> **练习 1-12〔延伸〕：基于跨层有向无环图（DAG）的关键路径分析——识别被隐藏通信与暴露主机调度气泡**`

- **第 2 章：模型架构（`1,370` 行 $\to$ `~1,540` 行，图 `40 -> 42`，习题 `14 -> 16`）**
  - **`2.5.3` 深化（原生稀疏注意力 DeepSeek NSA vs. Kimi MoBA 的硬件对齐法则与 `ECHO` OSDI '26 无损预取卸载）**：定量对比推理期事后剪枝（Quest / H2O，仅省 Decode 带宽、不省训练与 Prefill 算力）与**原生可训练稀疏注意力（DeepSeek NSA 的三分支：压缩粗粒度 Token + 选择性重要块 + 局部滑动窗口；以及 Kimi MoBA 的无参数门控块路由）**；推导为何 NSA 必须将选择块粒度 $l'$ 与共享 KV 页对齐至 Tensor Core 矩阵分块（如 $64$ 或 $128$ tokens），并结合 **ECHO（OSDI '26 #2）** 解析如何利用 NSA 压缩分支得分提前预测下一层选中块、实现主机 DRAM 到 HBM 的无损异步 KV 预取。
  - **`2.6.3` 深化（混合线性注意力 Kimi K3 / Gated DeltaNet 的循环状态检查点账本：`Marconi` + SGLang Unified Radix Cache + vLLM Hybrid KV Coordinator）**：完整落地 [`case-studies/hybrid-prefix-state.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/hybrid-prefix-state.md) 与 [`references/framework-history/2026-09-08/hybrid-state/README.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/references/framework-history/2026-09-08/hybrid-state/README.md) 的锁定真值——剖析 Kimi K3（`69` 层 KDA 线性注意力 + `24` 层 MLA）的双状态权衡：单个 KDA 循环状态检查点重达 **`428.5546875 MiB`**（`TP=8` 时每卡 `53.5693359375 MiB`），而 MLA 每 Token KV 仅 `1.0986328125 MiB`；推导为何不能每 Token 保存 KDA 快照，并结合 **Marconi（MLSys '25）**、**SGLang Unified Radix Cache（`5aab054e`）** 与 **vLLM Hybrid KV Cache Coordinator（`#37898`/`#45845`/`#49502`）** 复算从 `10,752` tokens 回退至最近对齐快照 `8,192` tokens 并重演 `2,560` tokens 的精确收支边界。
  - **新增配图（追加于 `2.6.3` 末尾，`图 2-40` 之后）**：
    - `图 2-41`（`ch02/ch02-deepseek-nsa-vs-moba-block-alignment.svg`）：DeepSeek NSA 三分支（压缩/选择/局部）、Kimi MoBA $64$-Token 块硬件对齐与 ECHO（OSDI '26）跨层无损预取流水线；
    - `图 2-42`（`ch02/ch02-hybrid-deltanet-marconi-state-checkpoint.svg`）：Kimi K3（`69` KDA + `24` MLA）混合模型在前缀复用与投机回滚下的双状态账本（`428.55 MiB` KDA 快照 vs. `1.10 MiB/tok` MLA KV）及 Marconi / SGLang 间隔检查点重演折中（`10,752 -> 8,192`）。
  - **新增习题（追加于 `练习 2-14` 之后）**：
    - `> **练习 2-15〔核心〕：DeepSeek NSA 三分支在 $128\mathrm{K}$ 上下文下的 KV 读取字节量、Tensor Core 块大小 $l'$ 对齐与 ECHO 预取带宽核算**`
    - `> **练习 2-16〔延伸〕：基于 Kimi K3 锁定参数（`428.5546875 MiB` KDA 快照 vs. `1.0986328125 MiB/tok` MLA KV），复算 `10,752 -> 8,192` 回退重演代价与最优快照间隔**`

### 3.2 Batch 2：第 3 章 + 第 4 章原位深化
- **第 3 章：推理与训练负载（`974` 行 $\to$ `~1,130` 行，图 `35 -> 37`，习题 `15 -> 17`）**
  - **`3.3.3` 深化（生产级推理负载合成科学 `ServeGen` MLSys/NSDI '26 与 `TokenScale` MLSys '26）**：结合 [`case-studies/workload-and-provisioning.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/workload-and-provisioning.md)，打破传统泊松到达假设（$CV^2 = 1.0$），引入 **ServeGen** 实测的客户端相关性重尾突发特征（如 $CV^2 = 4.0$ 在 Kingman 近似下使平均排队长度放大 **$2.5\times$**）、多模态图文交错前缀分布与推理输出双峰重尾分布，并对比 **TokenScale（MLSys '26）** 基于 $O(W)$ 滑动窗口词元速度的前瞻式弹性扩缩。
  - **`3.6.3` 深化（Computer-Use 与多轮 GUI Agent 负载的步间延迟放大规律 `OSWorld-Human` MLSys '26）**：基于 **OSWorld-Human（MLSys '26）**，定量拆解视觉 GUI 操作 Agent 单次任务的端到端耗时：为何模型推理本身之外，**环境截图捕获、高分辨率视觉编码、DOM/无障碍树解析与过密动作步数（$1.4\times\text{–}2.7\times$ 冗余步）** 占据主导，导出多轮视觉 Agent 的状态增长与动作晶化（Action Crystallization）收益模型。
  - **新增配图（追加于 `3.6.3` 末尾，`图 3-35` 之后）**：
    - `图 3-36`（`ch03/ch03-servegen-bursty-and-bimodal-workload.svg`）：ServeGen 真实生产负载特征（$CV^2=4.0$ 客户端突发到达致排队放大 $2.5\times$、双峰重尾输出分布）与 TokenScale 滑窗词元速度扩缩对比；
    - `图 3-37`（`ch03/ch03-osworld-computer-use-agent-timeline.svg`）：Computer-Use 视觉 GUI Agent（OSWorld-Human）的多轮交互时间线分解与冗余步数压缩收益。
  - **新增习题（追加于 `练习 3-15` 之后）**：
    - `> **练习 3-16〔核心〕：基于 Kingman 排队方程与 ServeGen 重尾突发参数（$CV_a^2 = 4.0$ vs. 泊松 $1.0$），核算给定利用率下的队列膨胀倍数与 KV 安全水位裕量**`
    - `> **练习 3-17〔延伸〕：视觉 Computer-Use Agent 多轮截图历史累积下的 Prefill 平方增长与滑窗摘要截断核算**`

- **第 4 章：加速器架构（`1,524` 行 $\to$ `~1,720` 行，图 `47 -> 49`，习题 `14 -> 16`）**
  - **`4.2.4` 深化（微缩放低精度格式硬件账本：NVFP4 vs. MXFP4 / MXFP6）**：精确对比 OCP 微缩放标准 **MXFP4**（`32` 元素共享 `1` 个 `E8M0` 2 的幂次缩放因子，每元素有效位数 $4 + 8/32 = 4.25\ \mathrm{bits}$）与 NVIDIA Blackwell/Rubin **NVFP4**（`16` 元素共享 `1` 个 `E4M3` 浮点缩放因子 + 张量级 FP32 二级缩放，每元素有效位数 $4 + 8/16 = 4.5\ \mathrm{bits}$）在动态范围、尾数量化信噪比、硬件移位器/乘法器开销与 HBM 带宽节省上的定量差异。
  - **`4.6.2` & `4.7.1` 深化（昇腾 910C/950PR/950DT 第三代达芬奇架构、AMD CDNA 4 MI355X 与 AWS Trainium2/3 全景补齐）**：
    - 基于 [`UB-ASCEND-NOTES.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/references/UB-ASCEND-NOTES.md)、[`ascend-950-official.txt`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/references/text/ascend-950-official.txt)（Table 4-2）与 [`case-studies/accelerator-architecture.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/accelerator-architecture.md)，完整增补 **昇腾 910A $\to$ 910B $\to$ 910C（双 die `128 GB` @ `3.2 TB/s`）$\to$ 950PR/950DT（第三代达芬奇架构 / 第三代 DaVinciCore）** 的微架构跃迁：950PR（`32/28` Cube + `64/56` Vector，`128 GB` HiBL 1.0 @ `1.6 TB/s`）与 950DT（`36/32/28` Cube + `72/64/56` Vector，**`144 GB` HiZQ 2.0 @ `4 TB/s`**，互连带宽 `2 TB/s`）；严格按官方 **Table 4-2** 锁定单 AI Core 片上存储层次（**`L1 Buffer 512 KB`、`L0A 64 KB`、`L0B 64 KB`、`L0C 256 KB`、`UB 512 KB`** 及全芯片 **`128 MB` L2 Cache**，`512B` Cache Line / `4x128B` Sector 低 4 位跨 4 Die / 16 Slice 交错），剖析 **Cube-Vector 核内直连**、**RegFile 双发射 SIMD + SIMD/SIMT 融合编程**、**`BufferID`（`get_buf`/`rel_buf`）硬件同步**与 **NDDMA** 分形转换；
    - 基于 [`HARDWARE-COVERAGE.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/references/HARDWARE-COVERAGE.md) 与 [`aws-trainium2.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/references/files/specs/aws-trainium2.md)/[`aws-trainium3.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/references/files/specs/aws-trainium3.md)，增补 **AMD MI350X/MI355X（CDNA 4，`288 GB` HBM3e @ `8 TB/s`，原生 FP4/FP6）**、**AWS Trainium2（8 个 NeuronCore-v3：每核 `28 MiB` SBUF + `2 MiB` PSUM、全芯片 `224 MiB` SBUF、`96 GiB` HBM3 @ `2.9 TB/s`、NeuronLink-v3 `1,280 GB/s`）** 与 **AWS Trainium3（8 个 NeuronCore-v4：每核 `32 MiB` SBUF + `2 MiB` PSUM、全芯片 `256 MiB` SBUF、Quad-MXFP8/MXFP4 `2.52 PFLOPS`、`144 GiB` HBM3e、NeuronLink-v4 `2,560 GB/s` + NeuronSwitch-v1）**，并如实注明 **Trainium3** 官方文档在 Overview 页（`4.9 TB/s` / `16` CC-Core）与 NKI 架构页（`4.7 TB/s` / `20` CC-Core）上的公开口径差异，以及 $8,192\times 8,192\times 8,192$ GEMM 跨芯片分块复用算例。
  - **新增配图（追加于 `4.8.4` 末尾，`图 4-47` 之后）**：
    - `图 4-48`（`ch04/ch04-nvfp4-vs-mxfp4-microscaling-layout.svg`）：NVFP4（16 元素 `E4M3` 块缩放 + 二级 FP32 缩放，4.5 bit/elem）与 OCP MXFP4（32 元素 `E8M0` 指数缩放，4.25 bit/elem）的内存字节布局与 Tensor Core 解包流水线对比；
    - `图 4-49`（`ch04/ch04-ascend950-davinci-and-trainium-sbuf.svg`）：华为昇腾 950 第三代达芬奇架构（`L1 512KB / L0A 64KB / L0B 64KB / L0C 256KB / UB 512KB / L2 128MB`、`BufferID` 同步与 NDDMA）与 AWS Trainium2/3（NeuronCore-v3 `28 MiB` / v4 `32 MiB` 软件管理 SBUF + `2 MiB` PSUM）的片上数据流对照。
  - **新增习题（追加于 `实验 4-14` 之后）**：
    - `> **实验 4-15 · 核心：核算 70B 与 671B 模型在 NVFP4（16 元素 E4M3 缩放）与 MXFP4（32 元素 E8M0 缩放）下的真实权重体积与带宽节省**`
    - `> **实验 4-16 · 延伸：基于昇腾 950 Table 4-2 存储参数（L1 512 KB, L0C 256 KB, UB 512 KB）与 Trainium2/3 SBUF（28/32 MiB），复算 $8192^3$ GEMM 与 FlashAttention 融合算子的片上分块上限与 HBM 流量消减**`

### 3.3 Batch 3：第 5 章 + 第 6 章原位深化
- **第 5 章：算子与运行时（`1,423` 行 $\to$ `~1,600` 行，图 `44 -> 46`，习题 `14 块/13 题号 -> 16 块/15 题号`）**
  - **`5.3.3` 深化（FlashAttention-3 / FlashAttention-4 周期账本与 DeepGEMM 细粒度 FP8 算子剖析）**：
    - 基于 [`case-studies/kernel-and-fleet-efficiency.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/kernel-and-fleet-efficiency.md) 与 **FlashAttention-4（MLSys '26）**，完整植入 B200 上 $128\times 128$ 注意力瓦片的硬件周期账本——揭示从 Hopper 到 Blackwell/Rubin 的**非对称硬件扩展瓶颈**（单瓦片 Tensor Core GEMM 仅需 `1,024` 周期、SMEM 读写 `768` 周期，而 SFU `exp2` 同样高达 `1,024` 周期成为并列瓶颈）；详解如何通过异步 TMEM 双缓冲、Warp-Group 乒乓流水与多项式软件模拟 `exp2`（消减 **`75%` SFU 压力**）重新平衡流水线；
    - 解析 **DeepSeek DeepGEMM** 如何针对细粒度 FP8 量化（激活 $1\times 128$ 块缩放、权重 $128\times 128$ 块缩放）利用 Tensor Core MMA 内部高精度累加 + CUDA Core 二级 FP32 缩放累加（Two-Level Accumulation），并通过运行时 JIT 编译将输入形状 `(M, N, K)` 作为编译期常量注入，省去寄存器索引计算并释放寄存器压力；同步介绍 **ThunderKittens 2.0 / TileLang** 的瓦片级原语抽象。
  - **`5.5.2` 深化（编译器辅助图捕获 `GraCE` MLSys/OSDI '26）**：基于 [`case-studies/graph-execution-tradeoffs.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/graph-execution-tradeoffs.md)，剖析动态控制流与 CPU–GPU 同步点如何打断传统 CUDA Graph 捕获，以及 **GraCE** 如何通过静态控制流重构与设备端条件分支将碎片化小图合并为端到端整图（并量化其 **`+66%` 编译时间换取运行时加速与 `+20%` 峰值显存开销** 的工程折中）。
  - **新增配图（追加于 `5.6.3` 末尾，`图 5-44` 之后）**：
    - `图 5-45`（`ch05/ch05-flashattention4-blackwell-sfu-pipeline.svg`）：FlashAttention-4 在 B200 上 $128\times 128$ 瓦片的周期分解（Tensor Core `1,024` vs. SMEM `768` vs. SFU `1,024` 周期）及多项式模拟削减 `75%` SFU 压力的 TMEM 双缓冲流水线；
    - `图 5-46`（`ch05/ch05-deepgemm-fp8-jit-and-grace-graph.svg`）：DeepGEMM 细粒度 FP8（$1\times 128$ / $128\times 128$）两级累加寄存器数据流与 GraCE 编译器辅助全图捕获（编译时间 $+66\%$ / 显存 $+20\%$ 折中）对比。
  - **新增习题（追加于 `实验 5-13` 之后，编号为 `5-14`、`5-15`）**：
    - `> **实验 5-14 · 核心：基于 FlashAttention-4 锁定周期账本（TC 1,024 / SMEM 768 / SFU 1,024 周期），核算多项式模拟分流 75% SFU 指令前后的瓦片关键路径周期与吞吐增益**`
    - `> **实验 5-15 · 延伸：推导 DeepGEMM 在 $K=7168$、块大小 $128$ 下二级 CUDA Core FP32 累加占总指令周期的比例与 JIT 形状特化收益**`

- **第 6 章：超节点（`1,484` 行 $\to$ `~1,660` 行，图 `61 -> 63`，习题 `16 -> 18`）**
  - **`6.4.1` 深化（灵衢 UB 2.0.1 正式规范 545 页深剖：SerDes 静态复用、内存语义与 CCU 集合卸载）**：基于 [`references/UB-ASCEND-NOTES.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/references/UB-ASCEND-NOTES.md) 与 [`ascend-950-official.txt`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/references/text/ascend-950-official.txt)（Table 3-1），系统解析 UB 2.0.1 的物理层与事务层：单芯片 `18 x 4` 路 `112G` PAM4 SerDes 如何合成 `2,016 GB/s` 双向（`1,008 GB/s` 单向）带宽，以及这 18 个 UB 端口如何与 **`PCIe 5.0 x16`（`128 GB/s` 双向，静态复用 `4` 个端口）** 和 **`2 x 400G UBoE / RoCE`（`200 GB/s` 双向，静态复用 `2` 个端口）** 共享 SerDes 通道；对比基于 Home Agent（HA）/ Caching Agent（CA）的 `64B` 细粒度 Load/Store 内存语义、NDDMA 异步块搬移与交换机/网卡内嵌 **CCU（Collective Communication Unit）** 的适用载荷尺寸分界。
  - **`6.6.3` 深化（超节点 SPMD 编译器自动切分：`PartIR` 与 `Shardy`）**：阐述在双芯粒（TPU7x）与超节点多级互连（D2D $\to$ NVLink/UB/ICI $\to$ DCN）下，编译器如何通过轴向网格标注（Mesh Axis Sharding）与代价模型自动搜索张量切分与集合通信重物化策略。
  - **新增配图（追加于 `6.7.4` 末尾，`图 6-61` 之后）**：
    - `图 6-62`（`ch06/ch06-ub201-serdes-mux-and-ccu-architecture.svg`）：灵衢 UB 2.0.1 协议栈全貌——`18 x 4` `112G` SerDes 在 UB Link、`PCIe 5.0 x16`（4 端口）与 `2x400G UBoE/RoCE`（2 端口）间的静态复用、HA/CA `64B` 内存语义与 CCU 集合通信卸载；
    - `图 6-63`（`ch06/ch06-shardy-partir-hierarchical-mesh-sharding.svg`）：面向“双芯粒 D2D $\to$ 机柜超节点 $\to$ 跨柜网络”三级带宽阶梯的 Shardy/PartIR 自动张量切分与重物化搜索树。
  - **新增习题（追加于 `实验 6-16` 之后）**：
    - `> **实验 6-17 · 核心：基于 UB 2.0.1 Table 3-1 规范，计算同时启用 PCIe 5.0 x16（复用 4 端口）与 2x400G UBoE（复用 2 端口）后剩余的 12 个 UB 端口双向带宽及 All-to-All 耗时变化**`
    - `> **实验 6-18 · 延伸：在三级分层互连（D2D 6x ICI / 1D ICI / DCN）下为 MoE 层寻找最小化通信暴露时间的 Shardy 网格切分方案**`

### 3.4 Batch 4：第 7 章 + 第 8 章原位深化
- **第 7 章：数据中心网络（`1,522` 行 $\to$ `~1,690` 行，图 `65 -> 67`，习题 `19 -> 21`）**
  - **`7.4.3` 深化（超以太网联盟 UEC 1.0 传输层协议栈、乱序包喷洒与 MoE 热点逻辑切分）**：对比传统 RoCEv2（依赖 PFC 无损流控 + ECMP 哈希极易引发大流碰撞与队头阻塞）与 **UEC 1.0（Ultra Ethernet Consortium）** 现代传输层：多路径逐包喷洒（Packet Spraying）打散哈希极化、接收端原生乱序重组（Out-of-Order Delivery）、基于 RTT 与显式标记的现代拥塞控制（NSCC/SMART），以及丢包时的选择性重传（Selective Retransmission）；同时结合 [`case-studies/moe-and-startup.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/moe-and-startup.md)（`64` 专家、`8` 卡、$B=2,048, k=2$ 下热专家 `640` vs. 均衡 `256` tokens 导致 $2.5\times$ 尾减速），解析零拷贝逻辑专家切分如何消除 All-to-All 长尾拥塞。
  - **`7.5.3` 深化（Broadcom Tomahawk 6 `102.4 Tbps` 单芯片与两层扁平 Clos 万卡组网 + `UCCL-Tran` OSDI '26 / `AutoCCL` NSDI '25）**：定量推导交换芯片容量翻倍对网络拓扑层数的决定性改变——基于 `102.4 Tbps`（如 `128 x 800G` 或 `256 x 400G` 高 Radix 端口）交换芯片，构建无阻塞 **两层（Leaf–Spine）扁平 Clos 网络**即可直接容纳 **`16,384` 张 `800G` GPU 或 `32,768` 张 `400G` GPU**，相比三层 Fat-Tree 砍掉一整层核心交换机与 $1/3$ 光模块成本及转发跳数；同时结合 [`case-studies/communication-tuning.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/communication-tuning.md) 对比 **UCCL-Tran（OSDI '26 #61）**、**UCCL（EuroSys '25 / NSDI '26）** 与 **AutoCCL（NSDI '25）** 的软件定义集合通信传输调优。
  - **新增配图（追加于 `7.6.4` 末尾，`图 7-65` 之后）**：
    - `图 7-66`（`ch07/ch07-rocev2-pfc-vs-uec10-packet-spraying.svg`）：RoCEv2（ECMP 哈希碰撞 + PFC 暂停扩散 + Go-Back-N）与 UEC 1.0（逐包喷洒 + 接收端乱序缓存 + 选择性重传）及 UCCL-Tran 多路径调度的微突发响应对比；
    - `图 7-67`（`ch07/ch07-tomahawk6-102t-two-tier-clos-scaling.svg`）：基于 `102.4 Tbps` 交换芯片的两层扁平 Clos（直连 `16,384` 卡 `@ 800G` / `32,768` 卡 `@ 400G`）与传统三层 Fat-Tree 的光模块数量、跳数及建设成本对比。
  - **新增习题（追加于 `实验 7-19` 之后）**：
    - `> **实验 7-20 · 核心：基于 `102.4 Tbps` 交换芯片（Radix $k=128$ @ 800G 或 $k=256$ @ 400G）推导两层 Clos 最大无阻塞 GPU 规模与省去的光模块数**`
    - `> **实验 7-21 · 延伸：核算 UEC 1.0 逐包喷洒在单链路微突发丢包 $0.1\%$ 时，选择性重传相比 RoCEv2 Go-Back-N 节省的重传字节与尾延迟**`

- **第 8 章：推理优化（`1,091` 行 $\to$ `~1,270` 行，图 `40 -> 42`，习题 `14 -> 16`）**
  - **`8.3.3` 深化（合入 [`root-dspark-execution/NOTES.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/2026-infra-survey/root-dspark-execution/NOTES.md) 全部锁定真值 + Ollama `v0.31` MLX MTP 动态回滚边界）**：
    - 完整落地 SGLang `c99d906e` 的 **DSpark 预算、紧凑窗口与执行图档位三本账**：详解 `local_verify_tier_num_tokens`、`schedule_layout`、`dp_global_verify_tier_num_tokens`、`_budget_aligned_to_graph_tier` 与 `run_compact` / `BuildRaggedVerifyWindow`（`dspark_verify_window.py` 前缀和索引与占位行填充）的源码级证据；
    - 复算四请求窗口 `7/3/2/2`（合计 `14` 行，前缀和边界 `0/7/10/12/14`）在档位 `8/16/32` 下的草稿预算（$14-4=10$）、本地执行图档位（`16` 行，末尾 `2` 行为占位值）、跨 DP rank 需求 `25` 时的共同档位（`32` 行），以及开启余量填充后草稿预算上限从 `10` 升至 `12`（本地 `16` 行档）或 `24`（共同 `32` 行档受 `28` 行完整窗口截断）的完整定量过程；同时对比 **Ollama `v0.31`（commit `83ed7d9965b1ee07e0f0b29fd46e47c31f0fcab8`，`x/mlxrunner/mtp.go` / `speculate.go` / `speculate_depth.go`）** 在混合循环注意力下的 Selective State Snapshot 状态克隆与拒绝回滚代价（[`parallel-mlx-speculation-body/NOTES.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/2026-infra-survey/parallel-mlx-speculation-body/NOTES.md)）。
  - **`8.4.4` 深化（结构化约束解码与快速跳跃生成：`XGrammar` / `Grammareng` OSDI '26 / `LLGuidance`）**：基于 [`case-studies/structured-generation.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/structured-generation.md)，解析上下文无关文法（CFG）下推自动机如何通过词表自适应拆分（针对 `128,256` 词表中占 **`37%` 的多字节/上下文相关词元** 结合 **Grammareng OSDI '26** 预编译位掩码与运行时检查）将每步 CPU 掩码生成压至微秒级，对比真实基准合规率（`Outlines 18.6%` vs. `XGrammar 58.3%` vs. `LLGuidance 68.3%`），并推导 **跳跃解码（Jump-Forward / Fast-Forward Tokens）** 将 JSON 固定键名与括号直接拼入 Prefill 一次验证的收益。
  - **新增配图（追加于 `8.6.3` 末尾，`图 8-40` 之后）**：
    - `图 8-41`（`ch08/ch08-dspark-ragged-verify-and-dp-tier.svg`）：SGLang DSpark 四请求窗口（`7/3/2/2` 合计 `14` 行）向 `16` 行/`32` 行共享 CUDA Graph 档位映射、前缀和索引（`0/7/10/12/14`）、末尾占位行与 `_budget_aligned_to_graph_tier` 余量回填机制；
    - `图 8-42`（`ch08/ch08-xgrammar-pda-bitmask-and-jump-forward.svg`）：XGrammar / Grammareng（OSDI '26）下推自动机位掩码预编译（处理 `37%` 多字节词元）与 JSON 确定性片段跳跃解码（Jump-Forward Tokens）步数消减原理。
  - **新增习题（追加于 `练习 8-14` 之后）**：
    - `> **练习 8-15 · 核心：基于 SGLang DSpark 规则，复算四请求窗口 `7/3/2/2` 与混合窗口 `8/1/3`（floor 24）在 `8/16/32/64` 档位下的执行图行数与草稿余量上限**`
    - `> **练习 8-16 · 延伸：核算 JSON 结构化输出中跳跃解码（Jump-Forward Tokens）消减 Decode 步数与附加 Prefill 验证词元的净延迟收益**`

### 3.5 Batch 5：第 9 章 + 第 10 章原位深化
- **第 9 章：分布式推理（`1,199` 行 $\to$ `~1,360` 行，图 `50 -> 52`，习题 `16 -> 18`）**
  - **`9.3.3` 深化（DeepSeek `DeepEP` 非对称域间转发与 0-SM BGDA 通信算子 + `SwiftEP` NSDI '26）**：深入剖析 **DeepEP** 在大规模 Prefill 与 Decode 阶段的两套定制内核：（1）Prefill 高吞吐内核利用“先经 RDMA 跨节点发送至目标节点对等 GPU、再经节点内 NVLink 转发至目标专家卡”的非对称两级层次路由，将跨节点 RDMA 流量收敛并与计算流水重叠；（2）Decode 低延迟内核通过 **IBGDA（GPUDirect Async）+ 硬件 Hook 后台完成检测**，在不占用任何 GPU SM 计算线程（**0-SM 通信监控**）的情况下实现微秒级 All-to-All Dispatch/Combine 与计算重叠；同步对比 **SwiftEP（NSDI '26）** 的动态流量整形。
  - **`9.4.3` 深化（Mooncake 分布式 KV 传输引擎与预测性早期拒绝 FAST '25 Best Paper）**：解析 Mooncake 以 KVCache 为中心的解耦架构——Conductor 全局调度器如何联合感知前缀缓存分布与 PD 实例负载，并在预填充队列过载但未来解码 KV 槽位将溢出的场景下执行**预测性早期拒绝（Predictive Early Rejection）**，避免浪费已完成 Prefill 却因无处存放 Decode KV 而被迫丢弃的沉没算力。
  - **新增配图（追加于 `9.7.3` 末尾，`图 9-50` 之后）**：
    - `图 9-51`（`ch09/ch09-deepep-asymmetric-forwarding-and-bgda.svg`）：DeepSeek DeepEP 架构拆解——左：Prefill 阶段 RDMA $\to$ NVLink 非对称两级层次转发；右：Decode 阶段 IBGDA 硬件直发与 0-SM Hook 后台通信重叠；
    - `图 9-52`（`ch09/ch09-mooncake-kv-centric-and-early-rejection.svg`）：Mooncake 以 KV 缓存为中心的全局池化传输引擎与防止 Prefill 算力沉没的预测性早期拒绝水位模型。
  - **新增习题（追加于 `练习 9-16` 之后）**：
    - `> **练习 9-17 · 核心：推导 DeepEP 在 8 卡 NVLink 节点间采用“RDMA + NVLink 两级转发”相比全网状直接 RDMA 的队列对（QP）缩减比与带宽增益**`
    - `> **练习 9-18 · 延伸：核算 Mooncake 在 Decode 显存池满载时触发预测性早期拒绝所挽回的无效 Prefill 算力与有效 Goodput 提升**`

- **第 10 章：训练系统（`1,271` 行 $\to$ `~1,450` 行，图 `52 -> 54`，习题 `16 -> 18`）**
  - **`10.5.3` 深化（跨并行拓扑自动重分片检查点 `ByteCheckpoint` NSDI '25 与无落盘状态复制 `Checkmate` NSDI '26）**：剖析当万卡集群因节点故障将训练规模从 $N$ 缩容或改变 $(TP, PP, DP)$ 配比时，传统分片检查点为何因形状绑定而被迫全量离线合并转换；解析 **ByteCheckpoint** 基于张量切分元数据图的在线异步流式重分片，以及 **Checkmate（NSDI '26）** 利用网络梯度与内存副本在秒级内恢复故障节点状态、跳过远端存储落盘读取的定量条件。
  - **`10.6.3` 深化（合入 [`root-weight-controller/NOTES.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/2026-infra-survey/root-weight-controller/NOTES.md) 全部锁定真值）**：
    - 完整落地 SGLang `c99d906e`（`tokenizer_manager.py`）与 vLLM RDT（`sharded_rdt_engine.py`）的**权重交接六阶段责任边界表**（请求进入持 reader lock $\to$ 设 `is_pause` 异步发暂停消息 $\to$ writer lock 提交更新 $\to$ 按 `elastic_worker_count` 收集并计算 `all(r.success)` $\to$ 成功后发布版本 $\to$ 显式调用 `continue_generation` 恢复）；
    - 定量推演**四响应部分失败（成功、成功、失败、成功）**的教学算例：说明为何 `all(r.success) == False` 只能阻止管理层发布新版本号，却不能推断前两个接收方自动回滚了已覆写的设备显存，以及调用方（如 SkyRL）如何通过剔除失败实例、重载一致性快照并清理陈旧 KV 缓存完成恢复。
  - **新增配图（追加于 `10.6.4` 末尾，`图 10-52` 之后）**：
    - `图 10-53`（`ch10/ch10-sglang-weight-update-six-stage-boundary.svg`）：SGLang / vLLM RDT 权重更新六阶段控制流时间线与“成功、成功、失败、成功”四 Worker 部分失败下的状态不一致边界及恢复协议；
    - `图 10-54`（`ch10/ch10-bytecheckpoint-resharding-and-checkmate.svg`）：ByteCheckpoint 跨并行维度在线张量重分片与 Checkmate（NSDI '26）基于网络内存副本的无落盘秒级故障恢复对比。
  - **新增习题（追加于 `习题 10-16` 之后）**：
    - `> **习题 10-17 · 核心：基于 SGLang `tokenizer_manager.py` 六阶段状态机，分析四接收方“3 成功 1 失败”时的等待延迟上限与防止混合新旧权重采样的恢复步骤**`
    - `> **习题 10-18 · 延伸：对比训练拓扑从 `TP=8, PP=8, DP=16` 缩容至 `TP=8, PP=8, DP=14` 时，传统离线合并检查点与 ByteCheckpoint 流式按需重分片的 I/O 字节数**`

### 3.6 Batch 6：第 11 章 + 第 12 章原位深化
- **第 11 章：资源调度与运行环境（`1,190` 行 $\to$ `~1,350` 行，图 `53 -> 55`，习题 `15 -> 17`）**
  - **`11.2.3` 深化（vLLM 实例冷启动延迟解剖 `Breaking the Ice` MLSys '26、Serverless 临时流水线扩容 `FaaScale` MLSys '26 / `HydraServe` NSDI '26 与生产机队 MPG）**：结合 [`case-studies/moe-and-startup.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/moe-and-startup.md) 与 [`case-studies/kernel-and-fleet-efficiency.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/kernel-and-fleet-efficiency.md)，定量拆解大模型容器从拉起进程到首 Token 的冷启动瀑布；复算 **FaaScale（MLSys '26 #59）**（及 **HydraServe NSDI '26 #23**）对 `671B` FP8（`671 GB`）模型在 `100 Gbps` vs. `400 Gbps` RDMA 下利用空闲碎片 GPU 构建临时逐层流水线并行（Temporary PP）实现秒级冷启动的账本，并植入 Google TPU 生产机队有效吞吐乘积 **MPG**（$SG=83.33\% \times RG=80.00\% \times PG=50.00\% = 33.33\%$）。
  - **`11.3.4` 深化（多任务异步 RL 混部调度 `Weave` OSDI '26 与抢占式 Spot 算力收割 `RLBoost` OSDI '26）**：结合 [`case-studies/preemptible-rollout-and-weight-readiness.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/preemptible-rollout-and-weight-readiness.md)，完整复算 **RLBoost（OSDI '26）** 锁定算例——在 `8x H200` 按需训练实例（`83.79 美元/h`）基础上动态收割 `6` 组 `8x H100` 抢占式 Spot 实例（每卡 `0.665 美元/h`、每组 `5.32 美元/h`）承担异步 Rollout，将端到端吞吐提升 **$1.381\times$**、成本效率提升 **$1.224\times$**（权重同步上界 `3.93 s`），以及 **Weave（OSDI '26）** 在多并发 RL 作业间的阶段交错编排。
  - **新增配图（追加于 `11.5.3` 末尾，`图 11-53` 之后）**：
    - `图 11-54`（`ch11/ch11-vllm-cold-start-waterfall-and-faascale.svg`）：vLLM 冷启动五阶段延迟瀑布（`Breaking the Ice`）、FaaScale（MLSys '26）`671 GB` 模型跨 `100G/400G` RDMA 临时流水线秒级扩容与机队 MPG（$33.33\%$）损耗分解；
    - `图 11-55`（`ch11/ch11-weave-rlboost-spot-gpu-harvesting.svg`）：Weave（OSDI '26）多作业 RL 阶段交错调度与 RLBoost（OSDI '26，`8xH200` 按需 + `6x8xH100` Spot 达成 $1.381\times$ 吞吐与 $1.224\times$ 成本效率）算力收割时间线。
  - **新增习题（追加于 `实验 11-15` 之后）**：
    - `> **实验 11-16 · 核心：基于 FaaScale 模型，核算 `671 GB` 权重在 `100 Gbps` 与 `400 Gbps` RDMA 下的逐层流式加载与临时流水线首 Token 启动时间**`
    - `> **实验 11-17 · 延伸：基于 RLBoost 锁定参数（`8xH200` 83.79 美元/h + `6x8xH100` Spot 5.32 美元/h，吞吐 $1.381\times$），复算每小时总账单与单位样本成本改善率（$1.224\times$）**`

- **第 12 章：端边云协同（`1,107` 行 $\to$ `~1,260` 行，图 `38 -> 40`，习题 `15 -> 17`）**
  - **`12.2.3` 深化（端侧 NPU 图编译器 `ExecuTorch` / `llm.npu` / `FlexSP` 与动态形状降级陷阱）**：深入剖析手机/PC 端侧 NPU（如 Apple Neural Engine、Qualcomm Hexagon NPU、MediaTek APU）为何偏爱固定形状静态张量图与 INT4/INT8 块量化，而在遇到动态长文本 Prefill 或未对齐算子时被迫回退至 CPU/GPU 引发跨内存域拷贝惩罚；详解 **llm.npu / FlexSP（ASPLOS/MLSys '26）** 的分块静态图拼接与 CPU–GPU–NPU 异构协同投机解码。
  - **`12.3.3` 深化（持续生成下的端侧热节流相变与多轮端云协同隐私脱敏路由）**：建立被动散热设备（笔记本/手机）在连续长输出下的外壳温度积分与功率墙骤降（如从 $35\ \mathrm{W}$ 跌至 $15\ \mathrm{W}$）分段模型，导出端侧算力随时间衰减时的动态云端卸载阈值。
  - **新增配图（追加于 `12.5.4` 末尾，`图 12-38` 之后）**：
    - `图 12-39`（`ch12/ch12-executorch-npu-static-chunk-vs-fallback.svg`）：端侧 NPU 静态分块图执行（ExecuTorch / llm.npu）vs. 动态形状触发 NPU $\to$ CPU 回退拷贝的延迟与功耗对比；
    - `图 12-40`（`ch12/ch12-passive-thermal-throttling-offload-shift.svg`）：无风扇终端持续推理下的温度爬升、热节流带宽骤降与端云协同卸载边界的动态漂移。
  - **新增习题（追加于 `拓展练习 12-15` 之后，保持第 12 章顶格加粗格式）**：
    - `**拓展练习 12-16　核算端侧 NPU 在静态块大小 $C=128$ 下处理任意长度提示词的尾块填充浪费率，以及回退 CPU 执行的盈亏交点。**`
    - `**拓展练习 12-17　推导被动散热终端在功率从 $30\ \mathrm{W}$ 节流至 $12\ \mathrm{W}$ 前后的最大可持续生成长度与端云切换时刻。**`

### 3.7 Batch 7：第 13 章 + 第 14 章深度扩充（双章齐升至 `~1,150` 行、`24` 幅图、`15` 道题）
- **第 13 章：AI 原生存储系统与数据底座（`823` 行 $\to$ `~1,150` 行，图 `18 -> 24`，习题 `12 -> 15`）**
  - **新增 `13.8` 节：十亿级小文件元数据瓶颈（`FalconFS` NSDI '26）、CXL/SSD 解耦向量检索（`DistVS` / `Helmsman`）与 CPU 协处理 GPU I/O（`CoPilotIO` / `Umap` / `MoonBright` OSDI '26）**：
    - **`13.8.1` 多模态海量小文件元数据风暴与服务端路径解析（`FalconFS` NSDI '26）**：剖析传统分布式文件系统中逐级目录 `lookup` RPC 造成的网络往返放大，以及 **FalconFS** 如何通过服务端复合路径解析、目录布局混合哈希与并发无锁更新将小文件元数据吞吐提升数倍；
    - **`13.8.2` 超大规模解耦式向量数据库（`DistVS` NSDI '26 & `Helmsman` OSDI '26）**：当十亿级高维向量索引超越单机 DRAM 时，如何跨主机内存、CXL 池化内存与 NVMe SSD 协同调度图遍历（Beam Search）与精确重排 I/O；
    - **`13.8.3` 当 GPUDirect Storage（GDS）反而不如 CPU 中转：`CoPilotIO`、`Umap` 与 `MoonBright`（OSDI '26）及 `kvcached`（MLSys '26）**：揭示在细粒度随机读、需实时解压/反序列化或高并发小 I/O 场景下，纯 GPU 直读因缺乏灵活预取与主机页缓存聚合而退化，阐述 **CoPilotIO（OSDI '26）** 如何利用空闲主机 CPU 核心作为 GPU I/O 协处理器，**Umap（OSDI '26 #76）** 如何重构分布式文件系统页粒度 `mmap` 矩阵访问以消除冗余拷贝与容器内存 OOM，并结合 **MoonBright（OSDI '26 #109）**（设备端 GPU 页表物化）与 **kvcached（MLSys '26）**（虚拟内存弹性 KV 缓存共享）建立统一虚拟存储视图。
  - **新增 6 幅 420pt 配图（追加于 `13.8` 各小节，`图 13-18` 之后）**：
    - `图 13-19`（`ch13/ch13-falconfs-server-side-path-resolution.svg`）：传统 POSIX 逐跳目录解析 vs. FalconFS 服务端单次复合路径解析与混合布局；
    - `图 13-20`（`ch13/ch13-distvs-helmsman-disaggregated-vector-search.svg`）：DistVS / Helmsman 跨 DRAM–CXL–SSD 的分层图索引遍历与异步预取流水线；
    - `图 13-21`（`ch13/ch13-copilotio-gds-vs-cpu-assisted-io.svg`）：大块顺序 GDS 直通 vs. 细粒度压缩数据下 CoPilotIO 主机 CPU 协处理预取的带宽交叉点；
    - `图 13-22`（`ch13/ch13-umap-moonbright-mmap-and-gpu-page-table.svg`）：Umap（OSDI '26）分布式文件系统页粒度 `mmap` 矩阵按需加载免容器 OOM 与 MoonBright 设备端 GPU 页表物化协同架构；
    - `图 13-23`（`ch13/ch13-multimodal-tar-webdataset-vs-lance-random.svg`）：WebDataset 顺序 Tar 分片与 Lance 列式随机点查在课程学习动态采样下的磁盘寻道对比；
    - `图 13-24`（`ch13/ch13-ai-storage-bandwidth-latency-iops-cube.svg`）：AI 存储底座全场景（预训练流、检查点、RAG 向量、多级 KV、沙箱镜像）的带宽—延迟—IOPS 三维需求立方体。
  - **新增习题与实验（`13-13` ~ `13-15`）**：
    - `> **练习 13-13：核算深层目录树（深度 $D=6$）下百万次小文件打开在传统逐级 RPC 与 FalconFS 服务端复合解析下的网络往返总耗时。**`
    - `> **练习 13-14：推导高维向量检索中将粗量化导航图驻留 DRAM、原始向量放 NVMe SSD 时的单查询 IOPS 需求与 SSD 队列深度上限。**`
    - `> **实验 13-15：运行 `experiments/ch13/13-01` 并扩展对比不同批量下主机预取缓冲与直接设备拷贝的吞吐交叉点。**`

- **第 14 章：模型与系统硬件协同设计（`819` 行 $\to$ `~1,150` 行，图 `18 -> 24`，习题 `12 -> 15`）**
  - **新增 `14.9` 节：架构模拟器、执行轨迹图（Chakra）与模型—通信协同跳层（`NeuSight` / `Phantora` / `FarSkip-Collective`）**：
    - **`14.9.1` 为什么不能靠拍脑袋选架构：算子瓦片级性能预测（`NeuSight` ASPLOS '25）与执行轨迹模拟器（MLCommons `Chakra` / `Phantora` NSDI '26 / `Charon` MLSys '26）**：阐述在流片或采购万卡前，如何用耦合计算—通信依赖图的执行轨迹（Execution Trace Graph）结合瓦片级硬件模型预测端到端 MFU 与网络拥塞，并量化模拟器相对真实硅片的误差边界；
    - **`14.9.2` 打破层间刚性同步：模型结构与集合通信的协同跳层与解耦（`FarSkip-Collective` MLSys '26）**：展示如何通过模型层间跨步残差连接设计，使第 $l$ 层的 AllReduce / All-to-All 通信不再阻塞第 $l+1$ 层输入，而延后至第 $l+2$ 层才消费，从而获得整整一层计算时间来完全掩盖跨节点慢速通信。
  - **新增 6 幅 420pt 配图（追加于 `14.9` 各小节，`图 14-18` 之后）**：
    - `图 14-19`（`ch14/ch14-chakra-execution-trace-graph-simulation.svg`）：MLCommons Chakra 计算—通信依赖有向无环图（ETG）与 Phantora/Charon 集群级时间线回放架构；
    - `图 14-20`（`ch14/ch14-neusight-tile-roofline-calibration.svg`）：理想峰值 Roofline、经验拟合与 NeuSight 瓦片级微架构建模在预测未见芯片性能时的误差对比；
    - `图 14-21`（`ch14/ch14-farskip-collective-delayed-dependency.svg`）：标准 Transformer 逐层阻塞通信 vs. FarSkip-Collective 跨层跳步连接赋予整层计算窗口掩盖 All-to-All 通信；
    - `图 14-22`（`ch14/ch14-moe-granularity-vs-network-radix-co-design.svg`）：MoE 专家细粒度切分（$G$ 增大）对模型困惑度收益与 All-to-All 小包通信税的帕累托最优拐点；
    - `图 14-23`（`ch14/ch14-kv-head-ratio-vs-tensor-parallel-limit.svg`）：GQA 查询头与 KV 头数量对最大张量并行度（$TP \le n_{kv}$ vs. KV 头复制）的硬约束相图；
    - `图 14-24`（`ch14/ch14-codesign-pre-silicon-to-production-loop.svg`）：从流片前 Chakra/NeuSight 仿真、算法—算子协同探索到生产集群预注册验证（实验 `14-01`/`14-02`）的完整工程飞轮。
  - **新增习题与实验（`14-13` ~ `14-15`，关联新建 `experiments/ch14/`）**：
    - `> **练习 14-13：推导 FarSkip-Collective 将集合通信依赖延后 $m=1$ 层时，完全掩盖跨节点 All-to-All 所需的最小层内计算耗时条件。**`
    - `> **练习 14-14：当模型从 $n_{kv}=8$ 缩减为 $n_{kv}=2$ 且采用 $TP=8$ 部署时，核算 KV 头复制带来的额外显存与 AllReduce 流量。**`
    - `> **实验 14-15：运行 `experiments/ch14/14-01` 与 `14-02`，复算生命周期过度训练（Chinchilla 偏移）最优参数规模与四因子逐级降本瀑布。**`

### 3.8 Batch 8：第 15 章 + 第 16 章深度扩充（双章齐升至 `~1,100` 行、`22` 幅图、`14` 道题）
- **第 15 章：吉瓦级 AI 数据中心、电力热学与跨地域协同训练（`725` 行 $\to$ `~1,120` 行，图 `16 -> 22`，习题 `10 -> 14`）**
  - **新增 `15.6` 节：两相液冷与硅片微流体散热、园区储能（BESS）电网调频套利、动静态能耗联合优化（`Kareus` OSDI '26）与跨洲骨干网训练（`Di-PS` NSDI '26）**：
    - **`15.6.1` 当单芯片迈向 $2\text{–}3\ \mathrm{kW}$：单相冷板极限、两相沸腾液冷与芯片内微通道直冷（Direct-to-Silicon Microfluidics）**：对比单相水冷（显热 $\dot{m} c_p \Delta T$ 受限于流速压降）与两相相变冷却（潜热 $h_{fg}$）及硅背面蚀刻微流道消除界面热阻（TIM）的极限热流密度（$>150\ \mathrm{W/cm^2}$）；
    - **`15.6.2` 园区百兆瓦级电池储能系统（BESS）的电网调频与峰谷电价套利账本**：量化 BESS 除削平毫秒级训练冲击波外，如何在分时电价（峰谷价差 $3\times$）与电网一次调频辅助服务中摊薄园区供电 CAPEX；
    - **`15.6.3` 动静态功耗联合优化（`Kareus` OSDI '26）与跨地域骨干网并行参数服务（`Di-PS` NSDI '26）**：剖析在低负载期漏电静态功耗（Static/Leakage Power）占比飙升时，为何单纯 DVFS 降频不如动态合并实例并使空闲卡进入深度休眠（**Kareus**）；并结合 **Di-PS（NSDI '26）** 推导跨洲数据中心间利用差分压缩与异步流水掩盖百毫秒级广域网 RTT 的吞吐边界。
  - **新增 6 幅 420pt 配图（追加于 `15.6` 各小节，`图 15-16` 之后）**：
    - `图 15-17`（`ch15/ch15-single-phase-vs-two-phase-microfluidics.svg`）：单相冷板、两相相变浸没/冷板与硅背面微通道直冷的热阻网络（$R_{\mathrm{TIM}}, R_{\mathrm{conv}}$）与千瓦级芯片结温边界；
    - `图 15-18`（`ch15/ch15-campus-bess-peak-shaving-and-arbitrage.svg`）：百兆瓦级园区储能（BESS）在 $50\ \mathrm{ms}$ 训练脉冲平滑、日内峰谷电价套利与电网掉电备电中的三重能量流；
    - `图 15-19`（`ch15/ch15-kareus-dynamic-static-energy-consolidation.svg`）：Kareus 动静态能耗分解——低负载下全卡浅度降频的高静态漏电税 vs. 负载紧缩合并 + 空闲卡深度休眠；
    - `图 15-20`（`ch15/ch15-dips-wan-backbone-parameter-sync.svg`）：Di-PS（NSDI '26）跨地域训练的广域网链路段带宽调度与层间异步梯度传输重叠；
    - `图 15-21`（`ch15/ch15-embodied-carbon-vs-operational-carbon-lifecycle.svg`）：AI 数据中心 3–5 年生命周期内隐含碳（Embodied Carbon：晶圆制造与基建）与运行碳（Operational Carbon：发电结构）的权衡曲线；
    - `图 15-22`（`ch15/ch15-gigawatt-campus-master-engineering-ledger.svg`）：吉瓦级 AI 园区从电网并网、储能平滑、微通道液冷到跨城光网协同的完整工程总账本。
  - **新增习题与实验（`15-11` ~ `15-14`，关联新建 `experiments/ch15/`）**：
    - `> **练习 15-11：对比硅背面微通道直冷（消除 $R_{\mathrm{TIM}} = 0.02\ \mathrm{^\circ C/W}$）与传统冷板在 $2,000\ \mathrm{W}$ 芯片功耗下的最大允许进液温度提升。**`
    - `> **练习 15-12：给定单卡静态漏电功耗 $P_{\mathrm{static}} = 180\ \mathrm{W}$、满载动态功耗 $P_{\mathrm{dyn}} = 520\ \mathrm{W}$，计算集群负载率为 $40\%$ 时 Kareus 合并休眠相比均匀分摊的节电率。**`
    - `> **练习 15-13：核算 $100\ \mathrm{MWh}$ 园区储能系统在峰谷电价差 $0.08\ \text{美元}/\mathrm{kWh}$、循环效率 $88\%$ 下的年套利收益与投资回收期。**`
    - `> **实验 15-14：运行 `experiments/ch15/15-01` 与 `15-02`，复现 TAPAS 热感知路由 30 请求回本曲线、LogicFolding 功率密度悖论及 DiLoCo 广域网带宽账本。**`

- **第 16 章：机密 AI 计算、可信执行环境与安全沙箱底座（`583` 行 $\to$ `~1,080` 行，图 `16 -> 22`，习题 `10 -> 14`）**
  - **新增 `16.6` 节：MoE 通信流量侧信道与恒定速率填充防御、可验证推理（TEE 证明 vs. zkML vs. 乐观抽检 OPML）与微虚拟机内存弹性重映射（`JANUS` / `Blowfish` / `Prism` OSDI '26）**：
    - **`16.6.1` 就算荷载全加密也会泄密：MoE 专家路由与投机解码步长的流量分析侧信道（Traffic Analysis Side-Channel）**：揭示即便启用了 NVLink/PCIe 硬件加密（TEE-I/O），攻击者仍可通过观察跨节点 All-to-All 发往各 EP rank 的数据包计数或 Token 吐出时间间隔反推领域类别与敏感提示词；推导**恒定速率哑元填充（Constant-Rate Dummy Token Padding）**在隐私保护度 $(\epsilon, \delta)$ 与额外带宽开销之间的定量折中；
    - **`16.6.2` 如何证明云端 API 没有偷偷把 70B 换成 8B 量化模型：可验证推理（Verifiable AI Inference）三路线对比**：建立（1）**硬件 TEE 远程证明签名**（延迟开销 $<5\%$，依赖芯片厂商硬件信任根）、（2）**零知识机器学习证明（zkML）**（密码学无条件可验证，但证明生成算力膨胀 $10^3\text{–}10^5\times$，仅适用小模型或局部审计）、（3）**乐观抽样复算与欺诈证明（Optimistic Sampling / OPML）** 的博弈论纳什均衡方程（给定抽检率 $p$、单次造假节省成本 $\Delta c$ 与罚没押金 $S_{\mathrm{slash}}$，导出诚实执行条件 $p \cdot S_{\mathrm{slash}} > (1-p)\Delta c$）；
    - **`16.6.3` 嵌套虚拟化低开销隔离（`JANUS` OSDI '26）与解耦内存下高密沙箱弹性气球回收（`Blowfish` / `Prism` OSDI '26）**：补充公有云虚拟机内运行不可信 AI 容器/Agent 时的硬件辅助嵌套虚拟化旁路（**JANUS**），以及基于解耦主机/CXL 内存池的高密微虚拟机跨实例弹性气球回收（**Blowfish OSDI '26 #50**）与跨模型 GPU 显存协调（**Prism OSDI '26 #5**）。
  - **新增 6 幅 420pt 配图（追加于 `16.6` 各小节，`图 16-16` 之后）**：
    - `图 16-17`（`ch16/ch16-moe-routing-traffic-side-channel-and-padding.svg`）：密文状态下 MoE All-to-All 专家负载偏斜泄露提示词特征 vs. 恒定桶大小哑元填充（Dummy Padding）防御；
    - `图 16-18`（`ch16/ch16-verifiable-inference-tee-zkml-opml-triangle.svg`）：可验证推理三路线（硬件 TEE 证明、密码学 zkML、乐观抽检欺诈证明 OPML）在证明开销、信任假设与模型规模上的对比；
    - `图 16-19`（`ch16/ch16-optimistic-verification-slashing-game.svg`）：乐观抽检可验证推理（OPML）的博弈树——抽检概率 $p$、造假收益 $\Delta c$ 与质押罚没金 $S_{\mathrm{slash}}$ 的诚实边界曲线；
    - `图 16-20`（`ch16/ch16-janus-nested-virtualization-fast-path.svg`）：传统二级页表嵌套虚拟化（$N_{\mathrm{guest}} \times N_{\mathrm{host}}$ 查表放大）vs. JANUS（OSDI '26）扁平化安全沙箱隔离；
    - `图 16-21`（`ch16/ch16-blowfish-elastic-sandbox-memory-pool.svg`）：Blowfish（OSDI '26 #50）在解耦内存池上的高密微虚拟机弹性气球回收与 Prism（OSDI '26 #5）跨模型显存协调机制；
    - `图 16-22`（`ch16/ch16-trusted-ai-infrastructure-defense-in-depth.svg`）：可信 AI 基础设施纵深防御全景（硬件 TEE-I/O + 流量填充 + 远程证明/抽检 + Firecracker 微虚机 + 幂等审计账本）。
  - **新增习题与实验（`16-11` ~ `16-14`，关联新建 `experiments/ch16/`）**：
    - `> **练习 16-11：在 16 卡 EP 部署中，若为防御 MoE 流量侧信道将每卡接收槽填充至热专家上界 $\mu(1+\alpha)$，计算由此增加的 All-to-All 带宽税。**`
    - `> **练习 16-12：推导乐观可验证推理（OPML）中，当模型偷换每次可省 $0.005$ 美元、平台质押金为 $1,000$ 美元时，确保服务商诚实执行的最小随机抽检率 $p_{\min}$。**`
    - `> **练习 16-13：核算 `virtio-pmem` + DAX 在单节点并发运行 `800` 个 microVM、共享 `64 GB` 只读基础镜像时节省的宿主机物理内存与 Guest `struct page` 开销。**`
    - `> **实验 16-14：运行 `experiments/ch16/16-01` 与 `16-02`，复算 DSec 38 万沙箱内存账本、PipeLLM 三缓冲加密流水线与 `I19` 审计日志中断排空水位。**`

---

## 四、新建第 17–20 章四大旗舰专题详细蓝图（Batches 9–10）

### 4.1 Batch 9 新建第 17 章：《多模态、视频生成（DiT）与世界模型基础设施》（`manuscripts/17-多模态、视频生成（DiT）与世界模型基础设施.md`，`~1,100` 行，`18` 幅配图，`12` 道习题）

- **`## 17.1 从一维词元序列到三维时空潜空间：多模态与世界模型的计算特征剧变`**
  - `### 17.1.1` **视觉/音频编码器与大语言模型拼装的算术强度断层**：结合 [`case-studies/multimodal-stage-placement.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/multimodal-stage-placement.md)，定量对比动态高分辨率切片（AnyRes / NaViT）与 `Qwen3-VL-4B` 锁定算例（单张 $640\times 640$ 图像经 $2\times 2$ 合并生成 `400` 个视觉 Token，DeepStack 编码特征张量 `[400, 10240]` 仅占 **`7.8125 MiB`**，而进入 36 层主干预填充后膨胀为 **`56.25 MiB` KV 缓存**，产生 **$7.2\times$ 跨机状态体积放大**）；
  - `### 17.1.2` **3D 扩散 Transformer（DiT）的时空 Token 爆炸与二次方注意力墙**：以生成一段 $10\ \mathrm{s}$、$24\ \mathrm{fps}$、$1080\mathrm{p}$（$1920\times 1080$）视频为例，经 3D 因果 VAE（时间压缩 $4\times$、空间压缩 $8\times 8$）和 $2\times 2$ Patchify 后，潜空间 Token 数高达 $N_{\mathrm{vid}} = \frac{240}{4} \times \frac{1080}{16} \times \frac{1920}{16} = 60 \times 68 \times 120 = 489,600$！单步全时空自注意力的复杂度 $O(N_{\mathrm{vid}}^2 d)$ 比普通文本 Prefill 高出两个数量级，且需迭代去噪 $K=20\text{–}50$ 步；
  - `### 17.1.3` **被忽视的显存杀手：3D 因果 VAE 解码器的特征图洪峰与时空分块（Spatial-Temporal Tiling）**：证明在去噪完成后，将潜变量还原为像素空间的 3D 卷积上采样中间激活往往突破单卡 $80\text{–}192\ \mathrm{GB}$ 显存，推导带重叠 Halo 边界的时空分块（Tiled VAE）与流式时间缓存的显存—重算账本。
- **`## 17.2 面向百万级时空序列的并行架构：从 DeepSpeed-Ulysses、Ring Attention 到 2D USP 与 db-SP`**
  - `### 17.2.1` **为何传统张量并行（TP）在长序列 DiT 中失去扩展性**：TP 每层对完整序列激活执行 AllReduce/ReduceScatter（通信量 $\propto N_{\mathrm{vid}}$ 且无法被小头维度分摊）；
  - `### 17.2.2` **DeepSpeed-Ulysses（头切分 All-to-All）与 Ring Attention（序列切分 P2P 环）的通信量与并行度硬上限**：
    - 精确对比：设序列长 $N$、隐藏维 $d$、并行度 $P$，Ulysses 在 QKV 投影后与注意力输出前各做一次 All-to-All，**每层单卡总通信量为 $\frac{4 N d}{P}$**（随 $1/P$ 线性下降，但受限于 $P \le n_{\mathrm{head}}$，在 GQA 头数较少时触顶）；Ring Attention 每卡持有长度 $N/P$ 的序列分片（本地 $K,V$ 块大小为 $\frac{2 N d}{P}$），沿环路传递 $P-1$ 步，**单步单卡通信量为 $\frac{2 N d}{P}$、全环 $P-1$ 步累计通信量为 $\frac{2(P-1) N d}{P} \approx 2 N d$**（累计通信量不随 $P$ 减小，但可通过 $P-1$ 步分块计算与 P2P 传输流水重叠，且彻底摆脱注意力头数 $n_{\mathrm{head}}$ 的上限约束）；
  - `### 17.2.3` **2D 统一序列并行（USP：$P_{\mathrm{Ulysses}} \times P_{\mathrm{Ring}}$）与双均衡序列并行（`db-SP` MLSys '26）**：将节点内高带宽 NVLink/UB 分配给 Ulysses（$P_U = 8$），跨节点 RoCE/IB 分配给 Ring Attention（$P_R$），并结合 **db-SP（MLSys '26）** 解决图文/视频掩码稀疏性导致的计算不均衡。
- **`## 17.3 利用扩散时间步冗余：异步位移块流水（DistriFusion / PipeFusion）与特征缓存（TeaCache）`**
  - `### 17.3.1` **相邻扩散步的激活缓慢演化特性与异步块流水（DistriFusion & PipeFusion）**：利用第 $t$ 步与第 $t-1$ 步特征高度相似（$\|x_t - x_{t-1}\| \ll \|x_t\|$）的性质，使用上一步陈旧的远端 Patch 激活立即启动本步注意力计算，同时在后台异步传输最新 Patch 激活，彻底消除同步通信停顿；
  - `### 17.3.2` **免训练时间步特征缓存（TeaCache / AdaCache / FastVideo）与低精度注意力（SageAttention-3 FP4）**：根据时间步嵌入的一阶多项式差分动态判断当前步是否跳过 DiT 主干重算、直接复用上一步残差，实现 $1.5\times\text{–}2.5\times$ 无损加速；
  - `### 17.3.3` **实时交互式世界模型（StreamDiffusionV2 / CausVid / GameNGen）的因果蒸馏与 30 FPS 帧预算**：将 50 步双向扩散蒸馏为 1–4 步因果自回归流式扩散，利用滚动 KV 缓存（Rolling KV Cache）与步间流水满足 $<33.3\ \mathrm{ms}$ 的实时游戏/仿真交互帧率。
- **`## 17.4 原生多模态服务的四阶段解耦：从 EPD（`TriInfer` MLSys '26）到 E-P-D-G（`vLLM-Omni`）池化架构`**
  - `### 17.4.1` **从混合 EPD 解耦（`TriInfer` MLSys '26 #127）走向 E-P-D-G 四池分离（`vLLM-Omni` / `ModServe`）**：结合 [`case-studies/multimodal-stage-placement.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/multimodal-stage-placement.md) 推导何时将视觉编码（**E**ncode）独立成池、何时与预填充（**P**refill）融合（取决于注意力与 FFN 代价阈值 $c_{\mathrm{attn}} / c_{\mathrm{ffn}}$），以及在原生全模态输出场景下进一步将自回归文本解码（**D**ecode）与视觉/语音/视频扩散生成（**G**enerate）解耦为四阶段异构资源池；
  - `### 17.4.2` **跨模态特征张量（ViT Embeddings）的高速传输、量化压缩与多级缓存复用**：基于 `Qwen3-VL-4B` 的 `7.8125 MiB` EC vs. `56.25 MiB` KV（$7.2\times$ 差距），核算多轮视觉问答中缓存图像编码后 Embedding 的显存压缩比与 RDMA 传输耗时。
- **`## 常见误区`、`## 练习与实验`（`17-1` ~ `17-12`）、`## 本章小结`**

#### 第 17 章 18 幅 420pt 配图清单（`图 17-1` ~ `图 17-18`）
- `图 17-1`（`ch17/ch17-1d-llm-vs-3d-spatiotemporal-dit-tokens.svg`）：一维文本 Token 序列与 3D 视频扩散模型（$T\times H\times W$ 潜空间分块，$489,600$ Tokens）的规模与算术强度对比；
- `图 17-2`（`ch17/ch17-anyres-navit-dynamic-image-tiling.svg`）：AnyRes 动态高分辨率切片、NaViT 序列打包与 `Qwen3-VL-4B` DeepStack 特征（`7.8125 MiB` EC $\to$ `56.25 MiB` KV，$7.2\times$ 放大）演化；
- `图 17-3`（`ch17/ch17-3d-causal-vae-memory-spike-and-tiling.svg`）：3D 因果 VAE 解码阶段的特征图显存洪峰（突破 $80\ \mathrm{GB}$）与带重叠 Halo 的时空分块（Spatial-Temporal Tiling）解码；
- `图 17-4`（`ch17/ch17-tp-vs-ulysses-vs-ring-attention-comm.svg`）：张量并行（TP）、DeepSpeed-Ulysses（每层 $\frac{4Nd}{P}$，$P \le n_{\mathrm{head}}$）与 Ring Attention（单步 $\frac{2Nd}{P}$、全环 $\frac{2(P-1)Nd}{P} \approx 2Nd$，无头数限制）对比；
- `图 17-5`（`ch17/ch17-2d-usp-hybrid-ulysses-ring-topology.svg`）：2D 统一序列并行（USP）：节点内 NVLink 跑 Ulysses All-to-All + 跨节点 RoCE 跑 Ring Attention P2P 重叠流水；
- `图 17-6`（`ch17/ch17-dbsp-dual-balanced-sequence-parallelism.svg`）：db-SP（MLSys '26）面向变长多模态与因果掩码的双均衡序列切分；
- `图 17-7`（`ch17/ch17-distrifusion-displaced-patch-parallelism.svg`）：DistriFusion 异步位移 Patch 并行——利用第 $t-1$ 步陈旧激活掩盖第 $t$ 步跨卡通信；
- `图 17-8`（`ch17/ch17-pipefusion-step-patch-2d-pipeline.svg`）：PipeFusion 将空间分块与去噪时间步交错流水化的气泡消除时序图；
- `图 17-9`（`ch17/ch17-teacache-timestep-residual-reuse.svg`）：TeaCache / AdaCache 基于时间步特征差分的自适应跳步复用与质量—延迟帕累托曲线；
- `图 17-10`（`ch17/ch17-sageattention3-fp4-quantized-attention.svg`）：SageAttention 平滑均值抑制离群值与 FP8/FP4 块量化注意力的硬件加速路径；
- `图 17-11`（`ch17/ch17-interactive-world-model-rolling-kv-30fps.svg`）：交互式世界模型（StreamDiffusionV2 / CausVid）的因果少步蒸馏与 $<33.3\ \mathrm{ms}$ 滚动 KV 帧生成流水线；
- `图 17-12`（`ch17/ch17-epdg-four-stage-disaggregated-serving.svg`）：从 TriInfer（MLSys '26）混合 EPD 解耦到 vLLM-Omni 原生全模态 E-P-D-G 四阶段异构池化架构；
- `图 17-13`（`ch17/ch17-vit-embedding-cache-vs-llm-kv-cache.svg`）：多轮多模态对话中缓存原始图片、缓存 ViT 视觉特征（`7.8125 MiB`）与缓存 LLM 全层 KV（`56.25 MiB`）的体积与重算延迟折中；
- `图 17-14`（`ch17/ch17-audio-video-text-interleaved-batching.svg`）：图文音交错流（Omni-modal）中模态专属编码器微批聚合与主干对齐调度；
- `图 17-15`（`ch17/ch17-multimodal-dataloader-gpu-decode-nvjpeg.svg`）：多模态预训练中 CPU 解码饥饿瓶颈与 DALI / nvJPEG / NVDEC 硬件视频解码卸载流水线；
- `图 17-16`（`ch17/ch17-dit-cfg-classifier-free-guidance-parallel.svg`）：分类器自由引导（CFG 双分支）与条件/无条件批量合并及尾部蒸馏对比；
- `图 17-17`（`ch17/ch17-video-dit-end-to-end-latency-breakdown.svg`）：$1080\mathrm{p}$ 视频生成从文本编码、50 步 DiT 去噪到 3D VAE 解码的端到端延迟与显存水位全貌；
- `图 17-18`（`ch17/ch17-multimodal-world-model-infra-synthesis.svg`）：多模态与世界模型基础设施的算法—并行—缓存—解耦四维总架构图。

---

### 4.2 Batch 9 新建第 18 章：《大规模后训练（RLVR）、异步强化学习与推理时扩展系统》（`manuscripts/18-大规模后训练（RLVR）、异步强化学习与推理时扩展系统.md`，`~1,100` 行，`18` 幅配图，`12` 道习题）

- **`## 18.1 从对齐微调到可验证奖励强化学习（RLVR）：计算图、显存账本与数据流突变`**
  - `### 18.1.1` **PPO 四模型（Actor、Critic、Reference、Reward）vs. GRPO / DAPO 无 Critic 组相对优势的显存与算力账本**：定量对比单步训练中 Actor 自回归生成（Rollout，访存带宽受限）、Reference/Reward 前向打分（Prefill 计算密集）与 Actor/Critic 反向传播（计算与优化器状态密集）的三重异构特征；证明 GRPO 每提示词采样 $G=8\text{–}64$ 条轨迹虽消除了同尺寸 Critic 模型（节省近半训练态显存），却将 Rollout 生成与 KV 占用放大了 $G$ 倍；
  - `### 18.1.2` **为什么在 RLVR（如 DeepSeek-R1 数学/代码推理训练）中 Rollout 生成耗时占端到端周期的 $70\%\text{–}90\%$**：建立单 Iteration 总时间 $T_{\mathrm{iter}} = T_{\mathrm{rollout}}(L_{\max}) + T_{\mathrm{verify}} + T_{\mathrm{ref\_fwd}} + T_{\mathrm{train}} + T_{\mathrm{sync}}$ 的定量分解。
- **`## 18.2 训推共置（Co-located）还是训推分离（Disaggregated）：`veRL` (`3D-HybridEngine`) vs. `RollArt` / `Weave`**
  - `### 18.2.1` **训推共置与零冗余重分片（`veRL` / HybridFlow EuroSys '25）**：剖析训练阶段最优并行（如 $TP_{\mathrm{tr}}=4, PP_{\mathrm{tr}}=4, DP_{\mathrm{tr}}=32$ 配合 ZeRO/FSDP）与推理阶段最优并行（如 $TP_{\mathrm{inf}}=2, PP_{\mathrm{inf}}=1$ 多实例并发）的拓扑冲突；详解 **`3D-HybridEngine`** 如何在同一组 GPU 上通过微秒级组内 All-Gather 重分片完成“训练态 $\to$ 推理态”原地切换，消除跨集群全量权重复制；
  - `### 18.2.2` **物理训推分离（`RollArt` / `OpenRLHF` / `AsyncFlow` / `HetRL`）与异构算力配比**：结合 [`case-studies/rl-scheduling-and-recovery.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/rl-scheduling-and-recovery.md)，当集群同时拥有推理特化卡（如 H20 / L40S / Groq LPX）与训练特化卡（如 H100 / B200）时，如何将 Rollout 池与 Learner 池物理分离，通过高速 RDMA 异步推送更新权重（衔接第 10 章 `10.6.3` 状态机），并导出最优训推卡数配比公式 $\frac{N_{\mathrm{rollout}}}{N_{\mathrm{train}}} = \frac{W_{\mathrm{rollout}} / \eta_{\mathrm{inf}}}{W_{\mathrm{train}} / \eta_{\mathrm{tr}}}$。
- **`## 18.3 攻克长思维链（Long-CoT）的长尾木桶效应：动态拼箱、长度偏差消除与异步部分采样`**
  - `### 18.3.1` **长思维链输出的幂律尾分布与世代尾气泡（Generation Tail Bubble）**：在同步 RL 中，同一批次内 $90\%$ 的轨迹已在 $2\mathrm{K}$ tokens 结束，仅 $5\%$ 的复杂推理轨迹拉长至 $32\mathrm{K}\text{–}64\mathrm{K}$ tokens，导致全集群数千张卡空转等待最慢一条轨迹；
  - `### 18.3.2` **尾部轨迹重分批与长度偏差消除（`RollPacker` OSDI/NSDI '26）、上下文并行动态升降档与自适应投机采样（`Seer` OSDI '26 / `DAS` MLSys '26）**：结合 [`case-studies/rollout-tail-and-sampling.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/rollout-tail-and-sampling.md)，当活跃请求数随完成而衰减时，动态收缩 DP 实例、将剩余超长轨迹热迁移拼箱并开启高并行度 FlashDecoding 或利用同批已完成短轨迹构建无草稿模型自适应投机树（**Seer**）；同时严格复算 **RollPacker** 揭示的动态按块重分批陷阱——**为何按微批求平均（Per-microbatch mean）会使长轨迹的词元梯度权重从 $1/2$ 扭曲为 $1/4$，必须改用全局有效总词元数归一化（Total-token normalization）保证数学等价**；
  - `### 18.3.3` **异步单步/多步偏离策略（One-Step Off-Policy）与陈旧度有界部分采样（Staleness-Bounded Partial Rollout）**：允许未完成的超长轨迹跨训练步继续生成（在版本 $v$ 启动、在版本 $v+\tau$ 完成），只要策略陈旧度 $\tau \le \tau_{\max}$ 即通过截断重要性采样（Truncated Importance Sampling）或近端掩码纳入训练，将 GPU 利用率从 $<45\%$ 推高至 $>88\%$；同步引入 **RobustRL（OSDI '26）** 的分角色无停机故障恢复。
- **`## 18.4 解耦式奖励验证底座（`DistRS` MLSys/NSDI '26）与测试时计算扩展（Test-Time Compute Scaling）系统`**
  - `### 18.4.1` **代码/形式化证明验证沙箱的突发拥塞与条件剩余时间抢占（`DistRS` MLSys/NSDI '26）**：结合 [`case-studies/reward-deadlines-and-feedback.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/reward-deadlines-and-feedback.md)，当数万条代码轨迹涌向单元测试或 Lean4 证明器时，如何将 CPU 沙箱验证与生成流式重叠，并基于条件期望剩余时间 $\mathbb{E}[S - e \mid S > e]$ 与超时阈值（如 **`216,000` tokens 沉没工作量边界**）裁剪死循环/超时验证任务；
  - `### 18.4.2` **推理时搜索（Best-of-N、MCTS、PRM 过程奖励引导树搜索）与投机思维树探索（`Breaking the Reward Barrier` OSDI '26）**：核算树状搜索分支共享公共前缀 KV 块（PagedAttention 引用计数写时复制 CoW）的显存节省，以及在重型 PRM 打分尚未返回时**投机性提前展开高概率子节点（Speculative ToT Exploration）** 的吞吐收益。
- **`## 常见误区`、`## 练习与实验`（`18-1` ~ `18-12`）、`## 本章小结`**

#### 第 18 章 18 幅 420pt 配图清单（`图 18-1` ~ `图 18-18`）
- `图 18-1`（`ch18/ch18-ppo-four-model-vs-grpo-memory-ledger.svg`）：PPO 四模型（Actor/Critic/Ref/RM）与 GRPO/DAPO（消去 Critic + 组采样 $G$ 条轨迹）的显存与计算量对比；
- `图 18-2`（`ch18/ch18-rlvr-iteration-timeline-rollout-dominance.svg`）：RLVR 单步训练时间线分解——为何自回归 Rollout 占据 $70\%\text{–}90\%$ 墙钟时间且算术强度极低；
- `图 18-3`（`ch18/ch18-verl-3d-hybrid-engine-zero-redundancy-reshard.svg`）：veRL（HybridFlow）`3D-HybridEngine` 训练态（ZeRO+TP+PP）与生成态（微张量并行）的零冗余内存重分片机制；
- `图 18-4`（`ch18/ch18-colocated-vs-disaggregated-rl-architecture.svg`）：训推共置（veRL）与物理训推分离（RollArt / OpenRLHF / HetRL）的架构与适用场景分界；
- `图 18-5`（`ch18/ch18-long-cot-power-law-straggler-bubble.svg`）：长思维链（Long-CoT）输出长度幂律分布引发的同步 Rollout 尾部巨大 GPU 空泡；
- `图 18-6`（`ch18/ch18-rollpacker-tail-migration-and-cp-scaling.svg`）：RollPacker 尾部存活请求跨实例紧缩迁移、动态上下文并行与“微批均值（$1/2 \to 1/4$ 长度偏差）vs. 全局总词元归一化”对比；
- `图 18-7`（`ch18/ch18-seer-divided-and-conquered-speculative-rollout.svg`）：Seer（OSDI '26）分段征服长尾 Rollout 与组内共享前缀投机解码加速；
- `图 18-8`（`ch18/ch18-staleness-bounded-partial-rollout-pipeline.svg`）：同步屏障 Rollout vs. 陈旧度有界（$\tau \le \tau_{\max}$）异步部分采样（Partial Rollout）的流水线吞吐跃升；
- `图 18-9`（`ch18/ch18-off-policy-importance-sampling-clipping.svg`）：异步 RL 中策略版本偏移 $\Delta v = \tau$ 下的重要性采样比率波动与稳定性截断边界；
- `图 18-10`（`ch18/ch18-robustrl-role-based-fault-isolation.svg`）：RobustRL（OSDI '26）区分无状态 Rollout 实例故障（秒级丢弃轨迹重采）与有状态 Learner 故障的非对称容错；
- `图 18-11`（`ch18/ch18-distrs-disaggregated-reward-sandbox-streaming.svg`）：DistRS 将代码编译/单测沙箱验证与长尾生成流式重叠及基于 $\mathbb{E}[S-e \mid S>e]$（`216,000` tokens 阈值）的超期裁剪；
- `图 18-12`（`ch18/ch18-test-time-scaling-prm-tree-search-cow-kv.svg`）：推理时扩展（Best-of-N / PRM 树搜索）利用 PagedAttention 写时复制（CoW）共享父节点 KV 缓存；
- `图 18-13`（`ch18/ch18-speculative-tot-reward-barrier-bypass.svg`）：Breaking the Reward Barrier（OSDI '26）在等待 PRM 评分期间投机性预展开高置信分支的时序对比；
- `图 18-14`（`ch18/ch18-multi-turn-tool-env-rl-state-suspension.svg`）：多轮工具调用 / SWE-bench 强化学习中等待外部环境返回时的 KV 换出挂起与高并发复用；
- `图 18-15`（`ch18/ch18-hetrl-heterogeneous-inference-training-matching.svg`）：HetRL 在带宽型推理卡与算力型训练卡之间的最优配比曲线；
- `图 18-16`（`ch18/ch18-dapo-dynamic-sampling-zero-variance-filter.svg`）：DAPO 动态过滤全对（$r=1$）与全错（$r=0$）零梯度提示词对有效梯度产出率的提升；
- `图 18-17`（`ch18/ch18-rl-post-training-throughput-roofline.svg`）：大规模 RLVR 系统从朴素同步共置到异步分池+尾部拼箱的四阶有效吞吐跃迁瀑布图；
- `图 18-18`（`ch18/ch18-rlvr-and-test-time-compute-closed-loop.svg`）：测试时搜索合成高价值轨迹 $\to$ 沙箱验证 $\to$ 异步 RLVR 更新 $\to$ 蒸馏回直觉模型的自我演进闭环。

---

### 4.3 Batch 10 新建第 19 章：《具身智能（VLA）、实时控制与大规模物理仿真底座》（`manuscripts/19-具身智能（VLA）、实时控制与大规模物理仿真底座.md`，`~1,050` 行，`16` 幅配图，`10` 道习题）

- **`## 19.1 物理世界不可回滚：具身智能（Embodied AI / VLA）的硬实时截止期与延迟退化法则`**
  - `### 19.1.1` **云端对话 SLO（软实时百毫秒）vs. 物理机器人/自动驾驶闭环控制（硬实时 $5\text{–}20\ \mathrm{ms}$ 截止期与 $<1\ \mathrm{ms}$ 抖动）**：在物理控制回路中，推理延迟 $\Delta t_{\mathrm{inf}}$ 直接转化为机械臂或车辆在旧观测下的盲飞位移 $\Delta x = v \cdot \Delta t_{\mathrm{inf}}$；
  - `### 19.1.2` **推理延迟如何摧毁操作成功率（`VLALagBench` / `VLA-Perf` 定量模型）**：展示为何在静态抓取中延迟仅拉长完成时间，而在动态目标追踪与精细接触操作中，当端到端感知—动作延迟超过 $100\ \mathrm{ms}$ 时，由于状态观测过时（Observation Staleness）导致动作轨迹与真实物理接触面失配，成功率从 $>80\%$ 断崖式跌破 $40\%$；
  - `### 19.1.3` **动作块预测（Action Chunking，一次预测未来 $H=16\text{–}50$ 步动作）与时间集成平滑（Temporal Ensembling）的开环盲区折中**。
- **`## 19.2 慢思考与快反射解耦：System 2（VLM）+ System 1（扩散/流匹配动作头）双频异步架构`**
  - `### 19.2.1` **双系统异构频率协同（$\pi_0$ / Helix / GR00T / RT-H 范式）**：**System 2**（7B–14B 多模态 VLM，运行于 $1\text{–}5\ \mathrm{Hz}$，周期 $200\text{–}1,000\ \mathrm{ms}$）负责高维多视角图像理解、任务分解与潜空间条件特征更新；**System 1**（轻量级 100M–300M Flow-Matching / Diffusion Policy 动作专家，运行于 $50\text{–}200\ \mathrm{Hz}$，周期 $5\text{–}20\ \mathrm{ms}$）以最新本体关节状态（Proprioception）和高频力觉反馈为输入，在固定步数（如 Flow-Matching 4–10 步 ODE 积分）内输出平滑关节力矩/位置指令；
  - `### 19.2.2` **在单颗车载/机器人 SoC（NVIDIA Jetson Thor / Drive Thor）上的硬件级确定性隔离**：解析如何利用 **GPU MIG（Multi-Instance GPU）切分或 SM 分区 + 锁页共享内存无锁环形缓冲（Lock-Free Ring Buffer）+ 实时 Linux（PREEMPT_RT / QNX）核绑定**，确保 System 2 的长耗时视觉 Prefill 绝不抢占 System 1 的 $5\ \mathrm{ms}$ 硬实时控制周期与内存带宽，严格界定最坏执行时间（WCET）。
- **`## 19.3 万级并行 GPU 物理仿真底座：Isaac Lab、Genesis 与 MuJoCo Warp 的三合一调度`**
  - `### 19.3.1` **为什么机器人不能只靠真机采集：单张 GPU 上并发运行 $4,096\text{–}16,384$ 个物理环境的吞吐革命**：消除传统 CPU 物理引擎（MuJoCo CPU）与 GPU 策略网络之间每步跨 PCIe 拷贝观测与动作的致命瓶颈，将**刚体/柔体碰撞约束求解（XPBD / 接触雅可比迭代）$\to$ 批量光线追踪视觉渲染（Vulkan / OptiX Ray-Tracing）$\to$ 策略网络批量推理与 PPO 更新**全部驻留于同一张 GPU 显存内；
  - `### 19.3.2` **物理仿真器、渲染器与神经网络在单卡 HBM 与 SM 上的三方资源博弈**：定量核算 $N_{\mathrm{env}} = 4,096$ 个环境下的接触点缓冲区、BVH 几何加速结构、多相机帧缓冲与策略激活显存账本，以及多 GPU 跨节点仿真时的环境分区与策略梯度同步。
- **`## 19.4 真机车队影子模式（Shadow Mode）、难例挖掘与 Sim-to-Real 闭环数据基础设施`**
  - `### 19.4.1` **端侧多传感器硬件脉冲同步（PTP / 相机曝光触发 + LiDAR + IMU + 关节编码器）与机载环形黑匣子缓冲**；
  - `### 19.4.2` **基于新颖度/不确定性触发的难例剪辑上传（Edge Triage）与云端 3D 高斯泼溅（3DGS）神经重建闭环仿真**。
- **`## 常见误区`、`## 练习与实验`（`19-1` ~ `19-10`）、`## 本章小结`**

#### 第 19 章 16 幅 420pt 配图清单（`图 19-1` ~ `图 19-16`）
- `图 19-1`（`ch19/ch19-cloud-llm-slo-vs-embodied-hard-realtime.svg`）：云端对话软实时 SLO 与物理世界闭环控制硬实时截止期（盲飞位移 $\Delta x = v \cdot \Delta t$）对比；
- `图 19-2`（`ch19/ch19-vlalagbench-latency-vs-dynamic-grasp-success.svg`）：VLALagBench 实证规律——静态操作 vs. 动态操作随推理延迟 $\Delta t_{\mathrm{inf}}$ 增大的成功率断崖曲线；
- `图 19-3`（`ch19/ch19-action-chunking-and-temporal-ensembling.svg`）：动作块预测（Action Chunking $H$ 步）与滑动窗口指数加权时间集成（Temporal Ensembling）时序图；
- `图 19-4`（`ch19/ch19-system2-vlm-system1-flow-matching-dual-loop.svg`）：System 2（$2\ \mathrm{Hz}$ 多模态 VLM 慢思考）+ System 1（$100\ \mathrm{Hz}$ Flow-Matching 动作头快反射）双频异步架构；
- `图 19-5`（`ch19/ch19-jetson-thor-mig-rtos-deterministic-isolation.svg`）：NVIDIA Jetson Thor / Drive Thor 上利用 GPU MIG / SM 分区与无锁共享内存隔离 System 2 干扰的 WCET 保障；
- `图 19-6`（`ch19/ch19-flow-matching-ode-steps-vs-control-jitter.svg`）：动作生成头中 DDPM 多步采样 vs. Flow-Matching 少步确定性 ODE 积分的周期稳定性对比；
- `图 19-7`（`ch19/ch19-cpu-sim-pcie-bottleneck-vs-gpu-native-sim.svg`）：传统 CPU 物理仿真 + PCIe 往返 vs. GPU 原生万级并行仿真（Isaac Lab / MuJoCo Warp / Genesis）的数据零搬移闭环；
- `图 19-8`（`ch19/ch19-physics-renderer-policy-three-way-sm-budget.svg`）：单张 GPU 内刚体接触求解器、批量光线追踪渲染器与 VLA 策略网络的三方显存与 SM 时间片分配；
- `图 19-9`（`ch19/ch19-contact-divergence-warp-inefficiency.svg`）：4,096 个并行仿真环境中部分机械臂发生复杂多点碰撞导致的 SIMT Warp 分支发散与重排序紧凑化；
- `图 19-10`（`ch19/ch19-multi-camera-tiled-rendering-zero-copy-tensor.svg`）：多环境相机画布拼接渲染直接映射为 PyTorch 张量（零拷贝 DMA-BUF）避免逐环境小 Kernel 启动；
- `图 19-11`（`ch19/ch19-multi-sensor-ptp-hardware-trigger-alignment.svg`）：多目相机曝光、激光雷达扫描、IMU 与关节力矩传感器的微秒级硬件 PTP 时间戳对齐；
- `图 19-12`（`ch19/ch19-shadow-mode-disagreement-edge-trigger.svg`）：真机影子模式（Shadow Mode）：利用在役策略与影子策略的动作分歧度触发高价值难例切片回传；
- `图 19-13`（`ch19/ch19-3dgs-neural-reconstruction-sim2real-loop.svg`）：利用路采/真机日志构建 3D 高斯泼溅（3DGS）可微数字孪生场景并用于闭环重仿真的完整流水线；
- `图 19-14`（`ch19/ch19-onboard-vs-mec-edge-offload-safety-fallback.svg`）：机载端侧计算与局域 5G/Wi-Fi 7 边缘卸载的抖动边界及本地安全回退控制器（Safety Shield）；
- `图 19-15`（`ch19/ch19-vla-early-exit-and-token-pruning-wcet.svg`）：面向硬实时截止期的多视角视觉 Token 动态剪枝（VLA-Pruner）与层早退机制；
- `图 19-16`（`ch19/ch19-embodied-ai-full-stack-infrastructure-map.svg`）：具身智能从万级 GPU 物理仿真、双频实时机载推理到车队影子数据飞轮的全栈基础设施总图。

---

### 4.4 Batch 10 新建第 20 章：《十万卡集群可靠性工程、静默数据损坏（SDC）防御与基准评测科学》（`manuscripts/20-十万卡集群可靠性工程、静默数据损坏（SDC）防御与基准评测科学.md`，`~1,100` 行，`16` 幅配图，`10` 道习题）

- **`## 20.1 十万卡规模定律的背面：从“几天一次故障”到“每小时一次中断”的可靠性数学`**
  - `### 20.1.1` **集群平均无故障时间 $\mathrm{MTBF}_{\mathrm{cluster}} = \frac{\mathrm{MTBF}_{\mathrm{node}}}{N}$ 与真实万卡故障账本**：深度拆解 Meta Llama 3 405B 在 `16,384` 张 H100 上历时 `54` 天的真实运行记录——共发生 `419` 次非计划中断（平均每 `2.9` 小时一次！），即使有效训练时间占比达 `>90%`，单是硬件故障（GPU HBM ECC 不可纠正错误、SRAM 奇偶校验、NVLink/光模块闪断、PSU/主机崩溃）就占到了 **`78%`**；推演当集群规模再扩 $6\times$ 至 `100,000` 卡时，平均每 `25–30` 分钟就会发生一次硬件异常；
  - `### 20.1.2` **热备节点池（Hot-Spare Pool）容量规划与排队论（Erlang-C / M/M/s/K 修复队列模型）**：基于 **MLSys '26（Understanding Failures and Optimizing Sparing Strategies for Large-Scale AI Clusters）**，给定节点失效率 $\lambda$、硬件平均修复/更换周期 $\mathrm{MTTR}_{\mathrm{repair}}$（如 $4\text{–}24\ \mathrm{h}$）与同拓扑超节点域约束，推导以 $99.9\%$ 概率保证训练不降级所需的最小热备节点比例。
- **`## 20.2 最危险的敌人不是宕机而是算错：静默数据损坏（SDC）物理机理与毫秒级在线拦截`**
  - `### 20.2.1` **为什么 GPU 不报错却悄悄毒化万卡权重：静默数据损坏（Silent Data Corruption）实证（`SDCs in the Wild` & `Safeguarding LLM Training at Scale: 35M GPU Hours` OSDI '26 #71）**：剖析宇宙射线单粒子翻转（SEU）、硅片老化边缘时序违例、低电压瞬态跌落（Voltage Droop）与温度热点如何导致 Tensor Core 累加器或无 ECC 保护的数据通路发生比特翻转（如 BF16 指数最高位由 `0` 翻为 `1`，数值瞬间放大 $2^{64}$ 倍或产生静默偏差），并通过一次 AllReduce 瞬间污染全集群数万张卡的梯度与动量状态；
  - `### 20.2.2` **为什么全量双重执行（DMR，$100\%$ 算力税）不可行，以及毫秒级在线 SDC 防御与精确定位三道防线（`Safeguarding LLM Training at Scale` OSDI '26 #71 + `OpGuard` OSDI '26 #72）**：
    1. **算子级代数不变量校验（Algorithm-Based Fault Tolerance, ABFT）**：利用矩阵乘法行和/列和恒等式 $\mathbf{1}^T (AB) = (\mathbf{1}^T A) B$，仅以 $O(MN + NK)$ 的向量外围计算（占 $O(MNK)$ 矩阵乘算力开销 **$<0.5\%$**）在线捕获 Tensor Core 内部硬件计算错误；
    2. **轻量级在线异常感知与跨秩统计校验（`Safeguarding LLM Training at Scale` OSDI '26 #71）**：总结 3,500 万 GPU 小时生产实战，将高频低开销的在线统计感知（在 AllReduce 归约前校验各 Rank 局部梯度范数 $\|g_i\|_2$ 与激活极值）与按需触发的确定性复核相分离，在毒化梯度混入全局缓冲前将其隔离；
    3. **消除良性非确定性实现位级对齐算子定位（`OpGuard` OSDI '26 #72）与微批次重跑热踢出**：当检测到数值偏离时，通过 **OpGuard** 控制原子加法顺序与通信归约树等良性软硬件非确定性，实现逐算子位级精确对齐（Bitwise Alignment），在微批次级局部重算（耗时几十毫秒）中精确定位首个产生 SDC 的故障算子与物理坏卡，立即将该卡踢出通信域并由热备卡接管，**避免回滚数小时前的全局检查点**。
- **`## 20.3 告别全集群重启：机队级健康自愈、慢节点（Straggler）取证与无重启弹性容错`**
  - `### 20.3.1` **从“一卡挂、万卡停”到亚秒级热迁移与副本组自治（`TorchFT` / `TrainMover` OSDI '26 / `EROICA` & `FLARE` NSDI '26）**：对比传统 `torchrun` 全局销毁重建通信组（耗时 $3\text{–}15\ \mathrm{min}$）与基于预初始化备用进程和 RDMA 虚拟地址重映射（**TrainMover**）的亚秒级无重启恢复；
  - `### 20.3.2` **比宕机更隐蔽的慢节点（Gray Failure / Straggler）与 `NCCL Flight Recorder` 环路死锁取证**：当某张卡因光模块误码率升高（BER 触发 FEC 重传）或冷板微堵降频导致每步慢 $15\%$ 时，如何通过跨 Rank 集合通信到达时间差（Arrival Skew Matrix）自动定位真正的根因慢节点（而非被其阻塞的下游无辜等待者）。
- **`## 20.4 AI 基础设施基准评测科学：如何识破跑分魔术并构建可复现、无偏剖析的工业级基准`**
  - `### 20.4.1` **为什么公开基准上的“$3\times$ 加速”上线后往往缩水甚至倒退：五大跑分失真机制（Benchmark Gaming）**：剖析（1）用随机词表 Token 测前缀缓存/投机解码、（2）脱离 TTFT/TPOT 百分位 SLO 谈极限吞吐、（3）忽略客户端突发相关性（**ServeGen**）、（4）测量窗口过短掩盖显存碎片与 GC 抖动、（5）隐藏反量化与精度退化（Perplexity 漂移）；
  - `### 20.4.2` **从算子与通信基准（`FlashInfer-Bench` MLSys '26 / `CCEval` NSDI '26）、分布漂移基准（`DriftBench` MLSys '26）到消除扁平负载采样失真的可信剖析（`Blink` OSDI '26 #23）与防污染评测（`LiveBench`）**：剖析为何基于硬件 PMU 采样的传统 Profiler 在短函数/扁平主机运行时负载上会出现严重归因偏差（*When Sampling Lies*），详解 **Blink（OSDI '26）** 如何恢复可信性能剖析，并构建兼具**算子硬件效率（FlashInfer-Bench）+ 网络集合通信基准（CCEval）+ 端到端 SLO Goodput（DriftBench）+ 模型质量防污染验证（LiveBench / KL 散度）**的四维黄金评测规范。
- **`## 20.5 全书二十章终极大收束：从一次矩阵乘法的五个追问，到驾驭十万卡智能工厂的工程哲学`**
  - `### 20.5.1` **全书二十章知识坐标系总览与双轨主线（Qwen3-8B Dense 轨 & DeepSeek V4.1 / MoE 轨）二十章收官总表**；
  - `### 20.5.2` **AI 基础设施架构师的十条第一性原理铁律**。
- **`## 常见误区`、`## 练习与实验`（`20-1` ~ `20-10`）、`## 本章小结`**

#### 第 20 章 16 幅 420pt 配图清单（`图 20-1` ~ `图 20-16`）
- `图 20-1`（`ch20/ch20-cluster-mtbf-scaling-and-llama3-failures.svg`）：集群规模从 $1\mathrm{K}$ 扩至 $100\mathrm{K}$ 卡时 $\mathrm{MTBF}_{\mathrm{cluster}}$ 从数天跌至 $25$ 分钟的双对数曲线及 Meta $16,384$ 卡 $419$ 次中断分类账本；
- `图 20-2`（`ch20/ch20-hot-spare-pool-queuing-dimensioning.svg`）：基于修复队列模型（MTBF vs. MTTR）的热备节点池（Hot-Spare Pool）配置比例与集群可用性曲线；
- `图 20-3`（`ch20/ch20-sdc-bitflip-allreduce-contamination.svg`）：单颗 GPU Tensor Core 静默比特翻转（SDC）如何逃过 HBM ECC 并通过一次 AllReduce 瞬间污染万卡权重；
- `图 20-4`（`ch20/ch20-abft-checksum-matrix-multiplication.svg`）：基于代数不变量（ABFT 行列校验和 $\mathbf{1}^T(AB) = (\mathbf{1}^T A)B$）以 $<0.5\%$ 算力开销在线拦截矩阵乘硬件错误；
- `图 20-5`（`ch20/ch20-safeguarding-sdc-and-opguard-bitwise-debug.svg`）：3500 万 GPU 小时在线 SDC 感知隔离（`Safeguarding LLM Training at Scale` OSDI '26）与控制良性非确定性的位级对齐故障算子定位（`OpGuard` OSDI '26）；
- `图 20-6`（`ch20/ch20-microbatch-rerun-vs-checkpoint-rollback.svg`）：检测到异常后原地重跑当前微批次（耗时 $50\ \mathrm{ms}$）区分瞬态软错误与永久硬件故障 vs. 盲目回滚数小时全局检查点；
- `图 20-7`（`ch20/ch20-trainmover-subsecond-hot-migration.svg`）：传统全局销毁重启（`torchrun` 耗时数分钟）vs. TrainMover（OSDI '26）利用预热备用进程与 RDMA 重映射实现亚秒级无重启接管；
- `图 20-8`（`ch20/ch20-nccl-flight-recorder-straggler-blame.svg`）：利用 `NCCL Flight Recorder` 与集合通信到达时间偏斜矩阵（Arrival Skew）从全网阻塞假象中揪出唯一根因慢节点；
- `图 20-9`（`ch20/ch20-optical-ber-fec-gray-failure-detection.svg`）：光模块劣化引发 FEC 纠错重传（灰故障 Gray Failure）导致通信有效带宽周期性塌陷的特征识别；
- `图 20-10`（`ch20/ch20-five-benchmark-gaming-traps.svg`）：AI 系统跑分中的五大失真陷阱（合成随机词元、无 SLO 极限吞吐、泊松到达假设、短窗口预热、精度静默退化）；
- `图 20-11`（`ch20/ch20-flashinfer-bench-cceval-and-blink-profiling.svg`）：FlashInfer-Bench（算子基准）、CCEval（集合通信基准）、DriftBench（负载漂移）与 Blink（OSDI '26 消除扁平负载 PMU 采样失真）的四维评测与剖析框架；
- `图 20-12`（`ch20/ch20-effective-training-time-ratio-ettr-model.svg`）：十万卡集群有效训练时间比（ETTR / Goodput）对检查点周期 $\tau_{\mathrm{ckpt}}$、恢复时间 $T_{\mathrm{rec}}$ 与故障率 $\lambda$ 的三维曲面；
- `图 20-13`（`ch20/ch20-automated-burn-in-fleet-health-pipeline.svg`）：新节点入网与故障修复后的自动化烤机（Burn-in：HPL/DCGM 压测 + 光链路裕量扫描 + SDC 校验）准入流水线；
- `图 20-14`（`ch20/ch20-twenty-chapter-dual-track-master-ledger.svg`）：全书 20 章双轨贯穿案例（Qwen3-8B Dense 轨与 DeepSeek V4.1 / MoE 轨）从第 1 章到第 20 章的完整工程参数演进总表；
- `图 20-15`（`ch20/ch20-twenty-chapter-architecture-knowledge-graph.svg`）：全书 20 章“模型与负载 $\to$ 芯片与系统 $\to$ 推理、训练与六大前沿底座”的知识体系全景图；
- `图 20-16`（`ch20/ch20-ten-first-principles-of-ai-infra.svg`）：AI 基础设施架构师的十条第一性原理终极法则总结图。

---

## 五、`experiments/ch14` – `experiments/ch20` 可独立运行实验套件设计

为解决当前 `experiments/` 目录仅覆盖 `ch01`–`ch13` 的结构性缺口，路径 D 将在对应批次新建 **`experiments/ch14/` 至 `experiments/ch20/`**，严格遵循 [`experiments/README.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/README.md) 规范（每个实验含仅依赖 Python 3.8+ 标准库的 `run.py`、`README.md`、`results/` 与 `verify.py`）：

| 实验目录 | 对应章节与习题 | 核心可复算内容与确定性校验指标 | 交付批次 |
| --- | --- | --- | :-: |
| [`experiments/ch14/14-01/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments) | 第 14 章 `实验 14-12` / `14-15` | **全生命周期过度训练最优配比与 K192 预注册证伪复核器**：复算计入推理生命周期 Token 量 $D_{\mathrm{inf}}$ 后的 Chinchilla 偏移曲线，并校验实验 `13-07` 的 10 个配对比值（中位数 $R=0.693141 < 0.70 \implies \text{refuted\_in\_scope}$）。 | Batch 7 |
| [`experiments/ch14/14-02/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments) | 第 14 章 `练习 14-11` / `14-13` | **2023–2026 百倍降本逐级条件比率链（Telescoping Waterfall）与 FarSkip 跳层通信掩盖模拟器**：复算 $S_0 \to S_3$（$20.90\ \mathrm{ms} \to 0.83\ \mathrm{ms}$）加入主机开销后的端到端加速比与跳层通信掩盖窗口。 | Batch 7 |
| [`experiments/ch15/15-01/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments) | 第 15 章 `实验 15-10` / `15-14` | **吉瓦级园区供电、*TAPAS* 热感知重配置（30 请求回本）与 LogicFolding 功率密度悖论复算器**：复算 `[102, 86] kW` 削峰、`3 s / 0.1 s = 30` 请求回本点、$(0.55/0.85)^2 \approx 0.42$ 降压与 $0.75/0.60 = 1.25\times$ 功率密度反升。 | Batch 8 |
| [`experiments/ch15/15-02/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments) | 第 15 章 `练习 15-12` / `15-13` | **DiLoCo 跨园区广域网训练带宽压缩（$500\times$）、Kareus 动静态能耗合并与园区储能（BESS）套利账本**。 | Batch 8 |
| [`experiments/ch16/16-01/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments) | 第 16 章 `实验 16-10` / `16-14` | **DeepSeek DSec 38 万微虚机沙箱内存账本（`virtio-pmem` DAX $-40.2\%$ 与 $1/64$ `struct page` 税）及 *PipeLLM* $192\ \mathrm{MiB}$ 三缓冲加密流水线模拟器**。 | Batch 8 |
| [`experiments/ch16/16-02/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments) | 第 16 章 `练习 16-11` / `16-12` | **算例 `I19` 审计日志中断排空（$81.92\ \mathrm{s}$ 溢出 / $76.92\ \mathrm{s}$ 排空）、MoE 流量填充带宽税与乐观可验证推理（OPML）纳什均衡计算器**。 | Batch 8 |
| [`experiments/ch17/17-01/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments) | 第 17 章 `实验 17-12` | **3D 视频 DiT（$489,600$ Tokens）2D 统一序列并行（Ulysses $\times$ Ring USP）通信量、3D VAE 分块显存峰值与 TeaCache 跳步加速比复算器**。 | Batch 9 |
| [`experiments/ch18/18-01/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments) | 第 18 章 `实验 18-12` | **RLVR（PPO vs. GRPO）显存账本、长思维链幂律尾气泡与陈旧度有界异步部分采样（Partial Rollout）吞吐模拟器**。 | Batch 9 |
| [`experiments/ch19/19-01/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments) | 第 19 章 `实验 19-10` | **具身智能 VLA 双频控制回路（System 2 + System 1）盲飞位移、VLALagBench 成功率退化与单卡 4,096 并行物理仿真显存/SM 预算器**。 | Batch 10 |
| [`experiments/ch20/20-01/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments) | 第 20 章 `实验 20-10` | **十万卡集群故障率（Young-Daly 最优检查点周期）、热备池 Erlang-C 排队模型与 ABFT 矩阵乘行列校验和 SDC 注入拦截实验**。 | Batch 10 |

---

## 六、出版级配图（`gpt-image-2.6-sunburst` + `420pt` 矢量规范）与全书 20 章构建适配清单

### 6.1 强制调用 `codex exec` + `gpt-image-2.6-sunburst`（`quality="high"`）与五大视觉美学标准
1. **模型与最高质量档位**：针对路径 D 新增的全部 **116 幅配图**（第 1–12 章新增 24 幅 + 第 13–16 章新增 24 幅 + 第 17–20 章新增 68 幅）以及更新的全书路线图 `ch01/figure-1-book-roadmap.svg`，在每一批次中必须通过 `codex exec` 调用 **`gpt-image-2.6-sunburst`** 模型，并显式指定 **`quality="high"`**（最高质量档位）生成高精度视觉构图参考或设计原型，再结合 [`manuscripts/figure_style/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/figure_style) 与各章绘图脚本输出 `420pt` 宽度的矢量 `.svg`、`.pdf` 与高分 `.png`（三格式齐全，100% 物理落盘，严禁 LFS 指针）。
2. **五大视觉维度硬指标（字体、对齐、排版留白、配色、自包含定量信息）**：
   - **字体（Typography）**：统一采用 `PingFang SC` / `Source Han Sans SC` / `Noto Sans CJK SC` 与 `Helvetica Neue` / `Arial` 无衬线字体族；主标题 `9.5–10.5 pt` 加粗，子面板标题 `8.5–9.0 pt` 加粗，数据标注与轴刻度 `7.5–8.5 pt`，图注说明 `7.0–7.5 pt`；同一层级字号全图严格一致，中英文与数字基线端正。
   - **对齐（Alignment）**：所有模块框、流水线阶段、坐标轴、图例与文本标签严格对齐点阵网格（Grid-Snapped）；多面板（Panel A / B / C）顶边、底边与列间距保持几何级对称。
   - **排版与留白（Layout & Whitespace）**：画布宽度严格锁定为 **`420pt`**（匹配 ElegantBook 正文版心宽度，1:1 零缩放嵌入），外边距充足，模块内边距（Padding）不低于 `6–8 pt`；**严禁任何文字压框线、文字越界、箭头穿过文字、图例遮挡曲线或双行标签重叠**。
   - **色彩体系（Color Palette）**：严格遵循 [`manuscripts/figure_style/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/figure_style) 学术出版级色板（深海蓝 `#1F4E79` / `#2B5B84`、青黛绿 `#1B7837` / `#2A9D8F`、琥珀橙 `#D95F02` / `#E76F51`、珊瑚红 `#C0392B`、岩灰 `#595959` 与柔和浅底色 `#F4F7FA` / `#FFF8F0`），用高对比度语义色突出关键瓶颈与拐点。
   - **Subagent 视觉巡检门禁（Visual Gate）**：每批次配图生成后，主 Agent 与 **Gemini 4 Argon High Review Subagent** 均须使用 `read_file` 逐张打开生成的 `.png` 光栅图像进行直接视觉审查；发现任何排版拥挤、文字截断或对齐瑕疵，立即修正重绘直至零瑕疵通过。

### 6.2 全书 20 章构建、校验与导航基础设施精确修改表（Batch 10 合龙时执行）

| 序号 | 目标文件 | 当前硬编码位置与行号 | 路径 D 精确修改方案 |
| :-: | --- | --- | --- |
| 1 | [`scripts/verify_core_principles.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/scripts/verify_core_principles.py) | 第 10 行：`for n in range(1,17):` | 改为 `for n in range(1,21):`（保持第 25 行 `if n<=12:` 对 `before/` 的历史兼容检查不变），自动校验全部 20 章的 `## 本章小结`、脚注闭合、相对链接、`figure-index.json`（731 幅图）与习题列表（304 块）。 |
| 2 | [`scripts/build_site.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/scripts/build_site.py) | 第 54–58 行：`if n <= 16` 与 `expected = list(range(17))` | 将简体中文版 `zh` 的章节上限改为 `if n <= 20` 与 `expected = list(range(21))`（含 `00-前言` 共 21 个章节级页面；英文/繁体/俄语社区译本保持 `range(13)` 不变）。 |
| 3 | [`scripts/check_site.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/scripts/check_site.py) | 第 46 行：`max_ch = 17 if name == '简体中文' else 13`；第 54 行打印信息 `(16 zh / 12 translations)` | 改为 `max_ch = 21 if name == '简体中文' else 13`，并将第 54 行打印信息更新为 `(20 zh / 12 translations)`。 |
| 4 | [`book/build_pdf.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/book/build_pdf.py) | 第 70 行：`choices=range(1,17)`；第 79 行：`range(1,17)` | 全部改为 `range(1,21)`，支持单章编译 `--chapter 17..20` 与全书 20 章合订本 PDF 编译。 |
| 5 | [`book/build_epub.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/book/build_epub.py) | 第 65 行：`for n in range(17)`；**第 79 行：`MANUSCRIPTS.glob('[01][0-9]-*.md')`** | 将第 65 行更新为 `for n in range(21)`；**同时必须将第 79 行 `MANUSCRIPTS.glob('[01][0-9]-*.md')` 改为 `MANUSCRIPTS.glob('[0-9][0-9]-*.md')`**（否则 `[01][0-9]` 无法匹配以 `2` 开头的 `20-十万卡集群...md`，会导致跨章锚点解析漏掉第 20 章！）。 |
| 6 | [`book/check_ci_pdf.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/book/check_ci_pdf.py) | 第 10 行：`'expected_text': ('AI Infra', '端边云')`；第 29 行：`expected_chapters = list(range(1, 17)) if args.edition == 'zh' else list(range(1, 13))` | 将第 10 行 `'zh'` 的 `'expected_text'` 更新为 `('AI Infra', '端边云', '十万卡集群')`，并将第 29 行 `'zh'` 的期望章节改为 `list(range(1, 21))`，确保 GitHub Actions CI 全书 20 章 PDF 书签、末章锚点与页数校验通过。 |
| 7 | [`manuscripts/ch01/build_roadmap.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/ch01/build_roadmap.py) | 第 18–27 行：画布高度与第三部分章节列表（当前列至第 16 章） | 扩展第三部分框高与列表，将 `17 多模态、视频 DiT 与世界模型`、`18 大规模后训练（RLVR）与推理时扩展`、`19 具身智能（VLA）与大规模物理仿真`、`20 十万卡集群可靠性与基准科学` 纳入 `figure-1-book-roadmap.{svg,png,pdf}`。 |
| 8 | `manuscripts/ch17/` ~ `manuscripts/ch20/` | 尚不存在 | 新建 `manuscripts/ch17/`、`ch18/`、`ch19/`、`ch20/` 四个目录，分别包含 `figure-index.json`、`figure-data.json`、`build.py` 及各自的 `.svg`/`.png`/`.pdf` 配图资产。 |
| 9 | [`website/index.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/website/index.md)、[`README.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/README.md)、[`manuscripts/README.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/README.md)、[`manuscripts/00-前言.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/00-%E5%89%8D%E8%A8%80.md)、[`case-studies/README.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/README.md)、[`experiments/README.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/README.md) | “十六章导读”、读者定制阅读路线与实验目录说明 | 全面升级为 **“二十章导读”**，更新三类读者（模型/应用、系统/网络、芯片/架构）在 20 章体系下的必读路线图（`D11-SP02`），并在 `case-studies/README.md` 与 `experiments/README.md` 中补齐第 14–20 章案例与可运行实验索引。 |

---

## 七、十批次推进路线图（10 Batches）与六步强制执行标准作业程序（SOP）

### 7.1 十批次推进总表（每 Batch 2 章，完成即验证、Subagent 审阅收敛、Git 提交并推送）

| 批次 | 覆盖章节 | 核心交付物概要 | 新增配图 | 新增习题 | 新增可运行实验目录 |
| :-: | --- | --- | :-: | :-: | :-: |
| **Batch 1** | **第 1 章 + 第 2 章** | **Ch01**：主机内核调度旁路与虚拟智能网卡卸载（`ghOSt` / `Wave` / NSDI '26 `ZOC`）、AI 全栈可观测性与关键路径剖析（MLSys '26 `XProf` / `ProfInfer`）、更新 20 章路线图 `图 1-1`；**Ch02**：原生可训练稀疏注意力（**DeepSeek NSA** vs. **Kimi MoBA** $64$-Token 块硬件对齐 + OSDI '26 `ECHO` 卸载预取）、混合线性注意力循环状态检查点账本（Kimi K3 `69` KDA + `24` MLA、`428.5546875 MiB` 快照 vs. `1.0986328125 MiB/tok`、`Marconi` / SGLang `Unified Radix Cache` / vLLM `Hybrid KV Cache Coordinator`）。 | +4 幅（`1-23`~`1-24`, `2-41`~`2-42`）+ 更新 `1-1` | +4 道（`1-11`~`1-12`, `2-15`~`2-16`） | — |
| **Batch 2** | **第 3 章 + 第 4 章** | **Ch03**：生产推理负载合成科学（NSDI '26 `ServeGen` 客户端突发相关性与双峰重尾分布）、Computer-Use 视觉 GUI Agent 步间延迟分解（MLSys '26 `OSWorld-Human`）；**Ch04**：微缩放低精度格式硬件账本（**NVFP4** 16 元素 `E4M3` 4.5b vs. **MXFP4** 32 元素 `E8M0` 4.25b）、**昇腾 910C/950PR/950DT（第三代达芬奇架构：`L1 512 KB`/`L0A 64 KB`/`L0B 64 KB`/`L0C 256 KB`/`UB 512 KB`、`128 MB L2`、RegFile 双发射 SIMD/SIMT 融合、`BufferID` 同步与满血/裁剪版 SKU）**、**AMD CDNA 4 MI355X** 与 **AWS Trainium2 (`28 MiB` SBUF + `2 MiB` PSUM/core) / Trainium3 (`32 MiB` SBUF + `2 MiB` PSUM/core, Quad-MXFP8/MXFP4, `4.9 vs. 4.7 TB/s` 披露勘误)**。 | +4 幅（`3-36`~`3-37`, `4-48`~`4-49`） | +4 道（`3-16`~`3-17`, `4-15`~`4-16`） | — |
| **Batch 3** | **第 5 章 + 第 6 章** | **Ch05**：**FlashAttention-4（MLSys '26）** Blackwell 非对称扩展与 SFU `exp2` 多项式模拟、**DeepSeek DeepGEMM** 细粒度 FP8（$1\times 128$/$128\times 128$）两级累加与 JIT 形状特化、编译器辅助 CUDA Graph 捕获（OSDI '26 `GraCE`）；**Ch06**：**灵衢 UB 2.0.1（545 页正式规范：`2,016 GB/s` 双向、18 个 x4 端口与 `PCIe 5.0 x16` 复用 4 端口 + `2x400G UBoE/RoCE` 复用 2 端口、HA/CA `64B` 内存语义与 CCU 集合通信卸载）**、超节点分级网格自动切分编译器（`Shardy` / `PartIR`）。 | +4 幅（`5-45`~`5-46`, `6-62`~`6-63`） | +4 道（`5-14`~`5-15`, `6-17`~`6-18`） | — |
| **Batch 4** | **第 7 章 + 第 8 章** | **Ch07**：**超以太网联盟 UEC 1.0**（逐包喷洒、乱序接收、选择性重传）、**Broadcom Tomahawk 6（`102.4 Tbps`）两层扁平 Clos 直连 `16K/32K` 卡** 与 `UCCL-Tran`（OSDI '26）/`AutoCCL`（NSDI '25）；**Ch08**：100% 合入 [`root-dspark-execution/NOTES.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/2026-infra-survey/root-dspark-execution/NOTES.md)（SGLang DSpark 四请求 `7/3/2/2` 紧凑验证窗口、DP 共同档位 `16/32` 与余量填充）、**XGrammar / XGrammar-2** 下推自动机位掩码与跳跃解码（Jump-Forward Tokens）。 | +4 幅（`7-66`~`7-67`, `8-41`~`8-42`） | +4 道（`7-20`~`7-21`, `8-15`~`8-16`） | — |
| **Batch 5** | **第 9 章 + 第 10 章** | **Ch09**：**DeepSeek DeepEP**（Prefill RDMA$\to$NVLink 非对称两级转发 + Decode IBGDA 0-SM Hook 后台通信重叠）与 `SwiftEP`、**Mooncake（FAST '25 Best Paper）** 全局 KV 传输引擎与预测性早期拒绝；**Ch10**：100% 合入 [`root-weight-controller/NOTES.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/2026-infra-survey/root-weight-controller/NOTES.md)（SGLang `tokenizer_manager.py` 六阶段状态机与四响应“3 成 1 败”恢复边界）、**ByteCheckpoint（NSDI '25）** 自动重分片与 **Checkmate（NSDI '26）** 无落盘秒级恢复。 | +4 幅（`9-51`~`9-52`, `10-53`~`10-54`） | +4 道（`9-17`~`9-18`, `10-17`~`10-18`） | — |
| **Batch 6** | **第 11 章 + 第 12 章** | **Ch11**：vLLM 冷启动五阶段延迟解剖（MLSys '26 `Breaking the Ice`）、Serverless 临时流水线扩容（MLSys '26 `FaaScale` / NSDI '26 `HydraServe`）、多作业 RL 混部与抢占式 Spot 算力收割（`Weave` / OSDI '26 `RLBoost`）；**Ch12**：端侧 NPU 静态分块图执行（`ExecuTorch` / `llm.npu`）与动态形状回退 CPU 惩罚、被动散热终端热节流漂移与端云动态卸载边界（含 Ollama `v0.31` MLX MTP vs. `v0.20.7` Metal GGUF）。 | +4 幅（`11-54`~`11-55`, `12-39`~`12-40`） | +4 道（`11-16`~`11-17`, `12-16`~`12-17`） | — |
| **Batch 7** | **第 13 章 + 第 14 章 + `experiments/ch14`** | **Ch13 扩至 ~1,150 行**：新增 `13.8`（`FalconFS` 服务端复合路径解析、`DistVS`/`Helmsman` 跨 DRAM–CXL–SSD 向量检索、`CoPilotIO` CPU 协处理 GPU I/O + `Umap`/`MoonBright`/`kvcached` 弹性虚拟内存页表）；**Ch14 扩至 ~1,150 行**：新增 `14.9`（MLCommons `Chakra` 执行轨迹图、`NeuSight` 瓦片级 Roofline 校准、`FarSkip-Collective` 跨层跳步掩盖通信）及可运行实验 `experiments/ch14/`。 | +12 幅（`13-19`~`13-24`, `14-19`~`14-24`） | +6 道（`13-13`~`13-15`, `14-13`~`14-15`） | `experiments/ch14/14-01`, `14-02` |
| **Batch 8** | **第 15 章 + 第 16 章 + `experiments/ch15..16`** | **Ch15 扩至 ~1,120 行**：新增 `15.6`（两相液冷与硅背面微通道直冷、百兆瓦园区储能 BESS 调频套利、`Kareus` 动静态能耗合并休眠、`Di-PS` 跨洲骨干网训练）；**Ch16 扩至 ~1,080 行**：新增 `16.6`（MoE 专家路由流量侧信道与恒定速率哑元填充、可验证推理 TEE/zkML/OPML 纳什均衡、`JANUS` 嵌套虚拟化 + `Blowfish`/`Prism` 弹性内存气球与跨模型显存协同）及 `experiments/ch15..16/`。 | +12 幅（`15-17`~`15-22`, `16-17`~`16-22`） | +8 道（`15-11`~`15-14`, `16-11`~`16-14`） | `experiments/ch15/15-01..02`, `ch16/16-01..02` |
| **Batch 9** | **新建第 17 章 + 新建第 18 章 + `experiments/ch17..18`** | **新建第 17 章（~1,100 行）**：多模态、视频生成（DiT $489,600$ Tokens）与世界模型基础设施（3D VAE 分块、2D USP `Ulysses x Ring` $\frac{2(P-1)Nd}{P}$ + `db-SP`、DistriFusion/PipeFusion、TeaCache、MLSys '26 `TriInfer` EPD 扩展至 E-P-D-G 四池分离）；**新建第 18 章（~1,100 行）**：大规模后训练（RLVR）、异步强化学习与推理时扩展系统（PPO vs. GRPO、veRL `3D-HybridEngine` vs. RollArt、RollPacker/Seer、陈旧度有界 Partial Rollout、DistRS、PRM 树搜索）及 `experiments/ch17..18/`。 | +36 幅（`17-1`~`17-18`, `18-1`~`18-18`） | +24 道（`17-1`~`17-12`, `18-1`~`18-12`） | `experiments/ch17/17-01`, `ch18/18-01` |
| **Batch 10** | **新建第 19 章 + 新建第 20 章 + `experiments/ch19..20` + 全书 20 章合龙** | **新建第 19 章（~1,050 行）**：具身智能（VLA）、硬实时控制（VLALagBench、System 2 + System 1 双频架构、Jetson Thor MIG/RTOS 隔离）与万级并行 GPU 物理仿真（Isaac Lab / Genesis / MuJoCo Warp）及影子模式飞轮；**新建第 20 章（~1,100 行）**：十万卡集群可靠性工程（Meta 16K 卡 419 次故障、热备池排队论）、静默数据损坏（SDC）在线拦截（ABFT `<0.5%` 校验、OSDI '26 `Safeguarding LLM Training at Scale` 3,500 万 GPU 小时验证 + `OpGuard` 良性非确定性对齐、微批次重跑）、无重启容错（`TrainMover` / `NCCL Flight Recorder`）、防作弊评测科学（`FlashInfer-Bench` / `CCEval` / `DriftBench` / `Blink` PMU 消除采样偏置）与全书 20 章总收束；同步更新构建校验脚本与全站导航。 | +32 幅（`19-1`~`19-16`, `20-1`~`20-16`） | +20 道（`19-1`~`19-10`, `20-1`~`20-10`） | `experiments/ch19/19-01`, `ch20/20-01` |

### 7.2 每一批次的六步强制执行标准作业程序（6-Step Batch SOP）

在后续执行上述每一个 Batch 时，必须严格按以下 6 个步骤闭环推进，任何一步未达标严禁进入下一 Batch：
1. **Step 1 · 锁定真值核对与算例复算**：读取该 Batch 涉及的所有 `references/`、`calculations/`、`experiments/`、`case-studies/` 与 `research/` 原始文件，逐一核对公式、字节数、延迟、吞吐、Git Commit SHA 与配置边界，确保零脱钩、零幻觉。
2. **Step 2 · 调用 `codex exec` + `gpt-image-2.6-sunburst` (`quality="high"`) 生成出版级 420pt 配图**：
   - 使用 `codex exec` 调用 `gpt-image-2.6-sunburst`（最高质量 `quality="high"`）辅助设计并结合 [`manuscripts/figure_style/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/figure_style) 生成 `420pt` 宽度的 `.svg`、`.pdf`、`.png` 三格式配图；
   - 更新对应章节的 `manuscripts/chXX/figure-index.json`（及 `figure-data.json` 与 `build.py`）；
   - 用 `read_file` 逐张检视 `.png` 光栅图像，核查字体、对齐、留白、配色与数值标注。
3. **Step 3 · 正文扩写与 [`refine-prose`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/.agents/skills/refine-prose/SKILL.md) 深度去 AI 腔润色**：
   - 严格遵循 [`.agents/skills/refine-prose/SKILL.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/.agents/skills/refine-prose/SKILL.md) 的铁律，消除一切生造词、欧化长句、空洞拔高与机械排比；
   - 严禁在 Markdown 正文中出现未转义的裸 `$` 货币符号（统一写为 `美元` 或在数学环境内写 `\mathrm{USD}`）。
4. **Step 4 · 自动化脚本全量验证**：
   - 运行 `python3 scripts/typography.py`、`python3 scripts/check_no_html.py` 与 `python3 scripts/verify_core_principles.py`；
   - 运行新建的 `experiments/chXX/XX-YY/run.py` 与 `verify.py` 确保实验 100% 独立可运行；
   - 运行 `python3 scripts/build_site.py && python3 scripts/check_site.py` 验证全站 HTML 构建、锚点跳转与图片光栅化零错误。
5. **Step 5 · 启动 Gemini 4 Argon High 推理强度 Subagent 独立审阅直至收敛**：
   - 调用 `invoke_subagent`（`TypeName: "self"`, `Model: "inherit"`）启动独立审阅 Subagent，对该 Batch 修改的章节正文、数学推导、新增习题/实验和每一张新增配图的 `.png` 视觉效果进行严苛审计；
   - 对 Subagent 提出的任何阻塞项或改进建议逐一修复，并通过 `send_message` 复审（最多 5 轮），直到 Subagent 明确输出 **`Status: Approved / Converged`**。
6. **Step 6 · Git 提交与远端推送部署**：
   - 执行 `git status` 确认无临时文件残留后，按清晰规范的 Commit Message 提交本 Batch 变更并执行 `git push origin main`。
