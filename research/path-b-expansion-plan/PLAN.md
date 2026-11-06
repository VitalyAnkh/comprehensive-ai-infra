# 路径 B 全书旗舰扩充主计划（16 章体系 · 2026 前沿芯片、电力热学、机密计算与全库证据 100% 闭环）

> **版本与状态**：v1.1（已通过 Gemini 4 Argon High Subagent 两轮审阅收敛，`Status: Approved / Converged`）  
> **目标规模**：全书从当前 **14 章（15,068 行 / 543 幅配图 / 175 道习题）** 升级为 **16 章旗舰体系（预计 ~20,280 行 / 615 幅 420pt 出版级配图 / 222 道可复算习题与实验）**  
> **核心准则**：零编造数据（100% 锚定 `references/`、`calculations/`、`experiments/`、`case-studies/`、`research/` 一手锁定真值）、非破坏性增量扩充（严禁破坏既有图号与习题号）、每章分批（8 个 Batch，每 Batch 2 章）执行 **`codex exec` + `gpt-image-2.6-sunburst` (`quality="high"`) 配图生成 + `refine-prose` 去 AI 腔深度润色 + Gemini 4 Argon High Subagent 审阅收敛（含 `.png` 视觉巡检）**。

---

## 一、全书 16 章总览与非破坏性编号不变量（Invariants）

### 1.1 为什么选择“第 1–14 章原位深化 + 第 15、16 章顺延新建”
1. **保护既有链接与交叉引用稳定性**：在路径 A（提交 `c57832bc`）完成后，全书第 1–14 章正文、`case-studies/README.md`（73 篇案例索引）、`experiments/`、`manuscripts/ch01..ch14/figure-index.json`（共 543 幅图）均已建立严密的双向链接。将新增的两大工业级专题放在章尾作为**第 15 章**与**第 16 章**，可确保第 1–14 章所有章号、H2/H3 节号、既有图号和既有习题号 100% 保持不变。
2. **形成更有纵深的三部曲与四大底座收束**：
   - **第一部分：模型与负载（第 1–3 章）**——从第一性原理、2023–2026 Token 成本下降百倍归因框架（`TC-01`–`TC-06`）与经典系统模式映射（`D11-01`–`D11-03`），到模型架构与推理/训练/RL 负载。
   - **第二部分：芯片与系统（第 4–7 章）**——涵盖 2026 最新一代加速器（NVIDIA Vera Rubin / Rubin GPU、Google TPU7x Ironwood、Groq 3 LPX）、算子与编译器运行时、超节点光电互连（CPO 与多轨交换端口预算 `D11-04`–`D11-05`）及数据中心网络（TPOT 反推通信预算与尾部联合验证 `D11-06`）。
   - **第三部分：推理、训练、基础设施底座与跨层协同（第 8–16 章）**——在单实例推理（第 8 章）、分布式推理（第 9 章）、训练系统（第 10 章，补齐 `D14-01`–`D14-03`）、资源调度（第 11 章）与端边云协同（第 12 章）之上，展开四大核心底座与终章闭环：
     - **第 13 章：AI 原生存储系统与数据底座**（由 426 行扩充至 ~1,050 行，补齐向量检索近存卸载、RAG 288 请求实测流水线、多级 KV 缓存池化与列式预训练数据底座）；
     - **第 14 章：模型与系统硬件协同设计**（由 410 行扩充至 ~1,080 行，补齐 Qwen3-8B 三候选方案设计评审、K192 预注册证伪账本、消费级/工作站与数据中心卡经济性分界、以及 2023–2026 Token 成本多因子正交归因总表 `TC-18`）；
     - **第 15 章（新建）：吉瓦级 AI 数据中心、电力热学与跨地域协同训练**（~980 行，覆盖吉瓦级园区供电拓扑、物理/经济可行边界 `D11-08`–`D11-09`、同步训练电网冲击与 Rubin 功率平滑、ASHRAE W45 液冷热力学与 *TAPAS* 热感知路由重配置、跨数据中心广域光网 DCI 与 DiLoCo 低频同步训练）；
     - **第 16 章（新建）：机密 AI 计算、可信执行环境与安全沙箱底座**（~980 行，覆盖多租户/主权 AI 威胁模型、Hopper/Blackwell/Rubin GPU TEE 与 PCIe TDISP/IDE 硬件级机密计算、*PipeLLM* 投机流水线加密、DeepSeek DSec 百万级 Agent 强化学习微虚拟机安全沙箱全解剖、以及可恢复任务审计账本与全书 16 章总收束）。

### 1.2 各章当前基线与路径 B 扩充目标总表

| 章号 | 章节文件 | 当前行数 | 目标行数 | 当前图数 (末图号) | 新增图数 (新图号区间) | 当前习题数 (末题号与格式) | 新增习题数 (新题号区间) | 所属 Batch |
| :-: | --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 01 | [`01-初识 AI Infra.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/01-%E5%88%9D%E8%AF%86%20AI%20Infra.md) | 612 | ~950 | 20 (`图 1-20`) | +2 (`图 1-21`~`1-22`) | 8 (`> **练习 1-8〔延伸〕...**`) | +2 (`练习 1-9`~`1-10`) | Batch 1 |
| 02 | [`02-模型架构.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/02-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84.md) | 1,314 | ~1,430 | 39 (`图 2-39`) | +1 (`图 2-40`) | 13 (`> **练习 2-13〔延伸〕...**`) | +1 (`练习 2-14`) | Batch 1 |
| 03 | [`03-推理与训练负载.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/03-%E6%8E%A8%E7%90%86%E4%B8%8E%E8%AE%AD%E7%BB%83%E8%B4%9F%E8%BD%BD.md) | 936 | ~1,080 | 34 (`图 3-34`) | +1 (`图 3-35`) | 14 (`> **练习 3-14〔延伸〕...**`) | +1 (`练习 3-15`) | Batch 2 |
| 04 | [`04-加速器架构.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/04-%E5%8A%A0%E9%80%9F%E5%99%A8%E6%9E%B6%E6%9E%84.md) | 1,451 | ~1,680 | 45 (`图 4-45`) | +2 (`图 4-46`~`4-47`) | 12 (`> **实验 4-12 · 延伸...**`) | +2 (`实验 4-13`~`4-14`) | Batch 2 |
| 05 | [`05-算子与运行时.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/05-%E7%AE%97%E5%AD%90%E4%B8%8E%E8%BF%90%E8%A1%8C%E6%97%B6.md) | 1,387 | ~1,520 | 43 (`图 5-43`) | +1 (`图 5-44`) | 13 (`> **实验 5-12 · 延伸...**`) | +1 (`实验 5-13`) | Batch 3 |
| 06 | [`06-超节点.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/06-%E8%B6%85%E8%8A%82%E7%82%B9.md) | 1,425 | ~1,600 | 59 (`图 6-59`) | +2 (`图 6-60`~`6-61`) | 15 (`> **实验 6-15 · 延伸...**`) | +1 (`实验 6-16`) | Batch 3 |
| 07 | [`07-数据中心网络.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/07-%E6%95%B0%E6%8D%AE%E4%B8%AD%E5%BF%83%E7%BD%91%E7%BB%9C.md) | 1,489 | ~1,620 | 64 (`图 7-64`) | +1 (`图 7-65`) | 18 (`> **实验 7-18 · 延伸...**`) | +1 (`实验 7-19`) | Batch 4 |
| 08 | [`08-推理优化.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/08-%E6%8E%A8%E7%90%86%E4%BC%98%E5%8C%96.md) | 1,066 | ~1,200 | 39 (`图 8-39`) | +1 (`图 8-40`) | 13 (`> **练习 8-13 · 分析...**`) | +1 (`练习 8-14`) | Batch 4 |
| 09 | [`09-分布式推理.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/09-%E5%88%86%E5%B8%83%E5%BC%8F%E6%8E%A8%E7%90%86.md) | 1,141 | ~1,280 | 49 (`图 9-49`) | +1 (`图 9-50`) | 15 (`> **练习 9-15 · ...**`) | +1 (`练习 9-16`) | Batch 5 |
| 10 | [`10-训练系统.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/10-%E8%AE%AD%E7%BB%83%E7%B3%BB%E7%BB%9F.md) | 1,194 | ~1,360 | 50 (`图 10-50`) | +2 (`图 10-51`~`10-52`) | 14 (`> **习题 10-14 · 分析...**`) | +2 (`习题 10-15`~`10-16`) | Batch 5 |
| 11 | [`11-资源调度与运行环境.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/11-%E8%B5%84%E6%BA%90%E8%B0%83%E5%BA%A6%E4%B8%8E%E8%BF%90%E8%A1%8C%E7%8E%AF%E5%A2%83.md) | 1,152 | ~1,290 | 52 (`图 11-52`) | +1 (`图 11-53`) | 14 (`> **实验 11-14 · 综合设计...**`) | +1 (`实验 11-15`) | Batch 6 |
| 12 | [`12-端边云协同.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/12-%E7%AB%AF%E8%BE%B9%E4%BA%91%E5%8D%8F%E5%90%8C.md) | 1,065 | ~1,180 | 37 (`图 12-37`) | +1 (`图 12-38`) | 14 (`**拓展练习 12-14　...**`) | +1 (`拓展练习 12-15`) | Batch 6 |
| 13 | [`13-AI 原生存储系统与数据底座.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/13-AI%20%E5%8E%9F%E7%94%9F%E5%AD%98%E5%82%A8%E7%B3%BB%E7%BB%9F%E4%B8%8E%E6%95%B0%E6%8D%AE%E5%BA%95%E5%BA%A7.md) | 426 | ~1,050 | 6 (`图 13-6`) | +12 (`图 13-7`~`13-18`) | 6 (`> **实验 13-6：...**`) | +6 (`练习/实验 13-7`~`13-12`) | Batch 7 |
| 14 | [`14-模型与系统硬件协同设计.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/14-%E6%A8%A1%E5%9E%8B%E4%B8%8E%E7%B3%BB%E7%BB%9F%E7%A1%AC%E4%BB%B6%E5%8D%8F%E5%90%8C%E8%AE%BE%E8%AE%A1.md) | 410 | ~1,080 | 6 (`图 14-6`) | +12 (`图 14-7`~`14-18`) | 6 (`> **实验 14-6：...**`) | +6 (`练习/实验 14-7`~`14-12`) | Batch 7 |
| 15 | [`15-吉瓦级 AI 数据中心、电力热学与跨地域协同训练.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/15-%E5%90%89%E7%93%A6%E7%BA%A7%20AI%20%E6%95%B0%E6%8D%AE%E4%B8%AD%E5%BF%83%E3%80%81%E7%94%B5%E5%8A%9B%E7%83%AD%E5%AD%A6%E4%B8%8E%E8%B7%A8%E5%9C%B0%E5%9F%9F%E5%8D%8F%E5%90%8C%E8%AE%AD%E7%BB%83.md) | 新建 | ~980 | 0 | +16 (`图 15-1`~`15-16`) | 0 | +10 (`练习/实验 15-1`~`15-10`) | Batch 8 |
| 16 | [`16-机密 AI 计算、可信执行环境与安全沙箱底座.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/16-%E6%9C%BA%E5%AF%86%20AI%20%E8%AE%A1%E7%AE%97%E3%80%81%E5%8F%AF%E4%BF%A1%E6%89%A7%E8%A1%8C%E7%8E%AF%E5%A2%83%E4%B8%8E%E5%AE%89%E5%85%A8%E6%B2%99%E7%AE%B1%E5%BA%95%E5%BA%A7.md) | 新建 | ~980 | 0 | +16 (`图 16-1`~`16-16`) | 0 | +10 (`练习/实验 16-1`~`16-10`) | Batch 8 |
| **合计** | **16 章正文（不含前言）** | **15,068** | **~20,280** | **543** | **+72（总计 615 幅）** | **175** | **+47（总计 222 道）** | **8 Batches** |

> [!IMPORTANT]
> **非破坏性编号与检验脚本兼容红线**：
> 1. **章内插图严格单调递增追加规则（兼容 `sync_figure_index`、`verify_teaching_book.py` 与 `verify_core_principles.py`）**：[`manuscripts/book_assets.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/book_assets.py) 中的 `sync_figure_index` 与 [`manuscripts/verify_teaching_book.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/verify_teaching_book.py) 强制要求正文中所有图片按阅读顺序满足 `number == f'{chapter}-{i}'`（`1..N` 严格连续递增），且 [`scripts/verify_core_principles.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/scripts/verify_core_principles.py) 第 19–22 行强制要求 `got == assets`。因此，对于第 1–14 章，**所有新增插图 `图 X-(last+1)..` 必须物理嵌入在该章最后一个既有插图 `图 X-last` 所在的节或后续新增节中**；前面小节（如 `1.1.1`、`2.5.3`、`4.6.1`、`5.5.1`、`6.5.1`、`7.6.3`、`8.6.2`、`9.5.1`、`10.3.2`、`11.4.4`、`12.1.5`、`14.1.3`）在正文中直接给出公式/表格并通过“如图 X-Y 所示”前向引用章末综合图，绝不在 `图 X-last` 之前插入新图破坏既有图号。
> 2. **习题格式兼容规则**：第 1–11、13–16 章的新增习题统一采用 `> **练习/实验/习题 X-Y...**` 引用块格式，且保留 [`archive/reviews/core-principles-revision-2026-09-10/before/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/archive/reviews/core-principles-revision-2026-09-10/before) 中的全部历史题号（满足 `verify_core_principles.py` 第 27–28 行 `set(exercise(before)) <= set(exercise(s))`）；第 12 章的新增练习 `12-15` 保持该章特有的 `**拓展练习 12-15　...**` 顶格加粗段落体例。

---

## 二、仓库未深挖一手资产与锁定真值 100% 吸收矩阵

本计划将仓库中已归档但尚未在正文中充分展开的 **7 大类一手研究资产** 100% 转化为正文推导、算例、配图与习题：

### 2.1 `research/token-cost-2023-2026/`（18 个成本归因锚点 `TC-01`–`TC-18`）
依据 [`research/token-cost-2023-2026/report.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/token-cost-2023-2026/report.md) 与 [`outline-placement.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/token-cost-2023-2026/outline-placement.md)，锁定以下四类成本定义与复算真值：
- **四套成本口径**：
  1. 公开 API 3:1 混合标价：$p_{\mathrm{mix}} = \frac{3 p_{\mathrm{in}} + p_{\mathrm{out}}}{4}$；
  2. 服务商满足 SLO 下的单位有效输出成本：$c_{\mathrm{out}} = \frac{K}{Y_{\mathrm{good}} \cdot 3600}$（其中 $K = k_{\mathrm{gpu}} + k_{\mathrm{host}} + k_{\mathrm{net}} + k_{\mathrm{power}} + k_{\mathrm{ops}}$）；
  3. 固定能力门槛与固定工作负载下的最低可用价格：$P^*(q, w, s, t) = \min_{m \in \mathcal{M}(t)} \{ p_m(w, t) \mid Q(m) \ge q,\ S(m, w) \le s \}$；
  4. 完整成功任务成本：$\mathrm{Cost\ per\ successful\ task} = \frac{\mathbb{E}[C_{\mathrm{attempt}}]}{p_{\mathrm{success}}}$。
- **固定能力前沿价格下降倍数（Epoch AI 基准）**：
  - `MMLU >= 64.8`（GPT-3.5 门槛）：2022-11 `text-davinci-003` $p_{\mathrm{mix}}=20.00\ \text{美元}/\text{M tok} \to$ 2023-03 `gpt-3.5-turbo` $2.00\ \text{美元} \to$ 2024-10 `Gemini 1.5 Flash-8B` $0.07\ \text{美元}$（距 2022-11 下降 **$286\times$**，距 2023-03 下降 **$29\times$**）；
  - `MMLU >= 86.0`（GPT-4-0314 门槛）：2023-03 $p_{\mathrm{mix}}=37.50\ \text{美元} \to$ 2025-02 `Gemini 2.0 Flash` $0.18\ \text{美元}$（下降 **$208\times$**）；
  - `GPQA Diamond >= 33`：$37.50\ \text{美元} \to$ 2024-12 `Phi-4` $0.12\ \text{美元}$（下降 **$313\times$**）；
  - `HumanEval >= 67`：$37.50\ \text{美元} \to$ 2024-07 `Llama 3.1 8B` $0.10\ \text{美元}$（下降 **$375\times$**）。
- **逐级条件比率分解（Telescoping Weight-Read Lower Bound）与“四重乘积谬误”**：
  - 单请求 $\text{batch}=1$ 仅看权重读取时间下界（$T_{\mathrm{weight}} = \frac{N_{\mathrm{active}} \cdot b_{\mathrm{w}}}{\mathrm{BW}_{\mathrm{mem}}}$）：
    - $S_0$（70B，`1 byte/param`，H100 $3.35\ \mathrm{TB/s}$）：$70\ \mathrm{GB} / 3.35\ \mathrm{TB/s} = 20.90\ \mathrm{ms}$；
    - $S_1$（8B 小模型或同活跃参数 MoE，`1 byte/param`，H100）：$8\ \mathrm{GB} / 3.35\ \mathrm{TB/s} = 2.39\ \mathrm{ms}$（相对 $S_0$ 缩小 **$8.75\times$**）；
    - $S_2$（8B，`0.5 byte/param` 权重量化，H100）：$4\ \mathrm{GB} / 3.35\ \mathrm{TB/s} = 1.19\ \mathrm{ms}$（相对 $S_1$ 缩小 **$2.00\times$**）；
    - $S_3$（8B，`0.5 byte/param`，换用 H200 $4.8\ \mathrm{TB/s}$）：$4\ \mathrm{GB} / 4.8\ \mathrm{TB/s} = 0.83\ \mathrm{ms}$（相对 $S_2$ 缩小 **$1.43\times$**）；
    - 累积读取下界缩小 $8.75 \times 2.00 \times 1.43 \approx 25.1\times$，但当读取下界降至 $0.83\ \mathrm{ms}$ 时，反量化、KV 访问、集合通信与主机 CPU 调度立即接管关键路径。

| 锚点编号 | 注入章节与节号 | 核心锁定事实与教学任务 |
| :-: | --- | --- |
| `TC-01` | 第 1 章 `1.1.3` / `1.3.4` | 引入 2023–2026 单位 Token 价格下降百倍的现象，区分四种成本定义（`p_mix`、`c_out`、固定能力前沿 $P^*$、成功任务成本），指出不能把单项优化倍率直接相乘。 |
| `TC-02` | 第 2 章 `2.6.2` | 能力密度提升（8B/9B 小模型追平早期 70B/GPT-4 部分基准）如何直接压低 $N_{\mathrm{active}}$ 与部署卡数台阶。 |
| `TC-03` | 第 2 章 `2.5.3` / `2.6.1` | GQA、MLA 与稀疏 MoE 在容量、每步读取量与跨卡通信上的协同减负边界。 |
| `TC-04` | 第 2 章 `2.5.3` / `2.6.3` | MTP（多 Token 预测）与原生投机头在模型层的参数/激活增量，及其与第 8 章运行期验证收支的接口。 |
| `TC-05` | 第 3 章 `3.2.2` | 推理/验证（Reasoning）负载下的成本反转算例：单价降为 $1/10$、单次消耗 Token 涨 $20\times$、成功率从 $50\%$ 升至 $80\%$ 时，单次尝试成本翻倍（$2\times$），成功任务成本变为 $2 \times \frac{0.5}{0.8} = 1.25\times$（反升 $+25\%$）。 |
| `TC-06` | 第 3 章 `3.6.3` | 推理感知的生命周期训练回本线：$N_{\mathrm{break\_even}} = \frac{\Delta C_{\mathrm{train}}}{\Delta c_{\mathrm{request}}}$（多花 $100$ 万美元训练使单次请求省 $0.002$ 美元，需 $5 \times 10^8$ 次请求回本）。 |
| `TC-07` | 第 4 章 `4.3.3` / `4.6.1` | 硬件代际演进（H100 $80\ \mathrm{GB}/3.35\ \mathrm{TB/s} \to$ H200 $141\ \mathrm{GB}/4.8\ \mathrm{TB/s} \to$ B200 $180\ \mathrm{GB}/8\ \mathrm{TB/s} \to$ Rubin $288\ \mathrm{GB}/22\ \mathrm{TB/s}$）如何移动算力墙与容量墙，以及为何不能将硬件倍数与软件倍数重复相乘。 |
| `TC-08` | 第 4 章 `4.2.4` | FP8 / NVFP4 低精度峰值翻倍向真实推理收益转化时的三道折扣：校准质量损失、反量化/缩放因子开销、以及小算子/通信未加速部分。 |
| `TC-09` | 第 5 章 `5.3.3` | FlashAttention-1/2/3 将注意力从 HBM 带宽受限推向 Tensor Core 与 SFU 协同流水线的演进脉络。 |
| `TC-10` | 第 5 章 `5.5.1` | 运行时主机 CPU 开销解耦实证：vLLM v0.6.0（`PR #6883` 异步 API 进程分离 + `PR #7000` `--num-scheduler-steps 10` 多步调度）消除单卡 H100 上 Llama-3 8B 达 $33\%$（API Server）+ $29\%$（Scheduler）的主机开销，吞吐提升 $2.7\times$（70B 提升 $1.8\times$）。 |
| `TC-11` | 第 8 章 `8.2.2` | Orca 连续批处理与 vLLM PagedAttention 将 KV 碎片率从 $60\%\text{–}80\%$ 压至 $<4\%$，通过提升有效并发 $B_{\mathrm{eff}}$ 分摊权重读取。 |
| `TC-12` | 第 8 章 `8.5.4` | 推测解码（Medusa / EAGLE-2 / DFlash）在高并发下的算力—带宽权衡：为何低 batch 下省时间、高 batch 下可能因验证算力争用而退化。 |
| `TC-13` | 第 8 章 `8.6.2` | 引擎与硬件对比的公平基准规范：固定 TTFT / TPOT / ITL 百分位 SLO 下的最大有效输出吞吐（Goodput $Y_{\mathrm{good}}$）。 |
| `TC-14` | 第 9 章 `9.2.3` | 预填充—解码分离（Splitwise / DistServe / Mooncake）消除阶段干扰、允许异构配比与独立批次大小，使满足 SLO 的有效吞吐提升 $1.4\times\text{–}4.48\times$。 |
| `TC-15` | 第 9 章 `9.5.3` | 前缀缓存命中率与分级 KV 池化对输入计费（Cached Input $50\%\text{–}90\%$ 折扣）及底层预填充算力节省的双重映射。 |
| `TC-16` | 第 11 章 `11.4.4` | 公开 API 价格与底层物理成本的背离：Batch API $50\%$ 折扣（填谷消峰）、离峰错峰定价、亏损补贴获客与商业毛利策略。 |
| `TC-17` | 第 11 章 `11.5.2` | 结合前缀缓存、模型路由、重试率与思考预算的完整 Agent 任务成本核算闭环。 |
| `TC-18` | 第 14 章 `14.8.1`–`14.8.2` | 全书成本归因总收束：用逐级条件比率链（Telescoping Attribution）与四重重复计数判据（上限叠加、分子分母混淆、瓶颈转移、阶段错配），严格解释 2023–2026 百倍成本下降的系统工程真相。 |

### 2.2 `research/draft11-outline-audit/` 与 `draft14-15-outline-audit/`（18 个结构与技术锚点）

| 审计编号 | 注入章节与节号 | 核心恢复内容与严格边界 |
| :-: | --- | --- |
| `D11-01` | 第 1 章 `1.1.1` | **传统云 vs. AI 数据中心六维对照表**：可编程性与应用接口、资源组织（一分为多 vs. 多合为一）、调度与公平、故障范围、利用率口径、通信形状。 |
| `D11-02` | 第 1 章 `1.1.2` | **四条跨层耦合导读地图**：算子 $\leftrightarrow$ 片上存储、并行策略 $\leftrightarrow$ 互联、模型架构 $\leftrightarrow$ 推理系统、超节点组织 $\leftrightarrow$ 封装/供电/物理部署。 |
| `D11-03` | 第 1 章 `1.3.4` | **六种经典计算机系统模式向 LLM 系统的映射表**：分层与缓存、重算换存储/搬运、内核旁路与零拷贝、重叠与流水、批量与分摊、解耦与池化。 |
| `D11-04` | 第 6 章 `6.5.1` | **交换端口预算（Radix）与拓扑层级推导**：严格区分交换芯片 radix、单台交换机盒式/框式端口数与整套多轨 Fabric 端口容量，演示给定端点链路数与收敛比下的拓扑级数计算。 |
| `D11-05` | 第 6 章 `6.5.2` | **铜缆（DAC/ACC）、可插拔光模块（Pluggable Optics）与共封装光学（CPO）的三维取舍**：比较连接密度、每比特功耗（$\mathrm{pJ/bit}$）、跨柜距离与现场维护性（结合 Spectrum-X Photonics / Quantum-X CPO 规格）。 |
| `D11-06` | 第 7 章 `7.6.3` | **从 TPOT 反推单次集合通信预算与联合尾部验证矩阵**：给出教学预算分配式 $\frac{f \cdot \mathrm{TPOT}}{L \cdot k}$（如 $L=61, k=2, f=20\%$ 时，$\mathrm{TPOT}=20\ \mathrm{ms} \implies 32.8\ \mu\mathrm{s}$，$\mathrm{TPOT}=5\ \mathrm{ms} \implies 8.2\ \mu\mathrm{s}$），并建立拓扑路径、专家偏斜、rank 就绪偏差与背景竞争流量的四因素正交验证矩阵。 |
| `D11-07` | 第 9 章 `9.5.1` | **多租户前缀缓存安全共享边界与 `cache_salt` 机制**：基于 vLLM 官方前缀缓存设计，量化隔离后命中率下降对 TTFT、显存驻留和成本的连带影响。 |
| `D11-08` | 第 15 章 `15.1.2` | **物理可行性 vs. 经济可行性双边界**：澄清 $P_{\mathrm{domain}} \times \mathrm{GB/W}$ 的分子分母范围一致性条件（防止将含 CPU/交换机/散热损耗的整域功率直接乘以芯片级 $\mathrm{GB/W}$ 高估容量）。 |
| `D11-09` | 第 15 章 `15.1.3` | **站点条件（电价、接入容量、电网改造工期）敏感性分析**：量化电价（如 $0.04$ vs. $0.12\ \text{美元}/\mathrm{kWh}$）、上线延期（折旧侵蚀）与机柜功率上限对并行与副本选型的反转效应。 |
| `D11-SP01` | 第 1 章 `1.4.4` & 第 14 章 `14.8.2` | **跨章逐步修正的双轨推算表（Qwen3-8B Dense 轨 + DeepSeek V4.1 / Qwen3-235B MoE 轨）**：集中展示第 1 章理想下界如何被第 2–16 章逐章加入的真实约束修正。 |
| `D11-SP02` | `00-前言.md` & `README.md` | **面向三类读者（模型/应用、系统/网络、芯片/架构）的 16 章定制阅读路径与必读小节地图**。 |
| `D11-SP03` | 第 1 章 `本章数字速查` & 第 4 章 `4.8` | **参数、公式适用条件与单位口径核查规范**（十进制 $\mathrm{GB}$ vs. 二进制 $\mathrm{GiB}$、单向 vs. 双向带宽、稠密 vs. 稀疏算力、芯片 TDP vs. 整机功率）。 |
| `D11-SP04` | 第 14 章 `14.7.1`–`14.7.2` | **版本化参数与可证伪预测账本规范**：结合实验 `13-07` 的真实冻结时间戳与证伪记录，确立十字段预测登记规范。 |
| `D14-01` | 第 10 章 `10.2.4` | **Megatron 序列并行（SP）与上下文并行（CP）的精确实现边界与激活/通信账本**，并与推理侧长上下文切分对比。 |
| `D14-02` | 第 10 章 `10.3.2` | **DualPipe 双向流水线调度 vs. 1F1B**：基于官方 DualPipe 归档，量化“用双倍模型分片/激活驻留换取流水气泡减半与正反向计算—通信相互掩盖”的收支条件。 |
| `D14-03` | 第 10 章 `10.3.4` | **训练侧路由负载均衡（DeepSeek-V3 无辅助损失动态偏置 + 极小序列级辅助损失）与推理侧 EPLB（固定逻辑路由下的物理专家复制与重排）的本质区别**。 |
| `D14-04` | 第 9 章 `9.6.3` | **大规模 EP 服务中单 Rank 故障的爆炸半径与恢复时间分解**：追踪 DP/TP/EP 通信组耦合范围、存活请求隔离、专家权重重载与积压排空时间线。 |
| `D14-05` | 第 14 章 `14.1.3` | **超越硬件彩票（Hardware Lottery）：系统角色改变如何逆转模型架构价值**（以块扩散模型 DFlash 从主生成模型转为并行投机草稿器为例）。 |

### 2.3 2026 前沿芯片与超节点一手归档参数（NVIDIA Vera Rubin / Google TPU7x / Groq 3 LPX）
- **NVIDIA Rubin GPU 与 Vera Rubin NVL72（[`references/text/nvidia-rubin-arch.txt`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/references/text/nvidia-rubin-arch.txt)、[`nvidia-rubin-system.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/references/files/specs/nvidia-rubin-system.md)）**：
  - **Rubin GPU 芯片规格**：双光罩（Dual Reticle-Sized Dies）通过 NV-HBI 互连，共 $336\text{B}$ 晶体管、$224$ 个 SM、$896$ 个第六代 Tensor Core；单卡推理 NVFP4 稠密峰值达 **$50\ \mathrm{PFLOPS}$**（第三代 Transformer Engine 支持自适应压缩、$K$ 维吞吐翻倍及 3-bit 查找表权重解压）；SFU 指数运算吞吐提升至 FP32 $2\times$ / BF16 $4\times$；显存配备 **$288\ \mathrm{GB}$ HBM4**（12-Hi 堆叠，接口位宽翻倍），峰值带宽高达 **$22\ \mathrm{TB/s}$**（较 Blackwell $8\ \mathrm{TB/s}$ 提升 $2.75\times$）；新增片上 TMEM 注意力得分 2:4 稀疏压缩、MoE 内联 TMA 描述符更新与跨 Kernel 瓦片级（Tile-level）生产者—消费者触发；单卡 **NVLink 6** 双向带宽 **$3,600\ \mathrm{GB/s}$**（支持硬件计数写 Counted Writes 加速 All-to-All 完成通知），NVLink-C2C 带宽 $1,800\ \mathrm{GB/s}$，PCIe Gen 6 带宽 $256\ \mathrm{GB/s}$；全系原生支持 **TEE-I/O 机密计算**。
  - **Vera CPU 规格**：$88$ 个 Olympus 定制 Arm 核（$176$ 线程，Spatial Multi-Threading），最高 **$1.5\ \mathrm{TB}$ SOCAMM LPDDR5X**，内存带宽 **$1.2\ \mathrm{TB/s}$**。
  - **Vera Rubin NVL72 整机柜**：$72$ 颗 Rubin GPU + $36$ 颗 Vera CPU（$18$ 个 Compute Tray + $9$ 个 NVLink 6 Switch Tray），聚合 NVFP4 算力 $3.6\ \mathrm{EFLOPS}$、HBM4 容量 $20.7\ \mathrm{TB}$、HBM4 聚合带宽 $1.58\ \mathrm{PB/s}$、NVLink 6 交换容量 $260\ \mathrm{TB/s}$；配 ConnectX-9 SuperNIC（每 GPU $1.6\ \mathrm{Tb/s}$）、BlueField-4 DPU 与 Spectrum-X Photonics CPO 交换机；支持 **$45^\circ\mathrm{C}$ 温水液冷（ASHRAE W45）**、智能功率平滑（储能 SoC 降低约 $10\%$ 平均功耗、削平约 $20\%$ 的 $50\ \mathrm{ms}$ 脉冲峰值功率）与 DSX MaxLPS（同兆瓦配额下多部署最高 $40\%$ GPU）。
  - **Groq 3 LPX 机柜（可选解码协处理）**：单柜 $256$ 颗 LPU，总计 $128\ \mathrm{GB}$ 片上 SRAM，聚合片上带宽 $40\ \mathrm{PB/s}$，机柜内 Scale-Up 带宽 $640\ \mathrm{TB/s}$。
- **Google TPU7x / Ironwood（[`references/text/google-tpu7x.txt`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/references/text/google-tpu7x.txt)）**：
  - 第七代 TPU，单 Pod 达 **$9,216$ 颗芯片**（3D Torus + OCS 光路交换）；采用**双芯粒（Dual-Chiplet）**架构取代 MegaCore（每个芯粒含 $1$ 个 TensorCore、$2$ 个第三代 SparseCore、$96\ \mathrm{GiB}$ HBM，芯粒间 D2D 互连带宽为 1D ICI 的 $6\times$，在 JAX 中暴露为 2 个逻辑设备）；整颗芯片合计 **$2,307\ \mathrm{TFLOPs}$ BF16 / $4,614\ \mathrm{TFLOPs}$ FP8**、**$192\ \mathrm{GiB}$ HBM**、**$7,380\ \mathrm{GB/s}$（$7.37\ \mathrm{TB/s}$）** HBM 带宽、**$1,200\ \mathrm{GB/s}$** 双向 ICI 互连与 $100\ \mathrm{Gbps}$ DCN；4 芯片主机配备 $224$ vCPU 与 $960\ \mathrm{GB}$ 内存。

### 2.4 `experiments/ch13/13-01/` 与 `13-07/` 真实运行记录与证伪账本
1. **RAG 向量检索 + Qwen3 生成 288 请求实测流水线（[`experiments/ch13/13-01/retrieval-generation/README.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch13/13-01/retrieval-generation/README.md)）**：
   - 语料与索引：`1,024` 篇合成技术文档（`895,296 B`），`all-MiniLM-L6-v2`（384 维，编码耗时 `11.302180 s`）；Faiss `IndexFlatIP`（`1,572,909 B`，构建 `0.000216 s`，$k \in \{1, 4, 16\}$）vs. `IndexHNSWFlat`（$M=8$, `efConstruction=40`, `efSearch` $\in \{4, 16, 64\}$，$k \in \{1, 4, 16\}$，序列化文件 `1,655,730 B`，构建 `0.023254 s`，共 12 个检索配置单元）。
   - 288 请求评测与预注册校准策略失配：全部 288 次请求中 `281/288` 严格通过（7 个不合格均为 `UNKNOWN`，对应检索上下文缺失目标站点文档）；按预注册校准策略（先在 8 题校准集满足 `8/8` 再选最小 $k$），`Flat`、`HNSW ef16` 与 `HNSW ef64` 均在校准集选中 $k=1$（`8/8` 通过），但在 16 题留出集（held-out evaluation）上均因 `q14` 对抗干扰项跌至 **`15/16`（15 题通过、1 题失败，未通过 `16/16` 严格留出集门槛）**；而 `HNSW ef4` 在任意 $k \in \{1, 4, 16\}$ 下校准集仅达 `7/8`（漏 `q05`），无法选出合格候选。
   - 6 个固定检索合格单元的真实 E2E 中位数与 KV 块分配：6 个固定配置单元（`Flat`、`HNSW ef16`、`HNSW ef64` 配合 $k \in \{4, 16\}$）均同时通过校准集 `8/8` 与留出集 `16/16`（`24/24` 全通过）；在 vLLM 预留的 `910` 个 KV 块（`2,146,959,360 B`，每块 `2,359,296 B`）中，$k=4$（输入中位数 `346.5` tokens）每请求峰值占用 `23` 块（`54,263,808 B`），留出集端到端中位数耗时分别为 `Flat k4 = 191.249 ms`、`HNSW ef16 k4 = 117.139 ms`、`HNSW ef64 k4 = 190.284 ms`；$k=16$（输入中位数 `1,201.5–1,202.0` tokens）每请求峰值占用 `77` 块（`181,665,792 B`），留出集端到端中位数耗时分别为 `Flat k16 = 368.577 ms`、`HNSW ef16 k16 = 149.898 ms`、`HNSW ef64 k16 = 200.894 ms`；同时暴露原资源守卫仅追踪带自定义 token 的主进程而漏掉 vLLM `EngineCore` 后代子进程（worker 自身 CUDA allocator 峰值达 `18,582,745,600 B`）的工程教训。
2. **Qwen3-8B 三候选方案系统设计评审（[`experiments/ch13/13-01/design-review/README.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch13/13-01/design-review/README.md)）**：
   - 对比 A（BF16 KV，`28/32` 通过，总耗时 `37.651 s`，长文档 `r0` 失败）、B（FP8 E4M3 KV，`26/32` 通过，总耗时 `38.640 s`）、C（FP8 KV + 查询向量还原为 BF16，`28/32` 通过，总耗时 `37.386 s`，但长文档失败转移至 `r1`）：证明**相同汇总得分（`28/32`）不等于相同故障模式**，且模拟量化包装器并未真正缩小底层 `12,884,115,456 B` 的 BF16 KV 显存池。
3. **K192 抢占恢复预注册条件预测与受控证伪（[`experiments/ch13/13-07/README.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch13/13-07/README.md)）**：
   - 预测清单于 `2026-09-09T07:28:10.963322+00:00` 冻结，早于 K192 执行（`07:30:02.180899+00:00`）；涵盖 30 条执行路径（5 轮 $\times$ 2 任务 $\times$ 3 策略），共 `19,516` 项机制校验、`9,420` 采样词元、`7,500` 唯一提交词元与 `1,920` 重复投递拦截。
   - 因 K96 存在反例（保留前缀窗口 $12.800\ \mathrm{s} >$ 从头重启窗口 $11.619\ \mathrm{s}$，比值 $R=1.101664$），预注册预测保守设定 K192 下 10 个配对单元的中位耗时比 $0.70 \le R \le 1.00$；实测 K192 的 10 个配对比值全数 $<1.0$（最小值 `0.373470`、最大值 `0.950270`），中位数 $R = 0.693141$（恢复更快，但跌破预注册下界 $0.70$），严格判定为 **`refuted_in_scope`**，构成全书方法论最硬核的诚实证伪范本。

### 2.5 工业级案例深剖与顶会锁定算例（`DSec`、`TAPAS`、`PipeLLM`、`I01`–`I21`）
- **DeepSeek DSec 百万级强化学习安全沙箱平台（[`case-studies/dsec-sandbox-platform.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/dsec-sandbox-platform.md)）**：单单元 `160` 台 CPU 节点 / `30,000` 核 / `250 TB` 内存，日创建 `300` 万沙箱、峰值并发 `38` 万、创建吞吐 $>5,000\ \mathrm{s}^{-1}$；支持四类隔离后端（**`FnCall`、运行在 QEMU/libvirt 虚拟机内的容器、`Firecracker microVM`、完整 QEMU VM**）；`virtio-pmem` + DAX 降峰值内存 $40.2\%$（含 $1/64$ `struct page` 元数据开销，即每 $128\ \mathrm{GB}$ 占 $2\ \mathrm{GB}$，瞬时 CPU 峰值由 $26.5\%$ 升至 $41.4\%$），DAMON + `virtio-balloon` 空闲页报告（默认 `2 MiB` 块）降时间积分内存 $21.2\%$，按需 EROFS 将 `8,192` 容器启动从 $>60\ \mathrm{min}$ 缩至 $35\ \mathrm{min}$（磁盘写从 $>1,600\ \mathrm{GB}$ 降至 $700\ \mathrm{GB}$），`SCHED_IDLE` + `PR_SCHED_CORE` 将噪声邻居延迟膨胀从 $+45.2\%$ 压至 $+17.3\%$，并完整记录 Agent 真实逃逸/破坏行为（`XFS_IOC_SWAPEXT`、`/proc/kpagecgroup` 死锁、`chronus` 伪造）。
- **TAPAS 热感知与功耗感知路由重配置（[`research/2026-infra-survey/verify_tapas_reading.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/2026-infra-survey/verify_tapas_reading.py)）**：机柜行功率 `[96+6, 80+6] = [102, 86] kW`（突破 `100 kW` 功率墙），请求延迟 `[0.8, 1.3] s`（突破 `1.0 s` 期限），串行重配置开销 `3 s` 换取单请求省 `0.1 s`，精确回本点为 **30 个请求**。
- **PipeLLM 机密 GPU 投机流水线加密（[`reading-asplos-2025.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/2026-infra-survey/reading-asplos-2025.md)）**：$64\ \mathrm{MiB}$ 张量跨 PCIe 弹跳缓冲加密的流水线周期、三缓冲显存账本与投机预加密命中率边界。
- **21 道 AI Infra 面试与架构设计改编题（[`interview-directions.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/2026-infra-survey/interview-directions.md) `I01`–`I21`）**：全数核对并补入对应章节的正文或章末练习，确保每一道 `I01`–`I21` 在书中都有可复算的完整解答闭环。

---

## 三、逐章扩充与新建章节详细蓝图（第 1 章 – 第 16 章）

### 3.1 第 1–12 章：原位深化、2026 前沿芯片升级与遗留占位全闭环

#### 第 1 章：初识 AI Infra（`612` 行 $\to$ `~950` 行，图 `20 -> 22`，习题 `8 -> 10`）
- **`1.1.1` 扩充（`D11-01`）**：增补“传统云数据中心 vs. AI 数据中心（AI 工厂）”六个维度的系统对比表与定量分析（可编程性接口、资源组织“一分为多 vs. 多合为一”、调度与公平、故障爆炸半径、有效利用率口径、通信流量形状）。
- **`1.1.2` 扩充（`D11-02`）**：增补“四条跨层耦合主线”导读（算子 $\leftrightarrow$ 片上存储、并行策略 $\leftrightarrow$ 互联拓扑、模型架构 $\leftrightarrow$ 推理服务、超节点组织 $\leftrightarrow$ 封装/供电/散热）。
- **`1.1.3` & `1.3.4` 扩充（`TC-01` + `D11-03`）**：
  - 引入 **2023–2026 单位 Token 成本下降百倍的四个核算口径**（公开标价 $p_{\mathrm{mix}}$、服务商单位有效输出成本 $c_{\mathrm{out}}$、固定能力前沿 $P^*(q,w,s,t)$ 与成功任务成本），并用 $S_0 \to S_3$ 逐级条件比率链（$20.90\ \mathrm{ms} \to 2.39\ \mathrm{ms} \to 1.19\ \mathrm{ms} \to 0.83\ \mathrm{ms}$）揭示为何不能将各层加速比简单相乘；
  - 增补**经典计算机系统六大设计模式在 LLM 系统中的映射表**（分层缓存、重算换搬运、内核旁路、重叠流水、批量分摊、解耦池化）。
- **`1.4.4` 与 `本章数字速查` 扩充（`D11-SP01` + `D11-SP03`）**：增加全书双轨贯穿案例（Qwen3-8B Dense 轨与 DeepSeek V4.1 / Qwen3-235B MoE 轨）的逐章约束修正总览表，以及单位口径核查表（$\mathrm{GB}$ vs. $\mathrm{GiB}$、单双向带宽、稠密/稀疏算力）。
- **新增配图（追加于 `1.4.4` 末尾）**：
  - `图 1-21`（`ch01/ch01-cloud-vs-ai-datacenter-and-patterns.svg`）：传统云与 AI 数据中心六维对比及六大经典系统模式向 LLM 基础设施的映射全景；
  - `图 1-22`（`ch01/ch01-token-cost-telescoping-chain.svg`）：2023–2026 固定能力前沿价格下降轨迹与 $S_0 \to S_3$ 权重读取下界逐级折叠（$20.90\ \mathrm{ms} \to 0.83\ \mathrm{ms}$）及主机/通信瓶颈接管示意图。
- **新增习题**：
  - `> **练习 1-9〔核心〕：四重乘积谬误与逐级条件比率分解（TC-01）**`
  - `> **练习 1-10〔延伸〕：修正“80 GB HBM 全扫耗时 24 μs”量纲错误并重算不同活跃参数下的真实带宽下界（D11-01）**`

#### 第 2 章：模型架构（`1,314` 行 $\to$ `~1,430` 行，图 `39 -> 40`，习题 `13 -> 14`）
- **`2.5.3` & `2.6.1`–`2.6.3` 扩充（`TC-02`、`TC-03`、`TC-04`）**：
  - 量化**模型能力密度跃迁**（如 8B/9B 追平早期 70B 基准）对活跃参数 $N_{\mathrm{active}}$、权重读取字节与最小部署卡数台阶的阶跃压缩；
  - 汇总 GQA、MLA、稀疏 MoE 与 MTP（多 Token 预测头）在模型层引入的额外参数/激活开销与推理期每步收益接口。
- **新增配图（追加于 `2.6.3` 末尾）**：
  - `图 2-40`（`ch02/ch02-capability-density-and-kv-pareto.svg`）：2023–2026 模型能力密度演进（等能力下活跃参数与每 Token 状态字节压缩）及 Dense / GQA / MLA / Hybrid 状态帕累托前沿。
- **新增习题**：
  - `> **练习 2-14〔延伸〕：等能力前沿下 70B Dense 向 8B Dense 及 235B-A22B MoE 迁移的显存台阶与带宽交叉点核算（TC-02/TC-03）**`

#### 第 3 章：推理与训练负载（`936` 行 $\to$ `~1,080` 行，图 `34 -> 35`，习题 `14 -> 15`）
- **`3.2.2` 扩充（`TC-05`）**：深入推导推理模型（Reasoning / Test-Time Compute）下的**单价下降与总账单反升悖论**：当单 Token 价格降至 $1/10$、但长思维链与采样使单任务 Token 数膨胀 $20\times$、成功率从 $50\%$ 提升至 $80\%$ 时，单次尝试成本变为 $2\times$，每成功任务成本变为 $2 \times \frac{0.5}{0.8} = 1.25\times$（$+25\%$），导出保持成功任务成本不涨的最大允许 Token 膨胀倍数通式。
- **`3.6.3` 扩充（`TC-06`）**：建立**推理感知的预训练/后训练投入回本方程** $N_{\mathrm{break\_even}} = \frac{\Delta C_{\mathrm{train}}}{\Delta c_{\mathrm{request}}}$，结合蒸馏、过度训练（Over-training）与小模型路由算例说明何时值得在训练侧多烧算力。
- **新增配图（追加于 `3.6.3` 末尾）**：
  - `图 3-35`（`ch03/ch03-reasoning-cost-reversal-and-breakeven.svg`）：测试时扩展（Test-Time Scaling）下单位 Token 降价与成功任务总成本反转曲面，以及训练多投入 $\Delta C_{\mathrm{train}}$ 对累计服务请求数 $N_{\mathrm{req}}$ 的盈亏平衡交点。
- **新增习题**：
  - `> **练习 3-15〔核心〕：长思维链推理模型的成功任务成本反转判据与训练蒸馏回本流量计算（TC-05/TC-06）**`

#### 第 4 章：加速器架构（`1,451` 行 $\to$ `~1,680` 行，图 `45 -> 47`，习题 `12 -> 14`）
- **`4.2.4` & `4.3.3` 扩充（`TC-07`、`TC-08`）**：剖析低精度（FP8 / NVFP4）峰值算力向端到端加速转化时的三道物理折扣（缩放因子/查找表开销、非矩阵算子 SFU 占比、HBM 带宽饱和）。
- **`4.6.1` 扩充（NVIDIA Rubin GPU 架构深剖）**：基于 [`references/text/nvidia-rubin-arch.txt`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/references/text/nvidia-rubin-arch.txt)，系统对比 Hopper（H100/H200）$\to$ Blackwell（B200/B300）$\to$ **Rubin GPU**（双光罩 $336\text{B}$ 晶体管、$224$ SM、$50\ \mathrm{PFLOPS}$ NVFP4、$288\ \mathrm{GB}$ HBM4 @ $22\ \mathrm{TB/s}$、SFU BF16 指数吞吐 $4\times$、TMEM 2:4 稀疏注意力得分压缩、MoE 内联 TMA 描述符更新、瓦片级跨 Kernel 触发与 $3,600\ \mathrm{GB/s}$ NVLink 6）如何针对 MoE 与百万上下文重塑片上配比。
- **`4.7.1` & `4.7.2` 扩充（Google TPU7x Ironwood 与 Groq 3 LPX 对比）**：基于 [`google-tpu7x.txt`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/references/text/google-tpu7x.txt)，解析 TPU7x 为何放弃单块 MegaCore 转向**双芯粒（Dual-Chiplet）**架构（每芯粒 $1$ TensorCore + $2$ SparseCore + $96\ \mathrm{GiB}$ HBM，D2D 带宽达 1D ICI 的 $6\times$，整芯片 $2,307\ \mathrm{TFLOPs}$ BF16 / $4,614\ \mathrm{TFLOPs}$ FP8、$192\ \mathrm{GiB}$ HBM @ $7.37\ \mathrm{TB/s}$、$1,200\ \mathrm{GB/s}$ 双向 ICI），并与 Groq 3 LPX（$256$ LPU、$128\ \mathrm{GB}$ SRAM @ $40\ \mathrm{PB/s}$）对比 HBM 路线与全片上 SRAM 路线的容量—带宽分界。
- **`4.8.2` 扩充（[`case-studies/macbook-rtx-pro6000.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/macbook-rtx-pro6000.md)）**：补充 Apple M2 Max（$96\ \mathrm{GB}$ 统一内存 @ $400\ \mathrm{GB/s}$）与工作站 RTX PRO 6000 Blackwell（$96\ \mathrm{GB}$ GDDR7 @ $1,792\ \mathrm{GB/s}$，$600\ \mathrm{W}$）读取 $4\ \mathrm{GiB}$ 权重的理想服务时间比（$10.74\ \mathrm{ms}$ vs. $2.40\ \mathrm{ms}$，比值 $4.48\times$）及各自适用边界。
- **新增配图（追加于 `4.8.4` 末尾）**：
  - `图 4-46`（`ch04/ch04-hopper-blackwell-rubin-scaling.svg`）：NVIDIA H100 $\to$ B200 $\to$ Rubin GPU 三代芯片的算力（BF16/FP8/NVFP4）、HBM 容量与带宽（$3.35 \to 8.0 \to 22.0\ \mathrm{TB/s}$）、SFU 指数吞吐及片上协作机制演进对照；
  - `图 4-47`（`ch04/ch04-tpu7x-dual-chiplet-vs-sram-asic.svg`）：Google TPU7x（Ironwood）双芯粒（Dual-Chiplet + SparseCore Gen3 + $192\ \mathrm{GiB}$ HBM）拓扑与 Apple M2 Max / RTX PRO 6000 / Groq 3 LPX 存储带宽阶梯对比。
- **新增习题**：
  - `> **实验 4-13 · 核心：Rubin GPU（22 TB/s HBM4, 50 PFLOPS NVFP4）与 B200 在长上下文 MoE Decode 下的算数强度嵴点迁移核算**`
  - `> **实验 4-14 · 延伸：TPU7x 双芯粒（Dual-Chiplet）D2D 互连与 1D ICI 带宽分级下的张量切分映射推导**`

#### 第 5 章：算子与运行时（`1,387` 行 $\to$ `~1,520` 行，图 `43 -> 44`，习题 `13 -> 14`）
- **`5.3.3` & `5.5.1` 扩充（`TC-09`、`TC-10`、`I01`、`I08`、`I20`）**：
  - 结合 [`research/token-cost-2023-2026/report.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/token-cost-2023-2026/report.md)，详述 **vLLM v0.6.0 主机运行时重构实证**：在单张 H100 运行 Llama-3 8B 时，旧同步架构中 API Server 占 $33\%$、Scheduler 占 $29\%$、GPU 执行仅占 $38\%$；通过 `PR #6883`（ZeroMQ 多进程异步解耦 API 与 EngineCore）和 `PR #7000`（`--num-scheduler-steps 10` 一次调度推进 10 步并采用异步输出拷贝），将 Llama-3 8B 吞吐提升 **$2.7\times$**、70B 提升 **$1.8\times$**；
  - 结合 [`case-studies/macbook-rtx-pro6000.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/macbook-rtx-pro6000.md)，说明 Ollama（`v0.20.7`）在 Apple Metal 统一内存下的 `newBufferWithBytesNoCopy` 零拷贝路径与量化内核调度边界。
- **新增配图（追加于 `5.6.3` 末尾）**：
  - `图 5-44`（`ch05/ch05-vllm-v060-multiprocess-and-multistep.svg`）：vLLM v0.6.0 主机运行时时间线拆解——从单进程 GIL 串行气泡（API $33\%$ + 调度 $29\%$）到多进程异步流水 + 10 步批量调度的 $2.7\times$ 吞吐跃升。
- **新增习题**：
  - `> **实验 5-13 · 核心：多步调度（Multi-Step Scheduling）的 GPU 空泡消除收益与异步输出步间滞后边界复算（TC-10 / I01）**`

#### 第 6 章：超节点（`1,425` 行 $\to$ `~1,600` 行，图 `59 -> 61`，习题 `15 -> 16`）
- **`6.5.1` 扩充（`D11-04`）**：补齐**交换芯片 Radix、交换机整机端口与多轨（Multi-Rail）Fabric 级数推导**，区分物理交换跳数、集合通信算法轮次与消息分块次数，阐明为何不能仅由“节点数 $>$ 单芯片端口数”直接断言多轨超节点的拓扑层数。
- **`6.5.2` & `6.5.3` & `6.5.4` 扩充（`D11-05` + Vera Rubin NVL72 + TPU7x 9216-chip 3D Torus）**：
  - 系统对比**无源/有源铜缆（DAC/ACC）、可插拔光模块（Pluggable Optics）与共封装光学（CPO，如 NVIDIA Spectrum-X Photonics / Quantum-X Photonics）**在每比特能耗（$\mathrm{pJ/bit}$）、端口密度、机柜内/跨机柜可达距离与故障隔离维护上的工程取舍；
  - 补充 **Vera Rubin NVL72**（$72\times 3.6\ \mathrm{TB/s} = 260\ \mathrm{TB/s}$ NVLink 6 全互连 + NVLink 硬件计数写 Counted Writes）与 **TPU7x $9,216$ 芯片 3D Torus + OCS 动态光交换**超节点对照。
- **新增配图（追加于 `6.7.4` 末尾）**：
  - `图 6-60`（`ch06/ch06-switch-radix-and-cpo-tradeoff.svg`）：交换芯片 Radix 到多轨 Fabric 的端口预算映射，以及铜缆（DAC）、可插拔光模块与共封装光学（CPO）的距离—功耗—密度三维取舍；
  - `图 6-61`（`ch06/ch06-vera-rubin-nvl72-vs-tpu7x-pod.svg`）：Vera Rubin NVL72（$260\ \mathrm{TB/s}$ NVLink 6 全互连机柜）与 Google TPU7x（$9,216$ 芯片 3D Torus + OCS 光路重配置）超节点架构对比。
- **新增习题**：
  - `> **实验 6-16 · 核心：给定交换芯片 Radix 与端点多轨网卡数的无阻塞两层/三层超节点规模推导及 CPO 光电功耗对比（D11-04/D11-05）**`

#### 第 7 章：数据中心网络（`1,489` 行 $\to$ `~1,620` 行，图 `64 -> 65`，习题 `18 -> 19`）
- **`7.6.3` 扩充（`D11-06` + `I12`）**：
  - 完整增补**从端到端 TPOT 反推单次集合通信预算的教学模型**：若单步 Decode 有 $L$ 个串行层、每层 $k$ 次不可隐藏的关键集合通信、分配给通信的预算占比为 $f$，则单次集合通信平均时间预算为 $\frac{f \cdot \mathrm{TPOT}}{L \cdot k}$（以 $L=61, k=2, f=20\%$ 为例：当 $\mathrm{TPOT}=20\ \mathrm{ms}$ 时预算仅 **$32.8\ \mu\mathrm{s}$**；当 $\mathrm{TPOT}=5\ \mathrm{ms}$ 时骤降至 **$8.2\ \mu\mathrm{s}$**）；
  - 建立 **MoE Decode All-to-All 尾延迟四因素联合验证矩阵**（拓扑与路径选择、专家负载偏斜、各 Rank 算子完成就绪偏差、共享队列竞争流量），阐明为何集合操作完成时间受最慢依赖 $\max_i(t_{\mathrm{ready},i} + t_{\mathrm{xfer},i})$ 支配而非“单消息 P99 乘以几百”。
- **新增配图（追加于 `7.6.5` 末尾）**：
  - `图 7-65`（`ch07/ch07-tpot-collective-budget-and-tail-matrix.svg`）：从 TPOT（$20\ \mathrm{ms} \to 5\ \mathrm{ms}$）反推每层 All-to-All 微秒级预算（$32.8\ \mu\mathrm{s} \to 8.2\ \mu\mathrm{s}$）及 Rank 就绪偏差与网络排队联合尾部验证矩阵。
- **新增习题**：
  - `> **实验 7-19 · 核心：61 层 MoE 模型下从 TPOT 反推 All-to-All 预算及 Rank 就绪偏差（Jitter）与交换机排队尾部的归因分离（D11-06 / I12）**`

#### 第 8 章：推理优化（`1,066` 行 $\to$ `~1,200` 行，图 `39 -> 40`，习题 `13 -> 14`）
- **`8.2.2`、`8.5.4`、`8.6.2` 扩充（`TC-11`、`TC-12`、`TC-13`、`I03`、`I10`）**：
  - 系统串联连续批处理與 PagedAttention（`TC-11`）、高并发下推测解码的算力回退阈值（`TC-12`），以及**满足双指标百分位 SLO（如 $\mathrm{P99\ TTFT} \le s_1,\ \mathrm{P99\ TPOT} \le s_2$）下的有效输出吞吐 $Y_{\mathrm{good}}$ 与单位 Token 成本 $c_{\mathrm{out}} = \frac{K}{Y_{\mathrm{good}} \cdot 3600}$ 曲线（`TC-13`）**。
- **新增配图（追加于 `8.6.3` 末尾）**：
  - `图 8-40`（`ch08/ch08-goodput-slo-frontier-and-unit-cost.svg`）：并发度增长下原始吞吐（Raw Throughput）与满足 TTFT/TPOT 百分位 SLO 的有效吞吐（Goodput $Y_{\mathrm{good}}$）背离曲线及最低单位 Token 成本工作点。
- **新增习题**：
  - `> **练习 8-14 · 核心：给定 TTFT 与 TPOT 双重 SLO 约束下的有效吞吐（Goodput）封顶与单位百万 Token 服务成本推导（TC-11/TC-12/TC-13）**`

#### 第 9 章：分布式推理（`1,141` 行 $\to$ `~1,280` 行，图 `49 -> 50`，习题 `15 -> 16`）
- **`9.2.3` & `9.5.3` 扩充（`TC-14`、`TC-15`）**：量化 PD 分离消除阶段干扰后将有效吞吐提升 $1.4\times\text{–}4.48\times$ 的收支边界，以及跨实例 KV 池化对 Cached Input 计费折扣的支撑。
- **`9.5.1` 扩充（`D11-07`）**：新增**多租户前缀缓存的安全隔离边界与 `cache_salt` 机制**（依据 vLLM 官方前缀缓存设计文档）：为何相同 Token 序列在不同租户/权限组间不能盲目共享 KV，以及引入租户命名空间盐值（`cache_salt`）后对全局命中率、TTFT 与显存容量的定量影响。
- **`9.6.3` 扩充（`D14-04`）**：新增**大规模专家并行（EP）服务中单 Rank 故障的爆炸半径与恢复时间分解**：对比紧密耦合的 All-to-All EP 组与独立 DP 副本在单卡掉线时的受影响请求比例、通信组重建、冗余专家顶替与积压队列排空时间线。
- **新增配图（追加于 `9.7.3` 末尾）**：
  - `图 9-50`（`ch09/ch09-tenant-cache-salt-and-ep-fault-radius.svg`）：多租户 `cache_salt` 前缀缓存隔离边界与大规模 EP 服务单 Rank 故障的通信组爆炸半径及降级恢复时间线。
- **新增习题**：
  - `> **练习 9-16 · 多租户前缀缓存 `cache_salt` 隔离后的命中率折损与大 EP 单 Rank 故障积压排空时间推导（D11-07 / D14-04）**`

#### 第 10 章：训练系统（`1,194` 行 $\to$ `~1,360` 行，图 `50 -> 52`，习题 `14 -> 16`）
- **`10.2.4` 扩充（`D14-01`）**：基于 Megatron-Core 官方文档，严格区分**序列并行（SP，切分 LayerNorm/Dropout 等非 TP 区的序列激活）与上下文并行（CP，沿序列维切分全部激活并通过 Ring/All-Gather 交换注意力 KV）**的显存峰值与跨卡通信账本。
- **`10.3.2` 扩充（`D14-02`）**：基于 DeepSeek 官方 [`DualPipe`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/draft14-15-outline-audit/sources/dualpipe-readme.md) 归档，深入对比 **1F1B 与 DualPipe 双向流水线调度**：如何通过双向同时注入正反向微批次，将流水线气泡从 $(P-1)$ 减半至 $(P/2 - 1)$ 并实现细粒度计算—通信相互掩盖，同时付出双份模型分片驻留与额外激活显存的精确代价。
- **`10.3.4` 扩充（`D14-03`）**：基于 [`deepseek-v3.txt`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/draft14-15-outline-audit/sources/deepseek-v3.txt) §2.1.2，严格辨析**训练侧专家负载均衡（动态路由偏置项更新 + 极小系数序列级辅助损失，直接改变前向选中哪些专家）与推理侧 EPLB（保持逻辑 top-$k$ 路由不变，仅做物理专家复制与重排）**的本质差异。
- **新增配图（追加于 `10.6.4` 末尾）**：
  - `图 10-51`（`ch10/ch10-dualpipe-vs-1f1b-schedule.svg`）：1F1B 与 DualPipe 双向流水线调度的气泡减半机制、正反向计算—通信重叠时序及显存驻留代价对照；
  - `图 10-52`（`ch10/ch10-training-bias-balance-vs-inference-eplb.svg`）：训练侧无辅助损失动态路由偏置（改变逻辑专家选择）与推理侧 EPLB（固定逻辑路由下的物理副本复制与搬移）的双轨机制对比。
- **新增习题**：
  - `> **习题 10-15 · 核心：1F1B 与 DualPipe 双向流水线的气泡率、激活驻留峰值与权重副本显存收支对比（D14-02）**`
  - `> **习题 10-16 · 分析：训练侧路由偏置均衡与推理侧 EPLB 的作用边界及 Megatron SP/CP 通信量复算（D14-01 / D14-03）**`

#### 第 11 章：资源调度与运行环境（`1,152` 行 $\to$ `~1,290` 行，图 `52 -> 53`，习题 `14 -> 15`）
- **`11.4.4` & `11.5.2` 扩充（`TC-16`、`TC-17`、`I11`、`I21`）**：
  - 深入剖析**公开 API 价格与底层服务商物理成本的映射与背离（`TC-16`）**：如何用夜间/闲时算力承载 Batch API（$50\%$ 折扣）以提升集群昼夜均摊利用率，以及前缀缓存折扣（$50\%\text{–}90\%$）与模型路由（`I21`：$84.93\%$ 成本交点与 6 秒时限下 $91.84\%$ 命中率门槛）如何共同决定完整 Agent 任务的期望账单（`TC-17`）。
- **新增配图（追加于 `11.5.3` 末尾）**：
  - `图 11-53`（`ch11/ch11-api-pricing-vs-provider-cost-and-routing.svg`）：服务商昼夜负载波谷与 Batch API 填谷经济学，以及双模型路由在成本交点（$84.93\%$）与 6 秒 SLO 交点（$91.84\%$）下的可行域切换。
- **新增习题**：
  - `> **实验 11-15 · 综合设计：昼夜潮汐流量下在线服务与半价 Batch API 混部配比及双模型缓存路由阈值复算（TC-16/TC-17/I21）**`

#### 第 12 章：端边云协同（`1,065` 行 $\to$ `~1,180` 行，图 `37 -> 38`，习题 `14 -> 15`）
- **`12.1.5` & `12.5.2` 扩充（[`case-studies/macbook-rtx-pro6000.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/macbook-rtx-pro6000.md)）**：
  - 补齐本地统一内存（MacBook Pro M2 Max $96\ \mathrm{GB}$ @ $400\ \mathrm{GB/s}$）、本地独立工作站（RTX PRO 6000 Blackwell $96\ \mathrm{GB}$ GDDR7 @ $1,792\ \mathrm{GB/s}$）与远端云服务三档方案在阶段延迟（Prefill vs. Decode）、可用模型容量上限（`recommendedMaxWorkingSetSize` vs. 独占显存）、隐私不出域与每任务整机能耗上的完整对比实验设计与决策判据。
- **新增配图（追加于 `12.5.4` 末尾）**：
  - `图 12-38`（`ch12/ch12-m2max-rtxpro6000-cloud-three-tier.svg`）：端侧统一内存（M2 Max 96GB）、桌面工作站（RTX PRO 6000 96GB GDDR7）与云端 H100 在不同上下文长度与并发下的延迟—容量—隐私三维决策边界。
- **新增习题（保持第 12 章顶格加粗格式）**：
  - `**拓展练习 12-15　96 GB 统一内存（M2 Max）与 96 GB 独立显存（RTX PRO 6000）在单请求长上下文与多请求并发下的延迟比及工作集溢出边界推导〔核心，★★〕。**`

---

### 3.2 第 13 章重磅扩充：AI 原生存储系统与数据底座（`426` 行 $\to$ `~1,050` 行，图 `6 -> 18`，习题 `6 -> 12`）

第 13 章目前仅 426 行、6 幅图、6 道题，远低于前 12 章平均体量（~1,185 行 / 43 幅图）。路径 B 将其扩充为约 **1,050 行、18 幅 420pt 出版级配图、12 道习题与实验**的完整深度章节，在保留既有 `13.1`–`13.5` 全部文本、图 `13-1`–`13-6` 与练习 `13-1`–`13-6` 的基础上，深化 `13.5` 并新增 `13.6` 与 `13.7` 两大核心节：

#### 1. 原有 `13.5` 节深化：列式预训练数据格式与万卡容器镜像按需加载
- **新增 `13.5.3`：列式存储格式（Parquet / Lance）随机跳转、*Youmu* 多模态流水线配比与按需 EROFS 镜像底座**：
  - 对比 Tar/WebDataset（顺序流式读取极快，但缺乏行级随机寻址与多模态按列投影能力）与 Lance / Parquet（支持列裁剪、按长度分桶采样与毫秒级随机 offset 寻址）在课程学习（Curriculum Learning）与长短序列混合打包中的 I/O 账本；
  - 融入 [`case-studies/dsec-sandbox-platform.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/dsec-sandbox-platform.md) 的生产级存储实证：当 `8,192` 个评测或训练容器并发启动且单镜像达 `20 GB` 时，全量拉取解包耗时 $>60\ \mathrm{min}$、每节点写入 $>1,600\ \mathrm{GB}$；改用 **只读 EROFS 块设备镜像 + 本地 SSD 页缓存按需拉取（On-demand Read）+ 内存 `overlayfs` 预构建索引（将复杂度从 $O(mN)$ 降至 $O(m+k)$）** 后，`8,192` 个容器启动缩短至 **`35 min`**，每节点磁盘写入降至 **`700 GB`**。

#### 2. 新增 `13.6` 节：向量检索存储引擎、近存卸载与 RAG 端到端数据底座
- **`13.6.1` 精确内积扫描（Faiss `IndexFlatIP`）、HNSW 图索引内存膨胀与 DiskANN 闪存受限检索**：
  - 完整落地 [`case-studies/retrieval-and-generation.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/retrieval-and-generation.md) 锁定算例：在 $N = 2^{25} = 33,554,432$ 条向量、维度 $d = 768$ 下，Faiss `IndexFlatIP` 精确存 FP32 占 $4d = 3,072\ \mathrm{B}$/向量，总计 **$96\ \mathrm{GiB}$**；若存为 FP16（`IndexFlatIP` 转换或自定义半精度表），每向量 $1,536\ \mathrm{B}$，总计 **$48\ \mathrm{GiB}$（$51.540\ \mathrm{GB}$）**；
  - 单查询全扫计算量 $2Nd = 51.540\ \mathrm{GFLOPs}$，读取 $51.540\ \mathrm{GB}$，**算术强度仅 $1\ \mathrm{FLOP/B}$**（当 $B=16$ 查询合批时升至 $16\ \mathrm{FLOP/B}$）；若经 $64\ \mathrm{GB/s}$ 主机链路把 $48\ \mathrm{GiB}$ 搬入加速器，仅传输下界即达 **$805.306\ \mathrm{ms}$**；而放到具备 $1\ \mathrm{TB/s}$ 内部聚合带宽的近存检索模块（IKS）本地扫描，读取下界降至 **$51.540\ \mathrm{ms}$**（8 分片广播查询仅 $8 \times 1,536 = 12,288\ \mathrm{B}$，每分片返回 $k=32$ 个 `4-byte score + 8-byte id = 12 B` 结果对，8 分片合计返回仅 **$8 \times 32 \times 12 = 3,072\ \mathrm{B}$**）；
  - 剖析 HNSW 图索引的邻接表内存开销（每向量额外 $2M \times 4\ \mathrm{B}$ 边指针）与 DiskANN / Starling 将压缩 PQ 向量放 DRAM、完整向量与邻接块对齐 $4\ \mathrm{KiB}$ NVMe LBA 扇区以压低 SSD 随机读轮次的机制。
- **`13.6.2` 真实 288 请求 Faiss + Qwen3 RAG 流水线实验与预注册校准失配审计（[`experiments/ch13/13-01/retrieval-generation/README.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch13/13-01/retrieval-generation/README.md)）**：
  - 完整呈现 `1,024` 篇技术文档（`895,296 B`）、`all-MiniLM-L6-v2`（384 维，编码耗时 `11.302180 s`）、Faiss `IndexFlatIP`（`1,572,909 B`，构建 `0.000216 s`）与 `IndexHNSWFlat`（$M=8$, `efConstruction=40`, `efSearch` $\in \{4, 16, 64\}$，序列化文件 `1,655,730 B`，构建 `0.023254 s`）在 Qwen3-8B vLLM 引擎上的 **288 次端到端请求实测（`281/288` 严格通过）**；
  - 剖析**为何预注册验证集校准规则会未通过留出集检验**：在 8 题校准集上，`Flat`、`HNSW ef16`、`HNSW ef64` 在 $k=1$ 时即可拿到 `8/8` 满分，但冻结 $k=1$ 进入 16 题留出集时均在 `q14` 失守、跌至 **`15/16`（15 题通过、1 题失败，未达 `16/16` 留出门槛）**，而 `HNSW ef4` 在校准集仅达 `7/8`（漏 `q05`）无法选出合格候选；反观固定 `Flat`/`HNSW ef16`/`HNSW ef64` $\times\ k \in \{4, 16\}$ 的 6 个实验单元在校准集 `8/8` 与留出集 `16/16`（合计 `24/24`）全部通过；
  - 核算 $k$ 值扩大对下游 vLLM KV 缓存池（总池 `910` 块 = `2,146,959,360 B`）与端到端延迟的真实影响：$k=4$（输入中位数 `346.5` tokens）峰值占用 `23` 块（`54,263,808 B`，留出集中位数耗时 `Flat = 191.249 ms`、`HNSW ef16 = 117.139 ms`、`HNSW ef64 = 190.284 ms`），$k=16$（输入中位数 `1,201.5–1,202.0` tokens）峰值升至 `77` 块（`181,665,792 B`，留出集中位数耗时 `Flat = 368.577 ms`、`HNSW ef16 = 149.898 ms`、`HNSW ef64 = 200.894 ms`）；并剖析原资源守卫仅追踪带自定义 token 的进程而漏掉 vLLM `EngineCore` 子进程（worker CUDA allocator 峰值达 `18,582,745,600 B`）的多进程观测缺口。

#### 3. 新增 `13.7` 节：多级 KV 缓存池化（HBM–DRAM–SSD）、非前缀融合与闪存耐久性边界
- **`13.7.1` 跨节点三级 KV 缓存池（Mooncake Store / LMCache / SGLang HiCache）的吞吐与命中率盈亏方程**：
  - 推导从 HBM $\to$ 主机 DRAM（PCIe Gen5 $\sim 64\ \mathrm{GB/s}$）$\to$ 远端 RDMA DRAM 池（$400\ \mathrm{Gbps} \sim 50\ \mathrm{GB/s}$）$\to$ 本地/远端 NVMe SSD 池（$7\text{–}14\ \mathrm{GB/s}$）拉取 KV 块替重算 Prefill 的临界上下文长度 $S_{\mathrm{crit}}$；
  - 解析 RAG 多文档拼接场景下的**非前缀 KV 复用（如 CacheBlend）**：当多个检索文档顺序任意组合导致位置编码（RoPE）与跨文档注意力失配时，如何仅对占 $10\%\text{–}15\%$ 的高敏感层/词元做增量重算，兼容 $85\%$ 以上的存储块复用。
- **`13.7.2` 持续 KV 换出下的企业级 SSD 写入寿命（DWPD）核算与准入节流**：
  - 给出定量算例：单台 8 卡推理节点若每秒持续向 4 块 $3.84\ \mathrm{TB}$ NVMe SSD 换出 $4\ \mathrm{GB/s}$ 新生成 KV 状态，单日写入量达 $4\ \mathrm{GB/s} \times 86,400\ \mathrm{s} = 345.6\ \mathrm{TB/day}$；计入写放大 $\mathrm{WAF}=1.5$ 后物理 NAND 写入达 $518.4\ \mathrm{TB/day}$，相当于每块盘每天全盘写入 **$33.75\ \mathrm{DWPD}$**，远超普通企业级 TLC SSD（$1\text{–}3\ \mathrm{DWPD}$）耐久上限！据此推导**只写高复用率前缀（Write-Admission Filter）、DRAM 写缓冲合并与高耐久 SLC/ZNS 分层**的必要性。

#### 4. 第 13 章新增 12 幅 420pt 配图（`图 13-7` ~ `图 13-18`）与 6 道新习题（`13-7` ~ `13-12`）
- **新增配图清单**：
  - `图 13-7`（`ch13/ch13-parquet-lance-vs-webdataset-io.svg`，置于 `13.5.3`）：WebDataset 顺序流与 Lance/Parquet 列式随机偏移寻址在课程学习与多模态投影下的 I/O 对比；
  - `图 13-8`（`ch13/ch13-dsec-erofs-ondemand-container-boot.svg`，置于 `13.5.3`）：DSec 平台 8,192 个容器并发启动——全量镜像拉取（$>60\ \mathrm{min}$, $>1,600\ \mathrm{GB}$ 写）vs. EROFS 按需块读取 + 内存 `overlayfs`（$35\ \mathrm{min}$, $700\ \mathrm{GB}$ 写）；
  - `图 13-9`（`ch13/ch13-faiss-flat-vs-iks-near-storage.svg`，置于 `13.6.1`）：$2^{25}\times 768$ FP16 向量表（$48\ \mathrm{GiB}$）经主机链路搬移（$805.3\ \mathrm{ms}$）vs. 8 分片近存 IKS 本地扫描（$51.5\ \mathrm{ms}$）的数据流与字节账本；
  - `图 13-10`（`ch13/ch13-hnsw-memory-vs-diskann-ssd-layout.svg`，置于 `13.6.1`）：HNSW 内存邻接图膨胀与 DiskANN 压缩 PQ 内存导航 + $4\ \mathrm{KiB}$ NVMe 扇区对齐贴合布局对比；
  - `图 13-11`（`ch13/ch13-rag-288-requests-calibration-trap.svg`，置于 `13.6.2`）：实验 `13-01` 的 288 请求 RAG 评测矩阵——校准集预选 $k=1$（`8/8`）在留出集失守（`15/16`，漏 `q14`）vs. 固定 $k=4/16$ 六个配置单元全通过（`24/24`）的分布偏移解剖；
  - `图 13-12`（`ch13/ch13-rag-topk-kv-blocks-and-rlimit-gap.svg`，置于 `13.6.2`）：检索 $k=1/4/16$ 对 vLLM 910 块 KV 池的占用（$23$ 块 vs. $77$ 块）及原资源守卫未覆盖 `EngineCore` 后代子进程（CUDA allocator 峰值 $18.58\ \mathrm{GB}$）的进程树边界；
  - `图 13-13`（`ch13/ch13-rag-end-to-end-timeline-breakdown.svg`，置于 `13.6.2`）：RAG 请求端到端关键路径分解（查询编码、向量检索、块拉取、Prefill 扩展与 Decode 步耗时随 $k$ 值的移动）；
  - `图 13-14`（`ch13/ch13-three-tier-kv-cache-hierarchy.svg`，置于 `13.7.1`）：HBM–主机 DRAM–RDMA 远端内存–NVMe SSD 多级 KV 缓存池（Mooncake / HiCache）的带宽阶梯与重算交点；
  - `图 13-15`（`ch13/ch13-cacheblend-non-prefix-kv-fusion.svg`，置于 `13.7.1`）：RAG 多文档非前缀拼接下全量重算、直接拼接失真与选择性稀疏重算（CacheBlend）的精度—延迟折中；
  - `图 13-16`（`ch13/ch13-ssd-dwpd-endurance-and-admission.svg`，置于 `13.7.2`）：KV 缓存高频换出下的 NVMe SSD 日写入量（$33.75\ \mathrm{DWPD}$ 寿命超限）与基于复用频次的写准入过滤器（Admission Filter）；
  - `图 13-17`（`ch13/ch13-storage-diagnosis-workflow.svg`，置于 `13.7.2`）：AI 存储子系统瓶颈五步定位图（元数据 IOPS、PCIe Root Complex 拷贝、NAND 读放大、GC 写尾延迟、多模态 CPU 解码饥饿）；
  - `图 13-18`（`ch13/ch13-full-stack-ai-data-plane.svg`，置于 `本章小结` 前）：贯穿预训练数据湖、向量知识库、3FS 检查点与多级 KV 缓存池的 AI 原生数据底座全景架构图。
- **新增习题与实验（`13-7` ~ `13-12`）**：
  - `> **练习 13-7：核算 $2^{25}\times 768$ 向量库在主机搬移与 8 分片近存 IKS 下的延迟比及批大小 $B$ 对算术强度的改变。**`
  - `> **练习 13-8：基于实验 13-01 数据，计算 RAG 检索从 $k=4$ 扩至 $k=16$ 时的 KV 块增量、显存占用与最大并发请求数折损。**`
  - `> **练习 13-9：推导多级 KV 缓存（DRAM / NVMe SSD）拉取耗时小于重算 Prefill 耗时的临界上下文长度 $S_{\mathrm{crit}}$。**`
  - `> **练习 13-10：计算 8 卡推理节点持续换出 KV 状态时的 SSD 日写入量（DWPD），并设计使寿命达标（$\le 3\ \mathrm{DWPD}$）的准入率上限。**`
  - `> **练习 13-11：对比全量镜像拉取与 DSec 按需 EROFS + 内存 `overlayfs` 在万级容器并发启动下的网络流量与 I/O 复杂度。**`
  - `> **实验 13-12：复现实验 `13-01`（`retrieval-generation`），验证不同检索 $k$ 值与 `EngineCore` 子进程显存/内存守卫边界。**`

---

### 3.3 第 14 章重磅扩充：模型与系统硬件协同设计（`410` 行 $\to$ `~1,080` 行，图 `6 -> 18`，习题 `6 -> 12`）

在保留第 14 章既有 `14.1`–`14.5` 全部文本、图 `14-1`–`14-6` 与练习 `14-1`–`14-6` 的基础上，深化 `14.1` 与 `14.5`，并新增 `14.6`（三候选方案设计评审与消费级/数据中心卡选型分界）、`14.7`（K192 预注册条件预测与受控证伪账本）与 `14.8`（2023–2026 Token 成本下降百倍的多因子正交归因总表 `TC-18`）：

#### 1. 原有 `14.1` 节深化（`D14-05` + 超越 Chinchilla 的推理感知训练分配）
- **新增 `14.1.3`：硬件彩票（Hardware Lottery）、系统角色转换（DFlash）与全生命周期算力分配方程**：
  - 引入 [`research/draft14-15-outline-audit/snapshots/13-架构协同设计.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/draft14-15-outline-audit/snapshots/13-%E6%9E%B6%E6%9E%84%E5%8D%8F%E5%90%8C%E8%AE%BE%E8%AE%A1.md) 与 `D14-05`：阐述为何脱离底层硬件谈“算法孤立优劣”会落入硬件彩票陷阱，同时指出**即便硬件不变，改变模型在系统中的角色（例如块扩散模型 DFlash 作为主生成模型受限于质量，但作为自回归主模型的并行投机草稿器却能一次前向起草多个位置）也会彻底逆转其工程价值**；
  - 联立预训练成本 $C_{\mathrm{train}}(N, D_{\mathrm{tr}})$ 与整个生命周期服务总请求 Token 数 $D_{\mathrm{inf}}$ 的总成本最小化方程：解释为何 Llama-3 / Qwen3 选择远超 Chinchilla 最优比例（$20\ \text{tokens/param}$）的 $15\mathrm{T}\text{–}36\mathrm{T}$ Token 过度训练。

#### 2. 新增 `14.6` 节：案例深剖四——多方案系统设计评审与跨档位硬件（H100 vs. RTX 4090 / PRO 6000）选型分界
- **`14.6.1` Qwen3-8B 三候选 KV 状态表示方案的严格受控评审（[`experiments/ch13/13-01/design-review/README.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch13/13-01/design-review/README.md)）**：
  - 完整呈现 8 任务 $\times$ 4 重复 = 32 请求的基准评审：候选 A（BF16 KV，`28/32` 通过，耗时 `37.651 s`）、候选 B（FP8 E4M3 KV，`26/32` 通过，耗时 `38.640 s`）、候选 C（FP8 KV + 查询向量还原为 BF16，`28/32` 通过，耗时 `37.386 s`）；
  - 揭示两条关键协同设计准则：（1）**总分相同（A 与 C 均为 `28/32`）不代表行为等价**——A 败于长文档 `r0`，C 败于长文档 `r1`；（2）**算法层模拟量化不等于系统层显存节省**——包装器在注意力内部做 FP8 舍入，但 vLLM 底层仍分配 `12,884,115,456 B` 的完整 BF16 KV 缓冲池，必须将存储格式、内核解包与尾项失败身份联合审计。
- **`14.6.2` 为什么不能只看 TFLOPS/$：数据中心卡（H100/B200）与消费级/工作站卡（RTX 4090 / RTX PRO 6000）的四场景翻转边界**：
  - 基于 [`snapshots/13-架构协同设计.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/draft14-15-outline-audit/snapshots/13-%E6%9E%B6%E6%9E%84%E5%8D%8F%E5%90%8C%E8%AE%BE%E8%AE%A1.md) §13.3.1，对比单卡算力单价（RTX 4090 / PRO 6000 极具吸引力）与显存容量（$24\ \mathrm{GB} / 96\ \mathrm{GB}$ vs. $80\text{–}288\ \mathrm{GB}$）、显存带宽（$1.0\text{–}1.79\ \mathrm{TB/s}$ vs. $3.35\text{–}22\ \mathrm{TB/s}$）、卡间互连（无 NVLink 仅 PCIe vs. $900\text{–}3,600\ \mathrm{GB/s}$ NVLink）及机房可靠性，推演四类场景（大规模预训练、中小模型微调、低延迟在线服务、离线批量推理）的最优硬件翻转点。

#### 3. 新增 `14.7` 节：案例深剖五——可证伪的条件预测账本：K192 抢占恢复实验的真实证伪（[`experiments/ch13/13-07/README.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch13/13-07/README.md)）
- **`14.7.1` 为什么架构预测必须预注册冻结（Pre-registration）与限定作用范围（`D11-SP04`）**：
  - 详解实验 `13-07` 的双阶段证据链：基于 K96 存在“保留前缀比从头重做更慢”（$12.800\ \mathrm{s} / 11.619\ \mathrm{s} = 1.101664$）的反例，研究者在 `2026-09-09T07:28:10.963322+00:00` 冻结预测清单（SHA-256 锁定），预测 K192 下 `preserve` 窗口相对 `restart` 窗口的中位耗时比落在 $0.70 \le R \le 1.00$，且 `19,516` 项机制校验（`7,500` 唯一提交词元 + `1,920` 重复投递拦截）零错误。
- **`14.7.2` 当实测比预测更快时为何仍判定为“受控证伪（`refuted_in_scope`）”**：
  - 剖析 `07:30:02.180899+00:00` 实测结果：机制校验 `19,516/19,516` 全过，但 K192 的 10 个配对单元比值全数 $<1.0$（最小值 `0.373470`、最大值 `0.950270`），中位数 $R = 0.693141$，因跌破预注册下界 $0.70$ 而被严格标记为 `refuted_in_scope`；以此确立本书**不事后改写阈值、区分机制不变量与定量区间偏离**的工程科学准则。

#### 4. 新增 `14.8` 节：2023–2026 推理成本下降百倍的多因子正交归因总收束（`TC-18`）
- **`14.8.1` 四类重复计数陷阱（Double-Counting Traps）**：
  - 系统总结 [`research/token-cost-2023-2026/report.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/token-cost-2023-2026/report.md) 的四条铁律：（1）把不同负载下的孤立上限倍率直接相乘；（2）分子分母混淆（硬件租金降、吞吐升、再乘商业降价）；（3）忽视阿姆达尔瓶颈转移（当模型缩至 8B FP8 时主机 CPU 调度占 $62\%$）；（4）阶段错配（把 Prefill 优化倍率乘到 Decode 主导负载上）。
- **`14.8.2` 全栈六层贡献与逐级条件比率（Telescoping Decomposition）总账本**：
  - 构建覆盖**模型能力密度与架构（第 2–3 章）$\to$ 芯片算力/带宽/精度（第 4 章）$\to$ 算子融合与运行时去瓶颈（第 5 章）$\to$ 超节点与网络（第 6–7 章）$\to$ 连续批处理、KV 分级池化与 PD/AF 分离（第 8–9、13 章）$\to$ 异步调度、错峰 Batch 与商业定价（第 11 章）**的正交归因总表，与 `D11-SP01` 双轨贯穿推算表合龙。

#### 5. 第 14 章新增 12 幅 420pt 配图（`图 14-7` ~ `图 14-18`）与 6 道新习题（`14-7` ~ `14-12`）
> **严格单调位置说明**：因既有 `图 14-6` 位于 `14.5.2`，为保证全文图号 `14-1..14-18` 按阅读顺序严格递增，`14.1.3` 正文通过前向引用指向 `图 14-7` 与 `图 14-8`，而 `图 14-7` 与 `图 14-8` 物理嵌入在 `14.5.2` 之后的新增小节 `14.5.3` 中，随后依次为 `14.6`（`图 14-9`~`14-10`）、`14.7`（`图 14-11`~`14-13`）与 `14.8`（`图 14-14`~`14-18`）。
- **新增配图清单**：
  - `图 14-7`（`ch14/ch14-hardware-lottery-and-role-reversal.svg`，置于 `14.5.3` 并由 `14.1.3` 前向引用）：硬件彩票（Hardware Lottery）象限与 DFlash 块扩散模型从“主生成模型”转为“并行投机草稿器”的系统角色价值逆转；
  - `图 14-8`（`ch14/ch14-beyond-chinchilla-lifecycle-compute.svg`，置于 `14.5.3` 并由 `14.1.3` 前向引用）：计入推理生命周期总 Token 量 $D_{\mathrm{inf}}$ 后的过度训练（Over-training）最优参数—数据配比偏移曲线；
  - `图 14-9`（`ch14/ch14-qwen3-three-candidate-design-review.svg`，置于 `14.6.1`）：实验 `13-01`（`design-review`）三候选方案（BF16 KV / FP8 KV / FP8 KV+BF16 Q）的得分、长文档失败项位移（`r0` vs. `r1`）与 $12.88\ \mathrm{GB}$ 显存池真相；
  - `图 14-10`（`ch14/ch14-datacenter-vs-consumer-gpu-crossover.svg`，置于 `14.6.2`）：H100/B200 数据中心卡与 RTX 4090 / RTX PRO 6000 在预训练、微调、在线服务与离线批处理四类场景下的成本—SLO 分界；
  - `图 14-11`（`ch14/ch14-preregistered-prediction-protocol.svg`，置于 `14.7.1`）：可证伪架构预测的十字段预注册冻结协议（冻结时间戳、SHA-256、条件范围、机制不变量与定量区间）；
  - `图 14-12`（`ch14/ch14-k96-k192-preemption-falsification.svg`，置于 `14.7.2`）：实验 `13-07` 中 K96 反例（$R=1.101664$）到 K192 实测（10 配对范围 $0.373470\text{–}0.950270$，中位数 $R=0.693141 < 0.70$ 触发 `refuted_in_scope`）及 `19,516` 项机制校验全通过的双层证据图；
  - `图 14-13`（`ch14/ch14-idempotent-checkpoint-dedup-timeline.svg`，置于 `14.7.2`）：`restart`、`append_only` 与 `idempotent_checkpoint` 在抢占恢复中拦截 `1,920` 个重复投递词元、确保 `7,500` 个唯一提交词元的时序对比；
  - `图 14-14`（`ch14/ch14-four-double-counting-traps.svg`，置于 `14.8.1`）：2023–2026 Token 成本归因中的四类重复计数谬误（上限叠乘、分子分母混同、阿姆达尔瓶颈转移、Prefill/Decode 阶段错配）；
  - `图 14-15`（`ch14/ch14-token-cost-2023-2026-telescoping-waterfall.svg`，置于 `14.8.2`）：2023–2026 固定能力前沿成本下降 $208\times\text{–}375\times$ 的全栈逐级条件比率瀑布图（`TC-18`）；
  - `图 14-16`（`ch14/ch14-dual-track-throughline-evolution.svg`，置于 `14.8.2`）：全书双轨贯穿算例（Qwen3-8B Dense 轨与 V4.1 Flash / 235B MoE 轨）跨章节逐步修正轨迹图（`D11-SP01`）；
  - `图 14-17`（`ch14/ch14-rubin-v41-co-design-mapping.svg`，置于 `14.8.2`）：前沿模型结构（MLA / 稀疏注意力 / 细粒度 MoE / MTP）与 2026 前沿芯片（Rubin HBM4 / TMEM 2:4 稀疏 / 内联 TMA / NVLink 6 计数写）的一一映射矩阵；
  - `图 14-18`（`ch14/ch14-co-design-closed-loop-synthesis.svg`，置于 `本章小结` 前）：模型算法—运行时软件—芯片微架构—超节点网络四维协同设计闭环总图。
- **新增习题与实验（`14-7` ~ `14-12`）**：
  - `> **练习 14-7：推导计入生命周期总推理 Token 量 $D_{\mathrm{inf}}$ 后的最优训练参数规模 $N^*$，解释为何高并发服务模型倾向于过度训练。**`
  - `> **练习 14-8：基于实验 `13-01`（`design-review`）结果，分析为何候选 A 与候选 C 均取得 `28/32` 却不能互换，以及如何验证量化是否真正释放了物理 KV 块。**`
  - `> **练习 14-9：计算 RTX PRO 6000（96 GB, 1.79 TB/s）与 H100（80 GB, 3.35 TB/s）在单流 Decode 与 8 卡 TP AllReduce 下的吞吐与成本翻转点。**`
  - `> **练习 14-10：剖析实验 `13-07` 中 K96（$R=1.101664$）与 K192（$R=0.693141$）的方向反转原因，并说明为何 $R=0.693141$ 优于预期却必须判为 `refuted_in_scope`。**`
  - `> **练习 14-11：给定模型小型化 $8.75\times$、权重量化 $2\times$、HBM 带宽提升 $1.43\times$ 及主机固定调度开销 $0.8\ \mathrm{ms}$，计算端到端真实加速比并与简单相乘（$25.1\times$）对比（TC-18）。**`
  - `> **实验 14-12：运行 `experiments/ch13/13-07` 预注册恢复检验脚本，改变抢占位置 $K$ 并输出完整的十字段预测复核报告。**`

---

### 3.4 新建第 15 章：《吉瓦级 AI 数据中心、电力热学与跨地域协同训练》（`manuscripts/15-吉瓦级 AI 数据中心、电力热学与跨地域协同训练.md`，`~980` 行，`16` 幅配图，`10` 道习题）

当单集群从万卡（$\sim 10\text{–}20\ \mathrm{MW}$）迈向十万卡乃至百万卡（$100\ \mathrm{MW}\text{–}1\ \mathrm{GW}$）时，**供电容量、瞬态功率摆幅（Power Swing）、液冷热力学边界与跨地域广域网光传输**取代单卡 FLOPS，成为决定 AI 基础设施能否建得成、跑得稳的第一性物理约束。本章全部推导严格锚定 [`nvidia-rubin-system.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/references/files/specs/nvidia-rubin-system.md)、[`ashrae-liquid-cooling.txt`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/references/text/ashrae-liquid-cooling.txt)、[`verify_tapas_reading.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/2026-infra-survey/verify_tapas_reading.py)、[`logicfolding-energy.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/logicfolding-energy.md) 与 `D11-08`–`D11-09`：

#### 章节小节结构与核心定量算例
- **`## 15.1 从兆瓦机房到吉瓦级“AI 工厂”：供电拓扑、物理可行性与站点经济学`**
  - `### 15.1.1` **园区供电链路与高压直流演进**：从变电站（$110\text{–}220\ \mathrm{kV}$）$\to$ 中压配电（$10\text{–}35\ \mathrm{kV}$）$\to$ 变压器/UPS $\to$ 机柜母排（$48\ \mathrm{V}$ 向 $800\ \mathrm{VDC}$ 演进，$I = P/V$ 降为 $1/16.7$、$I^2R$ 铜损降为 $1/278$），核算单机柜从风冷 $10\text{–}15\ \mathrm{kW}$ 跃升至 NVL72 级 $120\text{–}140\ \mathrm{kW}+$ 的供电转换损耗与 PUE 账本；
  - `### 15.1.2` **物理可行性 vs. 经济可行性的双边界（`D11-08`）**：严格界定 $P_{\mathrm{domain}} \times \mathrm{GB/W}$ 的分子分母口径——说明为何当 $\mathrm{GB/W}$ 取自加速卡 TDP、而 $P_{\mathrm{domain}}$ 包含主机 CPU、NVLink/光交换机、CDU 水泵与电源转换损耗时，直接相乘会严重高估超节点可装载的模型容量；
  - `### 15.1.3` **站点条件（电价、接入容量、并网工期）作为一等设计输入（`D11-09`）**：建立包含购电单价（如 $0.04$ vs. $0.12\ \text{美元}/\mathrm{kWh}$）、变电站扩容一次性资本开支、并网等待工期（芯片 36 个月经济寿命下的延期折旧损失）与机柜功率上限的 TCO 敏感性矩阵。
- **`## 15.2 同步训练的电网负载冲击（Power Swing）与功率平滑机制`**
  - `### 15.2.1` **万卡同步 Step 边界的 $\mathrm{d}P/\mathrm{d}t$ 功率冲击波**：定量刻画数据并行/流水线并行在 GEMM 满载期（每卡 $700\text{–}1,200\ \mathrm{W}$）与 AllReduce/Barrier 通信等待期（骤降至 $250\text{–}400\ \mathrm{W}$）之间以 $50\ \mathrm{ms}\text{–}1\ \mathrm{s}$ 周期齐步切换时，如何在十万卡园区产生数十兆瓦的周期性阶跃冲击，威胁上级变压器与电网频率稳定；
  - `### 15.2.2` **硬件级功率平滑与集群功率超配（Vera Rubin 智能功率平滑 & DSX MaxLPS）**：依据 [`nvidia-rubin-system.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/references/files/specs/nvidia-rubin-system.md)，解析机架级储能 SoC 闭环控制如何**降低约 $10\%$ 平均功耗、削平约 $20\%$ 的 $50\ \mathrm{ms}$ 峰值功率脉冲**，以及 DSX MaxLPS 如何在固定兆瓦电力配额下安全超配部署最高 **$+40\%$ 的 GPU 规模**；
  - `### 15.2.3` **运行时软件协同抑峰**：微批次错峰注入、空闲期内填充低算术强度校验核（Dummy/Padding Kernel）或动态 DVFS 锁频的能耗—吞吐得失核算。
- **`## 15.3 液冷热力学（ASHRAE W45）、功率密度悖论与热感知调度（*TAPAS*）`**
  - `### 15.3.1` **冷板式液冷与 CDU 换热回路热力学方程**：依据 [`ashrae-liquid-cooling.txt`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/references/text/ashrae-liquid-cooling.txt) 与 Vera Rubin 规范，建立 $Q = \dot{m} c_p \Delta T$ 流量—温差账本，解释为何支持 **$45^\circ\mathrm{C}$ 进水温度（ASHRAE W45）** 能够免除高能耗冷水机组（Chiller）、实现全年自然冷却（Free Cooling）；
  - `### 15.3.2` **总功耗下降为何散热压力反而上升：LogicFolding 功率密度悖论（[`case-studies/logicfolding-energy.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/logicfolding-energy.md)）**：复算三维折叠电路中电压从 $0.85\ \mathrm{V}$ 降至 $0.55\ \mathrm{V}$ 使动态功耗项降至 $(0.55/0.85)^2 \approx 0.42$，以及当总功耗降至 $0.75$、但投影面积缩至 $0.60$ 时，**单位面积功率密度反而升至 $0.75 / 0.60 = 1.25\times$（$+25\%$）** 的热阻与结温（$T_j = T_{\mathrm{coolant}} + P \cdot R_{\theta}$）边界；
  - `### 15.3.3` **热节流（Thermal Throttling）的同步木桶效应与 *TAPAS* 热感知重配置（ASPLOS '25）**：完整植入 [`verify_tapas_reading.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/2026-infra-survey/verify_tapas_reading.py) 的锁定算例——当两排机柜功率分别为 `[96+6, 80+6] = [102, 86] kW`（第一排突破 `100 kW` 供电/冷却上限）、两类请求延迟为 `[0.8, 1.3] s`（第二类突破 `1.0 s` SLO）时，通过重配置模型放置与请求路由将功率平衡至 `<= 100 kW` 并把单请求耗时压低 `0.1 s`；计入 `3 s` 串行重配置停顿后，精确导出**连续服务 $\ge 30$ 个请求（$3.0 / 0.1 = 30$）即收回重配置代价**。
- **`## 15.4 突破单园区电力墙：跨数据中心广域光网（DCI）与低频同步训练（DiLoCo）`**
  - `### 15.4.1` **当单站点电力封顶（如 $200\ \mathrm{MW}$）时：跨地域光传输的物理延迟与 BDP 账本**：光纤传播延迟 $\approx 5\ \mu\mathrm{s/km}$（相距 $1,000\ \mathrm{km}$ 的两座园区单向延迟 $5\ \mathrm{ms}$、$\mathrm{RTT} \approx 10\ \mathrm{ms}$）；核算在 $400\ \mathrm{Gbps}\text{–}1.6\ \mathrm{Tbps}$ DWDM 专线上的带宽时延积（$\mathrm{BDP} = \mathrm{BW} \times \mathrm{RTT}$）与为何不能直接把张量并行（TP）跨城部署；
  - `### 15.4.2` **跨园区流水线并行（Cross-DC PP）与分层数据并行（Hierarchical DP）的容忍边界**：推导微批次计算时间掩盖 $10\ \mathrm{ms}$ 广域网 RTT 所需的最小微批规模与梯度累积步数；
  - `### 15.4.3` **低频外循环同步（DiLoCo / Streaming DiLoCo）的带宽压缩与显存代价**：当内循环独立演进 $H = 500$ 步后再跨园区同步伪梯度（Pseudo-Gradient $\Delta \theta = \theta^{(0)} - \theta^{(H)}$）时，广域网平均带宽需求直降 **$500\times$**；同时严格核算外循环 Nesterov 动量优化器在每园区额外引入的主参数副本与动量状态显存开销，以及流式分块同步（Streaming DiLoCo）消除通信尖峰的机制。
- **`## 常见误区`、`## 练习与实验`（`练习 15-1` ~ `实验 15-10`）、`## 本章小结`**

#### 第 15 章新增 16 幅 420pt 配图（`图 15-1` ~ `图 15-16`）
- `图 15-1`（`ch15/ch15-gigawatt-ai-factory-power-chain.svg`）：吉瓦级 AI 工厂从 $220\ \mathrm{kV}$ 变电站到机柜 $800\ \mathrm{VDC}$ 母排与液冷 CDU 的全链路供电及损耗分解；
- `图 15-2`（`ch15/ch15-physical-vs-economic-feasibility.svg`）：物理可行域 vs. 经济可行域双边界及 $P_{\mathrm{domain}} \times \mathrm{GB/W}$ 口径错配高估警示图（`D11-08`）；
- `图 15-3`（`ch15/ch15-site-conditions-tco-sensitivity.svg`）：电价、并网延期折旧与机柜功率上限对集群 TCO 与最优并行方案的敏感性相图（`D11-09`）；
- `图 15-4`（`ch15/ch15-synchronous-training-power-swing.svg`）：万卡同步训练 GEMM 计算期与 AllReduce 通信期周期性切换引发的数十兆瓦 $\mathrm{d}P/\mathrm{d}t$ 电网冲击波形；
- `图 15-5`（`ch15/ch15-rubin-power-smoothing-and-maxlps.svg`）：Vera Rubin 机架级储能 SoC 功率平滑（削峰 $-20\%$、均功 $-10\%$）与 DSX MaxLPS 同配额 $+40\%$ GPU 超配机制；
- `图 15-6`（`ch15/ch15-runtime-staggered-microbatch-power.svg`）：微批次错峰注入与低功率填充算子抑制集群级功率跌落谷底的时序对比；
- `图 15-7`（`ch15/ch15-ashrae-w45-cdu-liquid-cooling-loop.svg`）：ASHRAE W45（$45^\circ\mathrm{C}$ 温水进液）冷板—分集水器—CDU—干冷器无冷机（Chiller-less）换热回路；
- `图 15-8`（`ch15/ch15-logicfolding-power-density-paradox.svg`）：LogicFolding 降压节能（$(0.55/0.85)^2 \approx 0.42$）与面积缩减更快导致的功率密度反升（$0.75/0.60 = 1.25\times$）及结温热阻模型；
- `图 15-9`（`ch15/ch15-thermal-throttling-straggler-amplification.svg`）：单颗过热 GPU 触发降频 $15\%$ 如何通过同步 AllReduce 屏障拖慢整组 $1,024$ 张卡的有效吞吐；
- `图 15-10`（`ch15/ch15-tapas-thermal-power-routing-breakeven.svg`）：*TAPAS* 热感知与功率封顶重配置（`[102, 86] kW` 削峰至 $\le 100\ \mathrm{kW}$，`3 s` 重配置在 $30$ 个请求处回本）；
- `图 15-11`（`ch15/ch15-multi-datacenter-dci-optical-latency.svg`）：跨 $1,000\ \mathrm{km}$ 广域光网（$\mathrm{RTT}=10\ \mathrm{ms}$）的传播延迟、DWDM 通道与不同并行维度的延迟容忍阶梯；
- `图 15-12`（`ch15/ch15-cross-dc-pipeline-and-hierarchical-dp.svg`）：跨园区流水线并行（Cross-DC PP）与园区内 NVLink/RoCE + 园区际 DWDM 分层数据并行时序；
- `图 15-13`（`ch15/ch15-diloco-inner-outer-optimization.svg`）：DiLoCo 内循环 $H=500$ 步独立更新与外循环 Nesterov 伪梯度同步将跨城带宽需求压缩 $500\times$ 的机制；
- `图 15-14`（`ch15/ch15-streaming-diloco-chunked-sync.svg`）：全量阻塞式外循环同步与 Streaming DiLoCo 分片流水重叠同步的广域网链路利用率对比；
- `图 15-15`（`ch15/ch15-single-campus-vs-multi-campus-decision.svg`）：单园区扩建等待电网扩容 vs. 多园区跨城协同训练的完成工期与成本交点决策树；
- `图 15-16`（`ch15/ch15-power-thermal-wan-co-design-summary.svg`）：从芯片结温、机柜液冷、变电站功率平滑到跨地域光网的吉瓦级 AI 基础设施全景图。

---

### 3.5 新建第 16 章：《机密 AI 计算、可信执行环境与安全沙箱底座》（`manuscripts/16-机密 AI 计算、可信执行环境与安全沙箱底座.md`，`~980` 行，`16` 幅配图，`10` 道习题）

当大模型进入金融、政务、医疗及百万并发代码/终端执行 Agent 强化学习时代，**“如何防止云平台/宿主机窃取高价值模型权重与用户隐私提示词（机密 AI 计算）”** 与 **“如何防止百万个自主演进的 RL Agent 逃逸容器、打挂宿主内核或伪造奖励信号（安全沙箱与可信审计底座）”** 成为 AI 基础设施不可或缺的安全双翼。本章 100% 锚定 [`case-studies/dsec-sandbox-platform.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/dsec-sandbox-platform.md)、[`case-studies/evaluation-and-agent-records.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/evaluation-and-agent-records.md)（`I19`）、[`reading-asplos-2025.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/2026-infra-survey/reading-asplos-2025.md)（*PipeLLM*）、[`nvidia-rubin-arch.txt`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/references/text/nvidia-rubin-arch.txt)（Rubin TEE-I/O）及第 4 章实验 `4-12` 的机密传输基础：

#### 章节小节结构与核心定量算例
- **`## 16.1 AI 基础设施的信任边界与双向威胁模型`**
  - `### 16.1.1` **双向安全威胁：保护模型与用户数据免受不可信宿主机窥探（机密计算）vs. 保护宿主机与内网免受不可信 Agent 破坏（安全沙箱）**；
  - `### 16.1.2` **全栈攻击面梳理**：权重文件落盘与内存明文驻留、PCIe/NVLink 总线嗅探、跨租户 KV 缓存侧信道、以及 RL 训练/评测中不可信代码调用内核系统调用与内网服务的越权路径。
- **`## 16.2 硬件级机密 GPU 计算：从 CPU+GPU 双 TEE 弹跳缓冲到 Rubin 原生 TEE-I/O`**
  - `### 16.2.1` **CPU TEE（Intel TDX / AMD SEV-SNP）与 GPU TEE 的远程证明（Remote Attestation）及硬件信任根**：密钥如何绑定至度量哈希并在进入 GPU HBM 前保持密文；
  - `### 16.2.2` **传统弹跳缓冲（Bounce Buffer）的 PCIe 双重拷贝与 CPU AES-GCM 加解密税 vs. 硬件内联加密（PCIe TDISP / IDE 与 NVLink 机密模式）**：延续第 4 章 `4.5.2` 基础，对比在 CPU 侧用软件/指令加密 $64\ \mathrm{MiB}$ 激活再经 PCIe DMA 拷入 GPU 的带宽折损，与 Blackwell / **Rubin 全系原生 TEE-I/O**（在 PCIe Gen 6 $256\ \mathrm{GB/s}$ 与 NVLink 6 $3,600\ \mathrm{GB/s}$ 链路控制器内直接完成线速认证加密、消除主机弹跳缓冲拷贝）的代际飞跃。
- **`## 16.3 机密大模型服务的投机流水线加密与状态保护（*PipeLLM* 与 *TensorTEE*）`**
  - `### 16.3.1` **机密卸载与跨卡流水中的加密墙：以 *PipeLLM*（ASPLOS '25）为例**：完整落地 [`reading-asplos-2025.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/2026-infra-survey/reading-asplos-2025.md) 锁定算例——针对单个 $64\ \mathrm{MiB}$ 张量的主机—设备机密传输，核算 `H2D -> Decrypt -> Compute -> Encrypt -> D2H` 五阶段流水线的周期瓶颈与**三缓冲（Triple-Buffering：$3 \times 64\ \mathrm{MiB} = 192\ \mathrm{MiB}$）显存驻留账本**；
  - `### 16.3.2` **一次性计数器模式（AES-CTR OTP）预生成掩码与投机预加密（Speculative Encryption）的命中率边界**：如何利用空闲阶段预先生成随机流密码掩码，将在线加密转化为按位异或（XOR），并核算投机分支被拒绝时的重加密带宽代价。
- **`## 16.4 百万级 Agent 强化学习安全沙箱底座：以 DeepSeek DSec 为例`**
  - `### 16.4.1` **为什么裸容器挡不住自主 RL Agent：四类隔离后端与真实越界/内核故障案例集**：完整植入 [`case-studies/dsec-sandbox-platform.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/dsec-sandbox-platform.md) 的生产记录——（1）梳理 DSec 的四类隔离后端（面向无状态任务的预热 **`FnCall`**、**运行在 QEMU/libvirt 虚拟机内的容器**、轻量级 **`Firecracker microVM`**、以及支持 Android 与 GPU 半虚拟化的**完整 QEMU VM**）；（2）共享内核下的真实破坏：Agent 使用 `XFS_IOC_SWAPEXT` 绕过访问控制损坏 XFS 元数据、覆盖 `/bin/bash`、或递归 `grep` 读取 `/proc/kpagecgroup` 触发宿主机内核缺陷与 `yes` 刷爆数十 GB 日志；（3）内网越界：扫描内网端口、读取或伪造发往 `chronus` socket 的 RPC、以及借 Go module proxy 获取参考实现；以此论证**硬件虚拟化隔离边界（Firecracker microVM / VM 内置容器）+ AppArmor + 动态 eBPF 网络白名单**的必然性；
  - `### 16.4.2` **单单元 160 节点并发运行 38 万沙箱的内存消除术（`virtio-pmem` + DAX 与 DAMON 主动气球回收）**：
    - 锁定生产规模：单 Cell 仅 `160` 台 CPU 节点（约 `30,000` 物理核、`250 TB` DRAM），日创建 `300` 万沙箱、峰值并发 `38` 万、创建速率 $>5,000\ \mathrm{s}^{-1}$（单机 $31.3\ \mathrm{s}^{-1}$），且 $90\%$ 容器与 microVM 平均 CPU 利用率 $\le 5\%$（单节点稳定运行 $\ge 3,200$ 个容器或 $\ge 800$ 个 microVM）；
    - 定量拆解只读镜像页去重：传统块设备导致宿主机与每个 microVM Guest OS 页缓存双重驻留；改用 **`virtio-pmem` + DAX** 让各 microVM 共享宿主机同一份页缓存，**削减 $40.2\%$ 峰值内存**！同时精确指出其隐藏代价：Guest 内核必须为 `pmem` 地址空间分配每 $4\ \mathrm{KiB}$ 页 `64 B` 的 `struct page` 描述符（占比 $1/64$，映射 $128\ \mathrm{GB}$ 设备即消耗 **$2\ \mathrm{GB}$ Guest 内存**，且瞬时 CPU 峰值由 $26.5\%$ 升至 $41.4\%$），因此生产中只读 EROFS 基础镜像与 toolkit 层采用 `virtio-pmem` + DAX，而较大可写磁盘采用 **DAMON 冷页回收 + `virtio-balloon` 空闲页报告（默认 order-9 即 `2 MiB` 块）**，将**时间积分内存（$\mathrm{GB}\cdot\mathrm{s}$）降低 $21.2\%$**；
  - `### 16.4.3` **超卖 CPU 下的噪声邻居隔离（`SCHED_IDLE` + Core Scheduling）与 eBPF 动态网络白名单**：
    - 当延迟敏感型（LS）任务与尽力而为型（BE，50% 负载）任务同机混部时，默认调度下棋 Agent 每步耗时膨胀 **$+45.2\%$**（仅用 `SCHED_IDLE` 最多改善 $3.4\%$）；将 BE 放入 `SCHED_IDLE` 并为 LS 开启核调度（`prctl(PR_SCHED_CORE)`）后，耗时膨胀压低至 **$+17.3\%$**（剩余干扰来自多核高负载降频、内存带宽与 LLC 争用）；
    - 结合 AppArmor（文件与 socket 限制）与按任务动态更新的 **eBPF 网络白名单**，阻断横向扫描与未授权连接。
- **`## 16.5 可恢复的任务审计账本、幂等工具执行与全书十六章总收束`**
  - `### 16.5.1` **日志后端中断下的不丢账、不重扣：Agent 事件流缓冲与排空定量模型（[`case-studies/evaluation-and-agent-records.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/evaluation-and-agent-records.md) / `I19`）**：
    - 完整复算锁定算例：当系统以 $10\ \mathrm{tasks/s}$ 运行、每任务产生 $20$ 条事件、每条事件平均 $2\ \mathrm{KiB}$ 时，稳态日志到达率为 **$400\ \mathrm{KiB/s}$**；若下游 Kafka/ClickHouse 审计后端中断 $120\ \mathrm{s}$，共积压 $24,000$ 条事件 = **$48,000\ \mathrm{KiB} = 46.875\ \mathrm{MiB}$**（若本地仅配 $32\ \mathrm{MiB} = 32,768\ \mathrm{KiB}$ 内存环形缓冲，将在 **$32,768 / 400 = 81.92\ \mathrm{s}$** 时溢出丢账！）；当后端恢复并以 $1\ \mathrm{MiB/s} = 1,024\ \mathrm{KiB/s}$ 速率消费、同时新事件仍以 $400\ \mathrm{KiB/s}$ 持续涌入时，净排空速率为 $1,024 - 400 = 624\ \mathrm{KiB/s}$，完全排空 $46.875\ \mathrm{MiB}$（即 $48,000\ \mathrm{KiB}$）积压需 **$48,000\ \mathrm{KiB} / 624\ \mathrm{KiB/s} \approx 76.92\ \mathrm{s}$**；
    - 严格区分**可采样的性能诊断 Trace（OpenTelemetry 尾部采样）**与**不可采样的计费账单、工具外部副作用幂等键（Idempotency Key）及 RL 轨迹版本准入日志**；
  - `### 16.5.2` **全书十六章总结：从一次矩阵乘法的五个追问，到吉瓦级可信 AI 计算机体系结构**：
    - 将全书 16 章收束为一张完整的知识坐标系：回顾“搬什么、搬多少、搬几次、经过哪里、谁必须等它”如何从单个算子寄存器分块（第 4–5 章）一路扩展到超节点与数据中心网络（第 6–7 章）、分布式推理与训练（第 8–10 章）、端边云与存储底座（第 11–13 章）、跨层软硬协同（第 14 章），最终抵达吉瓦级园区能量流（第 15 章）与硬件可信安全边界（第 16 章）。
- **`## 常见误区`、`## 练习与实验`（`练习 16-1` ~ `实验 16-10`）、`## 本章小结`**

#### 第 16 章新增 16 幅 420pt 配图（`图 16-1` ~ `图 16-16`）
- `图 16-1`（`ch16/ch16-confidential-ai-vs-sandbox-threat-model.svg`）：AI 基础设施的双向信任边界——保护模型/数据免受宿主窥探（机密计算 TEE）vs. 保护宿主/内网免受不可信 Agent 破坏（微虚拟机沙箱）；
- `图 16-2`（`ch16/ch16-cpu-gpu-tee-remote-attestation.svg`）：CPU TEE（TDX/SEV-SNP）与 GPU TEE 的远程证明、密钥注入及加密显存边界；
- `图 16-3`（`ch16/ch16-bounce-buffer-vs-rubin-tee-io.svg`）：传统 CPU 弹跳缓冲双重拷贝加密 vs. Blackwell/Rubin 原生 PCIe TDISP/IDE 与 NVLink 6 TEE-I/O 硬件内联线速加密路径对比；
- `图 16-4`（`ch16/ch16-pipellm-five-stage-confidential-pipeline.svg`）：*PipeLLM* 机密张量卸载的五阶段流水线（`H2D -> Decrypt -> Compute -> Encrypt -> D2H`）与 $192\ \mathrm{MiB}$ 三缓冲显存账本；
- `图 16-5`（`ch16/ch16-aes-ctr-otp-speculative-encryption.svg`）：AES-CTR 一次性掩码预生成与投机分支预加密的命中/回退收支模型；
- `图 16-6`（`ch16/ch16-agent-container-escape-failure-modes.svg`）：真实生产环境中自主 RL Agent 突破共享内核容器的故障模式（`XFS_IOC_SWAPEXT`、`/proc/kpagecgroup` 缺陷、`chronus` 内网 RPC 伪造）；
- `图 16-7`（`ch16/ch16-dsec-firecracker-microvm-architecture.svg`）：DeepSeek DSec 安全沙箱平台整体架构（160 节点 / 38 万并发沙箱、四类后端 `FnCall`/VM 容器/`Firecracker microVM`/完整 QEMU VM、按需 EROFS、DAMON 与 eBPF 白名单）；
- `图 16-8`（`ch16/ch16-virtio-pmem-dax-vs-struct-page-tax.svg`）：`virtio-pmem` + DAX 消除宿主—客户机双重页缓存（峰值内存 $-40.2\%$）与每 $128\ \mathrm{GB}$ 镜像消耗 $2\ \mathrm{GB}$ `struct page` 元数据的权衡边界；
- `图 16-9`（`ch16/ch16-damon-ept-balloon-reclaim.svg`）：宿主机 DAMON 扫描 KVM EPT 页表访问位驱动 `virtio-balloon` 回收闲置冷页（时间积分内存 $-21.2\%$）的闭环控制；
- `图 16-10`（`ch16/ch16-sched-idle-and-core-scheduling.svg`）：高优评测沙箱与低优合成数据沙箱混部下，CFS 默认调度（延迟膨胀 $+45.2\%$）vs. `SCHED_IDLE` + SMT 核调度（压低至 $+17.3\%$）对比；
- `图 16-11`（`ch16/ch16-ebpf-cgroup-network-allowlist.svg`）：基于 cgroup eBPF 的微虚拟机动态网络白名单与内网 RFC1918 零信任阻断流水线；
- `图 16-12`（`ch16/ch16-agent-audit-log-outage-and-drain.svg`）：算例 `I19` 中日志后端中断 $120\ \mathrm{s}$ 积压 $46.875\ \mathrm{MiB}$（$32\ \mathrm{MiB}$ 缓冲在 $81.92\ \mathrm{s}$ 溢出）及恢复后以 $624\ \mathrm{KiB/s}$ 净速率在 $76.92\ \mathrm{s}$ 排空的水位曲线；
- `图 16-13`（`ch16/ch16-idempotent-tool-commit-and-replay.svg`）：网络确认丢失与重试下，利用幂等键（Idempotency Key）与两阶段状态日志分离“结果重传”与“外部副作用重复执行”；
- `图 16-14`（`ch16/ch16-telemetry-sampling-vs-billing-ledger.svg`）：可采样性能诊断 Trace 与不可采样计费/RL 准入证据链的双通道分流架构；
- `图 16-15`（`ch16/ch16-sixteen-chapter-full-stack-map.svg`）：全书 16 章“从算子寄存器到吉瓦级可信数据中心”的五层追问与全栈知识地图；
- `图 16-16`（`ch16/ch16-ai-infra-architect-master-checklist.svg`）：AI 基础设施架构师的六维终极核查表（算力、存力、运力、电力热学、可信安全、全生命周期成本）。

---

## 四、出版级配图（`gpt-image-2.6-sunburst` + `420pt` 矢量规范）与版式美学强约束

针对本次路径 B 新增的 **72 幅配图**（第 1–12 章新增 16 幅 + 第 13 章新增 12 幅 + 第 14 章新增 12 幅 + 第 15 章新增 16 幅 + 第 16 章新增 16 幅），全面执行与路径 A 相同的最高出版级视觉工程标准：

### 4.1 强制调用 `codex exec` + `gpt-image-2.6-sunburst`（`quality="high"`）
1. **模型与质量等级**：在每一批次生成或重绘配图时，必须通过 `codex exec` 驱动 **`gpt-image-2.6-sunburst`** 模型，并显式指定 **`quality="high"`**（最高质量档位）生成高精度视觉构图参考或设计原型，再结合 [`manuscripts/figure_style/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/figure_style) 与各章 `build.py` 落地为精确受控的 `420pt` 矢量 `.svg`、`.pdf` 与高分 `.png`（三格式齐全，且全部真实物理落盘，严禁 Git LFS 指针残留）。
2. **五大视觉维度硬指标**：
   - **字体（Typography）**：统一采用 `PingFang SC` / `Source Han Sans SC` / `Noto Sans CJK SC` 与 `Helvetica Neue` / `Arial` 无衬线字体族；主标题 `9.5–10.5 pt` 加粗，子面板标题 `8.5–9.0 pt` 加粗，数据标注与轴刻度 `7.5–8.5 pt`，图注说明 `7.0–7.5 pt`；同一层级字号全图严格一致，严禁字号忽大忽小或中英文基线错位。
   - **对齐（Alignment）**：所有模块框、流水线阶段、坐标轴、图例与文本标签严格对齐像素/点阵网格（Grid-Snapped）；多面板（Panel A / B / C）顶边、底边与列间距保持几何级对称。
   - **排版与留白（Layout & Whitespace）**：画布宽度锁定为 **`420pt`**（匹配 ElegantBook 正文版心宽度，1:1 零缩放嵌入），外边距充足，模块内边距（Padding）不低于 `6–8 pt`；**严禁任何文字压线、文字越界、箭头穿过文字、图例遮挡曲线或双行标签重叠**。
   - **色彩体系（Color Palette）**：严格复用 [`manuscripts/figure_style/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/figure_style) 的学术出版级色板（深海蓝 `#1F4E79` / `#2B5B84`、青黛绿 `#1B7837` / `#2A9D8F`、琥珀橙 `#D95F02` / `#E76F51`、珊瑚红 `#C0392B`、岩灰 `#595959` 与柔和浅底色 `#F4F7FA` / `#FFF8F0`），用高对比度语义色突出瓶颈路径、反转交点与证伪边界。
   - **信息密度与自包含性（Self-Contained Visuals）**：每幅图必须直接标出关键锁定数值（如 `$22\ \mathrm{TB/s}$`、`$32.8\ \mu\mathrm{s}$`、`30 requests`、`refuted_in_scope ($R=0.693141$)`、`$-40.2\%$ Peak Memory`），让读者仅看图即可读懂核心定量结论。
3. **Subagent 视觉巡检门禁（Visual Gate）**：每批次配图生成后，必须使用 `read_file` 直接读取生成的 `.png` 光栅图进行人工级视觉检查，且由 **Gemini 4 Argon High Review Subagent** 逐张打开 `.png` 核查字体、对齐、留白、配色与数值准确性；发现哪怕一处文字挤压或对齐瑕疵，立即修正重绘直至零瑕疵通过。

---

## 五、全书 16 章构建、校验与导航基础设施适配清单

将全书从 14 章扩展至 16 章时，必须在 **Batch 8（收官批次）** 同步更新以下脚本与索引文件，确保全仓库构建、校验与 GitHub Pages 部署 100% 绿灯通过：

| 序号 | 目标文件 | 当前硬编码位置 | 路径 B 精确修改方案 |
| :-: | --- | --- | --- |
| 1 | [`scripts/verify_core_principles.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/scripts/verify_core_principles.py) | 第 10 行：`for n in range(1,15):`；第 27 行仅匹配 `^> \*\*(?:练习|实验) (\d+-\d+)` | 改为 `for n in range(1,17):`（保持第 23 行 `if n<=12:` 对 `before/` 的历史兼容检查不变），并新增覆盖第 1–16 章全部习题格式（含第 10 章 `习题` 与第 12 章无 `>` 的 `**拓展练习 12-X`）的统一清点正则 `r'^(?:>\s*)?\*\*(?:(?:练习|实验|习题|拓展练习)\s+)?(\d+-\d+)\b'`，确保全书 615 幅图与 222 道习题 100% 纳入自动校验。 |
| 2 | [`scripts/build_site.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/scripts/build_site.py) | 第 54–58 行：`if n <= 14` 与 `expected = list(range(15))` | 将简体中文版 `zh` 的章节上限改为 `if n <= 16` 与 `expected = list(range(17))`（含 `00-前言` 共 17 个页面；英文/繁体/俄语社区译本保持 `range(13)` 不变）。 |
| 3 | [`scripts/check_site.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/scripts/check_site.py) | 第 46 行：`max_ch = 15 if name == '简体中文' else 13`；第 54 行打印信息 | 改为 `max_ch = 17 if name == '简体中文' else 13`，并将第 54 行打印信息更新为 `(16 zh / 12 translations)`。 |
| 4 | [`book/build_pdf.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/book/build_pdf.py) | 第 70 行：`choices=range(1,15)`；第 79 行：`range(1,15)` | 全部改为 `range(1,17)`，支持单章编译 `--chapter 15/16` 与全书 16 章合订本 PDF 编译。 |
| 5 | [`book/build_epub.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/book/build_epub.py) | 第 65 行：`for n in range(13)` | 将简体中文版 `edition == 'zh'` 的章节范围更新为 `for n in range(17)`，使中文 EPUB 完整收录前言与第 1–16 章。 |
| 6 | [`book/check_ci_pdf.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/book/check_ci_pdf.py) | 第 29–36 行硬编码 `list(range(1, 13))` 与 `len(chapter_entries) < 12` | 区分语言版本：`expected_chapters = list(range(1, 17)) if args.edition == 'zh' else list(range(1, 13))`，并校验 `report['chapters'] == expected_chapters` 与 `len(chapter_entries) >= len(expected_chapters)`，确保 GitHub Actions CI 全书 PDF 校验通过。 |
| 7 | [`manuscripts/ch15/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts) & [`manuscripts/ch16/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts) | 尚不存在 | 新建 `manuscripts/ch15/` 与 `manuscripts/ch16/` 目录，分别包含 `figure-index.json`、`figure-data.json`、`build.py` 及各自的 16 套 `.svg`/`.png`/`.pdf` 配图资产。 |
| 8 | [`website/index.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/website/index.md)、[`README.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/README.md)、[`manuscripts/README.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/README.md)、[`manuscripts/00-前言.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/00-%E5%89%8D%E8%A8%80.md) | “十四章导读”与章节目录表（硬编码 14 章） | 全面升级为“十六章导读”，在第三部分加入第 15 章与第 16 章导读及三类读者定制阅读路径（`D11-SP02`），并更新 `ch01/figure-1-book-roadmap.svg`（全书结构路线图）以涵盖 16 章全貌。 |
| 9 | [`case-studies/README.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/README.md) & [`experiments/README.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/README.md) | 覆盖第 1–14 章 | 新增第 15 章与第 16 章案例分组索引，并补齐 `experiments/ch13/13-01`（`design-review`、`retrieval-generation`）与 `13-07` 的正文双向引用说明。 |

---

## 六、分批执行计划（8 个 Batch）与六步闭环质量门禁（SOP）

### 6.1 八批次推进路线图（每 Batch 2 章，完成即验证、提交并推送部署）

| 批次 | 覆盖章节 | 核心交付物概要 | 新增配图 | 新增习题 |
| :-: | --- | --- | :-: | :-: |
| **Batch 1** | **第 1 章 + 第 2 章** | Ch01 补齐云 vs. AI 六维对比（`D11-01`）、四条跨层耦合（`D11-02`）、六种经典系统模式映射（`D11-03`）、Token 成本四口径与逐级比率链（`TC-01`）、双轨贯穿总表（`D11-SP01`/`SP03`）；Ch02 补齐能力密度跃迁与状态压缩前沿（`TC-02`–`TC-04`）。 | +3 幅（`1-21`~`1-22`, `2-40`） | +3 道（`1-9`~`1-10`, `2-14`） |
| **Batch 2** | **第 3 章 + 第 4 章** | Ch03 补齐推理模型成本反升悖论（`TC-05`）与生命周期训练回本线（`TC-06`）；Ch04 全面植入 **NVIDIA Rubin GPU**（$50\ \mathrm{PFLOPS}$ NVFP4, $288\ \mathrm{GB}$ HBM4 @ $22\ \mathrm{TB/s}$）、**Google TPU7x Ironwood** 双芯粒架构、Groq 3 LPX 及 M2 Max vs. RTX PRO 6000 算例（`TC-07`–`TC-08`）。 | +3 幅（`3-35`, `4-46`~`4-47`） | +3 道（`3-15`, `4-13`~`4-14`） |
| **Batch 3** | **第 5 章 + 第 6 章** | Ch05 植入 vLLM v0.6.0 多进程解耦与 10 步批量调度实证（$2.7\times$ 吞吐跃升，`TC-09`–`TC-10`）及 Metal 统一内存零拷贝路径；Ch06 补齐交换芯片 Radix 与多轨拓扑推导（`D11-04`）、铜缆/可插拔光/CPO 三维取舍（`D11-05`）、**Vera Rubin NVL72**（$260\ \mathrm{TB/s}$ NVLink 6）与 TPU7x $9,216$ 卡 Pod。 | +3 幅（`5-44`, `6-60`~`6-61`） | +2 道（`5-13`, `6-16`） |
| **Batch 4** | **第 7 章 + 第 8 章** | Ch07 植入从 TPOT 反推单次集合通信微秒级预算（$32.8\ \mu\mathrm{s} \to 8.2\ \mu\mathrm{s}$）与 MoE 尾部四因素联合验证矩阵（`D11-06`）；Ch08 补齐连续批处理、高并发推测解码退化边界与双指标百分位 SLO 下的 Goodput 成本曲线（`TC-11`–`TC-13`）。 | +2 幅（`7-65`, `8-40`） | +2 道（`7-19`, `8-14`） |
| **Batch 5** | **第 9 章 + 第 10 章** | Ch09 补齐多租户前缀缓存 `cache_salt` 安全边界（`D11-07`）、大 EP 单 Rank 故障爆炸半径与恢复时间线（`D14-04`）及 PD/KV 池化收益（`TC-14`–`TC-15`）；Ch10 补齐 Megatron SP vs. CP（`D14-01`）、**DualPipe 双向流水线** vs. 1F1B（`D14-02`）、训练侧动态路由偏置 vs. 推理 EPLB（`D14-03`）。 | +3 幅（`9-50`, `10-51`~`10-52`） | +3 道（`9-16`, `10-15`~`10-16`） |
| **Batch 6** | **第 11 章 + 第 12 章** | Ch11 补齐 API 定价与底层物理成本背离、Batch API 昼夜填谷经济学及双模型缓存路由交点（`TC-16`–`TC-17`, `I21`）；Ch12 补齐 M2 Max（$96\ \mathrm{GB}$ 统一内存）、RTX PRO 6000（$96\ \mathrm{GB}$ GDDR7）与云端 H100 三档实验设计与部署决策边界。 | +2 幅（`11-53`, `12-38`） | +2 道（`11-15`, `12-15`） |
| **Batch 7** | **第 13 章 + 第 14 章** | **Ch13 扩充至 ~1,050 行**：新增 `13.5.3`（Lance/Parquet 与 DSec 按需 EROFS `35 min` 启动）、`13.6`（Faiss/HNSW/DiskANN、近存 IKS $51.5\ \mathrm{ms}$ 与实验 `13-01` 288 请求 RAG 实测及校准失配）、`13.7`（三级 KV 缓存池、CacheBlend 非前缀融合与 $33.75\ \mathrm{DWPD}$ 闪存寿命边界）；**Ch14 扩充至 ~1,080 行**：新增 `14.1.3`（硬件彩票与 DFlash 角色逆转 `D14-05`）、`14.6`（实验 `13-01` 三候选评审与数据中心/消费卡分界）、`14.7`（实验 `13-07` K192 预注册受控证伪 `refuted_in_scope`）、`14.8`（2023–2026 百倍成本下降正交归因总表 `TC-18`）。 | +24 幅（`13-7`~`13-18`, `14-7`~`14-18`） | +12 道（`13-7`~`13-12`, `14-7`~`14-12`） |
| **Batch 8** | **第 15 章 + 第 16 章 + 全站 16 章合龙** | **新建第 15 章（~980 行）**：吉瓦级园区供电、物理/经济可行域（`D11-08`/`09`）、同步训练 Power Swing 与 Rubin 功率平滑、ASHRAE W45 液冷、LogicFolding 功率密度悖论、*TAPAS* 热感知重配置（30 请求回本）、跨城 DCI 光网与 DiLoCo 低频同步训练；**新建第 16 章（~980 行）**：双向威胁模型、CPU/GPU TEE 与 Rubin 原生 TEE-I/O、*PipeLLM* 五阶段加密流水、**DeepSeek DSec 38 万并发微虚拟机沙箱全解剖**、算例 `I19` 审计日志中断排空模型與全书 16 章大收束；同步更新构建校验脚本与导航页。 | +32 幅（`15-1`~`15-16`, `16-1`~`16-16`）+ 更新 `图 0-1` | +20 道（`15-1`~`15-10`, `16-1`~`16-10`） |

### 6.2 每一批次的六步强制执行标准作业程序（6-Step Batch SOP）

在执行上述每一个 Batch 时，必须严格按以下 6 个步骤闭环推进，任何一步未达标严禁进入下一 Batch：

1. **Step 1 · 锁定真值核对与算例复算**：读取该 Batch 涉及的所有 `references/`、`calculations/`、`experiments/`、`case-studies/` 与 `research/` 原始文件，逐一核对公式、字节数、延迟、吞吐、时间戳与配置条件，确保零脱钩、零幻觉。
2. **Step 2 · 调用 `codex exec` + `gpt-image-2.6-sunburst` (`quality="high"`) 生成出版级 420pt 配图**：
   - 使用 `codex exec` 调用 `gpt-image-2.6-sunburst`（最高质量 `quality="high"`）辅助设计并结合 [`manuscripts/figure_style/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/figure_style) 生成 `420pt` 宽度的 `.svg`、`.pdf`、`.png` 三格式配图；
   - 更新对应章节的 `manuscripts/chXX/figure-index.json`（及 `ch13`–`ch16` 的 `figure-data.json` 与 `build.py`）；
   - 用 `read_file` 逐张检视 `.png` 光栅图像，人工核查字体、对齐、留白、色彩与数值标注。
3. **Step 3 · 正文扩写与 [`refine-prose`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/.agents/skills/refine-prose/SKILL.md) 深度去 AI 腔润色**：
   - 严格遵循 [`.agents/skills/refine-prose/SKILL.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/.agents/skills/refine-prose/SKILL.md) 的铁律，消除一切生造词、欧化长句、空洞拔高与机械排比，确保行文自然晓畅、逻辑层层递进；
   - 严禁在 Markdown 正文中出现未转义的裸 `$` 货币符号（统一写为 `美元` 或在数学环境内写 `\mathrm{USD}`），防止触发 KaTeX 跨段数学模式吞字。
4. **Step 4 · 自动化脚本全量验证**：
   - 运行 `python3 scripts/typography.py` 检查并修复中英文混排间距与标点规范；
   - 运行 `python3 scripts/verify_core_principles.py` 验证章末小结、脚注闭合、本地相对链接、`figure-index.json` 顺序与历史习题号保留；
   - 运行 `python3 scripts/build_site.py && python3 scripts/check_site.py` 验证全站 HTML 构建、锚点跳转與图片光栅化零错误。
5. **Step 5 · 启动 Gemini 4 Argon High 推理强度 Subagent 独立审阅直至收敛**：
   - 调用 `invoke_subagent`（`TypeName: "self"`, `Model: "inherit"`）启动独立审阅 Subagent，对该 Batch 修改的章节正文、数学推导、新增习题和每一张新增配图的 `.png` 视觉效果进行严苛审计；
   - 对 Subagent 提出的任何阻塞项或改进建议逐一修复，并通过 `send_message` 复审（最多 5 轮），直到 Subagent 明确输出 **`Status: Approved / Converged`**。
6. **Step 6 · Git 提交与远端推送部署**：
   - 执行 `git status` 确认无临时文件残留后，按清晰规范的 Commit Message 提交本 Batch 变更并执行 `git push origin main`，触发线上阅读网站更新。
