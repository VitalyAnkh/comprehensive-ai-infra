---
name: refine-prose
description: >-
  Deeply rewrite and polish Chinese technical book manuscripts in `manuscripts/` to eliminate
  AI-coined jargon, translationese, and awkward phrasing into clear, elegant, idiomatic modern
  written Chinese, followed by an automated multi-round Subagent review-and-fix loop (up to 5 rounds
  until convergence). Use this skill whenever asked to rewrite, polish, de-AI, or refine chapters
  of the book.
---

# 书稿深度改写与多轮审校闭环（Refine Prose）

本 Skill 用于将 `manuscripts/` 目录下的书稿从带有 AI 写作痕迹、生造词或生硬翻译腔的文本，深度改写为**讲解清晰、文雅克制、符合现代汉语习惯且零技术差错**的出版级中文技术专著书面语，并通过同配置（高推理强度）Subagent 执行严格的“审校 → 修复 → 复核”闭环。

---

## 一、 三大核心改写准则

### 1. 去 AI 腔、去生造词，保留行业标准术语
- **全面清除 AI 生造词与生硬拼凑词**（正文、表格、脚注、图注 `*图 X-Y...*` 以及图片 `![alt-text](...)` 均须同步清理）：

| 原稿常见生造词 / 拗口表达 | 标准改写方案 |
| :--- | :--- |
| `上下文交互` | `注意力核心运算（QK^T 与 AV）` / `上下文匹配与汇总` / `注意力计算` |
| `逻辑参数` | `参数` / `参数量` |
| `逻辑读取量` / `逻辑数据量` / `逻辑上的数据访问` | `理论读取量` / `理论状态访问量` / `算法层面的同一次数据访问` |
| `矩阵工作` / `同类工作` | `矩阵运算` / `矩阵运算量` / `同类算子运算` |
| `首输出` / `末输出` / `前一输出` | `首个输出 token` / `最后一个输出 token` / `上一步生成的输出 token` |
| `较长的读取项` / `计算项` | `耗时更长的权重读取` / `矩阵计算下界` |
| `每输出分摊` | `平均每个输出 token 分摊的耗时（或服务时间）` |
| `参考全层前向` | `全层前向基线` |
| `负载/存储（Load/Store）` | `加载/存储（Load/Store）` |
| `power-of-two-choices（二次随机选择）` | `power-of-two-choices（二选一随机择优，通用形式为 power-of-k-choices）` |

- **保留一线工程师通用的行业标准术语**：
  - 保持英文或中英约定俗成表达：`prefill`、`decode`、`token`、`batch size`、`KV 缓存`、`算子（kernel）`、`Roofline`、`MFU`、`MBU`、`TPOT`、`TTFT`、`AllReduce`、`GQA`、`MLA`、`DSA / CSA / HCA`、`KDA`、`MoE`、`SwiGLU`、`RoPE`、`logits`、`checkpoint` 等，切勿把行业标准术语硬译成生僻中文。

### 2. 讲解清晰与循序渐进的教学法（Pedagogy & Clarity）
- **从实际工程场景出发引导读者**：每个概念与公式的引入都应先给出真实生产场景或直观物理问题（例如在什么机器配置、什么模型规模下遇到了什么具体瓶颈），再推导核心机制与数学公式，最后落脚到可手算的数值算例与翻转条件（flip conditions）。
- **打破电报式短句**：将孤立罗列事实的短句重组为遵循“物理直觉 / 问题动机 → 核心机制 / 数学推导 → 具体算例与系统资源影响”的自然段落。
- **严密交待因果中间环节与数学成立条件**：
  - 解释机制时勿跳步（例如：除以 $\sqrt{d_h}$ 是为了防止特征维度较大时点积方差过大，导致后续 **Softmax 进入饱和区**而引发梯度消失）。
  - 阐述近似或比例结论时明确前提边界（例如：矩形注意力“多算近一半无用功”须指明在 $S=0$ 的从头 prefill 场景下；讨论隐藏维度 $d$ 减半、层数 $L$ 增至 4 倍时，须区分正比于 $Ld^2$ 的线性层与 FFN 运算量不变，而正比于 $Ld$ 的注意力核心运算与 KV 状态容量会翻倍）。
  - 当局部表格沿用计算脚本中的符号（如用 $N=8192$ 表示上下文长度）而全书用 $N$ 表示总参数量、$H$ 表示上下文长度时，须在首次出现处括注说明（如 `上下文总长度为 $N=8192$ 个 token（即前文式中的可见上下文长度 $H=8192$）`）。
  - 跨章节概念（如第 1 章铺垫第 2 章的 CED 架构）必须与后文具体层数（如 20 层编码器 + 20 层解码器）及执行机制严格对齐，避免前后矛盾或同章内大段重复定义。
- **丰富前沿案例、文献引用、配图与习题**：
  - 结合仓库 `case-studies/`、`research/`、`references/` 及前沿论文/技术博客，适时补充贴合章节主题的实际系统案例与最新工业界进展，并在文末规范添加脚注引用 `[^id]`。
  - 优化并扩充图片说明（`*图 X-Y　...*`）与 `![alt-text]`，使其能够独立讲清图中各要素的物理含义、颜色图例与核心结论；在抽象机制较难直观理解处，使用 `manuscripts/figure_style`（或生图工具）新增配套图示，并同步更新 `manuscripts/chXX/figure-index.json` 及后续图号。
  - 在保留原有全部练习与实验编号的基础上，结合新增案例与推导多添加具有启发性的量化计算习题（使用 `> **练习 X-Y · ...**` 或 `> **实验 X-Y · ...**` 格式）。

### 3. 文风雅致克制：现代汉语技术书面语红线
- **感叹号零容忍（`！` 清零）**：
  - 书稿正文、列表、图注和脚注中**一律不使用感叹号 `！`**，全部采用沉稳克制的句号 `。`。
  - 克制使用“足足”“区区”“竟然”“惊人”“天壤之别”等夸张感叹修饰词。
- **严禁网络俚语与过度口语化表达**：
  - 禁用网文/口语词及替换标准：
    - `眨眼算完` $\rightarrow$ `算完`
    - `提前剧透` $\rightarrow$ `提前下发`
    - `不能偷看未来的位置` $\rightarrow$ `不能访问未来的位置`
    - `一次吃进多个 token` $\rightarrow$ `一次性处理多个 token`
    - `显存里躺着的` / `躺进上下文` $\rightarrow$ `显存中已保存的` / `已写入上下文`
    - `正式封印` $\rightarrow$ `将该块定型` / `完成封块`
    - `无脑地……硬加` $\rightarrow$ `简单地将……直接累加`
    - `老老实实地` $\rightarrow$ `完整地` / `无条件地` / `逐一`
    - `直接抄作业` / `扫外面的汪洋大海` $\rightarrow$ `跨层复用` / `扫描全量历史索引`
    - `死死固定 / 死死封顶 / 死死卡在` $\rightarrow$ `始终固定` / `严格封顶` / `被限制在`
    - `大杂烩` $\rightarrow$ `单一混合表示`
    - `断崖式砍到了区区……` $\rightarrow$ `大幅缩减至……`
    - `统治力` $\rightarrow$ `架构优势集中显现`
    - `紧紧巴巴` $\rightarrow$ `仅够容纳`
- **简体中文规范**：严禁混入繁体字（如 `時間`、`個`、`為`、`與`、`對`、`這`、`裏`、`後`、`發`、`動`、`體`、`實`、`開`、`關`、`點`、`線`、`數`、`機`、`網`、`絡`、`優`、`顯`、`緩`、`陣`、`選`、`擇`、`邏`、`輯`、`參`、`讀`、`寫` 等）。

---

## 二、 结构与数值不变量（不可违背的硬约束）

1. **数值与公式 100% 守恒**：
   - 所有数学公式、符号定义、数值、单位（严格区分十进制 `GB / MB / TB/s` 与二进制 `GiB / MiB / KiB`）必须与原稿及 `calculations/` 完全一致。
   - 检查倍数与数量级表述（例如 $1.125\ \mathrm{GiB}$ 到 $0.069\ \mathrm{GiB}$ 是缩小至十六分之一，切勿误写为“十几个数量级”）。
2. **结构校验不变量（由 `scripts/verify_core_principles.py` 强制检验）**：
   - 每章（`01`–`12`）只能有唯一的一级标题 `# 第 X 章 ...`，且必须以唯一的 `## 本章小结` 作为最后一个二级标题结尾，不得包含参考文献小节标题。
   - 所有图片引用 `![alt](chXX/...)` 的路径、出现顺序与总数量必须与 `manuscripts/chXX/figure-index.json` 完全一致；图注格式必须匹配 `*图 X-Y　...*` 且从 1 到 $N$ 连续编号。
   - 所有练习与实验块引用必须保持 `> **练习 X-Y...` 或 `> **实验 X-Y...` 格式，编号与基线完全一致。
   - 所有脚注引用 `[^id]` 与文末定义 `[^id]: ...` 必须一一对应，无未定义、无重复定义，且正文中脚注标记必须紧贴其所支撑的具体数据或句子末尾。
   - 保留所有 HTML 锚点与生成标记（如 `<a id="..."></a>`、`<!-- MODEL-COMPARISON:START -->` 等）。

---

## 三、 标准执行与多轮 Subagent 审校闭环工作流

对每一章（或指定的一组章节）执行改写时，**必须严格按以下 4 个阶段推进**：

### 阶段 1：分段精读与深度改写（Deep Rewrite）
1. 按小节（每次约 250–350 行）使用 `read_file` 精读目标章节原文。
2. 按照“一、三大核心改写准则”与“二、结构与数值不变量”进行深度改写，使用 `edit_file` 分段替换。
3. 改写每一处图注 `*图 X-Y...*` 时，同步检查并更新其正上方的 `![alt-text](chXX/...)`。

### 阶段 2：本地结构、字符与标点自检（Pre-Commit Verification）
改写完整章后，在提交 Git 前必须运行以下检查命令：

```bash
# 1. 运行核心结构校验脚本，确认目标章节在 chapters 列表中无任何报错，随后还原脚本可能改动的归档 json
python3 scripts/verify_core_principles.py
git checkout archive/reviews/core-principles-revision-2026-09-10/core-validation.json

# 2. 检查目标章节是否残留感叹号（！）或常见繁体字
python3 -c '
import pathlib, re, sys
for p in sys.argv[1:]:
    t = pathlib.Path(p).read_text()
    excls = [i+1 for i, line in enumerate(t.splitlines()) if "！" in line]
    trad = set(re.findall(r"[時間個為與對這裏後發動體實業開關點線面積數計算機網絡優化顯存緩存陣列選擇邏輯參數讀寫]", t)) - {"存", "算", "化", "列", "面"}
    print(p, "exclamations:", excls, "trad:", trad)
' "manuscripts/<目标章节文件名>.md"

# 3. 检查空白字符与 diff 格式
git diff --check "manuscripts/<目标章节文件名>.md"
```

确认感叹号列表为空（`[]`）、繁体字集合为空（`set()`）、且目标章节无结构错误后，进入阶段 3。

### 阶段 3：提交初版改写（Initial Git Commit）
仅暂存当前改写的章节文件并提交：
```bash
git add "manuscripts/<目标章节文件名>.md"
git commit -m "docs(chXX): 深度改写第 X 章，去除 AI 生造词与翻译腔并优化技术讲解可读性"
```

### 阶段 4：多轮 Subagent 审校与修复循环（Iterative Review-and-Fix Loop，最多 5 轮）

初版提交完成后，**必须启动同配置 Subagent 进行迭代审校闭环**。设当前审校轮数 `round = 1`，最大审校轮数上限 `MAX_ROUNDS = 5`。

#### 4.1 第 1 轮：启动审校 Subagent（`invoke_subagent`）
调用 `invoke_subagent` 工具，参数要求如下：
- `TypeName`: `'self'`
- `Model`: `'inherit'`（继承当前主 Agent 的高推理强度模型，如 Gemini 4 Argon High）
- `Role`: `'ChXX Prose & Technical Reviewer'`
- `Prompt`: 使用以下标准审校指令模板（填入目标章节路径与最新 commit 哈希）：

```markdown
你是一位资深 AI 基础设施专家兼中文技术图书主编。请对最新提交（commit `<COMMIT_HASH>`）中深度改写的书稿进行全面、严苛的第 1 轮审校（Review）：
- `/Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/<目标章节文件名>.md`

注意：**请只做审校（Review），不要直接修改任何文件**。请通读该章全文并结合 `git diff` 逐段核查以下三大标准：
1. **讲解清晰（Clarity & Pedagogy）**：技术逻辑与因果链条是否严密透彻？有无因改写导致的概念跳跃、前提缺失或前后矛盾？
2. **语言易懂、文雅、符合现代汉语习惯（Idiomatic & Elegant Chinese）**：
   - 正文、表格、图注及图片 `![alt-text]` 中是否还残留任何“AI 腔”、“翻译腔”或生造词（如“上下文交互”“逻辑参数”“逻辑读取量”“矩阵工作”“首输出/末输出”“读取项/计算项”等）？
   - 是否存在任何感叹号 `！`、夸张副词、口语化或网络俚语表达？句式是否自然流畅、有无语病或重复啰嗦？
3. **没有错误（Zero Technical / Numerical / Structural Errors）**：
   - 逐一比对 `git diff`，检查所有数学公式、符号定义、数值、单位（GB vs GiB、MB vs MiB 等）、比例倍数是否与原稿及真实算术完全一致，有无事实偏差、算术错误、错别字或术语误译。
   - 检查图表引用、题注编号、练习编号、脚注引用位置与定义、内部锚点链接是否完整准确。

如果发现任何问题，请按行号给出结构化的问题清单（含原句、问题分析与具体修改建议）；如果全文已完全达标、没有任何遗留问题，请明确给出“终审通过（Approved / Converged）”结论。
```

> **重要提醒**：调用 `invoke_subagent`（或后续调用 `send_message`）后，**不要轮询状态**，直接结束当前回合等待系统在 Subagent 完成时自动唤醒。

#### 4.2 循环推进：修复 → 校验 → 提交 → 复核（直到收敛或 `round == 5`）
每次收到审校 Subagent 返回的报告后，按以下分支处理：

- **情形 A：已收敛（Converged）**
  - 若审校 Subagent 明确判定“零遗留问题 / 终审通过（Approved）”，则审校闭环结束，向用户汇报最终改写与审校结果。

- **情形 B：仍有待修复问题（Not Yet Converged）**
  1. **逐项修复**：严格对照审校报告中的每一条问题，使用 `edit_file` 在目标章节中逐一修正。
  2. **重新自检**：再次运行**阶段 2** 的全部校验脚本（`verify_core_principles.py`、感叹号/繁体字扫描、`git diff --check`），确保修复未引入任何新问题。
  3. **提交修复**：將修正后的文件提交到 Git：
     ```bash
     git add "manuscripts/<目标章节文件名>.md"
     git commit -m "docs(chXX): 根据第 <round> 轮审校反馈修正技术细节与文风表达"
     ```
  4. **判断是否继续下一轮审校**：
     - 若当前已完成轮数 `round >= 5`（即已满 5 轮审校与修复），则停止循环，向用户汇报 5 轮审校与修改的完整结果；
     - 若 `round < 5`，将 `round` 加 1（`round = round + 1`），调用 `send_message` 向**同一个审校 Subagent**（传入其 `conversationId`）发送第 `round` 轮复核请求：
       ```markdown
       我已经按照你上一轮的审校清单完成了全部修复，并提交为最新 commit `<NEW_COMMIT_HASH>`（可通过 `git show <NEW_COMMIT_HASH>` 查看本轮增量修改，并结合全文通读）。

       请对第 X 章进行第 <round> 轮全面审校复核，重点检查：
       1. 上一轮指出的所有技术细节、术语、错别字、图片 `![alt-text]` 和文风问题是否均已准确修复；
       2. 本轮修改是否引入了任何新的语病、口语词、感叹号或数值/结构偏差；
       3. 全文是否还有任何遗漏的可读性或严谨性问题。

       若仍有问题请继续列出逐行清单；若已完全收敛达标，请给出“终审通过（Approved）”结论。
       ```
     - 发送消息后停止工具调用，等待 Subagent 唤醒进入下一轮判断。
