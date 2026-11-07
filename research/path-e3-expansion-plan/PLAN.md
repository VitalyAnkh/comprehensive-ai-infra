# 《AI 基础设施系统工程》路径 E3（24 章百科全书大满贯终极版）完整执行与验收主计划

> **文档状态**：Path E3 Master Plan v1.1 — **Status: Approved / Converged**（已通过 Gemini 4 Argon High 独立 Subagent [`a4319e2a-c463-42aa-9c2c-27815921f1df`](conversation://a4319e2a-c463-42aa-9c2c-27815921f1df) 两轮严苛定量与架构审阅收敛）
> **基线仓库状态**：`HEAD = b2a871ef`（路径 D 全书 20 章收官提交，`23,590` 行章节正文 + `150` 行前言 = `23,740` 行，`731` 幅 `420pt` 矢量配图，`304` 个习题块 / `303` 个唯一习题编号，`123` 个可运行实验目录 `experiments/ch01..ch20`）
> **目标终态（路径 E3 · 24 章百科全书大满贯版）**：
> - **章节规模**：从 **20 章** 扩展至 **24 章**（在保留并原地深化前 20 章的基础上，新增 **第 21 章 自主智能体（Agentic Infra）与沙箱/分层记忆底座**、**第 22 章 全双工实时语音（Full-Duplex Speech）与原生全模态流式底座**、**第 23 章 百亿亿次 Token 数据工程、GPU 模糊去重与合成数据工厂底座**、**第 24 章 科学智能（AI for Science）基础设施：生物大分子、百万碱基基因组与物理场网格系统**），全书正文从 **`23,590` 行** 扩充至 **`~32,560` 行**（含前言约 `32,740` 行，净增 **`~8,970` 行**硬核系统推导、算例、源码与习题）；
> - **出版级 420pt 矢量配图**：从 **`731` 幅** 扩充至 **`835` 幅**（前 20 章每章新增 `2` 幅共 `+40` 幅，新建第 21–24 章每章 `16` 幅共 `+64` 幅，合计净增 **`104` 幅**，且同步重绘全书 24 章路线图 `ch01/figure-1-book-roadmap.svg` 与第 20/24 章全景收束图）；
> - **全书代码实现对称化**：消除当前第 1–3、6–15 章正文代码块为 `0` 的结构性落差，为第 1–20 章补齐 **`+52` 个** 工业级可验证系统代码块（PyTorch / Triton / TileLang / C++ / Python 模拟器），配合第 21–24 章新增的 **`+24` 个** 代码块，全书达到 **`113` 个** 核心系统实现代码块（**每章均拥有 `3–9` 个代码块，消灭零代码块章节**）；
> - **全书编号算例（`> **算例 X-Y**`）、表题（`*表 X-Y*`）与章末习题锁定值速查表对称化**：将第 19–20 章广受好评的 **`> **算例 X-Y（标题）**` 独立编号算例（每章至少 4–5 个）**、**`*表 X-Y　...*` 显式表题** 与 **章末习题核心锁定值速查表** 推广至全书全部 24 章；
> - **习题与可独立运行实验套件**：全书习题从 **`304` 块（`303` 个唯一 ID）** 扩充至 **`384` 块（`383` 个唯一 ID，净增 `80` 道：前 20 章每章 `+2` 道共 `+40` 道，第 21–24 章每章 `+10` 道共 `+40` 道）**；同时为第 17–20 章补齐第 2 个独立实验目录（`17-02`、`18-02`、`19-02`、`20-02`），并为第 21–24 章各建 2 个独立实验目录（`21-01..02`、`22-01..02`、`23-01..02`、`24-01..02`），全书可运行实验目录从 **`123` 个** 增至 **`135` 个**（**确保全部 24 章每章均拥有 $\ge 2$ 个零第三方依赖、可一键 `run.py && verify.py` 通过的实验目录**）。

---

## 一、全书 24 章基线盘点、非破坏性约束与目标规格总表

### 1.1 四大非破坏性校验红线（Non-Destructive Invariants）

在执行路径 E3 的全部 12 个批次中，必须 100% 遵守以下由仓库自动化审计脚本锁定的硬约束：

1. **图号单调递增与追加位置铁律（[`scripts/verify_core_principles.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/scripts/verify_core_principles.py) & [`manuscripts/verify_teaching_book.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/verify_teaching_book.py)）**：
   - 第 1–20 章每章已有的配图（`图 X-1` 至 `图 X-N`）在文中的相对顺序与编号严禁打乱或重排；
   - 第 1–20 章每章新增的 `2` 幅配图（`图 X-(N+1)` 与 `图 X-(N+2)`）**必须严格插入在该章当前最后一幅配图（`last_fig`）所在位置之后、`## 常见误区`（或章末习题节）之前**，并同步追加到 `manuscripts/chXX/figure-index.json` 末尾；
   - 特别注意第 1–12 章的图注格式与 Alt 文本规范：[`manuscripts/verify_teaching_book.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/verify_teaching_book.py) 第 10–11 行通过正则 `r'!\[[^\]]*\]\(([^)]+)\)'` 提取图片引用，并通过 `r'^\*图 {n}-(\d+)[：\s]'` 校验独立成行的斜体图注序列 `captions == [str(i) for i in range(1, len(refs)+1)]`。因此，**第 1–12 章新增配图的 `![...]` Alt 文本内严禁出现嵌套方括号 `[...]`（以免截断正则匹配），且图片下方必须紧跟一行 `*图 X-Y：...*`（或 `*图 X-Y ...*`）独立斜体图注**；同时严禁运行会覆写历史归档报告的 `manuscripts/verify_teaching_book.py` 而不还原工作区，且**严禁修改 `manuscripts/ch01..12/sources.json`、`manuscripts/ch01..12/manifest.json`、`manuscripts/ch05/teaching_revision.py` 以及 `calculations/results/core-principles.json` 中被 SHA-256 锁定的任何历史文件**。
2. **习题计数正则与算例/速查表防碰撞铁律（[`scripts/verify_core_principles.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/scripts/verify_core_principles.py) 第 23–30 行）**：
   - `verify_core_principles.py` 第 23 行使用正则 `r'^(?:>\s*)?\*\*(?:(?:练习|实验|习题|拓展练习)\s+)?(\d+-\d+)\b(?!\s*解答)'` 统计每章习题，并在第 29–30 行检查第 1–12 章原有的 `^> \*\*(?:练习|实验) (\d+-\d+)` 集合不能丢失任何已有 ID；
   - **关键防碰撞规范**：
     1. 正文新增或标准化编号算例时，必须统一写作 **`> **算例 X-Y（算例标题）**`**（包含“算例”二字，绝不触发上述习题正则）；
     2. 章末新增“习题核心锁定值与关键公式速查表”时，必须使用标准 Markdown 表格行（如 `| 练习 X-Y | 核心条件 | 锁定答案 |`，行首为 `|`，绝不以 `**X-Y**` 顶格开头，从而避免把速查表行重复计入 `all_ex`）；
     3. 各章新增的 `2` 道习题必须严格沿用被 `verify_core_principles.py` 第 23 行正则匹配的块引用加粗头部语法（统一使用 `> **练习 X-Y...**` / `> **实验 X-Y...**` / `> **习题 X-Y...**` / `> **拓展练习 X-Y...**`，严禁使用不带 `**` 的 `### 实验 X-Y` 标题以免漏计习题数），且紧跟在该章当前最后一道习题之后。
3. **章末节名与禁用标题铁律（[`scripts/verify_core_principles.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/scripts/verify_core_principles.py) 第 12–14 行）**：
   - 全部 24 章的最后一个二级标题（`## `）必须且只能是唯一的 **`## 本章小结`**；
   - 任何二级标题（`## `）严禁包含 `参考资料`、`参考文献`、`资料说明`、`资料与`、`注释与`、`文献与`、`参考与` 等字样。
4. **排版、无 HTML 与 LaTeX 数学定界符铁律（[`book/check_typography.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/book/check_typography.py) & [`scripts/check_no_html.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/scripts/check_no_html.py)）**：
   - 严禁在 Markdown 正文中使用任何未转义的裸 `$` 货币符号（一律写为 `美元` 或在数学公式内写 `\mathrm{USD}`）；
   - 严禁在 Markdown 中残留 `<div`、`<span`、`<br>` 等裸 HTML 标签；全书 `python3 scripts/check_no_html.py`、`python3 scripts/verify_core_principles.py` 与 SVG 版式/字体转曲检查（`width="420pt"` 且 `svg.fonttype == 'path'` 无裸 `<text>` 节点）必须零错误通过。

---

### 1.2 全书 24 章现状基线 vs. 路径 E3 目标规格精确对照总表

| 章节与文件 | 当前行数 | E3 目标行数 | 当前配图（末图位置） | E3 新增配图（目标总数） | 当前代码块 $\to$ E3 目标 | 编号算例（当前 $\to$ 目标） | 当前习题（格式） $\to$ E3 目标 | 实验目录（当前 $\to$ 目标） | 所属批次 |
| --- | :-: | :-: | --- | :-: | :-: | :-: | --- | :-: | :-: |
| [`01-初识 AI Infra.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/01-%E5%88%9D%E8%AF%86%20AI%20Infra.md) | `780` | `~1,020` | `24`（`1.4.4` L639 `图 1-24`） | `+2`: `图 1-25..26` + 重绘 `图 1-1`（**`26` 幅**） | `0` $\to$ **`3`** | `0` $\to$ **`4`** | `12`（`> **练习 1-12`）$\to$ **`14`**（`+1-13..14`） | `8`（`01-01..08`）$\to$ **`8`** | **Batch 1** |
| [`02-模型架构.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/02-%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84.md) | `1,399` | `~1,650` | `42`（`2.6.3` L1007 `图 2-42`） | `+2`: `图 2-43..44`（**`44` 幅**） | `0` $\to$ **`4`** | `0` $\to$ **`5`** | `16`（`> **练习 2-16`）$\to$ **`18`**（`+2-17..18`） | `9`（`02-01..09`）$\to$ **`9`** | **Batch 1** |
| [`03-推理与训练负载.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/03-%E6%8E%A8%E7%90%86%E4%B8%8E%E8%AE%AD%E7%BB%83%E8%B4%9F%E8%BD%BD.md) | `1,019` | `~1,260` | `37`（`3.6.3` L844 `图 3-37`） | `+2`: `图 3-38..39`（**`39` 幅**） | `0` $\to$ **`3`** | `0` $\to$ **`4`** | `17`（`> **练习 3-17`）$\to$ **`19`**（`+3-18..19`） | `10`（`03-01..10`）$\to$ **`10`** | **Batch 2** |
| [`04-加速器架构.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/04-%E5%8A%A0%E9%80%9F%E5%99%A8%E6%9E%B6%E6%9E%84.md) | `1,585` | `~1,850` | `49`（`4.8.4` L1430 `图 4-49`） | `+2`: `图 4-50..51`（**`51` 幅**） | `1` $\to$ **`4`** | `0` $\to$ **`5`** | `16`（`> **实验 4-16`）$\to$ **`18`**（`+4-17..18`） | `7`（`04-01..07`）$\to$ **`7`** | **Batch 2** |
| [`05-算子与运行时.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/05-%E7%AE%97%E5%AD%90%E4%B8%8E%E8%BF%90%E8%A1%8C%E6%97%B6.md) | `1,479` | `~1,720` | `46`（`5.6.3` L1292 `图 5-46`） | `+2`: `图 5-47..48`（**`48` 幅**） | `8` $\to$ **`9`** | `0` $\to$ **`5`** | `16` 块/`15` ID（`> **实验 5-15`）$\to$ **`18` 块/`17` ID**（`+5-16..17`） | `9`（`05-01..09`）$\to$ **`9`** | **Batch 3** |
| [`06-超节点.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/06-%E8%B6%85%E8%8A%82%E7%82%B9.md) | `1,535` | `~1,780` | `63`（`6.7.4` L1369 `图 6-63`） | `+2`: `图 6-64..65`（**`65` 幅**） | `0` $\to$ **`3`** | `0` $\to$ **`5`** | `18`（`> **实验 6-18`）$\to$ **`20`**（`+6-19..20`） | `10`（`06-01..10`）$\to$ **`10`** | **Batch 3** |
| [`07-数据中心网络.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/07-%E6%95%B0%E6%8D%AE%E4%B8%AD%E5%BF%83%E7%BD%91%E7%BB%9C.md) | `1,597` | `~1,840` | `67`（`7.6.4` L1409 `图 7-67`） | `+2`: `图 7-68..69`（**`69` 幅**） | `0` $\to$ **`3`** | `0` $\to$ **`5`** | `21`（`> **实验 7-21`）$\to$ **`23`**（`+7-22..23`） | `10`（`07-01..10`）$\to$ **`10`** | **Batch 4** |
| [`08-推理优化.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/08-%E6%8E%A8%E7%90%86%E4%BC%98%E5%8C%96.md) | `1,160` | `~1,400` | `42`（`8.6.3` L1029 `图 8-42`） | `+2`: `图 8-43..44`（**`44` 幅**） | `0` $\to$ **`4`** | `0` $\to$ **`5`** | `16`（`> **练习 8-16`）$\to$ **`18`**（`+8-17..18`） | `9`（`08-01..09`）$\to$ **`9`** | **Batch 4** |
| [`09-分布式推理.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/09-%E5%88%86%E5%B8%83%E5%BC%8F%E6%8E%A8%E7%90%86.md) | `1,246` | `~1,480` | `52`（`9.7.3` L1054 `图 9-52`） | `+2`: `图 9-53..54`（**`54` 幅**） | `0` $\to$ **`3`** | `0` $\to$ **`5`** | `18`（`> **练习 9-18`）$\to$ **`20`**（`+9-19..20`） | `10`（`09-01..10`）$\to$ **`10`** | **Batch 5** |
| [`10-训练系统.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/10-%E8%AE%AD%E7%BB%83%E7%B3%BB%E7%BB%9F.md) | `1,326` | `~1,570` | `54`（`10.6.4` L1137 `图 10-54`） | `+2`: `图 10-55..56`（**`56` 幅**） | `0` $\to$ **`4`** | `0` $\to$ **`5`** | `18`（`> **习题 10-18`）$\to$ **`20`**（`+10-19..20`） | `10`（`10-01..10`）$\to$ **`10`** | **Batch 5** |
| [`11-资源调度与运行环境.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/11-%E8%B5%84%E6%BA%90%E8%B0%83%E5%BA%A6%E4%B8%8E%E8%BF%90%E8%A1%8C%E7%8E%AF%E5%A2%83.md) | `1,218` | `~1,460` | `55`（`11.5.3` L1059 `图 11-55`） | `+2`: `图 11-56..57`（**`57` 幅**） | `0` $\to$ **`3`** | `0` $\to$ **`5`** | `17`（`> **实验 11-17`）$\to$ **`19`**（`+11-18..19`） | `10`（`11-01..10`）$\to$ **`10`** | **Batch 6** |
| [`12-端边云协同.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/12-%E7%AB%AF%E8%BE%B9%E4%BA%91%E5%8D%8F%E5%90%8C.md) | `1,143` | `~1,380` | `40`（`12.5.4` L879 `图 12-40`） | `+2`: `图 12-41..42`（**`42` 幅**） | `0` $\to$ **`3`** | `0` $\to$ **`4`** | `17`（`> **拓展练习 12-17`）$\to$ **`19`**（`+12-18..19`） | `8`（`12-01..08`）$\to$ **`8`** | **Batch 6** |
| [`13-AI 原生存储系统与数据底座.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/13-AI%20%E5%8E%9F%E7%94%9F%E5%AD%98%E5%82%A8%E7%B3%BB%E7%BB%9F%E4%B8%8E%E6%95%B0%E6%8D%AE%E5%BA%95%E5%BA%A7.md) | `989` | `~1,220` | `24`（`13.8.3` L836 `图 13-24`） | `+2`: `图 13-25..26`（**`26` 幅**） | `0` $\to$ **`3`** | `12`（含 `13-1..10`）$\to$ **`14`**（`+13-11..12`） | `15`（`> **实验 13-15`）$\to$ **`17`**（`+13-16..17`） | `3`（`01,06,07`）$\to$ **`3`** | **Batch 7** |
| [`14-模型与系统硬件协同设计.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/14-%E6%A8%A1%E5%9E%8B%E4%B8%8E%E7%B3%BB%E7%BB%9F%E7%A1%AC%E4%BB%B6%E5%8D%8F%E5%90%8C%E8%AE%BE%E8%AE%A1.md) | `971` | `~1,200` | `24`（`14.9.2` L837 `图 14-24`） | `+2`: `图 14-25..26`（**`26` 幅**） | `0` $\to$ **`3`** | `11`（含 `14-1..10`）$\to$ **`13`**（`+14-11..12`） | `15`（`> **实验 14-15`）$\to$ **`17`**（`+14-16..17`） | `2`（`14-01..02`）$\to$ **`2`** | **Batch 7** |
| [`15-吉瓦级 AI 数据中心、电力热学与跨地域协同训练.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/15-%E5%90%89%E7%93%A6%E7%BA%A7%20AI%20%E6%95%B0%E6%8D%AE%E4%B8%AD%E5%BF%83%E3%80%81%E7%94%B5%E5%8A%9B%E7%83%AD%E5%AD%A6%E4%B8%8E%E8%B7%A8%E5%9C%B0%E5%9F%9F%E5%8D%8F%E5%90%8C%E8%AE%AD%E7%BB%83.md) | `967` | `~1,200` | `22`（`15.6.3` L743 `图 15-22`） | `+2`: `图 15-23..24`（**`24` 幅**） | `0` $\to$ **`3`** | `10` $\to$ **`10`** | `14`（`> **实验 15-14`）$\to$ **`16`**（`+15-15..16`） | `2`（`15-01..02`）$\to$ **`2`** | **Batch 8** |
| [`16-机密 AI 计算、可信执行环境与安全沙箱底座.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/16-%E6%9C%BA%E5%AF%86%20AI%20%E8%AE%A1%E7%AE%97%E3%80%81%E5%8F%AF%E4%BF%A1%E6%89%A7%E8%A1%8C%E7%8E%AF%E5%A2%83%E4%B8%8E%E5%AE%89%E5%85%A8%E6%B2%99%E7%AE%B1%E5%BA%95%E5%BA%A7.md) | `884` | `~1,120` | `22`（`16.6.3` L629 `图 16-22`） | `+2`: `图 16-23..24`（**`24` 幅**） | `1` $\to$ **`3`** | `10` $\to$ **`10`** | `14`（`> **实验 16-14`）$\to$ **`16`**（`+16-15..16`） | `2`（`16-01..02`）$\to$ **`2`** | **Batch 8** |
| [`17-多模态、视频生成...md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/17-%E5%A4%9A%E6%A8%A1%E6%80%81%E3%80%81%E8%A7%86%E9%A2%91%E7%94%9F%E6%88%90%EF%BC%88DiT%EF%BC%89%E4%B8%8E%E4%B8%96%E7%95%8C%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80%E8%AE%BE%E6%96%BD.md) | `1,100` | `~1,280` | `18`（`17.4.2` L968 `图 17-18`） | `+2`: `图 17-19..20`（**`20` 幅**） | `7` $\to$ **`8`** | `6`（`17-1..6`）$\to$ **`7`**（`+17-7`） | `12`（`> **实验 17-12`）$\to$ **`14`**（`+17-13..14`） | `1` $\to$ **`2`**（`+17-02`） | **Batch 9** |
| [`18-大规模后训练...md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/18-%E5%A4%A7%E8%A7%84%E6%A8%A1%E5%90%8E%E8%AE%AD%E7%BB%83%EF%BC%88RLVR%EF%BC%89%E3%80%81%E5%BC%82%E6%AD%A5%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E4%B8%8E%E6%8E%A8%E7%90%86%E6%97%B6%E6%89%A9%E5%B1%95%E7%B3%BB%E7%BB%9F.md) | `1,085` | `~1,260` | `18`（`18.4.2` L923 `图 18-18`） | `+2`: `图 18-19..20`（**`20` 幅**） | `8` $\to$ **`9`** | `8`（`18-1..8`）$\to$ **`9`**（`+18-9`） | `12`（`> **实验 18-12`）$\to$ **`14`**（`+18-13..14`） | `1` $\to$ **`2`**（`+18-02`） | **Batch 9** |
| [`19-具身智能...md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/19-%E5%85%B7%E8%BA%AB%E6%99%BA%E8%83%BD%EF%BC%88VLA%EF%BC%89%E3%80%81%E5%AE%9E%E6%97%B6%E6%8E%A7%E5%88%B6%E4%B8%8E%E5%A4%A7%E8%A7%84%E6%A8%A1%E7%89%A9%E7%90%86%E4%BB%BF%E7%9C%9F%E5%BA%95%E5%BA%A7.md) | `1,071` | `~1,250` | `16`（`19.4.3` L917 `图 19-16`） | `+2`: `图 19-17..18`（**`18` 幅**） | `7` $\to$ **`8`** | `5` $\to$ **`6`** | `10`（`> **实验 19-10`）$\to$ **`12`**（`+19-11..12`） | `1` $\to$ **`2`**（`+19-02`） | **Batch 10** |
| [`20-十万卡集群可靠性...md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/20-%E5%8D%81%E4%B8%87%E5%8D%A1%E9%9B%86%E7%BE%A4%E5%8F%AF%E9%9D%A0%E6%80%A7%E5%B7%A5%E7%A8%8B%E3%80%81%E9%9D%99%E9%BB%98%E6%95%B0%E6%8D%AE%E6%8D%9F%E5%9D%8F%EF%BC%88SDC%EF%BC%89%E9%98%B2%E5%BE%A1%E4%B8%8E%E5%9F%BA%E5%87%86%E8%AF%84%E6%B5%8B%E7%A7%91%E5%AD%A6.md) | `1,036` | `~1,220` | `16`（`20.5.2` L850 `图 20-16`） | `+2`: `图 20-17..18`（**`18` 幅**） | `5` $\to$ **`6`** | `5` $\to$ **`6`** | `10`（`> **实验 20-10`）$\to$ **`12`**（`+20-11..12`） | `1` $\to$ **`2`**（`+20-02`） | **Batch 10** |
| **新建** `21-自主智能体基础设施（Agentic Infra）、Computer Use 沙箱与分层 KV 记忆底座.md` | `0` | **`~1,120`** | `0` | **`+16`: `图 21-1..16`** | `0` $\to$ **`6`** | `0` $\to$ **`5`** | `0` $\to$ **`10`**（`21-1..10`） | `0` $\to$ **`2`**（`21-01..02`） | **Batch 11** |
| **新建** `22-全双工实时语音（Full-Duplex Speech）、神经音频编解码与原生全模态流式底座.md` | `0` | **`~1,080`** | `0` | **`+16`: `图 22-1..16`** | `0` $\to$ **`6`** | `0` $\to$ **`5`** | `0` $\to$ **`10`**（`22-1..10`） | `0` $\to$ **`2`**（`22-01..02`） | **Batch 11** |
| **新建** `23-百亿亿次 Token 数据工程、GPU 加速模糊去重与合成数据工厂底座.md` | `0` | **`~1,080`** | `0` | **`+16`: `图 23-1..16`** | `0` $\to$ **`6`** | `0` $\to$ **`5`** | `0` $\to$ **`10`**（`23-1..10`） | `0` $\to$ **`2`**（`23-01..02`） | **Batch 12** |
| **新建** `24-科学智能（AI for Science）基础设施：生物大分子、百万碱基基因组与物理场网格系统.md` | `0` | **`~1,120`** | `0` | **`+16`: `图 24-1..16`** | `0` $\to$ **`6`** | `0` $\to$ **`5`** | `0` $\to$ **`10`**（`24-1..10`） | `0` $\to$ **`2`**（`24-01..02`） | **Batch 12** |
| **全书 24 章合计** | **`23,590`** | **`~32,560`** | **`731` 幅** | **`+104` 幅（共 `835` 幅）** | **`37` $\to$ `113`** | **`67` $\to$ `152`** | **`304` 块/`303` ID $\to$ `384` 块/`383` ID** | **`123` $\to$ `135`** | **12 Batches** |

---

## 二、第 1–12 章（核心基础篇与分布式系统篇）2026–2027 前沿深化、代码补齐与算例标准化蓝图（Batches 1–6）

### 2.1 Batch 1：第 1 章 + 第 2 章

#### 第 1 章：初识 AI Infra（`780` 行 $\to$ `~1,020` 行，图 `24 -> 26` + 重绘 `图 1-1` 为 24 章全景图，代码块 `0 -> 3`，习题 `12 -> 14`）
1. **在 `1.4.4` 节末尾（`图 1-24` 之后）新增 `### 1.4.5 2026–2027 全栈范式跃迁：从 Token 制造厂走向自主智能体（Agentic）、全双工语音、数据炼金术与科学发现底座`**：
   - 将第 1 章建立的“算力—访存—通信—能耗—可靠性”五维第一性原理统一框架延伸至全书新增的四大战略疆域（第 21–24 章）：对比**单轮对话（Read-Once KV）**、**多轮智能体（Write-Once-Read-Many 90%+ 前缀复用 + 沙箱微虚机分支）**、**全双工语音（$80\ \mathrm{ms}$ 硬帧率双流并发 + 打断回滚）**、**合成数据工厂（无 TTFT 约束的极限离线吞吐/美元）** 与 **AI4Science（$\mathcal{O}(N_{\mathrm{res}}^3)$ 三角注意力与百万碱基 FFT 卷积）** 的算术强度与系统瓶颈转移；
   - 建立 **AI 工厂全栈北极星指标体系**：将机队层的 **MPG（$\mathrm{MPG}_{\mathrm{fleet}} = \mathrm{SG}_{\mathrm{sched}} \times \mathrm{RG}_{\mathrm{runtime}} \times \mathrm{PG}_{\mathrm{kernel}}$，其中运行时有效比 $\mathrm{RG}_{\mathrm{runtime}} \equiv \mathrm{ETTR}_{\mathrm{cluster}}$ 已精确扣除检查点与故障回滚损耗）**、服务层的 **SLO-Goodput（达标吞吐比）** 与应用层的 **Task-Verification-Yield（单任务验证通过率）** 统一为无重复计数的端到端乘积链：
     $$\mathrm{Efficiency}_{\mathrm{E2E}} = \underbrace{\left(\mathrm{SG}_{\mathrm{sched}} \times \mathrm{ETTR}_{\mathrm{cluster}} \times \mathrm{PG}_{\mathrm{kernel}}\right)}_{\mathrm{MPG}_{\mathrm{fleet}}} \times \frac{\mathrm{Goodput}_{\mathrm{SLO}}}{\mathrm{Throughput}_{\mathrm{raw}}} \times P(\text{Task Verified})$$
2. **补齐 3 个核心系统代码块 + 4 个标准化 `> **算例 1-1..1-4**` + 显式表题 `*表 1-1..1-9*` + 章末习题速查表**：
   - **代码块 1**：五维第一性原理（Roofline 拐点、Little 定律并发在途字节、Bisection 通信比、集群 Young-Daly 有效产出率）统一核算器（Python）；
   - **代码块 2**：跨层关键路径有向无环图（`XProf` / `ProfInfer` Host $\to$ GPU $\to$ NIC 依赖边与 Slack 松弛度求解算法）；
   - **代码块 3**：全栈北极星效率链（$\mathrm{MPG}_{\mathrm{fleet}} \times \frac{\mathrm{Goodput}_{\mathrm{SLO}}}{\mathrm{Throughput}_{\mathrm{raw}}} \times P(\text{Verified})$）敏感度与木桶短板定位器。
3. **新增 2 幅 420pt 配图（追加于 `1.4.5`，`图 1-24` 之后）并更新 `图 1-1`**：
   - 更新 `图 1-1`（`ch01/figure-1-book-roadmap.svg`）：由 [`manuscripts/ch01/build_roadmap.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/ch01/build_roadmap.py) 生成全书 **24 章四大部分**（基础篇 Ch01–07、训推系统篇 Ch08–12、前沿底座上篇 Ch13–18、前沿底座下篇与全书收官 Ch19–24）全景导读图；
   - `图 1-25`（`ch01/ch01-five-workload-paradigms-arithmetic-intensity.svg`）：五大 AI 负载范式（自回归对话、多轮 Agent 沙箱、全双工语音流、离线合成数据工厂、AI4Science 三角注意力/物理网格）在算术强度—状态驻留时长—尾延迟敏感度三维空间的分布；
   - `图 1-26`（`ch01/ch01-north-star-efficiency-waterfall-chain.svg`）：从铭牌峰值算力（Peak PFLOPS）经机队 MPG（含集群 ETTR）、SLO Goodput 到单次验证成功任务的四级效率折损瀑布图。
4. **新增 2 道习题（`1-13`、`1-14`，追加于 `> **练习 1-12` 之后）**：
   - `> **练习 1-13〔延伸〕：五大 AI 负载范式下的系统主瓶颈迁移与北极星效率链核算**`
   - `> **练习 1-14〔延伸〕：结合 `experiments/ch01/` 验证跨层关键路径 Slack 松弛度与硬件局部升级的阿姆达尔收益上限**`

#### 第 2 章：模型架构（`1,399` 行 $\to$ `~1,650` 行，图 `42 -> 44`，代码块 `0 -> 4`，习题 `16 -> 18`）
1. **在 `2.6.3` 节末尾（`图 2-42` 之后）新增两节 2026–2027 颠覆性模型架构基础设施剖析**：
   - **`### 2.6.4` 打破自回归串行锁链：扩散大语言模型（Diffusion LLMs：`Mercury` / `LLaDA` / `Dream` / `Cola DLM`）的并行去噪计算图与块因果 KV 缓存**：
     - 剖析自回归（AR）解码为何每步只能处理 1 个 Token（算术强度 $\mathcal{I} \approx B \ll \mathcal{I}_{\mathrm{ridge}}$，死锁在 HBM 带宽墙上），而离散扩散语言模型（dLLM）在单步前向中同时并行更新长度为 $L_{\mathrm{blk}} = 64\text{–}512$ 的掩码块（经 $K_{\mathrm{denoise}} = 8\text{–}32$ 步去噪生成整段文本），将解码算术强度瞬间提升 $L_{\mathrm{blk}}$ 倍，直接从访存受限跃迁至 Tensor Core 计算密集区（实测突破 `1,000+ tokens/s`）；
     - 推导 **半自回归块扩散（Block Diffusion / `SGLang dLLM`）** 的显存与算力折中：对已定稿的历史块缓存精确 KV（无需重算历史），仅在当前活跃块 $L_{\mathrm{blk}}$ 内执行双向全注意力迭代去噪；给出 dLLM 相较 AR 解码实现端到端延迟反超的临界去噪步数判据：
       $$K_{\mathrm{denoise}} < \frac{L_{\mathrm{blk}}}{\max\!\left(1,\; \frac{L_{\mathrm{blk}}}{\mathcal{I}_{\mathrm{ridge}} / \mathcal{I}_{\mathrm{AR}}(B)}\right)}$$
   - **`### 2.6.5` 告别固定词表分词器：字节潜空间 Transformer（`BLT` Byte Latent Transformer）的熵驱动动态分块（Entropy-Based Dynamic Patching）与变长算子调度**：
     - 剖析传统 BPE 分词器的三大系统痛点（多语言分词膨胀率差异达 $3\text{–}5\times$、词表头 $V=150\mathrm{K}\text{–}256\mathrm{K}$ 占据海量显存带宽、无法直接处理原始字节流/二进制数据）；
     - 详解 **BLT（Byte Latent Transformer）** 的三级架构：（1）轻量级局部字节编码器（Local Byte Encoder，滑动窗口跨字节交叉注意力）、（2）基于微型字节语言模型预测熵 $H(b_t \mid b_{<t}) > \theta_H$ 动态切分潜空间 Patch 的主干全局 Transformer（高可预测片段如常见词干自动合并为长 Patch，高熵突变处按单字节精细处理）、（3）局部字节解码器；推导动态 Patch 长度方差对序列打包（Ragged Tensor / `varlen` FlashAttention）与 KV 缓存预算的影响。
2. **补齐 4 个核心系统代码块 + 5 个标准化 `> **算例 2-1..2-5**` + 显式表题 `*表 2-1..2-40*` + 章末习题速查表**：
   - **代码块 1**：MHA vs. GQA vs. MLA（矩阵吸收前后）vs. KDA（Kimi 线性注意力）单 Token 状态字节与算术强度精确核查器；
   - **代码块 2**：DeepSeek NSA（压缩 + 选择 + 滑窗三路）与 Kimi MoBA（64-Token 块无参门控）硬件块对齐模拟器；
   - **代码块 3**：半自回归块扩散语言模型（Block Diffusion LLM）与自回归（AR）解码的 Roofline 延迟对比实现；
   - **代码块 4**：混合线性+全注意力模型（69 层 KDA + 24 层 MLA）前缀快照间隔（`Marconi` / SGLang `Unified Radix Cache`）最优寻优器。
3. **新增 2 幅 420pt 配图（追加于 `2.6.4`–`2.6.5`，`图 2-42` 之后）**：
   - `图 2-43`（`ch02/ch02-autoregressive-vs-block-diffusion-llm-roofline.svg`）：自回归逐词解码（算术强度锁死于访存墙）与块扩散语言模型（dLLM：块内双向并行去噪 + 历史块 KV 缓存复用）的时序与 Roofline 跃迁对比；
   - `图 2-44`（`ch02/ch02-byte-latent-transformer-entropy-dynamic-patching.svg`）：字节潜空间 Transformer（BLT）基于条件信息熵阈值 $\theta_H$ 的动态字节打包（Dynamic Byte Patching）及三层异构主干数据流。
4. **新增 2 道习题（`2-17`、`2-18`，追加于 `> **练习 2-16` 之后）**：
   - `> **练习 2-17〔延伸〕：块扩散语言模型（Block dLLM）在 H200 上的算术强度跃迁与去噪步数盈亏平衡点核算**`
   - `> **练习 2-18〔延伸〕：字节潜空间模型（BLT）动态熵阈值 $\theta_H$ 对平均 Patch 压缩比与全局主干 FLOPs 的定量调控**`

---

### 2.2 Batch 2：第 3 章 + 第 4 章

#### 第 3 章：推理与训练负载（`1,019` 行 $\to$ `~1,260` 行，图 `37 -> 39`，代码块 `0 -> 3`，习题 `17 -> 19`）
1. **在 `3.6.3` 节末尾（`图 3-37` 之后）新增两节 2026–2027 生产负载新规律**：
   - **`### 3.6.4` 推理时计算扩展定律（Inference-Time Compute Scaling Laws）与全生命周期最佳模型尺寸左移**：
     - 当推理侧引入长思维链（Long-CoT）与并行验证搜索（Best-of-$N$）后，单次复杂查询消耗的推理 Token 数从 $500$ 飙升至 $32,000\text{–}128,000$；推导在给定总算力预算或总生命周期 TCO 下，为何“用更激进的训练 Token 倍数（如 $100\text{–}200\times$ Chinchilla）过度训练一个更小参数量的稠密或小激活 MoE 模型，再配合测试时搜索”能显著击败未经充分训练的超大稠密模型；
   - **`### 3.6.5` 多智能体编排风暴（Multi-Agent Orchestration Fan-Out）与级联尾延迟放大定律**：
     - 当一个用户请求在后台分裂为 $M=8\text{–}32$ 个并发子智能体（Subagents）并在屏障点（Barrier）汇合时，子任务的端到端完成时间由最慢的一个子智能体决定：$T_{\mathrm{barrier}} = \max_{1 \le i \le M} T_i$；
     - 定量推导在子任务耗时服从对数正态或帕累托重尾分布时，$M$ 路并发扇出如何将单智能体的 `P95` 延迟直接放大为编排屏障的 `P50` 中位延迟，并给出**对冲请求（Hedged Requests）与动态配额裁剪**的成本—延迟最优边界。
2. **补齐 3 个核心系统代码块 + 4 个标准化 `> **算例 3-1..3-4**` + 显式表题 `*表 3-1..3-12*` + 章末习题速查表**：
   - **代码块 1**：`ServeGen`（NSDI '26）突发相关到达流（MMPP）与双峰输出长度分布生成器及排队延迟仿真；
   - **代码块 2**：多智能体 $M$ 路并发扇出（Fan-out Barrier）尾延迟极值分布与对冲重试（Hedging）模拟器；
   - **代码块 3**：计入推理期长思维链（Test-Time Scaling）的全生命周期最优参数—训练词元配比求解器。
3. **新增 2 幅 420pt 配图（追加于 `3.6.4`–`3.6.5`，`图 3-37` 之后）**：
   - `图 3-38`（`ch03/ch03-inference-time-scaling-optimal-model-size-shift.svg`）：计入推理时思维链扩展（Test-Time Compute）后全生命周期总成本等高线与最优模型参数规模左移规律；
   - `图 3-39`（`ch03/ch03-multi-agent-fanout-tail-amplification-barrier.svg`）：多智能体 $M$ 路扇出屏障（Fan-out Barrier）对重尾延迟的极值放大效应与分位数对冲（Hedged Execution）收益。
4. **新增 2 道习题（`3-18`、`3-19`，追加于 `> **练习 3-17` 之后）**：
   - `> **练习 3-18〔延伸〕：核算 16 路并发子智能体（Subagents）屏障同步下的中位等待时间膨胀与 95 分位对冲开销**`
   - `> **练习 3-19〔延伸〕：当单请求平均输出从 512 tokens 增至 16,384 Long-CoT tokens 时，重算预训练与推理全生命周期盈亏平衡点**`

#### 第 4 章：加速器架构（`1,585` 行 $\to$ `~1,850` 行，图 `49 -> 51`，代码块 `1 -> 4`，习题 `16 -> 18`）
1. **在 `4.8.4` 节末尾（`图 4-49` 之后）新增两节 2026–2027 旗舰芯片与晶圆级/3D SRAM 架构深度剖析**：
   - **`### 4.8.5` HBM4 双倍位宽革命与 NVIDIA Vera Rubin（`VR200 NVL72` / `Rubin Ultra`）微架构拆解**：
     - **HBM4 物理层代际跃迁**：剖析为何 HBM3e 停留在单堆栈 `1024-bit` 接口，而 **HBM4** 将单堆栈 I/O 位宽翻倍至 **`2048-bit`**，并首次将 Base Die（基础逻辑裸片）从 DRAM 工艺切换为台积电先进逻辑工艺（允许集成自定义内存控制器、内建自测试与近存路由逻辑），使单颗 Rubin GPU（8 堆栈 HBM4，容量 `288 GB`）访存带宽从 B200 的 `8.0 TB/s` 暴涨至 **`13.0–22.0 TB/s`**（Rubin Ultra 12–16 堆栈 HBM4e 进一步达 `576 GB–1 TB`、`32+ TB/s`）；
     - **Vera Rubin 超异构协同**：88 核定制 ARM CPU（**Vera**，支持 `1.8 TB/s` NVLink-C2C 一致性互连）+ 第六代 **NVLink 6**（单 GPU 双向带宽 **`3.6 TB/s`**，整柜 NVL72 交换带宽 **`260 TB/s`**）+ 原生自适应压缩 **NVFP4**（单卡推理峰值 **`50 PFLOPS`**，训练 FP8 **`25–35 PFLOPS`**）；精确重算 Qwen3-235B 与 DeepSeek-V4.1 在 Rubin NVL72 上的 Roofline 拐点移动与 Decode 单步地板时间；
   - **`### 4.8.6` 绕过 HBM 带宽墙的激进路线：晶圆级与 3D 堆叠纯 SRAM 推理 ASIC（`Cerebras WSE-3` / `Groq LPU` / `d-Matrix Corsair`）及“GPU 预填充 + SRAM 芯片解码”异构分离架构**：
     - 定量对比 HBM 路线（容量大 `192–288 GB`/卡、带宽 `8–22 TB/s`、约 `0.08–0.15 美元/GB`）与片上 SRAM 路线（**Cerebras WSE-3**：单晶圆 `44 GB` 片上 SRAM、带宽高达 **`21 PB/s`**；**Groq LPU**：单芯片 `230 MB` SRAM、带宽 **`80 TB/s`**、编译期精确到时钟周期的无反应式确定性流水；**d-Matrix Corsair**：数字存内计算 DIMC SRAM，带宽 `150 TB/s`；**SambaNova SN40L**：SRAM + HBM + DDR 三级流式数据流）；
     - 推导 **异构硅片 PD 解耦（Heterogeneous Silicon Disaggregation）** 的总拥有成本（TCO）方程：由于 SRAM 极其昂贵且单芯片仅百兆字节（存放一个 70B FP8 模型需 `300+` 颗 Groq LPU 串联，绝不能浪费在计算密集的 Prefill 或长上下文 KV 存储上），工业界最优解是用 **高算力/高容量 HBM GPU 池负责 Prefill 与长上下文 KV 压缩**，再通过高速互连将激活/紧凑状态交接给 **超高速 SRAM ASIC 池执行极速 Decode（`1,500–3,000 tokens/s`）**！建立异构硅片配比与单位 Token 成本的闭合判据。
2. **补齐 3 个新增代码块（总计 4 个）+ 5 个标准化 `> **算例 4-1..4-5**` + 显式表题 `*表 4-1..4-16*` + 章末习题速查表**：
   - **代码块 2**：H100 / B200 / Vera Rubin / Ascend 950DT / Trainium3 / TPU v7 多代芯片 Roofline 与微缩放格式（NVFP4 vs. MXFP4）有效比特率计算器；
   - **代码块 3**：昇腾 950 第三代达芬奇架构片上缓冲（`L1 512 KB` $\to$ `L0A/L0B 64 KB` $\to$ `L0C 256 KB` $\to$ `UB 512 KB`）双缓冲流水与 `BufferID` 同步模拟器；
   - **代码块 4**：“HBM GPU 预填充池 + 纯 SRAM 芯片（Groq/Cerebras）解码池”异构硅片 PD 分离的机柜数量、延迟与单百万 Token 成本寻优器。
3. **新增 2 幅 420pt 配图（追加于 `4.8.5`–`4.8.6`，`图 4-49` 之后）**：
   - `图 4-50`（`ch04/ch04-hbm4-2048bit-logic-base-die-and-vera-rubin.svg`）：HBM3e（1024-bit）向 HBM4（2048-bit + 定制逻辑 Base Die）的物理接口跃迁及 NVIDIA Vera Rubin NVL72 超节点架构；
   - `图 4-51`（`ch04/ch04-sram-asic-vs-hbm-gpu-heterogeneous-pd-serving.svg`）：晶圆级/纯 SRAM 芯片（WSE-3 / Groq LPU）与 HBM GPU 的容量—带宽帕累托边界及“GPU Prefill + SRAM ASIC Decode”异构分离架构。
4. **新增 2 道习题（`4-17`、`4-18`，追加于 `> **实验 4-16` 之后）**：
   - `> **实验 4-17 · 核心：核算 HBM4（2048-bit 接口、22 TB/s）与 NVFP4 对 Vera Rubin 算术强度拐点 $\mathcal{I}_{\mathrm{ridge}}$ 及单步解码耗时的重塑**`
   - `> **实验 4-18 · 延伸：设计“B200 预填充池 + 纯 SRAM 芯片解码池”异构推理集群，求解满足 TPOT $\le 1\ \mathrm{ms}$ 的最小芯片配比与 TCO 盈亏点**`

---

### 2.3 Batch 3：第 5 章 + 第 6 章

#### 第 5 章：算子与运行时（`1,479` 行 $\to$ `~1,720` 行，图 `46 -> 48`，代码块 `8 -> 9`，习题 `16 块/15 ID -> 18 块/17 ID`）
1. **在 `5.6.3` 节末尾（`图 5-46` 之后）新增两节 2026–2027 算子 DSL 与全模型持久化超内核前沿**：
   - **`### 5.6.4` 下一代声明式算子原语：从 Triton 走向 `TileLang`（ICLR '26）与 `ThunderKittens 2.0 / ParallelKittens`**：
     - 剖析为何在 Hopper（`wgmma` + `TMA`）与 Blackwell（`tcgen05.mma` + `TMEM` + 2-CTA 协作）上，传统 Triton 编译器难以精确控制寄存器压力、异步屏障（`mbarrier`）与 Warp Group 角色分工（Producer vs. Consumer Warps）；
     - 对比 **`ThunderKittens`**（基于固定 $16\times 16$ 硬件寄存器/共享内存 Tile 的轻量 C++ 模板库，通过 `ParallelKittens` 原生封装 NVLink P2P 多模态计算通信重叠）与 **`TileLang`**（ICLR '26：将调度空间显式解耦为三级编程接口——声明式算子意图、`T.pipelined_for` 流水线深度与 `T.copy` 硬件 DMA 映射，仅用 80 行 Python 代码即可在 H100/B200/MI300X 上生成超越手写 C++ 性能的 MLA / FlashAttention-4 / 混合线性注意力内核）；
   - **`### 5.6.5` 代数张量超优化器（`Mirage`）与全模型单持久化大内核（`MPK` Mirage Persistent Mega-Kernel, OSDI '26）**：
     - 详解 **Mirage（多级代数等价超优化器）**：如何在内核图（Kernel Graph）、线程块图（Threadblock Graph）与寄存器子图（Thread Subgraph）三级抽象上通过有限域随机概率等价性验证（Probabilistic Equivalence Verification over $\mathbb{F}_p$），自动发现人类专家未曾想到的跨算子代数融合（如将 LayerNorm/RMSNorm 的平方求和吸收进前一层矩阵乘归约中）；
     - 详解 **`MPK`（Mirage Persistent Mega-Kernel）**：将大语言模型完整解码步（甚至包含跨卡 NVLink AllReduce / All-to-All）编译为**单个常驻 GPU 的持久化大内核（Single Mega-Kernel）**，利用片上共享内存/L2 作为微任务队列，让不同 SM 分工扮演计算 Worker 与通信 Worker，彻底消灭每层数十次 Kernel Launch 间隙、全局屏障与网格尾效应（Tail Effect），将单请求延迟推至逼近物理硬件极限。
2. **新增 1 个核心代码块（总计 9 个）+ 5 个标准化 `> **算例 5-1..5-5**` + 显式表题 `*表 5-1..5-14*` + 章末习题速查表**：
   - **代码块 9**：使用 `TileLang` / 持久化任务队列（Persistent Mega-Kernel `MPK`）实现“RMSNorm + QKV 投影 + 片上微任务同步”无主机介入流水线核心代码。
3. **新增 2 幅 420pt 配图（追加于 `5.6.4`–`5.6.5`，`图 5-46` 之后）**：
   - `图 5-47`（`ch05/ch05-tilelang-thunderkittens-abstraction-spectrum.svg`）：CUDA C++ / PTX、ThunderKittens $16\times 16$ 原语、TileLang 显式三级布局流水与 Triton 在硬件控制粒度与开发效率上的帕累托定位；
   - `图 5-48`（`ch05/ch05-mirage-mpk-persistent-megakernel-execution.svg`）：传统逐算子 Launch + NCCL 独立通信流（存在启动间隙与全局屏障气泡）vs. Mirage Persistent Mega-Kernel（`MPK`）单内核常驻 SM 细粒度计算—通信任务图执行。
4. **新增 2 道习题（`5-16`、`5-17`，追加于 `> **实验 5-15` 之后）**：
   - `> **实验 5-16：核算 80 层模型在传统逐层 9 次内核启动 vs. `MPK` 单持久化超内核下的主机启动税消除与尾 SM 气泡回收收益**`
   - `> **实验 5-17：对比 `TileLang` 显式双缓冲 TMA 流水与朴素单缓冲在 Blackwell `TMEM` 上的寄存器占用与带宽利用率**`

#### 第 6 章：超节点（`1,535` 行 $\to$ `~1,780` 行，图 `63 -> 65`，代码块 `0 -> 3`，习题 `18 -> 20`）
1. **在 `6.7.4` 节末尾（`图 6-63` 之后）新增两节 2026–2027 开放超节点与光电路交换重构专题**：
   - **`### 6.7.5` 开放标准全互联机柜：AMD Helios 72-GPU 机柜（Instinct MI400 / MI455X + HBM4 `432 GB`）与 UALink 1.0 内存语义交换**：
     - 深入拆解 AMD 面向 2026–2027 推出的 **Helios 整机柜超节点（72 颗 Instinct MI455X GPU）**：单芯片采用 2nm/3nm 3D 小芯片集成、搭载 **12 堆栈 HBM4（容量高达 `432 GB`/卡，带宽 `19.6 TB/s`）**，整机柜提供 **`31 TB` HBM4 总容量**与 **`1.4 EB/s` 级聚合内存带宽**；
     - 解析 **UALink 1.0（Ultra Accelerator Link）** 协议栈如何基于开放以太网物理层 SerDes（200G/通道）剥离冗长的 TCP/IP 与 RoCE 头部，构建 `64B` 固定流控单元（Flit）的低延迟 Load/Store 内存语义事务层（单卡双向带宽 `1.8–3.6 TB/s`，整机柜 Scale-Up 聚合带宽 `260 TB/s`），并与 NVIDIA NVLink 6 及华为灵衢 UB 2.0.1 进行帧开销、端到端跳数延迟（`<600 ns`）与开放生态三维对比；
   - **`### 6.7.6` 超节点拓扑的物理层自愈与动态切分：Google TPU v7（`Ironwood`）`9,216` 卡 3D Torus 与 `Palomar` 光电路交换机（OCS）**：
     - 剖析 Google 第七代 TPU **`Ironwood`（TPU v7）**：单芯片 `4,614 TFLOPS`（FP8）、`192 GiB` HBM3e（`7.37 TB/s`）、芯片间 `1.2 TB/s` 双向 ICI（Inter-Chip Interconnect），通过 `4\times 4\times 4` 液冷 Cube（64 芯片/柜）及柜间 **`Palomar` 压电/MEMS 光电路交换机（Optical Circuit Switch, OCS）** 组合成 **单 Pod `9,216` 颗芯片（`42.5 ExaFLOPS` FP8、`1.77 PB` HBM3e）** 的巨型 3D Torus 超节点；
     - 推导 **OCS 无光电转换（O-O-O）的三大工程优势**：（1）零 SerDes 重定时功耗（整台 136×136 端口 OCS 功耗仅 `~100 W`，比电交换机低 $95\%$）、（2）对传输速率（400G $\to$ 800G $\to$ 1.6T）透明升级、（3）当某个机柜发生故障或需要将 Torus 动态变形为适配特定模型 $TP \times PP \times DP$ 的扭曲环（Twisted Torus）时，毫秒级偏转微镜即可绕过坏柜重构物理拓扑。
2. **补齐 3 个核心系统代码块 + 5 个标准化 `> **算例 6-1..6-5**` + 显式表题 `*表 6-1..6-18*` + 章末习题速查表**：
   - **代码块 1**：NVLink 6（NVL72）、灵衢 UB 2.0.1（Atlas 900 SuperPoD）、UALink 1.0（AMD Helios 72）与 TPU v7 ICI+OCS 四大超节点协议有效载荷效率与 AllReduce / All-to-All 延迟计算器；
   - **代码块 2**：3D Torus 超节点中利用 OCS 光开关绕过故障 Cube 及扭曲环（Twisted Torus）伴随路由表生成器；
   - **代码块 3**：基于 `Shardy` / `PartIR` 分层代价模型的“超节点内高带宽域 + 超节点间光网”自动张量切分搜索器。
3. **新增 2 幅 420pt 配图（追加于 `6.7.5`–`6.7.6`，`图 6-63` 之后）**：
   - `图 6-64`（`ch06/ch06-ualink10-flit-protocol-and-amd-helios-rack.svg`）：UALink 1.0 内存语义事务层（64B Flit 封装）与 AMD Helios 72-GPU（`31 TB` HBM4）开放全互联机柜架构；
   - `图 6-65`（`ch06/ch06-tpu-v7-ironwood-9216-3d-torus-ocs-reconfig.svg`）：Google TPU v7（`Ironwood`）`9,216` 卡 3D Torus 超节点中 `Palomar` MEMS 光电路交换机（OCS）的动态拓扑变形与故障柜毫秒级绕行。
4. **新增 2 道习题（`6-19`、`6-20`，追加于 `> **实验 6-18` 之后）**：
   - `> **实验 6-19 · 核心：对比 NVIDIA Rubin NVL72（20.7 TB HBM4）、AMD Helios 72（31.1 TB HBM4）与华为 Atlas 950 超节点在部署万亿参数 MoE 时的单柜容量与跨柜流量消除率**`
   - `> **实验 6-20 · 延伸：核算 TPU v7 `9,216` 卡 3D Torus（$24\times 24\times 16$）在有无 OCS 光电路交换机时的环路二分带宽、能耗节省与坏柜绕行损耗**`

---

### 2.4 Batch 4：第 7 章 + 第 8 章

#### 第 7 章：数据中心网络（`1,597` 行 $\to$ `~1,840` 行，图 `67 -> 69`，代码块 `0 -> 3`，习题 `21 -> 23`）
1. **在 `7.6.5` 节之后新增两节 2026–2027 光互连与在网计算专题**（注：`图 7-67` 位于 `7.6.4` 末尾 L1409，`7.6.5` 无配图，故在 `7.6.5` 之后新增 `### 7.6.6` 与 `### 7.6.7` 放置 `图 7-68` 与 `图 7-69` 完全满足图序单调递增且小节号严格连续）：
   - **`### 7.6.6` 迈向 1.6T 时代的光电抉择：共封装光学（CPO / NVIDIA Quantum-X & Spectrum-X Photonics）vs. 线性直驱光模块（LPO）vs. 传统 DSP 可插拔光模块**：
     - 建立 10 万卡集群（含 `300,000+` 个 800G/1.6T 光端口）的**光互连功耗、误码率（BER）、链路闪断（Link Flap）与可维护性（MTTR）四维工程账本**：
       - **传统 DSP 可插拔模块**：每个 800G 模块功耗 `15–18 W`（1.6T 达 `25–30 W`），光模块总功耗高达集群网络功耗的 $55\%$，且高温下激光器老化导致频繁链路闪断；
       - **LPO（Linear Pluggable Optics）**：移除光模块内的高功耗 DSP 芯片，依赖交换芯片自身的强均衡 SerDes 直接驱动光学器件，单模块功耗立降 $40\%\text{–}50\%$、延迟降低 $100\ \mathrm{ns}$，但传输距离受限于 `<30–50 m` 且跨厂商互操作校准苛刻；
       - **CPO（Co-Packaged Optics，如台积电 COUPE 硅光引擎 + NVIDIA Spectrum-X / Broadcom Bailly）**：将硅光引擎直接封装在交换 ASIC 基板旁（电走线从 $30\ \mathrm{cm}$ 缩短至 $<1\ \mathrm{cm}$），配合外置可插拔激光源（ELSFP），将每比特传输能耗从 `18 pJ/bit` 压降至 **`5 pJ/bit`**（节省 $3.5\times$ 光电功耗，可靠性提升 $10\times$），同时剖析其现场返修（RMA）与液冷冷板耦合的工程挑战；
   - **`### 7.6.7` 网内计算（In-Network Computing：`SHARP v4`）与多租户万卡网络性能隔离（`mFabric` / 硬件流量整形）**：
     - 推导交换机网内树状聚合（In-Network AllReduce）将端到端梯度同步流量从 $2\frac{P-1}{P} M$ 减半至 $1\cdot M$ 的带宽增益，以及为何当多个训练/MoE 作业共享同一套 Clos/Rail 网络时，必须结合自适应逐包喷洒与发送端令牌桶整形消除跨作业微突发拥塞扩散（Victim Flow Starvation）。
2. **补齐 3 个核心系统代码块 + 5 个标准化 `> **算例 7-1..7-5**` + 显式表题 `*表 7-1..7-9*` + 章末习题速查表**：
   - **代码块 1**：万卡 2 层/3 层 Clos 与 Rail-Only 网络的端口数量、光模块选型（DAC / LPO / DSP Pluggable / CPO）总功耗与 TCO 核算器；
   - **代码块 2**：UEC 1.0 逐包喷洒（Packet Spraying） vs. RoCEv2 ECMP 哈希碰撞在 MoE All-to-All 下的尾延迟模拟器；
   - **代码块 3**：网内归约（SHARP）与 Ring / Tree AllReduce 在不同消息大小下的 α-β 延迟交叉点求解器。
3. **新增 2 幅 420pt 配图（追加于 `7.6.6`–`7.6.7`，`图 7-67` 之后）**：
   - `图 7-68`（`ch07/ch07-pluggable-dsp-vs-lpo-vs-cpo-photonics-ledger.svg`）：传统 DSP 可插拔光模块、线性直驱（LPO）与共封装硅光（CPO + 外置激光源 ELSFP）的物理通道损耗、pJ/bit 能耗与故障率对比；
   - `图 7-69`（`ch07/ch07-sharp-in-network-aggregation-and-tenant-isolation.svg`）：交换机网内计算（SHARP 树状归约通信量减半）与多租户微突发隔离机制。
4. **新增 2 道习题（`7-22`、`7-23`，追加于 `> **实验 7-21` 之后）**：
   - `> **实验 7-22 · 核心：核算 32,768 卡 1.6T 以太网集群采用传统 DSP 光模块、LPO 与 CPO 三种方案的光学总功耗（MW）、年电费与光链路闪断期望次数**`
   - `> **实验 7-23 · 延伸：推导网内聚合（SHARP）在分层超节点（节点内 NVLink + 节点间 800G 网络）下的端到端 AllReduce 加速比与交换机槽位资源上限**`

#### 第 8 章：推理优化（`1,160` 行 $\to$ `~1,400` 行，图 `42 -> 44`，代码块 `0 -> 4`，习题 `16 -> 18`）
1. **在 `8.6.3` 节末尾（`图 8-42` 之后）新增两节 2026–2027 推理引擎前沿架构并 100% 硬化锁定算例 `I10`**：
   - **`### 8.6.4` 有损 KV 压缩的输出长度膨胀反例（锁定算例 `I10` 完全体）与注意力流形保持校准**：
     - 完整展开 [`research/2026-infra-survey/interview-directions.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/2026-infra-survey/interview-directions.md) 第 50 行的 `I10` 生产级反直觉定理与锁定数值算例：单看每一步解码吞吐，激进的有损 KV 驱逐/低比特量化确实能将解码速率从 $50\ \mathrm{tok/s}$（单步 $20\ \mathrm{ms}$）提升至 $60\ \mathrm{tok/s}$（单步 $16.67\ \mathrm{ms}$）；然而在长链数学/代码推理中，丢失远端上下文细节会导致模型陷入自我怀疑、反复验算与啰嗦回溯，使输出 Token 数从 $L_0 = 200$ 膨胀至 $L_{\mathrm{comp}} = 300$（膨胀率 $\eta_{\mathrm{len}} = +50\%$）！在固定 $\mathrm{TTFT} = 0.5\ \mathrm{s}$（含首个输出 Token）下，未压缩基线端到端耗时为 $T_0 = 0.5 + \frac{200-1}{50} = \mathbf{4.48\ \mathrm{s}}$，而有损 KV 压缩后端到端耗时反而恶化至 $T_{\mathrm{comp}} = 0.5 + \frac{300-1}{60} \approx \mathbf{5.483\ \mathrm{s}}$（不降反升 $+22.4\%$）！给出严格的无退化准入不等式：
       $$\frac{L_{\mathrm{comp}} - 1}{L_0 - 1} = 1 + \eta_{\mathrm{len}} < \frac{v_{\mathrm{comp}}}{v_0} = \frac{T_{\mathrm{wgt}} + T_{\mathrm{KV}}^0}{T_{\mathrm{wgt}} + \gamma_{\mathrm{comp}} T_{\mathrm{KV}}^0}$$
   - **`### 8.6.5` 投机解码工业级演进闭环：从 EAGLE-3 多层特征融合、无草稿模型 N-gram/PLD 自投机到 `SGLang dLLM` 块扩散融合服务**：
     - 对比 EAGLE-3（移除特征回归损失、直接拼接目标模型低/中/高层隐藏态训练轻量草稿头，接受长度 $\tau$ 提升至 $4.5\text{–}6.2$ tokens/步）与代码编辑/RAG 场景下的零显存开销 Prompt Lookup Decoding（PLD，直接从输入上下文检索匹配后缀作为草稿），以及在 SGLang/vLLM 统一调度框架内同时承载自回归投机解码与块扩散生成（Block dLLM）的统一分页内存管理。
2. **补齐 4 个核心系统代码块 + 5 个标准化 `> **算例 8-1..8-5**` + 显式表题 `*表 8-1..8-6*` + 章末习题速查表**：
   - **代码块 1**：PagedAttention 块表分配、前缀 Radix Tree 引用计数与 Copy-on-Write（CoW）核心引擎；
   - **代码块 2**：SGLang `DSpark` 四请求不规则验证窗口（`7/3/2/2` $\to$ `total=14` tokens）与 DP 共同档位（`16/32`）Padding 核算器；
   - **代码块 3**：XGrammar 下推自动机（PDA）上下文无关字节位掩码预计算与跳跃解码（Jump-Forward）模拟器；
   - **代码块 4**：算例 `I10` 有损 KV 压缩下“单步加速（$50 \to 60\ \mathrm{tok/s}$）vs. 输出长度膨胀（$200 \to 300\ \mathrm{tokens}$，端到端 $4.48\ \mathrm{s} \to 5.483\ \mathrm{s}$）”延迟反转判定器。
3. **新增 2 幅 420pt 配图（追加于 `8.6.4`–`8.6.5`，`图 8-42` 之后）**：
   - `图 8-43`（`ch08/ch08-i10-kv-compression-length-inflation-paradox.svg`）：锁定算例 `I10`——有损 KV 压缩引发输出长度从 `200` 膨胀至 `300` tokens 导致单步加速（`50 -> 60 tok/s`）但端到端总延迟从 `4.48 s` 反升至 `5.483 s` 的相变边界；
   - `图 8-44`（`ch08/ch08-eagle3-multilayer-draft-vs-pld-lookup.svg`）：EAGLE-3 多层隐藏态融合树状投机与代码/RAG 场景无草稿模型前缀后缀检索投机（PLD）的架构与适用域对比。
4. **新增 2 道习题（`8-17`、`8-18`，追加于 `> **练习 8-16` 之后）**：
   - `> **练习 8-17 · 核心：基于算例 `I10`（$\mathrm{TTFT}=0.5\ \mathrm{s}$、基线 $50\ \mathrm{tok/s}$ 产出 $200$ tokens 耗时 $4.48\ \mathrm{s}$），复算压缩后 $60\ \mathrm{tok/s}$ 产出 $300$ tokens 的端到端耗时（$5.483\ \mathrm{s}$）及不发生延迟倒退的最大允许输出词元数 $L_{\max}$**`
   - `> **练习 8-18 · 实战：对比 EAGLE-3（草稿头开销 $1.2\ \mathrm{ms}$、平均接受长 $\tau=5.4$）与 PLD 前缀检索投机在代码补全负载下的有效加速比**`

---

### 2.5 Batch 5：第 9 章 + 第 10 章

#### 第 9 章：分布式推理（`1,246` 行 $\to$ `~1,480` 行，图 `52 -> 54`，代码块 `0 -> 3`，习题 `18 -> 20`）
1. **在 `9.7.3` 节末尾（`图 9-52` 之后）新增两节 2026–2027 分布式推理前沿专题**：
   - **`### 9.7.4` 弹性专家并行与零停机热重排（`ElasticEP` / `EPLB` 动态副本扩缩与权重预热切换）**：
     - 在大 EP（如 64–320 卡解码池）生产运行中，昼夜流量潮汐或突发热门话题会导致专家热度分布剧烈漂移；详解如何在不中断在线解码流的前提下，利用空闲 HBM 槽位后台异步拉取高频专家副本（Shadow Expert Warmup），并在单个 MoE 层边界通过原子指针切换更新路由映射表（Zero-Downtime Routing Table Swap）；
   - **`### 9.7.5` 跨机柜/跨可用区广域 PD 分离的 KV 增量流式传输与层间流水重叠（Layer-wise KV Streaming）**：
     - 证明当 Prefill 节点与 Decode 节点之间的网络带宽受限（如跨 Spine 或跨机房专线）时，无需等待全部 $L$ 层 Prefill 完成再一次性传输整块 KV 缓存；通过**逐层计算完即触发异步 RDMA Write（Layer-wise Pipelined KV Transfer）**，可将首字延迟（TTFT）中的 KV 传输暴露时间从 $\sum_{l=1}^L T_{\mathrm{xfer}}^{(l)}$ 压缩至最后一层传输时间 $T_{\mathrm{xfer}}^{(L)}$。
2. **补齐 3 个核心系统代码块 + 5 个标准化 `> **算例 9-1..9-5**` + 显式表题 `*表 9-1..9-10*` + 章末习题速查表**：
   - **代码块 1**：DeepEP 预填充两级非对称转发（RDMA $\to$ NVLink）与解码 IBGDA 零 SM 占用钩子（`hook`）通信预算器；
   - **代码块 2**：EPLB（Expert Parallelism Load Balancer）贪心分层装箱与冗余热专家副本分配算法实现；
   - **代码块 3**：Mooncake 预测性早期拒绝（Early Rejection）与逐层流水线 KV 传输（Layer-wise KV Streaming）TTFT 模拟器。
3. **新增 2 幅 420pt 配图（追加于 `9.7.4`–`9.7.5`，`图 9-52` 之后）**：
   - `图 9-53`（`ch09/ch09-eplb-shadow-expert-warmup-atomic-swap.svg`）：EPLB 在线流量漂移下的后台影子专家预热拉取与层间原子路由表无损切换时序；
   - `图 9-54`（`ch09/ch09-layerwise-pipelined-kv-transfer-ttft-hiding.svg`）：整块 KV 串行交接 vs. 逐层流水异步 RDMA 传输（Layer-wise KV Streaming）掩盖跨机传输延迟对比。
4. **新增 2 道习题（`9-19`、`9-20`，追加于 `> **练习 9-18` 之后）**：
   - `> **练习 9-19 · 计算：核算 60 层模型在 Prefill 计算耗时 $180\ \mathrm{ms}$、全量 KV 传输耗时 $120\ \mathrm{ms}$ 下，改为逐层流水传输后的 TTFT 缩减量与网卡队列深度需求**`
   - `> **练习 9-20 · 实战：实现 EPLB 热点专家副本重平衡算法，计算将 Top-4 热门专家双副本化后的最大单卡接收负载降幅**`

#### 第 10 章：训练系统（`1,326` 行 $\to$ `~1,570` 行，图 `54 -> 56`，代码块 `0 -> 4`，习题 `18 -> 20`）
1. **在 `10.6.4` 节末尾（`图 10-54` 之后）新增两节并 100% 补齐锁定算例 `I20` 与训推权重同步故障恢复闭环**：
   - **`### 10.6.5` 主机内存卸载的异步双缓冲流水账本（锁定算例 `I20` 完全体：`60.833 ms` 串行 vs. `53.833 ms` 单设备槽 vs. `35.604 ms` 双槽乒乓流水）**：
     - 精确植入 [`case-studies/host-transfer-and-buffer-lifetime.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/host-transfer-and-buffer-lifetime.md) 与 [`research/2026-infra-survey/interview-directions.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/2026-infra-survey/interview-directions.md) 第 60 行的 `I20` 生产算例：给定 **8 个 `64 MiB` 分片**（对应 Qwen3-8B 形状 `[8192, 4096]` 的 BF16 激活张量），每块需 **`1 ms` 主机侧准备**、经有效单向带宽 **`24 GiB/s`** 的 H2D 异步 DMA 引擎搬移（单块传输耗时 $\frac{64}{24\times 1024}\ \mathrm{s} \approx \mathbf{2.604167\ \mathrm{ms}}$），随后由 **GPU 消费计算 `4 ms`**：
       - 朴素同步串行（每块串行经历准备+拷贝+计算）：$8 \times (1 + 2.604167 + 4) = \mathbf{60.833\ \mathrm{ms}}$；
       - 单设备槽异步暂存（2 主机槽 + 1 设备槽）：虽然主机准备被掩盖，但因单个设备缓冲槽在 GPU 4 ms 计算期间不可被下一块 H2D 覆写，拷贝与计算仍被迫串行，总耗时 $1 + 2.604167 + 4 + 7 \times (2.604167 + 4) = \mathbf{53.833\ \mathrm{ms}}$；
       - **双槽乒乓缓冲（2 主机槽 `128 MiB` + 2 设备槽 `128 MiB`）**：彻底解耦主机准备、DMA 搬移与 GPU 计算，稳态步进周期由 $\max(1,\; 2.604167,\; 4) = 4\ \mathrm{ms}$ 主导，总耗时压降至 $1 + 2.604167 + 4 + 7 \times 4 = \mathbf{35.604\ \mathrm{ms}}$（加速 $1.71\times$；若因 PCIe 拓扑降级使有效带宽减半至 `12 GiB/s` 即单块传输 `5.2083 ms`，则瓶颈切回 DMA，双缓冲总耗时变为 $1 + 8 \times 5.208333 + 4 = \mathbf{46.667\ \mathrm{ms}}$）；
   - **`### 10.6.6` 二阶矩阵正交化优化器（`Muon` / Distributed Shampoo）的分布式通信革命与 RL 在线权重更新期间的单节点掉线重入状态机（vLLM RDT / `SkyRL`）**：
     - 剖析为何 **Muon（Newton-Schulz 正交化动量优化器，已在 Kimi K2/K3 等万亿模型预训练中规模化验证）** 能以 $1.3\text{–}1.5\times$ 样本效率击败 AdamW，以及分布式训练中如何通过张量分片收集—5 步 BF16 Newton-Schulz 矩阵乘迭代—重分发避免全参数 All-Gather 内存爆炸；
     - 补齐第 10 章与第 18 章的衔接闭环：在 RL 训练器向推理集群通过 NCCL/RDMA 广播新权重（`update_weights`）中途，若某个 Rollout 节点闪断掉线，如何通过 **通信域超时隔离 $\to$ 存活节点继续服务 $\to$ 重建恢复节点通过后台独立控制通道拉取最新版本号 $v$ 权重后再重新挂入负载均衡器（Re-admission）**。
2. **补齐 4 个核心系统代码块 + 5 个标准化 `> **算例 10-1..10-5**` + 显式表题 `*表 10-1..10-14*` + 章末习题速查表**：
   - **代码块 1**：ZeRO-1/2/3、FSDP2 与 Megatron 3D 并行（TP+PP+CP+EP）显存与通信量精确对账器；
   - **代码块 2**：DualPipe 双向交错流水线调度表与气泡率计算器；
   - **代码块 3**：锁定算例 `I20` 8×64 MiB 分片在串行（`60.833 ms`）、单设备槽（`53.833 ms`）、双槽乒乓缓冲（`35.604 ms`）与半带宽（`46.667 ms`）下的离散事件模拟器；
   - **代码块 4**：分布式 Muon 优化器 5 步 BF16 Newton-Schulz 正交化迭代与 SGLang/vLLM 权重更新六阶段状态机。
3. **新增 2 幅 420pt 配图（追加于 `10.6.5`–`10.6.6`，`图 10-54` 之后）**：
   - `图 10-55`（`ch10/ch10-i20-double-buffer-dma-staging-timeline.svg`）：锁定算例 `I20`——8 个 `64 MiB` 分片在串行（`60.833 ms`）、单设备槽阻塞（`53.833 ms`）与双槽乒乓流水（`35.604 ms`）下的精确甘特图对比；
   - `图 10-56`（`ch10/ch10-distributed-muon-newton-schulz-and-rl-readmission.svg`）：分布式 Muon 优化器分片 Newton-Schulz 正交化数据流及 RL 在线权重更新中途节点掉线与热重入（Re-admission）状态机。
4. **新增 2 道习题（`10-19`、`10-20`，追加于 `> **习题 10-18` 之后）**：
   - `> **习题 10-19 · 核心：完整复算算例 `I20` 中 8 个 `64 MiB` 分片（主机准备 `1 ms`、`24 GiB/s` 传输 `2.604 ms`、GPU 计算 `4 ms`）在串行、单设备槽与双槽缓冲下的完成时刻（验证 `60.833 ms`、`53.833 ms`、`35.604 ms` 及 `12 GiB/s` 下的 `46.667 ms`）**`
   - `> **习题 10-20 · 延伸：核算分布式 Muon 优化器对隐藏维 $4096 \times 14336$ 权重矩阵执行 5 步 BF16 Newton-Schulz 迭代的额外 FLOPs 占比（证明 $<1\%$）与通信开销**`

---

### 2.6 Batch 6：第 11 章 + 第 12 章

#### 第 11 章：资源调度与运行环境（`1,218` 行 $\to$ `~1,460` 行，图 `55 -> 57`，代码块 `0 -> 3`，习题 `17 -> 19`）
1. **在 `11.5.3` 节末尾（`图 11-55` 之后）新增两节并 100% 硬化锁定算例 `I16` 与 `I21`**：
   - **`### 11.5.4` 集群分配率幻觉 vs. 有效训练进展（锁定算例 `I16`：为何 `97.22%` 调度分配率下真实有效进展仅 `33.33%`）**：
     - 完整展开 [`research/2026-infra-survey/interview-directions.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/research/2026-infra-survey/interview-directions.md) 中的 `I16` 经典集群诊断算例：调度面板显示集群卡时分配率高达 **`97.22%`**，管理层误以为算力已被榨干；实则拆解全链路后发现，由于频繁抢占回滚、坏块重跑、数据加载饥饿与跨层同步等待，真正推动模型收敛的**有效训练进展（Effective Training Progress / MPG）仅为 `33.33%`**（近三分之二已分配卡时在空转或做无用功）！给出基于硬件计数器与 Step Goodput 的四阶审计矩阵；
   - **`### 11.5.5` 双模型前缀缓存路由的成本—延迟可行域反转（锁定算例 `I21`：`84.93%` 成本反超点与 `91.84%` 6 秒 SLO 达标阈值）与异构芯片池联合调度**：
     - 与本章既有正文（`11-资源调度与运行环境.md` L948）及 `interview-directions.md` L61 的模型命名严格对齐：**模型 A** 为基础单价低但啰嗦慢速的小模型（未命中输入 `1.0 美元/M`、命中输入 `0.1 美元/M`、输出 `4.0 美元/M`，成功率 $p_A = 0.80$，每次需生成 `2,000` tokens 且端到端耗时恒为 `10 s`），**模型 B** 为标价高 $2\times$ 但精炼快速的旗舰模型（未命中输入 `2.0 美元/M`、命中输入 `0.2 美元/M`、输出 `8.0 美元/M`，成功率 $p_B = 0.98$，每次仅需生成 `300` tokens，缓存命中时耗时 `4 s`、未命中时耗时 `12 s`）；严格推导随前缀缓存命中率 $h$ 上升，旗舰 **模型 B** 在 **$h > \frac{1291}{1520} \approx 84.93\%$** 时实现“每成功验证任务期望成本”反超廉价模型 A，并在 **$h \ge \frac{45}{49} \approx 91.84\%$** 时同时攻克 **期望延迟 $\le 6\ \mathrm{s}$ 且联合成功率 $\ge 90\%$** 的双约束可行域方程。
2. **补齐 3 个核心系统代码块 + 5 个标准化 `> **算例 11-1..11-5**` + 显式表题 `*表 11-1..11-7*` + 章末习题速查表**：
   - **代码块 1**：锁定算例 `I16` 机队分配率（`97.22%`）与四维损耗拆解（得出 `33.33%` 有效进展）审计器；
   - **代码块 2**：锁定算例 `I21` 双模型（廉价模型 A vs. 旗舰模型 B）前缀缓存命中率成本交叉点（`84.93%`）与 $6\ \mathrm{s}$ SLO 阈值（`91.84%`）求解器；
   - **代码块 3**：Serverless GPU 冷启动五阶段瀑布与 `HydraServe` / `RLBoost` 抢占式 Spot 算力弹性回收模拟器。
3. **新增 2 幅 420pt 配图（追加于 `11.5.4`–`11.5.5`，`图 11-55` 之后）**：
   - `图 11-56`（`ch11/ch11-i16-allocation-illusion-vs-effective-progress.svg`）：锁定算例 `I16`——`97.22%` 表面分配率如何经抢占回滚、I/O 等待与同步气泡层层剥蚀坍缩为 `33.33%` 真实有效训练进展；
   - `图 11-57`（`ch11/ch11-i21-dual-model-cache-hit-cost-slo-crossover.svg`）：锁定算例 `I21`——旗舰模型 B 随缓存命中率 $h$ 变化的单合格任务期望成本曲线（`84.93%` 反超模型 A）与 $6\ \mathrm{s}$ SLO 达标可行域（`91.84%` 阈值）。
4. **新增 2 道习题（`11-18`、`11-19`，追加于 `> **实验 11-17` 之后）**：
   - `> **实验 11-18：基于算例 `I16`，给定 1,000 卡集群的调度日志与检查点回滚记录，复算 `97.22%` 分配率与 `33.33%` 有效进展之间的各项损耗占比**`
   - `> **实验 11-19：基于算例 `I21`，推导廉价模型 A 与旗舰模型 B 在前缀缓存命中率 $h$ 下的单合格任务成本与延迟方程，精确验证 `1291/1520 ≈ 84.93%` 与 `45/49 ≈ 91.84%` 两个临界值**`

#### 第 12 章：端边云协同（`1,143` 行 $\to$ `~1,380` 行，图 `40 -> 42`，代码块 `0 -> 3`，习题 `17 -> 19`）
1. **在 `12.5.4` 节末尾（`图 12-40` 之后）新增两节 2026–2027 端侧 1-Bit / 三值化推理与端云隐私协同专题**：
   - **`### 12.5.5` 端侧 CPU/NPU 极低比特原生推理（`BitNet b1.58` / `llama.cpp` `TL1/TL2` 查表内核）与闪存按需流式加载（`LLM in a Flash`）**：
     - 剖析三值化权重 $\{-1, 0, +1\}$（每参数仅 `1.58 bits`）如何将浮点乘累加（FMA）彻底替换为纯整数加减法与 L1/寄存器级紧凑查表（Lookup Table, LUT，`TL1`/`TL2` 内核），使笔记本/手机 CPU 无需高功耗 GPU 即可跑出 `50–120 tokens/s` 且能耗降低 $70\%$；
     - 结合端侧 MoE 或稀疏激活模型，推导当模型总权重超出端侧 LPDDR 容量时，如何利用 UFS 4.0 / NVMe 闪存的顺序块读取特性与神经元共现捆绑（Row-Column Bundling）实现按需流式加载；
   - **`### 12.5.6` 端云协同投机解码（Edge-Draft + Cloud-Verify）与本地隐私信息脱敏网关（PII Redaction Gateway）**：
     - 当端侧 1.5B–3B 小模型能以 `<8 ms/tok` 本地生成草稿，而云端 235B/671B 大模型拥有极强逻辑能力时，如何通过局域/蜂窝网络批量上传 $K$ 个草稿 Token 交由云端单次并行验证；导出考虑无线 RTT 抖动（`20–80 ms`）后的最优上传批量 $K^*$，以及端侧小模型先做结构化隐私实体替换（PII Masking）再上云推理、返回后本地还原的零隐私泄露架构。
2. **补齐 3 个核心系统代码块 + 4 个标准化 `> **算例 12-1..12-4**` + 显式表题 `*表 12-1..12-8*` + 章末习题速查表**：
   - **代码块 1**：端侧被动散热热阻—热容微分方程（Thermal RC Model）与结温触发 DVFS 降频时刻模拟器；
   - **代码块 2**：`BitNet b1.58` 三值化权重（$\{-1,0,1\}$）4-元素打包查表（LUT `TL1`）无乘法矩阵向量乘核心实现；
   - **代码块 3**：端云协同投机解码（端侧草稿 + 无线 RTT + 云端并行验证）最优步长 $K^*$ 与流量开销求解器。
3. **新增 2 幅 420pt 配图（追加于 `12.5.5`–`12.5.6`，`图 12-40` 之后）**：
   - `图 12-41`（`ch12/ch12-bitnet-b158-ternary-lut-and-flash-bundling.svg`）：`BitNet b1.58` 三值化无乘法查表内核（`TL1`/`TL2`）内存布局与端侧闪存神经元行列捆绑流式预取；
   - `图 12-42`（`ch12/ch12-edge-draft-cloud-verify-and-pii-gateway.svg`）：端侧小模型草稿 + 云端大模型批量验证的跨网流水线及本地隐私实体脱敏还原网关（PII Redaction Gateway）。
4. **新增 2 道习题（`12-18`、`12-19`，追加于 `> **拓展练习 12-17` 之后）**：
   - `> **拓展练习 12-18：核算 7B 模型采用 `BitNet b1.58`（1.58 bits/param）相比 INT4（4.5 bits/param）在 LPDDR5X（`68 GB/s`）手机上的单步解码下界与整机续航提升**`
   - `> **拓展练习 12-19：在无线网络 RTT $= 35\ \mathrm{ms}$、端侧草稿速度 $6\ \mathrm{ms/tok}$、云端验证耗时 $12\ \mathrm{ms}$、平均接受率 $\alpha = 0.82$ 下，求端云协同投机解码的最优草稿长度 $K^*$**`

---

## 三、第 13–20 章（八大既有前沿专题）原地深化、代码补齐与实验扩容蓝图（Batches 7–10）

### 3.1 Batch 7：第 13 章 + 第 14 章

#### 第 13 章：AI 原生存储系统与数据底座（`989` 行 $\to$ `~1,220` 行，图 `24 -> 26`，代码块 `0 -> 3`，习题 `15 -> 17`）
1. **在 `13.8.3` 节末尾（`图 13-24` 之后）新增 `### 13.8.4 跨介质统一零拷贝数据面：NVIDIA NIXL（Inference Xfer Library）、PCIe P2P 盘到卡直通与纠删码（Erasure Coding）温冷检查点归档`**：
   - 解析 **NVIDIA NIXL** 如何将 GPU HBM、Host Pinned DRAM、本地 NVMe SSD（GDS）与远端 RDMA 存储池抽象为统一的非阻塞异步描述符（Transfer Descriptors），自动选择 NVLink、PCIe P2P、RoCEv2 或 GPUDirect Storage 路径；
   - 推导十万卡训练集群每小时产生 PB 级检查点时的**分层生命周期归档方程**：热层（最近 3 个 Step）驻留 NVMe 副本（3-Way Replication，秒级恢复），温冷层（历史里程碑 Checkpoint）后台异步转为 Reed-Solomon `RS(10, 4)` 纠删码（存储开销从 $3.0\times$ 压降至 $1.4\times$），核算编码 CPU/DPU 卸载带宽与恢复时间代价。
2. **补齐 3 个核心系统代码块 + 将既有 `* **算例 13-1..7**` 统一转为 `> **算例**` 块并新增 `算例 13-11..13-12`（共 14 个）+ 显式表题 `*表 13-1..13-9*` + 章末习题速查表**：
   - **代码块 1**：GPUDirect Storage（GDS `cuFileRead`）DMA 环形缓冲与 `CoPilotIO` CPU 协处理细粒度 I/O 合并调度器；
   - **代码块 2**：`FalconFS` 服务端复合路径解析（Server-side Compound Lookup）与十亿小文件 `tar/WebDataset` 索引偏移读取器；
   - **代码块 3**：多层 KV 缓存（HBM $\leftrightarrow$ Host DRAM $\leftrightarrow$ NVMe SSD）与检查点 `RS(10,4)` 纠删码归档收益计算器。
3. **新增 2 幅 420pt 配图（追加于 `13.8.4`，`图 13-24` 之后）**：
   - `图 13-25`（`ch13/ch13-nixl-unified-nonblocking-transfer-plane.svg`）：NIXL 跨 HBM、Host DRAM、本地 NVMe 与远端 RDMA 存储池的统一异步传输抽象与路径自动协商；
   - `图 13-26`（`ch13/ch13-checkpoint-replication-to-erasure-coding-lifecycle.svg`）：PB 级检查点从热层 3 副本秒级闪存暂存到温冷层 `RS(10,4)` 纠删码对象归档的容量与恢复带宽瀑布图。
4. **新增 2 道习题（`13-16`、`13-17`，追加于 `> **实验 13-15` 之后，严格保持 `> **练习/实验 X-Y` 格式）**：
   - `> **练习 13-16：核算 100,000 卡集群在每 30 分钟保存一次 `45 TB` 检查点、保留最近 4 个热副本并将每日 48 个历史快照转为 `RS(10,4)` 纠删码时的 30 天闪存容量节省。**`
   - `> **实验 13-17：对比 NIXL 异步描述符流水线在 `HBM <-> Host DRAM <-> NVMe` 三级换入换出中的尾延迟消减。**`

#### 第 14 章：模型与系统硬件协同设计（`971` 行 $\to$ `~1,200` 行，图 `24 -> 26`，代码块 `0 -> 3`，习题 `15 -> 17`）
1. **在 `14.9.2` 节末尾（`图 14-24` 之后）新增 `### 14.9.3 硬件感知神经架构搜索（HW-NAS）、异构层混合装配（Hybrid Layer Assembly）与跨代模型一键迁移蒸馏（Upcycling & Cross-Arch Distillation）`**：
   - 建立以**真实硬件延迟与显存驻留（而非纯 FLOPs）为目标函数**的混合架构协同搜索：给定目标芯片的 $(\mathcal{I}_{\mathrm{ridge}}, B_{\mathrm{HBM}}, C_{\mathrm{HBM}})$，求解全模型 $L$ 层中“多少层用滑动窗口/线性注意力（Mamba-2/KDA）、多少层保留全局注意力（MLA/GQA）、多少层插入 MoE 稀疏专家”的整数规划方程；
   - 解析从稠密模型向 MoE 或混合注意力架构升级时的**权重热启动复用（Sparse Upcycling + Attention Linearization Distillation）**，仅消耗从头预训练 $5\%\text{–}10\%$ 的算力即可完成系统友好型架构跃迁。
2. **补齐 3 个核心系统代码块 + 将既有 `* **算例 14-1..7**` 统一转为 `> **算例**` 块并新增 `算例 14-11..14-12`（共 13 个）+ 显式表题 `*表 14-1..14-10*` + 章末习题速查表**：
   - **代码块 1**：面向给定硬件 $(B_{\mathrm{HBM}}, F_{\mathrm{peak}}, C_{\mathrm{KV}})$ 的混合架构层配比（Linear Attention vs. Full Attention vs. MoE）整数规划搜索器；
   - **代码块 2**：2023–2026 百倍降本逐级条件比率链（Telescoping Waterfall $S_0 \to S_3$）与主机常数项校准器；
   - **代码块 3**：`Chakra` 执行轨迹图（ET）计算节点与集合通信节点重叠模拟及 `FarSkip-Collective` 跳层通信掩盖验证器。
3. **新增 2 幅 420pt 配图（追加于 `14.9.3`，`图 14-24` 之后）**：
   - `图 14-25`（`ch14/ch14-hardware-aware-hybrid-layer-allocation-pareto.svg`）：固定硬件显存与延迟预算下，混合层配比（线性递推层比例 $\rho_{\mathrm{lin}}$ vs. 全局注意力层 $1-\rho_{\mathrm{lin}}$）的质量—吞吐帕累托寻优曲面；
   - `图 14-26`（`ch14/ch14-sparse-upcycling-and-attention-distillation-pipeline.svg`）：稠密全注意力模型通过 Sparse Upcycling 与跨层注意力蒸馏低成本转化为系统友好型混合 MoE 架构的训练流水线。
4. **新增 2 道习题（`14-16`、`14-17`，追加于 `> **实验 14-15` 之后，严格保持 `> **练习/实验 X-Y` 格式）**：
   - `> **练习 14-16：给定单卡 KV 显存预算上限 `24 GiB` 与目标上下文长度 `256K`，求解 80 层模型中至少需将多少层替换为固定状态的线性注意力（KDA/Mamba-2）。**`
   - `> **实验 14-17：核算将 32B 稠密模型 Sparse Upcycling 为 8×32B（激活 2 专家）MoE 模型的初始化复制、路由器预热与算力回收周期。**`

---

### 3.2 Batch 8：第 15 章 + 第 16 章

#### 第 15 章：吉瓦级 AI 数据中心、电力热学与跨地域协同训练（`967` 行 $\to$ `~1,200` 行，图 `22 -> 24`，代码块 `0 -> 3`，习题 `14 -> 16`）
1. **在 `15.6.3` 节末尾（`图 15-22` 之后）新增 `### 15.6.4 800V 高压直流（HVDC）机架供电架构、固体氧化物燃料电池（SOFC）微网与余热回收区域供暖（PUE $\to$ ERE）`**：
   - 剖析当单机柜功率从 `40 kW` 跃升至 **`120–250 kW`（Rubin NVL72 / Rubin Ultra NVL144）** 时，传统 `48V` 直流母线所需的铜排电流高达 `2,500–5,200 A`（铜排重量超 `150 kg` 且 $I^2R$ 转换损耗激增）；详解 **$\pm 400\ \mathrm{V}$ / $800\ \mathrm{V}$ 高压直流（HVDC）机柜母线 + 芯片底部垂直供电（Vertical Power Delivery, VPD）** 如何将传输电流降低 $16\times$、铜耗降低 $90\%$；
   - 补充北美与欧洲吉瓦园区绕过电网并网排队（5–7 年）的**现场燃气轮机 / 固体氧化物燃料电池（SOFC）基荷微网**，以及利用液冷 $55\text{–}65\ \mathrm{^\circ C}$ 高温回水直接驱动热泵/吸收式制冷的能量复用效率（$\mathrm{ERE} = \frac{E_{\mathrm{total}} - E_{\mathrm{reused}}}{E_{\mathrm{IT}}} < 1.0$）。
2. **补齐 3 个核心系统代码块 + 显式表题 `*表 15-1..15-10*` + 章末习题速查表**：
   - **代码块 1**：`48V` vs. `800V HVDC` 机柜母线电流、铜排截面积、$I^2R$ 损耗与芯片背面微通道液冷结温 $T_j$ 求解器；
   - **代码块 2**：百兆瓦级园区储能（BESS）平抑 50 ms 训练功率骤降（$dP/dt$）与日内峰谷套利优化器；
   - **代码块 3**：`Kareus`（OSDI '26）动静态功耗紧缩合并与 `DiLoCo` / `Di-PS` 跨洲广域网外步同步带宽模拟器。
3. **新增 2 幅 420pt 配图（追加于 `15.6.4`，`图 15-22` 之后）**：
   - `图 15-23`（`ch15/ch15-48v-vs-800v-hvdc-and-vertical-power-delivery.svg`）：`120–250 kW` 超高密机柜中传统 `48V` 铜排瓶颈 vs. `800V HVDC` 高压直流母线与芯片背部垂直供电（VPD）的损耗对比；
   - `图 15-24`（`ch15/ch15-sofc-microgrid-and-heat-recovery-ere-loop.svg`）：吉瓦级 AI 园区现场 SOFC/燃气微网、BESS 储能与 $60\ \mathrm{^\circ C}$ 液冷余热回收（$\mathrm{ERE} < 1.0$）综合能量闭环。
4. **新增 2 道习题（`15-15`、`15-16`，追加于 `> **实验 15-14` 之后）**：
   - `> **练习 15-15：对比 `140 kW` 机柜在 `48V` 与 `800V HVDC` 供电下的母线电流、$I^2R$ 焦耳热损耗（设回路电阻 $0.8\ \mathrm{m\Omega}$）及所需铜排截面积。**`
   - `> **练习 15-16：核算 $100\ \mathrm{MW}$ 液冷数据中心（$\mathrm{PUE}=1.08$）回收 $65\%$ 散热功率用于园区供暖后的能量复用效率（ERE）与年碳减排量。**`

#### 第 16 章：机密 AI 计算、可信执行环境与安全沙箱底座（`884` 行 $\to$ `~1,120` 行，图 `22 -> 24`，代码块 `1 -> 3`，习题 `14 -> 16`）
1. **在 `16.6.3` 节末尾（`图 16-22` 之后）新增 `### 16.6.4 模型权重防盗取硬件绑定（Weight-Bound Encrypted Execution）、隐形抗剥离生成水印（`SynthID` 算子级融合）与提示词注入硬件级数据流隔离`**：
   - 解析私有部署场景下如何防止客户或托管机房从内存中导出百亿美金训练出的闭源模型权重：通过 **TPM / GPU 硬件熔丝密钥 + 远程证明授权租约（Lease-Bound Key Release）**，使模型权重在磁盘与主机内存中全程保持 AES-GCM-256 密文，仅在通过周期性心跳证明的 GPU TEE 内部解密执行；
   - 解析如何将大语言模型与图像/音频生成的密码学不可见水印（如基于伪随机哈希种子微调采样 Logits 的 **Tournament Sampling Watermark / SynthID**）直接融合进采样算子（Sampling Kernel）与 TEE 签名流中，使第三方无法在不破坏生成质量的前提下剥离水印。
2. **补齐 2 个新增代码块（总计 3 个）+ 显式表题 `*表 16-1..16-10*` + 章末习题速查表**：
   - **代码块 2**：MoE 跨节点 All-to-All 恒定速率哑元填充（Constant-Rate Dummy Padding）带宽税与乐观可验证推理（OPML）纳什均衡计算器；
   - **代码块 3**：采样算子内嵌密码学锦标赛水印（Cryptographic Tournament Watermarking）与 Z 分数统计检验实现。
3. **新增 2 幅 420pt 配图（追加于 `16.6.4`，`图 16-22` 之后）**：
   - `图 16-23`（`ch16/ch16-lease-bound-encrypted-weight-distribution.svg`）：私有化部署中基于远程证明租约（Lease-Bound KMS）与 GPU TEE 的闭源模型权重防导出硬件绑定流程；
   - `图 16-24`（`ch16/ch16-fused-cryptographic-logit-watermark-kernel.svg`）：在 TEE 采样算子内原位融合伪随机锦标赛水印（SynthID/Gumbel Watermark）及统计验证检测边界。
4. **新增 2 道习题（`16-15`、`16-16`，追加于 `> **实验 16-14` 之后）**：
   - `> **练习 16-15：推导 $m=4$ 层锦标赛采样水印在输出长度 $L=200$ tokens、熵值 $H=1.8\ \mathrm{bits/tok}$ 下的假阳性率上界与检测所需最小词元数。**`
   - `> **练习 16-16：设计支持断网宽限期 $\Delta t_{\mathrm{grace}} = 600\ \mathrm{s}$ 的租约绑定 TEE 权重解密状态机，核算密钥轮换与内存擦除（Zeroization）耗时。**`

---

### 3.3 Batch 9：第 17 章 + 第 18 章（含新建 `experiments/ch17/17-02` 与 `experiments/ch18/18-02`）

#### 第 17 章：多模态、视频生成（DiT）与世界模型基础设施（`1,100` 行 $\to$ `~1,280` 行，图 `18 -> 20`，代码块 `7 -> 8`，习题 `12 -> 14`，实验 `1 -> 2`）
1. **在 `17.4.2` 节末尾（`图 17-18` 之后）新增 `### 17.4.3 自回归与扩散统一架构（`Transfusion` / `Show-o` / `Ming-Omni`）的双模注意力掩码调度与 `3DGS`—`DiT` 混合神经渲染管线`**：
   - 剖析同一主干网络内**文本走因果掩码自回归（Causal AR）+ 图像/视频连续潜变量走块双向扩散（Bidirectional Block Diffusion）**（如 Meta `Transfusion`、ByteDance `Bagel`）时，如何通过分段混合掩码（Hybrid Causal-Bidirectional Mask）在单次 `FlashAttention-3/4` 调用中同时处理两种模态，以及推理期遇到 `<boi>`（图像起始符）时从逐词 Decode 无缝切换为高算术强度多步扩散去噪的池间调度；
   - 解析自动驾驶与具身世界模型中将 **3D 高斯泼溅（3DGS）显式几何脚手架** 与 **视频 DiT 外观生成** 结合的混合渲染流水线。
2. **新增 1 个 Python 代码块（总计 8 个代码块；注：`17-02` 的运行与验证命令直接合并入章末已有的 `bash` 代码块内，不额外增加代码块计数）+ 新增 `> **算例 17-7**`（并将既有 `#### 算例 17-1..6` 统一转为 `> **算例**` 块，共 7 个）+ 补齐显式表题 `*表 17-1..17-6*` + 章末习题速查表**；
3. **新增 2 幅 420pt 配图（追加于 `17.4.3`，`图 17-18` 之后）**：
   - `图 17-19`（`ch17/ch17-transfusion-hybrid-causal-bidirectional-mask.svg`）：统一多模态模型（Transfusion / Show-o）中文本因果掩码与图像块双向扩散掩码的混合排布及模式切换状态机；
   - `图 17-20`（`ch17/ch17-3dgs-scaffold-guided-world-model-dit.svg`）：显式 3DGS 几何光栅化提供深度/法线控制条件 + 视频 DiT 生成逼真物理交互画面的混合世界模型管线。
4. **新增 2 道习题（`17-13`、`17-14`）与第 2 个独立实验目录 [`experiments/ch17/17-02/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch17)**：
   - `> **练习 17-13：混合自回归—扩散计算图与模式切换调度。** ...`
   - `> **实验 17-14：运行 `experiments/ch17/17-02` 验证 E-P-D-G 四池配比与 Transfusion 混合掩码有效利用率。** ...`

#### 第 18 章：大规模后训练（RLVR）、异步强化学习与推理时扩展系统（`1,085` 行 $\to$ `~1,260` 行，图 `18 -> 20`，代码块 `8 -> 9`，习题 `12 -> 14`，实验 `1 -> 2`）
1. **在 `18.4.2` 节末尾（`图 18-18` 之后）新增 `### 18.4.3 多智能体课程强化学习、跨地域异构众包/闲置算力异步 RL（`Prime-RL` / `OpenDiLoCo-RL`）与过程奖励模型（PRM）在线自举防崩溃`**：
   - 剖析如何利用全球跨机房闲置推理集群或异构 GPU 异步生产 Rollout 轨迹，并通过压缩激活/轨迹 Token ID 与重要性比率上界过滤（IS Ratio Filtering）回传给中心 Learner 集群更新策略；
   - 解析长链 RLVR 中奖励黑客（Reward Hacking，如模型学会输出冗长伪代码骗取高分或格式崩溃）的在线统计熔断器（KL 散度突增检测 + 长度惩罚动态温度系数）。
2. **新增 1 个 Python 代码块（总计 9 个代码块；`18-02` 命令直接合并入章末已有的 `bash` 代码块内）+ 新增 `> **算例 18-9**`（并将既有 `#### 算例 18-1..8` 统一转为 `> **算例**` 块，共 9 个）+ 补齐显式表题 `*表 18-1..18-8*` + 章末习题速查表**；
3. **新增 2 幅 420pt 配图（追加于 `18.4.3`，`图 18-18` 之后）**：
   - `图 18-19`（`ch18/ch18-geo-distributed-async-rollout-harvester.svg`）：跨机房闲置推理算力异步收割 Rollout 轨迹、Token ID 极窄带宽回传与中心 Learner 重要性采样过滤架构；
   - `图 18-20`（`ch18/ch18-rlvr-reward-hacking-kl-divergence-breaker.svg`）：RLVR 训练中奖励黑客（长度失控与熵坍缩）的在线监控指标相图与自适应 KL 制动熔断机制。
4. **新增 2 道习题（`18-13`、`18-14`）与第 2 个独立实验目录 [`experiments/ch18/18-02/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch18)**：
   - `> **练习 18-13（跨地域异步 Rollout 带宽账本与策略陈旧度过滤率核算）：** ...`
   - `> **实验 18-14（运行 `experiments/ch18/18-02` 复算 DistRS 216,000 tokens 超期裁剪与 PRM CoW 树搜索显存节省）：** ...`

---

### 3.4 Batch 10：第 19 章 + 第 20 章（含新建 `experiments/ch19/19-02` 与 `experiments/ch20/20-02`）

#### 第 19 章：具身智能（VLA）、实时控制与大规模物理仿真底座（`1,071` 行 $\to$ `~1,250` 行，图 `16 -> 18`，代码块 `7 -> 8`，习题 `10 -> 12`，实验 `1 -> 2`）
1. **在 `19.4.3` 节末尾（`图 19-16` 之后）新增 `### 19.4.4 灵巧手高频视触觉融合（Visuotactile VLA，`1,000 Hz` 触觉阵列 + 全身力控 WBC）与跨本体具身基础模型（Cross-Embodiment Action Tokenizer：`FAST` 频域动作压缩）`**：
   - 剖析人形机器人全身控制（Whole-Body Control, WBC，`500–1,000 Hz` 关节力矩环）与高分辨率凝胶视触觉传感器（GelSight / Digit）接入 VLA 时的三级频率解耦（**System 2 VLM `2 Hz` $\to$ System 1 Flow-Matching `100 Hz` $\to$ 底层 MPC/WBC 全身动力学平衡环 `1,000 Hz`**）；
   - 解析 Physical Intelligence 提出的 **`FAST`（Frequency-space Action Sequence Tokenization，基于离散余弦变换 DCT 的频域动作分词器）** 为何能把高频连续动作轨迹压缩 $5\text{–}10\times$ 并消除自回归 VLA 在高采样率下的相邻步复读退化。
2. **新增 1 个 Python 代码块（总计 8 个代码块：`FAST` DCT 频域动作块编码器 + 三级控制环仿真；`19-02` 命令直接合并入章末已有的 `bash` 代码块内）+ 新增 `> **算例 19-6**`**；
3. **新增 2 幅 420pt 配图（追加于 `19.4.4`，`图 19-16` 之后）**：
   - `图 19-17`（`ch19/ch19-three-tier-vlm-flow-wbc-humanoid-control.svg`）：人形机器人三级实时控制架构（`2 Hz` VLM 语义规划 $\to$ `100 Hz` Flow-Matching 末端轨迹 $\to$ `1,000 Hz` WBC/MPC 全身力矩平衡与触觉反射）；
   - `图 19-18`（`ch19/ch19-fast-dct-frequency-action-tokenization.svg`）：时域逐帧离散化动作 Token 在高频采样下的边际信息坍缩 vs. `FAST`（离散余弦变换 DCT + BPE）频域动作压缩。
4. **新增 2 道习题（`19-11`、`19-12`）与第 2 个独立实验目录 [`experiments/ch19/19-02/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch19)**：
   - `> **练习 19-11（FAST 频域动作块 DCT 压缩率与自回归解码延迟缩减核算）**：...`
   - `> **实验 19-12（运行 `experiments/ch19/19-02` 复算真机 PTP 多传感器时标对齐、CBF 安全护盾拦截与 FAST 频域编码）**：...`

#### 第 20 章：十万卡集群可靠性工程、静默数据损坏（SDC）防御与基准评测科学（`1,036` 行 $\to$ `~1,220` 行，图 `16 -> 18`，代码块 `5 -> 6`，习题 `10 -> 12`，实验 `1 -> 2`）
1. **在 `20.5.2` 节末尾（`图 20-16` 之后）新增 `### 20.5.3 混沌故障注入演练（Chaos Fault Injection）、灰度金丝雀上线门禁与向第 21–24 章四大新兴疆域的延伸导引`**：
   - 解析十万卡机队如何通过主动混沌工程（在预生产/影子集群中向 `NCCL`、光模块驱动、HBM 页表与算子累加器确定性注入慢节点、丢包与单比特翻转）持续检验 `OpGuard`、`TrainMover` 与监控告警链路的真实召回率与切换耗时（GameDay 演练）；
   - 将原 `20.5` 的“二十章收束”无缝升级为 **“前二十章通用大模型基础设施总结与第 21–24 章四大新兴智能体/全双工/数据/科学疆域桥接”**（将全书最终 24 章大满贯总收束置于第 24 章 `24.5` 节，形成前后呼应的双枢纽结构）。
2. **新增 1 个 Python 代码块（总计 6 个代码块：NCCL 到达时间偏斜矩阵 Straggler 根因定位 + 混沌故障注入召回率评估器；`20-02` 命令直接合并入章末已有的 `bash` 代码块内）+ 新增 `> **算例 20-6**`**；
3. **新增 2 幅 420pt 配图（追加于 `20.5.3`，`图 20-16` 之后）**：
   - `图 20-17`（`ch20/ch20-chaos-fault-injection-and-canary-rollout-gate.svg`）：面向十万卡集群的混沌故障注入（SDC 比特翻转 / RoCE 灰丢包 / 慢节点）与金丝雀灰度发布验证闭环；
   - `图 20-18`（`ch20/ch20-core-cluster-to-four-new-frontiers-bridge.svg`）：从前 20 章通用大模型训练/推理底座迈向第 21–24 章（自主智能体、全双工语音、百亿亿次数据工厂、AI for Science）的架构映射桥接图。
4. **新增 2 道习题（`20-11`、`20-12`）与第 2 个独立实验目录 [`experiments/ch20/20-02/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch20)**：
   - `> **练习 20-11（集合通信 Arrival Skew Matrix 慢节点根因反演与光模块 FEC 重传灰故障定位）**：...`
   - `> **实验 20-12（运行 `experiments/ch20/20-02` 复算 Arrival Skew 慢节点定位、TrainMover 亚秒级热迁移与四大跑分失真修正）**：...`

---

## 四、新建第 21–24 章四大战略疆域旗舰章节详细蓝图（Batches 11–12）

### 4.1 Batch 11 新建第 21 章：《自主智能体基础设施（Agentic Infra）、Computer Use 沙箱与分层 KV 记忆底座》（`manuscripts/21-自主智能体基础设施（Agentic Infra）、Computer Use 沙箱与分层 KV 记忆底座.md`，`~1,120` 行，`16` 幅配图，`5` 个编号算例，`6` 个代码块，`10` 道习题，`2` 个实验目录 `21-01..02`）

- **`## 21.1 从单轮问答到百步自主执行：Agentic 负载的“一写多读（WORM）”物理法则`**
  - `### 21.1.1` **多轮工具循环（ReAct / CodeAct / SWE-Agent）的上下文雪崩与前缀复用不对称性**：定量解剖一个 40 轮自主编程/排障 Agent 的真实轨迹——每轮新增环境反馈 $\Delta L \approx 1,500\text{–}4,000$ tokens、仅生成动作指令 $L_{\mathrm{act}} \approx 120\text{–}300$ tokens；证明在 40 轮累计消耗的 `~320万` 预填充 Token 中，**高达 $92\%\text{–}96\%$ 是对历史轮次前缀的重复读取（Write-Once-Read-Many, WORM）**，而每两次推理之间却夹杂着长达 **$5\text{–}45\ \mathrm{s}$ 的外部工具执行空白期（Tool-Wait Hiatus）**；
  - `### 21.1.2` **工具等待期的显存驻留死锁：保留 HBM（高昂租金空转）vs. 丢弃重算（TTFT 爆炸）vs. 分层换出（Tiered Offload）的三岔口**：建立以工具等待时长 $t_{\mathrm{wait}}$、前缀长度 $L_{\mathrm{prefix}}$、HBM 单位秒租金 $c_{\mathrm{HBM}}$ 与 PCIe/RDMA 换入带宽 $B_{\mathrm{in}}$ 为自变量的三段式最优状态切换判据（衔接第 11 章 `2.16 s` 交叉点并推广至四级存储池）。
- **`## 21.2 构建“KV 缓存互联网”：跨节点多级 KV 路由（`NVIDIA Dynamo` / `LMCache` / `SGLang HiCache`）与张量流压缩（`CacheGen`）`**
  - `### 21.2.1` **四级分层 KV 缓存池（GPU HBM $\leftrightarrow$ Host Pinned DRAM $\leftrightarrow$ Local NVMe GDS $\leftrightarrow$ Remote RDMA Pool `Mooncake`/`InfiniStore`）的全局基数树（Distributed Radix Tree）索引**：详解 `NVIDIA Dynamo` KV 路由器与 `SGLang HiCache` 如何通过块级 SHA-256 哈希广播，在负载均衡层同时权衡**目标节点排队长度 $Q_j$** 与**本地/远端前缀命中块数 $H_j$**，最小化真实首字延迟 $\widehat{\mathrm{TTFT}}_j = T_{\mathrm{queue}}(Q_j) + T_{\mathrm{DMA}}(L_{\mathrm{hit}}^{\mathrm{host}}) + T_{\mathrm{RDMA}}(L_{\mathrm{hit}}^{\mathrm{remote}}) + T_{\mathrm{prefill}}(L_{\mathrm{miss}})$；
  - `### 21.2.2` **当网络带宽慢于重算时：面向传输的 KV 残差熵编码（`CacheGen`）与非对称块量化传输回本判据**：推导将 FP16/BF16 KV 缓存利用跨层/跨 Token 锚点差分与算术编码压缩至 `3.5–4.2 bits`（压缩比 $4\times$）后，在 $25\text{–}100\ \mathrm{Gbps}$ 跨机架网络上战胜本地 GPU 重算的临界带宽阈值。
- **`## 21.3 毫秒级分叉的 Computer-Use 与代码沙箱底座：微虚拟机快照克隆与写时复制（CoW）环境树`**
  - `### 21.3.1` **为什么容器（Docker）不够安全而传统虚拟机启动太慢：Firecracker / Cloud Hypervisor 微虚机快照（MicroVM Snapshot）与 `userfaultfd` 惰性缺页加载**：拆解冷启动一个带完整 Python/浏览器环境的沙箱需 `1.2–2.5 s`，而基于预热内存快照 + 写时复制（`MAP_PRIVATE` CoW）+ `userfaultfd` 按需拉取工作集页（Working-Set Page Hydration），可在 **`8–15 ms`** 内瞬间克隆出一个状态完全就绪的沙箱实例；
  - `### 21.3.2` **让物理环境也能像 KV 缓存一样树状回溯：面向 MCTS 环境搜索的沙箱文件系统（`OverlayFS` / `Btrfs` CoW）与内存状态同步分叉（Environment Forking）**：当智能体在 SWE-bench 或 GUI 操作中进行树状试错搜索（Tree-of-Actions）时，一旦某个分支执行了破坏性命令（如 `rm -rf` 或错误修改代码库），如何在 $10\ \mathrm{ms}$ 内将**模型 KV 缓存块表**与**沙箱微虚机内存/磁盘快照**同步回滚至父节点并分叉出新分支。
- **`## 21.4 模型上下文协议（MCP）网关、幂等事务外发箱（Transactional Outbox）与长期情节记忆图谱（`HippoRAG 2` / `Mem0`）`**
  - `### 21.4.1` **万级工具挂载下的上下文污染防御（Dynamic Tool Retrieval）与外部副作用幂等保护（Idempotency Key + Transactional Outbox）**：解决 Agent 超时重试或投机预执行导致重复下单/重复扣款的致命工程事故，并完整植入**锁定算例 `I19`（审计遥测日志中断排空：`400 KiB/s` 产生速率、`32 MiB` 缓冲在 `81.92 s` 满溢，在 `120 s` 中断累积 `46.875 MiB` 积压后以 `1 MiB/s` 上行带宽在 `76.92 s` 内排空）**；
  - `### 21.4.2` **突破上下文窗口的长期记忆底座：向量—知识图谱混合索引（`HippoRAG 2` 个性化 PageRank + `Mem0` 异步记忆巩固流水线）**。
- **`## 常见误区`、`## 练习与实验`（`21-1` ~ `21-10` + 速查表）、`## 本章小结`**

#### 第 21 章 16 幅 420pt 配图清单（`图 21-1` ~ `图 21-16`）
- `图 21-1`（`ch21/ch21-single-turn-chat-vs-multi-turn-agent-worm-loop.svg`）：单轮对话（Read-Once）与 40 轮自主智能体（WORM 95% 前缀复用 + 长工具等待空窗）的显存与算力时间线对比；
- `图 21-2`（`ch21/ch21-tool-wait-hbm-retain-recompute-offload-phase-diagram.svg`）：工具等待时长 $t_{\mathrm{wait}}$ 与前缀长度 $L_{\mathrm{prefix}}$ 构成的“HBM 驻留 vs. 分层换出 vs. 丢弃重算”三维最优决策相图；
- `图 21-3`（`ch21/ch21-four-tier-kv-cache-hierarchy-dynamo-lmcache.svg`）：GPU HBM $\leftrightarrow$ Host DRAM $\leftrightarrow$ Local NVMe $\leftrightarrow$ Remote RDMA Pool 四级分层 KV 缓存架构与带宽/容量金字塔；
- `图 21-4`（`ch21/ch21-kv-aware-global-radix-router-ttft-tradeoff.svg`）：KV 感知全局路由器（NVIDIA Dynamo / SGLang HiCache）在节点排队延迟与前缀缓存命中率之间的联合寻优；
- `图 21-5`（`ch21/ch21-cachegen-anchor-delta-entropy-kv-streaming.svg`）：CacheGen 利用层间/词间相关性进行锚点残差量化与算术编码流式传输的回本曲线；
- `图 21-6`（`ch21/ch21-microvm-snapshot-cow-userfaultfd-12ms-fork.svg`）：传统容器/虚机冷启动（`1.5 s`）vs. Firecracker 内存快照 + `userfaultfd` 惰性缺页加载实现 `12 ms` 极速克隆；
- `图 21-7`（`ch21/ch21-synchronized-kv-and-sandbox-mcts-tree-forking.svg`）：面向智能体树搜索（Tree-of-Actions）的 PagedAttention KV 块表与沙箱 OverlayFS/内存快照同步分叉与回滚；
- `图 21-8`（`ch21/ch21-computer-use-gui-webrtc-frame-diff-pipeline.svg`）：Computer-Use 桌面沙箱的脏矩形帧差分流（Dirty-Rect Streaming）、DOM 树混合提取与动作注入流水线；
- `图 21-9`（`ch21/ch21-mcp-gateway-dynamic-tool-rag-and-auth-isolation.svg`）：模型上下文协议（MCP）安全网关：按需语义检索挂载 Top-$K$ 工具 Schema 与租户级 OAuth 凭据隔离；
- `图 21-10`（`ch21/ch21-idempotent-outbox-and-speculative-tool-prefetch.svg`）：只读工具投机预取与写操作工具基于幂等键（Idempotency Key）和事务外发箱（Outbox）的精确一次执行保障；
- `图 21-11`（`ch21/ch21-i19-telemetry-backlog-buffer-overflow-and-drain.svg`）：锁定算例 `I19`——审计遥测断网期间 `400 KiB/s` 积压曲线（`81.92 s` 触顶 `32 MiB`、`120 s` 累积 `46.875 MiB`）与恢复后 `76.92 s` 排空过程；
- `图 21-12`（`ch21/ch21-prompt-injection-taint-tracking-dual-llm-firewall.svg`）：防御间接提示词注入（Indirect Prompt Injection）的特权控制流与非受信数据流双 LLM 污点追踪隔离架；
- `图 21-13`（`ch21/ch21-hipporag2-graph-vector-episodic-memory-consolidation.svg`）：HippoRAG 2 / Mem0 结合向量检索、知识图谱个性化 PageRank 与后台异步记忆提炼的长期记忆底座；
- `图 21-14`（`ch21/ch21-multi-agent-dag-scheduler-and-token-budget-allocation.svg`）：复杂软件工程任务中 Planner—Coder—Tester 多智能体 DAG 拓扑调度与动态 Token 预算熔断；
- `图 21-15`（`ch21/ch21-agentic-cluster-cpu-gpu-sandbox-pool-sizing.svg`）：大规模 Agentic 云平台中 GPU 推理池、大容量 DRAM/NVMe KV 缓存池与高密 CPU 沙箱池的黄金配比模型；
- `图 21-16`（`ch21/ch21-autonomous-agent-infrastructure-full-stack-map.svg`）：自主智能体基础设施从分层 KV 记忆、毫秒级 CoW 沙箱到幂等安全网关的全栈总图。

---

### 4.2 Batch 11 新建第 22 章：《全双工实时语音（Full-Duplex Speech）、神经音频编解码与原生全模态流式底座》（`manuscripts/22-全双工实时语音（Full-Duplex Speech）、神经音频编解码与原生全模态流式底座.md`，`~1,080` 行，`16` 幅配图，`5` 个编号算例，`6` 个代码块，`10` 道习题，`2` 个实验目录 `22-01..02`）

- **`## 22.1 跨越 $200\ \mathrm{ms}$ 人类对话感知阈值：从级联语音流水线到原生全双工（Full-Duplex S2S）`**
  - `### 22.1.1` **为什么传统 `VAD -> ASR -> LLM -> TTS` 四级串联注定延迟破秒且丢失副语言信息**：定量拆解级联架构的延迟账本（静音端点检测 VAD 尾音等待 `300–700 ms` + 流式 ASR 识别 `150–300 ms` + LLM 首字 TTFT `150–400 ms` + TTS 首包合成凑句 `200–400 ms` $\implies \mathrm{TTFA} \in [800, 1800]\ \mathrm{ms}$），且在转文字瞬间彻底抹除了语调、笑声、叹息与情绪停顿；
  - `### 22.1.2` **原生全双工语音模型（`Kyutai Moshi` / `Qwen2.5-Omni` / `GLM-4-Voice` / `GPT-4o Realtime`）的双轨常驻音频流与“内心独白（Inner Monologue）”对齐**：模型在每个固定音频帧步长（如 $\Delta t_{\mathrm{frame}} = 80\ \mathrm{ms}$，即 $12.5\ \mathrm{Hz}$）同时摄入 1 帧用户麦克风音频 Token 并吐出 1 帧助手扬声器音频 Token + 1 个内部文本锚点 Token，彻底消灭 VAD 等待（实现“边听边想边说”、随时附和 `uh-huh` 与 `<200 ms` 极速响应）。
- **`## 22.2 神经音频编解码器（Neural Audio Codecs）与“时间主干 + 深度残差”双层 Transformer 算子调度`**
  - `### 22.2.1` **残差向量量化（RVQ：`Mimi` / `EnCodec` / `SoundStream` / `SNAC`）的帧率—码本数—比特率方程**：剖析如何通过因果卷积编码器与蒸馏自 WavLM 的语义首码本（Semantic Codebook $q_1$）+ $Q-1$ 层声学残差码本（Acoustic Codebooks $q_2\dots q_Q$），在 $f_s = 12.5\ \mathrm{Hz}$ 超低帧率与 $Q=8$ 码本（每码本词表 $V_a = 2048$，即 $11\ \mathrm{bits}$）下实现仅 $1.1\ \mathrm{kbps}$ 的高保真语音流；
  - `### 22.2.2` **Temporal Transformer（跨时间步大模型）+ Depth Transformer（步内跨 $Q$ 层码本串行小模型）的 GPU 硬件困境与破局**：在每个 $80\ \mathrm{ms}$ 节拍内，7B/14B 的 Temporal Transformer 仅前向 1 次产出上下文向量 $h_t$，随后约 100M 参数的 Depth Transformer 必须沿码本维度串行自回归循环 $Q=8\text{–}32$ 步；证明**若直接逐码本启动常规内核，Depth 阶段的微矩阵乘（Batch 极小、隐藏维仅 `1024`）将触发严重的 Kernel Launch 阻塞与显存带宽饥饿**，详解如何通过 **单图全捕获（Full-Step CUDA Graph）+ 码本延迟交错（Delayed Codebook Pattern）+ 常驻 L2 缓存固定** 将单帧总计算耗时压入 `<12 ms`。
- **`## 22.3 随时打断（Barge-in）、网络抖动缓冲与“按客户端播放游标对齐”的 KV 缓存回滚协议`**
  - `### 22.3.1` **WebRTC / RTP 边缘音频网关、声学回声消除（AEC）与抗抖动缓冲（Adaptive Jitter Buffer）的时延预算**：拆解从端侧采集、Opus 帧封装、UDP/WebRTC 传输到服务端 GPU 推理并返回的完整 `160 ms` 端到端闭环账本；
  - `### 22.3.2` **全双工系统的核心一致性难题——服务端超前生成与客户端打断的游标失配（Playback-Cursor Aligned KV Rollback）**：服务端 GPU 往往比客户端实际播放进度超前生成了 $\Delta T_{\mathrm{lookahead}} = 400\text{–}1,200\ \mathrm{ms}$（即 $5\text{–}15$ 帧尚未被用户听到的语音）；一旦用户在客户端播放到 RTP 时间戳 $t_{\mathrm{play}}$ 时开口打断（Barge-in），服务端必须：(1) 立即通知边缘网关清空未播放的下行音频队列，(2) **将服务端 KV 缓存与内心独白文本历史精确截断回滚至 $t_{\mathrm{play}}$ 对应的帧索引 $k_{\mathrm{play}} = \lfloor t_{\mathrm{play}} / \Delta t_{\mathrm{frame}} \rfloor$**（丢弃模型自己说了一半但用户根本没听到的幻觉记忆），并从 $k_{\mathrm{play}}+1$ 无缝接纳用户新语音！
- **`## 22.4 实时音视频全模态流（Live Video + Full-Duplex Audio）的跨模态帧率对齐与滚动记忆压缩`**
  - `### 22.4.1` **高频音频流（$12.5\ \mathrm{Hz}$）与低频实时相机流（$1\text{–}2\ \mathrm{FPS}$）的交错分块预填充（Interleaved Chunked Prefill）与硬截止期保障**：防止每秒涌入的 $256\text{–}512$ 个视觉 Patch Token 在插入主干时撑爆当帧音频的 $80\ \mathrm{ms}$ 硬实时节拍；
  - `### 22.4.2` **长达 1 小时连续语音/视频通话的滚动 KV 压缩：声学残差码本过期剥离（Acoustic Token Eviction，仅保留文本/语义首码本锚点）**，将持续通话显存增长率降低 **$85\%$**。
- **`## 常见误区`、`## 练习与实验`（`22-1` ~ `22-10` + 速查表）、`## 本章小结`**

#### 第 22 章 16 幅 420pt 配图清单（`图 22-1` ~ `图 22-16`）
- `图 22-1`（`ch22/ch22-cascaded-asr-llm-tts-vs-native-full-duplex-s2s.svg`）：传统四级串联语音管线（`800–1,800 ms` 延迟 + 情绪丢失）与原生全双工 S2S（`160–200 ms` 双流并行）架构对比；
- `图 22-2`（`ch22/ch22-full-duplex-dual-audio-streams-and-inner-monologue.svg`）：全双工模型在 $80\ \mathrm{ms}$ 时钟节拍下同时处理用户音频流、系统音频流与内心独白文本流（Inner Monologue）的联合状态转移；
- `图 22-3`（`ch22/ch22-rvq-neural-audio-codec-semantic-acoustic-hierarchy.svg`）：神经音频编解码器（Mimi / EnCodec）中语义首码本（$q_1$）与多级声学残差码本（$q_2\dots q_Q$）的残差向量量化（RVQ）流程；
- `图 22-4`（`ch22/ch22-delayed-codebook-pattern-vs-depth-transformer.svg`）：RVQ 多码本生成的两种范式：时间步错位延迟模式（MusicGen Delayed Pattern）vs. 时间—深度双层 Transformer（Moshi RQ-Transformer）；
- `图 22-5`（`ch22/ch22-temporal-and-depth-transformer-gpu-roofline-trap.svg`）：单帧 $80\ \mathrm{ms}$ 内 Temporal 大模型（1 步）与 Depth 小模型（$Q$ 步串行微算子）的 GPU 启动开销陷阱与整步 CUDA Graph 融合；
- `图 22-6`（`ch22/ch22-webrtc-opus-aec-to-gpu-end-to-end-latency-budget.svg`）：从端侧麦克风 AEC、WebRTC/RTP 抖动缓冲到服务端 GPU 双层解码与声码器还原的 `160 ms` 全链路时延拆解；
- `图 22-7`（`ch22/ch22-barge-in-playback-cursor-aligned-kv-rollback.svg`）：用户打断（Barge-in）时的核心状态同步：清空网关未播缓冲 + 按客户端实际播放 RTP 时间戳 $t_{\mathrm{play}}$ 精确截断回滚服务端超前 KV 缓存；
- `图 22-8`（`ch22/ch22-streaming-causal-conv-vocoder-chunk-halo.svg`）：流式因果卷积声码器（Causal HiFi-GAN / Mimi Decoder）的帧间历史感受野缓存（Conv State Cache）与零爆音拼接；
- `图 22-9`（`ch22/ch22-live-video-1fps-and-audio-12hz-interleaved-pacing.svg`）：实时视频通话中将 $1\ \mathrm{FPS}$ 视觉编码 Token 拆分为微块交错插入 $12.5\ \mathrm{Hz}$ 音频节拍以消除帧抖动（Jitter Spike）；
- `图 22-10`（`ch22/ch22-acoustic-token-eviction-and-semantic-anchor-retention.svg`）：长时语音通话中的分层记忆修剪：剥离历史轮次高维声学码本与视觉帧、仅保留语义首码本与内心独白文本 KV；
- `图 22-11`（`ch22/ch22-dynamic-batching-under-strict-80ms-isochronous-tick.svg`）：等时硬节拍（Isochronous $80\ \mathrm{ms}$ Tick）下多路并发全双工会话的槽位对齐批处理与超时降级策略；
- `图 22-12`（`ch22/ch22-duplex-turn-taking-backchannel-and-overlap-state-machine.svg`）：全双工对话中的四态交互（倾听、附和 Backchannel、抢话重叠 Overlap、主动静音）概率控制；
- `图 22-13`（`ch22/ch22-edge-vad-aec-wake-word-and-cloud-duplex-handover.svg`）：端侧超低功耗唤醒/AEC 预处理与云端全双工大模型会话热接管协同架构；
- `图 22-14`（`ch22/ch22-voice-cloning-zero-shot-speaker-embedding-cache.svg`）：零样本音色克隆与情感控制中说话人参考声纹特征（Speaker Conditioning KV）的跨会话共享与安全防伪鉴权；
- `图 22-15`（`ch22/ch22-full-duplex-gpu-concurrency-vs-p99-frame-underrun.svg`）：单张 H100/B200 上并发全双工语音流路数 $B_{\mathrm{stream}}$ 与音频断流率（Frame Underrun Rate $<0.1\%$）的容量边界曲线；
- `图 22-16`（`ch22/ch22-full-duplex-omni-streaming-infrastructure-synthesis.svg`）：全双工实时语音与原生全模态流式基础设施从编解码、双层算子到打断回滚的全景架构图。

---

### 4.3 Batch 12 新建第 23 章：《百亿亿次 Token 数据工程、GPU 加速模糊去重与合成数据工厂底座》（`manuscripts/23-百亿亿次 Token 数据工程、GPU 加速模糊去重与合成数据工厂底座.md`，`~1,080` 行，`16` 幅配图，`5` 个编号算例，`6` 个代码块，`10` 道习题，`2` 个实验目录 `23-01..02`）

- **`## 23.1 从百亿网页爬虫到万亿精品词元：现代预训练数据漏斗（`FineWeb-2` / `DCLM` / `Nemotron-CC`）的系统经济学`**
  - `### 23.1.1` **百 PB 级原始网页（CommonCrawl WARC）的七级提纯漏斗与 $10\%$ 留存率账本**：定量追踪从 `100 PB` 压缩网页归档经 HTML 正文抽取（`Trafilatura` / `Resiliparse`）、语言识别（FastText LangID）、启发式规则过滤（Gopher / C4 / FineWeb 行尾标点与重复行规则）、全局模糊去重（Fuzzy Dedup）、模型质量打分分级（Model-Based Quality Tiering）到评测基准防污染净化的全流程 CPU 核时、网络洗牌字节与最终词元产出比；
  - `### 23.1.2` **为什么在 15T–50T Tokens 时代，“重复数据”比“低质噪声”更具毁灭性**：剖析重复文本导致的双下降现象（Double Descent）、记忆化背诵与上下文学习能力退化，以及“全局激进跨快照去重 vs. 高质量子集受控上采样（`Nemotron-CC`）”的消融铁律。
- **`## 23.2 十亿文档级模糊去重（Fuzzy Deduplication）的算法与 GPU 分布式加速（`NeMo Curator` / `DataTrove`）`**
  - `### 23.2.1` **MinHash + 局部敏感哈希（LSH）的 S 型碰撞曲线与分桶参数 $(b, r)$ 严格推导**：给定两篇文档的 $k$-shingle 集合 Jaccard 相似度 $s = \frac{|A \cap B|}{|A \cup B|}$，使用 $M = b \times r$ 个独立哈希函数（分为 $b$ 个 Band、每个 Band 含 $r$ 行），严格推导**至少在一个 Band 内完全一致并落入同一候选桶的概率**：
    $$P(\text{Candidate}\mid s) = 1 - \left(1 - s^r\right)^b, \quad s^* \approx \left(\frac{1}{b}\right)^{1/r}$$
    （例如取 $M=128, b=16, r=8$ 时，阈值拐点 $s^* \approx 0.707$，$s=0.80$ 的高度相似文档被捕获概率达 $1 - (1 - 0.80^8)^{16} \approx 94.7\%$，而 $s=0.50$ 的无关文档误碰率仅 $6.1\%$）；
  - `### 23.2.2` **从数万核 CPU 跑一周到 64 张 GPU 跑 2 小时：基于 `cuDF` 与分布式连通分量（GPU Connected Components / Union-Find）的全流水线卸载**：定量核算 50 亿篇文档生成 `128 × 4B = 512 B/doc` 签名（总计 `2.56 TB`）、跨节点 All-to-All 按 Band 哈希重分布（Shuffle）、构建相似度边表并运行分布式大星体收缩（Large-Star / Small-Star Connected Components）聚类删除重复簇的显存与网络账本；同时对比后缀数组（Suffix Array Exact Substring Dedup）在百字节级精确段落去重中的互补作用。
- **`## 23.3 模型驱动质量分级（`DCLM` / `FineWeb-Edu`）、评测集防污染（Decontamination）与在线动态退火配比（`DoReMi` / `RegMix`）`**
  - `### 23.3.1` **用大模型蒸馏轻量级打分器（LLM-as-Judge $\to$ BERT / FastText 分类器）的三级算力杠杆**：核算直接用 70B 模型给 500 亿篇文档打分的成本荒谬性 vs. 用 70B 标注 500 万样本训练轻量级分类器（单核每秒处理数千篇或 GPU 批量打分）的百倍降本路径；
  - `### 23.3.2` **评测基准防污染（13-gram 碰撞 + 稠密向量语义去污）与预训练三阶段动态配比退火（Web 主干期 $\to$ 高质量 STEM/代码加权期 $\to$ 数学/长链推理极速退火期 Mid-Training Annealing）**。
- **`## 23.4 万亿词元“合成数据工厂（Synthetic Data Factory）”：无延迟 SLO 约束下的极限吞吐/美元架构`**
  - `### 23.4.1` **在线推理服务（Latency-Critical）vs. 离线合成数据工厂（Throughput-First）的架构解耦**：当目标是合成 5 万亿个数学/代码/多语言推理 Token（如 `Cosmopedia` / `Phi-4` / `Nemotron-4` / `DeepSeek-R1` 蒸馏语料）而无需满足人类实时等待的 `<200 ms` TTFT 与 `<30 ms` TPOT 时，如何通过**超大批次（Batch Size $= 2048\text{–}8192$）将自回归解码彻底推入 Tensor Core 计算饱和区（MFU $>55\%$）**，并利用数据中心昼夜波谷闲置算力（Spot / Slack Capacity Harvesting）将每百万合成 Token 成本压降 **$8\text{–}15\times$**；
  - `### 23.4.2` **“生成—验证—过滤—重写”闭环流水线（Generator $\to$ Sandbox Verifier $\to$ Rubric Filter）的拒绝采样（Rejection Sampling）良率与最优算力分配**。
- **`## 常见误区`、`## 练习与实验`（`23-1` ~ `23-10` + 速查表）、`## 本章小结`**

#### 第 23 章 16 幅 420pt 配图清单（`图 23-1` ~ `图 23-16`）
- `图 23-1`（`ch23/ch23-100pb-commoncrawl-to-15t-tokens-curation-funnel.svg`）：从 `100 PB` CommonCrawl 原始 WARC 归档经七级提纯漏斗（抽取、语言识别、启发式、去重、质量分级、去污）产出 `15T` 精品词元的体积与算力瀑布；
- `图 23-2`（`ch23/ch23-minhash-lsh-band-row-s-curve-probability.svg`）：MinHash LSH 局部敏感哈希在不同 $(b, r)$ 分桶配置下的 S 型碰撞概率曲线 $P(s) = 1 - (1 - s^r)^b$ 与假阴/假阳权衡；
- `图 23-3`（`ch23/ch23-distributed-minhash-shuffle-and-gpu-union-find.svg`）：十亿级文档模糊去重的分布式流水线：Shingle 哈希 $\to$ 按 Band All-to-All 洗牌 $\to$ 候选边生成 $\to$ GPU 连通分量（Large-Star/Small-Star）聚簇代表元选取；
- `图 23-4`（`ch23/ch23-cpu-datatrove-vs-gpu-nemo-curator-throughput.svg`）：CPU 集群（DataTrove）与 GPU 加速流水线（NeMo Curator + cuDF + RAPIDS cuGraph）在百 TB 语料去重各阶段的耗时与 TCO 对比；
- `图 23-5`（`ch23/ch23-document-fuzzy-dedup-vs-suffix-array-substring-dedup.svg`）：文档级 MinHash 模糊去重与基于后缀数组（Suffix Array）的百字节级精确重复模板/导航栏剥离协同；
- `图 23-6`（`ch23/ch23-llm-rubric-distillation-to-fasttext-bert-classifier.svg`）：用 70B 教师模型标注百万样本蒸馏轻量级质量分类器（FineWeb-Edu / DCLM）的三级算力放大漏斗；
- `图 23-7`（`ch23/ch23-nemotron-cc-classifier-ensembling-and-rephrasing.svg`）：Nemotron-CC 范式：多分类器集成打分分桶与对高知识价值但表述残缺网页的合成改写抢救（Synthetic Rephrasing）；
- `图 23-8`（`ch23/ch23-benchmark-decontamination-ngram-and-embedding-hybrid.svg`）：防止评测集刷榜失真的双层去污管线（13-gram 子串匹配 + 稠密语义向量近邻剔除防止同义改写污染）；
- `图 23-9`（`ch23/ch23-doremi-regmix-domain-weight-optimization.svg`）：基于小型代理模型（Proxy Model）的领域配比寻优（DoReMi / RegMix）与三阶段预训练动态退火曲线；
- `图 23-10`（`ch23/ch23-online-serving-vs-offline-synthetic-factory-roofline.svg`）：在线低延迟服务（受限于小批量访存墙）与离线合成数据工厂（超大批量 + 跨层卸载推满 Tensor Core）的 Roofline 位置跃迁；
- `图 23-11`（`ch23/ch23-synthetic-data-generator-verifier-rejection-funnel.svg`）：推理与代码合成数据工厂的“多样化种子 $\to$ 高温批量采样 $\to$ 沙箱编译/测试验证 $\to$ 过程奖励过滤”良率账本；
- `图 23-12`（`ch23/ch23-slack-capacity-harvesting-for-synthetic-generation.svg`）：利用在线推理集群昼夜波谷与训练维护空窗自动拉起离线合成数据作业的零边际硬件成本收割架构；
- `图 23-13`（`ch23/ch23-pdf-scientific-ocr-layout-math-extraction-pipeline.svg`）：高质量学术论文/教材 PDF 语料的版面分析、公式 LaTeX 还原与多模态图文交错提取流水线；
- `图 23-14`（`ch23/ch23-tokenized-dataset-mmap-shuffling-and-sequence-packing.svg`）：万亿词元二进制索引（`.bin`/`.idx` mmap）的确定性全局伪随机洗牌与最佳适应序列打包（Best-Fit Cropping/Packing）零填充损耗；
- `图 23-15`（`ch23/ch23-data-lineage-copyright-optout-and-audit-ledger.svg`）：面向合规与可追溯性的语料血缘账本（Data Provenance）、版权退订剔除与训练批次精确回溯索引；
- `图 23-16`（`ch23/ch23-exascale-data-and-synthetic-factory-master-architecture.svg`）：百亿亿次 Token 数据工程与合成数据工厂从原始爬虫、GPU 去重到闭环合成退火的全栈总图。

---

### 4.4 Batch 12 新建第 24 章：《科学智能（AI for Science）基础设施：生物大分子、百万碱基基因组与物理场网格系统》（`manuscripts/24-科学智能（AI for Science）基础设施：生物大分子、百万碱基基因组与物理场网格系统.md`，`~1,120` 行，`16` 幅配图，`5` 个编号算例，`6` 个代码块，`10` 道习题，`2` 个实验目录 `24-01..02` + **全书 24 章终极大收束**）

- **`## 24.1 科学基础模型的算子与访存特征：为什么它们与自然语言 LLM 截然不同？`**
  - `### 24.1.1` **从一维离散词元到三维几何等变图、高阶关系张量与偏微分方程（PDE）连续场**：系统梳理四大科学基础模型家族的计算图基因差异——（1）生物大分子结构与相互作用（`AlphaFold 3` / `Boltz-1` / `Chai-1`：$N_{\mathrm{res}}^2$ 氨基酸对表示与 $\mathcal{O}(N_{\mathrm{res}}^3)$ 三角注意力）、（2）百万碱基基因组模型（`Evo 2` 40B：`1M` 单核苷酸分辨率长卷积—注意力混合主干）、（3）全球气象与流体物理场（`FourCastNet 3` / `GraphCast` / `Aurora`：球面谐波变换 SFNO 与高分辨率网格 Halo 交换）、（4）分子动力学神经势函数（`MACE` / `MatterGen`：三维空间截断半径图与球谐张量积）；
  - `### 24.1.2` **科学计算负载的两大独特系统瓶颈：参数量虽适中（100M–40B）但中间激活张量呈二次/三次方爆炸，以及双阶段异构硬件失配（CPU/SSD 序列检索 vs. GPU 几何扩散）**。
- **`## 24.2 生物大分子复合物预测系统（`AlphaFold 3` / `Boltz-1` / `OpenFold`）：MSA GPU 加速与 $\mathcal{O}(N_{\mathrm{res}}^3)$ 三角注意力并行`**
  - `### 24.2.1` **破除第一阶段 I/O 与 CPU 饥饿：将多序列比对（MSA）从 45 分钟 CPU 搜索压缩至 1.2 秒 GPU 扫描（`MMseqs2-GPU`）与两阶段流水解耦**：剖析为何传统 AlphaFold 流水线中 `JackHMMER`/`HHblits` 在數 TB 蛋白质序列库上的 CPU 检索占去端到端耗时的 $70\%\text{–}85\%$，致使昂贵的 H100/B200 空转等待；详解如何将 Stage 1（MSA 搜索池）与 Stage 2（Pairformer + 扩散结构生成池）物理分离，并利用 **`MMseqs2-GPU`** 将无间隙 $k$-mer 预过滤与向量化 Smith-Waterman 动态规划映射至 Tensor Core / DPX 指令实现百倍加速；
  - `### 24.2.2` **攻克 Pairformer 的 $\mathcal{O}(N_{\mathrm{res}}^2)$ 显存墙与 $\mathcal{O}(N_{\mathrm{res}}^3)$ 三角乘法/三角自注意力（Triangle Multiplication & Triangle Attention）**：当预测含 $N_{\mathrm{res}} = 5,120$ 个残基/原子的超大蛋白质—RNA—配体复合物时，单个二维残基对张量 $Z \in \mathbb{R}^{N_{\mathrm{res}} \times N_{\mathrm{res}} \times c_z}$（$c_z = 128$）及其在三角注意力中的中间激活即突破单卡 `80 GB` 显存！严格推导 **分块三角注意力核融合（Chunked Triangle Attention / `DS4Sci_EvoformerAttention`）** 与 **动态轴向序列并行（Dynamic Axial Parallelism, DAP：沿行轴 $i$ 与列轴 $j$ 正交切分二维对张量并通过子群 All-to-All 转置）** 的显存与通信账本。
- **`## 24.3 百万碱基基因组基础模型（`Evo 2` 40B）与长卷积—注意力混合并行底座`**
  - `### 24.3.1` **单核苷酸分辨率（Single-Nucleotide Resolution）下的百万上下文挑战與 `StripedHyena 2` 多尺度长卷积架构**：为什么基因组模型不能像文本那样用粗糙的 $k$-mer 分词（会掩盖单碱基突变 SNP 与剪接位点偏移），而必须直接处理长度 $L = 1,048,576$（`1M`）的 `A/C/G/T` 序列；
  - `### 24.3.2` **快速傅里叶卷积核融合（`FlashFFTConv`）与跨节点分层上下文并行**：利用卷积定理 $\mathbf{u} * \mathbf{k} = \mathcal{F}^{-1}(\mathcal{F}(\mathbf{u}) \odot \mathcal{F}(\mathbf{k}))$ 将 $\mathcal{O}(L^2)$ 长程依赖化为 $\mathcal{O}(L \log L)$，通过 Monarch 矩阵分解将 FFT 映射为 Tensor Core 密集矩阵乘并驻留片上 SRAM，配合跨卡 Ring/Ulysses 上下文并行支撑 `1M` 序列训练。
- **`## 24.4 地球气象物理场大模型（`FourCastNet` / `GraphCast`）与亿级原子分子动力学（`MACE`）的空间域分解系统`**
  - `### 24.4.1` **球面傅里叶神经算子（`SFNO`）的二维空间域切分、分布式球谐变换与局部卷积边界光环交换（Halo Exchange）**：推导在 $0.25^\circ \to 0.05^\circ$ 全球高分辨率气象网格上，如何沿纬度（$\theta$）与经度（$\phi$）执行 2D 空间域分解，使局部有限差分/消息传递的 Halo 边界通信与内部网格计算完全重叠，并利用 GPUDirect Storage 直读 PB 级 ERA5 Zarr/HDF5 气象再分析数组；
  - `### 24.4.2` **神经原子间势函数（`MACE` / `NequIP`）驱动的亿级原子分子动力学（MD）仿真：三维空间块分解（Spatial Domain Decomposition）与动态截断半径邻居表（Verlet Neighbor List）跨卡同步**。
- **`## 24.5 全书二十四章百科全书终极大收束：从一次矩阵乘法的五个追问，到统一驾驭语言、视觉、语音、行动、数据与科学发现的智能计算宏图`**
  - `### 24.5.1` **全书二十四章四大部分知识坐标系总览与核心工程定量账本总表（`*表 24-6*`）**；
  - `### 24.5.2` **从比特、瓦特、词元到物理行动与科学定律：AI 基础设施系统工程的终极结语**。
- **`## 常见误区`、`## 练习与实验`（`24-1` ~ `24-10` + 速查表）、`## 本章小结`**

#### 第 24 章 16 幅 420pt 配图清单（`图 24-1` ~ `图 24-16`）
- `图 24-1`（`ch24/ch24-llm-1d-tokens-vs-ai4science-four-structural-domains.svg`）：一维文本 Token 与四大科学智能领域（生物大分子 2D 对表示、基因组 1M 核苷酸序列、地球球面 PDE 网格、三维原子截断半径图）的计算图与复杂度对比；
- `图 24-2`（`ch24/ch24-alphafold3-boltz1-two-stage-disaggregated-pipeline.svg`）：生物大分子预测两阶段异构解耦架构：Stage 1（`MMseqs2-GPU` 序列检索池）与 Stage 2（Pairformer + 扩散结构生成池）独立扩缩容；
- `图 24-3`（`ch24/ch24-mmseqs2-gpu-kmer-prefilter-and-smith-waterman.svg`）：`MMseqs2-GPU` 利用片上共享内存 $k$-mer 查表与硬件 DPX 动态规划指令将 MSA 搜索从 45 分钟压降至 1.2 秒的核内流水；
- `图 24-4`（`ch24/ch24-pairformer-triangle-multiplication-and-attention-o-n3.svg`）：AlphaFold 3 / Boltz-1 Pairformer 中残基对表示 $Z_{ij}$ 的三角乘法更新与 $\mathcal{O}(N_{\mathrm{res}}^3)$ 三角自注意力几何依赖图；
- `图 24-5`（`ch24/ch24-chunked-triangle-attention-and-dynamic-axial-parallelism.svg`）：分块三角注意力核融合（消除 $\mathcal{O}(N_{\mathrm{res}}^3)$ 中间张量物化）与动态轴向并行（DAP 行列正交切分 + All-to-All 转置）；
- `图 24-6`（`ch24/ch24-all-atom-diffusion-module-multi-sample-batching.svg`）：全原子三维坐标扩散模块（All-Atom Diffusion）在共享同一躯干 Pairformer 特征下并行生成 $K=5\text{–}20$ 个候选构象的算术强度倍增；
- `图 24-7`（`ch24/ch24-evo2-stripedhyena2-multi-scale-hyena-attention-backbone.svg`）：百万碱基基因组模型（Evo 2 40B）的 `StripedHyena 2`（短/中/长三级门控卷积 + 稀疏旋转注意力）混合主干架构；
- `图 24-8`（`ch24/ch24-flashfftconv-monarch-decomposition-tensor-core.svg`）：`FlashFFTConv` 利用 Monarch 矩阵分解将长序列快速傅里叶变换（FFT）映射为 Tensor Core 矩阵乘并驻留 SRAM；
- `图 24-9`（`ch24/ch24-sfno-spherical-harmonics-and-2d-grid-decomposition.svg`）：全球气象大模型（FourCastNet 3 / Aurora）的球面傅里叶神经算子（SFNO）与经纬度二维空间域切分；
- `图 24-10`（`ch24/ch24-spatial-domain-halo-exchange-computation-overlap.svg`）：三维/二维物理网格并行中内部格点计算与边界幽灵区光环交换（Halo Exchange P2P）的双流完全掩盖时序；
- `图 24-11`（`ch24/ch24-era5-petabyte-zarr-gds-direct-streaming-dataloader.svg`）：面向 PB 级多维气象场（ERA5 Zarr/HDF5 分块存储）的 GPUDirect Storage 零拷贝按需切片加载流水线；
- `图 24-12`（`ch24/ch24-mace-equivariant-message-passing-and-verlet-neighbor-list.svg`）：等变神经原子势函数（MACE）在亿级原子分子动力学中的空间块分解、动态 Verlet 邻居表重构与跨卡边界原子迁移；
- `图 24-13`（`ch24/ch24-ai-in-the-loop-hybrid-hpc-pde-solver-and-surrogate.svg`）：传统 HPC 偏微分方程求解器与 AI 神经代理模型（Neural Surrogate）在同一个超算作业内零拷贝耦合的混合仿真架构；
- `图 24-14`（`ch24/ch24-autonomous-wet-lab-active-learning-closed-loop.svg`）：AI4Science 闭环发现工厂：生成式分子/材料设计（MatterGen）$\to$ 物理仿真筛选 $\to$ 自动化机器人湿实验验证 $\to$ 主动学习回流；
- `图 24-15`（`ch24/ch24-twenty-four-chapter-encyclopedia-knowledge-atlas.svg`）：全书 24 章四大部分（基础与硬件篇、训推系统篇、智能体与模态底座篇、数据/科学/机队工程篇）百科全书知识全景图；
- `图 24-16`（`ch24/ch24-unified-laws-across-all-24-chapters-synthesis.svg`）：贯穿全书 24 章的统一物理底座——从单个矩阵乘法的算术强度，到驾驭语言、全模态、机器人与科学发现的智能计算铁律。

---

## 五、新建 12 个独立可运行实验目录（`experiments/ch17..ch24`）工程规格表

为彻底消除第 17–20 章仅有单实验目录的不对称性，并为新建第 21–24 章提供同等完备的实验支撑，路径 E3 将新建 **12 个零第三方依赖（纯 Python 3.8+ 标准库）、含 `run.py` + `verify.py` + `README.md` + `results/baseline.json` 的独立实验目录**（使全书 24 章每章均拥有 $\ge 2$ 个实验目录，总计 **135 个实验目录**）：

| 新建实验目录 | 关联章节与习题 | 核心可复算物理模型与确定性断言指标 | 交付批次 |
| --- | --- | --- | :-: |
| [`experiments/ch17/17-02/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch17) | 第 17 章 `17-13` / `17-14` | **原生全模态 E-P-D-G 四池异构配比、Qwen3-VL `7.8125 MiB` EC vs. `56.25 MiB` KV（$7.2\times$ 放大）与 Transfusion 混合掩码效率复算器**。 | Batch 9 |
| [`experiments/ch18/18-02/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch18) | 第 18 章 `18-13` / `18-14` | **DistRS 解耦奖励沙箱条件剩余时间（`216,000` tokens 沉没工作量裁剪阈值）、RollPacker 总词元归一化偏差修正与 PRM 树搜索 CoW 显存节省复算器**。 | Batch 9 |
| [`experiments/ch19/19-02/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch19) | 第 19 章 `19-11` / `19-12` | **真机多传感器 PTP 硬件脉冲对齐、控制屏障函数（CBF）二次规划安全过滤与 `FAST` 离散余弦变换（DCT）频域动作块压缩模拟器**。 | Batch 10 |
| [`experiments/ch20/20-02/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch20) | 第 20 章 `20-11` / `20-12` | **集合通信到达时间偏斜矩阵（Arrival Skew Matrix）慢节点根因反演、`TrainMover` 亚秒级热迁移与五大跑分失真（Benchmark Gaming）修正复算器**。 | Batch 10 |
| [`experiments/ch21/21-01/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch21) | 第 21 章 `21-1` ~ `21-5` | **40 轮自主智能体 WORM 前缀复用率（$>92\%$）、四级 KV 缓存全局路由（Dynamo/HiCache）TTFT 寻优与 CacheGen 流式压缩回本计算器**。 | Batch 11 |
| [`experiments/ch21/21-02/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch21) | 第 21 章 `21-6` ~ `21-10` | **Firecracker 微虚机 `12 ms` CoW 快照分叉、MCTS 环境回溯显存账本与锁定算例 `I19`（`81.92 s` 溢出 / `76.92 s` 排空）确定性复算器**。 | Batch 11 |
| [`experiments/ch22/22-01/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch22) | 第 22 章 `22-1` ~ `22-5` | **级联语音管线 vs. 原生全双工 S2S（Moshi / Qwen-Omni）延迟拆解、RVQ 比特率方程与 Temporal+Depth 双层 Transformer $80\ \mathrm{ms}$ 帧预算复算器**。 | Batch 11 |
| [`experiments/ch22/22-02/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch22) | 第 22 章 `22-6` ~ `22-10` | **全双工语音随时打断（Barge-in）下按客户端播放 RTP 游标对齐的 KV 缓存精确回滚与长时通话声学码本剥离（$-85\%$ 显存）模拟器**。 | Batch 11 |
| [`experiments/ch23/23-01/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch23) | 第 23 章 `23-1` ~ `23-5` | **百 PB 网页提纯漏斗、MinHash LSH $(b=16, r=8)$ S 型碰撞曲线积分与分布式连通分量去重洗牌流量核算器**。 | Batch 12 |
| [`experiments/ch23/23-02/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch23) | 第 23 章 `23-6` ~ `23-10` | **万亿词元离线合成数据工厂（无 TTFT 约束超大批量 Roofline 跃迁）、沙箱拒绝采样良率与波谷闲置算力收割 TCO 计算器**。 | Batch 12 |
| [`experiments/ch24/24-01/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch24) | 第 24 章 `24-1` ~ `24-5` | **AlphaFold 3 / Boltz-1 两阶段解耦、`MMseqs2-GPU` 加速比与 Pairformer $\mathcal{O}(N_{\mathrm{res}}^2)$ 显存及 $\mathcal{O}(N_{\mathrm{res}}^3)$ 分块三角注意力/动态轴向并行核算器**。 | Batch 12 |
| [`experiments/ch24/24-02/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/ch24) | 第 24 章 `24-6` ~ `24-10` | **百万碱基基因组 `FlashFFTConv` 复杂度、全球气象大模型 2D 网格 Halo 交换重叠边界与全书 24 章工程总账本校验器**。 | Batch 12 |

---

## 六、出版级配图（`codex exec` + `gpt-image-2.6-sunburst` `quality="high"` + `420pt` 矢量规范）与全书 24 章构建适配表

### 6.1 出版级配图双重生成与视觉门禁（Visual Gate）
1. **对路径 E3 新增的全部 104 幅配图（第 1–20 章每章 `+2` 幅共 `40` 幅 + 第 21–24 章每章 `16` 幅共 `64` 幅）以及重绘的 `图 1-1`（24 章全景导读图）**：
   - 必须通过 `codex exec` 调用 **`gpt-image-2.6-sunburst`**（显式设置 **`quality="high"`** 最高质量档位）生成高精度视觉构图参考，并结合 [`manuscripts/figure_style/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/figure_style) 与各章 `build.py` 生成严格锁定 **`width="420pt"`** 的矢量 `.svg`、`.pdf` 与高分 `.png`（三格式 100% 物理落盘）；
   - **五大视觉红线**：（1）字体统一无衬线且最小字号不低于规范阈值、（2）网格级几何对齐、（3）充足留白且零文字压线/越界/重叠、（4）遵循深海蓝 `#1F4E79`、青黛绿 `#1B7837`、琥珀橙 `#D95F02`、珊瑚红 `#C0392B` 学术色板、（5）图内自带关键定量数字与拐点结论；
   - **逐张视觉巡检**：每批次生成后，主 Agent 与 **Gemini 4 Argon High Review Subagent** 必须使用 `read_file` 打开全部新增 `.png` 文件逐张目视审查，发现任何微小的排版瑕疵立即重绘修复。

### 6.2 全书 24 章构建、校验、PDF/EPUB 与站点导航脚本精确修改清单（Batch 12 合龙时执行）

| 序号 | 目标文件 | 当前硬编码位置与行号 | 路径 E3（24 章）精确修改方案 |
| :-: | --- | --- | --- |
| 1 | [`scripts/verify_core_principles.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/scripts/verify_core_principles.py) | 第 10 行：`for n in range(1,21):` | 改为 `for n in range(1,25):`，自动校验全部 24 章的 `## 本章小结`、脚注闭合、本地相对链接、`figure-index.json`（`835` 幅图）与习题列表（`384` 块 / `383` ID）。 |
| 2 | [`scripts/build_site.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/scripts/build_site.py) | 第 54–58 行：`if n <= 20` 与 `expected = list(range(21))` | 将简体中文版 `zh` 的章节上限改为 `if n <= 24` 与 `expected = list(range(25))`（含 `00-前言` 共 25 个章节级页面；外文社区译本保持 `range(13)` 不变）。 |
| 3 | [`scripts/check_site.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/scripts/check_site.py) | 第 46 行：`max_ch = 21 if name == '简体中文' else 13`；第 54 行打印信息 `(20 zh / 12 translations)` | 改为 `max_ch = 25 if name == '简体中文' else 13`，并将第 54 行打印信息更新为 `(24 zh / 12 translations)`。 |
| 4 | [`book/build_pdf.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/book/build_pdf.py) | 第 70 行：`choices=range(1,21)`；第 79 行：`range(1,21)` | 全部改为 `range(1,25)`，支持单章编译 `--chapter 21..24` 与全书 24 章合订本 PDF 编译。 |
| 5 | [`book/build_epub.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/book/build_epub.py) | 第 63–65 行：`for n in range(21)` | 改为 `for n in range(25)`（第 79 行已在路径 D 修复为 `MANUSCRIPTS.glob('[0-9][0-9]-*.md')`，天然支持 `21`–`24` 章跨章锚点解析）。 |
| 6 | [`book/check_ci_pdf.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/book/check_ci_pdf.py) | 第 10 行：`'expected_text': ('AI Infra', '端边云', '十万卡集群')`；第 29 行：`list(range(1, 21))` | 将第 10 行 `'zh'` 的 `'expected_text'` 更新为 `('AI Infra', '端边云', '十万卡集群', '科学智能')`，并将第 29 行 `'zh'` 的期望章节改为 `list(range(1, 25))`。 |
| 7 | [`manuscripts/ch01/build_roadmap.py`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/ch01/build_roadmap.py) | 第三部分章节列表（当前列至第 20 章） | 在 Batch 1 扩展画布与列表，将第 21–24 章（`21 自主智能体与分层 KV 记忆底座`、`22 全双工实时语音与原生全模态流`、`23 百亿亿次数据工程与合成数据工厂`、`24 科学智能（AI4Science）基础设施`）纳入 `figure-1-book-roadmap.{svg,png,pdf}`。 |
| 8 | `manuscripts/ch21/` ~ `manuscripts/ch24/` | 尚不存在 | 新建 `manuscripts/ch21/`、`ch22/`、`ch23/`、`ch24/` 四个目录，分别包含 `figure-index.json`、`figure-data.json`、`build.py` 及各自的 16 幅 `.svg`/`.png`/`.pdf` 配图资产。 |
| 9 | [`website/index.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/website/index.md)、[`README.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/README.md)、[`manuscripts/README.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/README.md)、[`manuscripts/00-前言.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/00-%E5%89%8D%E8%A8%80.md)、[`case-studies/README.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/case-studies/README.md)、[`experiments/README.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/experiments/README.md) | 当前为“二十章导读”与 20 章实验索引 | 全面升级为 **“二十四章百科全书导读”**，更新四类读者（模型/智能体/应用、推理/训练系统、硬件/芯片/网络、数据/科学计算）的定制阅读路线图，并补齐 135 个可运行实验的总索引。 |

---

## 七、十二批次推进路线图（12 Batches）与六步强制执行标准作业程序（6-Step Batch SOP）

### 7.1 十二批次推进总表（每 Batch 2 章，完成即验证、Subagent 审阅收敛、Git 提交并推送）

| 批次 | 覆盖章节 | 核心交付物概要 | 新增配图 | 新增代码块 | 新增习题 | 新增实验目录 |
| :-: | --- | --- | :-: | :-: | :-: | :-: |
| **Batch 1** | **第 1 章 + 第 2 章** | **Ch01**：五大 AI 负载范式演进、北极星效率乘积链（$\mathrm{MPG}_{\mathrm{fleet}}$〔含 $\mathrm{ETTR}$〕$\times \mathrm{SLO\text{-}Goodput} \times P(\text{Verified})$）、重绘 24 章路线图 `图 1-1`、补齐 3 个代码块与 4 个编号算例；**Ch02**：块扩散语言模型（**Diffusion LLMs：`Mercury`/`LLaDA`/`SGLang dLLM`**）Roofline 跃迁、无分词器 **字节潜空间模型（`BLT`）** 熵驱动动态分块、补齐 4 个代码块与 5 个编号算例。 | `+4` 幅（`1-25..26`, `2-43..44`）+ 重绘 `1-1` | `+7` 个 | `+4` 道（`1-13..14`, `2-17..18`） | — |
| **Batch 2** | **第 3 章 + 第 4 章** | **Ch03**：推理时计算扩展定律导致的最优模型规模左移、多智能体 $M$ 路并发扇出屏障（Fan-out Barrier）尾延迟极值放大与对冲、补齐 3 个代码块与 4 个算例；**Ch04**：**HBM4（2048-bit 双倍位宽 + 逻辑 Base Die）与 NVIDIA Vera Rubin NVL72**、**晶圆级/纯 SRAM 推理芯片（`Cerebras WSE-3` / `Groq LPU` / `d-Matrix Corsair`）及“HBM GPU Prefill + SRAM ASIC Decode”异构分离架构**、补齐 3 个代码块与 5 个算例。 | `+4` 幅（`3-38..39`, `4-50..51`） | `+6` 个 | `+4` 道（`3-18..19`, `4-17..18`） | — |
| **Batch 3** | **第 5 章 + 第 6 章** | **Ch05**：下一代算子 DSL（**`TileLang` ICLR '26 & `ThunderKittens 2.0`**）、代数张量超优化器（**`Mirage` & `MPK` 全模型单持久化超内核 OSDI '26**）、新增 1 个代码块与 5 个算例；**Ch06**：**AMD Helios 72-GPU 开放机柜（MI455X `432 GB` HBM4 + `UALink 1.0` 内存语义 Flit 协议）**、**Google TPU v7（`Ironwood`）`9,216` 卡 3D Torus 与 `Palomar` MEMS 光电路交换机（OCS）动态重构**、补齐 3 个代码块与 5 个算例。 | `+4` 幅（`5-47..48`, `6-64..65`） | `+4` 个 | `+4` 道（`5-16..17`, `6-19..20`） | — |
| **Batch 4** | **第 7 章 + 第 8 章** | **Ch07**：**1.6T 共封装硅光（CPO）vs. 线性直驱（LPO）vs. DSP 可插拔光模块**功耗/误码/MTTR 账本、网内归约（`SHARP v4`）与多租户隔离、补齐 3 个代码块与 5 个算例；**Ch08**：100% 硬化**锁定算例 `I10`（有损 KV 压缩引发输出长度膨胀 $\eta_{\mathrm{len}}$ 导致端到端延迟反升）**、EAGLE-3 多层特征投机 vs. PLD 前缀检索投机、补齐 4 个代码块与 5 个算例。 | `+4` 幅（`7-68..69`, `8-43..44`） | `+7` 个 | `+4` 道（`7-22..23`, `8-17..18`） | — |
| **Batch 5** | **第 9 章 + 第 10 章** | **Ch09**：弹性专家并行影子预热与零停机原子路由切换（`ElasticEP` / `EPLB`）、跨机房逐层流水异步 KV 传输（Layer-wise KV Streaming）、补齐 3 个代码块与 5 个算例；**Ch10**：100% 硬化**锁定算例 `I20`（8×64 MiB 分片串行 `60.833 ms` vs. 单设备槽 `53.833 ms` vs. 双槽乒乓 `35.604 ms`）**、分布式二阶正交化优化器（**`Muon` Newton-Schulz**）与 RL 权重同步掉线重入状态机、补齐 4 个代码块与 5 个算例。 | `+4` 幅（`9-53..54`, `10-55..56`） | `+7` 个 | `+4` 道（`9-19..20`, `10-19..20`） | — |
| **Batch 6** | **第 11 章 + 第 12 章** | **Ch11**：100% 硬化**锁定算例 `I16`（`97.22%` 表面分配率 vs. `33.33%` 真实有效训练进展）** 与 **锁定算例 `I21`（双模型缓存命中率 `84.93%` 成本交叉点与 `91.84%` 6 秒 SLO 阈值）**、补齐 3 个代码块与 5 个算例；**Ch12**：端侧三值化无乘法查表推理（**`BitNet b1.58` `TL1/TL2` LUT 内核**）与闪存行列捆绑预取、端云协同投机解码与本地 PII 隐私脱敏网关、补齐 3 个代码块与 4 个算例。 | `+4` 幅（`11-56..57`, `12-41..42`） | `+6` 个 | `+4` 道（`11-18..19`, `12-18..19`） | — |
| **Batch 7** | **第 13 章 + 第 14 章** | **Ch13**：**NVIDIA NIXL 统一零拷贝异步传输抽象**、PB 级检查点从热层 3 副本到温冷层 `RS(10,4)` 纠删码归档生命周期、补齐 3 个代码块与算例 `13-11..12`；**Ch14**：硬件感知混合架构整数规划搜索（HW-NAS）、稠密模型向混合 MoE 的 **Sparse Upcycling 与跨层注意力蒸馏**、补齐 3 个代码块与算例 `14-11..12`。 | `+4` 幅（`13-25..26`, `14-25..26`） | `+6` 个 | `+4` 道（`13-16..17`, `14-16..17`） | — |
| **Batch 8** | **第 15 章 + 第 16 章** | **Ch15**：`120–250 kW` 机柜 **`800V HVDC` 高压直流母线与芯片垂直供电（VPD）**、园区 SOFC 燃料电池微网与 $60\ \mathrm{^\circ C}$ 液冷余热回收（$\mathrm{ERE} < 1.0$）、补齐 3 个代码块；**Ch16**：私有化闭源模型基于远程证明租约的权重防导出绑定、采样算子内嵌密码学锦标赛水印（`SynthID` 融合）、补齐 2 个代码块。 | `+4` 幅（`15-23..24`, `16-23..24`） | `+5` 个 | `+4` 道（`15-15..16`, `16-15..16`） | — |
| **Batch 9** | **第 17 章 + 第 18 章 + `17-02`, `18-02`** | **Ch17**：自回归与扩散统一架构（`Transfusion` / `Show-o`）混合掩码调度、`3DGS`+`DiT` 混合神经渲染管线、新建 `experiments/ch17/17-02`；**Ch18**：跨地域闲置算力异步 Rollout 收割（`Prime-RL`）、RLVR 奖励黑客在线 KL 制动熔断、新建 `experiments/ch18/18-02`。 | `+4` 幅（`17-19..20`, `18-19..20`） | `+2` 个 | `+4` 道（`17-13..14`, `18-13..14`） | `17-02`, `18-02` |
| **Batch 10** | **第 19 章 + 第 20 章 + `19-02`, `20-02`** | **Ch19**：人形机器人三级实时控制环（`2 Hz` VLM $\to$ `100 Hz` Flow $\to$ `1,000 Hz` 全身力控 WBC + 视触觉）与 `FAST` DCT 频域动作分词器、新建 `experiments/ch19/19-02`；**Ch20**：十万卡混沌故障注入演练（Chaos Engineering）、灰度上线门禁与向第 21–24 章的延伸桥接、新建 `experiments/ch20/20-02`。 | `+4` 幅（`19-17..18`, `20-17..18`） | `+2` 个 | `+4` 道（`19-11..12`, `20-11..12`） | `19-02`, `20-02` |
| **Batch 11** | **新建第 21 章 + 新建第 22 章 + `21-01..02`, `22-01..02`** | **新建第 21 章（~1,120 行）**：自主智能体基础设施（WORM 95% 前缀复用、四级 KV 缓存全局路由 `Dynamo`/`LMCache`/`HiCache` + `CacheGen`、Firecracker 微虚机 `12 ms` CoW 快照分叉与环境回溯、MCP 网关、幂等外发箱与算例 `I19`、`HippoRAG 2` 长期记忆）；**新建第 22 章（~1,080 行）**：全双工实时语音与原生全模态流式底座（`<200 ms` 双流 S2S `Moshi`/`Qwen-Omni`、RVQ 编解码、Temporal+Depth 双层 Transformer CUDA Graph 融合、随时打断 Barge-in 按播放游标对齐的 KV 回滚、音视频交错与声学码本剥离）及 4 个实验目录。 | `+32` 幅（`21-1..16`, `22-1..16`） | `+12` 个 | `+20` 道（`21-1..10`, `22-1..10`） | `21-01..02`, `22-01..02` |
| **Batch 12** | **新建第 23 章 + 新建第 24 章 + `23-01..02`, `24-01..02` + 全书 24 章合龙** | **新建第 23 章（~1,080 行）**：百亿亿次 Token 数据工程、GPU 加速模糊去重（MinHash LSH $1-(1-s^r)^b$ + `NeMo Curator` GPU 连通分量）与万亿词元合成数据工厂（无 TTFT 约束超大批量 Roofline 跃迁 + 沙箱拒绝采样 + 波谷算力收割）；**新建第 24 章（~1,120 行）**：科学智能（AI4Science）基础设施（`AlphaFold 3`/`Boltz-1` 两阶段解耦 + `MMseqs2-GPU` + 分块三角注意力/动态轴向并行、百万碱基基因组 `Evo 2` + `FlashFFTConv`、气象大模型 `SFNO` 2D Halo 重叠、分子动力学 `MACE` 与 **全书 24 章百科全书终极大收束**）；同步升级全书构建/校验/PDF/EPUB/网站脚本至 24 章。 | `+32` 幅（`23-1..16`, `24-1..16`） | `+12` 个 | `+20` 道（`23-1..10`, `24-1..10`） | `23-01..02`, `24-01..02` |

---

### 7.2 每一批次的六步强制执行标准作业程序（6-Step Batch SOP）

在后续按计划执行上述每一个 Batch 时，必须严格按以下 6 个步骤闭环推进，任何一步未达标严禁进入下一 Batch：
1. **Step 1 · 锁定真值核对与算例复算**：读取该 Batch 涉及的所有 `references/`、`calculations/`、`experiments/`、`case-studies/` 与 `research/` 原始文件，逐一核对公式、字节数、延迟、吞吐、Git Commit SHA 与配置边界，确保零脱钩、零幻觉。
2. **Step 2 · 调用 `codex exec` + `gpt-image-2.6-sunburst` (`quality="high"`) 生成出版级 420pt 配图**：
   - 使用 `codex exec` 调用 `gpt-image-2.6-sunburst`（最高质量 `quality="high"`）辅助设计并结合 [`manuscripts/figure_style/`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/manuscripts/figure_style) 生成 `420pt` 宽度的 `.svg`、`.pdf`、`.png` 三格式配图；
   - 更新对应章节的 `manuscripts/chXX/figure-index.json`（及 `figure-data.json` 与 `build.py`）；
   - 用 `read_file` 逐张检视 `.png` 光栅图像，核查字体、对齐、留白、配色与数值标注。
3. **Step 3 · 正文扩写、代码块/编号算例/表题植入与 [`refine-prose`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/.agents/skills/refine-prose/SKILL.md) 深度去 AI 腔润色**：
   - 严格遵循 [`.agents/skills/refine-prose/SKILL.md`](file:///Users/quan.zheng/projects/dev/ai/ai-infra-book/.agents/skills/refine-prose/SKILL.md) 的铁律，消除一切生造词、欧化长句、空洞拔高与机械排比；
   - 严禁在 Markdown 正文中出现未转义的裸 `$` 货币符号（统一写为 `美元` 或在数学环境内写 `\mathrm{USD}`）。
4. **Step 4 · 自动化脚本全量验证**：
   - 运行 `python3 scripts/check_no_html.py`、`python3 scripts/verify_core_principles.py` 以及针对本批次新增配图与正文的版式/定界符检查（校验无裸 `$` 货币符号、SVG `width="420pt"` 且 `svg.fonttype == 'path'` 无裸 `<text>` 节点；合龙时运行 `python3 book/check_typography.py`）；
   - 运行本批次涉及或新建的 `experiments/chXX/XX-YY/run.py` 与 `verify.py` 确保实验 100% 独立可运行；
   - 运行 `python3 scripts/build_site.py && python3 scripts/check_site.py` 验证全站 HTML 构建、锚点跳转与图片光栅化零错误。
5. **Step 5 · 启动 Gemini 4 Argon High 推理强度 Subagent 独立审阅直至收敛**：
   - 调用 `invoke_subagent`（`TypeName: "self"`, `Model: "inherit"`）启动独立审阅 Subagent，对该 Batch 修改的章节正文、数学推导、代码块、新增习题/实验和每一张新增配图的 `.png` 视觉效果进行严苛审计；
   - 对 Subagent 提出的任何阻塞项或改进建议逐一修复，并通过 `send_message` 复审（最多 5 轮），直到 Subagent 明确输出 **`Status: Approved / Converged`**。
6. **Step 6 · Git 提交与远端推送部署**：
   - 执行 `git status` 确认无临时文件残留后，按清晰规范的 Commit Message 提交本 Batch 变更并执行 `git push origin main`。
