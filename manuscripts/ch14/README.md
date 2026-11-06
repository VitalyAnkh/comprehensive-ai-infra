# Chapter 14 Build & Verification Asset Pack (`manuscripts/ch14/`)

本目录包含《第 14 章：模型与系统硬件协同设计》（[manuscripts/14-模型与系统硬件协同设计.md](../14-%E6%A8%A1%E5%9E%8B%E4%B8%8E%E7%B3%BB%E7%BB%9F%E7%A1%AC%E4%BB%B6%E5%8D%8F%E5%90%8C%E8%AE%BE%E8%AE%A1.md)）的 420pt 出版级图表构建脚本、定量参数真值文件及自动化验证脚本。

## 核心文件清单

- `build.py`：生成本章 6 幅 `420pt` 矢量配图（`.svg` / `.png` / `.pdf`）、KaTeX 公式校验与离线预览页面。
- `verify.py`：校验正文章节结构、6 幅配图排版指标、引用链接完整性以及算例 14-1 至 14-4 的定量真值。
- `figure-data.json`：算例 14-1（DeepSeek V4.1 Flash 双轨演进）、算例 14-2（数字员工跨代 GPU 成本与归因审计）、算例 14-3（OpenTallas 专用 ASIC 四资源账本与晶圆级跳数边界）与算例 14-4（反事实翻转点）的锁定参数。
- `figure-index.json`：本章 6 幅配图的索引与教学角色说明。
- `reading-notes.md`：第 14 章核心文献、案例与代码库的导读笔记。
