#!/usr/bin/env python3
"""Build all 16 publication-grade figures (420pt) and validation manifests for Chapter 23."""
from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figure_style import COL, STYLE, Exporter, arrow, box, canvas, card, text
from preview_output import preview_path

plt.rcParams.update(STYLE)


def sha256(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b'version https://git-lfs.github.com/spec/v1'):
        m = re.search(rb'oid sha256:([0-9a-f]{64})', raw)
        if m:
            return m.group(1).decode('ascii')
    return hashlib.sha256(raw).hexdigest()


def draw_dual_band(
    ex: Exporter,
    stem: str,
    top_title: str,
    top_color: str,
    top_boxes: list[tuple[str, str]],
    bot_title: str,
    bot_color: str,
    bot_boxes: list[tuple[str, str]],
    footer: str,
) -> None:
    fig, ax = canvas(3.85)
    card(ax, 0.02, 0.50, 0.96, 0.46, top_title, color=top_color)
    if len(top_boxes) == 3:
        xs = [(0.028, 0.296), (0.352, 0.296), (0.676, 0.296)]
        for (x, w), (lbl, role) in zip(xs, top_boxes):
            box(ax, x, 0.58, w, 0.24, lbl, role, size=11)
        arrow(ax, (0.324, 0.70), (0.352, 0.70))
        arrow(ax, (0.648, 0.70), (0.676, 0.70))
    else:
        xs = [(0.03, 0.45), (0.52, 0.45)]
        for (x, w), (lbl, role) in zip(xs, top_boxes):
            box(ax, x, 0.58, w, 0.24, lbl, role, size=11)
        arrow(ax, (0.48, 0.70), (0.52, 0.70))

    card(ax, 0.02, 0.03, 0.96, 0.42, bot_title, color=bot_color)
    if len(bot_boxes) == 3:
        xs = [(0.028, 0.296), (0.352, 0.296), (0.676, 0.296)]
        for (x, w), (lbl, role) in zip(xs, bot_boxes):
            box(ax, x, 0.13, w, 0.20, lbl, role, size=11)
        arrow(ax, (0.324, 0.23), (0.352, 0.23))
        arrow(ax, (0.648, 0.23), (0.676, 0.23))
    else:
        xs = [(0.03, 0.45), (0.52, 0.45)]
        for (x, w), (lbl, role) in zip(xs, bot_boxes):
            box(ax, x, 0.13, w, 0.20, lbl, role, size=11)
        arrow(ax, (0.48, 0.23), (0.52, 0.23))

    text(ax, 0.50, 0.06, footer, size=11, ha='center', weight='bold', color=COL['ink'])
    ex.save(fig, stem)


def build_chapter() -> None:
    ex = Exporter(HERE)

    draw_dual_band(
        ex,
        'ch23-100pb-commoncrawl-to-15t-tokens-curation-funnel',
        '100 PB CommonCrawl 原始网页七级提纯漏斗（前四级过滤）',
        'blue',
        [
            ('① 原始归档 100 PB\n正文抽取留 18 PB\n剥离标签与广告脚本', 'blue'),
            ('② 语言与启发式过滤\n18 PB -> 9.0 PB\nGopher/FineWeb 规则', 'purple'),
            ('③ 全局模糊+精确去重\n9.0 PB -> 3.0 PB\n剔除 66.7% 重复冗余', 'orange'),
        ],
        '模型质量分级与去污收束（算例 23-1：产出 15.0T 精品词元）',
        'teal',
        [
            ('④ 质量分级与合成改写：\n3.0 PB -> 2.424 PB\nNemotron-CC 抢救高价值页', 'blue'),
            ('⑤ 评测基准 13-gram 去污：\n最终留存 2.40 PB（2.40%）\n25 个快照合 15.0T 词元', 'teal'),
        ],
        '从 100 PB 原始爬虫到 2.40 PB 精品语料，去重与分级决定上限',
    )

    draw_dual_band(
        ex,
        'ch23-minhash-lsh-band-row-s-curve-probability',
        'MinHash LSH 局部敏感哈希分桶参数（M = b x r = 128）',
        'purple',
        [
            ('5-gram 词片集合\n128 个独立哈希函数\n签名仅 512 B/doc', 'blue'),
            ('划分为 b=16 个分桶\n每桶含 r=8 行签名\n桶内全等即触发候选', 'purple'),
            ('拐点阈值 s*=0.7071\n公式 (1/b)^(1/r)\n陡峭分离相似与无关', 'teal'),
        ],
        'S 型碰撞概率曲线 P(s) = 1 - (1 - s^r)^b 数值特征（算例 23-2）',
        'teal',
        [
            ('无关文档（s = 0.50）：\n误碰率仅 6.07%\n避免平方级无效比对', 'orange'),
            ('近似重复文档（s = 0.80）：\n捕获率高达 94.70%\ns=0.90 时捕获率 99.99%', 'teal'),
        ],
        '调节 (b, r) 行列配比，在计算开销与召回率间精确塑形',
    )

    draw_dual_band(
        ex,
        'ch23-distributed-minhash-shuffle-and-gpu-union-find',
        '50 亿篇文档（2.56 TB 签名）分布式 GPU 模糊去重流水线',
        'blue',
        [
            ('① GPU 哈希签名\n50 亿文档 x 512 B\n总签名体积 2.56 TB', 'blue'),
            ('② 按分桶全互换洗牌\n64 卡洗牌 2.52 TB\n同桶文档发往同卡', 'purple'),
            ('③ cuGraph 连通分量\n大星/小星指针跳跃\n毫秒级收敛聚簇代表', 'teal'),
        ],
        '分布式图连通分量（Union-Find）消除传递性重复簇（算例 23-3）',
        'teal',
        [
            ('传递性闭包聚类（A~B, B~C）：\n大星/小星指针跳跃迭代\n将百万级重复网页缩为单簇', 'blue'),
            ('每簇仅留最高质量代表文档：\n剔除 66.7% 跨站转载与镜像\n避免预训练双下降与背诵退化', 'teal'),
        ],
        '将文本去重转化为分布式图连通分量求解，是百亿文档去重标准解法',
    )

    draw_dual_band(
        ex,
        'ch23-cpu-datatrove-vs-gpu-nemo-curator-throughput',
        'CPU 集群（DataTrove）vs. GPU 流水线（NeMo Curator）对比',
        'orange',
        [
            ('4,096 核 CPU 集群：\n哈希与磁盘洗牌受限\n耗时 148.0 小时', 'orange'),
            ('64 张 H100 GPU 集群：\ncuDF + NVLink 洗牌\n仅需 1.85 小时（80x）', 'teal'),
        ],
        '百 TB 语料去重端到端耗时与云租用总成本（TCO，算例 23-3）',
        'teal',
        [
            ('算力吞吐跃迁（快 80.0 倍）：\n从 6.17 天缩至 111 分钟\n支撑预训练配方日级快速迭代', 'blue'),
            ('云端租用总成本压降 4.50 倍：\nCPU 需 42,600 美元\nGPU 仅需 9,472 美元', 'teal'),
        ],
        'GPU 高带宽显存与高速互连洗牌在数据去重中具备压倒性优势',
    )

    draw_dual_band(
        ex,
        'ch23-document-fuzzy-dedup-vs-suffix-array-substring-dedup',
        '文档级 MinHash 模糊去重与段落级后缀数组精确去重协同',
        'blue',
        [
            ('第一层：MinHash\n粒度：整篇文档级\n剔除改写与转载文章', 'blue'),
            ('第二层：后缀数组\n粒度：50 词元子串\n线性构建全局索引', 'purple'),
            ('原位剥离模板噪声\n切除页眉导航菜单\n与免责声明重复尾', 'teal'),
        ],
        '双层混合去重对模型困惑度（PPL）与记忆化泄露的联合抑制',
        'teal',
        [
            ('防止长文被整篇误杀：\n正文独特但含公用页脚的网页\n经后缀数组剥离页脚得以保留', 'blue'),
            ('消除隐私与版权原样背诵：\n连续 50 词元重复率 < 0.1%\n显著改善上下文学习泛化能力', 'teal'),
        ],
        '文档级模糊去重抓宏观冗余，后缀数组子串去重精准剔除微观模板',
    )

    draw_dual_band(
        ex,
        'ch23-llm-rubric-distillation-to-fasttext-bert-classifier',
        '大模型质量打分蒸馏（70B 教师 -> 110M 轻量分类器）三级杠杆',
        'purple',
        [
            ('① 70B 教师精标种子\n标注 500 万篇文档\n仅耗 217.0 卡时', 'blue'),
            ('② 蒸馏 110M 分类器\n学习教育与推理评分\n训练耗 80.0 卡时', 'purple'),
            ('③ 全量 500 亿篇打分\n吞吐 180k tok/s/卡\n耗 38,580 卡时', 'teal'),
        ],
        '直接 70B 全量打分 vs. 蒸馏轻量分类器算力账本对比（算例 23-4）',
        'teal',
        [
            ('直接用 70B 打分 500 亿篇：\n需 2,170,138.9 GPU 小时\n折合成本高达 868 万美元', 'orange'),
            ('三级蒸馏管线总计：\n仅需 38,877.2 GPU 小时\n算力节省 55.82 倍（省 98.2%）', 'teal'),
        ],
        '用教师模型定义品味、轻量模型扫平百亿语料，是数据分级的黄金法则',
    )

    draw_dual_band(
        ex,
        'ch23-nemotron-cc-classifier-ensembling-and-rephrasing',
        'Nemotron-CC 多分类器集成打分与合成改写抢救（Rephrasing）',
        'blue',
        [
            ('多分类器互补集成\n教育打分+指令打分\n划分 20 个质量桶', 'blue'),
            ('高质桶受控上采样\nTop-5% 精品训 3 轮\n低质桶尽早丢弃', 'purple'),
            ('高价值残缺页抢救\n含稀缺知识但排版碎\n送入大模型合成改写', 'teal'),
        ],
        '突破“激进过滤导致知识枯竭”的合成改写闭环',
        'teal',
        [
            ('为什么不能盲目丢弃中低分网页：\n许多排障贴格式杂乱但含稀缺经验\n直接过滤会损失长尾专业知识', 'orange'),
            ('百科风格合成改写（Rephrase）：\n将破碎问答重写为结构化教材段落\n使有效精品词元总量扩充 1.8 倍', 'teal'),
        ],
        '分类器集成实现精细分层，合成改写将粗糙矿石冶炼成精品语料',
    )

    draw_dual_band(
        ex,
        'ch23-benchmark-decontamination-ngram-and-embedding-hybrid',
        '评测基准防污染（Benchmark Decontamination）双层净化管线',
        'orange',
        [
            ('构建全基准指纹库\nMMLU/GSM8K/代码库\n提取全部测试题文本', 'blue'),
            ('第一道防线：13-gram\n多模式串自动机匹配\n拦截字面精确重合', 'orange'),
            ('第二道防线：语义向量\n稠密余弦 > 0.85\n拦截同义改写与翻译', 'purple'),
        ],
        '防止预训练语料刷榜失真（Benchmark Leakage）的审计闭环',
        'teal',
        [
            ('为什么单靠 13-gram 远远不够：\n网页常含翻译成中文的英文测试题\n或变量名被替换的算法竞赛原题', 'orange'),
            ('双层混合去污代价与收益：\n仅剔除 0.99%（2.424->2.40 PB）\n确保评测分数反映真实泛化能力', 'teal'),
        ],
        '字面 n-gram 结合稠密向量检索，是守卫基准评测科学性的必由之路',
    )

    draw_dual_band(
        ex,
        'ch23-doremi-regmix-domain-weight-optimization',
        '基于代理模型的领域配比寻优（DoReMi / RegMix）与三阶段退火',
        'blue',
        [
            ('① 小代理模型实验\n用 280M/1B 跑网格\n拟合领域权重曲面', 'blue'),
            ('② 求解最优配比\n提升代码与理科权重\n压降泛娱乐网页比例', 'purple'),
            ('③ 迁移至千亿主干\n零超参试错直接命中\n验证集损失帕累托', 'teal'),
        ],
        '现代预训练三阶段动态数据退火（Curriculum Annealing）',
        'teal',
        [
            ('阶段 1–2（前 85% 训练词元）：\n万亿级多样性网页建立世界常识\n中段逐步拉升代码与长文配比', 'blue'),
            ('阶段 3（末段 15% 极速退火期）：\n注入 70% 合成数学与精校百科\n伴随学习率余弦归零实现跃升', 'teal'),
        ],
        '随训练阶段动态调整数据配比，能以相同算力榨出显著更高模型性能',
    )

    draw_dual_band(
        ex,
        'ch23-online-serving-vs-offline-synthetic-factory-roofline',
        '在线低延迟推理服务 vs. 离线合成数据工厂的 Roofline 跃迁',
        'purple',
        [
            ('在线实时服务（B=32）：\n受制于 TPOT < 25 ms\n卡在 HBM 访存带宽墙\nMFU 仅 11.2%（1,920 tok/s）', 'orange'),
            ('离线合成工厂（B=2,048）：\n无交互延迟 SLO 约束\n推入 Tensor Core 计算饱和区\nMFU 达 58.4%（13,824 tok/s）', 'teal'),
        ],
        '卸下实时延迟枷锁后的单卡吞吐与单位词元成本剧变（算例 23-5）',
        'teal',
        [
            ('单卡生成吞吐暴增 7.20 倍：\n结合前缀共享与分页 KV 卸载\n每次权重读取被数千序列摊薄', 'blue'),
            ('原始词元成本暴降 13.16 倍：\n从在线 0.579 美元/百万词元\n降至离线 0.044 美元/百万词元', 'teal'),
        ],
        '离线合成数据工厂用延迟容忍度换取极致算术强度与吞吐/美元',
    )

    draw_dual_band(
        ex,
        'ch23-synthetic-data-generator-verifier-rejection-funnel',
        '合成数据工厂“生成—沙箱验证—重写抢救”闭环漏斗（算例 23-5）',
        'blue',
        [
            ('① 多样化高温采样\n每题并发采 N=16 条\n原始 0.044 美元/M', 'blue'),
            ('② 沙箱编译与单测\n首轮通过率 31.25%\n拦截幻觉与语法错', 'purple'),
            ('③ 错误轨迹自纠重写\n挽回 40% 失败样本\n良率升至 58.75%', 'teal'),
        ],
        '可验证奖励过滤（RLVR / PRM）杜绝“模型自噬（Model Collapse）”',
        'teal',
        [
            ('为什么未经验证的合成数据有害：\n同分布无过滤自蒸馏放大高频偏见\n导致尾部知识丢失与方差坍缩', 'orange'),
            ('外部真值检验器注入新信息熵：\n编译器、定理证明器与 PRM 把关\n验证词元成本仅 0.075 美元/M', 'teal'),
        ],
        '生成器提供广度，沙箱验证器提供真值，自纠重写最大化算力良率',
    )

    draw_dual_band(
        ex,
        'ch23-slack-capacity-harvesting-for-synthetic-generation',
        '利用集群昼夜波谷与训练维修空窗收割零边际成本合成算力',
        'teal',
        [
            ('推理集群昼夜潮汐\n凌晨 00–08 点低谷\n闲置 40%–65% 算力', 'blue'),
            ('抢占式合成作业注入\n秒级拉起大批次生成\n填满 100% 功耗基线', 'purple'),
            ('百毫秒级优雅让路\n早高峰流量回升时\n按块落盘并立即腾退', 'teal'),
        ],
        '离线合成数据工厂作为数据中心“算力海绵（Compute Sponge）”',
        'blue',
        [
            ('吸收训练集群重调度碎片：\n万卡训练故障缩容或存检查点时\n空闲节点瞬间切入合成生成队列', 'blue'),
            ('等效 GPU 小时单价再降 45%：\n盘活沉没折旧卡时（2.20 美元/h）\n年产数万亿精品推理与代码词元', 'teal'),
        ],
        '离线合成任务具备无状态可抢占性，是削峰填谷提升机队利用率的利器',
    )

    draw_dual_band(
        ex,
        'ch23-pdf-scientific-ocr-layout-math-extraction-pipeline',
        '科学文献与教材 PDF 多模态版面解析及公式还原流水线',
        'blue',
        [
            ('① 版面检测与阅读序\n精准识别双栏、表格\n与跨页插图拓扑', 'blue'),
            ('② 公式与表格还原\n视觉语言小模型转写\nLaTeX 与 Markdown', 'purple'),
            ('③ 乱码与符号清洗\n修复字体映射缺失\n构建图文交错语料', 'teal'),
        ],
        '攻克高阶 STEM 知识壁垒（Nougat / MinerU / olmOCR 范式）',
        'teal',
        [
            ('传统纯文本提取器的灾难：\n分栏错乱、上下标丢失、积分乱码\n使模型学到支离破碎的错误公式', 'orange'),
            ('端到端视觉文档解析吞吐优化：\nTensorRT 加速轻量 VLM 光栅扫描\n单卡每日解析超 50 万页科技文献', 'teal'),
        ],
        '高精度 PDF 版面与公式还原是解锁人类高阶 STEM 知识宝库的钥匙',
    )

    draw_dual_band(
        ex,
        'ch23-tokenized-dataset-mmap-shuffling-and-sequence-packing',
        '万亿词元二进制索引（mmap）全局伪随机洗牌与最佳适应打包',
        'purple',
        [
            ('零拷贝内存映射读取\n.bin 流 + .idx 偏移\nOS 页缓存按需预取', 'blue'),
            ('O(1) 显存全局洗牌\n块级 Feistel 置换\n免物化百亿索引表', 'purple'),
            ('最佳适应序列打包\n线段树匹配剩余空间\n消除文档截断与填充', 'teal'),
        ],
        '朴素拼接截断（Concat-Chunk）vs. 最佳适应无截断打包（OBFD）',
        'teal',
        [
            ('朴素拼接截断的严重隐患：\n15%–25% 长代码与证明被拦腰截断\n破坏跨函数调用与长链因果完整', 'orange'),
            ('最佳适应无截断打包（OBFD）：\n填充率 < 0.5% 且保持文档完整边界\n配合变长注意力杜绝跨文档串扰', 'teal'),
        ],
        '无截断最佳适应打包兼顾了 99.5% 算力填充效率与长程因果语义完整性',
    )

    draw_dual_band(
        ex,
        'ch23-data-lineage-copyright-optout-and-audit-ledger',
        '全生命周期语料血缘账本（Data Provenance）与合规退订回溯',
        'blue',
        [
            ('抓取与过滤数字签名\n记录来源 URL、协议\n与许可证 SPDX 标签', 'blue'),
            ('块级 Merkle 哈希树\n绑定每个训练批次\n至原始文档 SHA256', 'purple'),
            ('版权退订与遗忘接口\n毫秒级定位目标分片\n支持增量退火擦除', 'teal'),
        ],
        '面向主权 AI 与企业合规审计的可验证数据治理底座',
        'teal',
        [
            ('亚秒级反查任意训练步语料来源：\n当模型出现异常输出或合规质询时\n经倒排哈希索引瞬间追溯源头文档', 'blue'),
            ('零停机动态黑名单原地屏蔽：\n在 .idx 索引层原地标记屏蔽位\n免重写数 PB 底层 .bin 数据文件', 'teal'),
        ],
        '可追溯的密码学数据血缘账本，是万亿词元大模型跨越合规红线的护城河',
    )

    draw_dual_band(
        ex,
        'ch23-exascale-data-and-synthetic-factory-master-architecture',
        '百亿亿次 Token 数据工程与合成数据工厂全栈架构总图',
        'teal',
        [
            ('支柱一：七级提纯漏斗\n100 PB -> 15T 词元\n2.40% 严苛留存率', 'blue'),
            ('支柱二：GPU 模糊去重\nMinHash + cuGraph\n80x 提速与 4.5x 降本', 'purple'),
            ('支柱三：合成数据工厂\nB=2048 推满 58.4%\n沙箱验证 0.075 美元/M', 'teal'),
        ],
        '从“清洗人类存量网页”到“算力冶炼合成智慧”的范式交接',
        'blue',
        [
            ('人类自然文本存量逼近枯竭：\nGPU 去重与分类蒸馏（55.8x 杠杆）\n榨干存量互联网每一滴高价值信号', 'blue'),
            ('合成数据开启第二增长曲线：\n将空闲算力转化为经沙箱验证的\n数学、代码与智能体轨迹驱动进化', 'teal'),
        ],
        '数据工程已演进为与预训练集群同等规模的核心 AI 基础设施',
    )

    draw_dual_band(
        ex,
        'ch23-mineru-olmocr-pdf-layout-and-formula-gpu-pipeline',
        '百万册学术 PDF 解析：数字页直抽与公式框动态批处理',
        'blue',
        [
            ('① CMap 字符探针\n65% 原生数字页\n正文零拷贝直抽', 'blue'),
            ('② 跨页子框微批\n公式/表格 B=256\n均摊仅 11.00 ms', 'purple'),
            ('③ 扫描页 olmOCR\n35% 乱码扫描页\n连续批处理 30.2 ms', 'teal'),
        ],
        '200 万册（5 亿页）文献库：加权 17.72 ms/页 vs 单体 VLM 625 ms/页',
        'teal',
        [
            ('单体 7B VLM 全页解码瓶颈：\n1.60 页/s · 86,805.6 GPU-h\n已有数字层重复自回归浪费算力', 'orange'),
            ('MinerU + olmOCR 混合双通路：\n56.43 页/s（提速 35.27 倍）\n2,461.1 GPU-h · 0.0135 USD/M', 'teal'),
        ],
        '区分原生数字页与扫描页并对公式子框跨页批处理，将吞吐推高 35.27 倍',
    )

    draw_dual_band(
        ex,
        'ch23-repo-level-ast-dependency-topological-packing',
        '代码仓库 64k 预训练：AST 依赖提取、Tarjan 破环与拓扑装箱',
        'purple',
        [
            ('① Tree-sitter\n解析 import/符号\n建跨文件有向图', 'blue'),
            ('② Tarjan 强连通\n凝聚循环导入环\n化简为严格 DAG', 'purple'),
            ('③ Kahn 拓扑装箱\n被调方排调用方前\n64k 装箱率 98.8%', 'teal'),
        ],
        '因果掩码下跨文件定义可见率：字母序 42.50% -> 拓扑序 95.00%',
        'teal',
        [
            ('路径字母序拼接（42.50%）：\napp/ 排在 core/ 与 utils/ 之前\n57.5% 跨文件调用沦为无定义盲猜', 'orange'),
            ('AST 拓扑序装箱（95.00%）：\n基础库与工具定义驻留左侧前缀\n跨文件 API 困惑度降 39.85%', 'teal'),
        ],
        '按 AST 依赖拓扑排序让因果自注意力自然贴合软件工程的真实调用流向',
    )

    outputs = ex.finish()

    fig_index = [
        {
            "figure_id": "图 23-1",
            "number": 1,
            "file": "ch23-100pb-commoncrawl-to-15t-tokens-curation-funnel.svg",
            "asset": "ch23/ch23-100pb-commoncrawl-to-15t-tokens-curation-funnel.svg",
            "section": "23.1.1",
            "caption": "从 100 PB CommonCrawl 原始 WARC 归档经七级提纯漏斗产出 15.0T 精品词元的体积与算力瀑布（算例 23-1）",
            "source_refs": ["experiments/ch23/23-01/run.py", "manuscripts/ch23/figure-data.json"],
            "TeachingRole": "展示百 PB 级原始网页经正文抽取、启发式过滤、去重、分级与去污仅留存 2.40% 的全链路账本"
        },
        {
            "figure_id": "图 23-2",
            "number": 2,
            "file": "ch23-minhash-lsh-band-row-s-curve-probability.svg",
            "asset": "ch23/ch23-minhash-lsh-band-row-s-curve-probability.svg",
            "section": "23.2.1",
            "caption": "MinHash LSH 局部敏感哈希在 (b=16, r=8) 分桶配置下的 S 型碰撞概率曲线 P(s) = 1 - (1 - s^r)^b（算例 23-2）",
            "source_refs": ["experiments/ch23/23-01/run.py", "manuscripts/ch23/figure-data.json"],
            "TeachingRole": "推导阈值拐点 s* = 0.7071 以及 s=0.80 捕获率 94.70%、s=0.50 误碰率 6.07% 的概率边界"
        },
        {
            "figure_id": "图 23-3",
            "number": 3,
            "file": "ch23-distributed-minhash-shuffle-and-gpu-union-find.svg",
            "asset": "ch23/ch23-distributed-minhash-shuffle-and-gpu-union-find.svg",
            "section": "23.2.2",
            "caption": "50 亿篇文档模糊去重的分布式流水线：Shingle 哈希、按 Band 洗牌与 GPU 连通分量（Large-Star/Small-Star）聚簇（算例 23-3）",
            "source_refs": ["experiments/ch23/23-01/run.py", "manuscripts/ch23/figure-data.json"],
            "TeachingRole": "揭示 2.56 TB 签名跨卡洗牌与分布式图连通分量求解传递性重复簇的系统机理"
        },
        {
            "figure_id": "图 23-4",
            "number": 4,
            "file": "ch23-cpu-datatrove-vs-gpu-nemo-curator-throughput.svg",
            "asset": "ch23/ch23-cpu-datatrove-vs-gpu-nemo-curator-throughput.svg",
            "section": "23.2.2",
            "caption": "CPU 集群（DataTrove）与 GPU 加速流水线（NeMo Curator + cuDF + cuGraph）在百 TB 语料去重中的耗时（80.0x 加速）与 TCO（降 4.50x）对比（算例 23-3）",
            "source_refs": ["experiments/ch23/23-01/run.py", "manuscripts/ch23/figure-data.json"],
            "TeachingRole": "对比 4,096 核 CPU 跑 148 小时与 64 卡 H100 跑 1.85 小时的吞吐与成本账本"
        },
        {
            "figure_id": "图 23-5",
            "number": 5,
            "file": "ch23-document-fuzzy-dedup-vs-suffix-array-substring-dedup.svg",
            "asset": "ch23/ch23-document-fuzzy-dedup-vs-suffix-array-substring-dedup.svg",
            "section": "23.2.2",
            "caption": "文档级 MinHash 模糊去重与基于后缀数组（Suffix Array）的 50 词元级精确重复模板/导航栏剥离协同",
            "source_refs": ["manuscripts/ch23/figure-data.json"],
            "TeachingRole": "阐明宏观整篇模糊去重与微观后缀数组子串切除的互补分工"
        },
        {
            "figure_id": "图 23-6",
            "number": 6,
            "file": "ch23-llm-rubric-distillation-to-fasttext-bert-classifier.svg",
            "asset": "ch23/ch23-llm-rubric-distillation-to-fasttext-bert-classifier.svg",
            "section": "23.3.1",
            "caption": "用 70B 教师模型标注 500 万样本蒸馏 110M 轻量级质量分类器（FineWeb-Edu / DCLM）实现 55.82 倍算力节省的三级漏斗（算例 23-4）",
            "source_refs": ["experiments/ch23/23-02/run.py", "manuscripts/ch23/figure-data.json"],
            "TeachingRole": "证明教师精标种子与轻量分类器全量推理相比直接用大模型打分节省 98.21% 算力"
        },
        {
            "figure_id": "图 23-7",
            "number": 7,
            "file": "ch23-nemotron-cc-classifier-ensembling-and-rephrasing.svg",
            "asset": "ch23/ch23-nemotron-cc-classifier-ensembling-and-rephrasing.svg",
            "section": "23.3.1",
            "caption": "Nemotron-CC 范式：多分类器集成打分分桶与对高知识价值但表述残缺网页的合成改写抢救（Synthetic Rephrasing）",
            "source_refs": ["manuscripts/ch23/figure-data.json"],
            "TeachingRole": "通过多分类器集成与百科风格合成改写突破激进过滤带来的独特词元枯竭墙"
        },
        {
            "figure_id": "图 23-8",
            "number": 8,
            "file": "ch23-benchmark-decontamination-ngram-and-embedding-hybrid.svg",
            "asset": "ch23/ch23-benchmark-decontamination-ngram-and-embedding-hybrid.svg",
            "section": "23.3.2",
            "caption": "防止评测集刷榜失真的双层去污管线（13-gram 子串匹配 + 稠密语义向量近邻剔除防止同义改写与跨语言污染）",
            "source_refs": ["manuscripts/ch23/figure-data.json"],
            "TeachingRole": "构建字面 n-gram 自动机与稠密向量检索结合的评测基准防泄漏防线"
        },
        {
            "figure_id": "图 23-9",
            "number": 9,
            "file": "ch23-doremi-regmix-domain-weight-optimization.svg",
            "asset": "ch23/ch23-doremi-regmix-domain-weight-optimization.svg",
            "section": "23.3.2",
            "caption": "基于小型代理模型（Proxy Model）的领域配比寻优（DoReMi / RegMix）与三阶段预训练动态退火曲线",
            "source_refs": ["manuscripts/ch23/figure-data.json"],
            "TeachingRole": "展示如何用小模型网格搜索最优领域配比并在预训练末段执行高纯度数据极速退火"
        },
        {
            "figure_id": "图 23-10",
            "number": 10,
            "file": "ch23-online-serving-vs-offline-synthetic-factory-roofline.svg",
            "asset": "ch23/ch23-online-serving-vs-offline-synthetic-factory-roofline.svg",
            "section": "23.4.1",
            "caption": "在线低延迟服务（B=32，MFU 11.2%）与离线合成数据工厂（B=2,048，MFU 58.4%，吞吐升 7.20 倍、成本降 13.16 倍）的 Roofline 跃迁（算例 23-5）",
            "source_refs": ["experiments/ch23/23-02/run.py", "manuscripts/ch23/figure-data.json"],
            "TeachingRole": "阐明卸下实时延迟枷锁后通过超大批量将自回归解码推入 Tensor Core 计算饱和区的经济学"
        },
        {
            "figure_id": "图 23-11",
            "number": 11,
            "file": "ch23-synthetic-data-generator-verifier-rejection-funnel.svg",
            "asset": "ch23/ch23-synthetic-data-generator-verifier-rejection-funnel.svg",
            "section": "23.4.2",
            "caption": "推理与代码合成数据工厂的“多样化种子 -> 高温批量采样 -> 沙箱测试验证（31.25%）-> 自纠重写（综合良率 58.75%）”闭环账本（算例 23-5）",
            "source_refs": ["experiments/ch23/23-02/run.py", "manuscripts/ch23/figure-data.json"],
            "TeachingRole": "利用外部确定性沙箱验证器与错误轨迹重写防止模型自噬并将验证词元成本锁定在 0.075 美元/M"
        },
        {
            "figure_id": "图 23-12",
            "number": 12,
            "file": "ch23-slack-capacity-harvesting-for-synthetic-generation.svg",
            "asset": "ch23/ch23-slack-capacity-harvesting-for-synthetic-generation.svg",
            "section": "23.4.1",
            "caption": "利用在线推理集群昼夜波谷与训练维护空窗自动拉起离线合成数据作业的零边际硬件成本收割架构",
            "source_refs": ["experiments/ch23/23-02/run.py", "manuscripts/ch23/figure-data.json"],
            "TeachingRole": "通过毫秒级抢占式波谷算力收割将离线合成 GPU 小时单价再压降 45%"
        },
        {
            "figure_id": "图 23-13",
            "number": 13,
            "file": "ch23-pdf-scientific-ocr-layout-math-extraction-pipeline.svg",
            "asset": "ch23/ch23-pdf-scientific-ocr-layout-math-extraction-pipeline.svg",
            "section": "23.1.1",
            "caption": "高质量学术论文与教材 PDF 语料的版面分析、公式 LaTeX 还原与多模态图文交错提取流水线",
            "source_refs": ["manuscripts/ch23/figure-data.json"],
            "TeachingRole": "解决传统 PDF 文本提取破坏分栏与数学公式的难题，解锁高阶 STEM 专业文献"
        },
        {
            "figure_id": "图 23-14",
            "number": 14,
            "file": "ch23-tokenized-dataset-mmap-shuffling-and-sequence-packing.svg",
            "asset": "ch23/ch23-tokenized-dataset-mmap-shuffling-and-sequence-packing.svg",
            "section": "23.3.2",
            "caption": "万亿词元二进制索引（.bin/.idx mmap）的确定性全局伪随机洗牌与最佳适应序列打包（Best-Fit Packing）零填充损耗",
            "source_refs": ["manuscripts/ch23/figure-data.json"],
            "TeachingRole": "通过 Feistel 块级置换与最佳适应无截断打包消除长文档腰斩与跨文档注意力串扰"
        },
        {
            "figure_id": "图 23-15",
            "number": 15,
            "file": "ch23-data-lineage-copyright-optout-and-audit-ledger.svg",
            "asset": "ch23/ch23-data-lineage-copyright-optout-and-audit-ledger.svg",
            "section": "23.3.2",
            "caption": "面向合规与可追溯性的语料血缘账本（Data Provenance）、版权退订剔除与训练批次精确回溯索引",
            "source_refs": ["manuscripts/ch23/figure-data.json"],
            "TeachingRole": "建立从每个训练 Batch 反向追溯至原始网页 SHA-256 与许可证标签的密码学审计链路"
        },
        {
            "figure_id": "图 23-16",
            "number": 16,
            "file": "ch23-exascale-data-and-synthetic-factory-master-architecture.svg",
            "asset": "ch23/ch23-exascale-data-and-synthetic-factory-master-architecture.svg",
            "section": "23.4.2",
            "caption": "百亿亿次 Token 数据工程与合成数据工厂从原始爬虫、GPU 去重到闭环合成退火的全栈总图",
            "source_refs": ["experiments/ch23/23-01/run.py", "experiments/ch23/23-02/run.py", "manuscripts/ch23/figure-data.json"],
            "TeachingRole": "收束第 23 章七级提纯漏斗、GPU 连通分量去重、分类器蒸馏与合成数据工厂的完整体系"
        },
        {
            "figure_id": "图 23-17",
            "number": 17,
            "file": "ch23-mineru-olmocr-pdf-layout-and-formula-gpu-pipeline.svg",
            "asset": "ch23/ch23-mineru-olmocr-pdf-layout-and-formula-gpu-pipeline.svg",
            "section": "23.4.2",
            "caption": "百万册级复杂学术 PDF 版面解析与公式还原（MinerU / olmOCR 混合快慢分流与跨页子区域 TensorRT 动态批处理，吞吐提速 35.27 倍）（算例 23-6）",
            "source_refs": ["experiments/ch23/23-03/run.py", "manuscripts/ch23/figure-data.json"],
            "TeachingRole": "通过 65% 原生数字页正文直抽与跨页公式框 TensorRT 动态批处理将单卡吞吐从 1.60 页/s 推高至 56.43 页/s"
        },
        {
            "figure_id": "图 23-18",
            "number": 18,
            "file": "ch23-repo-level-ast-dependency-topological-packing.svg",
            "asset": "ch23/ch23-repo-level-ast-dependency-topological-packing.svg",
            "section": "23.4.2",
            "caption": "代码仓库级跨文件 AST 依赖图提取、Tarjan 强连通破环与 Kahn 拓扑排序序列打包（因果依赖覆盖率从 42.50% 跃升至 95.00%）（算例 23-7）",
            "source_refs": ["experiments/ch23/23-03/run.py", "manuscripts/ch23/figure-data.json"],
            "TeachingRole": "消除路径字母序拼接导致的跨文件定义盲区，使 64k 代码预训练跨文件 API 困惑度降低 39.85%"
        }
    ]
    for row in fig_index:
        row['figure'] = f"23-{row['number']}"
        row['name'] = Path(row['file']).stem
    (HERE / 'figure-index.json').write_text(json.dumps(fig_index, ensure_ascii=False, indent=2) + '\n')

    md = HERE.parent / '23-百亿亿次 Token 数据工程、GPU 加速模糊去重与合成数据工厂底座.md'
    if md.exists():
        text_md = md.read_text()
        display = re.findall(r'\$\$(.+?)\$\$', text_md, re.S)
        inline = re.findall(r'(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)', text_md)
        formulas = [f.strip() for f in display + inline]
        katex_js = ROOT / 'manuscripts/ch06/vendor/katex/katex.js'
        js = (
            'const katex=require(process.argv[1]);const fs=require("fs");'
            'const a=JSON.parse(fs.readFileSync(0,"utf8"));const e=[];'
            'a.forEach((s,i)=>{try{katex.renderToString(s,{throwOnError:true,displayMode:i<'
            + str(len(display))
            + '})}catch(err){e.push({i,s,err:err.message})}});'
            'console.log(JSON.stringify({total:a.length,display:'
            + str(len(display))
            + ',inline:'
            + str(len(inline))
            + ',errors:e}));'
        )
        res = subprocess.run(
            ['node', '-e', js, str(katex_js)],
            input=json.dumps(formulas),
            text=True,
            capture_output=True,
            check=True,
        )
        math_val = json.loads(res.stdout)
        assert not math_val['errors'], math_val['errors']
        (HERE / 'math-validation.json').write_text(json.dumps(math_val, ensure_ascii=False, indent=2) + '\n')

        import markdown

        def repl_img(m):
            alt, rel = m.group(1), m.group(2)
            p = (md.parent / rel).with_suffix('.png')
            b64 = base64.b64encode(p.read_bytes()).decode()
            return f'<figure><img alt="{alt}" src="data:image/png;base64,{b64}"><figcaption>{alt}</figcaption></figure>'

        work = re.sub(r'!\[([^\]]*)\]\((ch23/[^)]+)\)', repl_img, text_md)
        body = markdown.markdown(work, extensions=['tables', 'fenced_code', 'footnotes', 'toc'])
        html = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>第 23 章：百亿亿次 Token 数据工程、GPU 加速模糊去重与合成数据工厂底座</title></head><body>{body}</body></html>\n'
        out_html = preview_path(md)
        out_html.write_text(html)

    sources = [
        HERE / 'build.py',
        HERE / 'figure-data.json',
        HERE / 'figure-index.json',
        ROOT / 'experiments/ch23/23-01/run.py',
        ROOT / 'experiments/ch23/23-02/run.py',
        ROOT / 'experiments/ch23/23-03/run.py',
    ]
    if md.exists():
        sources.append(md)
    (HERE / 'sources.json').write_text(
        json.dumps(
            {"sources": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in sources]},
            ensure_ascii=False,
            indent=2,
        )
        + '\n'
    )
    all_outputs = sorted(outputs) + [
        HERE / 'figure-data.json',
        HERE / 'figure-index.json',
        HERE / 'teaching-layout-validation.json',
    ]
    if (HERE / 'math-validation.json').exists():
        all_outputs.append(HERE / 'math-validation.json')
    (HERE / 'manifest.json').write_text(
        json.dumps(
            {"outputs": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in all_outputs]},
            ensure_ascii=False,
            indent=2,
        )
        + '\n'
    )
    print(f'Built {len(outputs)} figure files for Chapter 23; 0 layout warnings.')


if __name__ == '__main__':
    build_chapter()
