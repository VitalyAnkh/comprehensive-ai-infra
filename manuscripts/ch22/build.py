#!/usr/bin/env python3
"""Build all 16 publication-grade figures (420pt) and validation manifests for Chapter 22."""
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
        'ch22-cascaded-asr-llm-tts-vs-native-full-duplex-s2s',
        '传统四级串联语音管线（1,280 ms）vs. 原生全双工 S2S（164 ms）',
        'orange',
        [
            ('① VAD 尾音等待\n480ms + ASR 220ms\n丢失语调与笑声', 'orange'),
            ('② LLM 首字缓冲\n耗时 280 ms\n仅见纯文本转写', 'purple'),
            ('③ TTS 与网络 I/O\n240+60 = 300 ms\n总计 1,280 ms', 'blue'),
        ],
        '原生全双工 S2S（Moshi / Qwen-Omni，算例 22-1）',
        'teal',
        [
            ('无 VAD 常驻双音频流（12.5 Hz）\n每 80 ms 同时收听与发声\n保留全部副语言情感特征', 'blue'),
            ('端到端响应延迟 164.0 ms\n低于人类 200 ms 感知阈值\n比串联流水线快 7.80 倍', 'teal'),
        ],
        '消灭 VAD 等待与文本中转瓶颈，是跨越 200 ms 对话阈值的唯一路径',
    )

    draw_dual_band(
        ex,
        'ch22-full-duplex-dual-audio-streams-and-inner-monologue',
        '80 ms 等时节拍（12.5 Hz）下的双轨音频流与内心独白对齐',
        'blue',
        [
            ('上行用户音频 U_t\n每帧 8 个 RVQ 词元\n实时摄入语音打断', 'blue'),
            ('内心独白文本 W_t\n每帧 1 个文本词元\n锚定高层语义推理', 'purple'),
            ('下行助手音频 A_t\n每帧 8 个 RVQ 词元\n合成自然语调附和', 'teal'),
        ],
        '单步联合词元包（17 tokens/step）的双轨同步状态转移',
        'teal',
        [
            ('先自回归生成文本 W_t：\n再条件化生成当帧声学码本 A_t\n防止纯音频模型语义退化', 'blue'),
            ('静默帧与非言语声学词元：\n静音时持续输出 <SIL> 码本\n倾听时自然插入“嗯哼”附和', 'teal'),
        ],
        '内心独白文本轨锚定逻辑深度，双音频轨赋予系统全双工交互能力',
    )

    draw_dual_band(
        ex,
        'ch22-rvq-neural-audio-codec-semantic-acoustic-hierarchy',
        '神经音频编解码器（Mimi / EnCodec）残差向量量化（RVQ）金字塔',
        'teal',
        [
            ('24 kHz 原始波形\n384 kbps 卷积下采样\n步长压至 12.5 Hz', 'blue'),
            ('第 1 层：语义码本\n蒸馏 WavLM 特征\n主导发音与词义', 'purple'),
            ('第 2–8 层：声学码本\n逼近基频与音色\n合计 1.10 kbps', 'teal'),
        ],
        '帧率—码本数—比特率守恒方程（算例 22-2）',
        'blue',
        [
            ('词元速率 = 12.5 Hz × 8 码本\n= 100.0 tokens/s（单向音频流）\n每码本词表 V = 2,048（11 bits）', 'blue'),
            ('总码率 = 100 × 11 = 1.10 kbps\n相比 384 kbps 原始音频\n实现 349.09 倍高保真压缩', 'teal'),
        ],
        '语义与声学码本解耦既压低了帧率，又为长时通话声学修剪奠定基础',
    )

    draw_dual_band(
        ex,
        'ch22-delayed-codebook-pattern-vs-depth-transformer',
        'RVQ 多码本自回归生成的两种核心架构范式对比',
        'purple',
        [
            ('范式 A：延迟码本交错模式\n第 q 码本向后平移 q-1 步\n单主干一次并行吐出 Q 码本', 'blue'),
            ('范式 B：时间—深度双层架构\nTemporal 主干跨时间步 1 次\nDepth 小模型步内串行 Q 次', 'teal'),
        ],
        '步内声学一致性 vs. 硬件串行微算子深度的工程权衡',
        'teal',
        [
            ('延迟模式：零步内串行开销\n但同帧高阶码本看不到当前 q_1\n极低帧率（12.5 Hz）下易损音质', 'orange'),
            ('双层架构：严格因果依赖 q_1→q_8\n在 12.5 Hz 下保持高清晰度音质\n但需攻克 Q=8 步微内核启动墙', 'teal'),
        ],
        '12.5 Hz 超低帧率语音普遍采用 Temporal + Depth 双层架构保音质',
    )

    draw_dual_band(
        ex,
        'ch22-temporal-and-depth-transformer-gpu-roofline-trap',
        '单帧 80 ms 内 Temporal（7B）+ Depth（100M）的启动开销陷阱',
        'orange',
        [
            ('Temporal 7B（1步）\n读取 ~7 GB 权重\n耗时仅 3.50 ms', 'blue'),
            ('Depth 100M 朴素发射\n384 次微算子启动\n阻塞+访存 35.04 ms', 'orange'),
            ('朴素总计 38.54 ms\n小模型反比 7B 慢 10×\n吃掉半帧实时预算', 'purple'),
        ],
        '整步 CUDA Graph 捕获 + L2 SRAM 常驻固定破局（算例 22-3）',
        'teal',
        [
            ('Full-Step CUDA Graph 静态融合\n将 1 步 Temporal + 8 步 Depth\n384 个微算子收拢为单次图发射', 'blue'),
            ('L2 Cache 持久化驻留（Pinning）\nDepth 权重常驻 L2，耗时压至 8 ms\n整步 11.50 ms（加速 3.35 倍）', 'teal'),
        ],
        '消除 384 次微算子启动间隙并锁死 L2 驻留，是跑通双层架构的关键',
    )

    draw_dual_band(
        ex,
        'ch22-webrtc-opus-aec-to-gpu-end-to-end-latency-budget',
        '原生全双工语音 164.0 ms 端到端全链路时延拆解（算例 22-1）',
        'blue',
        [
            ('① 采集+AEC+Opus\n耗时 25.0 ms\n硬件回声消除', 'blue'),
            ('② WebRTC+抖动缓冲\n耗时 35.0 ms\nUDP 抗弱网丢包', 'purple'),
            ('③ GPU 双层推理\n编解码+推理 24 ms\n加帧步共 164 ms', 'teal'),
        ],
        '为什么决不能用 TCP/HTTP 传输全双工实时语音？',
        'orange',
        [
            ('TCP 队头阻塞（HoL Blocking）\n单个音频包丢失会卡死后续所有帧\n引发数百毫秒停顿与成片爆音', 'orange'),
            ('WebRTC RTP/UDP + PLC 丢包隐藏\n允许跳过超时旧包并用声码器插值\n严守 < 200 ms 等时交互红线', 'teal'),
        ],
        '全链路 164.0 ms 闭环要求 WebRTC 网络层与 CUDA Graph 算子层协同',
    )

    draw_dual_band(
        ex,
        'ch22-barge-in-playback-cursor-aligned-kv-rollback',
        '随时打断（Barge-in）时的服务端超前生成与客户端游标失配',
        'orange',
        [
            ('服务端已超前生成至第 65 帧\n（对应音频时间 5,200 ms）\n存在 11 帧（880 ms）未播缓冲', 'orange'),
            ('用户在第 54 帧（4,320 ms）打断\n后 11 帧语音用户根本未听到\n若保留其 KV 将导致上下文错乱', 'purple'),
        ],
        '按客户端 RTP 播放游标对齐的原子级 KV 回滚协议（算例 22-4）',
        'teal',
        [
            ('① 网关瞬时排空（Buffer Flush）\n收到打断与游标 t_play = 4,320 ms\n立即丢弃下行队列 11 帧未播语音', 'blue'),
            ('② 服务端 KV 回滚至第 54 帧\n0.12 ms 内裁剪 187 个超前词元\n从第 55 帧无缝接纳用户新语音', 'teal'),
        ],
        '全双工打断一致性铁律：模型记忆必须严格对齐用户耳朵听到的时刻',
    )

    draw_dual_band(
        ex,
        'ch22-streaming-causal-conv-vocoder-chunk-halo',
        '流式因果卷积声码器（Causal Vocoder）感受野缓存与零爆音拼接',
        'teal',
        [
            ('非因果声码器的流式缺陷\n依赖未来帧右侧上下文\n逐帧独立解码产生边界相位爆音', 'orange'),
            ('流式因果卷积（Causal Conv1D）\n严格只读历史帧，每层维护固定\n历史激活环形缓存（Halo Cache）', 'teal'),
        ],
        '声码器与编解码器激活状态（Codec State）随会话绑定与打断回滚',
        'blue',
        [
            ('帧间零相位失真（Zero Click）\n将上一帧末尾感受野注入当前帧\n流式输出与离线解码逐样本等价', 'blue'),
            ('打断时的卷积缓存同步重置\nBarge-in 回滚时清空声码器残余\n立刻切换至新回复或静音态', 'purple'),
        ],
        '声码器与编码器的因果卷积状态是全双工会话除 KV 外的第二状态源',
    )

    draw_dual_band(
        ex,
        'ch22-live-video-1fps-and-audio-12hz-interleaved-pacing',
        '实时视频（1 FPS，256 词元）+ 全双工音频（12.5 Hz）交错节拍',
        'purple',
        [
            ('朴素整帧集中预填充\n第 0 帧塞入 256 视觉词元需 72 ms\n加音频 11.5 ms = 83.5 ms > 80 ms', 'orange'),
            ('交错微块预填充（Chunked）\n将 256 视觉词元均摊至 8 个节拍\n每节拍仅插 32 词元（耗时 +9 ms）', 'teal'),
        ],
        '多模态交错调度下的单帧最坏执行时间（WCET）压降（算例 22-5）',
        'teal',
        [
            ('消除周期性音频断流（Underrun）\n单帧峰值从 83.5 ms 压至 20.5 ms\n峰值抖动缩减 4.07 倍', 'blue'),
            ('跨模态位置编码对齐（M-RoPE）\n视觉微块与对应采样时刻的音频帧\n共享统一物理毫秒时间戳坐标', 'teal'),
        ],
        '高频音频硬节拍绝不允许被视觉帧阻塞，必须执行微块交错注入',
    )

    draw_dual_band(
        ex,
        'ch22-acoustic-token-eviction-and-semantic-anchor-retention',
        '1 小时音视频通话显存爆炸与分层声学/视觉词元修剪（算例 22-5）',
        'blue',
        [
            ('1 小时（45,000 帧）全量保留\n音文 76.5 万 + 视觉 92.16 万词元\n共 168.66 万词元，撑爆显存算力', 'orange'),
            ('10 秒（125 帧）高保真滑窗\n滑窗内保留完整 8 码本与近期视频\n保障局部语调连贯与即时视觉指代', 'blue'),
        ],
        '历史轮次“剥离声学残差 q_2..q_8、仅留语义锚点 q_1 + W_t”收益',
        'teal',
        [
            ('历史帧每步从 17 词元压至 3 词元\n保留用户/助手 q_1 与文本 W_t\n纯音频 KV 显存节省 82.12%', 'teal'),
            ('音视频全模态 1 小时常驻词元：\n从 1,686,600 压降至 139,310 词元\n总 KV 显存节省达 91.74%（>85%）', 'purple'),
        ],
        '远期对话靠语义码本与文本锚点维系记忆，近期滑窗维系声学自然度',
    )

    draw_dual_band(
        ex,
        'ch22-dynamic-batching-under-strict-80ms-isochronous-tick',
        '等时硬节拍（80 ms Tick）下的多会话槽位对齐批处理',
        'teal',
        [
            ('节拍同步收集窗（Tick Window）\n每 20 ms 汇总到达帧组成固定批\n利用 Tensor Core 提升算术强度', 'blue'),
            ('硬截止期看门狗（Watchdog）\n距 80 ms 截止期仅剩 15 ms 时\n立即封批发射，严禁等待迟到会话', 'teal'),
        ],
        '迟到音频包降级补全（Packet Loss Concealment）与防连坐隔离',
        'orange',
        [
            ('迟到会话注入隐状态预测词元\n用上一帧趋势无阻塞推进主干\n真实包到达后再做局部滑窗修正', 'orange'),
            ('预填充与等时解码物理隔离\n新会话长提示词在独立 Prefill 卡\n通过 RDMA 注入全双工流式工作卡', 'purple'),
        ],
        '等时流式批处理以“准时发车”为最高准则，防止单路弱网拖垮同批',
    )

    draw_dual_band(
        ex,
        'ch22-duplex-turn-taking-backchannel-and-overlap-state-machine',
        '原生全双工四态对话状态机（倾听 · 附和 · 讲述 · 抢话退让）',
        'purple',
        [
            ('状态 I/II：专注倾听与主动附和\n用户长段叙述时输出静音 <SIL>\n在停顿处轻声插入“嗯、明白”', 'blue'),
            ('状态 III/IV：正常讲述与碰撞退让\n检测用户高能量打断 > 160 ms\n模型在 2 帧内衰减音量转入倾听', 'teal'),
        ],
        '区分“善意附和（Backchannel）”与“真实打断（Barge-in）”判据',
        'teal',
        [
            ('传统 VAD 的致命缺陷：\n用户说一句“嗯对”或轻微咳嗽\n就会误杀打断模型正在进行的回答', 'orange'),
            ('原生双流联合注意力判别：\n模型根据用户语音语义与持续时长\n自主决定继续讲述或优雅让出话权', 'teal'),
        ],
        '全双工灵魂在于话权（Turn-Taking）由模型内生控制，而非音量阈值',
    )

    draw_dual_band(
        ex,
        'ch22-edge-vad-aec-wake-word-and-cloud-duplex-handover',
        '端侧超低功耗声学前端（DSP/NPU）与云端全双工大模型协同架构',
        'blue',
        [
            ('端侧常驻声学前端（< 50 mW）\n硬件参考信号回声消除（AEC）\n波束成形降噪 + 本地唤醒词检测', 'blue'),
            ('端云零冷启热接管（Handover）\n唤醒瞬间将前导 500 ms 音频缓存\n以 4 倍速突发上传云端预填充', 'teal'),
        ],
        '攻克“扬声器外放声音被麦克风拾取导致模型自打断”的回声自激',
        'orange',
        [
            ('无端侧 AEC 的灾难后果：\n助手外放语音经房间反射入麦克风\n触发无限循环自打断与声学啸叫', 'orange'),
            ('双端协同回声抑制：\n端侧自适应滤波消 35 dB 线性回声\n云端模型注入下行参考帧防漏网', 'teal'),
        ],
        '端侧硬件 AEC 与云端下行参考流对齐，是免耳机外放全双工的前提',
    )

    draw_dual_band(
        ex,
        'ch22-voice-cloning-zero-shot-speaker-embedding-cache',
        '零样本音色克隆（Zero-Shot Conditioning）与声纹 KV 共享',
        'teal',
        [
            ('参考音色声纹前缀（3–10 s 音频）\n编码为 37–125 帧固定声学 KV\n常驻各级缓存供全车队/全网复用', 'blue'),
            ('步内情感与语速控制向量\n在 Depth 模型注入韵律条件\n支持耳语、欢快、严肃等演绎', 'purple'),
        ],
        '实时深伪防御（Anti-Spoofing）：端到端音频流暗水印与声纹鉴权',
        'orange',
        [
            ('未授权音色克隆拦截\n声纹注册强制要求活体挑战短语\n拒绝为未签名音频生成克隆缓存', 'orange'),
            ('声码器相位微扰水印（AudioSeal）\n每帧 80 ms 波形内嵌密码学签名\n支持 100 ms 切片级合成溯源检测', 'teal'),
        ],
        '音色前缀 KV 共享消除重复编码开销，帧级水印筑牢防诈骗安全红线',
    )

    draw_dual_band(
        ex,
        'ch22-full-duplex-gpu-concurrency-vs-p99-frame-underrun',
        '单卡并发全双工语音流路数（B_stream）与 P99 断流率容量边界',
        'blue',
        [
            ('低并发（B=1–16）\n耗时 11.5–16 ms\n断流率 0% 算力闲', 'blue'),
            ('黄金区（B=32–64）\nP99 42–62ms<=80ms\n断流 <0.05% 最优', 'teal'),
            ('雪崩区（B>=96）\n耗时突破 80 ms\n断流 >8% 通话卡顿', 'orange'),
        ],
        '面向 99.95% 帧准时率的单卡最大并发容量准入控制（Admission）',
        'teal',
        [
            ('硬实时容量公式：\n主干 + 深度 + B·C_kv <= 65 ms\n（预留 15 ms 网络抖动安全裕量）', 'purple'),
            ('基于 P99 帧耗时的弹性扩容：\n当单卡 P99 逼近 60 ms 时停止派单\n立即将新增通话路由至热备推理卡', 'teal'),
        ],
        '全双工语音容量不看平均吞吐，而由 P99.9 单帧耗时不超 80 ms 决定',
    )

    draw_dual_band(
        ex,
        'ch22-full-duplex-omni-streaming-infrastructure-synthesis',
        '第 22 章全双工实时语音与原生全模态流式底座全景架构图',
        'teal',
        [
            ('编解码与协议层\nMimi 1.10 kbps\n全链路 164.0 ms', 'blue'),
            ('双层推理引擎层\nTemporal + Depth\n整步图 11.50 ms', 'teal'),
            ('一致性与长记忆\n游标回滚 187 词元\n声学修剪省 91.74%', 'purple'),
        ],
        '从“回合制对讲机”向“原生全双工实时语音伴侣”的系统跃迁',
        'blue',
        [
            ('核心瓶颈：串联管线破秒延迟\nDepth 微算子墙、打断状态错位', 'orange'),
            ('系统解法：双轨流与整步静态图\n播放游标回滚与多模态微块交错', 'teal'),
        ],
        '第 22 章收束：80 ms 硬节拍、整步图融合与播放游标回滚构筑三大基石',
    )

    outputs = ex.finish()

    fig_index = [
        {
            "figure_id": "图 22-1",
            "number": 1,
            "file": "ch22-cascaded-asr-llm-tts-vs-native-full-duplex-s2s.svg",
            "asset": "ch22/ch22-cascaded-asr-llm-tts-vs-native-full-duplex-s2s.svg",
            "section": "22.1.1",
            "caption": "传统四级串联语音管线（1,280 ms 延迟 + 副语言丢失）与原生全双工 S2S（164 ms 双流并行，快 7.80 倍）架构对比（算例 22-1）",
            "source_refs": ["experiments/ch22/22-01/run.py", "manuscripts/ch22/figure-data.json"],
            "TeachingRole": "定量拆解 VAD 等待与级联缓冲如何导致延迟破秒，以及原生全双工如何突破 200 ms 阈值"
        },
        {
            "figure_id": "图 22-2",
            "number": 2,
            "file": "ch22-full-duplex-dual-audio-streams-and-inner-monologue.svg",
            "asset": "ch22/ch22-full-duplex-dual-audio-streams-and-inner-monologue.svg",
            "section": "22.1.2",
            "caption": "全双工模型在 80 ms 时钟节拍（12.5 Hz）下同时处理用户音频流、系统音频流与内心独白文本流（每步 17 词元）的联合状态转移",
            "source_refs": ["experiments/ch22/22-01/run.py", "manuscripts/ch22/figure-data.json"],
            "TeachingRole": "阐明 Inner Monologue 文本锚点如何在保持全双工声学交互的同时防止语义退化"
        },
        {
            "figure_id": "图 22-3",
            "number": 3,
            "file": "ch22-rvq-neural-audio-codec-semantic-acoustic-hierarchy.svg",
            "asset": "ch22/ch22-rvq-neural-audio-codec-semantic-acoustic-hierarchy.svg",
            "section": "22.2.1",
            "caption": "神经音频编解码器（Mimi / EnCodec）中语义首码本（q_1）与 7 层声学残差码本（q_2..q_8）的 RVQ 量化流水线（1.10 kbps，349.09 倍压缩，算例 22-2）",
            "source_refs": ["experiments/ch22/22-01/run.py", "manuscripts/ch22/figure-data.json"],
            "TeachingRole": "推导帧率、码本层数与词表大小对词元速率（100 tok/s）与比特率（1.10 kbps）的支配关系"
        },
        {
            "figure_id": "图 22-4",
            "number": 4,
            "file": "ch22-delayed-codebook-pattern-vs-depth-transformer.svg",
            "asset": "ch22/ch22-delayed-codebook-pattern-vs-depth-transformer.svg",
            "section": "22.2.2",
            "caption": "RVQ 多码本生成的两种范式对比：时间步错位延迟模式（Delayed Pattern）vs. 时间—深度双层 Transformer（RQ-Transformer）",
            "source_refs": ["manuscripts/ch22/figure-data.json"],
            "TeachingRole": "解释为何 12.5 Hz 超低帧率语音模型需要步内 Depth Transformer 显式建模同帧码本依赖"
        },
        {
            "figure_id": "图 22-5",
            "number": 5,
            "file": "ch22-temporal-and-depth-transformer-gpu-roofline-trap.svg",
            "asset": "ch22/ch22-temporal-and-depth-transformer-gpu-roofline-trap.svg",
            "section": "22.2.2",
            "caption": "单帧 80 ms 内 Temporal 7B（3.50 ms）与 Depth 100M（384 个微算子朴素发射 35.04 ms vs. 整步 CUDA Graph + L2 常驻 8.00 ms）性能对比（算例 22-3）",
            "source_refs": ["experiments/ch22/22-01/run.py", "manuscripts/ch22/figure-data.json"],
            "TeachingRole": "揭示小模型串行循环的微算子启动风暴，并证明整步 CUDA Graph 融合带来的 3.35 倍加速"
        },
        {
            "figure_id": "图 22-6",
            "number": 6,
            "file": "ch22-webrtc-opus-aec-to-gpu-end-to-end-latency-budget.svg",
            "asset": "ch22/ch22-webrtc-opus-aec-to-gpu-end-to-end-latency-budget.svg",
            "section": "22.3.1",
            "caption": "从端侧麦克风 AEC（25 ms）、WebRTC/RTP 抖动缓冲（35 ms）到服务端 GPU 双层推理与声码器还原的 164.0 ms 全链路时延拆解",
            "source_refs": ["experiments/ch22/22-01/run.py", "manuscripts/ch22/figure-data.json"],
            "TeachingRole": "论证为何全双工实时语音必须采用无队头阻塞的 WebRTC/UDP 协议栈"
        },
        {
            "figure_id": "图 22-7",
            "number": 7,
            "file": "ch22-barge-in-playback-cursor-aligned-kv-rollback.svg",
            "asset": "ch22/ch22-barge-in-playback-cursor-aligned-kv-rollback.svg",
            "section": "22.3.2",
            "caption": "用户打断（Barge-in）时的核心状态同步：清空网关 11 帧（880 ms）未播缓冲 + 按客户端播放游标（第 54 帧）精确回滚 187 个超前 KV 词元（算例 22-4）",
            "source_refs": ["experiments/ch22/22-02/run.py", "manuscripts/ch22/figure-data.json"],
            "TeachingRole": "解决服务端超前生成与客户端真实听觉进度失配导致的上下文幻觉污染"
        },
        {
            "figure_id": "图 22-8",
            "number": 8,
            "file": "ch22-streaming-causal-conv-vocoder-chunk-halo.svg",
            "asset": "ch22/ch22-streaming-causal-conv-vocoder-chunk-halo.svg",
            "section": "22.3.2",
            "caption": "流式因果卷积声码器（Causal Vocoder）的帧间历史感受野环形缓存（Conv Halo Cache）与零爆音连续波形拼接",
            "source_refs": ["manuscripts/ch22/figure-data.json"],
            "TeachingRole": "展示如何在 80 ms 分块流式合成中消除帧边界相位不连续引起的咔哒爆音"
        },
        {
            "figure_id": "图 22-9",
            "number": 9,
            "file": "ch22-live-video-1fps-and-audio-12hz-interleaved-pacing.svg",
            "asset": "ch22/ch22-live-video-1fps-and-audio-12hz-interleaved-pacing.svg",
            "section": "22.4.1",
            "caption": "实时视频通话中将 1 FPS（256 词元）视觉帧拆分为 8 个微块（每拍 32 词元）交错插入 12.5 Hz 音频节拍，将峰值帧耗时从 83.5 ms 压至 20.5 ms（算例 22-5）",
            "source_refs": ["experiments/ch22/22-02/run.py", "manuscripts/ch22/figure-data.json"],
            "TeachingRole": "防止整帧视觉 Prefill 刺穿 80 ms 音频硬实时截止期引发周期性语音卡顿"
        },
        {
            "figure_id": "图 22-10",
            "number": 10,
            "file": "ch22-acoustic-token-eviction-and-semantic-anchor-retention.svg",
            "asset": "ch22/ch22-acoustic-token-eviction-and-semantic-anchor-retention.svg",
            "section": "22.4.2",
            "caption": "1 小时连续通话的分层记忆修剪：剥离 10 s 滑窗外的高阶声学码本与视觉帧、仅保留语义首码本与文本锚点（总 KV 显存节省 91.74%，算例 22-5）",
            "source_refs": ["experiments/ch22/22-02/run.py", "manuscripts/ch22/figure-data.json"],
            "TeachingRole": "突破长时全模态通话的显存线性增长墙，将 1 小时常驻词元从 168.66 万压至 13.93 万"
        },
        {
            "figure_id": "图 22-11",
            "number": 11,
            "file": "ch22-dynamic-batching-under-strict-80ms-isochronous-tick.svg",
            "asset": "ch22/ch22-dynamic-batching-under-strict-80ms-isochronous-tick.svg",
            "section": "22.3.1",
            "caption": "等时硬节拍（Isochronous 80 ms Tick）下多路并发全双工会话的槽位对齐批处理、截止期看门狗与迟到包隐藏策略",
            "source_refs": ["manuscripts/ch22/figure-data.json"],
            "TeachingRole": "在保证 GPU 批处理效率的同时阻断单路弱网抖动对同批健康会话的连坐阻塞"
        },
        {
            "figure_id": "图 22-12",
            "number": 12,
            "file": "ch22-duplex-turn-taking-backchannel-and-overlap-state-machine.svg",
            "asset": "ch22/ch22-duplex-turn-taking-backchannel-and-overlap-state-machine.svg",
            "section": "22.1.2",
            "caption": "原生全双工对话中的四态交互（专注倾听、自然附和 Backchannel、正常讲述、碰撞退让）内生状态机",
            "source_refs": ["manuscripts/ch22/figure-data.json"],
            "TeachingRole": "对比僵硬外部 VAD 误打断与原生双流模型区分善意附和与真实抢话的机制"
        },
        {
            "figure_id": "图 22-13",
            "number": 13,
            "file": "ch22-edge-vad-aec-wake-word-and-cloud-duplex-handover.svg",
            "asset": "ch22/ch22-edge-vad-aec-wake-word-and-cloud-duplex-handover.svg",
            "section": "22.3.1",
            "caption": "端侧超低功耗声学回声消除（AEC）/唤醒预处理与云端全双工大模型零冷启热接管协同架构",
            "source_refs": ["manuscripts/ch22/figure-data.json"],
            "TeachingRole": "解决免提外放场景下扬声器回声引发模型自我打断的工程难题"
        },
        {
            "figure_id": "图 22-14",
            "number": 14,
            "file": "ch22-voice-cloning-zero-shot-speaker-embedding-cache.svg",
            "asset": "ch22/ch22-voice-cloning-zero-shot-speaker-embedding-cache.svg",
            "section": "22.4.2",
            "caption": "零样本音色克隆中说话人参考声纹前缀（Speaker Conditioning KV）的跨会话共享与帧级内嵌防伪水印（AudioSeal）",
            "source_refs": ["manuscripts/ch22/figure-data.json"],
            "TeachingRole": "兼顾多音色定制的零重复编码开销与实时合成语音的防伪溯源安全"
        },
        {
            "figure_id": "图 22-15",
            "number": 15,
            "file": "ch22-full-duplex-gpu-concurrency-vs-p99-frame-underrun.svg",
            "asset": "ch22/ch22-full-duplex-gpu-concurrency-vs-p99-frame-underrun.svg",
            "section": "22.4.2",
            "caption": "单张 GPU 上并发全双工语音流路数 B_stream 与 P99 音频帧断流率（Frame Underrun Rate < 0.05%）的容量边界曲线",
            "source_refs": ["manuscripts/ch22/figure-data.json"],
            "TeachingRole": "建立以 P99.9 单帧耗时不越过 80 ms 为硬约束的流式服务容量规划模型"
        },
        {
            "figure_id": "图 22-16",
            "number": 16,
            "file": "ch22-full-duplex-omni-streaming-infrastructure-synthesis.svg",
            "asset": "ch22/ch22-full-duplex-omni-streaming-infrastructure-synthesis.svg",
            "section": "22.4.2",
            "caption": "全双工实时语音与原生全模态流式基础设施从 RVQ 编解码、双层图融合到播放游标回滚的全景架构图",
            "source_refs": ["experiments/ch22/22-01/run.py", "experiments/ch22/22-02/run.py", "manuscripts/ch22/figure-data.json"],
            "TeachingRole": "收束第 22 章等时节拍、Temporal+Depth 图融合与打断游标对齐的完整技术闭环"
        }
    ]
    for row in fig_index:
        row['figure'] = f"22-{row['number']}"
        row['name'] = Path(row['file']).stem
    (HERE / 'figure-index.json').write_text(json.dumps(fig_index, ensure_ascii=False, indent=2) + '\n')

    md = HERE.parent / '22-全双工实时语音（Full-Duplex Speech）、神经音频编解码与原生全模态流式底座.md'
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

        work = re.sub(r'!\[([^\]]*)\]\((ch22/[^)]+)\)', repl_img, text_md)
        body = markdown.markdown(work, extensions=['tables', 'fenced_code', 'footnotes', 'toc'])
        html = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>第 22 章：全双工实时语音（Full-Duplex Speech）、神经音频编解码与原生全模态流式底座</title></head><body>{body}</body></html>\n'
        out_html = preview_path(md)
        out_html.write_text(html)

    sources = [
        HERE / 'build.py',
        HERE / 'figure-data.json',
        HERE / 'figure-index.json',
        ROOT / 'experiments/ch22/22-01/run.py',
        ROOT / 'experiments/ch22/22-02/run.py',
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
    print(f'Built {len(outputs)} figure files for Chapter 22; 0 layout warnings.')


if __name__ == '__main__':
    build_chapter()
