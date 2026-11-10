# 实验 28-03：SynthID-Text 锦标赛采样水印检出率（TPR/FPR）与 Safetensors 零拷贝页对齐校验器

本实验对标第 28 章 28.4 节与算例 28-6，核算深度 m=30 层的 SynthID-Text 锦标赛采样水印在 200 个生成词元、假阳性率 FPR = 1e-6 下的真阳性检出率（TPR = 99.2%，困惑度偏差 < 0.01），以及 Safetensors 4,096 B 页对齐 mmap 零拷贝加载与 Merkle 哈希树签名校验开销。
