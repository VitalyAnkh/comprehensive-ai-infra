# 第 23–24 章：SRAM/HBM 并发交叉点、RLVR 内核进化与 100 MW 智算工厂端到端全栈收官账本

- 计算标识：`ai-factory-capstone`（第 24 章）
- 场景输入：`{"catalog_devices_referenced": ["b200-sxm", "gb200-nvl72", "h100-sxm"], "facility_grid_mw": 100.0, "gpus_per_rack": 72, "in_train_hot_spares": 128, "pue": 1.12, "rack_kw": 165.0, "realized_train_tflops_per_gpu": 970.0, "serve_racks": 82, "spare_racks": 14, "train_racks": 400}`

## 核心量化结果

| 指标 | 数值 |
| --- | ---: |
| `facility_grid_mw` | 100.0 |
| `it_power_budget_mw` | 89.29 |
| `compute_racks_power_mw` | 81.84 |
| `fabric_storage_cpu_power_mw` | 7.45 |
| `total_72gpu_racks` | 496 |
| `total_gpus` | 35,712 |
| `active_training_gpus` | 28,672 |
| `serving_rollout_gpus` | 5,904 |
| `total_spare_gpus` | 1,136 |
| `spare_pool_ratio_pct` | 3.18 |
| `sustained_ettr_pct` | 97.6 |
| `sustained_training_eflops` | 27.81 |
| `flagship_10t_60t_flops` | 1.5e+26 |
| `flagship_10t_pretrain_days` | 62.4 |
| `flagship_10t_rlvr_days` | 18.5 |
| `flagship_10t_total_days` | 80.9 |
| `agile_1t_25t_flops` | 6.6e+24 |
| `agile_1t_pretrain_days` | 2.75 |
| `serving_daily_tokens_billions` | 216.5 |
| `three_year_total_tco_musd` | 2,030.82 |
| `amortized_usd_per_gpu_hour` | 2.16 |
| `serving_raw_partition_usd_per_1m_tok` | 1.41 |
| `serving_net_after_rl_share_usd_per_1m_tok` | 0.61 |
| `sram_wafer_b1_tok_per_s` | 1,850.0 |
| `hbm_4xh100_b1_tok_per_s` | 164.7 |
| `sram_vs_hbm_b1_speedup_x` | 11.23 |
| `sram_vs_hbm_tco_crossover_batch` | 14.2 |
| `hbm_b128_vs_sram_cost_advantage_x` | 6.8 |
| `rlvr_kernelbench_turn1_correct_pct` | 34.0 |
| `rlvr_kernelbench_turn8_correct_pct` | 94.8 |
| `rlvr_kernel_speedup_vs_eager_x` | 3.42 |
| `rlvr_kernel_speedup_vs_compile_x` | 1.58 |
| `rlvr_fused_hbm_saved_per_layer_gbps` | 38.4 |
| `rlvr_asymmetric_tile_accum_regs` | 96 |

## 建模口径与物理边界

- 算例 23-20：70B FP8 在全片上 SRAM 阵列上 B=1 达 1,850.0 tok/s（较 4xH100 的 164.7 tok/s 快 11.23x），单位词元成本盈亏平衡并发度为 B_cross = 14.2（B=128 时 HBM 具备 6.8x 成本优势）。
- 算例 24-7 与 24-8：8 轮硬件剖析反馈 RLVR 将 KernelBench 正确率从 34.0% 提至 94.8%，取得相对 Eager 3.42x / torch.compile 1.58x 加速，非对称 128x192 分块（96 累加寄存器）单层省去 38.4 GB/s HBM 落盘。
- 算例 24-12（100 MW 智算工厂）：PUE=1.12 下 89.29 MW IT 支撑 496 台 72-GPU 机柜（400 训练 + 82 推理 + 14 备件，共 35,712 卡；总备件池 1,136 卡 = 3.18%，ETTR=97.6%），28,672 张活跃训练卡输出 27.81 EFLOPS，62.4 天完成 10T/380B 旗舰预训练（+18.5 天 RLVR = 80.9 天），2.75 天完成 1T/44B 敏捷预训练。
