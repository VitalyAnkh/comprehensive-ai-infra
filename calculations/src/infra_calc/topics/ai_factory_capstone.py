"""Ch23 & Ch24: SRAM vs HBM crossover, RLVR kernel synthesis, and 100 MW AI Factory full-stack capstone."""
from __future__ import annotations

from .. import hardware


def calculate(
    facility_grid_mw: float = 100.0,
    pue: float = 1.12,
    fabric_storage_cpu_mw: float = 7.45,
    rack_kw: float = 165.0,
    gpus_per_rack: int = 72,
    train_racks: int = 400,
    serve_racks: int = 82,
    spare_racks: int = 14,
    in_train_hot_spares: int = 128,
    realized_train_tflops_per_gpu: float = 970.0,
) -> dict:
    hw_map = {d["id"]: d for d in hardware.catalog()["devices"]}

    # 1. Facility power & rack allocation (100 MW grid -> PUE 1.12 -> 89.29 MW IT = 81.84 MW compute + 7.45 MW fabric/3FS)
    it_power_mw = round(facility_grid_mw / pue, 2)                         # 89.29 MW
    compute_power_mw = round((train_racks + serve_racks + spare_racks) * rack_kw / 1000.0, 2)  # 81.84 MW
    total_racks = train_racks + serve_racks + spare_racks                  # 496 racks
    total_gpus = total_racks * gpus_per_rack                               # 35,712 GPUs

    # Partition accounting
    train_partition_gpus = train_racks * gpus_per_rack                     # 28,800 GPUs
    active_train_gpus = train_partition_gpus - in_train_hot_spares         # 28,672 GPUs (= 448 * 64)
    serve_gpus = serve_racks * gpus_per_rack                               # 5,904 GPUs
    dedicated_spare_gpus = spare_racks * gpus_per_rack                     # 1,008 GPUs
    total_spare_gpus = dedicated_spare_gpus + in_train_hot_spares          # 1,136 GPUs (3.18%)
    spare_ratio_pct = round(total_spare_gpus / total_gpus * 100.0, 2)
    ettr_pct = 97.6

    # 2. Sustained training throughput & frontier model schedules
    sustained_train_eflops = round(active_train_gpus * realized_train_tflops_per_gpu / 1e6, 2)  # 27.81 EFLOPS
    sustained_train_flops = active_train_gpus * realized_train_tflops_per_gpu * 1e12            # 2.781184e19 FLOP/s

    # Flagship 10T total / 380B active MoE (+36.67B long-context attention equiv) on 60T tokens = 1.50e26 FLOPs
    flagship_10t_flops = 1.50e26
    flagship_pretrain_days = round((flagship_10t_flops / sustained_train_flops) / 86400.0, 1)   # 62.4 days
    flagship_rlvr_posttrain_days = 18.5
    flagship_total_days = round(flagship_pretrain_days + flagship_rlvr_posttrain_days, 1)       # 80.9 days

    # Agile 1T total / 44B active MoE on 25T tokens = 6 * 44e9 * 25e12 = 6.60e24 FLOPs
    agile_1t_flops = 6.0 * 44e9 * 25e12                                    # 6.60e24 FLOPs
    agile_pretrain_days = round((agile_1t_flops / sustained_train_flops) / 86400.0, 2)          # 2.75 days

    # 3. Serving partition capacity & 3-year FinOps ledger
    serving_tokens_per_day_billions = 216.5  # > 200B tokens/day under TTFT <= 350ms, TPOT <= 20ms
    three_year_capex_musd = 1860.0
    three_year_power_opex_musd = round(facility_grid_mw * 8760 * 3 * 65.0 / 1e6, 2)  # $170.82M @ $65/MWh
    three_year_total_tco_musd = round(three_year_capex_musd + three_year_power_opex_musd, 2)    # $2,030.82M
    amortized_usd_per_gpu_hour = round((three_year_total_tco_musd * 1e6) / (total_gpus * 8760 * 3), 2)  # $2.16/GPU-h
    serving_usd_per_1m_tokens = round(
        (serve_gpus * amortized_usd_per_gpu_hour * 24.0) / (serving_tokens_per_day_billions * 1000.0),
        2,
    )  # $1.41/1M raw 24h serving partition amort (or $0.61/1M when 57% off-peak capacity runs async RL rollouts)
    serving_net_after_rl_share_usd_per_1m = round(serving_usd_per_1m_tokens * 0.433, 2)        # $0.61/1M

    # 4. Ch23 算例 23-20: 70B FP8 on 4xH100 (164.7 tok/s @ B=1, $16.00/h) vs Wafer-Scale SRAM (1,850.0 tok/s @ B=1, $44.28/h)
    hbm_4xh100_b1_tok_per_s = round(1000.0 / 6.07, 1)                      # 164.7 tok/s
    sram_wafer_b1_tok_per_s = 1850.0                                       # 1,850.0 tok/s
    sram_b1_speedup_x = round(sram_wafer_b1_tok_per_s / hbm_4xh100_b1_tok_per_s, 2)  # 11.23x
    tps_target_equal_cost = 5148.0 * (16.00 / 44.28)                       # 1,860.16 tok/s
    tco_crossover_batch = round(
        (tps_target_equal_cost * (6.07 - 0.118)) / (1000.0 - 0.118 * tps_target_equal_cost), 1
    )  # 14.2
    hbm_b128_usd_per_mtok = 0.98
    sram_b1_usd_per_mtok = 6.65
    hbm_b128_cost_advantage_x = round(sram_b1_usd_per_mtok / hbm_b128_usd_per_mtok, 1)  # 6.8x

    # 5. Ch24 算例 24-7 & 24-8: RLVR automated kernel synthesis evolution ledger (KernelBench 250 tasks)
    kernelbench_turn1_correct_pct = round(85 / 250 * 100.0, 1)             # 34.0%
    kernelbench_turn8_correct_pct = round(237 / 250 * 100.0, 1)            # 94.8%
    rlvr_kernel_speedup_vs_eager_x = 3.42
    rlvr_kernel_speedup_vs_compile_x = round(3.42 / 2.1646, 2)             # 1.58x
    rlvr_fused_hbm_saved_per_layer_gbps = round(1.6e6 * 24_000 / 1e9, 1)   # 38.4 GB/s (2,304.0 GB/s across 60 layers)
    rlvr_asymmetric_tile_accum_regs = (128 * 192) // 256                   # 96 regs/thread (0 B spill)

    return {
        "schema": "h2-topic-v1",
        "calculation": "ai-factory-capstone",
        "chapter": 24,
        "title": "第 23–24 章：SRAM/HBM 并发交叉点、RLVR 内核进化与 100 MW 智算工厂端到端全栈收官账本",
        "scenario": {
            "facility_grid_mw": facility_grid_mw,
            "pue": pue,
            "rack_kw": rack_kw,
            "gpus_per_rack": gpus_per_rack,
            "train_racks": train_racks,
            "serve_racks": serve_racks,
            "spare_racks": spare_racks,
            "in_train_hot_spares": in_train_hot_spares,
            "realized_train_tflops_per_gpu": realized_train_tflops_per_gpu,
            "catalog_devices_referenced": sorted(k for k in ("gb200-nvl72", "b200-sxm", "h100-sxm") if k in hw_map),
        },
        "summary": {
            "facility_grid_mw": facility_grid_mw,
            "it_power_budget_mw": it_power_mw,
            "compute_racks_power_mw": compute_power_mw,
            "fabric_storage_cpu_power_mw": fabric_storage_cpu_mw,
            "total_72gpu_racks": total_racks,
            "total_gpus": total_gpus,
            "active_training_gpus": active_train_gpus,
            "serving_rollout_gpus": serve_gpus,
            "total_spare_gpus": total_spare_gpus,
            "spare_pool_ratio_pct": spare_ratio_pct,
            "sustained_ettr_pct": ettr_pct,
            "sustained_training_eflops": sustained_train_eflops,
            "flagship_10t_60t_flops": flagship_10t_flops,
            "flagship_10t_pretrain_days": flagship_pretrain_days,
            "flagship_10t_rlvr_days": flagship_rlvr_posttrain_days,
            "flagship_10t_total_days": flagship_total_days,
            "agile_1t_25t_flops": agile_1t_flops,
            "agile_1t_pretrain_days": agile_pretrain_days,
            "serving_daily_tokens_billions": serving_tokens_per_day_billions,
            "three_year_total_tco_musd": three_year_total_tco_musd,
            "amortized_usd_per_gpu_hour": amortized_usd_per_gpu_hour,
            "serving_raw_partition_usd_per_1m_tok": serving_usd_per_1m_tokens,
            "serving_net_after_rl_share_usd_per_1m_tok": serving_net_after_rl_share_usd_per_1m,
            "sram_wafer_b1_tok_per_s": sram_wafer_b1_tok_per_s,
            "hbm_4xh100_b1_tok_per_s": hbm_4xh100_b1_tok_per_s,
            "sram_vs_hbm_b1_speedup_x": sram_b1_speedup_x,
            "sram_vs_hbm_tco_crossover_batch": tco_crossover_batch,
            "hbm_b128_vs_sram_cost_advantage_x": hbm_b128_cost_advantage_x,
            "rlvr_kernelbench_turn1_correct_pct": kernelbench_turn1_correct_pct,
            "rlvr_kernelbench_turn8_correct_pct": kernelbench_turn8_correct_pct,
            "rlvr_kernel_speedup_vs_eager_x": rlvr_kernel_speedup_vs_eager_x,
            "rlvr_kernel_speedup_vs_compile_x": rlvr_kernel_speedup_vs_compile_x,
            "rlvr_fused_hbm_saved_per_layer_gbps": rlvr_fused_hbm_saved_per_layer_gbps,
            "rlvr_asymmetric_tile_accum_regs": rlvr_asymmetric_tile_accum_regs,
        },
        "assumptions": [
            "算例 23-20：70B FP8 在全片上 SRAM 阵列上 B=1 达 1,850.0 tok/s（较 4xH100 的 164.7 tok/s 快 11.23x），单位词元成本盈亏平衡并发度为 B_cross = 14.2（B=128 时 HBM 具备 6.8x 成本优势）。",
            "算例 24-7 与 24-8：8 轮硬件剖析反馈 RLVR 将 KernelBench 正确率从 34.0% 提至 94.8%，取得相对 Eager 3.42x / torch.compile 1.58x 加速，非对称 128x192 分块（96 累加寄存器）单层省去 38.4 GB/s HBM 落盘。",
            "算例 24-12（100 MW 智算工厂）：PUE=1.12 下 89.29 MW IT 支撑 496 台 72-GPU 机柜（400 训练 + 82 推理 + 14 备件，共 35,712 卡；总备件池 1,136 卡 = 3.18%，ETTR=97.6%），28,672 张活跃训练卡输出 27.81 EFLOPS，62.4 天完成 10T/380B 旗舰预训练（+18.5 天 RLVR = 80.9 天），2.75 天完成 1T/44B 敏捷预训练。",
        ],
    }
