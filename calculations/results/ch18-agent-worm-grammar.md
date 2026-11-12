# 第 18 章：智能体 WORM 状态复用、CacheGen 跨网回本、XGrammar 位掩码与多智能体 DAG 尾延迟账本

- 计算标识：`agent-worm-grammar`（第 18 章）
- 场景输入：`{"batch_size": 64, "deepseek_v3_vocab_size": 129280, "fanout_k": 12, "gpu_decode_step_ms": 8.5, "naive_fsm_cpu_ms": 38.4, "normal_z99_ref": 2.3263, "qwen3_vocab_size": 151936, "single_branch_p99_s": 6.2, "worm_turns": 40}`

## 核心量化结果

| 指标 | 数值 |
| --- | ---: |
| `worm_40turn_unique_tokens` | 80,000 |
| `worm_40turn_cum_prefill_tokens` | 1,640,000 |
| `worm_40turn_prefix_reuse_pct` | 95.12 |
| `worm_prefill_to_decode_ratio` | 205.0 |
| `kv_64k_pcie_swap_in_ms` | 187.5 |
| `kv_64k_swap_vs_recompute_speedup_x` | 2.67 |
| `worm_15s_wait_hbm_rent_saving_pct` | 85.6 |
| `dynamo_kv_router_speedup_x` | 3.0 |
| `cachegen_compression_x` | 4.21 |
| `cachegen_50g_restore_ms` | 385.9 |
| `cachegen_vs_raw_transfer_speedup_x` | 4.01 |
| `cachegen_breakeven_bw_gibps` | 4.43 |
| `firecracker_fork_ms` | 12.0 |
| `firecracker_16branch_cow_saving_pct` | 89.06 |
| `firecracker_sync_rollback_ms` | 8.5 |
| `qwen3_bitmask_uint32_words` | 4,748 |
| `qwen3_bitmask_bytes_per_req` | 18,992 |
| `qwen3_batch64_bitmask_mib` | 1.16 |
| `deepseek_bitmask_uint32_words` | 4,040 |
| `deepseek_bitmask_bytes_per_req` | 16,160 |
| `naive_fsm_throughput_drop_pct` | 81.9 |
| `xgrammar_batch64_cpu_ms` | 2.1 |
| `xgrammar_effective_step_ms` | 8.5 |
| `xgrammar_throughput_loss_pct` | 0.000e+00 |
| `k16_required_branch_percentile` | 99.937 |
| `k12_barrier_join_p99_seconds` | 19.8 |
| `k12_quorum_slack_p99_seconds` | 7.4 |
| `speculative_tool_saved_latency_ms` | 210.7 |
| `speculative_tool_wasted_query_pct` | 16.3 |

## 建模口径与物理边界

- 算例 18-1 至 18-3：40 轮编程智能体（L0=2,000, dL=2,000）前缀复用率达 95.12%（P/D=205:1）；CacheGen 4.21x 压缩将 50G 跨网拉取从 1,549.1 ms 压至 385.9 ms（< 500 ms 重算线，回本带宽仅需 4.43 GiB/s）；Firecracker CoW 16 分支省内存 89.06%。
- 算例 18-8 至 18-10：Qwen3 (V=151,936) 词表位掩码占 18,992 B/请求（B=64 共 1.16 MiB），CPU 异步生成 2.10 ms 完全隐藏于 8.50 ms GPU 解码步之下；K=12 扇出尾延迟从 6.2 s 放大至 19.8 s，经关键路径调度与 10/12 定额收敛至 7.4 s。
