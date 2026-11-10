# Experiment 18-03: GUI Incremental ROI Vision Token Cache & Multi-Agent DAG Speculative Tool Execution

Reproduces:
- **Worked Example 21-6**: 30-step GUI agent (`OSWorld` / `WebArena`) incremental dirty-tile ROI vision token cache (`12.5%` dirty tiles) + interactive Set-of-Marks DOM pruning (`3,600 -> 450` tokens), compressing cumulative multimodal tokens from `180,000` to `24,600` (`7.32x` compression, `86.33%` saving) and accelerating ViT visual encoding from `1,350.00 ms` to `251.62 ms` (`5.37x` speedup).
- **Worked Example 21-7**: 4-stage multi-agent workflow DAG critical-path speculative tool execution (`600 ms` LLM + `1,400 ms` tool, `80%` hit probability, `1,050 ms` hit stage latency, `12.0 ms` Firecracker CoW rollback on miss), reducing expected critical-path latency from `8,000.00 ms` to `4,969.60 ms` (`1.61x` speedup, `37.88%` reduction) with a breakeven hit probability of `1.25%`.
- **Exercises 21-11 & 21-12**: 40-step 4K GUI incremental cache (`320,000 -> 35,680` tokens, `8.97x` compression; ViT `2,400.00 -> 372.00 ms`, `6.45x` speedup) and 5-stage DAG speculative execution (`10,000.00 -> 6,643.75 ms`, `1.51x` speedup).

## Run & Verify

```bash
python3 experiments/ch18/18-03/run.py
python3 experiments/ch18/18-03/verify.py
```
