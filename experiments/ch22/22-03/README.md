# Experiment 22-03: Real-Time 3D Avatar Lip-Sync (< 40 ms Jitter Alignment) & Global Edge WebRTC SFU Mesh

Reproduces:
- **Worked Example 22-6**: Real-time 3D Gaussian Avatar lip-sync rendering (`28.50 ms` rendering + `4.00 ms` NVENC = `32.50 ms` vs. `6.50 ms` causal vocoder + `18.00 ms` steady video RTP jitter skew), where dual-stream RTP `abs-capture-time` alignment buffer (`26.00 ms` audio hold delay) reduces steady audio-lead skew from `44.00 ms` to `18.00 ms` (`59.09%` reduction) and P99 skew from `59.00 ms` to `33.00 ms < 40.0 ms`, keeping end-to-end synchronized avatar TTFA at `190.00 ms < 200.0 ms`.
- **Worked Example 22-7**: Global Anycast edge WebRTC SFU termination (`12.0 ms` metro RTT + `10.0 ms` JB + `4.5 ms` spatial separation & Mimi RVQ encode) + transoceanic private fiber backbone (`68.0 ms` RTT + `4.0 ms` JB), reducing network & jitter path latency from `175.00 ms` to `98.50 ms` (`43.71%` reduction), reducing end-to-end cross-continental TTFA from `304.00 ms` to `181.50 ms` (`1.67x` speedup), and compressing bidirectional backbone bandwidth from `96.00 kbps` Opus to `2.20 kbps` RVQ tokens (`43.64x` compression, `97.71%` saving).
- **Exercises 22-11 & 22-12**: Holographic avatar lip-sync (`38.0 -> 16.0 ms` steady skew, `50.0 -> 28.0 ms` P99 skew, `182.00 ms` TTFA) and Tokyo-to-Oregon transpacific edge SFU mesh (`319.00 -> 185.50 ms` TTFA, `1.72x` speedup).

## Run & Verify

```bash
python3 experiments/ch22/22-03/run.py
python3 experiments/ch22/22-03/verify.py
```
