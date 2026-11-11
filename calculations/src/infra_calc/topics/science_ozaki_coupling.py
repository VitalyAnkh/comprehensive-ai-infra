"""Ch22: AI4Science 5-domain ledger, Pairformer/SFNO, signed centered Ozaki INT8 FP64, and DLPack coupling."""
from __future__ import annotations

from .. import hardware


def calculate(
    ozaki_slices: int = 7,
    int8_tensor_tops: float = 4500.0,
    native_fp64_tflops: float = 40.0,
    gemm_dim_n: int = 8192,
    coupled_field_gb: float = 384.0,
    coupled_gpus: int = 1024,
    rollout_steps: int = 500,
) -> dict:
    hw_map = {d["id"]: d for d in hardware.catalog()["devices"]}

    # 1. 算例 22-2: Pairformer N_res = 4,096 cubic triangle attention memory & 4-GPU DAP
    pair_naive_cubic_gib = round((4 * (4096**3) * 2) / (1024**3), 2)       # 512.00 GiB
    pair_dap4_workspace_gib = 0.25                                         # 0.25 GiB/GPU
    pair_workspace_reduction_x = round(pair_naive_cubic_gib / pair_dap4_workspace_gib, 1)  # 2,048.0x
    pair_dap_comm_to_comp_pct = round((2.85 / 34.50) * 100.0, 2)           # 8.26%

    # 2. 算例 22-4: Global 0.05-deg weather model (3,600 x 7,200) on 8x8=64 GPUs halo exchange
    local_h, local_w = 3600 // 8, 7200 // 8                                # 450 x 900
    halo_cells = 2 * (local_h + local_w) * 4                               # 10,800 cells
    halo_payload_mb = round((halo_cells * 512) / 1e6, 2)                   # 5.53 MB
    halo_comm_to_comp_pct = round((0.18 / 14.20) * 100.0, 2)               # 1.27% (0.00 ms exposed)

    # 3. 算例 22-8 (PLAN.md §5.6): Signed centered Ozaki scheme on INT8 Tensor Cores (4,500 TOPS vs 40 TFLOPS FP64)
    int8_max_mag = 127
    int32_max = (1 << 31) - 1                                              # 2,147,483,647
    max_k_without_int32_overflow = int32_max // (int8_max_mag * int8_max_mag)  # 133,144 (>> K=8,192)
    ozaki_int8_gemm_count = ozaki_slices * (ozaki_slices + 1) // 2         # S=7 -> 28 INT8 GEMMs
    effective_mantissa_bits = round(ozaki_slices * 7.4, 1)                 # 51.8 bits
    peak_tops_to_fp64_ratio_x = round(int8_tensor_tops / native_fp64_tflops, 1)  # 112.5x
    pure_compute_ceiling_x = round(peak_tops_to_fp64_ratio_x / ozaki_int8_gemm_count, 2)  # 4.02x
    ozaki_speedup_x = 3.68                                                 # measured end-to-end w/ slicing & accum
    ozaki_s7_equiv_fp64_tflops = round(native_fp64_tflops * ozaki_speedup_x, 2)  # 147.20 TFLOPS equiv

    # 4. 算例 22-9: 1,024-GPU coupled atmosphere-ocean / CFD + Neural Operator step (384 GB 3D field)
    fs_checkpoint_coupling_s = 58.2
    dlpack_insitu_coupling_s = 0.41
    insitu_speedup_x = round(fs_checkpoint_coupling_s / dlpack_insitu_coupling_s, 1)  # 142.0x

    # 5. 算例 22-10 (PLAN.md §5.6): Conservation projection layer across 500 rollout steps
    unconstrained_mass_drift_pct = 11.4
    projected_mass_drift_rel = 4.2e-13
    projection_step_overhead_pct = 3.1

    # 6. 5-domain AI4Science anchor matrix (replacing misplaced whole-book recap in Ch22.5)
    domains = [
        {"domain": "biomolecular_af3", "bottleneck": "Pairformer Triangle Attention O(N^3) memory", "dominant_precision": "BF16 + FP32 accum"},
        {"domain": "genomics_evo2_hyena", "bottleneck": "Million-nt FFT / state-space conv bandwidth", "dominant_precision": "BF16 + FP32 FFT"},
        {"domain": "weather_sfno_graphcast", "bottleneck": "Global spherical FFT & 2D Halo exchange", "dominant_precision": "BF16 + FP32 spectral"},
        {"domain": "dft_quantum_chemistry", "bottleneck": "Dense Hamiltonian diagonalization & FP64 GEMM", "dominant_precision": "FP64 / Ozaki INT8"},
        {"domain": "fusion_tokamak_control", "bottleneck": "Sub-ms deterministic magnetic control WCET", "dominant_precision": "FP16 / INT8 TensorRT"},
    ]

    return {
        "schema": "h2-topic-v1",
        "calculation": "science-ozaki-coupling",
        "chapter": 22,
        "title": "第 22 章：科学计算五大领域跨尺度账本、Ozaki INT8 模拟 FP64 与 DLPack 原位耦合",
        "scenario": {
            "ozaki_slices": ozaki_slices,
            "int8_tensor_tops": int8_tensor_tops,
            "native_fp64_tflops": native_fp64_tflops,
            "gemm_dim_n": gemm_dim_n,
            "coupled_field_gb": coupled_field_gb,
            "coupled_gpus": coupled_gpus,
            "rollout_steps": rollout_steps,
            "reference_devices": sorted(k for k in ("h100-sxm", "b200-sxm") if k in hw_map),
        },
        "summary": {
            "science_domains_covered": len(domains),
            "pairformer_4096_naive_gib": pair_naive_cubic_gib,
            "pairformer_4096_dap4_workspace_gib": pair_dap4_workspace_gib,
            "pairformer_workspace_reduction_x": pair_workspace_reduction_x,
            "pairformer_dap_comm_to_comp_pct": pair_dap_comm_to_comp_pct,
            "weather_64gpu_halo_mb": halo_payload_mb,
            "weather_64gpu_halo_comm_to_comp_pct": halo_comm_to_comp_pct,
            "ozaki_signed_int8_max_k_no_overflow": max_k_without_int32_overflow,
            "ozaki_s7_int8_gemms": ozaki_int8_gemm_count,
            "ozaki_s7_effective_mantissa_bits": effective_mantissa_bits,
            "ozaki_int8_to_fp64_peak_ratio_x": peak_tops_to_fp64_ratio_x,
            "ozaki_s7_compute_ceiling_x": pure_compute_ceiling_x,
            "ozaki_vs_native_fp64_speedup_x": ozaki_speedup_x,
            "ozaki_equiv_fp64_tflops": ozaki_s7_equiv_fp64_tflops,
            "coupling_384gb_filesystem_s": fs_checkpoint_coupling_s,
            "coupling_384gb_dlpack_insitu_s": dlpack_insitu_coupling_s,
            "coupling_insitu_speedup_x": insitu_speedup_x,
            "unconstrained_500step_mass_drift_pct": unconstrained_mass_drift_pct,
            "projected_500step_mass_drift_rel": projected_mass_drift_rel,
            "conservation_projection_overhead_pct": projection_step_overhead_pct,
        },
        "domains": domains,
        "assumptions": [
            "算例 22-2 与 22-4：Pairformer N=4,096 分块融合 + 4 卡 DAP 将工作区从 512.0 GiB 压至 0.25 GiB/GPU（2,048x，通信占比 8.26%）；全球 0.05° 气象 8x8 域分解单步 Halo 仅 5.53 MB（通信比 1.27%，双流重叠零暴露）。",
            "算例 22-8：有符号中心化 Ozaki 切片 [-127, 127] 在 K <= 133,144 时 INT32 严格零溢出；在 INT8=4,500 TOPS vs 原生 FP64=40 TFLOPS（112.5x）架构上，S=7（28 次 GEMM，理论上限 4.02x）实测加速 3.68x（等效 147.2 TFLOPS，51.8 bits 有效尾数）。",
            "算例 22-9 与 22-10：384 GB 耦合场经 DLPack 显存零拷贝将步间耦合从 58.2 s 压至 0.41 s（142x）；融合拉格朗日守恒投影以 +3.1% 开销消除 500 步自回归 11.4% 质量漂移（相对误差 < 1e-12）。",
        ],
    }
