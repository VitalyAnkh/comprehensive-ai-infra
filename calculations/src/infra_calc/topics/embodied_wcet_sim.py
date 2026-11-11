"""Ch20: System 1/2 WCET, 4,096-env GPU sim, NV12 ISP-DMA-BUF zero-copy, and 64-GPU World Model eval."""
from __future__ import annotations


def calculate(
    num_cameras: int = 6,
    width: int = 1920,
    height: int = 1200,
    fps: int = 60,
    nv12_bytes_per_pixel: float = 1.5,
    ros2_dram_traversals: int = 9,
    dmabuf_dram_traversals: int = 2,
    parallel_sim_envs: int = 4096,
    sim_hz: int = 60,
    world_model_gpus: int = 64,
    regression_scenarios: int = 1000,
) -> dict:
    # 1. 算例 20-3: System 1 (200 Hz) / System 2 (2 Hz) frequency ratio & MIG WCET isolation on Jetson Thor
    sys1_hz, sys2_hz = 200.0, 2.0
    freq_ratio = int(sys1_hz / sys2_hz)                                    # 100:1
    shared_stream_wcet_ms = 18.5
    shared_deadline_miss_pct = 28.4
    mig_static_graph_wcet_ms = 4.15                                        # 4.1-4.2 ms (< 5.0 ms deadline, 0.0% miss)

    # 2. 算例 20-4: 4,096 parallel environments PCIe bottleneck vs single-GPU native HBM allocation
    pcie_tier_a_gbps = round(parallel_sim_envs * 98_560 * sim_hz / 1e9, 2)   # 24.22 GB/s
    pcie_tier_b_compact_gbps = round(parallel_sim_envs * 176_544 * sim_hz / 1e9, 2)  # 43.39 GB/s (123.38 GB/s raw)
    pcie_tier_b_raw_gbps = round(parallel_sim_envs * 502_016 * sim_hz / 1e9, 2)      # 123.38 GB/s
    gpu_native_tier_a_gib = round((256.0 + 512.0 + 1536.0 + 6144.0) / 1024.0, 2)     # 8.25 GiB
    gpu_native_tier_b_gib = round((2048.0 + 4096.0 + 3600.0 + 16384.0) / 1024.0, 2)  # 25.52 GiB

    # 3. 算例 20-8: 6x 1920x1200@60fps NV12 ISP -> ViT zero-copy vs ROS 2 serialized copy
    bytes_per_frame = int(round(width * height * nv12_bytes_per_pixel))    # 3,456,000 B
    raw_ingress_gbps = round(num_cameras * fps * bytes_per_frame / 1e9, 5) # 1.24416 GB/s
    ros2_dram_gbps = round(raw_ingress_gbps * ros2_dram_traversals, 2)     # 11.20 GB/s
    dmabuf_dram_gbps = round(raw_ingress_gbps * dmabuf_dram_traversals, 2) # 2.49 GB/s
    dram_traffic_reduction_x = round(ros2_dram_traversals / dmabuf_dram_traversals, 2)  # 4.50x

    ros2_p99_sensor_to_vit_ms = 18.6
    ros2_p999_jitter_ms = 9.4
    dmabuf_p99_sensor_to_vit_ms = 0.85
    dmabuf_p999_jitter_ms = 0.14
    latency_reduction_x = round(ros2_p99_sensor_to_vit_ms / dmabuf_p99_sensor_to_vit_ms, 2)  # 21.88x

    # 4. 算例 20-9: Heterogeneous edge SoC (PVA stereo + GPU VLA + R52 1kHz MPC) SLC partitioning & memory QoS
    reflex_1khz_deadline_ms = 1.00
    unisolated_reflex_wcet_ms = 1.82
    slc_qos_isolated_reflex_wcet_ms = 0.41

    # 5. 算例 20-10 (PLAN.md §5.4): 64-GPU action-conditioned closed-loop World Model regression on 1,000 long-horizon scenarios
    rollouts_per_scenario = 100                                            # 100,000 closed-loop rollouts total
    total_rollouts = regression_scenarios * rollouts_per_scenario
    world_model_64gpu_hours = 0.40                                         # 24 minutes on 64 GPUs (25.6 GPU-h)
    world_model_eval_cost_usd = 768.0                                      # 64 GPUs * 0.40 h * $30.00/GPU-h (incl. storage/ops)
    physical_robot_eval_cost_usd = 111_360.0                               # 1,000 scenarios x 100 trials real-robot cell + operator reset
    eval_cost_compression_x = round(physical_robot_eval_cost_usd / world_model_eval_cost_usd, 1)  # 145.0x (145x)

    return {
        "schema": "h2-topic-v1",
        "calculation": "embodied-wcet-sim",
        "chapter": 20,
        "title": "第 20 章：具身双系统 WCET、万环境显存、NV12 零拷贝传感与 64 卡世界模型闭环评测账本",
        "scenario": {
            "num_cameras": num_cameras,
            "resolution": f"{width}x{height}",
            "fps": fps,
            "pixel_format": "NV12 (1.5 B/pixel)",
            "parallel_sim_envs": parallel_sim_envs,
            "world_model_gpus": world_model_gpus,
            "regression_scenarios": regression_scenarios,
            "total_closed_loop_rollouts": total_rollouts,
        },
        "summary": {
            "sys1_to_sys2_freq_ratio": freq_ratio,
            "jetson_shared_wcet_ms": shared_stream_wcet_ms,
            "jetson_shared_deadline_miss_pct": shared_deadline_miss_pct,
            "jetson_mig_isolated_wcet_ms": mig_static_graph_wcet_ms,
            "sim_4096env_pcie_tier_a_gbps": pcie_tier_a_gbps,
            "sim_4096env_pcie_tier_b_gbps": pcie_tier_b_compact_gbps,
            "sim_4096env_pcie_tier_b_raw_gbps": pcie_tier_b_raw_gbps,
            "sim_4096env_gpu_hbm_tier_a_gib": gpu_native_tier_a_gib,
            "sim_4096env_gpu_hbm_tier_b_gib": gpu_native_tier_b_gib,
            "nv12_bytes_per_frame": bytes_per_frame,
            "raw_camera_ingress_gbps": round(raw_ingress_gbps, 3),
            "ros2_serialized_dram_gbps": ros2_dram_gbps,
            "dmabuf_iceoryx_dram_gbps": dmabuf_dram_gbps,
            "dram_bandwidth_reduction_x": dram_traffic_reduction_x,
            "ros2_sensor_to_vit_p99_ms": ros2_p99_sensor_to_vit_ms,
            "ros2_sensor_p999_jitter_ms": ros2_p999_jitter_ms,
            "dmabuf_sensor_to_vit_p99_ms": dmabuf_p99_sensor_to_vit_ms,
            "dmabuf_sensor_p999_jitter_ms": dmabuf_p999_jitter_ms,
            "sensor_latency_speedup_x": latency_reduction_x,
            "reflex_1khz_deadline_ms": reflex_1khz_deadline_ms,
            "reflex_wcet_unisolated_ms": unisolated_reflex_wcet_ms,
            "reflex_wcet_slc_qos_isolated_ms": slc_qos_isolated_reflex_wcet_ms,
            "world_model_64gpu_eval_hours": world_model_64gpu_hours,
            "world_model_64gpu_cost_usd": world_model_eval_cost_usd,
            "physical_robot_eval_cost_usd": physical_robot_eval_cost_usd,
            "world_model_cost_compression_x": eval_cost_compression_x,
        },
        "assumptions": [
            "算例 20-3 与 20-4：System 1/2（200 Hz / 2 Hz = 100:1）经硬件 MIG + 静态图隔离将 WCET 从 18.5 ms（28.4% 违约）压至 4.15 ms（0% 违约）；4,096 环境全驻留 GPU HBM 仅占 8.25 GiB / 25.52 GiB，消除 24.22 / 43.39 GB/s 跨 PCIe 瓶颈。",
            "算例 20-8 至 20-10：6 路 1920x1200@60fps NV12（1.244 GB/s）经 ISP->DMA-BUF/NvSciBuf+Iceoryx 零拷贝将 DRAM 流量从 11.20 压至 2.49 GB/s、延迟从 18.6 ms 压至 0.85 ms；64 卡闭环世界模型对 1,000 个长程操作场景回归评测实现 145x 成本压缩。",
        ],
    }
