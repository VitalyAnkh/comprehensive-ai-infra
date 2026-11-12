#!/usr/bin/env python3
"""Experiment 20-04: Edge Multi-Camera ISP -> DMA-BUF Zero-Copy vs ROS 2 Serialized Copy WCET & Jitter."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_PATH = HERE / "expected" / "metrics.json"
SUMMARY_PATH = HERE / "results" / "summary.json"


def compute_metrics() -> dict:
    # 1. Worked Example 20-8: 6x 1920x1200@60fps NV12 sensor pipeline comparison
    num_cams = 6
    width, height = 1920, 1200
    fps = 60
    nv12_bpp = 1.5
    bytes_per_frame = int(round(width * height * nv12_bpp))  # 3,456,000 B
    raw_ingress_bps = num_cams * fps * bytes_per_frame  # 1,244,160,000 B/s
    raw_ingress_gbps = round(raw_ingress_bps / 1e9, 3)  # 1.244 GB/s
    raw_ingress_exact_gbps = round(raw_ingress_bps / 1e9, 5)  # 1.24416 GB/s

    ros2_traversals = 9  # 1 ISP write + 4 full-frame copies * (1 read + 1 write)
    dmabuf_traversals = 2  # 1 ISP write + 1 GPU ViT zero-copy read
    ros2_dram_gbps = round(raw_ingress_exact_gbps * ros2_traversals, 2)  # 11.20 GB/s
    dmabuf_dram_gbps = round(raw_ingress_exact_gbps * dmabuf_traversals, 2)  # 2.49 GB/s
    dram_bandwidth_reduction_x = round(ros2_traversals / dmabuf_traversals, 2)  # 4.50x

    ros2_stages_ms = {
        "isp_write_ms": 0.45,
        "v4l2_mmap_to_userspace_copy_ms": 2.85,
        "ros2_dds_serialize_loopback_ms": 8.40,
        "cpu_nv12_to_rgb_convert_ms": 4.30,
        "cuda_memcpy_htod_sync_ms": 2.60,
    }
    dmabuf_stages_ms = {
        "isp_unified_page_write_ms": 0.45,
        "nvscisync_hw_fence_signal_ms": 0.05,
        "iceoryx_64b_descriptor_pub_ms": 0.03,
        "gpu_vit_fused_nv12_gather_ms": 0.32,
    }
    ros2_p99_ms = round(sum(ros2_stages_ms.values()), 2)  # 18.60 ms
    ros2_p999_jitter_ms = 9.40
    dmabuf_p99_ms = round(sum(dmabuf_stages_ms.values()), 2)  # 0.85 ms
    dmabuf_p999_jitter_ms = 0.14
    sensor_latency_speedup_x = round(ros2_p99_ms / dmabuf_p99_ms, 2)  # 21.88x

    # 2. Exercise 20-15: 8x 4K (3840x2160@30fps) NV12 surround cameras at v = 1.20 m/s
    cams_4k, w_4k, h_4k, fps_4k = 8, 3840, 2160, 30
    frame_4k_bytes = int(round(w_4k * h_4k * nv12_bpp))  # 12,441,600 B
    raw_4k_gbps = round(cams_4k * fps_4k * frame_4k_bytes / 1e9, 6)  # 2.985984 GB/s
    ros2_4k_dram_gbps = round(raw_4k_gbps * ros2_traversals, 2)  # 26.87 GB/s
    dmabuf_4k_dram_gbps = round(raw_4k_gbps * dmabuf_traversals, 2)  # 5.97 GB/s
    saved_4k_dram_gbps = round(ros2_4k_dram_gbps - dmabuf_4k_dram_gbps, 2)  # 20.90 GB/s
    ros2_4k_p99_ms = 44.64
    dmabuf_4k_p99_ms = 1.85
    velocity_m_s = 1.20
    saved_blind_drift_mm = round(velocity_m_s * (ros2_4k_p99_ms - dmabuf_4k_p99_ms), 2)  # 51.35 mm

    # 3. Exercise 20-16: Iceoryx lock-free MemPool non-starvation slot sizing
    in_flight_isp_write_slots = 1
    latest_queue_slots = 1
    subscriber_hold_quotas = {"sys2_vlm": 1, "sys1_policy": 1, "shadow_nvenc_recorder": 2}
    min_slots_per_camera = in_flight_isp_write_slots + latest_queue_slots + sum(subscriber_hold_quotas.values())  # 6
    total_pool_slots = num_cams * min_slots_per_camera  # 36
    total_pool_bytes = total_pool_slots * bytes_per_frame  # 124,416,000 B
    total_pool_mib = round(total_pool_bytes / (1024 ** 2), 2)  # 118.65 MiB

    return {
        "experiment_id": "20-04",
        "sensor_pipeline_ex20_8": {
            "num_cameras": num_cams,
            "resolution": "1920x1200",
            "fps": fps,
            "nv12_bytes_per_frame": bytes_per_frame,
            "raw_camera_ingress_gbps": raw_ingress_gbps,
            "ros2_dram_traversals": ros2_traversals,
            "dmabuf_dram_traversals": dmabuf_traversals,
            "ros2_serialized_dram_gbps": ros2_dram_gbps,
            "dmabuf_iceoryx_dram_gbps": dmabuf_dram_gbps,
            "dram_bandwidth_reduction_x": dram_bandwidth_reduction_x,
            "ros2_stages_ms": ros2_stages_ms,
            "dmabuf_stages_ms": dmabuf_stages_ms,
            "ros2_sensor_to_vit_p99_ms": ros2_p99_ms,
            "ros2_sensor_p999_jitter_ms": ros2_p999_jitter_ms,
            "dmabuf_sensor_to_vit_p99_ms": dmabuf_p99_ms,
            "dmabuf_sensor_p999_jitter_ms": dmabuf_p999_jitter_ms,
            "sensor_latency_speedup_x": sensor_latency_speedup_x,
            "meets_sub_150us_jitter_sla": dmabuf_p999_jitter_ms < 0.15,
        },
        "exercise_20_15_4k_8cam": {
            "frame_4k_bytes": frame_4k_bytes,
            "raw_4k_ingress_gbps": round(raw_4k_gbps, 3),
            "ros2_4k_dram_gbps": ros2_4k_dram_gbps,
            "dmabuf_4k_dram_gbps": dmabuf_4k_dram_gbps,
            "saved_4k_dram_gbps": saved_4k_dram_gbps,
            "saved_blind_drift_mm": saved_blind_drift_mm,
        },
        "exercise_20_16_iceoryx_pool": {
            "min_slots_per_camera": min_slots_per_camera,
            "total_pool_slots": total_pool_slots,
            "total_pool_bytes": total_pool_bytes,
            "total_pool_mib": total_pool_mib,
        },
    }


def main() -> None:
    metrics = compute_metrics()
    payload = json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    for p in (EXPECTED_PATH, SUMMARY_PATH):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(payload, encoding="utf-8")
    print(payload.strip())


if __name__ == "__main__":
    main()
