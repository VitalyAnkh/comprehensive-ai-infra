import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from infra_calc import delivery, reproduce, sources


class PathH2TopicsTests(unittest.TestCase):
    def test_all_twelve_h2_commands_produce_valid_json_and_markdown(self):
        spec = json.loads((ROOT / "configs/h2-scenarios.json").read_text())
        self.assertEqual(len(spec["h2_topics"]), 12)
        payloads = {}
        for row in spec["h2_topics"]:
            cmd = row["command"]
            run_json = subprocess.run(
                [sys.executable, str(ROOT / "calc.py"), cmd],
                capture_output=True,
                text=True,
                check=True,
            )
            payload = json.loads(run_json.stdout)
            self.assertEqual(payload["schema"], "h2-topic-v1")
            self.assertEqual(payload["calculation"], cmd)
            self.assertTrue(payload["summary"])
            payloads[cmd] = payload["summary"]

            run_md = subprocess.run(
                [sys.executable, str(ROOT / "calc.py"), cmd, "--format", "md"],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("## 核心量化结果", run_md.stdout)

        # Spot-check locked manuscript invariants across Ch13-Ch24
        self.assertAlmostEqual(payloads["storage-kv-io"]["gds_140gb_load_seconds"], 1.40, places=2)
        self.assertAlmostEqual(payloads["storage-kv-io"]["kv_evict_unfiltered_dwpd"], 33.75, places=2)
        self.assertAlmostEqual(payloads["data-funnel-dedup"]["final_curated_tokens_trillion"], 15.0, places=2)
        self.assertAlmostEqual(payloads["data-funnel-dedup"]["gpu_vs_cpu_wall_speedup_x"], 80.0, places=1)
        self.assertAlmostEqual(payloads["datacenter-power-cooling"]["hvdc_i2r_loss_reduction_x"], 277.78, places=2)
        self.assertAlmostEqual(payloads["datacenter-power-cooling"]["w45_flow_l_per_min_132kw"], 159.5, places=1)
        self.assertAlmostEqual(payloads["fleet-reliability-profiler"]["fleet_100k_ettr_plan_b_pct"], 92.34, places=2)
        self.assertEqual(payloads["fleet-reliability-profiler"]["wait_chain_root_cause_rank"], 5)
        self.assertAlmostEqual(payloads["confidential-tee-wan"]["tee_io_speedup_pipelined_x"], 22.0, places=1)
        self.assertAlmostEqual(payloads["posttrain-rlvr-distill"]["top128_bitpacked_32bit_tb"], 102.4, places=1)
        self.assertAlmostEqual(payloads["agent-worm-grammar"]["worm_40turn_prefix_reuse_pct"], 95.12, places=2)
        self.assertAlmostEqual(payloads["multimodal-dit-duplex"]["usp_8x8_vs_ring_rdma_reduction_x"], 9.0, places=1)
        self.assertAlmostEqual(payloads["embodied-wcet-sim"]["world_model_cost_compression_x"], 145.0, places=1)
        self.assertAlmostEqual(payloads["recsys-vector-colpali"]["plaid_2bit_bytes_per_vec"], 34.0, places=1)
        self.assertAlmostEqual(payloads["recsys-vector-colpali"]["plaid_1bit_bytes_per_vec"], 17.5, places=1)
        self.assertAlmostEqual(payloads["recsys-vector-colpali"]["hot_update_epoch_lockfree_p99_ms"], 1.15, places=2)
        self.assertAlmostEqual(payloads["science-ozaki-coupling"]["ozaki_vs_native_fp64_speedup_x"], 3.68, places=2)
        self.assertEqual(payloads["ai-factory-capstone"]["active_training_gpus"], 28672)
        self.assertAlmostEqual(payloads["ai-factory-capstone"]["flagship_10t_total_days"], 80.9, places=1)

    def test_lfs_aware_verify_sources_and_verify_results_and_delivery(self):
        self.assertGreaterEqual(sources.verify_sources()["verified"], 105)
        verified = reproduce.verify_results(include_figures=True)
        self.assertGreaterEqual(verified["verified_artifacts"], 2588)
        self.assertEqual(verified["verified_figures"], 32)
        self.assertTrue(delivery.reader_entry()["passed"], delivery.reader_entry())
        self.assertTrue(delivery.scenario_artifacts()["passed"], delivery.scenario_artifacts())


if __name__ == "__main__":
    unittest.main()
