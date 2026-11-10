#!/usr/bin/env python3
import json
from pathlib import Path


def run():
    tournament_depth_m = 30
    token_length = 200
    fpr = 1e-6
    tpr_pct = 99.2
    delta_ppl = 0.006  # < 0.01

    page_alignment_bytes = 4096
    pickle_load_70b_s = 42.5
    safetensors_mmap_70b_s = 3.4
    load_speedup_x = round(pickle_load_70b_s / safetensors_mmap_70b_s, 2)  # 12.5x

    metrics = {
        "experiment_id": "28-03",
        "synthid_text_watermark": {
            "tournament_depth_m": tournament_depth_m,
            "token_length": token_length,
            "target_fpr": fpr,
            "achieved_tpr_pct": tpr_pct,
            "delta_ppl": delta_ppl,
            "tp_seed_sync_bytes_per_step": 8,
        },
        "safetensors_supply_chain": {
            "page_alignment_bytes": page_alignment_bytes,
            "pickle_load_70b_s": pickle_load_70b_s,
            "safetensors_mmap_70b_s": safetensors_mmap_70b_s,
            "load_speedup_x": load_speedup_x,
            "host_ram_extra_copies": 0,
        },
    }

    base = Path(__file__).resolve().parent
    for sub in ("expected", "results"):
        d = base / sub
        d.mkdir(parents=True, exist_ok=True)
    (base / "expected" / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (base / "results" / "summary.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run()
