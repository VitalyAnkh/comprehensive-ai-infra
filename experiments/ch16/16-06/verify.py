#!/usr/bin/env python3
import json
from pathlib import Path


def verify():
    base = Path(__file__).resolve().parent
    exp = json.loads((base / "expected" / "metrics.json").read_text(encoding="utf-8"))
    res = json.loads((base / "results" / "summary.json").read_text(encoding="utf-8"))
    assert exp == res, "Mismatch between expected and results"
    assert res["synthid_text_watermark"]["achieved_tpr_pct"] == 99.2
    assert res["synthid_text_watermark"]["delta_ppl"] < 0.01
    assert res["safetensors_supply_chain"]["page_alignment_bytes"] == 4096
    print("Experiment 16-06 verified successfully.")


if __name__ == "__main__":
    verify()
