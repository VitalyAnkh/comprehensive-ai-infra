#!/usr/bin/env python3
"""Verify Experiment 30-03."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    exp = json.loads((HERE / "expected" / "metrics.json").read_text())
    res = json.loads((HERE / "results" / "summary.json").read_text())
    assert exp == res, "Expected and results mismatch"

    g = res["grand_30chapter_encyclopedia_ledger"]
    assert g["total_chapters"] == 30
    assert g["architectural_parts"] == 5
    assert g["numbered_chapter_figures"] == 979
    assert g["total_figures"] == 980
    assert g["total_figure_asset_files"] == 2940
    assert g["python_code_blocks"] == 201
    assert g["total_typed_code_blocks"] == 202
    assert g["standardized_worked_examples"] == 228
    assert g["standardized_tables"] == 341
    assert g["exercise_blocks"] == 492
    assert g["unique_exercise_ids"] == 491
    assert g["academic_footnotes"] == 812
    assert g["runnable_experiment_suites"] == 165
    print("Experiment 30-03 verified successfully.")


if __name__ == "__main__":
    main()
