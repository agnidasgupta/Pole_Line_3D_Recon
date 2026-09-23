#!/usr/bin/env python3
"""Regression tests for strict/semantic Stage2 quality comparison."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from compare_v10_stage2_quality import compare_sessions


def write_fixture(root: Path, probability: float, line_span: float) -> None:
    obj = root / "stage2_objects" / "fixture_slice0"
    obj.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "file_id": "fixture", "component_id": "P00001", "slice_seq": 0,
        "refiner_probability": probability, "touches_xy_edge": False,
        "base_x": 10.0,
    }]).to_csv(obj / "slice_0_poles.csv", index=False)
    pd.DataFrame([
        {"class_name": "pole", "component_id": "P00001", "file_id": "fixture",
         "slice_seq": 0, "score_mean": probability, "component_accept": True},
        {"class_name": "line", "component_id": "S1E00001", "file_id": "fixture",
         "slice_seq": 0, "score_mean": 0.75, "component_accept": True},
    ]).to_csv(obj / "slice_0_components.csv", index=False)
    pd.DataFrame([{
        "file_id": "fixture", "component_id": "S1E00001", "slice_seq": 0,
        "horizontal_span_ft": line_span,
    }]).to_csv(obj / "slice_0_lines.csv", index=False)
    (obj / "slice_0_stage1_electrical_track_audit.json").write_text(
        json.dumps({"stage1_inferred_line_voxels": 1}) + "\n"
    )


def expect_failure(left: Path, right: Path, report: Path, expected: str) -> None:
    try:
        compare_sessions(left, right, report)
    except RuntimeError as error:
        assert expected in str(error), error
    else:
        raise AssertionError(f"comparison unexpectedly accepted: {expected}")


def main() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        baseline = root / "baseline"
        candidate = root / "candidate"
        report = root / "report.json"
        write_fixture(baseline, 0.8, 12.0)
        write_fixture(candidate, 0.8 + 1.0e-10, 12.0)
        result = compare_sessions(baseline, candidate, report)
        assert result["passed"] is True
        assert len(result["production_files_with_benign_serialization_differences"]) == 2

        write_fixture(candidate, 0.8 + 1.0e-10, 12.0 + 1.0e-10)
        expect_failure(baseline, candidate, report, "byte-exact electrical/line")

        write_fixture(candidate, 0.8 + 1.0e-3, 12.0)
        expect_failure(baseline, candidate, report, "beyond tolerance")

    print("V10_STAGE2_QUALITY_COMPARATOR_SELF_TEST_OK")


if __name__ == "__main__":
    main()
