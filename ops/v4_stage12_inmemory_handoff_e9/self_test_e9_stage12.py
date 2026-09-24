#!/usr/bin/env python3
"""Small synthetic test for E9's exact output and complete-run decision gates."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

import compare_e9_stage2_outputs as compare
import summarize_e9_stage12 as summary


def invoke(module, *args: str) -> None:
    previous = sys.argv
    try:
        sys.argv = [module.__file__, *args]
        module.main()
    finally:
        sys.argv = previous


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="e9-selftest-") as directory:
        root = Path(directory)
        control, candidate = root / "stage2_control", root / "stage2_candidate"
        for target in (control, candidate):
            target.mkdir()
            (target / "slice_poles.csv").write_text("x,y\n1,2\n")
            (target / "slice_audit.json").write_text(json.dumps({"accepted_stage1_line_voxels": 5, "stage1_npz": str(target / "source.npz")}))
            (target / "STAGE2_COMPLETED.json").write_text(json.dumps({"slices": 2, "totals": {"accepted_stage1_line_voxels": 5}, "manifest": str(target / "inference_manifest.csv")}))
            (target / "STAGE2_STAGE1_ELECTRICAL_TRACK_SUMMARY.json").write_text((target / "STAGE2_COMPLETED.json").read_text())
            pd.DataFrame([{"group_id": "a/b", "slice_seq": 1, "status": "completed", "stage1_npz": str(target / "source.npz")}]).to_csv(target / "inference_manifest.csv", index=False)
            pd.DataFrame([{"group_id": "a/b", "slice_seq": 1, "status": "completed", "stage1_npz": str(target / "source.npz")}]).to_csv(target / "stage2_manifest.csv", index=False)
        report = root / "comparison.json"
        invoke(compare, "--control", str(control), "--candidate", str(candidate), "--report", str(report))
        assert json.loads(report.read_text())["status"] == "PASS"
        (candidate / "slice_poles.csv").write_text("x,y\n1,3\n")
        try:
            invoke(compare, "--control", str(control), "--candidate", str(candidate), "--report", str(report))
        except RuntimeError:
            pass
        else:
            raise AssertionError("a changed Stage2 CSV must fail")
        assert json.loads(report.read_text())["status"] == "FAIL"

        stage1_dir, stage2_dir, candidate_dir, comparisons = [root / name for name in ("c1", "c2", "candidate", "comparisons")]
        for target in (stage1_dir, stage2_dir, candidate_dir, comparisons):
            target.mkdir()
        keys = [{"group_id": f"g{i}/s0", "slice_seq": seq} for i in range(30) for seq in (1, 2)]
        pd.DataFrame([{**key, "csv_read_ms": 10, "sparse_item_prep_ms": 5, "stage1_wall_ms": 100,
                       "stage1_artifact_write_ms": 10, "stage1_manifest_write_ms": 5} for key in keys]).to_csv(stage1_dir / "timing.csv", index=False)
        pd.DataFrame([{**key, "stage2_disk_slice_wall_ms": 100} for key in keys]).to_csv(stage2_dir / "timing.csv", index=False)
        pd.DataFrame([{**key, "stage1_exact": 1, "csv_read_ms": 10, "sparse_item_prep_ms": 5,
                       "stage1_inference_ms": 100, "stage2_reconstruction_ms": 80,
                       "stage2_write_and_orchestration_ms": 25, "stage2_callback_and_output_ms": 105,
                       "combined_stage1_stage2_ms": 220} for key in keys]).to_csv(candidate_dir / "timing.csv", index=False)
        for index in range(32):
            (comparisons / f"report_{index:02}.json").write_text('{"status":"PASS"}')
        args = ["--control-stage1-timings", str(stage1_dir), "--control-stage2-timings", str(stage2_dir),
                "--candidate-timings", str(candidate_dir), "--comparison-dir", str(comparisons), "--expected-slices", "60",
                "--json", str(root / "result.json"), "--markdown", str(root / "result.md")]
        invoke(summary, *args)
        accepted = json.loads((root / "result.json").read_text())
        assert accepted["decision"] == "ACCEPTED" and accepted["control_combined"]["count"] == 30
        (comparisons / "report_31.json").write_text('{"status":"FAIL"}')
        invoke(summary, *args)
        assert json.loads((root / "result.json").read_text())["decision"] == "NOT_ACCEPTED"
        (comparisons / "report_31.json").write_text('{"status":"PASS"}')
        incomplete = pd.read_csv(candidate_dir / "timing.csv").iloc[:-1]
        incomplete.to_csv(candidate_dir / "timing.csv", index=False)
        invoke(summary, *args)
        assert json.loads((root / "result.json").read_text())["decision"] == "NOT_ACCEPTED"
    print("E9_STAGE12_SELF_TEST_OK")


if __name__ == "__main__":
    main()
