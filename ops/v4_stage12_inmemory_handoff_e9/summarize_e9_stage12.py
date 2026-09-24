#!/usr/bin/env python3
"""Render E9 raw timing data as numeric summary plus an accept/reject decision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from v4_stage_contracts import atomic_json


def read_csvs(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(root.rglob("*.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if {"group_id", "slice_seq"} <= set(frame.columns):
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def describe(frame: pd.DataFrame, column: str) -> dict[str, float | int | None]:
    if column not in frame.columns:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None}
    values = pd.to_numeric(frame.get(column), errors="coerce").dropna().to_numpy(dtype=float)
    if not len(values):
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None}
    return {"count": int(len(values)), "mean_ms": float(np.mean(values)), "p50_ms": float(np.quantile(values, 0.50)), "p95_ms": float(np.quantile(values, 0.95))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-stage1-timings", required=True)
    parser.add_argument("--control-stage2-timings", required=True)
    parser.add_argument("--candidate-timings", required=True)
    parser.add_argument("--comparison-dir", required=True)
    parser.add_argument("--minimum-save-percent", type=float, default=1.0)
    parser.add_argument("--expected-sessions", type=int, default=30)
    parser.add_argument("--expected-slices", type=int, default=3738)
    parser.add_argument("--expected-comparisons", type=int, default=32)
    parser.add_argument("--json", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()
    c1 = read_csvs(Path(args.control_stage1_timings))
    c2 = read_csvs(Path(args.control_stage2_timings))
    cand = read_csvs(Path(args.candidate_timings))
    for frame in (c1, c2, cand):
        if frame.empty:
            raise RuntimeError("E9 timing input is empty")
        frame["slice_seq"] = pd.to_numeric(frame["slice_seq"], errors="raise").astype(int)
    control = c1.merge(c2, on=["group_id", "slice_seq"], suffixes=("_stage1", "_stage2"), how="inner")
    keys = ["group_id", "slice_seq"]
    if c1.duplicated(keys).any() or c2.duplicated(keys).any() or cand.duplicated(keys).any():
        raise RuntimeError("E9 duplicate timing slice keys; timing result is NOT_ACCEPTED")
    expected_keys = set(map(tuple, c1[keys].itertuples(index=False, name=None)))
    control_keys = set(map(tuple, control[keys].itertuples(index=False, name=None)))
    candidate_keys = set(map(tuple, cand[keys].itertuples(index=False, name=None)))
    complete_coverage = expected_keys == control_keys == candidate_keys
    session_count = int(c1["group_id"].nunique())
    candidate_all_count = len(cand)
    exact_count = int(pd.to_numeric(cand.get("stage1_exact"), errors="coerce").fillna(0).sum())
    control = control.sort_values(["group_id", "slice_seq"], kind="stable")
    cand = cand.sort_values(["group_id", "slice_seq"], kind="stable")
    # groupby.apply drops grouping columns in newer pandas; cumcount preserves
    # the original schema on both the Mac and the production Docker image.
    control = control.loc[control.groupby("group_id").cumcount().gt(0)].copy()
    cand = cand.loc[cand.groupby("group_id").cumcount().gt(0)].copy()
    control["combined_control_ms"] = (
        pd.to_numeric(control.get("csv_read_ms"), errors="coerce").fillna(0)
        + pd.to_numeric(control.get("sparse_item_prep_ms"), errors="coerce").fillna(0)
        + pd.to_numeric(control.get("stage1_wall_ms"), errors="coerce").fillna(0)
        + pd.to_numeric(control.get("stage1_artifact_write_ms"), errors="coerce").fillna(0)
        + pd.to_numeric(control.get("stage1_manifest_write_ms"), errors="coerce").fillna(0)
        + pd.to_numeric(control.get("stage2_disk_slice_wall_ms"), errors="coerce").fillna(0)
    )
    reports = list(Path(args.comparison_dir).rglob("*.json"))
    report_states = [json.loads(path.read_text()).get("status") for path in reports]
    comparison_pass = len(reports) == args.expected_comparisons and all(state == "PASS" for state in report_states)
    exact_required = int(candidate_all_count)
    control_summary = describe(control, "combined_control_ms")
    candidate_summary = describe(cand, "combined_stage1_stage2_ms")
    saved_percent = 100.0 * (float(control_summary["mean_ms"]) - float(candidate_summary["mean_ms"])) / float(control_summary["mean_ms"])
    accepted = (
        session_count == args.expected_sessions and len(c1) == args.expected_slices
        and len(c2) == args.expected_slices and candidate_all_count == args.expected_slices
        and complete_coverage and comparison_pass
        and exact_count == exact_required and saved_percent >= args.minimum_save_percent
    )
    raw = {
        "update_date_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
        "decision": "ACCEPTED" if accepted else "NOT_ACCEPTED",
        "acceptance_rules": {"expected_sessions": args.expected_sessions, "expected_slices": args.expected_slices, "expected_comparisons": args.expected_comparisons, "stage1_exact_rows": exact_required, "complete_slice_coverage": True, "stage2_output_comparisons_pass": True, "minimum_saved_percent": args.minimum_save_percent},
        "observed": {"sessions": session_count, "control_slices": len(c1), "candidate_slices": candidate_all_count, "complete_slice_coverage": complete_coverage, "stage2_comparisons": len(reports), "stage1_exact_rows": exact_count, "stage2_output_comparisons_pass": comparison_pass, "saved_percent": saved_percent},
        "control_combined": control_summary, "candidate_combined": candidate_summary,
        "control_breakdown": {name: describe(control, name) for name in ["csv_read_ms", "sparse_item_prep_ms", "stage1_wall_ms", "stage1_artifact_write_ms", "stage1_manifest_write_ms", "stage1_load_ms", "production_stage2_ms", "stage1_electrical_track_ms", "stage2_total_ms", "stage2_disk_slice_wall_ms", "combined_control_ms"]},
        "candidate_breakdown": {name: describe(cand, name) for name in ["csv_read_ms", "sparse_item_prep_ms", "stage1_inference_ms", "stage2_reconstruction_ms", "stage2_write_and_orchestration_ms", "stage2_callback_and_output_ms", "combined_stage1_stage2_ms"]},
        "raw_sources": {"control_stage1_timings": args.control_stage1_timings, "control_stage2_timings": args.control_stage2_timings, "candidate_timings": args.candidate_timings, "comparison_dir": args.comparison_dir},
    }
    atomic_json(raw, args.json)
    lines = ["# E9 Stage1-to-Stage2 in-memory handoff result", "", f"**Update date: {raw['update_date_utc']} UTC**", f"**Decision: {raw['decision']}**", "", "## Numeric summary", "", "| Metric | Disk control | In-memory candidate |", "| --- | ---: | ---: |", f"| Sessions | {session_count} | {cand['group_id'].nunique()} |", f"| Timed slices (first slice/session excluded) | {control_summary['count']} | {candidate_summary['count']} |", f"| Combined mean (ms) | {control_summary['mean_ms']:.3f} | {candidate_summary['mean_ms']:.3f} |", f"| Combined P50 (ms) | {control_summary['p50_ms']:.3f} | {candidate_summary['p50_ms']:.3f} |", f"| Combined P95 (ms) | {control_summary['p95_ms']:.3f} | {candidate_summary['p95_ms']:.3f} |", f"| Mean time saved | — | {saved_percent:.3f}% |", f"| Stage1 exact payload rows | — | {exact_count}/{exact_required} |", f"| Stage2 output comparisons | — | {sum(state == 'PASS' for state in report_states)}/{args.expected_comparisons} |", f"| Complete slice coverage | — | {'PASS' if complete_coverage else 'FAIL'} |", "", "## Control timing breakdown", ""]
    for name, value in raw["control_breakdown"].items():
        lines.append(f"- `{name}`: count={value['count']}, mean={value['mean_ms']}, P50={value['p50_ms']}, P95={value['p95_ms']}")
    lines.extend(["", "## Candidate timing breakdown", ""])
    for name, value in raw["candidate_breakdown"].items():
        lines.append(f"- `{name}`: count={value['count']}, mean={value['mean_ms']}, P50={value['p50_ms']}, P95={value['p95_ms']}")
    lines.extend(["", "## Raw numbers", "", "```json", json.dumps(raw, indent=2, sort_keys=True), "```", ""])
    Path(args.markdown).write_text("\n".join(lines))
    print(f"E9_DECISION={raw['decision']} saved_percent={saved_percent:.3f} stage1_exact={exact_count}/{exact_required} stage2_exact={comparison_pass}")


if __name__ == "__main__":
    main()
