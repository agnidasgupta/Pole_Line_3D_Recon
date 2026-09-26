#!/usr/bin/env python3
"""Summarize E11's paired component timing only; it never promotes production."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_rows(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(root.rglob("*.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if {"group_id", "slice_seq"} <= set(frame.columns):
            frames.append(frame)
    if not frames:
        raise RuntimeError(f"No E11 timing rows under {root}")
    out = pd.concat(frames, ignore_index=True)
    out["slice_seq"] = pd.to_numeric(out["slice_seq"], errors="raise").astype(int)
    if out.duplicated(["group_id", "slice_seq"]).any():
        raise RuntimeError(f"Duplicate E11 timing keys under {root}")
    return out.sort_values(["group_id", "slice_seq"], kind="stable")


def stat(frame: pd.DataFrame, name: str) -> dict[str, float | int | None]:
    values = pd.to_numeric(frame.get(name), errors="coerce").dropna().to_numpy(dtype=float) if name in frame else np.array([])
    if not len(values):
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None}
    return {"count": int(len(values)), "mean_ms": float(values.mean()), "p50_ms": float(np.quantile(values, .5)), "p95_ms": float(np.quantile(values, .95))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--control-stage1", required=True)
    p.add_argument("--control", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--comparisons", required=True)
    p.add_argument("--json", required=True)
    p.add_argument("--markdown", required=True)
    p.add_argument("--expected-repeats", type=int, default=2)
    args = p.parse_args()
    control_stage1 = read_rows(Path(args.control_stage1))
    control, candidate = read_rows(Path(args.control)), read_rows(Path(args.candidate))
    keys = ["group_id", "slice_seq"]
    control = control_stage1.merge(control, on=keys, suffixes=("_stage1", "_stage2"), validate="one_to_one")
    if set(map(tuple, control[keys].itertuples(index=False, name=None))) != set(map(tuple, candidate[keys].itertuples(index=False, name=None))):
        raise RuntimeError("E11 control/candidate coverage differs")
    # Exclude warm-up slice per representative run, matching E9/E10 policy.
    control = control.loc[control.groupby("group_id").cumcount().gt(0)].copy()
    candidate = candidate.loc[candidate.groupby("group_id").cumcount().gt(0)].copy()
    reports = [json.loads(path.read_text()) for path in sorted(Path(args.comparisons).glob("*.json"))]
    comparisons_ok = len(reports) == args.expected_repeats and all(r.get("status") == "PASS" for r in reports)
    exact_ok = int(pd.to_numeric(candidate.get("stage1_exact"), errors="coerce").fillna(0).sum()) == len(candidate)
    control["e11_disk_total_ms"] = pd.to_numeric(control["e11_disk_callback_ms_excluding_timing_write"], errors="raise")
    candidate["e11_candidate_total_ms"] = pd.to_numeric(candidate["stage2_callback_and_output_ms"], errors="raise")
    control["e11_disk_total_with_stage1_ms"] = sum(
        pd.to_numeric(control[name], errors="raise") for name in
        ("csv_read_ms", "sparse_item_prep_ms", "stage1_wall_ms", "stage1_artifact_write_ms", "stage1_manifest_write_ms", "e11_disk_total_ms")
    )
    candidate["e11_candidate_total_with_stage1_ms"] = sum(
        pd.to_numeric(candidate[name], errors="raise") for name in
        ("csv_read_ms", "sparse_item_prep_ms", "stage1_inference_ms", "stage1_manifest_write_ms", "e11_candidate_total_ms")
    )
    control_mean = float(control["e11_disk_total_ms"].mean())
    candidate_mean = float(candidate["e11_candidate_total_ms"].mean())
    control_total_mean = float(control["e11_disk_total_with_stage1_ms"].mean())
    candidate_total_mean = float(candidate["e11_candidate_total_with_stage1_ms"].mean())
    saved = 100.0 * (control_total_mean - candidate_total_mean) / control_total_mean
    credible = bool(comparisons_ok and exact_ok and saved >= 2.0)
    control_fields = ["csv_read_ms", "sparse_item_prep_ms", "stage1_wall_ms", "stage1_artifact_write_ms", "stage1_manifest_write_ms", "e11_disk_artifact_load_ms", "e11_disk_process_ms", "e11_disk_primary_output_write_ms", "e11_disk_audit_output_write_ms", "e11_disk_audit_json_write_ms", "e11_disk_manifest_update_ms", "e11_disk_unattributed_ms", "e11_disk_total_ms", "e11_disk_total_with_stage1_ms"]
    candidate_fields = ["csv_read_ms", "sparse_item_prep_ms", "stage1_inference_ms", "stage1_manifest_write_ms", "e11_candidate_process_ms", "e11_candidate_payload_materialization_ms", "e11_candidate_primary_output_write_ms", "e11_candidate_audit_output_write_ms", "e11_candidate_audit_json_write_ms", "e11_candidate_manifest_update_ms", "e11_candidate_timing_row_write_ms", "e11_candidate_unattributed_ms", "e11_candidate_total_ms", "e11_candidate_total_with_stage1_ms"]
    raw = {
        "experiment": "E11_stage12_timing_decomposition", "decision": "FULL30_ELIGIBLE" if credible else "STOP_NO_CREDIBLE_FULL30_GAIN",
        "quality": {"stage1_exact": exact_ok, "stage2_comparisons": len(reports), "stage2_comparisons_pass": comparisons_ok},
        "observed": {"timed_rows": len(candidate), "control_stage2_callback_mean_ms": control_mean, "candidate_stage2_callback_mean_ms": candidate_mean, "control_end_to_end_mean_ms": control_total_mean, "candidate_end_to_end_mean_ms": candidate_total_mean, "saved_percent": saved},
        "control_components": {name: stat(control, name) for name in control_fields},
        "candidate_components": {name: stat(candidate, name) for name in candidate_fields},
        "interpretation": "Exactness diagnostics are outside timed callback totals. E11 is a representative paired decomposition only; it never starts or approves a full-30 run.",
    }
    Path(args.json).write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    lines = ["# E11 Stage1-to-Stage2 timing decomposition", "", f"**Decision:** `{raw['decision']}`", "", "| Metric | Disk control | Persistent in-memory candidate |", "| --- | ---: | ---: |", f"| Timed rows | {len(control)} | {len(candidate)} |", f"| Stage2 callback mean (ms) | {control_mean:.3f} | {candidate_mean:.3f} |", f"| End-to-end mean (ms) | {control_total_mean:.3f} | {candidate_total_mean:.3f} |", f"| End-to-end mean time saved | — | {saved:.3f}% |", f"| Exact Stage1 payloads | — | {exact_ok} |", f"| Exact Stage2 comparisons | {comparisons_ok} ({len(reports)}/{args.expected_repeats}) | {comparisons_ok} ({len(reports)}/{args.expected_repeats}) |", "", "## Disk control components", ""]
    lines += [f"- `{name}`: {value}" for name, value in raw["control_components"].items()]
    lines += ["", "## Persistent in-memory components", ""]
    lines += [f"- `{name}`: {value}" for name, value in raw["candidate_components"].items()]
    lines += ["", raw["interpretation"], ""]
    Path(args.markdown).write_text("\n".join(lines))
    print(f"E11_DECISION={raw['decision']} saved_percent={saved:.3f} stage1_exact={exact_ok} stage2_exact={comparisons_ok}")


if __name__ == "__main__":
    main()
