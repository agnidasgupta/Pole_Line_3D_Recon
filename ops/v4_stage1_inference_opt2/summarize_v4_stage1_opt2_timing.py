#!/usr/bin/env python3
"""Write Opt2 per-session Stage-1 timing with accepted-baseline comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


COMPONENTS = [
    ("csv_read_ms", "CSV parse and dataframe materialization"),
    ("sparse_item_prep_ms", "validation, rounding, dedupe, sparse arrays"),
    ("core_schedule_ms", "occupied voxel to active-core schedule"),
    ("workspace_prepare_ms", "host launch/setup for reusable GPU workspace"),
    ("gpu_workspace_reset_ms", "first zero allocation or sparse prior-slice clear"),
    ("host_pin_ms", "pin sparse coordinate and distance arrays"),
    ("sparse_h2d_cuda_ms", "sparse coordinate/distance transfer to GPU"),
    ("gpu_sparse_scatter_ms", "scatter occupied values into GPU workspace"),
    ("gpu_patch_extract_ms", "extract accepted 64-cube patches"),
    ("gpu_feature_assembly_ms", "coordinate channels and channels-last layout"),
    ("gpu_model_ms", "accepted network forward and score fusion"),
    ("gpu_gather_plan_ms", "host occupied-row gather-index plan"),
    ("gpu_gather_ms", "gather core predictions into occupied-row arrays"),
    ("d2h_gather_ms", "single occupied-row result transfer to host"),
    ("stage1_wall_ms", "complete Stage1 prediction call"),
    ("stage1_artifact_write_ms", "atomic NPZ and metadata JSON write"),
    ("stage1_manifest_write_ms", "atomic session manifest update"),
    ("slice_total_ms", "CSV read through durable manifest/timing record"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-timing-dir", required=True)
    p.add_argument("--baseline-run", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def candidate_rows(root: Path):
    frames = []
    for path in sorted(root.glob("*.csv")):
        frame = pd.read_csv(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError(f"no candidate Stage1 timing rows under {root}")
    return pd.concat(frames, ignore_index=True)


def baseline_rows(run: Path):
    rows = []
    for path in sorted((run / "stage1").rglob("*_stage1.json")):
        payload = json.loads(path.read_text())
        timing = dict(payload.get("timing", {}))
        timing.update(
            {
                "group_id": payload.get("group_id", ""),
                "slice_seq": payload.get("slice_seq"),
                "relative_path": payload.get("relative_path", ""),
            }
        )
        rows.append(timing)
    if not rows:
        raise RuntimeError(f"no baseline Stage1 metadata timings under {run / 'stage1'}")
    return pd.DataFrame(rows)


def mean(frame, name):
    if name not in frame.columns:
        return None
    values = pd.to_numeric(frame[name], errors="coerce").dropna().to_numpy(float)
    return float(values.mean()) if len(values) else None


def comparable_total(frame):
    parts = []
    for name in ("csv_read_ms", "sparse_item_prep_ms", "stage1_wall_ms"):
        if name not in frame.columns:
            return None
        parts.append(pd.to_numeric(frame[name], errors="coerce"))
    values = sum(parts).dropna().to_numpy(float)
    return float(values.mean()) if len(values) else None


def format_row(name, baseline_value, candidate_value):
    base = "n/a" if baseline_value is None else f"{baseline_value:12.3f}"
    cand = "n/a" if candidate_value is None else f"{candidate_value:12.3f}"
    if baseline_value is None or candidate_value is None or baseline_value == 0:
        delta = saved = speedup = "n/a"
    else:
        delta = f"{candidate_value - baseline_value:12.3f}"
        saved = f"{100.0 * (baseline_value - candidate_value) / baseline_value:9.2f}%"
        speedup = f"{baseline_value / max(candidate_value, 1.0e-12):8.3f}x"
    return f"{name:31s} {base:>12s} {cand:>12s} {delta:>12s} {saved:>10s} {speedup:>9s}"


def main():
    a = parse_args()
    candidate = candidate_rows(Path(a.candidate_timing_dir).resolve())
    baseline = baseline_rows(Path(a.baseline_run).resolve())
    candidate["group_id"] = candidate["group_id"].astype(str)
    baseline["group_id"] = baseline["group_id"].astype(str)
    groups = sorted(candidate["group_id"].unique())
    missing_groups = set(groups) - set(baseline["group_id"].unique())
    if missing_groups:
        raise RuntimeError(f"candidate timing groups absent from baseline: {sorted(missing_groups)}")
    baseline = baseline[baseline["group_id"].isin(groups)].copy()

    lines = [
        "V4 STAGE1 OPT2 TIMING: SESSION AVERAGES PER SLICE",
        "",
        "Execution contract: accepted checkpoint/calibration, active_gpu, BF16, fixed batch shape,",
        "64^3 input patch, 48^3 output core, unchanged score fusion and thresholds;",
        "all verified-positive Pole/Line voxels retained with no class flips.",
        "Label 0 is unknown, not a verified negative; additions are reported for review.",
        "Score maximum absolute difference <= 1e-4 is a fidelity diagnostic, not a quality metric.",
        "Baseline-comparable total = csv_read_ms + sparse_item_prep_ms + stage1_wall_ms.",
        "Write timings are Opt2 instrumentation and therefore show baseline n/a.",
        "Positive saved% and speedup above 1.0x mean the candidate is faster.",
        "",
    ]

    overall_base = comparable_total(baseline)
    overall_candidate = comparable_total(candidate)
    lines.extend(
        [
            f"ALL_SESSIONS slices={len(candidate)} sessions={len(groups)}",
            format_row("baseline_comparable_total_ms", overall_base, overall_candidate),
            format_row("stage1_wall_ms", mean(baseline, "stage1_wall_ms"), mean(candidate, "stage1_wall_ms")),
            "",
        ]
    )

    for group_id in groups:
        cand = candidate[candidate.group_id.eq(group_id)].copy()
        base = baseline[baseline.group_id.eq(group_id)].copy()
        if len(cand) != len(base):
            raise RuntimeError(
                f"timing slice count differs for {group_id}: baseline={len(base)} candidate={len(cand)}"
            )
        lines.extend(
            [
                f"SESSION {group_id}",
                f"slices={len(cand)} baseline_slices={len(base)}",
                "component                       baseline_ms candidate_ms     delta_ms     saved%   speedup",
                format_row("baseline_comparable_total_ms", comparable_total(base), comparable_total(cand)),
            ]
        )
        for name, _description in COMPONENTS:
            lines.append(format_row(name, mean(base, name), mean(cand, name)))
        lines.append("")

    lines.extend(["COMPONENT DEFINITIONS", ""])
    for name, description in COMPONENTS:
        lines.append(f"{name}: {description}.")
    lines.append("")
    output = Path(a.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(f"V4_STAGE1_OPT2_TIMING_SUMMARY_OK output={output}")


if __name__ == "__main__":
    main()
