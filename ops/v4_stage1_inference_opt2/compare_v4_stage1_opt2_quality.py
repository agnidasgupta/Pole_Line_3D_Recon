#!/usr/bin/env python3
"""One-sided Stage-1 quality guard for execution-only Opt2 candidates.

The accepted output is a positive-only reference: Pole/Line predictions are
trusted positives, but label 0 is unknown rather than a trusted negative.
Consequently this guard rejects losses and Pole<->Line flips of known positives.
Candidate-only positives are reported as unverified and require review; they are
not called false positives. Numerical/structural checks are implementation
fidelity checks and are reported separately from inference quality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v4_realtime_core import label_from_scores, load_calibration


STRUCTURAL_ARRAYS = ("coords", "dist_values", "source_rows", "raw_labels")
SCORE_ARRAYS = ("pole", "line", "objectness")
META_FIELDS = (
    "contract_version", "stage", "id", "source", "relative_path", "geography",
    "session", "slice_seq", "group_id", "rows", "occupied_rows", "center_metadata",
    "evaluate_all_cores", "gpu_coord_channels", "fixed_batch_shape", "amp",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-stage1", required=True)
    p.add_argument("--candidate-stage1", required=True)
    p.add_argument("--baseline-export", required=True)
    p.add_argument("--candidate-export", required=True)
    p.add_argument("--calibration-json", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--score-atol", type=float, default=1.0e-4)
    return p.parse_args()


def relative_files(root: Path, suffix: str):
    return {
        str(path.relative_to(root)): path
        for path in root.rglob(f"*{suffix}")
        if path.is_file()
    }


def exact_equal(left, right):
    if left.dtype.kind in "fc" or right.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def finite_max_abs(left, right):
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    if np.any(np.isnan(a) != np.isnan(b)) or np.any(np.isinf(a) != np.isinf(b)):
        return float("inf")
    finite = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[finite] - b[finite]), initial=0.0))


def positive_delta(reference, candidate):
    reference = np.asarray(reference, dtype=np.int8)
    candidate = np.asarray(candidate, dtype=np.int8)
    if reference.shape != candidate.shape:
        raise RuntimeError("predicted-label shapes changed")
    known = reference > 0
    lost = known & (candidate == 0)
    flipped = known & (candidate > 0) & (candidate != reference)
    additions = (reference == 0) & (candidate > 0)
    return {
        "known_positive_reference": int(np.count_nonzero(known)),
        "known_positive_retained": int(np.count_nonzero(known & (candidate == reference))),
        "known_positive_lost_to_unknown": int(np.count_nonzero(lost)),
        "known_positive_class_flips": int(np.count_nonzero(flipped)),
        "unverified_candidate_additions": int(np.count_nonzero(additions)),
    }


def add_counts(total, current):
    for name, value in current.items():
        total[name] = total.get(name, 0) + int(value)


def compare_manifest(left_path: Path, right_path: Path):
    left = pd.read_csv(left_path)
    right = pd.read_csv(right_path)
    ignored = {"stage1_npz", "stage1_meta_json"}
    columns = [name for name in left.columns if name not in ignored]
    if [name for name in right.columns if name not in ignored] != columns:
        raise RuntimeError("Stage1 manifest schema/order changed")
    left = left[columns].sort_values(["group_id", "slice_seq"], kind="stable").reset_index(drop=True)
    right = right[columns].sort_values(["group_id", "slice_seq"], kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_exact=True, check_dtype=False)
    return len(left)


def compare_meta(left: Path, right: Path):
    a = json.loads(left.read_text())
    b = json.loads(right.read_text())
    differences = {name: (a.get(name), b.get(name)) for name in META_FIELDS if a.get(name) != b.get(name)}
    if differences:
        raise RuntimeError(f"Stage1 semantic metadata changed for {left.name}: {differences}")


def compare_csv_frame(left_path: Path, right_path: Path, score_atol: float):
    left = pd.read_csv(left_path)
    right = pd.read_csv(right_path)
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        raise RuntimeError(f"inference CSV shape/schema changed: {left_path.name}")
    max_abs = 0.0
    positive_counts = {}
    semantic_mismatches = 0
    score_columns = {"v4_pole_score", "v4_line_score", "v4_objectness"}
    for name in left.columns:
        if name in score_columns:
            a = pd.to_numeric(left[name], errors="raise").to_numpy(float)
            b = pd.to_numeric(right[name], errors="raise").to_numpy(float)
            delta = finite_max_abs(a, b)
            if delta > score_atol:
                raise RuntimeError(
                    f"inference score changed beyond fidelity tolerance in {name}: {delta} > {score_atol}"
                )
            max_abs = max(max_abs, delta)
        elif name == "v4_pred":
            add_counts(
                positive_counts,
                positive_delta(
                    pd.to_numeric(left[name], errors="raise").to_numpy(np.int8),
                    pd.to_numeric(right[name], errors="raise").to_numpy(np.int8),
                ),
            )
        elif name == "v4_semantic":
            semantic_mismatches += int(np.count_nonzero(left[name].to_numpy() != right[name].to_numpy()))
        else:
            pd.testing.assert_series_equal(
                left[name], right[name], check_exact=True, check_dtype=False, check_names=True
            )
    return max_abs, positive_counts, semantic_mismatches


def metric_artifact_status(base: Path, candidate: Path):
    """Record pseudo-metric equality without treating it as a quality gate."""
    left = pd.read_csv(base / "stage1_metrics_by_slice.csv")
    right = pd.read_csv(candidate / "stage1_metrics_by_slice.csv")
    frame_equal = False
    try:
        pd.testing.assert_frame_equal(left, right, check_exact=True, check_dtype=False)
        frame_equal = True
    except AssertionError:
        pass
    left_json = json.loads((base / "stage1_metrics_summary.json").read_text())
    right_json = json.loads((candidate / "stage1_metrics_summary.json").read_text())
    return {"by_slice_equal": frame_equal, "summary_equal": left_json == right_json}


def main():
    a = parse_args()
    if a.score_atol < 0.0 or a.score_atol > 1.0e-4:
        raise ValueError("--score-atol must remain between 0 and the 1e-4 fidelity ceiling")
    baseline = Path(a.baseline_stage1).resolve()
    candidate = Path(a.candidate_stage1).resolve()
    baseline_export = Path(a.baseline_export).resolve()
    candidate_export = Path(a.candidate_export).resolve()
    report_path = Path(a.report).resolve()
    for path in (baseline, candidate, baseline_export, candidate_export):
        if not path.is_dir():
            raise FileNotFoundError(path)

    manifest_rows = compare_manifest(baseline / "stage1_manifest.csv", candidate / "stage1_manifest.csv")
    base_npz = relative_files(baseline, "_stage1.npz")
    candidate_npz = relative_files(candidate, "_stage1.npz")
    if set(base_npz) != set(candidate_npz):
        raise RuntimeError(
            f"Stage1 NPZ inventory changed: missing={sorted(set(base_npz)-set(candidate_npz))[:10]} "
            f"extra={sorted(set(candidate_npz)-set(base_npz))[:10]}"
        )
    base_meta = relative_files(baseline, "_stage1.json")
    candidate_meta = relative_files(candidate, "_stage1.json")
    if set(base_meta) != set(candidate_meta):
        raise RuntimeError("Stage1 metadata inventory changed")

    calibration = load_calibration(a.calibration_json)
    score_max = {name: 0.0 for name in SCORE_ARRAYS}
    positive_counts = {}
    semantic_mismatches = 0
    occupied_rows = 0
    for relative in sorted(base_npz):
        with np.load(base_npz[relative]) as left, np.load(candidate_npz[relative]) as right:
            if set(left.files) != set(right.files):
                raise RuntimeError(f"NPZ member inventory changed: {relative}")
            for name in STRUCTURAL_ARRAYS:
                if name not in left.files or not exact_equal(left[name], right[name]):
                    raise RuntimeError(f"structural Stage1 array changed: {relative}:{name}")
            for name in SCORE_ARRAYS:
                delta = finite_max_abs(left[name], right[name])
                score_max[name] = max(score_max[name], delta)
                if delta > a.score_atol:
                    raise RuntimeError(
                        f"Stage1 {name} changed beyond fidelity tolerance: {relative} {delta} > {a.score_atol}"
                    )
            semantic_mismatches += int(np.count_nonzero(left["semantic"] != right["semantic"]))
            reference_label = label_from_scores(
                left["pole"], left["line"],
                calibration["pole_threshold"], calibration["line_threshold"],
            )
            candidate_label = label_from_scores(
                right["pole"], right["line"],
                calibration["pole_threshold"], calibration["line_threshold"],
            )
            add_counts(positive_counts, positive_delta(reference_label, candidate_label))
            occupied_rows += int(len(reference_label))

    for relative in sorted(base_meta):
        compare_meta(base_meta[relative], candidate_meta[relative])

    base_csv = relative_files(baseline_export, "_v4_inference.csv.gz")
    candidate_csv = relative_files(candidate_export, "_v4_inference.csv.gz")
    if set(base_csv) != set(candidate_csv):
        raise RuntimeError("Stage1 inference CSV inventory changed")
    csv_score_max = 0.0
    csv_positive_counts = {}
    csv_semantic_mismatches = 0
    for relative in sorted(base_csv):
        delta, counts, semantic_delta = compare_csv_frame(
            base_csv[relative], candidate_csv[relative], a.score_atol
        )
        csv_score_max = max(csv_score_max, delta)
        add_counts(csv_positive_counts, counts)
        csv_semantic_mismatches += semantic_delta

    metrics = metric_artifact_status(baseline_export, candidate_export)
    lost = positive_counts.get("known_positive_lost_to_unknown", 0)
    flips = positive_counts.get("known_positive_class_flips", 0)
    additions = positive_counts.get("unverified_candidate_additions", 0)
    if lost or flips:
        status = "REJECT_KNOWN_POSITIVE_REGRESSION"
        passed = False
    elif additions:
        status = "REVIEW_UNVERIFIED_ADDITIONS"
        passed = False
    else:
        status = "PASS_NO_KNOWN_POSITIVE_REGRESSION"
        passed = True

    report = {
        "guard_version": "v4-stage1-opt2-positive-only-reference-20260917",
        "reference_semantics": {
            "pole_and_line_labels": "verified_positive",
            "label_zero": "unknown_not_negative",
        },
        "passed_for_automatic_promotion": passed,
        "promotion_status": status,
        "score_atol_is_implementation_fidelity_not_quality": a.score_atol,
        "manifest_rows": int(manifest_rows),
        "npz_files": len(base_npz),
        "metadata_files": len(base_meta),
        "inference_csv_files": len(base_csv),
        "occupied_rows_compared": occupied_rows,
        "positive_reference_counts": positive_counts,
        "export_positive_reference_counts": csv_positive_counts,
        "unverified_additions_require_review": bool(additions),
        "score_max_abs": score_max,
        "inference_csv_score_max_abs": csv_score_max,
        "semantic_argmax_mismatches_fidelity_diagnostic": semantic_mismatches,
        "export_semantic_mismatches_fidelity_diagnostic": csv_semantic_mismatches,
        "structural_arrays_exact": list(STRUCTURAL_ARRAYS),
        "pseudo_metric_artifacts_diagnostic_only": metrics,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(status)
    print(json.dumps(report, sort_keys=True))
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
