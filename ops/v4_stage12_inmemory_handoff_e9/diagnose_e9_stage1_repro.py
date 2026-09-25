#!/usr/bin/env python3
"""Read-only, complete Stage 1 payload comparison for the E9 control failures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from v4_realtime_core import label_from_scores, load_calibration


SCORES = ("pole", "line", "objectness")
STRUCTURE = ("coords", "dist_values", "source_rows", "raw_labels", "semantic")


def inventory(root: Path) -> dict[str, Path]:
    return {str(p.relative_to(root)): p for p in root.rglob("*_stage1.npz") if p.is_file()}


def same(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError("shape mismatch")
    eq = a == b
    if a.dtype.kind in "fc" and b.dtype.kind in "fc":
        eq = eq | (np.isnan(a) & np.isnan(b))
    return eq


def aggregate() -> dict:
    return {"changed_values": 0, "affected_slices": 0, "max_abs": 0.0,
            "nonfinite_difference": 0, "dtype_or_shape_mismatch": 0}


def compare_one(reference: Path, candidate: Path, calibration: dict) -> dict:
    base_files, test_files = inventory(reference), inventory(candidate)
    result = {
        "reference": str(reference), "candidate": str(candidate),
        "reference_files": len(base_files), "candidate_files": len(test_files),
        "missing_files": sorted(base_files.keys() - test_files.keys())[:20],
        "extra_files": sorted(test_files.keys() - base_files.keys())[:20],
        "files_compared": 0,
        "structural_differences": {},
        "scores": {name: aggregate() for name in SCORES},
        "predicted_labels": {"known_positive_losses": 0,
                             "known_positive_class_flips": 0,
                             "unverified_additions": 0,
                             "changed_labels": 0, "affected_slices": 0},
        "changed_slices": [], "errors": [],
    }
    for rel in sorted(base_files.keys() & test_files.keys()):
        if len(result["errors"]) >= 20:
            break
        try:
            with np.load(base_files[rel], allow_pickle=False) as ref, np.load(test_files[rel], allow_pickle=False) as test:
                result["files_compared"] += 1
                detail = {"file": rel, "scores": {}}
                for name in sorted((set(ref.files) | set(test.files)) - set(SCORES)):
                    if name not in ref or name not in test:
                        result["structural_differences"][name] = result["structural_differences"].get(name, 0) + 1
                        continue
                    a, b = ref[name], test[name]
                    if a.dtype != b.dtype or a.shape != b.shape or not np.all(same(a, b)):
                        result["structural_differences"][name] = result["structural_differences"].get(name, 0) + 1
                for name in SCORES:
                    if name not in ref or name not in test:
                        result["errors"].append(f"{rel}: missing {name}")
                        continue
                    a, b = ref[name], test[name]
                    totals = result["scores"][name]
                    if a.dtype != b.dtype or a.shape != b.shape:
                        totals["dtype_or_shape_mismatch"] += 1
                        detail["scores"][name] = {"reference_shape": a.shape, "candidate_shape": b.shape,
                                                   "reference_dtype": str(a.dtype), "candidate_dtype": str(b.dtype)}
                        continue
                    changed = ~same(a, b)
                    count = int(np.count_nonzero(changed))
                    if count:
                        finite = changed & np.isfinite(a) & np.isfinite(b)
                        max_abs = float(np.max(np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64)))) if np.any(finite) else 0.0
                        nonfinite = int(np.count_nonzero(changed & ~finite))
                        totals["changed_values"] += count
                        totals["affected_slices"] += 1
                        totals["max_abs"] = max(totals["max_abs"], max_abs)
                        totals["nonfinite_difference"] += nonfinite
                        first = tuple(int(i) for i in np.argwhere(changed)[0])
                        detail["scores"][name] = {"changed_values": count, "max_abs": max_abs,
                                                   "nonfinite_difference": nonfinite,
                                                   "first_index": first,
                                                   "first_reference": str(a[first]),
                                                   "first_candidate": str(b[first])}
                if all(name in ref and name in test and ref[name].shape == test[name].shape for name in ("pole", "line")):
                    pthr, lthr = calibration["pole_threshold"], calibration["line_threshold"]
                    base_label = label_from_scores(ref["pole"], ref["line"], pthr, lthr)
                    test_label = label_from_scores(test["pole"], test["line"], pthr, lthr)
                    labels = result["predicted_labels"]
                    changed = base_label != test_label
                    num_changed = int(np.count_nonzero(changed))
                    labels["changed_labels"] += num_changed
                    labels["known_positive_losses"] += int(np.count_nonzero((base_label > 0) & (test_label == 0)))
                    labels["known_positive_class_flips"] += int(np.count_nonzero((base_label > 0) & (test_label > 0) & changed))
                    labels["unverified_additions"] += int(np.count_nonzero((base_label == 0) & (test_label > 0)))
                    labels["affected_slices"] += int(num_changed > 0)
                    if num_changed:
                        detail["predicted_label_changes"] = num_changed
                if detail["scores"] or "predicted_label_changes" in detail:
                    result["changed_slices"].append(detail)
        except Exception as exc:
            result["errors"].append(f"{rel}: {type(exc).__name__}: {exc}")
    result["changed_slice_count"] = len(result["changed_slices"])
    result["changed_slices"] = result["changed_slices"][:30]
    result["exact_stage1_payload"] = (
        not result["missing_files"] and not result["extra_files"] and not result["errors"]
        and not result["structural_differences"]
        and result["reference_files"] > 0
        and result["files_compared"] == result["reference_files"]
        and all(s["changed_values"] == s["dtype_or_shape_mismatch"] == 0 for s in result["scores"].values())
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--run", action="append", required=True, help="LABEL=STAGE1_DIRECTORY")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    calibration = load_calibration(args.calibration)
    report = {"purpose": "read_only_stage1_reproducibility_diagnostic_not_ground_truth_quality",
              "score_contract": "exact score arrays required (score_atol=0)",
              "reference": str(args.reference), "runs": {}}
    for spec in args.run:
        label, path = spec.split("=", 1)
        report["runs"][label] = compare_one(args.reference, Path(path), calibration)
        run = report["runs"][label]
        print(f"{label}: exact={run['exact_stage1_payload']} files={run['files_compared']}/{run['reference_files']} "
              f"slices_with_score_changes={run['changed_slice_count']} "
              f"labels_changed={run['predicted_labels']['changed_labels']}", flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"REPORT={args.report}", flush=True)


if __name__ == "__main__":
    main()
