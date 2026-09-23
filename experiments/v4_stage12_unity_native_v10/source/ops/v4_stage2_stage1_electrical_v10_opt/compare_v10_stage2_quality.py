#!/usr/bin/env python3
"""Strict Stage2 equivalence gate with bounded production-float tolerance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from pandas.errors import EmptyDataError


BYTE_EXACT_SUFFIXES = (
    "_lines.csv",
    "_line_vertices.csv",
    "_stage1_line_voxels.csv",
    "_accepted_line_voxels.csv",
    "_selected_fragment_bridges.csv",
    "_stage1_electrical_tracks.csv",
    "_pole_attachments.csv",
)
PRODUCTION_SEMANTIC_SUFFIXES = ("_poles.csv", "_components.csv")
PRIMARY_SUFFIXES = BYTE_EXACT_SUFFIXES + PRODUCTION_SEMANTIC_SUFFIXES
POLE_ATOL = 1.0e-9
POLE_RTOL = 1.0e-9

AUDIT_FIELDS = (
    "stage1_inferred_line_voxels",
    "accepted_stage1_line_voxels",
    "stage1_to_stage2_voxel_preservation",
    "raw_stage1_line_components",
    "joined_stage2_tracks",
    "selected_fragment_bridges",
    "same_lane_pole_terminal_group_merges",
    "geometry_support_samples",
    "geometry_supported_samples",
    "geometry_stage1_voxel_support_fraction",
    "geometry_outside_stage1_voxel_samples",
    "line_path_fragments",
    "point_line_fragments",
    "unjoined_singleton_line_voxels",
    "max_output_turn_deg",
    "max_selected_bridge_gap_ft",
    "max_track_radius_p95_ft",
    "pole_attachments",
    "pole_attachment_requires_stage1_voxel_contact",
    "attachment_geometry_vertices_added",
    "runtime_gt_usage",
    "synthetic_line_voxels",
    "pole_pair_inference",
    "line_refiner_used",
    "line_hysteresis_used",
    "line_to_line_bridge_near_pole_allowed",
    "parallel_lane_merge_allowed",
    "open_line_endpoints_preserved",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-session", required=True)
    p.add_argument("--candidate-session", required=True)
    p.add_argument("--report", required=True)
    return p.parse_args()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relative_files(root: Path, suffixes: tuple[str, ...]) -> dict[str, Path]:
    return {
        str(path.relative_to(root)): path
        for path in root.rglob("*")
        if path.is_file() and path.name.endswith(suffixes)
    }


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def row_keys(frame: pd.DataFrame, component_file: bool) -> list[str]:
    required = ["component_id"]
    if component_file:
        required.insert(0, "class_name")
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise RuntimeError(f"semantic production CSV missing identity columns: {missing}")
    return required + [name for name in ("file_id", "slice_seq") if name in frame.columns]


def keyed(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    work = frame.copy()
    for name in keys:
        work[name] = work[name].astype("string").fillna("<NA>")
    if work.duplicated(keys).any():
        duplicates = work.loc[work.duplicated(keys, keep=False), keys].head(10).to_dict("records")
        raise RuntimeError(f"duplicate production row identities: {duplicates}")
    return work.sort_values(keys, kind="stable").reset_index(drop=True)


def compare_frame_values(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    keys: list[str],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if list(left.columns) != list(right.columns):
        raise RuntimeError(
            f"production CSV schema/order changed: baseline={list(left.columns)} "
            f"candidate={list(right.columns)}"
        )
    if len(left) != len(right):
        raise RuntimeError(f"production row count changed: baseline={len(left)} candidate={len(right)}")
    if left.empty:
        return {"rows": 0, "numeric_cells": 0, "max_abs_delta": 0.0, "max_rel_delta": 0.0}

    l = keyed(left, keys)
    r = keyed(right, keys)
    left_ids = [tuple(row) for row in l[keys].itertuples(index=False, name=None)]
    right_ids = [tuple(row) for row in r[keys].itertuples(index=False, name=None)]
    if left_ids != right_ids:
        raise RuntimeError("production row identities changed")

    max_abs = 0.0
    max_rel = 0.0
    numeric_cells = 0
    for name in l.columns:
        if name in keys:
            continue
        left_numeric = is_numeric_dtype(l[name].dtype) and not is_bool_dtype(l[name].dtype)
        right_numeric = is_numeric_dtype(r[name].dtype) and not is_bool_dtype(r[name].dtype)
        if left_numeric != right_numeric:
            raise RuntimeError(f"production column type changed: {name}")
        if left_numeric:
            lv = pd.to_numeric(l[name], errors="raise").to_numpy(dtype=float)
            rv = pd.to_numeric(r[name], errors="raise").to_numpy(dtype=float)
            close = np.isclose(lv, rv, atol=atol, rtol=rtol, equal_nan=True)
            if not bool(np.all(close)):
                bad = np.flatnonzero(~close)[:5]
                details = [{"row": int(i), "baseline": float(lv[i]), "candidate": float(rv[i])} for i in bad]
                raise RuntimeError(f"production numeric values changed beyond tolerance in {name}: {details}")
            finite = np.isfinite(lv) & np.isfinite(rv)
            if bool(np.any(finite)):
                delta = np.abs(lv[finite] - rv[finite])
                denom = np.maximum(np.abs(lv[finite]), 1.0e-300)
                max_abs = max(max_abs, float(np.max(delta)))
                max_rel = max(max_rel, float(np.max(delta / denom)))
                numeric_cells += int(np.sum(finite))
        else:
            lv = l[name].astype("string").fillna("<NA>").tolist()
            rv = r[name].astype("string").fillna("<NA>").tolist()
            if lv != rv:
                raise RuntimeError(f"production categorical/boolean values changed in {name}")
    return {
        "rows": int(len(l)),
        "numeric_cells": int(numeric_cells),
        "max_abs_delta": float(max_abs),
        "max_rel_delta": float(max_rel),
    }


def compare_production_csv(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = read_csv(left_path)
    right = read_csv(right_path)
    is_components = left_path.name.endswith("_components.csv")
    keys = row_keys(left, is_components)
    if is_components:
        if "class_name" not in right.columns:
            raise RuntimeError("candidate components CSV lacks class_name")
        left_class = left["class_name"].astype(str)
        right_class = right["class_name"].astype(str)
        if not bool(left_class.isin(["line", "pole"]).all()) or not bool(right_class.isin(["line", "pole"]).all()):
            raise RuntimeError("components CSV contains an unexpected class_name")
        compare_frame_values(
            left[left_class.eq("line")].copy(),
            right[right_class.eq("line")].copy(),
            keys=keys,
            atol=0.0,
            rtol=0.0,
        )
        return compare_frame_values(
            left[left_class.eq("pole")].copy(),
            right[right_class.eq("pole")].copy(),
            keys=keys,
            atol=POLE_ATOL,
            rtol=POLE_RTOL,
        )
    return compare_frame_values(left, right, keys=keys, atol=POLE_ATOL, rtol=POLE_RTOL)


def compare_sessions(baseline: Path, candidate: Path, report_path: Path) -> dict[str, Any]:
    if not baseline.is_dir() or not candidate.is_dir():
        raise RuntimeError("baseline or candidate session directory missing")
    base_files = relative_files(baseline, PRIMARY_SUFFIXES)
    candidate_files = relative_files(candidate, PRIMARY_SUFFIXES)
    if set(base_files) != set(candidate_files):
        missing = sorted(set(base_files) - set(candidate_files))
        extra = sorted(set(candidate_files) - set(base_files))
        raise RuntimeError(f"primary output inventory differs; missing={missing} extra={extra}")

    production_differences = []
    for relative in sorted(base_files):
        left_path = base_files[relative]
        right_path = candidate_files[relative]
        left_hash = digest(left_path)
        right_hash = digest(right_path)
        if left_hash == right_hash:
            continue
        if left_path.name.endswith(BYTE_EXACT_SUFFIXES):
            raise RuntimeError(
                f"byte-exact electrical/line Stage2 output changed: "
                f"{relative} baseline={left_hash} candidate={right_hash}"
            )
        metrics = compare_production_csv(left_path, right_path)
        production_differences.append({"relative_path": relative, **metrics})

    base_audits = relative_files(baseline, ("_stage1_electrical_track_audit.json",))
    candidate_audits = relative_files(candidate, ("_stage1_electrical_track_audit.json",))
    if set(base_audits) != set(candidate_audits):
        raise RuntimeError("audit inventory differs")
    audit_mismatches = []
    for relative in sorted(base_audits):
        left = json.loads(base_audits[relative].read_text())
        right = json.loads(candidate_audits[relative].read_text())
        differences = {
            field: {"baseline": left.get(field), "candidate": right.get(field)}
            for field in AUDIT_FIELDS
            if left.get(field) != right.get(field)
        }
        if differences:
            audit_mismatches.append({"relative_path": relative, "differences": differences})
    if audit_mismatches:
        raise RuntimeError(f"semantic Stage2 audits changed: {audit_mismatches[:5]}")

    report = {
        "equivalence_version": "v10-stage2-quality-production-float-tolerance-opt1-fix2-20260910",
        "passed": True,
        "primary_files_compared": len(base_files),
        "byte_exact_suffixes": list(BYTE_EXACT_SUFFIXES),
        "production_semantic_suffixes": list(PRODUCTION_SEMANTIC_SUFFIXES),
        "production_numeric_atol": POLE_ATOL,
        "production_numeric_rtol": POLE_RTOL,
        "production_files_with_benign_serialization_differences": production_differences,
        "audits_compared": len(base_audits),
        "semantic_audit_mismatches": 0,
        "excluded_performance_diagnostics": [
            "fragment_bridge_candidates_csv",
            "timing_csv",
            "runtime_version",
            "disconnected_bridges_blocked_by_voxel_support",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    a = parse_args()
    report = compare_sessions(
        Path(a.baseline_session).resolve(),
        Path(a.candidate_session).resolve(),
        Path(a.report).resolve(),
    )
    print("V10_STAGE2_QUALITY_EQUIVALENCE_OK")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
