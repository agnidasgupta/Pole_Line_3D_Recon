#!/usr/bin/env python3
"""Select a Stage-1-label-preserving V4 Stage-2 profile with offline GT.

GT is used only here.  Runtime selection targets preservation of deployed
Stage-1 line labels, while GT prevents preservation of false Stage-1 labels from
becoming extra Stage-2 conductors.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from v4_realtime_pipeline import V4Stage2Processor
from v4_sparse_components import sparse_connected_labels
from v4_stage2_stage1_label import (
    Stage1LabelProfile,
    Stage1LabelStage2Processor,
    deployed_stage1_labels,
)
from v4_stage_contracts import load_stage1_artifact, stage1_paths
from v4_realtime_core import load_calibration


STAGE1_LABEL_WINDOW_FIX_VERSION = "ordinal-v2-20260903"

def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_root")
    p.add_argument("--stage2_bundle")
    p.add_argument("--calibration_json")
    p.add_argument("--output_dir")
    p.add_argument("--target_session", default="VELASCO_CUT_CP/session1")
    p.add_argument("--precision_slice_min", type=int, default=0)
    p.add_argument("--precision_slice_max", type=int, default=20)
    p.add_argument("--recall_slice_min", type=int, default=21)
    p.add_argument("--recall_slice_max", type=int, default=40)
    p.add_argument("--window_mode", choices=["ordinal", "slice_seq"], default="ordinal")
    p.add_argument("--guardrail_slices_per_session", type=int, default=2)
    p.add_argument("--top_profiles_for_guardrail", type=int, default=6)
    p.add_argument("--grid_size", type=int, nargs=3, default=[400, 400, 200])
    p.add_argument("--voxel_size_ft", type=float, default=0.5)
    p.add_argument("--near_tolerance_vox", type=float, default=1.75)
    p.add_argument("--complete_component_recall", type=float, default=0.80)
    p.add_argument("--min_target_true_voxels_recovered", type=int, default=5)
    p.add_argument("--min_target_true_preservation_gain", type=float, default=0.002)
    p.add_argument("--max_target_true_voxels_lost", type=int, default=0)
    p.add_argument("--max_precision_true_voxels_lost", type=int, default=0)
    p.add_argument("--max_precision_false_voxel_delta", type=int, default=0)
    p.add_argument("--max_precision_false_component_delta", type=int, default=0)
    p.add_argument("--max_precision_false_span_delta_ft", type=float, default=0.0)
    p.add_argument("--max_precision_near_drop", type=float, default=0.0025)
    p.add_argument("--max_recall_false_voxel_delta", type=int, default=0)
    p.add_argument("--max_recall_false_component_delta", type=int, default=0)
    p.add_argument("--max_recall_false_span_delta_ft", type=float, default=0.0)
    p.add_argument("--max_recall_near_drop", type=float, default=0.0025)
    p.add_argument("--max_guardrail_near_drop", type=float, default=0.005)
    p.add_argument("--max_guardrail_false_voxel_inflation", type=float, default=1.02)
    p.add_argument("--max_guardrail_false_voxel_delta", type=int, default=0)
    p.add_argument("--max_guardrail_false_component_delta", type=int, default=0)
    p.add_argument("--max_guardrail_false_span_delta_ft", type=float, default=0.0)
    p.add_argument("--max_output_voxel_inflation", type=float, default=2.25)
    p.add_argument("--self_test", action="store_true")
    return p.parse_args()


def host_to_container_legacy(path: Path) -> Path:
    text = str(path)
    prefix = "/workspace/voxel_poleline/outputs"
    if text == prefix:
        return Path("/outputs")
    if text.startswith(prefix + "/"):
        return Path("/outputs" + text[len(prefix):])
    return path


def read_session_manifest(stage1_root: Path, group_id: str) -> tuple[Path, pd.DataFrame]:
    found: list[tuple[Path, pd.DataFrame]] = []
    for path in sorted(stage1_root.rglob("stage1_manifest.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        required = {"group_id", "slice_seq", "relative_path", "stage1_npz", "stage1_meta_json", "status"}
        if not required <= set(frame.columns):
            continue
        rows = frame[
            frame["group_id"].astype(str).eq(group_id)
            & frame["status"].astype(str).eq("completed")
        ].copy()
        if not rows.empty:
            rows["slice_seq"] = pd.to_numeric(rows["slice_seq"], errors="raise").astype(int)
            found.append((path, rows.sort_values("slice_seq", kind="stable")))
    if len(found) != 1:
        raise RuntimeError(f"Expected one Stage1 manifest for {group_id}; found {len(found)}")
    return found[0]


def resolve_artifacts(row: Any, session_dir: Path) -> tuple[Path, Path]:
    npz = host_to_container_legacy(Path(str(row.stage1_npz)))
    meta = host_to_container_legacy(Path(str(row.stage1_meta_json)))
    if npz.is_file() and meta.is_file():
        return npz, meta
    fallback_npz, fallback_meta = stage1_paths(session_dir, str(row.relative_path))
    if fallback_npz.is_file() and fallback_meta.is_file():
        return fallback_npz, fallback_meta
    raise FileNotFoundError(f"Stage1 artifact unavailable: manifest={row.stage1_npz} fallback={fallback_npz}")


def select_window_rows(
    rows: pd.DataFrame,
    slice_min: int | None,
    slice_max: int | None,
    window_mode: str,
) -> pd.DataFrame:
    """Select target-session rows without confusing session ordinals with manifest slice_seq.

    The 3D review uses slice positions within a session (0, 1, 2, ...), while
    Stage1 manifests may retain global/absolute slice_seq values.  In ordinal
    mode we sort the completed target rows by slice_seq, assign a stable
    session_ordinal, and apply the requested window to that ordinal.  The
    original slice_seq is never rewritten and remains the artifact identity.
    """
    ordered = rows.copy()
    ordered["slice_seq"] = pd.to_numeric(ordered["slice_seq"], errors="raise").astype(int)
    ordered = ordered.sort_values("slice_seq", kind="stable").reset_index(drop=True)
    ordered["session_ordinal"] = np.arange(len(ordered), dtype=np.int64)

    if window_mode == "ordinal":
        key = ordered["session_ordinal"]
    elif window_mode == "slice_seq":
        key = ordered["slice_seq"]
    else:
        raise ValueError(f"unsupported window_mode: {window_mode}")

    mask = np.ones(len(ordered), dtype=bool)
    if slice_min is not None:
        mask &= key.to_numpy() >= int(slice_min)
    if slice_max is not None:
        mask &= key.to_numpy() <= int(slice_max)
    return ordered.loc[mask].copy()


def load_rows(
    manifest_path: Path,
    rows: pd.DataFrame,
    slice_min: int | None = None,
    slice_max: int | None = None,
    window_mode: str = "ordinal",
) -> list[dict[str, Any]]:
    selected = select_window_rows(rows, slice_min, slice_max, window_mode)
    out: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        npz, meta = resolve_artifacts(row, manifest_path.parent)
        item, pred, metadata = load_stage1_artifact(npz, meta)
        out.append({"row": row, "item": item, "pred": pred, "meta": metadata})
    return out


def coord_keys(coords: np.ndarray, grid: tuple[int, int, int]) -> np.ndarray:
    c = np.asarray(coords, dtype=np.int64)
    if not len(c):
        return np.empty(0, dtype=np.int64)
    gx, gy, _ = map(int, grid)
    return np.unique((c[:, 2] * gy + c[:, 1]) * gx + c[:, 0])


def accepted_line_components(result: dict[str, Any]) -> list[tuple[str, np.ndarray, dict[str, Any]]]:
    frame = result.get("components", pd.DataFrame())
    if frame.empty or not {"class_name", "component_accept", "component_id"} <= set(frame.columns):
        return []
    accepted = frame[
        frame["class_name"].astype(str).eq("line")
        & frame["component_accept"].astype(bool)
    ]
    points = result["raw_components"]["line_points"]
    out: list[tuple[str, np.ndarray, dict[str, Any]]] = []
    for row in accepted.to_dict("records"):
        cid = str(row["component_id"])
        if cid in points and len(points[cid]):
            out.append((cid, np.asarray(points[cid], dtype=np.int32), row))
    return out


def _near_metrics(gt: np.ndarray, pred: np.ndarray, tolerance: float) -> tuple[float, float]:
    if not len(gt) or not len(pred):
        return 0.0, 0.0
    gt_tree = cKDTree(gt.astype(float))
    pred_tree = cKDTree(pred.astype(float))
    pred_dist, _ = gt_tree.query(pred.astype(float), k=1, workers=-1)
    gt_dist, _ = pred_tree.query(gt.astype(float), k=1, workers=-1)
    return float(np.mean(pred_dist <= tolerance)), float(np.mean(gt_dist <= tolerance))


def evaluate_result(
    item: dict[str, Any],
    pred_scores: dict[str, np.ndarray],
    result: dict[str, Any],
    calibration: dict[str, Any],
    grid: tuple[int, int, int],
    near_tolerance: float,
    complete_component_recall: float,
) -> dict[str, Any]:
    coords = np.asarray(item["coords"], dtype=np.int32)
    gt_labels = np.asarray(item.get("raw_labels", np.zeros(len(coords))), dtype=np.int16)
    if len(gt_labels) != len(coords):
        raise ValueError("raw_labels and coords do not align")
    stage1_labels = deployed_stage1_labels(pred_scores, calibration)
    s1_line = stage1_labels == 2
    gt_line = gt_labels == 2
    valid_stage1 = s1_line & gt_line
    false_stage1 = s1_line & ~gt_line

    components = accepted_line_components(result)
    accepted_points = (
        np.concatenate([points for _, points, _ in components], axis=0)
        if components
        else np.empty((0, 3), dtype=np.int32)
    )
    accepted_keys = coord_keys(accepted_points, grid)
    s1_keys = coord_keys(coords[s1_line], grid)
    valid_keys = coord_keys(coords[valid_stage1], grid)
    false_keys = coord_keys(coords[false_stage1], grid)
    gt_keys = coord_keys(coords[gt_line], grid)

    accepted_outside_stage1 = int(
        np.setdiff1d(accepted_keys, s1_keys, assume_unique=True).size
    )

    retained_stage1 = int(np.intersect1d(accepted_keys, s1_keys, assume_unique=True).size)
    retained_valid = int(np.intersect1d(accepted_keys, valid_keys, assume_unique=True).size)
    retained_false = int(np.intersect1d(accepted_keys, false_keys, assume_unique=True).size)
    gt_tp = int(np.intersect1d(accepted_keys, gt_keys, assume_unique=True).size)
    gt_fp = int(len(accepted_keys) - gt_tp)
    gt_fn = int(len(gt_keys) - gt_tp)
    near_precision, near_recall = _near_metrics(coords[gt_line], accepted_points, near_tolerance)

    false_components = 0
    false_span_ft = 0.0
    gt_tree = cKDTree(coords[gt_line].astype(float)) if np.any(gt_line) else None
    for _, points, row in components:
        near_fraction = 0.0
        if gt_tree is not None and len(points):
            distance, _ = gt_tree.query(points.astype(float), k=1, workers=-1)
            near_fraction = float(np.mean(distance <= near_tolerance))
        if near_fraction < 0.05:
            false_components += 1
            false_span_ft += float(row.get("horizontal_span_ft", 0.0) or 0.0)

    valid_component_total = 0
    valid_component_complete = 0
    if np.any(valid_stage1):
        labels, count = sparse_connected_labels(coords[valid_stage1], grid)
        for cid in range(1, count + 1):
            points = coords[valid_stage1][labels == cid]
            if len(points) < 3:
                continue
            valid_component_total += 1
            pkeys = coord_keys(points, grid)
            recall = int(np.intersect1d(pkeys, accepted_keys, assume_unique=True).size) / max(len(pkeys), 1)
            if recall >= complete_component_recall:
                valid_component_complete += 1

    return {
        "stage1_line_voxels": int(len(s1_keys)),
        "stage1_valid_line_voxels": int(len(valid_keys)),
        "stage1_false_line_voxels": int(len(false_keys)),
        "accepted_voxels": int(len(accepted_keys)),
        "accepted_outside_stage1_line_voxels": accepted_outside_stage1,
        "retained_stage1_line_voxels": retained_stage1,
        "retained_valid_stage1_line_voxels": retained_valid,
        "retained_false_stage1_line_voxels": retained_false,
        "stage1_preservation": float(retained_stage1 / max(len(s1_keys), 1)),
        "valid_stage1_preservation": float(retained_valid / max(len(valid_keys), 1)),
        "gt_voxels": int(len(gt_keys)),
        "gt_tp": gt_tp,
        "gt_fp": gt_fp,
        "gt_fn": gt_fn,
        "gt_exact_precision": float(gt_tp / max(len(accepted_keys), 1)),
        "gt_exact_recall": float(gt_tp / max(len(gt_keys), 1)),
        "near_precision": near_precision,
        "near_recall": near_recall,
        "accepted_line_components": int(len(components)),
        "false_components": int(false_components),
        "false_component_span_ft": float(false_span_ft),
        "valid_stage1_components": int(valid_component_total),
        "valid_stage1_components_complete": int(valid_component_complete),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    sums = {
        key: float(sum(float(row.get(key, 0.0)) for row in rows))
        for key in [
            "stage1_line_voxels", "stage1_valid_line_voxels", "stage1_false_line_voxels",
            "accepted_voxels", "accepted_outside_stage1_line_voxels", "retained_stage1_line_voxels",
            "retained_valid_stage1_line_voxels", "retained_false_stage1_line_voxels",
            "gt_voxels", "gt_tp", "gt_fp", "gt_fn", "accepted_line_components",
            "false_components", "false_component_span_ft", "valid_stage1_components",
            "valid_stage1_components_complete",
        ]
    }
    sums.update({
        "slices": int(len(rows)),
        "stage1_preservation": float(sums["retained_stage1_line_voxels"] / max(sums["stage1_line_voxels"], 1.0)),
        "valid_stage1_preservation": float(sums["retained_valid_stage1_line_voxels"] / max(sums["stage1_valid_line_voxels"], 1.0)),
        "gt_exact_precision": float(sums["gt_tp"] / max(sums["accepted_voxels"], 1.0)),
        "gt_exact_recall": float(sums["gt_tp"] / max(sums["gt_voxels"], 1.0)),
        "near_precision": float(np.mean([row["near_precision"] for row in rows])),
        "near_recall": float(np.mean([row["near_recall"] for row in rows])),
    })
    return sums


def production_processor(bundle: str, grid: tuple[int, int, int], voxel: float) -> V4Stage2Processor:
    return V4Stage2Processor(
        bundle,
        grid_size=grid,
        voxel_size_ft=voxel,
        pole_candidate_threshold=0.15,
        line_candidate_threshold=0.08,
        line_weak_threshold=0.04,
        line_competition_ratio=0.55,
        pole_min_voxels=4,
        line_min_voxels=3,
        edge_width_vox=10,
    )


def generated_profiles() -> list[Stage1LabelProfile]:
    variants = [
        Stage1LabelProfile(
            name="strict_e045_m3",
            cross_section_eps_ft=0.45,
            cross_section_min_samples=3,
            noise_attach_max_ft=0.55,
            max_internal_gap_ft=0.50,
            min_longitudinal_coverage=0.85,
            min_horizontal_length_ft=1.50,
            min_linearity=0.90,
            max_radius_p90_ft=0.75,
            max_tortuosity=1.60,
            override_refiner_floor=0.15,
            override_min_horizontal_length_ft=4.0,
            override_min_longitudinal_coverage=0.90,
            override_min_linearity=0.94,
            override_max_radius_p90_ft=0.65,
            override_max_tortuosity=1.35,
        ),
        Stage1LabelProfile(
            name="strict_e055_m2",
            cross_section_eps_ft=0.55,
            cross_section_min_samples=2,
            noise_attach_max_ft=0.65,
            max_internal_gap_ft=0.50,
            min_longitudinal_coverage=0.80,
            min_horizontal_length_ft=1.25,
            min_linearity=0.86,
            max_radius_p90_ft=0.85,
            max_tortuosity=1.80,
            override_refiner_floor=0.10,
            override_min_horizontal_length_ft=3.5,
            override_min_longitudinal_coverage=0.86,
            override_min_linearity=0.92,
            override_max_radius_p90_ft=0.72,
            override_max_tortuosity=1.45,
        ),
        Stage1LabelProfile(
            name="balanced_e065_m3",
            cross_section_eps_ft=0.65,
            cross_section_min_samples=3,
            noise_attach_max_ft=0.75,
            max_internal_gap_ft=1.00,
            min_longitudinal_coverage=0.75,
            min_horizontal_length_ft=1.00,
            min_linearity=0.82,
            max_radius_p90_ft=1.00,
            max_tortuosity=2.00,
            override_refiner_floor=0.08,
            override_min_horizontal_length_ft=3.0,
            override_min_longitudinal_coverage=0.84,
            override_min_linearity=0.90,
            override_max_radius_p90_ft=0.80,
            override_max_tortuosity=1.60,
        ),
        Stage1LabelProfile(
            name="balanced_e075_m2",
            cross_section_eps_ft=0.75,
            cross_section_min_samples=2,
            noise_attach_max_ft=0.85,
            max_internal_gap_ft=1.00,
            min_longitudinal_coverage=0.70,
            min_horizontal_length_ft=1.00,
            min_linearity=0.78,
            max_radius_p90_ft=1.15,
            max_tortuosity=2.25,
            override_refiner_floor=0.05,
            override_min_horizontal_length_ft=3.0,
            override_min_longitudinal_coverage=0.82,
            override_min_linearity=0.88,
            override_max_radius_p90_ft=0.90,
            override_max_tortuosity=1.75,
        ),
        Stage1LabelProfile(
            name="recall_e085_m2",
            cross_section_eps_ft=0.85,
            cross_section_min_samples=2,
            noise_attach_max_ft=1.00,
            max_internal_gap_ft=1.50,
            min_longitudinal_coverage=0.65,
            min_horizontal_length_ft=0.75,
            min_linearity=0.74,
            max_radius_p90_ft=1.30,
            max_tortuosity=2.50,
            override_refiner_floor=0.03,
            override_min_horizontal_length_ft=2.5,
            override_min_longitudinal_coverage=0.78,
            override_min_linearity=0.86,
            override_max_radius_p90_ft=1.00,
            override_max_tortuosity=1.90,
        ),
        replace(Stage1LabelProfile(), name="no_override_balanced", enable_geometry_override=False),
    ]
    return variants


def evaluate_processor(
    processor: Any,
    cache: list[dict[str, Any]],
    calibration: dict[str, Any],
    grid: tuple[int, int, int],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for record in cache:
        row = record["row"]
        result = processor.process(record["item"], record["pred"], str(row.id), int(row.slice_seq))
        metrics = evaluate_result(
            record["item"], record["pred"], result, calibration, grid,
            args.near_tolerance_vox, args.complete_component_recall,
        )
        metrics["slice_seq"] = int(row.slice_seq)
        metrics["group_id"] = str(row.group_id)
        if hasattr(row, "session_ordinal"):
            metrics["session_ordinal"] = int(row.session_ordinal)
        rows.append(metrics)
    return aggregate(rows), rows


def select_evenly(rows: pd.DataFrame, count: int) -> pd.DataFrame:
    if count <= 0 or len(rows) <= count:
        return rows.copy()
    index = np.unique(np.linspace(0, len(rows) - 1, count, dtype=int))
    return rows.iloc[index].copy()


def discover_guardrail_cache(stage1_root: Path, target: str, per_session: int) -> list[dict[str, Any]]:
    if per_session <= 0:
        return []
    out: list[dict[str, Any]] = []
    for manifest in sorted(stage1_root.rglob("stage1_manifest.csv")):
        try:
            frame = pd.read_csv(manifest)
        except Exception:
            continue
        required = {"group_id", "slice_seq", "relative_path", "stage1_npz", "stage1_meta_json", "status"}
        if not required <= set(frame.columns):
            continue
        frame = frame[frame["status"].astype(str).eq("completed")].copy()
        for gid in sorted(frame["group_id"].dropna().astype(str).unique()):
            if gid == target:
                continue
            rows = frame[frame["group_id"].astype(str).eq(gid)].copy()
            rows["slice_seq"] = pd.to_numeric(rows["slice_seq"], errors="coerce")
            rows = rows.dropna(subset=["slice_seq"]).sort_values("slice_seq")
            used = 0
            for row in select_evenly(rows, per_session * 4).itertuples(index=False):
                try:
                    npz, meta = resolve_artifacts(row, manifest.parent)
                    item, pred, metadata = load_stage1_artifact(npz, meta)
                except Exception:
                    continue
                raw = np.asarray(item.get("raw_labels", []), dtype=np.int16)
                if np.any(raw == 2):
                    out.append({"row": row, "item": item, "pred": pred, "meta": metadata})
                    used += 1
                    if used >= per_session:
                        break
    return out


def profile_record(
    profile: Stage1LabelProfile,
    precision: dict[str, Any],
    recall: dict[str, Any],
    overall: dict[str, Any],
    baseline: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    bp = baseline["precision"]
    br = baseline["recall"]
    bo = baseline["overall"]
    return {
        "profile": profile.name,
        "profile_config": profile.to_dict(),
        "target_true_voxels_recovered": int(recall["retained_valid_stage1_line_voxels"] - br["retained_valid_stage1_line_voxels"]),
        "target_true_voxels_lost": int(max(br["retained_valid_stage1_line_voxels"] - recall["retained_valid_stage1_line_voxels"], 0)),
        "target_true_preservation_gain": float(recall["valid_stage1_preservation"] - br["valid_stage1_preservation"]),
        "target_complete_component_gain": int(recall["valid_stage1_components_complete"] - br["valid_stage1_components_complete"]),
        "precision_true_voxels_lost": int(max(bp["retained_valid_stage1_line_voxels"] - precision["retained_valid_stage1_line_voxels"], 0)),
        "precision_false_voxel_delta": int(precision["retained_false_stage1_line_voxels"] - bp["retained_false_stage1_line_voxels"]),
        "precision_false_component_delta": int(precision["false_components"] - bp["false_components"]),
        "precision_false_span_delta_ft": float(precision["false_component_span_ft"] - bp["false_component_span_ft"]),
        "precision_near_drop": float(bp["near_precision"] - precision["near_precision"]),
        "recall_false_voxel_delta": int(recall["retained_false_stage1_line_voxels"] - br["retained_false_stage1_line_voxels"]),
        "recall_false_component_delta": int(recall["false_components"] - br["false_components"]),
        "recall_false_span_delta_ft": float(recall["false_component_span_ft"] - br["false_component_span_ft"]),
        "recall_near_drop": float(br["near_precision"] - recall["near_precision"]),
        "output_voxel_inflation": float(overall["accepted_voxels"] / max(bo["accepted_voxels"], 1.0)),
        "precision": precision,
        "recall": recall,
        "overall": overall,
    }


def pass_target(record: dict[str, Any], args: argparse.Namespace) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if record["target_true_voxels_recovered"] < args.min_target_true_voxels_recovered:
        reasons.append("insufficient_true_stage1_voxel_recovery")
    if record["target_true_preservation_gain"] < args.min_target_true_preservation_gain:
        reasons.append("insufficient_true_stage1_preservation_gain")
    if record["target_true_voxels_lost"] > args.max_target_true_voxels_lost:
        reasons.append("lost_baseline_true_stage1_voxels")
    if record["precision_true_voxels_lost"] > args.max_precision_true_voxels_lost:
        reasons.append("lost_precision_window_true_stage1_voxels")
    if record["target_complete_component_gain"] < 1:
        reasons.append("no_complete_valid_stage1_component_gain")
    if record["precision_false_voxel_delta"] > args.max_precision_false_voxel_delta:
        reasons.append("precision_false_voxel_increase")
    if record["precision_false_component_delta"] > args.max_precision_false_component_delta:
        reasons.append("precision_false_component_increase")
    if record["precision_false_span_delta_ft"] > args.max_precision_false_span_delta_ft + 1e-9:
        reasons.append("precision_false_span_increase")
    if record["precision_near_drop"] > args.max_precision_near_drop:
        reasons.append("precision_near_drop")
    if record["recall_false_voxel_delta"] > args.max_recall_false_voxel_delta:
        reasons.append("recall_false_voxel_increase")
    if record["recall_false_component_delta"] > args.max_recall_false_component_delta:
        reasons.append("recall_false_component_increase")
    if record["recall_false_span_delta_ft"] > args.max_recall_false_span_delta_ft + 1e-9:
        reasons.append("recall_false_span_increase")
    if record["recall_near_drop"] > args.max_recall_near_drop:
        reasons.append("recall_near_drop")
    if record["output_voxel_inflation"] > args.max_output_voxel_inflation:
        reasons.append("output_voxel_inflation")
    return not reasons, reasons


def score_record(record: dict[str, Any]) -> float:
    return float(
        10000.0 * record["target_complete_component_gain"]
        + 20.0 * record["target_true_voxels_recovered"]
        - 100.0 * max(record["precision_true_voxels_lost"], 0)
        - 50.0 * max(record["precision_false_voxel_delta"], 0)
        - 3000.0 * max(record["precision_false_component_delta"], 0)
        - 100.0 * max(record["precision_false_span_delta_ft"], 0.0)
        - 50.0 * max(record["recall_false_voxel_delta"], 0)
        - 3000.0 * max(record["recall_false_component_delta"], 0)
        - 100.0 * max(record["recall_false_span_delta_ft"], 0.0)
    )


def write_profile_env(path: Path, profile: Stage1LabelProfile) -> None:
    with path.open("w") as handle:
        handle.write("SELECTION_STATUS=SELECTED_SAFE_STAGE1_LABEL_PROFILE\n")
        for key, value in profile.to_dict().items():
            if isinstance(value, bool):
                value = 1 if value else 0
            handle.write(f"STAGE1_LABEL_{key.upper()}={value}\n")


def self_test() -> None:
    profiles = generated_profiles()
    assert len(profiles) >= 5
    assert all(profile.cross_section_eps_ft > 0 for profile in profiles)
    assert all(profile.max_internal_gap_ft >= 0 for profile in profiles)
    demo = pd.DataFrame({"slice_seq": [100, 105, 110, 115], "group_id": ["g"] * 4})
    selected = select_window_rows(demo, 1, 2, "ordinal")
    assert selected["slice_seq"].tolist() == [105, 110]
    assert selected["session_ordinal"].tolist() == [1, 2]
    selected_seq = select_window_rows(demo, 105, 110, "slice_seq")
    assert selected_seq["slice_seq"].tolist() == [105, 110]
    print(f"V4_STAGE2_STAGE1_LABEL_SELECTOR_SELF_TEST_OK version={STAGE1_LABEL_WINDOW_FIX_VERSION}")


def main() -> None:
    args = cli()
    print(f"STAGE1_LABEL_SELECTOR_VERSION={STAGE1_LABEL_WINDOW_FIX_VERSION}", flush=True)
    if args.self_test:
        self_test()
        return
    required = [args.stage1_root, args.stage2_bundle, args.calibration_json, args.output_dir]
    if not all(required):
        raise SystemExit("--stage1_root, --stage2_bundle, --calibration_json and --output_dir are required")
    stage1_root = Path(args.stage1_root).resolve()
    bundle = str(Path(args.stage2_bundle).resolve())
    calibration_path = str(Path(args.calibration_json).resolve())
    out = Path(args.output_dir).resolve()
    if not stage1_root.is_dir():
        raise FileNotFoundError(stage1_root)
    if not Path(bundle).is_file():
        raise FileNotFoundError(bundle)
    if not Path(calibration_path).is_file():
        raise FileNotFoundError(calibration_path)
    out.mkdir(parents=True, exist_ok=True)
    (out / "SELECTOR_VERSION.txt").write_text(STAGE1_LABEL_WINDOW_FIX_VERSION + "\n")
    grid = tuple(map(int, args.grid_size))
    calibration = load_calibration(calibration_path)

    manifest_path, target_rows = read_session_manifest(stage1_root, args.target_session)

    precision_rows_selected = select_window_rows(
        target_rows, args.precision_slice_min, args.precision_slice_max, args.window_mode
    )
    recall_rows_selected = select_window_rows(
        target_rows, args.recall_slice_min, args.recall_slice_max, args.window_mode
    )

    window_map_parts = []
    if not precision_rows_selected.empty:
        part = precision_rows_selected[["group_id", "session_ordinal", "slice_seq", "relative_path"]].copy()
        part.insert(0, "window", "precision")
        window_map_parts.append(part)
    if not recall_rows_selected.empty:
        part = recall_rows_selected[["group_id", "session_ordinal", "slice_seq", "relative_path"]].copy()
        part.insert(0, "window", "recall")
        window_map_parts.append(part)
    if window_map_parts:
        pd.concat(window_map_parts, ignore_index=True).to_csv(out / "target_window_map.csv", index=False)

    if precision_rows_selected.empty or recall_rows_selected.empty:
        ordered = select_window_rows(target_rows, None, None, "ordinal")
        seq_min = int(ordered["slice_seq"].min()) if not ordered.empty else None
        seq_max = int(ordered["slice_seq"].max()) if not ordered.empty else None
        raise RuntimeError(
            "Target precision or recall window has no Stage1 rows: "
            f"mode={args.window_mode} target_rows={len(ordered)} "
            f"manifest_slice_seq_range={seq_min}..{seq_max} "
            f"precision={args.precision_slice_min}..{args.precision_slice_max} "
            f"recall={args.recall_slice_min}..{args.recall_slice_max}"
        )

    precision_cache = load_rows(
        manifest_path, target_rows, args.precision_slice_min, args.precision_slice_max, args.window_mode
    )
    recall_cache = load_rows(
        manifest_path, target_rows, args.recall_slice_min, args.recall_slice_max, args.window_mode
    )
    overall_cache = precision_cache + recall_cache

    baseline_proc = production_processor(bundle, grid, args.voxel_size_ft)
    baseline_precision, baseline_precision_rows = evaluate_processor(baseline_proc, precision_cache, calibration, grid, args)
    baseline_recall, baseline_recall_rows = evaluate_processor(baseline_proc, recall_cache, calibration, grid, args)
    baseline_overall = aggregate(baseline_precision_rows + baseline_recall_rows)
    baseline = {"precision": baseline_precision, "recall": baseline_recall, "overall": baseline_overall}

    records: list[dict[str, Any]] = []
    profiles = generated_profiles()
    for index, profile in enumerate(profiles, 1):
        processor = Stage1LabelStage2Processor(bundle, calibration_path, profile, grid, args.voxel_size_ft)
        precision, precision_rows = evaluate_processor(processor, precision_cache, calibration, grid, args)
        recall, recall_rows = evaluate_processor(processor, recall_cache, calibration, grid, args)
        overall = aggregate(precision_rows + recall_rows)
        record = profile_record(profile, precision, recall, overall, baseline)
        passed, reasons = pass_target(record, args)
        record["target_pass"] = bool(passed)
        record["target_reject_reasons"] = reasons
        record["target_score"] = score_record(record)
        records.append(record)
        print(
            f"[target {index}/{len(profiles)}] {profile.name} pass={passed} "
            f"recovered={record['target_true_voxels_recovered']} "
            f"component_gain={record['target_complete_component_gain']} "
            f"precision_false_delta={record['precision_false_voxel_delta']}",
            flush=True,
        )

    target_candidates = sorted(
        (record for record in records if record["target_pass"]),
        key=lambda record: record["target_score"],
        reverse=True,
    )
    guard_cache = discover_guardrail_cache(stage1_root, args.target_session, args.guardrail_slices_per_session)
    guard_baseline = None
    if guard_cache:
        guard_baseline, _ = evaluate_processor(baseline_proc, guard_cache, calibration, grid, args)

    selected: dict[str, Any] | None = None
    for record in target_candidates[: args.top_profiles_for_guardrail]:
        profile = Stage1LabelProfile(**record["profile_config"])
        if guard_cache and guard_baseline:
            processor = Stage1LabelStage2Processor(bundle, calibration_path, profile, grid, args.voxel_size_ft)
            guard, _ = evaluate_processor(processor, guard_cache, calibration, grid, args)
            near_drop = float(guard_baseline["near_precision"] - guard["near_precision"])
            false_inflation = float(
                guard["retained_false_stage1_line_voxels"]
                / max(guard_baseline["retained_false_stage1_line_voxels"], 1.0)
            )
            false_voxel_delta = int(guard["retained_false_stage1_line_voxels"] - guard_baseline["retained_false_stage1_line_voxels"])
            false_component_delta = int(guard["false_components"] - guard_baseline["false_components"])
            false_span_delta = float(guard["false_component_span_ft"] - guard_baseline["false_component_span_ft"])
            passed = bool(
                near_drop <= args.max_guardrail_near_drop
                and false_inflation <= args.max_guardrail_false_voxel_inflation
                and false_voxel_delta <= args.max_guardrail_false_voxel_delta
                and false_component_delta <= args.max_guardrail_false_component_delta
                and false_span_delta <= args.max_guardrail_false_span_delta_ft + 1e-9
            )
            record["guardrail"] = guard
            record["guardrail_baseline"] = guard_baseline
            record["guardrail_near_drop"] = near_drop
            record["guardrail_false_voxel_inflation"] = false_inflation
            record["guardrail_false_voxel_delta"] = false_voxel_delta
            record["guardrail_false_component_delta"] = false_component_delta
            record["guardrail_false_span_delta_ft"] = false_span_delta
            record["guardrail_pass"] = bool(passed)
        else:
            record["guardrail"] = None
            record["guardrail_baseline"] = None
            record["guardrail_near_drop"] = 0.0
            record["guardrail_false_voxel_inflation"] = 1.0
            record["guardrail_false_voxel_delta"] = 0
            record["guardrail_false_component_delta"] = 0
            record["guardrail_false_span_delta_ft"] = 0.0
            record["guardrail_pass"] = True
        print(
            f"[guardrail] {record['profile']} pass={record['guardrail_pass']} "
            f"near_drop={record['guardrail_near_drop']:.6f} "
            f"false_inflation={record['guardrail_false_voxel_inflation']:.4f}",
            flush=True,
        )
        if record["guardrail_pass"]:
            selected = record
            break

    flat = []
    for record in records:
        flat.append({
            "profile": record["profile"],
            "target_pass": record["target_pass"],
            "target_reject_reasons": ";".join(record["target_reject_reasons"]),
            "target_score": record["target_score"],
            "target_true_voxels_recovered": record["target_true_voxels_recovered"],
            "target_true_voxels_lost": record["target_true_voxels_lost"],
            "precision_true_voxels_lost": record["precision_true_voxels_lost"],
            "target_true_preservation_gain": record["target_true_preservation_gain"],
            "target_complete_component_gain": record["target_complete_component_gain"],
            "precision_false_voxel_delta": record["precision_false_voxel_delta"],
            "precision_false_component_delta": record["precision_false_component_delta"],
            "precision_false_span_delta_ft": record["precision_false_span_delta_ft"],
            "precision_near_drop": record["precision_near_drop"],
            "recall_false_voxel_delta": record["recall_false_voxel_delta"],
            "recall_false_component_delta": record["recall_false_component_delta"],
            "recall_false_span_delta_ft": record["recall_false_span_delta_ft"],
            "recall_near_drop": record["recall_near_drop"],
            "output_voxel_inflation": record["output_voxel_inflation"],
        })
    pd.DataFrame(flat).sort_values("target_score", ascending=False).to_csv(out / "profile_search.csv", index=False)
    (out / "baseline_metrics.json").write_text(json.dumps(baseline, indent=2, sort_keys=True))
    (out / "profile_search.json").write_text(json.dumps(records, indent=2, sort_keys=True))

    if selected is None:
        (out / "NO_SAFE_STAGE1_LABEL_PROFILE.txt").write_text(
            "NO_SAFE_STAGE1_LABEL_PROFILE\n"
            "No profile improved preservation of correct deployed Stage1 line labels in slices 21-40 "
            "while satisfying the slices 0-20 and cross-session GT guardrails.\n"
        )
        print("NO_SAFE_STAGE1_LABEL_PROFILE")
        raise SystemExit(3)

    profile = Stage1LabelProfile(**selected["profile_config"])

    # Persist per-slice before/after metrics for deterministic review. This is
    # offline evaluation only; the selected runtime processor still does not use GT.
    selected_processor = Stage1LabelStage2Processor(
        bundle, calibration_path, profile, grid, args.voxel_size_ft
    )
    _, selected_precision_rows = evaluate_processor(
        selected_processor, precision_cache, calibration, grid, args
    )
    _, selected_recall_rows = evaluate_processor(
        selected_processor, recall_cache, calibration, grid, args
    )
    baseline_slice = pd.DataFrame(baseline_precision_rows + baseline_recall_rows)
    selected_slice = pd.DataFrame(selected_precision_rows + selected_recall_rows)
    if not baseline_slice.empty and not selected_slice.empty:
        keys = ["group_id", "slice_seq"]
        if "session_ordinal" in baseline_slice.columns and "session_ordinal" in selected_slice.columns:
            keys.append("session_ordinal")
        comparison = baseline_slice.merge(
            selected_slice,
            on=keys,
            suffixes=("_baseline", "_selected"),
            validate="one_to_one",
        )
        window_key = comparison["session_ordinal"] if args.window_mode == "ordinal" else comparison["slice_seq"]
        comparison["window"] = np.where(
            window_key.between(args.precision_slice_min, args.precision_slice_max),
            "precision_control",
            "recall_recovery",
        )
        for metric in [
            "retained_valid_stage1_line_voxels",
            "retained_false_stage1_line_voxels",
            "valid_stage1_components_complete",
            "false_components",
            "false_component_span_ft",
            "accepted_voxels",
        ]:
            comparison[f"{metric}_delta"] = (
                comparison[f"{metric}_selected"] - comparison[f"{metric}_baseline"]
            )
        comparison.to_csv(out / "selected_slice_metrics.csv", index=False)

    (out / "selected_profile.json").write_text(json.dumps(profile.to_dict(), indent=2, sort_keys=True))
    write_profile_env(out / "selected_profile.env", profile)
    report = {
        "status": "SELECTED_SAFE_STAGE1_LABEL_PROFILE",
        "target_session": args.target_session,
        "precision_window": [args.precision_slice_min, args.precision_slice_max],
        "recall_window": [args.recall_slice_min, args.recall_slice_max],
        "stage1_label_definition": "v4_realtime_core.label_from_scores with accepted calibration thresholds",
        "window_mode": args.window_mode,
        "runtime_gt_usage": False,
        "synthetic_line_voxels": 0,
        "baseline": baseline,
        "selected": selected,
    }
    (out / "selection_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    with (out / "selection_report.txt").open("w") as handle:
        handle.write("SELECTED_SAFE_STAGE1_LABEL_PROFILE\n")
        handle.write(f"profile={profile.name}\n")
        handle.write(f"target_session={args.target_session}\n")
        handle.write(f"window_mode={args.window_mode}\n")
        handle.write(f"precision_window={args.precision_slice_min}-{args.precision_slice_max}\n")
        handle.write(f"recall_window={args.recall_slice_min}-{args.recall_slice_max}\n")
        for key in [
            "target_true_voxels_recovered", "target_true_voxels_lost",
            "precision_true_voxels_lost", "target_true_preservation_gain", "target_complete_component_gain",
            "precision_false_voxel_delta", "precision_false_component_delta",
            "precision_false_span_delta_ft", "precision_near_drop",
            "recall_false_voxel_delta", "recall_false_component_delta",
            "recall_false_span_delta_ft", "recall_near_drop",
            "output_voxel_inflation", "guardrail_near_drop",
            "guardrail_false_voxel_inflation", "guardrail_false_voxel_delta",
            "guardrail_false_component_delta", "guardrail_false_span_delta_ft",
        ]:
            handle.write(f"{key}={selected.get(key)}\n")
        handle.write("stage1_label_definition=calibrated_fused_v4_label_from_scores\n")
        handle.write("runtime_gt_usage=false\n")
        handle.write("synthetic_line_voxels=0\n")
    print("SELECTED_SAFE_STAGE1_LABEL_PROFILE")
    print(f"selected_profile={profile.name}")
    print(f"output_dir={out}")


if __name__ == "__main__":
    main()
