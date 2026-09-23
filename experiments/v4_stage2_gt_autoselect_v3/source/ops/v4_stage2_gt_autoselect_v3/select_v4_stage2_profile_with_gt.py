#!/usr/bin/env python3
"""Ground-truth-driven selector for V4 Stage-2 line candidate/refiner settings.

This is an OFFLINE experiment/selection tool. Ground-truth labels are used only
for scoring candidate settings. The selected Stage-2 run itself remains the
normal V4 runtime and does not consume GT labels as model features.

Primary objective
-----------------
Recover GT power-line voxels that the accepted V4 baseline misses on long,
line-like components that are either:
  * part of a parallel conductor bundle where a sibling lane is detected, or
  * geometrically bracketed by two detected poles in the session.

Guardrails
----------
Candidate profiles are rejected when they cause excessive line-voxel inflation
or materially reduce exact/near-GT precision on the target session or on a
deterministic sample of other labeled sessions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from v4_sparse_components import extract_sparse_components, pca_geometry, sparse_connected_labels
from v4_stage2_runtime import _physical_mask, load_bundle, local_X
from v4_stage_contracts import load_stage1_artifact, stage1_paths


@dataclass(frozen=True)
class CandidateSetting:
    name: str
    line_candidate_threshold: float
    line_weak_threshold: float
    line_competition_ratio: float


@dataclass(frozen=True)
class Profile:
    name: str
    line_candidate_threshold: float
    line_weak_threshold: float
    line_competition_ratio: float
    line_refiner_threshold: float


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_root")
    p.add_argument("--stage2_bundle")
    p.add_argument("--target_session", default="VELASCO_CUT_CP/session1")
    p.add_argument("--output_dir")
    p.add_argument("--grid_size", type=int, nargs=3, default=[400, 400, 200])
    p.add_argument("--voxel_size_ft", type=float, default=0.5)
    p.add_argument("--pole_candidate_threshold", type=float, default=0.15)
    p.add_argument("--pole_min_voxels", type=int, default=4)
    p.add_argument("--line_min_voxels", type=int, default=3)
    p.add_argument("--edge_width_vox", type=int, default=10)
    p.add_argument("--baseline_line_candidate_threshold", type=float, default=0.08)
    p.add_argument("--baseline_line_weak_threshold", type=float, default=0.04)
    p.add_argument("--baseline_line_competition_ratio", type=float, default=0.55)
    p.add_argument("--max_target_slices", type=int, default=0)
    p.add_argument("--guardrail_slices_per_session", type=int, default=3)
    p.add_argument("--top_target_profiles_for_guardrail", type=int, default=6)
    p.add_argument("--near_tolerance_vox", type=float, default=1.75)
    p.add_argument("--min_gt_component_voxels", type=int, default=6)
    p.add_argument("--min_gt_component_horizontal_ft", type=float, default=4.0)
    p.add_argument("--min_gt_component_linearity", type=float, default=0.60)
    p.add_argument("--max_gt_component_verticality", type=float, default=0.88)
    p.add_argument("--baseline_missed_component_recall", type=float, default=0.70)
    p.add_argument("--sibling_detected_recall", type=float, default=0.55)
    p.add_argument("--parallel_angle_deg", type=float, default=12.0)
    p.add_argument("--parallel_offset_ft", type=float, default=16.0)
    p.add_argument("--parallel_z_difference_ft", type=float, default=20.0)
    p.add_argument("--parallel_overlap_fraction", type=float, default=0.35)
    p.add_argument("--pole_corridor_ft", type=float, default=12.0)
    p.add_argument("--pole_z_tolerance_ft", type=float, default=30.0)
    p.add_argument("--pole_bracket_slack_ft", type=float, default=25.0)
    p.add_argument("--max_span_length_ft", type=float, default=450.0)
    p.add_argument("--max_target_near_precision_drop", type=float, default=0.05)
    p.add_argument("--max_guardrail_near_precision_drop", type=float, default=0.05)
    p.add_argument("--max_guardrail_exact_precision_drop", type=float, default=0.08)
    p.add_argument("--max_guardrail_recall_drop", type=float, default=0.01)
    p.add_argument("--max_accepted_voxel_inflation", type=float, default=3.0)
    p.add_argument("--min_target_recovered_voxels", type=int, default=10)
    p.add_argument("--min_target_recovery_fraction", type=float, default=0.01)
    p.add_argument("--self_test", action="store_true")
    return p.parse_args()


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    return v if np.isfinite(v) else default


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def linear_keys(coords: np.ndarray, grid_size: tuple[int, int, int]) -> np.ndarray:
    c = np.asarray(coords, dtype=np.int64)
    if c.size == 0:
        return np.empty(0, dtype=np.int64)
    gx, gy, _ = map(int, grid_size)
    return np.unique((c[:, 2] * gy + c[:, 1]) * gx + c[:, 0])


def count_intersection(a: np.ndarray, b: np.ndarray) -> int:
    if len(a) == 0 or len(b) == 0:
        return 0
    return int(np.intersect1d(a, b, assume_unique=True).size)


def union_component_keys(points: dict[str, np.ndarray], ids: Iterable[str], grid_size: tuple[int, int, int]) -> np.ndarray:
    arrays = [np.asarray(points[str(x)], dtype=np.int32) for x in ids if str(x) in points and len(points[str(x)])]
    if not arrays:
        return np.empty(0, dtype=np.int64)
    return linear_keys(np.concatenate(arrays, axis=0), grid_size)


def score_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f2 = 5.0 * precision * recall / max(4.0 * precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f2": float(f2),
        "iou": float(iou),
    }


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, float)
    bb = np.asarray(b, float)
    aa /= max(float(np.linalg.norm(aa)), 1e-12)
    bb /= max(float(np.linalg.norm(bb)), 1e-12)
    c = float(np.clip(abs(np.dot(aa, bb)), 0.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def select_evenly(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if n <= 0 or len(df) <= n:
        return df.copy()
    idx = np.unique(np.linspace(0, len(df) - 1, n, dtype=int))
    return df.iloc[idx].copy()


def read_manifest(path: Path, group_id: str | None = None, max_rows: int = 0) -> pd.DataFrame:
    d = pd.read_csv(path)
    required = {"group_id", "slice_seq", "relative_path", "stage1_npz", "stage1_meta_json", "status"}
    missing = sorted(required - set(d.columns))
    if missing:
        raise RuntimeError(f"Manifest missing columns {missing}: {path}")
    d = d[d["status"].astype(str).eq("completed")].copy()
    if group_id is not None:
        d = d[d["group_id"].astype(str).eq(group_id)].copy()
    d["slice_seq"] = pd.to_numeric(d["slice_seq"], errors="raise").astype(int)
    d = d.sort_values("slice_seq", kind="stable")
    if max_rows > 0:
        d = d.iloc[:max_rows].copy()
    return d


def discover_session_manifests(stage1_root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in sorted(stage1_root.rglob("stage1_manifest.csv")):
        try:
            d = pd.read_csv(p, usecols=["group_id", "status"])
        except Exception:
            continue
        d = d[d["status"].astype(str).eq("completed")]
        gids = sorted(set(d["group_id"].dropna().astype(str)))
        for gid in gids:
            if gid in out and out[gid] != p:
                raise RuntimeError(f"Duplicate Stage1 manifests for {gid}: {out[gid]} and {p}")
            out[gid] = p
    return out


def map_legacy_output_path(path: Path) -> Path:
    s = str(path)
    prefix = "/workspace/voxel_poleline/outputs"
    if s == prefix:
        return Path("/outputs")
    if s.startswith(prefix + "/"):
        return Path("/outputs" + s[len(prefix):])
    return path


def resolve_stage1_artifacts(row: Any, session_dir: Path) -> tuple[Path, Path]:
    npz = map_legacy_output_path(Path(str(row.stage1_npz)))
    meta = map_legacy_output_path(Path(str(row.stage1_meta_json)))
    if npz.is_file() and meta.is_file():
        return npz, meta
    fallback_npz, fallback_meta = stage1_paths(session_dir, str(row.relative_path))
    if fallback_npz.is_file() and fallback_meta.is_file():
        return fallback_npz, fallback_meta
    raise FileNotFoundError(
        f"Stage1 artifact not found. manifest_npz={row.stage1_npz!r} mapped_npz={npz} "
        f"fallback_npz={fallback_npz}"
    )


def component_acceptance(comps: dict[str, Any], bundle: dict[str, Any], class_name: str, threshold: float) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    key = "lines" if class_name == "line" else "poles"
    d = comps[key].copy()
    if d.empty:
        return d, np.zeros(0, dtype=float), []
    model = bundle[class_name + "_model"]
    prob = model.predict_proba(local_X(d, bundle["feature_columns"]))[:, 1]
    physical = _physical_mask(d, class_name)
    accepted = (prob >= float(threshold)) & physical
    d["refiner_probability"] = prob
    d["physical_ok"] = physical
    d["component_accept"] = accepted
    ids = d.loc[accepted, "component_id"].astype(str).tolist()
    return d, prob, ids


def near_counts(pred_coords: np.ndarray, gt_coords: np.ndarray, tolerance_vox: float) -> tuple[int, int, int]:
    pred = np.asarray(pred_coords, float)
    gt = np.asarray(gt_coords, float)
    if len(pred) == 0 and len(gt) == 0:
        return 0, 0, 0
    if len(pred) == 0:
        return 0, 0, int(len(gt))
    if len(gt) == 0:
        return 0, int(len(pred)), 0
    gt_tree = cKDTree(gt)
    pred_dist, _ = gt_tree.query(pred, k=1, workers=-1)
    near_pred = pred_dist <= float(tolerance_vox)
    pred_tp = int(np.sum(near_pred))
    pred_fp = int(len(pred) - pred_tp)
    pred_tree = cKDTree(pred)
    gt_dist, _ = pred_tree.query(gt, k=1, workers=-1)
    gt_hit = int(np.sum(gt_dist <= float(tolerance_vox)))
    gt_fn = int(len(gt) - gt_hit)
    # For near metrics, tp used for precision and recall cannot be one shared count
    # when many-to-one neighborhoods occur. We retain precision-side TP and store
    # recall separately elsewhere. Return precision-side counts plus recall FN.
    return pred_tp, pred_fp, gt_fn


def accepted_coords(points: dict[str, np.ndarray], ids: Iterable[str]) -> np.ndarray:
    arrays = [np.asarray(points[str(x)], dtype=np.int32) for x in ids if str(x) in points and len(points[str(x)])]
    if not arrays:
        return np.empty((0, 3), dtype=np.int32)
    p = np.concatenate(arrays, axis=0)
    return np.unique(p, axis=0)


def axis_xy_from_points(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, float)
    if len(p) < 2:
        return np.array([1.0, 0.0])
    xy = p[:, :2]
    c = np.median(xy, axis=0)
    try:
        _, _, vh = np.linalg.svd(xy - c, full_matrices=False)
        d = np.asarray(vh[0], float)
    except np.linalg.LinAlgError:
        d = np.array([1.0, 0.0])
    n = float(np.linalg.norm(d))
    return d / max(n, 1e-12)


def parallel_relation(a: dict[str, Any], b: dict[str, Any], args: argparse.Namespace) -> bool:
    d1 = np.asarray(a["axis_xy"], float)
    d2 = np.asarray(b["axis_xy"], float)
    if np.dot(d1, d2) < 0:
        d2 = -d2
    if angle_deg(d1, d2) > args.parallel_angle_deg:
        return False
    d = d1 + d2
    d /= max(float(np.linalg.norm(d)), 1e-12)
    perp = np.array([-d[1], d[0]])
    c1 = np.asarray(a["center_xy_ft"], float)
    c2 = np.asarray(b["center_xy_ft"], float)
    lateral = abs(float(np.dot(c2 - c1, perp)))
    if lateral > args.parallel_offset_ft:
        return False
    if abs(float(a["median_z_ft"]) - float(b["median_z_ft"])) > args.parallel_z_difference_ft:
        return False
    p1 = np.asarray(a["xy_ft"], float) @ d
    p2 = np.asarray(b["xy_ft"], float) @ d
    lo = max(float(np.min(p1)), float(np.min(p2)))
    hi = min(float(np.max(p1)), float(np.max(p2)))
    overlap = max(0.0, hi - lo)
    length = max(min(float(np.ptp(p1)), float(np.ptp(p2))), 1e-6)
    return overlap / length >= args.parallel_overlap_fraction


def dedupe_poles(poles: list[np.ndarray], radius_ft: float = 4.0) -> np.ndarray:
    if not poles:
        return np.empty((0, 3), dtype=float)
    kept: list[np.ndarray] = []
    for p in poles:
        q = np.asarray(p, float)
        if not np.isfinite(q).all():
            continue
        if not kept or min(float(np.linalg.norm(q[:2] - x[:2])) for x in kept) > radius_ft:
            kept.append(q)
    return np.vstack(kept) if kept else np.empty((0, 3), dtype=float)


def pole_bracketed(component: dict[str, Any], poles_world: np.ndarray, args: argparse.Namespace) -> bool:
    if len(poles_world) < 2:
        return False
    xy = np.asarray(component["world_xy_ft"], float)
    d = np.asarray(component["axis_xy"], float)
    c = np.median(xy, axis=0)
    s = (xy - c) @ d
    smin, smax = float(np.min(s)), float(np.max(s))
    perp = np.array([-d[1], d[0]])
    rel = poles_world[:, :2] - c
    t = rel @ d
    lateral = np.abs(rel @ perp)
    z_ok = np.abs(poles_world[:, 2] - float(component["world_median_z_ft"])) <= args.pole_z_tolerance_ft
    valid = (lateral <= args.pole_corridor_ft) & z_ok
    ids = np.flatnonzero(valid)
    if len(ids) < 2:
        return False
    for ii in range(len(ids)):
        for jj in range(ii + 1, len(ids)):
            a, b = float(t[ids[ii]]), float(t[ids[jj]])
            left, right = min(a, b), max(a, b)
            span = right - left
            if span < 10.0 or span > args.max_span_length_ft:
                continue
            if left <= smin + args.pole_bracket_slack_ft and right >= smax - args.pole_bracket_slack_ft:
                return True
    return False


def profile_candidates_from_evidence(seed_scores: list[float], voxel_scores: list[float], ratios: list[float]) -> list[CandidateSetting]:
    fixed = [
        CandidateSetting("baseline", 0.080, 0.040, 0.55),
        CandidateSetting("mild_1", 0.065, 0.030, 0.50),
        CandidateSetting("mild_2", 0.055, 0.025, 0.45),
        CandidateSetting("mid_1", 0.050, 0.015, 0.40),
        CandidateSetting("mid_2", 0.040, 0.015, 0.35),
        CandidateSetting("mid_3", 0.040, 0.010, 0.30),
        CandidateSetting("strong_1", 0.030, 0.010, 0.25),
        CandidateSetting("strong_2", 0.025, 0.005, 0.20),
        CandidateSetting("strong_3", 0.020, 0.005, 0.15),
        CandidateSetting("extreme", 0.010, 0.002, 0.08),
    ]

    def q(values: list[float], p: float, default: float) -> float:
        a = np.asarray([x for x in values if np.isfinite(x)], float)
        return float(np.quantile(a, p)) if len(a) else default

    empirical: list[CandidateSetting] = []
    if seed_scores and voxel_scores and ratios:
        levels = [
            ("empirical_mild", 0.75, 0.25, 0.50),
            ("empirical_mid", 0.50, 0.10, 0.25),
            ("empirical_strong", 0.25, 0.05, 0.10),
            ("empirical_extreme", 0.10, 0.01, 0.05),
        ]
        for name, ps, pv, pr in levels:
            candidate = float(np.clip(q(seed_scores, ps, 0.04), 0.005, 0.08))
            weak = float(np.clip(q(voxel_scores, pv, candidate * 0.4), 0.001, candidate))
            competition = float(np.clip(q(ratios, pr, 0.30), 0.03, 0.55))
            empirical.append(CandidateSetting(name, candidate, weak, competition))

    dedup: dict[tuple[float, float, float], CandidateSetting] = {}
    for s in fixed + empirical:
        c = round(float(s.line_candidate_threshold), 6)
        w = round(min(float(s.line_weak_threshold), c), 6)
        r = round(float(s.line_competition_ratio), 6)
        key = (c, w, r)
        if key not in dedup or s.name == "baseline":
            dedup[key] = CandidateSetting(s.name, c, w, r)
    return list(dedup.values())


def make_profiles(settings: list[CandidateSetting], base_refiner_threshold: float) -> list[Profile]:
    factors = [1.0, 0.80, 0.60, 0.40]
    profiles: list[Profile] = []
    seen: set[tuple[float, float, float, float]] = set()
    for s in settings:
        for factor in factors:
            rt = round(max(0.02, min(0.98, base_refiner_threshold * factor)), 6)
            key = (
                s.line_candidate_threshold,
                s.line_weak_threshold,
                s.line_competition_ratio,
                rt,
            )
            if key in seen:
                continue
            seen.add(key)
            suffix = "base_refiner" if abs(factor - 1.0) < 1e-12 else f"refiner_x{factor:.2f}"
            profiles.append(
                Profile(
                    f"{s.name}__{suffix}",
                    s.line_candidate_threshold,
                    s.line_weak_threshold,
                    s.line_competition_ratio,
                    rt,
                )
            )
    return profiles


def load_row_artifact(row: Any, session_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    npz, meta = resolve_stage1_artifacts(row, session_dir)
    return load_stage1_artifact(npz, meta)


def extract_profile_components(item: dict[str, Any], pred: dict[str, Any], p: Profile | CandidateSetting, args: argparse.Namespace) -> dict[str, Any]:
    return extract_sparse_components(
        item,
        pred,
        tuple(args.grid_size),
        args.voxel_size_ft,
        args.pole_candidate_threshold,
        p.line_candidate_threshold,
        p.line_weak_threshold,
        p.line_competition_ratio,
        args.pole_min_voxels,
        args.line_min_voxels,
        gt_points=None,
        edge_width_vox=args.edge_width_vox,
    )


def baseline_analysis(
    rows: pd.DataFrame,
    session_dir: Path,
    bundle: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[int, dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    baseline = Profile(
        "baseline__base_refiner",
        args.baseline_line_candidate_threshold,
        args.baseline_line_weak_threshold,
        args.baseline_line_competition_ratio,
        float(bundle["line_threshold"]),
    )
    grid = tuple(args.grid_size)
    by_seq: dict[int, dict[str, Any]] = {}
    component_rows: list[dict[str, Any]] = []
    pole_world: list[np.ndarray] = []
    total_gt = total_pred = total_tp = 0
    total_candidate_tp = 0
    exact_fp = 0
    near_pred_tp = near_pred_fp = near_gt_fn = 0
    labeled_slices = 0

    for row in rows.itertuples(index=False):
        item, pred, _ = load_row_artifact(row, session_dir)
        labels = np.asarray(item.get("raw_labels", np.zeros(len(item["coords"]))), dtype=np.int16)
        coords = np.asarray(item["coords"], dtype=np.int32)
        gt_idx = np.flatnonzero(labels == 2)
        if len(gt_idx) == 0:
            continue
        labeled_slices += 1
        comps = extract_profile_components(item, pred, baseline, args)
        line_df, _, line_ids = component_acceptance(comps, bundle, "line", baseline.line_refiner_threshold)
        pole_df, _, pole_ids = component_acceptance(comps, bundle, "pole", float(bundle["pole_threshold"]))
        candidate_keys = union_component_keys(comps["line_points"], comps["line_points"].keys(), grid)
        accepted_line_coords = accepted_coords(comps["line_points"], line_ids)
        accepted_keys = linear_keys(accepted_line_coords, grid)
        gt_coords = coords[gt_idx]
        gt_keys = linear_keys(gt_coords, grid)
        tp = count_intersection(gt_keys, accepted_keys)
        ctp = count_intersection(gt_keys, candidate_keys)
        total_gt += len(gt_keys)
        total_pred += len(accepted_keys)
        total_tp += tp
        total_candidate_tp += ctp
        exact_fp += len(accepted_keys) - tp
        n_tp, n_fp, n_fn = near_counts(accepted_line_coords, gt_coords, args.near_tolerance_vox)
        near_pred_tp += n_tp
        near_pred_fp += n_fp
        near_gt_fn += n_fn

        center = np.array(
            [
                safe_float(getattr(row, "center_x", 0.0), 0.0),
                safe_float(getattr(row, "center_y", 0.0), 0.0),
                safe_float(getattr(row, "center_z", 0.0), 0.0),
            ],
            dtype=float,
        )
        for pid in pole_ids:
            pts = np.asarray(comps["pole_points"].get(pid, np.empty((0, 3))), float)
            if len(pts) == 0:
                continue
            xy = np.median(pts[:, :2], axis=0)
            z = float(np.quantile(pts[:, 2], 0.85))
            pole_world.append((np.array([xy[0], xy[1], z]) + center) * args.voxel_size_ft)

        glabels, ncomp = sparse_connected_labels(gt_coords, grid)
        slice_components: list[dict[str, Any]] = []
        for cid in range(1, ncomp + 1):
            local_idx = np.flatnonzero(glabels == cid)
            pts = gt_coords[local_idx]
            if len(pts) < args.min_gt_component_voxels:
                continue
            geo = pca_geometry(pts, args.voxel_size_ft)
            axis = axis_xy_from_points(pts)
            xy_ft = pts[:, :2].astype(float) * args.voxel_size_ft
            horiz = float(np.linalg.norm(np.ptp(xy_ft, axis=0)))
            verticality = abs(float(np.asarray(geo["principal"], float)[2]))
            linearity = float(geo["linearity"])
            eligible = (
                horiz >= args.min_gt_component_horizontal_ft
                and linearity >= args.min_gt_component_linearity
                and verticality <= args.max_gt_component_verticality
            )
            comp_keys = linear_keys(pts, grid)
            base_tp = count_intersection(comp_keys, accepted_keys)
            cand_tp = count_intersection(comp_keys, candidate_keys)
            base_recall = base_tp / max(len(comp_keys), 1)
            cand_recall = cand_tp / max(len(comp_keys), 1)
            source_indices = gt_idx[local_idx]
            ls = np.asarray(pred["line"], float)[source_indices]
            ps = np.asarray(pred["pole"], float)[source_indices]
            ratio = ls / np.maximum(ps, 1e-6)
            world = (pts.astype(float) + center) * args.voxel_size_ft
            rec = {
                "slice_seq": int(row.slice_seq),
                "component_id": f"GT{cid:05d}",
                "n_voxels": int(len(pts)),
                "horizontal_span_ft": horiz,
                "linearity": linearity,
                "verticality": verticality,
                "baseline_candidate_recall": cand_recall,
                "baseline_accepted_recall": base_recall,
                "score_min": float(np.min(ls)),
                "score_p10": float(np.quantile(ls, 0.10)),
                "score_p50": float(np.quantile(ls, 0.50)),
                "score_p90": float(np.quantile(ls, 0.90)),
                "score_max": float(np.max(ls)),
                "competition_ratio_p10": float(np.quantile(ratio, 0.10)),
                "competition_ratio_p50": float(np.quantile(ratio, 0.50)),
                "eligible": bool(eligible),
                "axis_xy": axis,
                "center_xy_ft": np.median(xy_ft, axis=0),
                "xy_ft": xy_ft,
                "median_z_ft": float(np.median(pts[:, 2]) * args.voxel_size_ft),
                "world_xy_ft": world[:, :2],
                "world_median_z_ft": float(np.median(world[:, 2])),
                "keys": comp_keys,
                "points": pts,
                "line_scores": ls,
                "competition_ratios": ratio,
                "parallel_siblings": [],
                "sibling_detected": False,
                "pole_bracketed": False,
                "priority_target": False,
            }
            slice_components.append(rec)

        for i in range(len(slice_components)):
            for j in range(i + 1, len(slice_components)):
                a, b = slice_components[i], slice_components[j]
                if not (a["eligible"] and b["eligible"]):
                    continue
                if parallel_relation(a, b, args):
                    a["parallel_siblings"].append(j)
                    b["parallel_siblings"].append(i)
        for i, rec in enumerate(slice_components):
            rec["sibling_detected"] = any(
                slice_components[j]["baseline_accepted_recall"] >= args.sibling_detected_recall
                for j in rec["parallel_siblings"]
            )

        by_seq[int(row.slice_seq)] = {
            "row": row,
            "gt_keys": gt_keys,
            "gt_coords": gt_coords,
            "baseline_candidate_keys": candidate_keys,
            "baseline_accepted_keys": accepted_keys,
            "components": slice_components,
        }

    if labeled_slices == 0:
        raise RuntimeError(
            f"Target session {args.target_session} has no raw_labels==2 voxels in saved Stage1 artifacts"
        )

    poles = dedupe_poles(pole_world)
    priority_count = bundle_count = bracket_count = 0
    fallback_count = 0
    target_seed_scores: list[float] = []
    target_voxel_scores: list[float] = []
    target_ratios: list[float] = []

    for seq, sd in by_seq.items():
        priority_arrays: list[np.ndarray] = []
        bundle_arrays: list[np.ndarray] = []
        bracket_arrays: list[np.ndarray] = []
        fallback_arrays: list[np.ndarray] = []
        base = sd["baseline_accepted_keys"]
        for rec in sd["components"]:
            rec["pole_bracketed"] = bool(pole_bracketed(rec, poles, args)) if rec["eligible"] else False
            missed = rec["baseline_accepted_recall"] < args.baseline_missed_component_recall
            rec["priority_target"] = bool(rec["eligible"] and missed and (rec["sibling_detected"] or rec["pole_bracketed"]))
            missing_keys = np.setdiff1d(rec["keys"], base, assume_unique=True)
            if rec["eligible"] and missed and len(missing_keys):
                fallback_arrays.append(missing_keys)
                fallback_count += 1
            if rec["priority_target"] and len(missing_keys):
                priority_arrays.append(missing_keys)
                priority_count += 1
                target_seed_scores.append(rec["score_max"])
                # Scores correspond to all component voxels; use values from GT voxels
                # not accepted by baseline when possible.
                pts_for_scores = np.asarray(rec["points"], dtype=np.int64)
                gx, gy, _ = map(int, args.grid_size)
                ordered_keys = (pts_for_scores[:, 2] * gy + pts_for_scores[:, 1]) * gx + pts_for_scores[:, 0]
                missing_mask = ~np.isin(ordered_keys, base)
                target_voxel_scores.extend(np.asarray(rec["line_scores"])[missing_mask].tolist())
                target_ratios.extend(np.asarray(rec["competition_ratios"])[missing_mask].tolist())
            if rec["eligible"] and missed and rec["sibling_detected"] and len(missing_keys):
                bundle_arrays.append(missing_keys)
                bundle_count += 1
            if rec["eligible"] and missed and rec["pole_bracketed"] and len(missing_keys):
                bracket_arrays.append(missing_keys)
                bracket_count += 1

            component_rows.append(
                {
                    k: v
                    for k, v in rec.items()
                    if k
                    not in {
                        "axis_xy",
                        "center_xy_ft",
                        "xy_ft",
                        "world_xy_ft",
                        "keys",
                        "points",
                        "line_scores",
                        "competition_ratios",
                        "parallel_siblings",
                    }
                }
                | {"parallel_sibling_count": len(rec["parallel_siblings"])}
            )

        def cat(arrays: list[np.ndarray]) -> np.ndarray:
            return np.unique(np.concatenate(arrays)) if arrays else np.empty(0, dtype=np.int64)

        sd["priority_missed_keys"] = cat(priority_arrays)
        sd["bundle_missed_keys"] = cat(bundle_arrays)
        sd["bracketed_missed_keys"] = cat(bracket_arrays)
        sd["fallback_missed_keys"] = cat(fallback_arrays)

    priority_voxels = sum(len(x["priority_missed_keys"]) for x in by_seq.values())
    target_mode = "parallel_or_pole_bracketed"
    if priority_voxels == 0:
        target_mode = "fallback_long_line_components"
        for sd in by_seq.values():
            sd["priority_missed_keys"] = sd["fallback_missed_keys"]
        priority_voxels = sum(len(x["priority_missed_keys"]) for x in by_seq.values())
        target_seed_scores = [r["score_max"] for r in component_rows if r["eligible"] and r["baseline_accepted_recall"] < args.baseline_missed_component_recall]
        target_voxel_scores = []
        target_ratios = []
        for sd in by_seq.values():
            for rec in sd["components"]:
                if rec["eligible"] and rec["baseline_accepted_recall"] < args.baseline_missed_component_recall:
                    target_voxel_scores.extend(np.asarray(rec["line_scores"]).tolist())
                    target_ratios.extend(np.asarray(rec["competition_ratios"]).tolist())

    if priority_voxels == 0:
        raise RuntimeError("No baseline-missed eligible GT line voxels were found for profile selection")

    exact = score_metrics(total_tp, exact_fp, total_gt - total_tp)
    near_precision = near_pred_tp / max(near_pred_tp + near_pred_fp, 1)
    near_recall = (total_gt - near_gt_fn) / max(total_gt, 1)
    near_f2 = 5 * near_precision * near_recall / max(4 * near_precision + near_recall, 1e-12)
    summary = {
        "target_session": args.target_session,
        "target_mode": target_mode,
        "labeled_slices": labeled_slices,
        "detected_session_poles": int(len(poles)),
        "priority_target_components": int(priority_count),
        "bundle_target_components": int(bundle_count),
        "pole_bracketed_target_components": int(bracket_count),
        "fallback_target_components": int(fallback_count),
        "priority_missed_voxels": int(priority_voxels),
        "baseline_candidate_target_voxels": int(sum(count_intersection(sd["priority_missed_keys"], sd["baseline_candidate_keys"]) for sd in by_seq.values())),
        "baseline_accepted_voxels": int(total_pred),
        "baseline_candidate_gt_tp": int(total_candidate_tp),
        "baseline_exact": exact,
        "baseline_near_precision": float(near_precision),
        "baseline_near_recall": float(near_recall),
        "baseline_near_f2": float(near_f2),
        "target_seed_scores": target_seed_scores,
        "target_voxel_scores": target_voxel_scores,
        "target_competition_ratios": target_ratios,
    }
    return by_seq, pd.DataFrame(component_rows), summary


def evaluate_profile(
    profile: Profile,
    rows: pd.DataFrame,
    session_dir: Path,
    bundle: dict[str, Any],
    args: argparse.Namespace,
    target_by_seq: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    grid = tuple(args.grid_size)
    gt_total = pred_total = exact_tp = 0
    candidate_gt_tp = 0
    near_pred_tp = near_pred_fp = near_gt_fn = 0
    target_total = target_recovered = candidate_target_recovered = 0
    bundle_total = bundle_recovered = 0
    bracket_total = bracket_recovered = 0
    target_components = target_components_recovered = 0
    labeled_slices = 0

    for row in rows.itertuples(index=False):
        item, pred, _ = load_row_artifact(row, session_dir)
        coords = np.asarray(item["coords"], dtype=np.int32)
        labels = np.asarray(item.get("raw_labels", np.zeros(len(coords))), dtype=np.int16)
        gt_coords = coords[labels == 2]
        if len(gt_coords) == 0:
            continue
        labeled_slices += 1
        gt_keys = linear_keys(gt_coords, grid)
        comps = extract_profile_components(item, pred, profile, args)
        line_df, _, line_ids = component_acceptance(comps, bundle, "line", profile.line_refiner_threshold)
        candidate_keys = union_component_keys(comps["line_points"], comps["line_points"].keys(), grid)
        pred_coords = accepted_coords(comps["line_points"], line_ids)
        pred_keys = linear_keys(pred_coords, grid)
        tp = count_intersection(gt_keys, pred_keys)
        exact_tp += tp
        gt_total += len(gt_keys)
        pred_total += len(pred_keys)
        candidate_gt_tp += count_intersection(gt_keys, candidate_keys)
        n_tp, n_fp, n_fn = near_counts(pred_coords, gt_coords, args.near_tolerance_vox)
        near_pred_tp += n_tp
        near_pred_fp += n_fp
        near_gt_fn += n_fn

        if target_by_seq is not None and int(row.slice_seq) in target_by_seq:
            td = target_by_seq[int(row.slice_seq)]
            for name, total_name, recovered_name in [
                ("priority_missed_keys", "target", "target"),
                ("bundle_missed_keys", "bundle", "bundle"),
                ("bracketed_missed_keys", "bracket", "bracket"),
            ]:
                keys = td[name]
                recovered = count_intersection(keys, pred_keys)
                if total_name == "target":
                    target_total += len(keys)
                    target_recovered += recovered
                    candidate_target_recovered += count_intersection(keys, candidate_keys)
                elif total_name == "bundle":
                    bundle_total += len(keys)
                    bundle_recovered += recovered
                else:
                    bracket_total += len(keys)
                    bracket_recovered += recovered
            for rec in td["components"]:
                if not rec["priority_target"] and target_by_seq is not None:
                    # Fallback mode components are represented by priority_missed_keys;
                    # count only explicitly priority-tagged components here.
                    continue
                target_components += 1
                if count_intersection(rec["keys"], pred_keys) / max(len(rec["keys"]), 1) >= args.baseline_missed_component_recall:
                    target_components_recovered += 1

    if labeled_slices == 0:
        return {
            "profile": profile.name,
            "labeled_slices": 0,
            "line_candidate_threshold": profile.line_candidate_threshold,
            "line_weak_threshold": profile.line_weak_threshold,
            "line_competition_ratio": profile.line_competition_ratio,
            "line_refiner_threshold": profile.line_refiner_threshold,
        }

    exact_fp = pred_total - exact_tp
    exact = score_metrics(exact_tp, exact_fp, gt_total - exact_tp)
    near_precision = near_pred_tp / max(near_pred_tp + near_pred_fp, 1)
    near_recall = (gt_total - near_gt_fn) / max(gt_total, 1)
    near_f2 = 5 * near_precision * near_recall / max(4 * near_precision + near_recall, 1e-12)
    return {
        "profile": profile.name,
        "line_candidate_threshold": profile.line_candidate_threshold,
        "line_weak_threshold": profile.line_weak_threshold,
        "line_competition_ratio": profile.line_competition_ratio,
        "line_refiner_threshold": profile.line_refiner_threshold,
        "labeled_slices": labeled_slices,
        "accepted_voxels": int(pred_total),
        "candidate_gt_tp": int(candidate_gt_tp),
        "exact_tp": exact["tp"],
        "exact_fp": exact["fp"],
        "exact_fn": exact["fn"],
        "exact_precision": exact["precision"],
        "exact_recall": exact["recall"],
        "exact_f2": exact["f2"],
        "exact_iou": exact["iou"],
        "near_precision": float(near_precision),
        "near_recall": float(near_recall),
        "near_f2": float(near_f2),
        "near_pred_tp_count": int(near_pred_tp),
        "near_pred_fp_count": int(near_pred_fp),
        "near_gt_hit_count": int(gt_total - near_gt_fn),
        "gt_line_voxels": int(gt_total),
        "target_missed_voxels": int(target_total),
        "target_recovered_voxels": int(target_recovered),
        "target_recovery_fraction": float(target_recovered / max(target_total, 1)),
        "candidate_target_recovered_voxels": int(candidate_target_recovered),
        "bundle_missed_voxels": int(bundle_total),
        "bundle_recovered_voxels": int(bundle_recovered),
        "bracketed_missed_voxels": int(bracket_total),
        "bracketed_recovered_voxels": int(bracket_recovered),
        "target_components": int(target_components),
        "target_components_recovered": int(target_components_recovered),
    }


def build_guardrail_rows(manifests: dict[str, Path], target_gid: str, slices_per_session: int) -> list[tuple[pd.DataFrame, Path]]:
    out: list[tuple[pd.DataFrame, Path]] = []
    for gid, path in sorted(manifests.items()):
        if gid == target_gid:
            continue
        d = read_manifest(path, gid)
        if d.empty:
            continue
        d = select_evenly(d, slices_per_session)
        out.append((d, path.parent))
    return out


def evaluate_guardrail(profile: Profile, row_groups: list[tuple[pd.DataFrame, Path]], bundle: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    accum: dict[str, float] = {
        "labeled_slices": 0,
        "accepted_voxels": 0,
        "candidate_gt_tp": 0,
        "exact_tp": 0,
        "exact_fp": 0,
        "exact_fn": 0,
    }
    near_pred_tp = near_pred_fp = near_gt_fn = gt_total = 0
    for rows, session_dir in row_groups:
        r = evaluate_profile(profile, rows, session_dir, bundle, args, None)
        if int(r.get("labeled_slices", 0)) == 0:
            continue
        for k in accum:
            accum[k] += float(r.get(k, 0))
        near_pred_tp += int(r.get("near_pred_tp_count", 0))
        near_pred_fp += int(r.get("near_pred_fp_count", 0))
        slice_gt = int(r.get("gt_line_voxels", r["exact_tp"] + r["exact_fn"]))
        gt_total += slice_gt
        near_hit = int(r.get("near_gt_hit_count", 0))
        near_gt_fn += max(0, slice_gt - near_hit)
    exact = score_metrics(int(accum["exact_tp"]), int(accum["exact_fp"]), int(accum["exact_fn"]))
    near_precision = near_pred_tp / max(near_pred_tp + near_pred_fp, 1)
    near_recall = (gt_total - near_gt_fn) / max(gt_total, 1)
    near_f2 = 5 * near_precision * near_recall / max(4 * near_precision + near_recall, 1e-12)
    return {
        "profile": profile.name,
        "guardrail_labeled_slices": int(accum["labeled_slices"]),
        "guardrail_accepted_voxels": int(accum["accepted_voxels"]),
        "guardrail_candidate_gt_tp": int(accum["candidate_gt_tp"]),
        "guardrail_exact_tp": exact["tp"],
        "guardrail_exact_fp": exact["fp"],
        "guardrail_exact_fn": exact["fn"],
        "guardrail_exact_precision": exact["precision"],
        "guardrail_exact_recall": exact["recall"],
        "guardrail_exact_f2": exact["f2"],
        "guardrail_exact_iou": exact["iou"],
        "guardrail_near_precision": float(near_precision),
        "guardrail_near_recall": float(near_recall),
        "guardrail_near_f2": float(near_f2),
    }



def selected_component_recovery(
    profile: Profile,
    rows: pd.DataFrame,
    session_dir: Path,
    bundle: dict[str, Any],
    args: argparse.Namespace,
    target_by_seq: dict[int, dict[str, Any]],
) -> pd.DataFrame:
    grid = tuple(args.grid_size)
    out_rows: list[dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        seq = int(row.slice_seq)
        if seq not in target_by_seq:
            continue
        item, pred, _ = load_row_artifact(row, session_dir)
        comps = extract_profile_components(item, pred, profile, args)
        _, _, line_ids = component_acceptance(comps, bundle, "line", profile.line_refiner_threshold)
        candidate_keys = union_component_keys(comps["line_points"], comps["line_points"].keys(), grid)
        pred_keys = linear_keys(accepted_coords(comps["line_points"], line_ids), grid)
        td = target_by_seq[seq]
        priority_keys = td["priority_missed_keys"]
        for rec in td["components"]:
            missing_before = np.setdiff1d(rec["keys"], td["baseline_accepted_keys"], assume_unique=True)
            targeted_missing = count_intersection(missing_before, priority_keys)
            if targeted_missing == 0:
                continue
            selected_tp = count_intersection(rec["keys"], pred_keys)
            selected_candidate_tp = count_intersection(rec["keys"], candidate_keys)
            newly_recovered = count_intersection(missing_before, pred_keys)
            out_rows.append({
                "slice_seq": seq,
                "component_id": rec["component_id"],
                "n_voxels": rec["n_voxels"],
                "horizontal_span_ft": rec["horizontal_span_ft"],
                "linearity": rec["linearity"],
                "verticality": rec["verticality"],
                "sibling_detected": rec["sibling_detected"],
                "pole_bracketed": rec["pole_bracketed"],
                "baseline_candidate_recall": rec["baseline_candidate_recall"],
                "baseline_accepted_recall": rec["baseline_accepted_recall"],
                "selected_candidate_recall": selected_candidate_tp / max(len(rec["keys"]), 1),
                "selected_accepted_recall": selected_tp / max(len(rec["keys"]), 1),
                "baseline_missed_voxels": len(missing_before),
                "targeted_missed_voxels": targeted_missing,
                "newly_recovered_voxels": newly_recovered,
                "remaining_missed_voxels": max(0, len(missing_before) - newly_recovered),
            })
    return pd.DataFrame(out_rows)

def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def write_env(values: dict[str, Any], path: Path) -> None:
    lines = []
    for k, v in values.items():
        lines.append(f"{k}={shlex.quote(str(v))}")
    path.write_text("\n".join(lines) + "\n")


def self_test() -> None:
    class A:
        parallel_angle_deg = 12.0
        parallel_offset_ft = 16.0
        parallel_z_difference_ft = 20.0
        parallel_overlap_fraction = 0.35
        pole_corridor_ft = 12.0
        pole_z_tolerance_ft = 30.0
        pole_bracket_slack_ft = 25.0
        max_span_length_ft = 450.0

    a = {
        "axis_xy": np.array([1.0, 0.0]),
        "center_xy_ft": np.array([10.0, 0.0]),
        "xy_ft": np.array([[0.0, 0.0], [20.0, 0.0]]),
        "median_z_ft": 30.0,
    }
    b = {
        "axis_xy": np.array([1.0, 0.01]),
        "center_xy_ft": np.array([10.0, 5.0]),
        "xy_ft": np.array([[0.0, 5.0], [20.0, 5.0]]),
        "median_z_ft": 34.0,
    }
    assert parallel_relation(a, b, A())
    component = {
        "axis_xy": np.array([1.0, 0.0]),
        "world_xy_ft": np.array([[20.0, 0.0], [80.0, 0.0]]),
        "world_median_z_ft": 30.0,
    }
    poles = np.array([[0.0, 0.0, 25.0], [100.0, 0.0, 26.0]])
    assert pole_bracketed(component, poles, A())
    s = profile_candidates_from_evidence([0.04, 0.06], [0.005, 0.02], [0.15, 0.4])
    assert any(x.name == "baseline" for x in s)
    print("V4_STAGE2_GT_SELECTOR_SELF_TEST_OK")


def main() -> None:
    args = cli()
    if args.self_test:
        self_test()
        return

    missing_cli = [name for name in ("stage1_root", "stage2_bundle", "output_dir") if not getattr(args, name)]
    if missing_cli:
        raise SystemExit("Missing required arguments: " + ", ".join("--" + x for x in missing_cli))
    stage1_root = Path(args.stage1_root).resolve()
    bundle_path = Path(args.stage2_bundle).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not stage1_root.is_dir():
        raise FileNotFoundError(stage1_root)
    if not bundle_path.is_file():
        raise FileNotFoundError(bundle_path)

    manifests = discover_session_manifests(stage1_root)
    if args.target_session not in manifests:
        raise RuntimeError(
            f"Target session {args.target_session!r} not found. Available sessions: {sorted(manifests)}"
        )
    target_manifest = manifests[args.target_session]
    target_rows = read_manifest(target_manifest, args.target_session, args.max_target_slices)
    if target_rows.empty:
        raise RuntimeError(f"No completed Stage1 rows for {args.target_session}")
    bundle = load_bundle(bundle_path)
    base_refiner = float(bundle["line_threshold"])

    print(f"[selector] target={args.target_session} slices={len(target_rows)}")
    print(f"[selector] bundle_line_threshold={base_refiner:.6f}")
    target_by_seq, component_df, baseline_summary = baseline_analysis(
        target_rows, target_manifest.parent, bundle, args
    )
    component_df.to_csv(out / "baseline_gt_line_components.csv", index=False)
    write_json(
        {k: v for k, v in baseline_summary.items() if not isinstance(v, list)},
        out / "baseline_summary.json",
    )
    evidence = {
        "target_seed_score_quantiles": {},
        "target_voxel_score_quantiles": {},
        "target_competition_ratio_quantiles": {},
    }
    for key, values in [
        ("target_seed_score_quantiles", baseline_summary["target_seed_scores"]),
        ("target_voxel_score_quantiles", baseline_summary["target_voxel_scores"]),
        ("target_competition_ratio_quantiles", baseline_summary["target_competition_ratios"]),
    ]:
        a = np.asarray([x for x in values if np.isfinite(x)], float)
        evidence[key] = (
            {str(q): float(np.quantile(a, q)) for q in [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]}
            if len(a)
            else {}
        )
    write_json(evidence, out / "target_missed_voxel_score_distribution.json")

    settings = profile_candidates_from_evidence(
        baseline_summary["target_seed_scores"],
        baseline_summary["target_voxel_scores"],
        baseline_summary["target_competition_ratios"],
    )
    profiles = make_profiles(settings, base_refiner)
    pd.DataFrame([p.__dict__ for p in profiles]).to_csv(out / "profiles_evaluated.csv", index=False)
    print(f"[selector] profiles={len(profiles)} priority_missed_voxels={baseline_summary['priority_missed_voxels']}")

    target_results: list[dict[str, Any]] = []
    for i, profile in enumerate(profiles, 1):
        print(f"[selector-target] {i}/{len(profiles)} {profile.name}", flush=True)
        target_results.append(
            evaluate_profile(profile, target_rows, target_manifest.parent, bundle, args, target_by_seq)
        )
    target_df = pd.DataFrame(target_results)
    baseline_name = "baseline__base_refiner"
    if baseline_name not in set(target_df["profile"]):
        raise RuntimeError("Baseline profile missing from search")
    base_target = target_df[target_df["profile"].eq(baseline_name)].iloc[0]
    target_df["accepted_voxel_inflation"] = target_df["accepted_voxels"] / max(float(base_target["accepted_voxels"]), 1.0)
    target_df["target_near_precision_ok"] = target_df["near_precision"] >= max(
        0.0, float(base_target["near_precision"]) - args.max_target_near_precision_drop
    )
    target_df["target_inflation_ok"] = target_df["accepted_voxel_inflation"] <= args.max_accepted_voxel_inflation
    target_df["target_pre_safe"] = target_df["target_near_precision_ok"] & target_df["target_inflation_ok"]
    target_df = target_df.sort_values(
        [
            "target_pre_safe",
            "target_recovered_voxels",
            "bracketed_recovered_voxels",
            "bundle_recovered_voxels",
            "target_components_recovered",
            "near_f2",
            "exact_f2",
            "accepted_voxel_inflation",
        ],
        ascending=[False, False, False, False, False, False, False, True],
        kind="stable",
    )
    target_df.to_csv(out / "profile_search_target.csv", index=False)

    shortlist_names = target_df.head(args.top_target_profiles_for_guardrail)["profile"].astype(str).tolist()
    if baseline_name not in shortlist_names:
        shortlist_names.append(baseline_name)
    profile_map = {p.name: p for p in profiles}
    guardrail_groups = build_guardrail_rows(
        manifests, args.target_session, args.guardrail_slices_per_session
    )
    guardrail_results = []
    for i, name in enumerate(shortlist_names, 1):
        print(f"[selector-guardrail] {i}/{len(shortlist_names)} {name}", flush=True)
        guardrail_results.append(evaluate_guardrail(profile_map[name], guardrail_groups, bundle, args))
    guard_df = pd.DataFrame(guardrail_results)
    guard_df.to_csv(out / "profile_search_guardrail.csv", index=False)
    combined = target_df.merge(guard_df, on="profile", how="left")
    base_guard = combined[combined["profile"].eq(baseline_name)].iloc[0]

    combined["guardrail_precision_ok"] = combined["guardrail_near_precision"].fillna(base_guard["guardrail_near_precision"]) >= max(
        0.0, float(base_guard["guardrail_near_precision"]) - args.max_guardrail_near_precision_drop
    )
    combined["guardrail_exact_precision_ok"] = combined["guardrail_exact_precision"].fillna(base_guard["guardrail_exact_precision"]) >= max(
        0.0, float(base_guard["guardrail_exact_precision"]) - args.max_guardrail_exact_precision_drop
    )
    combined["guardrail_recall_ok"] = combined["guardrail_exact_recall"].fillna(base_guard["guardrail_exact_recall"]) >= max(
        0.0, float(base_guard["guardrail_exact_recall"]) - args.max_guardrail_recall_drop
    )
    combined["guardrail_evaluated"] = combined["profile"].isin(shortlist_names)
    combined["safe"] = (
        combined["guardrail_evaluated"]
        & combined["target_pre_safe"]
        & combined["guardrail_precision_ok"]
        & combined["guardrail_exact_precision_ok"]
        & combined["guardrail_recall_ok"]
    )
    target_total = int(baseline_summary["priority_missed_voxels"])
    min_gain = max(args.min_target_recovered_voxels, int(math.ceil(target_total * args.min_target_recovery_fraction)))
    combined["improves_target"] = combined["target_recovered_voxels"] >= min_gain
    combined["selectable"] = combined["safe"] & combined["improves_target"]
    combined = combined.sort_values(
        [
            "selectable",
            "target_recovered_voxels",
            "bracketed_recovered_voxels",
            "bundle_recovered_voxels",
            "target_components_recovered",
            "guardrail_near_f2",
            "guardrail_exact_f2",
            "near_precision",
            "accepted_voxel_inflation",
        ],
        ascending=[False, False, False, False, False, False, False, False, True],
        kind="stable",
    )
    combined.to_csv(out / "profile_search_combined.csv", index=False)

    selectable = combined[combined["selectable"]]
    if selectable.empty:
        selected_row = combined[combined["profile"].eq(baseline_name)].iloc[0]
        status = "NO_SAFE_GT_IMPROVEMENT"
        selected = False
    else:
        selected_row = selectable.iloc[0]
        status = "SELECTED_SAFE_GT_IMPROVEMENT"
        selected = True

    selected_profile = profile_map[str(selected_row["profile"])]
    detailed_recovery = selected_component_recovery(
        selected_profile, target_rows, target_manifest.parent, bundle, args, target_by_seq
    )
    detailed_recovery.to_csv(out / "selected_target_component_recovery.csv", index=False)
    candidate_recovery = int(selected_row.get("candidate_target_recovered_voxels", 0))
    accepted_recovery = int(selected_row.get("target_recovered_voxels", 0))
    if accepted_recovery > 0:
        bottleneck = "resolved_by_selected_profile"
    elif candidate_recovery > 0:
        bottleneck = "component_refiner_or_physical_gate"
    else:
        bottleneck = "candidate_generation_or_stage1_evidence"

    decision = {
        "selection_status": status,
        "selected": selected,
        "selection_basis": "GT voxels missed by baseline on parallel-bundle or pole-bracketed line components, with target and cross-session precision guardrails",
        "target_session": args.target_session,
        "target_mode": baseline_summary["target_mode"],
        "priority_missed_voxels": target_total,
        "minimum_required_recovered_voxels": int(min_gain),
        "diagnosed_bottleneck": bottleneck,
        "selected_profile": selected_profile.__dict__,
        "selected_metrics": {k: (v.item() if hasattr(v, "item") else v) for k, v in selected_row.to_dict().items()},
        "baseline_profile": profile_map[baseline_name].__dict__,
        "baseline_metrics": {k: (v.item() if hasattr(v, "item") else v) for k, v in combined[combined["profile"].eq(baseline_name)].iloc[0].to_dict().items()},
        "bundle_path": str(bundle_path),
        "bundle_sha256": sha256_file(bundle_path),
        "bundle_line_refiner_threshold": base_refiner,
        "ground_truth_source": "raw_labels array embedded in saved Stage1 NPZ; label==2",
        "ground_truth_not_used_by_runtime_features": True,
    }
    write_json(decision, out / "selected_profile.json")
    write_env(
        {
            "SELECTION_STATUS": status,
            "SELECTION_SELECTED": 1 if selected else 0,
            "SELECTED_PROFILE_NAME": selected_profile.name,
            "LINE_CANDIDATE_THRESHOLD": selected_profile.line_candidate_threshold,
            "LINE_WEAK_THRESHOLD": selected_profile.line_weak_threshold,
            "LINE_COMPETITION_RATIO": selected_profile.line_competition_ratio,
            "LINE_REFINER_THRESHOLD": selected_profile.line_refiner_threshold,
            "BASELINE_LINE_REFINER_THRESHOLD": base_refiner,
            "TARGET_SESSION": args.target_session,
        },
        out / "selected_profile.env",
    )
    report_lines = [
        "V4 Stage2 GT-driven profile selection",
        "=" * 72,
        f"status: {status}",
        f"target_session: {args.target_session}",
        f"target_mode: {baseline_summary['target_mode']}",
        f"priority_missed_voxels: {target_total}",
        f"minimum_required_recovered_voxels: {min_gain}",
        f"selected_profile: {selected_profile.name}",
        f"line_candidate_threshold: {selected_profile.line_candidate_threshold}",
        f"line_weak_threshold: {selected_profile.line_weak_threshold}",
        f"line_competition_ratio: {selected_profile.line_competition_ratio}",
        f"line_refiner_threshold: {selected_profile.line_refiner_threshold}",
        f"target_recovered_voxels: {accepted_recovery}",
        f"candidate_target_recovered_voxels: {candidate_recovery}",
        f"diagnosed_bottleneck: {bottleneck}",
        f"target_near_precision: {safe_float(selected_row.get('near_precision')):.6f}",
        f"guardrail_near_precision: {safe_float(selected_row.get('guardrail_near_precision')):.6f}",
        f"accepted_voxel_inflation: {safe_float(selected_row.get('accepted_voxel_inflation')):.6f}",
        "",
        "GT is used only for offline selection/evaluation. Runtime Stage2 still uses",
        "the saved V4 Stage1 scores, local geometry, the accepted refiner model,",
        "and the selected scalar thresholds; GT fields are not model inputs.",
    ]
    (out / "selection_report.txt").write_text("\n".join(report_lines) + "\n")
    print("\n".join(report_lines))
    print("V4_STAGE2_GT_PROFILE_SELECTION_COMPLETE")
    if not selected:
        print("NO_SAFE_GT_IMPROVEMENT: selected output run should not proceed")


if __name__ == "__main__":
    main()
