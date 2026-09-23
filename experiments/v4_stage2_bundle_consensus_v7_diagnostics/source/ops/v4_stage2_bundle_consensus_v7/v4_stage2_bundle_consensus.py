#!/usr/bin/env python3
"""Stage-2 line extraction that preserves deployed V4 Stage-1 line labels.

Runtime invariants
------------------
* Stage 1 is not rerun or modified.
* The deployed Stage-1 label decision is reproduced with ``label_from_scores``
  and the accepted calibration thresholds.
* Every emitted Stage-2 conductor point is an occupied Stage-1 coordinate whose
  deployed Stage-1 label is exactly class 2 (line).
* No pole pair is enumerated and no synthetic line voxel is created.
* GT labels are never read by this runtime processor.

The main correction is lane decomposition.  V4's original exact 26-neighbor
components can merge several parallel conductors through a few diagonal or
vegetation-supported links.  A single PCA/median parameterization can then
collapse those lanes.  This processor decomposes each Stage-1-labelled line
component in cross-section, splits unsupported longitudinal gaps, applies the
production physical/refiner gates, and permits a tightly constrained geometry
fallback for label-backed components rejected by the learned refiner.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from v4_realtime_core import label_from_scores, load_calibration
from v4_realtime_pipeline import V4Stage2Processor
from v4_sparse_components import extract_sparse_components, pca_geometry, sparse_connected_labels
from v4_stage2_local import add_edge_features
BUNDLE_CONSENSUS_RUNTIME_VERSION = "bundle-consensus-v7.1-fp-tolerance-20260903"
BUNDLE_CONSENSUS_NUMERIC_TOL = 1.0e-8
BUNDLE_CONSENSUS_DIAGNOSTIC_VERSION = "bundle-consensus-v7.2-gate-audit-20260903"


from v4_stage2_runtime import (
    LINE_OUTPUT_COLUMNS,
    POLE_OUTPUT_COLUMNS,
    VERTEX_OUTPUT_COLUMNS,
    _physical_mask,
    line_vertices,
    load_bundle,
    local_X,
    pole_param,
)


@dataclass(frozen=True)
class Stage1LabelProfile:
    name: str = "stage1_label_balanced"

    # Cross-section lane decomposition in feet.
    cross_section_eps_ft: float = 0.70
    cross_section_min_samples: int = 3
    cross_section_quantization_ft: float = 0.25
    mode_min_longitudinal_bins: int = 4
    mode_min_longitudinal_fraction: float = 0.15
    noise_attach_max_ft: float = 0.85
    max_lanes_per_component: int = 12

    # Longitudinal support and gap splitting.
    longitudinal_bin_ft: float = 0.50
    max_internal_gap_ft: float = 1.00
    min_longitudinal_coverage: float = 0.70
    min_voxels: int = 3
    min_horizontal_length_ft: float = 1.00

    # Candidate shape gate.  These are applied before the refiner.
    min_linearity: float = 0.70
    max_radius_p90_ft: float = 1.50
    max_tortuosity: float = 3.00
    max_verticality: float = 0.85
    max_vertical_horizontal_ratio: float = 2.50

    # Strict label-backed fallback if the production line refiner rejects a
    # candidate.  This is intentionally much stricter than the production
    # physical gate and never accepts an unlabelled voxel.
    enable_geometry_override: bool = True
    override_refiner_floor: float = 0.05
    override_min_voxels: int = 5
    override_min_horizontal_length_ft: float = 3.00
    override_min_longitudinal_coverage: float = 0.82
    override_max_internal_gap_ft: float = 0.50
    override_min_linearity: float = 0.90
    override_max_radius_p90_ft: float = 0.80
    override_max_tortuosity: float = 1.60
    override_max_verticality: float = 0.70

    # Residual-union controls. Production Stage2 output is preserved exactly;
    # only novel Stage1-label-backed lanes may be appended.
    residual_min_novel_voxels: int = 4
    residual_min_novel_fraction: float = 0.35
    residual_max_baseline_overlap_fraction: float = 0.50
    require_sibling_support: bool = True
    sibling_max_axis_angle_deg: float = 8.0
    sibling_min_cross_section_offset_ft: float = 0.50
    sibling_max_cross_section_offset_ft: float = 10.0
    sibling_min_longitudinal_overlap_fraction: float = 0.45

    # Bundle-consensus controls. A novel residual lane must agree with more
    # than one already accepted production conductor whenever enough siblings
    # are present. This suppresses vegetation branches that happen to run
    # parallel to a single line.
    bundle_min_parallel_siblings: int = 2
    bundle_min_endpoint_overlap_fraction: float = 0.70
    bundle_max_endpoint_extension_ft: float = 4.0
    bundle_spacing_ratio_min: float = 0.45
    bundle_spacing_ratio_max: float = 1.80

    # Production pole extraction remains unchanged.
    pole_candidate_threshold: float = 0.15
    pole_min_voxels: int = 4
    line_min_voxels: int = 3
    edge_width_vox: int = 10

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _AxisModel:
    center_xy: np.ndarray
    axis_xy: np.ndarray
    perp_xy: np.ndarray
    z_coef: np.ndarray

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = np.asarray(points, dtype=np.float64)
        xy = p[:, :2]
        s = (xy - self.center_xy) @ self.axis_xy
        t = (xy - self.center_xy) @ self.perp_xy
        z_pred = np.polyval(self.z_coef, s)
        return s, t, p[:, 2] - z_pred


def profile_from_dict(data: dict[str, Any]) -> Stage1LabelProfile:
    allowed = set(Stage1LabelProfile.__dataclass_fields__)
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown Stage1-label profile keys: {unknown}")
    return Stage1LabelProfile(**data)


def deployed_stage1_labels(pred: dict[str, np.ndarray], calibration: dict[str, Any]) -> np.ndarray:
    pole = np.asarray(pred["pole"], dtype=np.float32)
    line = np.asarray(pred["line"], dtype=np.float32)
    if len(pole) != len(line):
        raise ValueError("Stage1 pole/line score lengths differ")
    labels = label_from_scores(
        pole,
        line,
        float(calibration["pole_threshold"]),
        float(calibration["line_threshold"]),
    )
    return np.asarray(labels, dtype=np.int8)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    aa /= max(float(np.linalg.norm(aa)), 1e-12)
    bb /= max(float(np.linalg.norm(bb)), 1e-12)
    return float(np.degrees(np.arccos(np.clip(abs(float(np.dot(aa, bb))), 0.0, 1.0))))


def _fit_axis(points: np.ndarray) -> _AxisModel | None:
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 2 or not np.isfinite(p).all():
        return None
    xy = p[:, :2]
    center = np.median(xy, axis=0)
    keep = np.ones(len(p), dtype=bool)
    axis = np.array([1.0, 0.0], dtype=float)
    for _ in range(4):
        q = xy[keep] if np.any(keep) else xy
        if len(q) < 2:
            break
        center = np.median(q, axis=0)
        try:
            _, _, vh = np.linalg.svd(q - center, full_matrices=False)
            axis = np.asarray(vh[0], dtype=float)
        except np.linalg.LinAlgError:
            return None
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        perp = np.array([-axis[1], axis[0]], dtype=float)
        residual = np.abs((xy - center) @ perp)
        med = float(np.median(residual))
        mad = float(np.median(np.abs(residual - med)))
        limit = max(1.0, med + 3.0 * max(mad, 0.20))
        new_keep = residual <= limit
        if int(new_keep.sum()) < 2 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    perp = np.array([-axis[1], axis[0]], dtype=float)
    s = (xy - center) @ axis
    degree = 2 if len(p) >= 8 and float(np.ptp(s)) >= 6.0 else 1
    z_keep = keep.copy()
    coef: np.ndarray | None = None
    for _ in range(3):
        try:
            coef = np.polyfit(s[z_keep], p[z_keep, 2], deg=min(degree, max(int(z_keep.sum()) - 1, 1)))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            coef = None
            break
        residual = np.abs(p[:, 2] - np.polyval(coef, s))
        med = float(np.median(residual))
        mad = float(np.median(np.abs(residual - med)))
        limit = max(1.0, med + 3.5 * max(mad, 0.20))
        new_keep = residual <= limit
        if int(new_keep.sum()) < 3 or np.array_equal(new_keep, z_keep):
            break
        z_keep = new_keep
    if coef is None:
        coef = np.array([0.0, float(np.median(p[:, 2]))], dtype=float)
    return _AxisModel(center_xy=center, axis_xy=axis, perp_xy=perp, z_coef=np.asarray(coef, dtype=float))


def _coord_keys(coords: np.ndarray, grid_size: tuple[int, int, int]) -> np.ndarray:
    c = np.asarray(coords, dtype=np.int64)
    if not len(c):
        return np.empty(0, dtype=np.int64)
    gx, gy, _ = map(int, grid_size)
    return (c[:, 2] * gy + c[:, 1]) * gx + c[:, 0]


def _split_longitudinal_runs(
    indices: np.ndarray,
    coords: np.ndarray,
    model: _AxisModel,
    profile: Stage1LabelProfile,
    voxel_size_ft: float,
) -> list[tuple[np.ndarray, dict[str, float]]]:
    idx = np.unique(np.asarray(indices, dtype=np.int64))
    if len(idx) < profile.min_voxels:
        return []
    pts = np.asarray(coords[idx], dtype=np.float64)
    s, _, _ = model.project(pts)
    bin_vox = max(float(profile.longitudinal_bin_ft) / float(voxel_size_ft), 1.0)
    origin = float(np.min(s))
    bins = np.floor((s - origin) / bin_vox + 1e-9).astype(np.int64)
    occupied = np.unique(bins)
    if not len(occupied):
        return []
    allowed_missing = max(int(math.floor(profile.max_internal_gap_ft / profile.longitudinal_bin_ft)), 0)
    cuts = np.flatnonzero(np.diff(occupied) - 1 > allowed_missing) + 1
    runs: list[tuple[np.ndarray, dict[str, float]]] = []
    for run_bins in np.split(occupied, cuts):
        if not len(run_bins):
            continue
        mask = np.isin(bins, run_bins)
        ridx = idx[mask]
        if len(ridx) < profile.min_voxels:
            continue
        rs = s[mask]
        span_bins = int(run_bins[-1] - run_bins[0] + 1)
        coverage = float(len(run_bins) / max(span_bins, 1))
        missing = np.maximum(np.diff(run_bins) - 1, 0)
        max_gap_ft = float(missing.max() * profile.longitudinal_bin_ft) if len(missing) else 0.0
        length_ft = float(np.ptp(rs) * voxel_size_ft) if len(rs) > 1 else 0.0
        runs.append((ridx, {"longitudinal_coverage": coverage, "max_internal_gap_ft": max_gap_ft, "projected_length_ft": length_ft}))
    return runs


def _lane_clusters(
    component_indices: np.ndarray,
    coords: np.ndarray,
    profile: Stage1LabelProfile,
    voxel_size_ft: float,
) -> list[tuple[np.ndarray, int, dict[str, float]]]:
    """Decompose one 26-connected Stage1 line component into persistent lanes.

    A short vegetation/diagonal bridge can merge parallel conductors under exact
    26-connectivity. Plain DBSCAN on all cross-section samples is vulnerable to
    chaining through that bridge. We therefore identify cross-section modes only
    from bins that persist across a material fraction of longitudinal positions;
    short bridge bins are not eligible to connect two lane modes.
    """
    idx = np.asarray(component_indices, dtype=np.int64)
    pts = np.asarray(coords[idx], dtype=np.float64)
    model = _fit_axis(pts)
    if model is None:
        return []

    s_vox, transverse_vox, z_residual_vox = model.project(pts)
    s_ft = s_vox * float(voxel_size_ft)
    cross = np.column_stack([transverse_vox, z_residual_vox]) * float(voxel_size_ft)

    quant = max(float(profile.cross_section_quantization_ft), 1e-3)
    longitudinal_bin = max(float(profile.longitudinal_bin_ft), 1e-3)
    cross_bins = np.rint(cross / quant).astype(np.int64)
    unique_cross, inverse = np.unique(cross_bins, axis=0, return_inverse=True)
    s_bins = np.floor((s_ft - float(np.min(s_ft))) / longitudinal_bin + 1e-9).astype(np.int64)
    parent_longitudinal_bins = max(int(np.unique(s_bins).size), 1)
    required_persistence = max(
        int(profile.mode_min_longitudinal_bins),
        int(math.ceil(float(profile.mode_min_longitudinal_fraction) * parent_longitudinal_bins)),
    )

    persistence = np.zeros(len(unique_cross), dtype=np.int64)
    population = np.zeros(len(unique_cross), dtype=np.int64)
    for bin_index in range(len(unique_cross)):
        members = inverse == bin_index
        population[bin_index] = int(np.sum(members))
        persistence[bin_index] = int(np.unique(s_bins[members]).size)

    persistent = np.flatnonzero(persistence >= required_persistence)
    if not len(persistent):
        # Very short components cannot satisfy a long persistence gate. Use the
        # most persistent cross bin as one conservative mode rather than turning
        # every sparse cross-section point into a lane.
        persistent = np.array([int(np.argmax(persistence))], dtype=np.int64)

    persistent_centers = unique_cross[persistent].astype(float) * quant
    mode_labels = DBSCAN(
        eps=float(profile.cross_section_eps_ft),
        min_samples=1,
        metric="euclidean",
    ).fit_predict(persistent_centers)

    modes: list[tuple[np.ndarray, int, int]] = []
    for mode_id in sorted(int(x) for x in np.unique(mode_labels)):
        local = np.flatnonzero(mode_labels == mode_id)
        source_bins = persistent[local]
        weights = np.maximum(population[source_bins], 1).astype(float)
        center = np.average(unique_cross[source_bins].astype(float) * quant, axis=0, weights=weights)
        support_bins = int(np.max(persistence[source_bins]))
        population_sum = int(np.sum(population[source_bins]))
        modes.append((np.asarray(center, dtype=float), support_bins, population_sum))

    # Retain the most longitudinally persistent modes if branch clutter creates
    # excessive cross-section modes.
    modes.sort(key=lambda value: (value[1], value[2]), reverse=True)
    modes = modes[: int(profile.max_lanes_per_component)]
    modes.sort(key=lambda value: (float(value[0][0]), float(value[0][1])))
    if not modes:
        return []

    centers = np.asarray([mode[0] for mode in modes], dtype=float)
    distance = np.linalg.norm(cross[:, None, :] - centers[None, :, :], axis=2)
    nearest = np.argmin(distance, axis=1)
    nearest_distance = distance[np.arange(len(cross)), nearest]
    assigned = nearest_distance <= float(profile.noise_attach_max_ft)

    out: list[tuple[np.ndarray, int, dict[str, float]]] = []
    for lane_id in range(len(modes)):
        local = np.flatnonzero(assigned & (nearest == lane_id))
        if len(local) < profile.min_voxels:
            continue
        cidx = idx[local]
        lane_model = _fit_axis(coords[cidx])
        if lane_model is None:
            continue
        runs = _split_longitudinal_runs(cidx, coords, lane_model, profile, voxel_size_ft)
        for run_index, (run_indices, support) in enumerate(runs, 1):
            detail = dict(support)
            detail["cross_section_cluster"] = float(lane_id)
            detail["run_index"] = float(run_index)
            detail["parent_cross_section_clusters"] = float(len(modes))
            detail["mode_longitudinal_bins"] = float(modes[lane_id][1])
            detail["mode_population"] = float(modes[lane_id][2])
            detail["mode_required_longitudinal_bins"] = float(required_persistence)
            out.append((run_indices, lane_id, detail))
    return out


def _component_row(
    component_id: str,
    source_component_id: int,
    lane_cluster_id: int,
    points: np.ndarray,
    line_values: np.ndarray,
    semantic_values: np.ndarray,
    support: dict[str, float],
    voxel_size_ft: float,
) -> dict[str, Any]:
    pts = np.asarray(points, dtype=np.int32)
    vals = np.asarray(line_values, dtype=np.float32)
    sem = np.asarray(semantic_values, dtype=np.uint8)
    geo = pca_geometry(pts, voxel_size_ft)
    mins = pts.min(0)
    maxs = pts.max(0)
    spans = (maxs - mins + 1) * voxel_size_ft
    bbox = float(np.prod(maxs - mins + 1))
    direction = np.asarray(geo["principal"], dtype=float)
    verticality = float(abs(direction[2]))
    horizontal_head = float(math.sqrt(max(0.0, 1.0 - verticality * verticality)))
    endpoints = np.asarray(geo["endpoints"], dtype=float)
    row: dict[str, Any] = {
        "component_id": component_id,
        "class_name": "line",
        "n_voxels": int(len(pts)),
        "score_mean": float(vals.mean()),
        "score_std": float(vals.std()),
        "score_p10": float(np.quantile(vals, 0.10)),
        "score_p50": float(np.quantile(vals, 0.50)),
        "score_p90": float(np.quantile(vals, 0.90)),
        "score_max": float(vals.max()),
        "vertical_head_mean": verticality,
        "horizontal_head_mean": horizontal_head,
        "x_span_ft": float(spans[0]),
        "y_span_ft": float(spans[1]),
        "z_span_ft": float(spans[2]),
        "horizontal_span_ft": float(np.linalg.norm(spans[:2])),
        "bbox_density": float(len(pts) / max(bbox, 1.0)),
        "center_x": float(pts[:, 0].mean()),
        "center_y": float(pts[:, 1].mean()),
        "center_z": float(pts[:, 2].mean()),
        "center_z_ft": float(pts[:, 2].mean() * voxel_size_ft),
        "min_z_ft": float(mins[2] * voxel_size_ft),
        "max_z_ft": float(maxs[2] * voxel_size_ft),
        "principal_dx": float(direction[0]),
        "principal_dy": float(direction[1]),
        "principal_dz": float(direction[2]),
        "principal_verticality": verticality,
        "linearity": float(geo["linearity"]),
        "planarity": float(geo["planarity"]),
        "scattering": float(geo["scattering"]),
        "radius_p50_ft": float(geo["radius_p50_ft"]),
        "radius_p90_ft": float(geo["radius_p90_ft"]),
        "xy_path_length_ft": float(geo["xy_path_length_ft"]),
        "xy_endpoint_distance_ft": float(geo["xy_endpoint_distance_ft"]),
        "xy_tortuosity": float(geo["xy_tortuosity"]),
        "quadratic_rmse_ft": float(geo["quadratic_rmse_ft"]),
        "sag_estimate_ft": float(geo["sag_estimate_ft"]),
        "endpoint1_x": float(endpoints[0, 0]),
        "endpoint1_y": float(endpoints[0, 1]),
        "endpoint1_z": float(endpoints[0, 2]),
        "endpoint2_x": float(endpoints[1, 0]),
        "endpoint2_y": float(endpoints[1, 1]),
        "endpoint2_z": float(endpoints[1, 2]),
        "exact_gt_fraction": 0.0,
        "near_gt_fraction": 0.0,
        "proposal_kind": "stage1_inferred_line_lane",
        "source_stage1_component_id": int(source_component_id),
        "lane_cluster_id": int(lane_cluster_id),
        "stage1_line_label_fraction": 1.0,
        "semantic_line_fraction": float(np.mean(sem == 2)) if len(sem) else 0.0,
        **support,
    }
    return row


def _candidate_geometry_ok(row: pd.Series | dict[str, Any], profile: Stage1LabelProfile) -> bool:
    get = row.get if hasattr(row, "get") else lambda k, d=None: d
    h = float(get("horizontal_span_ft", 0.0))
    z = float(get("z_span_ft", 0.0))
    verticality = float(get("principal_verticality", 1.0))
    return bool(
        int(get("n_voxels", 0)) >= profile.min_voxels
        and h >= profile.min_horizontal_length_ft
        and float(get("longitudinal_coverage", 0.0)) >= profile.min_longitudinal_coverage
        and float(get("max_internal_gap_ft", 1e9)) <= profile.max_internal_gap_ft + 1e-9
        and float(get("linearity", 0.0)) >= profile.min_linearity
        and float(get("radius_p90_ft", 1e9)) <= profile.max_radius_p90_ft
        and float(get("xy_tortuosity", 1e9)) <= profile.max_tortuosity
        and verticality <= profile.max_verticality
        and z / max(h, 1e-6) <= profile.max_vertical_horizontal_ratio
    )


def _override_geometry_ok(d: pd.DataFrame, profile: Stage1LabelProfile) -> np.ndarray:
    if d.empty:
        return np.zeros(0, dtype=bool)
    def col(name: str, default: float) -> np.ndarray:
        return pd.to_numeric(d.get(name, pd.Series(default, index=d.index)), errors="coerce").fillna(default).to_numpy(float)
    return (
        (col("stage1_line_label_fraction", 0.0) >= 0.999999)
        & (col("n_voxels", 0.0) >= profile.override_min_voxels)
        & (col("horizontal_span_ft", 0.0) >= profile.override_min_horizontal_length_ft)
        & (col("longitudinal_coverage", 0.0) >= profile.override_min_longitudinal_coverage)
        & (col("max_internal_gap_ft", 1e9) <= profile.override_max_internal_gap_ft + 1e-9)
        & (col("linearity", 0.0) >= profile.override_min_linearity)
        & (col("radius_p90_ft", 1e9) <= profile.override_max_radius_p90_ft)
        & (col("xy_tortuosity", 1e9) <= profile.override_max_tortuosity)
        & (col("principal_verticality", 1.0) <= profile.override_max_verticality)
    )


def extract_stage1_label_line_components(
    item: dict[str, Any],
    pred: dict[str, np.ndarray],
    calibration: dict[str, Any],
    profile: Stage1LabelProfile,
    grid_size: tuple[int, int, int],
    voxel_size_ft: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    coords = np.asarray(item["coords"], dtype=np.int32)
    line_scores = np.asarray(pred["line"], dtype=np.float32)
    semantic = np.asarray(pred.get("semantic", np.zeros(len(coords))), dtype=np.uint8)
    labels = deployed_stage1_labels(pred, calibration)
    if not (len(coords) == len(line_scores) == len(semantic) == len(labels)):
        raise ValueError("Stage1 artifact arrays do not align")
    line_mask = labels == 2
    line_coords = coords[line_mask]
    source_indices = np.flatnonzero(line_mask)
    raw_labels, raw_count = sparse_connected_labels(line_coords, grid_size)

    rows: list[dict[str, Any]] = []
    points: dict[str, np.ndarray] = {}
    accepted_source_indices: dict[str, np.ndarray] = {}
    output_index = 0
    rejected_shape = 0
    raw_with_multiple_lanes = 0
    split_run_count = 0

    for raw_id in range(1, raw_count + 1):
        local = np.flatnonzero(raw_labels == raw_id)
        if len(local) < profile.min_voxels:
            continue
        raw_indices = source_indices[local]
        lanes = _lane_clusters(raw_indices, coords, profile, voxel_size_ft)
        lane_ids = {lane_id for _, lane_id, _ in lanes}
        if len(lane_ids) > 1:
            raw_with_multiple_lanes += 1
        if len(lanes) > len(lane_ids):
            split_run_count += len(lanes) - len(lane_ids)
        for run_indices, lane_id, support in lanes:
            output_index += 1
            cid = f"R{output_index:05d}"
            row = _component_row(
                cid,
                raw_id,
                lane_id,
                coords[run_indices],
                line_scores[run_indices],
                semantic[run_indices],
                support,
                voxel_size_ft,
            )
            if not _candidate_geometry_ok(row, profile):
                rejected_shape += 1
                continue
            rows.append(row)
            points[cid] = coords[run_indices].astype(np.int32, copy=True)
            accepted_source_indices[cid] = np.asarray(run_indices, dtype=np.int64)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = add_edge_features(frame, points, grid_size, edge_width_vox=profile.edge_width_vox)
    audit = {
        "stage1_occupied_voxels": int(len(coords)),
        "stage1_inferred_line_voxels": int(line_mask.sum()),
        "stage1_semantic_line_voxels": int(np.sum(semantic == 2)),
        "stage1_fused_semantic_agreement_voxels": int(np.sum((labels == 2) & (semantic == 2))),
        "raw_stage1_line_components": int(raw_count),
        "raw_components_with_multiple_lanes": int(raw_with_multiple_lanes),
        "longitudinal_extra_runs": int(split_run_count),
        "shape_rejected_candidates": int(rejected_shape),
        "final_label_backed_candidates": int(len(frame)),
        "runtime_gt_usage": False,
        "synthetic_line_voxels": 0,
    }
    return frame, points, {"audit": audit, "source_indices": accepted_source_indices, "stage1_labels": labels}


def _apply_stage2(
    comps: dict[str, Any],
    bundle: dict[str, Any],
    profile: Stage1LabelProfile,
    file_id: str,
    slice_seq: int,
    voxel_size_ft: float,
) -> dict[str, Any]:
    frames: list[pd.DataFrame] = []
    poles_out: list[dict[str, Any]] = []
    lines_out: list[dict[str, Any]] = []
    vertices_out: list[dict[str, Any]] = []
    for name, df, points in [
        ("pole", comps["poles"], comps["pole_points"]),
        ("line", comps["lines"], comps["line_points"]),
    ]:
        d = df.copy()
        if d.empty:
            continue
        clf = bundle[name + "_model"]
        threshold = float(bundle[name + "_threshold"])
        prob = clf.predict_proba(local_X(d, bundle["feature_columns"]))[:, 1]
        physical = _physical_mask(d, name)
        production_accept = prob >= threshold
        override = np.zeros(len(d), dtype=bool)
        if name == "line" and profile.enable_geometry_override:
            override = (prob >= profile.override_refiner_floor) & _override_geometry_ok(d, profile)
        accept = physical & (production_accept | override)
        d["refiner_probability"] = prob
        d["refiner_threshold"] = threshold
        d["physical_ok"] = physical
        d["production_refiner_accept"] = production_accept
        d["stage1_label_geometry_override"] = override
        d["component_accept"] = accept
        d["accept_mode"] = np.select(
            [accept & production_accept, accept & ~production_accept & override],
            ["production_refiner", "stage1_label_geometry_override"],
            default="rejected",
        )
        d["file_id"] = file_id
        d["slice_seq"] = int(slice_seq)
        frames.append(d)
        for row in d.loc[d["component_accept"].astype(bool)].itertuples(index=False):
            cid = str(row.component_id)
            pts = np.asarray(points[cid], dtype=float)
            if name == "pole":
                poles_out.append({
                    "file_id": file_id,
                    "component_id": cid,
                    "slice_seq": int(slice_seq),
                    "refiner_probability": float(row.refiner_probability),
                    "touches_xy_edge": bool(row.touches_xy_edge),
                    "radius_p90_ft": float(row.radius_p90_ft),
                    "verticality": float(row.principal_verticality),
                    **pole_param(pts, voxel_size_ft),
                })
            else:
                vertices = line_vertices(pts)
                lines_out.append({
                    "file_id": file_id,
                    "component_id": cid,
                    "slice_seq": int(slice_seq),
                    "refiner_probability": float(row.refiner_probability),
                    "horizontal_span_ft": float(row.horizontal_span_ft),
                    "vertical_span_ft": float(row.z_span_ft),
                    "verticality": float(row.principal_verticality),
                    "tortuosity": float(row.xy_tortuosity),
                    "vertex_count": int(len(vertices)),
                })
                for vertex_index, q in enumerate(vertices):
                    vertices_out.append({
                        "file_id": file_id,
                        "component_id": cid,
                        "slice_seq": int(slice_seq),
                        "vertex_index": int(vertex_index),
                        "x": float(q[0]),
                        "y": float(q[1]),
                        "z": float(q[2]),
                    })
    return {
        "components": pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
        "poles": pd.DataFrame(poles_out, columns=POLE_OUTPUT_COLUMNS),
        "lines": pd.DataFrame(lines_out, columns=LINE_OUTPUT_COLUMNS),
        "vertices": pd.DataFrame(vertices_out, columns=VERTEX_OUTPUT_COLUMNS),
    }


class _ReplacementStage1LabelStage2Processor:
    """V4 Stage-2 processor driven by exact deployed Stage-1 inferred labels."""

    def __init__(
        self,
        stage2_bundle: str,
        calibration_json: str,
        profile: Stage1LabelProfile,
        grid_size: tuple[int, int, int] = (400, 400, 200),
        voxel_size_ft: float = 0.5,
    ) -> None:
        self.grid = tuple(map(int, grid_size))
        self.voxel = float(voxel_size_ft)
        self.bundle = load_bundle(stage2_bundle)
        self.calibration = load_calibration(calibration_json)
        self.profile = profile

    def process(
        self,
        item: dict[str, Any],
        pred: dict[str, np.ndarray],
        file_id: str = "slice",
        slice_seq: int = 0,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        # Production pole extraction is retained exactly.  Its line result is
        # ignored because line candidates are rebuilt from deployed Stage-1 labels.
        baseline = extract_sparse_components(
            item,
            pred,
            self.grid,
            self.voxel,
            gt_points=None,
            pole_threshold=self.profile.pole_candidate_threshold,
            line_threshold=0.08,
            line_weak_threshold=0.04,
            line_competition_ratio=0.55,
            pole_min_voxels=self.profile.pole_min_voxels,
            line_min_voxels=self.profile.line_min_voxels,
            edge_width_vox=self.profile.edge_width_vox,
        )
        lines, line_points, detail = extract_stage1_label_line_components(
            item,
            pred,
            self.calibration,
            self.profile,
            self.grid,
            self.voxel,
        )
        comps = {
            "poles": baseline["poles"],
            "lines": lines,
            "pole_points": baseline["pole_points"],
            "line_points": line_points,
            "candidate_counts": {
                **baseline.get("candidate_counts", {}),
                **detail["audit"],
            },
        }
        component_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        result = _apply_stage2(comps, self.bundle, self.profile, file_id, slice_seq, self.voxel)
        refine_ms = (time.perf_counter() - t1) * 1000.0

        line_frame = result["components"]
        if not line_frame.empty and "class_name" in line_frame.columns:
            line_frame = line_frame[line_frame["class_name"].astype(str).eq("line")].copy()
        if line_frame.empty:
            accepted = line_frame.copy()
        elif "component_accept" not in line_frame.columns:
            raise RuntimeError("Stage2 component table is missing component_accept")
        else:
            accepted = line_frame[line_frame["component_accept"].astype(bool)].copy()
        accepted_ids = accepted["component_id"].astype(str).tolist() if not accepted.empty else []
        accepted_keys: set[int] = set()
        for cid in accepted_ids:
            pts = line_points.get(cid)
            if pts is not None and len(pts):
                accepted_keys.update(map(int, _coord_keys(pts, self.grid)))
        all_stage1_line_keys = set(map(int, _coord_keys(np.asarray(item["coords"])[detail["stage1_labels"] == 2], self.grid)))
        if not accepted_keys.issubset(all_stage1_line_keys):
            raise RuntimeError("Stage2 emitted line voxels not backed by deployed Stage1 line labels")

        modes = accepted["accept_mode"].value_counts().to_dict() if not accepted.empty and "accept_mode" in accepted.columns else {}
        audit = {
            **detail["audit"],
            "accepted_line_components": int(len(accepted_ids)),
            "accepted_stage1_line_voxels": int(len(accepted_keys)),
            "stage1_to_stage2_voxel_preservation": float(len(accepted_keys) / max(len(all_stage1_line_keys), 1)),
            "accepted_by_mode": {str(k): int(v) for k, v in modes.items()},
            "runtime_gt_usage": False,
            "synthetic_line_voxels": 0,
        }
        return {
            "raw_components": comps,
            **result,
            "stage1_labels": detail["stage1_labels"],
            "stage1_label_audit": audit,
            "timing": {
                "stage2_component_ms": component_ms,
                "stage2_refiner_parametric_ms": refine_ms,
                "stage2_total_ms": component_ms + refine_ms,
            },
        }



def _accepted_line_ids(frame: pd.DataFrame) -> list[str]:
    if frame.empty or not {"class_name", "component_accept", "component_id"} <= set(frame.columns):
        return []
    return frame[
        frame["class_name"].astype(str).eq("line")
        & frame["component_accept"].astype(bool)
    ]["component_id"].astype(str).tolist()


def _key_set(points: np.ndarray, grid_size: tuple[int, int, int]) -> set[int]:
    if points is None or not len(points):
        return set()
    return set(map(int, _coord_keys(np.asarray(points, dtype=np.int32), grid_size)))


def _line_support_descriptor(points: np.ndarray, voxel_size_ft: float) -> dict[str, Any] | None:
    pts = np.asarray(points, dtype=np.float64)
    model = _fit_axis(pts)
    if model is None or len(pts) < 2:
        return None
    s, _, _ = model.project(pts)
    return {
        "model": model,
        "points": pts,
        "s_min": float(np.min(s)),
        "s_max": float(np.max(s)),
        "span": float(np.ptp(s)),
        "voxel_size_ft": float(voxel_size_ft),
    }


def _sibling_support(
    candidate_points: np.ndarray,
    baseline_points: dict[str, np.ndarray],
    profile: Stage1LabelProfile,
    voxel_size_ft: float,
) -> dict[str, Any]:
    """V7.1 bundle consensus with diagnostic gate progression only."""
    candidate = _line_support_descriptor(candidate_points, voxel_size_ft)
    if candidate is None:
        return {
            "supported": False,
            "reason": "candidate_axis_unavailable",
            "baseline_components_total": int(len(baseline_points)),
            "axis_compatible_count": 0,
            "longitudinal_compatible_count": 0,
            "endpoint_overlap_compatible_count": 0,
            "endpoint_extension_compatible_count": 0,
            "offset_compatible_count": 0,
        }

    cmodel = candidate["model"]
    cpoints = candidate["points"]
    cs, _, _ = cmodel.project(cpoints)
    cmin, cmax = float(np.min(cs)), float(np.max(cs))
    cspan = max(cmax - cmin, 1e-9)

    compatible: list[dict[str, Any]] = []
    cross_points: list[np.ndarray] = []
    total = int(len(baseline_points))
    axis_count = longitudinal_count = endpoint_overlap_count = endpoint_extension_count = offset_count = 0
    min_axis_angle = float("inf")
    max_longitudinal_overlap = 0.0
    max_endpoint_overlap = 0.0
    min_endpoint_extension = float("inf")
    min_cross_offset = float("inf")
    max_cross_offset = 0.0

    def diag(reason: str, supported: bool = False, **extra: Any) -> dict[str, Any]:
        out = {
            "supported": bool(supported),
            "reason": reason,
            "baseline_components_total": total,
            "axis_compatible_count": int(axis_count),
            "longitudinal_compatible_count": int(longitudinal_count),
            "endpoint_overlap_compatible_count": int(endpoint_overlap_count),
            "endpoint_extension_compatible_count": int(endpoint_extension_count),
            "offset_compatible_count": int(offset_count),
            "best_axis_angle_deg": None if not np.isfinite(min_axis_angle) else float(min_axis_angle),
            "best_longitudinal_overlap_fraction": float(max_longitudinal_overlap),
            "best_endpoint_overlap_fraction": float(max_endpoint_overlap),
            "best_endpoint_extension_ft": None if not np.isfinite(min_endpoint_extension) else float(min_endpoint_extension),
            "min_cross_section_offset_ft_seen": None if not np.isfinite(min_cross_offset) else float(min_cross_offset),
            "max_cross_section_offset_ft_seen": float(max_cross_offset),
        }
        out.update(extra)
        return out

    for component_id, points in baseline_points.items():
        baseline = _line_support_descriptor(points, voxel_size_ft)
        if baseline is None:
            continue
        bmodel = baseline["model"]
        angle = _angle_deg(cmodel.axis_xy, bmodel.axis_xy)
        min_axis_angle = min(min_axis_angle, float(angle))
        if angle > float(profile.sibling_max_axis_angle_deg) + BUNDLE_CONSENSUS_NUMERIC_TOL:
            continue
        axis_count += 1

        bpoints = baseline["points"]
        bs = (bpoints[:, :2] - cmodel.center_xy) @ cmodel.axis_xy
        bmin, bmax = float(np.min(bs)), float(np.max(bs))
        overlap = max(0.0, min(cmax, bmax) - max(cmin, bmin))
        overlap_fraction = float(overlap / cspan)
        max_longitudinal_overlap = max(max_longitudinal_overlap, overlap_fraction)
        if overlap_fraction + BUNDLE_CONSENSUS_NUMERIC_TOL < float(profile.sibling_min_longitudinal_overlap_fraction):
            continue
        longitudinal_count += 1

        endpoint_overlap = float(overlap / max(cspan, bmax - bmin, 1e-9))
        max_endpoint_overlap = max(max_endpoint_overlap, endpoint_overlap)
        if endpoint_overlap + BUNDLE_CONSENSUS_NUMERIC_TOL < float(profile.bundle_min_endpoint_overlap_fraction):
            continue
        endpoint_overlap_count += 1

        extension_vox = max(0.0, cmin - bmin, bmin - cmin, cmax - bmax, bmax - cmax)
        endpoint_extension_ft = float(extension_vox * voxel_size_ft)
        min_endpoint_extension = min(min_endpoint_extension, endpoint_extension_ft)
        if endpoint_extension_ft > float(profile.bundle_max_endpoint_extension_ft) + BUNDLE_CONSENSUS_NUMERIC_TOL:
            continue
        endpoint_extension_count += 1

        bcenter = np.median(bpoints, axis=0)
        _, transverse, vertical = cmodel.project(bcenter.reshape(1, 3))
        cross = np.array([float(transverse[0]), float(vertical[0])], dtype=float) * float(voxel_size_ft)
        offset_ft = float(np.linalg.norm(cross))
        min_cross_offset = min(min_cross_offset, offset_ft)
        max_cross_offset = max(max_cross_offset, offset_ft)
        if offset_ft + BUNDLE_CONSENSUS_NUMERIC_TOL < float(profile.sibling_min_cross_section_offset_ft):
            continue
        if offset_ft > float(profile.sibling_max_cross_section_offset_ft) + BUNDLE_CONSENSUS_NUMERIC_TOL:
            continue
        offset_count += 1

        compatible.append({
            "component_id": str(component_id),
            "axis_angle_deg": float(angle),
            "offset_ft": float(offset_ft),
            "overlap_fraction": float(overlap_fraction),
            "endpoint_overlap_fraction": float(endpoint_overlap),
            "endpoint_extension_ft": float(endpoint_extension_ft),
        })
        cross_points.append(cross)

    min_siblings = max(1, int(profile.bundle_min_parallel_siblings))
    if len(compatible) < min_siblings:
        return diag(
            "insufficient_parallel_bundle_siblings",
            bundle_sibling_count=int(len(compatible)),
            bundle_min_parallel_siblings=int(min_siblings),
        )

    cp = np.vstack(cross_points)
    pairwise: list[float] = []
    for i in range(len(cp)):
        for j in range(i + 1, len(cp)):
            d = float(np.linalg.norm(cp[i] - cp[j]))
            if d > 1e-6:
                pairwise.append(d)
    nearest_candidate_spacing = float(np.min(np.linalg.norm(cp, axis=1)))
    spacing_reference = float(np.median(pairwise)) if pairwise else nearest_candidate_spacing
    spacing_ratio = float(nearest_candidate_spacing / max(spacing_reference, 1e-9))
    if (
        spacing_ratio + BUNDLE_CONSENSUS_NUMERIC_TOL < float(profile.bundle_spacing_ratio_min)
        or spacing_ratio > float(profile.bundle_spacing_ratio_max) + BUNDLE_CONSENSUS_NUMERIC_TOL
    ):
        return diag(
            "bundle_spacing_inconsistent",
            bundle_sibling_count=int(len(compatible)),
            bundle_nearest_spacing_ft=nearest_candidate_spacing,
            bundle_reference_spacing_ft=spacing_reference,
            bundle_spacing_ratio=spacing_ratio,
        )

    compatible.sort(key=lambda x: (x["overlap_fraction"], -x["axis_angle_deg"]), reverse=True)
    best = compatible[0]
    return diag(
        "supported",
        supported=True,
        sibling_component_id=best["component_id"],
        sibling_axis_angle_deg=best["axis_angle_deg"],
        sibling_cross_section_offset_ft=best["offset_ft"],
        sibling_longitudinal_overlap_fraction=best["overlap_fraction"],
        bundle_sibling_count=int(len(compatible)),
        bundle_nearest_spacing_ft=nearest_candidate_spacing,
        bundle_reference_spacing_ft=spacing_reference,
        bundle_spacing_ratio=spacing_ratio,
        bundle_endpoint_overlap_fraction=float(min(x["endpoint_overlap_fraction"] for x in compatible)),
        bundle_max_endpoint_extension_ft=float(max(x["endpoint_extension_ft"] for x in compatible)),
    )


def _empty_like(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.iloc[0:0].copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()


class Stage1BundleConsensusStage2Processor:
    """Preserve production Stage2 and append only safe residual Stage1 lanes.

    This architecture fixes the V5 failure mode: candidate lane decomposition no
    longer replaces production line components. Therefore production-accepted
    true lines cannot be lost by the experiment. New components must be novel,
    Stage1-label-backed, refiner/geometry accepted, and supported by a parallel
    production-accepted sibling conductor.
    """

    def __init__(
        self,
        stage2_bundle: str,
        calibration_json: str,
        profile: Stage1LabelProfile,
        grid_size: tuple[int, int, int] = (400, 400, 200),
        voxel_size_ft: float = 0.5,
    ) -> None:
        self.grid = tuple(map(int, grid_size))
        self.voxel = float(voxel_size_ft)
        self.bundle = load_bundle(stage2_bundle)
        self.calibration = load_calibration(calibration_json)
        self.profile = profile
        self.production = V4Stage2Processor(
            stage2_bundle,
            grid_size=self.grid,
            voxel_size_ft=self.voxel,
            pole_candidate_threshold=0.15,
            line_candidate_threshold=0.08,
            line_weak_threshold=0.04,
            line_competition_ratio=0.55,
            pole_min_voxels=profile.pole_min_voxels,
            line_min_voxels=profile.line_min_voxels,
            edge_width_vox=profile.edge_width_vox,
        )

    def process(
        self,
        item: dict[str, Any],
        pred: dict[str, np.ndarray],
        file_id: str = "slice",
        slice_seq: int = 0,
    ) -> dict[str, Any]:
        total_start = time.perf_counter()
        baseline = self.production.process(item, pred, file_id, slice_seq)
        baseline_ms = (time.perf_counter() - total_start) * 1000.0

        base_frame = baseline.get("components", pd.DataFrame()).copy()
        base_raw = baseline.get("raw_components", {})
        base_line_points = {
            str(k): np.asarray(v, dtype=np.int32)
            for k, v in base_raw.get("line_points", {}).items()
        }
        base_line_ids = _accepted_line_ids(base_frame)
        accepted_baseline_points = {
            cid: base_line_points[cid]
            for cid in base_line_ids
            if cid in base_line_points and len(base_line_points[cid])
        }
        baseline_keys: set[int] = set()
        for points in accepted_baseline_points.values():
            baseline_keys.update(_key_set(points, self.grid))

        candidate_start = time.perf_counter()
        residual_frame, residual_points, detail = extract_stage1_label_line_components(
            item,
            pred,
            self.calibration,
            self.profile,
            self.grid,
            self.voxel,
        )
        candidate_comps = {
            "poles": _empty_like(base_raw.get("poles", pd.DataFrame())),
            "lines": residual_frame,
            "pole_points": {},
            "line_points": residual_points,
        }
        candidate_result = _apply_stage2(
            candidate_comps,
            self.bundle,
            self.profile,
            file_id,
            slice_seq,
            self.voxel,
        )
        candidate_ms = (time.perf_counter() - candidate_start) * 1000.0

        candidate_components = candidate_result.get("components", pd.DataFrame()).copy()
        if not candidate_components.empty:
            candidate_components = candidate_components[
                candidate_components["class_name"].astype(str).eq("line")
            ].copy()

        selected_ids: list[str] = []
        selected_points: dict[str, np.ndarray] = {}
        candidate_audit: list[dict[str, Any]] = []
        claimed_residual_keys: set[int] = set()

        if not candidate_components.empty:
            candidate_components["residual_original_accept"] = candidate_components[
                "component_accept"
            ].astype(bool)
            candidate_components["residual_novel_voxels"] = 0
            candidate_components["residual_novel_fraction"] = 0.0
            candidate_components["residual_baseline_overlap_fraction"] = 0.0
            candidate_components["residual_sibling_supported"] = False
            candidate_components["residual_sibling_component_id"] = ""
            candidate_components["residual_sibling_axis_angle_deg"] = np.nan
            candidate_components["residual_sibling_cross_section_offset_ft"] = np.nan
            candidate_components["residual_sibling_longitudinal_overlap_fraction"] = np.nan
            candidate_components["residual_bundle_sibling_count"] = 0
            candidate_components["residual_bundle_spacing_ratio"] = np.nan
            candidate_components["residual_bundle_endpoint_overlap_fraction"] = np.nan
            candidate_components["residual_bundle_max_endpoint_extension_ft"] = np.nan
            candidate_components["residual_reject_reason"] = ""

            for index, row in candidate_components.iterrows():
                cid = str(row["component_id"])
                points = residual_points.get(cid, np.empty((0, 3), dtype=np.int32))
                keys = _key_set(points, self.grid)
                novel = keys - baseline_keys - claimed_residual_keys
                novel_count = int(len(novel))
                novel_fraction = float(novel_count / max(len(keys), 1))
                overlap_fraction = float(1.0 - novel_fraction)
                sibling = _sibling_support(
                    points,
                    accepted_baseline_points,
                    self.profile,
                    self.voxel,
                )

                reasons: list[str] = []
                if not bool(row.get("component_accept", False)):
                    reasons.append("refiner_or_geometry_rejected")
                if novel_count < int(self.profile.residual_min_novel_voxels):
                    reasons.append("insufficient_novel_voxels")
                if novel_fraction < float(self.profile.residual_min_novel_fraction):
                    reasons.append("insufficient_novel_fraction")
                if overlap_fraction > float(self.profile.residual_max_baseline_overlap_fraction):
                    reasons.append("excessive_baseline_overlap")
                if self.profile.require_sibling_support and not bool(sibling.get("supported", False)):
                    reasons.append("missing_parallel_production_sibling")

                final_accept = not reasons
                candidate_components.at[index, "component_accept"] = bool(final_accept)
                candidate_components.at[index, "residual_novel_voxels"] = novel_count
                candidate_components.at[index, "residual_novel_fraction"] = novel_fraction
                candidate_components.at[index, "residual_baseline_overlap_fraction"] = overlap_fraction
                candidate_components.at[index, "residual_sibling_supported"] = bool(sibling.get("supported", False))
                candidate_components.at[index, "residual_sibling_component_id"] = str(sibling.get("sibling_component_id", ""))
                candidate_components.at[index, "residual_sibling_axis_angle_deg"] = sibling.get("sibling_axis_angle_deg", np.nan)
                candidate_components.at[index, "residual_sibling_cross_section_offset_ft"] = sibling.get("sibling_cross_section_offset_ft", np.nan)
                candidate_components.at[index, "residual_sibling_longitudinal_overlap_fraction"] = sibling.get("sibling_longitudinal_overlap_fraction", np.nan)
                candidate_components.at[index, "residual_bundle_sibling_count"] = int(sibling.get("bundle_sibling_count", 0) or 0)
                candidate_components.at[index, "residual_bundle_spacing_ratio"] = sibling.get("bundle_spacing_ratio", np.nan)
                candidate_components.at[index, "residual_bundle_endpoint_overlap_fraction"] = sibling.get("bundle_endpoint_overlap_fraction", np.nan)
                candidate_components.at[index, "residual_bundle_max_endpoint_extension_ft"] = sibling.get("bundle_max_endpoint_extension_ft", np.nan)
                candidate_components.at[index, "residual_reject_reason"] = ";".join(reasons)
                if final_accept:
                    candidate_components.at[index, "accept_mode"] = (
                        "stage1_bundle_consensus_" + str(row.get("accept_mode", "accepted"))
                    )
                    selected_ids.append(cid)
                    selected_points[cid] = np.asarray(points, dtype=np.int32)
                    claimed_residual_keys.update(keys)
                else:
                    candidate_components.at[index, "accept_mode"] = "residual_rejected"

                candidate_audit.append({
                    "component_id": cid,
                    "accepted": bool(final_accept),
                    "refiner_or_geometry_accept": bool(row.get("component_accept", False)),
                    "refiner_probability": float(row.get("refiner_probability", 0.0) or 0.0),
                    "novel_voxels": novel_count,
                    "novel_fraction": novel_fraction,
                    "baseline_overlap_fraction": overlap_fraction,
                    "novel_voxels_pass": bool(novel_count >= int(self.profile.residual_min_novel_voxels)),
                    "novel_fraction_pass": bool(novel_fraction >= float(self.profile.residual_min_novel_fraction)),
                    "baseline_overlap_pass": bool(overlap_fraction <= float(self.profile.residual_max_baseline_overlap_fraction) + BUNDLE_CONSENSUS_NUMERIC_TOL),
                    "sibling_supported": bool(sibling.get("supported", False)),
                    "sibling_reason": str(sibling.get("reason", "")),
                    "sibling_component_id": str(sibling.get("sibling_component_id", "")),
                    "bundle_sibling_count": int(sibling.get("bundle_sibling_count", 0) or 0),
                    "bundle_min_parallel_siblings": int(sibling.get("bundle_min_parallel_siblings", self.profile.bundle_min_parallel_siblings) or self.profile.bundle_min_parallel_siblings),
                    "baseline_components_total": int(sibling.get("baseline_components_total", 0) or 0),
                    "axis_compatible_count": int(sibling.get("axis_compatible_count", 0) or 0),
                    "longitudinal_compatible_count": int(sibling.get("longitudinal_compatible_count", 0) or 0),
                    "endpoint_overlap_compatible_count": int(sibling.get("endpoint_overlap_compatible_count", 0) or 0),
                    "endpoint_extension_compatible_count": int(sibling.get("endpoint_extension_compatible_count", 0) or 0),
                    "offset_compatible_count": int(sibling.get("offset_compatible_count", 0) or 0),
                    "best_axis_angle_deg": sibling.get("best_axis_angle_deg", None),
                    "best_longitudinal_overlap_fraction": sibling.get("best_longitudinal_overlap_fraction", None),
                    "best_endpoint_overlap_fraction": sibling.get("best_endpoint_overlap_fraction", None),
                    "best_endpoint_extension_ft": sibling.get("best_endpoint_extension_ft", None),
                    "min_cross_section_offset_ft_seen": sibling.get("min_cross_section_offset_ft_seen", None),
                    "max_cross_section_offset_ft_seen": sibling.get("max_cross_section_offset_ft_seen", None),
                    "bundle_spacing_ratio": sibling.get("bundle_spacing_ratio", None),
                    "bundle_nearest_spacing_ft": sibling.get("bundle_nearest_spacing_ft", None),
                    "bundle_reference_spacing_ft": sibling.get("bundle_reference_spacing_ft", None),
                    "bundle_endpoint_overlap_fraction": sibling.get("bundle_endpoint_overlap_fraction", None),
                    "bundle_max_endpoint_extension_ft": sibling.get("bundle_max_endpoint_extension_ft", None),
                    "reject_reason": ";".join(reasons),
                })

        # Build only the newly approved line outputs. Production rows are copied
        # byte-for-byte at DataFrame level and are never reparameterized.
        residual_lines: list[dict[str, Any]] = []
        residual_vertices: list[dict[str, Any]] = []
        if not candidate_components.empty and selected_ids:
            selected_frame = candidate_components[
                candidate_components["component_id"].astype(str).isin(selected_ids)
            ]
            for row in selected_frame.itertuples(index=False):
                cid = str(row.component_id)
                points = np.asarray(selected_points[cid], dtype=float)
                vertices = line_vertices(points)
                residual_lines.append({
                    "file_id": file_id,
                    "component_id": cid,
                    "slice_seq": int(slice_seq),
                    "refiner_probability": float(row.refiner_probability),
                    "horizontal_span_ft": float(row.horizontal_span_ft),
                    "vertical_span_ft": float(row.z_span_ft),
                    "verticality": float(row.principal_verticality),
                    "tortuosity": float(row.xy_tortuosity),
                    "vertex_count": int(len(vertices)),
                })
                for vertex_index, q in enumerate(vertices):
                    residual_vertices.append({
                        "file_id": file_id,
                        "component_id": cid,
                        "slice_seq": int(slice_seq),
                        "vertex_index": int(vertex_index),
                        "x": float(q[0]),
                        "y": float(q[1]),
                        "z": float(q[2]),
                    })

        combined_components = pd.concat(
            [base_frame, candidate_components],
            ignore_index=True,
            sort=False,
        ) if not candidate_components.empty else base_frame
        baseline_lines = baseline.get("lines", pd.DataFrame(columns=LINE_OUTPUT_COLUMNS)).copy()
        if residual_lines:
            combined_lines = pd.concat(
                [baseline_lines, pd.DataFrame(residual_lines, columns=LINE_OUTPUT_COLUMNS)],
                ignore_index=True,
            )
        else:
            combined_lines = baseline_lines

        baseline_vertices = baseline.get("vertices", pd.DataFrame(columns=VERTEX_OUTPUT_COLUMNS)).copy()
        if residual_vertices:
            combined_vertices = pd.concat(
                [baseline_vertices, pd.DataFrame(residual_vertices, columns=VERTEX_OUTPUT_COLUMNS)],
                ignore_index=True,
            )
        else:
            combined_vertices = baseline_vertices

        combined_line_points = dict(base_line_points)
        combined_line_points.update(selected_points)
        combined_raw = dict(base_raw)
        combined_raw["line_points"] = combined_line_points
        if not residual_frame.empty:
            raw_lines = base_raw.get("lines", pd.DataFrame())
            combined_raw["lines"] = pd.concat(
                [raw_lines, residual_frame],
                ignore_index=True,
                sort=False,
            )
        candidate_counts = dict(base_raw.get("candidate_counts", {}))
        candidate_counts.update(detail.get("audit", {}))
        candidate_counts.update({
            "production_accepted_line_components": int(len(base_line_ids)),
            "residual_candidates": int(len(candidate_components)),
            "residual_accepted_components": int(len(selected_ids)),
            "residual_novel_voxels": int(sum(len(_key_set(points, self.grid) - baseline_keys) for points in selected_points.values())),
        })
        combined_raw["candidate_counts"] = candidate_counts

        final_keys = set(baseline_keys)
        for points in selected_points.values():
            final_keys.update(_key_set(points, self.grid))
        stage1_labels = np.asarray(detail["stage1_labels"], dtype=np.int8)
        stage1_line_keys = _key_set(np.asarray(item["coords"])[stage1_labels == 2], self.grid)
        residual_keys: set[int] = set()
        for points in selected_points.values():
            residual_keys.update(_key_set(points, self.grid))
        if not residual_keys.issubset(stage1_line_keys):
            raise RuntimeError("Residual union emitted a voxel not labelled line by deployed Stage1")
        if not baseline_keys.issubset(final_keys):
            raise RuntimeError("Residual union removed a production-accepted line voxel")

        audit = {
            **detail["audit"],
            "processor": "production_baseline_plus_stage1_bundle_consensus_v7",
            "production_accepted_line_components": int(len(base_line_ids)),
            "production_accepted_line_voxels": int(len(baseline_keys)),
            "residual_candidate_components": int(len(candidate_components)),
            "residual_accepted_components": int(len(selected_ids)),
            "residual_accepted_novel_voxels": int(len(final_keys - baseline_keys)),
            "final_accepted_line_voxels": int(len(final_keys)),
            "accepted_stage1_line_voxels": int(len(final_keys & stage1_line_keys)),
            "stage1_to_stage2_voxel_preservation": float(len(final_keys & stage1_line_keys) / max(len(stage1_line_keys), 1)),
            "accepted_line_components": int(len(base_line_ids) + len(selected_ids)),
            "production_voxel_preserved": bool(baseline_keys.issubset(final_keys)),
            "runtime_gt_usage": False,
            "synthetic_line_voxels": 0,
            "residual_candidate_audit": candidate_audit,
        }

        return {
            "raw_components": combined_raw,
            "components": combined_components,
            "poles": baseline.get("poles", pd.DataFrame(columns=POLE_OUTPUT_COLUMNS)),
            "lines": combined_lines,
            "vertices": combined_vertices,
            "stage1_labels": stage1_labels,
            "stage1_label_audit": audit,
            "timing": {
                "stage2_production_ms": float(baseline_ms),
                "stage2_residual_candidate_ms": float(candidate_ms),
                "stage2_component_ms": float(baseline.get("timing", {}).get("stage2_component_ms", 0.0)) + float(candidate_ms),
                "stage2_refiner_parametric_ms": float(baseline.get("timing", {}).get("stage2_refiner_parametric_ms", 0.0)),
                "stage2_total_ms": float((time.perf_counter() - total_start) * 1000.0),
            },
        }
