#!/usr/bin/env python3
"""V4 realtime Stage 3 clean-track reconstruction.

Stage 1/2 outputs are reused. Only Stage 3 uses slice centers/world coordinates.

Goals of this pass:
- never create zig-zag joins, self-loops, triangles, or polygonal conductor paths;
- each fragment endpoint can join at most one other fragment endpoint;
- reject joins that require lateral track switching or implausible Z switching;
- complete high-confidence fragmented conductors when two real poles bracket the same span corridor;
- infer an occluded pole only when two independently pole-anchored, non-parallel partial spans converge;
- keep pole heights inside a robust session height range instead of stretching poles to bad attachments;
- render every pole black and every conductor cyan.

Missing slice numbers remain valid. SPAN/PARTIAL_SPAN/OPEN_CONDUCTOR are retained in CSV
but are not represented with different plot colors.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

def atomic_json(obj, path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True))
    os.replace(tmp,path)


POLE_COLUMNS = [
    "file_id", "component_id", "slice_seq", "refiner_probability", "touches_xy_edge",
    "radius_p90_ft", "verticality", "base_x", "base_y", "base_z", "top_x", "top_y",
    "top_z", "height_ft", "tilt_ft",
]
LINE_COLUMNS = [
    "file_id", "component_id", "slice_seq", "refiner_probability", "horizontal_span_ft",
    "vertical_span_ft", "verticality", "tortuosity", "vertex_count",
]
VERTEX_COLUMNS = ["file_id", "component_id", "slice_seq", "vertex_index", "x", "y", "z"]
WORLD_POLE_COLUMNS = [
    "world_pole_id", "group_id", "world_x_ft", "world_y_ft",
    "base_x_ft", "base_y_ft", "base_z_ft", "top_x_ft", "top_y_ft", "top_z_ft",
    "height_ft", "observed_height_ft", "baseline_height_ft", "height_adjustment_ft",
    "min_allowed_height_ft", "max_allowed_height_ft",
    "attachment_min_z_ft", "attachment_max_z_ft",
    "pole_origin", "inferred_from_lines", "inference_support_count",
    "inference_supporting_tracks", "inference_supporting_slices", "inference_min_line_angle_deg",
    "source_components", "source_slices", "attached_chain_count",
]
CHAIN_COLUMNS = [
    "chain_id", "group_id", "chain_type", "pole1_id", "pole2_id", "visible_pole_supports",
    "span_length_ft", "horizontal_extent_ft", "vertical_extent_ft", "slice_min", "slice_max",
    "slice_range", "observed_slice_count", "observed_slices", "source_line_segments",
    "endpoint1_pole_dist_ft", "endpoint2_pole_dist_ft", "fragment_count", "accepted_join_count",
    "max_join_xy_ft", "max_join_z_ft", "max_join_lateral_ft", "smoothed_geometry",
    "span_completion_used", "span_completion_track_count",
]
VERTEX_WORLD_COLUMNS = ["chain_id", "group_id", "vertex_index", "world_x_ft", "world_y_ft", "world_z_ft"]
SUMMARY_COLUMNS = [
    "group_id", "source_pole_observations", "observed_merged_poles", "inferred_hidden_poles",
    "merged_poles", "session_median_observed_pole_height_ft",
    "line_segments", "accepted_fragment_joins", "conductor_chains", "spans", "partial_spans",
    "open_conductors", "pole_height_adjustments", "detached_fragment_bridges",
    "span_completion_paths", "height_attachments_rejected", "polygon_connections_prevented",
    "fragment_join_seconds", "span_completion_pre_seconds", "hidden_pole_seconds",
    "span_completion_post_seconds", "chain_build_and_attachment_seconds", "output_write_seconds",
    "elapsed_seconds",
]
JOIN_AUDIT_COLUMNS=["group_id","i","j","i_end","j_end","slice_gap","xy_gap_ft","z_gap_ft","distance_ft","lateral_ft","connector_angle_a_deg","connector_angle_b_deg","longitudinal_overlap_ft","join_mode","cost","source_a","source_b"]
HIDDEN_POLE_AUDIT_COLUMNS=["group_id","world_pole_id","world_x_ft","world_y_ft","supporting_track_count","supporting_tracks","supporting_anchor_poles","supporting_slices","min_line_angle_deg","max_attachment_z_spread_ft","max_endpoint_extrapolation_ft","nearest_slice_center_distance_ft","rule"]
SPAN_COMPLETION_AUDIT_COLUMNS=["group_id","pole1_id","pole2_id","source_track_indices","support_track_count","observed_coverage_fraction","span_length_ft","max_bridge_gap_ft","max_lateral_ft","max_z_error_ft","rule"]
POLYGON_AUDIT_COLUMNS=["group_id","track_idx","pole_start_id","pole_end_id","dropped_endpoint","reason","track_evidence_score"]
SKIPPED_CENTER_COLUMNS=["relative_path","reason"]
RESUME_AUDIT_COLUMNS=["group_id","resumed","detailed_audit_complete","note"]


def pa():
    p = argparse.ArgumentParser()
    p.add_argument("--inference_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--metadata_dir", default=None)
    p.add_argument("--centers_csv", default=None)
    p.add_argument("--world_units_to_ft", type=float, default=0.5)
    p.add_argument("--pole_merge_radius_ft", type=float, default=4.0)
    p.add_argument("--min_pole_separation_ft", type=float, default=10.0)
    p.add_argument("--max_span_length_ft", type=float, default=450.0)
    p.add_argument("--max_span_slices", type=int, default=9)
    p.add_argument("--slice_length_ft", type=float, default=50.0)

    # Fragment-to-fragment track stitching.
    p.add_argument("--fragment_join_radius_ft", type=float, default=14.0)
    p.add_argument("--missing_slice_extra_ft", type=float, default=8.0)
    p.add_argument("--max_join_angle_deg", type=float, default=18.0)
    p.add_argument("--max_connector_angle_deg", type=float, default=25.0)
    p.add_argument("--max_join_lateral_ft", type=float, default=1.5)
    p.add_argument("--max_join_vertical_ft", type=float, default=14.0)
    p.add_argument("--max_join_vertical_horizontal_ratio", type=float, default=0.35)
    p.add_argument("--max_z_extrap_error_ft", type=float, default=2.5)
    p.add_argument("--max_longitudinal_overlap_ft", type=float, default=3.0)
    p.add_argument("--overlap_xy_radius_ft", type=float, default=2.0)
    p.add_argument("--max_overlap_vertical_ft", type=float, default=2.0)

    # Conservative second-pass bridge for a genuinely detached fragment that sits
    # between two otherwise continuous conductor pieces. These tolerances are
    # intentionally tighter than the normal join tolerances except for distance.
    p.add_argument("--fragment_bridge_radius_ft", type=float, default=30.0)
    p.add_argument("--bridge_max_join_angle_deg", type=float, default=10.0)
    p.add_argument("--bridge_max_connector_angle_deg", type=float, default=15.0)
    p.add_argument("--bridge_max_lateral_ft", type=float, default=1.0)
    p.add_argument("--bridge_max_z_extrap_error_ft", type=float, default=2.0)

    # Topology guards. A conductor is an open path, never a loop/polygon.
    # Multiple conductors between the same pole pair are allowed, but a new
    # connection that closes a cycle through 3+ poles is not synthesized.
    p.add_argument("--disable_pole_graph_cycle_guard", action="store_true")
    p.add_argument("--self_intersection_tolerance_ft", type=float, default=0.25)

    # Pole attachment: XY/direction driven, not fixed-top-Z driven.
    p.add_argument("--pole_attachment_radius_ft", type=float, default=28.0)
    p.add_argument("--pole_attachment_close_ft", type=float, default=4.0)
    p.add_argument("--max_pole_attachment_angle_deg", type=float, default=40.0)
    p.add_argument("--max_pole_attachment_height_delta_ft", type=float, default=18.0)
    p.add_argument("--pole_top_margin_ft", type=float, default=1.5)
    p.add_argument("--allowed_pole_height_variation_ft", type=float, default=8.0)
    p.add_argument("--max_pole_height_adjust_ft", type=float, default=8.0)
    p.add_argument("--min_pole_height_ft", type=float, default=15.0)
    p.add_argument("--fixed_pole_height_ft", type=float, default=None)
    p.add_argument("--pole_height_quantile_low", type=float, default=0.10)
    p.add_argument("--pole_height_quantile_high", type=float, default=0.90)
    p.add_argument("--pole_height_range_margin_ft", type=float, default=2.0)
    p.add_argument("--pole_attachment_height_slack_ft", type=float, default=1.5)

    # Span-backed completion.  This is deliberately stronger than ordinary fragment
    # joining because a valid pole pair supplies two physical anchors.  Tracks are
    # still required to be monotonic, nearly collinear, Z-consistent, and disjoint.
    p.add_argument("--span_completion", action="store_true", default=True)
    p.add_argument("--disable_span_completion", action="store_true")
    p.add_argument("--span_completion_max_angle_deg", type=float, default=12.0)
    p.add_argument("--span_completion_max_connector_angle_deg", type=float, default=15.0)
    p.add_argument("--span_completion_max_lateral_ft", type=float, default=1.5)
    p.add_argument("--span_completion_max_z_error_ft", type=float, default=3.0)
    p.add_argument("--span_completion_max_gap_ft", type=float, default=100.0)
    p.add_argument("--span_completion_corridor_ft", type=float, default=10.0)
    p.add_argument("--span_completion_pole_extrap_ft", type=float, default=35.0)
    p.add_argument("--span_completion_min_tracks", type=int, default=2)
    p.add_argument("--span_completion_min_coverage", type=float, default=0.15)

    # Hidden-pole inference. A pole is only synthesized when >=2 distinct,
    # non-parallel conductor tracks terminate toward a common point that lies
    # inside observed slice support. Single open conductors never create poles.
    p.add_argument("--infer_hidden_poles", action="store_true", default=True)
    p.add_argument("--disable_hidden_pole_inference", action="store_true")
    p.add_argument("--hidden_pole_min_line_angle_deg", type=float, default=20.0)
    p.add_argument("--hidden_pole_max_endpoint_extrap_ft", type=float, default=20.0)
    p.add_argument("--hidden_pole_ray_backtrack_tolerance_ft", type=float, default=1.0)
    p.add_argument("--hidden_pole_max_attachment_z_spread_ft", type=float, default=15.0)
    p.add_argument("--hidden_pole_slice_support_radius_ft", type=float, default=35.0)
    p.add_argument("--hidden_pole_existing_pole_exclusion_ft", type=float, default=6.0)
    p.add_argument("--hidden_pole_cluster_radius_ft", type=float, default=4.0)
    p.add_argument("--hidden_pole_ground_reference_radius_ft", type=float, default=150.0)
    p.add_argument("--hidden_pole_min_track_evidence", type=float, default=2.0)
    p.add_argument("--hidden_pole_require_distinct_anchor_poles", type=int, default=1)

    # Clean geometry output.
    p.add_argument("--smooth_spacing_ft", type=float, default=2.0)
    p.add_argument("--min_chain_horizontal_ft", type=float, default=3.0)
    p.add_argument("--max_vertical_horizontal_ratio", type=float, default=1.0)
    p.add_argument("--session_filter", default=None)
    p.add_argument("--latest_slice", type=int, default=None)
    p.add_argument("--resume_sessions", action="store_true", help="Skip sessions with complete per-session outputs and rebuild aggregates from them.")
    p.add_argument("--disable_plots", action="store_true", help="Skip PNG rendering; recommended for incremental realtime updates. Reconstruction CSV/audit math is unchanged.")
    return p.parse_args()


def frame(rows, columns):
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).reindex(columns=columns)


def atomic_csv(df, path):
    """Write a CSV atomically so an interrupted process cannot leave a partial final file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _session_core_outputs_complete(gdir):
    gdir = Path(gdir)
    required = [
        gdir / "summary.json",
        gdir / "world_poles.csv",
        gdir / "conductor_chains.csv",
        gdir / "conductor_vertices.csv",
        gdir / "spans.csv",
    ]
    return all(x.exists() and x.stat().st_size > 0 for x in required)


def _read_csv_or_empty(path, columns):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    try:
        d = pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame(columns=columns)
    return d.reindex(columns=columns)


def _chain_counter_from_frame(cdf):
    best = 0
    if cdf is None or cdf.empty or "chain_id" not in cdf.columns:
        return best
    for x in cdf["chain_id"].dropna().astype(str):
        m = re.search(r"/C(\d+)$", x)
        if m:
            best = max(best, int(m.group(1)))
    return best


def _join_distance_for_mode(a, slice_gap, mode):
    gap = abs(int(slice_gap))
    missing = max(0, gap - 1)
    base = a.fragment_bridge_radius_ft if mode == "detached_bridge" else a.fragment_join_radius_ft
    return min(
        a.max_span_length_ft,
        base + missing * a.slice_length_ft + (a.missing_slice_extra_ft if missing else 0.0),
    )


def _candidate_fragment_pairs(frags, a, mode):
    """Spatially prune fragment pairs before the expensive geometry tests.

    The previous implementation tested every fragment pair twice. Dense sessions with
    thousands of line segments therefore became quadratic before any geometry could
    reject them. This index is exact with respect to the endpoint-distance gate: a pair
    is considered whenever at least one endpoint pair lies within the same distance
    that best_fragment_join() would permit for that slice gap.
    """
    if len(frags) < 2:
        return []
    by_slice = {}
    for i, f in enumerate(frags):
        seq = int(f["slice_seq"])
        pts = np.asarray(f["points"], float)
        if len(pts) < 2:
            continue
        for end_idx, q in ((0, pts[0]), (1, pts[-1])):
            by_slice.setdefault(seq, []).append((i, end_idx, np.asarray(q, float)))
    trees = {}
    for seq, rows in by_slice.items():
        xyz = np.vstack([r[2] for r in rows])
        trees[seq] = (rows, cKDTree(xyz))
    pairs = set()
    for i, f in enumerate(frags):
        seq = int(f["slice_seq"])
        pts = np.asarray(f["points"], float)
        if len(pts) < 2:
            continue
        for target_seq, (rows, tree) in trees.items():
            gap = abs(seq - int(target_seq))
            if gap > a.max_span_slices:
                continue
            radius = _join_distance_for_mode(a, gap, mode)
            for q in (pts[0], pts[-1]):
                for pos in tree.query_ball_point(np.asarray(q, float), radius):
                    j = int(rows[pos][0])
                    if j == i:
                        continue
                    aidx, bidx = (i, j) if i < j else (j, i)
                    pairs.add((aidx, bidx))
    return sorted(pairs)


def _fragment_midpoint_index(frags):
    mids = []
    valid_ids = []
    for i, f in enumerate(frags):
        pts = np.asarray(f["points"], float)
        if len(pts) == 0:
            continue
        mids.append(np.median(pts, axis=0))
        valid_ids.append(i)
    if not mids:
        return None, np.empty((0, 3)), []
    arr = np.vstack(mids)
    return cKDTree(arr[:, :2]), arr, valid_ids


def center_map(a, manifest):
    m = {}
    skipped = []
    if a.centers_csv:
        try:
            t = pd.read_csv(a.centers_csv)
        except EmptyDataError:
            t = pd.DataFrame(columns=["relative_path","center_x","center_y","center_z"])
        required={"relative_path","center_x","center_y","center_z"}
        if not t.empty and not required.issubset(t.columns):
            skipped.append({"relative_path":"<centers_csv>","reason":"centers_csv_missing_required_columns:"+",".join(sorted(required-set(t.columns)))})
        else:
            for _, r in t.iterrows():
                try:
                    vals=(float(r.center_x),float(r.center_y),float(r.center_z))
                    if np.isfinite(vals).all(): m[str(r.relative_path)] = vals
                except Exception:
                    skipped.append({"relative_path":str(getattr(r,"relative_path","<unknown>")),"reason":"invalid_centers_csv_row"})
    root = Path(a.metadata_dir).resolve() if a.metadata_dir else None
    manifest_has_centers = {"center_x", "center_y", "center_z"}.issubset(manifest.columns)
    for _, r in manifest.iterrows():
        rel = str(r.relative_path)
        if rel in m:
            continue
        # Optimized inference writes Stage-3-only center metadata directly into
        # inference_manifest.csv. Prefer it so reconstruction does not reopen
        # thousands of source CSVs merely to recover three constants.
        if manifest_has_centers:
            try:
                vals = (float(r.center_x), float(r.center_y), float(r.center_z))
                if np.isfinite(vals).all():
                    m[rel] = vals
                    continue
            except Exception:
                pass
        if root is None:
            skipped.append({"relative_path": rel, "reason": "no_center_metadata"})
            continue
        p = root / rel
        if not p.exists():
            skipped.append({"relative_path": rel, "reason": "metadata_file_missing"})
            continue
        try:
            d = pd.read_csv(p, nrows=256, usecols=lambda c: c in {"center_x", "center_y", "center_z"})
            if not {"center_x", "center_y", "center_z"}.issubset(d.columns):
                raise ValueError("center columns missing")
            vals = []
            for c in ("center_x", "center_y", "center_z"):
                x = pd.to_numeric(d[c], errors="coerce")
                x = x[np.isfinite(x)]
                if x.empty:
                    raise ValueError(c + " has no finite values")
                vals.append(float(x.median()))
            m[rel] = tuple(vals)
        except Exception as e:
            skipped.append({"relative_path": rel, "reason": str(e)})
    return m, skipped


def read_stage2_csv(path, columns, kind, audit):
    path = Path(path)
    if not path.exists():
        audit.append({"kind": kind, "path": str(path), "reason": "missing", "missing_columns": ""})
        return pd.DataFrame(columns=columns)
    try:
        if path.stat().st_size == 0:
            audit.append({"kind": kind, "path": str(path), "reason": "empty_file", "missing_columns": ""})
            return pd.DataFrame(columns=columns)
    except OSError as e:
        audit.append({"kind": kind, "path": str(path), "reason": f"stat_error:{e}", "missing_columns": ""})
        return pd.DataFrame(columns=columns)
    try:
        d = pd.read_csv(path)
    except EmptyDataError:
        audit.append({"kind": kind, "path": str(path), "reason": "no_columns", "missing_columns": ""})
        return pd.DataFrame(columns=columns)
    except Exception as e:
        audit.append({"kind": kind, "path": str(path), "reason": f"read_error:{type(e).__name__}:{e}", "missing_columns": ""})
        return pd.DataFrame(columns=columns)
    missing = [c for c in columns if c not in d.columns]
    if missing:
        audit.append({"kind": kind, "path": str(path), "reason": "missing_columns", "missing_columns": ";".join(missing)})
        return pd.DataFrame(columns=columns)
    return d


def unit(v, fallback=None):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.asarray(fallback if fallback is not None else [1.0, 0.0], float)
    return v / n


def angle_deg(a, b, absolute=False):
    a = unit(a)
    b = unit(b)
    dot = float(np.dot(a, b))
    if absolute:
        dot = abs(dot)
    return math.degrees(math.acos(np.clip(dot, -1.0, 1.0)))


def order_xy(p):
    p = np.asarray(p, float)
    if len(p) < 2:
        return p
    xy = p[:, :2]
    c = np.median(xy, axis=0)
    try:
        _, _, vh = np.linalg.svd(xy - c, full_matrices=False)
        d = unit(vh[0])
    except np.linalg.LinAlgError:
        d = np.array([1.0, 0.0])
    return p[np.argsort((xy - c) @ d)]


def fragment_direction(points):
    p = order_xy(points)
    if len(p) < 2:
        return np.array([1.0, 0.0])
    return unit(p[-1, :2] - p[0, :2])


def endpoint_outward_tangent(points, end_idx):
    p = np.asarray(points, float)
    if len(p) < 2:
        return np.array([1.0, 0.0])
    k = min(4, len(p) - 1)
    if end_idx == 0:
        # From interior toward first endpoint.
        return unit(p[0, :2] - p[k, :2])
    return unit(p[-1, :2] - p[-1-k, :2])


def endpoint_z_slope(points, end_idx):
    p = np.asarray(points, float)
    if len(p) < 2:
        return 0.0
    k = min(5, len(p) - 1)
    if end_idx == 0:
        a, b = p[k], p[0]
    else:
        a, b = p[-1-k], p[-1]
    ds = float(np.linalg.norm(b[:2] - a[:2]))
    if ds < 0.5:
        return 0.0
    return float((b[2] - a[2]) / ds)


def projected_overlap_ft(A, B, axis):
    axis = unit(axis)
    sa = A["points"][:, :2] @ axis
    sb = B["points"][:, :2] @ axis
    return float(min(sa.max(), sb.max()) - max(sa.min(), sb.min()))


def max_join_distance(a, slice_gap):
    # Missing slice numbers can create real gaps; alignment tests remain strict.
    missing = max(0, int(slice_gap) - 1)
    return min(a.max_span_length_ft, a.fragment_join_radius_ft + missing * a.slice_length_ft + (a.missing_slice_extra_ft if missing else 0.0))


def best_fragment_join(i, j, frags, a, mode="strict"):
    A, B = frags[i], frags[j]
    gap = abs(int(A["slice_seq"]) - int(B["slice_seq"]))
    if gap > a.max_span_slices:
        return None

    if mode == "detached_bridge":
        max_join_angle = a.bridge_max_join_angle_deg
        max_connector_angle = a.bridge_max_connector_angle_deg
        max_lateral = a.bridge_max_lateral_ft
        max_zerr = a.bridge_max_z_extrap_error_ft
        missing = max(0, int(gap) - 1)
        allowed_distance = min(
            a.max_span_length_ft,
            a.fragment_bridge_radius_ft + missing * a.slice_length_ft +
            (a.missing_slice_extra_ft if missing else 0.0),
        )
    else:
        max_join_angle = a.max_join_angle_deg
        max_connector_angle = a.max_connector_angle_deg
        max_lateral = a.max_join_lateral_ft
        max_zerr = a.max_z_extrap_error_ft
        allowed_distance = max_join_distance(a, gap)

    if angle_deg(A["direction"], B["direction"], absolute=True) > max_join_angle:
        return None

    best = None
    for ae in (0, 1):
        for be in (0, 1):
            pa = A["points"][0 if ae == 0 else -1]
            pb = B["points"][0 if be == 0 else -1]
            d = pb - pa
            xy = float(np.linalg.norm(d[:2]))
            dz = abs(float(d[2]))
            dist = float(np.linalg.norm(d))
            if dist > allowed_distance:
                continue

            ta = endpoint_outward_tangent(A["points"], ae)
            tb = endpoint_outward_tangent(B["points"], be)
            conn = unit(d[:2], ta)
            connector_a = angle_deg(ta, conn)
            connector_b = angle_deg(tb, -conn)

            avg = unit(ta - tb, ta)
            normal = np.array([-avg[1], avg[0]])
            lateral = abs(float(np.dot(d[:2], normal)))
            overlap = projected_overlap_ft(A, B, avg)

            # Even if two endpoints happen to be close, fragments with substantial
            # longitudinal overlap are parallel/duplicate conductors, not sequential
            # pieces. Joining them is a common source of triangular/zig-zag artifacts.
            if overlap > a.max_longitudinal_overlap_ft:
                continue

            if xy <= a.overlap_xy_radius_ft:
                if dz > a.max_overlap_vertical_ft:
                    continue
            else:
                if connector_a > max_connector_angle or connector_b > max_connector_angle:
                    continue
                if lateral > max_lateral:
                    continue
                if dz > a.max_join_vertical_ft:
                    continue
                if dz / max(xy, 1e-6) > a.max_join_vertical_horizontal_ratio:
                    continue
                slope_a = endpoint_z_slope(A["points"], ae)
                slope_b = endpoint_z_slope(B["points"], be)
                pred_b = float(pa[2] + slope_a * xy)
                pred_a = float(pb[2] + slope_b * xy)
                zerr = max(abs(float(pb[2]) - pred_b), abs(float(pa[2]) - pred_a))
                if zerr > max_zerr:
                    continue

            cost = (
                xy / max(a.fragment_join_radius_ft if mode == "strict" else a.fragment_bridge_radius_ft, 1.0)
                + 0.35 * gap
                + 0.7 * lateral / max(max_lateral, 0.25)
                + 0.5 * dz / max(a.max_join_vertical_ft, 1.0)
                + 0.35 * (connector_a + connector_b) / max(2 * max_connector_angle, 1.0)
                + (0.20 if mode == "detached_bridge" else 0.0)
            )
            row = {
                "i": i, "j": j, "i_end": ae, "j_end": be, "slice_gap": gap,
                "xy_gap_ft": xy, "z_gap_ft": dz, "distance_ft": dist,
                "lateral_ft": lateral, "connector_angle_a_deg": connector_a,
                "connector_angle_b_deg": connector_b, "longitudinal_overlap_ft": overlap,
                "join_mode": mode, "cost": cost,
            }
            if best is None or cost < best["cost"]:
                best = row
    return best


class UF:
    def __init__(self, n):
        self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a


def candidate_skips_intervening_fragment(c, frags, midpoint_index=None, lateral_tol_ft=3.0, z_tol_ft=5.0):
    """Reject a jump that bypasses an observed fragment between its endpoints.

    A KD-tree midpoint index limits the exact check to fragments that can actually
    lie near the candidate segment. This preserves the previous geometric rule while
    removing an O(N) scan for every proposed join.
    """
    A, B = frags[c["i"]], frags[c["j"]]
    pa = A["points"][0 if c["i_end"] == 0 else -1]
    pb = B["points"][0 if c["j_end"] == 0 else -1]
    v = pb[:2] - pa[:2]
    L2 = float(np.dot(v, v))
    if L2 < 4.0:
        return False
    if midpoint_index is None or midpoint_index[0] is None:
        candidate_ids = range(len(frags))
        midpoint_lookup = None
    else:
        tree, mids, valid_ids = midpoint_index
        center = 0.5 * (pa[:2] + pb[:2])
        radius = math.hypot(0.5 * math.sqrt(L2), lateral_tol_ft) + 0.5
        positions = tree.query_ball_point(center, radius)
        candidate_ids = [valid_ids[pos] for pos in positions]
        midpoint_lookup = {valid_ids[pos]: mids[pos] for pos in positions}
    for k in candidate_ids:
        if k in (c["i"], c["j"]):
            continue
        if midpoint_lookup is not None and k in midpoint_lookup:
            mid = midpoint_lookup[k]
        else:
            pts = np.asarray(frags[k]["points"], float)
            if len(pts) == 0:
                continue
            mid = np.median(pts, axis=0)
        t = float(np.dot(mid[:2] - pa[:2], v) / L2)
        if not (0.05 < t < 0.95):
            continue
        proj = pa[:2] + t * v
        lateral = float(np.linalg.norm(mid[:2] - proj))
        pred_z = float(pa[2] + t * (pb[2] - pa[2]))
        if lateral <= lateral_tol_ft and abs(float(mid[2]) - pred_z) <= z_tol_ft:
            return True
    return False


def select_one_to_one_joins(frags, a):
    """Build open, non-branching conductor paths using spatially pruned candidates.

    Endpoint degree remains <=1 and the fragment graph remains acyclic. The spatial
    index changes only how candidate pairs are found; best_fragment_join() still
    applies the full direction/lateral/Z/overlap checks.
    """
    endpoint_used = set()
    uf = UF(len(frags))
    accepted = []
    midpoint_index = _fragment_midpoint_index(frags)

    def consume(mode):
        candidates = []
        pair_ids = _candidate_fragment_pairs(frags, a, mode)
        for i, j in pair_ids:
            if uf.find(i) == uf.find(j):
                continue
            c = best_fragment_join(i, j, frags, a, mode=mode)
            if c is not None and not candidate_skips_intervening_fragment(c, frags, midpoint_index=midpoint_index):
                candidates.append(c)
        candidates.sort(key=lambda r: r["cost"])
        for c in candidates:
            ka = (c["i"], c["i_end"])
            kb = (c["j"], c["j_end"])
            if ka in endpoint_used or kb in endpoint_used:
                continue
            if uf.find(c["i"]) == uf.find(c["j"]):
                continue
            endpoint_used.add(ka)
            endpoint_used.add(kb)
            uf.union(c["i"], c["j"])
            accepted.append(c)

    consume("strict")
    consume("detached_bridge")
    return accepted


def build_track_components(frags, accepted):
    adj = {i: [] for i in range(len(frags))}
    for eidx, c in enumerate(accepted):
        adj[c["i"]].append((c["j"], c["i_end"], c["j_end"], eidx))
        adj[c["j"]].append((c["i"], c["j_end"], c["i_end"], eidx))

    seen = set()
    tracks = []
    for seed in range(len(frags)):
        if seed in seen:
            continue
        stack = [seed]
        comp = []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            for y, _, _, _ in adj[x]:
                if y not in seen:
                    stack.append(y)

        if len(comp) == 1:
            pts = frags[comp[0]]["points"].copy()
            tracks.append({"frags": comp, "points": pts, "join_ids": []})
            continue

        starts = [i for i in comp if len(adj[i]) == 1]
        cur = min(starts if starts else comp, key=lambda i: frags[i]["slice_seq"])
        prev = None
        ordered_points = []
        ordered_frags = []
        join_ids = []

        while cur is not None:
            neighbors = [x for x in adj[cur] if x[0] != prev]
            prev_connection_end = None
            if prev is not None:
                for nb, my_end, _, eidx in adj[cur]:
                    if nb == prev:
                        prev_connection_end = my_end
                        break
            p = frags[cur]["points"].copy()
            if prev is None:
                # For the first fragment, orient the connected endpoint to the end.
                if neighbors:
                    my_end = neighbors[0][1]
                    if my_end == 0:
                        p = p[::-1].copy()
            else:
                # Incoming connection must be the first endpoint.
                if prev_connection_end == 1:
                    p = p[::-1].copy()

            if ordered_points:
                # No invented zig-zag path: only append the next fragment's observations.
                # The smooth refit below bridges the unobserved gap.
                pass
            ordered_points.extend(p.tolist())
            ordered_frags.append(cur)

            if not neighbors:
                break
            nxt, _, _, eidx = neighbors[0]
            join_ids.append(eidx)
            prev, cur = cur, nxt
            if cur in ordered_frags:
                break

        tracks.append({
            "frags": ordered_frags,
            "points": np.asarray(ordered_points, float),
            "join_ids": join_ids,
        })
    return tracks


def robust_linear(x, y, degree=1, iterations=5):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < degree + 1 or np.ptp(x) < 1e-6:
        return np.array([0.0, float(np.nanmedian(y))]) if degree == 1 else np.array([0.0, 0.0, float(np.nanmedian(y))])
    X = np.vander(x, N=degree + 1)
    w = np.ones(len(x))
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(iterations):
        pred = X @ coef
        r = y - pred
        scale = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        u = np.abs(r) / (2.5 * scale)
        w = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-6))
        WX = X * w[:, None]
        coef = np.linalg.lstsq(WX, y * w, rcond=None)[0]
    return coef


def eval_poly(coef, x):
    return np.polyval(coef, np.asarray(x, float))


def track_axis(points, pole_xy1=None, pole_xy2=None):
    p = np.asarray(points, float)
    if pole_xy1 is not None and pole_xy2 is not None:
        d = np.asarray(pole_xy2, float) - np.asarray(pole_xy1, float)
        if np.linalg.norm(d) > 1e-6:
            return unit(d), np.asarray(pole_xy1, float)
    xy = p[:, :2]
    origin = np.median(xy, axis=0)
    try:
        _, _, vh = np.linalg.svd(xy - origin, full_matrices=False)
        axis = unit(vh[0])
    except np.linalg.LinAlgError:
        axis = np.array([1.0, 0.0])
    return axis, origin


def endpoint_tangent_from_track(points, at_start=True):
    p = np.asarray(points, float)
    if len(p) < 2:
        return np.array([1.0, 0.0])
    k = min(8, len(p) - 1)
    if at_start:
        return unit(p[0, :2] - p[k, :2])
    return unit(p[-1, :2] - p[-1-k, :2])


def build_pole_xy_index(poles):
    """Build a compact XY search index without changing any attachment rules."""
    if poles is None or poles.empty:
        return None
    idx = poles.index.to_numpy()
    xy = poles[["world_x_ft", "world_y_ft"]].to_numpy(float)
    good = np.isfinite(xy).all(axis=1)
    if not np.any(good):
        return None
    return {"tree": cKDTree(xy[good]), "indices": idx[good]}


def pole_attachment_candidate(endpoint, outward_tangent, poles, a, exclude=None, pole_index=None):
    """Return a geometrically supported pole attachment without stretching the pole unrealistically."""
    if poles.empty:
        return None
    exclude = set(exclude or [])
    best = None
    if pole_index is not None:
        local = pole_index["tree"].query_ball_point(np.asarray(endpoint[:2], float), r=float(a.pole_attachment_radius_ft))
        candidate_indices = [pole_index["indices"][j] for j in local]
    else:
        candidate_indices = list(poles.index)
    for idx in candidate_indices:
        if idx in exclude:
            continue
        r = poles.loc[idx]
        vec = np.array([float(r.world_x_ft) - endpoint[0], float(r.world_y_ft) - endpoint[1]])
        dxy = float(np.linalg.norm(vec))
        if dxy > a.pole_attachment_radius_ft:
            continue
        angle = 0.0 if dxy <= a.pole_attachment_close_ft else angle_deg(outward_tangent, vec)
        if dxy > a.pole_attachment_close_ft and angle > a.max_pole_attachment_angle_deg:
            continue

        base_z = float(r.base_z_ft)
        baseline_top = base_z + float(r.baseline_height_ft)
        max_h = float(r.get("max_allowed_height_ft", r.baseline_height_ft + a.max_pole_height_adjust_ft))
        max_top = base_z + max_h
        # Close XY is not sufficient evidence to make a 60-100 ft pole.  If the
        # conductor would require a pole above the robust session height range,
        # reject this attachment and leave the conductor partial/open.
        if float(endpoint[2]) > max_top + a.pole_attachment_height_slack_ft:
            continue
        if dxy > a.pole_attachment_close_ft and abs(float(endpoint[2]) - baseline_top) > a.max_pole_attachment_height_delta_ft:
            continue
        score = dxy + 0.10 * angle + 0.03 * max(0.0, float(endpoint[2]) - baseline_top)
        if best is None or score < best["score"]:
            best = {"idx": int(idx), "dxy": dxy, "angle": angle, "score": score}
    return best


def pole_height_bounds_from_observations(pg, session_median, a):
    if a.fixed_pole_height_ft is not None:
        h=float(a.fixed_pole_height_ft)
        return h,h
    vals=[]
    if pg is not None and not pg.empty and "height_ft" in pg.columns:
        x=pd.to_numeric(pg.height_ft,errors="coerce").to_numpy(float)
        vals=x[np.isfinite(x)&(x>0)].tolist()
    if len(vals)>=3:
        lo=float(np.quantile(vals, np.clip(a.pole_height_quantile_low,0,1))) - a.pole_height_range_margin_ft
        hi=float(np.quantile(vals, np.clip(a.pole_height_quantile_high,0,1))) + a.pole_height_range_margin_ft
    else:
        lo=float(session_median)-a.allowed_pole_height_variation_ft
        hi=float(session_median)+a.allowed_pole_height_variation_ft
    lo=max(float(a.min_pole_height_ft),lo)
    # Hard global guard around the robust median prevents outlier Stage-2 pole
    # components from defining an enormous permitted height range.
    hi=min(hi,float(session_median)+a.allowed_pole_height_variation_ft)
    hi=max(lo,hi)
    return lo,hi


def attachment_height_valid(pole_row, z, a):
    base=float(pole_row.base_z_ft)
    maxh=float(pole_row.get("max_allowed_height_ft",pole_row.baseline_height_ft+a.max_pole_height_adjust_ft))
    return float(z) <= base+maxh+a.pole_attachment_height_slack_ft


def _track_axis_features(raw, pole_a, pole_b):
    raw=np.asarray(raw,float)
    A=np.array([float(pole_a.world_x_ft),float(pole_a.world_y_ft)],float)
    B=np.array([float(pole_b.world_x_ft),float(pole_b.world_y_ft)],float)
    u=unit(B-A); n=np.array([-u[1],u[0]],float); L=float(np.linalg.norm(B-A))
    s=(raw[:,:2]-A)@u; lat=(raw[:,:2]-A)@n
    order=np.argsort(s); rr=raw[order]; ss=s[order]; ll=lat[order]
    zcoef=robust_linear(ss,rr[:,2],degree=1)
    return {"raw":rr,"s":ss,"lat":ll,"zcoef":zcoef,"u":u,"n":n,"A":A,"B":B,"L":L}


def _span_edge(fi,fj,a):
    """Directed, monotonic same-wire compatibility for two track fragments in one pole corridor."""
    si,sj=fi["s"],fj["s"]
    if len(si)<2 or len(sj)<2: return None
    gap=float(sj[0]-si[-1])
    if gap < -a.max_longitudinal_overlap_ft or gap > a.span_completion_max_gap_ft:
        return None
    # Parallel wires may occupy the same pole pair; never jump lanes.
    lat_i=float(np.median(fi["lat"])); lat_j=float(np.median(fj["lat"]))
    lateral=abs(lat_i-lat_j)
    if lateral>a.span_completion_max_lateral_ft: return None
    di=unit(fi["raw"][-1,:2]-fi["raw"][0,:2]); dj=unit(fj["raw"][-1,:2]-fj["raw"][0,:2])
    if angle_deg(di,dj,absolute=True)>a.span_completion_max_angle_deg: return None
    conn=fj["raw"][0,:2]-fi["raw"][-1,:2]
    if np.linalg.norm(conn)>1e-6 and angle_deg(fi["u"],conn,absolute=True)>a.span_completion_max_connector_angle_deg:
        return None
    s_mid=0.5*(float(si[-1])+float(sj[0]))
    zi=float(eval_poly(fi["zcoef"],s_mid)); zj=float(eval_poly(fj["zcoef"],s_mid))
    zerr=abs(zi-zj)
    if zerr>a.span_completion_max_z_error_ft: return None
    return {"gap_ft":max(0.0,gap),"lateral_ft":lateral,"z_error_ft":zerr}


def _pole_end_supported(feat,pole,at_start,a):
    raw=feat["raw"]; endpoint=raw[0] if at_start else raw[-1]
    target=np.array([float(pole.world_x_ft),float(pole.world_y_ft)])
    d=float(np.linalg.norm(target-endpoint[:2]))
    if d>a.span_completion_pole_extrap_ft: return False
    outward=endpoint_tangent_from_track(raw,at_start=at_start)
    vec=target-endpoint[:2]
    if d>1e-6 and angle_deg(outward,vec)>a.max_pole_attachment_angle_deg: return False
    return attachment_height_valid(pole,endpoint[2],a)


def complete_span_backed_tracks(track_records,poles,frags,a):
    """Merge partial/floating tracks only when a real pole pair brackets one coherent conductor lane.

    The pole pair makes long missing detections reconstructable, but the path remains a DAG in
    pole-to-pole progression, so this operation cannot create loops, triangles, or zig-zag rungs.
    """
    if a.disable_span_completion or not a.span_completion or len(track_records)<2 or len(poles)<2:
        return track_records,[]
    candidates=[]
    pole_indices=list(poles.index)
    pole_xy=poles.loc[pole_indices,["world_x_ft","world_y_ft"]].to_numpy(float)
    pole_tree=cKDTree(pole_xy)
    pole_pairs=sorted(pole_tree.query_pairs(r=a.max_span_length_ft))

    # Index three representatives per track (start/end/midpoint). A track only needs
    # detailed pole-pair geometry if at least one representative lies near that span.
    rep_xy=[]; rep_track=[]
    for ti,tr in enumerate(track_records):
        raw=np.asarray(tr["raw"],float)
        if len(raw)<2: continue
        for q in (raw[0,:2], raw[-1,:2], np.median(raw[:,:2],axis=0)):
            rep_xy.append(np.asarray(q,float)); rep_track.append(ti)
    rep_tree=cKDTree(np.vstack(rep_xy)) if rep_xy else None

    for ai,bi in pole_pairs:
            ia=pole_indices[ai]; A=poles.loc[ia]
            ib=pole_indices[bi]; B=poles.loc[ib]
            L=float(np.linalg.norm([float(B.world_x_ft-A.world_x_ft),float(B.world_y_ft-A.world_y_ft)]))
            if not (a.min_pole_separation_ft<=L<=a.max_span_length_ft): continue
            feats=[]
            if rep_tree is None:
                nearby_track_ids=[]
            else:
                mid=0.5*(pole_xy[ai]+pole_xy[bi])
                search_r=0.5*L+a.span_completion_corridor_ft+a.span_completion_pole_extrap_ft
                nearby_track_ids=sorted(set(rep_track[k] for k in rep_tree.query_ball_point(mid,search_r)))
            for ti in nearby_track_ids:
                tr=track_records[ti]
                raw=np.asarray(tr["raw"],float)
                if len(raw)<2: continue
                f=_track_axis_features(raw,A,B)
                # Candidate must progress along the pole-pair direction and remain inside one corridor.
                d=unit(f["raw"][-1,:2]-f["raw"][0,:2])
                if angle_deg(d,f["u"],absolute=True)>a.span_completion_max_angle_deg: continue
                if float(np.max(np.abs(f["lat"])))>a.span_completion_corridor_ft: continue
                if float(f["s"][-1]) < -a.span_completion_pole_extrap_ft or float(f["s"][0]) > L+a.span_completion_pole_extrap_ft: continue
                f.update({"track_idx":ti,"record":tr})
                feats.append(f)
            if len(feats)<a.span_completion_min_tracks: continue
            feats.sort(key=lambda f: float(f["s"][0]))
            N=len(feats)
            start=[_pole_end_supported(f,A,True,a) for f in feats]
            end=[_pole_end_supported(f,B,False,a) for f in feats]
            # Dynamic programming through monotonic fragments.  Benefit favors observed coverage
            # and Stage-2 confidence; missing gaps carry a penalty.
            dp=[-1e18]*N; prev=[None]*N
            for i,f in enumerate(feats):
                if start[i]:
                    length=max(0.0,float(f["s"][-1]-f["s"][0]))
                    dp[i]=4.0*length/max(L,1.0)+float(f["record"].get("evidence_score",0.0))
                for j in range(i):
                    if dp[j]<=-1e17: continue
                    e=_span_edge(feats[j],f,a)
                    if e is None: continue
                    length=max(0.0,float(f["s"][-1]-f["s"][0]))
                    val=dp[j]+4.0*length/max(L,1.0)+float(f["record"].get("evidence_score",0.0))-2.0*e["gap_ft"]/max(L,1.0)
                    if val>dp[i]: dp[i]=val; prev[i]=(j,e)
            ends=[i for i in range(N) if end[i] and dp[i]>-1e17]
            if not ends: continue
            k=max(ends,key=lambda i:dp[i]); path=[]; edges=[]
            while k is not None:
                path.append(k)
                pr=prev[k]
                if pr is None: break
                edges.append(pr[1]); k=pr[0]
            path=path[::-1]; edges=edges[::-1]
            track_ids=[feats[i]["track_idx"] for i in path]
            if len(track_ids)<a.span_completion_min_tracks: continue
            obs=sum(max(0.0,float(feats[i]["s"][-1]-feats[i]["s"][0])) for i in path)
            coverage=obs/max(L,1e-6)
            if coverage<a.span_completion_min_coverage: continue
            seqs=[]
            for i in path:
                seqs.extend([frags[x]["slice_seq"] for x in feats[i]["record"].get("frags",[])])
            if seqs and max(seqs)-min(seqs)>a.max_span_slices: continue
            candidates.append({"pole_a":ia,"pole_b":ib,"track_ids":track_ids,"path":path,"feats":feats,"edges":edges,"score":dp[path[-1]],"coverage":coverage,"span_length_ft":L})

    # Strongest disjoint paths first.  A single Stage-2 track cannot be borrowed by two spans.
    candidates.sort(key=lambda x:(x["score"],x["coverage"]),reverse=True)
    used=set(); accepted=[]; replacement={}
    for c in candidates:
        if any(t in used for t in c["track_ids"]): continue
        tids=c["track_ids"]
        if len(tids)<a.span_completion_min_tracks: continue
        A=poles.loc[c["pole_a"]]; B=poles.loc[c["pole_b"]]
        raw=[]; frag_ids=[]; join_ids=[]; src=[]
        for ti in tids:
            tr=track_records[ti]
            raw.append(np.asarray(tr["raw"],float)); frag_ids.extend(tr.get("frags",[])); join_ids.extend(tr.get("join_ids",[])); src.extend(tr.get("source_segments",[]))
        rr=np.vstack(raw)
        f=_track_axis_features(rr,A,B)
        merged={
            "track_idx":min(tids),"frags":sorted(set(frag_ids)),"raw":f["raw"],"join_ids":sorted(set(join_ids)),
            "source_segments":sorted(set(src)),"observed_start":{"idx":int(c["pole_a"]),"dxy":float(np.linalg.norm(f["raw"][0,:2]-np.array([A.world_x_ft,A.world_y_ft]))),"angle":0.0,"score":0.0},
            "observed_end":{"idx":int(c["pole_b"]),"dxy":float(np.linalg.norm(f["raw"][-1,:2]-np.array([B.world_x_ft,B.world_y_ft]))),"angle":0.0,"score":0.0},
            "start_slice":min([frags[x]["slice_seq"] for x in frag_ids]) if frag_ids else None,
            "end_slice":max([frags[x]["slice_seq"] for x in frag_ids]) if frag_ids else None,
            "evidence_score":max(float(track_records[t].get("evidence_score",0.0)) for t in tids)+c["coverage"],
            "span_completion_used":True,"span_completion_track_count":len(tids),
        }
        keep=min(tids); replacement[keep]=merged; used.update(tids); accepted.append({
            "pole1_id":str(A.world_pole_id),"pole2_id":str(B.world_pole_id),"source_track_indices":";".join(map(str,tids)),
            "support_track_count":len(tids),"observed_coverage_fraction":c["coverage"],"span_length_ft":c["span_length_ft"],
            "max_bridge_gap_ft":max([e["gap_ft"] for e in c["edges"]],default=0.0),
            "max_lateral_ft":max([e["lateral_ft"] for e in c["edges"]],default=0.0),
            "max_z_error_ft":max([e["z_error_ft"] for e in c["edges"]],default=0.0),
            "rule":"two_poles_bracket_monotonic_collinear_fragment_path",
        })
    out=[]
    for i,tr in enumerate(track_records):
        if i in replacement: out.append(replacement[i])
        elif i not in used:
            tr=dict(tr); tr.setdefault("span_completion_used",False); tr.setdefault("span_completion_track_count",1); out.append(tr)
    return out,accepted

def ray_intersection_2d(p1, d1, p2, d2):
    """Intersect two directed XY rays.

    Returns (xy, t1, t2, acute_angle_deg) where xy = p1+t1*d1 = p2+t2*d2.
    t values are in feet because d1/d2 are unit vectors. Near-parallel lines return None.
    """
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    d1 = unit(d1)
    d2 = unit(d2)
    raw_angle = angle_deg(d1, d2)
    acute = min(raw_angle, 180.0 - raw_angle)
    M = np.column_stack([d1, -d2])
    det = float(np.linalg.det(M))
    if abs(det) < 1e-6:
        return None
    try:
        t1, t2 = np.linalg.solve(M, p2 - p1)
    except np.linalg.LinAlgError:
        return None
    xy1 = p1 + float(t1) * d1
    xy2 = p2 + float(t2) * d2
    xy = 0.5 * (xy1 + xy2)
    return xy, float(t1), float(t2), float(acute)


def point_inside_observed_support(xy, endpoint_seqs, center_records, a):
    """Conservative guard against inventing poles beyond the acquired slices.

    A hidden pole candidate must fall near a known slice center from the same
    session. We additionally prefer centers near one of the supporting line
    endpoint slice indices. Missing slice numbers are allowed.
    """
    if not center_records:
        return False, float("inf"), None
    xy = np.asarray(xy, float)
    seqs = [int(x) for x in endpoint_seqs if x is not None]
    best = None
    for rec in center_records:
        seq = int(rec["slice_seq"])
        # The intersection should lie in or immediately adjacent to one of the
        # slices that supplied the conductor endpoints. This blocks extrapolation
        # to an unseen pole beyond the collected corridor.
        if seqs and min(abs(seq - s) for s in seqs) > 1:
            continue
        d = float(np.linalg.norm(xy - np.asarray(rec["xy_ft"], float)))
        if best is None or d < best[0]:
            best = (d, rec)
    if best is None:
        # Fallback to any observed session slice, still bounded by the support radius.
        for rec in center_records:
            d = float(np.linalg.norm(xy - np.asarray(rec["xy_ft"], float)))
            if best is None or d < best[0]:
                best = (d, rec)
    if best is None:
        return False, float("inf"), None
    return best[0] <= a.hidden_pole_slice_support_radius_ft, best[0], best[1]


def infer_hidden_poles(track_records, observed_poles, center_records, gid, session_median_h, height_bounds, a):
    """Infer an occluded pole only where two independently pole-anchored partial spans converge.

    This is stronger than the prior two-line rule: every voting conductor must already be
    anchored to a real reconstructed pole at its opposite end.  The anchor poles must be
    distinct, the unsupported rays must be non-parallel and converge inside observed slice
    support, and the required pole height must stay inside the robust session height range.
    """
    if a.disable_hidden_pole_inference or not a.infer_hidden_poles:
        return pd.DataFrame(columns=WORLD_POLE_COLUMNS), []
    hlow,hhigh=height_bounds
    endpoints=[]
    for tr in track_records:
        if float(tr.get("evidence_score",0.0)) < a.hidden_pole_min_track_evidence:
            continue
        raw=np.asarray(tr["raw"],float)
        if len(raw)<2: continue
        # Unsupported start can vote only if the far/end side is anchored; vice versa.
        specs=[("start",0,True,tr.get("observed_start"),tr.get("observed_end"),tr.get("start_slice")),
               ("end",-1,False,tr.get("observed_end"),tr.get("observed_start"),tr.get("end_slice"))]
        for end_name,idx,is_start,this_attach,far_attach,seq in specs:
            if this_attach is not None or far_attach is None:
                continue
            far_idx=int(far_attach["idx"])
            if far_idx not in observed_poles.index: continue
            far_pole=observed_poles.loc[far_idx]
            tangent=endpoint_tangent_from_track(raw,is_start)
            slope=endpoint_z_slope(raw,0 if is_start else 1)
            endpoints.append({
                "track_idx":int(tr["track_idx"]),"end_name":end_name,"point":raw[idx].copy(),"tangent":unit(tangent),
                "z_slope":float(slope),"slice_seq":None if seq is None else int(seq),"source_segments":list(tr.get("source_segments",[])),
                "far_pole_idx":far_idx,"far_pole_id":str(far_pole.world_pole_id),"evidence_score":float(tr.get("evidence_score",0.0)),
            })
    pair_candidates=[]
    if endpoints:
        endpoint_xy=np.vstack([np.asarray(e["point"][:2],float) for e in endpoints])
        endpoint_tree=cKDTree(endpoint_xy)
        max_pair_sep=2.0*(a.hidden_pole_max_endpoint_extrap_ft+a.hidden_pole_ray_backtrack_tolerance_ft)+1.0
        endpoint_pairs=sorted(endpoint_tree.query_pairs(r=max_pair_sep))
    else:
        endpoint_pairs=[]
    for i,j in endpoint_pairs:
            A=endpoints[i]; B=endpoints[j]
            if A["track_idx"]==B["track_idx"]: continue
            if int(a.hidden_pole_require_distinct_anchor_poles) and A["far_pole_idx"]==B["far_pole_idx"]: continue
            x=ray_intersection_2d(A["point"][:2],A["tangent"],B["point"][:2],B["tangent"])
            if x is None: continue
            xy,ta,tb,acute=x
            if acute<a.hidden_pole_min_line_angle_deg: continue
            if ta < -a.hidden_pole_ray_backtrack_tolerance_ft or tb < -a.hidden_pole_ray_backtrack_tolerance_ft: continue
            if ta>a.hidden_pole_max_endpoint_extrap_ft or tb>a.hidden_pole_max_endpoint_extrap_ft: continue
            za=float(A["point"][2]+A["z_slope"]*max(0.0,ta)); zb=float(B["point"][2]+B["z_slope"]*max(0.0,tb))
            if abs(za-zb)>a.hidden_pole_max_attachment_z_spread_ft: continue
            inside,support_dist,support_rec=point_inside_observed_support(xy,[A["slice_seq"],B["slice_seq"]],center_records,a)
            if not inside: continue
            if not observed_poles.empty:
                d_existing=np.linalg.norm(observed_poles[["world_x_ft","world_y_ft"]].to_numpy(float)-xy[None,:],axis=1)
                if len(d_existing) and float(np.min(d_existing))<=a.hidden_pole_existing_pole_exclusion_ft: continue
            # Circuit geometry: each partial span from its known far pole to the hidden candidate must itself be valid.
            good=True
            for E,z in ((A,za),(B,zb)):
                far=observed_poles.loc[E["far_pole_idx"]]
                sep=float(np.linalg.norm(np.asarray(xy)-np.array([far.world_x_ft,far.world_y_ft],float)))
                if not (a.min_pole_separation_ft<=sep<=a.max_span_length_ft): good=False; break
            if not good: continue
            pair_candidates.append({
                "xy":np.asarray(xy,float),"track_a":A["track_idx"],"track_b":B["track_idx"],"end_a":A["end_name"],"end_b":B["end_name"],
                "slice_a":A["slice_seq"],"slice_b":B["slice_seq"],"z_a":za,"z_b":zb,"line_angle_deg":acute,
                "extrap_a_ft":max(0.0,ta),"extrap_b_ft":max(0.0,tb),"support_center_distance_ft":support_dist,
                "support_slice_seq":None if support_rec is None else int(support_rec["slice_seq"]),"source_a":";".join(A["source_segments"]),
                "source_b":";".join(B["source_segments"]),"anchor_pole_a":A["far_pole_id"],"anchor_pole_b":B["far_pole_id"],
                "evidence_a":A["evidence_score"],"evidence_b":B["evidence_score"],
            })
    if not pair_candidates:
        return pd.DataFrame(columns=WORLD_POLE_COLUMNS),[]
    pts=np.vstack([r["xy"] for r in pair_candidates]); labels=DBSCAN(eps=a.hidden_pole_cluster_radius_ft,min_samples=1).fit_predict(pts)
    rows=[]; audit=[]; next_num=len(observed_poles)+1
    for lab in sorted(set(labels)):
        q=[r for r,x in zip(pair_candidates,labels) if x==lab]
        tracks=sorted(set([r["track_a"] for r in q]+[r["track_b"] for r in q])); anchors=sorted(set([r["anchor_pole_a"] for r in q]+[r["anchor_pole_b"] for r in q]))
        if len(tracks)<2 or (int(a.hidden_pole_require_distinct_anchor_poles) and len(anchors)<2): continue
        xy=np.median(np.vstack([r["xy"] for r in q]),axis=0)
        if not observed_poles.empty:
            d_existing=np.linalg.norm(observed_poles[["world_x_ft","world_y_ft"]].to_numpy(float)-xy[None,:],axis=1)
            if len(d_existing) and float(np.min(d_existing))<=a.hidden_pole_existing_pole_exclusion_ft: continue
        support_z=[z for r in q for z in (r["z_a"],r["z_b"])]; max_attach_z=float(max(support_z)); support_slices=sorted(set(int(x) for r in q for x in (r["slice_a"],r["slice_b"]) if x is not None))
        baseline_h=float(a.fixed_pole_height_ft) if a.fixed_pole_height_ft is not None else float(np.clip(session_median_h,hlow,hhigh))
        baseline_h=max(a.min_pole_height_ft,baseline_h)
        if not observed_poles.empty:
            pxy=observed_poles[["world_x_ft","world_y_ft"]].to_numpy(float); dist=np.linalg.norm(pxy-xy[None,:],axis=1); near=np.where(dist<=a.hidden_pole_ground_reference_radius_ft)[0]
            if len(near):
                nearest=near[np.argsort(dist[near])[:3]]; base_z=float(np.median(observed_poles.iloc[nearest].base_z_ft.to_numpy(float)))
            else: base_z=max_attach_z+a.pole_top_margin_ft-baseline_h
        else: base_z=max_attach_z+a.pole_top_margin_ft-baseline_h
        desired=max(a.min_pole_height_ft,max_attach_z+a.pole_top_margin_ft-base_z)
        # Reject rather than manufacture an implausibly tall hidden pole.
        if desired>hhigh+a.pole_attachment_height_slack_ft: continue
        height=float(np.clip(desired,hlow,hhigh))
        pid=f"{gid}/P{next_num:05d}"; next_num+=1; track_text=";".join(map(str,tracks)); slice_text=";".join(map(str,support_slices)); source_text=";".join(sorted(set(x for r in q for x in (r["source_a"],r["source_b"]) if x)))
        row={
            "world_pole_id":pid,"group_id":gid,"world_x_ft":float(xy[0]),"world_y_ft":float(xy[1]),"base_x_ft":float(xy[0]),"base_y_ft":float(xy[1]),"base_z_ft":base_z,
            "top_x_ft":float(xy[0]),"top_y_ft":float(xy[1]),"top_z_ft":base_z+height,"height_ft":height,"observed_height_ft":np.nan,"baseline_height_ft":baseline_h,
            "height_adjustment_ft":height-baseline_h,"min_allowed_height_ft":hlow,"max_allowed_height_ft":hhigh,
            "attachment_min_z_ft":float(min(support_z)),"attachment_max_z_ft":float(max(support_z)),"pole_origin":"inferred_hidden_from_two_anchored_spans","inferred_from_lines":True,
            "inference_support_count":len(tracks),"inference_supporting_tracks":track_text,"inference_supporting_slices":slice_text,"inference_min_line_angle_deg":float(min(r["line_angle_deg"] for r in q)),
            "source_components":source_text,"source_slices":slice_text,"attached_chain_count":0,
        }
        rows.append(row); audit.append({
            "group_id":gid,"world_pole_id":pid,"world_x_ft":float(xy[0]),"world_y_ft":float(xy[1]),"supporting_track_count":len(tracks),"supporting_tracks":track_text,
            "supporting_anchor_poles":";".join(anchors),"supporting_slices":slice_text,"min_line_angle_deg":float(min(r["line_angle_deg"] for r in q)),
            "max_attachment_z_spread_ft":float(max(support_z)-min(support_z)),"max_endpoint_extrapolation_ft":float(max(max(r["extrap_a_ft"],r["extrap_b_ft"]) for r in q)),
            "nearest_slice_center_distance_ft":float(min(r["support_center_distance_ft"] for r in q)),"rule":"two_distinct_pole_anchored_nonparallel_partial_spans_converge_inside_observed_support",
        })
    return frame(rows,WORLD_POLE_COLUMNS),audit

def fit_clean_track(points, a, pole_start=None, pole_end=None):
    """Return a smooth, monotonic conductor path with no point-to-point zig-zag joins."""
    p = np.asarray(points, float)
    if len(p) < 2:
        return p

    pxy1 = None if pole_start is None else np.array([pole_start.world_x_ft, pole_start.world_y_ft], float)
    pxy2 = None if pole_end is None else np.array([pole_end.world_x_ft, pole_end.world_y_ft], float)
    axis, origin = track_axis(p, pxy1, pxy2)
    normal = np.array([-axis[1], axis[0]])
    s = (p[:, :2] - origin) @ axis
    lateral = (p[:, :2] - origin) @ normal

    # Orient s consistently with the observed track order.
    if len(p) >= 2 and np.dot(p[-1, :2] - p[0, :2], axis) < 0:
        axis = -axis
        normal = -normal
        s = (p[:, :2] - origin) @ axis
        lateral = (p[:, :2] - origin) @ normal

    lat_coef = robust_linear(s, lateral, degree=1)
    z_degree = 2 if len(p) >= 8 and np.ptp(s) >= 20.0 else 1
    z_coef = robust_linear(s, p[:, 2], degree=z_degree)

    s0, s1 = float(np.min(s)), float(np.max(s))
    if pole_start is not None:
        sp = float((pxy1 - origin) @ axis)
        if abs(sp - s0) <= abs(sp - s1):
            s0 = sp
        else:
            s1 = sp
    if pole_end is not None:
        sp = float((pxy2 - origin) @ axis)
        if abs(sp - s0) <= abs(sp - s1):
            s0 = sp
        else:
            s1 = sp
    if s1 < s0:
        s0, s1 = s1, s0

    length = max(0.0, s1 - s0)
    n = max(2, int(math.ceil(length / max(a.smooth_spacing_ft, 0.5))) + 1)
    sg = np.linspace(s0, s1, n)
    lat = eval_poly(lat_coef, sg)
    z = eval_poly(z_coef, sg)
    xy = origin[None, :] + sg[:, None] * axis[None, :] + lat[:, None] * normal[None, :]
    out = np.column_stack([xy, z])

    # Snap attached endpoints to pole XY while keeping conductor Z continuous.
    if pole_start is not None:
        d0 = np.linalg.norm(out[0, :2] - pxy1)
        d1 = np.linalg.norm(out[-1, :2] - pxy1)
        idx = 0 if d0 <= d1 else -1
        out[idx, :2] = pxy1
    if pole_end is not None:
        d0 = np.linalg.norm(out[0, :2] - pxy2)
        d1 = np.linalg.norm(out[-1, :2] - pxy2)
        idx = 0 if d0 <= d1 else -1
        out[idx, :2] = pxy2

    # Ensure output follows the original track order, not arbitrary PCA sign.
    if np.linalg.norm(out[0, :2] - p[0, :2]) > np.linalg.norm(out[-1, :2] - p[0, :2]):
        out = out[::-1].copy()
    return out


def _orient2d(a, b, c):
    return float((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))


def _segments_intersect_strict(a, b, c, d, tol=0.25):
    """True for an interior XY crossing; shared/near endpoints are ignored."""
    a=np.asarray(a,float); b=np.asarray(b,float); c=np.asarray(c,float); d=np.asarray(d,float)
    for p in (a,b):
        for q in (c,d):
            if np.linalg.norm(p-q) <= tol:
                return False
    o1,o2=_orient2d(a,b,c),_orient2d(a,b,d)
    o3,o4=_orient2d(c,d,a),_orient2d(c,d,b)
    return (o1*o2 < -1e-9) and (o3*o4 < -1e-9)


def polyline_self_intersects_xy(points, tol=0.25):
    p=np.asarray(points,float)
    if len(p) < 4:
        return False
    for i in range(len(p)-1):
        for j in range(i+2, len(p)-1):
            if j == i+1:
                continue
            if _segments_intersect_strict(p[i,:2],p[i+1,:2],p[j,:2],p[j+1,:2],tol):
                return True
    return False


def track_evidence_score(tr, frags, accepted):
    inds=tr.get("frags",[])
    probs=[float(frags[i].get("score",0.0)) for i in inds]
    join_rows=[accepted[j] for j in tr.get("join_ids",[]) if j < len(accepted)]
    mean_prob=float(np.mean(probs)) if probs else 0.0
    bridge_penalty=sum(1 for x in join_rows if x.get("join_mode") == "detached_bridge")
    mean_cost=float(np.mean([x.get("cost",0.0) for x in join_rows])) if join_rows else 0.0
    return 3.0*mean_prob + 0.25*len(inds) - 0.20*bridge_penalty - 0.10*mean_cost


def plot_session(poles, chains, out2, out3, title):
    from matplotlib.lines import Line2D
    pole_color = "black"
    conductor_color = "cyan"
    legend = [
        Line2D([0], [0], color=pole_color, lw=3, marker="o", markersize=5, label="Pole"),
        Line2D([0], [0], color=conductor_color, lw=1.8, label="Conductor"),
    ]

    fig, ax = plt.subplots(figsize=(12, 8))
    if len(poles):
        ax.scatter(poles.world_x_ft, poles.world_y_ft, s=28, c=pole_color, zorder=5)
    for _, g in chains.groupby("chain_id", sort=False):
        g = g.sort_values("vertex_index")
        ax.plot(g.world_x_ft, g.world_y_ft, color=conductor_color, linewidth=1.6, zorder=2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("World X ft")
    ax.set_ylabel("World Y ft")
    ax.set_title(title)
    ax.legend(handles=legend, loc="best")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out2, dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    if len(poles):
        for _, r in poles.iterrows():
            ax.plot(
                [r.world_x_ft, r.world_x_ft], [r.world_y_ft, r.world_y_ft],
                [r.base_z_ft, r.top_z_ft], color=pole_color, linewidth=3,
            )
    for _, g in chains.groupby("chain_id", sort=False):
        g = g.sort_values("vertex_index")
        ax.plot(g.world_x_ft, g.world_y_ft, g.world_z_ft, color=conductor_color, linewidth=1.6)
    ax.set_xlabel("World X ft")
    ax.set_ylabel("World Y ft")
    ax.set_zlabel("World Z ft")
    ax.set_title(title)
    ax.legend(handles=legend, loc="best")
    fig.tight_layout()
    fig.savefig(out3, dpi=180)
    plt.close(fig)


def merge_poles(pg, gid, a):
    if pg.empty:
        default_h=float(a.fixed_pole_height_ft) if a.fixed_pole_height_ft is not None else 35.0
        lo=max(a.min_pole_height_ft,default_h-a.allowed_pole_height_variation_ft)
        hi=default_h+a.allowed_pole_height_variation_ft
        return pd.DataFrame(columns=WORLD_POLE_COLUMNS),default_h,(lo,hi)
    observed=pd.to_numeric(pg.height_ft,errors="coerce").to_numpy(float); observed=observed[np.isfinite(observed)&(observed>0)]
    geom=np.abs(pg.top_z_ft.to_numpy(float)-pg.base_z_ft.to_numpy(float)); geom=geom[np.isfinite(geom)&(geom>0)]
    session_median=float(np.median(observed)) if len(observed) else (float(np.median(geom)) if len(geom) else 35.0)
    hlow,hhigh=pole_height_bounds_from_observations(pg,session_median,a)
    xy=(pg[["base_x_ft","base_y_ft"]].to_numpy()+pg[["top_x_ft","top_y_ft"]].to_numpy())/2.0
    labels=DBSCAN(eps=a.pole_merge_radius_ft,min_samples=1).fit_predict(xy); rows=[]
    for cid in sorted(set(labels)):
        q=pg[labels==cid]; wx=float(((q.base_x_ft+q.top_x_ft)/2.0).median()); wy=float(((q.base_y_ft+q.top_y_ft)/2.0).median()); bz=float(q.base_z_ft.quantile(0.10))
        oh=pd.to_numeric(q.height_ft,errors="coerce"); oh=oh[np.isfinite(oh)&(oh>0)]; observed_h=float(oh.median()) if len(oh) else float(np.median(np.abs(q.top_z_ft-q.base_z_ft)))
        baseline_h=float(a.fixed_pole_height_ft) if a.fixed_pole_height_ft is not None else float(np.clip(observed_h,hlow,hhigh)); baseline_h=max(a.min_pole_height_ft,baseline_h)
        rows.append({
            "world_pole_id":f"{gid}/P{len(rows)+1:05d}","group_id":gid,"world_x_ft":wx,"world_y_ft":wy,"base_x_ft":wx,"base_y_ft":wy,"base_z_ft":bz,
            "top_x_ft":wx,"top_y_ft":wy,"top_z_ft":bz+baseline_h,"height_ft":baseline_h,"observed_height_ft":observed_h,"baseline_height_ft":baseline_h,
            "height_adjustment_ft":0.0,"min_allowed_height_ft":hlow,"max_allowed_height_ft":hhigh,"attachment_min_z_ft":np.nan,"attachment_max_z_ft":np.nan,
            "pole_origin":"observed_stage2","inferred_from_lines":False,"inference_support_count":0,"inference_supporting_tracks":"","inference_supporting_slices":"",
            "inference_min_line_angle_deg":np.nan,"source_components":";".join(q.source_key.astype(str)),"source_slices":";".join(map(str,sorted(set(q.slice_seq.astype(int))))),"attached_chain_count":0,
        })
    return pd.DataFrame(rows).reindex(columns=WORLD_POLE_COLUMNS),session_median,(hlow,hhigh)

def adjust_pole_heights(mp, attachments, a):
    """Adjust within each pole's robust allowed height range; never override the cap."""
    adjustments=0; rejected=0
    if mp.empty: return adjustments,rejected
    for idx in mp.index:
        zs=attachments.get(int(idx),[]); baseline=float(mp.at[idx,"baseline_height_ft"]); base=float(mp.at[idx,"base_z_ft"])
        if not zs: continue
        minh=float(mp.at[idx,"min_allowed_height_ft"]); maxh=float(mp.at[idx,"max_allowed_height_ft"])
        valid=[float(z) for z in zs if float(z)<=base+maxh+a.pole_attachment_height_slack_ft]
        rejected+=len(zs)-len(valid)
        if not valid: continue
        zmin,zmax=float(min(valid)),float(max(valid)); desired=zmax+a.pole_top_margin_ft-base
        new_height=float(np.clip(desired,minh,maxh)); new_top=base+new_height
        mp.at[idx,"top_z_ft"]=new_top; mp.at[idx,"height_ft"]=new_height; mp.at[idx,"height_adjustment_ft"]=new_height-baseline
        mp.at[idx,"attachment_min_z_ft"]=zmin; mp.at[idx,"attachment_max_z_ft"]=zmax
        if abs(new_height-baseline)>0.25: adjustments+=1
    return adjustments,rejected

def main():
    a = pa()
    inf = Path(a.inference_dir)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    object_audit = []

    man_path = inf / "inference_manifest.csv"
    if not man_path.exists():
        raise FileNotFoundError(f"Stage 3 requires inference manifest: {man_path}")
    try:
        man = pd.read_csv(man_path)
    except EmptyDataError as e:
        raise RuntimeError(f"Inference manifest has no columns: {man_path}") from e
    required_manifest={"id","relative_path","geography","session","slice_seq","group_id","pole_csv","line_csv","line_vertices_csv"}
    missing_manifest=sorted(required_manifest-set(man.columns))
    if missing_manifest:
        raise RuntimeError(f"Inference manifest missing required columns {missing_manifest}: {man_path}")
    if man.empty:
        raise RuntimeError(f"Inference manifest contains zero usable rows: {man_path}")
    if a.session_filter:
        man = man[man.group_id.astype(str) == a.session_filter].copy()
    if a.latest_slice is not None:
        if not a.session_filter:
            raise ValueError("--latest_slice requires --session_filter")
        man = man[(man.slice_seq <= a.latest_slice) & (man.slice_seq >= a.latest_slice - a.max_span_slices)].copy()

    centers, skipped = center_map(a, man)
    atomic_csv(frame(skipped, SKIPPED_CENTER_COLUMNS), out / "skipped_center_metadata.csv")

    # Session slice-center support is used only by Stage 3. Hidden poles may be
    # inferred only when the conductor intersection lies inside this acquired
    # support; open lines extrapolating beyond the slices cannot create poles.
    center_support_by_group = {}
    for _, r in man.iterrows():
        rel = str(r.relative_path)
        c = centers.get(rel)
        if c is None:
            continue
        center_support_by_group.setdefault(str(r.group_id), []).append({
            "relative_path": rel,
            "slice_seq": int(r.slice_seq),
            "xy_ft": (np.asarray(c[:2], float) * a.world_units_to_ft).tolist(),
        })

    allp = []
    allf = []
    for _, r in man.iterrows():
        rel = str(r.relative_path)
        c = centers.get(rel)
        if c is None:
            continue
        conv = a.world_units_to_ft
        center = np.array(c, float)
        relp = Path(rel)
        stem = relp.name[:-7] if relp.name.endswith(".csv.gz") else relp.stem
        fallback = inf / "stage2_objects" / relp.parent
        pfile = Path(str(r.pole_csv))
        lfile = Path(str(r.line_csv))
        vfile = Path(str(r.line_vertices_csv))
        if not pfile.exists():
            pfile = fallback / (stem + "_poles.csv")
        if not lfile.exists():
            lfile = fallback / (stem + "_line_segments.csv")
        if not vfile.exists():
            vfile = fallback / (stem + "_line_vertices.csv")

        p = read_stage2_csv(pfile, POLE_COLUMNS, "poles", object_audit)
        if not p.empty:
            for _, q in p.iterrows():
                b = (np.array([q.base_x, q.base_y, q.base_z]) + center) * conv
                t = (np.array([q.top_x, q.top_y, q.top_z]) + center) * conv
                allp.append({
                    "group_id": r.group_id, "geography": r.geography, "session": r.session,
                    "slice_seq": int(r.slice_seq), "relative_path": rel,
                    "source_key": f"{r.id}|{q.component_id}",
                    "base_x_ft": b[0], "base_y_ft": b[1], "base_z_ft": b[2],
                    "top_x_ft": t[0], "top_y_ft": t[1], "top_z_ft": t[2],
                    "height_ft": float(q.height_ft), "radius_p90_ft": float(q.radius_p90_ft),
                    "touches_xy_edge": q.touches_xy_edge,
                })

        l = read_stage2_csv(lfile, LINE_COLUMNS, "line_segments", object_audit)
        v = read_stage2_csv(vfile, VERTEX_COLUMNS, "line_vertices", object_audit)
        if not l.empty and not v.empty:
            # Build the component lookup once. The old implementation rescanned
            # the full vertex DataFrame for every line component.
            vertex_groups = {str(k): g.sort_values("vertex_index") for k, g in v.groupby(v.component_id.astype(str), sort=False)}
            for _, q in l.iterrows():
                z = vertex_groups.get(str(q.component_id))
                if z is None or z.empty:
                    object_audit.append({"kind":"line_vertices","path":str(vfile),"reason":"component_vertices_missing","missing_columns":str(q.component_id)})
                    continue
                local = z[["x", "y", "z"]].to_numpy(float)
                world = (local + center) * conv
                if len(world) >= 2:
                    world = order_xy(world)
                    allf.append({
                        "group_id": r.group_id, "geography": r.geography, "session": r.session,
                        "slice_seq": int(r.slice_seq), "relative_path": rel,
                        "source_key": f"{r.id}|{q.component_id}", "points": world,
                        "direction": fragment_direction(world), "score": float(q.refiner_probability),
                    })

    poles = pd.DataFrame(allp)
    group_ids = sorted(set([x["group_id"] for x in allf]) | set(poles.group_id.astype(str) if not poles.empty else []))

    summaries = []
    all_mp = []
    all_chains = []
    all_verts = []
    all_spans = []
    global_join_audit = []
    global_hidden_pole_audit = []
    global_span_completion_audit = []
    global_polygon_guard_audit = []
    chain_counter = 0
    resumed_sessions = 0
    resumed_sessions_missing_detailed_audits = 0
    resume_audit = []

    for gid in group_ids:
        gdir = out / "sessions" / gid
        if a.resume_sessions and _session_core_outputs_complete(gdir):
            try:
                summary = json.loads((gdir / "summary.json").read_text())
                mp0 = _read_csv_or_empty(gdir / "world_poles.csv", WORLD_POLE_COLUMNS)
                c0 = _read_csv_or_empty(gdir / "conductor_chains.csv", CHAIN_COLUMNS)
                v0 = _read_csv_or_empty(gdir / "conductor_vertices.csv", VERTEX_WORLD_COLUMNS)
                s0 = _read_csv_or_empty(gdir / "spans.csv", CHAIN_COLUMNS)
                summaries.append(summary)
                all_mp.extend(mp0.to_dict("records")); all_chains.extend(c0.to_dict("records")); all_verts.extend(v0.to_dict("records")); all_spans.extend(s0.to_dict("records"))
                chain_counter=max(chain_counter,_chain_counter_from_frame(c0))
                audit_specs=[
                    ("accepted_fragment_joins.csv",JOIN_AUDIT_COLUMNS,global_join_audit),
                    ("inferred_hidden_poles.csv",HIDDEN_POLE_AUDIT_COLUMNS,global_hidden_pole_audit),
                    ("span_completion_paths.csv",SPAN_COMPLETION_AUDIT_COLUMNS,global_span_completion_audit),
                    ("prevented_polygon_connections.csv",POLYGON_AUDIT_COLUMNS,global_polygon_guard_audit),
                ]
                missing_detail=False
                for name,cols,target in audit_specs:
                    ap=gdir/name
                    if ap.exists() and ap.stat().st_size>0:
                        target.extend(_read_csv_or_empty(ap,cols).to_dict("records"))
                    else:
                        missing_detail=True
                resumed_sessions += 1
                resumed_sessions_missing_detailed_audits += int(missing_detail)
                resume_audit.append({
                    "group_id":gid,"resumed":True,"detailed_audit_complete":not missing_detail,
                    "note":"legacy completed session had no per-session detailed audit CSVs" if missing_detail else "per-session detailed audits loaded",
                })
                print("[stage3-resume]",gid,"loaded completed session",summary)
                continue
            except Exception as exc:
                print("[stage3-resume]",gid,"existing session output could not be loaded; recomputing:",repr(exc))

        # A killed process can leave an empty/partial session directory. Remove only
        # that incomplete session before recomputing; completed sessions above stay intact.
        if a.resume_sessions and gdir.exists() and not _session_core_outputs_complete(gdir):
            shutil.rmtree(gdir)
        gdir.mkdir(parents=True, exist_ok=True)
        session_started=time.time()
        pg = poles[poles.group_id.astype(str) == gid].copy() if not poles.empty else pd.DataFrame()
        mp, session_median_h, session_height_bounds = merge_poles(pg, gid, a)
        frags = [x for x in allf if x["group_id"] == gid]
        polygon_audit_start=len(global_polygon_guard_audit)

        print(f"[stage3-start] {gid} poles_obs={len(pg)} line_segments={len(frags)} resume={a.resume_sessions}", flush=True)
        t_join=time.time()
        accepted = select_one_to_one_joins(frags, a)
        session_join_rows=[]
        for c in accepted:
            rr={"group_id": gid, **c, "source_a": frags[c["i"]]["source_key"], "source_b": frags[c["j"]]["source_key"]}
            global_join_audit.append(rr); session_join_rows.append(rr)
        fragment_join_seconds=time.time()-t_join
        print(f"[stage3-phase] {gid} fragment_join_seconds={fragment_join_seconds:.3f} accepted={len(accepted)}", flush=True)
        tracks = build_track_components(frags, accepted)

        # Build clean track records first. Hidden-pole inference is performed only
        # after line fragments have already been assembled into non-zig-zag tracks.
        # This prevents raw neighboring components from voting for fictitious poles.
        track_records = []
        pole_index = build_pole_xy_index(mp)
        for track_idx, tr in enumerate(tracks):
            inds = tr["frags"]
            raw = np.asarray(tr["points"], float)
            if len(raw) < 2:
                continue
            hspan_raw = float(np.linalg.norm(np.ptp(raw[:, :2], axis=0)))
            zspan_raw = float(np.ptp(raw[:, 2]))
            if hspan_raw < a.min_chain_horizontal_ft:
                continue
            if zspan_raw / max(hspan_raw, 1e-6) > a.max_vertical_horizontal_ratio:
                continue
            start_t = endpoint_tangent_from_track(raw, at_start=True)
            end_t = endpoint_tangent_from_track(raw, at_start=False)
            # Mark endpoints already explained by observed poles. Hidden poles are
            # considered only for the remaining unsupported endpoints.
            observed_start = pole_attachment_candidate(raw[0], start_t, mp, a, pole_index=pole_index)
            observed_end = pole_attachment_candidate(raw[-1], end_t, mp, a, pole_index=pole_index)
            track_records.append({
                "track_idx": track_idx, "frags": inds, "raw": raw,
                "join_ids": tr.get("join_ids", []),
                "start_slice": frags[inds[0]]["slice_seq"] if inds else None,
                "end_slice": frags[inds[-1]]["slice_seq"] if inds else None,
                "source_segments": [frags[i]["source_key"] for i in inds],
                "observed_start": observed_start, "observed_end": observed_end,
            })

        # Stronger tracks claim pole/topology connections first. This matters for
        # the polygon guard: if several plausible spans compete, the weaker
        # cycle-closing synthetic attachment is the one that is dropped.
        for tr in track_records:
            tr["evidence_score"] = track_evidence_score(
                {"frags": tr["frags"], "join_ids": tr.get("join_ids", [])}, frags, accepted
            )
        track_records.sort(key=lambda x: x["evidence_score"], reverse=True)

        # First complete fragmented spans that are already bracketed by two detected poles.
        # This recovers floating pieces without relaxing ordinary line-to-line joins globally.
        t_span=time.time()
        track_records, span_completion_audit_pre = complete_span_backed_tracks(track_records, mp, frags, a)
        span_completion_pre_seconds=time.time()-t_span
        print(f"[stage3-phase] {gid} span_completion_pre_seconds={span_completion_pre_seconds:.3f} paths={len(span_completion_audit_pre)}", flush=True)

        observed_merged_pole_count = len(mp)
        t_hidden=time.time()
        hidden_mp, hidden_audit = infer_hidden_poles(
            track_records, mp, center_support_by_group.get(gid, []), gid, session_median_h, session_height_bounds, a
        )
        hidden_pole_seconds=time.time()-t_hidden
        print(f"[stage3-phase] {gid} hidden_pole_seconds={hidden_pole_seconds:.3f} inferred={len(hidden_mp)}", flush=True)
        if not hidden_mp.empty:
            if mp.empty:
                mp = hidden_mp.copy().reindex(columns=WORLD_POLE_COLUMNS)
            else:
                mp = pd.concat([mp, hidden_mp], ignore_index=True).reindex(columns=WORLD_POLE_COLUMNS)
        global_hidden_pole_audit.extend(hidden_audit)
        pole_index = build_pole_xy_index(mp)

        # Re-run span-backed completion after adding hidden poles so each supporting
        # partial span can be carried through to the newly inferred pole.
        t_span=time.time()
        track_records, span_completion_audit_post = complete_span_backed_tracks(track_records, mp, frags, a)
        span_completion_post_seconds=time.time()-t_span
        print(f"[stage3-phase] {gid} span_completion_post_seconds={span_completion_post_seconds:.3f} paths={len(span_completion_audit_post)}", flush=True)
        span_completion_audit = span_completion_audit_pre + span_completion_audit_post
        for rr in span_completion_audit:
            global_span_completion_audit.append({"group_id": gid, **rr})

        t_chain=time.time()
        chains = []
        verts = []
        spans = []
        attachments = {}
        pole_graph = UF(len(mp))
        accepted_pole_pairs = set()

        for tr in track_records:
            inds = tr["frags"]
            raw = np.asarray(tr["raw"], float)

            # Attach track ends to either observed poles or strictly inferred hidden poles.
            start_t = endpoint_tangent_from_track(raw, at_start=True)
            end_t = endpoint_tangent_from_track(raw, at_start=False)
            # Preserve high-confidence pole anchors already established by the
            # span-completion/hidden-pole passes.  Otherwise search locally.
            # This lets a completed span bridge a slightly larger evidence gap
            # without inventing a new free-space connector.
            ca = tr.get("observed_start")
            cb = tr.get("observed_end")
            if ca is not None and int(ca.get("idx", -1)) not in mp.index:
                ca = None
            if cb is not None and int(cb.get("idx", -1)) not in mp.index:
                cb = None
            if ca is None:
                ca = pole_attachment_candidate(raw[0], start_t, mp, a, exclude=[cb["idx"]] if cb else None, pole_index=pole_index)
            if cb is None:
                cb = pole_attachment_candidate(raw[-1], end_t, mp, a, exclude=[ca["idx"]] if ca else None, pole_index=pole_index)

            # If the first choice prevents a valid second distinct pole, also try the opposite order.
            if ca is None and cb is not None:
                ca = pole_attachment_candidate(raw[0], start_t, mp, a, exclude=[cb["idx"]], pole_index=pole_index)
            if cb is None and ca is not None:
                cb = pole_attachment_candidate(raw[-1], end_t, mp, a, exclude=[ca["idx"]], pole_index=pole_index)

            # Pole graph topology guard. Multiple conductors between the same pole
            # pair are valid. But a newly synthesized attachment that closes a
            # 3+-pole cycle would draw a triangle/polygon, so detach the weaker end.
            if ca is not None and cb is not None and ca["idx"] != cb["idx"] and not a.disable_pole_graph_cycle_guard:
                ia, ib = int(ca["idx"]), int(cb["idx"])
                pair = tuple(sorted((ia, ib)))
                if pair not in accepted_pole_pairs and pole_graph.find(ia) == pole_graph.find(ib):
                    drop_start = float(ca["score"]) >= float(cb["score"])
                    global_polygon_guard_audit.append({
                        "group_id": gid, "track_idx": tr["track_idx"],
                        "pole_start_id": str(mp.loc[ia].world_pole_id),
                        "pole_end_id": str(mp.loc[ib].world_pole_id),
                        "dropped_endpoint": "start" if drop_start else "end",
                        "reason": "would_close_pole_graph_polygon",
                        "track_evidence_score": tr.get("evidence_score", np.nan),
                    })
                    if drop_start:
                        ca = None
                    else:
                        cb = None

            pole_start = mp.loc[ca["idx"]] if ca is not None else None
            pole_end = mp.loc[cb["idx"]] if cb is not None else None

            clean = fit_clean_track(raw, a, pole_start=pole_start, pole_end=pole_end)
            # A fitted endpoint can shift slightly in Z.  Never keep a pole attachment
            # that would require exceeding the pole's robust permitted height.
            refit = False
            if ca is not None and len(clean) and not attachment_height_valid(mp.loc[ca["idx"]], clean[0,2], a):
                ca = None; pole_start = None; refit = True
            if cb is not None and len(clean) and not attachment_height_valid(mp.loc[cb["idx"]], clean[-1,2], a):
                cb = None; pole_end = None; refit = True
            if refit:
                clean = fit_clean_track(raw, a, pole_start=pole_start, pole_end=pole_end)
            if len(clean) < 2:
                continue
            # Hard geometric rule: a single conductor can never cross itself.
            if polyline_self_intersects_xy(clean, a.self_intersection_tolerance_ft):
                global_polygon_guard_audit.append({
                    "group_id": gid, "track_idx": tr["track_idx"],
                    "pole_start_id": "" if ca is None else str(mp.loc[ca["idx"]].world_pole_id),
                    "pole_end_id": "" if cb is None else str(mp.loc[cb["idx"]].world_pole_id),
                    "dropped_endpoint": "track",
                    "reason": "self_intersecting_conductor_geometry_rejected",
                    "track_evidence_score": tr.get("evidence_score", np.nan),
                })
                continue
            hspan = float(np.linalg.norm(np.ptp(clean[:, :2], axis=0)))
            zspan = float(np.ptp(clean[:, 2]))
            if hspan < a.min_chain_horizontal_ft:
                continue

            attach_indices = []
            attach_dists = [999.0, 999.0]
            if ca is not None:
                attach_indices.append(ca["idx"])
                attach_dists[0] = ca["dxy"]
            if cb is not None:
                attach_indices.append(cb["idx"])
                attach_dists[1] = cb["dxy"]
            visible = sorted(set(attach_indices))

            kind = "OPEN_CONDUCTOR"
            spanlen = np.nan
            p1 = p2 = ""
            if ca is not None and cb is not None and ca["idx"] != cb["idx"]:
                A = mp.loc[ca["idx"]]
                B = mp.loc[cb["idx"]]
                sep = float(np.linalg.norm([B.world_x_ft - A.world_x_ft, B.world_y_ft - A.world_y_ft]))
                seqs = [frags[i]["slice_seq"] for i in inds]
                if a.min_pole_separation_ft <= sep <= a.max_span_length_ft and max(seqs) - min(seqs) <= a.max_span_slices:
                    kind = "SPAN"
                    spanlen = sep
                    p1, p2 = A.world_pole_id, B.world_pole_id
                    ia, ib = int(ca["idx"]), int(cb["idx"])
                    pair = tuple(sorted((ia, ib)))
                    if pair not in accepted_pole_pairs:
                        pole_graph.union(ia, ib)
                        accepted_pole_pairs.add(pair)
            elif len(visible) == 1:
                kind = "PARTIAL_SPAN"
                p1 = mp.loc[visible[0]].world_pole_id

            chain_counter += 1
            cid = f"{gid}/C{chain_counter:06d}"
            seqs = sorted(set(frags[i]["slice_seq"] for i in inds))
            join_rows = [accepted[j] for j in tr["join_ids"] if j < len(accepted)]
            row = {
                "chain_id": cid, "group_id": gid, "chain_type": kind,
                "pole1_id": p1, "pole2_id": p2, "visible_pole_supports": len(visible),
                "span_length_ft": spanlen, "horizontal_extent_ft": hspan, "vertical_extent_ft": zspan,
                "slice_min": min(seqs), "slice_max": max(seqs), "slice_range": max(seqs) - min(seqs),
                "observed_slice_count": len(seqs), "observed_slices": ";".join(map(str, seqs)),
                "source_line_segments": ";".join(frags[i]["source_key"] for i in inds),
                "endpoint1_pole_dist_ft": attach_dists[0], "endpoint2_pole_dist_ft": attach_dists[1],
                "fragment_count": len(inds), "accepted_join_count": len(join_rows),
                "max_join_xy_ft": max([x["xy_gap_ft"] for x in join_rows], default=0.0),
                "max_join_z_ft": max([x["z_gap_ft"] for x in join_rows], default=0.0),
                "max_join_lateral_ft": max([x["lateral_ft"] for x in join_rows], default=0.0),
                "smoothed_geometry": True,
                "span_completion_used": bool(tr.get("span_completion_used", False)),
                "span_completion_track_count": int(tr.get("span_completion_track_count", 1)),
            }
            chains.append(row)
            all_chains.append(row)
            if kind == "SPAN":
                spans.append(row)
                all_spans.append(row)

            # Each attached line endpoint is snapped to the pole XY at its own conductor Z.
            # The black pole is then extended/reduced to include the attachment height.
            if ca is not None:
                attachments.setdefault(ca["idx"], []).append(float(clean[0, 2]))
                mp.at[ca["idx"], "attached_chain_count"] += 1
            if cb is not None:
                attachments.setdefault(cb["idx"], []).append(float(clean[-1, 2]))
                mp.at[cb["idx"], "attached_chain_count"] += 1

            for vi, q in enumerate(clean):
                vr = {
                    "chain_id": cid, "group_id": gid, "vertex_index": vi,
                    "world_x_ft": q[0], "world_y_ft": q[1], "world_z_ft": q[2],
                }
                verts.append(vr)
                all_verts.append(vr)

        pole_adjustments, height_attachments_rejected = adjust_pole_heights(mp, attachments, a)
        chain_build_and_attachment_seconds=time.time()-t_chain

        cdf = frame(chains, CHAIN_COLUMNS)
        vdf = frame(verts, VERTEX_WORLD_COLUMNS)
        sdf = frame(spans, CHAIN_COLUMNS)
        mp = mp.reindex(columns=WORLD_POLE_COLUMNS)
        t_output=time.time()
        atomic_csv(mp, gdir / "world_poles.csv")
        atomic_csv(cdf, gdir / "conductor_chains.csv")
        atomic_csv(vdf, gdir / "conductor_vertices.csv")
        atomic_csv(sdf, gdir / "spans.csv")
        atomic_csv(frame(session_join_rows, JOIN_AUDIT_COLUMNS), gdir / "accepted_fragment_joins.csv")
        atomic_csv(frame(hidden_audit, HIDDEN_POLE_AUDIT_COLUMNS), gdir / "inferred_hidden_poles.csv")
        atomic_csv(frame([{"group_id":gid,**x} if "group_id" not in x else x for x in span_completion_audit], SPAN_COMPLETION_AUDIT_COLUMNS), gdir / "span_completion_paths.csv")
        session_polygon_rows=global_polygon_guard_audit[polygon_audit_start:]
        atomic_csv(frame(session_polygon_rows, POLYGON_AUDIT_COLUMNS), gdir / "prevented_polygon_connections.csv")
        if not vdf.empty or not mp.empty:
            if not a.disable_plots:
                plot_session(mp, vdf, gdir / "reconstruction_2d.png", gdir / "reconstruction_3d.png", f"{gid}: poles (black) and conductors (cyan)")
        output_write_seconds=time.time()-t_output
        elapsed_seconds=time.time()-session_started

        summary = {
            "group_id": gid,
            "source_pole_observations": len(pg),
            "observed_merged_poles": observed_merged_pole_count,
            "inferred_hidden_poles": len(hidden_mp),
            "merged_poles": len(mp),
            "session_median_observed_pole_height_ft": session_median_h,
            "line_segments": len(frags),
            "accepted_fragment_joins": len(accepted),
            "conductor_chains": len(cdf),
            "spans": int((cdf.chain_type == "SPAN").sum()) if not cdf.empty else 0,
            "partial_spans": int((cdf.chain_type == "PARTIAL_SPAN").sum()) if not cdf.empty else 0,
            "open_conductors": int((cdf.chain_type == "OPEN_CONDUCTOR").sum()) if not cdf.empty else 0,
            "pole_height_adjustments": pole_adjustments,
            "detached_fragment_bridges": sum(1 for x in accepted if x.get("join_mode") == "detached_bridge"),
            "span_completion_paths": len(span_completion_audit),
            "height_attachments_rejected": height_attachments_rejected,
            "polygon_connections_prevented": sum(1 for x in global_polygon_guard_audit if x.get("group_id") == gid),
            "fragment_join_seconds": fragment_join_seconds,
            "span_completion_pre_seconds": span_completion_pre_seconds,
            "hidden_pole_seconds": hidden_pole_seconds,
            "span_completion_post_seconds": span_completion_post_seconds,
            "chain_build_and_attachment_seconds": chain_build_and_attachment_seconds,
            "output_write_seconds": output_write_seconds,
            "elapsed_seconds": elapsed_seconds,
        }
        summaries.append(summary)
        atomic_json(summary, gdir / "summary.json")
        all_mp.extend(mp.to_dict("records"))
        print("[stage3-circuit-complete]", gid, summary, f"elapsed_seconds={elapsed_seconds:.3f}", flush=True)

    atomic_csv(frame(all_mp, WORLD_POLE_COLUMNS), out / "all_world_poles.csv")
    atomic_csv(frame(all_chains, CHAIN_COLUMNS), out / "all_conductor_chains.csv")
    atomic_csv(frame(all_verts, VERTEX_WORLD_COLUMNS), out / "all_conductor_vertices.csv")
    atomic_csv(frame(all_spans, CHAIN_COLUMNS), out / "all_spans.csv")
    atomic_csv(frame(summaries, SUMMARY_COLUMNS), out / "all_session_summaries.csv")
    atomic_csv(pd.DataFrame(object_audit, columns=["kind", "path", "reason", "missing_columns"]), out / "stage2_object_read_audit.csv")
    atomic_csv(frame(global_join_audit, JOIN_AUDIT_COLUMNS), out / "accepted_fragment_joins.csv")
    atomic_csv(frame(global_hidden_pole_audit, HIDDEN_POLE_AUDIT_COLUMNS), out / "inferred_hidden_poles.csv")
    atomic_csv(frame(global_span_completion_audit, SPAN_COMPLETION_AUDIT_COLUMNS), out / "span_completion_paths.csv")
    atomic_csv(frame(global_polygon_guard_audit, POLYGON_AUDIT_COLUMNS), out / "prevented_polygon_connections.csv")
    atomic_csv(frame(resume_audit, RESUME_AUDIT_COLUMNS), out / "resume_session_audit.csv")

    atomic_json({
        "completed": True,
        "reconstruction_version": "v4-realtime-circuit-complete",
        "sessions": len(summaries),
        "spans": len(all_spans),
        "conductor_chains": len(all_chains),
        "accepted_fragment_joins": int(sum(int(x.get("accepted_fragment_joins",0)) for x in summaries)),
        "inferred_hidden_poles": int(sum(int(x.get("inferred_hidden_poles",0)) for x in summaries)),
        "span_completion_paths": int(sum(int(x.get("span_completion_paths",0)) for x in summaries)),
        "polygon_connections_prevented": int(sum(int(x.get("polygon_connections_prevented",0)) for x in summaries)),
        "detached_fragment_bridges": int(sum(int(x.get("detached_fragment_bridges",0)) for x in summaries)),
        "resumed_sessions": resumed_sessions,
        "resumed_sessions_missing_detailed_audits": resumed_sessions_missing_detailed_audits,
        "skipped_center_metadata": len(skipped),
        "empty_or_unreadable_stage2_object_files": len(object_audit),
        "elapsed_seconds": time.time() - started,
        "rules": {
            "world_units_to_ft": a.world_units_to_ft,
            "min_pole_separation_ft": a.min_pole_separation_ft,
            "max_span_length_ft": a.max_span_length_ft,
            "max_span_slices": a.max_span_slices,
            "missing_intermediate_slices_allowed": True,
            "unattached_lines_retained": True,
            "one_to_one_fragment_endpoint_matching": True,
            "spatially_indexed_fragment_candidate_search": True,
            "spatially_indexed_intervening_fragment_guard": True,
            "spatially_indexed_span_completion": True,
            "spatially_indexed_hidden_pole_pairs": True,
            "session_level_resume": bool(a.resume_sessions),
            "fragment_graph_cycles_forbidden": True,
            "detached_fragment_bridge_pass": True,
            "pole_graph_cycles_forbidden_except_parallel_multiedges": not a.disable_pole_graph_cycle_guard,
            "self_intersecting_conductors_forbidden": True,
            "smooth_track_refit": True,
            "max_join_angle_deg": a.max_join_angle_deg,
            "max_connector_angle_deg": a.max_connector_angle_deg,
            "max_join_lateral_ft": a.max_join_lateral_ft,
            "max_z_extrap_error_ft": a.max_z_extrap_error_ft,
            "max_longitudinal_overlap_ft": a.max_longitudinal_overlap_ft,
            "fragment_bridge_radius_ft": a.fragment_bridge_radius_ft,
            "bridge_max_join_angle_deg": a.bridge_max_join_angle_deg,
            "bridge_max_connector_angle_deg": a.bridge_max_connector_angle_deg,
            "bridge_max_lateral_ft": a.bridge_max_lateral_ft,
            "bridge_max_z_extrap_error_ft": a.bridge_max_z_extrap_error_ft,
            "pole_attachment_radius_ft": a.pole_attachment_radius_ft,
            "pole_height_adjustment_from_attachments": True,
            "pole_height_hard_range_enforced": True,
            "pole_height_range_quantiles": [a.pole_height_quantile_low, a.pole_height_quantile_high],
            "span_backed_fragment_completion": not a.disable_span_completion,
            "span_completion_min_tracks": a.span_completion_min_tracks,
            "span_completion_min_coverage": a.span_completion_min_coverage,
            "hidden_pole_inference": not a.disable_hidden_pole_inference,
            "hidden_pole_requires_two_nonparallel_lines": True,
            "hidden_pole_requires_each_partial_span_anchored_to_existing_pole": True,
            "hidden_pole_requires_distinct_anchor_poles": bool(a.hidden_pole_require_distinct_anchor_poles),
            "hidden_pole_min_line_angle_deg": a.hidden_pole_min_line_angle_deg,
            "hidden_pole_max_endpoint_extrap_ft": a.hidden_pole_max_endpoint_extrap_ft,
            "hidden_pole_inside_observed_slice_support_required": True,
            "hidden_pole_slice_support_radius_ft": a.hidden_pole_slice_support_radius_ft,
            "single_open_line_never_creates_pole": True,
            "plot_pole_color": "black",
            "plot_conductor_color": "cyan",
        },
    }, out / "COMPLETED.json")


if __name__ == "__main__":
    main()
