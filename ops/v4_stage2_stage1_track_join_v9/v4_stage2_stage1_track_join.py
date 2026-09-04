#!/usr/bin/env python3
"""Stage-1 inferred-line track joining for V4 Stage 2 experiments.

The runtime never infers a conductor from poles.  It begins only with occupied
voxels whose deployed Stage-1 label is class 2 (line).  Exact 26-neighbour
connected components are first recovered, then geometrically compatible
fragments are joined into conductor tracks.  A bridge can span a short missing
run of Stage-1 labels, but both bridge endpoints must be observed Stage-1
class-2 voxels.  No synthetic line voxels are created.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np
import pandas as pd

STAGE1_TRACK_JOIN_RUNTIME_VERSION = "stage1-track-join-v9-20260904"


def resolve_deployed_stage1_labels(
    pred: dict[str, np.ndarray], calibration: dict[str, Any]
) -> tuple[np.ndarray, str]:
    pole = np.asarray(pred["pole"], dtype=np.float32)
    line = np.asarray(pred["line"], dtype=np.float32)
    if len(pole) != len(line):
        raise ValueError("Stage1 pole/line score lengths differ")
    for key in (
        "deployed_labels", "deployed_label", "inferred_labels", "inferred_label",
        "pred_labels", "pred_label",
    ):
        if key not in pred:
            continue
        arr = np.asarray(pred[key])
        if arr.ndim == 1 and len(arr) == len(pole) and np.issubdtype(arr.dtype, np.number):
            labels = arr.astype(np.int8, copy=False)
            if set(map(int, np.unique(labels))) <= {0, 1, 2}:
                return labels.copy(), f"pred.{key}"
    from v4_realtime_core import label_from_scores
    labels = label_from_scores(
        pole,
        line,
        float(calibration["pole_threshold"]),
        float(calibration["line_threshold"]),
    )
    return np.asarray(labels, dtype=np.int8), "v4_realtime_core.label_from_scores"


def _coord_keys(coords: np.ndarray, grid_size: tuple[int, int, int]) -> np.ndarray:
    c = np.asarray(coords, dtype=np.int64)
    if not len(c):
        return np.empty(0, dtype=np.int64)
    gx, gy, _ = map(int, grid_size)
    return (c[:, 2] * gy + c[:, 1]) * gx + c[:, 0]


def _forward_offsets_26() -> list[tuple[int, int, int]]:
    out = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                if (dz, dy, dx) > (0, 0, 0):
                    out.append((dx, dy, dz))
    if len(out) != 13:
        raise AssertionError(len(out))
    return out


_FORWARD_26 = _forward_offsets_26()


class UnionFind:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> bool:
        a = self.find(a)
        b = self.find(b)
        if a == b:
            return False
        if self.r[a] < self.r[b]:
            a, b = b, a
        self.p[b] = a
        if self.r[a] == self.r[b]:
            self.r[a] += 1
        return True


def connected_components_26(
    coords: np.ndarray,
    line_mask: np.ndarray,
    grid_size: tuple[int, int, int],
) -> tuple[np.ndarray, list[np.ndarray]]:
    c = np.asarray(coords, dtype=np.int32)
    m = np.asarray(line_mask, dtype=bool)
    line_idx = np.flatnonzero(m).astype(np.int64)
    if not len(line_idx):
        return line_idx, []
    gx, gy, gz = map(int, grid_size)
    keys = _coord_keys(c[line_idx], grid_size)
    key_to_local = {int(k): i for i, k in enumerate(keys)}
    uf = UnionFind(len(line_idx))
    for li, oi in enumerate(line_idx):
        x, y, z = map(int, c[int(oi)])
        for dx, dy, dz in _FORWARD_26:
            nx, ny, nz = x + dx, y + dy, z + dz
            if not (0 <= nx < gx and 0 <= ny < gy and 0 <= nz < gz):
                continue
            key = int((nz * gy + ny) * gx + nx)
            lj = key_to_local.get(key)
            if lj is not None:
                uf.union(li, lj)
    groups: dict[int, list[int]] = {}
    for li, oi in enumerate(line_idx):
        groups.setdefault(uf.find(li), []).append(int(oi))
    comps = [np.asarray(v, dtype=np.int64) for v in groups.values()]
    comps.sort(key=lambda a: (-len(a), int(a.min()) if len(a) else -1))
    return line_idx, comps


def _unit_xy(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    if len(p) < 2:
        return np.array([1.0, 0.0], dtype=float)
    xy = p[:, :2]
    q = xy - np.mean(xy, axis=0)
    try:
        _, _, vh = np.linalg.svd(q, full_matrices=False)
        d = np.asarray(vh[0], dtype=float)
    except np.linalg.LinAlgError:
        d = np.array([1.0, 0.0], dtype=float)
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return np.array([1.0, 0.0], dtype=float)
    d /= n
    # deterministic orientation
    if d[0] < 0 or (abs(d[0]) < 1e-12 and d[1] < 0):
        d = -d
    return d


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    if na < 1e-12 or nb < 1e-12:
        return 90.0
    dot = float(np.clip(abs(np.dot(aa / na, bb / nb)), -1.0, 1.0))
    return float(math.degrees(math.acos(dot)))


@dataclass
class Fragment:
    index: int
    voxel_indices: np.ndarray
    axis_xy: np.ndarray
    center_xy: np.ndarray
    t_min: float
    t_max: float
    endpoint_min_idx: int
    endpoint_max_idx: int
    horizontal_span_ft: float
    radius_p90_ft: float


def describe_fragment(
    component_index: int,
    voxel_indices: np.ndarray,
    coords: np.ndarray,
    voxel_size_ft: float,
) -> Fragment:
    pts = np.asarray(coords[voxel_indices], dtype=float)
    axis = _unit_xy(pts)
    center = np.mean(pts[:, :2], axis=0) if len(pts) else np.zeros(2)
    t = (pts[:, :2] - center) @ axis if len(pts) else np.zeros(0)
    if len(t):
        i0 = int(np.argmin(t))
        i1 = int(np.argmax(t))
        normal = np.array([-axis[1], axis[0]])
        lateral = np.abs((pts[:, :2] - center) @ normal) * float(voxel_size_ft)
        radius = float(np.quantile(lateral, 0.90)) if len(lateral) else 0.0
        span = float((t.max() - t.min()) * voxel_size_ft)
    else:
        i0 = i1 = 0
        radius = span = 0.0
    return Fragment(
        index=int(component_index),
        voxel_indices=np.asarray(voxel_indices, dtype=np.int64),
        axis_xy=axis,
        center_xy=center,
        t_min=float(t.min()) if len(t) else 0.0,
        t_max=float(t.max()) if len(t) else 0.0,
        endpoint_min_idx=int(voxel_indices[i0]),
        endpoint_max_idx=int(voxel_indices[i1]),
        horizontal_span_ft=span,
        radius_p90_ft=radius,
    )


def _bridge_metrics(
    a: Fragment,
    b: Fragment,
    coords: np.ndarray,
    voxel_size_ft: float,
) -> dict[str, Any]:
    options = []
    for side_a, ia in (("min", a.endpoint_min_idx), ("max", a.endpoint_max_idx)):
        for side_b, ib in (("min", b.endpoint_min_idx), ("max", b.endpoint_max_idx)):
            pa = np.asarray(coords[ia], dtype=float)
            pb = np.asarray(coords[ib], dtype=float)
            d3 = (pb - pa) * float(voxel_size_ft)
            gap = float(np.linalg.norm(d3))
            dxy_vox = pb[:2] - pa[:2]
            dxy_ft = dxy_vox * float(voxel_size_ft)
            h = float(np.linalg.norm(dxy_ft))
            avg_axis = a.axis_xy + (b.axis_xy if np.dot(a.axis_xy, b.axis_xy) >= 0 else -b.axis_xy)
            if np.linalg.norm(avg_axis) < 1e-12:
                avg_axis = a.axis_xy.copy()
            avg_axis = avg_axis / max(float(np.linalg.norm(avg_axis)), 1e-12)
            normal = np.array([-avg_axis[1], avg_axis[0]])
            lateral = float(abs(np.dot(dxy_ft, normal)))
            if h > 1e-12:
                bridge_axis = dxy_ft / h
                bridge_angle_a = _angle_deg(bridge_axis, a.axis_xy)
                bridge_angle_b = _angle_deg(bridge_axis, b.axis_xy)
            else:
                bridge_angle_a = bridge_angle_b = 90.0
            options.append({
                "a_side": side_a,
                "b_side": side_b,
                "a_endpoint_index": int(ia),
                "b_endpoint_index": int(ib),
                "gap_ft": gap,
                "horizontal_gap_ft": h,
                "vertical_gap_ft": float(abs(d3[2])),
                "lateral_offset_ft": lateral,
                "axis_angle_deg": _angle_deg(a.axis_xy, b.axis_xy),
                "bridge_angle_a_deg": bridge_angle_a,
                "bridge_angle_b_deg": bridge_angle_b,
            })
    options.sort(key=lambda r: (r["gap_ft"], r["lateral_offset_ft"]))
    return options[0]


def candidate_fragment_bridges(
    fragments: list[Fragment],
    coords: np.ndarray,
    voxel_size_ft: float,
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    max_gap = float(profile["max_gap_ft"])
    max_lateral = float(profile["max_lateral_ft"])
    max_angle = float(profile["max_axis_angle_deg"])
    max_bridge_angle = float(profile["max_bridge_angle_deg"])
    max_vertical = float(profile["max_vertical_gap_ft"])
    min_vox = int(profile.get("min_fragment_voxels", 2))
    rows = []
    for i in range(len(fragments)):
        a = fragments[i]
        if len(a.voxel_indices) < min_vox:
            continue
        for j in range(i + 1, len(fragments)):
            b = fragments[j]
            if len(b.voxel_indices) < min_vox:
                continue
            m = _bridge_metrics(a, b, coords, voxel_size_ft)
            passed = (
                m["gap_ft"] <= max_gap
                and m["lateral_offset_ft"] <= max_lateral
                and m["axis_angle_deg"] <= max_angle
                and m["bridge_angle_a_deg"] <= max_bridge_angle
                and m["bridge_angle_b_deg"] <= max_bridge_angle
                and m["vertical_gap_ft"] <= max_vertical
            )
            m.update({
                "fragment_a": int(a.index),
                "fragment_b": int(b.index),
                "fragment_a_voxels": int(len(a.voxel_indices)),
                "fragment_b_voxels": int(len(b.voxel_indices)),
                "passed": bool(passed),
                "selected": False,
            })
            rows.append(m)
    rows.sort(key=lambda r: (
        0 if r["passed"] else 1,
        r["gap_ft"],
        r["lateral_offset_ft"],
        r["axis_angle_deg"],
    ))
    return rows


def select_fragment_bridges(
    fragments: list[Fragment], candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Select a non-branching minimum-gap forest over compatible fragments."""
    uf = UnionFind(len(fragments))
    degree = [0] * len(fragments)
    selected = []
    for row in candidates:
        if not row["passed"]:
            continue
        a = int(row["fragment_a"])
        b = int(row["fragment_b"])
        if degree[a] >= 2 or degree[b] >= 2:
            continue
        if uf.find(a) == uf.find(b):
            continue
        uf.union(a, b)
        degree[a] += 1
        degree[b] += 1
        row["selected"] = True
        selected.append(row)
    return selected


def _track_groups(n_fragments: int, selected: list[dict[str, Any]]) -> list[list[int]]:
    uf = UnionFind(n_fragments)
    for r in selected:
        uf.union(int(r["fragment_a"]), int(r["fragment_b"]))
    groups: dict[int, list[int]] = {}
    for i in range(n_fragments):
        groups.setdefault(uf.find(i), []).append(i)
    out = list(groups.values())
    out.sort(key=lambda g: (min(g), len(g)))
    return out


def _ordered_observed_vertices(
    voxel_indices: np.ndarray,
    coords: np.ndarray,
    voxel_size_ft: float,
    vertex_bin_ft: float,
) -> np.ndarray:
    pts = np.asarray(coords[voxel_indices], dtype=float)
    if len(pts) <= 2:
        return pts.copy()
    axis = _unit_xy(pts)
    center = np.mean(pts[:, :2], axis=0)
    t_ft = ((pts[:, :2] - center) @ axis) * float(voxel_size_ft)
    order = np.argsort(t_ft, kind="stable")
    pts = pts[order]
    t_ft = t_ft[order]
    bin_ft = max(float(vertex_bin_ft), float(voxel_size_ft))
    b = np.floor((t_ft - t_ft.min()) / bin_ft + 1e-9).astype(int)
    verts = []
    for bid in np.unique(b):
        block = pts[b == bid]
        if len(block):
            verts.append(np.median(block, axis=0))
    if not verts:
        return pts[[0, -1]].copy()
    v = np.vstack(verts)
    # Ensure exact observed endpoints anchor the polyline.  Interior vertices are
    # medians of observed voxels; no synthetic voxel labels are created.
    v[0] = pts[0]
    if len(v) == 1:
        v = np.vstack([pts[0], pts[-1]])
    else:
        v[-1] = pts[-1]
    # Remove exact duplicate consecutive vertices.
    keep = [0]
    for i in range(1, len(v)):
        if not np.allclose(v[i], v[keep[-1]], atol=1e-12):
            keep.append(i)
    return v[keep]


def build_track_join_outputs(
    coords: np.ndarray,
    line_scores: np.ndarray,
    labels: np.ndarray,
    file_id: str,
    slice_seq: int,
    grid_size: tuple[int, int, int],
    voxel_size_ft: float,
    profile: dict[str, Any],
) -> dict[str, Any]:
    c = np.asarray(coords, dtype=np.int32)
    scores = np.asarray(line_scores, dtype=np.float32)
    lab = np.asarray(labels, dtype=np.int8)
    if not (len(c) == len(scores) == len(lab)):
        raise ValueError("Stage1 track-join arrays do not align")

    line_mask = lab == 2
    line_idx, components = connected_components_26(c, line_mask, grid_size)
    fragments = [describe_fragment(i, idx, c, voxel_size_ft) for i, idx in enumerate(components)]
    candidates = candidate_fragment_bridges(fragments, c, voxel_size_ft, profile)
    selected = select_fragment_bridges(fragments, candidates)
    groups = _track_groups(len(fragments), selected) if fragments else []

    selected_by_pair = {
        tuple(sorted((int(r["fragment_a"]), int(r["fragment_b"])))): r
        for r in selected
    }

    line_rows = []
    vertex_rows = []
    component_rows = []
    line_points: dict[str, np.ndarray] = {}
    track_rows = []
    track_voxel_count = 0
    singleton_voxels = 0
    max_bridge_gap = 0.0

    for track_no, group in enumerate(groups, 1):
        vox = np.unique(np.concatenate([fragments[g].voxel_indices for g in group])).astype(np.int64)
        if len(vox) < 2:
            singleton_voxels += int(len(vox))
            continue
        cid = f"S1T{track_no:05d}"
        verts = _ordered_observed_vertices(
            vox, c, voxel_size_ft, float(profile.get("vertex_bin_ft", 1.0))
        )
        if len(verts) < 2:
            singleton_voxels += int(len(vox))
            continue
        deltas_ft = np.diff(verts, axis=0) * float(voxel_size_ft)
        seg_lengths = np.linalg.norm(deltas_ft, axis=1)
        poly_len = float(seg_lengths.sum())
        direct = (verts[-1] - verts[0]) * float(voxel_size_ft)
        direct_len = float(np.linalg.norm(direct))
        horizontal_span = float(np.linalg.norm(direct[:2]))
        vertical_span = float(abs(direct[2]))
        verticality = float(vertical_span / max(direct_len, 1e-12))
        tort = float(poly_len / max(direct_len, 1e-12))
        bridge_rows = []
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                r = selected_by_pair.get(tuple(sorted((a, b))))
                if r is not None:
                    bridge_rows.append(r)
        bridge_count = len(bridge_rows)
        local_max_gap = max((float(r["gap_ft"]) for r in bridge_rows), default=0.0)
        max_bridge_gap = max(max_bridge_gap, local_max_gap)
        score_mean = float(np.mean(scores[vox])) if len(vox) else float("nan")

        line_rows.append({
            "file_id": str(file_id),
            "component_id": cid,
            "slice_seq": int(slice_seq),
            "refiner_probability": float("nan"),
            "horizontal_span_ft": horizontal_span,
            "vertical_span_ft": vertical_span,
            "verticality": verticality,
            "tortuosity": tort,
            "vertex_count": int(len(verts)),
        })
        for vi, q in enumerate(verts):
            vertex_rows.append({
                "file_id": str(file_id),
                "component_id": cid,
                "slice_seq": int(slice_seq),
                "vertex_index": int(vi),
                "x": float(q[0]), "y": float(q[1]), "z": float(q[2]),
            })
        component_rows.append({
            "component_id": cid,
            "class_name": "line",
            "n_voxels": int(len(vox)),
            "score_mean": score_mean,
            "component_accept": True,
            "accept_mode": "stage1_inferred_track_join",
            "stage1_line_label_fraction": 1.0,
            "raw_fragment_count": int(len(group)),
            "bridge_count": int(bridge_count),
            "max_bridge_gap_ft": float(local_max_gap),
            "synthetic_line_voxels": 0,
            "runtime_gt_usage": False,
            "file_id": str(file_id),
            "slice_seq": int(slice_seq),
        })
        line_points[cid] = c[vox].copy()
        track_rows.append({
            "component_id": cid,
            "raw_fragment_count": int(len(group)),
            "n_voxels": int(len(vox)),
            "vertex_count": int(len(verts)),
            "bridge_count": int(bridge_count),
            "max_bridge_gap_ft": float(local_max_gap),
            "horizontal_span_ft": horizontal_span,
            "vertical_span_ft": vertical_span,
            "score_mean": score_mean,
        })
        track_voxel_count += int(len(vox))

    # All class-2 voxels remain in the accepted voxel audit even if a single
    # isolated voxel cannot form a valid two-vertex Stage-2 line geometry.
    if track_voxel_count + singleton_voxels != len(line_idx):
        raise RuntimeError(
            f"Stage1 line voxel accounting mismatch: tracks={track_voxel_count} "
            f"singletons={singleton_voxels} stage1={len(line_idx)}"
        )

    return {
        "lines_rows": line_rows,
        "vertices_rows": vertex_rows,
        "components_rows": component_rows,
        "line_points": line_points,
        "line_indices": line_idx,
        "raw_components": components,
        "fragments": fragments,
        "bridge_candidates": candidates,
        "selected_bridges": selected,
        "track_rows": track_rows,
        "audit": {
            "stage1_inferred_line_voxels": int(len(line_idx)),
            "accepted_stage1_line_voxels": int(len(line_idx)),
            "stage1_to_stage2_voxel_preservation": 1.0,
            "raw_stage1_line_components": int(len(components)),
            "joined_stage2_tracks": int(len(line_rows)),
            "selected_fragment_bridges": int(len(selected)),
            "max_selected_bridge_gap_ft": float(max_bridge_gap),
            "unjoined_singleton_line_voxels": int(singleton_voxels),
            "runtime_gt_usage": False,
            "synthetic_line_voxels": 0,
            "pole_pair_inference": False,
            "line_refiner_used": False,
            "line_hysteresis_used": False,
            "line_geometry_source": "stage1_class2_track_join",
            "bridge_requires_stage1_class2_both_sides": True,
        },
    }


class Stage1TrackJoinStage2Processor:
    """Production poles + Stage-1 inferred conductor-track joining."""

    def __init__(
        self,
        stage2_bundle: str,
        calibration_json: str,
        profile: dict[str, Any],
        grid_size: tuple[int, int, int] = (400, 400, 200),
        voxel_size_ft: float = 0.5,
    ) -> None:
        from v4_realtime_core import load_calibration
        from v4_realtime_pipeline import V4Stage2Processor
        self.grid = tuple(map(int, grid_size))
        self.voxel = float(voxel_size_ft)
        self.calibration = load_calibration(calibration_json)
        self.profile = dict(profile)
        self.production = V4Stage2Processor(
            stage2_bundle,
            self.grid,
            self.voxel,
            0.15, 0.08, 0.04, 0.55, 4, 3, 10,
        )
        self.production_processor_module = self.production.__class__.__module__

    def process(
        self,
        item: dict[str, Any],
        pred: dict[str, np.ndarray],
        file_id: str = "slice",
        slice_seq: int = 0,
    ) -> dict[str, Any]:
        from v4_stage2_runtime import LINE_OUTPUT_COLUMNS, VERTEX_OUTPUT_COLUMNS
        start = time.perf_counter()
        baseline = self.production.process(item, pred, file_id, slice_seq)
        baseline_ms = (time.perf_counter() - start) * 1000.0
        labels, label_source = resolve_deployed_stage1_labels(pred, self.calibration)
        t = time.perf_counter()
        joined = build_track_join_outputs(
            np.asarray(item["coords"], dtype=np.int32),
            np.asarray(pred["line"], dtype=np.float32),
            labels,
            file_id,
            slice_seq,
            self.grid,
            self.voxel,
            self.profile,
        )
        join_ms = (time.perf_counter() - t) * 1000.0
        base_components = baseline.get("components", pd.DataFrame()).copy()
        if not base_components.empty and "class_name" in base_components.columns:
            pole_components = base_components[
                base_components["class_name"].astype(str).eq("pole")
            ].copy()
        else:
            pole_components = pd.DataFrame()
        line_components = pd.DataFrame(joined["components_rows"])
        components = pd.concat([pole_components, line_components], ignore_index=True, sort=False)
        raw = baseline.get("raw_components", {}).copy()
        raw["line_points"] = joined["line_points"]
        return {
            "raw_components": raw,
            "components": components,
            "poles": baseline["poles"].copy(),
            "lines": pd.DataFrame(joined["lines_rows"], columns=LINE_OUTPUT_COLUMNS),
            "vertices": pd.DataFrame(joined["vertices_rows"], columns=VERTEX_OUTPUT_COLUMNS),
            "stage1_labels": labels,
            "label_source": label_source,
            "line_indices": joined["line_indices"],
            "bridge_candidates": joined["bridge_candidates"],
            "selected_bridges": joined["selected_bridges"],
            "track_rows": joined["track_rows"],
            "stage1_track_join_audit": {
                **joined["audit"],
                "stage1_label_source": label_source,
                "production_poles_preserved": True,
                "production_line_outputs_replaced": True,
                "runtime_version": STAGE1_TRACK_JOIN_RUNTIME_VERSION,
                "production_processor_module": self.production_processor_module,
                "profile": self.profile,
            },
            "timing": {
                "production_stage2_ms": float(baseline_ms),
                "stage1_track_join_ms": float(join_ms),
                "stage2_total_ms": float(baseline_ms + join_ms),
            },
        }
