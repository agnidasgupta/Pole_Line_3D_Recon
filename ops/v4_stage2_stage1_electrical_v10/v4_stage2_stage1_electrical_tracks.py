#!/usr/bin/env python3
"""Electrical-safe Stage-1 inferred conductor track reconstruction for V4 Stage 2.

Principles:
- Runtime starts only from deployed Stage-1 class-2 (line) voxels.
- Production pole detections are preserved.
- Missing runs between fragments may be bridged only as same-lane, end-to-end continuation.
- Overlapping/parallel/converging lanes are never merged into one conductor track.
- Any line-to-line bridge that passes near a detected pole is forbidden.
- Track endpoints may attach independently to the detected pole surface.
- No synthetic line voxels and no runtime GT usage.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np
import pandas as pd

STAGE1_ELECTRICAL_TRACK_RUNTIME_VERSION = "stage1-electrical-tracks-v10-20260904"


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
    if len(pts):
        t = (pts[:, :2] - center) @ axis
        normal = np.array([-axis[1], axis[0]])
        lateral = np.abs((pts[:, :2] - center) @ normal) * float(voxel_size_ft)
        radius = float(np.quantile(lateral, 0.90)) if len(lateral) else 0.0
        span = float((t.max() - t.min()) * voxel_size_ft)
    else:
        radius = span = 0.0
    return Fragment(
        index=int(component_index),
        voxel_indices=np.asarray(voxel_indices, dtype=np.int64),
        axis_xy=axis,
        center_xy=center,
        horizontal_span_ft=span,
        radius_p90_ft=radius,
    )


def _point_segment_distance_xy_ft(
    point_xy_vox: np.ndarray,
    a_xy_vox: np.ndarray,
    b_xy_vox: np.ndarray,
    voxel_size_ft: float,
) -> float:
    p = np.asarray(point_xy_vox, dtype=float)
    a = np.asarray(a_xy_vox, dtype=float)
    b = np.asarray(b_xy_vox, dtype=float)
    ab = b - a
    d2 = float(np.dot(ab, ab))
    if d2 <= 1e-12:
        q = a
    else:
        u = float(np.clip(np.dot(p - a, ab) / d2, 0.0, 1.0))
        q = a + u * ab
    return float(np.linalg.norm((p - q) * float(voxel_size_ft)))


def _pole_center_xy_vox(row: Any) -> np.ndarray:
    base = np.array([float(row.base_x), float(row.base_y)], dtype=float)
    top = np.array([float(row.top_x), float(row.top_y)], dtype=float)
    return 0.5 * (base + top)


def _nearest_pole_to_bridge(
    pa: np.ndarray,
    pb: np.ndarray,
    poles: pd.DataFrame | None,
    voxel_size_ft: float,
    guard_radius_ft: float,
) -> tuple[bool, str, float]:
    if poles is None or poles.empty:
        return False, "", float("inf")
    best_id = ""
    best = float("inf")
    for r in poles.itertuples(index=False):
        center = _pole_center_xy_vox(r)
        d = _point_segment_distance_xy_ft(center, pa[:2], pb[:2], voxel_size_ft)
        if d < best:
            best = d
            best_id = str(getattr(r, "component_id", ""))
    return bool(best <= float(guard_radius_ft)), best_id, float(best)


def _bridge_metrics(
    a: Fragment,
    b: Fragment,
    coords: np.ndarray,
    voxel_size_ft: float,
    poles: pd.DataFrame | None,
    profile: dict[str, Any],
) -> dict[str, Any]:
    c = np.asarray(coords, dtype=float)
    aa = np.asarray(a.axis_xy, dtype=float)
    bb = np.asarray(b.axis_xy, dtype=float)
    if np.dot(aa, bb) < 0:
        bb = -bb
    axis = aa + bb
    if float(np.linalg.norm(axis)) < 1e-12:
        axis = aa.copy()
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    normal = np.array([-axis[1], axis[0]], dtype=float)

    pa_all = c[a.voxel_indices]
    pb_all = c[b.voxel_indices]
    a_xy_ft = pa_all[:, :2] * float(voxel_size_ft)
    b_xy_ft = pb_all[:, :2] * float(voxel_size_ft)
    t_a = a_xy_ft @ axis
    t_b = b_xy_ft @ axis
    n_a = a_xy_ft @ normal
    n_b = b_xy_ft @ normal
    amin, amax = float(np.min(t_a)), float(np.max(t_a))
    bmin, bmax = float(np.min(t_b)), float(np.max(t_b))
    overlap = max(0.0, min(amax, bmax) - max(amin, bmin))

    ca = float(np.median(t_a))
    cb = float(np.median(t_b))
    if ca <= cb:
        ia_local = int(np.argmax(t_a))
        ib_local = int(np.argmin(t_b))
        a_side, b_side = "max", "min"
        longitudinal_gap = max(0.0, bmin - amax)
    else:
        ia_local = int(np.argmin(t_a))
        ib_local = int(np.argmax(t_b))
        a_side, b_side = "min", "max"
        longitudinal_gap = max(0.0, amin - bmax)

    ia = int(a.voxel_indices[ia_local])
    ib = int(b.voxel_indices[ib_local])
    pa = c[ia]
    pb = c[ib]
    d3_ft = (pb - pa) * float(voxel_size_ft)
    gap_ft = float(np.linalg.norm(d3_ft))
    dxy_ft = d3_ft[:2]
    h = float(np.linalg.norm(dxy_ft))
    lane_center_offset = float(abs(np.median(n_a) - np.median(n_b)))
    endpoint_lateral_jump = float(abs(np.dot(dxy_ft, normal)))
    if h > 1e-12:
        bridge_axis = dxy_ft / h
        bridge_angle_a = _angle_deg(bridge_axis, aa)
        bridge_angle_b = _angle_deg(bridge_axis, bb)
    else:
        bridge_angle_a = bridge_angle_b = 90.0

    near_pole, pole_id, pole_dist = _nearest_pole_to_bridge(
        pa, pb, poles, voxel_size_ft, float(profile.get("pole_bridge_guard_radius_ft", 0.0))
    )
    return {
        "a_side": a_side,
        "b_side": b_side,
        "a_endpoint_index": ia,
        "b_endpoint_index": ib,
        "gap_ft": gap_ft,
        "longitudinal_gap_ft": float(longitudinal_gap),
        "longitudinal_overlap_ft": float(overlap),
        "horizontal_gap_ft": h,
        "vertical_gap_ft": float(abs(d3_ft[2])),
        "lane_center_offset_ft": lane_center_offset,
        "endpoint_lateral_jump_ft": endpoint_lateral_jump,
        "axis_angle_deg": _angle_deg(aa, bb),
        "bridge_angle_a_deg": bridge_angle_a,
        "bridge_angle_b_deg": bridge_angle_b,
        "near_pole_bridge_guard": bool(near_pole),
        "nearest_pole_component_id": pole_id,
        "nearest_pole_bridge_distance_ft": pole_dist,
    }


def candidate_fragment_bridges(
    fragments: list[Fragment],
    coords: np.ndarray,
    voxel_size_ft: float,
    profile: dict[str, Any],
    poles: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    max_gap = float(profile["max_gap_ft"])
    max_lane = float(profile["max_lane_offset_ft"])
    max_angle = float(profile["max_axis_angle_deg"])
    max_bridge_angle = float(profile["max_bridge_angle_deg"])
    max_vertical = float(profile["max_vertical_gap_ft"])
    max_overlap = float(profile["max_longitudinal_overlap_ft"])
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
            m = _bridge_metrics(a, b, coords, voxel_size_ft, poles, profile)
            checks = [
                (m["longitudinal_overlap_ft"] <= max_overlap, "longitudinal_overlap"),
                (m["gap_ft"] <= max_gap, "gap"),
                (m["lane_center_offset_ft"] <= max_lane, "lane_center_offset"),
                (m["endpoint_lateral_jump_ft"] <= max_lane, "endpoint_lateral_jump"),
                (m["axis_angle_deg"] <= max_angle, "axis_angle"),
                (m["bridge_angle_a_deg"] <= max_bridge_angle, "bridge_angle_a"),
                (m["bridge_angle_b_deg"] <= max_bridge_angle, "bridge_angle_b"),
                (m["vertical_gap_ft"] <= max_vertical, "vertical_gap"),
                (not m["near_pole_bridge_guard"], "near_pole_bridge_guard"),
            ]
            failed = [name for ok, name in checks if not ok]
            m.update({
                "fragment_a": int(a.index),
                "fragment_b": int(b.index),
                "fragment_a_voxels": int(len(a.voxel_indices)),
                "fragment_b_voxels": int(len(b.voxel_indices)),
                "passed": not failed,
                "reject_reason": "accepted" if not failed else failed[0],
                "all_failed_reasons": ";".join(failed),
                "selected": False,
                "selection_reject_reason": "",
            })
            rows.append(m)
    rows.sort(key=lambda r: (
        0 if r["passed"] else 1,
        r["longitudinal_gap_ft"],
        r["lane_center_offset_ft"],
        r["gap_ft"],
    ))
    return rows


def _track_radius_p95_ft(
    fragment_ids: set[int],
    fragments: list[Fragment],
    coords: np.ndarray,
    voxel_size_ft: float,
) -> float:
    if not fragment_ids:
        return 0.0
    idx = np.unique(np.concatenate([fragments[i].voxel_indices for i in sorted(fragment_ids)]))
    pts = np.asarray(coords[idx], dtype=float)
    if len(pts) < 2:
        return 0.0
    axis = _unit_xy(pts)
    center = np.mean(pts[:, :2], axis=0)
    normal = np.array([-axis[1], axis[0]])
    lateral = np.abs((pts[:, :2] - center) @ normal) * float(voxel_size_ft)
    return float(np.quantile(lateral, 0.95)) if len(lateral) else 0.0


def select_fragment_bridges(
    fragments: list[Fragment],
    candidates: list[dict[str, Any]],
    coords: np.ndarray,
    voxel_size_ft: float,
    profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Greedy electrically-safe end-to-end forest.

    Besides cycle/degree guards, each fragment endpoint side can be used only once and
    each tentative merged track must remain narrow enough to represent one conductor.
    """
    uf = UnionFind(len(fragments))
    degree = [0] * len(fragments)
    used_sides: set[tuple[int, str]] = set()
    members: dict[int, set[int]] = {i: {i} for i in range(len(fragments))}
    selected = []
    max_track_radius = float(profile["max_track_radius_ft"])

    for row in candidates:
        if not row["passed"]:
            continue
        a = int(row["fragment_a"])
        b = int(row["fragment_b"])
        sa = str(row["a_side"])
        sb = str(row["b_side"])
        if degree[a] >= 2 or degree[b] >= 2:
            row["selection_reject_reason"] = "degree"
            continue
        if (a, sa) in used_sides or (b, sb) in used_sides:
            row["selection_reject_reason"] = "endpoint_side_reuse"
            continue
        ra = uf.find(a)
        rb = uf.find(b)
        if ra == rb:
            row["selection_reject_reason"] = "cycle"
            continue
        merged = set(members.get(ra, {a})) | set(members.get(rb, {b}))
        radius = _track_radius_p95_ft(merged, fragments, coords, voxel_size_ft)
        row["merged_track_radius_p95_ft"] = float(radius)
        if radius > max_track_radius:
            row["selection_reject_reason"] = "track_lateral_drift"
            continue
        uf.union(a, b)
        nr = uf.find(a)
        for old in (ra, rb):
            if old != nr:
                members.pop(old, None)
        members[nr] = merged
        degree[a] += 1
        degree[b] += 1
        used_sides.add((a, sa))
        used_sides.add((b, sb))
        row["selected"] = True
        row["selection_reject_reason"] = "selected"
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
    v[0] = pts[0]
    if len(v) == 1:
        v = np.vstack([pts[0], pts[-1]])
    else:
        v[-1] = pts[-1]
    keep = [0]
    for i in range(1, len(v)):
        if not np.allclose(v[i], v[keep[-1]], atol=1e-12):
            keep.append(i)
    return v[keep]


def _track_endpoint_direction(verts: np.ndarray, end: str) -> np.ndarray:
    v = np.asarray(verts, dtype=float)
    if len(v) < 2:
        return np.array([1.0, 0.0], dtype=float)
    if end == "start":
        d = v[0, :2] - v[min(1, len(v)-1), :2]
    else:
        d = v[-1, :2] - v[max(0, len(v)-2), :2]
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-12 else np.array([1.0, 0.0], dtype=float)


def _pole_attachment_candidate(
    endpoint_vox: np.ndarray,
    outward_axis_xy: np.ndarray,
    poles: pd.DataFrame | None,
    voxel_size_ft: float,
    profile: dict[str, Any],
) -> dict[str, Any] | None:
    if poles is None or poles.empty:
        return None
    max_dist = float(profile["pole_attach_radius_ft"])
    max_angle = float(profile["pole_attach_max_angle_deg"])
    min_height_frac = float(profile["pole_attach_min_height_fraction"])
    standoff_min = float(profile.get("pole_surface_standoff_min_ft", 0.5))
    p = np.asarray(endpoint_vox, dtype=float)
    best = None
    for r in poles.itertuples(index=False):
        bx, by, bz = float(r.base_x), float(r.base_y), float(r.base_z)
        tx, ty, tz = float(r.top_x), float(r.top_y), float(r.top_z)
        zlo, zhi = min(bz, tz), max(bz, tz)
        if zhi - zlo <= 1e-9:
            continue
        frac = (float(p[2]) - zlo) / (zhi - zlo)
        if frac < min_height_frac or frac > 1.10:
            continue
        u = float(np.clip((float(p[2]) - bz) / (tz - bz) if abs(tz-bz) > 1e-9 else 1.0, 0.0, 1.0))
        pole_xy = np.array([bx + u*(tx-bx), by + u*(ty-by)], dtype=float)
        to_pole_vox = pole_xy - p[:2]
        h_ft = float(np.linalg.norm(to_pole_vox) * float(voxel_size_ft))
        if h_ft > max_dist or h_ft <= 1e-9:
            continue
        angle = _angle_deg(outward_axis_xy, to_pole_vox)
        if angle > max_angle:
            continue
        radius_ft = max(standoff_min, float(getattr(r, "radius_p90_ft", standoff_min)))
        radial = (p[:2] - pole_xy)
        nr = float(np.linalg.norm(radial))
        if nr <= 1e-12:
            radial = -outward_axis_xy
            nr = float(np.linalg.norm(radial))
        radial /= max(nr, 1e-12)
        anchor_xy = pole_xy + radial * (radius_ft / float(voxel_size_ft))
        anchor = np.array([anchor_xy[0], anchor_xy[1], p[2]], dtype=float)
        rec = {
            "pole_component_id": str(getattr(r, "component_id", "")),
            "endpoint_distance_to_pole_axis_ft": h_ft,
            "endpoint_to_pole_angle_deg": angle,
            "attachment_height_fraction": float(frac),
            "anchor_x": float(anchor[0]),
            "anchor_y": float(anchor[1]),
            "anchor_z": float(anchor[2]),
            "pole_surface_radius_ft": radius_ft,
            "anchor": anchor,
        }
        if best is None or (h_ft, angle) < (best[0], best[1]):
            best = (h_ft, angle, rec)
    return None if best is None else best[2]


def _attach_track_to_poles(
    verts: np.ndarray,
    poles: pd.DataFrame | None,
    voxel_size_ft: float,
    profile: dict[str, Any],
    component_id: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    v = np.asarray(verts, dtype=float)
    if len(v) < 2 or poles is None or poles.empty:
        return v, []
    rows: list[dict[str, Any]] = []
    start = _pole_attachment_candidate(v[0], _track_endpoint_direction(v, "start"), poles, voxel_size_ft, profile)
    end = _pole_attachment_candidate(v[-1], _track_endpoint_direction(v, "end"), poles, voxel_size_ft, profile)
    out = v.copy()
    if start is not None:
        rows.append({"component_id": component_id, "track_end": "start", **{k:v for k,v in start.items() if k != "anchor"}})
        out = np.vstack([start["anchor"], out])
    if end is not None:
        if start is None or str(end["pole_component_id"]) != str(start["pole_component_id"]) or len(v) > 3:
            rows.append({"component_id": component_id, "track_end": "end", **{k:v for k,v in end.items() if k != "anchor"}})
            out = np.vstack([out, end["anchor"]])
    return out, rows


def build_electrical_track_outputs(
    coords: np.ndarray,
    line_scores: np.ndarray,
    labels: np.ndarray,
    poles: pd.DataFrame,
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
        raise ValueError("Stage1 electrical-track arrays do not align")

    line_mask = lab == 2
    line_idx, components = connected_components_26(c, line_mask, grid_size)
    fragments = [describe_fragment(i, idx, c, voxel_size_ft) for i, idx in enumerate(components)]
    candidates = candidate_fragment_bridges(fragments, c, voxel_size_ft, profile, poles=poles)
    selected = select_fragment_bridges(fragments, candidates, c, voxel_size_ft, profile)
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
    attachment_rows: list[dict[str, Any]] = []
    track_voxel_count = 0
    singleton_voxels = 0
    max_bridge_gap = 0.0
    max_track_radius = 0.0

    for track_no, group in enumerate(groups, 1):
        vox = np.unique(np.concatenate([fragments[g].voxel_indices for g in group])).astype(np.int64)
        if len(vox) < 2:
            singleton_voxels += int(len(vox))
            continue
        cid = f"S1E{track_no:05d}"
        observed_verts = _ordered_observed_vertices(
            vox, c, voxel_size_ft, float(profile.get("vertex_bin_ft", 1.0))
        )
        if len(observed_verts) < 2:
            singleton_voxels += int(len(vox))
            continue
        verts, track_attachments = _attach_track_to_poles(
            observed_verts, poles, voxel_size_ft, profile, cid
        )
        attachment_rows.extend(track_attachments)

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
        track_radius = _track_radius_p95_ft(set(group), fragments, c, voxel_size_ft)
        max_track_radius = max(max_track_radius, track_radius)
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
            "accept_mode": "stage1_inferred_electrical_track",
            "stage1_line_label_fraction": 1.0,
            "raw_fragment_count": int(len(group)),
            "bridge_count": int(bridge_count),
            "pole_attachment_count": int(len(track_attachments)),
            "max_bridge_gap_ft": float(local_max_gap),
            "track_radius_p95_ft": float(track_radius),
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
            "pole_attachment_count": int(len(track_attachments)),
            "max_bridge_gap_ft": float(local_max_gap),
            "track_radius_p95_ft": float(track_radius),
            "horizontal_span_ft": horizontal_span,
            "vertical_span_ft": vertical_span,
            "score_mean": score_mean,
        })
        track_voxel_count += int(len(vox))

    if track_voxel_count + singleton_voxels != len(line_idx):
        raise RuntimeError(
            f"Stage1 line voxel accounting mismatch: tracks={track_voxel_count} "
            f"singletons={singleton_voxels} stage1={len(line_idx)}"
        )

    blocked_parallel = sum(1 for r in candidates if r.get("reject_reason") in {
        "longitudinal_overlap", "lane_center_offset", "endpoint_lateral_jump", "axis_angle"
    })
    blocked_pole = sum(1 for r in candidates if r.get("reject_reason") == "near_pole_bridge_guard")
    blocked_drift = sum(1 for r in candidates if r.get("selection_reject_reason") == "track_lateral_drift")

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
        "pole_attachment_rows": attachment_rows,
        "audit": {
            "stage1_inferred_line_voxels": int(len(line_idx)),
            "accepted_stage1_line_voxels": int(len(line_idx)),
            "stage1_to_stage2_voxel_preservation": 1.0,
            "raw_stage1_line_components": int(len(components)),
            "joined_stage2_tracks": int(len(line_rows)),
            "selected_fragment_bridges": int(len(selected)),
            "max_selected_bridge_gap_ft": float(max_bridge_gap),
            "max_track_radius_p95_ft": float(max_track_radius),
            "unjoined_singleton_line_voxels": int(singleton_voxels),
            "parallel_or_cross_lane_bridges_blocked": int(blocked_parallel),
            "near_pole_line_to_line_bridges_blocked": int(blocked_pole),
            "track_drift_bridges_blocked": int(blocked_drift),
            "pole_attachments": int(len(attachment_rows)),
            "runtime_gt_usage": False,
            "synthetic_line_voxels": 0,
            "pole_pair_inference": False,
            "line_refiner_used": False,
            "line_hysteresis_used": False,
            "line_geometry_source": "stage1_class2_electrical_tracks_plus_pole_surface_attachments",
            "bridge_requires_stage1_class2_both_sides": True,
            "line_to_line_bridge_near_pole_allowed": False,
            "parallel_lane_merge_allowed": False,
        },
    }


class Stage1ElectricalTrackStage2Processor:
    """Production poles + electrical-safe Stage-1 inferred conductor tracks."""

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
        joined = build_electrical_track_outputs(
            np.asarray(item["coords"], dtype=np.int32),
            np.asarray(pred["line"], dtype=np.float32),
            labels,
            baseline["poles"].copy(),
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
            "pole_attachment_rows": joined["pole_attachment_rows"],
            "stage1_electrical_track_audit": {
                **joined["audit"],
                "stage1_label_source": label_source,
                "production_poles_preserved": True,
                "production_line_outputs_replaced": True,
                "runtime_version": STAGE1_ELECTRICAL_TRACK_RUNTIME_VERSION,
                "production_processor_module": self.production_processor_module,
                "profile": self.profile,
            },
            "timing": {
                "production_stage2_ms": float(baseline_ms),
                "stage1_electrical_track_ms": float(join_ms),
                "stage2_total_ms": float(baseline_ms + join_ms),
            },
        }
