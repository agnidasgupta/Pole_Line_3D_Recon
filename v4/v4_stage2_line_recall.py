#!/usr/bin/env python3
from __future__ import annotations

import heapq
import json
import math
import os
from collections import deque
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_OFFSETS_26 = np.asarray(
    [(dx, dy, dz)
     for dx in (-1, 0, 1)
     for dy in (-1, 0, 1)
     for dz in (-1, 0, 1)
     if not (dx == 0 and dy == 0 and dz == 0)],
    dtype=np.int32,
)


@dataclass(frozen=True)
class LineRecallConfig:
    voxel_size_ft: float = 0.5
    weak_threshold: float = 0.08
    bridge_score_floor: float = 0.005
    bridge_max_gap_ft: float = 18.0
    bridge_max_axis_angle_deg: float = 12.0
    bridge_max_gap_angle_deg: float = 22.0
    bridge_corridor_radius_ft: float = 1.25
    bridge_max_tortuosity: float = 1.35
    bridge_min_mean_line_score: float = 0.02
    bridge_min_component_length_ft: float = 3.0
    bridge_min_component_voxels: int = 5
    bridge_max_expansions: int = 20000
    parallel_weak_threshold: float = 0.05
    parallel_min_component_voxels: int = 8
    parallel_min_length_ft: float = 6.0
    parallel_min_linearity: float = 0.86
    parallel_max_transverse_rms_ft: float = 0.9
    parallel_max_axis_angle_deg: float = 8.0
    parallel_max_offset_ft: float = 12.0
    parallel_min_overlap_fraction: float = 0.50
    parallel_min_mean_line_score: float = 0.035
    log_audit: bool = True

    @staticmethod
    def from_env(strong_threshold: float) -> "LineRecallConfig":
        def f(name: str, default: float) -> float:
            return float(os.environ.get(name, default))

        def i(name: str, default: int) -> int:
            return int(os.environ.get(name, default))

        weak_default = min(0.08, max(0.03, float(strong_threshold) * 0.45))
        return LineRecallConfig(
            voxel_size_ft=f("V4_LINE_RECALL_VOXEL_SIZE_FT", 0.5),
            weak_threshold=f("V4_LINE_RECALL_WEAK_THRESHOLD", weak_default),
            bridge_score_floor=f("V4_LINE_RECALL_BRIDGE_SCORE_FLOOR", 0.005),
            bridge_max_gap_ft=f("V4_LINE_RECALL_BRIDGE_MAX_GAP_FT", 18.0),
            bridge_max_axis_angle_deg=f("V4_LINE_RECALL_BRIDGE_MAX_AXIS_ANGLE_DEG", 12.0),
            bridge_max_gap_angle_deg=f("V4_LINE_RECALL_BRIDGE_MAX_GAP_ANGLE_DEG", 22.0),
            bridge_corridor_radius_ft=f("V4_LINE_RECALL_BRIDGE_CORRIDOR_RADIUS_FT", 1.25),
            bridge_max_tortuosity=f("V4_LINE_RECALL_BRIDGE_MAX_TORTUOSITY", 1.35),
            bridge_min_mean_line_score=f("V4_LINE_RECALL_BRIDGE_MIN_MEAN_SCORE", 0.02),
            bridge_min_component_length_ft=f("V4_LINE_RECALL_BRIDGE_MIN_COMPONENT_LENGTH_FT", 3.0),
            bridge_min_component_voxels=i("V4_LINE_RECALL_BRIDGE_MIN_COMPONENT_VOXELS", 5),
            bridge_max_expansions=i("V4_LINE_RECALL_BRIDGE_MAX_EXPANSIONS", 20000),
            parallel_weak_threshold=f("V4_LINE_RECALL_PARALLEL_WEAK_THRESHOLD", 0.05),
            parallel_min_component_voxels=i("V4_LINE_RECALL_PARALLEL_MIN_COMPONENT_VOXELS", 8),
            parallel_min_length_ft=f("V4_LINE_RECALL_PARALLEL_MIN_LENGTH_FT", 6.0),
            parallel_min_linearity=f("V4_LINE_RECALL_PARALLEL_MIN_LINEARITY", 0.86),
            parallel_max_transverse_rms_ft=f("V4_LINE_RECALL_PARALLEL_MAX_TRANSVERSE_RMS_FT", 0.9),
            parallel_max_axis_angle_deg=f("V4_LINE_RECALL_PARALLEL_MAX_AXIS_ANGLE_DEG", 8.0),
            parallel_max_offset_ft=f("V4_LINE_RECALL_PARALLEL_MAX_OFFSET_FT", 12.0),
            parallel_min_overlap_fraction=f("V4_LINE_RECALL_PARALLEL_MIN_OVERLAP_FRACTION", 0.50),
            parallel_min_mean_line_score=f("V4_LINE_RECALL_PARALLEL_MIN_MEAN_SCORE", 0.035),
            log_audit=os.environ.get("V4_LINE_RECALL_LOG", "1") not in {"0", "false", "False"},
        )


def _normalize(a: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(a))
    if n <= 1e-12:
        return np.zeros_like(a, dtype=np.float64)
    return np.asarray(a, dtype=np.float64) / n


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    aa = _normalize(a)
    bb = _normalize(b)
    c = float(np.clip(abs(np.dot(aa, bb)), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _unsigned_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    aa = _normalize(a)
    bb = _normalize(b)
    c = float(np.clip(np.dot(aa, bb), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _coord_map(coords: np.ndarray) -> Dict[Tuple[int, int, int], int]:
    return {tuple(map(int, c)): i for i, c in enumerate(np.asarray(coords, dtype=np.int32))}


def _seeded_hysteresis(coords: np.ndarray, strong: np.ndarray, weak: np.ndarray, cmap: Dict[Tuple[int, int, int], int]) -> np.ndarray:
    keep = np.asarray(strong, dtype=bool).copy()
    q: deque[int] = deque(np.flatnonzero(strong).tolist())
    while q:
        i = q.popleft()
        c = coords[i]
        for off in _OFFSETS_26:
            j = cmap.get((int(c[0] + off[0]), int(c[1] + off[1]), int(c[2] + off[2])))
            if j is None or keep[j] or not weak[j]:
                continue
            keep[j] = True
            q.append(j)
    return keep


def _components(coords: np.ndarray, mask: np.ndarray, cmap: Dict[Tuple[int, int, int], int]) -> List[np.ndarray]:
    active = np.asarray(mask, dtype=bool)
    unseen = set(np.flatnonzero(active).tolist())
    out: List[np.ndarray] = []
    while unseen:
        root = unseen.pop()
        comp = [root]
        q = deque([root])
        while q:
            i = q.popleft()
            c = coords[i]
            for off in _OFFSETS_26:
                j = cmap.get((int(c[0] + off[0]), int(c[1] + off[1]), int(c[2] + off[2])))
                if j is None or j not in unseen:
                    continue
                unseen.remove(j)
                comp.append(j)
                q.append(j)
        out.append(np.asarray(comp, dtype=np.int64))
    return out


def _geom(coords: np.ndarray, idx: np.ndarray, voxel_size_ft: float) -> Optional[dict]:
    if idx.size < 2:
        return None
    pts = coords[idx].astype(np.float64) * float(voxel_size_ft)
    center = pts.mean(axis=0)
    x = pts - center
    try:
        _, s, vh = np.linalg.svd(x, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    axis = _normalize(vh[0])
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    t = x @ axis
    i0_local = int(np.argmin(t))
    i1_local = int(np.argmax(t))
    residual = x - np.outer(t, axis)
    transverse = np.linalg.norm(residual, axis=1)
    eig = s * s
    linearity = float(eig[0] / max(float(np.sum(eig)), 1e-12))
    return {
        "idx": idx,
        "center": center,
        "axis": axis,
        "length_ft": float(t.max() - t.min()),
        "linearity": linearity,
        "transverse_rms_ft": float(np.sqrt(np.mean(transverse * transverse))),
        "endpoint_indices": (int(idx[i0_local]), int(idx[i1_local])),
        "t_min": float(t.min()),
        "t_max": float(t.max()),
    }


def _point_segment_distance_ft(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    den = float(np.dot(ab, ab))
    if den <= 1e-12:
        return float(np.linalg.norm(p - a))
    u = float(np.clip(np.dot(p - a, ab) / den, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + u * ab)))


def _astar_bridge(
    coords: np.ndarray,
    cmap: Dict[Tuple[int, int, int], int],
    line_score: np.ndarray,
    pole_score: Optional[np.ndarray],
    start_idx: int,
    goal_idx: int,
    cfg: LineRecallConfig,
) -> Optional[np.ndarray]:
    vs = float(cfg.voxel_size_ft)
    p0 = coords[start_idx].astype(np.float64) * vs
    p1 = coords[goal_idx].astype(np.float64) * vs
    direct = float(np.linalg.norm(p1 - p0))
    if direct <= 1e-9:
        return np.asarray([start_idx, goal_idx], dtype=np.int64)

    def heuristic(i: int) -> float:
        p = coords[i].astype(np.float64) * vs
        return float(np.linalg.norm(p1 - p))

    g = {start_idx: 0.0}
    parent: Dict[int, int] = {}
    pq: List[Tuple[float, float, int]] = [(heuristic(start_idx), 0.0, start_idx)]
    closed = set()
    expansions = 0

    while pq and expansions < int(cfg.bridge_max_expansions):
        _, gcost, i = heapq.heappop(pq)
        if i in closed:
            continue
        closed.add(i)
        expansions += 1
        if i == goal_idx:
            path = [i]
            while path[-1] != start_idx:
                path.append(parent[path[-1]])
            path.reverse()
            return np.asarray(path, dtype=np.int64)

        c = coords[i]
        for off in _OFFSETS_26:
            j = cmap.get((int(c[0] + off[0]), int(c[1] + off[1]), int(c[2] + off[2])))
            if j is None or j in closed:
                continue
            p = coords[j].astype(np.float64) * vs
            d_axis = _point_segment_distance_ft(p, p0, p1)
            if d_axis > float(cfg.bridge_corridor_radius_ft):
                continue
            ls = float(line_score[j])
            if j not in {start_idx, goal_idx} and ls < float(cfg.bridge_score_floor):
                # Do not wander through completely unsupported occupancy.
                continue
            step = vs * float(np.linalg.norm(off.astype(np.float64)))
            score_penalty = 0.75 * max(0.0, float(cfg.weak_threshold) - ls) / max(float(cfg.weak_threshold), 1e-6)
            corridor_penalty = 0.35 * d_axis / max(float(cfg.bridge_corridor_radius_ft), 1e-6)
            pole_penalty = 0.0
            if pole_score is not None:
                pole_penalty = 0.5 * max(0.0, float(pole_score[j]) - ls)
            ng = gcost + step + score_penalty + corridor_penalty + pole_penalty
            if ng >= g.get(j, float("inf")):
                continue
            g[j] = ng
            parent[j] = i
            heapq.heappush(pq, (ng + heuristic(j), ng, j))
    return None


def _bridge_components(
    coords: np.ndarray,
    current_mask: np.ndarray,
    line_score: np.ndarray,
    pole_score: Optional[np.ndarray],
    cmap: Dict[Tuple[int, int, int], int],
    cfg: LineRecallConfig,
) -> Tuple[np.ndarray, int, int]:
    add = np.zeros(len(coords), dtype=bool)
    comps = _components(coords, current_mask, cmap)
    geoms = []
    for comp in comps:
        if comp.size < int(cfg.bridge_min_component_voxels):
            continue
        g = _geom(coords, comp, cfg.voxel_size_ft)
        if g is None or g["length_ft"] < float(cfg.bridge_min_component_length_ft):
            continue
        geoms.append(g)

    candidates = 0
    accepted = 0
    for ai in range(len(geoms)):
        ga = geoms[ai]
        for bi in range(ai + 1, len(geoms)):
            gb = geoms[bi]
            if _angle_deg(ga["axis"], gb["axis"]) > float(cfg.bridge_max_axis_angle_deg):
                continue
            best = None
            for ea in ga["endpoint_indices"]:
                for eb in gb["endpoint_indices"]:
                    pa = coords[ea].astype(np.float64) * cfg.voxel_size_ft
                    pb = coords[eb].astype(np.float64) * cfg.voxel_size_ft
                    gap = float(np.linalg.norm(pb - pa))
                    if best is None or gap < best[0]:
                        best = (gap, ea, eb, pa, pb)
            if best is None:
                continue
            gap, ea, eb, pa, pb = best
            if gap <= 0.0 or gap > float(cfg.bridge_max_gap_ft):
                continue
            gap_vec = _normalize(pb - pa)
            if min(_unsigned_angle_deg(gap_vec, ga["axis"]), _unsigned_angle_deg(gap_vec, -ga["axis"])) > float(cfg.bridge_max_gap_angle_deg):
                continue
            if min(_unsigned_angle_deg(gap_vec, gb["axis"]), _unsigned_angle_deg(gap_vec, -gb["axis"])) > float(cfg.bridge_max_gap_angle_deg):
                continue
            candidates += 1
            path = _astar_bridge(coords, cmap, line_score, pole_score, ea, eb, cfg)
            if path is None or path.size < 2:
                continue
            pts = coords[path].astype(np.float64) * cfg.voxel_size_ft
            step_len = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
            if step_len / max(gap, 1e-6) > float(cfg.bridge_max_tortuosity):
                continue
            if float(np.mean(line_score[path])) < float(cfg.bridge_min_mean_line_score):
                continue
            add[path] = True
            accepted += 1
    return add, candidates, accepted


def _parallel_rescue(
    coords: np.ndarray,
    base_mask: np.ndarray,
    line_score: np.ndarray,
    cmap: Dict[Tuple[int, int, int], int],
    cfg: LineRecallConfig,
) -> Tuple[np.ndarray, int, int]:
    add = np.zeros(len(coords), dtype=bool)
    base_geoms = []
    for comp in _components(coords, base_mask, cmap):
        g = _geom(coords, comp, cfg.voxel_size_ft)
        if g is not None and g["length_ft"] >= float(cfg.parallel_min_length_ft):
            base_geoms.append(g)

    weak_pool = (line_score >= float(cfg.parallel_weak_threshold)) & (~base_mask)
    tested = 0
    accepted = 0
    for comp in _components(coords, weak_pool, cmap):
        if comp.size < int(cfg.parallel_min_component_voxels):
            continue
        g = _geom(coords, comp, cfg.voxel_size_ft)
        if g is None:
            continue
        if g["length_ft"] < float(cfg.parallel_min_length_ft):
            continue
        if g["linearity"] < float(cfg.parallel_min_linearity):
            continue
        if g["transverse_rms_ft"] > float(cfg.parallel_max_transverse_rms_ft):
            continue
        if float(np.mean(line_score[comp])) < float(cfg.parallel_min_mean_line_score):
            continue
        tested += 1
        for b in base_geoms:
            if _angle_deg(g["axis"], b["axis"]) > float(cfg.parallel_max_axis_angle_deg):
                continue
            delta = g["center"] - b["center"]
            along = float(np.dot(delta, b["axis"]))
            perp = delta - along * b["axis"]
            offset = float(np.linalg.norm(perp))
            if offset > float(cfg.parallel_max_offset_ft):
                continue
            # Project weak component into base-axis coordinates and demand overlap.
            wpts = coords[comp].astype(np.float64) * cfg.voxel_size_ft
            wt = (wpts - b["center"]) @ b["axis"]
            overlap = max(0.0, min(float(wt.max()), b["t_max"]) - max(float(wt.min()), b["t_min"]))
            frac = overlap / max(g["length_ft"], 1e-6)
            if frac < float(cfg.parallel_min_overlap_fraction):
                continue
            add[comp] = True
            accepted += 1
            break
    return add, tested, accepted


def recover_line_candidates(
    *,
    coords_xyz: np.ndarray,
    line_score: np.ndarray,
    strong_mask: np.ndarray,
    strong_threshold: float,
    pole_score: Optional[np.ndarray] = None,
    background_score: Optional[np.ndarray] = None,
    config: Optional[LineRecallConfig] = None,
) -> Tuple[np.ndarray, dict]:
    """Recover physically continuous line support without inventing empty voxels.

    All returned True entries refer to coordinates already present in coords_xyz.
    Ground-truth labels are never consumed.
    """
    coords = np.asarray(coords_xyz, dtype=np.int32)
    score = np.asarray(line_score, dtype=np.float64).reshape(-1)
    strong = np.asarray(strong_mask, dtype=bool).reshape(-1)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords_xyz must be [N,3], got {coords.shape}")
    if len(coords) != len(score) or len(score) != len(strong):
        raise ValueError(f"length mismatch coords={len(coords)} line_score={len(score)} strong={len(strong)}")
    pscore = None if pole_score is None else np.asarray(pole_score, dtype=np.float64).reshape(-1)
    if pscore is not None and len(pscore) != len(coords):
        pscore = None

    cfg = config or LineRecallConfig.from_env(float(strong_threshold))
    cmap = _coord_map(coords)

    weak = score >= float(cfg.weak_threshold)
    hysteresis = _seeded_hysteresis(coords, strong, weak, cmap)
    hysteresis_added = hysteresis & (~strong)

    bridge_add, bridge_candidates, bridge_accepted = _bridge_components(
        coords, hysteresis, score, pscore, cmap, cfg
    )
    after_bridge = hysteresis | bridge_add

    parallel_add, parallel_tested, parallel_accepted = _parallel_rescue(
        coords, after_bridge, score, cmap, cfg
    )
    after_parallel = after_bridge | parallel_add

    # A rescued geometric component is allowed to pull in immediately connected
    # weak support, exactly like hysteresis around an original strong seed.
    final_mask = _seeded_hysteresis(coords, after_parallel, weak, cmap)

    audit = {
        "strong_threshold": float(strong_threshold),
        "config": asdict(cfg),
        "occupied_voxels": int(len(coords)),
        "strong_voxels": int(strong.sum()),
        "weak_voxels": int(weak.sum()),
        "hysteresis_added_voxels": int(hysteresis_added.sum()),
        "bridge_added_voxels": int((bridge_add & ~hysteresis).sum()),
        "parallel_added_voxels": int((parallel_add & ~after_bridge).sum()),
        "final_line_candidate_voxels": int(final_mask.sum()),
        "bridge_candidate_pairs": int(bridge_candidates),
        "bridge_accepted_pairs": int(bridge_accepted),
        "parallel_components_tested": int(parallel_tested),
        "parallel_components_accepted": int(parallel_accepted),
    }
    if cfg.log_audit:
        print("[stage2-line-recall] " + json.dumps(audit, sort_keys=True), flush=True)
    return final_mask, audit


def _auto_array_candidate(local_vars, n: int):
    """Resolve the sparse occupied-voxel [N,3] coordinate array conservatively."""
    rows=[]
    reject_tokens=("prob","score","logit","feature","feat","physical","mask","label","target","pred")
    for name,value in local_vars.items():
        if name.startswith("_"):
            continue
        try:
            a=np.asarray(value)
        except Exception:
            continue
        if a.ndim != 2 or a.shape != (n,3):
            continue
        if not np.issubdtype(a.dtype,np.number):
            continue
        low=name.lower()
        score=0
        if "coord" in low: score += 100
        if "xyz" in low: score += 95
        if "voxel" in low: score += 70
        if low in {"pts","points","p","indices","index","idx"}: score += 45
        if "point" in low: score += 40
        if any(t in low for t in reject_tokens): score -= 80
        if np.issubdtype(a.dtype,np.integer): score += 25
        else:
            finite=np.isfinite(a)
            if finite.all() and np.allclose(a,np.rint(a),atol=1e-6): score += 15
        # Sparse voxel coordinates should normally have many unique rows.
        try:
            uniq=len(np.unique(a[:min(len(a),5000)],axis=0))
            if uniq >= min(len(a),5000)*0.95: score += 10
        except Exception:
            pass
        rows.append((score,name,a))
    rows.sort(key=lambda x:(-x[0],x[1]))
    meta=[{"name":name,"score":int(score),"dtype":str(a.dtype),"shape":list(a.shape)} for score,name,a in rows]
    if not rows:
        raise RuntimeError("V4 line recall: no numeric [N,3] coordinate candidate found in apply_stage2 locals")
    if len(rows)>1 and rows[0][0] <= rows[1][0] + 10:
        raise RuntimeError("V4 line recall: ambiguous [N,3] coordinate candidates: "+json.dumps(meta[:8]))
    if rows[0][0] < 20:
        raise RuntimeError("V4 line recall: low-confidence coordinate candidate: "+json.dumps(meta[:8]))
    return rows[0][1], np.asarray(rows[0][2],dtype=np.int32), meta


def _auto_optional_score(local_vars, n: int, token: str):
    rows=[]
    for name,value in local_vars.items():
        low=name.lower()
        if token not in low or not any(t in low for t in ("prob","score","logit","evidence")):
            continue
        try:
            a=np.asarray(value)
        except Exception:
            continue
        if a.size != n or not np.issubdtype(a.dtype,np.number):
            continue
        score=10
        if "prob" in low or "score" in low: score += 5
        rows.append((score,name,a.reshape(-1)))
    if not rows: return None,None
    rows.sort(key=lambda x:(-x[0],x[1]))
    return rows[0][1], np.asarray(rows[0][2],dtype=np.float64)


def recover_line_candidates_auto(
    *,
    local_vars,
    line_score,
    strong_mask,
    strong_threshold,
    physical_mask=None,
    config=None,
):
    """Runtime-safe hook for V4 apply_stage2.

    Resolves the occupied sparse coordinate array from apply_stage2 locals,
    preserves the original strong mask, and never promotes voxels outside the
    existing physical mask.
    """
    score=np.asarray(line_score,dtype=np.float64).reshape(-1)
    strong=np.asarray(strong_mask,dtype=bool).reshape(-1)
    n=len(score)
    if len(strong)!=n:
        raise ValueError(f"V4 line recall: strong/score mismatch {len(strong)} vs {n}")
    if physical_mask is None:
        physical=np.ones(n,dtype=bool)
    else:
        physical=np.asarray(physical_mask,dtype=bool).reshape(-1)
        if len(physical)!=n:
            raise ValueError(f"V4 line recall: physical/score mismatch {len(physical)} vs {n}")

    coord_name,coords,coord_meta=_auto_array_candidate(local_vars,n)
    pole_name,pole_score=_auto_optional_score(local_vars,n,"pole")
    bg_name,bg_score=_auto_optional_score(local_vars,n,"background")
    if bg_score is None:
        bg_name,bg_score=_auto_optional_score(local_vars,n,"bg")

    safe_score=score.copy()
    safe_score[~physical]=-np.inf
    safe_strong=strong & physical

    final_mask,audit=recover_line_candidates(
        coords_xyz=coords,
        line_score=safe_score,
        strong_mask=safe_strong,
        strong_threshold=float(strong_threshold),
        pole_score=pole_score,
        background_score=bg_score,
        config=config,
    )
    final_mask=np.asarray(final_mask,dtype=bool) & physical
    if not np.all(final_mask[safe_strong]):
        raise RuntimeError("V4 line recall invariant failed: original strong physical voxels were removed")
    audit.update({
        "auto_coords_var":coord_name,
        "auto_pole_score_var":pole_name,
        "auto_background_score_var":bg_name,
        "physical_voxels":int(physical.sum()),
        "nonphysical_promoted":int((final_mask & ~physical).sum()),
        "coord_candidates":coord_meta[:8],
    })
    if os.environ.get("V4_LINE_RECALL_LOG","1") not in {"0","false","False"}:
        print("[stage2-line-recall-auto] "+json.dumps(audit,sort_keys=True),flush=True)
    return final_mask,audit

