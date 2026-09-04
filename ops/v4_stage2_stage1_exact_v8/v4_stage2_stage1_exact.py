#!/usr/bin/env python3
"""Exact Stage-1 inferred-line passthrough for V4 Stage 2 experiments.

This module deliberately does *not* infer conductor spans from pole pairs.
Every emitted line edge joins two adjacent occupied Stage-1 voxels whose final
Stage-1 deployed label is class 2 (line). There is no Stage-2 line hysteresis,
no Stage-2 line refiner, no gap filling, and no GT access at runtime.

Production V4 Stage-2 pole outputs are retained unchanged. Line outputs are
replaced by exact Stage-1 line-mask adjacency edges for verification.
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd

STAGE1_EXACT_RUNTIME_VERSION = "stage1-exact-v8-20260903"


def _coord_keys(coords: np.ndarray, grid_size: tuple[int, int, int]) -> np.ndarray:
    c = np.asarray(coords, dtype=np.int64)
    if not len(c):
        return np.empty(0, dtype=np.int64)
    gx, gy, _ = map(int, grid_size)
    return (c[:, 2] * gy + c[:, 1]) * gx + c[:, 0]


def _forward_offsets_26() -> list[tuple[int, int, int]]:
    """Return one orientation of each of the 26 voxel-neighbour directions."""
    out: list[tuple[int, int, int]] = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                # Lexicographic positive half-space: every undirected edge once.
                if (dz, dy, dx) > (0, 0, 0):
                    out.append((dx, dy, dz))
    if len(out) != 13:
        raise AssertionError(f"expected 13 forward 26-neighbour offsets, got {len(out)}")
    return out


_FORWARD_26 = _forward_offsets_26()


def resolve_deployed_stage1_labels(
    pred: dict[str, np.ndarray],
    calibration: dict[str, Any],
) -> tuple[np.ndarray, str]:
    """Resolve the final Stage-1 deployed class labels without using GT.

    If the saved prediction artifact contains an explicit final/deployed label
    vector, use it. Otherwise reproduce production V4's calibrated decision via
    ``v4_realtime_core.label_from_scores``.
    """
    pole = np.asarray(pred["pole"], dtype=np.float32)
    line = np.asarray(pred["line"], dtype=np.float32)
    if len(pole) != len(line):
        raise ValueError("Stage1 pole/line score lengths differ")

    explicit_keys = (
        "deployed_labels",
        "deployed_label",
        "inferred_labels",
        "inferred_label",
        "pred_labels",
        "pred_label",
    )
    for key in explicit_keys:
        if key not in pred:
            continue
        arr = np.asarray(pred[key])
        if arr.ndim != 1 or len(arr) != len(pole):
            continue
        if not np.issubdtype(arr.dtype, np.number):
            continue
        labels = arr.astype(np.int8, copy=False)
        values = set(map(int, np.unique(labels)))
        if values <= {0, 1, 2}:
            return labels.copy(), f"pred.{key}"

    from v4_realtime_core import label_from_scores

    labels = label_from_scores(
        pole,
        line,
        float(calibration["pole_threshold"]),
        float(calibration["line_threshold"]),
    )
    return np.asarray(labels, dtype=np.int8), "v4_realtime_core.label_from_scores"


def exact_adjacency_edges(
    coords: np.ndarray,
    line_mask: np.ndarray,
    grid_size: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build exact undirected 26-neighbour edges among inferred line voxels.

    Returns ``(line_indices, edge_i, edge_j)`` where ``edge_i`` and ``edge_j``
    index the original ``coords`` array. No endpoint is created synthetically.
    """
    c = np.asarray(coords, dtype=np.int32)
    m = np.asarray(line_mask, dtype=bool)
    if len(c) != len(m):
        raise ValueError("coords and line_mask lengths differ")
    line_indices = np.flatnonzero(m).astype(np.int64)
    if not len(line_indices):
        return line_indices, np.empty(0, np.int64), np.empty(0, np.int64)

    gx, gy, gz = map(int, grid_size)
    line_coords = c[line_indices]
    keys = _coord_keys(line_coords, grid_size)
    key_to_original = {int(key): int(idx) for key, idx in zip(keys, line_indices)}

    ei: list[int] = []
    ej: list[int] = []
    for original_idx in line_indices:
        x, y, z = map(int, c[int(original_idx)])
        for dx, dy, dz in _FORWARD_26:
            nx, ny, nz = x + dx, y + dy, z + dz
            if not (0 <= nx < gx and 0 <= ny < gy and 0 <= nz < gz):
                continue
            key = int((nz * gy + ny) * gx + nx)
            other = key_to_original.get(key)
            if other is None:
                continue
            ei.append(int(original_idx))
            ej.append(int(other))

    return (
        line_indices,
        np.asarray(ei, dtype=np.int64),
        np.asarray(ej, dtype=np.int64),
    )


def build_exact_line_outputs(
    coords: np.ndarray,
    line_scores: np.ndarray,
    labels: np.ndarray,
    file_id: str,
    slice_seq: int,
    grid_size: tuple[int, int, int],
    voxel_size_ft: float,
) -> dict[str, Any]:
    """Create Stage-2 line outputs from exact adjacent Stage-1 line voxels."""
    c = np.asarray(coords, dtype=np.int32)
    scores = np.asarray(line_scores, dtype=np.float32)
    lab = np.asarray(labels, dtype=np.int8)
    if not (len(c) == len(scores) == len(lab)):
        raise ValueError("Stage1 exact-line arrays do not align")

    line_mask = lab == 2
    line_indices, edge_i, edge_j = exact_adjacency_edges(c, line_mask, grid_size)

    # Degree is used only for diagnostics. Every Stage-1 line voxel is retained
    # in the voxel audit even when it is isolated and therefore cannot form a
    # physically connected line edge without inventing another voxel.
    degree = np.zeros(len(c), dtype=np.int32)
    if len(edge_i):
        np.add.at(degree, edge_i, 1)
        np.add.at(degree, edge_j, 1)

    line_rows: list[dict[str, Any]] = []
    vertex_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    line_points: dict[str, np.ndarray] = {}
    max_step_ft = 0.0

    for n, (ia, ib) in enumerate(zip(edge_i, edge_j), 1):
        cid = f"S1E{n:07d}"
        pa = c[int(ia)].astype(np.float64)
        pb = c[int(ib)].astype(np.float64)
        delta_vox = pb - pa
        delta_ft = delta_vox * float(voxel_size_ft)
        step_ft = float(np.linalg.norm(delta_ft))
        max_step_ft = max(max_step_ft, step_ft)
        horizontal_ft = float(np.linalg.norm(delta_ft[:2]))
        vertical_ft = float(abs(delta_ft[2]))
        verticality = float(vertical_ft / max(step_ft, 1e-12))
        score_mean = float((scores[int(ia)] + scores[int(ib)]) * 0.5)

        line_rows.append({
            "file_id": str(file_id),
            "component_id": cid,
            "slice_seq": int(slice_seq),
            # Refiner is deliberately bypassed for exact Stage-1 line labels.
            "refiner_probability": float("nan"),
            "horizontal_span_ft": horizontal_ft,
            "vertical_span_ft": vertical_ft,
            "verticality": verticality,
            "tortuosity": 1.0,
            "vertex_count": 2,
        })
        for vertex_index, q in enumerate((pa, pb)):
            vertex_rows.append({
                "file_id": str(file_id),
                "component_id": cid,
                "slice_seq": int(slice_seq),
                "vertex_index": int(vertex_index),
                "x": float(q[0]),
                "y": float(q[1]),
                "z": float(q[2]),
            })
        component_rows.append({
            "component_id": cid,
            "class_name": "line",
            "n_voxels": 2,
            "score_mean": score_mean,
            "component_accept": True,
            "accept_mode": "stage1_exact_adjacent_pair",
            "stage1_line_label_fraction": 1.0,
            "synthetic_line_voxels": 0,
            "runtime_gt_usage": False,
            "edge_step_ft": step_ft,
            "file_id": str(file_id),
            "slice_seq": int(slice_seq),
        })
        edge_rows.append({
            "component_id": cid,
            "x1": int(pa[0]),
            "y1": int(pa[1]),
            "z1": int(pa[2]),
            "x2": int(pb[0]),
            "y2": int(pb[1]),
            "z2": int(pb[2]),
            "edge_step_ft": step_ft,
            "line_score_mean": score_mean,
        })
        line_points[cid] = np.vstack([pa, pb]).astype(np.int32)

    isolated_indices = line_indices[degree[line_indices] == 0]
    max_allowed = math.sqrt(3.0) * float(voxel_size_ft) + 1e-9
    if max_step_ft > max_allowed:
        raise RuntimeError(
            f"exact Stage1 edge exceeded one-voxel adjacency: {max_step_ft} > {max_allowed}"
        )

    line_voxel_keys = set(map(int, _coord_keys(c[line_indices], grid_size)))
    emitted_endpoint_indices = set(map(int, edge_i)) | set(map(int, edge_j))
    emitted_endpoint_keys = set(
        map(int, _coord_keys(c[np.asarray(sorted(emitted_endpoint_indices), dtype=np.int64)], grid_size))
    ) if emitted_endpoint_indices else set()
    if not emitted_endpoint_keys.issubset(line_voxel_keys):
        raise RuntimeError("emitted Stage2 line endpoint not backed by Stage1 inferred line voxel")

    return {
        "lines_rows": line_rows,
        "vertices_rows": vertex_rows,
        "components_rows": component_rows,
        "edge_rows": edge_rows,
        "line_points": line_points,
        "line_indices": line_indices,
        "isolated_indices": isolated_indices,
        "audit": {
            "stage1_inferred_line_voxels": int(len(line_indices)),
            "accepted_stage1_line_voxels": int(len(line_indices)),
            "stage1_to_stage2_voxel_preservation": 1.0 if len(line_indices) else 1.0,
            "exact_line_edges": int(len(edge_i)),
            "isolated_stage1_line_voxels": int(len(isolated_indices)),
            "max_exact_edge_step_ft": float(max_step_ft),
            "max_allowed_edge_step_ft": float(max_allowed),
            "runtime_gt_usage": False,
            "synthetic_line_voxels": 0,
            "pole_pair_inference": False,
            "line_hysteresis_used": False,
            "line_refiner_used": False,
            "line_geometry_source": "adjacent_stage1_inferred_class2_voxel_pairs",
        },
    }


class Stage1ExactStage2Processor:
    """Production poles + exact Stage-1 inferred line connectivity."""

    def __init__(
        self,
        stage2_bundle: str,
        calibration_json: str,
        grid_size: tuple[int, int, int] = (400, 400, 200),
        voxel_size_ft: float = 0.5,
    ) -> None:
        from v4_realtime_core import load_calibration
        from v4_stage2_runtime import V4Stage2Processor

        self.grid = tuple(map(int, grid_size))
        self.voxel = float(voxel_size_ft)
        self.calibration = load_calibration(calibration_json)
        self.production = V4Stage2Processor(
            stage2_bundle,
            grid_size=self.grid,
            voxel_size_ft=self.voxel,
            pole_candidate_threshold=0.15,
            line_candidate_threshold=0.08,
            line_weak_threshold=0.04,
            line_competition_ratio=0.55,
            pole_min_voxels=4,
            line_min_voxels=3,
            edge_width_vox=10,
        )

    def process(
        self,
        item: dict[str, Any],
        pred: dict[str, np.ndarray],
        file_id: str = "slice",
        slice_seq: int = 0,
    ) -> dict[str, Any]:
        from v4_stage2_runtime import LINE_OUTPUT_COLUMNS, VERTEX_OUTPUT_COLUMNS

        total_start = time.perf_counter()
        baseline = self.production.process(item, pred, file_id, slice_seq)
        baseline_ms = (time.perf_counter() - total_start) * 1000.0

        labels, label_source = resolve_deployed_stage1_labels(pred, self.calibration)
        exact_start = time.perf_counter()
        exact = build_exact_line_outputs(
            np.asarray(item["coords"], dtype=np.int32),
            np.asarray(pred["line"], dtype=np.float32),
            labels,
            file_id,
            slice_seq,
            self.grid,
            self.voxel,
        )
        exact_ms = (time.perf_counter() - exact_start) * 1000.0

        # Preserve production pole component rows exactly, but replace all line
        # component rows with exact Stage1-adjacency line rows.
        base_components = baseline.get("components", pd.DataFrame()).copy()
        if not base_components.empty and "class_name" in base_components.columns:
            pole_components = base_components[
                base_components["class_name"].astype(str).eq("pole")
            ].copy()
        else:
            pole_components = pd.DataFrame()
        exact_components = pd.DataFrame(exact["components_rows"])
        components = pd.concat([pole_components, exact_components], ignore_index=True, sort=False)

        lines = pd.DataFrame(exact["lines_rows"], columns=LINE_OUTPUT_COLUMNS)
        vertices = pd.DataFrame(exact["vertices_rows"], columns=VERTEX_OUTPUT_COLUMNS)

        raw = baseline.get("raw_components", {}).copy()
        raw["line_points"] = exact["line_points"]

        return {
            "raw_components": raw,
            "components": components,
            "poles": baseline["poles"].copy(),
            "lines": lines,
            "vertices": vertices,
            "stage1_labels": labels,
            "label_source": label_source,
            "line_indices": exact["line_indices"],
            "isolated_indices": exact["isolated_indices"],
            "exact_edge_rows": exact["edge_rows"],
            "stage1_exact_audit": {
                **exact["audit"],
                "stage1_label_source": label_source,
                "production_poles_preserved": True,
                "production_line_outputs_replaced": True,
                "runtime_version": STAGE1_EXACT_RUNTIME_VERSION,
            },
            "timing": {
                "production_stage2_ms": float(baseline_ms),
                "stage1_exact_line_ms": float(exact_ms),
                "stage2_total_ms": float(baseline_ms + exact_ms),
            },
        }
