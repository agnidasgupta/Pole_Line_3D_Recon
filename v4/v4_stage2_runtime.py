#!/usr/bin/env python3
"""Apply V4 slice-local Stage-2 refiners and parameterize accepted local objects."""
from __future__ import annotations
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from v4_stage2_local import LOCAL_FEATURE_COLUMNS
from v4_stage2_line_recall import recover_line_candidates_auto

POLE_OUTPUT_COLUMNS = ["file_id","component_id","slice_seq","refiner_probability","touches_xy_edge","radius_p90_ft","verticality","base_x","base_y","base_z","top_x","top_y","top_z","height_ft","tilt_ft"]
LINE_OUTPUT_COLUMNS = ["file_id","component_id","slice_seq","refiner_probability","horizontal_span_ft","vertical_span_ft","verticality","tortuosity","vertex_count"]
VERTEX_OUTPUT_COLUMNS = ["file_id","component_id","slice_seq","vertex_index","x","y","z"]


def local_X(d, feature_columns=LOCAL_FEATURE_COLUMNS):
    return d.reindex(columns=feature_columns, fill_value=0).replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)


def _line_axis(pts):
    p = np.asarray(pts, float)
    if len(p) < 2:
        return p, np.zeros(2, float), np.array([1.0, 0.0]), np.zeros(len(p), float)
    xy = p[:, :2]
    c = np.median(xy, axis=0)
    try:
        _, _, vh = np.linalg.svd(xy - c, full_matrices=False)
        d = vh[0]
    except np.linalg.LinAlgError:
        d = np.array([1.0, 0.0])
    d = d / max(float(np.linalg.norm(d)), 1e-9)
    s = (xy - c) @ d
    return p, c, d, s


def order_line(pts):
    p, _, _, s = _line_axis(pts)
    return p[np.argsort(s, kind="stable")] if len(p) >= 2 else p


def pole_param(pts, voxel=.5):
    p = np.asarray(pts, float)
    if len(p) == 0:
        return {"base_x":np.nan,"base_y":np.nan,"base_z":np.nan,"top_x":np.nan,"top_y":np.nan,"top_z":np.nan,"height_ft":0.0,"tilt_ft":0.0}
    z = p[:, 2]
    lo = np.quantile(z, .15)
    hi = np.quantile(z, .85)
    b = np.median(p[z <= lo], axis=0)
    t = np.median(p[z >= hi], axis=0)
    return {
        "base_x":float(b[0]), "base_y":float(b[1]), "base_z":float(z.min()),
        "top_x":float(t[0]), "top_y":float(t[1]), "top_z":float(z.max()),
        "height_ft":float((z.max() - z.min() + 1) * voxel),
        "tilt_ft":float(np.linalg.norm(t[:2] - b[:2]) * voxel),
    }


def line_vertices(pts, bin_vox=2.0):
    """Order and bin a line with one PCA/SVD instead of two."""
    p, _, _, s = _line_axis(pts)
    if len(p) < 2:
        return p
    order = np.argsort(s, kind="stable")
    p = p[order]
    s = s[order]
    bins = np.floor((s - s.min()) / float(bin_vox)).astype(np.int32)
    return np.asarray([np.median(p[bins == b], axis=0) for b in np.unique(bins)], float)


def load_bundle(path):
    b = joblib.load(path)
    needed = ["pole_model", "line_model", "pole_threshold", "line_threshold"]
    miss = [x for x in needed if x not in b]
    if miss:
        raise RuntimeError(f"Stage-2 bundle missing keys: {miss}")
    cols = list(b.get("feature_columns", LOCAL_FEATURE_COLUMNS))
    forbidden = [c for c in cols if c.startswith("world_") or c in {"center_x","center_y","center_z","exact_gt_fraction","near_gt_fraction"}]
    if forbidden:
        raise RuntimeError(f"Unsafe Stage-2 model features: {forbidden}")
    b["feature_columns"] = cols
    return b


def _physical_mask(d: pd.DataFrame, name: str) -> np.ndarray:
    """Vectorized equivalents of the existing loose Stage-2 physical gates."""
    if d.empty:
        return np.zeros(0, dtype=bool)
    h = pd.to_numeric(d["horizontal_span_ft"], errors="coerce").to_numpy(float)
    z = pd.to_numeric(d["z_span_ft"], errors="coerce").to_numpy(float)
    v = pd.to_numeric(d["principal_verticality"], errors="coerce").to_numpy(float)
    if name == "pole":
        radius = pd.to_numeric(d["radius_p90_ft"], errors="coerce").to_numpy(float)
        edge = d.get("touches_xy_edge", pd.Series(False, index=d.index)).astype(bool).to_numpy()
        min_h = np.where(edge, 4.0, 10.0)
        return np.isfinite(h) & np.isfinite(z) & np.isfinite(v) & np.isfinite(radius) & (z >= min_h) & (z <= 90.0) & (radius <= 3.5) & (v >= .65) & (h / np.maximum(z, 1e-6) <= .65)
    tort = pd.to_numeric(d["xy_tortuosity"], errors="coerce").to_numpy(float)
    return np.isfinite(h) & np.isfinite(z) & np.isfinite(v) & np.isfinite(tort) & (h >= .5) & (v <= .85) & (z / np.maximum(h, 1e-6) <= 2.5) & (tort <= 4.0)


def apply_stage2(comps, bundle, file_id, slice_seq, voxel_size=.5):
    frames = []
    poles_out = []
    lines_out = []
    verts_out = []
    for name, df, points in [
        ("pole", comps["poles"], comps["pole_points"]),
        ("line", comps["lines"], comps["line_points"]),
    ]:
        d = df.copy()
        if d.empty:
            continue
        clf = bundle[name + "_model"]
        thr = float(bundle[name + "_threshold"])
        prob = clf.predict_proba(local_X(d, bundle["feature_columns"]))[:, 1]
        physical = _physical_mask(d, name)
        acc = (prob >= thr) & physical
        acc, _v4_line_recall_audit = recover_line_candidates_auto(
            local_vars=locals(),
            line_score=prob,
            strong_mask=acc,
            strong_threshold=float(thr),
            physical_mask=physical,
        )
        d["refiner_probability"] = prob
        d["physical_ok"] = physical
        d["component_accept"] = acc
        d["accept_mode"] = np.where(acc, "refiner_plus_loose_hard_gate", "rejected")
        d["file_id"] = file_id
        d["slice_seq"] = int(slice_seq)
        frames.append(d)
        accepted = d.loc[d.component_accept.astype(bool)]
        for r in accepted.itertuples(index=False):
            key = str(r.component_id)
            pts = points[key]
            if name == "pole":
                poles_out.append({
                    "file_id":file_id, "component_id":key, "slice_seq":int(slice_seq),
                    "refiner_probability":float(r.refiner_probability),
                    "touches_xy_edge":bool(r.touches_xy_edge),
                    "radius_p90_ft":float(r.radius_p90_ft),
                    "verticality":float(r.principal_verticality),
                    **pole_param(pts, voxel_size),
                })
            else:
                vv = line_vertices(pts)
                lines_out.append({
                    "file_id":file_id, "component_id":key, "slice_seq":int(slice_seq),
                    "refiner_probability":float(r.refiner_probability),
                    "horizontal_span_ft":float(r.horizontal_span_ft),
                    "vertical_span_ft":float(r.z_span_ft),
                    "verticality":float(r.principal_verticality),
                    "tortuosity":float(r.xy_tortuosity), "vertex_count":int(len(vv)),
                })
                for vi, q in enumerate(vv):
                    verts_out.append({
                        "file_id":file_id, "component_id":key, "slice_seq":int(slice_seq),
                        "vertex_index":int(vi), "x":float(q[0]), "y":float(q[1]), "z":float(q[2]),
                    })
    return {
        "components":pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
        "poles":pd.DataFrame(poles_out, columns=POLE_OUTPUT_COLUMNS),
        "lines":pd.DataFrame(lines_out, columns=LINE_OUTPUT_COLUMNS),
        "vertices":pd.DataFrame(verts_out, columns=VERTEX_OUTPUT_COLUMNS),
    }


def atomic_csv(df, path, columns=None, **kwargs):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    d = df.copy()
    if d.empty and columns is not None:
        d = pd.DataFrame(columns=columns)
    tmp = p.with_name(p.name + f".tmp.{os.getpid()}")
    try:
        d.to_csv(tmp, index=False, **kwargs)
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
