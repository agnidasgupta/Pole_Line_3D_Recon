#!/usr/bin/env python3
"""Persistent V4 production pipeline with explicit Stage-1 and Stage-2 boundaries."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd

from v4_realtime_core import (
    setup_torch, load_v4_model, load_calibration, build_sparse_item_from_dataframe,
    predict_v4_sparse_rows, extract_center_metadata, label_from_scores,
)
from v4_sparse_components import extract_sparse_components
from v4_stage2_runtime import load_bundle, apply_stage2


class V4Stage2Processor:
    """Stage 2 only. No Stage-1 model is loaded and no world coordinates are consumed."""
    def __init__(
        self, stage2_bundle, grid_size=(400, 400, 200), voxel_size_ft=.5,
        pole_candidate_threshold=.15, line_candidate_threshold=.08,
        line_weak_threshold=.04, line_competition_ratio=.55,
        pole_min_voxels=4, line_min_voxels=3, edge_width_vox=10,
    ):
        self.grid = tuple(map(int, grid_size))
        self.voxel = float(voxel_size_ft)
        self.bundle = load_bundle(stage2_bundle)
        self.candidate = dict(
            pole_threshold=float(pole_candidate_threshold),
            line_threshold=float(line_candidate_threshold),
            line_weak_threshold=float(line_weak_threshold),
            line_competition_ratio=float(line_competition_ratio),
            pole_min_voxels=int(pole_min_voxels),
            line_min_voxels=int(line_min_voxels),
            edge_width_vox=int(edge_width_vox),
        )

    def process(self, item, pred, file_id="slice", slice_seq=0):
        t0 = time.perf_counter()
        comps = extract_sparse_components(
            item, pred, self.grid, self.voxel, gt_points=None, **self.candidate
        )
        component_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        s2 = apply_stage2(comps, self.bundle, file_id, slice_seq, self.voxel)
        refine_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "raw_components": comps,
            **s2,
            "timing": {
                "stage2_component_ms": component_ms,
                "stage2_refiner_parametric_ms": refine_ms,
                "stage2_total_ms": component_ms + refine_ms,
            },
        }


class V4RealtimePipeline:
    def __init__(
        self, model_path, calibration_json, stage2_bundle,
        grid_size=(400, 400, 200), voxel_size_ft=.5, core_size=48, batch_size=12,
        amp="bf16", compile_model=False, evaluate_all_cores=True,
        gpu_coord_channels=False, fixed_batch_shape=False,
        pole_candidate_threshold=.15, line_candidate_threshold=.08,
        line_weak_threshold=.04, line_competition_ratio=.55,
        pole_min_voxels=4, line_min_voxels=3, edge_width_vox=10,
    ):
        setup_torch()
        self.grid = tuple(map(int, grid_size))
        self.voxel = float(voxel_size_ft)
        self.core = int(core_size)
        self.batch = int(batch_size)
        self.amp = amp
        self.evaluate_all_cores = bool(evaluate_all_cores)
        self.gpu_coord_channels = bool(gpu_coord_channels)
        self.fixed_batch_shape = bool(fixed_batch_shape)
        self.model, self.cfg, self.compiled = load_v4_model(
            model_path, "cuda", bool(compile_model), "default"
        )
        self.cal = load_calibration(calibration_json)
        self.stage2 = V4Stage2Processor(
            stage2_bundle, self.grid, self.voxel,
            pole_candidate_threshold, line_candidate_threshold,
            line_weak_threshold, line_competition_ratio,
            pole_min_voxels, line_min_voxels, edge_width_vox,
        )

    def run_stage1(self, df, file_id="slice", slice_seq=0):
        t_all = time.perf_counter()
        t0 = time.perf_counter()
        item = build_sparse_item_from_dataframe(df, self.grid)
        prep_ms = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        pred = predict_v4_sparse_rows(
            item, self.model, self.cfg, self.cal, self.grid, self.core, self.batch,
            self.amp, evaluate_all_cores=self.evaluate_all_cores,
            gpu_coord_channels=self.gpu_coord_channels,
            fixed_batch_shape=self.fixed_batch_shape,
        )
        stage1_wall_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "item": item,
            "scores": pred,
            "center_metadata": extract_center_metadata(df),
            "timing": {
                "sparse_item_prep_ms": prep_ms,
                "stage1_wall_ms": stage1_wall_ms,
                "stage1_total_ms": (time.perf_counter() - t_all) * 1000.0,
                **pred["timing"],
            },
        }

    def run_stage2(self, item, pred, file_id="slice", slice_seq=0):
        return self.stage2.process(item, pred, file_id, slice_seq)

    def process_dataframe(self, df, file_id="slice", slice_seq=0, with_row_frame=False):
        t_all = time.perf_counter()
        s1 = self.run_stage1(df, file_id, slice_seq)
        s2 = self.run_stage2(s1["item"], s1["scores"], file_id, slice_seq)
        timing = {
            **s1["timing"],
            **s2["timing"],
            "stage12_total_ms": (time.perf_counter() - t_all) * 1000.0,
        }
        result = {
            "item": s1["item"], "scores": s1["scores"],
            "center_metadata": s1["center_metadata"],
            "raw_components": s2["raw_components"],
            "components": s2["components"], "poles": s2["poles"],
            "lines": s2["lines"], "vertices": s2["vertices"],
            "timing": timing,
        }
        if with_row_frame:
            result["row_frame"] = self._row_frame(df, s1["item"], s1["scores"], s2["raw_components"], s2)
        return result

    def _row_frame(self, df, item, pred, comps, s2):
        out = df.copy()
        n = len(out)
        ps = np.zeros(n, np.float32)
        ls = np.zeros(n, np.float32)
        s1 = np.full(n, -1, np.int8)
        s2lab = np.full(n, -1, np.int8)
        coords = item["coords"]
        gx, gy, gz = self.grid
        if len(coords):
            key = ((coords[:, 2].astype(np.int64) * gy + coords[:, 1]) * gx + coords[:, 0])
            order = np.argsort(key)
            sk = key[order]
            rc = item["row_coords"]
            valid = item["valid_rows"]
            ridx = np.flatnonzero(valid)
            c = rc[valid]
            q = ((c[:, 2].astype(np.int64) * gy + c[:, 1]) * gx + c[:, 0])
            pos = np.searchsorted(sk, q)
            ok = pos < len(sk)
            hit = np.zeros(len(pos), bool)
            hit[ok] = sk[pos[ok]] == q[ok]
            src = ridx[hit]
            uu = order[pos[hit]]
            ps[src] = pred["pole"][uu]
            ls[src] = pred["line"][uu]
            s1[src] = label_from_scores(
                pred["pole"][uu], pred["line"][uu],
                self.cal["pole_threshold"], self.cal["line_threshold"],
            )
            coord_to_label = {}
            cf = s2.get("components", pd.DataFrame())
            if not cf.empty and "component_accept" in cf.columns:
                accepted = set(cf.loc[cf.component_accept.astype(bool), "component_id"].astype(str))
                for name, points, lab in [
                    ("line", comps["line_points"], 2), ("pole", comps["pole_points"], 1)
                ]:
                    for cid, pts in points.items():
                        if str(cid) not in accepted:
                            continue
                        for x, y, z in np.asarray(pts, int):
                            coord_to_label[int((z * gy + y) * gx + x)] = lab
            if coord_to_label:
                s2lab[src] = np.array([coord_to_label.get(int(k), 0) for k in q[hit]], dtype=np.int8)
            else:
                s2lab[src] = 0
        out["v4_pole_score"] = ps.astype(np.float16)
        out["v4_line_score"] = ls.astype(np.float16)
        out["v4_stage1_label"] = s1
        out["v4_stage2_label"] = s2lab
        return out
