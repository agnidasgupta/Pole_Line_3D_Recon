#!/usr/bin/env python3
"""E11: component timing of the E10 persistent Stage-1 -> Stage-2 handoff.

E11 does not alter inference, calibration, thresholds, reconstruction, or any
durable Stage-2 schema.  It runs the same direct processor/write path as E10
and adds per-slice timing accounting outside the exactness check.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd

import run_v4_stage2_stage1_electrical_tracks as stage2
from run_e10_stage12_persistent import PersistentStage2Sink
from run_e9_stage12_inmemory import exact_array_report, parse_e9_args, write_csv_row
from v4_stage_contracts import CONTRACT_VERSION, STAGE2_MANIFEST_COLUMNS, atomic_csv, atomic_json, upsert_manifest_row
from v4_stage2_runtime import LINE_OUTPUT_COLUMNS, POLE_OUTPUT_COLUMNS, VERTEX_OUTPUT_COLUMNS


T = TypeVar("T")


class E11TimedPersistentSink(PersistentStage2Sink):
    """E10's direct writer with transparent component timing."""

    def _measure(self, timings: dict[str, float], name: str, fn: Callable[[], T]) -> T:
        start = time.perf_counter()
        try:
            return fn()
        finally:
            timings[name] = timings.get(name, 0.0) + (time.perf_counter() - start) * 1000.0

    def _run_stage2_one(self, item: dict[str, Any], pred: dict[str, Any], metadata: dict[str, Any]) -> float:
        timings: dict[str, float] = {}
        started = time.perf_counter()
        seq = int(metadata["slice_seq"])
        gid = str(metadata["group_id"])
        relative_path = str(metadata["relative_path"])
        pole_csv, line_csv, vertex_csv = stage2.stage2_paths(self.stage2_root, relative_path)
        diag = stage2.diagnostic_paths(pole_csv)

        result = self._measure(
            timings, "e11_candidate_process_ms",
            lambda: self.processor.process(item, pred, str(metadata["id"]), seq),
        )
        self.last_stage2_ms = timings["e11_candidate_process_ms"]

        def write_primary() -> None:
            atomic_csv(result["poles"], pole_csv, POLE_OUTPUT_COLUMNS)
            atomic_csv(result["lines"], line_csv, LINE_OUTPUT_COLUMNS)
            atomic_csv(result["vertices"], vertex_csv, VERTEX_OUTPUT_COLUMNS)
            atomic_csv(result["components"], diag["components"], columns=list(result["components"].columns))
        self._measure(timings, "e11_candidate_primary_output_write_ms", write_primary)

        payload_start = time.perf_counter()
        coords = np.asarray(item["coords"], dtype=np.int32)
        labels = np.asarray(result["stage1_labels"], dtype=np.int8)
        line_idx = np.flatnonzero(labels == 2)
        line_scores = np.asarray(pred["line"], dtype=np.float32)
        pole_scores = np.asarray(pred["pole"], dtype=np.float32)
        semantic = np.asarray(pred.get("semantic", np.zeros(len(coords))), dtype=np.uint8)
        voxel_frame = pd.DataFrame({
            "x": coords[line_idx, 0], "y": coords[line_idx, 1], "z": coords[line_idx, 2],
            "v4_pole_score": pole_scores[line_idx], "v4_line_score": line_scores[line_idx],
            "v4_semantic_head": semantic[line_idx], "v4_deployed_label": np.full(len(line_idx), 2, dtype=np.int8),
        })
        accepted_idx = np.asarray(result["accepted_line_indices"], dtype=np.int64)
        accepted_frame = pd.DataFrame({
            "x": coords[accepted_idx, 0], "y": coords[accepted_idx, 1], "z": coords[accepted_idx, 2],
            "v4_pole_score": pole_scores[accepted_idx], "v4_line_score": line_scores[accepted_idx],
            "v4_semantic_head": semantic[accepted_idx], "v4_deployed_label": np.full(len(accepted_idx), 2, dtype=np.int8),
        })
        input_keys = set(map(tuple, coords[line_idx].tolist()))
        accepted_keys = set(map(tuple, coords[accepted_idx].tolist()))
        if input_keys != accepted_keys or len(accepted_idx) != len(line_idx):
            raise RuntimeError("E11 Stage1 line voxel identity check failed")
        candidate_raw = pd.DataFrame(result["bridge_candidates"])
        candidate_columns = list(dict.fromkeys(stage2.BRIDGE_COLUMNS + list(candidate_raw.columns)))
        timings["e11_candidate_payload_materialization_ms"] = (time.perf_counter() - payload_start) * 1000.0

        def write_audit_csvs() -> None:
            atomic_csv(voxel_frame, diag["stage1_voxels"], columns=list(voxel_frame.columns))
            atomic_csv(accepted_frame, diag["accepted_voxels"], columns=list(accepted_frame.columns))
            atomic_csv(pd.DataFrame(result["selected_bridges"], columns=stage2.BRIDGE_COLUMNS), diag["bridges"], columns=stage2.BRIDGE_COLUMNS)
            atomic_csv(pd.DataFrame(result["bridge_candidates"], columns=candidate_columns), diag["bridge_candidates"], columns=candidate_columns)
            atomic_csv(pd.DataFrame(result["track_rows"], columns=stage2.TRACK_COLUMNS), diag["tracks"], columns=stage2.TRACK_COLUMNS)
            atomic_csv(pd.DataFrame(result["pole_attachment_rows"], columns=stage2.ATTACHMENT_COLUMNS), diag["pole_attachments"], columns=stage2.ATTACHMENT_COLUMNS)
        self._measure(timings, "e11_candidate_audit_output_write_ms", write_audit_csvs)

        audit = {
            **result["stage1_electrical_track_audit"], "contract_version": CONTRACT_VERSION, "stage": 2,
            "group_id": gid, "id": str(metadata["id"]), "slice_seq": seq,
            "stage1_npz": "in_memory://stage1_payload", "stage1_meta_json": "in_memory://stage1_metadata",
            "pole_csv": str(pole_csv), "line_csv": str(line_csv), "line_vertices_csv": str(vertex_csv),
            "stage1_line_voxels_csv": str(diag["stage1_voxels"]), "accepted_line_voxels_csv": str(diag["accepted_voxels"]),
            "selected_fragment_bridges_csv": str(diag["bridges"]), "fragment_bridge_candidates_csv": str(diag["bridge_candidates"]),
            "stage1_electrical_tracks_csv": str(diag["tracks"]), "pole_attachments_csv": str(diag["pole_attachments"]),
        }
        self._measure(timings, "e11_candidate_audit_json_write_ms", lambda: atomic_json(audit, diag["audit"]))
        self.audits.append(audit)

        manifest_row = {
            "contract_version": CONTRACT_VERSION, "id": str(metadata["id"]), "source": str(metadata["source"]),
            "relative_path": relative_path, "geography": str(metadata["geography"]), "session": str(metadata["session"]),
            "slice_seq": seq, "group_id": gid, "center_x": float(metadata["center_metadata"]["center_x"]),
            "center_y": float(metadata["center_metadata"]["center_y"]), "center_z": float(metadata["center_metadata"]["center_z"]),
            "stage1_npz": "in_memory://stage1_payload", "output_csv": "", "pole_csv": str(pole_csv),
            "line_csv": str(line_csv), "line_vertices_csv": str(vertex_csv), "rows": int(metadata["rows"]),
            "accepted_poles": int(len(result["poles"])), "accepted_line_segments": int(len(result["lines"])), "status": "completed",
        }
        def write_manifests() -> None:
            frame = upsert_manifest_row(self.stage2_root / "inference_manifest.csv", manifest_row, STAGE2_MANIFEST_COLUMNS)
            atomic_csv(frame, self.stage2_root / "stage2_manifest.csv", STAGE2_MANIFEST_COLUMNS)
        self._measure(timings, "e11_candidate_manifest_update_ms", write_manifests)

        timing_row = {
            "group_id": gid, "slice_seq": seq, "stage1_load_ms": 0.0,
            "production_stage2_ms": float(result["timing"]["production_stage2_ms"]),
            "stage1_electrical_track_ms": float(result["timing"]["stage1_electrical_track_ms"]),
            "stage2_total_ms": self.last_stage2_ms,
            "stage1_inferred_line_voxels": int(audit["stage1_inferred_line_voxels"]),
            "accepted_stage1_line_voxels": int(audit["accepted_stage1_line_voxels"]),
            "stage1_to_stage2_voxel_preservation": float(audit["stage1_to_stage2_voxel_preservation"]),
            "raw_stage1_line_components": int(audit["raw_stage1_line_components"]),
            "joined_stage2_tracks": int(audit["joined_stage2_tracks"]), "selected_fragment_bridges": int(audit["selected_fragment_bridges"]),
            "max_selected_bridge_gap_ft": float(audit["max_selected_bridge_gap_ft"]),
            "unjoined_singleton_line_voxels": int(audit["unjoined_singleton_line_voxels"]),
            "parallel_or_cross_lane_bridges_blocked": int(audit.get("parallel_or_cross_lane_bridges_blocked", 0)),
            "near_pole_line_to_line_bridges_blocked": int(audit.get("near_pole_line_to_line_bridges_blocked", 0)),
            "track_drift_bridges_blocked": int(audit.get("track_drift_bridges_blocked", 0)),
            "pole_attachments": int(audit.get("pole_attachments", 0)), "synthetic_line_voxels": 0,
            "runtime_gt_usage": False, "pole_pair_inference": False,
        }
        self._measure(timings, "e11_candidate_timing_row_write_ms", lambda: stage2.upsert_timing(self.stage2_timing_csv, timing_row))
        total = (time.perf_counter() - started) * 1000.0
        measured = sum(timings.values())
        timings["e11_candidate_callback_ms"] = total
        timings["e11_candidate_unattributed_ms"] = total - measured
        self.last_e11_components = timings
        return total

    def consume(self, _npz: Path, _meta: Path, item: dict[str, Any], pred: dict[str, Any], metadata: dict[str, Any]) -> None:
        validation_start = time.perf_counter()
        report = exact_array_report(item, pred, self.baseline_root, str(metadata["relative_path"]))
        report.update({"group_id": metadata["group_id"], "slice_seq": int(metadata["slice_seq"]), "relative_path": metadata["relative_path"]})
        atomic_json(report, self.quality_dir / f"slice_{int(metadata['slice_seq'])}.stage1_exact.json")
        validation_ms = (time.perf_counter() - validation_start) * 1000.0
        if report["status"] != "PASS":
            raise RuntimeError(f"E11 Stage1 payload differs from production: {metadata['relative_path']}")
        callback_ms = self._run_stage2_one(item, pred, metadata)
        stage1_timing = dict(metadata.get("timing", {}))
        row = {
            "group_id": metadata["group_id"], "slice_seq": int(metadata["slice_seq"]), "relative_path": metadata["relative_path"],
            "rows": int(metadata["rows"]), "occupied_rows": int(metadata["occupied_rows"]),
            "csv_read_ms": float(stage1_timing.get("csv_read_ms", 0.0)), "sparse_item_prep_ms": float(stage1_timing.get("sparse_item_prep_ms", 0.0)),
            "stage1_inference_ms": float(stage1_timing.get("stage1_wall_ms", 0.0)), "stage2_callback_and_output_ms": callback_ms,
            "stage2_reconstruction_ms": self.last_stage2_ms, "stage1_exact": 1,
            "refiner_workers_pinned": int(self.refiner_workers_pinned), "validation_ms_excluded": validation_ms,
            **self.last_e11_components,
        }
        row["combined_stage1_stage2_ms"] = row["csv_read_ms"] + row["sparse_item_prep_ms"] + row["stage1_inference_ms"] + callback_ms
        write_csv_row(self.timing_csv, row)
        self.rows.append(self._manifest_row(metadata))


def main() -> None:
    args, forwarded = parse_e9_args()
    required = [Path(args.e9_stage2_bundle), Path(args.e9_stage2_profile), Path(args.e9_baseline_stage1_dir)]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"E11 required paths are missing: {missing}")
    sink = E11TimedPersistentSink(args)
    import run_v4_stage1_opt2 as stage1
    import sys
    saved_save, saved_argv = stage1.save_stage1_artifact, sys.argv[:]
    stage1.save_stage1_artifact = sink.consume
    sys.argv = [sys.argv[0], *forwarded]
    try:
        stage1.main()
        timing_arg = forwarded.index("--timing_csv")
        sink.include_stage1_manifest_timing(Path(forwarded[timing_arg + 1]))
        sink.finalize()
        print("E11_STAGE12_PROFILE_SESSION_OK", flush=True)
    finally:
        stage1.save_stage1_artifact, sys.argv = saved_save, saved_argv


if __name__ == "__main__":
    main()
