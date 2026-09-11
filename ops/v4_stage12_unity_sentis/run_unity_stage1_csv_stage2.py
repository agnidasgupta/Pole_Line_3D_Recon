#!/usr/bin/env python3
"""Run the accepted V10 Stage-2 code from a Unity/Sentis Stage-1 CSV handoff.

Run only inside the repository Docker image. It deliberately imports the exact
optimized Python processor instead of maintaining a second geometry implementation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v4_stage2_runtime import LINE_OUTPUT_COLUMNS, POLE_OUTPUT_COLUMNS, VERTEX_OUTPUT_COLUMNS
from v4_stage_contracts import atomic_csv, atomic_json, stage2_paths
from v4_stage2_stage1_electrical_tracks_opt import Stage1ElectricalTrackStage2Processor


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--relative-path", required=True)
    p.add_argument("--file-id", required=True)
    p.add_argument("--slice-seq", required=True, type=int)
    p.add_argument("--group-id", required=True)
    p.add_argument("--stage2-bundle", required=True)
    p.add_argument("--calibration-json", required=True)
    p.add_argument("--profile-json", required=True)
    p.add_argument("--grid-size", type=int, nargs=3, default=(400, 400, 200))
    p.add_argument("--voxel-size-ft", type=float, default=.5)
    return p.parse_args()


def main() -> None:
    args = arguments()
    frame = pd.read_csv(args.stage1_csv)
    required = {"x", "y", "z", "pole", "line", "deployed_label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Unity Stage1 CSV missing columns: {missing}")
    coords = frame[["x", "y", "z"]].apply(pd.to_numeric, errors="raise").to_numpy(np.int32)
    pole = pd.to_numeric(frame["pole"], errors="raise").to_numpy(np.float32)
    line = pd.to_numeric(frame["line"], errors="raise").to_numpy(np.float32)
    deployed = pd.to_numeric(frame["deployed_label"], errors="raise").to_numpy(np.int8)
    if not (len(coords) == len(pole) == len(line) == len(deployed)):
        raise RuntimeError("Unity Stage1 arrays do not align")
    if not set(map(int, np.unique(deployed))) <= {0, 1, 2}:
        raise RuntimeError("deployed_label must contain only 0, 1, 2")

    item = {
        "sparse_native": True,
        "coords": coords,
        "dist_values": pd.to_numeric(frame.get("dist_values", 0), errors="coerce").fillna(0).to_numpy(np.float32)
        if "dist_values" in frame else np.zeros(len(frame), np.float32),
        "source_rows": pd.to_numeric(frame.get("source_row", np.arange(len(frame))), errors="coerce").fillna(-1).to_numpy(np.int64)
        if "source_row" in frame else np.arange(len(frame), dtype=np.int64),
        "raw_labels": np.zeros(len(frame), np.int16),
        "raw_hardneg": np.zeros(len(frame), np.uint8),
        "z_sorted": coords[:, 2] if len(coords) else np.zeros(0, np.int32),
        "has_gt": False,
    }
    pred = {
        "pole": pole, "line": line, "deployed_labels": deployed,
        "semantic": np.zeros(len(frame), np.uint8),
        "objectness": np.zeros(len(frame), np.float32), "timing": {},
    }
    profile = json.loads(Path(args.profile_json).read_text())
    processor = Stage1ElectricalTrackStage2Processor(
        args.stage2_bundle, args.calibration_json, profile,
        tuple(args.grid_size), args.voxel_size_ft,
    )
    result = processor.process(item, pred, args.file_id, args.slice_seq)
    root = Path(args.output_dir).resolve()
    pole_path, line_path, vertex_path = stage2_paths(root, args.relative_path)
    atomic_csv(result["poles"], pole_path, POLE_OUTPUT_COLUMNS)
    atomic_csv(result["lines"], line_path, LINE_OUTPUT_COLUMNS)
    atomic_csv(result["vertices"], vertex_path, VERTEX_OUTPUT_COLUMNS)
    stem = pole_path.name[:-len("_poles.csv")]
    components_path = pole_path.with_name(stem + "_components.csv")
    audit_path = pole_path.with_name(stem + "_stage1_electrical_track_audit.json")
    atomic_csv(result["components"], components_path, list(result["components"].columns))
    audit = {
        **result["stage1_electrical_track_audit"],
        "group_id": args.group_id,
        "slice_seq": args.slice_seq,
        "file_id": args.file_id,
        "source": "unity_sentis_stage1_csv",
        "pole_csv": str(pole_path),
        "line_csv": str(line_path),
        "line_vertices_csv": str(vertex_path),
    }
    atomic_json(audit, audit_path)
    print("V10_UNITY_STAGE1_CSV_STAGE2_OK")
    print(json.dumps({
        "poles": len(result["poles"]), "tracks": len(result["lines"]),
        "stage1_line_voxels": audit["stage1_inferred_line_voxels"],
        "preservation": audit["stage1_to_stage2_voxel_preservation"],
        "output_root": str(root),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
