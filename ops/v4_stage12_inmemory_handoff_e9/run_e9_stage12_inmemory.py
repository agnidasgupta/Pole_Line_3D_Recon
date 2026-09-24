#!/usr/bin/env python3
"""E9: exact Stage-1 to Stage-2 in-memory handoff.

The accepted Stage-1 E7 implementation remains the inference engine.  This
wrapper intercepts only its durable Stage-1 artifact callback, verifies the
payload against the accepted production artifact in memory, and supplies that
same payload to the unmodified V10 Stage-2 runner.  Stage-2's normal CSV/JSON
outputs are still written atomically; candidate Stage-1 NPZ/JSON files are not.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_v4_stage1_opt2 as stage1
import run_v4_stage2_stage1_electrical_tracks as stage2
from v4_stage_contracts import STAGE1_MANIFEST_COLUMNS, atomic_json, load_stage1_artifact, stage1_paths
from v4_stage2_stage1_electrical_tracks import Stage1ElectricalTrackStage2Processor


def parse_e9_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--e9-stage2-output-dir", required=True)
    parser.add_argument("--e9-stage2-bundle", required=True)
    parser.add_argument("--e9-stage2-profile", required=True)
    parser.add_argument("--e9-baseline-stage1-dir", required=True)
    parser.add_argument("--e9-timing-csv", required=True)
    parser.add_argument("--e9-stage2-timing-csv", required=True)
    parser.add_argument("--e9-quality-dir", required=True)
    parser.add_argument("--e9-stage2-calibration-json", required=True)
    parser.add_argument("--e9-grid-size", nargs=3, type=int, default=[400, 400, 200])
    parser.add_argument("--e9-voxel-size-ft", type=float, default=0.5)
    parser.add_argument("--e9-session-filter", required=True)
    return parser.parse_known_args()


def write_csv_row(path: Path, row: dict[str, Any]) -> None:
    """Atomically replace a slice row so an automatic retry cannot double-count it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0:
        prior = pd.read_csv(path)
        keep = ~((prior["group_id"].astype(str) == str(row["group_id"])) &
                 (prior["slice_seq"].astype(int) == int(row["slice_seq"])))
        frame = pd.concat([prior.loc[keep], pd.DataFrame([row])], ignore_index=True)
    else:
        frame = pd.DataFrame([row])
    frame = frame.sort_values(["group_id", "slice_seq"], kind="stable")
    tmp = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def exact_array_report(item: dict[str, Any], pred: dict[str, Any], baseline_root: Path, relative_path: str) -> dict[str, Any]:
    npz, meta = stage1_paths(baseline_root, relative_path)
    base_item, base_pred, _ = load_stage1_artifact(npz, meta)
    fields = {
        "coords": (item["coords"], base_item["coords"]),
        "dist_values": (item.get("dist_values"), base_item.get("dist_values")),
        "source_rows": (item.get("source_rows"), base_item.get("source_rows")),
        "raw_labels": (item.get("raw_labels"), base_item.get("raw_labels")),
        "pole": (pred["pole"], base_pred["pole"]),
        "line": (pred["line"], base_pred["line"]),
        "semantic": (pred.get("semantic"), base_pred.get("semantic")),
        "objectness": (pred.get("objectness"), base_pred.get("objectness")),
    }
    details: dict[str, Any] = {}
    passed = True
    for name, (candidate, baseline) in fields.items():
        a = np.asarray(candidate)
        b = np.asarray(baseline)
        same_shape = a.shape == b.shape and a.dtype == b.dtype
        exact = bool(same_shape and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes())
        max_abs = None
        if a.shape == b.shape and np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
            max_abs = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0
        details[name] = {"exact": exact, "candidate_dtype": str(a.dtype), "baseline_dtype": str(b.dtype), "shape": list(a.shape), "max_abs": max_abs}
        passed = passed and exact
    return {"status": "PASS" if passed else "FAIL", "baseline_npz": str(npz), "fields": details}


def pin_refiner_workers(root: Any) -> int:
    """Force sklearn refiner objects to one worker for deterministic CSV formatting."""
    visited: set[int] = set()
    changed = 0

    def walk(obj: Any, depth: int = 0) -> None:
        nonlocal changed
        if obj is None or depth > 6 or id(obj) in visited:
            return
        visited.add(id(obj))
        if hasattr(obj, "n_jobs"):
            try:
                if getattr(obj, "n_jobs") != 1:
                    setattr(obj, "n_jobs", 1)
                    changed += 1
            except Exception:
                pass
        if isinstance(obj, dict):
            for value in obj.values():
                walk(value, depth + 1)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                walk(value, depth + 1)
        elif hasattr(obj, "__dict__") and obj.__class__.__module__.split(".")[0] not in {"numpy", "pandas", "torch"}:
            for value in vars(obj).values():
                walk(value, depth + 1)

    walk(root)
    return changed


class InMemoryStage2Sink:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.stage2_root = Path(args.e9_stage2_output_dir).resolve()
        self.baseline_root = Path(args.e9_baseline_stage1_dir).resolve()
        self.timing_csv = Path(args.e9_timing_csv).resolve()
        self.stage2_timing_csv = Path(args.e9_stage2_timing_csv).resolve()
        self.quality_dir = Path(args.e9_quality_dir).resolve()
        self.stage2_root.mkdir(parents=True, exist_ok=True)
        self.quality_dir.mkdir(parents=True, exist_ok=True)
        profile = json.loads(Path(args.e9_stage2_profile).read_text())
        self.processor = Stage1ElectricalTrackStage2Processor(
            args.e9_stage2_bundle,
            args.e9_stage2_calibration_json,
            profile,
            tuple(args.e9_grid_size),
            args.e9_voxel_size_ft,
        )
        self.refiner_workers_pinned = pin_refiner_workers(self.processor)
        self.tmp = Path(tempfile.mkdtemp(prefix="e9-stage12-", dir="/dev/shm"))
        self.rows: list[dict[str, Any]] = []
        self.audits: list[dict[str, Any]] = []

    def _manifest_row(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "contract_version": "v4-production-1", "id": str(metadata["id"]),
            "source": str(metadata["source"]), "relative_path": str(metadata["relative_path"]),
            "geography": str(metadata["geography"]), "session": str(metadata["session"]),
            "slice_seq": int(metadata["slice_seq"]), "group_id": str(metadata["group_id"]),
            **dict(metadata.get("center_metadata", {})),
            "stage1_npz": "in_memory://stage1_payload", "stage1_meta_json": "in_memory://stage1_metadata",
            "rows": int(metadata["rows"]), "occupied_rows": int(metadata["occupied_rows"]), "status": "completed",
        }

    def _run_stage2_one(self, item: dict[str, Any], pred: dict[str, Any], metadata: dict[str, Any]) -> float:
        row = self._manifest_row(metadata)
        manifest = pd.DataFrame([row]).reindex(columns=STAGE1_MANIFEST_COLUMNS)
        stage1_dir = self.tmp / f"slice_{int(row['slice_seq'])}"
        stage1_dir.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(stage1_dir / "stage1_manifest.csv", index=False)
        saved_argv = sys.argv[:]
        saved_load = stage2.load_stage1_artifact
        saved_resolve = stage2.resolve_stage1_artifacts
        saved_factory = stage2.Stage1ElectricalTrackStage2Processor
        stage2.load_stage1_artifact = lambda *_args, **_kwargs: (item, pred, metadata)
        stage2.resolve_stage1_artifacts = lambda *_args, **_kwargs: (Path("in_memory://stage1_payload"), Path("in_memory://stage1_metadata"))
        stage2.Stage1ElectricalTrackStage2Processor = lambda *_args, **_kwargs: self.processor
        sys.argv = [
            "run_v4_stage2_stage1_electrical_tracks.py", "--stage1_dir", str(stage1_dir),
            "--output_dir", str(self.stage2_root), "--session_filter", self.args.e9_session_filter,
            "--stage2_bundle", self.args.e9_stage2_bundle,
            "--calibration_json", self.args.e9_stage2_calibration_json,
            "--profile_json", self.args.e9_stage2_profile,
            "--timing_csv", str(self.stage2_timing_csv),
            "--write_voxel_audit", "1", "--resume", "0", "--max_slices", "0",
        ]
        start = time.perf_counter()
        try:
            stage2.main()
        finally:
            sys.argv = saved_argv
            stage2.load_stage1_artifact = saved_load
            stage2.resolve_stage1_artifacts = saved_resolve
            stage2.Stage1ElectricalTrackStage2Processor = saved_factory
        return (time.perf_counter() - start) * 1000.0

    def consume(self, _npz: Path, _meta: Path, item: dict[str, Any], pred: dict[str, Any], metadata: dict[str, Any]) -> None:
        report = exact_array_report(item, pred, self.baseline_root, str(metadata["relative_path"]))
        report.update({"group_id": metadata["group_id"], "slice_seq": int(metadata["slice_seq"]), "relative_path": metadata["relative_path"]})
        atomic_json(report, self.quality_dir / f"slice_{int(metadata['slice_seq'])}.stage1_exact.json")
        if report["status"] != "PASS":
            raise RuntimeError(f"E9 Stage1 in-memory payload differs from production: {metadata['relative_path']}")
        stage2_callback_ms = self._run_stage2_one(item, pred, metadata)
        stage2_rows = pd.read_csv(self.stage2_timing_csv)
        measured = stage2_rows.loc[
            stage2_rows["group_id"].astype(str).eq(str(metadata["group_id"]))
            & pd.to_numeric(stage2_rows["slice_seq"], errors="raise").eq(int(metadata["slice_seq"]))
        ]
        if len(measured) != 1:
            raise RuntimeError("E9 Stage2 timing row missing or duplicate")
        stage2_compute_ms = float(measured.iloc[0]["stage2_total_ms"])
        pole_csv, _, _ = stage2.stage2_paths(self.stage2_root, str(metadata["relative_path"]))
        audit_file = stage2.diagnostic_paths(pole_csv)["audit"]
        self.audits.append(json.loads(audit_file.read_text()))
        timing = dict(metadata.get("timing", {}))
        row = {
            "group_id": metadata["group_id"], "slice_seq": int(metadata["slice_seq"]),
            "relative_path": metadata["relative_path"], "rows": int(metadata["rows"]),
            "occupied_rows": int(metadata["occupied_rows"]),
            "csv_read_ms": float(timing.get("csv_read_ms", 0.0)),
            "sparse_item_prep_ms": float(timing.get("sparse_item_prep_ms", 0.0)),
            "stage1_inference_ms": float(timing.get("stage1_wall_ms", 0.0)),
            "stage2_reconstruction_ms": stage2_compute_ms,
            "stage2_write_and_orchestration_ms": float(stage2_callback_ms - stage2_compute_ms),
            "stage2_callback_and_output_ms": float(stage2_callback_ms),
            "combined_stage1_stage2_ms": float(timing.get("csv_read_ms", 0.0) + timing.get("sparse_item_prep_ms", 0.0) + timing.get("stage1_wall_ms", 0.0) + stage2_callback_ms),
            "stage1_exact": 1, "refiner_workers_pinned": int(self.refiner_workers_pinned),
        }
        write_csv_row(self.timing_csv, row)
        self.rows.append(self._manifest_row(metadata))

    def finalize(self) -> None:
        if not self.rows or len(self.audits) != len(self.rows):
            raise RuntimeError("E9 Stage2 slice/audit count mismatch")
        completed_path = self.stage2_root / "STAGE2_COMPLETED.json"
        completed = json.loads(completed_path.read_text())
        totals = {}
        for key in completed["totals"]:
            if key == "stage1_to_stage2_voxel_preservation":
                continue
            totals[key] = sum(int(a.get(key, 0)) for a in self.audits)
        total_line = totals["stage1_inferred_line_voxels"]
        accepted_line = totals["accepted_stage1_line_voxels"]
        totals["stage1_to_stage2_voxel_preservation"] = accepted_line / max(total_line, 1) if total_line else 1.0
        if total_line != accepted_line:
            raise RuntimeError("E9 Stage2 voxel preservation invariant failed")
        completed["slices"] = len(self.rows)
        completed["totals"] = totals
        atomic_json(completed, completed_path)
        atomic_json(completed, self.stage2_root / "STAGE2_STAGE1_ELECTRICAL_TRACK_SUMMARY.json")
        summary = {
            "completed": True, "stage": "stage1_to_stage2_in_memory", "group_id": self.args.e9_session_filter,
            "slices": len(self.rows), "stage1_artifacts_written": False,
            "stage2_outputs_written": True, "stage1_handoff": "in_memory", "refiner_workers_pinned": self.refiner_workers_pinned,
        }
        atomic_json(summary, self.stage2_root / "E9_STAGE12_SESSION_COMPLETED.json")
        shutil.rmtree(self.tmp, ignore_errors=True)


def main() -> None:
    e9, forwarded = parse_e9_args()
    required = [Path(e9.e9_stage2_bundle), Path(e9.e9_stage2_profile), Path(e9.e9_baseline_stage1_dir)]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"E9 required paths are missing: {missing}")
    sink = InMemoryStage2Sink(e9)
    saved_save = stage1.save_stage1_artifact
    saved_argv = sys.argv[:]
    stage1.save_stage1_artifact = sink.consume
    sys.argv = [sys.argv[0], *forwarded]
    try:
        stage1.main()
        sink.finalize()
        print("E9_STAGE12_INMEMORY_SESSION_OK", flush=True)
    finally:
        sys.argv = saved_argv
        stage1.save_stage1_artifact = saved_save


if __name__ == "__main__":
    main()
