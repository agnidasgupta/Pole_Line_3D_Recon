#!/usr/bin/env python3
"""Run Stage-2-only electrical-safe Stage-1 inferred conductor tracks from saved Stage1."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from v4_stage2_runtime import LINE_OUTPUT_COLUMNS, POLE_OUTPUT_COLUMNS, VERTEX_OUTPUT_COLUMNS
from v4_stage_contracts import (
    CONTRACT_VERSION,
    STAGE1_MANIFEST_COLUMNS,
    STAGE2_MANIFEST_COLUMNS,
    atomic_csv,
    atomic_json,
    load_stage1_artifact,
    stage1_paths,
    stage2_paths,
    upsert_manifest_row,
)
from v4_stage2_stage1_electrical_tracks import STAGE1_ELECTRICAL_TRACK_RUNTIME_VERSION, Stage1ElectricalTrackStage2Processor

HOST_OUTPUT_PREFIX = Path("/workspace/voxel_poleline/outputs")
CONTAINER_OUTPUT_PREFIX = Path("/outputs")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage2 joining of deployed Stage1 line-label fragments into tracks")
    p.add_argument("--stage1_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--session_filter", required=True)
    p.add_argument("--stage2_bundle", required=True)
    p.add_argument("--calibration_json", required=True)
    p.add_argument("--profile_json", required=True)
    p.add_argument("--timing_csv", required=True)
    p.add_argument("--grid_size", type=int, nargs=3, default=[400, 400, 200])
    p.add_argument("--voxel_size_ft", type=float, default=0.5)
    p.add_argument("--resume", type=int, choices=[0, 1], default=1)
    p.add_argument("--max_slices", type=int, default=0)
    p.add_argument("--write_voxel_audit", type=int, choices=[0, 1], default=1)
    return p.parse_args()


def map_legacy_host_path(path: Path) -> Path:
    text = str(path)
    prefix = str(HOST_OUTPUT_PREFIX)
    if text == prefix:
        return CONTAINER_OUTPUT_PREFIX
    if text.startswith(prefix + "/"):
        return Path(str(CONTAINER_OUTPUT_PREFIX) + text[len(prefix):])
    return path


def resolve_stage1_artifacts(row: Any, stage1_root: Path) -> tuple[Path, Path]:
    npz = map_legacy_host_path(Path(str(row.stage1_npz)))
    meta = map_legacy_host_path(Path(str(row.stage1_meta_json)))
    if npz.is_file() and meta.is_file():
        return npz, meta
    fallback_npz, fallback_meta = stage1_paths(stage1_root, str(row.relative_path))
    if fallback_npz.is_file() and fallback_meta.is_file():
        return fallback_npz, fallback_meta
    raise FileNotFoundError(
        f"Stage1 artifact unavailable: manifest_npz={row.stage1_npz!r} mapped_npz={npz} fallback_npz={fallback_npz}"
    )


def diagnostic_paths(pole_csv: Path) -> dict[str, Path]:
    suffix = "_poles.csv"
    name = pole_csv.name
    stem = name[:-len(suffix)] if name.endswith(suffix) else pole_csv.stem
    return {
        "components": pole_csv.with_name(f"{stem}_components.csv"),
        "audit": pole_csv.with_name(f"{stem}_stage1_electrical_track_audit.json"),
        "stage1_voxels": pole_csv.with_name(f"{stem}_stage1_line_voxels.csv"),
        "accepted_voxels": pole_csv.with_name(f"{stem}_accepted_line_voxels.csv"),
        "bridges": pole_csv.with_name(f"{stem}_selected_fragment_bridges.csv"),
        "bridge_candidates": pole_csv.with_name(f"{stem}_fragment_bridge_candidates.csv"),
        "tracks": pole_csv.with_name(f"{stem}_stage1_electrical_tracks.csv"),
        "pole_attachments": pole_csv.with_name(f"{stem}_pole_attachments.csv"),
    }


def load_existing_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame(columns=STAGE2_MANIFEST_COLUMNS)
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame(columns=STAGE2_MANIFEST_COLUMNS)


def completed_pairs(frame: pd.DataFrame) -> set[tuple[str, int]]:
    if frame.empty or not {"group_id", "slice_seq", "status"} <= set(frame.columns):
        return set()
    rows = frame[frame["status"].astype(str).eq("completed")]
    return {(str(r.group_id), int(r.slice_seq)) for r in rows.itertuples(index=False)}


def upsert_timing(path: Path, row: dict[str, Any]) -> None:
    if path.is_file() and path.stat().st_size:
        try:
            frame = pd.read_csv(path)
        except EmptyDataError:
            frame = pd.DataFrame()
    else:
        frame = pd.DataFrame()
    if not frame.empty and {"group_id", "slice_seq"} <= set(frame.columns):
        seq = pd.to_numeric(frame["slice_seq"], errors="coerce")
        frame = frame.loc[~(frame["group_id"].astype(str).eq(str(row["group_id"])) & seq.eq(int(row["slice_seq"])))].copy()
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    frame["slice_seq"] = pd.to_numeric(frame["slice_seq"], errors="raise").astype(int)
    frame = frame.sort_values(["group_id", "slice_seq"], kind="stable")
    atomic_csv(frame, path, columns=list(frame.columns))


def main() -> None:
    args = parse_args()
    if args.max_slices < 0:
        raise ValueError("--max_slices must be nonnegative")
    stage1_root = Path(args.stage1_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    bundle = Path(args.stage2_bundle).resolve()
    calibration = Path(args.calibration_json).resolve()
    profile_path = Path(args.profile_json).resolve()
    timing_path = Path(args.timing_csv).resolve()
    for path in (bundle, calibration, profile_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not stage1_root.is_dir():
        raise FileNotFoundError(stage1_root)
    output_root.mkdir(parents=True, exist_ok=True)
    timing_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_path = stage1_root / "stage1_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(manifest_path)
    missing = sorted(set(STAGE1_MANIFEST_COLUMNS) - set(manifest.columns))
    if missing:
        raise RuntimeError(f"Stage1 manifest missing columns {missing}: {manifest_path}")
    manifest = manifest[
        manifest["group_id"].astype(str).eq(args.session_filter)
        & manifest["status"].astype(str).eq("completed")
    ].copy()
    manifest["slice_seq"] = pd.to_numeric(manifest["slice_seq"], errors="raise").astype(int)
    manifest = manifest.sort_values("slice_seq", kind="stable")
    if args.max_slices > 0:
        manifest = manifest.iloc[: args.max_slices].copy()
    if manifest.empty:
        raise RuntimeError(f"No completed Stage1 slices for {args.session_filter}")

    profile = json.loads(profile_path.read_text())
    processor = Stage1ElectricalTrackStage2Processor(
        str(bundle), str(calibration), profile, tuple(args.grid_size), args.voxel_size_ft
    )
    output_manifest = output_root / "inference_manifest.csv"
    done = completed_pairs(load_existing_manifest(output_manifest))
    session_audits: list[dict[str, Any]] = []

    for index, row in enumerate(manifest.itertuples(index=False), 1):
        seq = int(row.slice_seq)
        pole_csv, line_csv, vertex_csv = stage2_paths(output_root, str(row.relative_path))
        diag = diagnostic_paths(pole_csv)
        ready = all(path.is_file() for path in diag.values()) if args.write_voxel_audit else diag["components"].is_file() and diag["audit"].is_file()
        if args.resume and (args.session_filter, seq) in done and pole_csv.is_file() and line_csv.is_file() and vertex_csv.is_file() and ready:
            print(f"[stage2-stage1-electrical] {index}/{len(manifest)} reuse seq={seq}", flush=True)
            session_audits.append(json.loads(diag["audit"].read_text()))
            continue

        stage1_npz, stage1_meta = resolve_stage1_artifacts(row, stage1_root)
        t0 = time.perf_counter()
        item, pred, metadata = load_stage1_artifact(stage1_npz, stage1_meta)
        load_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        result = processor.process(item, pred, str(row.id), seq)
        stage2_ms = (time.perf_counter() - t1) * 1000.0

        atomic_csv(result["poles"], pole_csv, POLE_OUTPUT_COLUMNS)
        atomic_csv(result["lines"], line_csv, LINE_OUTPUT_COLUMNS)
        atomic_csv(result["vertices"], vertex_csv, VERTEX_OUTPUT_COLUMNS)
        atomic_csv(result["components"], diag["components"], columns=list(result["components"].columns))

        coords = np.asarray(item["coords"], dtype=np.int32)
        labels = np.asarray(result["stage1_labels"], dtype=np.int8)
        line_idx = np.flatnonzero(labels == 2)
        line_scores = np.asarray(pred["line"], dtype=np.float32)
        pole_scores = np.asarray(pred["pole"], dtype=np.float32)
        semantic = np.asarray(pred.get("semantic", np.zeros(len(coords))), dtype=np.uint8)

        if args.write_voxel_audit:
            voxel_frame = pd.DataFrame({
                "x": coords[line_idx, 0],
                "y": coords[line_idx, 1],
                "z": coords[line_idx, 2],
                "v4_pole_score": pole_scores[line_idx],
                "v4_line_score": line_scores[line_idx],
                "v4_semantic_head": semantic[line_idx],
                "v4_deployed_label": np.full(len(line_idx), 2, dtype=np.int8),
            })
            atomic_csv(voxel_frame, diag["stage1_voxels"], columns=list(voxel_frame.columns))
            # Track joining preserves the Stage1 class-2 voxel set exactly.
            atomic_csv(voxel_frame, diag["accepted_voxels"], columns=list(voxel_frame.columns))
            bridge_frame = pd.DataFrame(result["selected_bridges"])
            atomic_csv(bridge_frame, diag["bridges"], columns=list(bridge_frame.columns))
            candidate_frame = pd.DataFrame(result["bridge_candidates"])
            atomic_csv(candidate_frame, diag["bridge_candidates"], columns=list(candidate_frame.columns))
            track_frame = pd.DataFrame(result["track_rows"])
            atomic_csv(track_frame, diag["tracks"], columns=list(track_frame.columns))
            attach_frame = pd.DataFrame(result["pole_attachment_rows"])
            atomic_csv(attach_frame, diag["pole_attachments"], columns=list(attach_frame.columns))

        audit = {
            **result["stage1_electrical_track_audit"],
            "contract_version": CONTRACT_VERSION,
            "stage": 2,
            "group_id": args.session_filter,
            "id": str(row.id),
            "slice_seq": seq,
            "stage1_npz": str(stage1_npz),
            "stage1_meta_json": str(stage1_meta),
            "pole_csv": str(pole_csv),
            "line_csv": str(line_csv),
            "line_vertices_csv": str(vertex_csv),
            "stage1_line_voxels_csv": str(diag["stage1_voxels"]) if args.write_voxel_audit else "",
            "accepted_line_voxels_csv": str(diag["accepted_voxels"]) if args.write_voxel_audit else "",
            "selected_fragment_bridges_csv": str(diag["bridges"]) if args.write_voxel_audit else "",
            "fragment_bridge_candidates_csv": str(diag["bridge_candidates"]) if args.write_voxel_audit else "",
            "stage1_electrical_tracks_csv": str(diag["tracks"]) if args.write_voxel_audit else "",
            "pole_attachments_csv": str(diag["pole_attachments"]) if args.write_voxel_audit else "",
        }
        atomic_json(audit, diag["audit"])
        session_audits.append(audit)

        manifest_row = {
            "contract_version": CONTRACT_VERSION,
            "id": str(row.id),
            "source": str(row.source),
            "relative_path": str(row.relative_path),
            "geography": str(row.geography),
            "session": str(row.session),
            "slice_seq": seq,
            "group_id": str(row.group_id),
            "center_x": float(row.center_x),
            "center_y": float(row.center_y),
            "center_z": float(row.center_z),
            "stage1_npz": str(stage1_npz),
            "output_csv": "",
            "pole_csv": str(pole_csv),
            "line_csv": str(line_csv),
            "line_vertices_csv": str(vertex_csv),
            "rows": int(row.rows),
            "accepted_poles": int(len(result["poles"])),
            "accepted_line_segments": int(len(result["lines"])),
            "status": "completed",
        }
        frame = upsert_manifest_row(output_manifest, manifest_row, STAGE2_MANIFEST_COLUMNS)
        atomic_csv(frame, output_root / "stage2_manifest.csv", STAGE2_MANIFEST_COLUMNS)
        done.add((args.session_filter, seq))

        upsert_timing(timing_path, {
            "group_id": args.session_filter,
            "slice_seq": seq,
            "stage1_load_ms": load_ms,
            "production_stage2_ms": float(result["timing"]["production_stage2_ms"]),
            "stage1_electrical_track_ms": float(result["timing"]["stage1_electrical_track_ms"]),
            "stage2_total_ms": stage2_ms,
            "stage1_inferred_line_voxels": int(audit["stage1_inferred_line_voxels"]),
            "accepted_stage1_line_voxels": int(audit["accepted_stage1_line_voxels"]),
            "stage1_to_stage2_voxel_preservation": float(audit["stage1_to_stage2_voxel_preservation"]),
            "raw_stage1_line_components": int(audit["raw_stage1_line_components"]),
            "joined_stage2_tracks": int(audit["joined_stage2_tracks"]),
            "selected_fragment_bridges": int(audit["selected_fragment_bridges"]),
            "max_selected_bridge_gap_ft": float(audit["max_selected_bridge_gap_ft"]),
            "unjoined_singleton_line_voxels": int(audit["unjoined_singleton_line_voxels"]),
            "parallel_or_cross_lane_bridges_blocked": int(audit.get("parallel_or_cross_lane_bridges_blocked", 0)),
            "near_pole_line_to_line_bridges_blocked": int(audit.get("near_pole_line_to_line_bridges_blocked", 0)),
            "track_drift_bridges_blocked": int(audit.get("track_drift_bridges_blocked", 0)),
            "pole_attachments": int(audit.get("pole_attachments", 0)),
            "synthetic_line_voxels": 0,
            "runtime_gt_usage": False,
            "pole_pair_inference": False,
        })

        print(
            f"[stage2-stage1-electrical] {index}/{len(manifest)} seq={seq} "
            f"poles={len(result['poles'])} tracks={len(result['lines'])} "
            f"stage1_line={audit['stage1_inferred_line_voxels']} "
            f"preservation={audit['stage1_to_stage2_voxel_preservation']:.4f} "
            f"raw_components={audit['raw_stage1_line_components']} "
            f"bridges={audit['selected_fragment_bridges']} stage2={stage2_ms:.1f}ms",
            flush=True,
        )

    total_stage1 = sum(int(a.get("stage1_inferred_line_voxels", 0)) for a in session_audits)
    total_accepted = sum(int(a.get("accepted_stage1_line_voxels", 0)) for a in session_audits)
    summary = {
        "contract_version": CONTRACT_VERSION,
        "stage": 2,
        "completed": True,
        "processor": STAGE1_ELECTRICAL_TRACK_RUNTIME_VERSION,
        "group_id": args.session_filter,
        "slices": int(len(manifest)),
        "manifest": str(output_manifest),
        "timing_csv": str(timing_path),
        "runtime_gt_usage": False,
        "synthetic_line_voxels": 0,
        "pole_pair_inference": False,
        "line_hysteresis_used": False,
        "line_refiner_used": False,
        "totals": {
            "stage1_inferred_line_voxels": int(total_stage1),
            "accepted_stage1_line_voxels": int(total_accepted),
            "stage1_to_stage2_voxel_preservation": float(total_accepted / max(total_stage1, 1)) if total_stage1 else 1.0,
            "raw_stage1_line_components": int(sum(int(a.get("raw_stage1_line_components", 0)) for a in session_audits)),
            "joined_stage2_tracks": int(sum(int(a.get("joined_stage2_tracks", 0)) for a in session_audits)),
            "selected_fragment_bridges": int(sum(int(a.get("selected_fragment_bridges", 0)) for a in session_audits)),
            "unjoined_singleton_line_voxels": int(sum(int(a.get("unjoined_singleton_line_voxels", 0)) for a in session_audits)),
            "parallel_or_cross_lane_bridges_blocked": int(sum(int(a.get("parallel_or_cross_lane_bridges_blocked", 0)) for a in session_audits)),
            "near_pole_line_to_line_bridges_blocked": int(sum(int(a.get("near_pole_line_to_line_bridges_blocked", 0)) for a in session_audits)),
            "track_drift_bridges_blocked": int(sum(int(a.get("track_drift_bridges_blocked", 0)) for a in session_audits)),
            "pole_attachments": int(sum(int(a.get("pole_attachments", 0)) for a in session_audits)),
        },
    }
    if summary["totals"]["stage1_inferred_line_voxels"] != summary["totals"]["accepted_stage1_line_voxels"]:
        raise RuntimeError("Stage1 electrical-track voxel preservation invariant failed")
    atomic_json(summary, output_root / "STAGE2_COMPLETED.json")
    atomic_json(summary, output_root / "STAGE2_STAGE1_ELECTRICAL_TRACK_SUMMARY.json")
    print("V4_STAGE2_STAGE1_ELECTRICAL_TRACK_SESSION_OK", flush=True)


if __name__ == "__main__":
    main()
