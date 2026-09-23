#!/usr/bin/env python3
"""Run V4 Stage 2 from saved Stage-1 artifacts using production-preserving Stage-1 bundle consensus.

This experiment preserves the production Stage-2 result exactly and appends only
novel, geometry/refiner-approved, Stage-1-line-labelled lanes that have parallel
sibling support from a production-accepted conductor. Ground truth is never consulted.
"""
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
from v4_stage2_bundle_consensus import Stage1BundleConsensusStage2Processor, profile_from_dict


HOST_OUTPUT_PREFIX = Path("/workspace/voxel_poleline/outputs")
CONTAINER_OUTPUT_PREFIX = Path("/outputs")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage2-only V4 bundle-consensus experiment preserving production output"
    )
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
    """Map persisted host paths in manifests to the mounted Docker namespace."""
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
        "Stage1 artifact unavailable: "
        f"manifest_npz={row.stage1_npz!r} mapped_npz={npz} "
        f"fallback_npz={fallback_npz}"
    )


def diagnostic_paths(pole_csv: Path) -> tuple[Path, Path, Path, Path]:
    name = pole_csv.name
    suffix = "_poles.csv"
    stem = name[:-len(suffix)] if name.endswith(suffix) else pole_csv.stem
    return (
        pole_csv.with_name(f"{stem}_components.csv"),
        pole_csv.with_name(f"{stem}_stage1_label_audit.json"),
        pole_csv.with_name(f"{stem}_stage1_line_voxels.csv"),
        pole_csv.with_name(f"{stem}_accepted_line_voxels.csv"),
    )


def load_existing_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame(columns=STAGE2_MANIFEST_COLUMNS)
    try:
        frame = pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame(columns=STAGE2_MANIFEST_COLUMNS)
    return frame


def completed_pairs(frame: pd.DataFrame) -> set[tuple[str, int]]:
    required = {"group_id", "slice_seq", "status"}
    if frame.empty or not required <= set(frame.columns):
        return set()
    rows = frame[frame["status"].astype(str).eq("completed")]
    return {
        (str(row.group_id), int(row.slice_seq))
        for row in rows.itertuples(index=False)
    }


def upsert_timing(path: Path, row: dict[str, Any]) -> pd.DataFrame:
    if path.is_file() and path.stat().st_size:
        try:
            frame = pd.read_csv(path)
        except EmptyDataError:
            frame = pd.DataFrame()
    else:
        frame = pd.DataFrame()
    if not frame.empty and {"group_id", "slice_seq"} <= set(frame.columns):
        seq = pd.to_numeric(frame["slice_seq"], errors="coerce")
        keep = ~(
            frame["group_id"].astype(str).eq(str(row["group_id"]))
            & seq.eq(int(row["slice_seq"]))
        )
        frame = frame.loc[keep].copy()
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    frame["slice_seq"] = pd.to_numeric(frame["slice_seq"], errors="raise").astype(int)
    frame = frame.sort_values(["group_id", "slice_seq"], kind="stable")
    atomic_csv(frame, path, columns=list(frame.columns))
    return frame


def main() -> None:
    args = parse_args()
    if args.max_slices < 0:
        raise ValueError("--max_slices must be nonnegative")

    stage1_root = Path(args.stage1_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    bundle_path = Path(args.stage2_bundle).resolve()
    calibration_path = Path(args.calibration_json).resolve()
    profile_path = Path(args.profile_json).resolve()
    timing_path = Path(args.timing_csv).resolve()

    if not stage1_root.is_dir():
        raise FileNotFoundError(stage1_root)
    if not bundle_path.is_file():
        raise FileNotFoundError(bundle_path)
    if not calibration_path.is_file():
        raise FileNotFoundError(calibration_path)
    if not profile_path.is_file():
        raise FileNotFoundError(profile_path)

    output_root.mkdir(parents=True, exist_ok=True)
    timing_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_path = stage1_root / "stage1_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    try:
        manifest = pd.read_csv(manifest_path)
    except EmptyDataError as exc:
        raise RuntimeError(f"Stage1 manifest has no rows: {manifest_path}") from exc
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

    profile_data = json.loads(profile_path.read_text())
    profile = profile_from_dict(profile_data)
    processor = Stage1BundleConsensusStage2Processor(
        str(bundle_path),
        str(calibration_path),
        profile,
        tuple(args.grid_size),
        args.voxel_size_ft,
    )

    output_manifest = output_root / "inference_manifest.csv"
    existing = load_existing_manifest(output_manifest)
    done = completed_pairs(existing)

    session_audits: list[dict[str, Any]] = []
    for index, row in enumerate(manifest.itertuples(index=False), 1):
        seq = int(row.slice_seq)
        pole_csv, line_csv, vertex_csv = stage2_paths(output_root, str(row.relative_path))
        components_csv, audit_json, stage1_voxels_csv, accepted_voxels_csv = diagnostic_paths(pole_csv)

        voxel_audit_ready = (
            not args.write_voxel_audit
            or (stage1_voxels_csv.is_file() and accepted_voxels_csv.is_file())
        )
        if (
            args.resume
            and (args.session_filter, seq) in done
            and pole_csv.is_file()
            and line_csv.is_file()
            and vertex_csv.is_file()
            and components_csv.is_file()
            and audit_json.is_file()
            and voxel_audit_ready
        ):
            print(f"[stage2-bundle-consensus] {index}/{len(manifest)} reuse seq={seq}", flush=True)
            session_audits.append(json.loads(audit_json.read_text()))
            continue

        stage1_npz, stage1_meta = resolve_stage1_artifacts(row, stage1_root)
        start = time.perf_counter()
        item, pred, metadata = load_stage1_artifact(stage1_npz, stage1_meta)
        load_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        result = processor.process(item, pred, str(row.id), seq)
        stage2_ms = (time.perf_counter() - start) * 1000.0

        atomic_csv(result["poles"], pole_csv, POLE_OUTPUT_COLUMNS)
        atomic_csv(result["lines"], line_csv, LINE_OUTPUT_COLUMNS)
        atomic_csv(result["vertices"], vertex_csv, VERTEX_OUTPUT_COLUMNS)
        atomic_csv(result["components"], components_csv, columns=list(result["components"].columns))

        if args.write_voxel_audit:
            coords = np.asarray(item["coords"], dtype=np.int32)
            stage1_line_mask = np.asarray(result["stage1_labels"], dtype=np.int8) == 2
            stage1_voxels = pd.DataFrame({
                "x": coords[stage1_line_mask, 0],
                "y": coords[stage1_line_mask, 1],
                "z": coords[stage1_line_mask, 2],
                "v4_pole_score": np.asarray(pred["pole"], dtype=np.float32)[stage1_line_mask],
                "v4_line_score": np.asarray(pred["line"], dtype=np.float32)[stage1_line_mask],
                "v4_semantic_head": np.asarray(pred.get("semantic", np.zeros(len(coords))), dtype=np.uint8)[stage1_line_mask],
                "v4_deployed_label": np.full(int(stage1_line_mask.sum()), 2, dtype=np.int8),
            })
            atomic_csv(
                stage1_voxels,
                stage1_voxels_csv,
                columns=["x", "y", "z", "v4_pole_score", "v4_line_score", "v4_semantic_head", "v4_deployed_label"],
            )

            accepted_rows: list[dict[str, Any]] = []
            component_frame = result["components"]
            if not component_frame.empty and {"class_name", "component_accept", "component_id"} <= set(component_frame.columns):
                accepted_ids_for_voxels = component_frame[
                    component_frame["class_name"].astype(str).eq("line")
                    & component_frame["component_accept"].astype(bool)
                ]["component_id"].astype(str)
                for component_id in accepted_ids_for_voxels:
                    for x, y, z in np.asarray(result["raw_components"]["line_points"].get(component_id, np.empty((0, 3))), dtype=np.int32):
                        accepted_rows.append({
                            "component_id": component_id,
                            "x": int(x),
                            "y": int(y),
                            "z": int(z),
                        })
            atomic_csv(
                pd.DataFrame(accepted_rows),
                accepted_voxels_csv,
                columns=["component_id", "x", "y", "z"],
            )

        audit = {
            **result["stage1_label_audit"],
            "contract_version": CONTRACT_VERSION,
            "stage": 2,
            "group_id": args.session_filter,
            "id": str(row.id),
            "slice_seq": seq,
            "profile": profile.to_dict(),
            "stage1_npz": str(stage1_npz),
            "stage1_meta_json": str(stage1_meta),
            "components_csv": str(components_csv),
            "pole_csv": str(pole_csv),
            "line_csv": str(line_csv),
            "line_vertices_csv": str(vertex_csv),
            "stage1_line_voxels_csv": str(stage1_voxels_csv) if args.write_voxel_audit else "",
            "accepted_line_voxels_csv": str(accepted_voxels_csv) if args.write_voxel_audit else "",
        }
        atomic_json(audit, audit_json)
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

        timing_row = {
            "group_id": args.session_filter,
            "slice_seq": seq,
            "stage1_load_ms": load_ms,
            "stage2_component_ms": float(result["timing"]["stage2_component_ms"]),
            "stage2_refiner_parametric_ms": float(result["timing"]["stage2_refiner_parametric_ms"]),
            "stage2_total_ms": stage2_ms,
            "stage1_inferred_line_voxels": int(audit["stage1_inferred_line_voxels"]),
            "accepted_stage1_line_voxels": int(audit["accepted_stage1_line_voxels"]),
            "stage1_to_stage2_voxel_preservation": float(audit["stage1_to_stage2_voxel_preservation"]),
            "raw_stage1_line_components": int(audit["raw_stage1_line_components"]),
            "raw_components_with_multiple_lanes": int(audit["raw_components_with_multiple_lanes"]),
            "longitudinal_extra_runs": int(audit["longitudinal_extra_runs"]),
            "accepted_line_components": int(audit.get("accepted_line_components", len(result["lines"]))),
            "production_accepted_line_components": int(audit.get("production_accepted_line_components", 0)),
            "production_accepted_line_voxels": int(audit.get("production_accepted_line_voxels", 0)),
            "residual_candidate_components": int(audit.get("residual_candidate_components", 0)),
            "residual_accepted_components": int(audit.get("residual_accepted_components", 0)),
            "residual_accepted_novel_voxels": int(audit.get("residual_accepted_novel_voxels", 0)),
            "production_voxel_preserved": bool(audit.get("production_voxel_preserved", False)),
            "synthetic_line_voxels": 0,
            "runtime_gt_usage": False,
        }
        upsert_timing(timing_path, timing_row)

        print(
            f"[stage2-bundle-consensus] {index}/{len(manifest)} seq={seq} "
            f"poles={len(result['poles'])} lines={len(result['lines'])} "
            f"stage1_line={audit['stage1_inferred_line_voxels']} "
            f"retained={audit['accepted_stage1_line_voxels']} "
            f"preservation={audit['stage1_to_stage2_voxel_preservation']:.4f} "
            f"load={load_ms:.1f}ms stage2={stage2_ms:.1f}ms",
            flush=True,
        )

    summary = {
        "contract_version": CONTRACT_VERSION,
        "stage": 2,
        "completed": True,
        "processor": "production_baseline_plus_stage1_bundle_consensus_v7",
        "group_id": args.session_filter,
        "slices": int(len(manifest)),
        "profile": profile.to_dict(),
        "manifest": str(output_manifest),
        "timing_csv": str(timing_path),
        "runtime_gt_usage": False,
        "synthetic_line_voxels": 0,
        "write_voxel_audit": bool(args.write_voxel_audit),
        "totals": {
            "stage1_inferred_line_voxels": int(sum(int(x.get("stage1_inferred_line_voxels", 0)) for x in session_audits)),
            "accepted_stage1_line_voxels": int(sum(int(x.get("accepted_stage1_line_voxels", 0)) for x in session_audits)),
            "accepted_line_components": int(sum(int(x.get("accepted_line_components", 0)) for x in session_audits)),
        },
    }
    total_stage1 = summary["totals"]["stage1_inferred_line_voxels"]
    total_accepted = summary["totals"]["accepted_stage1_line_voxels"]
    summary["totals"]["stage1_to_stage2_voxel_preservation"] = float(
        total_accepted / max(total_stage1, 1)
    )
    atomic_json(summary, output_root / "STAGE2_COMPLETED.json")
    atomic_json(summary, output_root / "STAGE2_BUNDLE_CONSENSUS_SUMMARY.json")
    print("V4_STAGE2_BUNDLE_CONSENSUS_SESSION_OK", flush=True)


if __name__ == "__main__":
    main()
