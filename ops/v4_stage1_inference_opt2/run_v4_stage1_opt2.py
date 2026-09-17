#!/usr/bin/env python3
"""Isolated V4 Stage-1 execution experiment with unchanged production inference."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from run_v4_realtime_session import discover
from v4_realtime_core import (
    build_sparse_item_from_dataframe,
    extract_center_metadata,
    load_calibration,
    load_v4_model,
    setup_torch,
)
from v4_stage_contracts import (
    CONTRACT_VERSION,
    STAGE1_MANIFEST_COLUMNS,
    atomic_json,
    load_stage1_artifact,
    safe_id,
    save_stage1_artifact,
    stage1_paths,
    upsert_manifest_row,
)
from v4_realtime_core_opt2 import V4SparseGpuWorkspace, predict_v4_sparse_rows_opt


TIMING_COLUMNS = [
    "group_id", "slice_seq", "relative_path", "rows", "occupied_rows",
    "csv_read_ms", "sparse_item_prep_ms", "core_schedule_ms",
    "workspace_prepare_ms", "gpu_workspace_reset_ms", "host_pin_ms",
    "sparse_h2d_cuda_ms", "gpu_sparse_scatter_ms", "gpu_patch_extract_ms",
    "gpu_feature_assembly_ms", "gpu_model_ms", "gpu_gather_plan_ms",
    "gpu_gather_ms", "d2h_gather_ms", "stage1_wall_ms",
    "stage1_artifact_write_ms", "stage1_manifest_write_ms", "slice_total_ms",
    "active_cores", "total_possible_cores", "batches", "workspace_reused",
    "host_output_reused", "pinned_d2h", "gpu_volume_mode", "fixed_batch_shape",
    "batch_size", "channels_last", "compile_model", "compile_mode",
    "prune_embedding_head", "optimization_version",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--session_filter", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--calibration_json", required=True)
    p.add_argument("--timing_csv", required=True)
    p.add_argument("--progress_json", required=True)
    p.add_argument("--grid_size", type=int, nargs=3, default=[400, 400, 200])
    p.add_argument("--core_size", type=int, default=48)
    p.add_argument("--batch_size", type=int, choices=[12], default=12)
    p.add_argument("--amp", choices=["bf16"], default="bf16")
    p.add_argument("--compile_model", type=int, choices=[0], default=0)
    p.add_argument(
        "--compile_mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
    )
    p.add_argument("--channels_last", type=int, choices=[1], default=1)
    p.add_argument("--pinned_d2h", type=int, choices=[0, 1], default=1)
    p.add_argument("--prune_embedding_head", type=int, choices=[0], default=0)
    p.add_argument("--warmup_iterations", type=int, choices=[0], default=0)
    p.add_argument("--evaluate_all_cores", type=int, choices=[0], default=0)
    p.add_argument("--gpu_coord_channels", type=int, choices=[1], default=1)
    p.add_argument("--fixed_batch_shape", type=int, choices=[1], default=1)
    p.add_argument("--resume", type=int, choices=[0, 1], default=1)
    p.add_argument("--max_slices", type=int, default=0)
    a = p.parse_args()
    return a


def prepare_model_for_experiment(a):
    model, cfg, compiled = load_v4_model(a.model_path, "cuda", False, "default")
    if compiled:
        raise RuntimeError("production-preserving experiment unexpectedly compiled the model")
    return model, cfg, False, 0.0


def completed_manifest_keys(path: Path):
    if not path.is_file() or not path.stat().st_size:
        return set()
    try:
        frame = pd.read_csv(path)
    except EmptyDataError:
        return set()
    if not {"group_id", "slice_seq", "status"}.issubset(frame.columns):
        return set()
    frame = frame[frame.status.astype(str).eq("completed")]
    return {(str(row.group_id), int(row.slice_seq)) for row in frame.itertuples(index=False)}


def append_timing(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TIMING_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in TIMING_COLUMNS})
        stream.flush()


def manifest_row(a, geo, sess, sid, seq, src, rel, npz, meta_path, center, rows, occupied):
    return {
        "contract_version": CONTRACT_VERSION,
        "id": sid,
        "source": str(src),
        "relative_path": rel,
        "geography": geo,
        "session": sess,
        "slice_seq": seq,
        "group_id": a.session_filter,
        **center,
        "stage1_npz": str(npz),
        "stage1_meta_json": str(meta_path),
        "rows": int(rows),
        "occupied_rows": int(occupied),
        "status": "completed",
    }


def main():
    a = parse_args()
    input_root = Path(a.input_dir).resolve()
    output_root = Path(a.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = discover(input_root, a.session_filter)
    if a.max_slices > 0:
        rows = rows[:a.max_slices]
    geo, sess = a.session_filter.split("/", 1)
    manifest_path = output_root / "stage1_manifest.csv"
    done = completed_manifest_keys(manifest_path)

    setup_torch()
    model, cfg, compiled, warmup_ms = prepare_model_for_experiment(a)
    print(
        f"[stage1-opt2] model_ready compiled={compiled} batch={a.batch_size} "
        f"channels_last={a.channels_last} pinned_d2h={a.pinned_d2h} "
        f"prune_embedding={a.prune_embedding_head} warmup_ms={warmup_ms:.1f}",
        flush=True,
    )
    calibration = load_calibration(a.calibration_json)
    workspace = V4SparseGpuWorkspace()

    for index, (seq, src, rel) in enumerate(rows, 1):
        slice_t0 = time.perf_counter()
        sid = safe_id(rel.replace("/", "__"))
        npz, meta_path = stage1_paths(output_root, rel)
        progress = {
            "state": "running",
            "group_id": a.session_filter,
            "slice_index": index,
            "slice_count": len(rows),
            "slice_seq": int(seq),
            "relative_path": rel,
            "updated_unix": time.time(),
        }
        atomic_json(progress, a.progress_json)

        if a.resume and (a.session_filter, seq) in done and npz.is_file() and meta_path.is_file():
            print(f"[stage1-opt2] {index}/{len(rows)} reuse seq={seq} artifact={npz}", flush=True)
            continue

        if a.resume and npz.is_file() and meta_path.is_file():
            _, _, prior_meta = load_stage1_artifact(npz, meta_path)
            center = dict(prior_meta.get("center_metadata", {}))
            row = manifest_row(
                a, geo, sess, str(prior_meta.get("id", sid)), seq,
                prior_meta.get("source", src), rel, npz, meta_path, center,
                prior_meta.get("rows", 0), prior_meta.get("occupied_rows", 0),
            )
            upsert_manifest_row(manifest_path, row, STAGE1_MANIFEST_COLUMNS)
            done.add((a.session_filter, seq))
            print(f"[stage1-opt2] {index}/{len(rows)} repaired_manifest seq={seq}", flush=True)
            continue

        t0 = time.perf_counter()
        frame = pd.read_csv(src)
        read_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        item = build_sparse_item_from_dataframe(frame, a.grid_size)
        prep_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        pred = predict_v4_sparse_rows_opt(
            item,
            model,
            cfg,
            calibration,
            a.grid_size,
            a.core_size,
            a.batch_size,
            a.amp,
            channels_last=bool(a.channels_last),
            evaluate_all_cores=False,
            gpu_coord_channels=True,
            fixed_batch_shape=True,
            workspace=workspace,
            pinned_d2h=bool(a.pinned_d2h),
            require_compiled=bool(a.compile_model),
        )
        infer_ms = (time.perf_counter() - t0) * 1000.0
        center = extract_center_metadata(frame)
        timing = {
            "csv_read_ms": read_ms,
            "sparse_item_prep_ms": prep_ms,
            "stage1_wall_ms": infer_ms,
            **pred["timing"],
        }
        metadata = {
            "id": sid,
            "source": str(src),
            "relative_path": rel,
            "geography": geo,
            "session": sess,
            "slice_seq": seq,
            "group_id": a.session_filter,
            "rows": len(frame),
            "occupied_rows": len(item["coords"]),
            "center_metadata": center,
            "timing": timing,
            "model_path": str(a.model_path),
            "calibration_json": str(a.calibration_json),
            "compiled": compiled,
            "batch_size": int(a.batch_size),
            "channels_last": bool(a.channels_last),
            "compile_mode": a.compile_mode if a.compile_model else "disabled",
            "pinned_d2h": bool(a.pinned_d2h),
            "prune_embedding_head": bool(a.prune_embedding_head),
            "warmup_iterations": int(a.warmup_iterations),
            "model_warmup_ms": float(warmup_ms),
            "evaluate_all_cores": False,
            "gpu_coord_channels": True,
            "fixed_batch_shape": True,
            "amp": a.amp,
        }

        t0 = time.perf_counter()
        save_stage1_artifact(npz, meta_path, item, pred, metadata)
        artifact_write_ms = (time.perf_counter() - t0) * 1000.0
        row = manifest_row(
            a, geo, sess, sid, seq, src, rel, npz, meta_path, center,
            len(frame), len(item["coords"]),
        )
        t0 = time.perf_counter()
        upsert_manifest_row(manifest_path, row, STAGE1_MANIFEST_COLUMNS)
        manifest_write_ms = (time.perf_counter() - t0) * 1000.0
        done.add((a.session_filter, seq))

        timing_row = {
            "group_id": a.session_filter,
            "slice_seq": seq,
            "relative_path": rel,
            "rows": len(frame),
            "occupied_rows": len(item["coords"]),
            **timing,
            "batch_size": int(a.batch_size),
            "channels_last": int(a.channels_last),
            "compile_model": int(a.compile_model),
            "compile_mode": a.compile_mode if a.compile_model else "disabled",
            "prune_embedding_head": int(a.prune_embedding_head),
            "stage1_artifact_write_ms": artifact_write_ms,
            "stage1_manifest_write_ms": manifest_write_ms,
            "slice_total_ms": (time.perf_counter() - slice_t0) * 1000.0,
        }
        append_timing(Path(a.timing_csv), timing_row)
        print(
            f"[stage1-opt2] {index}/{len(rows)} seq={seq} occupied={len(item['coords'])} "
            f"infer={infer_ms:.1f}ms write={artifact_write_ms:.1f}ms "
            f"workspace_reused={pred['timing'].get('workspace_reused', 0)}",
            flush=True,
        )

    atomic_json(
        {
            "completed": True,
            "contract_version": CONTRACT_VERSION,
            "stage": 1,
            "group_id": a.session_filter,
            "slices": len(rows),
            "manifest": str(manifest_path),
        },
        output_root / "STAGE1_COMPLETED.json",
    )
    atomic_json(
        {
            "state": "completed",
            "group_id": a.session_filter,
            "slice_count": len(rows),
            "updated_unix": time.time(),
        },
        a.progress_json,
    )
    print("V4_STAGE1_OPT2_SESSION_OK", flush=True)


if __name__ == "__main__":
    main()
