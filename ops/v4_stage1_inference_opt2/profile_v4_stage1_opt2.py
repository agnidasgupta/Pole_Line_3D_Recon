#!/usr/bin/env python3
"""Profile an accepted E0/E3/E4/E5/E5b run without writing Stage1 artifacts."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch

from run_v4_realtime_session import discover
from v4_realtime_core import (
    build_sparse_item_from_dataframe,
    load_calibration,
    load_v4_model,
    setup_torch,
)
from v4_realtime_core_opt2 import V4SparseGpuWorkspace, predict_v4_sparse_rows_opt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--session_filter", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--calibration_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--slice_ordinal", type=int, default=-1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--retain_gather_host_buffers", type=int, choices=[0, 1], default=0)
    parser.add_argument("--precompute_batch_gather_plans", type=int, choices=[0, 1], default=0)
    parser.add_argument("--cache_coordinate_channels", type=int, choices=[0, 1], default=0)
    parser.add_argument("--cache_reference_coordinate_channels", type=int, choices=[0, 1], default=0)
    parser.add_argument("--grid_size", type=int, nargs=3, default=[400, 400, 200])
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        parser.error("warmup must be >= 0 and iterations must be >= 1")
    if args.cache_coordinate_channels and args.cache_reference_coordinate_channels:
        parser.error("E5 and E5b coordinate caches are mutually exclusive")
    return args


def main():
    args = parse_args()
    discovered = discover(Path(args.input_dir).resolve(), args.session_filter)
    if not discovered:
        raise SystemExit(f"no slices discovered for {args.session_filter}")
    ordinal = len(discovered) // 2 if args.slice_ordinal < 0 else args.slice_ordinal
    if ordinal < 0 or ordinal >= len(discovered):
        raise SystemExit(f"slice_ordinal {ordinal} outside [0,{len(discovered)-1}]")
    seq, source, relative_path = discovered[ordinal]

    frame = pd.read_csv(source)
    item = build_sparse_item_from_dataframe(frame, args.grid_size)
    setup_torch()
    model, cfg, compiled = load_v4_model(args.model_path, "cuda", False, "default")
    if compiled:
        raise RuntimeError("production-equivalent profiler unexpectedly compiled the model")
    calibration = load_calibration(args.calibration_json)
    workspace = V4SparseGpuWorkspace()

    def infer_once():
        return predict_v4_sparse_rows_opt(
            item, model, cfg, calibration, args.grid_size, 48, 12, "bf16",
            channels_last=True, evaluate_all_cores=False,
            gpu_coord_channels=True, fixed_batch_shape=True,
            workspace=workspace, pinned_d2h=False, require_compiled=False,
            detailed_cuda_timing=True,
            retain_gather_host_buffers=bool(args.retain_gather_host_buffers),
            precompute_batch_gather_plans=bool(args.precompute_batch_gather_plans),
            cache_coordinate_channels=bool(args.cache_coordinate_channels),
            cache_reference_coordinate_channels=bool(args.cache_reference_coordinate_channels),
        )

    for _ in range(args.warmup):
        infer_once()
    torch.cuda.synchronize()

    wall_ms = []
    component_rows = []
    torch.cuda.cudart().cudaProfilerStart()
    if args.cache_reference_coordinate_channels:
        variant = "e5b_reference_coordinate_cache"
    elif args.cache_coordinate_channels:
        variant = "e5_coordinate_input_cache"
    elif args.precompute_batch_gather_plans:
        variant = "e4_precomputed_batch_gather_plans"
    elif args.retain_gather_host_buffers:
        variant = "e3_retain_gather_host_buffers"
    else:
        variant = "e0_control"
    torch.cuda.nvtx.range_push(f"stage1_opt2_{variant}_profile")
    try:
        for index in range(args.iterations):
            torch.cuda.nvtx.range_push(f"stage1_opt2_iteration_{index:03d}")
            started = time.perf_counter()
            try:
                prediction = infer_once()
            finally:
                torch.cuda.nvtx.range_pop()
            wall_ms.append((time.perf_counter() - started) * 1000.0)
            component_rows.append(prediction["timing"])
    finally:
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
    torch.cuda.synchronize()

    keys = sorted({key for row in component_rows for key in row if key.endswith("_ms")})
    mean_components = {}
    for key in keys:
        values = [float(row[key]) for row in component_rows if row.get(key) is not None]
        values = [value for value in values if math.isfinite(value)]
        mean_components[key] = sum(values) / len(values) if values else None

    output = {
        "group_id": args.session_filter,
        "slice_ordinal": int(ordinal),
        "slice_seq": int(seq),
        "relative_path": relative_path,
        "source": str(source),
        "rows": int(len(frame)),
        "occupied_rows": int(len(item["coords"])),
        "warmup_iterations": int(args.warmup),
        "profile_iterations": int(args.iterations),
        "mean_wall_ms": sum(wall_ms) / len(wall_ms),
        "min_wall_ms": min(wall_ms),
        "max_wall_ms": max(wall_ms),
        "mean_timing_components_ms": mean_components,
        "checkpoint_config": cfg,
        "model_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "compiled": False,
        "runtime": {
            "mode": "active_gpu",
            "amp": "bf16",
            "batch_size": 12,
            "channels_last": True,
            "pinned_d2h": False,
            "detailed_cuda_timing": True,
            "retain_gather_host_buffers": bool(args.retain_gather_host_buffers),
            "precompute_batch_gather_plans": bool(args.precompute_batch_gather_plans),
            "cache_coordinate_channels": bool(args.cache_coordinate_channels),
            "cache_reference_coordinate_channels": bool(args.cache_reference_coordinate_channels),
            "full_model_heads": True,
            "patch_size": int(cfg.get("patch_size", 64)),
            "core_size": 48,
            "fixed_batch_shape": True,
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
