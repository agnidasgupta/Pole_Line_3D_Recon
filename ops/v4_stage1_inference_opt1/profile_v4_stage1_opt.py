#!/usr/bin/env python3
"""Profile one representative slice without writing Stage-1 result artifacts."""
from __future__ import annotations

import argparse
import json
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
from v4_realtime_core_opt import V4SparseGpuWorkspace, predict_v4_sparse_rows_opt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--session_filter", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--calibration_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--slice_ordinal", type=int, default=-1,
                        help="zero-based discovered row; -1 selects the median row")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--grid_size", type=int, nargs=3, default=[400, 400, 200])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.warmup < 0 or args.iterations < 1:
        raise SystemExit("warmup must be >= 0 and iterations must be >= 1")
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
    calibration = load_calibration(args.calibration_json)
    workspace = V4SparseGpuWorkspace()

    def infer_once():
        return predict_v4_sparse_rows_opt(
            item, model, cfg, calibration, args.grid_size, 48, 12, "bf16",
            evaluate_all_cores=False, gpu_coord_channels=True,
            fixed_batch_shape=True, workspace=workspace,
        )

    for _ in range(args.warmup):
        infer_once()
    torch.cuda.synchronize()

    wall_ms = []
    component_rows = []
    torch.cuda.nvtx.range_push("stage1_opt_profile")
    try:
        for index in range(args.iterations):
            torch.cuda.nvtx.range_push(f"stage1_opt_iteration_{index:03d}")
            started = time.perf_counter()
            try:
                prediction = infer_once()
            finally:
                torch.cuda.nvtx.range_pop()
            wall_ms.append((time.perf_counter() - started) * 1000.0)
            component_rows.append(prediction["timing"])
    finally:
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

    keys = sorted({key for row in component_rows for key in row if key.endswith("_ms")})
    mean_components = {
        key: sum(float(row.get(key, 0.0)) for row in component_rows) / len(component_rows)
        for key in keys
    }
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
        "compiled": bool(compiled),
        "runtime": {
            "mode": "active_gpu",
            "amp": "bf16",
            "batch_size": 12,
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
