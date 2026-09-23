#!/usr/bin/env python3
"""Reproducible V4 Stage-1 training for the deployed five-channel 3-D network.

This is intentionally a training-only entry point. Runtime inference remains in
``run_v4_stage1.py`` and ``v4_realtime_core.py``. All dataset paths are expected
to be container paths; the Nebius launcher mounts the output/data tree at
``/outputs``.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from precision_common import (
    BalancedVoxelPatchDataset,
    fixed_class_weights,
    fuse_scores,
    maybe_compile_model,
    precision_recovery_loss,
    safe_model_state,
    search_thresholds_class_specific,
    update_score_hist,
    write_json_atomic,
)
from voxel_common import MultiHeadVoxelNet3D, read_json, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resume_checkpoint", default="")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--samples_per_epoch", type=int, default=12000)
    p.add_argument("--eval_samples", type=int, default=3072)
    p.add_argument("--batch_size", type=int, default=3)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--cache_items", type=int, default=2)
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--base_channels", type=int, default=16)
    p.add_argument("--use_coord_channels", type=int, default=1)
    p.add_argument("--use_dist", type=int, default=1)
    p.add_argument("--amp", choices=("none", "bf16", "fp16"), default="bf16")
    p.add_argument("--channels_last", type=int, default=1)
    p.add_argument("--compile_model", type=int, default=0)
    p.add_argument("--compile_mode", default="reduce-overhead")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1.0e-4)
    p.add_argument("--min_lr", type=float, default=2.0e-6)
    p.add_argument("--weight_decay", type=float, default=1.0e-4)
    p.add_argument("--early_stopping_patience", type=int, default=10)
    p.add_argument("--scheduler_patience", type=int, default=3)
    p.add_argument("--pole_pos_prob", type=float, default=.25)
    p.add_argument("--line_pos_prob", type=float, default=.25)
    p.add_argument("--hardneg_prob", type=float, default=.40)
    p.add_argument("--class_weight_other", type=float, default=1.0)
    p.add_argument("--class_weight_pole", type=float, default=1.5)
    p.add_argument("--class_weight_line", type=float, default=1.5)
    p.add_argument("--hardneg_weight", type=float, default=6.0)
    p.add_argument("--alpha_pos", type=float, default=.55)
    p.add_argument("--gamma_pos", type=float, default=1.0)
    p.add_argument("--gamma_neg", type=float, default=3.0)
    p.add_argument("--tversky_alpha_fp", type=float, default=.55)
    p.add_argument("--tversky_beta_fn", type=float, default=.45)
    p.add_argument("--lambda_sem", type=float, default=1.0)
    p.add_argument("--lambda_binary", type=float, default=.75)
    p.add_argument("--lambda_tversky", type=float, default=.60)
    p.add_argument("--lambda_objectness", type=float, default=.10)
    p.add_argument("--lambda_fp_penalty", type=float, default=.20)
    p.add_argument("--lambda_cross_class", type=float, default=.15)
    p.add_argument("--score_sem_weight", type=float, default=.55)
    p.add_argument("--score_binary_weight", type=float, default=.35)
    p.add_argument("--score_object_weight", type=float, default=.10)
    p.add_argument("--score_bins", type=int, default=101)
    p.add_argument("--threshold_min", type=float, default=.05)
    p.add_argument("--threshold_max", type=float, default=.98)
    p.add_argument("--threshold_steps", type=int, default=48)
    p.add_argument("--pole_target_precision", type=float, default=.80)
    p.add_argument("--pole_target_recall", type=float, default=.85)
    p.add_argument("--pole_target_iou", type=float, default=.62)
    p.add_argument("--line_target_precision", type=float, default=.58)
    p.add_argument("--line_target_recall", type=float, default=.88)
    p.add_argument("--line_target_iou", type=float, default=.52)
    p.add_argument("--line_recall_weight", type=float, default=2.5)
    return p.parse_args()


def amp_context(amp: str):
    if amp == "none":
        return nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if amp == "bf16" else torch.float16)


def loader(dataset, args: argparse.Namespace) -> DataLoader:
    options = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    if args.num_workers > 0:
        options["prefetch_factor"] = 2
    return DataLoader(dataset, **options)


@torch.no_grad()
def evaluate(model, data_loader, args: argparse.Namespace) -> tuple[dict, list[dict]]:
    model.eval()
    hist = np.zeros((3, args.score_bins, args.score_bins), dtype=np.int64)
    for batch in data_loader:
        x = batch["x"].cuda(non_blocking=True)
        if args.channels_last:
            x = x.contiguous(memory_format=torch.channels_last_3d)
        y = batch["labels"].cuda(non_blocking=True)
        with amp_context(args.amp):
            out = model(x)
        pole, line = fuse_scores(
            out, args.score_sem_weight, args.score_binary_weight, args.score_object_weight
        )
        update_score_hist(hist, y, pole, line, args.score_bins)
    return search_thresholds_class_specific(
        hist,
        args.threshold_min,
        args.threshold_max,
        args.threshold_steps,
        pole_target_precision=args.pole_target_precision,
        pole_target_recall=args.pole_target_recall,
        pole_target_iou=args.pole_target_iou,
        line_target_precision=args.line_target_precision,
        line_target_recall=args.line_target_recall,
        line_target_iou=args.line_target_iou,
        line_recall_weight=args.line_recall_weight,
    )


def load_optional_checkpoint(model: torch.nn.Module, path: str) -> None:
    if not path:
        return
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state", checkpoint)
    clean = {}
    for key, value in state.items():
        for prefix in ("module.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        clean[key] = value
    model.load_state_dict(clean, strict=True)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("V4 Stage-1 training requires a CUDA GPU")
    if args.patch_size != 64:
        raise ValueError("the deployed V4 contract requires patch_size=64")
    if args.grad_accum < 1 or args.batch_size < 1:
        raise ValueError("batch_size and grad_accum must be positive")

    output = Path(args.output_dir)
    train_output = output / "train"
    validation_output = output / "full_val"
    train_output.mkdir(parents=True, exist_ok=True)
    validation_output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    manifests = Path(args.dataset_dir) / "manifests"
    summary = read_json(str(manifests / "summary.json"))
    train_records = read_json(str(manifests / "train.json"))
    val_records = read_json(str(manifests / "val.json"))
    if not train_records or not val_records:
        raise RuntimeError("train/validation manifests must both be nonempty")
    grid = tuple(map(int, summary["grid_size_xyz"]))
    in_channels = 1 + 3 * int(bool(args.use_coord_channels)) + int(bool(args.use_dist))

    train_ds = BalancedVoxelPatchDataset(
        train_records, grid, args.patch_size, args.samples_per_epoch,
        use_coord_channels=bool(args.use_coord_channels), use_dist=bool(args.use_dist),
        pos_pole_prob=args.pole_pos_prob, pos_line_prob=args.line_pos_prob,
        hardneg_prob=args.hardneg_prob, cache_items=args.cache_items,
        deterministic=False, seed=args.seed, jitter=8,
    )
    val_ds = BalancedVoxelPatchDataset(
        val_records, grid, args.patch_size, args.eval_samples,
        use_coord_channels=bool(args.use_coord_channels), use_dist=bool(args.use_dist),
        pos_pole_prob=.25, pos_line_prob=.25, hardneg_prob=.40,
        cache_items=args.cache_items, deterministic=True, seed=args.seed + 10000, jitter=0,
    )
    train_loader = loader(train_ds, args)
    val_loader = loader(val_ds, args)

    model = MultiHeadVoxelNet3D(in_ch=in_channels, base=args.base_channels)
    load_optional_checkpoint(model, args.resume_checkpoint)
    model = model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last_3d)
    model, compiled = maybe_compile_model(model, bool(args.compile_model), args.compile_mode)
    class_weights = fixed_class_weights(
        args.class_weight_other, args.class_weight_pole, args.class_weight_line, "cuda"
    )
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True
        )
    except Exception:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=.5, patience=args.scheduler_patience, min_lr=args.min_lr
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")

    started = time.perf_counter()
    history = []
    best_score = -float("inf")
    no_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        batches = 0
        for step, batch in enumerate(train_loader, 1):
            x = batch["x"].cuda(non_blocking=True)
            y = batch["labels"].cuda(non_blocking=True)
            hard = batch["hardneg"].cuda(non_blocking=True)
            if args.channels_last:
                x = x.contiguous(memory_format=torch.channels_last_3d)
            with amp_context(args.amp):
                output_heads = model(x)
                loss, _ = precision_recovery_loss(output_heads, y, hard, class_weights, args)
                scaled_loss = loss / args.grad_accum
            scaler.scale(scaled_loss).backward()
            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach())
            batches += 1

        best, threshold_rows = evaluate(model, val_loader, args)
        scheduler.step(float(best["score"]))
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, batches),
            "score": float(best["score"]),
            "pole_threshold": float(best["pole_threshold"]),
            "line_threshold": float(best["line_threshold"]),
            "pole_precision": float(best["pole_precision"]),
            "pole_recall": float(best["pole_recall"]),
            "pole_iou": float(best["pole_iou"]),
            "line_precision": float(best["line_precision"]),
            "line_recall": float(best["line_recall"]),
            "line_iou": float(best["line_iou"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(train_output / "v4_stage1_history.csv", index=False)
        pd.DataFrame(threshold_rows).drop(columns=["cm", "class_metrics"]).to_csv(
            validation_output / "threshold_search.csv", index=False
        )
        print(json.dumps(row), flush=True)

        if row["score"] > best_score + 1.0e-9:
            best_score = row["score"]
            no_improvement = 0
            config = {
                **vars(args), "grid_size_xyz": list(grid), "in_channels": in_channels,
                "compiled_during_training": bool(compiled),
            }
            torch.save(
                {"model_state": safe_model_state(model), "config": config, "epoch": epoch,
                 "patch_metrics": best},
                train_output / "precision_best.pt",
            )
            calibration = {
                "pole_threshold": row["pole_threshold"],
                "line_threshold": row["line_threshold"],
                "score_weights": {
                    "semantic": args.score_sem_weight,
                    "binary": args.score_binary_weight,
                    "objectness": args.score_object_weight,
                },
                "selection_metrics": best,
                "warning": "Patch validation calibration. Run the repository full-scene validation gate before promotion.",
            }
            write_json_atomic(calibration, str(validation_output / "calibration.json"))
        else:
            no_improvement += 1
        if no_improvement >= args.early_stopping_patience:
            break

    summary_out = {
        "completed": True,
        "elapsed_seconds": time.perf_counter() - started,
        "best_score": best_score,
        "epochs_completed": len(history),
        "checkpoint": str(train_output / "precision_best.pt"),
        "calibration": str(validation_output / "calibration.json"),
        "quality_rule": "new checkpoints are experimental until full-scene Stage1 and strict Stage2 equivalence gates pass",
    }
    write_json_atomic(summary_out, str(output / "STAGE1_TRAINING_COMPLETED.json"))
    print(json.dumps(summary_out, indent=2))


if __name__ == "__main__":
    main()
