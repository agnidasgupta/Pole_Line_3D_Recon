#!/usr/bin/env python3
"""Train dual-scale geometry model with balanced FP/FN replay."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from contextlib import nullcontext

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from voxel_common import read_json, seed_everything, MultiHeadVoxelNet3D
from precision_common import (
    class_metrics_from_cm,
    fuse_scores,
    maybe_compile_model,
    safe_model_state,
    search_thresholds,
    search_thresholds_class_specific,
    target_score,
    update_cm,
    update_score_hist,
    write_json_atomic,
)
from v6_common import SpanAwareGeoNet3D, SpanAwarePatchDataset, load_from_v4_checkpoint, v6_loss


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--replay_dir", default=None)
    ap.add_argument("--resume_checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--samples_per_epoch", type=int, default=16000)
    ap.add_argument("--eval_samples", type=int, default=3072)
    ap.add_argument("--batch_size", type=int, default=3)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--patch_size", type=int, default=64)
    ap.add_argument("--context_xy", type=int, default=256)
    ap.add_argument("--context_z", type=int, default=128)
    ap.add_argument("--base_channels", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min_lr", type=float, default=2e-6)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--freeze_fine_epochs", type=int, default=2)
    ap.add_argument("--early_stopping_patience", type=int, default=10)
    ap.add_argument("--scheduler_patience", type=int, default=3)
    ap.add_argument("--num_workers", type=int, default=10)
    ap.add_argument("--cache_items", type=int, default=1)
    ap.add_argument("--amp", choices=["none", "bf16", "fp16"], default="bf16")
    ap.add_argument("--channels_last", type=int, default=1)
    ap.add_argument("--compile_model", type=int, default=1)
    ap.add_argument("--compile_mode", default="reduce-overhead")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_coord_channels", type=int, default=1)
    ap.add_argument("--use_dist", type=int, default=1)

    # V4 positive-teacher distillation: transfer V4 line sensitivity without
    # copying V4 background/negative decisions.
    ap.add_argument("--v4_teacher_checkpoint", default=None)
    ap.add_argument("--v4_teacher_calibration_json", default=None)
    ap.add_argument("--lambda_v4_line_teacher", type=float, default=0.30)
    ap.add_argument("--v4_teacher_min_line_score", type=float, default=0.30)
    ap.add_argument("--v4_teacher_line_over_pole_margin", type=float, default=0.05)
    ap.add_argument("--v4_teacher_vertical_margin", type=float, default=0.15)
    ap.add_argument("--orientation_support_length", type=int, default=11)

    ap.add_argument("--pole_pos_prob", type=float, default=0.25)
    ap.add_argument("--line_pos_prob", type=float, default=0.30)
    ap.add_argument("--pole_fp_prob", type=float, default=0.0)
    ap.add_argument("--line_fp_prob", type=float, default=0.0)
    ap.add_argument("--pole_fn_prob", type=float, default=0.05)
    ap.add_argument("--line_fn_prob", type=float, default=0.05)

    ap.add_argument("--class_weight_other", type=float, default=1.0)
    ap.add_argument("--class_weight_pole", type=float, default=1.5)
    ap.add_argument("--class_weight_line", type=float, default=1.5)
    ap.add_argument("--hardneg_weight", type=float, default=1.0)
    ap.add_argument("--replay_weight", type=float, default=6.0)
    ap.add_argument("--alpha_pos", type=float, default=0.60)
    ap.add_argument("--gamma_pos", type=float, default=1.0)
    ap.add_argument("--gamma_neg", type=float, default=2.0)
    ap.add_argument("--lambda_sem", type=float, default=1.0)
    ap.add_argument("--lambda_binary", type=float, default=0.75)
    ap.add_argument("--lambda_iou", type=float, default=0.60)
    ap.add_argument("--lambda_objectness", type=float, default=0.10)
    ap.add_argument("--lambda_replay_fp", type=float, default=0.0)
    ap.add_argument("--lambda_replay_fn", type=float, default=0.35)
    ap.add_argument("--lambda_cross_class", type=float, default=0.15)
    ap.add_argument("--lambda_orientation", type=float, default=0.20)
    ap.add_argument("--lambda_physics", type=float, default=0.25)
    ap.add_argument("--orientation_margin", type=float, default=0.05)
    ap.add_argument("--geometry_unlabeled_discount", type=float, default=0.85,
                    help="Downweight background loss on strongly pole/line-like geometry because GT catenary/pole labels can be incomplete")
    ap.add_argument("--boundary_radius", type=float, default=2.5)
    ap.add_argument("--soft_sigma", type=float, default=1.25)

    ap.add_argument("--edge_asset_prob", type=float, default=0.15,
                    help="Sample labelled pole/line voxels near local XY slice boundaries; uses no world/center information")
    ap.add_argument("--edge_width_vox", type=int, default=10, help="Boundary band for edge-asset sampling")
    ap.add_argument("--score_sem_weight", type=float, default=0.55)
    ap.add_argument("--score_binary_weight", type=float, default=0.35)
    ap.add_argument("--score_object_weight", type=float, default=0.10)
    ap.add_argument("--score_bins", type=int, default=101)
    ap.add_argument("--threshold_min", type=float, default=0.05)
    ap.add_argument("--threshold_max", type=float, default=0.98)
    ap.add_argument("--threshold_steps", type=int, default=48)
    # Retain legacy common targets for compatibility, but teacher-recall model
    # selection uses the class-specific targets below.
    ap.add_argument("--target_precision", type=float, default=0.95)
    ap.add_argument("--target_recall", type=float, default=0.95)
    ap.add_argument("--target_iou", type=float, default=0.80)
    ap.add_argument("--pole_target_precision", type=float, default=0.80)
    ap.add_argument("--pole_target_recall", type=float, default=0.85)
    ap.add_argument("--pole_target_iou", type=float, default=0.62)
    ap.add_argument("--line_target_precision", type=float, default=0.58)
    ap.add_argument("--line_target_recall", type=float, default=0.88)
    ap.add_argument("--line_target_iou", type=float, default=0.52)
    ap.add_argument("--line_recall_weight", type=float, default=2.5)
    ap.add_argument("--keep_candidates", type=int, default=3)
    return ap.parse_args()


def setup_torch():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def autocast_ctx(args):
    if args.amp == "none":
        return nullcontext()
    dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def make_loader(ds, args):
    kw = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
              pin_memory=True, persistent_workers=args.num_workers > 0)
    if args.num_workers > 0:
        kw["prefetch_factor"] = 2
    return DataLoader(ds, **kw)


@torch.no_grad()
def evaluate(model, dl, args):
    model.eval()
    hist = np.zeros((3, args.score_bins, args.score_bins), dtype=np.int64)
    raw_cm = np.zeros((3, 3), dtype=np.int64)
    loss_sum = 0.0
    batches = 0
    for batch in tqdm(dl, desc="deterministic V6 Stage-1 validation", leave=False):
        xf = batch["x_fine"].cuda(non_blocking=True)
        xc = batch["x_coarse"].cuda(non_blocking=True)
        if args.channels_last:
            xf = xf.contiguous(memory_format=torch.channels_last_3d)
            xc = xc.contiguous(memory_format=torch.channels_last_3d)
        y = batch["labels"].cuda(non_blocking=True)
        hard = batch["hardneg"].cuda(non_blocking=True)
        boundary = batch["boundary_ignore"].cuda(non_blocking=True)
        pole_soft = batch["pole_soft"].cuda(non_blocking=True)
        line_soft = batch["line_soft"].cuda(non_blocking=True)
        replay_masks = batch["replay_masks"].cuda(non_blocking=True)
        with autocast_ctx(args):
            out = model(xf, xc)
            loss, _ = v6_loss(out, xf, y, hard, boundary, pole_soft, line_soft, replay_masks, args)
        loss_sum += float(loss.detach())
        batches += 1
        update_cm(raw_cm, y, out["semantic"].argmax(dim=1))
        ps, ls = fuse_scores(out, args.score_sem_weight, args.score_binary_weight, args.score_object_weight)
        update_score_hist(hist, y, ps, ls, args.score_bins)
    best, rows = search_thresholds_class_specific(
        hist, args.threshold_min, args.threshold_max, args.threshold_steps,
        pole_target_precision=args.pole_target_precision,
        pole_target_recall=args.pole_target_recall,
        pole_target_iou=args.pole_target_iou,
        line_target_precision=args.line_target_precision,
        line_target_recall=args.line_target_recall,
        line_target_iou=args.line_target_iou,
        line_recall_weight=args.line_recall_weight)
    raw_rows, raw_miou = class_metrics_from_cm(raw_cm)
    best["loss"] = loss_sum / max(batches, 1)
    best["raw_cm"] = raw_cm.tolist()
    best["raw_class_metrics"] = raw_rows
    best["raw_miou"] = raw_miou
    return best, rows


def save_plots(rows, outdir):
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "v6_stage1_history.csv"), index=False)
    plt.figure(figsize=(9, 5))
    plt.plot(df.epoch, df.train_loss, label="train")
    plt.plot(df.epoch, df.val_loss, label="validation")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.grid(True, alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_curve.png"), dpi=180); plt.close()
    plt.figure(figsize=(11, 7))
    for c in ["pole_precision","pole_recall","pole_iou","line_precision","line_recall","line_iou"]:
        plt.plot(df.epoch, df[c], label=c)
    plt.axhline(.95, linestyle="--", linewidth=1)
    plt.axhline(.80, linestyle=":", linewidth=1)
    plt.ylim(0,1.01); plt.xlabel("epoch"); plt.ylabel("metric"); plt.grid(True, alpha=.3); plt.legend(ncol=2); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "learning_curve.png"), dpi=180); plt.close()
    if "train_teacher_line_distill" in df.columns:
        fig,ax1=plt.subplots(figsize=(9,5))
        ax1.plot(df.epoch,df.train_teacher_line_distill,label="V4 positive-teacher loss")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("teacher distillation loss"); ax1.grid(True,alpha=.3)
        ax2=ax1.twinx()
        if "train_teacher_positive_fraction" in df.columns:
            ax2.plot(df.epoch,df.train_teacher_positive_fraction,linestyle="--",label="teacher-positive fraction")
            ax2.set_ylabel("fraction of patch voxels")
        lines=ax1.get_lines()+ax2.get_lines(); ax1.legend(lines,[x.get_label() for x in lines],loc="best")
        fig.tight_layout(); fig.savefig(os.path.join(outdir,"v4_teacher_distillation_curve.png"),dpi=180); plt.close(fig)


def load_v4_teacher(args, in_ch):
    path = args.v4_teacher_checkpoint or args.resume_checkpoint
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    teacher_in = int(ckpt.get("in_channels", in_ch))
    if teacher_in != in_ch:
        raise RuntimeError(f"V4 teacher input channels {teacher_in} != V6 fine input channels {in_ch}")
    teacher = MultiHeadVoxelNet3D(in_ch=teacher_in, base=int(cfg.get("base_channels", args.base_channels)))
    state = ckpt.get("model_state", ckpt)
    missing, unexpected = teacher.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"V4 teacher checkpoint mismatch missing={missing} unexpected={unexpected}")
    teacher = teacher.cuda().eval()
    if args.channels_last:
        teacher = teacher.to(memory_format=torch.channels_last_3d)
    for q in teacher.parameters():
        q.requires_grad_(False)
    score_weights = {"semantic": .55, "binary": .35, "objectness": .10}
    calibration_threshold = 0.0
    cal_path = args.v4_teacher_calibration_json
    if cal_path and os.path.exists(cal_path):
        cal = read_json(cal_path)
        score_weights.update(cal.get("score_weights", {}))
        calibration_threshold = float(cal.get("line_threshold", 0.0))
    effective_min = max(float(args.v4_teacher_min_line_score), calibration_threshold)
    args.v4_teacher_min_line_score = effective_min
    return teacher, {"checkpoint":path,"calibration":cal_path,"score_weights":score_weights,
                     "calibration_line_threshold":calibration_threshold,"effective_min_line_score":effective_min}


def main():
    args = parse_args()
    if args.context_xy % args.patch_size != 0 or args.context_z % args.patch_size != 0:
        raise ValueError("context_xy and context_z must be divisible by patch_size")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    os.makedirs(args.output_dir, exist_ok=True)
    candidates_dir = os.path.join(args.output_dir, "candidates")
    os.makedirs(candidates_dir, exist_ok=True)
    setup_torch(); seed_everything(args.seed)

    summary = read_json(os.path.join(args.dataset_dir, "manifests", "summary.json"))
    train_records = read_json(os.path.join(args.dataset_dir, "manifests", "train.json"))
    val_records = read_json(os.path.join(args.dataset_dir, "manifests", "val.json"))
    grid_size = summary["grid_size_xyz"]
    in_ch = 1 + (3 if args.use_coord_channels else 0) + (1 if args.use_dist else 0)

    train_ds = SpanAwarePatchDataset(
        records=train_records, grid_size=grid_size, replay_dir=args.replay_dir,
        patch_size=args.patch_size, context_xy=args.context_xy, context_z=args.context_z,
        samples_per_epoch=args.samples_per_epoch, use_coord_channels=args.use_coord_channels, use_dist=args.use_dist,
        pole_pos_prob=args.pole_pos_prob, line_pos_prob=args.line_pos_prob, pole_fp_prob=args.pole_fp_prob,
        line_fp_prob=args.line_fp_prob, pole_fn_prob=args.pole_fn_prob, line_fn_prob=args.line_fn_prob,
        edge_asset_prob=args.edge_asset_prob, edge_width_vox=args.edge_width_vox, cache_items=args.cache_items,
        deterministic=False, seed=args.seed, jitter=6, replay_radius=3,
        boundary_radius=args.boundary_radius, soft_sigma=args.soft_sigma)
    # Validation is deterministic and entirely local. No world/center metadata is loaded.
    val_ds = SpanAwarePatchDataset(
        records=val_records, grid_size=grid_size, replay_dir=None, patch_size=args.patch_size,
        context_xy=args.context_xy, context_z=args.context_z, samples_per_epoch=args.eval_samples,
        use_coord_channels=args.use_coord_channels, use_dist=args.use_dist,
        pole_pos_prob=.25, line_pos_prob=.30, pole_fp_prob=0., line_fp_prob=0., pole_fn_prob=0., line_fn_prob=0.,
        edge_asset_prob=.15, edge_width_vox=args.edge_width_vox, cache_items=args.cache_items, deterministic=True,
        seed=args.seed+10000, jitter=0, replay_radius=0, boundary_radius=args.boundary_radius, soft_sigma=args.soft_sigma)
    train_dl = make_loader(train_ds, args)
    val_dl = make_loader(val_ds, args)

    model = SpanAwareGeoNet3D(in_ch=in_ch, base=args.base_channels)
    transfer = load_from_v4_checkpoint(model, args.resume_checkpoint)
    print("V4 transfer:", transfer)
    teacher, teacher_info = load_v4_teacher(args, in_ch)
    print("V4 positive teacher:", json.dumps(teacher_info, indent=2))
    model = model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last_3d)
    model, compiled = maybe_compile_model(model, bool(args.compile_model), args.compile_mode)
    print("GPU:", torch.cuda.get_device_name(0), "torch.compile:", compiled)

    try:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True)
    except Exception:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=.5,
                                                       patience=args.scheduler_patience, min_lr=args.min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    history = []
    candidates = []
    no_improve = 0
    best_score = -1e18
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        freeze = epoch <= args.freeze_fine_epochs
        for p in base_model.fine_encoder.parameters():
            p.requires_grad = not freeze
        model.train(); opt.zero_grad(set_to_none=True)
        total = 0.0; steps = 0; part_sum = {}
        for step, batch in enumerate(tqdm(train_dl, desc=f"V6 Stage-1 train epoch {epoch}"), start=1):
            xf = batch["x_fine"].cuda(non_blocking=True)
            xc = batch["x_coarse"].cuda(non_blocking=True)
            if args.channels_last:
                xf = xf.contiguous(memory_format=torch.channels_last_3d)
                xc = xc.contiguous(memory_format=torch.channels_last_3d)
            y = batch["labels"].cuda(non_blocking=True)
            hard = batch["hardneg"].cuda(non_blocking=True)
            boundary = batch["boundary_ignore"].cuda(non_blocking=True)
            pole_soft = batch["pole_soft"].cuda(non_blocking=True)
            line_soft = batch["line_soft"].cuda(non_blocking=True)
            replay_masks = batch["replay_masks"].cuda(non_blocking=True)
            with torch.inference_mode(), autocast_ctx(args):
                tout = teacher(xf)
                tps, tls = fuse_scores(
                    tout,
                    float(teacher_info["score_weights"].get("semantic", .55)),
                    float(teacher_info["score_weights"].get("binary", .35)),
                    float(teacher_info["score_weights"].get("objectness", .10)),
                )
            with autocast_ctx(args):
                out = model(xf, xc)
                loss, parts = v6_loss(
                    out, xf, y, hard, boundary, pole_soft, line_soft, replay_masks, args,
                    teacher_pole_score=tps, teacher_line_score=tls
                )
                scaled = loss / args.grad_accum
            scaler.scale(scaled).backward()
            if step % args.grad_accum == 0 or step == len(train_dl):
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            total += float(loss.detach()); steps += 1
            for k,v in parts.items(): part_sum[k] = part_sum.get(k,0.0) + v

        val, search_rows = evaluate(model, val_dl, args)
        score = float(val["score"])
        sched.step(score)
        row = {
            "epoch": epoch, "train_loss": total/max(steps,1), "val_loss": val["loss"],
            "score": score, "total_target_gap": val["total_target_gap"],
            "pole_threshold": val["pole_threshold"], "line_threshold": val["line_threshold"],
            "pole_precision": val["pole_precision"], "pole_recall": val["pole_recall"], "pole_iou": val["pole_iou"],
            "line_precision": val["line_precision"], "line_recall": val["line_recall"], "line_iou": val["line_iou"],
            "lr": opt.param_groups[0]["lr"], "fine_encoder_frozen": int(freeze),
        }
        for k,v in part_sum.items(): row["train_"+k] = v/max(steps,1)
        history.append(row); save_plots(history, args.output_dir)
        print(json.dumps(row))

        ckpt_path = os.path.join(candidates_dir, f"epoch_{epoch:03d}.pt")
        torch.save({"model_state": safe_model_state(model), "epoch": epoch, "patch_metrics": val,
                    "config": {**vars(args), "grid_size_xyz": list(grid_size)}, "transfer": transfer, "v4_teacher": teacher_info}, ckpt_path)
        candidates.append({"epoch": epoch, "score": score, "path": ckpt_path,
                           "target_gap": val["total_target_gap"],
                           "pole_iou": val["pole_iou"], "line_iou": val["line_iou"]})
        candidates.sort(key=lambda x: x["score"], reverse=True)
        while len(candidates) > args.keep_candidates:
            old = candidates.pop()
            if os.path.exists(old["path"]): os.remove(old["path"])
        write_json_atomic({"candidates": candidates, "config": vars(args)},
                          os.path.join(args.output_dir, "candidate_manifest.json"))

        if score > best_score + 1e-6:
            best_score = score; no_improve = 0
        else:
            no_improve += 1
        if no_improve >= args.early_stopping_patience:
            print("Early stopping after", epoch)
            break

    result = {
        "completed": True, "elapsed_seconds": time.perf_counter()-start,
        "candidates": candidates, "history_rows": len(history), "config": vars(args),
        "v4_teacher": teacher_info,
        "warning": "Patch metrics select candidates only. Final model selection must use exhaustive full-validation evaluation. Strict GT precision can understate teacher-supported conductor quality where GT is incomplete.",
    }
    write_json_atomic(result, os.path.join(args.output_dir, "training_summary.json"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
