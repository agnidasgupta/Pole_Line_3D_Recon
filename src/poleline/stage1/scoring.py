#!/usr/bin/env python3
"""Shared utilities for precision-recovery fine-tuning and deterministic calibration."""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from voxel_common import (
    IGNORE_INDEX,
    NUM_CLASSES,
    LRUFileCache,
    VoxelPatchDataset,
    extract_patch,
    make_coord_channels,
    load_npz_dense,
)

CLASS_NAMES = ["other_0", "pole_1", "powerline_2"]


class BalancedVoxelPatchDataset(VoxelPatchDataset):
    """Patch sampler whose requested class probabilities are actually honored.

    The earlier sampler picked a random file before picking a class. Because only a
    minority of files contain poles, a nominal pole probability of 0.30 produced far
    fewer than 30% pole-centered patches. This sampler first picks the category, then
    selects an eligible file.
    """

    def __init__(
        self,
        records,
        grid_size,
        patch_size=64,
        samples_per_epoch=2000,
        mode="seg",
        use_coord_channels=True,
        use_dist=True,
        pos_pole_prob=0.25,
        pos_line_prob=0.25,
        hardneg_prob=0.40,
        cache_items=4,
        deterministic=False,
        seed=42,
        jitter=8,
    ):
        super().__init__(
            records=records,
            grid_size=grid_size,
            patch_size=patch_size,
            samples_per_epoch=samples_per_epoch,
            mode=mode,
            use_coord_channels=use_coord_channels,
            use_dist=use_dist,
            pos_pole_prob=pos_pole_prob,
            pos_line_prob=pos_line_prob,
            hardneg_prob=hardneg_prob,
            cache_items=cache_items,
        )
        total = pos_pole_prob + pos_line_prob + hardneg_prob
        if total > 1.0 + 1e-9:
            raise ValueError("Sampling probabilities must sum to <= 1.0")
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self.jitter = int(jitter)
        self.pole_records = [i for i, r in enumerate(records) if int(r.get("label_counts", {}).get("1", 0)) > 0]
        self.line_records = [i for i, r in enumerate(records) if int(r.get("label_counts", {}).get("2", 0)) > 0]
        self.hard_records = [i for i, r in enumerate(records) if int(r.get("hardneg_count", 0)) > 0]
        if not self.hard_records:
            # Stage-5 manifests may not carry an aggregate hardneg count in every copy.
            # Keep all records eligible and verify the raw hardneg array after loading.
            self.hard_records = list(range(len(records)))

    def _rngs(self, idx: int):
        if self.deterministic:
            seed = self.seed + int(idx) * 1000003
            return random.Random(seed), np.random.default_rng(seed)
        return random, np.random.default_rng(np.random.randint(0, 2**32 - 1))

    def _record_for_category(self, category: str, py_rng):
        if category == "pole" and self.pole_records:
            return self.records[py_rng.choice(self.pole_records)]
        if category == "line" and self.line_records:
            return self.records[py_rng.choice(self.line_records)]
        if category == "hard" and self.hard_records:
            return self.records[py_rng.choice(self.hard_records)]
        return py_rng.choice(self.records)

    def _choose(self, idx: int):
        py_rng, np_rng = self._rngs(idx)
        r = py_rng.random()
        if r < self.pos_pole_prob:
            category = "pole"
        elif r < self.pos_pole_prob + self.pos_line_prob:
            category = "line"
        elif r < self.pos_pole_prob + self.pos_line_prob + self.hardneg_prob:
            category = "hard"
        else:
            category = "random"

        # A small bounded retry makes hard-negative selection robust if a manifest
        # lacks per-file hard-negative counts.
        for _ in range(8):
            rec = self._record_for_category(category, py_rng)
            item = self._load(rec)
            coords = item["coords"]
            labels = item["raw_labels"]
            raw_hard = item.get("raw_hardneg")
            choices = None
            if category == "pole":
                choices = np.flatnonzero(labels == 1)
            elif category == "line":
                choices = np.flatnonzero(labels == 2)
            elif category == "hard" and raw_hard is not None:
                choices = np.flatnonzero(raw_hard > 0)
            if choices is None or len(choices) == 0:
                if category == "hard":
                    continue
                choices = np.arange(coords.shape[0])
            pick = int(np_rng.choice(choices))
            center = coords[pick].astype(np.int64).copy()
            if not self.deterministic and self.jitter > 0:
                delta = np_rng.integers(-self.jitter, self.jitter + 1, size=3)
                center += delta.astype(np.int64)
            return rec, item, center

        rec = py_rng.choice(self.records)
        item = self._load(rec)
        pick = int(np_rng.integers(0, max(1, item["coords"].shape[0])))
        center = item["coords"][pick].astype(np.int64)
        return rec, item, center

    def __getitem__(self, idx):
        rec, item, center = self._choose(idx)
        occ = extract_patch(item["occ"], center, self.patch_size, fill_value=0.0)
        labels = extract_patch(item["labels"], center, self.patch_size, fill_value=IGNORE_INDEX)
        hard = extract_patch(item["hardneg"], center, self.patch_size, fill_value=0.0)
        feats = [occ]
        if self.use_coord_channels:
            xch, ych, zch = make_coord_channels(center, self.grid_size, self.patch_size)
            feats.extend([xch, ych, zch])
        if self.use_dist:
            feats.append(extract_patch(item["dist"], center, self.patch_size, fill_value=0.0))
        x = np.stack(feats, axis=0).astype(np.float32)
        return {
            "x": torch.from_numpy(x),
            "labels": torch.from_numpy(labels.astype(np.int64)),
            "hardneg": torch.from_numpy(hard.astype(np.float32)),
            "center": torch.tensor(center.astype(np.int64)),
            "file_id": rec.get("id", ""),
        }


def fixed_class_weights(other: float, pole: float, line: float, device=None):
    return torch.tensor([other, pole, line], dtype=torch.float32, device=device)


def asymmetric_focal_bce(
    logits,
    targets,
    mask,
    voxel_weights=None,
    alpha_pos=0.55,
    gamma_pos=1.0,
    gamma_neg=3.0,
):
    """Asymmetric focal BCE that keeps hard negatives influential.

    Compared with alpha=0.90 in the prior run, alpha_pos=0.55 removes much of the
    positive bias while still protecting recall through positive sampling and the
    Tversky term.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pos = targets > 0.5
    focal = torch.where(pos, (1.0 - p).clamp_min(1e-6) ** gamma_pos, p.clamp_min(1e-6) ** gamma_neg)
    alpha = torch.where(pos, torch.full_like(targets, alpha_pos), torch.full_like(targets, 1.0 - alpha_pos))
    loss = bce * focal * alpha
    if voxel_weights is not None:
        loss = loss * voxel_weights
    loss = loss * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def soft_tversky_loss(logits, targets, mask, alpha_fp=0.55, beta_fn=0.45, smooth=1.0):
    probs = torch.sigmoid(logits) * mask
    targets = targets * mask
    dims = tuple(range(2, probs.ndim))
    tp = (probs * targets).sum(dim=dims)
    fp = (probs * (1.0 - targets) * mask).sum(dim=dims)
    fn = ((1.0 - probs) * targets).sum(dim=dims)
    score = (tp + smooth) / (tp + alpha_fp * fp + beta_fn * fn + smooth)
    return 1.0 - score.mean()


def weighted_ce_fixed(logits, labels, class_weights, voxel_weights=None):
    loss = F.cross_entropy(logits, labels.long(), weight=class_weights, ignore_index=IGNORE_INDEX, reduction="none")
    valid = labels != IGNORE_INDEX
    if voxel_weights is not None:
        loss = loss * voxel_weights
        denom = (valid.float() * voxel_weights).sum().clamp_min(1.0)
    else:
        denom = valid.sum().clamp_min(1)
    return loss[valid].sum() / denom


def precision_recovery_loss(out, labels, hard, class_weights, args):
    valid = (labels != IGNORE_INDEX).float()
    hard_weight = 1.0 + hard * (args.hardneg_weight - 1.0)
    sem = weighted_ce_fixed(out["semantic"], labels, class_weights, hard_weight)
    pole_t = (labels == 1).float().unsqueeze(1)
    line_t = (labels == 2).float().unsqueeze(1)
    obj_t = ((labels == 1) | (labels == 2)).float().unsqueeze(1)
    mask = valid.unsqueeze(1)
    w = hard_weight.unsqueeze(1)

    pole_bce = asymmetric_focal_bce(out["pole"], pole_t, mask, w, args.alpha_pos, args.gamma_pos, args.gamma_neg)
    line_bce = asymmetric_focal_bce(out["line"], line_t, mask, w, args.alpha_pos, args.gamma_pos, args.gamma_neg)
    pole_tv = soft_tversky_loss(out["pole"], pole_t, mask, args.tversky_alpha_fp, args.tversky_beta_fn)
    line_tv = soft_tversky_loss(out["line"], line_t, mask, args.tversky_alpha_fp, args.tversky_beta_fn)
    obj = asymmetric_focal_bce(out["objectness"], obj_t, mask, w, 0.55, 1.0, 3.0)

    # Explicitly suppress asset scores on mined background and cross-class voxels.
    pprob = torch.sigmoid(out["pole"])
    lprob = torch.sigmoid(out["line"])
    hard_mask = (hard > 0).float().unsqueeze(1) * mask
    fp_pen = ((pprob + lprob) * hard_mask).sum() / hard_mask.sum().clamp_min(1.0)
    cross_mask = ((labels == 1) | (labels == 2)).float().unsqueeze(1)
    cross_pen = (
        (lprob * pole_t).sum() + (pprob * line_t).sum()
    ) / cross_mask.sum().clamp_min(1.0)

    total = (
        args.lambda_sem * sem
        + args.lambda_binary * (pole_bce + line_bce)
        + args.lambda_tversky * (pole_tv + line_tv)
        + args.lambda_objectness * obj
        + args.lambda_fp_penalty * fp_pen
        + args.lambda_cross_class * cross_pen
    )
    parts = {
        "sem": float(sem.detach()),
        "pole_bce": float(pole_bce.detach()),
        "line_bce": float(line_bce.detach()),
        "pole_tversky": float(pole_tv.detach()),
        "line_tversky": float(line_tv.detach()),
        "objectness": float(obj.detach()),
        "fp_penalty": float(fp_pen.detach()),
        "cross_class": float(cross_pen.detach()),
    }
    return total, parts


def fuse_scores(out, sem_weight=0.55, binary_weight=0.35, object_weight=0.10):
    sem = torch.softmax(out["semantic"].float(), dim=1)
    pole = torch.sigmoid(out["pole"].float()).squeeze(1)
    line = torch.sigmoid(out["line"].float()).squeeze(1)
    obj = torch.sigmoid(out["objectness"].float()).squeeze(1)
    pole_score = sem_weight * sem[:, 1] + binary_weight * pole + object_weight * obj
    line_score = sem_weight * sem[:, 2] + binary_weight * line + object_weight * obj
    return pole_score.clamp(0, 1), line_score.clamp(0, 1)


def update_score_hist(hist, labels, pole_score, line_score, bins=101):
    valid = labels != IGNORE_INDEX
    if not torch.any(valid):
        return
    y = labels[valid].to(torch.int64)
    pb = torch.clamp((pole_score[valid] * (bins - 1)).round().to(torch.int64), 0, bins - 1)
    lb = torch.clamp((line_score[valid] * (bins - 1)).round().to(torch.int64), 0, bins - 1)
    flat = y * bins * bins + pb * bins + lb
    bc = torch.bincount(flat, minlength=NUM_CLASSES * bins * bins)
    hist += bc.reshape(NUM_CLASSES, bins, bins).detach().cpu().numpy()


def update_cm(cm, labels, pred):
    valid = labels != IGNORE_INDEX
    if not torch.any(valid):
        return
    y = labels[valid].to(torch.int64)
    p = pred[valid].to(torch.int64)
    cm += torch.bincount(y * NUM_CLASSES + p, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES).cpu().numpy()


def class_metrics_from_cm(cm):
    cm = np.asarray(cm, dtype=np.int64)
    rows = []
    for i, name in enumerate(CLASS_NAMES):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(cm[i].sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        iou = tp / max(tp + fp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        f2 = 5 * precision * recall / max(4 * precision + recall, 1e-12)
        rows.append({
            "class_id": i,
            "class_name": name,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f2": f2,
            "iou": iou,
        })
    return rows, float(np.mean([r["iou"] for r in rows]))


def cm_from_hist(hist: np.ndarray, pole_threshold: float, line_threshold: float):
    bins = hist.shape[1]
    scores = np.linspace(0.0, 1.0, bins, dtype=np.float64)
    ps, ls = np.meshgrid(scores, scores, indexing="ij")
    pole_ok = ps >= pole_threshold
    line_ok = ls >= line_threshold
    pred = np.zeros_like(ps, dtype=np.int8)
    only_p = pole_ok & ~line_ok
    only_l = line_ok & ~pole_ok
    both = pole_ok & line_ok
    pred[only_p] = 1
    pred[only_l] = 2
    pred[both & (ps >= ls)] = 1
    pred[both & (ls > ps)] = 2
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for actual in range(NUM_CLASSES):
        h = hist[actual]
        for cls in range(NUM_CLASSES):
            cm[actual, cls] = int(h[pred == cls].sum())
    return cm


def target_score(rows, target_precision=0.95, target_recall=0.95, target_iou=0.80):
    focus = [rows[1], rows[2]]
    gaps = []
    for r in focus:
        gaps.extend([
            max(0.0, target_precision - r["precision"]),
            max(0.0, target_recall - r["recall"]),
            max(0.0, target_iou - r["iou"]),
        ])
    min_metric = min(min(r["precision"], r["recall"], r["iou"]) for r in focus)
    mean_iou = float(np.mean([r["iou"] for r in focus]))
    mean_pr = float(np.mean([r["precision"] + r["recall"] for r in focus]) / 2.0)
    all_met = all(g <= 1e-12 for g in gaps)
    score = (100.0 if all_met else 0.0) + 3.0 * min_metric + 2.0 * mean_iou + mean_pr - 12.0 * sum(gaps)
    return score, all_met, float(sum(gaps))


def search_thresholds(
    hist: np.ndarray,
    threshold_min=0.20,
    threshold_max=0.95,
    threshold_steps=31,
    target_precision=0.95,
    target_recall=0.95,
    target_iou=0.80,
):
    thresholds = np.linspace(threshold_min, threshold_max, threshold_steps)
    best = None
    rows_out = []
    for pt in thresholds:
        for lt in thresholds:
            cm = cm_from_hist(hist, float(pt), float(lt))
            metrics, miou = class_metrics_from_cm(cm)
            score, all_met, total_gap = target_score(metrics, target_precision, target_recall, target_iou)
            row = {
                "pole_threshold": float(pt),
                "line_threshold": float(lt),
                "score": float(score),
                "all_targets_met": bool(all_met),
                "total_target_gap": float(total_gap),
                "miou": miou,
                "pole_precision": metrics[1]["precision"],
                "pole_recall": metrics[1]["recall"],
                "pole_iou": metrics[1]["iou"],
                "line_precision": metrics[2]["precision"],
                "line_recall": metrics[2]["recall"],
                "line_iou": metrics[2]["iou"],
                "cm": cm.tolist(),
                "class_metrics": metrics,
            }
            rows_out.append(row)
            if best is None or row["score"] > best["score"]:
                best = row
    return best, rows_out


def prediction_from_scores(pole_score, line_score, pole_threshold, line_threshold):
    pred = torch.zeros_like(pole_score, dtype=torch.long)
    pole_ok = pole_score >= pole_threshold
    line_ok = line_score >= line_threshold
    pred[pole_ok & ~line_ok] = 1
    pred[line_ok & ~pole_ok] = 2
    both = pole_ok & line_ok
    pred[both & (pole_score >= line_score)] = 1
    pred[both & (line_score > pole_score)] = 2
    return pred


def safe_model_state(model):
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    return model.state_dict()


def maybe_compile_model(model, enabled: bool, mode: str = "reduce-overhead"):
    if not enabled or not hasattr(torch, "compile"):
        return model, False
    try:
        compiled = torch.compile(model, mode=mode, dynamic=False)
        return compiled, True
    except Exception as exc:
        print(f"torch.compile unavailable; continuing in eager mode: {exc}")
        return model, False


def write_json_atomic(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def target_score_class_specific(
    rows,
    pole_target_precision=0.85,
    pole_target_recall=0.90,
    pole_target_iou=0.65,
    line_target_precision=0.60,
    line_target_recall=0.88,
    line_target_iou=0.55,
    line_recall_weight=2.5,
):
    """Target-aware score with separate pole/line goals.

    V4 realtime evaluation can use this instead of forcing the same precision target on
    both asset classes.  The line target intentionally prioritizes recall because
    reviewed V4 'false positives' include legitimate/unlabelled conductors.
    """
    pole = rows[1]
    line = rows[2]
    pgaps = [
        max(0.0, pole_target_precision - pole["precision"]),
        max(0.0, pole_target_recall - pole["recall"]),
        max(0.0, pole_target_iou - pole["iou"]),
    ]
    lgaps = [
        max(0.0, line_target_precision - line["precision"]),
        max(0.0, line_target_recall - line["recall"]),
        max(0.0, line_target_iou - line["iou"]),
    ]
    all_met = all(g <= 1e-12 for g in pgaps + lgaps)
    # Preserve pole quality while explicitly rewarding line recall and IoU.
    score = (
        (100.0 if all_met else 0.0)
        + 2.0 * pole["precision"]
        + 2.5 * pole["recall"]
        + 3.0 * pole["iou"]
        + 1.0 * line["precision"]
        + float(line_recall_weight) * line["recall"]
        + 3.0 * line["iou"]
        - 10.0 * sum(pgaps)
        - 8.0 * (lgaps[0] + lgaps[2])
        - 12.0 * lgaps[1]
    )
    return float(score), bool(all_met), float(sum(pgaps) + sum(lgaps))


def search_thresholds_class_specific(
    hist: np.ndarray,
    threshold_min=0.01,
    threshold_max=0.95,
    threshold_steps=95,
    pole_target_precision=0.85,
    pole_target_recall=0.90,
    pole_target_iou=0.65,
    line_target_precision=0.60,
    line_target_recall=0.88,
    line_target_iou=0.55,
    line_recall_weight=2.5,
):
    thresholds = np.linspace(threshold_min, threshold_max, threshold_steps)
    best = None
    rows_out = []
    for pt in thresholds:
        for lt in thresholds:
            cm = cm_from_hist(hist, float(pt), float(lt))
            metrics, miou = class_metrics_from_cm(cm)
            score, all_met, total_gap = target_score_class_specific(
                metrics,
                pole_target_precision=pole_target_precision,
                pole_target_recall=pole_target_recall,
                pole_target_iou=pole_target_iou,
                line_target_precision=line_target_precision,
                line_target_recall=line_target_recall,
                line_target_iou=line_target_iou,
                line_recall_weight=line_recall_weight,
            )
            row = {
                "pole_threshold": float(pt),
                "line_threshold": float(lt),
                "score": float(score),
                "all_targets_met": bool(all_met),
                "total_target_gap": float(total_gap),
                "miou": miou,
                "pole_precision": metrics[1]["precision"],
                "pole_recall": metrics[1]["recall"],
                "pole_iou": metrics[1]["iou"],
                "line_precision": metrics[2]["precision"],
                "line_recall": metrics[2]["recall"],
                "line_iou": metrics[2]["iou"],
                "cm": cm.tolist(),
                "class_metrics": metrics,
            }
            rows_out.append(row)
            if best is None or row["score"] > best["score"]:
                best = row
    return best, rows_out


def update_score_hist_torch(hist: torch.Tensor, labels, pole_score, line_score, bins=101):
    """GPU-resident score histogram accumulation to avoid per-batch D2H sync."""
    valid = labels != IGNORE_INDEX
    if not torch.any(valid):
        return
    y = labels[valid].to(torch.int64)
    pb = torch.clamp((pole_score[valid] * (bins - 1)).round().to(torch.int64), 0, bins - 1)
    lb = torch.clamp((line_score[valid] * (bins - 1)).round().to(torch.int64), 0, bins - 1)
    flat = y * bins * bins + pb * bins + lb
    hist.add_(torch.bincount(flat, minlength=NUM_CLASSES * bins * bins).reshape(NUM_CLASSES, bins, bins))


def update_cm_torch(cm: torch.Tensor, labels, pred):
    """GPU-resident confusion-matrix accumulation to avoid per-batch D2H sync."""
    valid = labels != IGNORE_INDEX
    if not torch.any(valid):
        return
    y = labels[valid].to(torch.int64)
    p = pred[valid].to(torch.int64)
    cm.add_(torch.bincount(y * NUM_CLASSES + p, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES))
