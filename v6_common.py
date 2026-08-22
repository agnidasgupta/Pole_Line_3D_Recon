#!/usr/bin/env python3
"""V6 span-aware dual-scale model, dataset, and physics-aware losses.

The fine branch operates at native 64^3 resolution. The context branch covers a
wide, anisotropic region (default 256 x 256 x 128 voxels in X/Y/Z) and pools it
to the same 64^3 tensor shape. This gives the network substantially more
horizontal conductor context without discarding pole-height detail.
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset

from voxel_common import (
    IGNORE_INDEX,
    NUM_CLASSES,
    Encoder3D,
    UpBlock3D,
    LRUFileCache,
    load_npz_dense,
)

REPLAY_TYPES = {"pole_fp": 0, "line_fp": 1, "pole_fn": 2, "line_fn": 3}


def safe_id(rec: Dict) -> str:
    return rec.get("id", os.path.basename(rec["npz_path"]).replace(".npz", "")).replace("/", "__")


def extract_patch_shape(arr: np.ndarray, center_xyz: Sequence[int], shape_zyx: Sequence[int], fill_value=0):
    """Extract a rectangular [Z,Y,X] patch centered on x,y,z with padding."""
    x, y, z = [int(v) for v in center_xyz]
    pz, py, px = [int(v) for v in shape_zyx]
    z0, y0, x0 = z - pz // 2, y - py // 2, x - px // 2
    z1, y1, x1 = z0 + pz, y0 + py, x0 + px
    out = np.full((pz, py, px), fill_value, dtype=arr.dtype)
    sz, sy, sx = arr.shape
    az0, ay0, ax0 = max(0, z0), max(0, y0), max(0, x0)
    az1, ay1, ax1 = min(sz, z1), min(sy, y1), min(sx, x1)
    if az1 <= az0 or ay1 <= ay0 or ax1 <= ax0:
        return out
    oz0, oy0, ox0 = az0 - z0, ay0 - y0, ax0 - x0
    out[oz0:oz0 + az1 - az0, oy0:oy0 + ay1 - ay0, ox0:ox0 + ax1 - ax0] = arr[az0:az1, ay0:ay1, ax0:ax1]
    return out


def _pool_rect(a: np.ndarray, out_size: int, mode: str) -> np.ndarray:
    z, y, x = a.shape
    if z % out_size or y % out_size or x % out_size:
        raise ValueError(f"Context shape {a.shape} must be divisible by output size {out_size}")
    fz, fy, fx = z // out_size, y // out_size, x // out_size
    r = a.reshape(out_size, fz, out_size, fy, out_size, fx)
    if mode == "max":
        return r.max(axis=(1, 3, 5))
    if mode == "mean":
        return r.mean(axis=(1, 3, 5))
    raise ValueError(mode)


def _coord_channels(center_xyz, grid_size, shape_zyx, output_size=None):
    gx, gy, gz = [int(v) for v in grid_size]
    x, y, z = [float(v) for v in center_xyz]
    pz, py, px = [int(v) for v in shape_zyx]
    if output_size is None:
        xs = x - px / 2 + np.arange(px, dtype=np.float32)
        ys = y - py / 2 + np.arange(py, dtype=np.float32)
        zs = z - pz / 2 + np.arange(pz, dtype=np.float32)
    else:
        out = int(output_size)
        fx, fy, fz = px / out, py / out, pz / out
        xs = x - px / 2 + (np.arange(out, dtype=np.float32) + 0.5) * fx
        ys = y - py / 2 + (np.arange(out, dtype=np.float32) + 0.5) * fy
        zs = z - pz / 2 + (np.arange(out, dtype=np.float32) + 0.5) * fz
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    return tuple(
        np.clip((a / max(n - 1, 1)) * 2.0 - 1.0, -1.5, 1.5).astype(np.float32)
        for a, n in ((xx, gx), (yy, gy), (zz, gz))
    )


def _soft_target_and_boundary(labels: np.ndarray, class_id: int, radius: float, sigma: float):
    valid = labels != IGNORE_INDEX
    positive = labels == class_id
    soft = np.zeros(labels.shape, dtype=np.float32)
    boundary = np.zeros(labels.shape, dtype=np.float32)
    if positive.any():
        dist = distance_transform_edt(~positive).astype(np.float32)
        soft = np.exp(-0.5 * (dist / max(float(sigma), 1e-3)) ** 2).astype(np.float32)
        soft[dist > max(float(radius) * 2.0, 1.0)] = 0.0
        soft[positive] = 1.0
        soft[~valid] = 0.0
        boundary[(labels == 0) & (dist <= float(radius))] = 1.0
    return soft, boundary


def build_v6_features(
    item: Dict,
    center_xyz: np.ndarray,
    grid_size: Tuple[int, int, int],
    patch_size: int,
    context_xy: int,
    context_z: int,
    use_coord_channels: bool,
    use_dist: bool,
    boundary_radius: float = 2.5,
    soft_sigma: float = 1.25,
):
    fine_shape = (patch_size, patch_size, patch_size)
    ctx_shape = (context_z, context_xy, context_xy)
    fine_occ = extract_patch_shape(item["occ"], center_xyz, fine_shape, 0.0)
    fine = [fine_occ]
    if use_coord_channels:
        fine.extend(_coord_channels(center_xyz, grid_size, fine_shape))
    if use_dist:
        fine.append(extract_patch_shape(item["dist"], center_xyz, fine_shape, 0.0))

    ctx_occ = extract_patch_shape(item["occ"], center_xyz, ctx_shape, 0.0)
    coarse = [_pool_rect(ctx_occ, patch_size, "max")]
    if use_coord_channels:
        coarse.extend(_coord_channels(center_xyz, grid_size, ctx_shape, output_size=patch_size))
    if use_dist:
        coarse.append(_pool_rect(extract_patch_shape(item["dist"], center_xyz, ctx_shape, 0.0), patch_size, "mean"))

    labels = extract_patch_shape(item["labels"], center_xyz, fine_shape, IGNORE_INDEX).astype(np.int64)
    hard = extract_patch_shape(item["hardneg"], center_xyz, fine_shape, 0.0).astype(np.float32)
    pole_soft, pole_boundary = _soft_target_and_boundary(labels, 1, boundary_radius, soft_sigma)
    line_soft, line_boundary = _soft_target_and_boundary(labels, 2, boundary_radius, soft_sigma)
    boundary_ignore = np.maximum(pole_boundary, line_boundary).astype(np.float32)
    return (
        np.stack(fine).astype(np.float32),
        np.stack(coarse).astype(np.float32),
        labels,
        hard,
        boundary_ignore,
        pole_soft,
        line_soft,
    )


def build_v6_inference_features(
    item: Dict, center_xyz: np.ndarray, grid_size: Tuple[int,int,int], patch_size: int,
    context_xy: int, context_z: int, use_coord_channels: bool, use_dist: bool,
):
    """Build only model inputs; avoids label distance transforms during inference."""
    fine_shape=(patch_size,patch_size,patch_size); ctx_shape=(context_z,context_xy,context_xy)
    fine_occ=extract_patch_shape(item["occ"],center_xyz,fine_shape,0.0); fine=[fine_occ]
    if use_coord_channels: fine.extend(_coord_channels(center_xyz,grid_size,fine_shape))
    if use_dist: fine.append(extract_patch_shape(item["dist"],center_xyz,fine_shape,0.0))
    ctx_occ=extract_patch_shape(item["occ"],center_xyz,ctx_shape,0.0); coarse=[_pool_rect(ctx_occ,patch_size,"max")]
    if use_coord_channels: coarse.extend(_coord_channels(center_xyz,grid_size,ctx_shape,output_size=patch_size))
    if use_dist: coarse.append(_pool_rect(extract_patch_shape(item["dist"],center_xyz,ctx_shape,0.0),patch_size,"mean"))
    return np.stack(fine).astype(np.float32), np.stack(coarse).astype(np.float32)


class SpanAwarePatchDataset(Dataset):
    """Balanced category-first sampling with optional FP/FN replay."""

    def __init__(
        self,
        records: List[Dict],
        grid_size,
        replay_dir: Optional[str],
        patch_size=64,
        context_xy=256,
        context_z=128,
        samples_per_epoch=18000,
        use_coord_channels=True,
        use_dist=True,
        pole_pos_prob=0.20,
        line_pos_prob=0.20,
        pole_fp_prob=0.15,
        line_fp_prob=0.15,
        pole_fn_prob=0.15,
        line_fn_prob=0.15,
        edge_asset_prob=0.15,
        edge_width_vox=10,
        cache_items=1,
        deterministic=False,
        seed=42,
        jitter=6,
        replay_radius=3,
        boundary_radius=2.5,
        soft_sigma=1.25,
    ):
        self.records = records
        self.grid_size = tuple(int(v) for v in grid_size)
        self.replay_dir = replay_dir
        self.patch_size = int(patch_size)
        self.context_xy = int(context_xy)
        self.context_z = int(context_z)
        self.samples_per_epoch = int(samples_per_epoch)
        self.use_coord_channels = bool(use_coord_channels)
        self.use_dist = bool(use_dist)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self.jitter = int(jitter)
        self.replay_radius = int(replay_radius)
        self.edge_asset_prob = float(edge_asset_prob)
        self.edge_width_vox = int(edge_width_vox)
        self.boundary_radius = float(boundary_radius)
        self.soft_sigma = float(soft_sigma)
        self.cache = LRUFileCache(max_items=int(cache_items))
        self.replay_cache = LRUFileCache(max_items=max(8, int(cache_items) * 8))
        self.probs = [
            ("pole_pos", float(pole_pos_prob)), ("line_pos", float(line_pos_prob)),
            ("pole_fp", float(pole_fp_prob)), ("line_fp", float(line_fp_prob)),
            ("pole_fn", float(pole_fn_prob)), ("line_fn", float(line_fn_prob)),
        ]
        if sum(v for _, v in self.probs) + self.edge_asset_prob > 1.0 + 1e-9:
            raise ValueError("Sampling probabilities plus edge_asset_prob must sum to <= 1")
        self.by_id = {safe_id(r): r for r in records}
        self.pole_records = [r for r in records if int(r.get("label_counts", {}).get("1", 0)) > 0]
        self.line_records = [r for r in records if int(r.get("label_counts", {}).get("2", 0)) > 0]
        self.replay_records = defaultdict(list)
        index_path = os.path.join(replay_dir, "replay_index.json") if replay_dir else ""
        if index_path and os.path.exists(index_path):
            idx = json.load(open(index_path))
            for row in idx.get("files", []):
                sid = row["id"]
                if sid not in self.by_id:
                    continue
                for name in REPLAY_TYPES:
                    if int(row.get("counts", {}).get(name, 0)) > 0:
                        self.replay_records[name].append(self.by_id[sid])

    def __len__(self):
        return self.samples_per_epoch

    def _rng(self, idx):
        seed = self.seed + int(idx) * 1000003 if self.deterministic else int(np.random.randint(0, 2**32 - 1))
        return random.Random(seed), np.random.default_rng(seed)

    def _load(self, rec):
        p = rec["npz_path"]
        item = self.cache.get(p)
        if item is None:
            item = load_npz_dense(p, self.grid_size, use_dist=self.use_dist)
            self.cache.put(p, item)
        return item

    def _load_replay(self, rec):
        if not self.replay_dir:
            return None
        p = os.path.join(self.replay_dir, "done", safe_id(rec) + ".npz")
        arr = self.replay_cache.get(p)
        if arr is None and os.path.exists(p):
            d = np.load(p)
            arr = {"coords": d["coords"].astype(np.int32), "types": d["types"].astype(np.int8)}
            self.replay_cache.put(p, arr)
        return arr

    def _select(self, idx):
        py, nr = self._rng(idx)
        u, acc, category = py.random(), 0.0, "random"
        if u < self.edge_asset_prob:
            category = "edge_asset"
            acc = self.edge_asset_prob
        for name, p in self.probs:
            if category == "edge_asset":
                break
            acc += p
            if u < acc:
                category = name
                break
        if category == "edge_asset":
            asset_records = self.pole_records + self.line_records
            rec = py.choice(asset_records if asset_records else self.records)
        elif category == "pole_pos" and self.pole_records:
            rec = py.choice(self.pole_records)
        elif category == "line_pos" and self.line_records:
            rec = py.choice(self.line_records)
        elif category in REPLAY_TYPES and self.replay_records.get(category):
            rec = py.choice(self.replay_records[category])
        else:
            category, rec = "random", py.choice(self.records)
        item = self._load(rec)
        replay_type = -1
        if category == "edge_asset":
            labels=item["raw_labels"]; coords=item["coords"]
            assets=np.flatnonzero((labels==1)|(labels==2))
            gx,gy,_=self.grid_size; w=self.edge_width_vox
            if len(assets):
                acoords=coords[assets]
                edge=(acoords[:,0] < w)|(acoords[:,0] >= gx-w)|(acoords[:,1] < w)|(acoords[:,1] >= gy-w)
                choices=assets[edge]
            else:
                choices=np.array([],dtype=np.int64)
            if len(choices): center=coords[int(nr.choice(choices))].astype(np.int64)
            else:
                category="random"; center=coords[int(nr.integers(0,len(coords)))].astype(np.int64)
        elif category == "pole_pos":
            choices = np.flatnonzero(item["raw_labels"] == 1)
            center = item["coords"][int(nr.choice(choices))].astype(np.int64)
        elif category == "line_pos":
            choices = np.flatnonzero(item["raw_labels"] == 2)
            center = item["coords"][int(nr.choice(choices))].astype(np.int64)
        elif category in REPLAY_TYPES:
            replay = self._load_replay(rec)
            t = REPLAY_TYPES[category]
            choices = np.flatnonzero(replay["types"] == t) if replay is not None else np.array([], dtype=np.int64)
            if len(choices):
                center = replay["coords"][int(nr.choice(choices))].astype(np.int64)
                replay_type = t
            else:
                category = "random"
                center = item["coords"][int(nr.integers(0, len(item["coords"])))].astype(np.int64)
        else:
            center = item["coords"][int(nr.integers(0, len(item["coords"])))].astype(np.int64)
        center = center.copy()
        if not self.deterministic and self.jitter > 0 and category in {"pole_pos", "line_pos", "edge_asset", "random"}:
            center += nr.integers(-self.jitter, self.jitter + 1, size=3).astype(np.int64)
        return rec, item, center, replay_type, category

    def __getitem__(self, idx):
        rec, item, center, replay_type, category = self._select(idx)
        fine, coarse, labels, hard, boundary, pole_soft, line_soft = build_v6_features(
            item, center, self.grid_size, self.patch_size, self.context_xy, self.context_z,
            self.use_coord_channels, self.use_dist, self.boundary_radius, self.soft_sigma,
        )
        masks = np.zeros((4,) + hard.shape, dtype=np.float32)
        if replay_type >= 0:
            c, r = self.patch_size // 2, self.replay_radius
            masks[replay_type, max(0,c-r):c+r+1, max(0,c-r):c+r+1, max(0,c-r):c+r+1] = 1.0
        return {
            "x_fine": torch.from_numpy(fine), "x_coarse": torch.from_numpy(coarse),
            "labels": torch.from_numpy(labels), "hardneg": torch.from_numpy(hard),
            "boundary_ignore": torch.from_numpy(boundary),
            "pole_soft": torch.from_numpy(pole_soft), "line_soft": torch.from_numpy(line_soft),
            "replay_masks": torch.from_numpy(masks), "category": category,
            "file_id": safe_id(rec),
        }


class GeometryStem(nn.Module):
    def forward(self, x):
        occ = x[:, :1]
        local = F.avg_pool3d(occ, 3, 1, 1)
        vertical = F.avg_pool3d(occ, (15, 3, 3), 1, (7, 1, 1))
        along_y = F.avg_pool3d(occ, (3, 15, 3), 1, (1, 7, 1))
        along_x = F.avg_pool3d(occ, (3, 3, 15), 1, (1, 1, 7))
        horizontal = torch.maximum(along_x, along_y)
        orient_delta = vertical - horizontal
        return torch.cat([x, local, vertical, along_y, along_x, horizontal, orient_delta], dim=1)


class IdentityFuse(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels * 2, channels, 1, bias=False)
        with torch.no_grad():
            self.conv.weight.zero_()
            for i in range(channels):
                self.conv.weight[i, i, 0, 0, 0] = 0.5
                self.conv.weight[i, channels + i, 0, 0, 0] = 0.5
    def forward(self, a, b):
        return self.conv(torch.cat([a, b], dim=1))


class SpanAwareGeoNet3D(nn.Module):
    def __init__(self, in_ch=5, base=16, emb_dim=8):
        super().__init__()
        self.input_channels = int(in_ch)
        self.geo = GeometryStem()
        geo_ch = in_ch + 6
        self.fine_encoder = Encoder3D(geo_ch, base)
        self.coarse_encoder = Encoder3D(geo_ch, base)
        self.fuse1, self.fuse2, self.fuse3 = IdentityFuse(base), IdentityFuse(base*2), IdentityFuse(base*4)
        self.up2 = UpBlock3D(base*4, base*2, base*2)
        self.up1 = UpBlock3D(base*2, base, base)
        self.semantic_head = nn.Conv3d(base, NUM_CLASSES, 1)
        self.pole_head = nn.Conv3d(base, 1, 1)
        self.line_head = nn.Conv3d(base, 1, 1)
        self.objectness_head = nn.Conv3d(base, 1, 1)
        self.verticality_head = nn.Conv3d(base, 1, 1)
        self.horizontality_head = nn.Conv3d(base, 1, 1)
        self.embedding_head = nn.Conv3d(base, emb_dim, 1)

    def forward(self, x_fine, x_coarse):
        f1, f2, f3 = self.fine_encoder(self.geo(x_fine))
        c1, c2, c3 = self.coarse_encoder(self.geo(x_coarse))
        s1, s2, s3 = self.fuse1(f1,c1), self.fuse2(f2,c2), self.fuse3(f3,c3)
        y = self.up1(self.up2(s3, s2), s1)
        return {
            "semantic": self.semantic_head(y), "pole": self.pole_head(y), "line": self.line_head(y),
            "objectness": self.objectness_head(y), "verticality": self.verticality_head(y),
            "horizontality": self.horizontality_head(y), "embedding": self.embedding_head(y),
        }


def load_from_v4_checkpoint(model: SpanAwareGeoNet3D, checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    old = ckpt.get("model_state", ckpt)
    new = model.state_dict()
    copied = []
    def copy(dst, src, adapt=False):
        if src not in old or dst not in new:
            return
        a, b = old[src], new[dst]
        if a.shape == b.shape:
            b.copy_(a); copied.append(dst)
        elif adapt and a.ndim == 5 and b.ndim == 5 and a.shape[0] == b.shape[0]:
            b.zero_(); n = min(a.shape[1], b.shape[1]); b[:, :n].copy_(a[:, :n]); copied.append(dst)
    for branch in ("fine_encoder", "coarse_encoder"):
        for k in list(new):
            if k.startswith(branch + "."):
                suffix = k[len(branch)+1:]
                copy(k, "encoder." + suffix, adapt=suffix == "e1.net.0.weight")
    for prefix in ("up2", "up1", "semantic_head", "pole_head", "line_head", "objectness_head", "embedding_head"):
        for k in list(new):
            if k.startswith(prefix + "."):
                copy(k, k)
    model.load_state_dict(new)
    return {"copied_tensors": len(copied), "source": checkpoint_path}


def _soft_iou(logits, targets, mask, smooth=1.0):
    p = torch.sigmoid(logits) * mask
    t = targets * mask
    dims = tuple(range(2, p.ndim))
    inter = (p*t).sum(dims); union = (p+t-p*t).sum(dims)
    return 1.0 - ((inter+smooth)/(union+smooth)).mean()


def _asym_bce(logits, targets, mask, weights, alpha_pos=.58, gamma_pos=1., gamma_neg=2.5):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits); pos = targets > .5
    focal = torch.where(pos, (1-p).clamp_min(1e-6)**gamma_pos, p.clamp_min(1e-6)**gamma_neg)
    alpha = torch.where(pos, torch.full_like(targets, alpha_pos), torch.full_like(targets, 1-alpha_pos))
    loss = bce*focal*alpha*weights*mask
    return loss.sum() / (weights*mask).sum().clamp_min(1.0)


def directional_supports_3d(volume: torch.Tensor, length: int = 11):
    """Rotation-friendlier local line supports.

    Returns (vertical_support, horizontal_support).  Horizontal support is the
    strongest response over X, Y, +45 degree, and -45 degree XY directions.
    The small max-pools tolerate one-voxel Z/XY drift without using world data.
    """
    if volume.ndim != 5 or volume.shape[1] != 1:
        raise ValueError("directional_supports_3d expects [B,1,D,H,W]")
    length = int(length)
    if length < 3 or length % 2 == 0:
        raise ValueError("length must be odd and >= 3")
    r = length // 2

    # Tolerate small Z changes first, then measure continuity in XY.
    hsrc = F.max_pool3d(volume, kernel_size=(3,1,1), stride=1, padding=(1,0,0))
    k = torch.zeros((4,1,1,length,length), device=volume.device, dtype=volume.dtype)
    c = r
    for i in range(length):
        k[0,0,0,c,i] = 1.0 / length              # X
        k[1,0,0,i,c] = 1.0 / length              # Y
        k[2,0,0,i,i] = 1.0 / length              # +45 deg
        k[3,0,0,i,length-1-i] = 1.0 / length     # -45 deg
    h = F.conv3d(hsrc, k, padding=(0,r,r)).amax(dim=1, keepdim=True)

    # Tolerate a small XY footprint before measuring vertical continuity.
    vsrc = F.max_pool3d(volume, kernel_size=(1,3,3), stride=1, padding=(0,1,1))
    kv = torch.ones((1,1,length,1,1), device=volume.device, dtype=volume.dtype) / length
    v = F.conv3d(vsrc, kv, padding=(r,0,0))
    return v.clamp(0,1), h.clamp(0,1)


def _positive_v4_line_teacher_loss(
    out, labels, valid_mask, teacher_pole_score, teacher_line_score,
    horizontal_support, vertical_support, args
):
    """Positive-only V4 distillation for GT-background voxels.

    V4 is used only when it strongly supports a line and does not simultaneously
    support a pole.  There is deliberately no negative teacher loss: V4 false
    positives are not copied into V6.2 as background supervision.
    """
    if teacher_line_score is None:
        z = out["line"].sum() * 0.0
        return z, {"teacher_positive_fraction": 0.0, "teacher_score_mean": 0.0}

    tline = teacher_line_score.detach().float()
    tpole = teacher_pole_score.detach().float() if teacher_pole_score is not None else torch.zeros_like(tline)
    background = labels == 0
    # Reject only strongly vertical local geometry.  Horizontal support includes
    # diagonal directions, so diagonal conductors are not penalized as in the
    # older X/Y-only physics term.
    not_strongly_vertical = vertical_support.squeeze(1) <= (horizontal_support.squeeze(1) + float(args.v4_teacher_vertical_margin))
    mask = (
        valid_mask
        & background
        & (tline >= float(args.v4_teacher_min_line_score))
        & (tline >= tpole + float(args.v4_teacher_line_over_pole_margin))
        & not_strongly_vertical
    )
    if not torch.any(mask):
        z = out["line"].sum() * 0.0
        return z, {"teacher_positive_fraction": 0.0, "teacher_score_mean": 0.0}

    w = ((tline - float(args.v4_teacher_min_line_score)) / max(1e-6, 1.0 - float(args.v4_teacher_min_line_score))).clamp(0,1)
    w = (0.25 + 0.75 * w).detach()
    m = mask.float()
    target = tline.clamp(float(args.v4_teacher_min_line_score), 0.995)

    line_bce = F.binary_cross_entropy_with_logits(out["line"].squeeze(1).float(), target, reduction="none")
    sem_logp = F.log_softmax(out["semantic"].float(), dim=1)[:,2]
    sem_nll = -sem_logp
    obj_bce = F.binary_cross_entropy_with_logits(out["objectness"].squeeze(1).float(), target, reduction="none")
    denom = (w*m).sum().clamp_min(1.0)
    loss = ((line_bce + 0.50*sem_nll + 0.25*obj_bce) * w * m).sum() / (1.75 * denom)
    frac = float(mask.float().mean().detach())
    mean_score = float(tline[mask].mean().detach())
    return loss, {"teacher_positive_fraction": frac, "teacher_score_mean": mean_score}


def v6_loss(out, x_fine, labels, hard, boundary_ignore, pole_soft, line_soft, replay_masks, args,
            teacher_pole_score=None, teacher_line_score=None):
    valid = labels != IGNORE_INDEX
    semantic_mask = valid & (boundary_ignore < .5)
    pole_fp, line_fp, pole_fn, line_fn = [replay_masks[:, i:i+1] for i in range(4)]
    replay_any = replay_masks.max(dim=1).values
    base_w = 1.0 + hard*(args.hardneg_weight-1.0)
    # Label-noise tolerance: physically coherent horizontal/vertical background can be
    # a real asset omitted or displaced by catenary-derived GT. Downweight, never
    # promote, those background negatives. World continuity decides weak positives
    # later in the component refiner.
    occ=x_fine[:,0:1]
    v_occ,h_occ=directional_supports_3d(occ,length=int(getattr(args,"orientation_support_length",11)))
    pole_like=F.relu(v_occ-h_occ-args.orientation_margin).squeeze(1).clamp(0,1)
    line_like=F.relu(h_occ-v_occ-args.orientation_margin).squeeze(1).clamp(0,1)
    background=(labels==0).float()
    ambiguous=(torch.maximum(pole_like,line_like)*background).clamp(0,1)
    noise_weight=(1.0-args.geometry_unlabeled_discount*ambiguous).clamp(min=.05)
    sem_w = base_w*(1.0+replay_any*(args.replay_weight-1.0))*noise_weight
    cw = torch.tensor([args.class_weight_other,args.class_weight_pole,args.class_weight_line],device=labels.device)
    ce = F.cross_entropy(out["semantic"], labels, weight=cw, ignore_index=IGNORE_INDEX, reduction="none")
    sem = (ce*sem_w*semantic_mask.float()).sum()/(sem_w*semantic_mask.float()).sum().clamp_min(1.)

    mask = valid.float().unsqueeze(1); exact_mask = semantic_mask.float().unsqueeze(1)
    pole_t=(labels==1).float().unsqueeze(1); line_t=(labels==2).float().unsqueeze(1)
    obj_t=((labels==1)|(labels==2)).float().unsqueeze(1); bw=base_w.unsqueeze(1)
    pole_neg_noise=(1.0-args.geometry_unlabeled_discount*pole_like.unsqueeze(1)*(1-pole_t)).clamp(min=.05)
    line_neg_noise=(1.0-args.geometry_unlabeled_discount*line_like.unsqueeze(1)*(1-line_t)).clamp(min=.05)
    pole_w=bw*(1+pole_fp*(args.replay_weight-1)+pole_fn*(args.replay_weight-1))*pole_neg_noise
    line_w=bw*(1+line_fp*(args.replay_weight-1)+line_fn*(args.replay_weight-1))*line_neg_noise
    pole_bce=_asym_bce(out["pole"],pole_t,exact_mask,pole_w,args.alpha_pos,args.gamma_pos,args.gamma_neg)
    line_bce=_asym_bce(out["line"],line_t,exact_mask,line_w,args.alpha_pos,args.gamma_pos,args.gamma_neg)
    obj_bce=_asym_bce(out["objectness"],obj_t,exact_mask,bw,.55,1.,2.5)
    pole_iou=_soft_iou(out["pole"],pole_soft.unsqueeze(1),mask)
    line_iou=_soft_iou(out["line"],line_soft.unsqueeze(1),mask)

    # Orientation supervision is exact on labelled assets and lightly negative on nearby valid background.
    # Plausible unlabeled assets are not reliable negatives. Supervise labelled assets fully,
    # but downweight orientation-negative background when local occupancy already looks asset-like.
    orient_weight=(1.0-args.geometry_unlabeled_discount*ambiguous).clamp(min=.05).unsqueeze(1)*mask
    orient_weight=torch.maximum(orient_weight, (pole_t+line_t).clamp(max=1.0))
    vertical_bce = F.binary_cross_entropy_with_logits(out["verticality"], pole_t, reduction="none")
    horizontal_bce = F.binary_cross_entropy_with_logits(out["horizontality"], line_t, reduction="none")
    orientation = ((vertical_bce + horizontal_bce)*orient_weight).sum()/(2*orient_weight.sum().clamp_min(1.))

    pprob=torch.sigmoid(out["pole"]); lprob=torch.sigmoid(out["line"])
    fp_replay=(pprob*pole_fp+lprob*line_fp).sum()/(pole_fp+line_fp).sum().clamp_min(1.)
    fn_replay=((1-pprob)*pole_fn+(1-lprob)*line_fn).sum()/(pole_fn+line_fn).sum().clamp_min(1.)
    cross=((lprob*pole_t+pprob*line_t)*mask).sum()/((pole_t+line_t)*mask).sum().clamp_min(1.)

    line_vertical=(lprob*F.relu(v_occ-h_occ-args.orientation_margin)*mask).sum()/mask.sum().clamp_min(1.)
    pole_horizontal=(pprob*F.relu(h_occ-v_occ-args.orientation_margin)*mask).sum()/mask.sum().clamp_min(1.)
    p_smooth=F.avg_pool3d(pprob,(9,3,3),1,(4,1,1))
    _,l_smooth=directional_supports_3d(lprob,length=int(getattr(args,"orientation_support_length",11)))
    pole_cont=((pprob-p_smooth).abs()*pole_t).sum()/pole_t.sum().clamp_min(1.)
    line_cont=((lprob-l_smooth).abs()*line_t).sum()/line_t.sum().clamp_min(1.)
    physics=line_vertical+pole_horizontal+0.5*(pole_cont+line_cont)

    teacher_line,teacher_diag=_positive_v4_line_teacher_loss(
        out,labels,valid,teacher_pole_score,teacher_line_score,h_occ,v_occ,args
    )

    total=(args.lambda_sem*sem + args.lambda_binary*(pole_bce+line_bce) + args.lambda_iou*(pole_iou+line_iou)
           +args.lambda_objectness*obj_bce + args.lambda_replay_fp*fp_replay + args.lambda_replay_fn*fn_replay
           +args.lambda_cross_class*cross + args.lambda_orientation*orientation + args.lambda_physics*physics
           +args.lambda_v4_line_teacher*teacher_line)
    parts={k:float(v.detach()) for k,v in {
        "sem":sem,"pole_bce":pole_bce,"line_bce":line_bce,"pole_iou_loss":pole_iou,"line_iou_loss":line_iou,
        "objectness":obj_bce,"replay_fp":fp_replay,"replay_fn":fn_replay,"cross":cross,"orientation":orientation,
        "line_vertical_penalty":line_vertical,"pole_horizontal_penalty":pole_horizontal,
        "pole_continuity":pole_cont,"line_continuity":line_cont,"physics":physics,
        "teacher_line_distill":teacher_line}.items()}
    parts.update(teacher_diag)
    return total, parts
