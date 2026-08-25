import os
import json
import math
import time
import random
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


IGNORE_INDEX = -100
NUM_CLASSES = 3  # 0 background/other occupied structure, 1 pole, 2 powerline


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def write_json(obj, path: str) -> None:
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def class_counts_from_records(records: List[Dict]) -> Dict[str, int]:
    counts = {"0": 0, "1": 0, "2": 0}
    for r in records:
        for k in counts:
            counts[k] += int(r.get('label_counts', {}).get(k, 0))
    return counts


def compute_class_weights(counts: Dict[str, int], max_weight: float = 25.0) -> torch.Tensor:
    vals = np.array([max(1, counts.get(str(i), 0)) for i in range(NUM_CLASSES)], dtype=np.float64)
    inv = vals.sum() / (NUM_CLASSES * vals)
    inv = np.clip(inv, 1.0, max_weight)
    # Poles and lines are safety-critical; bias them above ordinary inverse-frequency weights.
    inv[1] = max(inv[1], 8.0)
    inv[2] = max(inv[2], 8.0)
    return torch.tensor(inv, dtype=torch.float32)


def focal_bce_with_logits(logits, targets, mask=None, alpha=0.75, gamma=2.0, weights=None):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    probs = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, probs, 1.0 - probs)
    alpha_t = torch.where(targets > 0.5, torch.full_like(targets, alpha), torch.full_like(targets, 1.0 - alpha))
    loss = alpha_t * ((1.0 - pt).clamp(min=1e-6) ** gamma) * bce
    if weights is not None:
        loss = loss * weights
    if mask is not None:
        loss = loss * mask
        denom = mask.sum().clamp(min=1.0)
        return loss.sum() / denom
    return loss.mean()


def weighted_ce(logits, labels, class_weights, voxel_weights=None, ignore_index=IGNORE_INDEX):
    loss = F.cross_entropy(logits, labels.long(), weight=class_weights.to(logits.device), ignore_index=ignore_index, reduction='none')
    valid = (labels != ignore_index).float()
    if voxel_weights is not None:
        loss = loss * voxel_weights
        valid = valid * voxel_weights.clamp(min=0.0)
    return (loss * (labels != ignore_index).float()).sum() / valid.sum().clamp(min=1.0)


def prototype_contrastive_loss(emb, labels, max_samples=4096, margin=1.5, ignore_index=IGNORE_INDEX):
    """Simple class-prototype loss over sampled occupied voxels.

    This does not label trees/buildings/streetlights. It gives the model an embedding
    space where pole and line voxels separate from hard background voxels. Hard
    negatives are upweighted by the semantic/focal losses, then this head makes
    those background embeddings less pole-like.
    """
    # emb: [B,E,D,H,W], labels: [B,D,H,W]
    B, E = emb.shape[:2]
    emb_flat = emb.permute(0, 2, 3, 4, 1).reshape(-1, E)
    lab_flat = labels.reshape(-1)
    valid = lab_flat != ignore_index
    idx = valid.nonzero(as_tuple=False).flatten()
    if idx.numel() < 32:
        return emb.sum() * 0.0
    if idx.numel() > max_samples:
        perm = torch.randperm(idx.numel(), device=idx.device)[:max_samples]
        idx = idx[perm]
    e = F.normalize(emb_flat[idx], dim=1)
    y = lab_flat[idx]
    centers = []
    present = []
    intra = 0.0
    n_present = 0
    for c in range(NUM_CLASSES):
        m = y == c
        if m.sum() >= 4:
            center = F.normalize(e[m].mean(dim=0, keepdim=True), dim=1).squeeze(0)
            centers.append(center)
            present.append(c)
            intra = intra + ((e[m] - center) ** 2).sum(dim=1).mean()
            n_present += 1
    if n_present == 0:
        return emb.sum() * 0.0
    loss = intra / n_present
    if len(centers) > 1:
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                d = torch.norm(centers[i] - centers[j], p=2)
                loss = loss + F.relu(margin - d) ** 2
    return loss


class LRUFileCache:
    def __init__(self, max_items=8):
        self.max_items = max_items
        self.cache = OrderedDict()

    def get(self, path):
        if path in self.cache:
            val = self.cache.pop(path)
            self.cache[path] = val
            return val
        return None

    def put(self, path, val):
        self.cache[path] = val
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)


def load_npz_dense(npz_path: str, grid_size: Tuple[int, int, int], use_dist: bool = True):
    """Return dense occupancy/features/labels/hardneg arrays.

    grid_size is (X,Y,Z). Dense tensors are stored as [Z,Y,X].
    """
    data = np.load(npz_path)
    coords = data['coords'].astype(np.int32)  # [N,3] x,y,z
    labels = data['labels'].astype(np.int16)
    dist = data['dist'].astype(np.float32) if ('dist' in data and use_dist) else None
    hardneg = data['hardneg'].astype(np.uint8) if 'hardneg' in data else np.zeros(len(labels), dtype=np.uint8)

    gx, gy, gz = grid_size
    occ = np.zeros((gz, gy, gx), dtype=np.float32)
    lab = np.full((gz, gy, gx), IGNORE_INDEX, dtype=np.int16)
    hard = np.zeros((gz, gy, gx), dtype=np.float32)
    dist_grid = np.zeros((gz, gy, gx), dtype=np.float32)

    x = np.clip(coords[:, 0], 0, gx - 1)
    y = np.clip(coords[:, 1], 0, gy - 1)
    z = np.clip(coords[:, 2], 0, gz - 1)
    occ[z, y, x] = 1.0
    lab[z, y, x] = labels
    hard[z, y, x] = hardneg.astype(np.float32)
    if dist is not None:
        dmax = max(float(np.nanmax(np.abs(dist))), 1.0)
        dist_grid[z, y, x] = np.nan_to_num(dist / dmax, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        'occ': occ,
        'labels': lab,
        'hardneg': hard,
        'dist': dist_grid,
        'coords': coords,
        'raw_labels': labels,
        'raw_hardneg': hardneg,
    }



def load_npz_sparse(npz_path: str, grid_size: Tuple[int, int, int], use_dist: bool = True):
    """Load a prepared slice without expanding it to a full dense volume.

    The returned arrays preserve the dense loader's semantics, including
    per-file ``dist`` normalization and "last row wins" behavior for duplicate
    voxel coordinates.  Coordinates are sorted by Z so patch extraction can
    narrow the candidate rows with ``searchsorted`` before applying XY bounds.
    """
    with np.load(npz_path) as data:
        coords = data['coords'].astype(np.int32, copy=False)
        labels = data['labels'].astype(np.int16, copy=False)
        dist = data['dist'].astype(np.float32, copy=False) if ('dist' in data and use_dist) else None
        hardneg = data['hardneg'].astype(np.uint8, copy=False) if 'hardneg' in data else np.zeros(len(labels), dtype=np.uint8)

    gx, gy, gz = [int(v) for v in grid_size]
    valid = (
        (coords[:, 0] >= 0) & (coords[:, 0] < gx) &
        (coords[:, 1] >= 0) & (coords[:, 1] < gy) &
        (coords[:, 2] >= 0) & (coords[:, 2] < gz)
    )
    coords = coords[valid]
    labels = labels[valid]
    hardneg = hardneg[valid]
    if dist is not None:
        dist = dist[valid]

    # Dense assignment in load_npz_dense() keeps the last value at duplicate
    # coordinates.  Reproduce that exactly before sorting for sparse lookup.
    if len(coords):
        linear = ((coords[:, 2].astype(np.int64) * gy + coords[:, 1]) * gx + coords[:, 0])
        rev = linear[::-1]
        _, rev_first = np.unique(rev, return_index=True)
        keep = np.sort(len(linear) - 1 - rev_first)
        duplicate_rows_dropped = int(len(coords) - len(keep))
        coords = coords[keep]
        labels = labels[keep]
        hardneg = hardneg[keep]
        if dist is not None:
            dist = dist[keep]
    else:
        duplicate_rows_dropped = 0

    if dist is None:
        dist_norm = np.zeros(len(coords), dtype=np.float32)
    else:
        dmax = max(float(np.nanmax(np.abs(dist))) if len(dist) else 0.0, 1.0)
        dist_norm = np.nan_to_num(dist / dmax, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    order = np.argsort(coords[:, 2], kind='stable') if len(coords) else np.zeros(0, dtype=np.int64)
    coords = coords[order]
    labels = labels[order]
    hardneg = hardneg[order]
    dist_norm = dist_norm[order]
    z_sorted = coords[:, 2] if len(coords) else np.zeros(0, dtype=np.int32)
    return {
        'coords': coords,
        'raw_labels': labels,
        'raw_hardneg': hardneg,
        'dist_values': dist_norm,
        'z_sorted': z_sorted,
        'duplicate_rows_dropped': duplicate_rows_dropped,
        'sparse_native': True,
    }


def sparse_rows_in_box(item: Dict, start_xyz, shape_zyx):
    """Return row indices whose x/y/z coordinates fall in a rectangular box."""
    x0, y0, z0 = [int(v) for v in start_xyz]
    pz, py, px = [int(v) for v in shape_zyx]
    x1, y1, z1 = x0 + px, y0 + py, z0 + pz
    zvals = item['z_sorted']
    lo = int(np.searchsorted(zvals, z0, side='left'))
    hi = int(np.searchsorted(zvals, z1, side='left'))
    if hi <= lo:
        return np.zeros(0, dtype=np.int64)
    c = item['coords'][lo:hi]
    m = (c[:, 0] >= x0) & (c[:, 0] < x1) & (c[:, 1] >= y0) & (c[:, 1] < y1)
    return np.flatnonzero(m).astype(np.int64) + lo


def extract_patch(arr, center_xyz, patch_size, fill_value=0):
    """Extract [Z,Y,X] patch centered on x,y,z; pad as needed."""
    x, y, z = [int(v) for v in center_xyz]
    P = patch_size
    r0 = P // 2
    z0, z1 = z - r0, z - r0 + P
    y0, y1 = y - r0, y - r0 + P
    x0, x1 = x - r0, x - r0 + P
    out = np.full((P, P, P), fill_value, dtype=arr.dtype)
    sz, sy, sx = arr.shape
    az0, az1 = max(0, z0), min(sz, z1)
    ay0, ay1 = max(0, y0), min(sy, y1)
    ax0, ax1 = max(0, x0), min(sx, x1)
    oz0, oy0, ox0 = az0 - z0, ay0 - y0, ax0 - x0
    out[oz0:oz0 + (az1 - az0), oy0:oy0 + (ay1 - ay0), ox0:ox0 + (ax1 - ax0)] = arr[az0:az1, ay0:ay1, ax0:ax1]
    return out


def make_coord_channels(center_xyz, grid_size, patch_size):
    gx, gy, gz = grid_size
    x, y, z = [int(v) for v in center_xyz]
    P = patch_size
    r0 = P // 2
    xs = np.arange(x - r0, x - r0 + P, dtype=np.float32)
    ys = np.arange(y - r0, y - r0 + P, dtype=np.float32)
    zs = np.arange(z - r0, z - r0 + P, dtype=np.float32)
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing='ij')
    xch = np.clip((xx / max(gx - 1, 1)) * 2.0 - 1.0, -1.5, 1.5)
    ych = np.clip((yy / max(gy - 1, 1)) * 2.0 - 1.0, -1.5, 1.5)
    zch = np.clip((zz / max(gz - 1, 1)) * 2.0 - 1.0, -1.5, 1.5)
    return xch.astype(np.float32), ych.astype(np.float32), zch.astype(np.float32)


class VoxelPatchDataset(Dataset):
    def __init__(self, records, grid_size, patch_size=64, samples_per_epoch=2000,
                 mode='seg', use_coord_channels=True, use_dist=True,
                 pos_pole_prob=0.35, pos_line_prob=0.25, hardneg_prob=0.25,
                 cache_items=4):
        self.records = records
        self.grid_size = tuple(grid_size)
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.mode = mode
        self.use_coord_channels = bool(use_coord_channels)
        self.use_dist = bool(use_dist)
        self.pos_pole_prob = pos_pole_prob
        self.pos_line_prob = pos_line_prob
        self.hardneg_prob = hardneg_prob
        self.cache = LRUFileCache(max_items=cache_items)

    def __len__(self):
        return self.samples_per_epoch

    def _load(self, rec):
        path = rec['npz_path']
        item = self.cache.get(path)
        if item is None:
            item = load_npz_dense(path, self.grid_size, use_dist=self.use_dist)
            self.cache.put(path, item)
        return item

    def _choose_record_and_center(self):
        rec = random.choice(self.records)
        item = self._load(rec)
        coords = item['coords']
        labels = item['raw_labels']
        hardneg = item.get('hardneg')
        r = random.random()
        choices = None
        if self.mode == 'seg':
            if r < self.pos_pole_prob and np.any(labels == 1):
                choices = np.where(labels == 1)[0]
            elif r < self.pos_pole_prob + self.pos_line_prob and np.any(labels == 2):
                choices = np.where(labels == 2)[0]
            elif r < self.pos_pole_prob + self.pos_line_prob + self.hardneg_prob and hardneg is not None:
                raw_hard = item.get('raw_hardneg')
                if raw_hard is not None and np.any(raw_hard > 0):
                    choices = np.where(raw_hard > 0)[0]
        if choices is None or len(choices) == 0:
            choices = np.arange(coords.shape[0])
        idx = int(np.random.choice(choices))
        return rec, item, coords[idx]

    def __getitem__(self, idx):
        rec, item, center = self._choose_record_and_center()
        occ = extract_patch(item['occ'], center, self.patch_size, fill_value=0.0)
        labels = extract_patch(item['labels'], center, self.patch_size, fill_value=IGNORE_INDEX)
        hard = extract_patch(item['hardneg'], center, self.patch_size, fill_value=0.0)
        feats = [occ]
        if self.use_coord_channels:
            xch, ych, zch = make_coord_channels(center, self.grid_size, self.patch_size)
            feats.extend([xch, ych, zch])
        if self.use_dist:
            dist = extract_patch(item['dist'], center, self.patch_size, fill_value=0.0)
            feats.append(dist)
        x = np.stack(feats, axis=0).astype(np.float32)
        return {
            'x': torch.from_numpy(x),
            'labels': torch.from_numpy(labels.astype(np.int64)),
            'hardneg': torch.from_numpy(hard.astype(np.float32)),
            'center': torch.tensor(center.astype(np.int64)),
            'file_id': rec.get('id', '')
        }


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.SiLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class Encoder3D(nn.Module):
    def __init__(self, in_ch=5, base=16):
        super().__init__()
        self.e1 = ConvBlock3D(in_ch, base)
        self.e2 = ConvBlock3D(base, base * 2)
        self.e3 = ConvBlock3D(base * 2, base * 4)
        self.pool = nn.MaxPool3d(2)
    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        return s1, s2, s3


class UpBlock3D(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock3D(out_ch + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        # Crop/pad to skip shape if needed.
        dz = skip.shape[2] - x.shape[2]
        dy = skip.shape[3] - x.shape[3]
        dx = skip.shape[4] - x.shape[4]
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2, dz // 2, dz - dz // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class MaskedOccupancyAutoencoder3D(nn.Module):
    def __init__(self, in_ch=5, base=16):
        super().__init__()
        self.encoder = Encoder3D(in_ch, base)
        self.up2 = UpBlock3D(base * 4, base * 2, base * 2)
        self.up1 = UpBlock3D(base * 2, base, base)
        self.out = nn.Conv3d(base, 1, 1)
    def forward(self, x):
        s1, s2, s3 = self.encoder(x)
        y = self.up2(s3, s2)
        y = self.up1(y, s1)
        return self.out(y)


class MultiHeadVoxelNet3D(nn.Module):
    """Better-output-heads model for 0/1/2 labels.

    Heads:
    - semantic: 3 classes 0 other/occupied background, 1 pole, 2 powerline
    - pole: binary pole-vs-rest, heavily weighted for recall
    - line: binary powerline-vs-rest
    - objectness: labelled utility asset objectness, label in {1,2}
    - embedding: auxiliary representation used to separate poles/lines from hard negatives
    """
    def __init__(self, in_ch=5, base=16, emb_dim=8):
        super().__init__()
        self.encoder = Encoder3D(in_ch, base)
        self.up2 = UpBlock3D(base * 4, base * 2, base * 2)
        self.up1 = UpBlock3D(base * 2, base, base)
        self.semantic_head = nn.Conv3d(base, NUM_CLASSES, 1)
        self.pole_head = nn.Conv3d(base, 1, 1)
        self.line_head = nn.Conv3d(base, 1, 1)
        self.objectness_head = nn.Conv3d(base, 1, 1)
        self.embedding_head = nn.Conv3d(base, emb_dim, 1)
    def forward(self, x):
        s1, s2, s3 = self.encoder(x)
        y = self.up2(s3, s2)
        y = self.up1(y, s1)
        return {
            'semantic': self.semantic_head(y),
            'pole': self.pole_head(y),
            'line': self.line_head(y),
            'objectness': self.objectness_head(y),
            'embedding': self.embedding_head(y),
        }


def copy_encoder_weights_from_mae(seg_model, mae_state: Dict):
    # Supports checkpoints saved as {'model_state': ...} or raw state dict.
    state = mae_state.get('model_state', mae_state)
    enc = {k.replace('encoder.', ''): v for k, v in state.items() if k.startswith('encoder.')}
    if not enc:
        return False
    missing, unexpected = seg_model.encoder.load_state_dict(enc, strict=False)
    return True


def get_model_state_dict(model):
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()


def build_input_channels(use_coord_channels=True, use_dist=True):
    return 1 + (3 if use_coord_channels else 0) + (1 if use_dist else 0)


def make_masked_input(x, mask_ratio=0.6):
    """Mask occupancy channel only. Keep coordinate/dist channels visible."""
    x_masked = x.clone()
    B, C, D, H, W = x.shape
    occ = x[:, :1]
    # Prefer masking occupied voxels plus a random background sample.
    rand = torch.rand((B, 1, D, H, W), device=x.device)
    mask = rand < mask_ratio
    x_masked[:, :1] = occ.masked_fill(mask, 0.0)
    return x_masked, mask.float()


def patch_to_points_prediction(df, logits_np, grid_size, patch_size, center_xyz, use_yxz=False):
    # Reserved for future row-level patch stitching.
    raise NotImplementedError


def dense_occupancy_from_sparse(item: Dict, grid_size: Tuple[int, int, int], dtype=np.uint8):
    """Create only the dense occupancy mask required by component labeling."""
    gx, gy, gz = [int(v) for v in grid_size]
    occ = np.zeros((gz, gy, gx), dtype=dtype)
    c = item.get('coords')
    if c is not None and len(c):
        occ[c[:, 2], c[:, 1], c[:, 0]] = 1
    return occ
