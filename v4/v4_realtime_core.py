#!/usr/bin/env python3
"""V4 realtime Stage-1 utilities.

Stage 1 is strictly slice-local.  The model sees only local x/y/z occupancy,
local normalized coordinate channels, and local dist_center_ft when present.
Session/world center metadata is never fed into the network.

The optimized inference path preserves the V4 64^3 patch / 48^3 core math but
only evaluates cores containing at least one occupied source voxel and gathers
scores only at occupied rows before D2H transfer.
"""
from __future__ import annotations

import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from precision_common import fuse_scores, maybe_compile_model
from voxel_common import MultiHeadVoxelNet3D, build_input_channels, sparse_rows_in_box, make_coord_channels

IGNORE_INDEX = -100


def setup_torch() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def autocast_ctx(amp: str):
    if amp == "none" or not torch.cuda.is_available():
        return nullcontext()
    if amp == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cuda", dtype=torch.float16)


def _dedupe_last(coords, labels, dist_values, row_source_indices, gx, gy):
    if not len(coords):
        return coords, labels, dist_values, row_source_indices
    lin = ((coords[:, 2].astype(np.int64) * int(gy) + coords[:, 1]) * int(gx) + coords[:, 0])
    _, rev_first = np.unique(lin[::-1], return_index=True)
    keep = np.sort(len(lin) - 1 - rev_first)
    return coords[keep], labels[keep], dist_values[keep], row_source_indices[keep]


def build_sparse_item_from_dataframe(df: pd.DataFrame, grid_size=(400, 400, 200)) -> Dict:
    """Build the sparse V4 input from one slice.

    Only x/y/z, optional label, and optional dist_center_ft participate in Stage 1.
    center_x/center_y/center_z are intentionally ignored here and extracted
    separately for Stage 3.
    """
    gx, gy, gz = map(int, grid_size)
    required = {"x", "y", "z"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"V4 Stage 1 missing required local columns: {missing}")

    xyz = df[["x", "y", "z"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    finite = np.isfinite(xyz).all(axis=1)
    row_coords = np.zeros((len(df), 3), np.int32)
    row_coords[finite] = np.rint(xyz[finite]).astype(np.int32)
    valid = finite & (
        (row_coords[:, 0] >= 0) & (row_coords[:, 0] < gx) &
        (row_coords[:, 1] >= 0) & (row_coords[:, 1] < gy) &
        (row_coords[:, 2] >= 0) & (row_coords[:, 2] < gz)
    )

    row_labels = np.full(len(df), IGNORE_INDEX, np.int16)
    has_gt = "label" in df.columns
    if has_gt:
        y = pd.to_numeric(df["label"], errors="coerce").to_numpy(float)
        good = valid & np.isfinite(y) & np.isin(y, [0, 1, 2])
        row_labels[good] = y[good].astype(np.int16)

    source_rows = np.flatnonzero(valid).astype(np.int64)
    coords = row_coords[valid].copy()
    labels = row_labels[valid].copy() if has_gt else np.zeros(len(coords), np.int16)

    dist_values = np.zeros(len(coords), np.float32)
    if "dist_center_ft" in df.columns:
        d = pd.to_numeric(df["dist_center_ft"], errors="coerce").to_numpy(float)
        d_valid = d[valid]
        dmax = max(float(np.nanmax(np.abs(d_valid))) if len(d_valid) else 0.0, 1.0)
        dist_values = np.nan_to_num(d_valid / dmax, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    coords, labels, dist_values, source_rows = _dedupe_last(
        coords, labels, dist_values, source_rows, gx, gy
    )
    if len(coords):
        order = np.argsort(coords[:, 2], kind="stable")
        coords = coords[order]
        labels = labels[order]
        dist_values = dist_values[order]
        source_rows = source_rows[order]
    else:
        order = np.zeros(0, np.int64)

    return {
        "sparse_native": True,
        "coords": coords,
        "raw_labels": labels,
        "raw_hardneg": np.zeros(len(coords), np.uint8),
        "dist_values": dist_values,
        "z_sorted": coords[:, 2] if len(coords) else np.zeros(0, np.int32),
        "source_rows": source_rows,
        "row_coords": row_coords,
        "row_labels": row_labels,
        "valid_rows": valid,
        "has_gt": has_gt,
    }


def build_sparse_item_from_npz(npz_path: str, grid_size=(400, 400, 200)) -> Dict:
    """Load one prepared local slice for offline Stage-2 mining.

    The NPZ contains only slice-local coords/labels/dist/hardneg. It is therefore
    safe for Stage 1/2 and cannot expose center/world metadata.
    """
    gx, gy, gz = map(int, grid_size)
    with np.load(npz_path) as d:
        coords = d["coords"].astype(np.int32, copy=False)
        labels = d["labels"].astype(np.int16, copy=False)
        dist = d["dist"].astype(np.float32, copy=False) if "dist" in d else np.zeros(len(coords), np.float32)
    valid = (coords[:,0]>=0)&(coords[:,0]<gx)&(coords[:,1]>=0)&(coords[:,1]<gy)&(coords[:,2]>=0)&(coords[:,2]<gz)
    coords=coords[valid]; labels=labels[valid]; dist=dist[valid]
    source_rows=np.arange(len(coords),dtype=np.int64)
    dmax=max(float(np.nanmax(np.abs(dist))) if len(dist) else 0.0,1.0)
    dist=np.nan_to_num(dist/dmax,nan=0.0,posinf=0.0,neginf=0.0).astype(np.float32,copy=False)
    coords,labels,dist,source_rows=_dedupe_last(coords,labels,dist,source_rows,gx,gy)
    if len(coords):
        order=np.argsort(coords[:,2],kind="stable"); coords=coords[order]; labels=labels[order]; dist=dist[order]; source_rows=source_rows[order]
    return {"sparse_native":True,"coords":coords,"raw_labels":labels,"raw_hardneg":np.zeros(len(coords),np.uint8),
            "dist_values":dist,"z_sorted":coords[:,2] if len(coords) else np.zeros(0,np.int32),
            "source_rows":source_rows,"has_gt":True}


def extract_center_metadata(df: pd.DataFrame) -> Dict[str, float]:
    """Stage-3-only metadata. Never pass these values into Stage 1 or Stage 2."""
    out = {"center_x": np.nan, "center_y": np.nan, "center_z": np.nan}
    for c in out:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
            v = v[np.isfinite(v)]
            if len(v):
                out[c] = float(np.median(v))
    return out


def load_v4_model(
    checkpoint: str,
    device: str = "cuda",
    compile_model: bool = False,
    compile_mode: str = "default",
):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("config", {})) if isinstance(ckpt, dict) else {}
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    # Be tolerant of DataParallel / compiled-state prefixes.
    clean = {}
    for k, v in state.items():
        nk = k
        for pref in ("module.", "_orig_mod."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        clean[nk] = v
    use_coord = bool(int(cfg.get("use_coord_channels", cfg.get("use_coord", 1))))
    use_dist = bool(int(cfg.get("use_dist", 1)))
    in_ch = int(cfg.get("in_channels", cfg.get("in_ch", build_input_channels(use_coord, use_dist))))
    base = int(cfg.get("base_channels", cfg.get("base", 16)))
    model = MultiHeadVoxelNet3D(in_ch=in_ch, base=base)
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing or unexpected:
        # Heads/encoder must match. A silent partial V4 load is unsafe.
        raise RuntimeError(f"V4 checkpoint mismatch: missing={missing} unexpected={unexpected}")
    model = model.to(device).eval()
    compiled = False
    if device.startswith("cuda") and compile_model:
        try:
            model, compiled = maybe_compile_model(model, True, compile_mode)
        except Exception as exc:
            print("WARNING: V4 torch.compile setup failed; using eager mode:", repr(exc))
    cfg.setdefault("use_coord_channels", int(use_coord))
    cfg.setdefault("use_dist", int(use_dist))
    cfg.setdefault("base_channels", base)
    cfg.setdefault("patch_size", 64)
    return model, cfg, compiled


def load_calibration(path: str) -> Dict:
    obj = json.load(open(path))
    weights = obj.get("score_weights", {})
    return {
        "pole_threshold": float(obj.get("pole_threshold", obj.get("threshold", 0.2125))),
        "line_threshold": float(obj.get("line_threshold", obj.get("threshold", 0.2125))),
        "score_sem_weight": float(weights.get("semantic", 0.55)),
        "score_binary_weight": float(weights.get("binary", 0.35)),
        "score_object_weight": float(weights.get("objectness", 0.10)),
        "raw": obj,
    }


def build_v4_patch(
    item: Dict,
    center_xyz: Sequence[int],
    grid_size=(400, 400, 200),
    patch_size: int = 64,
    use_coord_channels: bool = True,
    use_dist: bool = True,
) -> np.ndarray:
    """Materialize one V4 64^3 patch directly from sparse rows."""
    x, y, z = map(int, center_xyz)
    p = int(patch_size)
    start = (x - p // 2, y - p // 2, z - p // 2)
    ridx = sparse_rows_in_box(item, start, (p, p, p))
    occ = np.zeros((p, p, p), np.float32)
    dist = np.zeros_like(occ)
    if len(ridx):
        c = item["coords"][ridx]
        lx = c[:, 0] - start[0]
        ly = c[:, 1] - start[1]
        lz = c[:, 2] - start[2]
        occ[lz, ly, lx] = 1.0
        if use_dist:
            dist[lz, ly, lx] = item["dist_values"][ridx]
    feats = [occ]
    if use_coord_channels:
        feats.extend(make_coord_channels(np.asarray(center_xyz), tuple(grid_size), p))
    if use_dist:
        feats.append(dist)
    return np.stack(feats, axis=0).astype(np.float32, copy=False)


def build_v4_data_patch(item: Dict, center_xyz: Sequence[int], patch_size: int=64, use_dist: bool=True) -> np.ndarray:
    """Materialize only occupancy (+ dist) on CPU. Coordinate channels are generated on GPU."""
    x,y,z=map(int,center_xyz); p=int(patch_size); start=(x-p//2,y-p//2,z-p//2)
    ridx=sparse_rows_in_box(item,start,(p,p,p)); occ=np.zeros((p,p,p),np.float32); dist=np.zeros_like(occ)
    if len(ridx):
        c=item["coords"][ridx]; lx=c[:,0]-start[0]; ly=c[:,1]-start[1]; lz=c[:,2]-start[2]; occ[lz,ly,lx]=1.0
        if use_dist: dist[lz,ly,lx]=item["dist_values"][ridx]
    return np.stack([occ,dist],axis=0) if use_dist else occ[None]


def assemble_v4_channels_gpu(data_tensor: torch.Tensor, centers_xyz: Sequence[Sequence[int]], grid_size, patch_size: int, use_coord: bool, use_dist: bool):
    """Assemble the original V4 [occ,x,y,z,dist] channels on CUDA in FP32."""
    if not use_coord: return data_tensor
    B=data_tensor.shape[0]; p=int(patch_size); gx,gy,gz=map(int,grid_size); dev=data_tensor.device
    centers=torch.as_tensor(np.asarray(centers_xyz,np.int64),device=dev,dtype=torch.float32)
    a=torch.arange(p,device=dev,dtype=torch.float32); r=p//2
    xv=(centers[:,0,None]+a[None,:]-r); yv=(centers[:,1,None]+a[None,:]-r); zv=(centers[:,2,None]+a[None,:]-r)
    xv=((xv/max(gx-1,1))*2.0-1.0).clamp(-1.5,1.5); yv=((yv/max(gy-1,1))*2.0-1.0).clamp(-1.5,1.5); zv=((zv/max(gz-1,1))*2.0-1.0).clamp(-1.5,1.5)
    xch=xv[:,None,None,None,:].expand(B,1,p,p,p); ych=yv[:,None,None,:,None].expand(B,1,p,p,p); zch=zv[:,None,:,None,None].expand(B,1,p,p,p)
    feats=[data_tensor[:,0:1],xch,ych,zch]
    if use_dist: feats.append(data_tensor[:,1:2])
    return torch.cat(feats,dim=1)


def active_core_groups(coords: np.ndarray, grid_size=(400, 400, 200), core_size: int = 48):
    """Map occupied sparse rows to the exact non-overlapping V4 output cores.

    This is a pure scheduling optimization: every occupied voxel belongs to exactly
    one core and receives the same patch-context prediction as the full tiler.
    """
    coords = np.asarray(coords, dtype=np.int32)
    if not len(coords):
        return []
    csz = int(core_size)
    keys = coords // csz
    # Unique in deterministic z/y/x order.
    ukeys = np.unique(keys, axis=0)
    ukeys = ukeys[np.lexsort((ukeys[:, 0], ukeys[:, 1], ukeys[:, 2]))]
    groups = []
    for key in ukeys:
        m = np.all(keys == key, axis=1)
        rows = np.flatnonzero(m).astype(np.int64)
        origin = key.astype(np.int64) * csz
        center = origin + csz // 2
        groups.append({"key": tuple(map(int, key)), "origin": origin, "center": center, "rows": rows})
    return groups


def all_core_groups(coords: np.ndarray, grid_size=(400,400,200), core_size: int=48):
    """Reference full V4 tiling schedule, including empty output cores."""
    c=np.asarray(coords,np.int32); csz=int(core_size); gx,gy,gz=map(int,grid_size); groups=[]
    if len(c): keys=c//csz
    else: keys=np.zeros((0,3),np.int32)
    for z in range(0,gz,csz):
        for y in range(0,gy,csz):
            for x in range(0,gx,csz):
                key=np.array([x//csz,y//csz,z//csz],np.int32); rows=np.flatnonzero(np.all(keys==key,axis=1)).astype(np.int64) if len(c) else np.zeros(0,np.int64)
                origin=np.array([x,y,z],np.int64); groups.append({"key":tuple(map(int,key)),"origin":origin,"center":origin+csz//2,"rows":rows})
    return groups


def _safe_cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def predict_v4_sparse_rows(
    item: Dict,
    model,
    cfg: Dict,
    calibration: Dict,
    grid_size=(400, 400, 200),
    core_size: int = 48,
    batch_size: int = 12,
    amp: str = "bf16",
    channels_last: bool = True,
    return_timing: bool = True,
    evaluate_all_cores: bool = True,
    gpu_coord_channels: bool = False,
):
    """Predict V4 pole/line scores only at occupied input rows.

    The network still executes the original 64^3 V4 patch math. Empty output cores
    are skipped, and only occupied positions are gathered before D2H transfer.
    """
    gx, gy, gz = map(int, grid_size)
    patch = int(cfg.get("patch_size", 64))
    if patch < core_size or (patch - core_size) % 2:
        raise ValueError("V4 patch_size-core_size must be non-negative and even")
    pad = (patch - int(core_size)) // 2
    sl = slice(pad, pad + int(core_size))
    groups = all_core_groups(item["coords"], grid_size, core_size) if evaluate_all_cores else active_core_groups(item["coords"], grid_size, core_size)
    n = len(item["coords"])
    pole = np.zeros(n, np.float32)
    line = np.zeros(n, np.float32)
    semantic = np.zeros(n, np.uint8)
    objectness = np.zeros(n, np.float32)
    timing = {
        "active_cores": len(groups), "total_possible_cores": int(math.ceil(gx/core_size)*math.ceil(gy/core_size)*math.ceil(gz/core_size)),
        "patch_build_ms": 0.0, "h2d_ms": 0.0, "gpu_feature_assembly_ms": 0.0, "gpu_model_ms": 0.0, "d2h_gather_ms": 0.0,
        "batches": 0,
    }
    use_coord = bool(int(cfg.get("use_coord_channels", cfg.get("use_coord", 1))))
    use_dist = bool(int(cfg.get("use_dist", 1)))
    fallback_done = False

    for start in range(0, len(groups), int(batch_size)):
        bg = groups[start:start + int(batch_size)]
        t0 = time.perf_counter()
        if gpu_coord_channels and use_coord:
            built = [build_v4_data_patch(item, g["center"], patch, use_dist) for g in bg]
        else:
            built = [build_v4_patch(item, g["center"], grid_size, patch, use_coord, use_dist) for g in bg]
        timing["patch_build_ms"] += (time.perf_counter() - t0) * 1000.0

        _safe_cuda_sync(); t0 = time.perf_counter()
        xb = torch.from_numpy(np.stack(built)).to("cuda", non_blocking=True)
        _safe_cuda_sync(); timing["h2d_ms"] += (time.perf_counter() - t0) * 1000.0

        _safe_cuda_sync(); t0 = time.perf_counter()
        if gpu_coord_channels and use_coord:
            xb = assemble_v4_channels_gpu(xb,[g["center"] for g in bg],grid_size,patch,use_coord,use_dist)
        if channels_last:
            xb = xb.contiguous(memory_format=torch.channels_last_3d)
        _safe_cuda_sync(); timing["gpu_feature_assembly_ms"] += (time.perf_counter() - t0) * 1000.0

        _safe_cuda_sync(); t0 = time.perf_counter()
        try:
            with torch.inference_mode(), autocast_ctx(amp):
                out = model(xb)
        except Exception as exc:
            if hasattr(model, "_orig_mod") and not fallback_done:
                print("WARNING: compiled V4 inference failed; switching to eager:", repr(exc))
                model = model._orig_mod
                fallback_done = True
                with torch.inference_mode(), autocast_ctx(amp):
                    out = model(xb)
            else:
                raise
        ps, ls = fuse_scores(
            out,
            calibration["score_sem_weight"],
            calibration["score_binary_weight"],
            calibration["score_object_weight"],
        )
        sem = out["semantic"].argmax(dim=1)
        obj = torch.sigmoid(out["objectness"].float()).squeeze(1)
        _safe_cuda_sync(); timing["gpu_model_ms"] += (time.perf_counter() - t0) * 1000.0

        # Gather only occupied rows from the output cores while still on GPU.
        _safe_cuda_sync(); t0 = time.perf_counter()
        offsets = []
        dest_rows = []
        core_vol = int(core_size) ** 3
        for bi, g in enumerate(bg):
            rr = g["rows"]
            local = item["coords"][rr].astype(np.int64) - g["origin"][None, :]
            if np.any(local < 0) or np.any(local >= int(core_size)):
                raise RuntimeError("active-core assignment produced an out-of-core row")
            flat = local[:, 2] * int(core_size) * int(core_size) + local[:, 1] * int(core_size) + local[:, 0]
            offsets.append(flat + bi * core_vol)
            dest_rows.append(rr)
        if offsets:
            take = torch.from_numpy(np.concatenate(offsets).astype(np.int64, copy=False)).to("cuda", non_blocking=True)
            dest = np.concatenate(dest_rows)
            pcore = ps[:, sl, sl, sl].contiguous().view(-1)
            lcore = ls[:, sl, sl, sl].contiguous().view(-1)
            score_pair = torch.stack([pcore.index_select(0, take), lcore.index_select(0, take)], dim=1).to(torch.float32)
            scpu = score_pair.cpu().numpy()
            semcpu = sem[:, sl, sl, sl].contiguous().view(-1).index_select(0, take).to(torch.uint8).cpu().numpy()
            objcpu = obj[:, sl, sl, sl].contiguous().view(-1).index_select(0, take).to(torch.float32).cpu().numpy()
            pole[dest] = scpu[:, 0]
            line[dest] = scpu[:, 1]
            semantic[dest] = semcpu
            objectness[dest] = objcpu
        _safe_cuda_sync(); timing["d2h_gather_ms"] += (time.perf_counter() - t0) * 1000.0
        timing["batches"] += 1

    if len(item["coords"]) and sum(len(g["rows"]) for g in groups) != len(item["coords"]):
        raise RuntimeError("V4 active-core scheduling did not cover every occupied row")
    return {
        "pole": pole, "line": line, "semantic": semantic, "objectness": objectness,
        "timing": timing,
    }


def label_from_scores(pole: np.ndarray, line: np.ndarray, pole_threshold: float, line_threshold: float) -> np.ndarray:
    out = np.zeros(len(pole), np.int8)
    p = np.asarray(pole) >= float(pole_threshold)
    l = np.asarray(line) >= float(line_threshold)
    out[p & ~l] = 1
    out[l & ~p] = 2
    both = p & l
    out[both & (pole >= line)] = 1
    out[both & (line > pole)] = 2
    return out


def row_level_scores(item: Dict, pred: Dict, total_rows: int, calibration: Dict) -> pd.DataFrame:
    """Return compact row-aligned V4 scores for diagnostics/evaluation."""
    out = pd.DataFrame({
        "source_row": item["source_rows"].astype(np.int64),
        "x": item["coords"][:, 0], "y": item["coords"][:, 1], "z": item["coords"][:, 2],
        "v4_pole_score": pred["pole"], "v4_line_score": pred["line"],
        "v4_semantic": pred["semantic"], "v4_objectness": pred["objectness"],
        "v4_pred": label_from_scores(pred["pole"], pred["line"], calibration["pole_threshold"], calibration["line_threshold"]),
    })
    if item.get("has_gt", False):
        out["label"] = item["raw_labels"]
    return out


@dataclass
class V4RuntimeConfig:
    grid_size: Tuple[int, int, int] = (400, 400, 200)
    voxel_size_ft: float = 0.5
    core_size: int = 48
    batch_size: int = 12
    amp: str = "bf16"
    channels_last: bool = True
