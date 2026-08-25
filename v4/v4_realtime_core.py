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


def _cuda_event_pair():
    if not torch.cuda.is_available():
        return None, None
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _event_ms(pair) -> float:
    a, b = pair
    if a is None or b is None:
        return 0.0
    return float(a.elapsed_time(b))


def _prepare_group_gather(groups, coords: np.ndarray, core_size: int) -> None:
    """Precompute occupied output-core offsets once per slice on CPU."""
    c = np.asarray(coords, dtype=np.int64)
    core_vol = int(core_size) ** 3
    for g in groups:
        rr = np.asarray(g["rows"], dtype=np.int64)
        if not len(rr):
            g["_flat_core"] = np.zeros(0, np.int64)
            continue
        local = c[rr] - np.asarray(g["origin"], dtype=np.int64)[None, :]
        if np.any(local < 0) or np.any(local >= int(core_size)):
            raise RuntimeError("V4 core assignment produced an out-of-core occupied row")
        g["_flat_core"] = (
            local[:, 2] * int(core_size) * int(core_size)
            + local[:, 1] * int(core_size)
            + local[:, 0]
        ).astype(np.int64, copy=False)
        if np.any(g["_flat_core"] < 0) or np.any(g["_flat_core"] >= core_vol):
            raise RuntimeError("V4 core gather offset is outside the output core")


def _pad_batch_numpy(built, centers, target: int):
    """Pad only the final active-core batch to a fixed CUDA batch shape.

    The added samples are exact duplicates of the last real patch and are discarded.
    No result from a padded sample can be written into the occupied-row outputs.
    """
    if not built or len(built) >= int(target):
        return built, centers
    need = int(target) - len(built)
    last = built[-1]
    last_center = centers[-1]
    built = list(built) + [last] * need
    centers = list(centers) + [last_center] * need
    return built, centers


def _run_model_scores(model, xb, calibration, amp: str, fallback_state: Dict):
    try:
        with torch.inference_mode(), autocast_ctx(amp):
            out = model(xb)
    except Exception as exc:
        if hasattr(model, "_orig_mod") and not fallback_state.get("done", False):
            print("WARNING: compiled V4 inference failed; switching to eager:", repr(exc))
            model = model._orig_mod
            fallback_state["done"] = True
            fallback_state["model"] = model
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
    return model, ps, ls, sem, obj


def _predict_v4_cpu_patch_reference(
    item: Dict,
    model,
    cfg: Dict,
    calibration: Dict,
    grid_size,
    core_size: int,
    batch_size: int,
    amp: str,
    channels_last: bool,
    evaluate_all_cores: bool,
    fixed_batch_shape: bool,
):
    """Original V4 CPU patch construction path retained as the equivalence reference."""
    gx, gy, gz = map(int, grid_size)
    patch = int(cfg.get("patch_size", 64))
    pad = (patch - int(core_size)) // 2
    sl = slice(pad, pad + int(core_size))
    groups = all_core_groups(item["coords"], grid_size, core_size) if evaluate_all_cores else active_core_groups(item["coords"], grid_size, core_size)
    _prepare_group_gather(groups, item["coords"], core_size)
    n = len(item["coords"])
    pole = np.zeros(n, np.float32)
    line = np.zeros(n, np.float32)
    semantic = np.zeros(n, np.uint8)
    objectness = np.zeros(n, np.float32)
    timing = {
        "active_cores": len(groups),
        "total_possible_cores": int(math.ceil(gx/core_size)*math.ceil(gy/core_size)*math.ceil(gz/core_size)),
        "patch_build_ms": 0.0,
        "host_batch_pack_ms": 0.0,
        "host_pin_ms": 0.0,
        "h2d_ms": 0.0,
        "sparse_h2d_cuda_ms": 0.0,
        "gpu_sparse_scatter_ms": 0.0,
        "gpu_patch_extract_ms": 0.0,
        "gpu_feature_assembly_ms": 0.0,
        "gpu_model_ms": 0.0,
        "gpu_gather_ms": 0.0,
        "d2h_gather_ms": 0.0,
        "cuda_sync_count": 0,
        "gpu_volume_mode": 0,
        "fixed_batch_shape": int(bool(fixed_batch_shape)),
        "batches": 0,
    }
    use_coord = bool(int(cfg.get("use_coord_channels", cfg.get("use_coord", 1))))
    use_dist = bool(int(cfg.get("use_dist", 1)))
    fallback_state = {"done": False, "model": model}
    core_vol = int(core_size) ** 3

    for start in range(0, len(groups), int(batch_size)):
        bg = groups[start:start + int(batch_size)]
        real_count = len(bg)
        t0 = time.perf_counter()
        built = [build_v4_patch(item, g["center"], grid_size, patch, use_coord, use_dist) for g in bg]
        timing["patch_build_ms"] += (time.perf_counter() - t0) * 1000.0
        centers = [g["center"] for g in bg]
        if fixed_batch_shape and not evaluate_all_cores:
            built, centers = _pad_batch_numpy(built, centers, int(batch_size))

        t0 = time.perf_counter()
        host_batch = np.stack(built)
        timing["host_batch_pack_ms"] += (time.perf_counter() - t0) * 1000.0

        _safe_cuda_sync(); timing["cuda_sync_count"] += 1; t0 = time.perf_counter()
        xb = torch.from_numpy(host_batch).to("cuda", non_blocking=True)
        _safe_cuda_sync(); timing["cuda_sync_count"] += 1
        timing["h2d_ms"] += (time.perf_counter() - t0) * 1000.0

        _safe_cuda_sync(); timing["cuda_sync_count"] += 1; t0 = time.perf_counter()
        if channels_last:
            xb = xb.contiguous(memory_format=torch.channels_last_3d)
        _safe_cuda_sync(); timing["cuda_sync_count"] += 1
        timing["gpu_feature_assembly_ms"] += (time.perf_counter() - t0) * 1000.0

        _safe_cuda_sync(); timing["cuda_sync_count"] += 1; t0 = time.perf_counter()
        model, ps, ls, sem, obj = _run_model_scores(model, xb, calibration, amp, fallback_state)
        _safe_cuda_sync(); timing["cuda_sync_count"] += 1
        timing["gpu_model_ms"] += (time.perf_counter() - t0) * 1000.0

        _safe_cuda_sync(); timing["cuda_sync_count"] += 1; t0 = time.perf_counter()
        offsets = []
        dest_rows = []
        for bi, g in enumerate(bg[:real_count]):
            rr = np.asarray(g["rows"], np.int64)
            if not len(rr):
                continue
            offsets.append(g["_flat_core"] + bi * core_vol)
            dest_rows.append(rr)
        if offsets:
            take = torch.from_numpy(np.concatenate(offsets).astype(np.int64, copy=False)).to("cuda", non_blocking=True)
            dest = np.concatenate(dest_rows)
            pcore = ps[:real_count, sl, sl, sl].contiguous().view(-1)
            lcore = ls[:real_count, sl, sl, sl].contiguous().view(-1)
            score_pair = torch.stack([pcore.index_select(0, take), lcore.index_select(0, take)], dim=1).to(torch.float32)
            scpu = score_pair.cpu().numpy()
            semcpu = sem[:real_count, sl, sl, sl].contiguous().view(-1).index_select(0, take).to(torch.uint8).cpu().numpy()
            objcpu = obj[:real_count, sl, sl, sl].contiguous().view(-1).index_select(0, take).to(torch.float32).cpu().numpy()
            pole[dest] = scpu[:, 0]
            line[dest] = scpu[:, 1]
            semantic[dest] = semcpu
            objectness[dest] = objcpu
        _safe_cuda_sync(); timing["cuda_sync_count"] += 1
        timing["d2h_gather_ms"] += (time.perf_counter() - t0) * 1000.0
        timing["batches"] += 1

    if len(item["coords"]) and sum(len(g["rows"]) for g in groups) != len(item["coords"]):
        raise RuntimeError("V4 core scheduling did not cover every occupied row")
    return {"pole": pole, "line": line, "semantic": semantic, "objectness": objectness, "timing": timing}


def _pin_numpy_tensor(array: np.ndarray, dtype=None):
    """Copy a small NumPy array into pinned host memory when CUDA supports it."""
    src = torch.from_numpy(array)
    if dtype is not None:
        src = src.to(dtype=dtype)
    try:
        dst = torch.empty(src.shape, dtype=src.dtype, pin_memory=True)
        dst.copy_(src)
        return dst, True
    except Exception:
        return src.contiguous(), False


def _predict_v4_gpu_sparse_volume(
    item: Dict,
    model,
    cfg: Dict,
    calibration: Dict,
    grid_size,
    core_size: int,
    batch_size: int,
    amp: str,
    channels_last: bool,
    evaluate_all_cores: bool,
    fixed_batch_shape: bool,
):
    """Production V4 path: one sparse H2D per slice, GPU scatter, GPU patch extraction.

    This preserves the original V4 64^3 patch / 48^3 output-core math.  Unlike the
    reference path, it never materializes dense patches on the CPU and never transfers
    a dense patch batch over PCIe.  Fused occupied-row scores remain on the GPU across
    all batches and are copied to the host exactly once at the end of the slice.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("V4 GPU sparse-volume inference requires CUDA")
    gx, gy, gz = map(int, grid_size)
    patch = int(cfg.get("patch_size", 64))
    pad = (patch - int(core_size)) // 2
    if patch < int(core_size) or (patch - int(core_size)) % 2:
        raise ValueError("V4 patch_size-core_size must be non-negative and even")
    sl = slice(pad, pad + int(core_size))
    use_coord = bool(int(cfg.get("use_coord_channels", cfg.get("use_coord", 1))))
    use_dist = bool(int(cfg.get("use_dist", 1)))
    groups = all_core_groups(item["coords"], grid_size, core_size) if evaluate_all_cores else active_core_groups(item["coords"], grid_size, core_size)
    _prepare_group_gather(groups, item["coords"], core_size)
    n = len(item["coords"])
    timing = {
        "active_cores": len(groups),
        "total_possible_cores": int(math.ceil(gx/core_size)*math.ceil(gy/core_size)*math.ceil(gz/core_size)),
        "patch_build_ms": 0.0,
        "host_batch_pack_ms": 0.0,
        "host_pin_ms": 0.0,
        "h2d_ms": 0.0,
        "sparse_h2d_cuda_ms": 0.0,
        "gpu_sparse_scatter_ms": 0.0,
        "gpu_patch_extract_ms": 0.0,
        "gpu_feature_assembly_ms": 0.0,
        "gpu_model_ms": 0.0,
        "gpu_gather_ms": 0.0,
        "d2h_gather_ms": 0.0,
        "cuda_sync_count": 1,
        "gpu_volume_mode": 1,
        "fixed_batch_shape": int(bool(fixed_batch_shape)),
        "batches": 0,
    }
    if n == 0:
        return {
            "pole": np.zeros(0, np.float32), "line": np.zeros(0, np.float32),
            "semantic": np.zeros(0, np.uint8), "objectness": np.zeros(0, np.float32),
            "timing": timing,
        }

    # A full patch of zero border covers first/last output cores even when a 48-voxel
    # core starts outside the nominal grid extent (for example x=384 in a 400 grid).
    border = patch
    channels = 2 if use_dist else 1
    device = torch.device("cuda")
    data_volume = torch.zeros(
        (channels, gz + 2 * border, gy + 2 * border, gx + 2 * border),
        device=device, dtype=torch.float32,
    )

    t0 = time.perf_counter()
    coords_np = np.asarray(item["coords"], dtype=np.int64)
    dist_np = np.asarray(item["dist_values"], dtype=np.float32)
    coords_host, pinned_coords = _pin_numpy_tensor(coords_np, torch.int64)
    dist_host = None
    pinned_dist = False
    if use_dist:
        dist_host, pinned_dist = _pin_numpy_tensor(dist_np, torch.float32)
    timing["host_pin_ms"] = (time.perf_counter() - t0) * 1000.0
    timing["host_pinned"] = int(bool(pinned_coords and (pinned_dist if use_dist else True)))

    h2d = _cuda_event_pair(); h2d[0].record()
    coords_gpu = coords_host.to(device, non_blocking=bool(pinned_coords))
    dist_gpu = dist_host.to(device, non_blocking=bool(pinned_dist)) if use_dist else None
    h2d[1].record()

    scatter = _cuda_event_pair(); scatter[0].record()
    xx = coords_gpu[:, 0] + border
    yy = coords_gpu[:, 1] + border
    zz = coords_gpu[:, 2] + border
    data_volume[0, zz, yy, xx] = 1.0
    if use_dist:
        data_volume[1, zz, yy, xx] = dist_gpu
    scatter[1].record()

    pole_gpu = torch.zeros(n, device=device, dtype=torch.float32)
    line_gpu = torch.zeros(n, device=device, dtype=torch.float32)
    semantic_gpu = torch.zeros(n, device=device, dtype=torch.uint8)
    objectness_gpu = torch.zeros(n, device=device, dtype=torch.float32)
    fallback_state = {"done": False, "model": model}
    core_vol = int(core_size) ** 3
    patch_events = []
    feature_events = []
    model_events = []
    gather_events = []

    for start in range(0, len(groups), int(batch_size)):
        bg = groups[start:start + int(batch_size)]
        real_count = len(bg)
        if not real_count:
            continue
        centers = [np.asarray(g["center"], dtype=np.int64) for g in bg]

        pe = _cuda_event_pair(); pe[0].record()
        patches = []
        for center in centers:
            sx = int(center[0]) - patch // 2 + border
            sy = int(center[1]) - patch // 2 + border
            sz = int(center[2]) - patch // 2 + border
            q = data_volume[:, sz:sz+patch, sy:sy+patch, sx:sx+patch]
            if tuple(q.shape[-3:]) != (patch, patch, patch):
                raise RuntimeError(f"GPU V4 patch extraction returned shape {tuple(q.shape)} for center {center.tolist()}")
            patches.append(q)
        xb = torch.stack(patches, dim=0)
        padded_centers = list(centers)
        if fixed_batch_shape and not evaluate_all_cores and real_count < int(batch_size):
            need = int(batch_size) - real_count
            xb = torch.cat([xb, xb[-1:].expand(need, -1, -1, -1, -1)], dim=0)
            padded_centers.extend([centers[-1]] * need)
        pe[1].record(); patch_events.append(pe)

        fe = _cuda_event_pair(); fe[0].record()
        xb = assemble_v4_channels_gpu(xb, padded_centers, grid_size, patch, use_coord, use_dist) if use_coord else xb
        if channels_last:
            xb = xb.contiguous(memory_format=torch.channels_last_3d)
        fe[1].record(); feature_events.append(fe)

        me = _cuda_event_pair(); me[0].record()
        model, ps, ls, sem, obj = _run_model_scores(model, xb, calibration, amp, fallback_state)
        me[1].record(); model_events.append(me)

        ge = _cuda_event_pair(); ge[0].record()
        offsets = []
        dest_rows = []
        for bi, g in enumerate(bg):
            rr = np.asarray(g["rows"], dtype=np.int64)
            if not len(rr):
                continue
            offsets.append(g["_flat_core"] + bi * core_vol)
            dest_rows.append(rr)
        if offsets:
            take_np = np.concatenate(offsets).astype(np.int64, copy=False)
            dest_np = np.concatenate(dest_rows).astype(np.int64, copy=False)
            take_host, take_pinned = _pin_numpy_tensor(take_np, torch.int64)
            dest_host, dest_pinned = _pin_numpy_tensor(dest_np, torch.int64)
            take = take_host.to(device, non_blocking=bool(take_pinned))
            dest = dest_host.to(device, non_blocking=bool(dest_pinned))
            pcore = ps[:real_count, sl, sl, sl].contiguous().view(-1)
            lcore = ls[:real_count, sl, sl, sl].contiguous().view(-1)
            semcore = sem[:real_count, sl, sl, sl].contiguous().view(-1)
            objcore = obj[:real_count, sl, sl, sl].contiguous().view(-1)
            pole_gpu.index_copy_(0, dest, pcore.index_select(0, take).to(torch.float32))
            line_gpu.index_copy_(0, dest, lcore.index_select(0, take).to(torch.float32))
            semantic_gpu.index_copy_(0, dest, semcore.index_select(0, take).to(torch.uint8))
            objectness_gpu.index_copy_(0, dest, objcore.index_select(0, take).to(torch.float32))
        ge[1].record(); gather_events.append(ge)
        timing["batches"] += 1

    # One occupied-row result transfer per slice.  This is intentionally the only
    # synchronization in the production GPU-volume path.
    d2h_t0 = time.perf_counter()
    score_cpu = torch.stack([pole_gpu, line_gpu, objectness_gpu], dim=1).cpu().numpy()
    semantic_cpu = semantic_gpu.cpu().numpy()
    torch.cuda.synchronize()
    timing["d2h_gather_ms"] = (time.perf_counter() - d2h_t0) * 1000.0
    timing["sparse_h2d_cuda_ms"] = _event_ms(h2d)
    timing["h2d_ms"] = timing["sparse_h2d_cuda_ms"]
    timing["gpu_sparse_scatter_ms"] = _event_ms(scatter)
    timing["gpu_patch_extract_ms"] = float(sum(_event_ms(x) for x in patch_events))
    timing["gpu_feature_assembly_ms"] = float(sum(_event_ms(x) for x in feature_events))
    timing["gpu_model_ms"] = float(sum(_event_ms(x) for x in model_events))
    timing["gpu_gather_ms"] = float(sum(_event_ms(x) for x in gather_events))

    if sum(len(g["rows"]) for g in groups) != n:
        raise RuntimeError("V4 GPU core scheduling did not cover every occupied row")
    return {
        "pole": score_cpu[:, 0].astype(np.float32, copy=False),
        "line": score_cpu[:, 1].astype(np.float32, copy=False),
        "semantic": semantic_cpu.astype(np.uint8, copy=False),
        "objectness": score_cpu[:, 2].astype(np.float32, copy=False),
        "timing": timing,
    }


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
    fixed_batch_shape: bool = False,
):
    """Predict V4 scores at occupied rows with a gated reference/production path.

    ``gpu_coord_channels=False`` retains the original CPU patch implementation and is
    the numerical reference. ``gpu_coord_channels=True`` uses the production sparse-
    volume implementation: one sparse H2D transfer, GPU scatter/patch extraction,
    GPU coordinate channels, and one occupied-score D2H transfer.
    """
    patch = int(cfg.get("patch_size", 64))
    if patch < int(core_size) or (patch - int(core_size)) % 2:
        raise ValueError("V4 patch_size-core_size must be non-negative and even")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if gpu_coord_channels:
        return _predict_v4_gpu_sparse_volume(
            item, model, cfg, calibration, grid_size, int(core_size), int(batch_size), amp,
            bool(channels_last), bool(evaluate_all_cores), bool(fixed_batch_shape),
        )
    return _predict_v4_cpu_patch_reference(
        item, model, cfg, calibration, grid_size, int(core_size), int(batch_size), amp,
        bool(channels_last), bool(evaluate_all_cores), bool(fixed_batch_shape),
    )
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
