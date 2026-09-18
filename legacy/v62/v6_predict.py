#!/usr/bin/env python3
"""Shared full-scene prediction utilities for V6."""
from __future__ import annotations

import concurrent.futures as futures
from contextlib import nullcontext
from typing import Dict, Sequence, Tuple

import numpy as np
import torch

from precision_common import fuse_scores, maybe_compile_model
from v6_common import SpanAwareGeoNet3D, build_v6_inference_features


def setup_torch():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def autocast_ctx(amp: str):
    if amp == "none":
        return nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if amp == "bf16" else torch.float16)


def tile_centers(grid_size: Sequence[int], core_size: int):
    gx, gy, gz = [int(v) for v in grid_size]
    return [np.array([x + core_size // 2, y + core_size // 2, z + core_size // 2], dtype=np.int64)
            for z in range(0, gz, core_size) for y in range(0, gy, core_size) for x in range(0, gx, core_size)]


def load_v6_model(model_path: str, device="cuda", compile_model=True, compile_mode="reduce-overhead"):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    in_ch = 1 + (3 if int(cfg.get("use_coord_channels", 1)) else 0) + (1 if int(cfg.get("use_dist", 1)) else 0)
    model = SpanAwareGeoNet3D(in_ch=in_ch, base=int(cfg.get("base_channels", 16)))
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    compiled = False
    if device.startswith("cuda") and compile_model:
        try:
            model, compiled = maybe_compile_model(model, True, compile_mode)
        except Exception as exc:
            print("WARNING: torch.compile setup failed, using eager mode:", repr(exc))
    return model, cfg, compiled


def _build(item, center, grid_size, cfg):
    fine, coarse = build_v6_inference_features(
        item, center, tuple(grid_size), int(cfg.get("patch_size", 64)),
        int(cfg.get("context_xy", 256)), int(cfg.get("context_z", 128)),
        bool(int(cfg.get("use_coord_channels", 1))), bool(int(cfg.get("use_dist", 1))),
    )
    return fine, coarse


def predict_dense_scores(
    item: Dict,
    model,
    cfg: Dict,
    grid_size: Tuple[int, int, int],
    core_size=48,
    batch_size=5,
    build_workers=6,
    amp="bf16",
    channels_last=True,
    score_sem_weight=.55,
    score_binary_weight=.35,
    score_object_weight=.10,
):
    """Return dense float16 pole/line/orientation scores and semantic argmax."""
    gx, gy, gz = [int(v) for v in grid_size]
    patch = int(cfg.get("patch_size", 64))
    if patch < core_size or (patch - core_size) % 2:
        raise ValueError("patch_size-core_size must be non-negative and even")
    pad = (patch - core_size) // 2
    sl = slice(pad, pad + core_size)
    centers = tile_centers(grid_size, core_size)
    pole = np.zeros((gz, gy, gx), dtype=np.float16)
    line = np.zeros((gz, gy, gx), dtype=np.float16)
    vertical = np.zeros((gz, gy, gx), dtype=np.float16)
    horizontal = np.zeros((gz, gy, gx), dtype=np.float16)
    semantic = np.zeros((gz, gy, gx), dtype=np.uint8)
    fallback_done = False
    with futures.ThreadPoolExecutor(max_workers=max(1, int(build_workers))) as pool:
        for start in range(0, len(centers), int(batch_size)):
            bc = centers[start:start + int(batch_size)]
            built = list(pool.map(lambda c: _build(item, c, grid_size, cfg), bc))
            xf = torch.from_numpy(np.stack([b[0] for b in built])).cuda(non_blocking=True)
            xc = torch.from_numpy(np.stack([b[1] for b in built])).cuda(non_blocking=True)
            if channels_last:
                xf = xf.contiguous(memory_format=torch.channels_last_3d)
                xc = xc.contiguous(memory_format=torch.channels_last_3d)
            try:
                with torch.inference_mode(), autocast_ctx(amp):
                    out = model(xf, xc)
            except Exception as exc:
                if hasattr(model, "_orig_mod") and not fallback_done:
                    print("WARNING: compiled inference failed; retrying permanently in eager mode:", repr(exc))
                    model = model._orig_mod
                    fallback_done = True
                    with torch.inference_mode(), autocast_ctx(amp):
                        out = model(xf, xc)
                else:
                    raise
            ps, ls = fuse_scores(out, score_sem_weight, score_binary_weight, score_object_weight)
            vs = torch.sigmoid(out["verticality"].float()).squeeze(1)
            hs = torch.sigmoid(out["horizontality"].float()).squeeze(1)
            sem = out["semantic"].argmax(dim=1)
            for bi, c in enumerate(bc):
                x0, y0, z0 = [int(v) - core_size // 2 for v in c]
                x1, y1, z1 = min(gx, x0 + core_size), min(gy, y0 + core_size), min(gz, z0 + core_size)
                vx, vy, vz = x1 - x0, y1 - y0, z1 - z0
                pole[z0:z1, y0:y1, x0:x1] = ps[bi, sl, sl, sl][:vz, :vy, :vx].detach().cpu().numpy().astype(np.float16)
                line[z0:z1, y0:y1, x0:x1] = ls[bi, sl, sl, sl][:vz, :vy, :vx].detach().cpu().numpy().astype(np.float16)
                vertical[z0:z1, y0:y1, x0:x1] = vs[bi, sl, sl, sl][:vz, :vy, :vx].detach().cpu().numpy().astype(np.float16)
                horizontal[z0:z1, y0:y1, x0:x1] = hs[bi, sl, sl, sl][:vz, :vy, :vx].detach().cpu().numpy().astype(np.float16)
                semantic[z0:z1, y0:y1, x0:x1] = sem[bi, sl, sl, sl][:vz, :vy, :vx].detach().cpu().numpy().astype(np.uint8)
    return {"pole": pole, "line": line, "verticality": vertical, "horizontality": horizontal, "semantic": semantic}
