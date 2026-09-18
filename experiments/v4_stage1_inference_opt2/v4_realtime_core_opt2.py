#!/usr/bin/env python3
"""Quality-gated Stage-1 execution experiments derived from accepted Opt1.

This module intentionally leaves the accepted V4 network, checkpoint loading,
calibration, BF16 autocast, patch/core geometry, batch size, memory format, score
fusion, thresholds and output serialization unchanged. Opt2 permits only outer
execution-plumbing changes that must reproduce production outputs exactly.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Sequence

import numpy as np
import torch

import v4_realtime_core as reference


OPTIMIZATION_VERSION = "v4-stage1-active-gpu-opt2-e5-coordinate-input-cache-20260918"


def active_core_groups_opt(
    coords: np.ndarray,
    grid_size=(400, 400, 200),
    core_size: int = 48,
):
    """Create the accepted z/y/x core schedule without K full-array scans.

    The accepted implementation forms each unique core and then evaluates
    ``np.all(keys == key, axis=1)`` for every core, which is O(K*N).  Stable sorting
    once by a bounded linear core id is O(N log N), preserves ascending source-row
    order inside each core, and produces the identical z/y/x core order.
    """
    c = np.asarray(coords, dtype=np.int32)
    if not len(c):
        return []
    gx, gy, gz = map(int, grid_size)
    csz = int(core_size)
    if csz <= 0:
        raise ValueError("core_size must be positive")
    if np.any(c < 0) or np.any(c[:, 0] >= gx) or np.any(c[:, 1] >= gy) or np.any(c[:, 2] >= gz):
        raise RuntimeError("active-core scheduling received an out-of-grid coordinate")

    keys = c // csz
    nx = int(math.ceil(gx / csz))
    ny = int(math.ceil(gy / csz))
    flat = ((keys[:, 2].astype(np.int64) * ny + keys[:, 1]) * nx + keys[:, 0])
    order = np.argsort(flat, kind="stable")
    sorted_flat = flat[order]
    starts = np.r_[0, np.flatnonzero(sorted_flat[1:] != sorted_flat[:-1]) + 1]
    stops = np.r_[starts[1:], len(order)]

    groups = []
    for start, stop in zip(starts, stops):
        rows = order[int(start):int(stop)].astype(np.int64, copy=False)
        key = keys[rows[0]].astype(np.int64, copy=False)
        origin = key * csz
        groups.append(
            {
                "key": tuple(map(int, key)),
                "origin": origin,
                "center": origin + csz // 2,
                "rows": rows,
            }
        )
    return groups


def active_core_schedule_with_batch_plans_opt(
    coords: np.ndarray,
    grid_size=(400, 400, 200),
    core_size: int = 48,
    batch_size: int = 12,
):
    """Build the accepted core schedule and identical gather plans in one pass.

    Stable sorting by the accepted z/y/x linear core id makes every core a
    contiguous span.  E4 derives each batch's destination rows and flattened
    core offsets directly from those spans, avoiding the later per-batch Python
    list construction and ``np.concatenate`` calls.  Row order and batch-slot
    offsets are byte-for-byte equivalent to the E0 gather-plan construction.
    """
    c = np.asarray(coords, dtype=np.int32)
    if not len(c):
        return [], []
    gx, gy, gz = map(int, grid_size)
    csz = int(core_size)
    bsz = int(batch_size)
    if csz <= 0 or bsz <= 0:
        raise ValueError("core_size and batch_size must be positive")
    if np.any(c < 0) or np.any(c[:, 0] >= gx) or np.any(c[:, 1] >= gy) or np.any(c[:, 2] >= gz):
        raise RuntimeError("active-core scheduling received an out-of-grid coordinate")

    keys = c // csz
    nx = int(math.ceil(gx / csz))
    ny = int(math.ceil(gy / csz))
    flat = ((keys[:, 2].astype(np.int64) * ny + keys[:, 1]) * nx + keys[:, 0])
    order = np.argsort(flat, kind="stable").astype(np.int64, copy=False)
    sorted_flat = flat[order]
    starts = np.r_[0, np.flatnonzero(sorted_flat[1:] != sorted_flat[:-1]) + 1]
    stops = np.r_[starts[1:], len(order)]

    groups = []
    for start, stop in zip(starts, stops):
        rows = order[int(start):int(stop)]
        key = keys[rows[0]].astype(np.int64, copy=False)
        origin = key * csz
        groups.append(
            {
                "key": tuple(map(int, key)),
                "origin": origin,
                "center": origin + csz // 2,
                "rows": rows,
            }
        )

    core_vol = csz ** 3
    group_ids = np.repeat(
        np.arange(len(groups), dtype=np.int64),
        (stops - starts).astype(np.int64, copy=False),
    )
    sorted_keys = keys[order].astype(np.int64, copy=False)
    local = c[order].astype(np.int64, copy=False) - sorted_keys * csz
    flat_core_all = ((local[:, 2] * csz + local[:, 1]) * csz + local[:, 0])
    take_all = flat_core_all + (group_ids % bsz) * core_vol

    for group, start, stop in zip(groups, starts, stops):
        group["_flat_core"] = flat_core_all[int(start):int(stop)]

    plans = []
    for group_start in range(0, len(groups), bsz):
        group_stop = min(group_start + bsz, len(groups))
        row_start = int(starts[group_start])
        row_stop = int(stops[group_stop - 1])
        plans.append(
            (
                take_all[row_start:row_stop].astype(np.int64, copy=False),
                order[row_start:row_stop].astype(np.int64, copy=False),
            )
        )
    return groups, plans


def exact_padding(grid_size: Sequence[int], patch_size: int, core_size: int):
    """Return minimal (low, high) x/y/z zero padding for the accepted tiler."""
    patch = int(patch_size)
    csz = int(core_size)
    if patch < csz or (patch - csz) % 2:
        raise ValueError("patch_size-core_size must be non-negative and even")
    low = max(0, patch // 2 - csz // 2)
    lows = np.asarray([low, low, low], dtype=np.int64)
    highs = []
    for extent in map(int, grid_size):
        last_origin = ((extent - 1) // csz) * csz
        last_center = last_origin + csz // 2
        highs.append(max(0, last_center + patch // 2 - extent))
    return lows, np.asarray(highs, dtype=np.int64)


@dataclass
class V4SparseGpuWorkspace:
    """Reusable sparse-volume scratch storage for sequential slice inference."""

    volume: torch.Tensor | None = None
    previous_coords: torch.Tensor | None = None
    spec: tuple | None = None
    low_pad: np.ndarray | None = None
    high_pad: np.ndarray | None = None
    host_scores: torch.Tensor | None = None
    host_semantic: torch.Tensor | None = None
    host_output_capacity: int = 0
    model_input: torch.Tensor | None = None
    model_input_spec: tuple | None = None
    coordinate_lines: dict = field(default_factory=dict)

    def reset(self) -> None:
        self.volume = None
        self.previous_coords = None
        self.spec = None
        self.low_pad = None
        self.high_pad = None
        self.host_scores = None
        self.host_semantic = None
        self.host_output_capacity = 0
        self.model_input = None
        self.model_input_spec = None
        self.coordinate_lines.clear()

    def acquire(self, grid_size, patch_size: int, core_size: int, channels: int, device):
        gx, gy, gz = map(int, grid_size)
        low, high = exact_padding((gx, gy, gz), patch_size, core_size)
        shape = (
            int(channels),
            gz + int(low[2]) + int(high[2]),
            gy + int(low[1]) + int(high[1]),
            gx + int(low[0]) + int(high[0]),
        )
        spec = (shape, str(torch.device(device)), torch.float32)
        allocated = self.volume is None or self.spec != spec
        if allocated:
            self.volume = torch.zeros(shape, device=device, dtype=torch.float32)
            self.previous_coords = None
            self.spec = spec
            self.low_pad = low
            self.high_pad = high
        return self.volume, self.low_pad, self.previous_coords, allocated

    def remember(self, coords_gpu: torch.Tensor) -> None:
        self.previous_coords = coords_gpu

    def acquire_host_outputs(self, rows: int):
        """Return reusable pinned CPU output buffers with at least rows capacity."""
        rows = int(rows)
        allocated = (
            self.host_scores is None
            or self.host_semantic is None
            or self.host_output_capacity < rows
        )
        if allocated:
            self.host_scores = torch.empty(
                (rows, 3), dtype=torch.float32, device="cpu", pin_memory=True
            )
            self.host_semantic = torch.empty(
                rows, dtype=torch.uint8, device="cpu", pin_memory=True
            )
            self.host_output_capacity = rows
        return self.host_scores[:rows], self.host_semantic[:rows], allocated

    def acquire_model_input(self, batch: int, channels: int, patch: int, device):
        """Return a reusable FP32 channels-last model-input tensor."""
        shape = (int(batch), int(channels), int(patch), int(patch), int(patch))
        spec = (shape, str(torch.device(device)), torch.float32)
        allocated = self.model_input is None or self.model_input_spec != spec
        if allocated:
            self.model_input = torch.empty(
                shape,
                device=device,
                dtype=torch.float32,
                memory_format=torch.channels_last_3d,
            )
            self.model_input_spec = spec
        return self.model_input, allocated

    def coordinate_line(
        self,
        axis: int,
        center: int,
        extent: int,
        patch: int,
        device,
    ):
        """Cache the exact FP32 one-dimensional V4 coordinate-channel values."""
        key = (
            int(axis), int(center), int(extent), int(patch), str(torch.device(device))
        )
        line = self.coordinate_lines.get(key)
        hit = line is not None
        if line is None:
            center_tensor = torch.tensor(
                float(center), device=device, dtype=torch.float32
            )
            offsets = torch.arange(int(patch), device=device, dtype=torch.float32)
            line = center_tensor + offsets - int(patch) // 2
            line = ((line / max(int(extent) - 1, 1)) * 2.0 - 1.0).clamp(
                -1.5, 1.5
            )
            self.coordinate_lines[key] = line
        return line, hit


def assemble_v4_channels_cached_opt(
    data_tensor: torch.Tensor,
    centers_xyz: Sequence[Sequence[int]],
    grid_size,
    patch_size: int,
    use_coord: bool,
    use_dist: bool,
    workspace: V4SparseGpuWorkspace,
):
    """Build the exact V4 input in a reusable channels-last buffer.

    E5 caches only deterministic coordinate lines. Dynamic occupancy and
    distance channels are copied from the unchanged E0 patch tensor. The final
    tensor has the same values, shape, dtype and channels-last layout as E0.
    """
    if not use_coord:
        raise ValueError("E5 coordinate caching requires production coordinate channels")
    batch = int(data_tensor.shape[0])
    patch = int(patch_size)
    gx, gy, gz = map(int, grid_size)
    channels = 5 if use_dist else 4
    output, allocated = workspace.acquire_model_input(
        batch, channels, patch, data_tensor.device
    )

    axis_lines = [[], [], []]
    hits = 0
    misses = 0
    for center in centers_xyz:
        values = tuple(map(int, center))
        for axis, (value, extent) in enumerate(zip(values, (gx, gy, gz))):
            line, hit = workspace.coordinate_line(
                axis, value, extent, patch, data_tensor.device
            )
            axis_lines[axis].append(line)
            hits += int(hit)
            misses += int(not hit)

    xv = torch.stack(axis_lines[0], dim=0)
    yv = torch.stack(axis_lines[1], dim=0)
    zv = torch.stack(axis_lines[2], dim=0)
    output[:, 0].copy_(data_tensor[:, 0])
    output[:, 1].copy_(xv[:, None, None, :].expand(batch, patch, patch, patch))
    output[:, 2].copy_(yv[:, None, :, None].expand(batch, patch, patch, patch))
    output[:, 3].copy_(zv[:, :, None, None].expand(batch, patch, patch, patch))
    if use_dist:
        output[:, 4].copy_(data_tensor[:, 1])
    return output, int(not allocated), hits, misses


class _NoopCudaEvent:
    def record(self) -> None:
        return None


_NOOP_EVENT_PAIR = (_NoopCudaEvent(), _NoopCudaEvent())


def _event_pair(enabled: bool = True):
    if not enabled:
        return _NOOP_EVENT_PAIR
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _event_ms(pair) -> float:
    if pair is _NOOP_EVENT_PAIR:
        return float("nan")
    return float(pair[0].elapsed_time(pair[1]))


def _predict_active_gpu_opt(
    item: Dict,
    model,
    cfg: Dict,
    calibration: Dict,
    grid_size,
    core_size: int,
    batch_size: int,
    amp: str,
    channels_last: bool,
    fixed_batch_shape: bool,
    workspace: V4SparseGpuWorkspace,
    pinned_d2h: bool,
    require_compiled: bool,
    detailed_cuda_timing: bool,
    retain_gather_host_buffers: bool,
    precompute_batch_gather_plans: bool,
    cache_coordinate_channels: bool,
):
    if not torch.cuda.is_available():
        raise RuntimeError("optimized V4 Stage 1 requires CUDA")
    gx, gy, gz = map(int, grid_size)
    patch = int(cfg.get("patch_size", 64))
    pad = (patch - int(core_size)) // 2
    if patch < int(core_size) or (patch - int(core_size)) % 2:
        raise ValueError("patch_size-core_size must be non-negative and even")
    sl = slice(pad, pad + int(core_size))
    use_coord = bool(int(cfg.get("use_coord_channels", cfg.get("use_coord", 1))))
    use_dist = bool(int(cfg.get("use_dist", 1)))

    schedule_t0 = time.perf_counter()
    batch_gather_plans = None
    if precompute_batch_gather_plans:
        groups, batch_gather_plans = active_core_schedule_with_batch_plans_opt(
            item["coords"], grid_size, core_size, batch_size
        )
    else:
        groups = active_core_groups_opt(item["coords"], grid_size, core_size)
        reference._prepare_group_gather(groups, item["coords"], core_size)
    schedule_ms = (time.perf_counter() - schedule_t0) * 1000.0
    n = len(item["coords"])
    timing = {
        "optimization_version": OPTIMIZATION_VERSION,
        "active_cores": len(groups),
        "total_possible_cores": int(math.ceil(gx/core_size)*math.ceil(gy/core_size)*math.ceil(gz/core_size)),
        "core_schedule_ms": schedule_ms,
        "patch_build_ms": 0.0,
        "host_batch_pack_ms": 0.0,
        "host_pin_ms": 0.0,
        "workspace_prepare_ms": 0.0,
        "gpu_workspace_reset_ms": 0.0,
        "h2d_ms": 0.0,
        "sparse_h2d_cuda_ms": 0.0,
        "gpu_sparse_scatter_ms": 0.0,
        "gpu_patch_extract_ms": 0.0,
        "gpu_feature_assembly_ms": 0.0,
        "gpu_model_ms": 0.0,
        "gpu_gather_plan_ms": 0.0,
        "gpu_gather_ms": 0.0,
        "d2h_gather_ms": 0.0,
        "cuda_sync_count": 1,
        "gpu_volume_mode": 1,
        "workspace_reused": 0,
        "host_output_reused": 0,
        "pinned_d2h": int(bool(pinned_d2h)),
        "detailed_cuda_timing": int(bool(detailed_cuda_timing)),
        "retain_gather_host_buffers": int(bool(retain_gather_host_buffers)),
        "precompute_batch_gather_plans": int(bool(precompute_batch_gather_plans)),
        "cache_coordinate_channels": int(bool(cache_coordinate_channels)),
        "coordinate_cache_hits": 0,
        "coordinate_cache_misses": 0,
        "model_input_reused": 0,
        "fixed_batch_shape": int(bool(fixed_batch_shape)),
        "batches": 0,
    }
    if n == 0:
        return {
            "pole": np.zeros(0, np.float32),
            "line": np.zeros(0, np.float32),
            "semantic": np.zeros(0, np.uint8),
            "objectness": np.zeros(0, np.float32),
            "timing": timing,
        }

    channels = 2 if use_dist else 1
    device = torch.device("cuda")
    reset_event = _event_pair(detailed_cuda_timing)
    reset_event[0].record()
    prepare_t0 = time.perf_counter()
    data_volume, low_pad, previous_coords, allocated = workspace.acquire(
        grid_size, patch, core_size, channels, device
    )
    timing["workspace_prepare_ms"] = (time.perf_counter() - prepare_t0) * 1000.0
    timing["workspace_reused"] = int(not allocated)
    if previous_coords is not None and len(previous_coords):
        px = previous_coords[:, 0] + int(low_pad[0])
        py = previous_coords[:, 1] + int(low_pad[1])
        pz = previous_coords[:, 2] + int(low_pad[2])
        data_volume[:, pz, py, px] = 0.0
    reset_event[1].record()

    pin_t0 = time.perf_counter()
    coords_np = np.asarray(item["coords"], dtype=np.int64)
    dist_np = np.asarray(item["dist_values"], dtype=np.float32)
    coords_host, pinned_coords = reference._pin_numpy_tensor(coords_np, torch.int64)
    dist_host = None
    pinned_dist = False
    if use_dist:
        dist_host, pinned_dist = reference._pin_numpy_tensor(dist_np, torch.float32)
    timing["host_pin_ms"] = (time.perf_counter() - pin_t0) * 1000.0
    timing["host_pinned"] = int(bool(pinned_coords and (pinned_dist if use_dist else True)))

    h2d = _event_pair(detailed_cuda_timing)
    h2d[0].record()
    coords_gpu = coords_host.to(device, non_blocking=bool(pinned_coords))
    dist_gpu = dist_host.to(device, non_blocking=bool(pinned_dist)) if use_dist else None
    h2d[1].record()

    scatter = _event_pair(detailed_cuda_timing)
    scatter[0].record()
    xx = coords_gpu[:, 0] + int(low_pad[0])
    yy = coords_gpu[:, 1] + int(low_pad[1])
    zz = coords_gpu[:, 2] + int(low_pad[2])
    data_volume[0, zz, yy, xx] = 1.0
    if use_dist:
        data_volume[1, zz, yy, xx] = dist_gpu
    scatter[1].record()
    # Record immediately so a later exception cannot leave uncleared live voxels.
    workspace.remember(coords_gpu)

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
    # The gather indices are copied asynchronously from pinned host memory.
    # E3 retains those source tensors until the existing end-of-slice CUDA
    # synchronization instead of allowing the pinned allocator to reclaim them
    # between batches.  CUDA operations, model execution and output decisions
    # remain unchanged.
    gather_host_refs = []

    for batch_index, start in enumerate(range(0, len(groups), int(batch_size))):
        bg = groups[start:start + int(batch_size)]
        real_count = len(bg)
        if not real_count:
            continue
        centers = [np.asarray(g["center"], dtype=np.int64) for g in bg]

        pe = _event_pair(detailed_cuda_timing)
        pe[0].record()
        patches = []
        for center in centers:
            sx = int(center[0]) - patch // 2 + int(low_pad[0])
            sy = int(center[1]) - patch // 2 + int(low_pad[1])
            sz = int(center[2]) - patch // 2 + int(low_pad[2])
            q = data_volume[:, sz:sz+patch, sy:sy+patch, sx:sx+patch]
            if tuple(q.shape[-3:]) != (patch, patch, patch):
                raise RuntimeError(
                    f"optimized GPU patch shape {tuple(q.shape)} for center {center.tolist()}"
                )
            patches.append(q)
        xb = torch.stack(patches, dim=0)
        padded_centers = list(centers)
        if fixed_batch_shape and real_count < int(batch_size):
            need = int(batch_size) - real_count
            xb = torch.cat([xb, xb[-1:].expand(need, -1, -1, -1, -1)], dim=0)
            padded_centers.extend([centers[-1]] * need)
        pe[1].record()
        patch_events.append(pe)

        fe = _event_pair(detailed_cuda_timing)
        fe[0].record()
        if cache_coordinate_channels:
            xb, input_reused, cache_hits, cache_misses = assemble_v4_channels_cached_opt(
                xb, padded_centers, grid_size, patch, use_coord, use_dist, workspace
            )
            timing["model_input_reused"] += int(input_reused)
            timing["coordinate_cache_hits"] += int(cache_hits)
            timing["coordinate_cache_misses"] += int(cache_misses)
        elif use_coord:
            xb = reference.assemble_v4_channels_gpu(
                xb, padded_centers, grid_size, patch, use_coord, use_dist
            )
        if channels_last and not cache_coordinate_channels:
            xb = xb.contiguous(memory_format=torch.channels_last_3d)
        fe[1].record()
        feature_events.append(fe)

        me = _event_pair(detailed_cuda_timing)
        me[0].record()
        model, ps, ls, sem, obj = reference._run_model_scores(
            model, xb, calibration, amp, fallback_state
        )
        if require_compiled and fallback_state.get("done", False):
            raise RuntimeError("compiled execution failed and attempted eager fallback")
        me[1].record()
        model_events.append(me)

        gather_plan_t0 = time.perf_counter()
        if batch_gather_plans is not None:
            take_np, dest_np = batch_gather_plans[batch_index]
        else:
            offsets = []
            dest_rows = []
            for bi, group in enumerate(bg):
                rr = np.asarray(group["rows"], dtype=np.int64)
                if len(rr):
                    offsets.append(group["_flat_core"] + bi * core_vol)
                    dest_rows.append(rr)
            take_np = np.concatenate(offsets).astype(np.int64, copy=False)
            dest_np = np.concatenate(dest_rows).astype(np.int64, copy=False)
        timing["gpu_gather_plan_ms"] += (time.perf_counter() - gather_plan_t0) * 1000.0

        ge = _event_pair(detailed_cuda_timing)
        ge[0].record()
        if len(take_np):
            take_host, take_pinned = reference._pin_numpy_tensor(take_np, torch.int64)
            dest_host, dest_pinned = reference._pin_numpy_tensor(dest_np, torch.int64)
            if retain_gather_host_buffers:
                gather_host_refs.extend((take_host, dest_host))
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
        ge[1].record()
        gather_events.append(ge)
        timing["batches"] += 1

    d2h_t0 = time.perf_counter()
    score_device = torch.stack([pole_gpu, line_gpu, objectness_gpu], dim=1)
    if pinned_d2h:
        score_host, semantic_host, allocated = workspace.acquire_host_outputs(n)
        score_host.copy_(score_device, non_blocking=True)
        semantic_host.copy_(semantic_gpu, non_blocking=True)
        torch.cuda.synchronize()
        # Detach returned arrays from reusable staging before the next slice.
        score_cpu = score_host.numpy().copy()
        semantic_cpu = semantic_host.numpy().copy()
        timing["host_output_reused"] = int(not allocated)
    else:
        score_cpu = score_device.cpu().numpy()
        semantic_cpu = semantic_gpu.cpu().numpy()
        torch.cuda.synchronize()
    # All asynchronous index copies are complete after the synchronization
    # above, so E3 can now release the retained pinned host tensors safely.
    gather_host_refs.clear()
    timing["d2h_gather_ms"] = (time.perf_counter() - d2h_t0) * 1000.0
    timing["gpu_workspace_reset_ms"] = _event_ms(reset_event)
    timing["sparse_h2d_cuda_ms"] = _event_ms(h2d)
    timing["h2d_ms"] = timing["sparse_h2d_cuda_ms"]
    timing["gpu_sparse_scatter_ms"] = _event_ms(scatter)
    timing["gpu_patch_extract_ms"] = float(sum(_event_ms(x) for x in patch_events))
    timing["gpu_feature_assembly_ms"] = float(sum(_event_ms(x) for x in feature_events))
    timing["gpu_model_ms"] = float(sum(_event_ms(x) for x in model_events))
    timing["gpu_gather_ms"] = float(sum(_event_ms(x) for x in gather_events))

    if sum(len(group["rows"]) for group in groups) != n:
        raise RuntimeError("optimized V4 core scheduling did not cover every occupied row")
    return {
        "pole": score_cpu[:, 0].astype(np.float32, copy=False),
        "line": score_cpu[:, 1].astype(np.float32, copy=False),
        "semantic": semantic_cpu.astype(np.uint8, copy=False),
        "objectness": score_cpu[:, 2].astype(np.float32, copy=False),
        "timing": timing,
    }


def predict_v4_sparse_rows_opt(
    item: Dict,
    model,
    cfg: Dict,
    calibration: Dict,
    grid_size=(400, 400, 200),
    core_size: int = 48,
    batch_size: int = 12,
    amp: str = "bf16",
    channels_last: bool = True,
    evaluate_all_cores: bool = False,
    gpu_coord_channels: bool = True,
    fixed_batch_shape: bool = True,
    workspace: V4SparseGpuWorkspace | None = None,
    pinned_d2h: bool = False,
    require_compiled: bool = False,
    detailed_cuda_timing: bool = True,
    retain_gather_host_buffers: bool = False,
    precompute_batch_gather_plans: bool = False,
    cache_coordinate_channels: bool = False,
):
    """Run an isolated active-GPU candidate under production-output equivalence."""
    if evaluate_all_cores or not gpu_coord_channels:
        raise ValueError("opt2 is restricted to accepted active_gpu mode")
    if int(batch_size) != 12:
        raise ValueError("production-preserving opt2 fixes batch_size=12")
    if not channels_last:
        raise ValueError("production-preserving opt2 fixes channels_last=1")
    if not fixed_batch_shape:
        raise ValueError("production-preserving opt2 requires fixed_batch_shape=1")
    if amp != "bf16":
        raise ValueError("production-preserving opt2 fixes amp=bf16")
    if workspace is None:
        workspace = V4SparseGpuWorkspace()
    return _predict_active_gpu_opt(
        item, model, cfg, calibration, grid_size, int(core_size), int(batch_size), amp,
        bool(channels_last), bool(fixed_batch_shape), workspace,
        bool(pinned_d2h), bool(require_compiled), bool(detailed_cuda_timing),
        bool(retain_gather_host_buffers), bool(precompute_batch_gather_plans),
        bool(cache_coordinate_channels),
    )
