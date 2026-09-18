#!/usr/bin/env python3
"""Deterministic scheduler, batch grouping, and D2H regression tests for Opt2."""
from __future__ import annotations

import numpy as np
import torch

import v4_realtime_core as reference
from v4_realtime_core_opt2 import (
    V4SparseGpuWorkspace,
    active_core_groups_opt,
    active_core_schedule_with_batch_plans_opt,
    assemble_v4_channels_cached_opt,
    exact_padding,
    predict_v4_sparse_rows_opt,
)


def assert_groups_equal(coords, grid, core):
    left = reference.active_core_groups(coords, grid, core)
    right = active_core_groups_opt(coords, grid, core)
    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert a["key"] == b["key"]
        assert np.array_equal(a["origin"], b["origin"])
        assert np.array_equal(a["center"], b["center"])
        assert np.array_equal(a["rows"], b["rows"])


def assert_e4_plans_equal(coords, grid, core, batch_size=12):
    expected = reference.active_core_groups(coords, grid, core)
    reference._prepare_group_gather(expected, coords, core)
    actual, plans = active_core_schedule_with_batch_plans_opt(
        coords, grid, core, batch_size
    )
    assert len(expected) == len(actual)
    assert len(plans) == (len(expected) + batch_size - 1) // batch_size
    core_vol = int(core) ** 3
    for batch_index, start in enumerate(range(0, len(expected), batch_size)):
        batch = expected[start:start + batch_size]
        expected_take = np.concatenate(
            [group["_flat_core"] + index * core_vol for index, group in enumerate(batch)]
        ).astype(np.int64, copy=False)
        expected_dest = np.concatenate(
            [np.asarray(group["rows"], dtype=np.int64) for group in batch]
        ).astype(np.int64, copy=False)
        actual_take, actual_dest = plans[batch_index]
        assert np.array_equal(expected_take, actual_take)
        assert np.array_equal(expected_dest, actual_dest)


class DummyModel(torch.nn.Module):
    def forward(self, x):
        occ = x[:, 0]
        xcoord = x[:, 1]
        ycoord = x[:, 2]
        zcoord = x[:, 3]
        dist = x[:, 4]
        semantic = torch.stack(
            [-(occ + dist), 2.0 * occ + 0.125 * xcoord, occ + 0.25 * ycoord],
            dim=1,
        )
        return {
            "semantic": semantic,
            "pole": (occ + 0.125 * xcoord).unsqueeze(1),
            "line": (dist + 0.25 * ycoord).unsqueeze(1),
            "objectness": (occ + 0.0625 * zcoord).unsqueeze(1),
        }


def item_for(coords, seed):
    rng = np.random.default_rng(seed)
    coords = np.asarray(coords, dtype=np.int32)
    order = np.argsort(coords[:, 2], kind="stable")
    coords = coords[order]
    return {
        "coords": coords,
        "dist_values": rng.normal(size=len(coords)).astype(np.float32)[order],
    }


def main():
    rng = np.random.default_rng(20260914)
    grid = (97, 83, 65)
    for count in (0, 1, 2, 31, 257, 4096):
        coords = np.column_stack(
            [rng.integers(0, grid[axis], size=count) for axis in range(3)]
        ).astype(np.int32)
        assert_groups_equal(coords, grid, 48)
        assert_e4_plans_equal(coords, grid, 48)

    low, high = exact_padding((400, 400, 200), 64, 48)
    assert tuple(low) == (8, 8, 8), (low, high)
    assert tuple(high) == (40, 40, 48), (low, high)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Stage1 opt2 self-test")
    model = DummyModel().cuda().eval()
    cfg = {"patch_size": 64, "use_coord_channels": 1, "use_dist": 1}
    calibration = {
        "score_sem_weight": 0.55,
        "score_binary_weight": 0.35,
        "score_object_weight": 0.10,
    }
    workspace = V4SparseGpuWorkspace()
    e3_workspace = V4SparseGpuWorkspace()
    e4_workspace = V4SparseGpuWorkspace()
    e5_workspace = V4SparseGpuWorkspace()
    assembly_workspace = V4SparseGpuWorkspace()
    assembly_data = torch.randn(
        (12, 2, 64, 64, 64), device="cuda", dtype=torch.float32
    )
    assembly_centers = [
        np.asarray([24 + 48 * (index % 4), 24 + 48 * ((index // 4) % 3), 24])
        for index in range(12)
    ]
    expected_input = reference.assemble_v4_channels_gpu(
        assembly_data, assembly_centers, (400, 400, 200), 64, True, True
    ).contiguous(memory_format=torch.channels_last_3d)
    actual_input, reused, _, misses = assemble_v4_channels_cached_opt(
        assembly_data, assembly_centers, (400, 400, 200), 64, True, True,
        assembly_workspace,
    )
    assert torch.equal(expected_input, actual_input)
    assert actual_input.is_contiguous(memory_format=torch.channels_last_3d)
    assert reused == 0 and misses > 0
    actual_input, reused, _, misses = assemble_v4_channels_cached_opt(
        assembly_data, assembly_centers, (400, 400, 200), 64, True, True,
        assembly_workspace,
    )
    assert torch.equal(expected_input, actual_input)
    assert reused == 1 and misses == 0
    boundary = np.asarray(
        [[0, 0, 0], [96, 82, 64], [48, 48, 48], [80, 60, 20], [7, 75, 63]],
        dtype=np.int32,
    )
    for seed in (1, 2):
        random_coords = np.column_stack(
            [rng.integers(0, grid[axis], size=300) for axis in range(3)]
        ).astype(np.int32)
        item = item_for(np.vstack([boundary, random_coords]), seed)
        expected = reference.predict_v4_sparse_rows(
            item, model, cfg, calibration, grid_size=grid, core_size=48,
            batch_size=12, amp="bf16", evaluate_all_cores=False,
            gpu_coord_channels=True, fixed_batch_shape=True,
        )
        actual = predict_v4_sparse_rows_opt(
            item, model, cfg, calibration, grid_size=grid, core_size=48,
            batch_size=12, amp="bf16", evaluate_all_cores=False,
            gpu_coord_channels=True, fixed_batch_shape=True, workspace=workspace,
            pinned_d2h=True,
        )
        for name in ("pole", "line", "objectness"):
            delta = float(np.max(np.abs(expected[name] - actual[name]), initial=0.0))
            assert delta == 0.0, (seed, name, delta)
        assert np.array_equal(expected["semantic"], actual["semantic"])
        e3 = predict_v4_sparse_rows_opt(
            item, model, cfg, calibration, grid_size=grid, core_size=48,
            batch_size=12, amp="bf16", evaluate_all_cores=False,
            gpu_coord_channels=True, fixed_batch_shape=True,
            workspace=e3_workspace, pinned_d2h=False,
            detailed_cuda_timing=True, retain_gather_host_buffers=True,
        )
        for name in ("pole", "line", "objectness"):
            delta = float(np.max(np.abs(expected[name] - e3[name]), initial=0.0))
            assert delta == 0.0, (seed, "e3", name, delta)
        assert np.array_equal(expected["semantic"], e3["semantic"])
        assert e3["timing"]["retain_gather_host_buffers"] == 1
        e4 = predict_v4_sparse_rows_opt(
            item, model, cfg, calibration, grid_size=grid, core_size=48,
            batch_size=12, amp="bf16", evaluate_all_cores=False,
            gpu_coord_channels=True, fixed_batch_shape=True,
            workspace=e4_workspace, pinned_d2h=False,
            detailed_cuda_timing=True, retain_gather_host_buffers=False,
            precompute_batch_gather_plans=True,
        )
        for name in ("pole", "line", "objectness"):
            delta = float(np.max(np.abs(expected[name] - e4[name]), initial=0.0))
            assert delta == 0.0, (seed, "e4", name, delta)
        assert np.array_equal(expected["semantic"], e4["semantic"])
        assert e4["timing"]["precompute_batch_gather_plans"] == 1
        e5 = predict_v4_sparse_rows_opt(
            item, model, cfg, calibration, grid_size=grid, core_size=48,
            batch_size=12, amp="bf16", evaluate_all_cores=False,
            gpu_coord_channels=True, fixed_batch_shape=True,
            workspace=e5_workspace, pinned_d2h=False,
            detailed_cuda_timing=True, retain_gather_host_buffers=False,
            precompute_batch_gather_plans=False,
            cache_coordinate_channels=True,
        )
        for name in ("pole", "line", "objectness"):
            delta = float(np.max(np.abs(expected[name] - e5[name]), initial=0.0))
            assert delta == 0.0, (seed, "e5", name, delta)
        assert np.array_equal(expected["semantic"], e5["semantic"])
        assert e5["timing"]["cache_coordinate_channels"] == 1
    assert actual["timing"]["workspace_reused"] == 1
    assert actual["timing"]["pinned_d2h"] == 1
    print("V4_STAGE1_OPT2_SELF_TEST_OK")


if __name__ == "__main__":
    main()
