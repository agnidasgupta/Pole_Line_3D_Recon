#!/usr/bin/env python3
"""Deterministic scheduler, batch grouping, and D2H regression tests for Opt2."""
from __future__ import annotations

import numpy as np
import torch

import v4_realtime_core as reference
from v4_realtime_core_opt2 import (
    V4SparseGpuWorkspace,
    active_core_groups_opt,
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
    assert actual["timing"]["workspace_reused"] == 1
    assert actual["timing"]["pinned_d2h"] == 1
    print("V4_STAGE1_OPT2_SELF_TEST_OK")


if __name__ == "__main__":
    main()
