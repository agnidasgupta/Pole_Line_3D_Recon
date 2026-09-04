#!/usr/bin/env python3
from __future__ import annotations

import math
import numpy as np

from v4_stage2_stage1_exact import exact_adjacency_edges, build_exact_line_outputs


def main() -> None:
    grid = (20, 20, 20)
    # A true inferred line chain plus two poles with no line voxels between them.
    coords = np.array([
        [1, 1, 1], [2, 1, 1], [3, 2, 1], [4, 3, 2],
        [10, 10, 1], [15, 10, 1],
        [8, 8, 8],
    ], dtype=np.int32)
    labels = np.array([2, 2, 2, 2, 1, 1, 2], dtype=np.int8)
    line_scores = np.array([.9, .8, .85, .75, .1, .1, .7], dtype=np.float32)
    mask = labels == 2
    line_idx, ei, ej = exact_adjacency_edges(coords, mask, grid)
    assert len(line_idx) == 5
    # Chain contains three adjacent edges; isolated [8,8,8] has none.
    assert len(ei) == 3, (ei, ej)
    # No edge may connect the two poles because they are not line-labelled.
    pole_indices = {4, 5}
    assert not any(int(a) in pole_indices or int(b) in pole_indices for a, b in zip(ei, ej))

    out = build_exact_line_outputs(coords, line_scores, labels, "x", 0, grid, .5)
    audit = out["audit"]
    assert audit["stage1_inferred_line_voxels"] == 5
    assert audit["accepted_stage1_line_voxels"] == 5
    assert audit["stage1_to_stage2_voxel_preservation"] == 1.0
    assert audit["isolated_stage1_line_voxels"] == 1
    assert audit["synthetic_line_voxels"] == 0
    assert audit["pole_pair_inference"] is False
    assert audit["line_hysteresis_used"] is False
    assert audit["line_refiner_used"] is False
    assert audit["max_exact_edge_step_ft"] <= math.sqrt(3.0) * .5 + 1e-9
    print("V4_STAGE2_STAGE1_EXACT_SELF_TEST_OK")


if __name__ == "__main__":
    main()
