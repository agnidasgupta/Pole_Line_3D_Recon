#!/usr/bin/env python3
"""Deterministic tests for deployed-Stage1-label Stage2 reconstruction."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import joblib
import numpy as np

from v4_stage2_local import LOCAL_FEATURE_COLUMNS
from v4_stage2_stage1_label import (
    Stage1LabelProfile,
    Stage1LabelStage2Processor,
    _fit_axis,
    _split_longitudinal_runs,
    deployed_stage1_labels,
)


class FixedProbabilityModel:
    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        n = len(values)
        p = np.full(n, self.probability, dtype=np.float64)
        return np.column_stack([1.0 - p, p])


def coordinate_set(result: dict) -> set[tuple[int, int, int]]:
    frame = result["components"]
    accepted: set[tuple[int, int, int]] = set()
    if frame.empty:
        return accepted
    line_rows = frame[
        frame["class_name"].astype(str).eq("line")
        & frame["component_accept"].astype(bool)
    ]
    for component_id in line_rows["component_id"].astype(str):
        for point in result["raw_components"]["line_points"][component_id]:
            accepted.add(tuple(map(int, point)))
    return accepted


def main() -> None:
    grid = (80, 80, 60)
    voxel = 0.5

    lower = np.array([[x, 20, 20] for x in range(10, 51)], dtype=np.int32)
    upper = np.array([[x, 20, 24] for x in range(10, 51)], dtype=np.int32)
    # A short vertical bridge merges both conductors under exact 26-connectivity.
    bridge = np.array([[30, 20, z] for z in (21, 22, 23)], dtype=np.int32)
    coords = np.unique(np.concatenate([lower, upper, bridge], axis=0), axis=0)

    pole_score = np.zeros(len(coords), dtype=np.float32)
    line_score = np.full(len(coords), 0.90, dtype=np.float32)
    semantic = np.zeros(len(coords), dtype=np.uint8)
    # Semantic-head argmax intentionally disagrees at one true fused-line voxel.
    semantic[0] = 2

    item = {
        "sparse_native": True,
        "coords": coords,
        "dist_values": np.zeros(len(coords), dtype=np.float32),
        "source_rows": np.arange(len(coords), dtype=np.int64),
        "raw_labels": np.full(len(coords), 2, dtype=np.int16),
        "raw_hardneg": np.zeros(len(coords), dtype=np.uint8),
        "z_sorted": coords[:, 2],
        "has_gt": True,
    }
    pred = {
        "pole": pole_score,
        "line": line_score,
        "semantic": semantic,
        "objectness": np.ones(len(coords), dtype=np.float32),
        "timing": {},
    }

    profile = Stage1LabelProfile(
        name="self_test",
        cross_section_eps_ft=0.55,
        cross_section_min_samples=2,
        cross_section_quantization_ft=0.25,
        mode_min_longitudinal_bins=4,
        mode_min_longitudinal_fraction=0.15,
        noise_attach_max_ft=0.65,
        max_internal_gap_ft=0.50,
        min_longitudinal_coverage=0.90,
        min_horizontal_length_ft=4.0,
        min_linearity=0.90,
        max_radius_p90_ft=0.70,
        max_tortuosity=1.40,
        enable_geometry_override=True,
        override_refiner_floor=0.05,
        override_min_voxels=5,
        override_min_horizontal_length_ft=4.0,
        override_min_longitudinal_coverage=0.90,
        override_max_internal_gap_ft=0.50,
        override_min_linearity=0.90,
        override_max_radius_p90_ft=0.70,
        override_max_tortuosity=1.40,
    )

    with tempfile.TemporaryDirectory(prefix="v4-stage1-label-test-") as temp:
        root = Path(temp)
        calibration_path = root / "calibration.json"
        bundle_path = root / "bundle.joblib"
        calibration_path.write_text(json.dumps({"pole_threshold": 0.20, "line_threshold": 0.20}))
        joblib.dump(
            {
                "pole_model": FixedProbabilityModel(0.95),
                "line_model": FixedProbabilityModel(0.95),
                "pole_threshold": 0.50,
                "line_threshold": 0.50,
                "feature_columns": list(LOCAL_FEATURE_COLUMNS),
            },
            bundle_path,
        )

        processor = Stage1LabelStage2Processor(
            str(bundle_path), str(calibration_path), profile, grid, voxel
        )
        result = processor.process(item, pred, "synthetic", 1)

        audit = result["stage1_label_audit"]
        if audit["raw_stage1_line_components"] != 1:
            raise AssertionError(audit)
        if audit["raw_components_with_multiple_lanes"] < 1:
            raise AssertionError(f"parallel lanes were not decomposed: {audit}")
        if audit["accepted_line_components"] < 2:
            raise AssertionError(f"expected at least two accepted lanes: {audit}")
        if audit["runtime_gt_usage"] is not False or audit["synthetic_line_voxels"] != 0:
            raise AssertionError(audit)

        accepted = coordinate_set(result)
        inferred = {tuple(map(int, point)) for point in coords}
        if not accepted.issubset(inferred):
            raise AssertionError("accepted output contains a synthetic coordinate")

        # Ground truth is deliberately changed while scores/coordinates stay fixed.
        # Runtime Stage2 output must remain identical.
        item_no_gt = dict(item)
        item_no_gt["raw_labels"] = np.zeros(len(coords), dtype=np.int16)
        item_no_gt["has_gt"] = True
        result_no_gt = processor.process(item_no_gt, pred, "synthetic", 1)
        if coordinate_set(result_no_gt) != accepted:
            raise AssertionError("runtime Stage2 output changed when only GT changed")

        # The deployed fused decision, not semantic-head argmax, is authoritative.
        modified = {key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value for key, value in pred.items()}
        modified["line"][0] = 0.01
        modified["semantic"][0] = 2
        labels = deployed_stage1_labels(modified, {"pole_threshold": 0.20, "line_threshold": 0.20})
        if labels[0] == 2:
            raise AssertionError("semantic-head label incorrectly overrode deployed fused decision")
        modified["line"][1] = 0.90
        modified["semantic"][1] = 0
        labels = deployed_stage1_labels(modified, {"pole_threshold": 0.20, "line_threshold": 0.20})
        if labels[1] != 2:
            raise AssertionError("deployed fused line decision was not preserved")

        # Unsupported longitudinal gaps are split rather than rendered as one line.
        gap_points = np.array(
            [[x, 10, 10] for x in range(0, 11)]
            + [[x, 10, 10] for x in range(16, 27)],
            dtype=np.int32,
        )
        axis = _fit_axis(gap_points)
        if axis is None:
            raise AssertionError("axis fit failed")
        runs = _split_longitudinal_runs(
            np.arange(len(gap_points)), gap_points, axis, profile, voxel
        )
        if len(runs) != 2:
            raise AssertionError(f"unsupported gap was not split: runs={len(runs)}")

    print("V4_STAGE2_STAGE1_LABEL_SELF_TEST_OK")


if __name__ == "__main__":
    main()
