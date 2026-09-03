#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import joblib
import numpy as np

from v4_stage2_local import LOCAL_FEATURE_COLUMNS
from v4_stage2_bundle_consensus import (
    Stage1LabelProfile,
    Stage1BundleConsensusStage2Processor,
    _sibling_support,
)


class FixedProbabilityModel:
    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        p = np.full(len(values), self.probability, dtype=np.float64)
        return np.column_stack([1.0 - p, p])


def accepted_points(result: dict) -> set[tuple[int, int, int]]:
    frame = result["components"]
    points = result["raw_components"]["line_points"]
    out: set[tuple[int, int, int]] = set()
    rows = frame[
        frame["class_name"].astype(str).eq("line")
        & frame["component_accept"].astype(bool)
    ]
    for cid in rows["component_id"].astype(str):
        for p in np.asarray(points[cid], dtype=int):
            out.add(tuple(map(int, p)))
    return out


def main() -> None:
    grid = (80, 80, 60)
    voxel = 0.5
    lower_a = np.array([[x, 20, 20] for x in range(10, 51)], dtype=np.int32)
    lower_b = np.array([[x, 20, 22] for x in range(10, 51)], dtype=np.int32)
    upper = np.array([[x, 20, 24] for x in range(10, 51)], dtype=np.int32)
    coords = np.unique(np.concatenate([lower_a, lower_b, upper], axis=0), axis=0)

    # Two production-accepted parallel lanes establish bundle consensus. Upper
    # is a valid deployed Stage1 line label but below the production strong
    # threshold, so it cannot seed production hysteresis by itself.
    strong = {tuple(x) for x in np.concatenate([lower_a, lower_b], axis=0)}
    line = np.array([
        0.90 if tuple(p) in strong else 0.05
        for p in coords
    ], dtype=np.float32)
    pole = np.zeros(len(coords), dtype=np.float32)
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
        "pole": pole,
        "line": line,
        "semantic": np.zeros(len(coords), dtype=np.uint8),
        "objectness": np.ones(len(coords), dtype=np.float32),
        "timing": {},
    }

    profile = Stage1LabelProfile(
        name="bundle_consensus_self_test",
        cross_section_eps_ft=0.55,
        cross_section_min_samples=2,
        noise_attach_max_ft=0.65,
        max_internal_gap_ft=0.50,
        min_longitudinal_coverage=0.90,
        min_horizontal_length_ft=4.0,
        min_linearity=0.90,
        max_radius_p90_ft=0.70,
        max_tortuosity=1.40,
        residual_min_novel_voxels=5,
        residual_min_novel_fraction=0.80,
        residual_max_baseline_overlap_fraction=0.20,
        require_sibling_support=True,
        sibling_max_axis_angle_deg=4.0,
        sibling_min_cross_section_offset_ft=1.0,
        sibling_max_cross_section_offset_ft=4.0,
        sibling_min_longitudinal_overlap_fraction=0.80,
        bundle_min_parallel_siblings=2,
        bundle_min_endpoint_overlap_fraction=0.90,
        bundle_max_endpoint_extension_ft=1.0,
        bundle_spacing_ratio_min=0.75,
        bundle_spacing_ratio_max=1.25,
    )

    baseline_bundle = {"L00001": lower_a, "L00002": lower_b}
    sibling = _sibling_support(upper, baseline_bundle, profile, voxel)
    if not sibling.get("supported"):
        raise AssertionError(f"parallel sibling support failed: {sibling}")
    if int(sibling.get("bundle_sibling_count", 0)) < 2:
        raise AssertionError(f"bundle sibling count failed: {sibling}")
    diagonal = np.array([[20 + i, 20 + i, 24] for i in range(12)], dtype=np.int32)
    if _sibling_support(diagonal, baseline_bundle, profile, voxel).get("supported"):
        raise AssertionError("crossing branch incorrectly received sibling support")

    with tempfile.TemporaryDirectory(prefix="v4-bundle-consensus-test-") as temp:
        root = Path(temp)
        calibration = root / "calibration.json"
        bundle = root / "bundle.joblib"
        calibration.write_text(json.dumps({"pole_threshold": 0.20, "line_threshold": 0.03}))
        joblib.dump({
            "pole_model": FixedProbabilityModel(0.95),
            "line_model": FixedProbabilityModel(0.95),
            "pole_threshold": 0.50,
            "line_threshold": 0.50,
            "feature_columns": list(LOCAL_FEATURE_COLUMNS),
        }, bundle)

        processor = Stage1BundleConsensusStage2Processor(
            str(bundle), str(calibration), profile, grid, voxel
        )
        result = processor.process(item, pred, "synthetic", 1)
        audit = result["stage1_label_audit"]
        if not audit["production_voxel_preserved"]:
            raise AssertionError(audit)
        if audit["residual_accepted_components"] < 1:
            raise AssertionError(f"weak parallel residual lane not recovered: {audit}")
        if audit["residual_accepted_novel_voxels"] < len(upper):
            raise AssertionError(f"not all novel upper-lane voxels recovered: {audit}")
        accepted = accepted_points(result)
        expected = {tuple(map(int, p)) for p in np.concatenate([lower_a, lower_b, upper], axis=0)}
        if not expected.issubset(accepted):
            raise AssertionError("final result did not preserve baseline and residual lanes")
        if audit["runtime_gt_usage"] is not False or audit["synthetic_line_voxels"] != 0:
            raise AssertionError(audit)

        # Runtime result must not depend on GT labels.
        no_gt = dict(item)
        no_gt["raw_labels"] = np.zeros(len(coords), dtype=np.int16)
        result_no_gt = processor.process(no_gt, pred, "synthetic", 1)
        if accepted_points(result_no_gt) != accepted:
            raise AssertionError("runtime result changed when only GT changed")

    print("V4_STAGE2_BUNDLE_CONSENSUS_SELF_TEST_OK")


if __name__ == "__main__":
    main()
