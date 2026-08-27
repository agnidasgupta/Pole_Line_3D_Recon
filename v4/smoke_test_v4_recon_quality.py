#!/usr/bin/env python3
"""Focused smoke tests for the first V4 reconstruction-quality patch."""
from types import SimpleNamespace
import numpy as np
import pandas as pd

import reconstruct_v4_stage3 as s3


def args():
    return SimpleNamespace(
        disable_span_completion=False,
        span_completion=True,
        min_pole_separation_ft=10.0,
        max_span_length_ft=450.0,
        span_completion_corridor_ft=10.0,
        span_completion_pole_extrap_ft=35.0,
        span_completion_max_angle_deg=12.0,
        span_completion_min_tracks=2,
        span_completion_min_coverage=0.15,
        span_completion_single_track_min_coverage=0.30,
        span_completion_max_connector_angle_deg=15.0,
        span_completion_max_lateral_ft=1.5,
        span_completion_max_z_error_ft=3.0,
        span_completion_max_gap_ft=100.0,
        max_longitudinal_overlap_ft=3.0,
        max_span_slices=9,
        max_pole_attachment_angle_deg=40.0,
        max_pole_height_adjust_ft=8.0,
        pole_attachment_height_slack_ft=1.5,
        pole_attachment_radius_ft=28.0,
        unsupported_track_min_evidence=2.0,
        unsupported_track_boundary_margin_slices=1,
        suppress_unsupported_interior_tracks=False,
    )


def poles():
    return pd.DataFrame(
        [
            {
                "world_pole_id": "P0",
                "world_x_ft": 0.0,
                "world_y_ft": 0.0,
                "base_z_ft": 0.0,
                "baseline_height_ft": 40.0,
                "max_allowed_height_ft": 50.0,
            },
            {
                "world_pole_id": "P1",
                "world_x_ft": 100.0,
                "world_y_ft": 0.0,
                "base_z_ft": 0.0,
                "baseline_height_ft": 40.0,
                "max_allowed_height_ft": 50.0,
            },
        ]
    )


def track(x0, x1, evidence=3.0, track_idx=0):
    xs = np.linspace(x0, x1, 9)
    raw = np.column_stack(
        [xs, np.zeros_like(xs), np.full_like(xs, 30.0)]
    )
    return {
        "track_idx": track_idx,
        "frags": [0],
        "raw": raw,
        "join_ids": [],
        "source_segments": ["synthetic/S0"],
        "observed_start": None,
        "observed_end": None,
        "start_slice": 4,
        "end_slice": 4,
        "evidence_score": evidence,
    }


def test_single_track_two_pole_completion():
    a = args()
    tr = track(20.0, 80.0, evidence=3.0)
    out, audit = s3.complete_span_backed_tracks(
        [tr], poles(), [{"slice_seq": 4}], a
    )
    assert len(out) == 1, out
    assert out[0].get("span_completion_used") is True, out[0]
    assert out[0].get("span_completion_track_count") == 1, out[0]
    assert out[0].get("observed_start") is not None, out[0]
    assert out[0].get("observed_end") is not None, out[0]
    assert len(audit) == 1, audit
    assert audit[0]["support_track_count"] == 1, audit[0]
    assert (
        audit[0]["rule"]
        == "two_poles_bracket_single_supported_parametric_track"
    ), audit[0]
    print("PASS single-track two-pole completion")


def test_single_track_without_two_pole_support_is_not_completed():
    a = args()
    tr = track(40.0, 60.0, evidence=3.0)
    out, audit = s3.complete_span_backed_tracks(
        [tr], poles(), [{"slice_seq": 4}], a
    )
    assert len(out) == 1, out
    assert out[0].get("span_completion_used") is False, out[0]
    assert audit == [], audit
    print("PASS unsupported single track not stretched between poles")


def test_unsupported_floating_classification_and_opt_in_suppression():
    a = args()
    tr = track(40.0, 60.0, evidence=1.0)
    clean = np.asarray(tr["raw"], float)
    row = s3.classify_conductor_support(
        "OPEN_CONDUCTOR",
        tr,
        clean,
        poles(),
        [4, 5],
        None,
        None,
        0,
        9,
        a,
    )
    assert row["classification"] == "INTERIOR_UNSUPPORTED_FLOATING", row
    assert row["suppressed"] is False, row

    a.suppress_unsupported_interior_tracks = True
    row2 = s3.classify_conductor_support(
        "OPEN_CONDUCTOR",
        tr,
        clean,
        poles(),
        [4, 5],
        None,
        None,
        0,
        9,
        a,
    )
    assert row2["classification"] == "INTERIOR_UNSUPPORTED_FLOATING", row2
    assert row2["suppressed"] is True, row2
    print("PASS unsupported floating classification and opt-in suppression")


def test_boundary_open_is_retained():
    a = args()
    a.suppress_unsupported_interior_tracks = True
    tr = track(40.0, 60.0, evidence=0.5)
    row = s3.classify_conductor_support(
        "OPEN_CONDUCTOR",
        tr,
        np.asarray(tr["raw"], float),
        poles(),
        [0],
        None,
        None,
        0,
        9,
        a,
    )
    assert row["classification"] == "BOUNDARY_OPEN", row
    assert row["suppressed"] is False, row
    print("PASS boundary-open track retained")


def main():
    marker = getattr(s3, "V4_RECON_QUALITY_PATCH_VERSION", None)
    assert marker == "v4-recon-quality-single-track-and-support-audit-v1", marker
    test_single_track_two_pole_completion()
    test_single_track_without_two_pole_support_is_not_completed()
    test_unsupported_floating_classification_and_opt_in_suppression()
    test_boundary_open_is_retained()
    print("V4_RECON_QUALITY_SMOKE_OK")


if __name__ == "__main__":
    main()
