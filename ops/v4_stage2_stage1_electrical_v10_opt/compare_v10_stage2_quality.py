#!/usr/bin/env python3
"""Byte-exact primary-output and semantic-audit equivalence gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PRIMARY_SUFFIXES = (
    "_poles.csv",
    "_lines.csv",
    "_line_vertices.csv",
    "_components.csv",
    "_stage1_line_voxels.csv",
    "_accepted_line_voxels.csv",
    "_selected_fragment_bridges.csv",
    "_stage1_electrical_tracks.csv",
    "_pole_attachments.csv",
)

AUDIT_FIELDS = (
    "stage1_inferred_line_voxels",
    "accepted_stage1_line_voxels",
    "stage1_to_stage2_voxel_preservation",
    "raw_stage1_line_components",
    "joined_stage2_tracks",
    "selected_fragment_bridges",
    "same_lane_pole_terminal_group_merges",
    "geometry_support_samples",
    "geometry_supported_samples",
    "geometry_stage1_voxel_support_fraction",
    "geometry_outside_stage1_voxel_samples",
    "line_path_fragments",
    "point_line_fragments",
    "unjoined_singleton_line_voxels",
    "max_output_turn_deg",
    "max_selected_bridge_gap_ft",
    "max_track_radius_p95_ft",
    "pole_attachments",
    "pole_attachment_requires_stage1_voxel_contact",
    "attachment_geometry_vertices_added",
    "runtime_gt_usage",
    "synthetic_line_voxels",
    "pole_pair_inference",
    "line_refiner_used",
    "line_hysteresis_used",
    "line_to_line_bridge_near_pole_allowed",
    "parallel_lane_merge_allowed",
    "open_line_endpoints_preserved",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-session", required=True)
    p.add_argument("--candidate-session", required=True)
    p.add_argument("--report", required=True)
    return p.parse_args()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relative_files(root: Path, suffixes: tuple[str, ...]) -> dict[str, Path]:
    return {
        str(path.relative_to(root)): path
        for path in root.rglob("*")
        if path.is_file() and path.name.endswith(suffixes)
    }


def main():
    a = parse_args()
    baseline = Path(a.baseline_session).resolve()
    candidate = Path(a.candidate_session).resolve()
    if not baseline.is_dir() or not candidate.is_dir():
        raise RuntimeError("baseline or candidate session directory missing")

    base_files = relative_files(baseline, PRIMARY_SUFFIXES)
    candidate_files = relative_files(candidate, PRIMARY_SUFFIXES)
    if set(base_files) != set(candidate_files):
        missing = sorted(set(base_files) - set(candidate_files))
        extra = sorted(set(candidate_files) - set(base_files))
        raise RuntimeError(f"primary output inventory differs; missing={missing} extra={extra}")
    mismatches = []
    for relative in sorted(base_files):
        left = digest(base_files[relative])
        right = digest(candidate_files[relative])
        if left != right:
            mismatches.append({"relative_path": relative, "baseline_sha256": left, "candidate_sha256": right})
    if mismatches:
        raise RuntimeError(f"primary Stage2 outputs changed: {mismatches[:10]}")

    base_audits = relative_files(baseline, ("_stage1_electrical_track_audit.json",))
    candidate_audits = relative_files(candidate, ("_stage1_electrical_track_audit.json",))
    if set(base_audits) != set(candidate_audits):
        raise RuntimeError("audit inventory differs")
    audit_mismatches = []
    for relative in sorted(base_audits):
        left = json.loads(base_audits[relative].read_text())
        right = json.loads(candidate_audits[relative].read_text())
        differences = {
            field: {"baseline": left.get(field), "candidate": right.get(field)}
            for field in AUDIT_FIELDS
            if left.get(field) != right.get(field)
        }
        if differences:
            audit_mismatches.append({"relative_path": relative, "differences": differences})
    if audit_mismatches:
        raise RuntimeError(f"semantic Stage2 audits changed: {audit_mismatches[:5]}")

    report = {
        "equivalence_version": "v10-stage2-quality-byte-exact-opt1-20260910",
        "passed": True,
        "primary_files_compared": len(base_files),
        "audits_compared": len(base_audits),
        "primary_output_mismatches": 0,
        "semantic_audit_mismatches": 0,
        "excluded_performance_diagnostics": [
            "fragment_bridge_candidates_csv",
            "timing_csv",
            "runtime_version",
            "disconnected_bridges_blocked_by_voxel_support",
        ],
    }
    Path(a.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("V10_STAGE2_QUALITY_EQUIVALENCE_OK")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
