#!/usr/bin/env python3
"""Exact comparison of all durable Stage-2 data outputs for E9."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from v4_stage_contracts import atomic_json


VOLATILE_JSON_KEYS = {
    "stage1_npz", "stage1_meta_json", "pole_csv", "line_csv", "line_vertices_csv",
    "stage1_line_voxels_csv", "accepted_line_voxels_csv", "selected_fragment_bridges_csv",
    "fragment_bridge_candidates_csv", "stage1_electrical_tracks_csv", "pole_attachments_csv",
    "stage1_handoff", "stage1_input_mode", "timing_csv", "manifest",
}
EXCLUDED_NAMES = {"E9_STAGE12_SESSION_COMPLETED.json"}
MANIFEST_PATH_COLUMNS = {"stage1_npz", "pole_csv", "line_csv", "line_vertices_csv", "output_csv"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: normalize(value) for key, value in sorted(obj.items()) if key not in VOLATILE_JSON_KEYS}
    if isinstance(obj, list):
        return [normalize(value) for value in obj]
    return obj


def normalized_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = [{key: value for key, value in row.items() if key not in MANIFEST_PATH_COLUMNS}
                for row in csv.DictReader(stream)]
    return sorted(rows, key=lambda row: (row.get("group_id", ""), int(row.get("slice_seq", "0"))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    control = Path(args.control)
    candidate = Path(args.candidate)
    left = {path.relative_to(control).as_posix(): path for path in control.rglob("*") if path.is_file() and path.name not in EXCLUDED_NAMES}
    right = {path.relative_to(candidate).as_posix(): path for path in candidate.rglob("*") if path.is_file() and path.name not in EXCLUDED_NAMES}
    all_paths = sorted(set(left) | set(right))
    diffs = []
    required = {"inference_manifest.csv", "stage2_manifest.csv", "STAGE2_COMPLETED.json", "STAGE2_STAGE1_ELECTRICAL_TRACK_SUMMARY.json"}
    for name in sorted(required):
        if name not in left or name not in right:
            diffs.append({"file": name, "reason": "REQUIRED_OUTPUT_MISSING"})
    if not all_paths:
        diffs.append({"file": "*", "reason": "EMPTY_STAGE2_OUTPUT_TREES"})
    compared = 0
    for rel in all_paths:
        a, b = left.get(rel), right.get(rel)
        if a is None or b is None:
            diffs.append({"file": rel, "reason": "MISSING", "control": bool(a), "candidate": bool(b)})
            continue
        compared += 1
        if a.name in {"inference_manifest.csv", "stage2_manifest.csv"}:
            try:
                same = normalized_manifest(a) == normalized_manifest(b)
            except Exception as exc:
                diffs.append({"file": rel, "reason": "MANIFEST_READ_ERROR", "error": str(exc)})
                continue
            if not same:
                diffs.append({"file": rel, "reason": "MANIFEST_CONTENT_MISMATCH", "control_sha256": digest(a), "candidate_sha256": digest(b)})
        elif a.suffix.lower() == ".json":
            try:
                same = normalize(json.loads(a.read_text())) == normalize(json.loads(b.read_text()))
            except Exception as exc:
                diffs.append({"file": rel, "reason": "JSON_READ_ERROR", "error": str(exc)})
                continue
            if not same:
                diffs.append({"file": rel, "reason": "JSON_CONTENT_MISMATCH", "control_sha256": digest(a), "candidate_sha256": digest(b)})
        elif digest(a) != digest(b):
            diffs.append({"file": rel, "reason": "BYTE_CONTENT_MISMATCH", "control_sha256": digest(a), "candidate_sha256": digest(b)})
    report = {
        "comparison": "all durable Stage2 CSV outputs byte-exact; manifests and JSON outputs exact after removal of location-only handoff fields",
        "control": str(control), "candidate": str(candidate), "files_compared": compared,
        "differences": diffs, "status": "PASS" if not diffs else "FAIL",
    }
    atomic_json(report, args.report)
    print(f"E9_STAGE2_OUTPUT_COMPARISON={report['status']} files={compared} differences={len(diffs)}")
    if diffs:
        raise RuntimeError(f"E9 Stage2 output comparison failed; report={args.report}")


if __name__ == "__main__":
    main()
