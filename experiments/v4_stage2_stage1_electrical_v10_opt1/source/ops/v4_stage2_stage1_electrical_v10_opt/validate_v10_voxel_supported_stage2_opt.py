#!/usr/bin/env python3
"""Strict post-run validator. Run only in the approved Docker image."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--session-dir", required=True)
    p.add_argument("--report", required=True)
    return p.parse_args()


def key(row):
    return (int(row.x), int(row.y), int(row.z))


def supported(point, cells):
    q = np.asarray(point, dtype=float)
    lo = np.ceil(q - 0.500001).astype(int)
    hi = np.floor(q + 0.500001).astype(int)
    return any(
        (x, y, z) in cells
        for x in range(lo[0], hi[0] + 1)
        for y in range(lo[1], hi[1] + 1)
        for z in range(lo[2], hi[2] + 1)
    )


def validate_one(audit_path):
    audit = json.loads(audit_path.read_text())
    stem = audit_path.name.removesuffix("_stage1_electrical_track_audit.json")
    parent = audit_path.parent
    files = {
        "stage1": parent / f"{stem}_stage1_line_voxels.csv",
        "accepted": parent / f"{stem}_accepted_line_voxels.csv",
        "vertices": parent / f"{stem}_line_vertices.csv",
        "bridges": parent / f"{stem}_selected_fragment_bridges.csv",
        "attachments": parent / f"{stem}_pole_attachments.csv",
    }
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise RuntimeError(f"missing diagnostic files: {missing}")

    stage1 = pd.read_csv(files["stage1"])
    accepted = pd.read_csv(files["accepted"])
    for name, frame in (("stage1", stage1), ("accepted", accepted)):
        absent = sorted({"x", "y", "z"} - set(frame.columns))
        if absent:
            raise RuntimeError(f"{name} missing columns {absent}: {audit_path}")
    stage1_keys = {key(r) for r in stage1.itertuples(index=False)}
    accepted_keys = {key(r) for r in accepted.itertuples(index=False)}
    if stage1_keys != accepted_keys or len(stage1) != len(accepted):
        raise RuntimeError(f"voxel identity mismatch: {audit_path}")

    vertices = pd.read_csv(files["vertices"])
    needed = {"component_id", "vertex_index", "x", "y", "z"}
    if not needed <= set(vertices.columns):
        raise RuntimeError(f"vertex schema mismatch: {audit_path}")
    samples = supported_samples = 0
    endpoints = {}
    max_turn = 0.0
    for cid, group in vertices.groupby("component_id", sort=False):
        group = group.sort_values("vertex_index")
        v = group[["x", "y", "z"]].to_numpy(dtype=float)
        if not len(v):
            continue
        endpoints[str(cid)] = (v[0], v[-1])
        if len(v) == 1:
            samples += 1
            supported_samples += int(supported(v[0], stage1_keys))
        for first, second in zip(v[:-1], v[1:]):
            distance = float(np.linalg.norm(second - first))
            n = max(1, int(math.ceil(distance / 0.25)))
            for i in range(n + 1):
                q = first + (i / n) * (second - first)
                samples += 1
                supported_samples += int(supported(q, stage1_keys))
        for i in range(1, len(v) - 1):
            x, y = v[i] - v[i - 1], v[i + 1] - v[i]
            if np.linalg.norm(x) and np.linalg.norm(y):
                dot = float(np.clip(np.dot(x/np.linalg.norm(x), y/np.linalg.norm(y)), -1, 1))
                max_turn = max(max_turn, math.degrees(math.acos(dot)))
    if samples != supported_samples:
        raise RuntimeError(
            f"geometry left Stage1 support: {audit_path} "
            f"supported={supported_samples}/{samples}"
        )

    bridges = pd.read_csv(files["bridges"])
    if len(bridges):
        raise RuntimeError(f"disconnected bridges selected: {audit_path}")
    attachments = pd.read_csv(files["attachments"])
    required_attachment = {
        "component_id", "track_end", "pole_component_id", "anchor_x", "anchor_y", "anchor_z",
        "contact_pole_voxel_x", "contact_pole_voxel_y", "contact_pole_voxel_z",
        "attachment_support_mode",
    }
    if not required_attachment <= set(attachments.columns):
        raise RuntimeError(f"attachment schema mismatch: {audit_path}")
    seen = set()
    for row in attachments.itertuples(index=False):
        anchor = np.asarray([row.anchor_x, row.anchor_y, row.anchor_z], dtype=float)
        if str(row.component_id) not in endpoints:
            raise RuntimeError(f"attachment references missing track: {audit_path}")
        expected = endpoints[str(row.component_id)][0 if str(row.track_end) == "start" else 1]
        if not np.allclose(anchor, expected, atol=1.0e-9) or not supported(anchor, stage1_keys):
            raise RuntimeError(f"unsupported attachment anchor: {audit_path}")
        pole_voxel = np.asarray([
            row.contact_pole_voxel_x, row.contact_pole_voxel_y, row.contact_pole_voxel_z
        ], dtype=float)
        if np.max(np.abs(pole_voxel - anchor)) > 1.000001:
            raise RuntimeError(f"non-contact pole attachment: {audit_path}")
        if str(row.attachment_support_mode) != "direct_stage1_line_pole_voxel_contact":
            raise RuntimeError(f"wrong attachment mode: {audit_path}")
        attachment_key = (
            str(row.pole_component_id), round(float(row.anchor_x), 6),
            round(float(row.anchor_y), 6), round(float(row.anchor_z), 6),
        )
        if attachment_key in seen:
            raise RuntimeError(f"duplicate pole attachment anchor: {audit_path}")
        seen.add(attachment_key)

    required_audit = {
        "runtime_gt_usage": False,
        "synthetic_line_voxels": 0,
        "selected_fragment_bridges": 0,
        "geometry_outside_stage1_voxel_samples": 0,
        "pole_attachment_requires_stage1_voxel_contact": True,
        "open_line_endpoints_preserved": True,
    }
    for field, expected in required_audit.items():
        if audit.get(field) != expected:
            raise RuntimeError(f"audit invariant {field}={audit.get(field)!r}: {audit_path}")
    return {
        "slice_seq": int(audit["slice_seq"]), "stage1_line_voxels": len(stage1),
        "geometry_samples": samples, "attachments": len(attachments),
        "max_turn_deg": float(max_turn),
    }


def main():
    a = args()
    root = Path(a.session_dir).resolve()
    audits = sorted(root.rglob("*_stage1_electrical_track_audit.json"))
    if not audits:
        raise RuntimeError(f"no Stage2 audit files under {root}")
    rows = [validate_one(path) for path in audits]
    report = {
        "validator": "v10-voxel-supported-stage2-20260908",
        "passed": True, "slices": len(rows),
        "stage1_line_voxels": sum(r["stage1_line_voxels"] for r in rows),
        "geometry_samples": sum(r["geometry_samples"] for r in rows),
        "pole_attachments": sum(r["attachments"] for r in rows),
        "max_turn_deg": max((r["max_turn_deg"] for r in rows), default=0.0),
        "per_slice": rows,
    }
    Path(a.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("V10_VOXEL_SUPPORTED_STAGE2_VALIDATION_OK")
    print(json.dumps({k: v for k, v in report.items() if k != "per_slice"}, sort_keys=True))


if __name__ == "__main__":
    main()
