#!/usr/bin/env python3
"""Write one human-readable per-session mean timing report."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


TIMING_FIELDS = (
    "stage1_load_ms",
    "production_pole_component_ms",
    "production_pole_refiner_parametric_ms",
    "production_stage2_ms",
    "stage1_label_resolve_ms",
    "line_connected_components_ms",
    "line_trace_ms",
    "line_assignment_audit_ms",
    "line_geometry_ms",
    "disconnected_bridge_pair_enumeration_ms",
    "stage1_electrical_track_ms",
    "stage2_total_ms",
    "stage2_output_write_ms",
    "slice_wall_ms",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timing-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--baseline-timing-dir", default="")
    return p.parse_args()


def rows(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def mean(values):
    finite = [float(v) for v in values if v not in (None, "")]
    return sum(finite) / len(finite) if finite else None


def main():
    a = parse_args()
    timing_dir = Path(a.timing_dir).resolve()
    paths = sorted(timing_dir.glob("*.csv"))
    if not paths:
        raise RuntimeError(f"no session timing CSV files under {timing_dir}")
    baseline_dir = Path(a.baseline_timing_dir).resolve() if a.baseline_timing_dir else None
    all_rows = []
    lines = [
        "V10 STAGE2 VOXEL-SUPPORTED OPTIMIZATION TIMING",
        "Units: milliseconds; values are arithmetic means per processed slice.",
        "Quality gate: byte-exact primary Stage2 outputs plus semantic audit equivalence.",
        "",
    ]
    for path in paths:
        session_rows = rows(path)
        if not session_rows:
            raise RuntimeError(f"empty timing file: {path}")
        all_rows.extend(session_rows)
        gid = session_rows[0].get("group_id", path.stem)
        lines.extend([f"SESSION {gid}", f"slices={len(session_rows)}"])
        for field in TIMING_FIELDS:
            value = mean([row.get(field) for row in session_rows])
            lines.append(f"avg_{field}={'NA' if value is None else f'{value:.3f}'}")
        if baseline_dir is not None:
            baseline_path = baseline_dir / path.name
            if baseline_path.is_file():
                baseline_rows = rows(baseline_path)
                old = mean([row.get("stage2_total_ms") for row in baseline_rows])
                new = mean([row.get("stage2_total_ms") for row in session_rows])
                lines.append(f"baseline_avg_stage2_total_ms={'NA' if old is None else f'{old:.3f}'}")
                lines.append(f"stage2_compute_speedup_x={'NA' if not old or not new else f'{old/new:.3f}'}")
        lines.append("")

    lines.extend(["ALL_SESSIONS_WEIGHTED_BY_SLICE", f"slices={len(all_rows)}"])
    for field in TIMING_FIELDS:
        value = mean([row.get(field) for row in all_rows])
        lines.append(f"avg_{field}={'NA' if value is None else f'{value:.3f}'}")
    Path(a.output).write_text("\n".join(lines) + "\n")
    print("V10_STAGE2_TIMING_SESSION_AVERAGES_OK")


if __name__ == "__main__":
    main()
