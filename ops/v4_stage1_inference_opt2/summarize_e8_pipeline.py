#!/usr/bin/env python3
"""Summarize bounded E8 pipeline timing without touching inference outputs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

FIELDS = (
    "stage1_wall_ms",
    "slice_total_ms",
    "csv_read_ms",
    "sparse_item_prep_ms",
    "stage1_artifact_write_ms",
    "stage1_manifest_write_ms",
    "d2h_gather_ms",
)

def value(row, name):
    try:
        result = float(row.get(name, "nan"))
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", required=True)
    p.add_argument("--exclude-first", type=int, default=1)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    root = Path(a.run_root)
    timing_files = sorted((root / "timings" / "stage1").glob("*.csv"))
    rows = []
    for path in timing_files:
        with path.open(newline="") as stream:
            rows.extend(csv.DictReader(stream))
    rows = rows[a.exclude_first:]
    means = {}
    for field in FIELDS:
        samples = [v for row in rows if (v := value(row, field)) is not None]
        means[field] = sum(samples) / len(samples) if samples else None
    payload = {
        "run_root": str(root),
        "timing_files": len(timing_files),
        "timed_slices": len(rows),
        "excluded_first_slices": a.exclude_first,
        "means_ms": means,
    }
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))

if __name__ == "__main__":
    main()
