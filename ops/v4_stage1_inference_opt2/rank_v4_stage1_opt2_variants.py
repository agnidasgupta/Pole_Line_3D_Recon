#!/usr/bin/env python3
"""Rank Opt2 runs that reproduced V4 production outputs exactly."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("roots", nargs="+", help="one or more Opt2 run roots")
    p.add_argument("--output", help="optional CSV output path")
    p.add_argument("--group-id", help="compare timing only for one matching session")
    return p.parse_args()


def read_info(path: Path):
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def main():
    a = parse_args()
    rows = []
    for raw in a.roots:
        root = Path(raw).resolve()
        info = read_info(root / "RUN_INFO.txt")
        timing_paths = sorted((root / "timings" / "stage1").glob("*.csv"))
        if not timing_paths:
            raise RuntimeError(f"no timing CSVs in {root}")
        timing = pd.concat([pd.read_csv(path) for path in timing_paths], ignore_index=True)
        if a.group_id:
            timing = timing[timing["group_id"].astype(str).eq(a.group_id)].copy()
            if timing.empty:
                raise RuntimeError(f"group_id {a.group_id!r} is absent from {root}")
        reports = [json.loads(path.read_text()) for path in sorted((root / "status").glob("*.production_equivalence.json"))]
        if not reports:
            raise RuntimeError(f"no production-equivalence reports in {root}")
        passed = all(bool(report.get("passed_for_automatic_promotion")) for report in reports)
        additions = sum(
            int(report.get("positive_reference_counts", {}).get("unverified_candidate_additions", 0))
            for report in reports
        )
        lost = sum(
            int(report.get("positive_reference_counts", {}).get("known_positive_lost_to_unknown", 0))
            for report in reports
        )
        flips = sum(
            int(report.get("positive_reference_counts", {}).get("known_positive_class_flips", 0))
            for report in reports
        )
        row = {
            "variant": info.get("variant_name", root.name),
            "production_equivalence": "PASS" if passed else "BLOCK",
            "sessions": int(timing["group_id"].astype(str).nunique()),
            "known_positive_losses": lost,
            "known_positive_class_flips": flips,
            "unverified_additions": additions,
            "stage1_wall_ms": pd.to_numeric(timing["stage1_wall_ms"], errors="coerce").mean(),
            "gpu_model_ms": pd.to_numeric(timing["gpu_model_ms"], errors="coerce").mean(),
            "gpu_feature_assembly_ms": pd.to_numeric(timing["gpu_feature_assembly_ms"], errors="coerce").mean(),
            "d2h_gather_ms": pd.to_numeric(timing["d2h_gather_ms"], errors="coerce").mean(),
            "batch_size": info.get("batch_size"),
            "compile_model": info.get("compile_model"),
            "compile_mode": info.get("compile_mode"),
            "channels_last": info.get("channels_last"),
            "pinned_d2h": info.get("pinned_d2h"),
            "detailed_cuda_timing": info.get("detailed_cuda_timing"),
            "retain_gather_host_buffers": info.get("retain_gather_host_buffers", "0"),
            "precompute_batch_gather_plans": info.get("precompute_batch_gather_plans", "0"),
            "cache_coordinate_channels": info.get("cache_coordinate_channels", "0"),
            "cache_reference_coordinate_channels": info.get("cache_reference_coordinate_channels", "0"),
            "prune_embedding_head": info.get("prune_embedding_head"),
            "run_root": str(root),
        }
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(
        ["production_equivalence", "stage1_wall_ms"], ascending=[False, True]
    )
    print(frame.to_string(index=False))
    if a.output:
        output = Path(a.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        print(f"VARIANT_RANKING_WRITTEN={output}")


if __name__ == "__main__":
    main()
