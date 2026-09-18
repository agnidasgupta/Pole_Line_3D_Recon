#!/usr/bin/env python3
"""Time Stage-3 reconstruction for one session.

Two measurements are supported:
  batch   - reconstruct the complete selected session once and report total and
            amortized seconds per source slice.
  rolling - for every source slice, run Stage 3 with --latest_slice. Stage 3 then
            uses that slice plus the previous max_span_slices slice indices, which
            measures actual rolling/realtime reconstruction latency per slice.
  both    - run both measurements.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PHASE_RE = re.compile(r"^\[stage3-phase\]\s+(.+?)\s+(fragment_join_seconds|span_completion_pre_seconds|hidden_pole_seconds|span_completion_post_seconds)=([0-9.]+)(?:\s+.*)?$")
COMPLETE_RE = re.compile(r"^\[stage3-circuit-complete\]\s+(.+?)\s+.*\s+elapsed_seconds=([0-9.]+)\s*$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True)
    p.add_argument("--inference_dir", required=True)
    p.add_argument("--metadata_dir", default="/data/voxel_csv_combined")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--code_dir", default="/workspace/voxel_poleline")
    p.add_argument("--mode", choices=["batch", "rolling", "both"], default="both")
    p.add_argument("--max_span_slices", type=int, default=9)
    p.add_argument("--world_units_to_ft", type=float, default=0.5)
    p.add_argument("--keep_rolling_outputs", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    f = pos - lo
    return xs[lo] * (1 - f) + xs[hi] * f


def read_session_rows(inference_dir: Path, session: str) -> list[dict[str, Any]]:
    path = inference_dir / "inference_manifest.csv"
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing/empty inference manifest: {path}")
    with path.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if str(r.get("group_id", "")) == session]
    if not rows:
        raise RuntimeError(f"Inference manifest has no rows for exact session {session!r}")
    for r in rows:
        r["slice_seq"] = int(float(r["slice_seq"]))
    rows.sort(key=lambda r: (r["slice_seq"], r.get("relative_path", "")))
    return rows


def base_cmd(a: argparse.Namespace, script: Path, out: Path) -> list[str]:
    return [
        sys.executable, str(script),
        "--inference_dir", str(Path(a.inference_dir).resolve()),
        "--output_dir", str(out),
        "--metadata_dir", str(Path(a.metadata_dir).resolve()),
        "--session_filter", a.session,
        "--world_units_to_ft", str(a.world_units_to_ft),
        "--max_span_slices", str(a.max_span_slices),
    ]


def run_logged(cmd: list[str], cwd: Path, log_path: Path) -> tuple[int, float, dict[str, float], float | None]:
    phases: dict[str, float] = {}
    algorithm_elapsed: float | None = None
    t0 = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", buffering=1) as log:
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for raw in proc.stdout:
            print(raw, end="")
            log.write(raw)
            line = raw.rstrip("\n")
            m = PHASE_RE.match(line)
            if m:
                phases[m.group(2)] = float(m.group(3))
            m = COMPLETE_RE.match(line)
            if m and m.group(1) == cmd[cmd.index("--session_filter") + 1]:
                algorithm_elapsed = float(m.group(2))
        rc = proc.wait()
    return rc, time.perf_counter() - t0, phases, algorithm_elapsed


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})


def main() -> int:
    a = parse_args()
    code_dir = Path(a.code_dir).resolve()
    script = code_dir / "reconstruct_v62_stage3.py"
    inference_dir = Path(a.inference_dir).resolve()
    metadata_dir = Path(a.metadata_dir).resolve()
    out = Path(a.output_dir).resolve()
    if not script.exists():
        raise FileNotFoundError(f"Missing reconstruction script: {script}")
    if not metadata_dir.exists():
        raise FileNotFoundError(f"Missing metadata/raw input directory: {metadata_dir}")
    rows = read_session_rows(inference_dir, a.session)
    seqs = sorted({int(r["slice_seq"]) for r in rows})
    out.mkdir(parents=True, exist_ok=True)

    print(f"[recon-timing] session={a.session} manifest_rows={len(rows)} unique_slice_seq={len(seqs)}")
    print(f"[recon-timing] mode={a.mode} max_span_slices={a.max_span_slices}")

    summary: dict[str, Any] = {
        "session": a.session,
        "manifest_row_count": len(rows),
        "unique_slice_count": len(seqs),
        "slice_sequences": seqs,
        "mode": a.mode,
        "max_span_slices": a.max_span_slices,
        "batch": None,
        "rolling": None,
    }

    if a.mode in {"batch", "both"}:
        batch_out = out / "batch_reconstruction"
        if batch_out.exists():
            shutil.rmtree(batch_out)
        cmd = base_cmd(a, script, batch_out)
        print("[recon-timing] batch command:", shlex.join(cmd))
        if not a.dry_run:
            rc, wall, phases, algo = run_logged(cmd, code_dir, out / "batch_reconstruction.log")
            if rc != 0:
                raise RuntimeError(f"Batch reconstruction failed with exit code {rc}")
            denom = max(len(seqs), 1)
            summary["batch"] = {
                "total_wall_seconds": wall,
                "stage3_reported_algorithm_seconds": algo,
                "equivalent_wall_seconds_per_slice": wall / denom,
                "equivalent_algorithm_seconds_per_slice": (algo / denom) if algo is not None else None,
                "phase_seconds": phases,
                "phase_seconds_per_slice": {k: v / denom for k, v in phases.items()},
                "note": "Stage 3 reconstructs a session jointly, so batch seconds_per_slice is an amortized equivalent, not an independently isolated slice measurement.",
            }

    rolling_rows: list[dict[str, Any]] = []
    if a.mode in {"rolling", "both"}:
        rolling_root = out / "rolling_tmp"
        rolling_root.mkdir(parents=True, exist_ok=True)
        previous_seq: int | None = None
        for order, seq in enumerate(seqs, 1):
            rout = rolling_root / f"slice_{seq}"
            if rout.exists():
                shutil.rmtree(rout)
            cmd = base_cmd(a, script, rout) + ["--latest_slice", str(seq)]
            print(f"[recon-timing] rolling {order}/{len(seqs)} latest_slice={seq}")
            if a.dry_run:
                print("[recon-timing] command:", shlex.join(cmd))
                continue
            rc, wall, phases, algo = run_logged(cmd, code_dir, out / "rolling_logs" / f"slice_{seq}.log")
            if rc != 0:
                raise RuntimeError(f"Rolling reconstruction failed at latest_slice={seq} with exit code {rc}")
            # The Stage-3 filter uses numeric slice-index range, not count of observed slices.
            low = seq - a.max_span_slices
            window_rows = [r for r in rows if low <= int(r["slice_seq"]) <= seq]
            window_unique = len({int(r["slice_seq"]) for r in window_rows})
            rolling_rows.append({
                "slice_order": order,
                "slice_seq": seq,
                "previous_observed_slice_seq": previous_seq if previous_seq is not None else "",
                "observed_slices_in_stage3_window": window_unique,
                "manifest_rows_in_stage3_window": len(window_rows),
                "wall_seconds": wall,
                "stage3_reported_algorithm_seconds": algo if algo is not None else "",
                "fragment_join_seconds": phases.get("fragment_join_seconds", ""),
                "span_completion_pre_seconds": phases.get("span_completion_pre_seconds", ""),
                "hidden_pole_seconds": phases.get("hidden_pole_seconds", ""),
                "span_completion_post_seconds": phases.get("span_completion_post_seconds", ""),
            })
            previous_seq = seq
            if not a.keep_rolling_outputs:
                shutil.rmtree(rout, ignore_errors=True)

        times = [float(r["wall_seconds"]) for r in rolling_rows]
        algo_times = [float(r["stage3_reported_algorithm_seconds"]) for r in rolling_rows if r["stage3_reported_algorithm_seconds"] != ""]
        summary["rolling"] = {
            "completed_slice_count": len(rolling_rows),
            "total_wall_seconds": sum(times),
            "mean_wall_seconds_per_slice": statistics.fmean(times) if times else None,
            "median_wall_seconds_per_slice": statistics.median(times) if times else None,
            "p95_wall_seconds_per_slice": percentile(times, 0.95) if times else None,
            "min_wall_seconds_per_slice": min(times) if times else None,
            "max_wall_seconds_per_slice": max(times) if times else None,
            "mean_stage3_algorithm_seconds_per_slice": statistics.fmean(algo_times) if algo_times else None,
            "timing_definition": f"Actual rolling Stage-3 latency for each latest slice. Each run uses the latest slice plus observed slices whose numeric slice_seq lies within latest-{a.max_span_slices}..latest. Missing slice numbers are allowed.",
        }
        write_csv(
            out / "slice_reconstruction_timing.csv",
            rolling_rows,
            ["slice_order", "slice_seq", "previous_observed_slice_seq", "observed_slices_in_stage3_window", "manifest_rows_in_stage3_window", "wall_seconds", "stage3_reported_algorithm_seconds", "fragment_join_seconds", "span_completion_pre_seconds", "hidden_pole_seconds", "span_completion_post_seconds"],
        )
        if not a.keep_rolling_outputs:
            shutil.rmtree(rolling_root, ignore_errors=True)

    (out / "session_reconstruction_timing.json").write_text(json.dumps(summary, indent=2))
    print("\n[recon-timing] summary")
    print(json.dumps(summary, indent=2))
    if rolling_rows:
        print(f"[recon-timing] per-slice CSV: {out / 'slice_reconstruction_timing.csv'}")
    print(f"[recon-timing] summary JSON: {out / 'session_reconstruction_timing.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
