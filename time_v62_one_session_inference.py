#!/usr/bin/env python3
"""Run V6.2 Stage-1+2 inference on exactly one session and time every slice.

This is a timing/launcher wrapper around the existing, runtime-hardened
infer_v62_stage1_stage2.py. The model is loaded exactly once. Per-slice wall
clock time is measured between consecutive completion messages from the
underlying inference loop, after model loading/compile has completed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SESSION_PARENT_RE = re.compile(r"^(session\d+)_slice(\d+)$", re.I)
DONE_RE = re.compile(r"^\[infer-v62\]\s+(\d+)/(\d+)\s+(.+?)\s+poles=(\d+)\s+lines=(\d+)\s*$")
RESUME_RE = re.compile(r"^\[infer-v62\]\s+(\d+)\s+resume\s+(.+?)\s*$")
MODEL_READY_RE = re.compile(r"^files\s+(\d+)\s+compile\s+(.+?)\s*$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--session", required=True, help="Exact group id, e.g. 59768101-C4990BB-2026/session3")
    p.add_argument("--input_dir", default="/data/voxel_csv_combined")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--code_dir", default="/workspace/voxel_poleline")
    p.add_argument("--model_path", required=True)
    p.add_argument("--calibration_json", required=True)
    p.add_argument("--local_refiner_bundle", required=True)
    p.add_argument("--grid_size", type=int, nargs=3, default=[400, 400, 200])
    p.add_argument("--voxel_size_ft", type=float, default=0.5)
    p.add_argument("--pole_candidate_threshold", type=float, default=0.15)
    p.add_argument("--line_candidate_threshold", type=float, default=0.08)
    p.add_argument("--line_weak_threshold", type=float, default=0.04)
    p.add_argument("--line_competition_ratio", type=float, default=0.55)
    p.add_argument("--line_min_voxels", type=int, default=3)
    p.add_argument("--edge_width_vox", type=int, default=10)
    p.add_argument("--core_size", type=int, default=48)
    p.add_argument("--batch_size", type=int, default=5)
    p.add_argument("--build_workers", type=int, default=6)
    p.add_argument("--amp", default="bf16")
    p.add_argument("--compile_model", type=int, default=1)
    p.add_argument("--resume", type=int, choices=[0, 1], default=0,
                   help="Use 0 for a clean timing benchmark. Use 1 only to continue an interrupted session.")
    p.add_argument("--compression_level", type=int, default=1)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def safe_id(rel: str) -> str:
    return re.sub(r"\.csv(?:\.gz)?$", "", rel.replace("/", "__").replace("\\", "__"), flags=re.I)


def session_parts(session: str) -> tuple[str, str]:
    if "/" not in session:
        raise ValueError("--session must be '<geography>/<sessionN>', for example '59768101-C4990BB-2026/session3'")
    geo, sess = session.rsplit("/", 1)
    if not re.fullmatch(r"session\d+", sess, flags=re.I):
        raise ValueError(f"Invalid session suffix {sess!r}; expected sessionN")
    return geo, sess.lower()


def discover_session(input_dir: Path, session: str) -> list[dict[str, Any]]:
    geo, sess = session_parts(session)
    rows: list[dict[str, Any]] = []
    candidates = list(input_dir.rglob("*.csv")) + list(input_dir.rglob("*.csv.gz"))
    for src in candidates:
        try:
            rel = src.relative_to(input_dir)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) < 3 or parts[0] != geo:
            continue
        m = SESSION_PARENT_RE.match(parts[-2])
        if not m or m.group(1).lower() != sess:
            continue
        seq = int(m.group(2))
        rels = str(rel)
        rows.append({
            "source_relpath": rels,
            "id": safe_id(rels),
            "group_id": f"{geo}/{sess}",
            "slice_seq": seq,
        })
    rows.sort(key=lambda r: (int(r["slice_seq"]), str(r["source_relpath"])))
    return rows


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    f = pos - lo
    return xs[lo] * (1.0 - f) + xs[hi] * f


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in columns})


def main() -> int:
    a = parse_args()
    input_dir = Path(a.input_dir).resolve()
    code_dir = Path(a.code_dir).resolve()
    out = Path(a.output_dir).resolve()
    infer_script = code_dir / "infer_v62_stage1_stage2.py"

    for p, label in [
        (input_dir, "input directory"),
        (infer_script, "inference script"),
        (Path(a.model_path), "Stage-1 model"),
        (Path(a.calibration_json), "calibration JSON"),
        (Path(a.local_refiner_bundle), "Stage-2 refiner bundle"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {label}: {p}")

    records = discover_session(input_dir, a.session)
    if not records:
        raise RuntimeError(f"No source slices found for exact session {a.session!r} under {input_dir}")

    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "session_manifest.json"
    manifest.write_text(json.dumps(records, indent=2))
    timing_csv = out / "slice_inference_timing.csv"
    summary_json = out / "session_inference_timing.json"
    log_path = out / "timed_inference.log"

    seq_by_rel = {str(r["source_relpath"]): int(r["slice_seq"]) for r in records}

    cmd = [
        sys.executable, str(infer_script),
        "--input_dir", str(input_dir),
        "--output_dir", str(out),
        "--model_path", str(a.model_path),
        "--calibration_json", str(a.calibration_json),
        "--local_refiner_bundle", str(a.local_refiner_bundle),
        "--manifest_json", str(manifest),
        "--grid_size", *map(str, a.grid_size),
        "--voxel_size_ft", str(a.voxel_size_ft),
        "--pole_candidate_threshold", str(a.pole_candidate_threshold),
        "--line_candidate_threshold", str(a.line_candidate_threshold),
        "--line_weak_threshold", str(a.line_weak_threshold),
        "--line_competition_ratio", str(a.line_competition_ratio),
        "--line_min_voxels", str(a.line_min_voxels),
        "--edge_width_vox", str(a.edge_width_vox),
        "--core_size", str(a.core_size),
        "--batch_size", str(a.batch_size),
        "--build_workers", str(a.build_workers),
        "--amp", str(a.amp),
        "--compile_model", str(a.compile_model),
        "--resume", str(a.resume),
        "--compression_level", str(a.compression_level),
    ]

    print(f"[session-timing] session={a.session}")
    print(f"[session-timing] slices={len(records)} first_seq={records[0]['slice_seq']} last_seq={records[-1]['slice_seq']}")
    print("[session-timing] command:", shlex.join(cmd))
    if a.dry_run:
        return 0

    rows: list[dict[str, Any]] = []
    process_started = time.perf_counter()
    model_ready_at: float | None = None
    previous_slice_done_at: float | None = None
    failure: str | None = None

    with log_path.open("w", buffering=1) as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(code_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            now = time.perf_counter()
            print(raw_line, end="")
            log.write(raw_line)
            line = raw_line.rstrip("\n")

            if model_ready_at is None and MODEL_READY_RE.match(line):
                model_ready_at = now
                previous_slice_done_at = now
                continue

            m = DONE_RE.match(line)
            if m:
                order = int(m.group(1))
                total = int(m.group(2))
                rel = m.group(3)
                poles = int(m.group(4))
                lines = int(m.group(5))
                start_marker = previous_slice_done_at if previous_slice_done_at is not None else process_started
                elapsed = now - start_marker
                previous_slice_done_at = now
                rows.append({
                    "slice_order": order,
                    "total_slices": total,
                    "slice_seq": seq_by_rel.get(rel, ""),
                    "relative_path": rel,
                    "status": "completed",
                    "accepted_poles": poles,
                    "accepted_line_segments": lines,
                    "wall_seconds": elapsed,
                    "cumulative_seconds_after_model_ready": (now - model_ready_at) if model_ready_at is not None else "",
                })
                print(f"[slice-timing] {order}/{total} seq={seq_by_rel.get(rel,'?')} seconds={elapsed:.3f}")
                continue

            m = RESUME_RE.match(line)
            if m:
                order = int(m.group(1))
                rel = m.group(2)
                now_elapsed = 0.0 if previous_slice_done_at is None else now - previous_slice_done_at
                previous_slice_done_at = now
                rows.append({
                    "slice_order": order,
                    "total_slices": len(records),
                    "slice_seq": seq_by_rel.get(rel, ""),
                    "relative_path": rel,
                    "status": "resumed_existing_output",
                    "accepted_poles": "",
                    "accepted_line_segments": "",
                    "wall_seconds": now_elapsed,
                    "cumulative_seconds_after_model_ready": (now - model_ready_at) if model_ready_at is not None else "",
                })

        rc = proc.wait()
        if rc != 0:
            failure = f"Underlying inference exited with code {rc}"

    ended = time.perf_counter()
    total_wall = ended - process_started
    startup = (model_ready_at - process_started) if model_ready_at is not None else None
    completed_times = [float(r["wall_seconds"]) for r in rows if r["status"] == "completed"]
    summary = {
        "completed": failure is None,
        "session": a.session,
        "source_slice_count": len(records),
        "timed_completed_slice_count": len(completed_times),
        "resumed_slice_count": sum(r["status"] != "completed" for r in rows),
        "model_load_and_compile_seconds": startup,
        "slice_loop_wall_seconds": sum(completed_times),
        "total_process_wall_seconds": total_wall,
        "mean_slice_seconds": statistics.fmean(completed_times) if completed_times else None,
        "median_slice_seconds": statistics.median(completed_times) if completed_times else None,
        "p95_slice_seconds": percentile(completed_times, 0.95) if completed_times else None,
        "min_slice_seconds": min(completed_times) if completed_times else None,
        "max_slice_seconds": max(completed_times) if completed_times else None,
        "throughput_slices_per_second": (len(completed_times) / sum(completed_times)) if completed_times and sum(completed_times) > 0 else None,
        "timing_definition": "Per-slice wall time is measured from the previous slice completion to the current slice completion after the model-ready line. It includes CSV read, dense Stage-1 inference, Stage-2 component/refiner work, output CSV/NPZ writes, and per-slice bookkeeping. Model load/torch.compile is reported separately.",
        "failure": failure,
    }
    write_csv(
        timing_csv,
        rows,
        ["slice_order", "total_slices", "slice_seq", "relative_path", "status", "accepted_poles", "accepted_line_segments", "wall_seconds", "cumulative_seconds_after_model_ready"],
    )
    summary_json.write_text(json.dumps(summary, indent=2))
    print("\n[session-timing] inference timing summary")
    print(json.dumps(summary, indent=2))
    print(f"[session-timing] per-slice CSV: {timing_csv}")
    print(f"[session-timing] summary JSON: {summary_json}")
    return 0 if failure is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
