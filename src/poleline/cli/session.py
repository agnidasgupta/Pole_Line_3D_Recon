#!/usr/bin/env python3
"""Production V4 Stage1 -> Stage2 -> rolling Stage3 orchestrator.

Every stage has a durable atomic boundary.  In-process handoff is used for latency,
but a later stage can always be run independently from the saved upstream artifacts.

Physical span contract: each sequence increment is 50 ft; a 450-ft maximum span is
therefore a maximum sequence gap of 9.  A closed interval [S-9, S] may contain up to
10 observed slice centers when every sequence is present.
"""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from v4_realtime_pipeline import V4RealtimePipeline
from v4_stage2_runtime import POLE_OUTPUT_COLUMNS, LINE_OUTPUT_COLUMNS, VERTEX_OUTPUT_COLUMNS
from v4_stage_contracts import (
    CONTRACT_VERSION, STAGE1_MANIFEST_COLUMNS, STAGE2_MANIFEST_COLUMNS,
    atomic_csv, atomic_json, load_stage1_artifact, save_stage1_artifact,
    safe_id, stage1_paths, stage2_paths, upsert_manifest_row,
)

SESSION_RE = re.compile(r"^(session\d+)_slice(\d+)$", re.I)
_STAGE3_MODULE = None


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--session_filter", required=True, help="geography/sessionN; Nebius launcher discovers this automatically")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--calibration_json", required=True)
    p.add_argument("--stage2_bundle", required=True)
    p.add_argument("--stage3_script", default=str(Path(__file__).with_name("reconstruct_v4_stage3.py")))
    p.add_argument("--grid_size", type=int, nargs=3, default=[400, 400, 200])
    p.add_argument("--voxel_size_ft", type=float, default=.5)
    p.add_argument("--core_size", type=int, default=48)
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--amp", choices=["fp16", "bf16", "none"], default="bf16")
    p.add_argument("--compile_model", type=int, default=0)
    p.add_argument("--evaluate_all_cores", type=int, default=1)
    p.add_argument("--gpu_coord_channels", type=int, default=0)
    p.add_argument("--fixed_batch_shape", type=int, default=0)
    p.add_argument("--pole_candidate_threshold", type=float, default=.15)
    p.add_argument("--line_candidate_threshold", type=float, default=.08)
    p.add_argument("--line_weak_threshold", type=float, default=.04)
    p.add_argument("--line_competition_ratio", type=float, default=.55)
    p.add_argument("--pole_min_voxels", type=int, default=4)
    p.add_argument("--line_min_voxels", type=int, default=3)
    p.add_argument("--edge_width_vox", type=int, default=10)
    p.add_argument("--max_sequence_gap", type=int, default=9)
    p.add_argument("--max_span_length_ft", type=float, default=450.0)
    p.add_argument("--slice_length_ft", type=float, default=50.0)
    p.add_argument("--stage3_every_slice", type=int, default=1)
    p.add_argument("--stage3_execution", choices=["inprocess", "subprocess"], default="inprocess")
    p.add_argument("--stage3_inmemory_cache", type=int, default=1)
    p.add_argument("--write_row_csv", type=int, default=0)
    p.add_argument("--resume", type=int, default=1)
    p.add_argument("--max_slices", type=int, default=0)
    a = p.parse_args()
    if int(a.max_sequence_gap) != 9 or abs(float(a.max_span_length_ft) - 450.0) > 1e-9 or abs(float(a.slice_length_ft) - 50.0) > 1e-9:
        raise ValueError("Production Stage3 requires 9 sequence intervals x 50 ft = 450 ft maximum span")
    if int(a.stage3_every_slice) != 1:
        raise ValueError("Production realtime contract requires Stage3 after every arriving slice")
    return a


def discover(root, gid):
    root = Path(root).resolve()
    parts = str(gid).split("/")
    if len(parts) != 2 or not parts[0] or not re.fullmatch(r"session\d+", parts[1], re.I):
        raise ValueError(f"session_filter must be geography/sessionN, got {gid!r}")
    geo, sess = parts
    base = root / geo
    if not base.exists():
        raise FileNotFoundError(base)
    rows = []
    for p in sorted(base.rglob("*.csv")) + sorted(base.rglob("*.csv.gz")):
        m = SESSION_RE.match(p.parent.name)
        if not m or m.group(1).lower() != sess.lower():
            continue
        seq = int(m.group(2))
        rows.append((seq, p, str(p.relative_to(root))))
    rows.sort(key=lambda x: (x[0], x[2]))
    if not rows:
        raise RuntimeError(f"No slices found for {gid} under {root}")
    seqs = [x[0] for x in rows]
    dup = sorted({x for x in seqs if seqs.count(x) > 1})
    if dup:
        raise RuntimeError(f"Multiple CSV sources for slice sequence(s) {dup[:20]}; resolve before realtime replay")
    return rows


def _read_manifest(path, columns):
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    try:
        d = pd.read_csv(p)
    except EmptyDataError:
        return pd.DataFrame(columns=columns)
    for c in columns:
        if c not in d.columns:
            d[c] = np.nan
    return d.reindex(columns=columns)


def _manifest_row_by_seq(df, gid, seq):
    if df.empty:
        return None
    q = df[(df.group_id.astype(str) == str(gid)) & (pd.to_numeric(df.slice_seq, errors="coerce") == int(seq))]
    if q.empty:
        return None
    return q.iloc[-1].to_dict()


def _stage3_argv(a, out, gid, seq, dest):
    cmd = [
        a.stage3_script,
        "--inference_dir", str(out),
        "--output_dir", str(dest),
        "--session_filter", gid,
        "--latest_slice", str(seq),
        "--max_sequence_gap", str(a.max_sequence_gap),
        "--max_span_length_ft", str(a.max_span_length_ft),
        "--slice_length_ft", str(a.slice_length_ft),
        "--disable_plots",
    ]
    if int(a.stage3_inmemory_cache):
        cmd.append("--realtime_inmemory_cache")
    return cmd


def _ensure_stage3_module(a):
    global _STAGE3_MODULE
    if _STAGE3_MODULE is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("v4_realtime_stage3_module", a.stage3_script)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import Stage3 module: {a.stage3_script}")
        _STAGE3_MODULE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_STAGE3_MODULE)
    return _STAGE3_MODULE


def run_stage3(a, out, gid, seq, dest, stage2_payload=None):
    """Run one rolling Stage3 update, using RAM handoff when available."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    log = dest / "stage3_stdout.log"
    argv = _stage3_argv(a, out, gid, seq, dest)
    t0 = time.perf_counter()
    if a.stage3_execution == "subprocess":
        proc = subprocess.run([sys.executable, *argv], text=True, capture_output=True)
        ms = (time.perf_counter() - t0) * 1000.0
        log.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(f"Stage3 failed seq={seq} code={proc.returncode}; see {log}")
        return ms

    mod = _ensure_stage3_module(a)
    if stage2_payload is not None and int(a.stage3_inmemory_cache):
        mod.realtime_cache_put_stage2(
            stage2_payload["record"], stage2_payload["center_xyz"],
            stage2_payload["poles"], stage2_payload["lines"], stage2_payload["vertices"],
            world_units_to_ft=.5, max_sequence_gap=a.max_sequence_gap,
        )
    old_argv = sys.argv[:]
    so = io.StringIO()
    se = io.StringIO()
    try:
        sys.argv = argv
        with contextlib.redirect_stdout(so), contextlib.redirect_stderr(se):
            mod.main()
    except Exception:
        se.write("\n--- PYTHON TRACEBACK ---\n" + traceback.format_exc())
        raise
    finally:
        sys.argv = old_argv
        log.write_text(so.getvalue() + "\n--- STDERR ---\n" + se.getvalue())
    return (time.perf_counter() - t0) * 1000.0


def stage3_breakdown(dest, gid):
    p = Path(dest) / "sessions" / Path(gid) / "summary.json"
    if not p.is_file():
        return {}
    try:
        d = json.loads(p.read_text())
    except Exception:
        return {}
    mapping = {
        "elapsed_seconds":"stage3_algorithm_ms",
        "fragment_join_seconds":"stage3_fragment_join_ms",
        "span_completion_pre_seconds":"stage3_span_completion_pre_ms",
        "hidden_pole_seconds":"stage3_hidden_pole_ms",
        "span_completion_post_seconds":"stage3_span_completion_post_ms",
        "chain_build_and_attachment_seconds":"stage3_chain_build_attachment_ms",
        "output_write_seconds":"stage3_output_write_ms",
    }
    out = {}
    for src, dst in mapping.items():
        try:
            out[dst] = float(d[src]) * 1000.0
        except Exception:
            pass
    return out


def _write_failure(out, gid, seq, rel, exc):
    payload = {
        "completed":False, "contract_version":CONTRACT_VERSION, "group_id":gid,
        "slice_seq":int(seq), "relative_path":rel,
        "exception_type":type(exc).__name__, "exception":str(exc),
        "traceback":traceback.format_exc(), "timestamp_unix":time.time(),
        "output_dir":str(Path(out).resolve()),
    }
    atomic_json(payload, Path(out) / "errors" / f"slice_{int(seq):06d}_FAILED.json")
    atomic_json(payload, Path(out) / "FAILED.json")


def main():
    a = args()
    inp = Path(a.input_dir).resolve()
    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = discover(inp, a.session_filter)
    rows = rows[:a.max_slices] if a.max_slices > 0 else rows
    geo, sess = a.session_filter.split("/", 1)

    pipe = V4RealtimePipeline(
        a.model_path, a.calibration_json, a.stage2_bundle,
        a.grid_size, a.voxel_size_ft, a.core_size, a.batch_size, a.amp,
        bool(a.compile_model), bool(a.evaluate_all_cores), bool(a.gpu_coord_channels),
        bool(a.fixed_batch_shape), a.pole_candidate_threshold, a.line_candidate_threshold,
        a.line_weak_threshold, a.line_competition_ratio, a.pole_min_voxels,
        a.line_min_voxels, a.edge_width_vox,
    )

    stage1_manifest_path = out / "stage1_manifest.csv"
    stage2_manifest_path = out / "inference_manifest.csv"
    stage2_alias_path = out / "stage2_manifest.csv"
    timing_path = out / "realtime_slice_timing.csv"
    stage1_manifest = _read_manifest(stage1_manifest_path, STAGE1_MANIFEST_COLUMNS)
    stage2_manifest = _read_manifest(stage2_manifest_path, STAGE2_MANIFEST_COLUMNS)
    if timing_path.is_file() and timing_path.stat().st_size:
        try:
            timings = pd.read_csv(timing_path)
        except EmptyDataError:
            timings = pd.DataFrame()
    else:
        timings = pd.DataFrame()
    timing_rows = timings.to_dict("records") if not timings.empty else []

    prev = None
    for i, (seq, src, rel) in enumerate(rows, 1):
        arrival_t0 = time.perf_counter()
        if prev is not None and seq <= prev:
            raise RuntimeError("Slice sequence must be strictly increasing")
        prev = seq
        sid = safe_id(rel.replace("/", "__"))
        s1_npz, s1_meta = stage1_paths(out, rel)
        polecsv, linecsv, vertcsv = stage2_paths(out, rel)
        stem = src.name[:-7] if src.name.endswith(".csv.gz") else src.stem
        rowcsv = out / "row_scores" / Path(rel).parent / f"{stem}_v4_realtime.csv.gz"

        tm = {
            "slice_order":i, "slice_seq":seq, "relative_path":rel,
            "csv_read_ms":0.0, "stage1_artifact_load_ms":0.0,
            "stage1_artifact_write_ms":0.0, "stage1_manifest_write_ms":0.0,
            "stage2_artifact_write_ms":0.0, "stage2_manifest_write_ms":0.0,
            "stage3_incremental_ms":0.0, "resume_from":"none",
        }
        try:
            s2row = _manifest_row_by_seq(stage2_manifest, a.session_filter, seq)
            # Repair a manifest-only interruption without recomputing Stage 2. All
            # three object artifacts are atomic, so their joint presence is a safe
            # durable boundary when the upstream Stage-1 manifest is complete.
            if a.resume and (not s2row or str(s2row.get("status", "")) != "completed") and polecsv.is_file() and linecsv.is_file() and vertcsv.is_file():
                s1_for_repair = _manifest_row_by_seq(stage1_manifest, a.session_filter, seq)
                if s1_for_repair and str(s1_for_repair.get("status", "")) == "completed":
                    try:
                        pc = pd.read_csv(polecsv); lc = pd.read_csv(linecsv)
                        s2row = {
                            "contract_version":CONTRACT_VERSION, "id":sid, "source":str(src),
                            "relative_path":rel, "geography":geo, "session":sess, "slice_seq":seq,
                            "group_id":a.session_filter,
                            "center_x":float(s1_for_repair["center_x"]), "center_y":float(s1_for_repair["center_y"]), "center_z":float(s1_for_repair["center_z"]),
                            "stage1_npz":str(s1_npz), "output_csv":"", "pole_csv":str(polecsv),
                            "line_csv":str(linecsv), "line_vertices_csv":str(vertcsv),
                            "rows":int(float(s1_for_repair.get("rows", 0))), "accepted_poles":len(pc),
                            "accepted_line_segments":len(lc), "status":"completed",
                        }
                        stage2_manifest = upsert_manifest_row(stage2_manifest_path, s2row, STAGE2_MANIFEST_COLUMNS)
                        atomic_csv(stage2_manifest, stage2_alias_path, STAGE2_MANIFEST_COLUMNS)
                        print(f"[v4-production] repaired Stage2 manifest seq={seq} from durable object artifacts", flush=True)
                    except Exception as repair_exc:
                        print(f"WARNING: Stage2 manifest repair skipped seq={seq}: {repair_exc}", flush=True)
                        s2row = None
            stage2_complete = bool(
                a.resume and s2row and str(s2row.get("status", "")) == "completed"
                and polecsv.is_file() and linecsv.is_file() and vertcsv.is_file()
            )
            stage2_payload = None
            center = None

            if stage2_complete:
                tm["resume_from"] = "stage2"
                center = {
                    "center_x":float(s2row["center_x"]), "center_y":float(s2row["center_y"]),
                    "center_z":float(s2row["center_z"]),
                }
            else:
                s1row = _manifest_row_by_seq(stage1_manifest, a.session_filter, seq)
                # Stage-1 NPZ/JSON are committed atomically before the manifest. If
                # only the manifest write was interrupted, repair it and reuse scores.
                if a.resume and (not s1row or str(s1row.get("status", "")) != "completed") and s1_npz.is_file() and s1_meta.is_file():
                    item0, pred0, meta0 = load_stage1_artifact(s1_npz, s1_meta)
                    center0 = dict(meta0.get("center_metadata", {}))
                    repaired = {
                        "contract_version":CONTRACT_VERSION, "id":str(meta0.get("id", sid)),
                        "source":str(meta0.get("source", src)), "relative_path":rel,
                        "geography":geo, "session":sess, "slice_seq":seq, "group_id":a.session_filter,
                        "center_x":center0.get("center_x", np.nan), "center_y":center0.get("center_y", np.nan), "center_z":center0.get("center_z", np.nan),
                        "stage1_npz":str(s1_npz), "stage1_meta_json":str(s1_meta),
                        "rows":int(meta0.get("rows", 0)), "occupied_rows":int(meta0.get("occupied_rows", len(item0["coords"]))),
                        "status":"completed",
                    }
                    stage1_manifest = upsert_manifest_row(stage1_manifest_path, repaired, STAGE1_MANIFEST_COLUMNS)
                    s1row = repaired
                    print(f"[v4-production] repaired Stage1 manifest seq={seq} from durable score artifact", flush=True)
                stage1_complete = bool(
                    a.resume and s1row and str(s1row.get("status", "")) == "completed"
                    and s1_npz.is_file() and s1_meta.is_file()
                )
                if stage1_complete:
                    tm["resume_from"] = "stage1"
                    t0 = time.perf_counter()
                    item, pred, meta = load_stage1_artifact(s1_npz, s1_meta)
                    tm["stage1_artifact_load_ms"] = (time.perf_counter() - t0) * 1000.0
                    center = dict(meta.get("center_metadata", {}))
                    if not center and s1row is not None:
                        center = {"center_x":float(s1row["center_x"]), "center_y":float(s1row["center_y"]), "center_z":float(s1row["center_z"])}
                    s1_timing = dict(meta.get("timing", {}))
                    tm.update({k: v for k, v in s1_timing.items() if k not in tm})
                else:
                    t0 = time.perf_counter()
                    df = pd.read_csv(src)
                    tm["csv_read_ms"] = (time.perf_counter() - t0) * 1000.0
                    s1 = pipe.run_stage1(df, sid, seq)
                    item, pred, center = s1["item"], s1["scores"], s1["center_metadata"]
                    tm.update(s1["timing"])
                    meta = {
                        "id":sid, "source":str(src), "relative_path":rel, "geography":geo,
                        "session":sess, "slice_seq":seq, "group_id":a.session_filter,
                        "rows":len(df), "occupied_rows":len(item["coords"]),
                        "center_metadata":center, "timing":s1["timing"],
                        "model_path":str(Path(a.model_path)), "calibration_json":str(Path(a.calibration_json)),
                        "evaluate_all_cores":bool(a.evaluate_all_cores),
                        "gpu_coord_channels":bool(a.gpu_coord_channels),
                        "fixed_batch_shape":bool(a.fixed_batch_shape), "amp":a.amp,
                    }
                    t0 = time.perf_counter()
                    save_stage1_artifact(s1_npz, s1_meta, item, pred, meta)
                    tm["stage1_artifact_write_ms"] = (time.perf_counter() - t0) * 1000.0
                    s1_manifest_row = {
                        "contract_version":CONTRACT_VERSION, "id":sid, "source":str(src),
                        "relative_path":rel, "geography":geo, "session":sess, "slice_seq":seq,
                        "group_id":a.session_filter, **center, "stage1_npz":str(s1_npz),
                        "stage1_meta_json":str(s1_meta), "rows":len(df),
                        "occupied_rows":len(item["coords"]), "status":"completed",
                    }
                    t0 = time.perf_counter()
                    stage1_manifest = upsert_manifest_row(
                        stage1_manifest_path, s1_manifest_row, STAGE1_MANIFEST_COLUMNS
                    )
                    tm["stage1_manifest_write_ms"] = (time.perf_counter() - t0) * 1000.0

                if not np.isfinite([center.get("center_x"), center.get("center_y"), center.get("center_z")]).all():
                    raise RuntimeError(f"Stage3 center metadata missing/nonfinite in {rel}")

                s2 = pipe.run_stage2(item, pred, sid, seq)
                tm.update(s2["timing"])
                t0 = time.perf_counter()
                atomic_csv(s2["poles"], polecsv, POLE_OUTPUT_COLUMNS)
                atomic_csv(s2["lines"], linecsv, LINE_OUTPUT_COLUMNS)
                atomic_csv(s2["vertices"], vertcsv, VERTEX_OUTPUT_COLUMNS)
                tm["stage2_artifact_write_ms"] = (time.perf_counter() - t0) * 1000.0

                if a.write_row_csv and not stage1_complete:
                    # Row diagnostics are optional and intentionally outside the normal
                    # production contract; reconstruct from the in-memory raw frame only.
                    rf = pipe._row_frame(df, item, pred, s2["raw_components"], s2)
                    rowcsv.parent.mkdir(parents=True, exist_ok=True)
                    atomic_csv(rf, rowcsv, compression={"method":"gzip", "compresslevel":1})

                s2row = {
                    "contract_version":CONTRACT_VERSION, "id":sid, "source":str(src),
                    "relative_path":rel, "geography":geo, "session":sess, "slice_seq":seq,
                    "group_id":a.session_filter, **center, "stage1_npz":str(s1_npz),
                    "output_csv":str(rowcsv) if a.write_row_csv and rowcsv.is_file() else "",
                    "pole_csv":str(polecsv), "line_csv":str(linecsv),
                    "line_vertices_csv":str(vertcsv),
                    "rows":int(s1row["rows"]) if stage1_complete and s1row else len(df),
                    "accepted_poles":len(s2["poles"]),
                    "accepted_line_segments":len(s2["lines"]), "status":"completed",
                }
                t0 = time.perf_counter()
                stage2_manifest = upsert_manifest_row(
                    stage2_manifest_path, s2row, STAGE2_MANIFEST_COLUMNS
                )
                atomic_csv(stage2_manifest, stage2_alias_path, STAGE2_MANIFEST_COLUMNS)
                tm["stage2_manifest_write_ms"] = (time.perf_counter() - t0) * 1000.0
                stage2_payload = {
                    "record":s2row,
                    "center_xyz":np.asarray([center["center_x"], center["center_y"], center["center_z"]], float),
                    "poles":s2["poles"], "lines":s2["lines"], "vertices":s2["vertices"],
                }

            stage3_dest = out / "stage3_incremental" / safe_id(a.session_filter) / f"slice_{seq:06d}"
            stage3_marker = stage3_dest / "COMPLETED.json"
            if a.resume and stage3_marker.is_file():
                tm["resume_from"] = "stage3"
                stage3_ms = 0.0
                stage3_detail = stage3_breakdown(stage3_dest, a.session_filter)
            else:
                stage3_dest.mkdir(parents=True, exist_ok=True)
                stage3_ms = run_stage3(a, out, a.session_filter, seq, stage3_dest, stage2_payload)
                stage3_detail = stage3_breakdown(stage3_dest, a.session_filter)
            tm["stage3_incremental_ms"] = stage3_ms
            tm.update(stage3_detail)
            tm["stage3_wrapper_overhead_ms"] = max(
                0.0, stage3_ms - stage3_detail.get("stage3_algorithm_ms", stage3_ms)
            ) if stage3_ms else 0.0

            # Window members are completed Stage-2 slices from this session only.
            acquired = sorted(set(
                int(r["slice_seq"]) for r in stage2_manifest.to_dict("records")
                if str(r.get("group_id")) == a.session_filter
                and seq - a.max_sequence_gap <= int(r["slice_seq"]) <= seq
                and str(r.get("status")) == "completed"
            ))
            atomic_json({
                "contract_version":CONTRACT_VERSION, "group_id":a.session_filter,
                "latest_slice":seq, "window_first_seq":seq-a.max_sequence_gap,
                "window_observed_slices":acquired, "window_observed_slice_count":len(acquired),
                "max_sequence_gap":a.max_sequence_gap,
                "max_observed_slice_centers":a.max_sequence_gap + 1,
                "slice_length_ft":a.slice_length_ft,
                "max_span_length_ft":a.max_span_length_ft,
                "output_dir":str(stage3_dest),
            }, out / "stage3_incremental" / safe_id(a.session_filter) / "LATEST.json")

            tm["stage3_window_observed_slices"] = len(acquired)
            tm["stage3_window_first_seq"] = seq - a.max_sequence_gap
            tm["stage3_output"] = str(stage3_dest)
            tm["arrival_to_publish_ms"] = (time.perf_counter() - arrival_t0) * 1000.0
            tm["end_to_end_update_ms"] = tm["arrival_to_publish_ms"]
            t0 = time.perf_counter()
            timing_rows = [r for r in timing_rows if int(r.get("slice_seq", -1)) != seq] + [tm]
            timing_rows = sorted(timing_rows, key=lambda r: int(r["slice_seq"]))
            atomic_csv(pd.DataFrame(timing_rows), timing_path)
            telemetry_ms = (time.perf_counter() - t0) * 1000.0
            print(
                f"[v4-production] {i}/{len(rows)} seq={seq} resume={tm['resume_from']} "
                f"stage1={tm.get('stage1_wall_ms',0.0):.1f}ms "
                f"stage2={tm.get('stage2_total_ms',0.0):.1f}ms stage3={stage3_ms:.1f}ms "
                f"publish={tm['arrival_to_publish_ms']:.1f}ms telemetry={telemetry_ms:.1f}ms",
                flush=True,
            )
        except Exception as exc:
            _write_failure(out, a.session_filter, seq, rel, exc)
            raise

    report = {
        "completed":True, "contract_version":CONTRACT_VERSION,
        "group_id":a.session_filter, "slices":len(rows),
        "first_slice_seq":rows[0][0], "last_slice_seq":rows[-1][0],
        "stage1_slice_local":True, "stage2_slice_local":True,
        "stages_independently_replayable":True,
        "stage1_artifact_manifest":str(stage1_manifest_path),
        "stage2_artifact_manifest":str(stage2_alias_path),
        "stage3_rolling_past_only":True,
        "stage3_max_sequence_gap":a.max_sequence_gap,
        "stage3_max_observed_slice_centers":a.max_sequence_gap + 1,
        "stage3_slice_length_ft":a.slice_length_ft,
        "stage3_max_span_length_ft":a.max_span_length_ft,
        "model_compiled":pipe.compiled, "amp":a.amp,
        "evaluate_all_cores":bool(a.evaluate_all_cores),
        "gpu_coord_channels":bool(a.gpu_coord_channels),
        "fixed_batch_shape":bool(a.fixed_batch_shape),
        "stage3_execution":a.stage3_execution,
        "stage3_inmemory_cache":bool(a.stage3_inmemory_cache),
    }
    if timing_rows:
        x = np.array([float(r["arrival_to_publish_ms"]) for r in timing_rows if float(r.get("arrival_to_publish_ms", 0)) > 0])
        if len(x):
            report["incremental_latency_ms"] = {
                "mean":float(x.mean()), "p50":float(np.quantile(x,.5)),
                "p95":float(np.quantile(x,.95)), "max":float(x.max()),
            }
    atomic_json(report, out / "COMPLETED.json")
    failed = out / "FAILED.json"
    if failed.exists():
        failed.unlink()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
