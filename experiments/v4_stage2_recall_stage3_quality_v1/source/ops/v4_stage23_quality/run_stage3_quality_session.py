#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path


def truthy(v):
    return str(v).strip().lower() in {"1", "true", "yes", "y", "completed", "ok"}


def read_sequences(manifest: Path, group_id: str):
    rows = list(csv.DictReader(manifest.open(newline="")))
    out = []
    for r in rows:
        gid = str(r.get("group_id", r.get("session_filter", ""))).strip()
        if gid and gid != group_id:
            continue
        status = str(r.get("status", r.get("completed", "completed"))).strip()
        if "status" in r and status and status.lower() not in {"completed", "ok", "done", "success", "1", "true"}:
            continue
        val = r.get("slice_seq", r.get("sequence", r.get("seq")))
        if val is None or str(val).strip() == "":
            continue
        out.append(int(float(val)))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--session_filter", required=True)
    ap.add_argument("--stage3_script", required=True)
    ap.add_argument("--max_sequence_gap", type=int, default=9)
    ap.add_argument("--slice_length_ft", type=float, default=50.0)
    ap.add_argument("--max_span_length_ft", type=float, default=450.0)
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--disable_plots", type=int, default=1)
    args = ap.parse_args()

    s2 = Path(args.stage2_dir)
    manifest = s2 / "inference_manifest.csv"
    if not manifest.is_file():
        found = list(s2.rglob("inference_manifest.csv"))
        if len(found) != 1:
            raise SystemExit(f"Expected exactly one inference_manifest.csv under {s2}; found {len(found)}")
        manifest = found[0]
    seqs = read_sequences(manifest, args.session_filter)
    if not seqs:
        raise SystemExit(f"No slice_seq rows for {args.session_filter} in {manifest}")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    run_rows = []
    for n, seq in enumerate(seqs, 1):
        slice_out = out_root / f"slice_{seq:06d}"
        marker = slice_out / "QUALITY_STAGE3_COMPLETED.json"
        if args.resume and marker.is_file():
            print(f"[quality-stage3] resume {n}/{len(seqs)} seq={seq}", flush=True)
            continue
        slice_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            args.stage3_script,
            "--inference_dir", str(s2),
            "--output_dir", str(slice_out),
            "--session_filter", args.session_filter,
            "--latest_slice", str(seq),
            "--max_sequence_gap", str(args.max_sequence_gap),
            "--slice_length_ft", str(args.slice_length_ft),
            "--max_span_length_ft", str(args.max_span_length_ft),
        ]
        if args.disable_plots:
            cmd.append("--disable_plots")
        # Intentionally DO NOT pass --realtime_inmemory_cache. Every newest slice
        # is reconstructed from the complete legal Stage-2 window from scratch.
        print(f"[quality-stage3] run {n}/{len(seqs)} seq={seq}", flush=True)
        p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (slice_out / "stage3_stdout.log").write_text(p.stdout)
        if p.returncode != 0:
            print(p.stdout, file=sys.stderr)
            raise SystemExit(f"Stage3 failed seq={seq} rc={p.returncode}")
        marker.write_text(json.dumps({
            "group_id": args.session_filter,
            "slice_seq": seq,
            "execution": "fresh_process_full_legal_window",
            "max_sequence_gap": args.max_sequence_gap,
            "slice_length_ft": args.slice_length_ft,
            "max_span_length_ft": args.max_span_length_ft,
        }, indent=2) + "\n")
        run_rows.append({"group_id": args.session_filter, "slice_seq": seq, "status": "completed"})

    (out_root / "QUALITY_STAGE3_SESSION_COMPLETED.json").write_text(json.dumps({
        "group_id": args.session_filter,
        "slice_count": len(seqs),
        "first_slice": seqs[0],
        "last_slice": seqs[-1],
        "fresh_recompute": True,
    }, indent=2) + "\n")
    print(f"QUALITY_STAGE3_SESSION_OK group={args.session_filter} slices={len(seqs)}")


if __name__ == "__main__":
    main()
