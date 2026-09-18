#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

INT_RE = re.compile(r"-?\d+")


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def ints_from_text(x) -> list[int]:
    if x is None:
        return []
    return [int(v) for v in INT_RE.findall(str(x))]


def resolve_artifact_path(value, replay_root: Path, manifest_parent: Path) -> Path:
    """Resolve durable Stage-2 paths without assuming the verifier CWD.

    Production currently writes absolute paths. Independent/recovery workflows may
    intentionally relocate artifacts and use relative paths, so the verifier also
    accepts paths relative to the manifest directory or replay root.
    """
    p = Path(str(value))
    if p.is_absolute():
        return p
    for base in (manifest_parent, replay_root):
        q = (base / p).resolve()
        if q.exists():
            return q
    return (manifest_parent / p).resolve()


def rolling_contract_ok(comp: dict, errs: list[str], warns: list[str]) -> None:
    """Validate current replay metadata while accepting the pre-hotfix legacy key."""
    for key in ("stage1_slice_local", "stage2_slice_local"):
        if comp.get(key) is not True:
            errs.append(f"{key} is not true")

    if "stage3_rolling_past_only" in comp:
        if comp.get("stage3_rolling_past_only") is not True:
            errs.append("stage3_rolling_past_only is not true")
    elif "stage3_multi_slice_only" in comp:
        # Older replay metadata used this field. It is weaker than the current
        # contract, but accepting it keeps old diagnostic bundles verifiable.
        if comp.get("stage3_multi_slice_only") is not True:
            errs.append("legacy stage3_multi_slice_only is not true")
        else:
            warns.append(
                "legacy replay metadata uses stage3_multi_slice_only; "
                "current replays should emit stage3_rolling_past_only"
            )
    else:
        errs.append(
            "replay metadata missing Stage3 rolling-window contract field "
            "(stage3_rolling_past_only)"
        )

    if "stages_independently_replayable" in comp and comp.get("stages_independently_replayable") is not True:
        errs.append("stages_independently_replayable is not true")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay_dir", required=True)
    ap.add_argument("--max_sequence_gap", type=int, default=9)
    ap.add_argument("--slice_length_ft", type=float, default=50.0)
    ap.add_argument("--max_span_length_ft", type=float, default=450.0)
    ap.add_argument("--require_all_stage3_snapshots", type=int, default=1)
    a = ap.parse_args()

    if a.max_sequence_gap != 9 or abs(a.slice_length_ft * 9 - a.max_span_length_ft) > 1e-9:
        raise SystemExit("ERROR production contract must be 9 sequence gaps x 50 ft = 450 ft")

    root = Path(a.replay_dir).resolve()
    manifest_path = root / "inference_manifest.csv"
    timing_path = root / "realtime_slice_timing.csv"
    complete_path = root / "COMPLETED.json"
    for p in (manifest_path, timing_path, complete_path):
        if not p.is_file():
            raise SystemExit(f"ERROR missing {p}")

    man = read_csv(manifest_path)
    tim = read_csv(timing_path)
    with complete_path.open() as f:
        comp = json.load(f)

    errs: list[str] = []
    warns: list[str] = []

    if not comp.get("completed"):
        errs.append("replay COMPLETED.json does not report completed=true")

    rolling_contract_ok(comp, errs, warns)

    if int(comp.get("stage3_max_sequence_gap", -1)) != a.max_sequence_gap:
        errs.append("max sequence gap contract mismatch")
    if int(comp.get("stage3_max_observed_slice_centers", -1)) != a.max_sequence_gap + 1:
        errs.append("max observed slice-center contract mismatch")
    if float(comp.get("stage3_max_span_length_ft", -1)) != a.max_span_length_ft:
        errs.append("max span length contract mismatch")

    if man.empty:
        errs.append("manifest empty")
    elif "slice_seq" not in man.columns:
        errs.append("manifest missing slice_seq")
    else:
        seq_num = pd.to_numeric(man["slice_seq"], errors="coerce")
        if seq_num.isna().any():
            errs.append("manifest contains invalid slice_seq")
        seq = seq_num.dropna().astype(int)
        if seq.duplicated().any():
            errs.append(f"duplicate slice_seq values: {sorted(seq[seq.duplicated()].unique())[:20]}")
        if not seq.is_monotonic_increasing:
            errs.append("manifest slice_seq is not increasing")

        for col in ("pole_csv", "line_csv", "line_vertices_csv"):
            if col not in man.columns:
                errs.append(f"manifest missing {col}")
                continue
            for v in man[col].astype(str):
                p = resolve_artifact_path(v, root, manifest_path.parent)
                if not p.is_file() or p.stat().st_size == 0:
                    errs.append(f"missing/empty Stage2 output: {p}")
                    if len(errs) > 50:
                        break

    if tim.empty:
        errs.append("timing CSV empty")
    elif "slice_seq" not in tim.columns:
        errs.append("timing CSV missing slice_seq")
    else:
        tseq = pd.to_numeric(tim["slice_seq"], errors="coerce")
        if tseq.isna().any():
            errs.append("timing CSV contains invalid slice_seq")
        tseq = tseq.dropna().astype(int)
        if tseq.duplicated().any():
            errs.append("timing CSV has duplicate slice_seq rows")
        if not man.empty and "slice_seq" in man.columns:
            mseq = set(pd.to_numeric(man["slice_seq"], errors="coerce").dropna().astype(int))
            missing = mseq - set(tseq)
            if missing:
                warns.append(f"timing rows missing for {len(missing)} manifest slices; first={sorted(missing)[:10]}")
        if "stage3_window_observed_slices" in tim.columns:
            w = pd.to_numeric(tim["stage3_window_observed_slices"], errors="coerce").dropna()
            if len(w) and int(w.max()) > a.max_sequence_gap + 1:
                errs.append(f"rolling window exceeded {a.max_sequence_gap + 1} observed slice centers")
        if "stage3_window_first_seq" in tim.columns:
            wf = pd.to_numeric(tim["stage3_window_first_seq"], errors="coerce")
            paired = pd.DataFrame({"slice_seq": pd.to_numeric(tim["slice_seq"], errors="coerce"), "first": wf}).dropna()
            if not paired.empty:
                bad = paired[paired["first"].astype(int) != paired["slice_seq"].astype(int) - a.max_sequence_gap]
                if not bad.empty:
                    errs.append(
                        "timing CSV contains an incorrect Stage3 window lower bound; "
                        f"first bad rows={bad.head(5).to_dict('records')}"
                    )

    stage3_root = root / "stage3_incremental"
    latest = list(stage3_root.rglob("LATEST.json")) if stage3_root.exists() else []
    if len(latest) != 1:
        errs.append(f"expected one LATEST.json for one replay session, found {len(latest)}")

    snapshots = [p for p in stage3_root.rglob("slice_*") if p.is_dir()] if stage3_root.exists() else []
    expected_seqs = (
        set(pd.to_numeric(man["slice_seq"], errors="coerce").dropna().astype(int))
        if not man.empty and "slice_seq" in man.columns
        else set()
    )
    found_seqs: set[int] = set()
    gid = str(comp.get("group_id", ""))

    for snap in snapshots:
        m = re.search(r"slice_(\d+)$", snap.name)
        if not m:
            continue
        latest_seq = int(m.group(1))
        found_seqs.add(latest_seq)
        cp = snap / "COMPLETED.json"
        if not cp.is_file():
            errs.append(f"missing Stage3 COMPLETED.json: {snap}")
            continue
        with cp.open() as f:
            c = json.load(f)
        rules = c.get("rules", {})
        if float(rules.get("max_span_length_ft", -1)) != a.max_span_length_ft:
            errs.append(f"{snap}: max_span_length_ft mismatch")
        if int(rules.get("max_sequence_gap", rules.get("max_span_slices", -1))) != a.max_sequence_gap:
            errs.append(f"{snap}: max_sequence_gap mismatch")
        if int(rules.get("max_observed_slice_centers", a.max_sequence_gap + 1)) != a.max_sequence_gap + 1:
            errs.append(f"{snap}: max observed centers mismatch")
        if float(rules.get("slice_length_ft", a.slice_length_ft)) != a.slice_length_ft:
            errs.append(f"{snap}: slice_length_ft mismatch")
        if float(rules.get("sequence_gap_times_slice_length_ft", a.max_span_length_ft)) != a.max_span_length_ft:
            errs.append(f"{snap}: sequence-gap distance mismatch")

        lo = latest_seq - a.max_sequence_gap
        session_dir = snap / "sessions" / Path(gid)
        ch = read_csv(session_dir / "conductor_chains.csv") if session_dir.exists() else pd.DataFrame()
        if not ch.empty:
            for col in ("slice_min", "slice_max", "start_slice", "end_slice"):
                if col not in ch.columns:
                    continue
                vals = pd.to_numeric(ch[col], errors="coerce").dropna().astype(int)
                if col in ("slice_min", "start_slice") and (vals < lo).any():
                    errs.append(f"{snap}: conductor uses slice older than {lo}")
                if col in ("slice_max", "end_slice") and (vals > latest_seq).any():
                    errs.append(f"{snap}: conductor uses future slice > {latest_seq}")

        wp = read_csv(session_dir / "world_poles.csv") if session_dir.exists() else pd.DataFrame()
        if not wp.empty and "source_slices" in wp.columns:
            for x in wp["source_slices"].dropna():
                ss = ints_from_text(x)
                if any(v < lo for v in ss):
                    errs.append(f"{snap}: pole source slice older than {lo}")
                if any(v > latest_seq for v in ss):
                    errs.append(f"{snap}: pole source slice in future of {latest_seq}")

    if a.require_all_stage3_snapshots and expected_seqs - found_seqs:
        errs.append(
            f"missing Stage3 snapshots for {len(expected_seqs-found_seqs)} slices; "
            f"first={sorted(expected_seqs-found_seqs)[:10]}"
        )

    report = {
        "verification_version": "v4-replay-verifier-hotfix-20260825",
        "replay_dir": str(root),
        "manifest_rows": len(man),
        "timing_rows": len(tim),
        "stage3_snapshots": len(found_seqs),
        "max_sequence_gap": a.max_sequence_gap,
        "max_observed_slice_centers": a.max_sequence_gap + 1,
        "slice_length_ft": a.slice_length_ft,
        "max_span_length_ft": a.max_span_length_ft,
        "errors": errs,
        "warnings": warns,
        "ok": not errs,
    }
    print(json.dumps(report, indent=2))
    (root / "REALTIME_REPLAY_VERIFICATION.json").write_text(json.dumps(report, indent=2))
    if errs:
        raise SystemExit("V4_REALTIME_REPLAY_VERIFICATION_FAILED")
    print("V4_REALTIME_REPLAY_VERIFICATION_OK")


if __name__ == "__main__":
    main()
