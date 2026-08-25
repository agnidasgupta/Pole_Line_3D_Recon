#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


def write_csv(path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=columns).to_csv(path, index=False)


def build_replay(root: Path, *, current_key: bool = True, bad_window: bool = False) -> None:
    gid = "59768101-C4990BB/2026_session3"
    rows = []
    timing = []
    for seq in (10, 11, 12):
        obj = root / "stage2_objects" / f"slice_{seq:06d}"
        pole = obj / "poles.csv"
        line = obj / "lines.csv"
        vert = obj / "vertices.csv"
        write_csv(pole, ["component_id"])
        write_csv(line, ["component_id"])
        write_csv(vert, ["component_id"])
        rows.append(
            {
                "slice_seq": seq,
                "group_id": gid,
                "pole_csv": str(pole),
                "line_csv": str(line),
                "line_vertices_csv": str(vert),
            }
        )
        timing.append(
            {
                "slice_seq": seq,
                "stage3_window_observed_slices": 11 if bad_window and seq == 12 else min(3, seq - 9),
                "stage3_window_first_seq": seq - 9,
            }
        )
        snap = root / "stage3_incremental" / "59768101-C4990BB__2026_session3" / f"slice_{seq:06d}"
        session = snap / "sessions" / "59768101-C4990BB" / "2026_session3"
        write_csv(session / "conductor_chains.csv", ["slice_min", "slice_max"])
        write_csv(session / "world_poles.csv", ["source_slices"])
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "COMPLETED.json").write_text(
            json.dumps(
                {
                    "completed": True,
                    "rules": {
                        "max_span_length_ft": 450.0,
                        "max_span_slices": 9,
                        "max_sequence_gap": 9,
                        "max_observed_slice_centers": 10,
                        "slice_length_ft": 50.0,
                        "sequence_gap_times_slice_length_ft": 450.0,
                    },
                }
            )
        )

    latest_dir = root / "stage3_incremental" / "59768101-C4990BB__2026_session3"
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / "LATEST.json").write_text(json.dumps({"latest_slice": 12}))

    pd.DataFrame(rows).to_csv(root / "inference_manifest.csv", index=False)
    pd.DataFrame(timing).to_csv(root / "realtime_slice_timing.csv", index=False)
    comp = {
        "completed": True,
        "group_id": gid,
        "stage1_slice_local": True,
        "stage2_slice_local": True,
        "stages_independently_replayable": True,
        "stage3_max_sequence_gap": 9,
        "stage3_max_observed_slice_centers": 10,
        "stage3_max_span_length_ft": 450.0,
    }
    if current_key:
        comp["stage3_rolling_past_only"] = True
    else:
        comp["stage3_multi_slice_only"] = True
    (root / "COMPLETED.json").write_text(json.dumps(comp))


def run_verify(root: Path) -> subprocess.CompletedProcess[str]:
    script = Path(__file__).with_name("verify_v4_realtime_replay.py")
    return subprocess.run(
        [
            sys.executable,
            str(script),
            "--replay_dir",
            str(root),
            "--max_sequence_gap",
            "9",
            "--slice_length_ft",
            "50",
            "--max_span_length_ft",
            "450",
        ],
        text=True,
        capture_output=True,
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "current"
        root.mkdir()
        build_replay(root, current_key=True)
        p = run_verify(root)
        assert p.returncode == 0, p.stdout + p.stderr
        report = json.loads((root / "REALTIME_REPLAY_VERIFICATION.json").read_text())
        assert report["ok"] is True
        assert not report["errors"]

        legacy = Path(td) / "legacy"
        legacy.mkdir()
        build_replay(legacy, current_key=False)
        p = run_verify(legacy)
        assert p.returncode == 0, p.stdout + p.stderr
        report = json.loads((legacy / "REALTIME_REPLAY_VERIFICATION.json").read_text())
        assert report["ok"] is True
        assert any("legacy replay metadata" in x for x in report["warnings"])

        bad = Path(td) / "bad_window"
        bad.mkdir()
        build_replay(bad, current_key=True, bad_window=True)
        p = run_verify(bad)
        assert p.returncode != 0, "invalid 11-center window unexpectedly passed"
        report = json.loads((bad / "REALTIME_REPLAY_VERIFICATION.json").read_text())
        assert report["ok"] is False
        assert any("rolling window exceeded 10" in x for x in report["errors"])

    print("V4_REPLAY_VERIFIER_CONTRACT_SMOKE_OK")


if __name__ == "__main__":
    main()
