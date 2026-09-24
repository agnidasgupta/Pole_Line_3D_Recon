#!/usr/bin/env python3
"""Apply the E8 result gate to control/candidate timing summaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

def load(path):
    return json.loads(Path(path).read_text())

def mean(records, key):
    values = [r["means_ms"].get(key) for r in records]
    values = [v for v in values if isinstance(v, (int, float))]
    return sum(values) / len(values) if values else None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--control", action="append", required=True)
    p.add_argument("--candidate", action="append", required=True)
    p.add_argument("--minimum-save-percent", type=float, default=1.0)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    controls = [load(path) for path in a.control]
    candidates = [load(path) for path in a.candidate]
    control_total = mean(controls, "slice_total_ms")
    candidate_total = mean(candidates, "slice_total_ms")
    if control_total is None or candidate_total is None or control_total <= 0:
        status = "REJECTED_INCOMPLETE_TIMING"
        saved = None
    else:
        saved = 100.0 * (control_total - candidate_total) / control_total
        status = "PASS" if saved >= a.minimum_save_percent else "REJECTED_NO_MEANINGFUL_GAIN"
    payload = {
        "status": status,
        "minimum_save_percent": a.minimum_save_percent,
        "control_mean_slice_total_ms": control_total,
        "candidate_mean_slice_total_ms": candidate_total,
        "saved_percent": saved,
        "controls": controls,
        "candidates": candidates,
    }
    Path(a.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))

if __name__ == "__main__":
    main()
