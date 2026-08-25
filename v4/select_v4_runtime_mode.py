#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

MODE_FLAGS = {
    "full_cpu":   (1, 0, 0),
    "active_cpu": (0, 0, 1),
    "full_gpu":   (1, 1, 0),
    "active_gpu": (0, 1, 1),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate_json", required=True)
    ap.add_argument("--output_env", required=True)
    a = ap.parse_args()
    with open(a.gate_json) as f:
        d = json.load(f)
    summary = d.get("summary", {})
    reported = str(d.get("recommended_runtime", ""))

    passing = [
        name for name in ("active_gpu", "active_cpu", "full_gpu")
        if bool(summary.get(name, {}).get("pass_recommended", False))
    ]
    if reported in passing:
        mode = reported
    elif passing:
        mode = max(passing, key=lambda n: float(summary.get(n, {}).get("speedup_mean", 0.0)))
    else:
        mode = "full_cpu"

    all_cores, gpu_coords, fixed_batch = MODE_FLAGS[mode]
    out = Path(a.output_env)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"export V4_RUNTIME_MODE={mode}\n"
        f"export EVALUATE_ALL_CORES={all_cores}\n"
        f"export GPU_COORD_CHANNELS={gpu_coords}\n"
        f"export V4_FIXED_BATCH_SHAPE={fixed_batch}\n"
        f"export V4_RUNTIME_GATE_JSON={Path(a.gate_json).resolve()}\n"
    )
    print(json.dumps({
        "selected": mode,
        "evaluate_all_cores": all_cores,
        "gpu_coord_channels": gpu_coords,
        "fixed_batch_shape": fixed_batch,
        "summary": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
