#!/usr/bin/env python3
"""Equivalence smoke test for the minimal persistent Stage-3 join experiment."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import reconstruct_v4_stage3 as s3
from stage3_incremental_runtime import Stage3IncrementalJoiner, Stage3Session


def args():
    old = sys.argv[:]
    sys.argv = ["prog", "--inference_dir", "/tmp/x", "--output_dir", "/tmp/y"]
    try:
        return s3.pa()
    finally:
        sys.argv = old


def make_frag(name, seq, p0, p1, n=8):
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = np.asarray(p0, float)[None, :] * (1.0 - t) + np.asarray(p1, float)[None, :] * t
    pts = s3.order_xy(pts)
    return {
        "group_id": "G", "geography": "geo", "session": "s",
        "slice_seq": int(seq), "relative_path": f"{name}.csv",
        "source_key": str(name), "points": pts,
        "direction": s3.fragment_direction(pts), "score": 0.9,
    }


def canon(frags, accepted):
    return {
        (
            frags[c["i"]]["source_key"], int(c["i_end"]),
            frags[c["j"]]["source_key"], int(c["j_end"]),
            str(c["join_mode"]), round(float(c["cost"]), 12),
        )
        for c in accepted
    }


def assert_equivalent(a, slices, label):
    inc = Stage3IncrementalJoiner(a, s3)
    window = []
    for seq, rows in slices:
        got = inc.add_slice(seq, rows)
        got_frags = inc.window_fragments()
        window.extend((seq, dict(f)) for f in rows)
        window = [(ss, f) for ss, f in window if ss >= seq - a.max_span_slices]
        ref_frags = [dict(f) for _, f in window]
        ref = s3.select_one_to_one_joins(ref_frags, a)
        if canon(got_frags, got) != canon(ref_frags, ref):
            raise AssertionError(f"incremental != batch: {label} seq={seq}")
    print(f"PASS {label}")


def displacement(a):
    return [
        (1, [make_frag("A", 1, (0, 0, 30), (40, 0, 30))]),
        (2, [make_frag("B", 2, (50, 0, 30), (90, 0, 30))]),
        (3, [make_frag("D", 3, (101, .4, 30), (140, .4, 30))]),
        (4, [make_frag("E", 4, (92, .05, 30), (131, .05, 30))]),
    ]


def random_corridor(seed, n_slices=30):
    rng = np.random.default_rng(seed)
    out = []
    n_cond = int(rng.integers(2, 4))
    y0 = rng.uniform(-2, 2, n_cond)
    z0 = rng.uniform(28, 34, n_cond)
    fid = 0
    for seq in range(n_slices):
        if rng.random() < .12:
            continue
        rows = []
        x0 = 50.0 * seq
        for k in range(n_cond):
            if rng.random() < .15:
                continue
            p0 = (x0 + rng.uniform(0, 6), y0[k] + rng.normal(0, .15), z0[k] + rng.normal(0, .3))
            p1 = (x0 + 50 - rng.uniform(0, 6), y0[k] + rng.normal(0, .15), z0[k] + rng.normal(0, .3))
            rows.append(make_frag(f"f{fid}", seq, p0, p1)); fid += 1
        for _ in range(int(rng.integers(0, 3))):
            cx, cy, cz = x0 + rng.uniform(0, 50), rng.uniform(-3, 3), rng.uniform(27, 35)
            rows.append(make_frag(
                f"f{fid}", seq, (cx, cy, cz),
                (cx + rng.uniform(4, 10), cy + rng.normal(0, .3), cz + rng.normal(0, .3))
            )); fid += 1
        out.append((seq, rows))
    return out


def bootstrap_resume(a):
    stream = displacement(a) + [
        (11, [make_frag("F", 11, (150, 0, 30), (190, 0, 30))]),
        (13, []),  # legal missing sequence and zero-fragment arrival
    ]
    window = []
    for seq, rows in stream:
        window.extend((seq, dict(f)) for f in rows)
        window = [(ss, f) for ss, f in window if ss >= seq - a.max_span_slices]
    frags = [f for _, f in window]
    sess = Stage3Session(a, "G", s3)
    got = sess.select_for_window(13, frags)
    ref = s3.select_one_to_one_joins([dict(f) for f in frags], a)
    if canon(sess.joiner.window_fragments(), got) != canon(frags, ref):
        raise AssertionError("bootstrap/resume window != batch")
    print("PASS bootstrap/resume + missing/empty slice")


def main():
    a = args()
    assert_equivalent(a, displacement(a), "newcomer displacement")
    for seed in (1, 2, 3, 7, 11, 42, 1234, 20260826):
        assert_equivalent(a, random_corridor(seed), f"random corridor seed={seed}")
    bootstrap_resume(a)
    print("ALL MINIMAL INCREMENTAL JOIN EQUIVALENCE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
