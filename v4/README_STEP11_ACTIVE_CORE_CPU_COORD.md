# V4 realtime Step 11 — isolate active-core scheduling from GPU coordinate generation

The original combined equivalence test compared two changes at once:

- reference: full core tiling + CPU coordinate channels
- optimized: active core tiling + GPU coordinate channels

If labels match but score deltas exceed the strict tolerance, that does not tell us which optimization caused the drift.

This patch exposes `GPU_COORD_CHANNELS` independently and adds a four-way H100 gate:

1. `full_cpu` — exact reference
2. `active_cpu` — only skip empty output cores
3. `full_gpu` — only move coordinate generation to GPU
4. `active_gpu` — both optimizations

Production recommendation is `active_cpu` only if it matches `full_cpu` under the score tolerance with zero Stage-1 label mismatches. Otherwise use `full_cpu`.

Stages 1 and 2 remain one-slice-only. Stage 3 remains the only multi-slice stage and is unchanged.
