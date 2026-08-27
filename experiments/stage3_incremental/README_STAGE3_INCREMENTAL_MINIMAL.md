# V4 Stage 3 Minimal Incremental Join Experiment

This package is for branch `v4-stage3-incremental` only.

It does not replace the Stage 3 reconstruction algorithm. It makes only the fragment candidate graph / pair scoring persistent while preserving the existing batch/global downstream functions and output schemas.

Files:
- `v4/stage3_incremental_runtime.py` - persistent Stage3Session and incremental joiner.
- `v4/smoke_test_v4_stage3_incremental_joiner.py` - batch-equivalence smoke test.
- `tools/apply_stage3_incremental_minimal.py` - marker-checked two-location patcher for the branch's current `v4/reconstruct_v4_stage3.py`.
- `reconstruct_v4_stage3_minimal.diff` - review-only diff against the accepted Hotfix1 Stage 3 source.

Experiment 1 intentionally keeps these global/current-window operations unchanged:
- merge_poles
- intervening-fragment guard evaluation
- final strict/bridge greedy arbitration
- build_track_components
- span completion
- hidden-pole inference
- chain construction / pole attachment
- all output CSV/audit schemas

The persistent joiner retains only `[S-9,S]`, supports missing sequence numbers and empty-fragment slices, and can bootstrap from the authoritative current rolling window after a process restart or resume.
