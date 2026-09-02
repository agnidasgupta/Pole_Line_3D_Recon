# V4 Stage-2 GT auto-selection V3

This experiment leaves V4 Stage 1, Stage-2 source code, the learned ExtraTrees models, and Stage 3 unchanged.

It uses `raw_labels` embedded in saved Stage-1 NPZ artifacts only for offline profile selection. Runtime Stage 2 uses the normal V4 entry point and selected scalar thresholds.

## Programmatic target

The selector finds GT line components that the accepted V4 baseline misses and that are either:

- part of a parallel conductor bundle where another GT lane is already detected; or
- geometrically bracketed by two detected poles in the session.

It sweeps candidate/weak/competition thresholds plus line-refiner thresholds, then rejects profiles that materially reduce exact or near-GT precision on the target session or a deterministic sample of other labeled sessions.

If no safe profile improves recovery, the workflow writes `NO_SAFE_GT_IMPROVEMENT.txt`, packages diagnostics, and does not run a misleading Stage-2 output.

## Default target

`VELASCO_CUT_CP/session1`

## Nebius run

```bash
RUN_SCOPE=target \
nohup /workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_gt_autoselect_v3/ops/v4_stage2_gt_autoselect_v3/run_v4_stage2_gt_autoselect_v3_on_nebius.sh \
  > "$HOME/v4_stage2_gt_autoselect_v3_master.log" 2>&1 &

echo $! > "$HOME/LATEST_V4_STAGE2_GT_AUTOSELECT_V3_PID.txt"
```

Use `RUN_SCOPE=all` only after the target selector returns `SELECTED_SAFE_GT_IMPROVEMENT`; the decision itself remains programmatic.
