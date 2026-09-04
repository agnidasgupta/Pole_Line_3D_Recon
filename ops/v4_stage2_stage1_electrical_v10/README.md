# V4 Stage2 Stage1 Electrical Tracks V10

Stage2-only experiment. Production `v4/` remains unchanged. Stage1 is not rerun and Stage3 is not run.

## Why V10

V9 successfully connected fragmented Stage1 class-2 line voxels, but its bridge selection could merge two nearby conductors. The defect had three parts:

1. V9 chose the shortest of four fragment-endpoint combinations rather than requiring true end-to-end continuation.
2. V9 allowed enough lateral/angle tolerance for a nearby parallel or converging line to be mistaken for the continuation.
3. V9 had no pole-neighborhood topology rule, so two lines approaching the same pole could be joined to each other before reaching the pole.

## V10 electrical rules

- Runtime starts only from deployed Stage1 `label == 2` voxels.
- Production pole detections are preserved.
- Same-line bridges require longitudinal end-to-end continuation.
- Longitudinally overlapping fragments cannot be merged.
- Signed lane-center displacement is tightly bounded.
- Each fragment endpoint side can be used by at most one bridge.
- A tentative merged track must remain within the learned single-conductor lateral radius.
- Any line-to-line bridge passing through a detected pole neighborhood is forbidden.
- Separate conductor tracks may independently attach to the detected pole surface.
- Pole attachments use distinct surface points based on each line's approach direction/height; tracks remain separate components.
- No pole-pair enumeration.
- No runtime GT.
- No synthetic line voxels.

## Output contract

Unchanged:

```
/workspace/voxel_poleline/outputs/poleline_voxel_run_session_groups/v4_stage23_quality/full_run_v2_<UTC>/
  stage2/<SID>/
  stage3/                  # reserved, empty
  selection/
  logs/stage2/
  logs/stage3/             # reserved, empty
  timing/stage2/
  status/
  RUN_INFO.txt
  session_map.tsv
  PHASE2_STAGE2_OK.txt
  STAGE2_ONLY_COMPLETE.txt
```

## Diagnostics per slice

- `*_stage1_line_voxels.csv`
- `*_accepted_line_voxels.csv`
- `*_fragment_bridge_candidates.csv`
- `*_selected_fragment_bridges.csv`
- `*_stage1_electrical_tracks.csv`
- `*_pole_attachments.csv`
- `*_stage1_electrical_track_audit.json`

Important audit counters include:

- `parallel_or_cross_lane_bridges_blocked`
- `near_pole_line_to_line_bridges_blocked`
- `track_drift_bridges_blocked`
- `pole_attachments`

Run the target session first, inspect Velasco ordinals 20-39, then run all saved sessions only after the target looks electrically correct.
