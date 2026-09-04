# V4 Stage2 Stage1-Exact V8

Purpose: verify and preserve Stage-1 inferred line labels in Stage 2 without any
pole-pair line inference, GT runtime use, Stage-2 line hysteresis, or Stage-2 line
refiner rejection.

## Runtime invariant

For each saved Stage-1 slice:

1. Resolve the final deployed Stage-1 labels (explicit saved deployed label when
   available, otherwise production `label_from_scores` with accepted calibration).
2. Select exactly `label == 2` occupied voxels.
3. Preserve production V4 pole outputs unchanged.
4. Replace Stage-2 line geometry with edges between 26-neighbour Stage-1 class-2
   voxels only. Each edge is at most one voxel diagonal (`sqrt(3)*0.5 ft`).
5. Retain isolated Stage-1 class-2 voxels in the voxel audit but do not invent a
   connection for them.
6. Require exact voxel preservation = 1.0.

No GT is read. No pole pair is enumerated. No gap is bridged. No synthetic line
endpoint is generated. This is a verification-oriented Stage-2 experiment.

Canonical output directory remains:

`.../v4_stage23_quality/full_run_v2_<UTC>/stage2/<SID>`

with sibling directories `stage3/`, `selection/`, `logs/`, `timing/`, `status/`.
Stage3 is reserved and not run.
