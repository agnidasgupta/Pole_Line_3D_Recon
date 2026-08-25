# Pole / Line 3D Reconstruction — V4 Realtime Production Candidate

This repository is intentionally **V4-only**. It packages the accepted V4 `MultiHeadVoxelNet3D` checkpoint into a production-oriented three-stage realtime pipeline.

## Non-negotiable realtime contract

| Stage | Input allowed | State/history allowed | Purpose |
|---|---|---|---|
| Stage 1 | exactly the newly arrived slice | none | V4 voxel pole/power-line scoring |
| Stage 2 | exactly that same slice | none | local sparse components + local learned refiner + pole/line parameterization |
| Stage 3 | current slice plus already-acquired past slices only | rolling past window only | full pole/conductor reconstruction |

Stage 3 is executed **after every new slice**. For newest sequence `S`, it may read only manifest rows satisfying:

```text
S - 9 <= slice_seq <= S
```

The maximum physical span remains 450 ft. Missing sequence numbers are allowed; future slices are prohibited. The production runner rejects all-session reconstruction and rejects any change from the 9-increment / 450-ft rolling contract.

## Accepted Stage-1 model

The package does **not retrain Stage 1**. It uses the accepted V4 checkpoint and calibration:

```text
/outputs/poleline_voxel_run_session_groups/precision_v4/train/precision_best.pt
/outputs/poleline_voxel_run_session_groups/precision_v4/full_val/calibration.json
```

The accepted V4 network remains:

```text
model             MultiHeadVoxelNet3D
patch              64 x 64 x 64
output core        48 x 48 x 48
base channels      16
coordinate input   local x/y/z only
extra input        dist_center_ft
normalization      GroupNorm
```

No center/world/session coordinate is used by Stage 1 or Stage 2. Center metadata is carried forward only so Stage 3 can convert local detections to world coordinates.

## Why the runtime is being optimized around the accepted weights

The attached 100-slice V4 raw benchmark measured the reference Stage-1 implementation at approximately:

```text
mean end-to-end including raw CSV read       1,890.0 ms/slice
mean end-to-end excluding raw CSV read       1,830.0 ms/slice
CPU input preprocessing                         19.6 ms
CPU patch construction                         354.8 ms
host -> device                                  703.4 ms
GPU model + score fusion/threshold              603.9 ms
device -> host                                   28.8 ms
CPU dense scatter                               108.4 ms
```

That benchmark shows substantial time outside convolution execution. This candidate therefore optimizes scheduling, transfers, sparse local processing, persistent process state, and incremental reconstruction without altering accepted Stage-1 weights.

## Stage 1 runtime variants

Four V4-equivalent runtime variants exist only to isolate where speed is gained:

```text
full_cpu    reference full 64^3 tiling + original CPU coordinate-channel construction
active_cpu  skip empty 48^3 output cores + original CPU coordinate-channel construction
full_gpu    full tiling + GPU coordinate-channel construction
active_gpu  active-core scheduling + GPU coordinate-channel construction
```

The production selector is deliberately conservative:

- `full_cpu` is always safe.
- `active_cpu` is selected automatically only when the H100 equivalence gate shows score deltas within tolerance **and zero Stage-1 label mismatches**.
- GPU-coordinate variants are measured but are not selected automatically.

Run `run_v4_runtime_variant_gate_on_nebius.sh` before Stage-2 mining. The selected result is stored in:

```text
/outputs/poleline_voxel_run_session_groups/v4_realtime/diagnostics/v4_runtime_mode.env
```

Stage-2 mining and realtime replay automatically source that same file, ensuring Stage 2 is trained on exactly the Stage-1 runtime path used in production.

## Stage 2

Stage 2 is strictly current-slice-local:

1. candidate pole/line voxels are formed from Stage-1 scores;
2. exact sparse 26-neighbor connected components are constructed without allocating a dense 400x400x200 component-label volume;
3. local geometric/component descriptors are computed;
4. learned `ExtraTreesClassifier` pole and line refiners accept/reject components;
5. accepted objects are parameterized into pole CSVs, line-segment CSVs, and line-vertex CSVs.

GT overlap information may be used **offline only to construct Stage-2 training targets**. GT overlap fields, center fields, and world fields are not model features. The source-contract validator enforces this.

## Stage 3

Stage 3 is the only multi-slice stage. It runs after every committed Stage-1/2 slice and reads only the acquired rolling window. The Python reconstruction module stays imported in the persistent replay process by default (`STAGE3_EXECUTION=inprocess`), avoiding process/import startup on every update.

Important invariants retained by the deterministic reconstruction include:

- world transform only in Stage 3;
- 0.5-ft voxel/world conversion;
- minimum pole separation 10 ft;
- maximum unsupported span 450 ft / 9 sequence increments;
- missing sequence numbers accepted;
- one-to-one fragment endpoint matching;
- detached bridge only under tighter geometry;
- pole attachment constrained by XY/direction/height geometry;
- bounded pole-height adjustment;
- span completion only from pole-backed evidence;
- hidden poles require multiple distinct nonparallel converging pole-anchored tracks;
- a single open conductor never creates a hidden pole;
- self-loops, synthetic polygons/triangles, and self-intersecting conductor paths are prevented;
- unattached conductors are retained.

The production runner does **not** permit an all-session Stage-3 finalize pass.

## Timing generated per arriving slice

`realtime_slice_timing.csv` includes the important realtime costs, including:

```text
csv_read_ms
sparse_item_prep_ms
stage1_wall_ms
patch_build_ms
h2d_ms
gpu_feature_assembly_ms
gpu_model_ms
d2h_gather_ms
stage2_component_ms
stage2_refiner_parametric_ms
stage12_total_ms
stage3_incremental_ms
stage3_algorithm_ms
stage3_fragment_join_ms
stage3_span_completion_pre_ms
stage3_hidden_pole_ms
stage3_span_completion_post_ms
stage3_chain_build_attachment_ms
stage3_output_write_ms
stage3_wrapper_overhead_ms
stage3_window_observed_slices
end_to_end_update_ms
```

`summarize_v4_realtime_timing.py` produces P50/P95/mean/min/max values and also groups Stage-3 latency by the number of observed slices in the rolling window.

## First production-validation sequence

Do not push this candidate to the repository `main` branch before the H100 test. The intended order is:

```text
build V4-only Docker image
  -> Nebius preflight
  -> four-way Stage-1 runtime equivalence gate
  -> Stage-1 batch-size sweep
  -> train Stage-2 on selected Stage-1 runtime
  -> 5-slice realtime replay
  -> full-session realtime replay
  -> verify strict past-only Stage-3 snapshots
  -> review P50/P95 timings + Stage-2 metrics + reconstruction outputs
  -> push/merge accepted code
```

See `NEBIUS_V4_REALTIME_RUNBOOK.md` for exact commands and `GITHUB_V4_REALTIME_INSTRUCTIONS.md` for the branch workflow.
