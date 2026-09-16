# V4 Stage 1 quality-preserving optimization experiment

This directory is an isolated experiment. It does not modify the accepted `v4/`
model, training code, checkpoint, calibration, thresholds, output writers, or Stage 2.

## Fixed quality contract

- the same `precision_best.pt` and `calibration.json` paths recorded by the accepted
  full-data run; their current SHA-256 values are recorded in the experiment, and
  output equivalence is established independently for every session;
- CUDA `active_gpu` execution, BF16, batch size 12, fixed batch shape;
- 64 x 64 x 64 input patches and 48 x 48 x 48 output cores;
- identical active-core order and identical occupied-row order;
- unchanged score fusion, class thresholds, NPZ schema, metadata schema, manifests,
  inference CSV schema, and metrics;
- zero predicted-label mismatches across every occupied voxel in all 30 sessions;
- exact coordinates, source rows, labels, distance features, and semantic argmax;
- pole, line, and objectness score maximum absolute difference at most `1e-4` to
  accommodate deterministic GPU-library float variation between separate runs.

Every session is compared automatically with the accepted baseline before its
`status/<SID>.stage1.ok` marker is written. Any structural, label, metric, or excessive
score difference stops promotion.

## Complexity reductions

| Block | Accepted implementation | Opt1 implementation | Quality effect |
|---|---:|---:|---|
| Active-core membership | K scans of N occupied voxels, O(KN) | one stable sort and group pass, O(N log N) | identical z/y/x groups and row order |
| Dense GPU workspace reset | zero about 0.73 GB every slice, O(grid volume) | allocate once, then clear prior sparse coordinates, O(N) | identical zeros and occupied values |
| Boundary padding | fixed 64 voxels on all six faces | exact low/high padding: x 8/40, y 8/40, z 8/48 | identical 64-cube patch values |
| Model/checkpoint/calibration | unchanged | unchanged | none |

The reusable workspace is session-local and is reset when its shape/device contract
changes. A failed slice cannot leave stale occupied voxels because its live coordinate
set is recorded immediately after scatter and cleared before the next slice.

## Inference model architecture

The accepted checkpoint is a compact 3D U-Net-like `MultiHeadVoxelNet3D`. The default
checkpoint configuration is `in_ch=5`, `base=16`, and `emb_dim=8`. The checkpoint
configuration and the generated `MODEL_INVENTORY.json` are authoritative if a future
checkpoint differs from these defaults.

Input channels are occupancy, normalized local x/y/z coordinates, and normalized
distance-to-center. Each active 48-cube output core is inferred with an eight-voxel
context halo on every side, giving a 64 x 64 x 64 input patch. The tensor flow is:

| Stage | Operation | Output shape per patch | Trainable parameters |
|---|---|---:|---:|
| Encoder 1 | two 3-cube convolutions, GroupNorm, SiLU | 16 x 64 x 64 x 64 | 9,136 |
| Pool 1 | 2-cube max pool | 16 x 32 x 32 x 32 | 0 |
| Encoder 2 | two 3-cube convolutions, GroupNorm, SiLU | 32 x 32 x 32 x 32 | 41,600 |
| Pool 2 | 2-cube max pool | 32 x 16 x 16 x 16 | 0 |
| Encoder 3 | two 3-cube convolutions, GroupNorm, SiLU | 64 x 16 x 16 x 16 | 166,144 |
| Decoder 2 | 2-cube transpose convolution, concat Encoder 2, ConvBlock | 32 x 32 x 32 x 32 | 99,488 |
| Decoder 1 | 2-cube transpose convolution, concat Encoder 1, ConvBlock | 16 x 64 x 64 x 64 | 24,912 |
| Semantic head | 1-cube convolution, 16 to 3 | 3 x 64 x 64 x 64 | 51 |
| Pole head | 1-cube convolution, 16 to 1 | 1 x 64 x 64 x 64 | 17 |
| Line head | 1-cube convolution, 16 to 1 | 1 x 64 x 64 x 64 | 17 |
| Objectness head | 1-cube convolution, 16 to 1 | 1 x 64 x 64 x 64 | 17 |
| Embedding head | 1-cube convolution, 16 to 8 | 8 x 64 x 64 x 64 | 136 |

Total default trainable parameters: **341,518**. Convolution weights exclude bias inside
`ConvBlock`; GroupNorm contributes learned scale and offset. Transpose convolutions and
the five 1-cube heads include bias. The embedding tensor is produced by the current
forward pass but is not used by score fusion. Pole and line scores are respectively:

`0.55 * semantic probability + 0.35 * binary-head probability + 0.10 * objectness probability`

Calibration supplies the final thresholds. Opt1 does not change any model operation,
weight, BF16 autocast policy, fusion coefficient, or threshold.

The exact checkpoint inventory is generated inside Docker during profiling and contains
every leaf layer, parameter shape, and count. It is written as both
`MODEL_INVENTORY.txt` and `MODEL_INVENTORY.json`.

## Training configuration represented by the checkpoint

The V4 training entry point uses 64-cube patches, five input channels, AdamW
(`lr=1e-4`, `weight_decay=1e-4`), BF16, channels-last 3D tensors, batch size 3 with two
gradient-accumulation steps, and a ReduceLROnPlateau schedule. Its multi-task objective
combines semantic asymmetric focal loss, pole/line binary losses, Tversky terms,
objectness, false-positive penalty, and pole/line cross-class penalty. The checkpoint and
calibration remain fixed in this experiment; no retraining or recalibration occurs.

## Why Opt1 is quality-equivalent

Opt1 changes data movement and indexing only. Stable sorting replaces repeated active-core
membership scans; a session-local dense GPU workspace is reused and only the previous
sparse locations are cleared; exact asymmetric padding replaces unnecessary full-volume
padding. The dense two-channel float32 workspace falls from 731,529,216 bytes to
411,041,792 bytes (43.8% less). All 30 sessions must still pass the voxel-by-voxel
equivalence gate before the completion marker is written.

## Output layout

The experiment root is isolated at:

`outputs/poleline_voxel_run_session_groups/v4_production/stage1_opt_experiments/<UTC>`

Below that root, the original Stage 1 structure is retained:

```text
stage1/<SID>/stage1_manifest.csv
stage1/<SID>/stage1_scores/<original-relative-path>/*_stage1.npz
stage1/<SID>/stage1_scores/<original-relative-path>/*_stage1.json
stage1_inference/<SID>/inference_manifest.csv
stage1_inference/<SID>/inference_rows/<original-relative-path>/*_v4_inference.csv.gz
stage1_inference/<SID>/stage1_metrics_by_slice.csv
stage1_inference/<SID>/stage1_metrics_summary.json
```

Experiment-only diagnostics are under `logs/`, `status/`, and `timings/stage1/`.

## Mac: install and push the experiment

Use the existing experimental worktree. Do not use the main worktree that already has
the V10 branch checked out elsewhere.

```bash
REPO=/Users/agni/dev/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10
ARCHIVE=/Users/agni/Downloads/v4_stage1_inference_opt1_source_20260914.tar.gz
BRANCH=v4-stage1-inference-opt1

git -C "$REPO" status --short

if git -C "$REPO" show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git -C "$REPO" switch "$BRANCH"
else
  git -C "$REPO" switch -c "$BRANCH"
fi

tar -xzf "$ARCHIVE" -C "$REPO"
chmod +x "$REPO"/ops/v4_stage1_inference_opt1/*.sh
git -C "$REPO" diff --check
git -C "$REPO" status --short
git -C "$REPO" add ops/v4_stage1_inference_opt1
git -C "$REPO" commit -m "Optimize V4 Stage1 without changing inference quality"
git -C "$REPO" push -u origin "$BRANCH"
```

If the first `status` command is not empty, stop and preserve those unrelated changes
before switching branches.

## Nebius: fetch and launch

All Python, NumPy, pandas, PyTorch, and comparison commands are run inside Docker.

```bash
REPO=/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10
BRANCH=v4-stage1-inference-opt1

cd "$REPO"
git status --short
git remote set-url github https://github.com/agnidasgupta/Pole_Line_3D_Recon.git
git fetch github "$BRANCH"

if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git switch "$BRANCH"
  git merge --ff-only "github/$BRANCH"
else
  git switch -c "$BRANCH" --track "github/$BRANCH"
fi

bash ops/v4_stage1_inference_opt1/launch_v4_stage1_opt_experiment.sh
```

The launcher creates a fresh UTC run root, verifies the baseline runtime contract and
model/calibration hashes, compiles every Python file in Docker, runs the deterministic
CUDA self-test, and then processes all 30 sessions. The first completed session is
already a real-data end-to-end quality gate.

## Monitor and diagnose a stall

```bash
REPO=/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10
bash "$REPO/ops/v4_stage1_inference_opt1/monitor_v4_stage1_opt_experiment.sh"
```

Interpretation:

- `STATE=RUNNING` and `HEALTH=ACTIVE_OR_WAITING_NORMALLY`: continue waiting.
- `STATE=RUNNING` and `HEALTH=CHECK_POSSIBLE_STALL`: no output changed for 15 minutes.
  Check the displayed progress JSON, GPU utilization, Docker container, and recent log.
- `STATE=EXITED` and `READY_TO_PACKAGE=YES`: all 30 sessions passed equivalence and the
  timing report exists.
- `STATE=EXITED` and `READY_TO_PACKAGE=NO`: inspect the failure signals and the relevant
  `status/*.stage1.failed`, `logs/stage1/*.log`, and `logs/stage1_export/*.log` files.

Do not infer completion only from a missing PID. The authoritative completion marker is
`STAGE1_OPT_EQUIVALENT_COMPLETE.txt` together with 30 `.stage1.ok` and 30
`.quality_equivalence.json` files.

## Timing decision

Review:

```bash
RUN_ROOT=$(cat /home/agni/LATEST_V4_STAGE1_OPT_RUN.txt)
sed -n '1,220p' "$RUN_ROOT/STAGE1_TIMING_SESSION_AVERAGES.txt"
```

The text file reports each timing component as an average per slice for every session,
plus baseline mean, candidate mean, delta, saved percentage, and speedup. Keep opt1 only
if every quality gate passes and `baseline_comparable_total_ms` improves materially.
`gpu_model_ms` is expected to remain nearly constant because the model itself is fixed.

## Package and download to Mac

On Nebius:

```bash
REPO=/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10
bash "$REPO/ops/v4_stage1_inference_opt1/package_v4_stage1_opt_results.sh"
```

On Mac:

```bash
NEBIUS=nebius-va
DEST=/Users/agni/Downloads
ARCHIVE=$(ssh "$NEBIUS" 'cat /home/agni/LATEST_V4_STAGE1_OPT_ARCHIVE.txt')

scp "$NEBIUS:$ARCHIVE" "$NEBIUS:$ARCHIVE.sha256" "$DEST/"
cd "$DEST"
shasum -a 256 -c "$(basename "$ARCHIVE").sha256"
```

The results archive starts at `poleline_voxel_run_session_groups/...`, so extracting it
under a Mac output root preserves the same complete directory hierarchy:

```bash
MAC_OUTPUTS=/Users/agni/Downloads/v4_stage1_opt1_outputs
mkdir -p "$MAC_OUTPUTS"
tar -xzf "/Users/agni/Downloads/$(basename "$ARCHIVE")" -C "$MAC_OUTPUTS"
```

No Python is run on the Mac in this workflow.

## Nsight GPU profiling on Nebius

Profile only after the 30-session equivalence run is complete. The profiler selects the
median discovered slice from `VELASCO_CUT_CP/session1`, performs three warm-up iterations,
and captures five measured iterations. It writes beneath the completed run at
`profiling/nsight_<UTC>`; it never modifies Stage-1 result artifacts.

First verify that the Docker image contains the Nsight CLIs:

```bash
docker run --rm va-v4-realtime:torch241-cu121 bash -lc '
  command -v nsys
  command -v ncu || true
'
```

`nsys` is required. `ncu` is optional because kernel replay is intrusive and much slower.
If `nsys` is absent, create an Nsight-enabled derivative of the same image rather than
changing PyTorch, CUDA, the checkpoint, or calibration used by inference.

```bash
REPO=/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10
bash "$REPO/ops/v4_stage1_inference_opt1/build_v4_stage1_nsight_image.sh"
```

The derivative adds only the official `nsight-systems-cli` package. Continue to use the
original image for production/equivalence inference. Select the derivative only for the
profiling command by setting `IMAGE=va-v4-realtime:torch241-cu121-nsight`.

Run the timeline capture:

```bash
REPO=/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10
IMAGE=va-v4-realtime:torch241-cu121-nsight \
  bash "$REPO/ops/v4_stage1_inference_opt1/run_v4_stage1_nsight_profile.sh"
```

For an additional bounded Nsight Compute capture of the first 50 launched kernels:

```bash
REPO=/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10
RUN_NCU=1 IMAGE=va-v4-realtime:torch241-cu121-nsight \
  bash "$REPO/ops/v4_stage1_inference_opt1/run_v4_stage1_nsight_profile.sh"
```

The archive contains the `.nsys-rep`, optional `.ncu-rep`, `NSYS_STATS.txt`, exact model
inventory, profiler log, and slice/runtime summary. Download it on Mac with:

```bash
NEBIUS=nebius-va
DEST=/Users/agni/Downloads
PROFILE_ARCHIVE=$(ssh "$NEBIUS" 'cat /home/agni/LATEST_V4_STAGE1_OPT_PROFILE_ARCHIVE.txt')
scp "$NEBIUS:$PROFILE_ARCHIVE" "$NEBIUS:$PROFILE_ARCHIVE.sha256" "$DEST/"
cd "$DEST"
shasum -a 256 -c "$(basename "$PROFILE_ARCHIVE").sha256"
```

Open the `.nsys-rep` in Nsight Systems to inspect GPU occupancy over time, kernel gaps,
CUDA API synchronization, transfers, and NVTX iteration boundaries. Use `NSYS_STATS.txt`
for sortable kernel/API totals. Open `.ncu-rep` in Nsight Compute only when per-kernel
memory, launch, occupancy, or instruction metrics are needed.
