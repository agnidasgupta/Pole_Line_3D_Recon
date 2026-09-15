# V4 Stage 1 quality-preserving optimization experiment

This directory is an isolated experiment. It does not modify the accepted `v4/`
model, training code, checkpoint, calibration, thresholds, output writers, or Stage 2.

## Fixed quality contract

- accepted `precision_best.pt` and `calibration.json`, verified by SHA-256 against the
  accepted full-data run;
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
