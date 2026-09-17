# V4 Stage 1 Opt2: positive-safe performance experiments

This directory is an isolated experiment pack derived from the accepted Opt1
active-GPU runner. It does not replace Opt1. The profiler showed that the network
forward dominates Stage 1 (about 151 ms of a 176 ms slice), with 2,225 kernel
launches per measured iteration and substantial GroupNorm, copy, and memory-layout
conversion work. The experiments therefore target compilation/fusion, tensor
layout, batch amortization, the unused embedding head, and final D2H staging.

## Quality rule

The accepted Stage 1 result is a positive-only reference:

- an existing Pole or Line prediction is a verified positive;
- label 0 means unknown, not a verified negative;
- losing a verified positive or changing Pole to Line (or Line to Pole) rejects a
  candidate;
- a candidate-only Pole/Line prediction is an unverified addition, not an assumed
  false positive; it blocks automatic promotion until reviewed;
- score drift up to `1e-4`, exact structure, and pseudo-metric comparisons are
  execution-fidelity diagnostics, not definitions of inference quality.

The automatic promotion state is therefore `PASS_NO_KNOWN_POSITIVE_REGRESSION`
only when all known positives are retained, no class flips occur, and there are no
unreviewed additions. If additions occur, the report says
`REVIEW_UNVERIFIED_ADDITIONS` and preserves their count for review.

## What changes and what stays fixed

Every experiment keeps the checkpoint, calibration, BF16 autocast, active-core
selection, 64-cube patch, 48-cube output core, coordinate/distance channels, score
fusion, thresholds, source rows, and artifact schema fixed.

The experiment switches are:

| Variable | Values | Purpose |
| --- | --- | --- |
| `COMPILE_MODEL` | `0`, `1` | Test `torch.compile` fusion and launch reduction. |
| `COMPILE_MODE` | `reduce-overhead`, `default`, `max-autotune` | Select compiler strategy. |
| `CHANNELS_LAST` | `1`, `0` | Determine whether native NCDHW avoids costly layout conversions. |
| `BATCH_SIZE` | `12`, `16`, `24`, `32` | Amortize launches; test one size at a time. |
| `PRUNE_EMBEDDING_HEAD` | `0`, `1` | Skip the independent embedding output, which Stage 1 never reads. |
| `PINNED_D2H` | `0`, `1` | Stage occupied-row results through reusable pinned host buffers. |

Compilation fails closed if it cannot activate or falls back to eager execution.

## Install on a new branch

On the Mac, extract the supplied ZIP into the existing repository and create a
separate experiment branch:

```bash
REPO=/Users/agni/dev/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10
ZIP=/Users/agni/Downloads/V4_Stage1_Opt2_Positive_Safe_Experiments.zip
BRANCH=v4-stage1-inference-opt2

git -C "$REPO" status --short
git -C "$REPO" switch v4-stage1-inference-opt1
git -C "$REPO" pull --ff-only origin v4-stage1-inference-opt1
git -C "$REPO" switch -c "$BRANCH"

unzip -o "$ZIP" -d "$REPO"
chmod +x "$REPO"/ops/v4_stage1_inference_opt2/*.sh

git -C "$REPO" diff --check
git -C "$REPO" status --short
git -C "$REPO" add ops/v4_stage1_inference_opt2
git -C "$REPO" commit -m "Add positive-safe Stage1 Opt2 experiments"
git -C "$REPO" push -u origin "$BRANCH"
```

Only the new `ops/v4_stage1_inference_opt2` directory should be committed.

On Nebius:

```bash
REPO=/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10
BRANCH=v4-stage1-inference-opt2

git -C "$REPO" status --short
git -C "$REPO" fetch github "$BRANCH"
git -C "$REPO" switch "$BRANCH"
git -C "$REPO" merge --ff-only "github/$BRANCH"
chmod +x "$REPO"/ops/v4_stage1_inference_opt2/*.sh
```

Do not continue if the Nebius worktree contains unrelated changes.

## Run the most useful experiments

Run variants sequentially on the H100. Concurrent runs create GPU contention and
make timing comparisons invalid. The first pass uses the representative
`VELASCO_CUT_CP/session1` session and its complete set of slices.

Set shared variables:

```bash
REPO=/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10
OPS="$REPO/ops/v4_stage1_inference_opt2"
export ONLY_GROUP_ID='VELASCO_CUT_CP/session1'
export EXPECTED_SESSIONS=1
export SCORE_ATOL=1e-4
export WARMUP_ITERATIONS=3
```

### E0 — control

This validates the Opt2 harness without enabling new execution changes:

```bash
VARIANT_NAME=e0_control_eager_b12_cl1 \
COMPILE_MODEL=0 COMPILE_MODE=reduce-overhead \
BATCH_SIZE=12 CHANNELS_LAST=1 PINNED_D2H=0 PRUNE_EMBEDDING_HEAD=0 \
bash "$OPS/launch_v4_stage1_opt2_experiment.sh"
```

Wait for it to exit before starting the next variant:

```bash
bash "$OPS/monitor_v4_stage1_opt2_experiment.sh"
```

### E1 — compile/fuse the model (highest priority)

```bash
VARIANT_NAME=e1_compile_reduce_b12_cl1 \
COMPILE_MODEL=1 COMPILE_MODE=reduce-overhead \
BATCH_SIZE=12 CHANNELS_LAST=1 PINNED_D2H=0 PRUNE_EMBEDDING_HEAD=0 \
bash "$OPS/launch_v4_stage1_opt2_experiment.sh"
```

This directly targets the dominant model time and high kernel-launch count. If it
passes and improves timing, optionally compare `COMPILE_MODE=max-autotune` as a
separate variant; its compilation warm-up is longer.

### E2 — remove layout-conversion pressure

```bash
VARIANT_NAME=e2_compile_reduce_b12_cl0 \
COMPILE_MODEL=1 COMPILE_MODE=reduce-overhead \
BATCH_SIZE=12 CHANNELS_LAST=0 PINNED_D2H=0 PRUNE_EMBEDDING_HEAD=0 \
bash "$OPS/launch_v4_stage1_opt2_experiment.sh"
```

Compare E2 with E1. Keep the faster layout; do not assume channels-last is faster
for this 3D GroupNorm-heavy network.

### E3 — skip the unused embedding head

Use the winning `CHANNELS_LAST` value from E1/E2:

```bash
WINNING_LAYOUT=1  # set to 0 if E2 won

VARIANT_NAME=e3_compile_pruned_b12 \
COMPILE_MODEL=1 COMPILE_MODE=reduce-overhead \
BATCH_SIZE=12 CHANNELS_LAST="$WINNING_LAYOUT" PINNED_D2H=0 PRUNE_EMBEDDING_HEAD=1 \
bash "$OPS/launch_v4_stage1_opt2_experiment.sh"
```

The embedding head is downstream of the shared trunk and is not consumed by score
fusion or artifact writing. The positive guard still verifies the candidate.

### E4 — batch-size sweep

Run `16` first, then `24` only if `16` completes without CUDA OOM. Test `32` only
if memory remains comfortable. Example for batch 16:

```bash
VARIANT_NAME=e4_compile_pruned_b16 \
COMPILE_MODEL=1 COMPILE_MODE=reduce-overhead \
BATCH_SIZE=16 CHANNELS_LAST="$WINNING_LAYOUT" PINNED_D2H=0 PRUNE_EMBEDDING_HEAD=1 \
bash "$OPS/launch_v4_stage1_opt2_experiment.sh"
```

Repeat with a new `VARIANT_NAME` and `BATCH_SIZE=24`. Larger batches may reduce
launch overhead but can lose on memory traffic, so retain only measured wins.

### E5 — pinned final D2H staging

Use the best compile/layout/batch settings and change only `PINNED_D2H`:

```bash
VARIANT_NAME=e5_winner_pinned_d2h \
COMPILE_MODEL=1 COMPILE_MODE=reduce-overhead \
BATCH_SIZE=16 CHANNELS_LAST="$WINNING_LAYOUT" PINNED_D2H=1 PRUNE_EMBEDDING_HEAD=1 \
bash "$OPS/launch_v4_stage1_opt2_experiment.sh"
```

This targets the long host-visible D2H call, although profiling showed that the
physical GPU copy itself was small. Treat it as a lower-priority measured test.

## Inspect and rank results

For every completed run, require the log to contain:

```text
PASS_NO_KNOWN_POSITIVE_REGRESSION
V4_STAGE1_OPT2_ALL_SESSIONS_POSITIVE_SAFE_AND_OK
```

Inspect the report directly:

```bash
RUN_ROOT=$(cat /home/agni/LATEST_V4_STAGE1_OPT2_RUN.txt)
jq '{promotion_status, positive_reference_counts, score_max_abs,
     pseudo_metric_artifacts_diagnostic_only}' \
  "$RUN_ROOT"/status/*.positive_guard.json

cat "$RUN_ROOT/STAGE1_TIMING_SESSION_AVERAGES.txt"
```

Save the printed run root for each variant, then rank them:

```bash
python "$OPS/rank_v4_stage1_opt2_variants.py" \
  /path/to/e0_run_root \
  /path/to/e1_run_root \
  /path/to/e2_run_root \
  /path/to/e3_run_root \
  /path/to/e4_run_root \
  --output /home/agni/v4_stage1_opt2_variant_ranking.csv
```

Promote only a variant whose automatic guard is `PASS`, whose timing win repeats,
and whose known-positive loss/class-flip/addition counts are all zero. An addition
is not presumed wrong; it requires targeted review before promotion.

## Confirm the winner on all 30 sessions

Rerun the single winning configuration with a new variant name and no session
filter:

```bash
unset ONLY_GROUP_ID
export EXPECTED_SESSIONS=30

VARIANT_NAME=winner_full30 \
COMPILE_MODEL=1 COMPILE_MODE=reduce-overhead \
BATCH_SIZE=16 CHANNELS_LAST="$WINNING_LAYOUT" PINNED_D2H=1 PRUNE_EMBEDDING_HEAD=1 \
bash "$OPS/launch_v4_stage1_opt2_experiment.sh"
```

Replace the example batch and D2H choices with the actual single-session winners.
Monitor with the same monitor script. Package only after `READY_TO_PACKAGE=YES`:

```bash
bash "$OPS/package_v4_stage1_opt2_results.sh"
```

The experiment writes only to a new UTC/variant output root and never modifies the
accepted baseline or Opt1 source directory.
