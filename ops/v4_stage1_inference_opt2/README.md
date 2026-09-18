# V4 Stage 1 Opt2: production-preserving inference experiments

This directory reduces execution overhead around the accepted V4 Stage 1 model.
It does not change the inference procedure, model, or definition of quality.

## Inference contract

Every candidate keeps all of the following unchanged:

- complete `MultiHeadVoxelNet3D`, including every trained head;
- accepted checkpoint and calibration;
- eager PyTorch, active-GPU execution, BF16, batch 12, channels-last;
- 64-cube input patches and 48-cube output cores;
- patch order, model inputs, score fusion, thresholds and output decisions;
- Stage 1 artifact schema and serialization;
- no compilation, pruning, relabeling, threshold tuning, ONNX export or warm-up.

`SCORE_ATOL=0` is an implementation-regression gate against saved production
outputs. It is not evaluation against incomplete labels. An unlabelled voxel is
not known to be background: visually verified lines and poles can appear as
metric false positives and must not be suppressed by an optimization.

## Accepted control and experiment status

E0 reproduced production exactly over all 3,738 slices in 30 sessions:

```text
stage1_wall_ms:             282.852 -> 112.298 ms (60.30%, 2.519x)
baseline_comparable_total:  379.022 -> 209.723 ms (44.67%, 1.807x)
production output:          exact, SCORE_ATOL=0
```

| Variant | Result | Decision |
|---|---|---|
| E0 optimized control | Exact on 3,738 slices / 30 sessions | Accepted control |
| E1 pinned D2H | Exact, but slower than E0 | Rejected |
| E2 remove detailed CUDA events | Production score drift in session 14 | Rejected |
| E3 retain gather host buffers | Exact on one session; 0.23% faster unprofiled, 0.16% slower profiled | Rejected as noise |
| E4 precomputed batch gather plans | Pending | Next isolated experiment |

Rejected variants must not be resumed, packaged or promoted.

## E0 Nsight conclusion

The attached E0 profile used the representative slice
`VELASCO_CUT_CP/session1_slice293/slice_293.csv` with 200,634 occupied rows,
three warm-up iterations and five measured iterations.

```text
mean profiled wall:          176.667 ms
GPU model:                   151.343 ms
core schedule:                15.290 ms
D2H gather:                   12.852 ms
feature assembly:              5.175 ms
gather plan:                   0.613 ms
cudaStreamSynchronize:        70 calls / 520.297 ms across five iterations
ordinary kernel launches: 10,425 calls across five iterations
```

The model is the main cost and must remain unchanged. CUDA-event recording was
only about 0.4% of CUDA API time, so E2's failed attempt to remove it was both
unsafe and low-value. E3 did not remove the stream waits. The safest remaining
host-side target is schedule/gather-plan construction.

## E4 method

E0 stable-sorts occupied rows by the accepted z/y/x linear core id, constructs
active-core groups, then later rebuilds each batch's gather lists in Python and
concatenates them. E4 derives the same gather arrays directly from the contiguous
stable-sort spans:

- destination rows remain in identical stable source-row order;
- each row receives the identical batch-slot core offset;
- active-core order, batch boundaries and padded final batch remain identical;
- model calls, CUDA kernels, fusion, thresholds and output writes are unchanged;
- E3 buffer retention remains disabled, so E4 is tested against E0 alone.

E0 remains the default. E4 is enabled only with
`PRECOMPUTE_BATCH_GATHER_PLANS=1`.

## Install and update from Mac

```bash
cd /Users/agni/Downloads
shasum -a 256 -c V4_Stage1_Opt2_E4_Precomputed_Gather_Plans_v6.zip.sha256

REPO=/Users/agni/dev/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10
BRANCH=v4-stage1-inference-opt2
ZIP=/Users/agni/Downloads/V4_Stage1_Opt2_E4_Precomputed_Gather_Plans_v6.zip

git -C "$REPO" status --short
git -C "$REPO" fetch origin "$BRANCH"
git -C "$REPO" switch "$BRANCH"
git -C "$REPO" merge --ff-only "origin/$BRANCH"
unzip -o "$ZIP" -d "$REPO"
chmod +x "$REPO"/ops/v4_stage1_inference_opt2/*.sh

git -C "$REPO" diff --check
git -C "$REPO" diff --stat
git -C "$REPO" status --short
```

Review before committing. The package changes only this README, the root README,
and Opt2 experiment/profiler files.

```bash
git -C "$REPO" add README.md ops/v4_stage1_inference_opt2
git -C "$REPO" diff --cached --check
git -C "$REPO" diff --cached --name-only
git -C "$REPO" commit -m "Add production-gated Stage1 E4 gather-plan experiment"
git -C "$REPO" push origin "$BRANCH"
```

## Fetch on Nebius

```bash
REPO=/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10
BRANCH=v4-stage1-inference-opt2

git -C "$REPO" status --short
git -C "$REPO" fetch github "$BRANCH"
git -C "$REPO" switch "$BRANCH"
git -C "$REPO" merge --ff-only "github/$BRANCH"

OPS="$REPO/ops/v4_stage1_inference_opt2"
chmod +x "$OPS"/*.sh
```

Do not continue through a dirty worktree or a non-fast-forward merge.

## Run the E4 representative-session gate

E4 must first use the same session as E0. Python is executed only inside the
provided Docker workflow.

```bash
export ONLY_GROUP_ID='VELASCO_CUT_CP/session1'
export EXPECTED_SESSIONS=1

VARIANT_NAME=e4_precomputed_batch_gather_plans \
IMAGE=va-v4-realtime:torch241-cu121 \
COMPILE_MODEL=0 COMPILE_MODE=default \
BATCH_SIZE=12 CHANNELS_LAST=1 PINNED_D2H=0 \
PRUNE_EMBEDDING_HEAD=0 WARMUP_ITERATIONS=0 \
DETAILED_CUDA_TIMING=1 RETAIN_GATHER_HOST_BUFFERS=0 \
PRECOMPUTE_BATCH_GATHER_PLANS=1 \
SCORE_ATOL=0 RESUME=1 \
bash "$OPS/launch_v4_stage1_opt2_experiment.sh"
```

Monitor:

```bash
bash "$OPS/monitor_v4_stage1_opt2_experiment.sh"
```

For a one-session gate, `READY_TO_PACKAGE=NO` is expected because packaging is
reserved for a complete 30-session result. Require these actual success signals:

```text
ACCEPTED=1 FAILED=0 PRODUCTION_EQUIVALENT=1
STATE=EXITED
V4_STAGE1_OPT2_ALL_SESSIONS_PRODUCTION_EQUIVALENT_AND_OK
```

Save the E4 root:

```bash
E4_ROOT=$(cat /home/agni/LATEST_V4_STAGE1_OPT2_RUN.txt)
printf '%s\n' "$E4_ROOT" > /home/agni/V4_STAGE1_OPT2_E4_ROOT.txt

grep -E 'variant_name=|detailed_cuda_timing=|retain_gather_host_buffers=|precompute_batch_gather_plans=' \
  "$E4_ROOT/RUN_INFO.txt"
```

Expected E4 metadata includes:

```text
detailed_cuda_timing=1
retain_gather_host_buffers=0
precompute_batch_gather_plans=1
```

## Compare E4 with E0 inside Docker

Use the timing-enabled accepted E0 representative run, not E3:

```bash
HOST_OUTPUTS=/workspace/voxel_poleline/outputs
E0_ROOT=$(cat /home/agni/V4_STAGE1_OPT2_E0_PROFILE_SOURCE_ROOT.txt)
E4_ROOT=$(cat /home/agni/V4_STAGE1_OPT2_E4_ROOT.txt)
E0_C="/outputs${E0_ROOT#$HOST_OUTPUTS}"
E4_C="/outputs${E4_ROOT#$HOST_OUTPUTS}"

docker run --rm \
  --mount "type=bind,source=$OPS,target=/workspace/opt,readonly" \
  --mount "type=bind,source=$HOST_OUTPUTS,target=/outputs" \
  va-v4-realtime:torch241-cu121 \
  python /workspace/opt/rank_v4_stage1_opt2_variants.py \
    "$E0_C" "$E4_C" \
    --group-id 'VELASCO_CUT_CP/session1' \
    --output /outputs/v4_stage1_opt2_e0_e4_ranking.csv
```

E4 is eligible for further testing only if production equivalence is `PASS`,
known-positive losses and class flips are zero, and timing improves consistently.
Repeat the representative run once if the improvement is below 1%; treat an
inconsistent or sub-1% result as noise and retain E0.

## Profile an accepted E4 representative run

Profile only after the representative equivalence gate passes:

```bash
E4_ROOT=$(cat /home/agni/V4_STAGE1_OPT2_E4_ROOT.txt)
PROFILE_IMAGE=va-v4-realtime:torch241-cu121-nsight
PROFILE_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
PROFILE_LOG="/home/agni/V4_STAGE1_OPT2_E4_NSYS_${PROFILE_STAMP}.log"

nohup env \
  PROFILE_VARIANT=e4 \
  IMAGE="$PROFILE_IMAGE" \
  RUN_ROOT="$E4_ROOT" \
  SESSION_FILTER='VELASCO_CUT_CP/session1' \
  PROFILE_TIMEOUT_SECONDS=1800 \
  bash "$OPS/run_v4_stage1_opt2_nsight_profile.sh" \
  > "$PROFILE_LOG" 2>&1 < /dev/null &

PROFILE_PID=$!
printf '%s\n' "$PROFILE_PID" > /home/agni/LATEST_V4_STAGE1_OPT2_E4_PROFILE_PID.txt
printf '%s\n' "$PROFILE_LOG" > /home/agni/LATEST_V4_STAGE1_OPT2_E4_PROFILE_LOG.txt
echo "PROFILE_PID=$PROFILE_PID"
echo "PROFILE_LOG=$PROFILE_LOG"
```

Monitor:

```bash
PID=$(cat /home/agni/LATEST_V4_STAGE1_OPT2_E4_PROFILE_PID.txt)
LOG=$(cat /home/agni/LATEST_V4_STAGE1_OPT2_E4_PROFILE_LOG.txt)

if kill -0 "$PID" 2>/dev/null; then
  echo STATE=RUNNING
  ps -o pid,ppid,stat,etime,%cpu,%mem,cmd -p "$PID"
else
  echo STATE=EXITED
fi
tail -n 120 "$LOG"
```

Success requires `NSIGHT_PROFILE_OK` and a nonempty archive pointer:

```text
/home/agni/LATEST_V4_STAGE1_OPT2_E4_PROFILE_ARCHIVE.txt
```

Profiling archives contain reports and inventories only. They exclude Stage 1
NPZs, inference CSVs, source data, checkpoints and calibration files.

## Full 30-session gate

Run this only if both repeated unprofiled timing and the E4 Nsight trace support
an improvement over E0:

```bash
unset ONLY_GROUP_ID
export EXPECTED_SESSIONS=30

VARIANT_NAME=e4_precomputed_batch_gather_plans_full30 \
IMAGE=va-v4-realtime:torch241-cu121 \
COMPILE_MODEL=0 COMPILE_MODE=default \
BATCH_SIZE=12 CHANNELS_LAST=1 PINNED_D2H=0 \
PRUNE_EMBEDDING_HEAD=0 WARMUP_ITERATIONS=0 \
DETAILED_CUDA_TIMING=1 RETAIN_GATHER_HOST_BUFFERS=0 \
PRECOMPUTE_BATCH_GATHER_PLANS=1 \
SCORE_ATOL=0 RESUME=1 \
bash "$OPS/launch_v4_stage1_opt2_experiment.sh"
```

Promotion requires all 30 sessions and all 3,738 slices to complete with exact
saved-production scores and predictions. Any mismatch rejects E4; it must not be
explained away using the incomplete annotation set.
