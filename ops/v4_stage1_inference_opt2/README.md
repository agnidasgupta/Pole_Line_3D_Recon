# V4 Stage 1 Opt2: production-equivalent inference plumbing

This directory reduces execution overhead around the accepted V4 Stage 1 model.
It does not change the inference procedure or redefine quality.

## Fixed contract

- complete accepted `MultiHeadVoxelNet3D` and all heads;
- accepted checkpoint and calibration;
- eager PyTorch, active-GPU, BF16, batch 12, channels-last;
- 64-cube patches and 48-cube output cores;
- identical inputs, fusion, thresholds, output decisions and serialization;
- no compilation, pruning, relabeling, threshold tuning or ONNX export;
- exact comparison with production output (`SCORE_ATOL=0`).

Exact comparison is an implementation regression check. It does not treat the
incomplete dataset labels as truth. A metric false positive may be a visually
confirmed real pole or line and must not be removed by an optimization.

## Accepted E0 result

The production-equivalent E0 candidate passed all 3,738 slices in 30 sessions:

```text
stage1_wall_ms:             282.852 -> 112.298 ms (60.30%, 2.519x)
baseline_comparable_total:  379.022 -> 209.723 ms (44.67%, 1.807x)
production output:          exact, score_atol=0
```

Pinned D2H was exactly equivalent but slower overall and is rejected.

## Experiment status

- **E0 accepted control:** exact production equivalence over 3,738 slices in 30 sessions.
- **E1 pinned D2H:** exact, but slower than E0; rejected.
- **E2 no detailed CUDA events:** rejected. The full run stopped in session 14
  after a production pole score changed by `0.00016323942691087723`.
- **E3 pinned gather-buffer lifetime:** next candidate. It retains E0 CUDA
  events and all inference operations, but keeps asynchronous gather-index
  source buffers alive until the existing end-of-slice synchronization.

E2 must not be resumed, packaged, profiled or promoted. Its failure is a direct
production-output comparison and is unrelated to incomplete ground-truth labels.

## Why E3 was selected

The accepted timing-enabled Nsight trace recorded five iterations of a
representative slice:

```text
mean wall time:                 175.833 ms
GPU model time:                 151.159 ms
cudaStreamSynchronize:          70 calls, 526.975 ms API time across 5 iterations
cudaEventRecord:             1,390 calls,   3.701 ms API time across 5 iterations
```

Each iteration contains ten long stream waits of roughly 11–12 ms, matching its
ten inference batches. Event recording itself is less than 1% of CUDA API time.
The pattern is consistent with pinned gather-index buffers being reclaimed while
their asynchronous H2D copies are still in flight. E3 retains those host tensors
until the existing final synchronization. It does not alter model inputs, CUDA
kernels, scores, thresholds or output serialization.

## Install the E0 profiler and E3 update

On the Mac:

```bash
cd /Users/agni/Downloads
shasum -a 256 -c V4_Stage1_Opt2_E0_Profile_E3_v5.zip.sha256

REPO=/Users/agni/dev/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10
BRANCH=v4-stage1-inference-opt2
ZIP=/Users/agni/Downloads/V4_Stage1_Opt2_E0_Profile_E3_v5.zip

git -C "$REPO" status --short
git -C "$REPO" switch "$BRANCH"
unzip -o "$ZIP" -d "$REPO"
chmod +x "$REPO"/ops/v4_stage1_inference_opt2/*.sh
git -C "$REPO" diff --check
git -C "$REPO" add ops/v4_stage1_inference_opt2
git -C "$REPO" commit -m "Profile Stage1 E0 and add production-gated E3"
git -C "$REPO" push origin "$BRANCH"
```

On Nebius:

```bash
REPO=/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10
BRANCH=v4-stage1-inference-opt2
git -C "$REPO" status --short
git -C "$REPO" fetch github "$BRANCH"
git -C "$REPO" switch "$BRANCH"
git -C "$REPO" merge --ff-only "github/$BRANCH"
chmod +x "$REPO"/ops/v4_stage1_inference_opt2/*.sh
OPS="$REPO/ops/v4_stage1_inference_opt2"
```

## Profile the accepted E0 control

The profiler refuses a run without the production-equivalent completion marker
and requires `detailed_cuda_timing=1`. Older accepted E0 run metadata is allowed
to omit `retain_gather_host_buffers`; omission means the E0 value `0`.

```bash
E0_ROOT=$(cat /home/agni/V4_STAGE1_OPT2_E0_ROOT.txt)
test -s "$E0_ROOT/STAGE1_OPT2_PRODUCTION_EQUIVALENT_COMPLETE.txt"

PROFILE_IMAGE=va-v4-realtime:torch241-cu121-nsight
docker run --rm "$PROFILE_IMAGE" nsys --version

PROFILE_VARIANT=e0 \
IMAGE="$PROFILE_IMAGE" \
RUN_ROOT="$E0_ROOT" \
SESSION_FILTER='VELASCO_CUT_CP/session1' \
PROFILE_TIMEOUT_SECONDS=1800 \
bash "$OPS/run_v4_stage1_opt2_nsight_profile.sh"
```

Success requires `NSIGHT_PROFILE_OK`, a nonempty `.nsys-rep`, and the pointer:

```text
/home/agni/LATEST_V4_STAGE1_OPT2_E0_PROFILE_ARCHIVE.txt
```

## E3 representative-session gate

E3 changes only pinned gather-index buffer lifetime. Detailed CUDA timing remains
enabled so the execution and allocation pattern stays aligned with E0.

```bash
export ONLY_GROUP_ID='VELASCO_CUT_CP/session1'
export EXPECTED_SESSIONS=1

VARIANT_NAME=e3_retain_gather_host_buffers \
IMAGE=va-v4-realtime:torch241-cu121 \
COMPILE_MODEL=0 COMPILE_MODE=default \
BATCH_SIZE=12 CHANNELS_LAST=1 PINNED_D2H=0 \
PRUNE_EMBEDDING_HEAD=0 WARMUP_ITERATIONS=0 \
DETAILED_CUDA_TIMING=1 RETAIN_GATHER_HOST_BUFFERS=1 \
SCORE_ATOL=0 RESUME=1 \
bash "$OPS/launch_v4_stage1_opt2_experiment.sh"
```

Monitor with:

```bash
bash "$OPS/monitor_v4_stage1_opt2_experiment.sh"
```

Require all of the following before comparing timing:

```text
ACCEPTED=1 FAILED=0 PRODUCTION_EQUIVALENT=1
READY_TO_PACKAGE=YES
V4_STAGE1_OPT2_ALL_SESSIONS_PRODUCTION_EQUIVALENT_AND_OK
```

Save the run root:

```bash
E3_ROOT=$(cat /home/agni/LATEST_V4_STAGE1_OPT2_RUN.txt)
printf '%s\n' "$E3_ROOT" > /home/agni/V4_STAGE1_OPT2_E3_ROOT.txt
```

Compare E0 and E3 only inside Docker. Repeat the representative run if the
improvement is below about 1%, because that difference may be run-to-run noise.

```bash
HOST_OUTPUTS=/workspace/voxel_poleline/outputs
E0_ROOT=$(cat /home/agni/V4_STAGE1_OPT2_E0_ROOT.txt)
E3_ROOT=$(cat /home/agni/V4_STAGE1_OPT2_E3_ROOT.txt)
E0_C="/outputs${E0_ROOT#$HOST_OUTPUTS}"
E3_C="/outputs${E3_ROOT#$HOST_OUTPUTS}"

docker run --rm \
  --mount "type=bind,source=$OPS,target=/workspace/opt,readonly" \
  --mount "type=bind,source=$HOST_OUTPUTS,target=/outputs" \
  va-v4-realtime:torch241-cu121 \
  python /workspace/opt/rank_v4_stage1_opt2_variants.py \
    "$E0_C" "$E3_C" \
    --group-id 'VELASCO_CUT_CP/session1' \
    --output /outputs/v4_stage1_opt2_e0_e3_ranking.csv
```

## E3 full-dataset gate

Run this only after the representative session is exact and consistently faster:

```bash
unset ONLY_GROUP_ID
export EXPECTED_SESSIONS=30

VARIANT_NAME=e3_retain_gather_host_buffers_full30 \
IMAGE=va-v4-realtime:torch241-cu121 \
COMPILE_MODEL=0 COMPILE_MODE=default \
BATCH_SIZE=12 CHANNELS_LAST=1 PINNED_D2H=0 \
PRUNE_EMBEDDING_HEAD=0 WARMUP_ITERATIONS=0 \
DETAILED_CUDA_TIMING=1 RETAIN_GATHER_HOST_BUFFERS=1 \
SCORE_ATOL=0 RESUME=1 \
bash "$OPS/launch_v4_stage1_opt2_experiment.sh"
```

Promotion requires 30 accepted and production-equivalent sessions and exact
scores with `SCORE_ATOL=0`. If any session fails, retain E0.

## Profile an accepted E3 run

Only after the full E3 gate succeeds:

```bash
E3_ROOT=$(cat /home/agni/LATEST_V4_STAGE1_OPT2_RUN.txt)
PROFILE_IMAGE=va-v4-realtime:torch241-cu121-nsight

PROFILE_VARIANT=e3 \
IMAGE="$PROFILE_IMAGE" \
RUN_ROOT="$E3_ROOT" \
SESSION_FILTER='VELASCO_CUT_CP/session1' \
PROFILE_TIMEOUT_SECONDS=1800 \
bash "$OPS/run_v4_stage1_opt2_nsight_profile.sh"
```

The E3 archive pointer is
`/home/agni/LATEST_V4_STAGE1_OPT2_E3_PROFILE_ARCHIVE.txt`. Profiling archives
contain reports and inventories only; they exclude Stage 1 NPZs, inference CSVs,
datasets and model files.
