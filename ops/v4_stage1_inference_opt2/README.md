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

## Install the no-event experiment update

On the Mac:

```bash
cd /Users/agni/Downloads
shasum -a 256 -c V4_Stage1_Opt2_No_CUDA_Events_v4.zip.sha256

REPO=/Users/agni/dev/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10
BRANCH=v4-stage1-inference-opt2
ZIP=/Users/agni/Downloads/V4_Stage1_Opt2_No_CUDA_Events_v4.zip

git -C "$REPO" status --short
git -C "$REPO" switch "$BRANCH"
unzip -o "$ZIP" -d "$REPO"
chmod +x "$REPO"/ops/v4_stage1_inference_opt2/*.sh
git -C "$REPO" diff --check
git -C "$REPO" add ops/v4_stage1_inference_opt2
git -C "$REPO" commit -m "Test Stage1 without detailed CUDA timing events"
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

## E2: disable detailed CUDA timing events

First test the complete representative session. This changes only instrumentation;
`stage1_wall_ms` remains measured around the synchronized inference call.

```bash
export ONLY_GROUP_ID='VELASCO_CUT_CP/session1'
export EXPECTED_SESSIONS=1

VARIANT_NAME=e2_no_detailed_cuda_events \
IMAGE=va-v4-realtime:torch241-cu121 \
COMPILE_MODEL=0 COMPILE_MODE=default \
BATCH_SIZE=12 CHANNELS_LAST=1 PINNED_D2H=0 \
PRUNE_EMBEDDING_HEAD=0 WARMUP_ITERATIONS=0 \
DETAILED_CUDA_TIMING=0 SCORE_ATOL=0 \
bash "$OPS/launch_v4_stage1_opt2_experiment.sh"
```

Monitor with `bash "$OPS/monitor_v4_stage1_opt2_experiment.sh"`. Require exact
production equivalence. CUDA-event component fields are intentionally `n/a`; use
`stage1_wall_ms` and comparable total to judge the experiment.

If E2 is repeatedly faster than the accepted E0 result, confirm it across all 30
sessions with the same settings after unsetting `ONLY_GROUP_ID` and setting
`EXPECTED_SESSIONS=30`. Otherwise retain E0.

## Nsight Systems profile for E2

Use the existing Nsight-enabled derivative of the accepted image. The profiler
uses the full production model and fixes `detailed_cuda_timing=false`.

```bash
PROFILE_IMAGE=va-v4-realtime:torch241-cu121-nsight
docker run --rm "$PROFILE_IMAGE" nsys --version

RUN_ROOT=$(cat /home/agni/LATEST_V4_STAGE1_OPT2_RUN.txt)

IMAGE="$PROFILE_IMAGE" \
RUN_ROOT="$RUN_ROOT" \
SESSION_FILTER='VELASCO_CUT_CP/session1' \
PROFILE_TIMEOUT_SECONDS=1800 \
bash "$OPS/run_v4_stage1_opt2_nsight_profile.sh"
```

Success ends with `NSIGHT_PROFILE_OK` and writes a profiling-only archive pointer
to `/home/agni/LATEST_V4_STAGE1_OPT2_PROFILE_ARCHIVE.txt`.

