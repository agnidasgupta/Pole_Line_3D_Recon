# V4 Stage 1 Opt2: production-preserving inference experiments

This directory reduces execution overhead around the accepted V4 Stage 1 model.
It does not change the production model or redefine inference quality.

## Fixed production contract

- accepted checkpoint and calibration;
- complete `MultiHeadVoxelNet3D` and every trained head;
- eager PyTorch, active-GPU, BF16, batch 12 and channels-last;
- 64-cube input patches and 48-cube output cores;
- identical core order, model inputs, fusion, thresholds and serialization;
- no compilation, pruning, relabeling, ONNX export or warm-up;
- exact saved-production comparison with `SCORE_ATOL=0`.

Exact comparison is an implementation regression gate, not evaluation against
incomplete labels. An unlabelled voxel is not necessarily background. Visually
confirmed poles and lines that lack labels must not be removed by optimization.

## Accepted result and experiment history

E0 reproduced production exactly over 3,738 slices in 30 sessions:

```text
stage1_wall_ms:             282.852 -> 112.298 ms (60.30%, 2.519x)
baseline_comparable_total:  379.022 -> 209.723 ms (44.67%, 1.807x)
```

| Variant | Result | Decision |
|---|---|---|
| E0 optimized control | Exact on 3,738 slices / 30 sessions | Accepted control |
| E1 pinned D2H | Exact, slower overall | Rejected |
| E2 remove detailed CUDA events | Production score drift | Rejected |
| E3 retain gather host buffers | Exact, no repeatable benefit | Rejected |
| E4 precomputed gather plans | Exact twice; mean 0.37% faster | Rejected as operationally insignificant |
| E5 coordinate/input cache | Pending | Current isolated experiment |

E4's two-run mean was 121.745 ms for E0 and 121.300 ms for E4, saving
only 0.445 ms per slice. It must not be promoted, profiled or run over all 30
sessions.

## Why E5

The accepted E0 profile measured approximately 5.175 ms of feature assembly per
profiled slice. Every batch recalculates deterministic x/y/z coordinate channels,
concatenates them with occupancy and distance, and then materializes a
channels-last tensor.

E5 caches the exact FP32 coordinate vectors and writes the unchanged five input
channels directly into a reusable channels-last buffer. It preserves:

- every coordinate value and channel position;
- occupancy and normalized-distance values;
- batch padding and active-core order;
- the complete model forward and all five heads;
- BF16 autocast, score fusion and output decisions.

E0 remains the default. E5 is enabled only by
`CACHE_COORDINATE_CHANNELS=1`. E4 and E3 remain disabled during E5.

See [`V4_STAGE1_INFERENCE_COMPLEXITY_AUDIT.md`](V4_STAGE1_INFERENCE_COMPLEXITY_AUDIT.md)
for model-level and future training-level findings.

## Install from Mac

```bash
cd /Users/agni/Downloads
shasum -a 256 -c V4_Stage1_Opt2_E5_Coordinate_Input_Cache_v7.zip.sha256

REPO=/Users/agni/dev/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10
BRANCH=v4-stage1-inference-opt2
ZIP=/Users/agni/Downloads/V4_Stage1_Opt2_E5_Coordinate_Input_Cache_v7.zip

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

Review and push:

```bash
git -C "$REPO" add README.md ops/v4_stage1_inference_opt2
git -C "$REPO" diff --cached --check
git -C "$REPO" diff --cached --name-only
git -C "$REPO" commit -m "Add production-gated Stage1 E5 coordinate cache"
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

Stop if the worktree is dirty or the update is not a fast-forward.

## Run the representative E5 gate

Python runs only inside Docker through this launcher:

```bash
ONLY_GROUP_ID='VELASCO_CUT_CP/session1' \
EXPECTED_SESSIONS=1 \
VARIANT_NAME=e5_coordinate_input_cache \
IMAGE=va-v4-realtime:torch241-cu121 \
COMPILE_MODEL=0 COMPILE_MODE=default \
BATCH_SIZE=12 CHANNELS_LAST=1 PINNED_D2H=0 \
PRUNE_EMBEDDING_HEAD=0 WARMUP_ITERATIONS=0 \
DETAILED_CUDA_TIMING=1 RETAIN_GATHER_HOST_BUFFERS=0 \
PRECOMPUTE_BATCH_GATHER_PLANS=0 \
CACHE_COORDINATE_CHANNELS=1 \
SCORE_ATOL=0 RESUME=1 \
bash "$OPS/launch_v4_stage1_opt2_experiment.sh"
```

Monitor:

```bash
bash "$OPS/monitor_v4_stage1_opt2_experiment.sh"
```

For a one-session run, `READY_TO_PACKAGE=NO` is expected. Require:

```text
ACCEPTED=1 FAILED=0 PRODUCTION_EQUIVALENT=1
STATE=EXITED
V4_STAGE1_OPT2_ALL_SESSIONS_PRODUCTION_EQUIVALENT_AND_OK
```

Save and inspect the run:

```bash
E5_ROOT=$(cat /home/agni/LATEST_V4_STAGE1_OPT2_RUN.txt)
printf '%s\n' "$E5_ROOT" > /home/agni/V4_STAGE1_OPT2_E5_ROOT.txt

grep -E \
  'variant_name=|detailed_cuda_timing=|retain_gather_host_buffers=|precompute_batch_gather_plans=|cache_coordinate_channels=' \
  "$E5_ROOT/RUN_INFO.txt"
```

Expected:

```text
detailed_cuda_timing=1
retain_gather_host_buffers=0
precompute_batch_gather_plans=0
cache_coordinate_channels=1
```

## Compare E5 with E0

```bash
HOST_OUTPUTS=/workspace/voxel_poleline/outputs
E0_ROOT=$(cat /home/agni/V4_STAGE1_OPT2_E0_PROFILE_SOURCE_ROOT.txt)
E5_ROOT=$(cat /home/agni/V4_STAGE1_OPT2_E5_ROOT.txt)
E0_C="/outputs${E0_ROOT#$HOST_OUTPUTS}"
E5_C="/outputs${E5_ROOT#$HOST_OUTPUTS}"

docker run --rm \
  --mount "type=bind,source=$OPS,target=/workspace/opt,readonly" \
  --mount "type=bind,source=$HOST_OUTPUTS,target=/outputs" \
  va-v4-realtime:torch241-cu121 \
  python /workspace/opt/rank_v4_stage1_opt2_variants.py \
    "$E0_C" "$E5_C" \
    --group-id 'VELASCO_CUT_CP/session1' \
    --output /outputs/v4_stage1_opt2_e0_e5_ranking.csv
```

Also compare the targeted timing:

```bash
echo '===== E0 ====='
grep -E '^(gpu_feature_assembly_ms|stage1_wall_ms|baseline_comparable_total_ms)' \
  "$E0_ROOT/STAGE1_TIMING_SESSION_AVERAGES.txt"

echo '===== E5 ====='
grep -E '^(gpu_feature_assembly_ms|stage1_wall_ms|baseline_comparable_total_ms)' \
  "$E5_ROOT/STAGE1_TIMING_SESSION_AVERAGES.txt"
```

Any saved-production mismatch rejects E5 immediately. If E5 passes but improves
Stage 1 wall time by less than 1%, repeat one fresh E0/E5 pair. Reject an
inconsistent or sub-1% result as noise.

## Profile an accepted E5 representative run

Profile only after exact equivalence and a repeatable timing improvement:

```bash
E5_ROOT=$(cat /home/agni/V4_STAGE1_OPT2_E5_ROOT.txt)
PROFILE_IMAGE=va-v4-realtime:torch241-cu121-nsight
PROFILE_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
PROFILE_LOG="/home/agni/V4_STAGE1_OPT2_E5_NSYS_${PROFILE_STAMP}.log"

nohup env \
  PROFILE_VARIANT=e5 \
  IMAGE="$PROFILE_IMAGE" \
  RUN_ROOT="$E5_ROOT" \
  SESSION_FILTER='VELASCO_CUT_CP/session1' \
  PROFILE_TIMEOUT_SECONDS=1800 \
  bash "$OPS/run_v4_stage1_opt2_nsight_profile.sh" \
  > "$PROFILE_LOG" 2>&1 < /dev/null &

PROFILE_PID=$!
printf '%s\n' "$PROFILE_PID" > /home/agni/LATEST_V4_STAGE1_OPT2_E5_PROFILE_PID.txt
printf '%s\n' "$PROFILE_LOG" > /home/agni/LATEST_V4_STAGE1_OPT2_E5_PROFILE_LOG.txt
```

Success requires `NSIGHT_PROFILE_OK` and a nonempty pointer at:

```text
/home/agni/LATEST_V4_STAGE1_OPT2_E5_PROFILE_ARCHIVE.txt
```

## Full 30-session gate

Run only after repeatable representative timing and Nsight support E5:

```bash
unset ONLY_GROUP_ID

EXPECTED_SESSIONS=30 \
VARIANT_NAME=e5_coordinate_input_cache_full30 \
IMAGE=va-v4-realtime:torch241-cu121 \
COMPILE_MODEL=0 COMPILE_MODE=default \
BATCH_SIZE=12 CHANNELS_LAST=1 PINNED_D2H=0 \
PRUNE_EMBEDDING_HEAD=0 WARMUP_ITERATIONS=0 \
DETAILED_CUDA_TIMING=1 RETAIN_GATHER_HOST_BUFFERS=0 \
PRECOMPUTE_BATCH_GATHER_PLANS=0 \
CACHE_COORDINATE_CHANNELS=1 \
SCORE_ATOL=0 RESUME=1 \
bash "$OPS/launch_v4_stage1_opt2_experiment.sh"
```

Promotion requires exact saved-production equivalence for all 30 sessions and
all 3,738 slices. Any mismatch rejects E5.
