# V4 Stage 1 Opt2 E6/E7 gated runbook

E6 and E7 preserve the accepted V4 production inference contract:

- the same checkpoint and calibration;
- active GPU, BF16, batch 12 and channels-last;
- all five model heads;
- 64-cube patches and 48-cube output cores;
- identical score fusion, thresholds, decisions, artifacts and directory layout;
- exact saved-output comparison with `SCORE_ATOL=0`.

The comparison is against accepted V4 production output. It is not an
evaluation against incomplete labels; an unlabelled voxel may still be a real
pole or line.

## v10.1 self-test correction

The v10 self-test incorrectly required `cuda_graph_captured=1` on both of two
calls sharing one E7 workspace. The first call correctly captures the graph;
the second correctly reuses it and reports `cuda_graph_captured=0`. V10.1
requires exactly one capture across the two calls and successful replay on each
call. This correction changes only the test invariant. It does not change the
E5b, E6 or E7 inference procedure.

## Why E6 includes the gather-buffer lifetime safeguard

The earlier E2 timing-disabled experiment drifted. Detailed CUDA events had
incidentally extended the lifetime of pinned host tensors used for asynchronous
gather-index H2D copies. E6 removes the events only while enabling the already
exact E3 lifetime safeguard. The tensors are retained until the existing final
CUDA synchronization. No values, copies, kernels, model operations or output
decisions are changed.

E6 settings:

```text
DETAILED_CUDA_TIMING=0
RETAIN_GATHER_HOST_BUFFERS=1
CACHE_REFERENCE_COORDINATE_CHANNELS=1
USE_CUDA_GRAPH=0
```

E7 adds one isolated change to accepted E6: capture and replay of the unchanged
fixed-shape batch-12 model forward and score-fusion operators.

```text
USE_CUDA_GRAPH=1
```

The graph retains the complete five-head output. Variable-length occupied-row
gathering and D2H output transfer remain outside the graph.

## Required execution order

1. Run a fresh matched E5b control on the session that exposed E5 drift.
2. Run E6 on the same session.
3. Require exact outputs, then repeat the E5b/E6 pair.
4. Run E7 only if E6 passes both exactness checks.
5. Repeat a fresh E6/E7 pair.
6. Run the faster exact candidate across all 30 sessions / 3,738 slices.
7. Begin the asynchronous I/O experiment only after the full gate passes.

E6 passed twice, E7 passed twice with verified CUDA Graph capture/replay, and
E7 then passed the full 30-session / 3,738-slice gate. The documented next
phase is E8; see [`E6_E7_ACCEPTANCE_REPORT.md`](E6_E7_ACCEPTANCE_REPORT.md).

## Common Nebius variables

```bash
REPO=/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10
OPS="$REPO/ops/v4_stage1_inference_opt2"
IMAGE=va-v4-realtime:torch241-cu121
GID='NYSEGDistVegMgnt_AUBURN_-_SPRUCE_HAVEN_FARMS_TAP_520-3p_2029/session2'
```

Python is invoked only inside Docker by the supplied launch and ranking scripts.

## Fresh matched E5b control

Run immediately before E6:

```bash
ONLY_GROUP_ID="$GID" \
EXPECTED_SESSIONS=1 \
VARIANT_NAME=e5b_e6_matched_control \
IMAGE="$IMAGE" \
COMPILE_MODEL=0 COMPILE_MODE=default \
BATCH_SIZE=12 CHANNELS_LAST=1 PINNED_D2H=0 \
PRUNE_EMBEDDING_HEAD=0 WARMUP_ITERATIONS=0 \
DETAILED_CUDA_TIMING=1 RETAIN_GATHER_HOST_BUFFERS=0 \
PRECOMPUTE_BATCH_GATHER_PLANS=0 \
CACHE_COORDINATE_CHANNELS=0 \
CACHE_REFERENCE_COORDINATE_CHANNELS=1 \
USE_CUDA_GRAPH=0 \
SCORE_ATOL=0 RESUME=1 \
bash "$OPS/launch_v4_stage1_opt2_experiment.sh"

E5B_CONTROL_ROOT=$(cat /home/agni/LATEST_V4_STAGE1_OPT2_RUN.txt)
printf '%s\n' "$E5B_CONTROL_ROOT" \
  > /home/agni/V4_STAGE1_OPT2_E5B_E6_CONTROL_ROOT.txt
```

Wait for and verify completion before launching E6. The E6 comparison uses this
fresh root rather than an older run from a different GPU-temperature/cache state.

## Launch E6

```bash
ONLY_GROUP_ID="$GID" \
EXPECTED_SESSIONS=1 \
VARIANT_NAME=e6_e5b_timing_off_safe_gather \
IMAGE="$IMAGE" \
COMPILE_MODEL=0 COMPILE_MODE=default \
BATCH_SIZE=12 CHANNELS_LAST=1 PINNED_D2H=0 \
PRUNE_EMBEDDING_HEAD=0 WARMUP_ITERATIONS=0 \
DETAILED_CUDA_TIMING=0 RETAIN_GATHER_HOST_BUFFERS=1 \
PRECOMPUTE_BATCH_GATHER_PLANS=0 \
CACHE_COORDINATE_CHANNELS=0 \
CACHE_REFERENCE_COORDINATE_CHANNELS=1 \
USE_CUDA_GRAPH=0 \
SCORE_ATOL=0 RESUME=1 \
bash "$OPS/launch_v4_stage1_opt2_experiment.sh"
```

Monitor:

```bash
bash "$OPS/monitor_v4_stage1_opt2_experiment.sh"
```

Require:

```text
ACCEPTED=1 FAILED=0 PRODUCTION_EQUIVALENT=1
STATE=EXITED
V4_STAGE1_OPT2_ALL_SESSIONS_PRODUCTION_EQUIVALENT_AND_OK
```

Save the completed root:

```bash
E6_ROOT=$(cat /home/agni/LATEST_V4_STAGE1_OPT2_RUN.txt)
printf '%s\n' "$E6_ROOT" > /home/agni/V4_STAGE1_OPT2_E6_ROOT.txt

grep -E \
  'variant_name=|detailed_cuda_timing=|retain_gather_host_buffers=|cache_reference_coordinate_channels=|use_cuda_graph=' \
  "$E6_ROOT/RUN_INFO.txt"
```

Expected values are `0`, `1`, `1`, and `0` respectively.

Compare E6 with the immediately preceding E5b control:

```bash
HOST_OUTPUTS=/workspace/voxel_poleline/outputs
E5B_CONTROL_ROOT=$(cat /home/agni/V4_STAGE1_OPT2_E5B_E6_CONTROL_ROOT.txt)
E6_ROOT=$(cat /home/agni/V4_STAGE1_OPT2_E6_ROOT.txt)
E5B_CONTROL_C="/outputs${E5B_CONTROL_ROOT#$HOST_OUTPUTS}"
E6_C="/outputs${E6_ROOT#$HOST_OUTPUTS}"

docker run --rm \
  --mount "type=bind,source=$OPS,target=/workspace/opt,readonly" \
  --mount "type=bind,source=$HOST_OUTPUTS,target=/outputs" \
  "$IMAGE" python /workspace/opt/rank_v4_stage1_opt2_variants.py \
    "$E5B_CONTROL_C" "$E6_C" --group-id "$GID" \
    --output /outputs/v4_stage1_opt2_e5b_e6_ranking.csv
```

Repeat the matched control/E6 pair once with new variant names. Do not launch
E7 if either E6 run changes any saved production score, class, structural array,
manifest row or exported inference value.

## Launch E7 after E6 passes

```bash
ONLY_GROUP_ID="$GID" \
EXPECTED_SESSIONS=1 \
VARIANT_NAME=e7_e5b_cuda_graph_replay \
IMAGE="$IMAGE" \
COMPILE_MODEL=0 COMPILE_MODE=default \
BATCH_SIZE=12 CHANNELS_LAST=1 PINNED_D2H=0 \
PRUNE_EMBEDDING_HEAD=0 WARMUP_ITERATIONS=0 \
DETAILED_CUDA_TIMING=0 RETAIN_GATHER_HOST_BUFFERS=1 \
PRECOMPUTE_BATCH_GATHER_PLANS=0 \
CACHE_COORDINATE_CHANNELS=0 \
CACHE_REFERENCE_COORDINATE_CHANNELS=1 \
USE_CUDA_GRAPH=1 \
SCORE_ATOL=0 RESUME=1 \
bash "$OPS/launch_v4_stage1_opt2_experiment.sh"
```

After the monitor reports success:

```bash
E7_ROOT=$(cat /home/agni/LATEST_V4_STAGE1_OPT2_RUN.txt)
printf '%s\n' "$E7_ROOT" > /home/agni/V4_STAGE1_OPT2_E7_ROOT.txt
```

## Compare E6 and E7

```bash
HOST_OUTPUTS=/workspace/voxel_poleline/outputs
E6_ROOT=$(cat /home/agni/V4_STAGE1_OPT2_E6_ROOT.txt)
E7_ROOT=$(cat /home/agni/V4_STAGE1_OPT2_E7_ROOT.txt)
E6_C="/outputs${E6_ROOT#$HOST_OUTPUTS}"
E7_C="/outputs${E7_ROOT#$HOST_OUTPUTS}"
```

Report end-to-end means including graph capture on the first inferred slice:

```bash
docker run --rm \
  --mount "type=bind,source=$OPS,target=/workspace/opt,readonly" \
  --mount "type=bind,source=$HOST_OUTPUTS,target=/outputs" \
  "$IMAGE" python /workspace/opt/rank_v4_stage1_opt2_variants.py \
    "$E6_C" "$E7_C" --group-id "$GID" \
    --output /outputs/v4_stage1_opt2_e6_e7_ranking.csv
```

Also report steady-state means with the one-time capture slice excluded:

```bash
docker run --rm \
  --mount "type=bind,source=$OPS,target=/workspace/opt,readonly" \
  --mount "type=bind,source=$HOST_OUTPUTS,target=/outputs" \
  "$IMAGE" python /workspace/opt/rank_v4_stage1_opt2_variants.py \
    "$E6_C" "$E7_C" --group-id "$GID" --exclude-first-slices 1 \
    --output /outputs/v4_stage1_opt2_e6_e7_steady_state_ranking.csv
```

Do not infer model-component timings for E6/E7: detailed events are deliberately
disabled, so `gpu_model_ms` and related columns are expected to be unavailable.
Use `stage1_wall_ms` and `baseline_comparable_total_ms`.

## Profile an exact E7 candidate

Profile only after the repeated paired gate remains exact:

```bash
E7_ROOT=$(cat /home/agni/V4_STAGE1_OPT2_E7_ROOT.txt)
PROFILE_IMAGE=va-v4-realtime:torch241-cu121-nsight
PROFILE_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
PROFILE_LOG="/home/agni/V4_STAGE1_OPT2_E7_NSYS_${PROFILE_STAMP}.log"

nohup env \
  PROFILE_VARIANT=e7 \
  IMAGE="$PROFILE_IMAGE" \
  RUN_ROOT="$E7_ROOT" \
  SESSION_FILTER="$GID" \
  PROFILE_TIMEOUT_SECONDS=1800 \
  bash "$OPS/run_v4_stage1_opt2_nsight_profile.sh" \
  > "$PROFILE_LOG" 2>&1 < /dev/null &

PROFILE_PID=$!
printf '%s\n' "$PROFILE_PID" > /home/agni/LATEST_V4_STAGE1_OPT2_E7_PROFILE_PID.txt
printf '%s\n' "$PROFILE_LOG" > /home/agni/LATEST_V4_STAGE1_OPT2_E7_PROFILE_LOG.txt
```

Monitor:

```bash
PID=$(cat /home/agni/LATEST_V4_STAGE1_OPT2_E7_PROFILE_PID.txt)
LOG=$(cat /home/agni/LATEST_V4_STAGE1_OPT2_E7_PROFILE_LOG.txt)

if kill -0 "$PID" 2>/dev/null; then
  echo STATE=RUNNING
  ps -o pid,ppid,stat,etime,%cpu,%mem,cmd -p "$PID"
else
  echo STATE=EXITED
fi

tail -n 120 "$LOG"
```

Require `NSIGHT_PROFILE_OK` and a nonempty
`/home/agni/LATEST_V4_STAGE1_OPT2_E7_PROFILE_ARCHIVE.txt`. Confirm that the
number of per-pass launch calls decreases; the raw report does not replace the
saved-output equivalence gate.

## Full gate for the winner

Set `USE_CUDA_GRAPH=1` for E7 or `0` for E6, and use the matching variant name.

```bash
unset ONLY_GROUP_ID

EXPECTED_SESSIONS=30 \
VARIANT_NAME=e7_e5b_cuda_graph_replay_full30 \
IMAGE="$IMAGE" \
COMPILE_MODEL=0 COMPILE_MODE=default \
BATCH_SIZE=12 CHANNELS_LAST=1 PINNED_D2H=0 \
PRUNE_EMBEDDING_HEAD=0 WARMUP_ITERATIONS=0 \
DETAILED_CUDA_TIMING=0 RETAIN_GATHER_HOST_BUFFERS=1 \
PRECOMPUTE_BATCH_GATHER_PLANS=0 \
CACHE_COORDINATE_CHANNELS=0 \
CACHE_REFERENCE_COORDINATE_CHANNELS=1 \
USE_CUDA_GRAPH=1 \
SCORE_ATOL=0 RESUME=1 \
bash "$OPS/launch_v4_stage1_opt2_experiment.sh"
```

Promotion requires 30 accepted sessions, zero failures, 30 exact production
equivalence reports and exactly 3,738 timing rows.

## Subsequent experiments

Do not combine the next two changes.

- E8 overlaps CSV read/sparse preparation for slice N+1, GPU inference for N,
  and durable artifact writing for N-1. It must retain the exact existing
  `stage1/<SID>/stage1_scores`, `stage1_inference/<SID>/inference_rows`, timing,
  status and manifest paths and schemas.
- E9 adds in-memory Stage 1 to Stage 2 delivery only after E8 passes. The same
  Stage 1 NPZ/JSON and Stage 2 CSV/manifest artifacts must still be written to
  the same paths. The in-memory route removes disk reopening from the critical
  path; it does not remove or rename durable artifacts.

Each phase repeats the one-session paired gate, repeat gate and 30-session exact
saved-output gate. E8/E9 implementation must not begin from a rejected E6/E7
root.
