#!/usr/bin/env bash
set -euo pipefail

TOOL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP_REPO=${EXP_REPO:-$(cd "$TOOL_DIR/../.." && pwd)}
HOST_OUTPUTS=${HOST_OUTPUTS:-/workspace/voxel_poleline/outputs}
HOST_INPUT=${HOST_INPUT:-/data/voxel_csv_combined}
IMAGE=${IMAGE:-va-v4-realtime:torch241-cu121-nsight}
MODEL=${MODEL_HOST:-$HOST_OUTPUTS/poleline_voxel_run_session_groups/precision_v4/train/precision_best.pt}
CALIBRATION=${CALIBRATION_HOST:-$HOST_OUTPUTS/poleline_voxel_run_session_groups/precision_v4/full_val/calibration.json}
SESSION_FILTER=${SESSION_FILTER:-VELASCO_CUT_CP/session1}
NSYS_SET=${NSYS_SET:-cuda,nvtx,osrt}
PROFILE_TIMEOUT_SECONDS=${PROFILE_TIMEOUT_SECONDS:-1800}
PROFILE_VARIANT=${PROFILE_VARIANT:-e0}

fail() { echo "ERROR: $*" >&2; exit 1; }
[[ "$PROFILE_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] || fail "PROFILE_TIMEOUT_SECONDS must be an integer"
case "$PROFILE_VARIANT" in
  e0) RETAIN_GATHER_HOST_BUFFERS=0; PRECOMPUTE_BATCH_GATHER_PLANS=0; CACHE_COORDINATE_CHANNELS=0 ;;
  e3) RETAIN_GATHER_HOST_BUFFERS=1; PRECOMPUTE_BATCH_GATHER_PLANS=0; CACHE_COORDINATE_CHANNELS=0 ;;
  e4) RETAIN_GATHER_HOST_BUFFERS=0; PRECOMPUTE_BATCH_GATHER_PLANS=1; CACHE_COORDINATE_CHANNELS=0 ;;
  e5) RETAIN_GATHER_HOST_BUFFERS=0; PRECOMPUTE_BATCH_GATHER_PLANS=0; CACHE_COORDINATE_CHANNELS=1 ;;
  *) fail "PROFILE_VARIANT must be e0, e3, e4, or e5" ;;
esac
for path in "$TOOL_DIR/profile_v4_stage1_opt2.py" "$TOOL_DIR/inventory_v4_stage1_model.py" "$HOST_INPUT" "$MODEL" "$CALIBRATION"; do
  [ -e "$path" ] || fail "missing required path: $path"
done

RUN_ROOT=${RUN_ROOT:-$(cat /home/agni/V4_STAGE1_OPT2_ACCEPTED_RUN.txt)}
[ -s "$RUN_ROOT/STAGE1_OPT2_PRODUCTION_EQUIVALENT_COMPLETE.txt" ] || \
  fail "accepted production-equivalent marker missing: $RUN_ROOT"
[ -s "$RUN_ROOT/RUN_INFO.txt" ] || fail "RUN_INFO.txt missing: $RUN_ROOT"
grep -qx 'detailed_cuda_timing=1' "$RUN_ROOT/RUN_INFO.txt" || \
  fail "profile source run did not use accepted detailed_cuda_timing=1"
if [ "$PROFILE_VARIANT" = e3 ]; then
  grep -qx 'retain_gather_host_buffers=1' "$RUN_ROOT/RUN_INFO.txt" || \
    fail "E3 profile source run did not retain gather host buffers"
else
  if grep -q '^retain_gather_host_buffers=' "$RUN_ROOT/RUN_INFO.txt"; then
    grep -qx 'retain_gather_host_buffers=0' "$RUN_ROOT/RUN_INFO.txt" || \
      fail "E0 profile source run unexpectedly enabled E3"
  fi
fi
if [ "$PROFILE_VARIANT" = e5 ]; then
  grep -qx 'cache_coordinate_channels=1' "$RUN_ROOT/RUN_INFO.txt" || \
    fail "E5 profile source run did not enable coordinate/input caching"
else
  if grep -q '^cache_coordinate_channels=' "$RUN_ROOT/RUN_INFO.txt"; then
    grep -qx 'cache_coordinate_channels=0' "$RUN_ROOT/RUN_INFO.txt" || \
      fail "$PROFILE_VARIANT profile source unexpectedly enabled E5"
  fi
fi
if [ "$PROFILE_VARIANT" = e4 ]; then
  grep -qx 'precompute_batch_gather_plans=1' "$RUN_ROOT/RUN_INFO.txt" || \
    fail "E4 profile source run did not precompute batch gather plans"
else
  if grep -q '^precompute_batch_gather_plans=' "$RUN_ROOT/RUN_INFO.txt"; then
    grep -qx 'precompute_batch_gather_plans=0' "$RUN_ROOT/RUN_INFO.txt" || \
      fail "$PROFILE_VARIANT profile source unexpectedly enabled E4"
  fi
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
PROFILE_ROOT="$RUN_ROOT/profiling/nsight_${PROFILE_VARIANT}_cuda_timing_$STAMP"
PROFILE_C="/outputs${PROFILE_ROOT#$HOST_OUTPUTS}"
mkdir -p "$PROFILE_ROOT"
printf '%s\n' "$PROFILE_ROOT" > /home/agni/LATEST_V4_STAGE1_OPT2_PROFILE.txt

tool_probe=$(docker run --rm "$IMAGE" bash -lc '
  printf "nsys="; command -v nsys || true
')
printf '%s\n' "$tool_probe" | tee "$PROFILE_ROOT/PROFILER_TOOLS.txt"
nsys_bin=$(printf '%s\n' "$tool_probe" | sed -n 's/^nsys=//p')
[ -n "$nsys_bin" ] || fail "nsys is absent from $IMAGE"

docker run --rm --gpus all \
  --mount "type=bind,source=$EXP_REPO/v4,target=/workspace/v4,readonly" --mount "type=bind,source=$EXP_REPO/v4/..,target=/workspace/poleline_repo,readonly" \
  --mount "type=bind,source=$TOOL_DIR,target=/workspace/opt,readonly" \
  --mount "type=bind,source=$HOST_OUTPUTS,target=/outputs" \
  --workdir /workspace/v4 -e PYTHONPATH=/workspace/v4:/workspace/opt \
  "$IMAGE" python /workspace/opt/inventory_v4_stage1_model.py \
    --model_path "/outputs${MODEL#$HOST_OUTPUTS}" \
    --output_json "$PROFILE_C/MODEL_INVENTORY.json" \
    --output_txt "$PROFILE_C/MODEL_INVENTORY.txt"

common=(
  --rm --gpus all --ipc=host --cap-add SYS_ADMIN --security-opt seccomp=unconfined
  --mount "type=bind,source=$EXP_REPO/v4,target=/workspace/v4,readonly" --mount "type=bind,source=$EXP_REPO/v4/..,target=/workspace/poleline_repo,readonly"
  --mount "type=bind,source=$TOOL_DIR,target=/workspace/opt,readonly"
  --mount "type=bind,source=$HOST_OUTPUTS,target=/outputs"
  --mount "type=bind,source=$HOST_INPUT,target=/data/voxel_csv_combined,readonly"
  --workdir /workspace/v4 -e PYTHONPATH=/workspace/v4:/workspace/opt
  -e PYTHONPYCACHEPREFIX=/tmp/pycache -e MPLCONFIGDIR=/tmp/matplotlib
)
profile_args=(
  python /workspace/opt/profile_v4_stage1_opt2.py
  --input_dir /data/voxel_csv_combined --session_filter "$SESSION_FILTER"
  --model_path "/outputs${MODEL#$HOST_OUTPUTS}"
  --calibration_json "/outputs${CALIBRATION#$HOST_OUTPUTS}"
  --output_json "$PROFILE_C/PROFILE_SUMMARY.json" --warmup 3 --iterations 5
  --retain_gather_host_buffers "$RETAIN_GATHER_HOST_BUFFERS"
  --precompute_batch_gather_plans "$PRECOMPUTE_BATCH_GATHER_PLANS"
  --cache_coordinate_channels "$CACHE_COORDINATE_CHANNELS"
)

timeout --signal=TERM --kill-after=60 "$PROFILE_TIMEOUT_SECONDS" \
docker run "${common[@]}" "$IMAGE" "$nsys_bin" profile \
  --trace="$NSYS_SET" --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --force-overwrite=true \
  --output="$PROFILE_C/stage1_opt2_nsys" "${profile_args[@]}" \
  2>&1 | tee "$PROFILE_ROOT/NSYS_PROFILE.log"

[ -s "$PROFILE_ROOT/stage1_opt2_nsys.nsys-rep" ] || \
  fail "Nsight completed without creating stage1_opt2_nsys.nsys-rep"

if ! docker run --rm \
  --mount "type=bind,source=$HOST_OUTPUTS,target=/outputs" \
  "$IMAGE" "$nsys_bin" stats --force-export=true \
  --report nvtx_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
  --format column "$PROFILE_C/stage1_opt2_nsys.nsys-rep" \
  > "$PROFILE_ROOT/NSYS_STATS.txt"; then
  echo "WARNING: requested stats set unsupported; using default reports" \
    > "$PROFILE_ROOT/NSYS_STATS.txt"
  docker run --rm \
    --mount "type=bind,source=$HOST_OUTPUTS,target=/outputs" \
    "$IMAGE" "$nsys_bin" stats --force-export=true --format column \
    "$PROFILE_C/stage1_opt2_nsys.nsys-rep" \
    >> "$PROFILE_ROOT/NSYS_STATS.txt" 2>&1 || true
fi

sha256sum "$PROFILE_ROOT"/* > "$PROFILE_ROOT/SHA256SUMS.txt"
ARCHIVE="/home/agni/v4_stage1_opt2_${PROFILE_VARIANT}_cuda_timing_nsight_${STAMP}.tar.gz"
tar -C "$HOST_OUTPUTS" -czf "$ARCHIVE" "${PROFILE_ROOT#$HOST_OUTPUTS/}"
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
printf '%s\n' "$ARCHIVE" > /home/agni/LATEST_V4_STAGE1_OPT2_PROFILE_ARCHIVE.txt
printf '%s\n' "$ARCHIVE" > "/home/agni/LATEST_V4_STAGE1_OPT2_${PROFILE_VARIANT^^}_PROFILE_ARCHIVE.txt"
echo "NSIGHT_PROFILE_OK profile_root=$PROFILE_ROOT"
echo "archive=$ARCHIVE"
