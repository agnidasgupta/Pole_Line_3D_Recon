#!/usr/bin/env bash
set -euo pipefail

TOOL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP_REPO=${EXP_REPO:-$(cd "$TOOL_DIR/../.." && pwd)}
HOST_OUTPUTS=${HOST_OUTPUTS:-/workspace/voxel_poleline/outputs}
HOST_INPUT=${HOST_INPUT:-/data/voxel_csv_combined}
IMAGE=${IMAGE:-va-v4-realtime:torch241-cu121}
MODEL=${MODEL_HOST:-$HOST_OUTPUTS/poleline_voxel_run_session_groups/precision_v4/train/precision_best.pt}
CALIBRATION=${CALIBRATION_HOST:-$HOST_OUTPUTS/poleline_voxel_run_session_groups/precision_v4/full_val/calibration.json}
SESSION_FILTER=${SESSION_FILTER:-VELASCO_CUT_CP/session1}
RUN_NCU=${RUN_NCU:-0}
NSYS_SET=${NSYS_SET:-cuda,nvtx,osrt}

fail() { echo "ERROR: $*" >&2; exit 1; }
[ "$RUN_NCU" = 0 ] || [ "$RUN_NCU" = 1 ] || fail "RUN_NCU must be 0 or 1"
for path in "$TOOL_DIR/profile_v4_stage1_opt.py" "$TOOL_DIR/inventory_v4_stage1_model.py" "$HOST_INPUT" "$MODEL" "$CALIBRATION"; do
  [ -e "$path" ] || fail "missing required path: $path"
done

RUN_ROOT=${RUN_ROOT:-$(cat /home/agni/LATEST_V4_STAGE1_OPT_RUN.txt)}
[ -s "$RUN_ROOT/STAGE1_OPT_EQUIVALENT_COMPLETE.txt" ] || fail "completed opt1 marker missing: $RUN_ROOT"
[ ! -s /home/agni/LATEST_V4_STAGE1_OPT_PID.txt ] || {
  prior_pid=$(cat /home/agni/LATEST_V4_STAGE1_OPT_PID.txt)
  ! kill -0 "$prior_pid" 2>/dev/null || fail "Stage1 experiment still running as PID $prior_pid"
}

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
PROFILE_ROOT="$RUN_ROOT/profiling/nsight_$STAMP"
PROFILE_C="/outputs${PROFILE_ROOT#$HOST_OUTPUTS}"
mkdir -p "$PROFILE_ROOT"
printf '%s\n' "$PROFILE_ROOT" > /home/agni/LATEST_V4_STAGE1_OPT_PROFILE.txt

docker run --rm --gpus all \
  --mount "type=bind,source=$EXP_REPO/v4,target=/workspace/v4,readonly" \
  --mount "type=bind,source=$TOOL_DIR,target=/workspace/opt,readonly" \
  --mount "type=bind,source=$HOST_OUTPUTS,target=/outputs" \
  --workdir /workspace/v4 -e PYTHONPATH=/workspace/v4:/workspace/opt \
  "$IMAGE" python /workspace/opt/inventory_v4_stage1_model.py \
    --model_path "/outputs${MODEL#$HOST_OUTPUTS}" \
    --output_json "$PROFILE_C/MODEL_INVENTORY.json" \
    --output_txt "$PROFILE_C/MODEL_INVENTORY.txt"

tool_probe=$(docker run --rm "$IMAGE" bash -lc '
  printf "nsys="; command -v nsys || true
  printf "ncu="; command -v ncu || true
')
printf '%s\n' "$tool_probe" | tee "$PROFILE_ROOT/PROFILER_TOOLS.txt"
nsys_bin=$(printf '%s\n' "$tool_probe" | sed -n 's/^nsys=//p')
[ -n "$nsys_bin" ] || fail "nsys is absent from $IMAGE; use an Nsight-enabled derivative of the same image"
ncu_bin=$(printf '%s\n' "$tool_probe" | sed -n 's/^ncu=//p')
[ "$RUN_NCU" = 0 ] || [ -n "$ncu_bin" ] || fail "RUN_NCU=1 but ncu is absent from $IMAGE"

common=(
  --rm --gpus all --ipc=host --cap-add SYS_ADMIN --security-opt seccomp=unconfined
  --mount "type=bind,source=$EXP_REPO/v4,target=/workspace/v4,readonly"
  --mount "type=bind,source=$TOOL_DIR,target=/workspace/opt,readonly"
  --mount "type=bind,source=$HOST_OUTPUTS,target=/outputs"
  --mount "type=bind,source=$HOST_INPUT,target=/data/voxel_csv_combined,readonly"
  --workdir /workspace/v4 -e PYTHONPATH=/workspace/v4:/workspace/opt
  -e PYTHONPYCACHEPREFIX=/tmp/pycache -e MPLCONFIGDIR=/tmp/matplotlib
)
profile_args=(
  python /workspace/opt/profile_v4_stage1_opt.py
  --input_dir /data/voxel_csv_combined --session_filter "$SESSION_FILTER"
  --model_path "/outputs${MODEL#$HOST_OUTPUTS}"
  --calibration_json "/outputs${CALIBRATION#$HOST_OUTPUTS}"
  --output_json "$PROFILE_C/PROFILE_SUMMARY.json" --warmup 3 --iterations 5
)

docker run "${common[@]}" "$IMAGE" "$nsys_bin" profile \
  --trace="$NSYS_SET" --sample=none --cpuctxsw=none \
  --capture-range=nvtx --nvtx-capture=stage1_opt_profile \
  --capture-range-end=stop --force-overwrite=true \
  --output="$PROFILE_C/stage1_opt_nsys" "${profile_args[@]}" \
  2>&1 | tee "$PROFILE_ROOT/NSYS_PROFILE.log"

docker run --rm \
  --mount "type=bind,source=$HOST_OUTPUTS,target=/outputs" \
  "$IMAGE" "$nsys_bin" stats --force-export=true \
  --report nvtx_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
  --format column "$PROFILE_C/stage1_opt_nsys.nsys-rep" \
  > "$PROFILE_ROOT/NSYS_STATS.txt"

if [ "$RUN_NCU" = 1 ]; then
  docker run "${common[@]}" "$IMAGE" "$ncu_bin" \
    --target-processes application-only --nvtx \
    --nvtx-include 'stage1_opt_profile/' --set basic --launch-count 50 \
    --force-overwrite --export "$PROFILE_C/stage1_opt_ncu" \
    python /workspace/opt/profile_v4_stage1_opt.py \
      --input_dir /data/voxel_csv_combined --session_filter "$SESSION_FILTER" \
      --model_path "/outputs${MODEL#$HOST_OUTPUTS}" \
      --calibration_json "/outputs${CALIBRATION#$HOST_OUTPUTS}" \
      --output_json "$PROFILE_C/NCU_PROFILE_SUMMARY.json" --warmup 1 --iterations 1 \
    2>&1 | tee "$PROFILE_ROOT/NCU_PROFILE.log"
fi

sha256sum "$PROFILE_ROOT"/* > "$PROFILE_ROOT/SHA256SUMS.txt"
tar -C "$HOST_OUTPUTS" -czf "/home/agni/v4_stage1_opt1_nsight_${STAMP}.tar.gz" \
  "${PROFILE_ROOT#$HOST_OUTPUTS/}"
sha256sum "/home/agni/v4_stage1_opt1_nsight_${STAMP}.tar.gz" \
  > "/home/agni/v4_stage1_opt1_nsight_${STAMP}.tar.gz.sha256"
printf '%s\n' "/home/agni/v4_stage1_opt1_nsight_${STAMP}.tar.gz" \
  > /home/agni/LATEST_V4_STAGE1_OPT_PROFILE_ARCHIVE.txt
echo "NSIGHT_PROFILE_OK profile_root=$PROFILE_ROOT"
echo "archive=/home/agni/v4_stage1_opt1_nsight_${STAMP}.tar.gz"
