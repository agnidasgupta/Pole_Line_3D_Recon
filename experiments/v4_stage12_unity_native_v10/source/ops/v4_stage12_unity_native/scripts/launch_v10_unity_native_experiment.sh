#!/usr/bin/env bash
set -euo pipefail

: "${UNITY_SERVER:?Set UNITY_SERVER to the uploaded V10Stage12GpuPlayer executable}"

OUTPUTS=/workspace/voxel_poleline/outputs
STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-$OUTPUTS/poleline_voxel_run_session_groups/v4_stage23_quality/unity_native_v10_${STAMP}}"
LOG="${RUN_LOG:-$RUN_ROOT/UNITY_PLAYER.log}"
RESUME="${RESUME:-1}"
EXPECTED_SESSIONS="${EXPECTED_SESSIONS:-30}"
REFERENCE_ROOT="${REFERENCE_ROOT:-}"
V10_UNITY_GPU="${V10_UNITY_GPU:-1}"

input_count=0
[[ -n "${INPUT_MANIFEST:-}" ]] && input_count=$((input_count+1))
[[ -n "${INPUT_CSV:-}" ]] && input_count=$((input_count+1))
[[ -n "${INPUT_DIRECTORY:-}" ]] && input_count=$((input_count+1))
[[ "$input_count" -eq 1 ]] || {
  echo "ERROR: set exactly one of INPUT_MANIFEST, INPUT_CSV, or INPUT_DIRECTORY" >&2
  exit 1
}

mkdir -p "$RUN_ROOT"
args=(
  -batchmode
  --v10-run-root "$RUN_ROOT"
  --v10-resume "$RESUME"
  --v10-expected-sessions "$EXPECTED_SESSIONS"
)
if [[ -n "${INPUT_MANIFEST:-}" ]]; then
  args+=(--v10-manifest "$INPUT_MANIFEST")
elif [[ -n "${INPUT_CSV:-}" ]]; then
  : "${GROUP_ID:?Set GROUP_ID for raw CSV input}"
  : "${SLICE_SEQ:?Set SLICE_SEQ for raw CSV input}"
  args+=(--v10-input-csv "$INPUT_CSV" --v10-group-id "$GROUP_ID" --v10-slice-seq "$SLICE_SEQ")
  [[ -n "${RELATIVE_PATH:-}" ]] && args+=(--v10-relative-path "$RELATIVE_PATH")
else
  : "${GROUP_ID:?Set GROUP_ID for raw input-directory mode}"
  args+=(--v10-input-directory "$INPUT_DIRECTORY" --v10-group-id "$GROUP_ID")
  [[ -n "${START_SLICE_SEQ:-}" ]] && args+=(--v10-start-slice-seq "$START_SLICE_SEQ")
fi
if [[ "$V10_UNITY_GPU" == 1 ]]; then
  command -v nvidia-smi >/dev/null || {
    echo "ERROR: nvidia-smi is unavailable; GPU production launch refused" >&2
    exit 1
  }
  nvidia-smi -L
  args=(-force-vulkan "${args[@]}")
else
  args=(-nographics "${args[@]}")
fi
if [[ -n "$REFERENCE_ROOT" ]]; then
  args+=(--v10-reference-root "$REFERENCE_ROOT")
fi

nohup "$UNITY_SERVER" "${args[@]}" -logFile "$LOG" </dev/null >/dev/null 2>&1 &
pid=$!
printf '%s\n' "$pid" > /home/agni/LATEST_V10_UNITY_NATIVE_PID.txt
printf '%s\n' "$LOG" > /home/agni/LATEST_V10_UNITY_NATIVE_LOG.txt
printf '%s\n' "$RUN_ROOT" > /home/agni/LATEST_V10_UNITY_NATIVE_RUN.txt

sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  if [[ -f "$RUN_ROOT/STAGE12_COMPLETE.txt" ]]; then
    echo "COMPLETED_IMMEDIATELY pid=$pid"
    echo "run_root=$RUN_ROOT"
    exit 0
  fi
  echo "START_FAILED pid=$pid"
  tail -n 120 "$LOG" || true
  exit 1
fi
echo "STARTED pid=$pid"
echo "run_root=$RUN_ROOT"
echo "log=$LOG"
echo "gpu_requested=$V10_UNITY_GPU"
