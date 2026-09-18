#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
HANDOFF_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
UNITY_SERVER="${UNITY_SERVER:-$HANDOFF_ROOT/runtime/Linux/V10Stage12GpuPlayer}"
STATE_DIR="${STATE_DIR:-$HANDOFF_ROOT/state}"
OUTPUT_BASE="${OUTPUT_BASE:-$HANDOFF_ROOT/output/poleline_voxel_run_session_groups/v4_stage23_quality}"
STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-$OUTPUT_BASE/unity_native_v10_${STAMP}}"
RUN_LOG="${RUN_LOG:-$RUN_ROOT/UNITY_PLAYER.log}"
RESUME="${RESUME:-1}"
EXPECTED_SESSIONS="${EXPECTED_SESSIONS:-1}"
REFERENCE_ROOT="${REFERENCE_ROOT:-}"

test -x "$UNITY_SERVER" || {
  echo "ERROR: Unity GPU player is missing or not executable: $UNITY_SERVER" >&2
  exit 1
}
command -v nvidia-smi >/dev/null 2>&1 || {
  echo "ERROR: nvidia-smi is unavailable; NVIDIA GPU execution refused" >&2
  exit 1
}
nvidia-smi -L

input_count=0
[[ -n "${INPUT_MANIFEST:-}" ]] && input_count=$((input_count+1))
[[ -n "${INPUT_CSV:-}" ]] && input_count=$((input_count+1))
[[ -n "${INPUT_DIRECTORY:-}" ]] && input_count=$((input_count+1))
[[ "$input_count" -eq 1 ]] || {
  echo "ERROR: set exactly one of INPUT_MANIFEST, INPUT_CSV, or INPUT_DIRECTORY" >&2
  exit 1
}

mkdir -p "$STATE_DIR" "$RUN_ROOT"
args=(
  -force-vulkan
  -batchmode
  --v10-run-root "$RUN_ROOT"
  --v10-resume "$RESUME"
  --v10-expected-sessions "$EXPECTED_SESSIONS"
)

if [[ -n "${INPUT_MANIFEST:-}" ]]; then
  test -e "$INPUT_MANIFEST" || { echo "ERROR: missing INPUT_MANIFEST=$INPUT_MANIFEST" >&2; exit 1; }
  args+=(--v10-manifest "$INPUT_MANIFEST")
elif [[ -n "${INPUT_CSV:-}" ]]; then
  : "${GROUP_ID:?Set GROUP_ID for INPUT_CSV mode}"
  : "${SLICE_SEQ:?Set SLICE_SEQ for INPUT_CSV mode}"
  test -f "$INPUT_CSV" || { echo "ERROR: missing INPUT_CSV=$INPUT_CSV" >&2; exit 1; }
  args+=(--v10-input-csv "$INPUT_CSV" --v10-group-id "$GROUP_ID" --v10-slice-seq "$SLICE_SEQ")
  [[ -n "${RELATIVE_PATH:-}" ]] && args+=(--v10-relative-path "$RELATIVE_PATH")
elif [[ -n "${INPUT_DIRECTORY:-}" ]]; then
  : "${GROUP_ID:?Set GROUP_ID for INPUT_DIRECTORY mode}"
  test -d "$INPUT_DIRECTORY" || { echo "ERROR: missing INPUT_DIRECTORY=$INPUT_DIRECTORY" >&2; exit 1; }
  args+=(--v10-input-directory "$INPUT_DIRECTORY" --v10-group-id "$GROUP_ID")
  [[ -n "${START_SLICE_SEQ:-}" ]] && args+=(--v10-start-slice-seq "$START_SLICE_SEQ")
fi

[[ -n "$REFERENCE_ROOT" ]] && args+=(--v10-reference-root "$REFERENCE_ROOT")

nohup "$UNITY_SERVER" "${args[@]}" -logFile "$RUN_LOG" </dev/null >/dev/null 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$STATE_DIR/LATEST_PID.txt"
printf '%s\n' "$RUN_LOG" > "$STATE_DIR/LATEST_LOG.txt"
printf '%s\n' "$RUN_ROOT" > "$STATE_DIR/LATEST_RUN.txt"

sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  if [[ -f "$RUN_ROOT/STAGE12_COMPLETE.txt" ]]; then
    echo "COMPLETED_IMMEDIATELY pid=$pid"
    echo "run_root=$RUN_ROOT"
    exit 0
  fi
  echo "START_FAILED pid=$pid" >&2
  tail -n 120 "$RUN_LOG" 2>/dev/null || true
  exit 1
fi

echo "STARTED pid=$pid"
echo "run_root=$RUN_ROOT"
echo "log=$RUN_LOG"
echo "backend=Sentis_GPUCompute"
echo "graphics=Vulkan"
