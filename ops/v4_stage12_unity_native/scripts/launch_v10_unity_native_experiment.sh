#!/usr/bin/env bash
set -euo pipefail

: "${UNITY_SERVER:?Set UNITY_SERVER to the uploaded V10Stage12Server executable}"
: "${INPUT_MANIFEST:?Set INPUT_MANIFEST to the Unity input_manifest.csv}"

OUTPUTS=/workspace/voxel_poleline/outputs
STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-$OUTPUTS/poleline_voxel_run_session_groups/v4_stage23_quality/unity_native_v10_${STAMP}}"
LOG="${RUN_LOG:-$RUN_ROOT/UNITY_PLAYER.log}"
RESUME="${RESUME:-1}"
EXPECTED_SESSIONS="${EXPECTED_SESSIONS:-30}"
REFERENCE_ROOT="${REFERENCE_ROOT:-}"

mkdir -p "$RUN_ROOT"
args=(
  -batchmode
  --v10-manifest "$INPUT_MANIFEST"
  --v10-run-root "$RUN_ROOT"
  --v10-resume "$RESUME"
  --v10-expected-sessions "$EXPECTED_SESSIONS"
)
if [[ "${UNITY_HEADLESS:-1}" == 1 ]]; then
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
