#!/usr/bin/env bash
set -euo pipefail
TOOL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP_REPO=$(cd "$TOOL_DIR/../.." && pwd)
STAMP=${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
VARIANT_NAME=${VARIANT_NAME:?VARIANT_NAME is required}
RUN_ID="${STAMP}_${VARIANT_NAME}"
LOG="/home/agni/V4_STAGE1_OPT2_${RUN_ID}.launch.log"
nohup env EXP_REPO="$EXP_REPO" RUN_STAMP="$STAMP" VARIANT_NAME="$VARIANT_NAME" \
  RESUME="${RESUME:-1}" EXPECTED_SESSIONS="${EXPECTED_SESSIONS:-30}" \
  ONLY_GROUP_ID="${ONLY_GROUP_ID:-}" SESSION_TIMEOUT_SECONDS="${SESSION_TIMEOUT_SECONDS:-7200}" \
  SCORE_ATOL="${SCORE_ATOL:-0}" COMPILE_MODEL="${COMPILE_MODEL:-0}" \
  COMPILE_MODE="${COMPILE_MODE:-default}" BATCH_SIZE="${BATCH_SIZE:-12}" \
  CHANNELS_LAST="${CHANNELS_LAST:-1}" PINNED_D2H="${PINNED_D2H:-0}" \
  DETAILED_CUDA_TIMING="${DETAILED_CUDA_TIMING:-1}" \
  RETAIN_GATHER_HOST_BUFFERS="${RETAIN_GATHER_HOST_BUFFERS:-0}" \
  PRUNE_EMBEDDING_HEAD="${PRUNE_EMBEDDING_HEAD:-0}" WARMUP_ITERATIONS="${WARMUP_ITERATIONS:-0}" \
  bash "$TOOL_DIR/run_v4_stage1_opt2_experiment.sh" > "$LOG" 2>&1 < /dev/null &
PID=$!
ROOT="/workspace/voxel_poleline/outputs/poleline_voxel_run_session_groups/v4_production/stage1_opt2_experiments/$RUN_ID"
printf '%s\n' "$PID" > /home/agni/LATEST_V4_STAGE1_OPT2_PID.txt
printf '%s\n' "$LOG" > /home/agni/LATEST_V4_STAGE1_OPT2_LAUNCH_LOG.txt
printf '%s\n' "$ROOT" > /home/agni/LATEST_V4_STAGE1_OPT2_RUN.txt
echo "V4_STAGE1_OPT2_LAUNCHED"
echo "PID=$PID"
echo "VARIANT_NAME=$VARIANT_NAME"
echo "RUN_ROOT=$ROOT"
echo "LAUNCH_LOG=$LOG"
