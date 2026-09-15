#!/usr/bin/env bash
set -euo pipefail
TOOL_DIR=/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10/ops/v4_stage1_inference_opt1
STAMP=${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
LOG=/home/agni/V4_STAGE1_OPT_${STAMP}.launch.log
nohup env RUN_STAMP="$STAMP" RESUME="${RESUME:-1}" EXPECTED_SESSIONS="${EXPECTED_SESSIONS:-30}" \
  SESSION_TIMEOUT_SECONDS="${SESSION_TIMEOUT_SECONDS:-7200}" SCORE_ATOL="${SCORE_ATOL:-1e-4}" \
  bash "$TOOL_DIR/run_v4_stage1_opt_experiment.sh" > "$LOG" 2>&1 < /dev/null &
PID=$!
ROOT=/workspace/voxel_poleline/outputs/poleline_voxel_run_session_groups/v4_production/stage1_opt_experiments/$STAMP
printf '%s\n' "$PID" > /home/agni/LATEST_V4_STAGE1_OPT_PID.txt
printf '%s\n' "$LOG" > /home/agni/LATEST_V4_STAGE1_OPT_LAUNCH_LOG.txt
printf '%s\n' "$ROOT" > /home/agni/LATEST_V4_STAGE1_OPT_RUN.txt
echo "V4_STAGE1_OPT1_LAUNCHED"
echo "PID=$PID"
echo "RUN_ROOT=$ROOT"
echo "LAUNCH_LOG=$LOG"
