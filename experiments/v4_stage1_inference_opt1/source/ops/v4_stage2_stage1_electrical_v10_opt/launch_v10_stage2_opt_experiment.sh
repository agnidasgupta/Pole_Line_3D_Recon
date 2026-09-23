#!/usr/bin/env bash
set -euo pipefail
TOOL_DIR=/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10/ops/v4_stage2_stage1_electrical_v10_opt
QUALITY_BASELINE_ROOT=${QUALITY_BASELINE_ROOT:-$(head -n 1 /home/agni/LATEST_V10_FAULT_TOLERANT_STAGE2_RUN.txt)}
STAMP=${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
LOG=/home/agni/V10_STAGE2_OPT_${STAMP}.launch.log
nohup env RUN_STAMP="$STAMP" RESUME="${RESUME:-1}" EXPECTED_SESSIONS="${EXPECTED_SESSIONS:-30}" \
  SESSION_TIMEOUT_SECONDS="${SESSION_TIMEOUT_SECONDS:-7200}" \
  QUALITY_BASELINE_ROOT="$QUALITY_BASELINE_ROOT" \
  bash "$TOOL_DIR/run_v10_stage2_opt_experiment.sh" \
  > "$LOG" 2>&1 < /dev/null &
PID=$!
ROOT=/workspace/voxel_poleline/outputs/poleline_voxel_run_session_groups/v4_stage23_quality/fault_tolerant_v10_${STAMP}
printf '%s\n' "$PID" > /home/agni/LATEST_V10_STAGE2_OPT_PID.txt
printf '%s\n' "$LOG" > /home/agni/LATEST_V10_STAGE2_OPT_LAUNCH_LOG.txt
printf '%s\n' "$ROOT" > /home/agni/LATEST_V10_STAGE2_OPT_RUN.txt
printf '%s\n' "$QUALITY_BASELINE_ROOT" > /home/agni/LATEST_V10_STAGE2_OPT_BASELINE.txt
echo "V10_STAGE2_OPT_LAUNCHED"
echo "PID=$PID"
echo "RUN_ROOT=$ROOT"
echo "QUALITY_BASELINE_ROOT=$QUALITY_BASELINE_ROOT"
echo "LAUNCH_LOG=$LOG"
