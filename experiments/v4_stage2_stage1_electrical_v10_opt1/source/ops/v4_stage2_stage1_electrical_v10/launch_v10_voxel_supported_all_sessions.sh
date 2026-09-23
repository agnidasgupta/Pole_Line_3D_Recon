#!/usr/bin/env bash
set -euo pipefail
TOOL_DIR=/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10/ops/v4_stage2_stage1_electrical_v10
STAMP=${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
LOG=/home/agni/V10_VOXEL_SUPPORTED_STAGE2_${STAMP}.launch.log
nohup env RUN_STAMP="$STAMP" RESUME="${RESUME:-1}" EXPECTED_SESSIONS="${EXPECTED_SESSIONS:-30}" \
  SESSION_TIMEOUT_SECONDS="${SESSION_TIMEOUT_SECONDS:-7200}" \
  bash "$TOOL_DIR/run_v10_voxel_supported_fault_tolerant_stage2.sh" \
  > "$LOG" 2>&1 < /dev/null &
PID=$!
ROOT=/workspace/voxel_poleline/outputs/poleline_voxel_run_session_groups/v4_stage23_quality/fault_tolerant_v10_${STAMP}
printf '%s\n' "$PID" > /home/agni/LATEST_V10_FAULT_TOLERANT_STAGE2_PID.txt
printf '%s\n' "$LOG" > /home/agni/LATEST_V10_FAULT_TOLERANT_STAGE2_LAUNCH_LOG.txt
printf '%s\n' "$ROOT" > /home/agni/LATEST_V10_FAULT_TOLERANT_STAGE2_RUN.txt
printf '%s\n' "$ROOT/FAULT_TOLERANT_DRIVER.log" > /home/agni/LATEST_V10_FAULT_TOLERANT_STAGE2_LOG.txt
echo "V10_STAGE2_LAUNCHED"
echo "PID=$PID"
echo "RUN_ROOT=$ROOT"
echo "LAUNCH_LOG=$LOG"
