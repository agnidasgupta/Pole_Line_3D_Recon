#!/usr/bin/env bash
set -euo pipefail
REPO=${EXP_REPO:-/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10}
STAMP=${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
LAUNCH_LOG=/home/agni/V10_UNITY_EXPORT_LAUNCH_$STAMP.log
nohup env RUN_STAMP="$STAMP" EXP_REPO="$REPO" \
  bash "$REPO/ops/v4_stage12_unity_sentis/build_v10_unity_export_on_nebius.sh" \
  >"$LAUNCH_LOG" 2>&1 </dev/null &
PID=$!
printf '%s\n' "$PID" >/home/agni/LATEST_V10_UNITY_EXPORT_PID.txt
printf '%s\n' "$LAUNCH_LOG" >/home/agni/LATEST_V10_UNITY_EXPORT_LAUNCH_LOG.txt
printf 'STARTED pid=%s launch_log=%s stamp=%s\n' "$PID" "$LAUNCH_LOG" "$STAMP"
