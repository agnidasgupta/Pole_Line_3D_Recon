#!/usr/bin/env bash
set -euo pipefail
PID_FILE="$HOME/LATEST_V4_STAGE2_STAGE1_TRACK_JOIN_V9_PID.txt"
RUN_FILE="$HOME/LATEST_V4_STAGE2_STAGE1_TRACK_JOIN_V9_RUN.txt"
LOG_FILE="${MASTER_LOG:-$HOME/v4_stage2_stage1_track_join_v9.log}"
if [ -f "$PID_FILE" ]; then
  PID=$(cat "$PID_FILE")
  if ps -p "$PID" >/dev/null 2>&1; then echo "process=ACTIVE pid=$PID"; else echo "process=NOT_ACTIVE pid=$PID"; fi
fi
if [ -f "$RUN_FILE" ]; then
  RUN_HOST=$(cat "$RUN_FILE")
  echo "run=$RUN_HOST"
  if [ -d "$RUN_HOST/status" ]; then
    DONE=$(find "$RUN_HOST/status" -maxdepth 1 -type f -name '*.stage2.ok' | wc -l | tr -d ' ')
    echo "completed_sessions=$DONE"
  fi
  [ -f "$RUN_HOST/PHASE2_STAGE2_OK.txt" ] && cat "$RUN_HOST/PHASE2_STAGE2_OK.txt"
fi
[ -f "$LOG_FILE" ] && tail -40 "$LOG_FILE"
