#!/usr/bin/env bash
set -euo pipefail
PID=$(cat /home/agni/LATEST_V10_UNITY_EXPORT_PID.txt)
LAUNCH_LOG=$(cat /home/agni/LATEST_V10_UNITY_EXPORT_LAUNCH_LOG.txt)
RUN_ROOT=$(cat /home/agni/LATEST_V10_UNITY_EXPORT_RUN.txt 2>/dev/null || true)
NOW=$(date +%s)
if kill -0 "$PID" 2>/dev/null; then
  echo "RUNNING pid=$PID"
  ps -o pid,ppid,stat,etime,%cpu,%mem,cmd -p "$PID"
else
  echo "EXITED pid=$PID"
fi
if [ -n "$RUN_ROOT" ] && [ -d "$RUN_ROOT" ]; then
  echo "run_root=$RUN_ROOT"
  find "$RUN_ROOT" -maxdepth 4 -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' | sort | tail -20
  [ -s "$RUN_ROOT/UNITY_EXPORT_COMPLETE.txt" ] && echo READY_TO_PACKAGE || true
  LOG="$RUN_ROOT/ONNX_EXPORT.log"
else
  LOG="$LAUNCH_LOG"
fi
if [ -s "$LOG" ]; then
  MODIFIED=$(stat -c %Y "$LOG")
  AGE=$((NOW-MODIFIED))
  SIZE=$(stat -c %s "$LOG")
  echo "log=$LOG bytes=$SIZE last_change_seconds=$AGE"
  if kill -0 "$PID" 2>/dev/null && [ "$AGE" -gt 900 ]; then echo "POSSIBLY_STALLED no log change for over 15 minutes"; fi
  tail -n 80 "$LOG"
fi
echo "containers:"
docker ps --format 'table {{.ID}}\t{{.Status}}\t{{.Image}}\t{{.Command}}'
