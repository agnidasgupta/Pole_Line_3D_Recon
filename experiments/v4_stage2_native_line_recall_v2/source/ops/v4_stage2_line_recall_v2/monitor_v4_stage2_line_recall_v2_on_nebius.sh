#!/usr/bin/env bash
set -euo pipefail

RUN_PTR="$HOME/LATEST_V4_STAGE2_LINE_RECALL_V2_RUN.txt"
PID_PTR="$HOME/LATEST_V4_STAGE2_LINE_RECALL_V2_PID.txt"
LOG_PTR="$HOME/v4_stage2_line_recall_v2_master.log"

[ -f "$RUN_PTR" ] || { echo "No run pointer: $RUN_PTR"; exit 1; }
RUN_HOST=$(cat "$RUN_PTR")
PID=""
[ -f "$PID_PTR" ] && PID=$(cat "$PID_PTR")

EXPECTED=0
if [ -f "$RUN_HOST/session_map.tsv" ]; then
  EXPECTED=$(wc -l < "$RUN_HOST/session_map.tsv" | tr -d ' ')
fi
DONE=$(find "$RUN_HOST/status" -type f -name '*.ok' 2>/dev/null | wc -l | tr -d ' ')

echo "============================================================"
echo "V4 STAGE2 LINE-RECALL V2 STATUS"
echo "Run:       $RUN_HOST"
echo "PID:       ${PID:-unknown}"
if [ -n "$PID" ] && ps -p "$PID" >/dev/null 2>&1; then
  echo "Process:   RUNNING"
else
  echo "Process:   NOT RUNNING"
fi
echo "Completed: $DONE / $EXPECTED"
if [ -f "$RUN_HOST/ALL_STAGE2_V2_OK.txt" ]; then
  echo "Overall:   COMPLETE + PACKAGED"
  cat "$RUN_HOST/ALL_STAGE2_V2_OK.txt"
elif [ -f "$RUN_HOST/PHASE2_STAGE2_OK.txt" ]; then
  echo "Overall:   STAGE2 COMPLETE; PACKAGING MAY BE ACTIVE"
  cat "$RUN_HOST/PHASE2_STAGE2_OK.txt"
else
  echo "Overall:   RUNNING OR FAILED"
fi
echo "============================================================"

echo
echo "--- latest master log ---"
tail -80 "$LOG_PTR" 2>/dev/null || true

echo
echo "--- latest session log ---"
LATEST=$(find "$RUN_HOST/logs" -type f -name '*.log' -print0 2>/dev/null | xargs -0 ls -1t 2>/dev/null | head -1 || true)
if [ -n "$LATEST" ]; then
  echo "$LATEST"
  tail -60 "$LATEST"
fi
