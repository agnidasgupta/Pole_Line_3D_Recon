#!/usr/bin/env bash
set -euo pipefail
RUN_PTR="$HOME/LATEST_V4_STAGE2_STAGE1_EXACT_V8_RUN.txt"
PID_PTR="$HOME/LATEST_V4_STAGE2_STAGE1_EXACT_V8_PID.txt"
MASTER_LOG="${MASTER_LOG:-$HOME/v4_stage2_stage1_exact_v8_master.log}"

if [ -f "$PID_PTR" ]; then
  PID=$(cat "$PID_PTR")
  if ps -p "$PID" >/dev/null 2>&1; then echo "status=ACTIVE pid=$PID"; else echo "status=NOT_ACTIVE pid=$PID"; fi
fi
if [ -f "$RUN_PTR" ]; then
  RUN=$(cat "$RUN_PTR")
  echo "run=$RUN"
  if [ -f "$RUN/session_map.tsv" ]; then
    total=$(wc -l < "$RUN/session_map.tsv" | tr -d ' ')
    done=$(find "$RUN/status" -maxdepth 1 -type f -name '*.stage2.ok' 2>/dev/null | wc -l | tr -d ' ')
    echo "sessions=$done/$total"
  fi
  test -f "$RUN/PHASE2_STAGE2_OK.txt" && cat "$RUN/PHASE2_STAGE2_OK.txt" || true
  test -f "$RUN/STAGE2_ONLY_COMPLETE.txt" && cat "$RUN/STAGE2_ONLY_COMPLETE.txt" || true
fi
if [ -f "$MASTER_LOG" ]; then
  echo "--- log tail ---"
  tail -60 "$MASTER_LOG"
fi
