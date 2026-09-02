#!/usr/bin/env bash
set -euo pipefail

RUN_PTR="$HOME/LATEST_V4_STAGE2_GT_AUTOSELECT_V3_RUN.txt"
PID_PTR="$HOME/LATEST_V4_STAGE2_GT_AUTOSELECT_V3_PID.txt"
LOG_PTR="$HOME/v4_stage2_gt_autoselect_v3_master.log"

[ -f "$RUN_PTR" ] || { echo "No run pointer: $RUN_PTR"; exit 1; }
RUN_HOST=$(cat "$RUN_PTR")
PID=""
[ -f "$PID_PTR" ] && PID=$(cat "$PID_PTR")

EXPECTED=0
[ -f "$RUN_HOST/session_map.tsv" ] && EXPECTED=$(( $(wc -l < "$RUN_HOST/session_map.tsv") - 1 ))
[ "$EXPECTED" -lt 0 ] && EXPECTED=0
DONE=$(find "$RUN_HOST/status" -type f -name '*.ok' 2>/dev/null | wc -l | tr -d ' ')

echo "============================================================"
echo "V4 STAGE2 GT AUTOSELECT V3 STATUS"
echo "Run:       $RUN_HOST"
echo "PID:       ${PID:-unknown}"
if [ -n "$PID" ] && ps -p "$PID" >/dev/null 2>&1; then
  echo "Process:   RUNNING"
else
  echo "Process:   NOT RUNNING"
fi
if [ -f "$RUN_HOST/selection/selected_profile.env" ]; then
  # shellcheck disable=SC1090
  source "$RUN_HOST/selection/selected_profile.env"
  echo "Selection: ${SELECTION_STATUS:-unknown}"
  echo "Profile:   ${SELECTED_PROFILE_NAME:-unknown}"
  echo "Threshold: candidate=${LINE_CANDIDATE_THRESHOLD:-?} weak=${LINE_WEAK_THRESHOLD:-?} competition=${LINE_COMPETITION_RATIO:-?} refiner=${LINE_REFINER_THRESHOLD:-?}"
else
  echo "Selection: not finished"
fi
echo "Stage2:    $DONE / $EXPECTED sessions"
if [ -f "$RUN_HOST/ALL_STAGE2_GT_AUTOSELECT_V3_OK.txt" ]; then
  echo "Overall:   COMPLETE + PACKAGED"
  cat "$RUN_HOST/ALL_STAGE2_GT_AUTOSELECT_V3_OK.txt"
elif [ -f "$RUN_HOST/NO_SAFE_GT_IMPROVEMENT.txt" ]; then
  echo "Overall:   NO SAFE IMPROVEMENT; DIAGNOSTICS PACKAGED"
  cat "$RUN_HOST/NO_SAFE_GT_IMPROVEMENT.txt"
elif [ -f "$RUN_HOST/PHASE2_STAGE2_OK.txt" ]; then
  echo "Overall:   STAGE2 COMPLETE; PACKAGING MAY BE ACTIVE"
else
  echo "Overall:   SELECTING/RUNNING/FAILED"
fi
echo "============================================================"

echo
echo "--- selection report ---"
cat "$RUN_HOST/selection/selection_report.txt" 2>/dev/null || true

echo
echo "--- latest master log ---"
tail -100 "$LOG_PTR" 2>/dev/null || true

echo
echo "--- latest Stage2 session log ---"
LATEST=$(find "$RUN_HOST/logs" -type f -name 'stage2__*.log' -print0 2>/dev/null | xargs -0 ls -1t 2>/dev/null | head -1 || true)
if [ -n "$LATEST" ]; then
  echo "$LATEST"
  tail -60 "$LATEST"
fi
