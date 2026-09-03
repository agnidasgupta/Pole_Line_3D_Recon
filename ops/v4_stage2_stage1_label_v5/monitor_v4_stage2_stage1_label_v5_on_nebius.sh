#!/usr/bin/env bash
set -euo pipefail

RUN_POINTER="${RUN_POINTER:-$HOME/LATEST_V4_STAGE2_STAGE1_LABEL_V5_RUN.txt}"
PID_POINTER="${PID_POINTER:-$HOME/LATEST_V4_STAGE2_STAGE1_LABEL_V5_PID.txt}"
MASTER_LOG="${MASTER_LOG:-$HOME/v4_stage2_stage1_label_v5_master.log}"

if [ ! -f "$RUN_POINTER" ]; then
  echo "No Stage2 Stage1-label run pointer exists yet: $RUN_POINTER"
  exit 1
fi
RUN_HOST=$(cat "$RUN_POINTER")
[ -d "$RUN_HOST" ] || { echo "Run directory missing: $RUN_HOST"; exit 1; }

TOTAL=0
DONE=0
MANIFESTS=0
TIMINGS=0
[ -f "$RUN_HOST/session_map.tsv" ] && TOTAL=$(wc -l < "$RUN_HOST/session_map.tsv" | tr -d ' ')
[ -d "$RUN_HOST/status" ] && DONE=$(find "$RUN_HOST/status" -maxdepth 1 -type f -name '*.stage2.ok' | wc -l | tr -d ' ')
[ -d "$RUN_HOST/stage2" ] && MANIFESTS=$(find "$RUN_HOST/stage2" -mindepth 2 -maxdepth 2 -type f -name inference_manifest.csv | wc -l | tr -d ' ')
[ -d "$RUN_HOST/timing/stage2" ] && TIMINGS=$(find "$RUN_HOST/timing/stage2" -maxdepth 1 -type f -name '*.csv' | wc -l | tr -d ' ')

PROCESS="UNKNOWN"
if [ -f "$PID_POINTER" ]; then
  PID=$(cat "$PID_POINTER")
  if ps -p "$PID" >/dev/null 2>&1; then
    PROCESS="RUNNING (PID $PID)"
  else
    PROCESS="NOT RUNNING (PID $PID)"
  fi
else
  PID=""
  PROCESS="PID POINTER MISSING"
fi

SELECTION="PENDING"
if [ -f "$RUN_HOST/selection/selection_report.txt" ]; then
  SELECTION=$(head -1 "$RUN_HOST/selection/selection_report.txt")
elif [ -f "$RUN_HOST/NO_SAFE_STAGE1_LABEL_PROFILE.txt" ]; then
  SELECTION="NO_SAFE_STAGE1_LABEL_PROFILE"
fi

PHASE2="INCOMPLETE"
[ -f "$RUN_HOST/PHASE2_STAGE2_OK.txt" ] && PHASE2="COMPLETE"
OVERALL="INCOMPLETE"
[ -f "$RUN_HOST/STAGE2_ONLY_COMPLETE.txt" ] && OVERALL="COMPLETE + PACKAGED"

cat <<EOF
============================================================
V4 STAGE2 STAGE1-LABEL V5 STATUS
============================================================
Run:          $RUN_HOST
Process:      $PROCESS
Selection:    $SELECTION
Sessions:     $DONE / $TOTAL
Manifests:    $MANIFESTS / $TOTAL
Timings:      $TIMINGS / $TOTAL
Phase2:       $PHASE2
Overall:      $OVERALL
Master log:   $MASTER_LOG
============================================================
EOF

if [ -f "$MASTER_LOG" ]; then
  echo
  echo "=== Latest master-log output ==="
  tail -60 "$MASTER_LOG"
fi

if [ -f "$RUN_HOST/STAGE2_ONLY_COMPLETE.txt" ]; then
  echo
  echo "=== Completion markers ==="
  cat "$RUN_HOST/PHASE2_STAGE2_OK.txt"
  cat "$RUN_HOST/STAGE2_ONLY_COMPLETE.txt"
  if [ -f "$HOME/LATEST_V4_STAGE2_STAGE1_LABEL_V5_ARCHIVE.txt" ]; then
    ARCHIVE=$(cat "$HOME/LATEST_V4_STAGE2_STAGE1_LABEL_V5_ARCHIVE.txt")
    echo "Archive: $ARCHIVE"
    ls -lh "$ARCHIVE" "${ARCHIVE}.sha256"
  fi
elif [ "$PROCESS" != "RUNNING"* ] && [ "$SELECTION" != "NO_SAFE_STAGE1_LABEL_PROFILE" ]; then
  echo
  echo "WARNING: process is not running and final completion marker is absent."
  exit 2
fi
