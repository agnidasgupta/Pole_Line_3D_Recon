#!/usr/bin/env bash
set -euo pipefail

PID=$(cat /home/agni/LATEST_V10_UNITY_NATIVE_PID.txt)
LOG=$(cat /home/agni/LATEST_V10_UNITY_NATIVE_LOG.txt)
RUN_ROOT=$(cat /home/agni/LATEST_V10_UNITY_NATIVE_RUN.txt)
HEARTBEAT="$RUN_ROOT/RUN_HEARTBEAT.txt"
STALL_SECONDS="${STALL_SECONDS:-300}"

if kill -0 "$PID" 2>/dev/null; then
  process=RUNNING
else
  process=EXITED
fi

age=-1
if [[ -f "$HEARTBEAT" ]]; then
  now=$(date +%s)
  changed=$(stat -c %Y "$HEARTBEAT")
  age=$((now-changed))
fi

state=STARTING
if [[ -f "$RUN_ROOT/STAGE12_COMPLETE.txt" ]]; then
  state=COMPLETE
elif [[ -f "$RUN_ROOT/STAGE12_FAILED.txt" || -f "$RUN_ROOT/FATAL_ERROR.txt" ]]; then
  state=FAILED
elif [[ "$process" == RUNNING && "$age" -ge 0 && "$age" -le "$STALL_SECONDS" ]]; then
  state=ACTIVE
elif [[ "$process" == RUNNING ]]; then
  state=STALLED
elif [[ "$process" == EXITED ]]; then
  state=EXITED_INCOMPLETE
fi

echo "process=$process pid=$PID"
echo "state=$state heartbeat_age_seconds=$age stall_threshold_seconds=$STALL_SECONDS"
echo "run_root=$RUN_ROOT"
[[ -f "$HEARTBEAT" ]] && cat "$HEARTBEAT"
accepted=0
failed_sessions=0
failed_slices=0
if [[ -d "$RUN_ROOT/status" ]]; then
  accepted=$(find "$RUN_ROOT/status" -maxdepth 1 -type f -name '*.stage2.ok' | wc -l | tr -d ' ')
  failed_sessions=$(find "$RUN_ROOT/status" -maxdepth 1 -type f -name '*.stage2.failed' | wc -l | tr -d ' ')
  failed_slices=$(find "$RUN_ROOT/status" -maxdepth 1 -type f -name '*.slice_*.failed' | wc -l | tr -d ' ')
fi
echo "accepted_sessions=$accepted"
echo "failed_sessions=$failed_sessions"
echo "failed_slices=$failed_slices"
tail -n 100 "$LOG" 2>/dev/null || true

case "$state" in
  COMPLETE) exit 0 ;;
  ACTIVE|STARTING) exit 10 ;;
  *) exit 1 ;;
esac
