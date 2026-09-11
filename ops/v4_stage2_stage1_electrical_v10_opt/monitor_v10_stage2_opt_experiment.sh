#!/usr/bin/env bash
set -u
PID_FILE=/home/agni/LATEST_V10_STAGE2_OPT_PID.txt
RUN_FILE=/home/agni/LATEST_V10_STAGE2_OPT_RUN.txt
[ -s "$PID_FILE" ] && [ -s "$RUN_FILE" ] || { echo "NO_LATEST_RUN_POINTERS"; exit 2; }
PID=$(tr -cd '0-9' < "$PID_FILE")
RUN_ROOT=$(head -n 1 "$RUN_FILE")
DRIVER="$RUN_ROOT/FAULT_TOLERANT_DRIVER.log"
accepted=$(find "$RUN_ROOT/status" -maxdepth 1 -type f -name '*.stage2.ok' 2>/dev/null | wc -l | tr -d ' ')
failed=$(find "$RUN_ROOT/status" -maxdepth 1 -type f -name '*.stage2.failed' 2>/dev/null | wc -l | tr -d ' ')
equivalent=$(find "$RUN_ROOT/status" -maxdepth 1 -type f -name '*.quality_equivalence.json' 2>/dev/null | wc -l | tr -d ' ')
latest_epoch=$(find "$RUN_ROOT" -type f -printf '%T@\n' 2>/dev/null | sort -nr | head -n 1 | cut -d. -f1)
now=$(date +%s)
age=$((now-${latest_epoch:-now}))
echo "RUN_ROOT=$RUN_ROOT"
echo "ACCEPTED=$accepted FAILED=$failed EQUIVALENT=$equivalent LAST_OUTPUT_AGE_SECONDS=$age"
process_command=""
if [ -r "/proc/$PID/cmdline" ]; then
  process_command=$(tr '\0' ' ' < "/proc/$PID/cmdline")
fi
if kill -0 "$PID" 2>/dev/null && [[ "$process_command" == *run_v10_stage2_opt_experiment.sh* ]]; then
  echo "STATE=RUNNING"
  echo "PROCESS_COMMAND=$process_command"
  ps -o pid,ppid,stat,etime,%cpu,%mem,cmd -p "$PID" --ppid "$PID"
  docker ps --format 'table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Image}}'
  if [ "$age" -gt 900 ]; then
    echo "HEALTH=CHECK_POSSIBLE_STALL (no output change for more than 15 minutes)"
  else
    echo "HEALTH=ACTIVE_OR_WAITING_NORMALLY"
  fi
else
  echo "STATE=EXITED"
  if [ -s "$RUN_ROOT/STAGE2_ONLY_COMPLETE.txt" ] && [ -s "$RUN_ROOT/STAGE2_TIMING_SESSION_AVERAGES.txt" ] && [ "$accepted" -eq 30 ] && [ "$failed" -eq 0 ] && [ "$equivalent" -eq 30 ]; then
    echo "READY_TO_PACKAGE=YES"
    cat "$RUN_ROOT/STAGE2_ONLY_COMPLETE.txt"
  else
    echo "READY_TO_PACKAGE=NO"
    echo "The run exited incomplete; inspect failed status files and logs."
  fi
fi
echo "===== RECENT DRIVER LOG ====="
tail -n 80 "$DRIVER" 2>/dev/null || true
echo "===== FAILURE SIGNALS ====="
grep -nEi 'ERROR|Traceback|RuntimeError|timed out|SESSION_REJECTED|INCOMPLETE|mismatch|equivalence|stalled|killed' "$DRIVER" 2>/dev/null | tail -n 60 || true
