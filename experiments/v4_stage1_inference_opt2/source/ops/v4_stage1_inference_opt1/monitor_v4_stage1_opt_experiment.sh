#!/usr/bin/env bash
set -u
PID_FILE=/home/agni/LATEST_V4_STAGE1_OPT_PID.txt
RUN_FILE=/home/agni/LATEST_V4_STAGE1_OPT_RUN.txt
LAUNCH_FILE=/home/agni/LATEST_V4_STAGE1_OPT_LAUNCH_LOG.txt
[ -s "$PID_FILE" ] && [ -s "$RUN_FILE" ] || { echo "NO_LATEST_STAGE1_OPT_POINTERS"; exit 2; }
PID=$(tr -cd '0-9' < "$PID_FILE")
RUN_ROOT=$(head -n 1 "$RUN_FILE")
DRIVER="$RUN_ROOT/STAGE1_OPT_DRIVER.log"
accepted=$(find "$RUN_ROOT/status" -maxdepth 1 -type f -name '*.stage1.ok' 2>/dev/null | wc -l | tr -d ' ')
failed=$(find "$RUN_ROOT/status" -maxdepth 1 -type f -name '*.stage1.failed' 2>/dev/null | wc -l | tr -d ' ')
equivalent=$(find "$RUN_ROOT/status" -maxdepth 1 -type f -name '*.quality_equivalence.json' 2>/dev/null | wc -l | tr -d ' ')
latest_epoch=$(find "$RUN_ROOT" -type f -printf '%T@\n' 2>/dev/null | sort -nr | head -n 1 | cut -d. -f1)
now=$(date +%s)
age=$((now-${latest_epoch:-now}))
echo "RUN_ROOT=$RUN_ROOT"
echo "ACCEPTED=$accepted FAILED=$failed EQUIVALENT=$equivalent LAST_OUTPUT_AGE_SECONDS=$age"
process_command=""
if [ -r "/proc/$PID/cmdline" ]; then process_command=$(tr '\0' ' ' < "/proc/$PID/cmdline"); fi
if kill -0 "$PID" 2>/dev/null && [[ "$process_command" == *run_v4_stage1_opt_experiment.sh* ]]; then
  echo "STATE=RUNNING"
  echo "PROCESS_COMMAND=$process_command"
  ps -o pid,ppid,stat,etime,%cpu,%mem,cmd -p "$PID" --ppid "$PID"
  docker ps --format 'table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Image}}'
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || true
  if [ "$age" -gt 900 ]; then
    echo "HEALTH=CHECK_POSSIBLE_STALL"
    echo "No output changed for more than 15 minutes; inspect GPU utilization and the current progress JSON below."
  else
    echo "HEALTH=ACTIVE_OR_WAITING_NORMALLY"
  fi
else
  echo "STATE=EXITED"
  if [ -s "$RUN_ROOT/STAGE1_OPT_EQUIVALENT_COMPLETE.txt" ] && \
     [ -s "$RUN_ROOT/STAGE1_TIMING_SESSION_AVERAGES.txt" ] && \
     [ "$accepted" -eq 30 ] && [ "$failed" -eq 0 ] && [ "$equivalent" -eq 30 ]; then
    echo "READY_TO_PACKAGE=YES"
    cat "$RUN_ROOT/STAGE1_OPT_EQUIVALENT_COMPLETE.txt"
  else
    echo "READY_TO_PACKAGE=NO"
  fi
fi
echo "===== CURRENT PROGRESS ====="
find "$RUN_ROOT/status" -maxdepth 1 -type f -name '*.progress.json' -printf '%T@ %p\n' 2>/dev/null \
  | sort -nr | head -1 | cut -d' ' -f2- | xargs -r cat
echo
echo "===== RECENT LAUNCH LOG ====="
if [ -s "$LAUNCH_FILE" ]; then
  LAUNCH_LOG=$(head -n 1 "$LAUNCH_FILE")
  echo "LAUNCH_LOG=$LAUNCH_LOG"
  tail -n 100 "$LAUNCH_LOG" 2>/dev/null || true
else
  echo "LAUNCH_LOG_POINTER_MISSING"
fi
echo "===== RECENT DRIVER LOG ====="
tail -n 100 "$DRIVER" 2>/dev/null || true
echo "===== FAILURE SIGNALS ====="
if [ -n "${LAUNCH_LOG:-}" ]; then
  grep -nEi 'ERROR|Traceback|RuntimeError|timed out|SESSION_REJECTED|INCOMPLETE|mismatch|stalled|killed|out of memory' \
    "$LAUNCH_LOG" "$DRIVER" 2>/dev/null | tail -n 80 || true
else
  grep -nEi 'ERROR|Traceback|RuntimeError|timed out|SESSION_REJECTED|INCOMPLETE|mismatch|stalled|killed|out of memory' \
    "$DRIVER" 2>/dev/null | tail -n 80 || true
fi
