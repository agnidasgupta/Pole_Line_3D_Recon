#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STATE_DIR="$HERE/.v4_nebius_state"
LOG=$(cat "$STATE_DIR/build_log.txt" 2>/dev/null || true)
PID=$(cat "$STATE_DIR/build_pid.txt" 2>/dev/null || true)
[[ -n "$LOG" ]] || { echo 'No V4 build state found. Run build_v4_realtime_image_on_nebius.sh first.'; exit 2; }
echo "Build pid: ${PID:-unknown}"
echo "Build log: $LOG"
if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then echo 'Status: RUNNING'; else echo 'Status: NOT RUNNING'; fi
echo '--- last 80 build lines ---'
tail -80 "$LOG" 2>/dev/null || true
if grep -q 'V4_DOCKER_IMAGE_BUILD_OK' "$LOG" 2>/dev/null; then echo 'BUILD_RESULT=PASS'; else echo 'BUILD_RESULT=NOT_YET_PASSED'; fi
