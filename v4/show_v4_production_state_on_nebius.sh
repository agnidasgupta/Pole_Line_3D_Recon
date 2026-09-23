#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/v4_nebius_common.sh"
v4_print_context
NAME=$(cat "$V4_RUN_ROOT_HOST/current_container.txt" 2>/dev/null || true)
if [[ -n "$NAME" ]]; then
  echo "Container: $NAME"
  "${V4_DOCKER[@]}" ps -a --filter "name=^/${NAME}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' || true
  echo "--- last 80 container log lines ---"
  "${V4_DOCKER[@]}" logs --tail 80 "$NAME" 2>&1 || true
fi
PREFLIGHT_ROOT=$(cat "$V4_RUN_ROOT_HOST/LATEST_PREFLIGHT_ROOT.txt" 2>/dev/null || true)
if [[ -n "$PREFLIGHT_ROOT" ]]; then
  echo "Latest preflight root: $PREFLIGHT_ROOT"
  [[ -f "$PREFLIGHT_ROOT/preflight.log" ]] && { echo "--- last 60 preflight log lines ---"; tail -60 "$PREFLIGHT_ROOT/preflight.log" || true; }
  [[ -f "$PREFLIGHT_ROOT/PREFLIGHT_OK.txt" ]] && { echo "--- preflight result ---"; cat "$PREFLIGHT_ROOT/PREFLIGHT_OK.txt"; }
  [[ -f "$PREFLIGHT_ROOT/PREFLIGHT_FAILED.txt" ]] && { echo "--- preflight failure ---"; cat "$PREFLIGHT_ROOT/PREFLIGHT_FAILED.txt"; }
fi
TEST_ROOT=$(cat "$V4_RUN_ROOT_HOST/LATEST_TEST_ROOT.txt" 2>/dev/null || cat "$V4_SESSION_ROOT_HOST/LATEST_TEST_ROOT.txt" 2>/dev/null || true)
if [[ -n "$TEST_ROOT" ]]; then
  echo "Latest acceptance root: $TEST_ROOT"
  [[ -f "$TEST_ROOT/acceptance.log" ]] && { echo "--- last 80 acceptance log lines ---"; tail -80 "$TEST_ROOT/acceptance.log" || true; }
  [[ -f "$TEST_ROOT/recovery.log" ]] && { echo "--- last 80 recovery log lines ---"; tail -80 "$TEST_ROOT/recovery.log" || true; }
  [[ -f "$TEST_ROOT/RECOVERY_ACCEPTANCE_OK.txt" ]] && { echo "--- recovery result ---"; cat "$TEST_ROOT/RECOVERY_ACCEPTANCE_OK.txt"; }
  [[ -f "$TEST_ROOT/FAILED.txt" ]] && { echo "--- acceptance failure ---"; cat "$TEST_ROOT/FAILED.txt"; }
fi
echo "--- deployment gate/acceptance ---"
echo "Gate fingerprint match:       $V4_GATE_FINGERPRINT_MATCH"
echo "Acceptance fingerprint match: $V4_ACCEPTANCE_FINGERPRINT_MATCH"
[[ -f "$V4_DIAG_ROOT_HOST/PRODUCTION_ACCEPTANCE_OK.txt" ]] && cat "$V4_DIAG_ROOT_HOST/PRODUCTION_ACCEPTANCE_OK.txt"
echo "--- durable markers ---"
find "$V4_RUN_ROOT_HOST" -maxdepth 5 -type f \( -name 'COMPLETED.json' -o -name 'FAILED.json' -o -name 'FAILED.txt' -o -name 'STAGE*_COMPLETED.json' -o -name '*timing*.txt' -o -name '*timing*.json' -o -name 'REALTIME_REPLAY_VERIFICATION.json' -o -name 'RECOVERY_ACCEPTANCE_OK.txt' \) -print 2>/dev/null | sort || true
echo "--- failure files ---"
find "$V4_RUN_ROOT_HOST" -type f \( -path '*/errors/*' -o -name 'FAILED.json' -o -name 'FAILED.txt' \) -print 2>/dev/null | sort | tail -30 || true
echo "Recovery context: $V4_RUN_ROOT_HOST/run_context.env"
echo "Latest code backup: $(cat "$V4_SESSION_ROOT_HOST/backups/LATEST_CODE_BACKUP.txt" 2>/dev/null || echo '<none>')"
