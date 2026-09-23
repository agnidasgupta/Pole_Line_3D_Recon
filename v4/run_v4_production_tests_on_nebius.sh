#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/v4_nebius_common.sh"
v4_print_context
BACKUP=$(v4_backup_code); echo "Code backup: $BACKUP"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
TEST_ROOT_HOST="$V4_RUN_ROOT_HOST/tests/$STAMP"
mkdir -p "$TEST_ROOT_HOST"
TEST_ROOT=$(v4_to_container_output "$TEST_ROOT_HOST")
NAME=${NAME:-v4-accept-$(printf '%s' "$V4_SESSION_SAFE" | tr -cd 'A-Za-z0-9_.-' | cut -c1-40)-${V4_DEPLOY_SHORT:0:8}}
CMD="GATE_FILES='${GATE_FILES:-32}' BATCH_SWEEP_FILES='${BATCH_SWEEP_FILES:-8}' BATCH_SIZES='${BATCH_SIZES:-8,12,16,20,24,32}' AMP='${AMP:-bf16}' SCORE_TOL='${SCORE_TOL:-1e-4}' bash v4_production_acceptance_inside_docker.sh '$TEST_ROOT' '$V4_SESSION_FILTER' '$V4_MODEL' '$V4_CAL' '$V4_STAGE2' '$V4_DIAG_ROOT' '${QUICK_SLICES:-12}' '${FULL_SESSION:-1}' '$V4_DEPLOY_FINGERPRINT'"
v4_detached_run "$NAME" "$TEST_ROOT/acceptance.log" "$CMD"
printf '%s\n' "$TEST_ROOT_HOST" > "$V4_SESSION_ROOT_HOST/LATEST_TEST_ROOT.txt"
printf '%s\n' "$TEST_ROOT_HOST" > "$V4_RUN_ROOT_HOST/LATEST_TEST_ROOT.txt"
echo "Acceptance test root: $TEST_ROOT_HOST"
